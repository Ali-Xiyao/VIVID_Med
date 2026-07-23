"""Upload files through the SUES login node's PTY-only SSH workaround.

The current login node accepts SSH authentication but stalls non-PTY exec and
SFTP channels.  This client requests a PTY, switches it to raw/no-echo mode,
and then starts OpenSSH's sftp-server explicitly.  Files are written through a
``.rcsd-part`` sibling and atomically renamed after size validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import posixpath
import stat
import time
from pathlib import Path, PurePosixPath

import paramiko


READY = b"RCSD_RAW_SFTP_READY\n"


def sha256_local(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_remote(sftp: paramiko.SFTPClient, path: str) -> str:
    digest = hashlib.sha256()
    with sftp.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_raw_sftp(args: argparse.Namespace) -> tuple[paramiko.SSHClient, paramiko.SFTPClient]:
    client = paramiko.SSHClient()
    client.load_system_host_keys(str(args.known_hosts))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(
        hostname=args.host,
        username=args.user,
        key_filename=str(args.key),
        timeout=args.timeout,
        banner_timeout=args.timeout,
        auth_timeout=args.timeout,
        channel_timeout=args.timeout,
        allow_agent=False,
        look_for_keys=False,
    )
    transport = client.get_transport()
    if transport is None:
        raise RuntimeError("SSH transport was not created")
    transport.set_keepalive(30)
    channel = transport.open_session(timeout=args.timeout)
    channel.settimeout(args.timeout)
    channel.get_pty(term="dumb", width=80, height=24)
    command = (
        "stty raw -echo -onlcr -icrnl -inlcr; "
        "printf 'RCSD_RAW_SFTP_READY\\n'; "
        f"exec {args.sftp_server}"
    )
    channel.exec_command(command)
    received = b""
    while READY not in received:
        chunk = channel.recv(1024)
        if not chunk:
            raise RuntimeError(f"raw PTY closed before ready marker: {received!r}")
        received += chunk
    trailing = received.split(READY, 1)[1]
    if trailing:
        raise RuntimeError(f"unexpected bytes before SFTP handshake: {trailing!r}")
    return client, paramiko.SFTPClient(channel)


def mkdir_p(sftp: paramiko.SFTPClient, path: str) -> None:
    pure = PurePosixPath(path)
    current = "/" if pure.is_absolute() else ""
    for part in pure.parts:
        if part == "/":
            continue
        current = posixpath.join(current, part)
        try:
            mode = sftp.stat(current).st_mode
            if mode is None or not stat.S_ISDIR(mode):
                raise NotADirectoryError(current)
        except FileNotFoundError:
            sftp.mkdir(current)
        except OSError as error:
            if getattr(error, "errno", None) == 2:
                sftp.mkdir(current)
            else:
                raise


def remote_size(sftp: paramiko.SFTPClient, path: str) -> int | None:
    try:
        return int(sftp.stat(path).st_size)
    except FileNotFoundError:
        return None
    except OSError as error:
        if getattr(error, "errno", None) == 2:
            return None
        raise


def upload_file(
    sftp: paramiko.SFTPClient,
    local: Path,
    remote: str,
    *,
    verify_sha256: bool,
    announce: bool = True,
) -> str:
    local_size = local.stat().st_size
    existing_size = remote_size(sftp, remote)
    if existing_size == local_size:
        if not verify_sha256 or sha256_remote(sftp, remote) == sha256_local(local):
            return "verified-existing"

    mkdir_p(sftp, posixpath.dirname(remote))
    partial = f"{remote}.rcsd-part"
    offset = remote_size(sftp, partial) or 0
    if offset > local_size:
        sftp.remove(partial)
        offset = 0

    mode = "r+b" if offset else "wb"
    started = time.monotonic()
    last_report = started
    with local.open("rb") as source, sftp.open(partial, mode) as target:
        source.seek(offset)
        target.seek(offset)
        target.set_pipelined(True)
        transferred = offset
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            target.write(chunk)
            transferred += len(chunk)
            now = time.monotonic()
            if now - last_report >= 30:
                print(f"PROGRESS|{local}|{transferred}|{local_size}", flush=True)
                last_report = now

    if remote_size(sftp, partial) != local_size:
        raise RuntimeError(f"remote size mismatch after upload: {remote}")
    if verify_sha256 and sha256_remote(sftp, partial) != sha256_local(local):
        raise RuntimeError(f"remote SHA-256 mismatch after upload: {remote}")
    if existing_size is not None:
        sftp.remove(remote)
    try:
        sftp.posix_rename(partial, remote)
    except OSError:
        sftp.rename(partial, remote)
    elapsed = max(time.monotonic() - started, 1e-6)
    if announce:
        print(
            f"UPLOADED|{local}|{remote}|{local_size}|{local_size / elapsed:.1f}Bps",
            flush=True,
        )
    return "uploaded"


def upload_manifest(
    sftp: paramiko.SFTPClient,
    manifest: Path,
    local_root: Path,
    remote_root: str,
) -> None:
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for index, row in enumerate(rows, 1):
        relative = row["relative_path"]
        local = local_root / Path(relative)
        if not local.is_file():
            raise FileNotFoundError(local)
        expected_size = int(row["bytes"])
        expected_hash = row["sha256"].lower()
        if local.stat().st_size != expected_size or sha256_local(local) != expected_hash:
            raise RuntimeError(f"local manifest mismatch: {relative}")
        remote = posixpath.join(remote_root, PurePosixPath(relative).as_posix())
        status = upload_file(sftp, local, remote, verify_sha256=True)
        print(f"MANIFEST|{index}/{len(rows)}|{status}|{relative}", flush=True)
    try:
        manifest_relative = manifest.resolve().relative_to(local_root.resolve()).as_posix()
    except ValueError:
        return
    manifest_remote = posixpath.join(remote_root, manifest_relative)
    status = upload_file(sftp, manifest, manifest_remote, verify_sha256=True)
    print(f"MANIFEST_AUTHORITY|{status}|{manifest_relative}", flush=True)


def upload_tree(sftp: paramiko.SFTPClient, local_root: Path, remote_root: str) -> None:
    if not local_root.is_dir():
        raise NotADirectoryError(local_root)
    files = sorted(path for path in local_root.rglob("*") if path.is_file())
    for index, local in enumerate(files, 1):
        relative = local.relative_to(local_root).as_posix()
        remote = posixpath.join(remote_root, relative)
        status = upload_file(
            sftp,
            local,
            remote,
            verify_sha256=False,
            announce=False,
        )
        if index == 1 or index % 100 == 0 or index == len(files):
            print(f"TREE|{index}/{len(files)}|{status}|{relative}", flush=True)


def parse_mapping(value: str) -> tuple[Path, str]:
    try:
        local, remote = value.split("::", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("mapping must be LOCAL::REMOTE") from error
    return Path(local), remote


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="172.20.52.10")
    parser.add_argument("--user", default="dqxy11")
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--known-hosts", type=Path, required=True)
    parser.add_argument("--sftp-server", default="/usr/libexec/openssh/sftp-server")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--local-root", type=Path, default=Path.cwd())
    parser.add_argument("--remote-root")
    parser.add_argument("--file", action="append", type=parse_mapping, default=[])
    parser.add_argument("--tree", action="append", type=parse_mapping, default=[])
    args = parser.parse_args()
    if bool(args.manifest) != bool(args.remote_root):
        parser.error("--manifest and --remote-root must be supplied together")
    if not args.manifest and not args.file and not args.tree:
        parser.error("provide a manifest, file, or tree")

    client, sftp = open_raw_sftp(args)
    try:
        if args.manifest:
            upload_manifest(sftp, args.manifest, args.local_root, args.remote_root)
        for local, remote in args.file:
            if not local.is_file():
                raise FileNotFoundError(local)
            status = upload_file(sftp, local, remote, verify_sha256=False)
            print(f"FILE|{status}|{local}|{remote}", flush=True)
        for local, remote in args.tree:
            upload_tree(sftp, local, remote)
    finally:
        sftp.close()
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
