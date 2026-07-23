# SUES SSH/SFTP recovery (2026-07-22)

## Symptom

`dqxy11@172.20.52.10` completed TCP negotiation and public-key
authentication, but normal exec and SFTP sessions stalled immediately after
the session channel opened. Verbose SSH showed a zero remote receive window.

## Evidence

- TCP port 22 returned `SSH-2.0-OpenSSH_8.7`.
- The configured ED25519 key was accepted.
- Plain exec, default SCP/SFTP, legacy SCP, and Paramiko non-PTY channels all
  failed at session startup.
- `ssh -tt -o IPQoS=none sues-hpc ...` succeeded and reached login node
  `mu01`.
- Remote account process/file limits were healthy, `~/.ssh/rc` was absent,
  and `/usr/libexec/openssh/sftp-server` existed.

## Workaround

For interactive commands, use:

```powershell
ssh sues-hpc-tty
```

For file transfer, standard SFTP cannot simply request a PTY because terminal
line processing corrupts the binary SFTP protocol. Use:

```powershell
python scripts/upload_rawpty_sftp.py --help
```

The uploader requests a PTY, switches it to raw/no-echo mode, explicitly
starts OpenSSH's SFTP server, writes resumable `.rcsd-part` files, validates
size, and atomically renames completed files. Manifest uploads additionally
verify SHA-256 before and after transfer.

## Verified probe

A 277-byte project file was uploaded to `/tmp`, read back through the same
channel, and matched its local SHA-256. The probe was then removed.

This is a client-side compatibility workaround, not a server-side repair. An
administrator should still inspect why non-PTY sessions on `mu01` stall.
