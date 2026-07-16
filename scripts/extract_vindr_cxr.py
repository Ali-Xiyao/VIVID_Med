"""Safely and resumably extract the official VinDr-CXR ZIP package."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUBLIC_ROOT = Path(r"H:\Xiyao_Wang\000_Public Dataset")
DEFAULT_ARCHIVE = DEFAULT_PUBLIC_ROOT / (
    "vindr-cxr-an-open-dataset-of-chest-x-rays-with-radiologist-annotations-1.0.0.zip"
)
DEFAULT_PROGRESS = ROOT / "outputs/logs/vindr_cxr_extract_progress.json"
COPY_BUFFER_BYTES = 16 * 1024 * 1024
FREE_SPACE_RESERVE_BYTES = 20 * 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_PUBLIC_ROOT)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--list-only", action="store_true")
    return parser.parse_args()


def safe_output_path(destination: Path, member_name: str) -> Path:
    relative = PurePosixPath(member_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe ZIP member path: {member_name!r}")
    output = destination.joinpath(*relative.parts)
    output.resolve().relative_to(destination.resolve())
    return output


def write_progress(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    archive = args.archive.resolve()
    destination = args.destination.resolve()
    progress_path = args.progress.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as package:
        members = package.infolist()
        file_members = [member for member in members if not member.is_dir()]
        total_bytes = sum(member.file_size for member in file_members)
        total_files = len(file_members)
        for member in members:
            safe_output_path(destination, member.filename)

        free_bytes = shutil.disk_usage(destination).free
        existing_bytes = sum(
            member.file_size
            for member in file_members
            if (candidate := safe_output_path(destination, member.filename)).is_file()
            and candidate.stat().st_size == member.file_size
        )
        required_bytes = total_bytes - existing_bytes
        summary = {
            "archive": str(archive),
            "destination": str(destination),
            "total_files": total_files,
            "total_uncompressed_bytes": total_bytes,
            "already_complete_bytes": existing_bytes,
            "required_bytes": required_bytes,
            "free_bytes_before": free_bytes,
            "reserve_bytes": FREE_SPACE_RESERVE_BYTES,
        }
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        if args.list_only:
            return
        if free_bytes < required_bytes + FREE_SPACE_RESERVE_BYTES:
            raise OSError(
                f"Insufficient free space: free={free_bytes}, required={required_bytes}, "
                f"reserve={FREE_SPACE_RESERVE_BYTES}"
            )

        started = time.time()
        completed_files = 0
        completed_bytes = 0
        skipped_files = 0
        for index, member in enumerate(file_members, start=1):
            output = safe_output_path(destination, member.filename)
            if output.is_file() and output.stat().st_size == member.file_size:
                skipped_files += 1
                completed_files += 1
                completed_bytes += member.file_size
                continue

            output.parent.mkdir(parents=True, exist_ok=True)
            partial = output.with_name(output.name + ".part")
            if partial.exists():
                partial.unlink()
            with package.open(member, "r") as source, partial.open("wb") as target:
                shutil.copyfileobj(source, target, COPY_BUFFER_BYTES)
            if partial.stat().st_size != member.file_size:
                raise IOError(
                    f"Extracted size mismatch for {member.filename}: "
                    f"expected={member.file_size}, actual={partial.stat().st_size}"
                )
            os.replace(partial, output)
            completed_files += 1
            completed_bytes += member.file_size

            if index % 100 == 0 or index == total_files:
                elapsed = time.time() - started
                payload = {
                    **summary,
                    "status": "extracting",
                    "completed_files": completed_files,
                    "skipped_files": skipped_files,
                    "completed_bytes": completed_bytes,
                    "elapsed_seconds": elapsed,
                    "last_member": member.filename,
                }
                write_progress(progress_path, payload)
                print(
                    f"files={completed_files}/{total_files} "
                    f"bytes={completed_bytes}/{total_bytes} elapsed={elapsed:.1f}s",
                    flush=True,
                )

        elapsed = time.time() - started
        top_level = PurePosixPath(file_members[0].filename).parts[0]
        marker = destination / top_level / "_extraction_complete.json"
        final_payload = {
            **summary,
            "status": "complete",
            "completed_files": completed_files,
            "skipped_files": skipped_files,
            "completed_bytes": completed_bytes,
            "elapsed_seconds": elapsed,
            "extraction_marker": str(marker),
            "zip_crc_note": "Every member was read through zipfile; CRC is checked while extracting.",
        }
        marker.write_text(json.dumps(final_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_progress(progress_path, final_payload)
        print(json.dumps(final_payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
