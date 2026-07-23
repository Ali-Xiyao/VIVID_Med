"""Estimate a patient/study manifest's referenced pixel footprint.

This is read-only and does not open image contents. It supports deciding
whether a canonical-frontal subset should be packaged before server transfer.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--image-root", type=Path)
    args = parser.parse_args()

    files = 0
    missing = 0
    total_bytes = 0
    with args.manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "image_path" not in (reader.fieldnames or []):
            raise ValueError("manifest requires image_path")
        for row in reader:
            path = Path(row["image_path"])
            if args.image_root is not None:
                path = args.image_root / path
            if not path.is_file():
                missing += 1
                continue
            files += 1
            total_bytes += path.stat().st_size
    gib = total_bytes / (1024 ** 3)
    print(f"files={files} missing={missing} bytes={total_bytes} GiB={gib:.3f}")
    return 2 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
