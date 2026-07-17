"""Create a passing joint train/val/calibration/test dataset lock before formal BiVES-CXR training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bives_cxr.dataset_lock import build_dataset_lock, dataset_lock_sha256, validate_dataset_lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "val", "calibration", "test"):
        parser.add_argument(f"--{split}", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    options = {
        "check_images": True,
        "require_complete_statements": True,
        "check_decodable": True,
        "reject_constant_images": True,
        "require_provenance": True,
        "verify_image_sha256": True,
        "require_matching_protocol": True,
    }
    lock = build_dataset_lock(
        {split: getattr(args, split) for split in ("train", "val", "calibration", "test")},
        data_root=args.data_root,
        audit_options=options,
    )
    lock_sha256 = dataset_lock_sha256(lock)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # A lock is not usable until the same verifier accepts what was written.
    validate_dataset_lock(
        args.output,
        {split: getattr(args, split) for split in ("train", "val", "calibration", "test")},
        data_root=args.data_root,
        audit_options=options,
    )
    print(json.dumps({"status": "pass", "dataset_lock_sha256": lock_sha256}, ensure_ascii=False))


if __name__ == "__main__":
    main()
