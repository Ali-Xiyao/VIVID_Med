"""Audit BiVES-CXR manifests before server training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bives_cxr.audit import audit_manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--check-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--require-complete-statements",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--check-decodable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reject-constant-images",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--require-provenance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--verify-image-sha256",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-matching-protocol",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifests = {"train": args.train, "val": args.val}
    if args.calibration is not None:
        manifests["calibration"] = args.calibration
    if args.test is not None:
        manifests["test"] = args.test
    report = audit_manifests(
        manifests,
        data_root=args.data_root,
        check_images=args.check_images,
        require_complete_statements=args.require_complete_statements,
        check_decodable=args.check_decodable,
        reject_constant_images=args.reject_constant_images,
        require_provenance=args.require_provenance,
        verify_image_sha256=args.verify_image_sha256,
        require_matching_protocol=args.require_matching_protocol,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if report["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
