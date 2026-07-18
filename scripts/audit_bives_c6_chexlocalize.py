"""Build the hashed prior registry or audit a local CheXlocalize test release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bives_cxr.c6_intake import (  # noqa: E402
    audit_chexlocalize_test_release,
    build_chexpert_prior_access_registry,
)


PUBLIC_ROOT = Path(r"H:\Xiyao_Wang\000_Public Dataset")
DEFAULT_DATASET_ROOT = PUBLIC_ROOT / "CheXlocalize"
DEFAULT_VALID_CSV = PUBLIC_ROOT / "CheXpert-v1.0-small" / "valid.csv"
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "bives_cxr" / "c6_chexlocalize_intake"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    registry = subparsers.add_parser("build-prior-registry")
    registry.add_argument("--valid-csv", type=Path, default=DEFAULT_VALID_CSV)
    registry.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "prior_chexpert_access_registry.json",
    )

    audit = subparsers.add_parser("audit-test-release")
    audit.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    audit.add_argument("--license-record", type=Path, required=True)
    audit.add_argument(
        "--prior-registry",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "prior_chexpert_access_registry.json",
    )
    audit.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "chexlocalize_test_intake.json",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.command == "build-prior-registry":
        payload = build_chexpert_prior_access_registry(args.valid_csv)
        summary = {
            "status": "prior_access_registry_ready",
            "output": str(args.output),
            "counts": payload["counts"],
            "canonical_artifact_sha256": payload["canonical_artifact_sha256"],
            "contains_raw_identifiers": payload["contains_raw_identifiers"],
        }
    else:
        payload = audit_chexlocalize_test_release(
            args.dataset_root,
            license_record=args.license_record,
            prior_registry=args.prior_registry,
        )
        summary = payload
    write_json(args.output, payload)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
