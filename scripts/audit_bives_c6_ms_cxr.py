"""Build the frozen MIMIC registry or audit an acquired MS-CXR v1.1.0 release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bives_cxr.c6_ms_cxr import (  # noqa: E402
    audit_ms_cxr_test_release,
    build_mimic_prior_access_registry,
    preflight_ms_cxr_test_release,
)


PUBLIC_ROOT = Path(r"H:\Xiyao_Wang\000_Public Dataset")
DEFAULT_DATASET_ROOT = PUBLIC_ROOT / "MS-CXR"
DEFAULT_PACKAGE_ARCHIVE = (
    DEFAULT_DATASET_ROOT
    / "ms-cxr-making-the-most-of-text-semantics-to-improve-biomedical-vision-language-processing-1.1.0.zip"
)
DEFAULT_METADATA = PUBLIC_ROOT / "mimic_cxr_other" / "mimic-cxr-2.0.0-metadata.csv.gz"
DEFAULT_IMAGES_ROOT = PUBLIC_ROOT / "mimic-cxr" / "mimic-cxr" / "mimic-cxr-images"
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "bives_cxr" / "c6_ms_cxr_intake"
DEFAULT_PRIOR_MANIFESTS = (
    ROOT / "local_runs" / "bives_cxr" / "p0_intake" / "mimic_candidates_5k.jsonl",
    ROOT / "local_runs" / "bives_cxr" / "weak_sc_v1" / "weak_sc_train.jsonl",
    ROOT / "local_runs" / "bives_cxr" / "weak_sc_v1" / "weak_sc_val.jsonl",
    ROOT / "local_runs" / "bives_cxr" / "proxy_p0_sc_v3_input" / "train_proxy.jsonl",
    ROOT / "local_runs" / "bives_cxr" / "proxy_p0_sc_v3_input" / "val_proxy.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    registry = subparsers.add_parser("build-prior-registry")
    registry.add_argument(
        "--manifest", type=Path, action="append", dest="manifests"
    )
    registry.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "prior_mimic_access_registry.json",
    )

    audit = subparsers.add_parser("audit-test-release")
    audit.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    audit.add_argument("--mimic-metadata", type=Path, default=DEFAULT_METADATA)
    audit.add_argument("--mimic-images-root", type=Path, default=DEFAULT_IMAGES_ROOT)
    audit.add_argument("--license-record", type=Path, required=True)
    audit.add_argument("--package-archive", type=Path, default=DEFAULT_PACKAGE_ARCHIVE)
    audit.add_argument(
        "--prior-registry",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "prior_mimic_access_registry.json",
    )

    preflight = subparsers.add_parser(
        "preflight-test-release",
        help="validate release structure without asserting credential/CITI/DUA status",
    )
    preflight.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    preflight.add_argument("--mimic-metadata", type=Path, default=DEFAULT_METADATA)
    preflight.add_argument(
        "--mimic-images-root", type=Path, default=DEFAULT_IMAGES_ROOT
    )
    preflight.add_argument(
        "--package-archive", type=Path, default=DEFAULT_PACKAGE_ARCHIVE
    )
    preflight.add_argument(
        "--prior-registry",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "prior_mimic_access_registry.json",
    )
    preflight.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "ms_cxr_test_structure_preflight.json",
    )
    audit.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "ms_cxr_test_intake.json",
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
        payload = build_mimic_prior_access_registry(
            args.manifests or DEFAULT_PRIOR_MANIFESTS
        )
        summary = {
            "status": "prior_access_registry_ready",
            "output": str(args.output),
            "counts": payload["counts"],
            "identifier_set_sha256": payload["identifier_set_sha256"],
            "canonical_artifact_sha256": payload["canonical_artifact_sha256"],
            "contains_raw_identifiers": payload["contains_raw_identifiers"],
        }
    elif args.command == "audit-test-release":
        payload = audit_ms_cxr_test_release(
            args.dataset_root,
            mimic_metadata=args.mimic_metadata,
            mimic_images_root=args.mimic_images_root,
            license_record=args.license_record,
            package_archive=args.package_archive,
            prior_registry=args.prior_registry,
        )
        summary = payload
    else:
        payload = preflight_ms_cxr_test_release(
            args.dataset_root,
            mimic_metadata=args.mimic_metadata,
            mimic_images_root=args.mimic_images_root,
            package_archive=args.package_archive,
            prior_registry=args.prior_registry,
        )
        summary = payload
    write_json(args.output, payload)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
