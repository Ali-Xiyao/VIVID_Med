"""Prepare the ignored C6F MS-CXR manifest, geometry, and dataset lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bives_cxr.c6_ms_cxr_eval import (  # noqa: E402
    build_dataset_lock,
    build_ms_cxr_geometry,
    build_ms_cxr_manifest,
)


PUBLIC_ROOT = Path(r"H:\Xiyao_Wang\000_Public Dataset")
DATASET_ROOT = PUBLIC_ROOT / "MS-CXR"
DEFAULT_OUTPUT = ROOT / "local_runs" / "bives_cxr" / "c6_ms_cxr_postc5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DATASET_ROOT / "MS_CXR_Local_Alignment_v1.1.0.json",
    )
    parser.add_argument(
        "--mimic-metadata",
        type=Path,
        default=PUBLIC_ROOT / "mimic_cxr_other" / "mimic-cxr-2.0.0-metadata.csv.gz",
    )
    parser.add_argument(
        "--mimic-images-root",
        type=Path,
        default=PUBLIC_ROOT / "mimic-cxr" / "mimic-cxr" / "mimic-cxr-images",
    )
    parser.add_argument(
        "--strict-intake",
        type=Path,
        default=ROOT / "local_runs" / "bives_cxr" / "c6_ms_cxr_intake" / "ms_cxr_test_intake.json",
    )
    parser.add_argument(
        "--authority",
        type=Path,
        default=ROOT / "refine-logs" / "C6F_MS_CXR_POST_C5_EVALUATION_AUTHORITY_20260718.md",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "refine-logs" / "C6F_MS_CXR_QWEN35_2B_EVAL_CONFIG_20260718.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--geometry-workers", type=int, default=8)
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "ms_cxr_postc5_manifest.jsonl"
    geometry_rows_path = args.output_dir / "ms_cxr_postc5_geometry_rows.jsonl"
    geometry_lock_path = args.output_dir / "ms_cxr_postc5_geometry_lock.json"
    dataset_lock_path = args.output_dir / "ms_cxr_postc5_dataset_lock.json"
    mask_dir = args.output_dir / "geometry_masks"

    rows = build_ms_cxr_manifest(
        annotations_path=args.annotations,
        mimic_metadata_path=args.mimic_metadata,
        mimic_images_root=args.mimic_images_root,
    )
    write_jsonl(manifest_path, rows)
    geometry_rows, geometry_summary = build_ms_cxr_geometry(
        rows, mask_dir=mask_dir, workers=args.geometry_workers
    )
    write_jsonl(geometry_rows_path, geometry_rows)
    write_json(geometry_lock_path, geometry_summary)

    release_paths = [
        args.annotations,
        args.mimic_metadata,
        DATASET_ROOT / "license_record.json",
        DATASET_ROOT
        / "ms-cxr-making-the-most-of-text-semantics-to-improve-biomedical-vision-language-processing-1.1.0.zip",
        ROOT
        / "local_runs"
        / "bives_cxr"
        / "c6_ms_cxr_intake"
        / "prior_mimic_access_registry.json",
    ]
    source_paths = [
        ROOT / "bives_cxr" / "c6_ms_cxr_eval.py",
        ROOT / "bives_cxr" / "pixel_interventions.py",
        ROOT / "bives_cxr" / "rescue_protocol.py",
        ROOT / "bives_cxr" / "polarity_runtime.py",
        ROOT / "scripts" / "prepare_bives_c6_ms_cxr_evaluation.py",
        ROOT / "scripts" / "evaluate_bives_c6_ms_cxr.py",
    ]
    for path in [*release_paths, *source_paths, args.strict_intake, args.authority, args.config]:
        if not path.is_file():
            raise FileNotFoundError(path)
    dataset_lock = build_dataset_lock(
        manifest_path=manifest_path,
        geometry_rows_path=geometry_rows_path,
        geometry_summary=geometry_summary,
        strict_intake_path=args.strict_intake,
        authority_path=args.authority,
        config_path=args.config,
        source_paths=source_paths,
        release_paths=release_paths,
    )
    write_json(dataset_lock_path, dataset_lock)
    print(
        json.dumps(
            {
                "status": (
                    "C6F_PREOPEN_DATA_LOCK_PASS"
                    if geometry_summary["status"] == "pass"
                    else "C6F_PREOPEN_GEOMETRY_FAIL"
                ),
                "manifest": str(manifest_path),
                "dataset_lock": str(dataset_lock_path),
                "geometry_lock": str(geometry_lock_path),
                "rows": len(rows),
                "geometry_eligible": geometry_summary["eligible"],
                "canonical_artifact_sha256": dataset_lock["canonical_artifact_sha256"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if geometry_summary["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
