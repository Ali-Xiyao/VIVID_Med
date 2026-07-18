"""Build the local score-free C6G MS-CXR geometry lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bives_cxr.c6g_geometry import (  # noqa: E402
    build_c6g_geometry,
    derive_frozen_thresholds,
    validate_frozen_thresholds,
    verify_c6f_immutability,
    write_c6g_artifacts,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "local_runs/bives_cxr/c6g_ms_cxr_geometry",
    )
    parser.add_argument("--geometry-workers", type=int, default=8)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            ROOT
            / "local_runs/bives_cxr/c6_ms_cxr_postc5/ms_cxr_postc5_manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--c4-geometry-rows",
        type=Path,
        default=(
            ROOT
            / "local_runs/bives_cxr/vindr_connected_control_geometry/connected_geometry_rows.jsonl"
        ),
    )
    parser.add_argument(
        "--c5-geometry-rows",
        type=Path,
        default=(
            ROOT
            / "local_runs/bives_cxr/connected_control_c5_confirmation/geometry_rows.jsonl"
        ),
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=ROOT / "refine-logs/C6G_MS_CXR_GEOMETRY_THRESHOLDS_20260718.json",
    )
    parser.add_argument(
        "--authority",
        type=Path,
        default=ROOT / "refine-logs/C6G_MS_CXR_GEOMETRY_ONLY_AUTHORITY_20260718.md",
    )
    parser.add_argument(
        "--protocol-plan",
        type=Path,
        default=ROOT / "BiVES_C6G_MS_CXR_geometry_protocol_plan.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"C6G output directory is not empty; use a new replay directory: {args.output_dir}"
        )
    frozen_thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    derived_thresholds = derive_frozen_thresholds(
        args.c4_geometry_rows,
        args.c5_geometry_rows,
    )
    validate_frozen_thresholds(frozen_thresholds, derived_thresholds)
    c6f_hashes = verify_c6f_immutability(
        {
            "authority": ROOT
            / "refine-logs/C6F_MS_CXR_POST_C5_EVALUATION_AUTHORITY_20260718.md",
            "config": ROOT
            / "refine-logs/C6F_MS_CXR_QWEN35_2B_EVAL_CONFIG_20260718.yaml",
            "execution_log": ROOT
            / "refine-logs/C6F_MS_CXR_PREOPEN_GEOMETRY_EXECUTION_LOG_20260718.md",
            "manifest": args.manifest,
            "geometry_rows": ROOT
            / "local_runs/bives_cxr/c6_ms_cxr_postc5/ms_cxr_postc5_geometry_rows.jsonl",
            "geometry_lock": ROOT
            / "local_runs/bives_cxr/c6_ms_cxr_postc5/ms_cxr_postc5_geometry_lock.json",
            "dataset_lock": ROOT
            / "local_runs/bives_cxr/c6_ms_cxr_postc5/ms_cxr_postc5_dataset_lock.json",
        }
    )
    rows = _read_jsonl(args.manifest)
    geometry_rows, summary = build_c6g_geometry(
        rows,
        mask_dir=args.output_dir / "geometry_masks",
        thresholds=frozen_thresholds,
        workers=args.geometry_workers,
    )
    artifacts = write_c6g_artifacts(
        output_dir=args.output_dir,
        geometry_rows=geometry_rows,
        summary=summary,
        source_manifest_path=args.manifest,
        authority_path=args.authority,
        protocol_plan_path=args.protocol_plan,
        threshold_path=args.thresholds,
        c6f_hashes=c6f_hashes,
    )
    report = {
        "status": summary["status"],
        "rows": summary["rows"],
        "eligible": summary["eligible"],
        "infeasible": summary["infeasible"],
        "evaluation_gate_open_geometry": summary["evaluation_gate_open_geometry"],
        "model_evaluation_authorized": False,
        "gpu_authorized": False,
        "image_decode_authorized": False,
        "scores_accessed": False,
        "geometry_rows": str(artifacts["geometry_rows"]),
        "candidate_certificates": str(artifacts["candidate_certificates"]),
        "geometry_lock": str(artifacts["geometry_lock"]),
        "geometry_lock_canonical_sha256": artifacts["lock"]["canonical_sha256"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
