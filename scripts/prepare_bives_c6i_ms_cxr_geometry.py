"""Build the score-free C6I actual-input-space MS-CXR geometry release."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bives_cxr.c6g_geometry import (  # noqa: E402
    derive_frozen_thresholds,
    validate_frozen_thresholds,
)
from bives_cxr.c6i_input_geometry import (  # noqa: E402
    build_c6i_geometry,
    read_jsonl,
    write_c6i_geometry_artifacts,
)


C6F_ROOT = ROOT / "local_runs/bives_cxr/c6_ms_cxr_postc5"
C6G_ROOT = ROOT / "local_runs/bives_cxr/c6g_ms_cxr_geometry_final"
DEFAULT_ROOT = ROOT / "local_runs/bives_cxr/c6i_ms_cxr_actual_input_geometry"


def predecessor_paths(manifest: Path) -> dict[str, Path]:
    return {
        "c6f_authority": ROOT / "refine-logs/C6F_MS_CXR_POST_C5_EVALUATION_AUTHORITY_20260718.md",
        "c6f_config": ROOT / "refine-logs/C6F_MS_CXR_QWEN35_2B_EVAL_CONFIG_20260718.yaml",
        "c6f_execution_log": ROOT / "refine-logs/C6F_MS_CXR_PREOPEN_GEOMETRY_EXECUTION_LOG_20260718.md",
        "c6f_manifest": manifest,
        "c6f_geometry_rows": C6F_ROOT / "ms_cxr_postc5_geometry_rows.jsonl",
        "c6f_geometry_lock": C6F_ROOT / "ms_cxr_postc5_geometry_lock.json",
        "c6f_dataset_lock": C6F_ROOT / "ms_cxr_postc5_dataset_lock.json",
        "c6g_authority": ROOT / "refine-logs/C6G_MS_CXR_GEOMETRY_ONLY_AUTHORITY_20260718.md",
        "c6g_thresholds": ROOT / "refine-logs/C6G_MS_CXR_GEOMETRY_THRESHOLDS_20260718.json",
        "c6g_execution_log": ROOT / "refine-logs/C6G_MS_CXR_GEOMETRY_EXECUTION_LOG_20260718.md",
        "c6g_lock": C6G_ROOT / "c6g_geometry_lock.json",
        "c6g_rows": C6G_ROOT / "c6g_geometry_rows.jsonl",
        "c6g_certificates": C6G_ROOT / "c6g_candidate_certificates.jsonl",
        "c6h_authority": ROOT / "refine-logs/C6H_MS_CXR_ONE_TIME_EVALUATION_AUTHORITY_20260718.md",
        "c6h_config": ROOT / "refine-logs/C6H_MS_CXR_QWEN35_2B_EVAL_CONFIG_20260718.yaml",
        "c6h_preopen_log": ROOT / "refine-logs/C6H_MS_CXR_PREOPEN_EXECUTION_LOG_20260718.md",
        "c6h_failure": ROOT / "refine-logs/C6H_MS_CXR_PRE_SCORE_PIXEL_ALIGNMENT_FAILURE_20260718.md",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--geometry-workers", type=int, default=8)
    parser.add_argument("--manifest", type=Path, default=C6F_ROOT / "ms_cxr_postc5_manifest.jsonl")
    parser.add_argument(
        "--c4-geometry-rows",
        type=Path,
        default=ROOT / "local_runs/bives_cxr/vindr_connected_control_geometry/connected_geometry_rows.jsonl",
    )
    parser.add_argument(
        "--c5-geometry-rows",
        type=Path,
        default=ROOT / "local_runs/bives_cxr/connected_control_c5_confirmation/geometry_rows.jsonl",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=ROOT / "refine-logs/C6G_MS_CXR_GEOMETRY_THRESHOLDS_20260718.json",
    )
    parser.add_argument(
        "--authority",
        type=Path,
        default=ROOT / "refine-logs/C6I_MS_CXR_ACTUAL_INPUT_RECOVERY_AUTHORITY_20260718.md",
    )
    return parser.parse_args()


def clean_commit() -> str:
    for command in (
        ["git", "diff", "--quiet", "HEAD", "--"],
        ["git", "diff", "--cached", "--quiet"],
    ):
        if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
            raise ValueError("C6I final geometry requires a clean committed tracked tree")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"C6I output directory is not empty: {args.output_dir}")
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    validate_frozen_thresholds(
        thresholds,
        derive_frozen_thresholds(args.c4_geometry_rows, args.c5_geometry_rows),
    )
    predecessors = predecessor_paths(args.manifest)
    for path in [args.manifest, args.thresholds, args.authority, *predecessors.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)
    source_commit = clean_commit()
    rows, summary = build_c6i_geometry(
        read_jsonl(args.manifest),
        mask_dir=args.output_dir / "geometry_masks",
        thresholds=thresholds,
        workers=args.geometry_workers,
    )
    artifacts = write_c6i_geometry_artifacts(
        output_dir=args.output_dir,
        geometry_rows=rows,
        summary=summary,
        manifest_path=args.manifest,
        authority_path=args.authority,
        threshold_path=args.thresholds,
        predecessor_paths=predecessors,
        implementation_paths={
            "c6i_geometry_module": ROOT / "bives_cxr/c6i_input_geometry.py",
            "c6i_geometry_entrypoint": ROOT / "scripts/prepare_bives_c6i_ms_cxr_geometry.py",
        },
        source_commit=source_commit,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "rows": summary["rows"],
                "eligible": summary["eligible"],
                "infeasible": summary["infeasible"],
                "actual_image_sizes": summary["actual_image_sizes"],
                "evaluation_gate_open_geometry": summary["evaluation_gate_open_geometry"],
                "model_evaluation_authorized": False,
                "gpu_authorized": False,
                "scores_accessed": False,
                "geometry_lock": str(artifacts["geometry_lock"]),
                "geometry_lock_canonical_sha256": artifacts["lock"]["canonical_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if summary["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
