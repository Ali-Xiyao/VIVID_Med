"""Build the ignored, fail-closed C6H pre-open lock without loading a model."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bives_cxr.c6h_ms_cxr_eval import build_c6h_lock  # noqa: E402


C6F_ROOT = ROOT / "local_runs" / "bives_cxr" / "c6_ms_cxr_postc5"
C6G_ROOT = ROOT / "local_runs" / "bives_cxr" / "c6g_ms_cxr_geometry_final"
DEFAULT_ROOT = ROOT / "local_runs" / "bives_cxr" / "c6h_ms_cxr_one_time"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authority",
        type=Path,
        default=ROOT
        / "refine-logs"
        / "C6H_MS_CXR_ONE_TIME_EVALUATION_AUTHORITY_20260718.md",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "refine-logs" / "C6H_MS_CXR_QWEN35_2B_EVAL_CONFIG_20260718.yaml",
    )
    parser.add_argument("--manifest", type=Path, default=C6F_ROOT / "ms_cxr_postc5_manifest.jsonl")
    parser.add_argument(
        "--strict-intake",
        type=Path,
        default=ROOT
        / "local_runs"
        / "bives_cxr"
        / "c6_ms_cxr_intake"
        / "ms_cxr_test_intake.json",
    )
    parser.add_argument("--c6g-lock", type=Path, default=C6G_ROOT / "c6g_geometry_lock.json")
    parser.add_argument("--c6g-rows", type=Path, default=C6G_ROOT / "c6g_geometry_rows.jsonl")
    parser.add_argument(
        "--c6g-certificates",
        type=Path,
        default=C6G_ROOT / "c6g_candidate_certificates.jsonl",
    )
    parser.add_argument("--c6g-mask-dir", type=Path, default=C6G_ROOT / "geometry_masks")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def clean_commit() -> str:
    for command in (
        ["git", "diff", "--quiet", "HEAD", "--"],
        ["git", "diff", "--cached", "--quiet"],
    ):
        if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
            raise ValueError("C6H pre-open lock requires a clean committed tracked tree")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> None:
    args = parse_args()
    source_commit = clean_commit()
    c6f_paths = {
        "authority": ROOT / "refine-logs/C6F_MS_CXR_POST_C5_EVALUATION_AUTHORITY_20260718.md",
        "config": ROOT / "refine-logs/C6F_MS_CXR_QWEN35_2B_EVAL_CONFIG_20260718.yaml",
        "execution_log": ROOT / "refine-logs/C6F_MS_CXR_PREOPEN_GEOMETRY_EXECUTION_LOG_20260718.md",
        "manifest": args.manifest,
        "geometry_rows": C6F_ROOT / "ms_cxr_postc5_geometry_rows.jsonl",
        "geometry_lock": C6F_ROOT / "ms_cxr_postc5_geometry_lock.json",
        "dataset_lock": C6F_ROOT / "ms_cxr_postc5_dataset_lock.json",
    }
    source_paths = [
        ROOT / "bives_cxr/c6h_ms_cxr_eval.py",
        ROOT / "bives_cxr/c6_ms_cxr_eval.py",
        ROOT / "bives_cxr/c6g_geometry.py",
        ROOT / "bives_cxr/pixel_interventions.py",
        ROOT / "bives_cxr/polarity_runtime.py",
        ROOT / "scripts/evaluate_bives_c6_ms_cxr.py",
        ROOT / "scripts/prepare_bives_c6h_ms_cxr_evaluation.py",
        ROOT / "scripts/evaluate_bives_c6h_ms_cxr.py",
    ]
    artifact_paths = [
        ROOT / "local_runs/bives_cxr/qwen35_2b_sc_b2_sparse_k16_seed17/best.pt",
        ROOT / "local_runs/bives_cxr/qwen35_2b_weak_sc_cache/cache_lock.json",
        ROOT / "configs/bives_cxr/qwen35_2b_sc_b2_sparse_k16.yaml",
    ]
    for path in [
        args.authority,
        args.config,
        args.manifest,
        args.strict_intake,
        args.c6g_lock,
        args.c6g_rows,
        args.c6g_certificates,
        *c6f_paths.values(),
        *source_paths,
        *artifact_paths,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
    lock = build_c6h_lock(
        source_commit=source_commit,
        authority_path=args.authority,
        config_path=args.config,
        manifest_path=args.manifest,
        strict_intake_path=args.strict_intake,
        c6g_lock_path=args.c6g_lock,
        c6g_rows_path=args.c6g_rows,
        c6g_certificates_path=args.c6g_certificates,
        c6g_mask_dir=args.c6g_mask_dir,
        c6f_paths=c6f_paths,
        source_paths=source_paths,
        artifact_paths=artifact_paths,
    )
    output_path = args.output_dir / "c6h_preopen_lock.json"
    if output_path.exists():
        raise ValueError("C6H pre-open lock already exists; use a fresh output directory")
    write_json(output_path, lock)
    print(
        json.dumps(
            {
                "status": "C6H_PREOPEN_LOCK_PASS",
                "output": str(output_path),
                "source_commit": source_commit,
                "canonical_artifact_sha256": lock["canonical_artifact_sha256"],
                "rows": lock["counts"]["rows"],
                "model_loaded": False,
                "gpu_used": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
