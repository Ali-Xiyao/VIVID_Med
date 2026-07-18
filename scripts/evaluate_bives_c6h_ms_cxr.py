"""Run the authorized one-time local Qwen3.5-2B C6H MS-CXR evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bives_cxr.backbones import Qwen35VisionAdapter, load_qwen35_visual_and_processor  # noqa: E402
from bives_cxr.c6_ms_cxr_eval import (  # noqa: E402
    FINDINGS,
    IMAGE_SIZE,
    OPERATORS,
    evaluate_survival_gate,
    summarize_operator,
    validate_ms_cxr_manifest,
)
from bives_cxr.c6h_ms_cxr_eval import validate_c6h_lock, validate_c6h_protocol  # noqa: E402
from bives_cxr.pixel_interventions import (  # noqa: E402
    deterministic_random_mask,
    patch_gate_to_pixel_mask,
    replace_with_local_ring_mean,
    replace_with_masked_gaussian_blur,
)
from bives_cxr.polarity_runtime import (  # noqa: E402
    load_locked_polarity_checkpoint,
)
from bives_cxr.provenance import canonical_json_sha256, file_sha256  # noqa: E402
from scripts.evaluate_bives_c6_ms_cxr import (  # noqa: E402
    read_jsonl,
    score_original,
    score_variants,
    snapshot_model_files,
    write_json,
    write_jsonl,
)


FORMAT_VERSION = "bives_c6h_ms_cxr_evaluation_v1"
C6F_ROOT = ROOT / "local_runs" / "bives_cxr" / "c6_ms_cxr_postc5"
C6G_ROOT = ROOT / "local_runs" / "bives_cxr" / "c6g_ms_cxr_geometry_final"
DEFAULT_ROOT = ROOT / "local_runs" / "bives_cxr" / "c6h_ms_cxr_one_time"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "refine-logs" / "C6H_MS_CXR_QWEN35_2B_EVAL_CONFIG_20260718.yaml",
    )
    parser.add_argument(
        "--authority",
        type=Path,
        default=ROOT
        / "refine-logs"
        / "C6H_MS_CXR_ONE_TIME_EVALUATION_AUTHORITY_20260718.md",
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
    parser.add_argument("--preopen-lock", type=Path, default=DEFAULT_ROOT / "c6h_preopen_lock.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "evaluation")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def committed_code_identity() -> str:
    for command in (
        ["git", "diff", "--quiet", "HEAD", "--"],
        ["git", "diff", "--cached", "--quiet"],
    ):
        if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
            raise ValueError("C6H requires every tracked change committed before opening")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_c6h_protocol(config)
    rows = read_jsonl(args.manifest)
    validate_ms_cxr_manifest(rows)
    geometry_rows = read_jsonl(args.c6g_rows)
    geometry = {str(row["sample_id"]): row for row in geometry_rows}
    lock = validate_c6h_lock(
        args.preopen_lock,
        authority_path=args.authority,
        config_path=args.config,
        manifest_path=args.manifest,
        strict_intake_path=args.strict_intake,
        c6g_lock_path=args.c6g_lock,
        c6g_rows_path=args.c6g_rows,
        c6g_certificates_path=args.c6g_certificates,
        c6g_mask_dir=args.c6g_mask_dir,
    )
    git_commit = committed_code_identity()
    if git_commit != lock["source_commit"]:
        raise ValueError("C6H committed source differs from pre-open lock")
    checkpoint_path = ROOT / config["checkpoint"]["path"]
    cache_lock_path = ROOT / config["training_cache_lock"]["path"]
    training_config_path = ROOT / config["training_config"]["path"]
    model_path = Path(config["model"]["path"])
    for path, expected in (
        (checkpoint_path, config["checkpoint"]["sha256"]),
        (cache_lock_path, config["training_cache_lock"]["sha256"]),
        (training_config_path, config["training_config"]["sha256"]),
    ):
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"C6H frozen artifact mismatch: {path}")
    if snapshot_model_files(model_path) != config["model"]["snapshot_sha256"]:
        raise ValueError("C6H Qwen3.5-2B snapshot changed")
    device = torch.device(args.device or config["model"]["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("C6H requires an available local CUDA device")
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if free_bytes < 20 * 1024**3:
        raise ValueError(f"C6H requires at least 20 GiB free on {device}")

    identity = {
        "format_version": FORMAT_VERSION,
        "git_commit": git_commit,
        "config_sha256": file_sha256(args.config),
        "authority_sha256": file_sha256(args.authority),
        "preopen_lock_sha256": file_sha256(args.preopen_lock),
        "preopen_lock_canonical_sha256": lock["canonical_artifact_sha256"],
        "c6g_lock_canonical_sha256": lock["c6g_lock_canonical_sha256"],
        "model_snapshot_sha256": config["model"]["snapshot_sha256"],
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "dtype": config["model"]["dtype"],
        "rows": 29,
        "model_evaluation_authorized": True,
        "one_time_execution_authorized": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    marker_path = args.output_dir / "EVALUATION_OPENED.json"
    metrics_path = args.output_dir / "metrics_final.json"
    progress_path = args.output_dir / "progress.json"
    if metrics_path.exists():
        raise ValueError("C6H is already complete and cannot be rerun")
    if marker_path.exists():
        if json.loads(marker_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("existing C6H opening marker identity differs")
    else:
        write_json(marker_path, identity)

    completed: dict[str, dict[str, Any]] = {}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != identity:
            raise ValueError("existing C6H progress identity differs")
        completed = dict(progress.get("completed", {}))

    torch.manual_seed(20260718)
    torch.cuda.manual_seed_all(20260718)
    torch.use_deterministic_algorithms(True)
    polarity_model, statement_to_index, checkpoint = load_locked_polarity_checkpoint(
        checkpoint_path, device
    )
    if (
        checkpoint.get("variant") != "B2_sparse_exact_k"
        or int(checkpoint.get("step", -1)) != 450
        or checkpoint.get("cache_lock_sha256") != config["training_cache_lock"]["sha256"]
        or statement_to_index != {"consolidation": 0, "pleural_effusion": 1}
    ):
        raise ValueError("C6H checkpoint semantic identity changed")
    visual, processor, qwen_config = load_qwen35_visual_and_processor(
        model_path,
        dtype=config["model"]["dtype"],
        attention_implementation="eager",
    )
    visual = visual.to(device).eval()
    adapter = Qwen35VisionAdapter(
        visual,
        spatial_merge_size=int(qwen_config["vision_config"]["spatial_merge_size"]),
    ).to(device).eval()
    score_kwargs = {
        "polarity_model": polarity_model,
        "statement_to_index": statement_to_index,
        "adapter": adapter,
        "visual": visual,
        "processor": processor,
        "device": device,
    }

    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        sample_id = str(row["sample_id"])
        if sample_id in completed:
            continue
        image_path = Path(str(row["image_path"]))
        if file_sha256(image_path) != row["official_image_sha256"]:
            raise ValueError(f"C6H image hash mismatch: {sample_id}")
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        if image.size != (int(row["native_columns"]), int(row["native_rows"])):
            raise ValueError(f"C6H JPG/native geometry mismatch: {sample_id}")
        original, extracted = score_original(image, row, **score_kwargs)
        geometry_row = geometry[sample_id]
        mask_path = args.c6g_mask_dir / str(geometry_row["mask_file"])
        if file_sha256(mask_path) != geometry_row["mask_sha256"]:
            raise ValueError(f"C6H geometry mask changed: {sample_id}")
        with np.load(mask_path, allow_pickle=False) as payload:
            target = payload["target_mask"].astype(bool)
            control = payload["control_mask"].astype(bool)
            content = payload["content_mask"].astype(bool)
        letterboxed = extracted["letterboxed_image"]
        mean_target, mean_control, blur_target, blur_control = score_variants(
            [
                replace_with_local_ring_mean(letterboxed, target, content),
                replace_with_local_ring_mean(letterboxed, control, content),
                replace_with_masked_gaussian_blur(letterboxed, target, content),
                replace_with_masked_gaussian_blur(letterboxed, control, content),
            ],
            row,
            **score_kwargs,
        )
        evidence_mask = patch_gate_to_pixel_mask(
            extracted["gate"], extracted["grid_hw"], IMAGE_SIZE
        ) & content
        random_mask = deterministic_random_mask(
            int(evidence_mask.sum()),
            content,
            seed_key=f"{sample_id}::c6h-localization::{config['evaluation']['bootstrap_seed']}",
        )
        target_area = int(target.sum())
        completed[sample_id] = {
            "sample_id": sample_id,
            "unit_id": row["unit_id"],
            "canonical_statement_id": row["canonical_statement_id"],
            "source_split": "publisher_test",
            "positive_only": True,
            "box_area_quartile": int(geometry_row["box_area_quartile"]),
            "target_area_pixels": target_area,
            "control_area_pixels": int(control.sum()),
            "original": original,
            "topk_target_coverage": float((evidence_mask & target).sum() / target_area),
            "random_target_coverage": float((random_mask & target).sum() / target_area),
            "local_mean": {
                "target_score": mean_target,
                "control_score": mean_control,
                "target_effect": float(original["support_probability"] - mean_target),
                "control_effect": float(original["support_probability"] - mean_control),
                "tcig": float(mean_control - mean_target),
            },
            "masked_gaussian_blur": {
                "target_score": blur_target,
                "control_score": blur_control,
                "target_effect": float(original["support_probability"] - blur_target),
                "control_effect": float(original["support_probability"] - blur_control),
                "tcig": float(blur_control - blur_target),
            },
        }
        completed[sample_id]["topk_localization_gain"] = (
            completed[sample_id]["topk_target_coverage"]
            - completed[sample_id]["random_target_coverage"]
        )
        write_json(
            progress_path,
            {
                "identity": identity,
                "status": "in_progress",
                "completed_count": len(completed),
                "total": len(rows),
                "completed": completed,
            },
        )
        print(f"C6H_SCORE {index}/{len(rows)}", flush=True)

    final_rows = [completed[key] for key in sorted(completed)]
    if len(final_rows) != 29:
        raise ValueError("C6H result table is incomplete")
    operator_results = {
        operator: summarize_operator(
            final_rows,
            operator,
            bootstrap_replicates=int(config["evaluation"]["bootstrap_replicates"]),
            bootstrap_seed=int(config["evaluation"]["bootstrap_seed"]),
        )
        for operator in OPERATORS
    }
    gate = evaluate_survival_gate(operator_results)
    result = {
        "format_version": FORMAT_VERSION,
        "status": "pass_independent_external_mechanism" if gate["pass"] else "fail_final_stop",
        "formal_result": False,
        "independent_external_result": True,
        "clinical_validation": False,
        "positive_only": True,
        "classification_metrics_computed": False,
        "training_performed": False,
        "model_evaluation_authorized": True,
        "one_time_execution_authorized": True,
        "identity": identity,
        "rows": len(final_rows),
        "patients": len({row["unit_id"] for row in final_rows}),
        "per_finding_counts": {
            finding: sum(row["canonical_statement_id"] == finding for row in final_rows)
            for finding in FINDINGS
        },
        "operators": operator_results,
        "survival_gate": gate,
        "runtime_seconds": float(time.perf_counter() - started),
        "route_decision": "terminal_stop_after_c6h_pass_or_fail",
        "scale_decision": "qwen35_4b_9b_not_authorized",
    }
    rows_path = args.output_dir / "evaluation_rows.jsonl"
    write_jsonl(rows_path, final_rows)
    result["evaluation_rows_sha256"] = file_sha256(rows_path)
    result["canonical_artifact_sha256"] = canonical_json_sha256(result)
    write_json(metrics_path, result)
    write_json(
        progress_path,
        {"identity": identity, "status": "complete", "completed_count": 29, "total": 29},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
