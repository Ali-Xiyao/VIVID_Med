"""Run the one-time local Qwen3.5-2B C6F mechanism evaluation on MS-CXR JPGs."""

from __future__ import annotations

import argparse
import hashlib
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
    validate_dataset_lock,
    validate_ms_cxr_manifest,
)
from bives_cxr.pixel_interventions import (  # noqa: E402
    LOCAL_MEAN_RING_WIDTH,
    MASKED_GAUSSIAN_SIGMA,
    MASKED_GAUSSIAN_TRUNCATE,
    deterministic_random_mask,
    patch_gate_to_pixel_mask,
    replace_with_local_ring_mean,
    replace_with_masked_gaussian_blur,
)
from bives_cxr.polarity_runtime import (  # noqa: E402
    extract_qwen35_patch_batch,
    extract_qwen35_patches,
    load_locked_polarity_checkpoint,
    score_statements,
)
from bives_cxr.provenance import canonical_json_sha256, file_sha256  # noqa: E402
from bives_cxr.rescue_protocol import COORDINATE_ZONE_CONTROL_VERSION  # noqa: E402


FORMAT_VERSION = "bives_c6f_ms_cxr_evaluation_v1"
DEFAULT_ROOT = ROOT / "local_runs" / "bives_cxr" / "c6_ms_cxr_postc5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "refine-logs" / "C6F_MS_CXR_QWEN35_2B_EVAL_CONFIG_20260718.yaml",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "ms_cxr_postc5_manifest.jsonl")
    parser.add_argument("--dataset-lock", type=Path, default=DEFAULT_ROOT / "ms_cxr_postc5_dataset_lock.json")
    parser.add_argument("--geometry-rows", type=Path, default=DEFAULT_ROOT / "ms_cxr_postc5_geometry_rows.jsonl")
    parser.add_argument("--mask-dir", type=Path, default=DEFAULT_ROOT / "geometry_masks")
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "evaluation")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def snapshot_model_files(model_root: Path) -> str:
    names = {"config.json", "model.safetensors.index.json"}
    names.update(path.name for path in model_root.glob("*.safetensors"))
    files = {
        name: file_sha256(model_root / name)
        for name in sorted(names)
        if (model_root / name).is_file()
    }
    return canonical_json_sha256(files)


def committed_code_identity() -> str:
    for args in (
        ["git", "diff", "--quiet", "HEAD", "--"],
        ["git", "diff", "--cached", "--quiet"],
    ):
        completed = subprocess.run(args, cwd=ROOT, check=False)
        if completed.returncode != 0:
            raise ValueError("C6F requires every tracked code change committed before opening")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def score_original(
    image: Image.Image,
    row: dict[str, Any],
    *,
    polarity_model: torch.nn.Module,
    statement_to_index: dict[str, int],
    adapter: torch.nn.Module,
    visual: torch.nn.Module,
    processor: Any,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    finding = str(row["canonical_statement_id"])
    extracted = extract_qwen35_patches(
        image,
        str(row["statement_text"]),
        adapter=adapter,
        visual=visual,
        processor=processor,
        device=device,
        image_size=IMAGE_SIZE,
    )
    output = score_statements(
        polarity_model,
        extracted["patch_tokens"],
        extracted["valid_mask"],
        [statement_to_index[finding]],
    )
    score = {
        "support_probability": float(output["support_probability"][0].cpu()),
        "signed_evidence": float(output["signed_evidence"][0].cpu()),
        "topk_indices": torch.where(output["gate"][0].detach().cpu() > 0.5)[0].tolist(),
    }
    return score, {**extracted, "gate": output["gate"][0].detach().cpu() > 0.5}


def score_variants(
    images: list[Image.Image],
    row: dict[str, Any],
    *,
    polarity_model: torch.nn.Module,
    statement_to_index: dict[str, int],
    adapter: torch.nn.Module,
    visual: torch.nn.Module,
    processor: Any,
    device: torch.device,
) -> list[float]:
    finding = str(row["canonical_statement_id"])
    extracted_batch = extract_qwen35_patch_batch(
        images,
        [str(row["statement_text"])] * len(images),
        adapter=adapter,
        visual=visual,
        processor=processor,
        device=device,
        image_size=IMAGE_SIZE,
    )
    result = []
    for extracted in extracted_batch:
        output = score_statements(
            polarity_model,
            extracted["patch_tokens"],
            extracted["valid_mask"],
            [statement_to_index[finding]],
        )
        result.append(float(output["support_probability"][0].cpu()))
    return result


def validate_protocol(config: dict[str, Any]) -> None:
    if config["model"]["family"] != "Qwen3.5" or config["model"]["scale"] != "2B":
        raise ValueError("C6F authorizes only Qwen3.5-2B")
    if config["checkpoint"]["variant"] != "B2_sparse_exact_k" or int(config["checkpoint"]["topk"]) != 16:
        raise ValueError("C6F requires frozen B2 exact-K=16")
    intervention = config["intervention"]
    observed = (
        int(intervention["image_size"]),
        str(intervention["control_version"]),
        int(intervention["local_mean_ring_width"]),
        float(intervention["masked_gaussian_sigma"]),
        float(intervention["masked_gaussian_truncate"]),
        float(intervention["dilation_fraction"]),
    )
    expected = (
        IMAGE_SIZE,
        COORDINATE_ZONE_CONTROL_VERSION,
        LOCAL_MEAN_RING_WIDTH,
        MASKED_GAUSSIAN_SIGMA,
        MASKED_GAUSSIAN_TRUNCATE,
        0.0,
    )
    if observed != expected or config["evaluation"].get("allow_classification_metrics") is not False:
        raise ValueError("C6F frozen intervention/claim boundary changed")
    if config["scale"].get("qwen35_4b_authorized") or config["scale"].get("qwen35_9b_authorized"):
        raise ValueError("C6F scale lock changed")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_protocol(config)
    rows = read_jsonl(args.manifest)
    validate_ms_cxr_manifest(rows)
    geometry_rows = read_jsonl(args.geometry_rows)
    geometry = {str(row["sample_id"]): row for row in geometry_rows}
    if len(geometry) != 29 or any(not row.get("feasible") for row in geometry_rows):
        raise ValueError("C6F requires 29/29 score-free geometry rows")
    lock = validate_dataset_lock(
        args.dataset_lock,
        manifest_path=args.manifest,
        geometry_rows_path=args.geometry_rows,
        strict_intake_path=args.strict_intake,
        authority_path=args.authority,
        config_path=args.config,
    )
    for path_string, expected in {**lock["source_sha256"], **lock["release_sha256"]}.items():
        path = Path(path_string)
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"C6F locked source/release changed: {path}")

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
            raise ValueError(f"C6F frozen artifact mismatch: {path}")
    if snapshot_model_files(model_path) != config["model"]["snapshot_sha256"]:
        raise ValueError("C6F Qwen3.5-2B snapshot changed")
    git_commit = committed_code_identity()
    device = torch.device(args.device or config["model"]["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("C6F requires an available local CUDA device")

    identity = {
        "format_version": FORMAT_VERSION,
        "git_commit": git_commit,
        "config_sha256": file_sha256(args.config),
        "authority_sha256": file_sha256(args.authority),
        "dataset_lock_sha256": file_sha256(args.dataset_lock),
        "dataset_lock_canonical_sha256": lock["canonical_artifact_sha256"],
        "model_snapshot_sha256": config["model"]["snapshot_sha256"],
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "device": str(device),
        "dtype": config["model"]["dtype"],
        "rows": 29,
        "model_evaluation_authorized": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    marker_path = args.output_dir / "EVALUATION_OPENED.json"
    metrics_path = args.output_dir / "metrics_final.json"
    progress_path = args.output_dir / "progress.json"
    if metrics_path.exists():
        raise ValueError("C6F evaluation is already complete and cannot be rerun")
    if marker_path.exists():
        if json.loads(marker_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("existing C6F opening marker identity differs")
    else:
        write_json(marker_path, identity)

    completed: dict[str, dict[str, Any]] = {}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != identity:
            raise ValueError("existing C6F progress identity differs")
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
        raise ValueError("C6F checkpoint semantic identity changed")
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
            raise ValueError(f"C6F image hash mismatch: {sample_id}")
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        if image.size != (int(row["native_columns"]), int(row["native_rows"])):
            raise ValueError(f"C6F JPG/native geometry mismatch: {sample_id}")
        original, extracted = score_original(image, row, **score_kwargs)
        geometry_row = geometry[sample_id]
        mask_path = args.mask_dir / str(geometry_row["mask_file"])
        if file_sha256(mask_path) != geometry_row["mask_sha256"]:
            raise ValueError(f"C6F geometry mask changed: {sample_id}")
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
            seed_key=f"{sample_id}::c6f-localization::{config['evaluation']['bootstrap_seed']}",
        )
        target_area = int(target.sum())
        target_coverage = float((evidence_mask & target).sum() / target_area)
        random_coverage = float((random_mask & target).sum() / target_area)
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
            "topk_target_coverage": target_coverage,
            "random_target_coverage": random_coverage,
            "topk_localization_gain": target_coverage - random_coverage,
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
        print(f"C6F_SCORE {index}/{len(rows)}", flush=True)

    final_rows = [completed[key] for key in sorted(completed)]
    if len(final_rows) != 29:
        raise ValueError("C6F result table is incomplete")
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
        "status": (
            "pass_independent_external_mechanism"
            if gate["pass"]
            else "fail_final_stop"
        ),
        "formal_result": False,
        "independent_external_result": True,
        "clinical_validation": False,
        "positive_only": True,
        "classification_metrics_computed": False,
        "training_performed": False,
        "model_evaluation_authorized": True,
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
        "scale_decision": "qwen35_4b_9b_remain_blocked_pending_new_user_authority",
    }
    rows_path = args.output_dir / "evaluation_rows.jsonl"
    write_jsonl(rows_path, final_rows)
    result["evaluation_rows_sha256"] = file_sha256(rows_path)
    result["canonical_artifact_sha256"] = canonical_json_sha256(result)
    write_json(metrics_path, result)
    write_json(
        progress_path,
        {
            "identity": identity,
            "status": "complete",
            "completed_count": 29,
            "total": 29,
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
