"""Evaluate locked B2 target/control/evidence-only VinDr pixel interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bives_cxr.backbones import Qwen35VisionAdapter, load_qwen35_visual_and_processor  # noqa: E402
from bives_cxr.dicom import load_cxr_dicom  # noqa: E402
from bives_cxr.expert_sc import file_sha256, read_expert_sc_manifest  # noqa: E402
from bives_cxr.pixel_interventions import (  # noqa: E402
    delete_pixels,
    deterministic_disjoint_control_mask,
    deterministic_random_mask,
    paired_intervention_metrics,
    patch_gate_to_pixel_mask,
    retain_pixels,
    transform_mask_to_letterbox,
    union_box_mask,
)
from bives_cxr.polarity_runtime import (  # noqa: E402
    extract_qwen35_patch_batch,
    extract_qwen35_patches,
    load_locked_polarity_checkpoint,
    score_statements,
)
from scripts.evaluate_bives_vindr_sc import require_integrity_gate, snapshot_model_files  # noqa: E402


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_dilations(value: str) -> list[float]:
    result = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not result or result[0] < 0 or result[-1] > 0.25:
        raise ValueError("dilations must be non-empty fractions in [0, 0.25]")
    return result


def audit_control_geometry(
    rows: list[dict[str, Any]],
    dilations: list[float],
    image_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Freeze an outcome-independent exact-area-control feasibility cohort."""

    eligible = []
    excluded = []
    for row in rows:
        metadata = pydicom.dcmread(
            row["image_path"],
            stop_before_pixels=True,
            specific_tags=["Rows", "Columns"],
        )
        width = int(metadata.Columns)
        height = int(metadata.Rows)
        scale = min(image_size / width, image_size / height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        left = (image_size - resized_width) // 2
        top = (image_size - resized_height) // 2
        content_box = (
            left,
            top,
            left + resized_width,
            top + resized_height,
        )
        content_mask = np.zeros((image_size, image_size), dtype=bool)
        content_mask[top : top + resized_height, left : left + resized_width] = True
        checks = []
        for dilation in dilations:
            target = transform_mask_to_letterbox(
                union_box_mask(
                    width,
                    height,
                    row["bounding_boxes"],
                    dilation_fraction=dilation,
                ),
                content_box,
                image_size,
            ) & content_mask
            target_area = int(target.sum())
            disjoint_area = int((content_mask & ~target).sum())
            checks.append(
                {
                    "dilation_fraction": dilation,
                    "target_area_pixels": target_area,
                    "disjoint_content_pixels": disjoint_area,
                    "feasible": disjoint_area >= target_area,
                }
            )
        if all(check["feasible"] for check in checks):
            eligible.append(row)
        else:
            excluded.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "canonical_statement_id": str(row["canonical_statement_id"]),
                    "reason": "target_union_exceeds_disjoint_exact_area_capacity",
                    "checks": checks,
                }
            )
    return eligible, excluded


@torch.no_grad()
def score_image(
    image,
    statement_text: str,
    statement_index: int,
    *,
    polarity_model,
    adapter,
    visual,
    processor,
    device: torch.device,
    image_size: int,
) -> tuple[float, dict[str, Any], dict[str, torch.Tensor]]:
    extracted = extract_qwen35_patches(
        image,
        statement_text,
        adapter=adapter,
        visual=visual,
        processor=processor,
        device=device,
        image_size=image_size,
    )
    output = score_statements(
        polarity_model,
        extracted["patch_tokens"],
        extracted["valid_mask"],
        [statement_index],
    )
    return float(output["support_probability"][0].cpu()), extracted, output


@torch.no_grad()
def score_image_batch(
    images,
    statement_text: str,
    statement_index: int,
    *,
    polarity_model,
    adapter,
    visual,
    processor,
    device: torch.device,
    image_size: int,
) -> list[float]:
    """Score intervention variants in one visual-tower forward."""

    extracted_batch = extract_qwen35_patch_batch(
        images,
        [statement_text] * len(images),
        adapter=adapter,
        visual=visual,
        processor=processor,
        device=device,
        image_size=image_size,
    )
    scores = []
    for extracted in extracted_batch:
        output = score_statements(
            polarity_model,
            extracted["patch_tokens"],
            extracted["valid_mask"],
            [statement_index],
        )
        scores.append(float(output["support_probability"][0].cpu()))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--integrity-audit", type=Path, required=True)
    parser.add_argument("--expert-sc-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-cache-lock", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--dilations", default="0,0.1")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    parser.add_argument("--original-score-atol", type=float, default=1e-6)
    args = parser.parse_args()

    if not 0.0 < args.original_score_atol <= 1e-2:
        raise ValueError("original-score-atol must be in (0, 1e-2]")

    require_integrity_gate(args.integrity_audit)
    expert_metrics_path = args.expert_sc_dir / "metrics_final.json"
    expert_predictions_path = args.expert_sc_dir / "predictions.jsonl"
    expert_metrics = json.loads(expert_metrics_path.read_text(encoding="utf-8"))
    if not expert_metrics.get("evaluation_only") or expert_metrics.get("external_test_used_for_selection"):
        raise ValueError("expert S/C prerequisite must be evaluation-only and selection-free")
    expert_predictions = {
        row["sample_id"]: row
        for row in (
            json.loads(line)
            for line in expert_predictions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    training_lock = json.loads(args.training_cache_lock.read_text(encoding="utf-8"))
    if snapshot_model_files(args.model_path) != training_lock.get("model_snapshot_sha256"):
        raise ValueError("Qwen3.5 model snapshot differs from the training cache lock")
    device = torch.device(args.device)
    polarity_model, statement_to_index, checkpoint = load_locked_polarity_checkpoint(
        args.checkpoint, device
    )
    if checkpoint.get("variant") != "B2_sparse_exact_k":
        raise ValueError("intervention evaluation requires locked B2 sparse exact-K")
    if checkpoint.get("cache_lock_sha256") != file_sha256(args.training_cache_lock):
        raise ValueError("B2 checkpoint/training-cache binding failed")

    positive_rows = [row for row in read_expert_sc_manifest(args.manifest) if int(row["binary_label"]) == 1]
    if not positive_rows or any(not row.get("bounding_boxes") for row in positive_rows):
        raise ValueError("all positive expert rows must have target boxes")
    dilations = parse_dilations(args.dilations)
    positive_rows, geometry_excluded_rows = audit_control_geometry(
        positive_rows,
        dilations,
        args.image_size,
    )
    if not positive_rows:
        raise ValueError("no positive rows support exact-area disjoint controls")
    identity = {
        "manifest_sha256": file_sha256(args.manifest),
        "integrity_audit_sha256": file_sha256(args.integrity_audit),
        "expert_metrics_sha256": file_sha256(expert_metrics_path),
        "expert_predictions_sha256": file_sha256(expert_predictions_path),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "training_cache_lock_sha256": file_sha256(args.training_cache_lock),
        "model_snapshot_sha256": training_lock["model_snapshot_sha256"],
        "image_size": int(args.image_size),
        "dilations": dilations,
        "original_score_atol": float(args.original_score_atol),
        "control_geometry_version": "bives_exact_area_disjoint_all_dilations_v1",
        "geometry_excluded_rows": geometry_excluded_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = args.output_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    progress_path = args.output_dir / "progress.json"
    completed: dict[str, dict[str, Any]] = {}
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != identity:
            raise ValueError("existing intervention progress has a different identity")
        completed = dict(progress.get("completed", {}))

    visual, processor, qwen_config = load_qwen35_visual_and_processor(
        args.model_path, dtype=args.dtype, attention_implementation="eager"
    )
    visual = visual.to(device).eval()
    adapter = Qwen35VisionAdapter(
        visual,
        spatial_merge_size=int(qwen_config["vision_config"]["spatial_merge_size"]),
    ).to(device).eval()
    total_tasks = len(positive_rows) * len(dilations)
    for row_index, row in enumerate(sorted(positive_rows, key=lambda item: item["sample_id"]), start=1):
        sample_id = str(row["sample_id"])
        pending = [d for d in dilations if f"{sample_id}::d{d:g}" not in completed]
        if not pending:
            continue
        image_path = Path(row["image_path"])
        if not str(row.get("official_image_sha256") or ""):
            raise ValueError(f"VinDr official image SHA is missing: {sample_id}")
        image, _ = load_cxr_dicom(image_path)
        statement_id = str(row["canonical_statement_id"])
        original_score, extracted, original_output = score_image(
            image,
            str(row["statement_text"]),
            statement_to_index[statement_id],
            polarity_model=polarity_model,
            adapter=adapter,
            visual=visual,
            processor=processor,
            device=device,
            image_size=args.image_size,
        )
        expected_score = float(expert_predictions[sample_id]["support_probability"])
        original_score_abs_diff = abs(original_score - expected_score)
        if original_score_abs_diff > args.original_score_atol:
            raise ValueError(
                "original intervention score does not reproduce expert inference: "
                f"{sample_id}; single={original_score:.12g}; "
                f"expert_batch={expected_score:.12g}; "
                f"abs_diff={abs(original_score - expected_score):.12g}"
            )
        gate = original_output["gate"][0].detach().cpu() > 0.5
        evidence_mask = patch_gate_to_pixel_mask(gate, extracted["grid_hw"], args.image_size)
        content_mask = np.zeros((args.image_size, args.image_size), dtype=bool)
        left, top, right, bottom = extracted["content_box"]
        content_mask[top:bottom, left:right] = True
        evidence_mask &= content_mask
        for dilation in pending:
            task_key = f"{sample_id}::d{dilation:g}"
            native_target = union_box_mask(
                image.width,
                image.height,
                row["bounding_boxes"],
                dilation_fraction=dilation,
            )
            target_mask = transform_mask_to_letterbox(
                native_target, extracted["content_box"], args.image_size
            )
            control_mask = deterministic_disjoint_control_mask(
                target_mask,
                content_mask,
                seed_key=f"{task_key}::control::{args.bootstrap_seed}",
            )
            random_localization = deterministic_random_mask(
                int(evidence_mask.sum()),
                content_mask,
                seed_key=f"{task_key}::random-localization::{args.bootstrap_seed}",
            )
            letterboxed = extracted["letterboxed_image"]
            target_score, control_score, keep_score = score_image_batch(
                [
                    delete_pixels(letterboxed, target_mask),
                    delete_pixels(letterboxed, control_mask),
                    retain_pixels(letterboxed, evidence_mask),
                ],
                str(row["statement_text"]),
                statement_to_index[statement_id],
                polarity_model=polarity_model,
                adapter=adapter,
                visual=visual,
                processor=processor,
                device=device,
                image_size=args.image_size,
            )
            mask_name = hashlib.sha256(task_key.encode("utf-8")).hexdigest() + ".npz"
            mask_path = masks_dir / mask_name
            temporary = mask_path.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                target_mask=target_mask,
                control_mask=control_mask,
                evidence_mask=evidence_mask,
                random_localization_mask=random_localization,
            )
            os.replace(temporary, mask_path)
            target_area = int(target_mask.sum())
            completed[task_key] = {
                "sample_id": sample_id,
                "unit_id": str(row["unit_id"]),
                "canonical_statement_id": statement_id,
                "dilation_fraction": dilation,
                "original_score": original_score,
                "expert_batch_original_score": expected_score,
                "original_score_abs_diff": original_score_abs_diff,
                "target_drop_score": target_score,
                "control_drop_score": control_score,
                "keep_score": keep_score,
                "target_deletion_effect": original_score - target_score,
                "control_deletion_effect": original_score - control_score,
                "tcig": control_score - target_score,
                "target_area_pixels": target_area,
                "control_area_pixels": int(control_mask.sum()),
                "evidence_area_pixels": int(evidence_mask.sum()),
                "topk_target_coverage": float((evidence_mask & target_mask).sum() / target_area),
                "random_target_coverage": float((random_localization & target_mask).sum() / target_area),
                "topk_localization_gain": float(
                    ((evidence_mask & target_mask).sum() - (random_localization & target_mask).sum())
                    / target_area
                ),
                "mask_file": str(mask_path.relative_to(args.output_dir).as_posix()),
                "mask_file_sha256": file_sha256(mask_path),
            }
            write_json_atomic(
                progress_path,
                {
                    "identity": identity,
                    "status": "in_progress",
                    "completed_count": len(completed),
                    "total_tasks": total_tasks,
                    "completed": completed,
                },
            )
            print(json.dumps({"completed_tasks": len(completed), "total_tasks": total_tasks}))

    all_rows = [completed[key] for key in sorted(completed)]
    results_by_dilation = {}
    for dilation in dilations:
        subset = [row for row in all_rows if float(row["dilation_fraction"]) == dilation]
        results_by_dilation[f"{dilation:g}"] = paired_intervention_metrics(
            subset,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
    primary = results_by_dilation[f"{dilations[0]:g}"]
    per_finding_gate = {}
    for finding, point in primary["per_finding"].items():
        ci = primary["image_cluster_bootstrap_95ci"]["per_finding"][finding]
        per_finding_gate[finding] = {
            "mean_tcig": point["mean_tcig"],
            "tcig_ci_lower_gt_zero": ci["mean_tcig"]["lower"] > 0,
            "mean_topk_localization_gain": point.get("mean_topk_localization_gain"),
            "localization_gain_ci_lower_gt_zero": ci.get(
                "mean_topk_localization_gain", {"lower": float("-inf")}
            )["lower"]
            > 0,
        }
    result = {
        "formal_result": False,
        "evaluation_only": True,
        "external_test_used_for_selection": False,
        "checkpoint_step": int(checkpoint["step"]),
        "identity": identity,
        "positive_rows": len(positive_rows),
        "geometry_excluded_count": len(geometry_excluded_rows),
        "geometry_excluded_rows": geometry_excluded_rows,
        "max_original_score_abs_diff": max(
            float(row["original_score_abs_diff"]) for row in all_rows
        ),
        "results_by_dilation": results_by_dilation,
        "primary_dilation": dilations[0],
        "per_finding_gate": per_finding_gate,
        "target_control_and_localization_pass_all_findings": all(
            row["tcig_ci_lower_gt_zero"] and row["localization_gain_ci_lower_gt_zero"]
            for row in per_finding_gate.values()
        ),
    }
    rows_path = args.output_dir / "intervention_rows.jsonl"
    with rows_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    result["intervention_rows_sha256"] = file_sha256(rows_path)
    write_json_atomic(args.output_dir / "metrics_final.json", result)
    write_json_atomic(
        progress_path,
        {
            "identity": identity,
            "status": "complete",
            "completed_count": len(completed),
            "total_tasks": total_tasks,
            "completed": completed,
            "metrics_final": "metrics_final.json",
        },
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
