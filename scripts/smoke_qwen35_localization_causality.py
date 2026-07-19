"""Run a real-Qwen3.5, synthetic-image localization-causality gate locally."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bives_cxr.localization_causality import (
    build_precomputed_audit_row,
    build_target_specific_controls,
    intervention_strength_metrics,
)
from bives_cxr.pixel_interventions import (
    replace_with_local_ring_mean,
    replace_with_masked_gaussian_blur,
)
from bives_cxr.qwen35_localization_audit import (
    QWEN35_2B_SNAPSHOT_SHA256,
    Qwen35StatementScorer,
    deterministic_top_cell_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def synthetic_case(size: int = 448) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    image = Image.new("RGB", (size, size), (24, 24, 24))
    draw = ImageDraw.Draw(image)
    draw.ellipse((56, 38, 214, 410), fill=(64, 64, 64), outline=(170, 170, 170), width=4)
    draw.ellipse((234, 38, 392, 410), fill=(68, 68, 68), outline=(170, 170, 170), width=4)
    draw.rectangle((252, 275, 378, 392), fill=(118, 118, 118))
    expert = np.zeros((size, size), dtype=bool)
    expert[275:392, 252:378] = True
    content = np.ones((size, size), dtype=bool)
    return image, expert, content


def occlusion_map(
    scorer: Qwen35StatementScorer,
    image: Image.Image,
    statement: str,
    *,
    grid_size: int,
    content_mask: np.ndarray,
) -> tuple[float, np.ndarray]:
    if grid_size < 2:
        raise ValueError("occlusion grid must be at least 2x2")
    height, width = content_mask.shape
    y_edges = np.linspace(0, height, grid_size + 1, dtype=np.int64)
    x_edges = np.linspace(0, width, grid_size + 1, dtype=np.int64)
    perturbed: list[Image.Image] = []
    for row in range(grid_size):
        for column in range(grid_size):
            mask = np.zeros_like(content_mask)
            mask[y_edges[row] : y_edges[row + 1], x_edges[column] : x_edges[column + 1]] = True
            perturbed.append(
                replace_with_local_ring_mean(image, mask, content_mask, ring_width=8)
            )
    values = scorer.score([image, *perturbed], statement)
    original = values[0]
    sensitivity = original - np.asarray(values[1:], dtype=np.float64)
    return original, sensitivity.reshape(grid_size, grid_size)


def operator_images(
    image: Image.Image,
    masks: dict[str, np.ndarray],
    content: np.ndarray,
    operator: Callable[[Image.Image, np.ndarray, np.ndarray], Image.Image],
) -> dict[str, Image.Image]:
    return {role: operator(image, mask, content) for role, mask in masks.items()}


def main() -> None:
    args = parse_args()
    if args.grid_size != 4:
        raise ValueError("the frozen development gate requires a 4x4 occlusion grid")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scorer = Qwen35StatementScorer(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        attention_implementation="eager",
    )
    image, expert, content = synthetic_case()
    statement = "A right pleural effusion is present."
    original_score, sensitivity = occlusion_map(
        scorer,
        image,
        statement,
        grid_size=args.grid_size,
        content_mask=content,
    )
    explanation = deterministic_top_cell_mask(
        sensitivity,
        image_height=image.height,
        image_width=image.width,
    )
    controls, certificates = build_target_specific_controls(
        expert,
        explanation,
        content_mask=content,
        seed_key="qwen35-synthetic-development-right-effusion",
    )
    masks = {"X": expert, "C_X": controls["C_X"], "E": explanation, "C_E": controls["C_E"]}
    operators: dict[str, Callable[[Image.Image, np.ndarray, np.ndarray], Image.Image]] = {
        "local_mean_ring8": lambda source, mask, valid: replace_with_local_ring_mean(
            source, mask, valid, ring_width=8
        ),
        "masked_gaussian_blur_sigma8": lambda source, mask, valid: replace_with_masked_gaussian_blur(
            source, mask, valid, sigma=8.0, truncate=3.0
        ),
    }
    thresholds = {
        "max_normalized_centroid_distance": 1.0,
        "max_log_perimeter_ratio": 1.0,
        "max_masked_l1_difference": 1.0,
        "max_masked_rms_difference": 1.0,
        "max_ssim_difference": 1.0,
        "max_edge_difference": 1.0,
    }
    rows: list[dict[str, Any]] = []
    original_array = np.asarray(image.convert("RGB"))
    explanation_map = np.kron(sensitivity, np.ones((112, 112), dtype=np.float64))
    for operator_id, operator in operators.items():
        variants = operator_images(image, masks, content, operator)
        variant_scores = scorer.score(
            [variants[role] for role in ("X", "C_X", "E", "C_E")],
            statement,
        )
        scores = {
            "s0": original_score,
            **dict(zip(("sX", "sCX", "sE", "sCE"), variant_scores, strict=True)),
        }
        strength = {
            role: intervention_strength_metrics(
                original_array,
                np.asarray(variants[role]),
                intervention_mask=masks[role],
                content_mask=content,
            )
            for role in masks
        }
        rows.append(
            build_precomputed_audit_row(
                identity={
                    "row_id": f"synthetic-qwen35-right-effusion-{operator_id}",
                    "dataset_id": "synthetic_qwen35_development_v1",
                    "dataset_role": "synthetic_development",
                    "patient_id": "synthetic-patient-001",
                    "image_id": "synthetic-cxr-001",
                    "pathology_id": "pleural_effusion_right",
                    "model_id": f"qwen35-2b-{QWEN35_2B_SNAPSHOT_SHA256[:12]}",
                    "explanation_id": "occlusion_local_mean_4x4_top1",
                    "operator_id": operator_id,
                },
                scores=scores,
                expert_mask=expert,
                explanation_mask=explanation,
                expert_control_mask=controls["C_X"],
                explanation_control_mask=controls["C_E"],
                content_mask=content,
                strength_metrics=strength,
                strength_thresholds=thresholds,
                explanation_map=explanation_map,
            )
        )
    result = {
        "schema_version": "cxr_localization_qwen35_synthetic_gate_v1",
        "formal_result": False,
        "test_opened": False,
        "real_model_loaded": True,
        "real_patient_data_used": False,
        "device": args.device,
        "dtype": args.dtype,
        "model": scorer.identity(),
        "explanation": {
            "method": "occlusion_local_mean",
            "grid": [4, 4],
            "region": "stable_row_major_top1_cell",
            "sensitivity": sensitivity.tolist(),
        },
        "control_certificates": certificates,
        "rows": rows,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(scorer.device)),
    }
    result["canonical_sha256"] = canonical_sha256(result)
    output_path = args.output_dir / "qwen35_synthetic_gate.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "pass", "output": str(output_path), **result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
