#!/usr/bin/env python
"""Run one patient-disjoint local GPU shard of CheXlocalize development."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bives_cxr.chexlocalize_validation import patient_disjoint_shard  # noqa: E402
from bives_cxr.localization_causality import (  # noqa: E402
    build_precomputed_audit_row,
    build_target_specific_controls,
    intervention_strength_metrics,
)
from bives_cxr.pixel_interventions import (  # noqa: E402
    replace_with_local_ring_mean,
    replace_with_masked_gaussian_blur,
)
from bives_cxr.provenance import canonical_json_sha256, file_sha256  # noqa: E402
from bives_cxr.qwen35_localization_audit import (  # noqa: E402
    QWEN35_2B_SNAPSHOT_SHA256,
    Qwen35StatementScorer,
    deterministic_top_cell_mask,
)
from bives_cxr.qwen35_preprocessing import letterbox_image  # noqa: E402


DEFAULT_INPUT = ROOT / "local_runs/cxr_localization_causality/chexlocalize_qwen35_development"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lock = json.loads((args.input_dir / "development_lock.json").read_text(encoding="utf-8"))
    manifest_path = args.input_dir / "development_manifest.jsonl"
    validate_lock(lock, manifest_path)
    all_rows = read_jsonl(manifest_path)
    rows = patient_disjoint_shard(all_rows, args.shard_index, args.shard_count)
    if not rows:
        raise ValueError("CheXlocalize development shard is empty")
    scorer = Qwen35StatementScorer(lock["model_path"], device=args.device, dtype=args.dtype, attention_implementation="eager")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    exclusions = []
    for source in rows:
        try:
            audit_rows.extend(evaluate_sample(source, scorer, args.input_dir))
        except ValueError as error:
            exclusions.append({"sample_hash": source["sample_id"], "reason": str(error)})
    rows_path = args.output_dir / "audit_rows.jsonl"
    write_jsonl(rows_path, audit_rows)
    result = {
        "schema_version": "chexlocalize-qwen35-development-shard-v1",
        "status": "complete_nonformal",
        "formal_result": False,
        "confirmatory_evidence": False,
        "test_opened": False,
        "patient_level_claim": True,
        "cluster_unit": "patient_id_hash",
        "device": args.device,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "input_lock_canonical_sha256": lock["canonical_sha256"],
        "model": scorer.identity(),
        "source_samples": len(rows),
        "source_patients": len({row["patient_id_hash"] for row in rows}),
        "audit_rows": len(audit_rows),
        "exclusions": exclusions,
        "rows_sha256": file_sha256(rows_path),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(scorer.device)),
    }
    result["canonical_sha256"] = canonical_json_sha256(result)
    write_json(args.output_dir / "shard_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def evaluate_sample(source: dict[str, Any], scorer: Qwen35StatementScorer, input_dir: Path) -> list[dict[str, Any]]:
    image_path = Path(str(source["image_path"]))
    if file_sha256(image_path) != source["official_image_sha256"]:
        raise ValueError("source JPEG hash changed")
    with Image.open(image_path) as loaded:
        image = loaded.convert("RGB")
    letterboxed, content_box = letterbox_image(image, 448)
    if list(content_box) != source["score_free_audit"]["content_box"]:
        raise ValueError("source JPEG preprocessing geometry changed")
    mask_path = input_dir / "expert_masks" / source["mask_file"]
    if file_sha256(mask_path) != source["mask_sha256"]:
        raise ValueError("score-free expert mask changed")
    with np.load(mask_path, allow_pickle=False) as payload:
        expert = payload["expert_mask"].astype(bool)
        content = payload["content_mask"].astype(bool)
    statement = str(source["statement_text"])
    original_score, sensitivity = occlusion_map(scorer, letterboxed, statement, grid_size=4, content_mask=content)
    explanation = deterministic_top_cell_mask(sensitivity, image_height=448, image_width=448) & content
    controls, _ = build_target_specific_controls(expert, explanation, content_mask=content, seed_key=f"chexlocalize-qwen35-development:{source['sample_id']}")
    masks = {"X": expert, "C_X": controls["C_X"], "E": explanation, "C_E": controls["C_E"]}
    operators: dict[str, Callable[[Image.Image, np.ndarray, np.ndarray], Image.Image]] = {
        "local_mean_ring8": lambda src, mask, valid: replace_with_local_ring_mean(src, mask, valid, ring_width=8),
        "masked_gaussian_blur_sigma8": lambda src, mask, valid: replace_with_masked_gaussian_blur(src, mask, valid, sigma=8.0, truncate=3.0),
    }
    thresholds = {key: 1.0 for key in ("max_normalized_centroid_distance", "max_log_perimeter_ratio", "max_masked_l1_difference", "max_masked_rms_difference", "max_ssim_difference", "max_edge_difference")}
    original_array = np.asarray(letterboxed.convert("RGB"))
    explanation_map = np.repeat(np.repeat(sensitivity, 112, axis=0), 112, axis=1)
    output = []
    for operator_id, operator in operators.items():
        variants = {role: operator(letterboxed, mask, content) for role, mask in masks.items()}
        variant_scores = scorer.score([variants[role] for role in ("X", "C_X", "E", "C_E")], statement)
        scores = {"s0": original_score, **dict(zip(("sX", "sCX", "sE", "sCE"), variant_scores, strict=True))}
        strength = {role: intervention_strength_metrics(original_array, np.asarray(variants[role]), intervention_mask=masks[role], content_mask=content) for role in masks}
        row = build_precomputed_audit_row(
            identity={"row_id": f"chexlocalize-qwen35-{source['sample_id'][:20]}-{operator_id}", "dataset_id": "chexlocalize_validation_development_v1_0", "dataset_role": "development", "patient_id": source["patient_id_hash"], "image_id": source["image_id_hash"], "pathology_id": source["canonical_statement_id"], "model_id": f"qwen35-2b-{QWEN35_2B_SNAPSHOT_SHA256[:12]}", "explanation_id": "occlusion_local_mean_4x4_top1", "operator_id": operator_id},
            scores=scores, expert_mask=expert, explanation_mask=explanation,
            expert_control_mask=controls["C_X"], explanation_control_mask=controls["C_E"],
            content_mask=content, strength_metrics=strength, strength_thresholds=thresholds,
            explanation_map=explanation_map,
        )
        row["patient_level_claim"] = True
        row["cluster_unit_type"] = "patient_id_hash"
        row["source_split"] = "publisher_validation_prior_exposed_development"
        output.append(row)
    return output


def occlusion_map(scorer: Qwen35StatementScorer, image: Image.Image, statement: str, *, grid_size: int, content_mask: np.ndarray) -> tuple[float, np.ndarray]:
    height, width = content_mask.shape
    y_edges = np.linspace(0, height, grid_size + 1, dtype=np.int64)
    x_edges = np.linspace(0, width, grid_size + 1, dtype=np.int64)
    perturbed = []
    for row in range(grid_size):
        for column in range(grid_size):
            mask = np.zeros_like(content_mask)
            mask[y_edges[row]:y_edges[row + 1], x_edges[column]:x_edges[column + 1]] = True
            mask &= content_mask
            perturbed.append(replace_with_local_ring_mean(image, mask, content_mask, ring_width=8))
    values = scorer.score([image, *perturbed], statement)
    return values[0], (values[0] - np.asarray(values[1:], dtype=np.float64)).reshape(grid_size, grid_size)


def validate_lock(lock: dict[str, Any], manifest_path: Path) -> None:
    if lock.get("status") != "score_free_ready" or lock.get("formal_result") is not False:
        raise ValueError("CheXlocalize development lock is not ready")
    if lock.get("test_opened") is not False or lock.get("patient_level_claim") is not True:
        raise ValueError("CheXlocalize development boundary changed")
    if lock.get("model_snapshot_sha256") != QWEN35_2B_SNAPSHOT_SHA256:
        raise ValueError("CheXlocalize Qwen3.5 model identity changed")
    if file_sha256(manifest_path) != lock.get("manifest_sha256"):
        raise ValueError("CheXlocalize development manifest changed")
    body = dict(lock)
    recorded = body.pop("canonical_sha256", None)
    if canonical_json_sha256(body) != recorded:
        raise ValueError("CheXlocalize development lock canonical hash changed")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
