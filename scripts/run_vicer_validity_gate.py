"""Run the locked local VICER V0 Qwen3.5 intervention-validity matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bives_cxr.backbones import Qwen35VisionAdapter, load_qwen35_visual_and_processor  # noqa: E402
from bives_cxr.qwen35_preprocessing import content_mask_for_grid, letterbox_image  # noqa: E402
from scripts.cache_qwen35_patch_tokens import read_image  # noqa: E402
from vicer_cxr.intervention_bank import apply_v0_intervention  # noqa: E402
from vicer_cxr.validity import (  # noqa: E402
    VALIDITY_FINDINGS,
    canonical_sha256,
    file_sha256,
    linear_margin,
    summarize_v0_rows,
    validate_v0_manifest,
)


OPERATORS = {
    "masked_gaussian_blur": (2.0, 4.0, 8.0, 16.0),
    "local_ring_mean": (0.25, 0.5, 0.75, 1.0),
    "low_frequency_replacement": (0.25, 0.5, 0.75, 1.0),
}


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_head(path: Path) -> dict[str, np.ndarray | float]:
    with np.load(path, allow_pickle=False) as value:
        return {
            "scaler_mean": value["scaler_mean"],
            "scaler_scale": value["scaler_scale"],
            "weight": value["weight"],
            "intercept": float(value["intercept"][0]),
        }


def normalize_tokens(tokens: torch.Tensor) -> np.ndarray:
    value = tokens.float()
    value = torch.nn.functional.layer_norm(value, (value.shape[-1],))
    return value.cpu().numpy().astype(np.float64)


def patch_mask_from_pixels(
    mask: np.ndarray, grid_hw: tuple[int, int], valid_mask: np.ndarray
) -> np.ndarray:
    grid_h, grid_w = map(int, grid_hw)
    height, width = mask.shape
    result = np.zeros((grid_h, grid_w), dtype=bool)
    for y in range(grid_h):
        y0, y1 = y * height // grid_h, (y + 1) * height // grid_h
        for x in range(grid_w):
            x0, x1 = x * width // grid_w, (x + 1) * width // grid_w
            result[y, x] = bool(mask[y0:y1, x0:x1].any())
    flattened = result.reshape(-1) & valid_mask.astype(bool)
    if not flattened.any():
        raise ValueError("VICER intervention mask contains no valid patch")
    return flattened


def pool(tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = tokens[mask]
    if selected.size == 0:
        raise ValueError("VICER pooled feature is empty")
    return selected.mean(axis=0)


def encode_images(
    images: list[Any],
    *,
    visual: torch.nn.Module,
    adapter: Qwen35VisionAdapter,
    processor: Any,
    device: torch.device,
    image_size: int,
) -> list[dict[str, Any]]:
    letterboxed = []
    content_boxes = []
    for image in images:
        prepared, content_box = letterbox_image(image, image_size)
        letterboxed.append(prepared)
        content_boxes.append(content_box)
    messages = [
        [{"role": "user", "content": [{"type": "image", "image": image}]}]
        for image in letterboxed
    ]
    texts = [
        processor.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
        for message in messages
    ]
    batch = processor(text=texts, images=letterboxed, return_tensors="pt", padding=True)
    pixel_values = batch["pixel_values"].to(device=device, dtype=next(visual.parameters()).dtype)
    grid = batch["image_grid_thw"].to(device)
    with torch.no_grad():
        patches = adapter(pixel_values, grid)
    result = []
    for index, content_box in enumerate(content_boxes):
        valid = patches.valid_mask[index].detach().cpu()
        content = content_mask_for_grid(batch["image_grid_thw"][index], content_box, image_size)
        valid &= content
        result.append(
            {
                "tokens": patches.tokens[index].detach().cpu(),
                "valid_mask": valid.numpy(),
                "grid_hw": patches.grid_hw[index],
                "content_box": content_box,
            }
        )
    return result


def realism_score(feature: np.ndarray, reference: dict[str, np.ndarray | float]) -> float:
    centered = feature - np.asarray(reference["mean"])
    projected = centered @ np.asarray(reference["components"]).T
    standardized = projected / np.maximum(np.asarray(reference["scales"]), 1e-12)
    distance = float(np.sqrt(np.mean(standardized**2)))
    excess = max(0.0, distance - float(reference["distance_q95"]))
    return float(math.exp(-excess))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-lock", type=Path, required=True)
    parser.add_argument("--geometry-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--head-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--minimum-critic-auroc", type=float, default=0.60)
    parser.add_argument("--minimum-verifier-auroc", type=float, default=0.60)
    parser.add_argument("--minimum-q-remove", type=float, default=0.02)
    parser.add_argument("--minimum-q-preserve", type=float, default=0.98)
    parser.add_argument("--minimum-q-realism", type=float, default=0.50)
    parser.add_argument("--minimum-monotonic-spearman", type=float, default=0.80)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.50)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    validate_v0_manifest(rows)
    eval_rows = [row for row in rows if row["v0_role"] == "validity_eval"]
    data_lock = json.loads(args.data_lock.read_text(encoding="utf-8"))
    geometry_lock = json.loads((args.geometry_dir / "geometry_lock.json").read_text(encoding="utf-8"))
    cache_lock = json.loads((args.cache_dir / "cache_lock.json").read_text(encoding="utf-8"))
    head_lock_path = args.head_dir / "head_lock.json"
    head_lock = json.loads(head_lock_path.read_text(encoding="utf-8"))
    if not head_lock.get("head_gate_pass"):
        raise ValueError("VICER V0 independent head calibration gate failed before model scoring")
    if len(eval_rows) != len(VALIDITY_FINDINGS) * 8:
        raise ValueError("VICER V0 evaluation denominator changed")
    if geometry_lock.get("records") != len(eval_rows) or geometry_lock.get("model_or_score_opened") is not False:
        raise ValueError("VICER V0 score-free geometry lock changed")
    if cache_lock.get("status") != "complete" or cache_lock.get("manifest_sha256") != file_sha256(args.manifest):
        raise ValueError("VICER V0 cache lock changed")

    geometry_rows = [
        json.loads(line)
        for line in (args.geometry_dir / "geometry_rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    geometry = {str(row["sample_id"]): row for row in geometry_rows}
    index_rows = [
        json.loads(line)
        for line in (args.cache_dir / "index.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    index = {str(row["sample_id"]): row for row in index_rows}
    critic_heads = {
        finding: load_head(args.head_dir / f"{finding}_critic.npz") for finding in VALIDITY_FINDINGS
    }
    verifier_heads = {
        finding: load_head(args.head_dir / f"{finding}_verifier.npz") for finding in VALIDITY_FINDINGS
    }
    with np.load(args.head_dir / "realism_reference.npz", allow_pickle=False) as value:
        realism_reference = {
            "mean": value["mean"],
            "components": value["components"],
            "scales": value["scales"],
            "distance_q95": float(value["distance_q95"][0]),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "v0_rows.jsonl"
    progress_path = args.output_dir / "progress.json"
    identity = {
        "schema_version": "vicer-v0-run-identity-v1",
        "manifest_sha256": file_sha256(args.manifest),
        "data_lock_canonical_sha256": data_lock["canonical_sha256"],
        "geometry_lock_canonical_sha256": geometry_lock["canonical_sha256"],
        "cache_lock_canonical_sha256": cache_lock["canonical_sha256"],
        "head_lock_canonical_sha256": head_lock["canonical_sha256"],
        "model_snapshot_sha256": cache_lock["model_snapshot_sha256"],
        "operators": {key: list(value) for key, value in OPERATORS.items()},
        "thresholds": {
            "minimum_critic_auroc": args.minimum_critic_auroc,
            "minimum_verifier_auroc": args.minimum_verifier_auroc,
            "minimum_q_remove": args.minimum_q_remove,
            "minimum_q_preserve": args.minimum_q_preserve,
            "minimum_q_realism": args.minimum_q_realism,
            "minimum_monotonic_spearman": args.minimum_monotonic_spearman,
            "minimum_valid_fraction": args.minimum_valid_fraction,
        },
    }
    identity["canonical_sha256"] = canonical_sha256(identity)
    completed: dict[tuple[str, str, float], dict[str, Any]] = {}
    if rows_path.is_file():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                if row.get("run_identity_sha256") != identity["canonical_sha256"]:
                    raise ValueError("existing VICER V0 rows belong to another identity")
                completed[(row["sample_id"], row["operator_family"], float(row["strength"]))] = row

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    visual, processor, config = load_qwen35_visual_and_processor(
        args.model_path, dtype=args.dtype, attention_implementation="eager"
    )
    visual = visual.to(device).eval()
    adapter = Qwen35VisionAdapter(
        visual, spatial_merge_size=int(config["vision_config"]["spatial_merge_size"])
    ).to(device).eval()

    for sample_index, row in enumerate(eval_rows, start=1):
        geometry_row = geometry[str(row["sample_id"])]
        mask_file = args.geometry_dir / geometry_row["mask_file"]
        if file_sha256(mask_file) != geometry_row["mask_file_sha256"]:
            raise ValueError("VICER V0 mask artifact changed")
        with np.load(mask_file, allow_pickle=False) as masks:
            target_mask = masks["target"].astype(bool)
            control_mask = masks["control"].astype(bool)
            content_mask = masks["content"].astype(bool)
        cache_item = args.cache_dir / index[str(row["sample_id"])]["cache_file"]
        original_payload = torch.load(cache_item, map_location="cpu", weights_only=False)
        original_tokens = normalize_tokens(original_payload["patch_tokens"])
        original_valid = original_payload["valid_mask"].bool().numpy()
        grid_hw = tuple(original_payload["grid_hw"])
        target_patch = patch_mask_from_pixels(target_mask, grid_hw, original_valid)
        outside_target = original_valid & ~target_patch
        original_local = pool(original_tokens, target_patch)
        original_global = pool(original_tokens, original_valid)
        finding = str(row["canonical_statement_id"])
        critic_head = critic_heads[finding]
        verifier_head = verifier_heads[finding]
        original_critic_margin = float(linear_margin(original_local[None, :], critic_head)[0])
        original_verifier_margin = float(linear_margin(original_global[None, :], verifier_head)[0])
        source_image = read_image(Path(row["image_path"]))
        letterboxed, content_box = letterbox_image(source_image, args.image_size)
        if tuple(map(int, content_box)) != tuple(original_payload["content_box"]):
            raise ValueError("VICER original/intervention letterbox geometry changed")

        for family, strengths in OPERATORS.items():
            for strength in strengths:
                key = (str(row["sample_id"]), family, float(strength))
                if key in completed:
                    continue
                target_image, target_operator = apply_v0_intervention(
                    letterboxed,
                    target_mask,
                    content_mask,
                    family=family,
                    strength=float(strength),
                )
                control_image, control_operator = apply_v0_intervention(
                    letterboxed,
                    control_mask,
                    content_mask,
                    family=family,
                    strength=float(strength),
                )
                encoded_target, encoded_control = encode_images(
                    [target_image, control_image],
                    visual=visual,
                    adapter=adapter,
                    processor=processor,
                    device=device,
                    image_size=args.image_size,
                )
                if tuple(encoded_target["grid_hw"]) != grid_hw or tuple(encoded_control["grid_hw"]) != grid_hw:
                    raise ValueError("VICER intervention Qwen patch grid changed")
                target_tokens = normalize_tokens(encoded_target["tokens"])
                control_tokens = normalize_tokens(encoded_control["tokens"])
                target_local = pool(target_tokens, target_patch)
                target_global = pool(target_tokens, original_valid)
                control_global = pool(control_tokens, original_valid)
                target_critic_margin = float(linear_margin(target_local[None, :], critic_head)[0])
                target_verifier_margin = float(linear_margin(target_global[None, :], verifier_head)[0])
                control_verifier_margin = float(linear_margin(control_global[None, :], verifier_head)[0])
                q_remove = float(expit(original_critic_margin) - expit(target_critic_margin))
                cosine = np.sum(original_tokens[outside_target] * target_tokens[outside_target], axis=1)
                cosine /= np.maximum(
                    np.linalg.norm(original_tokens[outside_target], axis=1)
                    * np.linalg.norm(target_tokens[outside_target], axis=1),
                    1e-12,
                )
                q_preserve = float(np.clip(np.mean(cosine), 0.0, 1.0))
                q_realism = realism_score(target_global, realism_reference)
                target_effect = original_verifier_margin - target_verifier_margin
                control_effect = original_verifier_margin - control_verifier_margin
                target_control_gap = target_effect - control_effect
                valid_intervention = bool(
                    q_remove >= args.minimum_q_remove
                    and q_preserve >= args.minimum_q_preserve
                    and q_realism >= args.minimum_q_realism
                )
                result_row = {
                    "schema_version": "vicer-v0-dose-response-row-v1",
                    "run_identity_sha256": identity["canonical_sha256"],
                    "sample_id": row["sample_id"],
                    "image_id": row["image_id"],
                    "canonical_statement_id": finding,
                    "operator_family": family,
                    "strength": float(strength),
                    "target_operator": target_operator,
                    "control_operator": control_operator,
                    "q_remove": q_remove,
                    "q_preserve": q_preserve,
                    "q_realism": q_realism,
                    "valid_intervention": valid_intervention,
                    "original_critic_margin": original_critic_margin,
                    "target_critic_margin": target_critic_margin,
                    "original_verifier_margin": original_verifier_margin,
                    "target_verifier_margin": target_verifier_margin,
                    "control_verifier_margin": control_verifier_margin,
                    "target_effect": float(target_effect),
                    "control_effect": float(control_effect),
                    "target_control_gap": float(target_control_gap),
                    "critic_calibration_auroc": float(head_lock["findings"][finding]["critic"]["calibration_auroc"]),
                    "verifier_calibration_auroc": float(head_lock["findings"][finding]["verifier"]["calibration_auroc"]),
                    "validity_uses_target_verifier_effect": False,
                    "patient_level_claim": False,
                    "chexlocalize_test_opened": False,
                }
                with rows_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(result_row, sort_keys=True, ensure_ascii=False) + "\n")
                completed[key] = result_row
                write_json_atomic(
                    progress_path,
                    {
                        "identity": identity,
                        "status": "in_progress",
                        "completed_rows": len(completed),
                        "total_rows": len(eval_rows) * sum(len(values) for values in OPERATORS.values()),
                        "last_sample_index": sample_index,
                    },
                )
                print(json.dumps({"completed": len(completed), "sample": row["sample_id"], "family": family, "strength": strength}))

    expected = len(eval_rows) * sum(len(values) for values in OPERATORS.values())
    if len(completed) != expected:
        raise ValueError("VICER V0 result matrix is incomplete")
    ordered_rows = [completed[key] for key in sorted(completed)]
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in ordered_rows),
        encoding="utf-8",
        newline="\n",
    )
    summary = summarize_v0_rows(
        ordered_rows,
        minimum_critic_auroc=args.minimum_critic_auroc,
        minimum_verifier_auroc=args.minimum_verifier_auroc,
        minimum_monotonic_spearman=args.minimum_monotonic_spearman,
        minimum_preservation=args.minimum_q_preserve,
        minimum_realism=args.minimum_q_realism,
        minimum_valid_fraction=args.minimum_valid_fraction,
    )
    summary.update(
        {
            "run_identity": identity,
            "rows_sha256": file_sha256(rows_path),
            "data_role": "new VinDr-train image-disjoint development only",
            "single_reader_positive_allowed": True,
            "patient_level_claim": False,
            "chexlocalize_test_opened": False,
            "selector_started": False,
        }
    )
    summary["canonical_sha256"] = canonical_sha256(summary)
    write_json_atomic(args.output_dir / "v0_result.json", summary)
    write_json_atomic(
        progress_path,
        {"identity": identity, "status": "complete", "completed_rows": expected, "result": "v0_result.json"},
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
