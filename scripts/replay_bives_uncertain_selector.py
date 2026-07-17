"""Zero-training aligned selector/evidence replay for a real uncertain pair.

This diagnostic intentionally does not optimise parameters or change the
BiVES decoder, losses, K budget, or Qwen3.5 backbone. It separates the
evidence field from the exact-K selector using the train/validation uncertain
images actually consumed by a completed local mechanism gate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bives_cxr.backbones import Qwen35VisionAdapter, load_qwen35_visual_and_processor
from bives_cxr.data import BiVESManifestDataset, read_manifest
from bives_cxr.model import BiVESModelConfig
from scripts.train_bives_cxr import (
    BiVESExperiment,
    Qwen35BiVESCollator,
    load_checkpoint_model_state,
    load_config,
    move_to_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-uncertain-image", type=Path)
    parser.add_argument("--val-uncertain-image", type=Path)
    parser.add_argument("--rotation-degrees", type=float, default=1.0)
    return parser.parse_args()


def resolve_uncertain_image(config: dict[str, Any], split: str) -> Path:
    rows = read_manifest(config["data"][f"{split}_manifest"])
    for row in rows:
        if row["state"] == "uncertain":
            path = Path(str(row["image_path"]))
            return path if path.is_absolute() else Path(config["data"]["data_root"]) / path
    raise ValueError(f"{split} manifest has no uncertain row")


def theta_for_pil_rotation(
    degrees: float,
    *,
    inverse: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return grid_sample output-to-input coordinates for positive PIL.rotate."""

    angle = math.radians(float(degrees)) * (-1.0 if inverse else 1.0)
    cosine, sine = math.cos(angle), math.sin(angle)
    return torch.tensor(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0]], dtype=dtype, device=device
    ).unsqueeze(0)


def warp_grid(source: torch.Tensor, theta: torch.Tensor, *, mode: str = "bilinear") -> torch.Tensor:
    """Warp ``[channels, height, width]`` source values into target coordinates."""

    if source.ndim != 3:
        raise ValueError("source must have shape [channels, height, width]")
    source_batch = source.unsqueeze(0)
    theta = theta.to(device=source.device, dtype=source.dtype)
    grid = functional.affine_grid(theta, source_batch.shape, align_corners=False)
    return functional.grid_sample(
        source_batch, grid, mode=mode, padding_mode="zeros", align_corners=False
    ).squeeze(0)


def pil_rotation_alignment_report(
    grid_h: int = 28, grid_w: int = 28, degrees: float = 1.0
) -> dict[str, float | bool]:
    """Lock the grid-sample direction against a real positive PIL rotation."""

    patch = 16
    source = torch.zeros((1, grid_h, grid_w), dtype=torch.float32)
    source[0, 5, grid_w - 7] = 1.0
    image = Image.fromarray((source[0].numpy() * 255.0).astype(np.uint8), mode="L")
    image = image.resize((grid_w * patch, grid_h * patch), resample=Image.Resampling.NEAREST)
    rotated = image.rotate(float(degrees), resample=Image.Resampling.BICUBIC, fillcolor=0)
    observed = torch.from_numpy(np.asarray(rotated, dtype=np.float32) / 255.0)
    observed = observed.reshape(grid_h, patch, grid_w, patch).mean(dim=(1, 3)).unsqueeze(0)
    forward = warp_grid(source, theta_for_pil_rotation(degrees))
    reverse = warp_grid(source, theta_for_pil_rotation(degrees, inverse=True))
    forward_mse = float(torch.mean((forward - observed) ** 2))
    reverse_mse = float(torch.mean((reverse - observed) ** 2))
    return {
        "degrees": float(degrees),
        "forward_mse": forward_mse,
        "inverse_mse": reverse_mse,
        "direction_verified": bool(forward_mse < reverse_mse),
    }


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty_like(order, dtype=np.float64)
    result[order] = np.arange(values.size, dtype=np.float64)
    return result


def correlation(a: torch.Tensor, b: torch.Tensor, valid: torch.Tensor) -> dict[str, float | int]:
    mask = valid.bool().detach().cpu().numpy()
    first, second = a.detach().float().cpu().numpy()[mask], b.detach().float().cpu().numpy()[mask]
    if first.size < 2 or float(np.std(first)) == 0.0 or float(np.std(second)) == 0.0:
        return {"pearson": float("nan"), "spearman": float("nan"), "count": int(first.size)}
    return {
        "pearson": float(np.corrcoef(first, second)[0, 1]),
        "spearman": float(np.corrcoef(ranks(first), ranks(second))[0, 1]),
        "count": int(first.size),
    }


def exact_mask(scores: torch.Tensor, valid: torch.Tensor, topk: int) -> torch.Tensor:
    if int(valid.bool().sum()) < topk:
        raise ValueError("exact-K diagnostic requires at least K valid patches")
    masked = scores.detach().float().masked_fill(~valid.bool(), -torch.inf)
    indices = torch.topk(masked, k=topk).indices
    result = torch.zeros_like(masked)
    result[indices] = 1.0
    return result


def relaxed_gate(logits: torch.Tensor, valid: torch.Tensor, topk: int, temperature: float) -> torch.Tensor:
    masked = logits.detach().float().masked_fill(~valid.bool(), -torch.inf)
    threshold = torch.topk(masked, k=topk).values[-1]
    return torch.sigmoid((masked - threshold) / float(temperature)).masked_fill(~valid.bool(), 0.0)


def aggregate_evidence(
    evidence_pm: torch.Tensor, weights: torch.Tensor, decoder: torch.nn.Module
) -> dict[str, float | list[float]]:
    weights = weights.detach().float().clamp_min(0.0)
    aggregate = (evidence_pm.detach().float() * weights.unsqueeze(-1)).sum(dim=0) / weights.sum().clamp_min(1e-8)
    positive, negative = float(aggregate[0]), float(aggregate[1])
    decoded = decoder(aggregate[0].reshape(1), aggregate[1].reshape(1))
    return {
        "evidence_pos": positive,
        "evidence_neg": negative,
        "delta": positive - negative,
        "total": positive + negative,
        "rho": (positive - negative) / (positive + negative + 1e-8),
        "state_probs": decoded["state_probs"][0].detach().float().cpu().tolist(),
    }


def mask_indices(mask: torch.Tensor) -> list[int]:
    return torch.where(mask.detach().cpu() > 0.5)[0].tolist()


def jaccard(first: torch.Tensor, second: torch.Tensor) -> float:
    first_set, second_set = set(mask_indices(first)), set(mask_indices(second))
    return len(first_set & second_set) / max(1, len(first_set | second_set))


def recall_at_one_patch(reference: torch.Tensor, candidate: torch.Tensor, grid_h: int, grid_w: int) -> float:
    neighborhood: set[int] = set()
    for index in mask_indices(reference):
        y, x = divmod(index, grid_w)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < grid_h and 0 <= xx < grid_w:
                    neighborhood.add(yy * grid_w + xx)
    selected = set(mask_indices(candidate))
    return len(selected & neighborhood) / max(1, len(selected))


def k_margin(logits: torch.Tensor, valid: torch.Tensor, topk: int) -> float:
    values = torch.sort(logits.detach().float().masked_fill(~valid.bool(), -torch.inf), descending=True).values
    return float("nan") if topk >= int(valid.bool().sum()) else float(values[topk - 1] - values[topk])


def contribution_summary(signed: torch.Tensor, total: torch.Tensor, mask: torch.Tensor) -> dict[str, float | int | None]:
    selected_signed, selected_total = signed.detach().float()[mask.bool()], total.detach().float()[mask.bool()]
    if selected_signed.numel() == 0:
        return {"patches": 0, "signed_mean": float("nan"), "signed_sum": 0.0, "total_mean": float("nan"), "total_sum": 0.0, "max_abs_signed_patch": None, "max_abs_signed_value": float("nan")}
    local_index = int(selected_signed.abs().argmax())
    global_indices = torch.where(mask.bool())[0]
    return {
        "patches": int(selected_signed.numel()),
        "signed_mean": float(selected_signed.mean()),
        "signed_sum": float(selected_signed.sum()),
        "total_mean": float(selected_total.mean()),
        "total_sum": float(selected_total.sum()),
        "max_abs_signed_patch": int(global_indices[local_index]),
        "max_abs_signed_value": float(selected_signed[local_index]),
    }


def build_experiment(config: dict[str, Any], checkpoint: dict[str, Any], device: torch.device) -> tuple[BiVESExperiment, Any]:
    visual_model, processor, qwen_config = load_qwen35_visual_and_processor(
        config["model"]["path"], dtype=str(config["model"].get("dtype", "bf16")), attention_implementation=str(config["model"].get("attention_implementation", "eager"))
    )
    for parameter in visual_model.parameters():
        parameter.requires_grad = False
    adapter = Qwen35VisionAdapter(visual_model, spatial_merge_size=int(qwen_config["vision_config"]["spatial_merge_size"])).to(device)
    head_config = BiVESModelConfig(
        visual_dim=int(qwen_config["vision_config"]["hidden_size"]), statement_dim=int(config["model"].get("statement_dim", 512)), fusion_dim=int(config["bives"].get("fusion_dim", 512)), evidence_max=float(config["bives"].get("evidence_max", 8.0)), gate_mode=str(config["bives"]["mask"].get("type", "soft_topk")), topk=int(config["bives"]["mask"].get("topk", 16)), gate_temperature=float(config["bives"]["mask"].get("temperature", 0.5)), decoder_type=str(config["bives"]["decoder"].get("type", "")), tau_a=float(config["bives"]["decoder"].get("tau_a", 1.0)), tau_p=float(config["bives"]["decoder"].get("tau_p", 1.0)), uncertainty_mass=float(config["bives"]["decoder"].get("uncertainty_mass", 1.0)), num_controls=int(config["bives"].get("interventions", {}).get("num_controls", 2)), control_mode=str(config["bives"].get("interventions", {}).get("control_mode", "random_disjoint")), contextual_layers=int(config["bives"].get("contextual_layers", 1)), contextual_heads=int(config["bives"].get("contextual_heads", 1)), contextual_dropout=float(config["bives"].get("contextual_dropout", 0.0))
    )
    experiment = BiVESExperiment(adapter, num_statements=len(checkpoint["statement_to_index"]), statement_dim=int(config["model"].get("statement_dim", 512)), head_config=head_config).to(device)
    load_checkpoint_model_state(experiment, checkpoint)
    experiment.eval()
    return experiment, processor


def write_pair_manifest(path: Path, train_path: Path, val_path: Path) -> None:
    rows = []
    for split, image_path in (("train", train_path), ("val", val_path)):
        rows.append({"sample_id": f"uncertain-direct-pair-{split}", "patient_id": f"uncertain-direct-pair-{split}", "image_path": str(image_path.resolve().as_posix()), "group_id": "uncertain-direct-pair", "canonical_statement_id": "local-overfit-synthetic-statement", "statement_text": "Synthetic local mechanism statement.", "state": "uncertain", "source_dataset": "local_mechanism_gate_direct_pair"})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_path = args.train_uncertain_image or resolve_uncertain_image(config, "train")
    val_path = args.val_uncertain_image or resolve_uncertain_image(config, "val")
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError(f"uncertain pair not found: train={train_path}, val={val_path}")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "direct_pair_manifest.jsonl"
    write_pair_manifest(manifest, train_path, val_path)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(str(config.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")))
    experiment, processor = build_experiment(config, checkpoint, device)
    dataset = BiVESManifestDataset(manifest, data_root=".", statement_to_index=checkpoint["statement_to_index"])
    collator = Qwen35BiVESCollator(processor, image_size=int(config["data"].get("image_size", 448)), include_group_indices=False, split="selector_replay", training_seed=int(config.get("seed", 17)), evaluation_control_seed=int(config.get("evaluation", {}).get("control_seed", 20260717)))
    batch = move_to_device(next(iter(DataLoader(dataset, batch_size=2, collate_fn=collator))), device)
    patches = experiment.backbone(batch["pixel_values"], batch["image_grid_thw"])
    if len(set(patches.grid_hw)) != 1:
        raise ValueError(f"direct uncertain pair requires matching patch grids, got {patches.grid_hw}")
    grid_h, grid_w = patches.grid_hw[0]
    statements = experiment.statement_table(batch["statement_indices"]).to(dtype=experiment.head.visual_projection.weight.dtype)
    valid = patches.valid_mask & batch["content_valid_mask"].bool()
    original = experiment.head.score_tokens(patches.tokens.to(dtype=experiment.head.visual_projection.weight.dtype), statements, valid)
    topk, temperature = int(config["bives"]["mask"].get("topk", 16)), float(config["bives"]["mask"].get("temperature", 0.5))
    train, val = 0, 1
    theta_train_to_val = theta_for_pil_rotation(args.rotation_degrees, device=device)
    theta_val_to_train = theta_for_pil_rotation(args.rotation_degrees, inverse=True, device=device)
    train_evidence, val_evidence = original["evidence_pm"][train], original["evidence_pm"][val]
    train_logits, val_logits = original["gate_logits"][train], original["gate_logits"][val]
    train_valid, val_valid = valid[train], valid[val]
    train_hard, val_hard = exact_mask(train_logits, train_valid, topk), exact_mask(val_logits, val_valid, topk)
    train_relaxed, val_relaxed = relaxed_gate(train_logits, train_valid, topk, temperature), relaxed_gate(val_logits, val_valid, topk, temperature)

    def gridify(values: torch.Tensor) -> torch.Tensor:
        return values.reshape(1, grid_h, grid_w) if values.ndim == 1 else values.transpose(0, 1).reshape(values.shape[1], grid_h, grid_w)

    aligned_train_evidence = warp_grid(gridify(train_evidence), theta_train_to_val).reshape(2, -1).transpose(0, 1)
    aligned_train_logits = warp_grid(gridify(train_logits), theta_train_to_val).reshape(-1)
    aligned_train_tokens = warp_grid(gridify(patches.tokens[train].detach().float()), theta_train_to_val).reshape(patches.tokens.shape[-1], -1).transpose(0, 1)
    aligned_train_valid = warp_grid(gridify(train_valid.float()), theta_train_to_val, mode="nearest").reshape(-1) > 0.5
    common_valid_val = val_valid & aligned_train_valid
    mapped_train_mask = exact_mask(warp_grid(gridify(train_hard), theta_train_to_val).reshape(-1), val_valid, topk)
    mapped_train_relaxed = warp_grid(gridify(train_relaxed), theta_train_to_val).reshape(-1).masked_fill(~val_valid, 0.0)
    aligned_val_valid = warp_grid(gridify(val_valid.float()), theta_val_to_train, mode="nearest").reshape(-1) > 0.5
    mapped_val_mask = exact_mask(warp_grid(gridify(val_hard), theta_val_to_train).reshape(-1), train_valid, topk)
    r_tt, r_vv = aggregate_evidence(train_evidence, train_hard, experiment.head.decoder), aggregate_evidence(val_evidence, val_hard, experiment.head.decoder)
    r_vt, r_tv = aggregate_evidence(val_evidence, mapped_train_mask, experiment.head.decoder), aggregate_evidence(train_evidence, mapped_val_mask, experiment.head.decoder)
    selector_delta, field_delta = float(r_vv["delta"]) - float(r_vt["delta"]), float(r_vt["delta"]) - float(r_tt["delta"])
    selector_fraction = abs(selector_delta) / (abs(selector_delta) + abs(field_delta) + 1e-8)
    uncertain_tolerance = 0.2
    if abs(float(r_vv["rho"])) <= uncertain_tolerance:
        diagnosis = "uncertain_failure_not_reproduced"
    elif abs(float(r_vt["rho"])) <= uncertain_tolerance:
        diagnosis = "selector_dominant"
    else:
        diagnosis = "evidence_field_or_synthetic_definition_dominant"
    train_signed, val_signed = train_evidence[:, 0] - train_evidence[:, 1], val_evidence[:, 0] - val_evidence[:, 1]
    train_total, val_total = train_evidence.sum(dim=-1), val_evidence.sum(dim=-1)
    intersection, added, removed = val_hard.bool() & mapped_train_mask.bool(), val_hard.bool() & ~mapped_train_mask.bool(), mapped_train_mask.bool() & ~val_hard.bool()
    k_values = [k for k in (8, 16, 32, 64) if k <= int(train_valid.sum()) and k <= int(val_valid.sum())]
    k_sweep: dict[str, Any] = {}
    for k in k_values:
        train_k, val_k = exact_mask(train_logits, train_valid, k), exact_mask(val_logits, val_valid, k)
        mapped_train_k = exact_mask(warp_grid(gridify(train_k), theta_train_to_val).reshape(-1), val_valid, k)
        mapped_val_k = exact_mask(warp_grid(gridify(val_k), theta_val_to_train).reshape(-1), train_valid, k)
        k_sweep[str(k)] = {"r_tt": aggregate_evidence(train_evidence, train_k, experiment.head.decoder), "r_vv": aggregate_evidence(val_evidence, val_k, experiment.head.decoder), "r_vt": aggregate_evidence(val_evidence, mapped_train_k, experiment.head.decoder), "r_tv": aggregate_evidence(train_evidence, mapped_val_k, experiment.head.decoder)}
    soft = {"train": aggregate_evidence(train_evidence, train_relaxed, experiment.head.decoder), "val": aggregate_evidence(val_evidence, val_relaxed, experiment.head.decoder), "val_with_aligned_train_gate": aggregate_evidence(val_evidence, mapped_train_relaxed, experiment.head.decoder)}
    all_pool = {"train": aggregate_evidence(train_evidence, train_valid.float(), experiment.head.decoder), "val": aggregate_evidence(val_evidence, val_valid.float(), experiment.head.decoder)}
    token_cosine = functional.cosine_similarity(aligned_train_tokens[common_valid_val], patches.tokens[val].detach().float()[common_valid_val], dim=-1)
    report = {
        "formal_result": False, "training_steps": 0, "checkpoint": str(args.checkpoint), "train_uncertain_image": str(train_path), "val_uncertain_image": str(val_path), "grid_h": int(grid_h), "grid_w": int(grid_w), "topk": topk, "rotation_degrees": float(args.rotation_degrees), "rotation_direction_check": pil_rotation_alignment_report(grid_h, grid_w, args.rotation_degrees),
        "cross_replay": {"r_tt": r_tt, "r_vv": r_vv, "r_vt": r_vt, "r_tv": r_tv},
        "selector_field_decomposition": {"selector_delta": selector_delta, "field_delta": field_delta, "selector_fraction": selector_fraction, "uncertain_rho_tolerance": uncertain_tolerance, "diagnosis": diagnosis},
        "aligned_stability": {"topk_jaccard": jaccard(mapped_train_mask, val_hard), "recall_at_one_patch": recall_at_one_patch(mapped_train_mask, val_hard, grid_h, grid_w), "gate_logits": correlation(aligned_train_logits, val_logits, common_valid_val), "signed_evidence": correlation(aligned_train_evidence[:, 0] - aligned_train_evidence[:, 1], val_signed, common_valid_val), "total_evidence": correlation(aligned_train_evidence.sum(dim=-1), val_total, common_valid_val), "qwen_patch_token_cosine_mean": float(token_cosine.mean()), "qwen_patch_token_cosine_min": float(token_cosine.min()), "common_valid_patches": int(common_valid_val.sum()), "train_k_margin": k_margin(train_logits, train_valid, topk), "val_k_margin": k_margin(val_logits, val_valid, topk)},
        "pooling": {"soft": soft, "all_valid": all_pool, "k_sweep": k_sweep},
        "val_patch_contributions": {"common": contribution_summary(val_signed, val_total, intersection), "added_by_val_selector": contribution_summary(val_signed, val_total, added), "removed_from_train_selector": contribution_summary(val_signed, val_total, removed)},
    }
    tensors = {"qwen_tokens_train": patches.tokens[train].detach().float().cpu(), "qwen_tokens_val": patches.tokens[val].detach().float().cpu(), "gate_logits_train": train_logits.detach().float().cpu(), "gate_logits_val": val_logits.detach().float().cpu(), "relaxed_gate_train": train_relaxed.detach().float().cpu(), "relaxed_gate_val": val_relaxed.detach().float().cpu(), "evidence_pos_train": train_evidence[:, 0].detach().float().cpu(), "evidence_neg_train": train_evidence[:, 1].detach().float().cpu(), "evidence_pos_val": val_evidence[:, 0].detach().float().cpu(), "evidence_neg_val": val_evidence[:, 1].detach().float().cpu(), "signed_evidence_train": train_signed.detach().float().cpu(), "signed_evidence_val": val_signed.detach().float().cpu(), "total_evidence_train": train_total.detach().float().cpu(), "total_evidence_val": val_total.detach().float().cpu(), "hard_mask_train": train_hard.detach().cpu(), "hard_mask_val": val_hard.detach().cpu(), "mapped_train_mask_on_val": mapped_train_mask.detach().cpu(), "mapped_val_mask_on_train": mapped_val_mask.detach().cpu(), "valid_mask_train": train_valid.detach().cpu(), "valid_mask_val": val_valid.detach().cpu(), "grid_hw": torch.tensor([grid_h, grid_w]), "affine_train_to_val": theta_train_to_val.detach().cpu(), "affine_val_to_train": theta_val_to_train.detach().cpu()}
    torch.save(tensors, output_dir / "selector_evidence_arrays.pt")
    (output_dir / "selector_evidence_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output_dir / "selector_evidence_report.json"), "diagnosis": diagnosis}, indent=2))


if __name__ == "__main__":
    main()
