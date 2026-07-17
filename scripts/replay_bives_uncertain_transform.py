"""Zero-training replay for BiVES uncertain transform and selector stability."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
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


VARIANTS = (
    "U0_posterize",
    "U1_posterize_contrast",
    "U2_posterize_rotate_nearest",
    "U3_posterize_contrast_rotate_bicubic",
    "U4_rotate_bicubic_posterize",
    "U5_rotate_bicubic_posterize_contrast",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-image",
        type=Path,
        help="Optional raw/near-raw source image. Defaults to the train support image from the config manifest.",
    )
    return parser.parse_args()


def median_fill(image: Image.Image) -> int:
    return int(np.median(np.asarray(image.convert("L"))))


def rotate_bicubic(image: Image.Image) -> Image.Image:
    return image.rotate(1.0, resample=Image.Resampling.BICUBIC, fillcolor=median_fill(image))


def rotate_nearest(image: Image.Image) -> Image.Image:
    return image.rotate(1.0, resample=Image.Resampling.NEAREST, fillcolor=median_fill(image))


def variant_image(base: Image.Image, name: str) -> Image.Image:
    image = base.convert("L")
    posterized = ImageOps.posterize(ImageOps.autocontrast(image), 3)
    if name == "U0_posterize":
        out = posterized
    elif name == "U1_posterize_contrast":
        out = ImageEnhance.Contrast(posterized).enhance(0.92)
    elif name == "U2_posterize_rotate_nearest":
        out = rotate_nearest(posterized)
    elif name == "U3_posterize_contrast_rotate_bicubic":
        out = rotate_bicubic(ImageEnhance.Contrast(posterized).enhance(0.92))
    elif name == "U4_rotate_bicubic_posterize":
        out = ImageOps.posterize(ImageOps.autocontrast(rotate_bicubic(image)), 3)
    elif name == "U5_rotate_bicubic_posterize_contrast":
        out = ImageEnhance.Contrast(
            ImageOps.posterize(ImageOps.autocontrast(rotate_bicubic(image)), 3)
        ).enhance(0.92)
    else:
        raise ValueError(f"unknown variant: {name}")
    return out.convert("RGB")


def find_default_source(config: dict[str, Any]) -> Path:
    rows = read_manifest(config["data"]["train_manifest"])
    for row in rows:
        if row["state"] == "support":
            image_path = Path(str(row["image_path"]))
            if not image_path.is_absolute():
                image_path = Path(config["data"]["data_root"]) / image_path
            return image_path
    raise ValueError("train manifest has no support row to use as the replay source image")


def write_manifest(path: Path, image_dir: Path) -> None:
    rows = []
    for index, name in enumerate(VARIANTS):
        rows.append(
            {
                "sample_id": name,
                "patient_id": f"replay-patient-{index}",
                "image_path": str((image_dir / f"{name}.png").as_posix()),
                "group_id": "uncertain-transform-replay",
                "canonical_statement_id": "local-overfit-synthetic-statement",
                "statement_text": "Synthetic local mechanism statement.",
                "state": "uncertain",
                "source_dataset": "local_mechanism_gate_replay",
            }
        )
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks_out = np.empty_like(order, dtype=np.float64)
    ranks_out[order] = np.arange(values.size, dtype=np.float64)
    return ranks_out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size or a.size < 2:
        return float("nan")
    ra = ranks(a)
    rb = ranks(b)
    if float(np.std(ra)) == 0.0 or float(np.std(rb)) == 0.0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def topk_indices(scores: torch.Tensor, valid: torch.Tensor, k: int) -> list[int]:
    masked = scores.detach().float().clone()
    masked[~valid.bool()] = -torch.inf
    return torch.topk(masked, k=k).indices.cpu().tolist()


def aggregate_for_indices(evidence_pm: torch.Tensor, indices: list[int]) -> dict[str, float]:
    selected = evidence_pm.detach().float().cpu()[indices]
    pos = float(selected[:, 0].mean())
    neg = float(selected[:, 1].mean())
    total = pos + neg
    rho = (pos - neg) / (total + 1e-8)
    return {"evidence_pos": pos, "evidence_neg": neg, "total": total, "rho": rho}


def k_margin(scores: torch.Tensor, valid: torch.Tensor, k: int) -> float:
    masked = scores.detach().float().clone()
    masked[~valid.bool()] = -torch.inf
    values = torch.sort(masked, descending=True).values
    if k >= int(valid.sum().item()):
        return float("nan")
    return float((values[k - 1] - values[k]).cpu())


def jaccard(a: list[int], b: list[int]) -> float:
    sa = set(a)
    sb = set(b)
    return len(sa & sb) / max(1, len(sa | sb))


def recall_at_one_patch(reference: list[int], candidate: list[int], grid_h: int, grid_w: int) -> float:
    dilated: set[int] = set()
    for index in reference:
        y, x = divmod(index, grid_w)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy = y + dy
                xx = x + dx
                if 0 <= yy < grid_h and 0 <= xx < grid_w:
                    dilated.add(yy * grid_w + xx)
    return len(set(candidate) & dilated) / max(1, len(candidate))


def rotate_indices(indices: list[int], grid_h: int, grid_w: int, degrees: float = 1.0) -> list[int]:
    angle = math.radians(degrees)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    cy = (grid_h - 1) / 2.0
    cx = (grid_w - 1) / 2.0
    mapped: list[int] = []
    for index in indices:
        y, x = divmod(index, grid_w)
        yy = y - cy
        xx = x - cx
        xr = xx * cos_a - yy * sin_a + cx
        yr = xx * sin_a + yy * cos_a + cy
        mx = min(max(int(round(xr)), 0), grid_w - 1)
        my = min(max(int(round(yr)), 0), grid_h - 1)
        mapped.append(my * grid_w + mx)
    return sorted(set(mapped))


def build_experiment(
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[BiVESExperiment, Any]:
    visual_model, processor, qwen_config = load_qwen35_visual_and_processor(
        config["model"]["path"],
        dtype=str(config["model"].get("dtype", "bf16")),
        attention_implementation=str(config["model"].get("attention_implementation", "eager")),
    )
    for parameter in visual_model.parameters():
        parameter.requires_grad = False
    visual_adapter = Qwen35VisionAdapter(
        visual_model,
        spatial_merge_size=int(qwen_config["vision_config"]["spatial_merge_size"]),
    ).to(device)
    head_config = BiVESModelConfig(
        visual_dim=int(qwen_config["vision_config"]["hidden_size"]),
        statement_dim=int(config["model"].get("statement_dim", 512)),
        fusion_dim=int(config["bives"].get("fusion_dim", 512)),
        evidence_max=float(config["bives"].get("evidence_max", 8.0)),
        gate_mode=str(config["bives"]["mask"].get("type", "soft_topk")),
        topk=int(config["bives"]["mask"].get("topk", 16)),
        gate_temperature=float(config["bives"]["mask"].get("temperature", 0.5)),
        decoder_type=str(config["bives"]["decoder"].get("type", "")),
        tau_a=float(config["bives"]["decoder"].get("tau_a", 1.0)),
        tau_p=float(config["bives"]["decoder"].get("tau_p", 1.0)),
        uncertainty_mass=float(config["bives"]["decoder"].get("uncertainty_mass", 1.0)),
        num_controls=int(config["bives"].get("interventions", {}).get("num_controls", 4)),
        control_mode=str(config["bives"].get("interventions", {}).get("control_mode", "random_disjoint")),
        contextual_layers=int(config["bives"].get("contextual_layers", 1)),
        contextual_heads=int(config["bives"].get("contextual_heads", 4)),
        contextual_dropout=float(config["bives"].get("contextual_dropout", 0.0)),
    )
    experiment = BiVESExperiment(
        visual_adapter,
        num_statements=len(checkpoint["statement_to_index"]),
        statement_dim=int(config["model"].get("statement_dim", 512)),
        head_config=head_config,
    ).to(device)
    load_checkpoint_model_state(experiment, checkpoint)
    experiment.eval()
    return experiment, processor


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "variants"
    image_dir.mkdir(exist_ok=True)
    source_path = args.source_image or find_default_source(config)
    with Image.open(source_path) as source:
        base = source.copy()
    unique_counts: dict[str, int] = {}
    for name in VARIANTS:
        image = variant_image(base, name)
        image.save(image_dir / f"{name}.png")
        unique_counts[name] = int(np.unique(np.asarray(image.convert("L"))).size)
    manifest = output_dir / "replay_manifest.jsonl"
    write_manifest(manifest, image_dir)

    device = torch.device(str(config.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    experiment, processor = build_experiment(config, checkpoint, device)
    dataset = BiVESManifestDataset(
        manifest,
        data_root=".",
        statement_to_index=checkpoint["statement_to_index"],
    )
    collator = Qwen35BiVESCollator(
        processor,
        image_size=int(config["data"].get("image_size", 448)),
        include_group_indices=False,
        split="replay",
        training_seed=int(config.get("seed", 17)),
        evaluation_control_seed=int(config.get("evaluation", {}).get("control_seed", 20260717)),
    )
    batch = move_to_device(next(iter(DataLoader(dataset, batch_size=len(dataset), collate_fn=collator))), device)
    patches = experiment.backbone(batch["pixel_values"], batch["image_grid_thw"])
    statement_embeddings = experiment.statement_table(batch["statement_indices"]).to(
        dtype=experiment.head.visual_projection.weight.dtype
    )
    valid_mask = patches.valid_mask & batch["content_valid_mask"].bool()
    original = experiment.head.score_tokens(
        patches.tokens.to(dtype=experiment.head.visual_projection.weight.dtype),
        statement_embeddings,
        valid_mask,
    )
    grid_h, grid_w = patches.grid_hw[0]
    topk = int(config["bives"]["mask"].get("topk", 16))
    k_values = [k for k in (8, 16, 32, 64) if k <= int(valid_mask[0].sum().item())]
    reference_indices = topk_indices(original["gate_logits"][0], valid_mask[0], topk)
    reference_tokens = patches.tokens[0].detach().float().cpu()
    records = []
    for index, name in enumerate(VARIANTS):
        scores = original["gate_logits"][index]
        evidence_pm = original["evidence_pm"][index]
        valid = valid_mask[index]
        own_indices = topk_indices(scores, valid, topk)
        mapped = (
            reference_indices
            if name in {"U0_posterize", "U1_posterize_contrast"}
            else rotate_indices(reference_indices, grid_h, grid_w)
        )
        mapped = [idx for idx in mapped if idx < int(valid.numel()) and bool(valid[idx])]
        if len(mapped) < topk:
            mapped = mapped + [idx for idx in reference_indices if idx not in mapped][: topk - len(mapped)]
        mapped = mapped[:topk]
        token_cos = torch.nn.functional.cosine_similarity(
            reference_tokens[valid_mask[0].detach().cpu()],
            patches.tokens[index].detach().float().cpu()[valid.detach().cpu()],
            dim=-1,
        )
        gate_ref = original["gate_logits"][0].detach().float().cpu()[valid_mask[0].detach().cpu()].numpy()
        gate_cur = scores.detach().float().cpu()[valid.detach().cpu()].numpy()
        records.append(
            {
                "variant": name,
                "unique_gray_values": unique_counts[name],
                "state_probs": original["state_probs"][index].detach().float().cpu().tolist(),
                "own_mask": {
                    **aggregate_for_indices(evidence_pm, own_indices),
                    "topk_indices": own_indices,
                    "k_margin": k_margin(scores, valid, topk),
                },
                "mapped_u0_mask": {
                    **aggregate_for_indices(evidence_pm, mapped),
                    "topk_indices": mapped,
                    "jaccard_vs_u0": jaccard(reference_indices, own_indices),
                    "recall_at_one_patch_vs_u0": recall_at_one_patch(reference_indices, own_indices, grid_h, grid_w),
                },
                "k_sweep": {
                    str(k): aggregate_for_indices(evidence_pm, topk_indices(scores, valid, k))
                    for k in k_values
                },
                "gate_logit_spearman_vs_u0": spearman(gate_ref, gate_cur),
                "qwen_patch_token_cosine_vs_u0_mean": float(token_cos.mean()),
                "qwen_patch_token_cosine_vs_u0_min": float(token_cos.min()),
            }
        )
    payload = {
        "formal_result": False,
        "training_steps": 0,
        "checkpoint": str(args.checkpoint),
        "source_image": str(source_path),
        "grid_h": int(grid_h),
        "grid_w": int(grid_w),
        "topk": topk,
        "variants": list(VARIANTS),
        "records": records,
    }
    (output_dir / "replay_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "output": str(output_dir / "replay_report.json")}, indent=2))


if __name__ == "__main__":
    main()
