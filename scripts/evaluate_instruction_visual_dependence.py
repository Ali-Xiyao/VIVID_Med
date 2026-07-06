"""Evaluate instruction-model loss under visual input perturbations."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import CXRInstructionDataset, instruction_collate_fn
from scripts.train_cxr_instruction import create_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: str) -> dict[str, Any]:
    state = torch.load(checkpoint_path, map_location=device)
    model.vit.load_state_dict(state["vit"])
    model.projector.load_state_dict(state["projector"])
    return {
        "global_step": int(state.get("global_step", -1)),
        "best_val_loss": float(state.get("best_val_loss", float("nan"))),
    }


def create_eval_loader(config: dict[str, Any], max_samples: int | None, batch_size: int | None) -> DataLoader:
    data_cfg = config["data"]
    instruction_path = data_cfg.get("val_instruction_path")
    if not instruction_path:
        raise ValueError("config data.val_instruction_path is required for visual-dependence eval")

    dataset = CXRInstructionDataset(
        data_root=data_cfg["data_root"],
        instruction_jsonl_path=instruction_path,
        image_size=int(data_cfg.get("image_size", 224)),
        is_train=False,
        max_samples=max_samples if max_samples is not None else data_cfg.get("max_val_samples"),
        prompt_template=data_cfg.get("prompt_template", "Question: {question}\nAnswer: "),
    )
    eval_batch_size = batch_size or int(config["training"].get("eval_batch_size", config["training"]["batch_size"]))
    return DataLoader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=instruction_collate_fn,
        pin_memory=True,
    )


def build_mismatch_indices(records: list[dict[str, Any]]) -> list[int]:
    n = len(records)
    if n < 2:
        raise ValueError("image_shuffle requires at least two records")
    offset = max(1, n // 2)
    partners: list[int] = []
    for idx, record in enumerate(records):
        sample_id = record.get("sample_id")
        partner = (idx + offset) % n
        for _ in range(n):
            if records[partner].get("sample_id") != sample_id:
                break
            partner = (partner + 1) % n
        else:
            raise ValueError("image_shuffle requires at least two distinct sample_id values")
        partners.append(partner)
    return partners


def perturb_images(images: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "normal":
        return images
    if mode == "question_only":
        return torch.zeros_like(images)
    raise ValueError(f"Unknown eval mode: {mode}")


@torch.no_grad()
def evaluate_mode(
    model: torch.nn.Module,
    loader: DataLoader,
    mode: str,
    device: str,
    use_bf16: bool,
    limit_batches: int | None,
    mismatch_indices: list[int] | None = None,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    num_batches = 0
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 and str(device).startswith("cuda")
        else nullcontext()
    )

    for batch_idx, batch in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        if mode == "image_shuffle":
            if mismatch_indices is None:
                raise ValueError("mismatch_indices is required for image_shuffle")
            shuffled_images = [
                loader.dataset[mismatch_indices[int(record_index)]]["image"]
                for record_index in batch["record_indices"]
            ]
            images = torch.stack(shuffled_images).to(device, non_blocking=True)
        else:
            images = batch["images"].to(device, non_blocking=True)
            images = perturb_images(images, mode)
        with autocast_ctx:
            output = model(
                images=images,
                prompt_text=batch["prompt_texts"],
                target_text=batch["target_jsons"],
            )
        batch_size = len(batch["target_jsons"])
        total_loss += float(output["loss"].detach().cpu()) * batch_size
        total_examples += batch_size
        num_batches += 1

    if total_examples == 0:
        raise RuntimeError("No examples were evaluated")
    return {
        "mode": mode,
        "loss": total_loss / total_examples,
        "examples": total_examples,
        "batches": num_batches,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "loss", "delta_vs_normal", "examples", "batches"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--modes", nargs="+", default=["normal", "question_only", "image_shuffle"])
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit-batches", type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args.config)
    device = args.device or config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    loader = create_eval_loader(config, max_samples=args.max_samples, batch_size=args.batch_size)
    model = create_model(config, str(device))
    checkpoint_meta = load_checkpoint(model, args.checkpoint, str(device))

    use_bf16 = bool(config.get("training", {}).get("bf16", True))
    mismatch_indices = None
    if "image_shuffle" in args.modes:
        mismatch_indices = build_mismatch_indices(loader.dataset.records)
    rows = [
        evaluate_mode(
            model=model,
            loader=loader,
            mode=mode,
            device=str(device),
            use_bf16=use_bf16,
            limit_batches=args.limit_batches,
            mismatch_indices=mismatch_indices,
        )
        for mode in args.modes
    ]

    normal_loss = next((row["loss"] for row in rows if row["mode"] == "normal"), None)
    for row in rows:
        row["delta_vs_normal"] = None if normal_loss is None else row["loss"] - normal_loss

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_meta": checkpoint_meta,
        "seed": args.seed,
        "device": str(device),
        "modes": rows,
        "interpretation": "Positive delta_vs_normal means the perturbation made teacher-forced answer loss worse.",
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(args.output.with_suffix(".csv"), rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
