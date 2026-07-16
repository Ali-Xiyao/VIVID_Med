"""Evaluate Qwen3-VL instruction loss under visual input perturbations."""

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
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.clinical_instruction_dataset import ClinicalInstructionDataset, Qwen3VLInstructionCollator
from scripts.train_qwen3vl_clinical_instruction import (
    compute_weighted_lm_loss,
    load_model_and_processor,
    load_trainable_checkpoint,
    model_inputs,
    move_tensors_to_device,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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


class PerturbedInstructionDataset(Dataset):
    def __init__(self, base: ClinicalInstructionDataset, mode: str, mismatch_indices: list[int] | None = None) -> None:
        self.base = base
        self.mode = mode
        self.mismatch_indices = mismatch_indices
        self.records = base.records
        if mode == "hard_shuffle":
            self.indices = [
                idx
                for idx, record in enumerate(base.records)
                if record.get("hard_negative_image_path")
            ]
            if not self.indices:
                raise ValueError("hard_shuffle mode requires records with hard_negative_image_path")
        else:
            self.indices = list(range(len(base.records)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        base_idx = self.indices[idx]
        item = dict(self.base[base_idx])
        if self.mode == "normal":
            item["eval_mode"] = self.mode
            item["original_image_path"] = item["image_path"]
            item["perturbed_image_path"] = item["image_path"]
            return item
        if self.mode == "question_only":
            image = item["image"]
            item["image"] = Image.new("RGB", image.size, (0, 0, 0))
            item["eval_mode"] = self.mode
            item["original_image_path"] = item["image_path"]
            item["perturbed_image_path"] = "black_image"
            return item
        if self.mode == "image_shuffle":
            if self.mismatch_indices is None:
                raise ValueError("mismatch_indices is required for image_shuffle")
            partner_idx = self.mismatch_indices[base_idx]
            partner = self.base[partner_idx]
            item["image"] = partner["image"]
            item["eval_mode"] = self.mode
            item["original_image_path"] = item["image_path"]
            item["perturbed_image_path"] = partner["image_path"]
            item["perturbed_sample_id"] = partner["sample_id"]
            return item
        if self.mode == "hard_shuffle":
            record = self.records[base_idx]
            hard_path = Path(str(record.get("hard_negative_image_path") or ""))
            if not hard_path.is_absolute():
                hard_path = self.base.data_root / hard_path
            item["image"] = self.base._load_image(hard_path)
            item["eval_mode"] = self.mode
            item["original_image_path"] = item["image_path"]
            item["perturbed_image_path"] = str(hard_path)
            item["perturbed_sample_id"] = record.get("hard_negative_sample_id")
            return item
        raise ValueError(f"Unknown mode: {self.mode}")


def create_loader(
    config: dict[str, Any],
    processor: Any,
    mode: str,
    max_samples: int | None,
    batch_size: int | None,
) -> DataLoader:
    data_cfg = config["data"]
    val_path = data_cfg.get("val_instruction_path")
    if not val_path:
        raise ValueError("data.val_instruction_path is required")
    base = ClinicalInstructionDataset(
        data_root=data_cfg.get("data_root", "."),
        instruction_jsonl_path=val_path,
        max_samples=max_samples if max_samples is not None else data_cfg.get("max_val_samples"),
    )
    mismatch_indices = build_mismatch_indices(base.records) if mode == "image_shuffle" else None
    dataset = PerturbedInstructionDataset(base=base, mode=mode, mismatch_indices=mismatch_indices)
    collator = Qwen3VLInstructionCollator(
        processor=processor,
        max_length=data_cfg.get("max_length"),
        loss_weighting=config.get("training", {}).get("loss_weighting"),
        loss_masking=config.get("training", {}).get("loss_masking"),
    )
    eval_batch_size = batch_size or int(config["training"].get("eval_batch_size", config["training"]["batch_size"]))
    return DataLoader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=collator,
        pin_memory=True,
    )


@torch.no_grad()
def evaluate_mode(
    model: torch.nn.Module,
    loader: DataLoader,
    mode: str,
    device: torch.device,
    use_bf16: bool,
    limit_batches: int | None,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    example_count = 0
    batch_count = 0
    per_answer_type: dict[str, list[float]] = {}
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 and str(device).startswith("cuda")
        else nullcontext()
    )
    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Evaluating {mode}", leave=False)):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        batch = move_tensors_to_device(batch, device)
        with autocast_ctx:
            outputs = model(**model_inputs(batch), use_cache=False)
            loss = compute_weighted_lm_loss(outputs.logits, batch["labels"], batch["loss_weights"])
        batch_size = int(batch["labels"].shape[0])
        value = float(loss.detach().cpu())
        losses.append(value)
        example_count += batch_size
        batch_count += 1
        for answer_type in batch["answer_types"]:
            per_answer_type.setdefault(str(answer_type), []).append(value)
    if example_count == 0:
        raise RuntimeError(f"No examples evaluated for mode {mode}")
    return {
        "mode": mode,
        "loss": float(np.mean(losses)),
        "examples": example_count,
        "batches": batch_count,
        "answer_type_loss": {key: float(np.mean(values)) for key, values in sorted(per_answer_type.items())},
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "loss", "delta_vs_normal", "examples", "batches"])
        writer.writeheader()
        writer.writerows(
            {
                "mode": row["mode"],
                "loss": row["loss"],
                "delta_vs_normal": row["delta_vs_normal"],
                "examples": row["examples"],
                "batches": row["batches"],
            }
            for row in rows
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--modes", nargs="+", default=["normal", "question_only", "image_shuffle"])
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit-batches", type=int)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    requested_device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    model, processor = load_model_and_processor(config, device)
    checkpoint_meta = load_trainable_checkpoint(args.checkpoint, model, device)

    use_bf16 = str(config.get("model", {}).get("dtype", "bf16")).lower() in {"bf16", "bfloat16"}
    rows = []
    for mode in args.modes:
        loader = create_loader(
            config=config,
            processor=processor,
            mode=mode,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
        )
        rows.append(evaluate_mode(model, loader, mode, device, use_bf16, args.limit_batches))

    normal_loss = next((row["loss"] for row in rows if row["mode"] == "normal"), None)
    for row in rows:
        row["delta_vs_normal"] = None if normal_loss is None else row["loss"] - normal_loss

    payload = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_meta": {
            "global_step": checkpoint_meta.get("global_step"),
            "best_val_loss": checkpoint_meta.get("best_val_loss"),
        },
        "seed": args.seed,
        "device": str(device),
        "max_samples": args.max_samples,
        "limit_batches": args.limit_batches,
        "modes": rows,
        "interpretation": "Positive delta_vs_normal means the visual perturbation increased teacher-forced answer loss.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(args.output.with_suffix(".csv"), rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
