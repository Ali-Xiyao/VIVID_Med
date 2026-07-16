"""Score hard-negative instruction rows by wrong-image teacher-forced NLL."""

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
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.clinical_instruction_dataset import ClinicalInstructionDataset, Qwen3VLInstructionCollator
from scripts.train_qwen3vl_clinical_instruction import (
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


class HardNegativeImageDataset(Dataset):
    def __init__(
        self,
        base: ClinicalInstructionDataset,
        shard_index: int | None = None,
        num_shards: int = 1,
    ) -> None:
        self.base = base
        indices = [
            idx for idx, record in enumerate(base.records)
            if record.get("hard_negative_image_path")
        ]
        if shard_index is not None or num_shards > 1:
            if num_shards < 1:
                raise ValueError("--num-shards must be >= 1")
            if shard_index is None:
                shard_index = 0
            if shard_index < 0 or shard_index >= num_shards:
                raise ValueError("--shard-index must be in [0, num_shards)")
            indices = [base_idx for pos, base_idx in enumerate(indices) if pos % num_shards == shard_index]
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        base_idx = self.indices[idx]
        item = dict(self.base[base_idx])
        record = self.base.records[base_idx]
        hard_path = Path(str(record.get("hard_negative_image_path") or ""))
        if not hard_path.is_absolute():
            hard_path = self.base.data_root / hard_path
        item["image"] = self.base._load_image(hard_path)
        item["scored_original_image_path"] = item["image_path"]
        item["image_path"] = str(hard_path)
        item["scored_hard_negative_image_path"] = str(hard_path)
        return item


@torch.no_grad()
def per_example_nll(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    use_bf16: bool,
) -> list[dict[str, float]]:
    batch = move_tensors_to_device(batch, device)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 and str(device).startswith("cuda")
        else nullcontext()
    )
    with autocast_ctx:
        outputs = model(**model_inputs(batch), use_cache=False)
    logits = outputs.logits.float()
    labels = batch["labels"]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    vocab_size = int(shift_logits.shape[-1])
    token_losses = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)
    valid = shift_labels != -100
    rows: list[dict[str, float]] = []
    for index in range(int(shift_labels.shape[0])):
        mask = valid[index]
        token_count = int(mask.sum().item())
        if token_count == 0:
            rows.append({"nll": float("nan"), "token_count": 0})
            continue
        rows.append(
            {
                "nll": float(token_losses[index][mask].mean().detach().cpu()),
                "token_count": token_count,
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize_checkpoint_meta(checkpoint: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("global_step", "best_val_loss", "model_path"):
        if key in checkpoint:
            value = checkpoint.get(key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                summary[key] = value
            else:
                summary[key] = str(value)
    parameter_groups = checkpoint.get("parameter_groups")
    if isinstance(parameter_groups, dict):
        summary["parameter_groups"] = parameter_groups
    state = checkpoint.get("trainable_state_dict")
    if isinstance(state, dict):
        summary["trainable_state_keys"] = len(state)
    return summary


def score_dataset(
    model: torch.nn.Module,
    processor: Any,
    config: dict[str, Any],
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    use_bf16: bool,
) -> list[dict[str, Any]]:
    collator = Qwen3VLInstructionCollator(
        processor=processor,
        max_length=config["data"].get("max_length"),
        loss_masking=config.get("training", {}).get("loss_masking"),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collator)
    rows: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc="Scoring hard negatives"):
        metadata = list(batch["metadata"])
        image_paths = list(batch["image_paths"])
        losses = per_example_nll(model, batch, device, use_bf16)
        for meta, image_path, loss in zip(metadata, image_paths, losses):
            rows.append(
                {
                    "instruction_id": meta.get("instruction_id"),
                    "sample_id": meta.get("sample_id"),
                    "image_path": meta.get("image_path"),
                    "finding": meta.get("finding"),
                    "state": meta.get("state"),
                    "answer": meta.get("answer") or meta.get("answer_short"),
                    "hard_negative_image_path": image_path,
                    "hard_negative_sample_id": meta.get("hard_negative_sample_id"),
                    "hard_negative_reason": meta.get("hard_negative_reason"),
                    "hard_negative_nll": loss["nll"],
                    "token_count": loss["token_count"],
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--instruction-jsonl", type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
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
    data_cfg = config["data"]
    instruction_path = args.instruction_jsonl or Path(str(data_cfg["train_instruction_path"]))
    base = ClinicalInstructionDataset(
        data_root=str(data_cfg.get("data_root", ".")),
        instruction_jsonl_path=str(instruction_path),
        max_samples=args.max_samples,
    )
    dataset = HardNegativeImageDataset(base, shard_index=args.shard_index, num_shards=args.num_shards)
    model, processor = load_model_and_processor(config, device)
    checkpoint_meta = load_trainable_checkpoint(args.checkpoint, model, device)
    use_bf16 = str(config.get("model", {}).get("dtype", "bf16")).lower() in {"bf16", "bfloat16"}
    rows = score_dataset(model, processor, config, dataset, device, args.batch_size, use_bf16)
    rows.sort(key=lambda row: float(row.get("hard_negative_nll") or float("inf")))
    write_jsonl(args.output_jsonl, rows)
    if args.output_csv:
        write_csv(args.output_csv, rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "output_jsonl": str(args.output_jsonl),
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "checkpoint_meta": summarize_checkpoint_meta(checkpoint_meta),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
