"""Evaluate Qwen3-VL robustness to deterministic question paraphrases."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.clinical_instruction_dataset import ClinicalInstructionDataset, Qwen3VLInstructionCollator
from scripts.evaluate_qwen3vl_counterfactual_diagnostics import score_batch
from scripts.train_qwen3vl_clinical_instruction import load_model_and_processor, load_trainable_checkpoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_ws(text: Any) -> str:
    return " ".join(str(text or "").split())


def split_question_and_options(question: str) -> tuple[str, str]:
    match = re.search(r"(?m)^\s*A[\.)]\s+", question or "")
    if not match:
        return normalize_ws(question), ""
    return normalize_ws(question[: match.start()]), question[match.start() :].strip()


def clinical_rewrite(question: str) -> str:
    stem, options = split_question_and_options(question)
    rewritten = stem
    replacements = [
        (r"^Which statement best describes\b", "Select the statement that best describes"),
        (r"^Which statement is best supported by\b", "Choose the statement most supported by"),
        (r"^Which statement is most consistent with\b", "Choose the statement most consistent with"),
        (r"^Which finding\b", "Identify which finding"),
        (r"^Where are\b", "At what location are"),
        (r"^Where is\b", "At what location is"),
        (r"^What is the status of\b", "How should the status of"),
        (r"^What is\b", "State what"),
        (r"^On which side\b", "Which side"),
        (r"^Is there evidence of\b", "Does this chest X-ray show evidence of"),
        (r"^Is there\b", "Does this chest X-ray show"),
        (r"^Does the report\b", "Is the report"),
        (r"^Return the fixed UMS JSON\b", "Provide the fixed UMS JSON"),
    ]
    for pattern, repl in replacements:
        new_text = re.sub(pattern, repl, rewritten, flags=re.IGNORECASE)
        if new_text != rewritten:
            rewritten = new_text
            break
    if rewritten == stem and stem:
        rewritten = f"For this chest X-ray, {stem[:1].lower() + stem[1:]}"
    if options:
        return f"{rewritten}\n{options}"
    return rewritten


def style_rewrite(question: str) -> str:
    stem, options = split_question_and_options(question)
    rewritten = f"Please answer this clinical chest X-ray question: {stem}"
    if options:
        return f"{rewritten}\n{options}"
    return rewritten


class ListDataset(Dataset):
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]


def create_dataset(config: dict[str, Any], max_samples: int | None) -> ClinicalInstructionDataset:
    data_cfg = config["data"]
    return ClinicalInstructionDataset(
        data_root=data_cfg.get("data_root", "."),
        instruction_jsonl_path=data_cfg["val_instruction_path"],
        max_samples=max_samples if max_samples is not None else data_cfg.get("max_val_samples"),
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in sorted({row["variant"] for row in rows}):
        variant_rows = [row for row in rows if row["variant"] == variant]
        deltas = np.asarray([float(row["delta_vs_original"]) for row in variant_rows], dtype=np.float64)
        original = np.asarray([float(row["original_nll"]) for row in variant_rows], dtype=np.float64)
        variant_nll = np.asarray([float(row["variant_nll"]) for row in variant_rows], dtype=np.float64)
        result[variant] = {
            "n": len(variant_rows),
            "original_nll_mean": float(original.mean()),
            "variant_nll_mean": float(variant_nll.mean()),
            "mean_delta_vs_original": float(deltas.mean()),
            "median_delta_vs_original": float(np.median(deltas)),
            "relative_delta_vs_original": float(deltas.mean() / original.mean()) if original.mean() else None,
            "variant_worse_rate": float((deltas > 0).mean()),
        }
    return result


def grouped_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "null")].append(row)
    return {group: summarize(value) for group, value in sorted(grouped.items())}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
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
    model.eval()
    use_bf16 = str(config.get("model", {}).get("dtype", "bf16")).lower() in {"bf16", "bfloat16"}
    dataset = create_dataset(config, args.max_samples)
    collator = Qwen3VLInstructionCollator(
        processor=processor,
        max_length=config["data"].get("max_length"),
        loss_weighting=None,
        loss_masking=config.get("training", {}).get("loss_masking"),
    )

    base_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
        collate_fn=collator,
        pin_memory=True,
    )
    original_scores: dict[str, dict[str, float]] = {}
    original_meta: dict[str, dict[str, Any]] = {}
    for batch in tqdm(base_loader, desc="Scoring original"):
        scores = score_batch(model, batch, device, use_bf16)
        for idx, score in enumerate(scores):
            instruction_id = str(batch["instruction_ids"][idx])
            original_scores[instruction_id] = score
            original_meta[instruction_id] = {
                "instruction_id": instruction_id,
                "sample_id": batch["sample_ids"][idx],
                "answer_type": batch["answer_types"][idx],
                "visual_dependency": batch["visual_dependencies"][idx],
                "finding": batch["findings"][idx],
                "state": batch["states"][idx],
            }

    rows: list[dict[str, Any]] = []
    for variant_name, rewrite in {"style_rewrite": style_rewrite, "clinical_rewrite": clinical_rewrite}.items():
        items: list[dict[str, Any]] = []
        for idx in range(len(dataset)):
            item = dict(dataset[idx])
            item["question"] = rewrite(item["question"])
            items.append(item)
        loader = DataLoader(
            ListDataset(items),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collator,
            pin_memory=True,
        )
        for batch in tqdm(loader, desc=f"Scoring {variant_name}"):
            scores = score_batch(model, batch, device, use_bf16)
            for idx, variant_score in enumerate(scores):
                instruction_id = str(batch["instruction_ids"][idx])
                original = original_scores[instruction_id]
                meta = original_meta[instruction_id]
                rows.append(
                    {
                        **meta,
                        "variant": variant_name,
                        "original_nll": original["nll"],
                        "variant_nll": variant_score["nll"],
                        "delta_vs_original": variant_score["nll"] - original["nll"],
                        "original_token_count": original["token_count"],
                        "variant_token_count": variant_score["token_count"],
                    }
                )

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
        "batch_size": args.batch_size,
        "summary": {
            "overall": summarize(rows),
            "by_answer_type": grouped_summary(rows, "answer_type"),
            "by_visual_dependency": grouped_summary(rows, "visual_dependency"),
        },
        "interpretation": "Positive delta_vs_original means the paraphrased question increased teacher-forced answer NLL.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(args.output.with_name(args.output.stem + "_rows.csv"), rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
