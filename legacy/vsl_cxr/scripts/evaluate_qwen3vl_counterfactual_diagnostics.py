"""Evaluate Qwen3-VL answer-type and counterfactual option NLL diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
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


OPTION_RE = re.compile(r"(?ms)^\s*([A-D])([\.)])\s*(.+?)(?=^\s*[A-D][\.)]\s*|\Z)")
LETTER_RE = re.compile(r"^\s*([A-D])(?:[\.)]?\s*)?$", re.IGNORECASE)


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


def option_from_field(letter: str, text: Any) -> dict[str, str] | None:
    option_text = normalize_ws(text)
    if not option_text:
        return None
    return {
        "letter": letter,
        "delimiter": ".",
        "text": option_text,
        "target": f"{letter}. {option_text}",
    }


def parse_options(question: str, answer: str, metadata: dict[str, Any] | None = None) -> tuple[str | None, list[dict[str, str]]]:
    options: list[dict[str, str]] = []
    for match in OPTION_RE.finditer(question or ""):
        letter, delimiter, text = match.groups()
        option_text = normalize_ws(text)
        if option_text:
            options.append(
                {
                    "letter": letter,
                    "delimiter": delimiter,
                    "text": option_text,
                    "target": f"{letter}{delimiter} {option_text}",
                }
            )
    if len(options) < 2 and metadata:
        field_options: list[dict[str, str]] = []
        for letter in ["A", "B", "C", "D"]:
            option = option_from_field(letter, metadata.get(f"option_{letter.lower()}"))
            if option:
                field_options.append(option)
        if len(field_options) >= 2:
            options = field_options
    if len(options) < 2:
        return None, options

    answer_text = normalize_ws(answer)
    letter_only_answer = False
    prefix = re.match(r"^\s*([A-D])(?:[\.)]|\s*$)", answer_text, flags=re.IGNORECASE)
    if prefix:
        letter_only_answer = bool(LETTER_RE.match(answer_text))
        correct_letter = prefix.group(1).upper()
        if letter_only_answer:
            for option in options:
                option["target"] = option["letter"]
        return correct_letter, options

    lower_answer = answer_text.lower()
    for option in options:
        if lower_answer.startswith(option["text"].lower()):
            return option["letter"], options
    for option in options:
        if option["text"].lower() in lower_answer:
            return option["letter"], options
    return None, options


class ListDataset(Dataset):
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]


def create_dataset(config: dict[str, Any], max_samples: int | None) -> ClinicalInstructionDataset:
    data_cfg = config["data"]
    val_path = data_cfg.get("val_instruction_path")
    if not val_path:
        raise ValueError("data.val_instruction_path is required")
    return ClinicalInstructionDataset(
        data_root=data_cfg.get("data_root", "."),
        instruction_jsonl_path=val_path,
        max_samples=max_samples if max_samples is not None else data_cfg.get("max_val_samples"),
    )


def create_collator(config: dict[str, Any], processor: Any) -> Qwen3VLInstructionCollator:
    return Qwen3VLInstructionCollator(
        processor=processor,
        max_length=config["data"].get("max_length"),
        loss_weighting=None,
        loss_masking=config.get("training", {}).get("loss_masking"),
    )


@torch.no_grad()
def score_batch(
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
    for idx in range(int(shift_labels.shape[0])):
        mask = valid[idx]
        token_count = int(mask.sum().item())
        if token_count == 0:
            rows.append({"nll": float("nan"), "token_count": 0})
            continue
        loss_sum = float(token_losses[idx][mask].sum().detach().cpu())
        rows.append({"nll": loss_sum / token_count, "token_count": token_count})
    return rows


def summarize_numeric(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def grouped_numeric(rows: list[dict[str, Any]], group_key: str, value_key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key) or "null")].append(row)
    return {key: summarize_numeric(value, value_key) for key, value in sorted(grouped.items())}


def summarize_counterfactual(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if int(row.get("negative_count") or 0) > 0]
    total_records = len(rows)
    no_option_records = sum(1 for row in rows if row.get("option_parse_status") == "no_options")
    correct_letter_failures = sum(1 for row in rows if row.get("option_parse_status") == "correct_letter_missing")
    if not valid_rows:
        return {
            "total_records": total_records,
            "option_formatted_records": 0,
            "no_option_records": no_option_records,
            "correct_letter_failures": correct_letter_failures,
        }
    accuracy = np.asarray([1.0 if row["correct_is_best"] else 0.0 for row in valid_rows], dtype=np.float64)
    gaps = np.asarray([float(row["best_negative_nll"]) - float(row["correct_option_nll"]) for row in valid_rows], dtype=np.float64)
    return {
        "total_records": total_records,
        "option_formatted_records": len(valid_rows),
        "no_option_records": no_option_records,
        "correct_letter_failures": correct_letter_failures,
        "pairwise_accuracy": float(accuracy.mean()),
        "mean_best_negative_minus_correct_nll": float(gaps.mean()),
        "median_best_negative_minus_correct_nll": float(np.median(gaps)),
    }


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
    collator = create_collator(config, processor)

    normal_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
        collate_fn=collator,
        pin_memory=True,
    )

    answer_rows: list[dict[str, Any]] = []
    for batch in tqdm(normal_loader, desc="Scoring answers"):
        scores = score_batch(model, batch, device, use_bf16)
        for idx, score in enumerate(scores):
            meta = batch["metadata"][idx]
            answer_rows.append(
                {
                    "instruction_id": batch["instruction_ids"][idx],
                    "sample_id": batch["sample_ids"][idx],
                    "answer_type": batch["answer_types"][idx],
                    "visual_dependency": batch["visual_dependencies"][idx],
                    "finding": batch["findings"][idx],
                    "state": batch["states"][idx],
                    "counterfactual_type": meta.get("counterfactual_type"),
                    "nll": score["nll"],
                    "token_count": score["token_count"],
                }
            )

    counterfactual_rows: list[dict[str, Any]] = []
    candidate_items: list[dict[str, Any]] = []
    candidate_meta: list[dict[str, Any]] = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        meta = item["metadata"]
        if meta.get("answer_type") != "counterfactual_choice":
            continue
        answer_for_parse = next(
            (
                value
                for value in [
                    str(meta.get("answer") or ""),
                    str(meta.get("answer_short") or ""),
                    str(meta.get("positive_option") or ""),
                ]
                if value
            ),
            "",
        )
        correct_letter, options = parse_options(str(meta.get("question") or ""), answer_for_parse, meta)
        if len(options) < 2:
            option_parse_status = "no_options"
        elif correct_letter is None:
            option_parse_status = "correct_letter_missing"
        else:
            option_parse_status = "parsed"
        base_row = {
            "instruction_id": item["instruction_id"],
            "sample_id": item["sample_id"],
            "finding": item["finding"],
            "state": item["state"],
            "visual_dependency": item["visual_dependency"],
            "counterfactual_type": meta.get("counterfactual_type"),
            "correct_letter": correct_letter,
            "option_count": len(options),
            "option_parse_status": option_parse_status,
            "parse_failed": option_parse_status != "parsed",
        }
        if option_parse_status != "parsed":
            counterfactual_rows.append(
                {
                    **base_row,
                    "negative_count": 0,
                    "correct_option_nll": None,
                    "best_negative_nll": None,
                    "correct_is_best": None,
                }
            )
            continue
        for option in options:
            candidate = dict(item)
            candidate["answer"] = option["target"]
            candidate_items.append(candidate)
            candidate_meta.append(
                {
                    **base_row,
                    "candidate_letter": option["letter"],
                    "is_correct": option["letter"] == correct_letter,
                }
            )

    if candidate_items:
        candidate_loader = DataLoader(
            ListDataset(candidate_items),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collator,
            pin_memory=True,
        )
        scored_candidates: list[dict[str, Any]] = []
        offset = 0
        for batch in tqdm(candidate_loader, desc="Scoring counterfactual options"):
            scores = score_batch(model, batch, device, use_bf16)
            for score in scores:
                scored_candidates.append({**candidate_meta[offset], **score})
                offset += 1

        by_instruction: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in scored_candidates:
            by_instruction[str(item["instruction_id"])].append(item)

        for items in by_instruction.values():
            correct_items = [item for item in items if item["is_correct"]]
            negative_items = [item for item in items if not item["is_correct"]]
            base = dict(items[0])
            if not correct_items or not negative_items:
                counterfactual_rows.append(
                    {
                        **{key: base.get(key) for key in [
                            "instruction_id",
                            "sample_id",
                            "finding",
                            "state",
                            "visual_dependency",
                            "counterfactual_type",
                            "correct_letter",
                            "option_count",
                            "option_parse_status",
                            "parse_failed",
                        ]},
                        "negative_count": len(negative_items),
                        "correct_option_nll": None,
                        "best_negative_nll": None,
                        "correct_is_best": None,
                    }
                )
                continue
            correct_nll = float(correct_items[0]["nll"])
            best_negative_nll = min(float(item["nll"]) for item in negative_items)
            best_nll = min(float(item["nll"]) for item in items)
            counterfactual_rows.append(
                {
                    "instruction_id": base["instruction_id"],
                    "sample_id": base["sample_id"],
                    "finding": base["finding"],
                    "state": base["state"],
                    "visual_dependency": base["visual_dependency"],
                    "counterfactual_type": base["counterfactual_type"],
                    "correct_letter": base["correct_letter"],
                    "option_count": base["option_count"],
                    "option_parse_status": base["option_parse_status"],
                    "parse_failed": False,
                    "negative_count": len(negative_items),
                    "correct_option_nll": correct_nll,
                    "best_negative_nll": best_negative_nll,
                    "correct_minus_best_negative_nll": correct_nll - best_negative_nll,
                    "best_negative_minus_correct_nll": best_negative_nll - correct_nll,
                    "correct_is_best": correct_nll <= best_nll,
                }
            )

    grouped_cf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in counterfactual_rows:
        grouped_cf[str(row.get("counterfactual_type") or "null")].append(row)
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
            "answer_nll": {
                "overall": summarize_numeric(answer_rows, "nll"),
                "by_answer_type": grouped_numeric(answer_rows, "answer_type", "nll"),
                "by_visual_dependency": grouped_numeric(answer_rows, "visual_dependency", "nll"),
                "by_counterfactual_type": grouped_numeric(answer_rows, "counterfactual_type", "nll"),
            },
            "counterfactual_option_nll": {
                "overall": summarize_counterfactual(counterfactual_rows),
                "by_counterfactual_type": {
                    key: summarize_counterfactual(value) for key, value in sorted(grouped_cf.items())
                },
            },
        },
        "interpretation": (
            "For answer_nll, lower is better under teacher forcing. For counterfactual_option_nll, "
            "positive best_negative_minus_correct_nll and high pairwise_accuracy mean lower NLL for the correct option."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(args.output.with_name(args.output.stem + "_answer_rows.csv"), answer_rows)
    write_csv(args.output.with_name(args.output.stem + "_counterfactual_rows.csv"), counterfactual_rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
