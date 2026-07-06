"""Evaluate instruction answer-type and counterfactual-choice NLL diagnostics."""

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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import CXRInstructionDataset, instruction_collate_fn
from scripts.evaluate_instruction_visual_dependence import load_checkpoint
from scripts.train_cxr_instruction import create_model


OPTION_RE = re.compile(
    r"(?ms)^\s*([A-D])([\.)])\s*(.+?)(?=^\s*[A-D][\.)]\s*|\Z)"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_eval_loader(config: dict[str, Any], max_samples: int | None, batch_size: int) -> DataLoader:
    data_cfg = config["data"]
    instruction_path = data_cfg.get("val_instruction_path")
    if not instruction_path:
        raise ValueError("config data.val_instruction_path is required")

    dataset = CXRInstructionDataset(
        data_root=data_cfg["data_root"],
        instruction_jsonl_path=instruction_path,
        image_size=int(data_cfg.get("image_size", 224)),
        is_train=False,
        max_samples=max_samples if max_samples is not None else data_cfg.get("max_val_samples"),
        prompt_template=data_cfg.get("prompt_template", "Question: {question}\nAnswer: "),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=instruction_collate_fn,
        pin_memory=True,
    )


def normalize_ws(text: str) -> str:
    return " ".join(str(text or "").split())


def parse_options(question: str, answer: str) -> tuple[str | None, list[dict[str, str]]]:
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

    if len(options) < 2:
        return None, options

    answer_text = normalize_ws(answer)
    prefix = re.match(r"^\s*([A-D])[\.)]", answer_text)
    if prefix:
        return prefix.group(1), options

    lower_answer = answer_text.lower()
    for option in options:
        if lower_answer.startswith(option["text"].lower()):
            return option["letter"], options
    for option in options:
        if option["text"].lower() in lower_answer:
            return option["letter"], options
    return None, options


@torch.no_grad()
def score_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    prompt_texts: list[str],
    target_texts: list[str],
    device: str,
    use_bf16: bool,
) -> list[dict[str, float]]:
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 and str(device).startswith("cuda")
        else nullcontext()
    )
    with autocast_ctx:
        output = model(
            images=images.to(device, non_blocking=True),
            prompt_text=prompt_texts,
            target_text=target_texts,
        )

    logits = output["logits"].float()
    labels = output["labels"]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    batch_size, seq_len, vocab_size = shift_logits.shape

    token_losses = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(batch_size, seq_len)
    valid = shift_labels != -100

    rows: list[dict[str, float]] = []
    for idx in range(batch_size):
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
    arr = np.array(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def grouped_summary(rows: list[dict[str, Any]], group_key: str, value_key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key) or "null")].append(row)
    return {key: summarize_numeric(value, value_key) for key, value in sorted(grouped.items())}


def summarize_counterfactual(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if row.get("negative_count", 0) > 0]
    total_records = len(rows)
    no_option_records = sum(1 for row in rows if row.get("option_parse_status") == "no_options")
    correct_letter_failures = sum(
        1 for row in rows if row.get("option_parse_status") == "correct_letter_missing"
    )
    if not valid_rows:
        return {
            "total_records": total_records,
            "option_formatted_records": 0,
            "no_option_records": no_option_records,
            "correct_letter_failures": correct_letter_failures,
        }
    accuracies = np.array([1.0 if row["correct_is_best"] else 0.0 for row in valid_rows], dtype=np.float64)
    gaps = np.array([float(row["best_negative_nll"]) - float(row["correct_option_nll"]) for row in valid_rows])
    return {
        "total_records": total_records,
        "option_formatted_records": len(valid_rows),
        "no_option_records": no_option_records,
        "correct_letter_failures": correct_letter_failures,
        "pairwise_accuracy": float(accuracies.mean()),
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
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device")
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
    model.eval()
    use_bf16 = bool(config.get("training", {}).get("bf16", True))

    answer_rows: list[dict[str, Any]] = []
    counterfactual_rows: list[dict[str, Any]] = []

    for batch in loader:
        normal_scores = score_batch(
            model=model,
            images=batch["images"],
            prompt_texts=batch["prompt_texts"],
            target_texts=batch["target_jsons"],
            device=str(device),
            use_bf16=use_bf16,
        )
        for idx, score in enumerate(normal_scores):
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

        candidate_images: list[torch.Tensor] = []
        candidate_prompts: list[str] = []
        candidate_targets: list[str] = []
        candidate_meta: list[dict[str, Any]] = []

        for idx, meta in enumerate(batch["metadata"]):
            if meta.get("answer_type") != "counterfactual_choice":
                continue
            correct_letter, options = parse_options(str(meta.get("question") or ""), str(meta.get("answer") or ""))
            if len(options) < 2:
                option_parse_status = "no_options"
            elif correct_letter is None:
                option_parse_status = "correct_letter_missing"
            else:
                option_parse_status = "parsed"
            base_row = {
                "instruction_id": batch["instruction_ids"][idx],
                "sample_id": batch["sample_ids"][idx],
                "finding": batch["findings"][idx],
                "state": batch["states"][idx],
                "visual_dependency": batch["visual_dependencies"][idx],
                "counterfactual_type": meta.get("counterfactual_type"),
                "correct_letter": correct_letter,
                "option_count": len(options),
                "option_parse_status": option_parse_status,
                "parse_failed": option_parse_status != "parsed",
            }
            if base_row["parse_failed"]:
                row = dict(base_row)
                row.update(
                    {
                        "negative_count": 0,
                        "correct_option_nll": None,
                        "best_negative_nll": None,
                        "correct_is_best": None,
                    }
                )
                counterfactual_rows.append(row)
                continue

            for option in options:
                candidate_images.append(batch["images"][idx])
                candidate_prompts.append(batch["prompt_texts"][idx])
                candidate_targets.append(option["target"])
                candidate_meta.append(
                    {
                        **base_row,
                        "candidate_letter": option["letter"],
                        "is_correct": option["letter"] == correct_letter,
                    }
                )

        if candidate_targets:
            images = torch.stack(candidate_images)
            scores = score_batch(
                model=model,
                images=images,
                prompt_texts=candidate_prompts,
                target_texts=candidate_targets,
                device=str(device),
                use_bf16=use_bf16,
            )
            by_instruction: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for meta, score in zip(candidate_meta, scores):
                item = dict(meta)
                item.update(score)
                by_instruction[str(item["instruction_id"])].append(item)

            for items in by_instruction.values():
                correct_items = [item for item in items if item["is_correct"]]
                negative_items = [item for item in items if not item["is_correct"]]
                if not correct_items or not negative_items:
                    base = dict(items[0])
                    counterfactual_rows.append(
                        {
                            key: base.get(key)
                            for key in [
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
                            ]
                        }
                        | {
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
                base = dict(correct_items[0])
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

    summary = {
        "answer_nll": {
            "overall": summarize_numeric(answer_rows, "nll"),
            "by_answer_type": grouped_summary(answer_rows, "answer_type", "nll"),
            "by_visual_dependency": grouped_summary(answer_rows, "visual_dependency", "nll"),
            "by_counterfactual_type": grouped_summary(answer_rows, "counterfactual_type", "nll"),
        },
        "counterfactual_option_nll": {
            "overall": summarize_counterfactual(counterfactual_rows),
            "by_counterfactual_type": {
                key: summarize_counterfactual(value)
                for key, value in sorted(
                    (
                        (group_key, group_rows)
                        for group_key, group_rows in defaultdict(list, {}).items()
                    )
                )
            },
        },
    }

    grouped_cf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in counterfactual_rows:
        grouped_cf[str(row.get("counterfactual_type") or "null")].append(row)
    summary["counterfactual_option_nll"]["by_counterfactual_type"] = {
        key: summarize_counterfactual(value) for key, value in sorted(grouped_cf.items())
    }

    payload = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_meta": checkpoint_meta,
        "seed": args.seed,
        "device": str(device),
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "summary": summary,
        "interpretation": (
            "For answer_nll, lower is better under teacher forcing. "
            "For counterfactual_option_nll, positive best_negative_minus_correct_nll "
            "and high pairwise_accuracy mean the model gives lower NLL to the correct option."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(args.output.with_name(args.output.stem + "_answer_rows.csv"), answer_rows)
    write_csv(args.output.with_name(args.output.stem + "_counterfactual_rows.csv"), counterfactual_rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
