"""Evaluate instruction-model robustness to deterministic question paraphrases."""

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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import CXRInstructionDataset, instruction_collate_fn
from scripts.evaluate_instruction_counterfactual_diagnostics import score_batch
from scripts.evaluate_instruction_visual_dependence import load_checkpoint
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


def split_question_and_options(question: str) -> tuple[str, str]:
    match = re.search(r"(?m)^\s*A[\.)]\s+", question or "")
    if not match:
        return normalize_ws(question), ""
    stem = normalize_ws(question[: match.start()])
    options = question[match.start() :].strip()
    return stem, options


def clinical_rewrite(question: str) -> str:
    stem, options = split_question_and_options(question)
    rewritten = stem
    replacements = [
        (r"^Which statement best describes\\b", "Select the statement that best describes"),
        (r"^Which statement is best supported by\\b", "Choose the statement most supported by"),
        (r"^Which statement is most consistent with\\b", "Choose the statement most consistent with"),
        (r"^Which finding\\b", "Identify which finding"),
        (r"^Where are\\b", "At what location are"),
        (r"^Where is\\b", "At what location is"),
        (r"^What is the status of\\b", "How should the status of"),
        (r"^What is\\b", "State what"),
        (r"^On which side\\b", "Which side"),
        (r"^Is there evidence of\\b", "Does this chest X-ray show evidence of"),
        (r"^Is there\\b", "Does this chest X-ray show"),
        (r"^Does the report\\b", "Is the report"),
    ]
    for pattern, repl in replacements:
        new_text = re.sub(pattern, repl, rewritten, flags=re.IGNORECASE)
        if new_text != rewritten:
            rewritten = new_text
            break
    if rewritten == stem:
        rewritten = f"In this chest X-ray case, {stem[:1].lower() + stem[1:]}" if stem else question
    if options:
        return f"{rewritten}\n{options}"
    return rewritten


def style_rewrite(question: str) -> str:
    stem, options = split_question_and_options(question)
    rewritten = f"Please answer this clinical chest X-ray question: {stem}"
    if options:
        return f"{rewritten}\n{options}"
    return rewritten


def make_prompt(prompt_template: str, question: str) -> str:
    return prompt_template.format(question=question)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in sorted({row["variant"] for row in rows}):
        variant_rows = [row for row in rows if row["variant"] == variant]
        deltas = np.array([float(row["delta_vs_original"]) for row in variant_rows], dtype=np.float64)
        variant_nll = np.array([float(row["variant_nll"]) for row in variant_rows], dtype=np.float64)
        original_nll = np.array([float(row["original_nll"]) for row in variant_rows], dtype=np.float64)
        result[variant] = {
            "n": len(variant_rows),
            "original_nll_mean": float(original_nll.mean()),
            "variant_nll_mean": float(variant_nll.mean()),
            "mean_delta_vs_original": float(deltas.mean()),
            "median_delta_vs_original": float(np.median(deltas)),
            "relative_delta_vs_original": float(deltas.mean() / original_nll.mean()) if original_nll.mean() else None,
            "variant_worse_rate": float((deltas > 0).mean()),
        }
    return result


def grouped_summary(rows: list[dict[str, Any]], group_key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key) or "null")].append(row)
    return {key: summarize(value) for key, value in sorted(grouped.items())}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
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
    prompt_template = config["data"].get("prompt_template", "Question: {question}\nAnswer: ")
    model = create_model(config, str(device))
    checkpoint_meta = load_checkpoint(model, args.checkpoint, str(device))
    model.eval()
    use_bf16 = bool(config.get("training", {}).get("bf16", True))

    rows: list[dict[str, Any]] = []
    for batch in loader:
        original_scores = score_batch(
            model=model,
            images=batch["images"],
            prompt_texts=batch["prompt_texts"],
            target_texts=batch["target_jsons"],
            device=str(device),
            use_bf16=use_bf16,
        )
        questions = [str(meta.get("question") or "") for meta in batch["metadata"]]
        variants = {
            "style_rewrite": [style_rewrite(question) for question in questions],
            "clinical_rewrite": [clinical_rewrite(question) for question in questions],
        }
        for variant_name, variant_questions in variants.items():
            variant_prompts = [make_prompt(prompt_template, question) for question in variant_questions]
            variant_scores = score_batch(
                model=model,
                images=batch["images"],
                prompt_texts=variant_prompts,
                target_texts=batch["target_jsons"],
                device=str(device),
                use_bf16=use_bf16,
            )
            for idx, (original, variant) in enumerate(zip(original_scores, variant_scores)):
                rows.append(
                    {
                        "instruction_id": batch["instruction_ids"][idx],
                        "sample_id": batch["sample_ids"][idx],
                        "variant": variant_name,
                        "answer_type": batch["answer_types"][idx],
                        "visual_dependency": batch["visual_dependencies"][idx],
                        "finding": batch["findings"][idx],
                        "state": batch["states"][idx],
                        "original_nll": original["nll"],
                        "variant_nll": variant["nll"],
                        "delta_vs_original": variant["nll"] - original["nll"],
                        "original_token_count": original["token_count"],
                        "variant_token_count": variant["token_count"],
                    }
                )

    payload = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_meta": checkpoint_meta,
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
