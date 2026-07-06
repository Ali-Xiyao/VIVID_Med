"""Evaluate derived answerability/uncertainty diagnostics for no-LM UMS.

The no-LM classifier is trained as a 4-state model:
null, absent, uncertain, present. This script loads an existing checkpoint and
derives schema diagnostics without training:

- answerability: target = state != null, probability = 1 - p_null
- uncertainty: target = state == uncertain, probability = p_uncertain
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import CheXpertUMSDataset, get_val_transforms  # noqa: E402
from data.chexpert_dataset import collate_fn  # noqa: E402
from evaluation.metrics import compute_classification_metrics, compute_reliability_metrics  # noqa: E402
from scripts.train_ums_classifier import (  # noqa: E402
    STATE_TO_INDEX,
    UMSStateClassifier,
    get_schema_auxiliary_targets,
    labels_to_state_targets,
    load_config,
)


FINAL_DIR = ROOT / "outputs" / "final_tables"


SUMMARY_COLUMNS = [
    "target",
    "checkpoint",
    "split",
    "n_samples",
    "n_labels",
    "support_total",
    "support_positive",
    "prevalence",
    "prob_mean",
    "pred_rate",
    "accuracy",
    "macro_f1",
    "micro_f1",
    "macro_auc",
    "brier_score",
    "ece",
    "mce",
]


PER_LABEL_COLUMNS = [
    "target",
    "label",
    "support_total",
    "support_positive",
    "prevalence",
    "prob_mean",
    "pred_rate",
    "accuracy",
    "f1",
    "precision",
    "recall",
    "auc",
]


def finite_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else ""
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: finite_value(row.get(column, "")) for column in columns})


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for column in columns:
            value = finite_value(row.get(column, ""))
            if isinstance(value, float):
                value = f"{value:.6f}"
            values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body]) + "\n"


def build_val_loader(config: dict[str, Any], split: str) -> tuple[DataLoader, list[str]]:
    if split != "val":
        raise ValueError("Only --split val is supported for this diagnostic export.")

    data_cfg = config["data"]
    dataset = CheXpertUMSDataset(
        data_root=data_cfg["data_root"],
        ums_jsonl_path=data_cfg["val_ums_path"],
        transform=get_val_transforms(data_cfg["image_size"]),
        is_train=False,
        use_common_labels_only=data_cfg.get("use_common_labels_only", False),
        selected_labels=data_cfg.get("selected_labels"),
        max_samples=data_cfg.get("max_val_samples", 1000),
        dense_subset_top_k=data_cfg.get("val_dense_top_k"),
        dense_subset_min_answerable=data_cfg.get("val_dense_min_answerable"),
    )
    loader = DataLoader(
        dataset,
        batch_size=config["training"].get("eval_batch_size", config["training"]["batch_size"]),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 0),
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, list(dataset.label_names)


def choose_device(config: dict[str, Any], requested: str | None) -> torch.device:
    value = requested or str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if value.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    if value == "cuda":
        return torch.device("cuda")
    return torch.device(value)


def load_model(config: dict[str, Any], checkpoint_path: Path, num_labels: int, device: torch.device) -> UMSStateClassifier:
    model_cfg = config["model"]
    model = UMSStateClassifier(
        model_name=model_cfg["vit_model_name"],
        pretrained=False,
        num_labels=num_labels,
        drop_rate=float(model_cfg.get("drop_rate", 0.0)),
        drop_path_rate=float(model_cfg.get("drop_path_rate", 0.1)),
        auxiliary_targets=get_schema_auxiliary_targets(config),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model")
    if state is None:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain a 'model' state_dict.")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint load mismatch: missing={missing}, unexpected={unexpected}")
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def collect_state_probs(model: UMSStateClassifier, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    all_targets = []
    all_probs = []
    for batch in tqdm(loader, desc="Evaluating no-LM derivatives"):
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=-1)
        targets = labels_to_state_targets(labels)
        all_targets.append(targets.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_targets, axis=0), np.concatenate(all_probs, axis=0)


def summarize_target(
    *,
    name: str,
    target: np.ndarray,
    prob: np.ndarray,
    label_names: list[str],
    checkpoint: Path,
    split: str,
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y_true = target.astype(float)
    y_prob = prob.astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        label_names=label_names,
        threshold=threshold,
    )
    reliability = compute_reliability_metrics(y_true=y_true, y_prob=y_prob)

    summary = {
        "target": name,
        "checkpoint": checkpoint.as_posix(),
        "split": split,
        "n_samples": int(y_true.shape[0]),
        "n_labels": int(y_true.shape[1]),
        "support_total": int(y_true.size),
        "support_positive": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "prob_mean": float(y_prob.mean()),
        "pred_rate": float(y_pred.mean()),
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1": metrics.get("macro_f1"),
        "micro_f1": metrics.get("micro_f1"),
        "macro_auc": metrics.get("macro_auc"),
        "brier_score": reliability.get("brier_score"),
        "ece": reliability.get("ece"),
        "mce": reliability.get("mce"),
    }

    per_label_rows = []
    per_label = metrics.get("per_label", {})
    for index, label in enumerate(label_names):
        label_metrics = per_label.get(label, {})
        label_true = y_true[:, index]
        label_prob = y_prob[:, index]
        label_pred = y_pred[:, index]
        per_label_rows.append(
            {
                "target": name,
                "label": label,
                "support_total": int(label_true.size),
                "support_positive": int(label_true.sum()),
                "prevalence": float(label_true.mean()),
                "prob_mean": float(label_prob.mean()),
                "pred_rate": float(label_pred.mean()),
                "accuracy": label_metrics.get("accuracy"),
                "f1": label_metrics.get("f1"),
                "precision": label_metrics.get("precision"),
                "recall": label_metrics.get("recall"),
                "auc": label_metrics.get("auc"),
            }
        )
    return summary, per_label_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate derived no-LM schema diagnostics")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["val"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not Path(config["data"]["val_ums_path"]).exists():
        raise FileNotFoundError(config["data"]["val_ums_path"])

    threshold = float(args.threshold if args.threshold is not None else config.get("evaluation", {}).get("threshold", 0.5))
    device = choose_device(config, args.device)
    loader, label_names = build_val_loader(config, args.split)
    model = load_model(config, checkpoint, len(label_names), device)
    state_true, state_prob = collect_state_probs(model, loader, device)

    answerability_target = (state_true != STATE_TO_INDEX["null"]).astype(int)
    answerability_prob = 1.0 - state_prob[..., STATE_TO_INDEX["null"]]
    uncertainty_target = (state_true == STATE_TO_INDEX["uncertain"]).astype(int)
    uncertainty_prob = state_prob[..., STATE_TO_INDEX["uncertain"]]

    summary_rows = []
    per_label_rows = []
    for name, target, prob in [
        ("answerability", answerability_target, answerability_prob),
        ("uncertainty", uncertainty_target, uncertainty_prob),
    ]:
        summary, label_rows = summarize_target(
            name=name,
            target=target,
            prob=prob,
            label_names=label_names,
            checkpoint=checkpoint,
            split=args.split,
            threshold=threshold,
        )
        summary_rows.append(summary)
        per_label_rows.extend(label_rows)

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(FINAL_DIR / "no_lm_schema_derivative_metrics.csv", summary_rows, SUMMARY_COLUMNS)
    write_csv(FINAL_DIR / "no_lm_schema_derivative_per_label.csv", per_label_rows, PER_LABEL_COLUMNS)

    text = "# no-LM Derived Schema Metrics\n\n"
    text += (
        "Validation-only diagnostics from the existing no-LM 4-state checkpoint. "
        "These metrics are derived from 4-state logits and are not separate S2/S3 training objectives.\n\n"
    )
    text += "## Summary\n\n"
    text += markdown_table(summary_rows, SUMMARY_COLUMNS)
    text += "\n## Boundary\n\n"
    text += (
        "- answerability uses `1 - p_null` and target `state != null`.\n"
        "- uncertainty uses `p_uncertain` and target `state == uncertain`.\n"
        "- These rows can support answerability/uncertainty diagnostics, not a claim that no-LM has explicit S2/S3 schema serialization.\n"
    )
    (FINAL_DIR / "no_lm_schema_derivative_metrics.md").write_text(text, encoding="utf-8")

    manifest = {
        "config": args.config,
        "checkpoint": checkpoint.as_posix(),
        "split": args.split,
        "threshold": threshold,
        "state_to_index": STATE_TO_INDEX,
        "outputs": [
            "outputs/final_tables/no_lm_schema_derivative_metrics.csv",
            "outputs/final_tables/no_lm_schema_derivative_metrics.md",
            "outputs/final_tables/no_lm_schema_derivative_per_label.csv",
        ],
    }
    (FINAL_DIR / "no_lm_schema_derivative_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote derived schema metrics for {len(label_names)} labels to {FINAL_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
