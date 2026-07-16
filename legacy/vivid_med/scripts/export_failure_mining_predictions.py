"""Export eval-only LP probabilities and mine failure cases.

This script does not train or resume models. It loads existing linear-probe
checkpoints, runs the validation split, and writes per-sample/per-label
probabilities for failure-case mining.
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
import timm
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import CheXpertUMSDataset, get_val_transforms  # noqa: E402
from data.chexpert_dataset import collate_fn  # noqa: E402


METHODS = [
    {
        "key": "frozen_lm",
        "name": "Frozen-LM UMS no-SPD LP",
        "checkpoint": "outputs/lp_A_ums_12label/best.pt",
        "config": "configs/lp_A_ums_12label.yaml",
    },
    {
        "key": "no_lm",
        "name": "no-LM UMS state-classifier LP",
        "checkpoint": "outputs/lp_ums_classifier_no_llm_12label_full/best.pt",
        "config": "configs/lp_ums_classifier_no_llm_12label.yaml",
    },
    {
        "key": "bce",
        "name": "BCE ViT-B",
        "checkpoint": "outputs/baseline_vit_full14/best.pt",
        "config": "",
    },
    {
        "key": "random_lm",
        "name": "Random-LM UMS LP",
        "checkpoint": "outputs/lp_ums_random_lm_12label/best.pt",
        "config": "configs/lp_ums_random_lm_12label.yaml",
    },
]


PREDICTION_COLUMNS = [
    "sample_id",
    "image_path",
    "label",
    "label_index",
    "ums_state",
    "answerable",
    "null_or_uncertain_flag",
    "y_true_binary",
    "frozen_lm_prob",
    "no_lm_prob",
    "bce_prob",
    "random_lm_prob",
    "frozen_lm_pred",
    "no_lm_pred",
    "bce_pred",
    "random_lm_pred",
    "frozen_lm_correct",
    "no_lm_correct",
    "bce_correct",
    "random_lm_correct",
    "confidence_gap_frozen_minus_no_lm",
]


SUMMARY_COLUMNS = [
    "failure_class",
    "count",
    "share_of_valid_binary_fields",
    "top_fields",
    "rare_count",
    "uncertain_heavy_count",
    "high_null_count",
    "example_sample_id",
    "example_image_path",
    "interpretation",
]


RARE_FIELDS = {"Fracture", "Lung Lesion", "Pneumonia"}
UNCERTAIN_HEAVY_FIELDS = {"Atelectasis", "Consolidation", "Pneumonia"}
HIGH_NULL_FIELDS = {"Fracture", "Lung Lesion", "Pneumonia", "Cardiomegaly"}


def finite(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if np.isfinite(value) else ""
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: finite(row.get(column, "")) for column in columns})


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for column in columns:
            value = finite(row.get(column, ""))
            if isinstance(value, float):
                value = f"{value:.6f}"
            values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body]) + "\n"


def choose_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    return torch.device(requested)


def build_loader(split: str, max_samples: int | None, batch_size: int, num_workers: int) -> tuple[DataLoader, list[str]]:
    if split != "val":
        raise ValueError("Only --split val is supported in this export script.")
    dataset = CheXpertUMSDataset(
        data_root="./data/dataset",
        ums_jsonl_path="./data/dataset/processed/chexpert_ums_val.jsonl",
        transform=get_val_transforms(224),
        is_train=False,
        use_common_labels_only=False,
        selected_labels=None,
        max_samples=max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, list(dataset.label_names)


def load_lp_model(checkpoint_path: Path, num_labels: int, device: torch.device) -> torch.nn.Module:
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=num_labels,
        drop_rate=0.0,
        drop_path_rate=0.0,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if state is None:
        raise KeyError(f"{checkpoint_path} does not contain checkpoint['model']")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def collect_probs(models: dict[str, torch.nn.Module], loader: DataLoader, device: torch.device) -> dict[str, Any]:
    method_probs: dict[str, list[np.ndarray]] = {key: [] for key in models}
    labels = []
    answerable = []
    sample_ids = []
    image_paths = []

    for batch in tqdm(loader, desc="Exporting LP probabilities"):
        images = batch["images"].to(device)
        labels.append(batch["labels"].numpy())
        answerable.append(batch["answerable"].numpy())
        sample_ids.extend(batch["sample_ids"])
        image_paths.extend(batch["original_paths"])
        for key, model in models.items():
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            method_probs[key].append(probs)

    return {
        "probs": {key: np.concatenate(parts, axis=0) for key, parts in method_probs.items()},
        "labels": np.concatenate(labels, axis=0),
        "answerable": np.concatenate(answerable, axis=0),
        "sample_ids": sample_ids,
        "image_paths": image_paths,
    }


def state_from_label(value: float) -> str:
    if np.isnan(value):
        return "null"
    if value == -1:
        return "uncertain"
    if value == 1:
        return "present"
    if value == 0:
        return "absent"
    return "unknown"


def true_class_confidence(prob: float, y_true: int) -> float:
    return float(prob if y_true == 1 else 1.0 - prob)


def build_prediction_rows(bundle: dict[str, Any], label_names: list[str], threshold: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = bundle["labels"]
    answerable = bundle["answerable"]
    probs = bundle["probs"]
    for sample_idx, sample_id in enumerate(bundle["sample_ids"]):
        for label_idx, label_name in enumerate(label_names):
            raw_label = float(labels[sample_idx, label_idx])
            state = state_from_label(raw_label)
            valid_binary = state in {"present", "absent"}
            y_true = int(raw_label) if valid_binary else None

            row: dict[str, Any] = {
                "sample_id": sample_id,
                "image_path": bundle["image_paths"][sample_idx],
                "label": label_name,
                "label_index": label_idx,
                "ums_state": state,
                "answerable": bool(answerable[sample_idx, label_idx]),
                "null_or_uncertain_flag": state in {"null", "uncertain"},
                "y_true_binary": y_true,
            }

            for key in ["frozen_lm", "no_lm", "bce", "random_lm"]:
                prob = float(probs[key][sample_idx, label_idx])
                pred = int(prob >= threshold)
                row[f"{key}_prob"] = prob
                row[f"{key}_pred"] = pred
                row[f"{key}_correct"] = (pred == y_true) if valid_binary else None

            if valid_binary:
                frozen_conf = true_class_confidence(row["frozen_lm_prob"], y_true)
                no_lm_conf = true_class_confidence(row["no_lm_prob"], y_true)
                row["confidence_gap_frozen_minus_no_lm"] = frozen_conf - no_lm_conf
            else:
                row["confidence_gap_frozen_minus_no_lm"] = None
            rows.append(row)
    return rows


def top_fields(rows: list[dict[str, Any]], limit: int = 5) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    return "; ".join(f"{field}:{count}" for field, count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:limit])


def summarize_case(name: str, rows: list[dict[str, Any]], valid_count: int, interpretation: str) -> dict[str, Any]:
    example = rows[0] if rows else {}
    return {
        "failure_class": name,
        "count": len(rows),
        "share_of_valid_binary_fields": (len(rows) / valid_count) if valid_count else 0.0,
        "top_fields": top_fields(rows),
        "rare_count": sum(1 for row in rows if row["label"] in RARE_FIELDS),
        "uncertain_heavy_count": sum(1 for row in rows if row["label"] in UNCERTAIN_HEAVY_FIELDS),
        "high_null_count": sum(1 for row in rows if row["label"] in HIGH_NULL_FIELDS),
        "example_sample_id": example.get("sample_id", ""),
        "example_image_path": example.get("image_path", ""),
        "interpretation": interpretation,
    }


def mine_cases(rows: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    valid = [row for row in rows if row["y_true_binary"] in {0, 1}]
    frozen_better = [
        row for row in valid
        if row["frozen_lm_correct"] is True and row["no_lm_correct"] is False
    ]
    no_lm_better = [
        row for row in valid
        if row["no_lm_correct"] is True and row["frozen_lm_correct"] is False
    ]
    all_fail = [
        row for row in valid
        if all(row[f"{key}_correct"] is False for key in ["frozen_lm", "no_lm", "bce", "random_lm"])
    ]
    random_lm_failures = [
        row for row in valid
        if row["random_lm_correct"] is False
        and any(row[f"{key}_correct"] is True for key in ["frozen_lm", "no_lm", "bce"])
    ]

    case_sets = {
        "frozen_better_than_no_lm": frozen_better,
        "no_lm_better_than_frozen": no_lm_better,
        "all_methods_fail": all_fail,
        "random_lm_failure_examples": random_lm_failures,
    }
    summary = [
        summarize_case(
            "frozen_better_than_no_lm",
            frozen_better,
            len(valid),
            "Candidate settings where frozen-LM LP fixes no-LM LP errors.",
        ),
        summarize_case(
            "no_lm_better_than_frozen",
            no_lm_better,
            len(valid),
            "Settings where no-LM LP fixes frozen-LM LP errors.",
        ),
        summarize_case(
            "all_methods_fail",
            all_fail,
            len(valid),
            "Hard binary field instances missed by all compared LP heads.",
        ),
        summarize_case(
            "random_lm_failure_examples",
            random_lm_failures,
            len(valid),
            "Instances where random-LM LP fails while at least one non-random control succeeds.",
        ),
    ]
    return case_sets, summary


def limit_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: abs(float(row.get("confidence_gap_frozen_minus_no_lm") or 0.0)),
        reverse=True,
    )[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LP probabilities for failure mining")
    parser.add_argument("--split", default="val", choices=["val"])
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--case-limit", type=int, default=500)
    args = parser.parse_args()

    for spec in METHODS:
        checkpoint = ROOT / spec["checkpoint"]
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

    device = choose_device(args.device)
    loader, label_names = build_loader(
        split=args.split,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    models = {}
    for spec in METHODS:
        models[spec["key"]] = load_lp_model(ROOT / spec["checkpoint"], len(label_names), device)

    bundle = collect_probs(models, loader, device)
    rows = build_prediction_rows(bundle, label_names, args.threshold)
    case_sets, summary_rows = mine_cases(rows)

    failure_dir = ROOT / "outputs" / "failure_cases"
    final_dir = ROOT / "outputs" / "final_tables"
    write_csv(failure_dir / "lp_failure_mining_predictions.csv", rows, PREDICTION_COLUMNS)
    for name, case_rows in case_sets.items():
        write_csv(failure_dir / f"{name}.csv", limit_rows(case_rows, args.case_limit), PREDICTION_COLUMNS)
    write_csv(final_dir / "failure_case_summary.csv", summary_rows, SUMMARY_COLUMNS)

    text = "# Failure Case Summary\n\n"
    text += (
        "Eval-only failure mining from existing LP checkpoints. Correctness is computed only for binary "
        "present/absent fields; null and uncertain fields are retained in the long prediction table but "
        "excluded from correct/incorrect case definitions.\n\n"
    )
    text += markdown_table(summary_rows, SUMMARY_COLUMNS)
    text += "\n## Provenance\n\n"
    for spec in METHODS:
        config_text = spec["config"] or "missing_config_default_full14_loader"
        text += f"- {spec['key']}: `{spec['checkpoint']}`; config `{config_text}`\n"
    text += "- Validation split: `data/dataset/processed/chexpert_ums_val.jsonl`\n"
    text += "- Label set: full CheXpert 14 labels from `CheXpertUMSDataset`.\n"
    (final_dir / "failure_case_summary.md").write_text(text, encoding="utf-8")

    manifest = {
        "split": args.split,
        "max_samples": args.max_samples,
        "threshold": args.threshold,
        "label_names": label_names,
        "methods": METHODS,
        "outputs": [
            "outputs/failure_cases/lp_failure_mining_predictions.csv",
            "outputs/failure_cases/frozen_better_than_no_lm.csv",
            "outputs/failure_cases/no_lm_better_than_frozen.csv",
            "outputs/failure_cases/all_methods_fail.csv",
            "outputs/failure_cases/random_lm_failure_examples.csv",
            "outputs/final_tables/failure_case_summary.csv",
            "outputs/final_tables/failure_case_summary.md",
        ],
        "correctness_policy": "only present/absent labels are binary-valid; null/uncertain are excluded",
    }
    (final_dir / "failure_case_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} prediction rows and {len(summary_rows)} summary rows")


if __name__ == "__main__":
    main()
