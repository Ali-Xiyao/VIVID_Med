"""
Export a small schema manual-audit sheet from a UMS JSONL file.

The output is a CSV with image paths, metadata, per-finding states, and
answerability flags. It is intended as an EMNLP-style audit artifact; it does
not replace human review, but prepares a reproducible review set.
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def state_counts(schema: Dict[str, Any], labels: List[str]) -> Dict[str, int]:
    counts = {"present": 0, "absent": 0, "uncertain": 0, "null": 0}
    findings = schema.get("findings", {})
    for label in labels:
        item = findings.get(label, {})
        state = item.get("state") if isinstance(item, dict) else None
        if state is None:
            counts["null"] += 1
        else:
            counts[str(state)] = counts.get(str(state), 0) + 1
    return counts


def audit_row(schema: Dict[str, Any], data_root: Path, labels: List[str]) -> Dict[str, Any]:
    extensions = schema.get("extensions", {})
    original_path = extensions.get("original_path", "")
    findings = schema.get("findings", {})
    answerability = schema.get("answerability", {})
    uncertainty = schema.get("uncertainty", {})
    counts = state_counts(schema, labels)

    row: Dict[str, Any] = {
        "sample_id": extensions.get("sample_id"),
        "dataset": extensions.get("dataset"),
        "original_path": original_path,
        "absolute_image_path": str(data_root / original_path) if original_path else "",
        "patient_sex": extensions.get("patient_sex"),
        "patient_age": extensions.get("patient_age"),
        "study_view": schema.get("study_view"),
        "present_count": counts["present"],
        "absent_count": counts["absent"],
        "uncertain_count": counts["uncertain"],
        "null_count": counts["null"],
        "answerable_count": sum(1 for label in labels if bool(answerability.get(label, False))),
        "audit_notes": "",
    }

    for label in labels:
        item = findings.get(label, {})
        state = item.get("state") if isinstance(item, dict) else None
        row[f"{label}__state"] = "null" if state is None else state
        row[f"{label}__answerable"] = bool(answerability.get(label, False))
        row[f"{label}__uncertain"] = uncertainty.get(label)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a UMS schema manual-audit sheet")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--output-summary", type=str, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_yaml(args.config)
    data_cfg = config["data"]
    data_root = Path(data_cfg["data_root"])
    labels = list(data_cfg.get("selected_labels") or [])
    ums_path_key = "val_ums_path" if args.split == "val" else "train_ums_path"
    ums_path = Path(data_cfg[ums_path_key])

    schemas = load_jsonl(ums_path)
    rng = random.Random(args.seed)
    if args.sample_size < len(schemas):
        selected = rng.sample(schemas, args.sample_size)
    else:
        selected = schemas

    rows = [audit_row(schema, data_root=data_root, labels=labels) for schema in selected]
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    aggregate = {
        "config": args.config,
        "ums_path": str(ums_path),
        "split": args.split,
        "sample_size": len(rows),
        "seed": args.seed,
        "selected_labels": labels,
        "mean_answerable_count": sum(row["answerable_count"] for row in rows) / max(len(rows), 1),
        "mean_null_count": sum(row["null_count"] for row in rows) / max(len(rows), 1),
        "mean_present_count": sum(row["present_count"] for row in rows) / max(len(rows), 1),
        "mean_uncertain_count": sum(row["uncertain_count"] for row in rows) / max(len(rows), 1),
    }
    output_summary = Path(args.output_summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(output_summary, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)

    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
