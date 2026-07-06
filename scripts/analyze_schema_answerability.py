"""
Analyze state and answerability distributions in a UMS JSONL split.

This produces per-finding counts/rates for present, absent, uncertain, null,
and answerable fields. It is a lightweight EMNLP/base-study data diagnostic.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


STATES = ("present", "absent", "uncertain", "null")


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


def init_label_stats(labels: List[str]) -> Dict[str, Dict[str, int]]:
    return {
        label: {
            "n": 0,
            "present": 0,
            "absent": 0,
            "uncertain": 0,
            "null": 0,
            "answerable": 0,
            "unanswerable": 0,
            "uncertainty_true": 0,
        }
        for label in labels
    }


def analyze(schemas: List[Dict[str, Any]], labels: List[str]) -> Dict[str, Any]:
    per_label = init_label_stats(labels)
    overall = {
        "n_samples": len(schemas),
        "n_label_slots": len(schemas) * len(labels),
        "present": 0,
        "absent": 0,
        "uncertain": 0,
        "null": 0,
        "answerable": 0,
        "unanswerable": 0,
    }

    for schema in schemas:
        findings = schema.get("findings", {})
        answerability = schema.get("answerability", {})
        uncertainty = schema.get("uncertainty", {})
        for label in labels:
            stats = per_label[label]
            stats["n"] += 1
            item = findings.get(label, {})
            state = item.get("state") if isinstance(item, dict) else None
            state_key = "null" if state is None else str(state)
            if state_key not in STATES:
                state_key = "null"
            stats[state_key] += 1
            overall[state_key] += 1

            if bool(answerability.get(label, False)):
                stats["answerable"] += 1
                overall["answerable"] += 1
            else:
                stats["unanswerable"] += 1
                overall["unanswerable"] += 1

            if bool(uncertainty.get(label, False)):
                stats["uncertainty_true"] += 1

    per_label_rates = {}
    for label, stats in per_label.items():
        n = max(stats["n"], 1)
        per_label_rates[label] = {
            **stats,
            "answerable_rate": stats["answerable"] / n,
            "null_rate": stats["null"] / n,
            "present_rate": stats["present"] / n,
            "absent_rate": stats["absent"] / n,
            "uncertain_rate": stats["uncertain"] / n,
            "uncertainty_true_rate": stats["uncertainty_true"] / n,
        }

    slot_count = max(overall["n_label_slots"], 1)
    overall_rates = {
        **overall,
        "answerable_rate": overall["answerable"] / slot_count,
        "null_rate": overall["null"] / slot_count,
        "present_rate": overall["present"] / slot_count,
        "absent_rate": overall["absent"] / slot_count,
        "uncertain_rate": overall["uncertain"] / slot_count,
    }
    return {"overall": overall_rates, "per_label": per_label_rates}


def write_label_csv(path: Path, per_label: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "n",
        "answerable",
        "answerable_rate",
        "present",
        "present_rate",
        "absent",
        "absent_rate",
        "uncertain",
        "uncertain_rate",
        "null",
        "null_rate",
        "uncertainty_true",
        "uncertainty_true_rate",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label, stats in per_label.items():
            writer.writerow({"label": label, **{name: stats.get(name) for name in fieldnames if name != "label"}})


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze UMS answerability and state distributions")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    data_cfg = config["data"]
    labels = list(data_cfg.get("selected_labels") or [])
    ums_path = Path(data_cfg["val_ums_path"] if args.split == "val" else data_cfg["train_ums_path"])
    schemas = load_jsonl(ums_path)
    result = {
        "config": args.config,
        "split": args.split,
        "ums_path": str(ums_path),
        "selected_labels": labels,
        **analyze(schemas, labels),
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    write_label_csv(Path(args.output_csv), result["per_label"])
    print(json.dumps({"overall": result["overall"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
