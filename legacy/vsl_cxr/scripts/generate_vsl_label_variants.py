"""Derive VSL-CXR label-ablation datasets from the audited D6 VSL-4class set."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN_DIR = ROOT / "outputs" / "instruction_data" / "vsl_cxr"
FINAL_DIR = ROOT / "outputs" / "final_tables"


VARIANTS = {
    "vsl_2class": {
        "dataset_id": "D6-VSL-2class",
        "labels": {"support", "contradict"},
        "remap": {},
        "description": "support vs contradict visual-sufficiency subset derived from D6",
    },
    "vsl_3class": {
        "dataset_id": "D6-VSL-3class",
        "labels": {"support", "contradict", "uncertain"},
        "remap": {},
        "description": "support / contradict / uncertain visual-sufficiency subset derived from D6",
    },
    "vsl_4class_balanced": {
        "dataset_id": "D6-VSL-4class-balanced",
        "labels": {"support", "contradict", "uncertain", "insufficient"},
        "remap": {},
        "description": "four-class class-balanced sampling subset derived from D6",
    },
    "vsl_4class_field_balanced": {
        "dataset_id": "D6-VSL-4class-field-balanced",
        "labels": {"support", "contradict", "uncertain", "insufficient"},
        "remap": {},
        "description": "four-class finding-balanced sampling subset derived from D6",
    },
}


def root_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def repo_rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = [str(row.get(key, "")).replace("|", "\\|").replace("\n", " ") for key in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def balance_by_label(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("sufficiency_label") or ""), []).append(row)
    non_empty = [bucket for bucket in grouped.values() if bucket]
    if not non_empty:
        return []
    target = min(len(bucket) for bucket in non_empty)
    rng = random.Random(seed)
    balanced: list[dict[str, Any]] = []
    for label in sorted(grouped):
        bucket = list(grouped[label])
        rng.shuffle(bucket)
        balanced.extend(bucket[:target])
    rng.shuffle(balanced)
    return balanced


def balance_by_finding(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("finding") or "unknown"), []).append(row)
    non_empty = [bucket for bucket in grouped.values() if bucket]
    if not non_empty:
        return []
    target = min(len(bucket) for bucket in non_empty)
    rng = random.Random(seed)
    balanced: list[dict[str, Any]] = []
    for finding in sorted(grouped):
        bucket = list(grouped[finding])
        rng.shuffle(bucket)
        balanced.extend(bucket[:target])
    rng.shuffle(balanced)
    return balanced


def derive_rows(rows: list[dict[str, Any]], variant_name: str, split: str, seed: int) -> list[dict[str, Any]]:
    spec = VARIANTS[variant_name]
    allowed = spec["labels"]
    remap = spec["remap"]
    out: list[dict[str, Any]] = []
    for row in rows:
        source_label = str(row.get("sufficiency_label") or row.get("answer") or "")
        if source_label not in allowed:
            continue
        label = str(remap.get(source_label, source_label))
        new_row = dict(row)
        new_row["answer"] = label
        new_row["answer_type"] = label
        new_row["sufficiency_label"] = label
        new_row["source"] = "vsl_cxr_label_ablation_from_d6"
        new_row["source_dataset_id"] = "D6-VSL-4class"
        new_row["vsl_dataset_id"] = spec["dataset_id"]
        new_row["vsl_label_version"] = f"{variant_name}_from_d6_v1"
        quality_flags = list(new_row.get("quality_flags") or [])
        quality_flags.extend([f"{variant_name}_from_d6_v1", f"vsl_{label}"])
        new_row["quality_flags"] = sorted(set(str(flag) for flag in quality_flags))
        new_row["instruction_id"] = f"d6_{variant_name}_{split}_{len(out):07d}"
        metadata = dict(new_row.get("metadata") or {})
        metadata["source_instruction_id"] = row.get("instruction_id")
        metadata["source_sufficiency_label"] = source_label
        metadata["variant"] = variant_name
        new_row["metadata"] = metadata
        out.append(new_row)
    if variant_name == "vsl_4class_balanced":
        out = balance_by_label(out, seed=seed)
        for idx, row in enumerate(out):
            row["instruction_id"] = f"d6_{variant_name}_{split}_{idx:07d}"
    if variant_name == "vsl_4class_field_balanced":
        out = balance_by_finding(out, seed=seed)
        for idx, row in enumerate(out):
            row["instruction_id"] = f"d6_{variant_name}_{split}_{idx:07d}"
    return out


def summarize(variant_name: str, split: str, rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    counts = Counter(str(row.get("sufficiency_label")) for row in rows)
    return {
        "variant": variant_name,
        "dataset_id": VARIANTS[variant_name]["dataset_id"],
        "split": split,
        "artifact": repo_rel(path),
        "rows": len(rows),
        "support": counts.get("support", 0),
        "contradict": counts.get("contradict", 0),
        "uncertain": counts.get("uncertain", 0),
        "insufficient": counts.get("insufficient", 0),
        "findings": len(Counter(str(row.get("finding") or "unknown") for row in rows)),
        "source_dataset": "D6-VSL-4class",
        "description": VARIANTS[variant_name]["description"],
        "status": "materialized" if rows else "empty",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_IN_DIR / "d6_vsl_4class_train.jsonl")
    parser.add_argument("--val", type=Path, default=DEFAULT_IN_DIR / "d6_vsl_4class_val.jsonl")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_IN_DIR)
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=sorted(VARIANTS))
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = read_jsonl(root_path(args.train))
    val_rows = read_jsonl(root_path(args.val))
    summary_rows: list[dict[str, Any]] = []

    for variant_name in args.variants:
        variant_train = derive_rows(train_rows, variant_name, "train", seed=args.seed)
        variant_val = derive_rows(val_rows, variant_name, "val", seed=args.seed + 1)
        train_path = root_path(args.out_dir) / f"d6_{variant_name}_train.jsonl"
        val_path = root_path(args.out_dir) / f"d6_{variant_name}_val.jsonl"
        write_jsonl(train_path, variant_train)
        write_jsonl(val_path, variant_val)
        summary_rows.append(summarize(variant_name, "train", variant_train, train_path))
        summary_rows.append(summarize(variant_name, "val", variant_val, val_path))
        print(f"{variant_name}: train_rows={len(variant_train)} val_rows={len(variant_val)}")

    columns = [
        "variant",
        "dataset_id",
        "split",
        "artifact",
        "rows",
        "support",
        "contradict",
        "uncertain",
        "insufficient",
        "findings",
        "source_dataset",
        "description",
        "status",
    ]
    summary_csv = FINAL_DIR / "vsl_cxr_label_variant_manifest.csv"
    summary_md = FINAL_DIR / "vsl_cxr_label_variant_manifest.md"
    write_csv(summary_csv, summary_rows, columns)
    summary_md.write_text(
        "# VSL-CXR Label Variant Manifest\n\n"
        "Derived from the structurally audited D6 VSL-4class dataset. These rows support the Phase 2 label-ablation gates; "
        "manual correctness remains inherited from the D6 manual-audit boundary.\n\n"
        + md_table(summary_rows, columns)
        + "\n",
        encoding="utf-8",
    )
    print(f"summary={repo_rel(summary_csv)}")


if __name__ == "__main__":
    main()
