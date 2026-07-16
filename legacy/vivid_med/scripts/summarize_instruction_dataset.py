"""Summarize clinical instruction JSONL datasets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_distribution_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["category", "value", "count"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def add_counter(rows: list[dict[str, Any]], category: str, counter: Counter[str]) -> None:
    for value, count in counter.most_common():
        rows.append({"category": category, "value": value, "count": count})


def md_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| Category | Value | Count |", "| --- | --- | ---: |"]
    for row in rows:
        lines.append(f"| {row['category']} | {row['value']} | {row['count']} |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--md", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    sample_ids = {str(row.get("sample_id")) for row in records}
    rows: list[dict[str, Any]] = []

    add_counter(rows, "answer_type", Counter(str(row.get("answer_type")) for row in records))
    add_counter(rows, "finding", Counter(str(row.get("finding")) for row in records))
    add_counter(rows, "state", Counter(str(row.get("state")) for row in records))
    add_counter(rows, "answerability", Counter(str(row.get("answerability")) for row in records))
    add_counter(rows, "visual_dependency", Counter(str(row.get("visual_dependency")) for row in records))
    add_counter(rows, "quality_flag", Counter(flag for row in records for flag in row.get("quality_flags", [])))
    add_counter(rows, "source_version", Counter(str(row.get("source_version")) for row in records))

    write_distribution_csv(args.csv, rows)
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(
        "\n".join(
            [
                "# Instruction Dataset Stats",
                "",
                f"- Input: `{args.input}`",
                f"- Records: {len(records)}",
                f"- Unique samples: {len(sample_ids)}",
                f"- QA per image: {len(records) / max(len(sample_ids), 1):.2f}",
                "",
                md_table(rows),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"records": len(records), "unique_samples": len(sample_ids)}, indent=2))


if __name__ == "__main__":
    main()
