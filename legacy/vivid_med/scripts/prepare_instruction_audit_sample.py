"""Create manual audit CSV samples for clinical instruction records."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


AUDIT_COLUMNS = [
    "instruction_id",
    "sample_id",
    "image_path",
    "report",
    "question",
    "answer",
    "finding",
    "state",
    "answerability",
    "uncertainty",
    "evidence_phrase",
    "evidence_source",
    "answer_type",
    "visual_dependency",
    "counterfactual_type",
    "quality_flags",
    "valid_report_supported",
    "hallucinated_location",
    "null_as_absent_error",
    "clinically_meaningful",
    "notes",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    rng = random.Random(args.seed)
    if len(rows) > args.sample_size:
        rows = rng.sample(rows, args.sample_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key, "") for key in AUDIT_COLUMNS}
            out["quality_flags"] = ";".join(row.get("quality_flags", []))
            writer.writerow(out)

    print(json.dumps({"output": str(args.output), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
