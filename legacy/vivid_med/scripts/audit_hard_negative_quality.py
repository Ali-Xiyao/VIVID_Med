"""Audit hard-negative quality and possible false negatives in SHUF-style rows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from case_study_modules_common import FINAL_DIR, read_jsonl, rel, write_csv_rows, write_md_table


COLUMNS = [
    "case_study",
    "case_id",
    "instruction_id",
    "sample_id",
    "finding",
    "positive_image",
    "hard_negative_image",
    "issue",
    "risk",
    "source",
]


def audit_row(row: dict[str, Any], source: Path) -> dict[str, Any] | None:
    negative = row.get("hard_negative_image_path") or ""
    negative_ids = row.get("hard_negative_sample_ids") or []
    negative_paths = row.get("hard_negative_image_paths") or []
    if not negative and negative_paths:
        negative = negative_paths[0]
    issue = ""
    risk = "medium"
    if not negative:
        issue = "missing_hard_negative"
        risk = "high"
    elif str(row.get("sample_id")) == str(row.get("hard_negative_sample_id")):
        issue = "same_sample_negative"
        risk = "high"
    elif negative_paths and len(set(negative_paths)) < len(negative_paths):
        issue = "duplicate_negative_paths"
        risk = "medium"
    elif negative_ids and str(row.get("sample_id")) in {str(item) for item in negative_ids}:
        issue = "positive_sample_in_negative_list"
        risk = "high"
    elif str(row.get("finding") or "").lower() not in str(row.get("hard_negative_reason") or "").lower():
        issue = "needs_manual_false_negative_review"
        risk = "medium"
    if not issue:
        return None
    return {
        "case_study": "CS5_false_or_weak_hard_negative",
        "case_id": 0,
        "instruction_id": row.get("instruction_id", ""),
        "sample_id": row.get("sample_id", ""),
        "finding": row.get("finding", ""),
        "positive_image": row.get("image_path", ""),
        "hard_negative_image": negative,
        "issue": issue,
        "risk": risk,
        "source": rel(source),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="*",
        type=Path,
        default=[
            Path("outputs/instruction_data/next_stage/shuf_k4_train.jsonl"),
            Path("outputs/instruction_data/next_stage/mined_shuf_train.jsonl"),
            Path("outputs/instruction_data/next_stage/sameq_shuf_3k_train.jsonl"),
        ],
    )
    parser.add_argument("--max-rows-per-input", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output-csv", type=Path, default=FINAL_DIR / "case_study_hard_negative_quality.csv")
    parser.add_argument("--output-md", type=Path, default=FINAL_DIR / "case_study_hard_negative_quality.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases: list[dict[str, Any]] = []
    for path in args.inputs:
        for row in read_jsonl(path, max_rows=args.max_rows_per_input):
            case = audit_row(row, path)
            if case is not None:
                case["case_id"] = len(cases) + 1
                cases.append(case)
                if len(cases) >= args.limit:
                    break
        if len(cases) >= args.limit:
            break
    note = "This is an automatic data-quality audit. Rows marked for review may be acceptable after image/report inspection."
    write_csv_rows(args.output_csv, cases, COLUMNS)
    write_md_table(args.output_md, "Hard Negative Quality Audit", cases, COLUMNS, note)


if __name__ == "__main__":
    main()

