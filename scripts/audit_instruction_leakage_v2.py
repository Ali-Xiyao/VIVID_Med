"""Audit clinical instruction JSONL files for leakage and balance issues.

This script implements the next-stage "Leakage audit 2.0" gate. It is
heuristic by design: rows are flagged for review/rejection when the question
appears to reveal the answer, option order is imbalanced, evidence spans leak
into the prompt, or duplicates make template shortcuts too easy.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ANSWER_WORDS = {"yes", "no", "present", "absent", "uncertain", "null", "a", "b", "c", "d"}
LOCATION_WORDS = {"left", "right", "bilateral", "upper", "lower", "basilar", "apical", "midlung"}
SEVERITY_WORDS = {"mild", "moderate", "severe", "small", "large", "trace"}


def read_jsonl(path: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if max_rows is not None and len(rows) >= max_rows:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text(f"# {title}\n\nNo rows.\n", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    lines = [f"# {title}", "", "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def contains_phrase(container: str, phrase: Any, min_len: int = 5) -> bool:
    text = normalize_text(container)
    needle = normalize_text(phrase)
    return len(needle) >= min_len and needle in text


def option_length_imbalance(row: dict[str, Any]) -> float:
    option_a = str(row.get("option_a") or "")
    option_b = str(row.get("option_b") or "")
    if not option_a or not option_b:
        return 0.0
    longer = max(len(option_a), len(option_b))
    shorter = max(1, min(len(option_a), len(option_b)))
    return longer / shorter


def answer_leaks_from_question(row: dict[str, Any]) -> bool:
    answer = normalize_text(row.get("answer_short") or row.get("answer"))
    question = normalize_text(row.get("question"))
    answer_type = normalize_text(row.get("answer_type"))
    option_list_question = (
        {"present", "absent", "uncertain"}.issubset(set(re.findall(r"[a-z]+", question)))
        or {"definite", "uncertain"}.issubset(set(re.findall(r"[a-z]+", question)))
    )
    if answer_type in {"p2_field_query", "state_qa", "uncertainty"} and option_list_question:
        return False
    if not answer or answer in {"a", "b"}:
        return False
    if answer in ANSWER_WORDS and re.search(rf"\b{re.escape(answer)}\b", question):
        return True
    if "laterality" in answer_type or "location" in answer_type:
        return answer in LOCATION_WORDS and re.search(rf"\b{re.escape(answer)}\b", question)
    if "severity" in answer_type:
        return answer in SEVERITY_WORDS and re.search(rf"\b{re.escape(answer)}\b", question)
    return False


def row_flags(row: dict[str, Any], duplicate_count: int) -> list[str]:
    question = str(row.get("question") or "")
    answer_type = normalize_text(row.get("answer_type"))
    flags: list[str] = []
    if contains_phrase(question, row.get("evidence_span")) or contains_phrase(question, row.get("evidence_phrase")):
        flags.append("question_contains_evidence_span")
    if answer_leaks_from_question(row):
        flags.append("question_contains_exact_answer")
    if "report says" in normalize_text(question) or "according to the report" in normalize_text(question):
        flags.append("question_mentions_report_says")
    if ("laterality" in answer_type or "location" in answer_type) and row.get("location"):
        if re.search(rf"\b{re.escape(normalize_text(row.get('location')))}\b", normalize_text(question)):
            flags.append("location_question_contains_location")
    if "severity" in answer_type and row.get("severity"):
        if re.search(rf"\b{re.escape(normalize_text(row.get('severity')))}\b", normalize_text(question)):
            flags.append("severity_question_contains_severity")
    if option_length_imbalance(row) >= 1.8:
        flags.append("ab_option_length_imbalance")
    if duplicate_count > 1:
        flags.append("duplicate_question_for_image")
    if normalize_text(row.get("answer_type")) == "counterfactual_choice" and not (row.get("option_a") and row.get("option_b")):
        flags.append("counterfactual_missing_ab_options")
    return flags


def audit_rows(rows: list[dict[str, Any]], dataset_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    duplicate_counter = Counter((str(row.get("sample_id")), normalize_text(row.get("question"))) for row in rows)
    detail_rows: list[dict[str, Any]] = []
    answer_counter = Counter(normalize_text(row.get("answer_short") or row.get("answer")) for row in rows)
    flag_counter: Counter[str] = Counter()
    accepted = 0
    for index, row in enumerate(rows):
        flags = row_flags(row, duplicate_counter[(str(row.get("sample_id")), normalize_text(row.get("question")))])
        flag_counter.update(flags)
        if not flags:
            accepted += 1
        detail_rows.append(
            {
                "dataset": dataset_name,
                "row_index": index,
                "instruction_id": row.get("instruction_id", ""),
                "sample_id": row.get("sample_id", ""),
                "answer_type": row.get("answer_type", ""),
                "answer": row.get("answer_short") or row.get("answer", ""),
                "flags": ";".join(flags),
                "recommended_action": "accept" if not flags else ("reject" if any(flag.startswith("question_contains") for flag in flags) else "flag"),
            }
        )
    total = len(rows)
    a_count = answer_counter.get("a", 0)
    b_count = answer_counter.get("b", 0)
    summary_rows = [
        {
            "dataset": dataset_name,
            "instructions": total,
            "accepted": accepted,
            "accepted_pct": f"{(accepted / total * 100.0):.4f}" if total else "0",
            "leakage_or_flag_pct": f"{(((total - accepted) / total) * 100.0):.4f}" if total else "0",
            "answer_a_pct": f"{(a_count / max(1, a_count + b_count) * 100.0):.4f}",
            "duplicate_question_pairs": sum(1 for count in duplicate_counter.values() if count > 1),
        }
    ]
    for flag, count in flag_counter.most_common():
        summary_rows.append(
            {
                "dataset": dataset_name,
                "instructions": total,
                "accepted": "",
                "accepted_pct": "",
                "leakage_or_flag_pct": "",
                "answer_a_pct": "",
                "duplicate_question_pairs": "",
                "flag": flag,
                "flag_count": count,
                "flag_pct": f"{(count / total * 100.0):.4f}" if total else "0",
            }
        )
    return summary_rows, detail_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--detail-csv", type=Path)
    parser.add_argument("--summary-md", type=Path)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input, max_rows=args.max_rows)
    dataset_name = args.dataset_name or args.input.stem
    summary_rows, detail_rows = audit_rows(rows, dataset_name)
    write_csv(args.summary_csv, summary_rows)
    if args.detail_csv:
        write_csv(args.detail_csv, detail_rows)
    if args.summary_md:
        write_md(args.summary_md, "Instruction Leakage Audit 2.0", summary_rows)
    print(json.dumps({"input": str(args.input), "rows": len(rows), "summary_csv": str(args.summary_csv)}, indent=2))


if __name__ == "__main__":
    main()
