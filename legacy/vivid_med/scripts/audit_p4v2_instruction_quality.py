"""Audit P4-v2 D6/D7 instruction quality, leakage, and hard-shuffle pairs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md_table(path: Path, title: str, rows: list[dict[str, Any]], columns: list[str], note: str = "") -> None:
    lines = [f"# {title}", ""]
    if note:
        lines.extend([note, ""])
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").casefold()).strip()


def audit_row(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    reject: list[str] = []
    flag: list[str] = []
    question = norm(row.get("question"))
    evidence = norm(row.get("evidence_span"))
    answer = norm(row.get("answer_short") or row.get("answer"))
    if evidence and evidence in question:
        reject.append("question_contains_evidence_span")
    if "report says" in question or "report mentions" in question or "according to the report" in question:
        reject.append("question_mentions_report")
    if row.get("answer_type") == "counterfactual_choice":
        options = re.findall(r"\b([AB])\.\s*([^\n]+)", str(row.get("question") or ""))
        if len(options) < 2:
            reject.append("ab_options_missing")
        if answer not in {"a", "b"}:
            reject.append("answer_not_ab")
        if len(options) >= 2:
            lengths = [len(text.split()) for _, text in options[:2]]
            if min(lengths) > 0 and max(lengths) / min(lengths) > 2.5:
                flag.append("ab_option_length_imbalance")
    elif answer in {"yes", "no"} and answer in question.split():
        flag.append("question_contains_yes_no_token")
    finding = norm(row.get("finding"))
    if finding and row.get("answer_type") in {"finding_verification", "uncertainty"} and finding not in question:
        flag.append("question_missing_finding")
    return reject, flag


def summarize_dataset(label: str, path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    rejected = 0
    flagged = 0
    reject_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    answer_counter = Counter(str(row.get("answer_short") or row.get("answer")) for row in rows)
    ab_total = answer_counter.get("A", 0) + answer_counter.get("B", 0)
    for row in rows:
        reject, flag = audit_row(row)
        if reject:
            rejected += 1
            reject_counts.update(reject)
        if flag:
            flagged += 1
            flag_counts.update(flag)
    hard_negative_attached = sum(1 for row in rows if row.get("hard_negative_image_path"))
    return {
        "dataset": label,
        "path": str(path),
        "records": len(rows),
        "images": len({str(row.get("sample_id")) for row in rows}),
        "avg_qa_per_image": round(len(rows) / max(1, len({str(row.get("sample_id")) for row in rows})), 4),
        "accepted_percent": round(100 * (len(rows) - rejected) / max(1, len(rows)), 4),
        "rejected_percent": round(100 * rejected / max(1, len(rows)), 4),
        "leakage_percent": round(100 * reject_counts.get("question_contains_evidence_span", 0) / max(1, len(rows)), 4),
        "flagged_percent": round(100 * flagged / max(1, len(rows)), 4),
        "ab_answer_a_percent": round(100 * answer_counter.get("A", 0) / max(1, ab_total), 4) if ab_total else "",
        "hard_negative_attached_percent": round(100 * hard_negative_attached / max(1, len(rows)), 4),
        "top_reject": reject_counts.most_common(1)[0][0] if reject_counts else "",
        "top_flag": flag_counts.most_common(1)[0][0] if flag_counts else "",
    }


def distribution_rows(label: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for field in ["answer_type", "counterfactual_type", "answer_short", "finding", "state", "location", "hard_negative_reason"]:
        counter = Counter(str(row.get(field) or "none") for row in rows)
        for value, count in counter.most_common():
            out.append({"dataset": label, "field": field, "value": value, "count": count})
    return out


def manual_template_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows[:limit], start=1):
        out.append(
            {
                "audit_id": idx,
                "sample_id": row.get("sample_id"),
                "question": row.get("question"),
                "answer": row.get("answer_short") or row.get("answer"),
                "evidence_span": row.get("evidence_span"),
                "finding": row.get("finding"),
                "answer_type": row.get("answer_type"),
                "visual_dependency": row.get("visual_dependency"),
                "Correct?": "",
                "Leakage?": "",
                "Hallucination?": "",
                "Notes": "",
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--summary-csv", type=Path, default=Path("outputs/final_tables/instruction_leakage_audit.csv"))
    parser.add_argument("--summary-md", type=Path, default=Path("outputs/final_tables/instruction_leakage_audit.md"))
    parser.add_argument("--distribution-csv", type=Path, default=Path("outputs/final_tables/instruction_distribution.csv"))
    parser.add_argument("--distribution-md", type=Path, default=Path("outputs/final_tables/instruction_distribution.md"))
    parser.add_argument("--manual-template", type=Path, default=Path("outputs/final_tables/manual_audit_template.csv"))
    parser.add_argument("--d6-audit-csv", type=Path, default=Path("outputs/final_tables/d6_instruction_audit.csv"))
    parser.add_argument("--d7-audit-csv", type=Path, default=Path("outputs/final_tables/d7_shuffle_pair_audit.csv"))
    parser.add_argument("--manual-limit", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.input) != len(args.label):
        raise SystemExit("--input and --label counts must match")
    summaries: list[dict[str, Any]] = []
    distributions: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    d6_rows: list[dict[str, Any]] = []
    d7_rows: list[dict[str, Any]] = []
    for label, path in zip(args.label, args.input):
        rows = read_jsonl(path)
        summaries.append(summarize_dataset(label, path, rows))
        distributions.extend(distribution_rows(label, rows))
        all_rows.extend(rows)
        if label.lower().startswith("d6"):
            d6_rows.extend(rows)
        if label.lower().startswith("d7"):
            d7_rows.extend(rows)
    summary_cols = [
        "dataset",
        "records",
        "images",
        "avg_qa_per_image",
        "accepted_percent",
        "rejected_percent",
        "leakage_percent",
        "flagged_percent",
        "ab_answer_a_percent",
        "hard_negative_attached_percent",
        "top_reject",
        "top_flag",
        "path",
    ]
    write_csv(args.summary_csv, summaries, summary_cols)
    write_md_table(args.summary_md, "P4-v2 Instruction Leakage Audit", summaries, summary_cols)
    dist_cols = ["dataset", "field", "value", "count"]
    write_csv(args.distribution_csv, distributions, dist_cols)
    write_md_table(args.distribution_md, "P4-v2 Instruction Distribution", distributions, dist_cols)
    write_csv(args.manual_template, manual_template_rows(all_rows, args.manual_limit))
    if d6_rows:
        write_csv(args.d6_audit_csv, [summarize_dataset("D6-all", Path("combined"), d6_rows)], summary_cols)
    if d7_rows:
        write_csv(args.d7_audit_csv, [summarize_dataset("D7-all", Path("combined"), d7_rows)], summary_cols)
    print(json.dumps({"summaries": summaries, "distribution_rows": len(distributions)}, indent=2))


if __name__ == "__main__":
    main()
