"""Filter and validate clinical instruction JSONL records."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


VALID_STATES = {"present", "absent", "uncertain", "null", None}
VALID_ANSWERABILITY = {"answerable", "not_answerable"}
VALID_VISUAL_DEP = {"high", "medium", "low"}


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()


def evidence_is_substring(evidence: Any, report: Any) -> bool:
    evidence_text = normalize_text(evidence)
    report_text = normalize_text(report)
    return bool(evidence_text) and evidence_text in report_text


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_state(value: Any) -> str:
    if value in {"present", "absent", "uncertain"}:
        return str(value)
    return "null"


def validation_errors(row: dict[str, Any]) -> list[str]:
    errors = []
    for field in ["instruction_id", "sample_id", "image_path", "question", "answer", "finding", "answer_type"]:
        if row.get(field) in {None, ""}:
            errors.append(f"missing_{field}")

    state = normalize_state(row.get("state"))
    if row.get("state") not in VALID_STATES:
        errors.append("invalid_state")
    if row.get("answerability") not in VALID_ANSWERABILITY:
        errors.append("invalid_answerability")
    if row.get("visual_dependency") not in VALID_VISUAL_DEP:
        errors.append("invalid_visual_dependency")

    answer = str(row.get("answer") or "").lower()
    if state == "null" and (" is absent" in answer or " absent." in answer):
        errors.append("null_as_absent_error")

    report = row.get("report")
    evidence = row.get("evidence_phrase")
    source = row.get("evidence_source")
    if source == "report_substring":
        if not report or not evidence:
            errors.append("report_substring_missing_report_or_evidence")
        elif not evidence_is_substring(evidence, report):
            errors.append("evidence_not_substring")

    if row.get("laterality") and not report:
        errors.append("laterality_without_report")
    if row.get("severity") and not report:
        errors.append("severity_without_report")

    if len(str(row.get("answer") or "").split()) > 80:
        errors.append("answer_too_long")

    return errors


def markdown_report(stats: dict[str, Any]) -> str:
    lines = [
        "# Clinical Instruction Filter Report",
        "",
        f"- Input records: {stats['input_records']}",
        f"- Kept records: {stats['kept_records']}",
        f"- Dropped records: {stats['dropped_records']}",
        f"- Duplicate records dropped: {stats['duplicate_records']}",
        "",
        "## Drop Reasons",
        "",
        "| Reason | Count |",
        "| --- | ---: |",
    ]
    for reason, count in stats["drop_reasons"].most_common():
        lines.append(f"| {reason} | {count} |")
    lines += [
        "",
        "## Kept Distributions",
        "",
        "### Answer Type",
        "",
        "| Answer type | Count |",
        "| --- | ---: |",
    ]
    for key, count in stats["answer_type"].most_common():
        lines.append(f"| {key} | {count} |")
    lines += ["", "### Quality Flags", "", "| Flag | Count |", "| --- | ---: |"]
    for key, count in stats["quality_flags"].most_common():
        lines.append(f"| {key} | {count} |")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--dropped", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    seen = set()
    drop_reasons: Counter[str] = Counter()
    duplicates = 0

    for row in rows:
        key = (row.get("sample_id"), row.get("question"), row.get("answer"))
        if key in seen:
            duplicates += 1
            drop_reasons["duplicate"] += 1
            continue
        seen.add(key)

        errors = validation_errors(row)
        if errors:
            dropped_row = dict(row)
            dropped_row["filter_errors"] = errors
            dropped.append(dropped_row)
            drop_reasons.update(errors)
            continue
        kept.append(row)

    write_jsonl(args.output, kept)
    if args.dropped:
        write_jsonl(args.dropped, dropped)

    stats = {
        "input_records": len(rows),
        "kept_records": len(kept),
        "dropped_records": len(dropped),
        "duplicate_records": duplicates,
        "drop_reasons": drop_reasons,
        "answer_type": Counter(str(row.get("answer_type")) for row in kept),
        "quality_flags": Counter(flag for row in kept for flag in row.get("quality_flags", [])),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown_report(stats), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if not isinstance(v, Counter)}, indent=2))


if __name__ == "__main__":
    main()
