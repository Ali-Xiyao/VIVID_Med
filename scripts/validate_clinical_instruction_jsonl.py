"""Validate clinical instruction JSONL files for Qwen3-VL training."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.clinical_instruction_dataset import normalize_answer_type


ALLOWED_ANSWER_TYPES = {
    "fixed_json_schema",
    "finding_verification",
    "evidence_phrase",
    "laterality_location",
    "severity",
    "uncertainty",
    "answerability",
    "image_report_consistency",
    "counterfactual_choice",
    "hard_negative_laterality",
    "hard_negative_state",
    "temporal_comparison",
    "device_position",
}

ALLOWED_FINDINGS = {
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "global",
}

ALLOWED_STATES = {"present", "absent", "uncertain", "null", "not_applicable", None}
ALLOWED_VISUAL_DEPENDENCY = {"high", "medium", "low", None}
EVIDENCE_REQUIRED_TYPES = {
    "finding_verification",
    "evidence_phrase",
    "laterality_location",
    "severity",
    "uncertainty",
    "counterfactual_choice",
    "hard_negative_laterality",
    "hard_negative_state",
}
LATERALITY_TERMS = {
    "left": ["left"],
    "right": ["right"],
    "bilateral": ["bilateral", "both", "bilaterally"],
}
SEVERITY_TERMS = {
    "mild": ["mild", "minimal", "slight", "small", "tiny", "trace"],
    "moderate": ["moderate"],
    "severe": ["severe", "large", "marked", "extensive"],
}
ABSENT_CLAIM_PATTERNS = [
    r"\bis absent\b",
    r"\babsent\.",
    r"\bno evidence of\b",
    r"\bthere is no\b",
    r"\bnot present\b",
]


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def compact_key(value: Any) -> str:
    return re.sub(r"\W+", " ", normalize_text(value)).strip()


def first_non_empty(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in {None, ""}:
            return value
    return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                rows.append({"_line_no": line_no, "_json_error": str(exc)})
                continue
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            clean = {key: value for key, value in row.items() if not key.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False, sort_keys=True) + "\n")


def evidence_is_substring(evidence: Any, report: Any) -> bool:
    evidence_text = normalize_text(evidence)
    report_text = normalize_text(report)
    return bool(evidence_text) and evidence_text in report_text


def report_supports_any(report: Any, terms: list[str]) -> bool:
    report_text = normalize_text(report)
    return any(re.search(rf"\b{re.escape(term)}\b", report_text) for term in terms)


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized["answer_type"] = normalize_answer_type(row.get("answer_type"))
    normalized["report_text"] = first_non_empty(row, ["report_text", "report"], "")
    normalized["evidence_span"] = first_non_empty(row, ["evidence_span", "evidence_phrase"], None)
    normalized["source"] = first_non_empty(row, ["source", "source_mode"], None)
    return normalized


def validate_row(row: dict[str, Any], check_images: bool, data_root: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if row.get("_json_error"):
        return [f"json_decode_error:{row['_json_error']}"], warnings

    row = normalize_row(row)
    for field in ["sample_id", "image_path", "question", "answer", "answer_type"]:
        if first_non_empty(row, [field]) in {None, ""}:
            errors.append(f"missing_{field}")

    answer_type = row.get("answer_type")
    if answer_type not in ALLOWED_ANSWER_TYPES:
        errors.append("invalid_answer_type")

    finding = row.get("finding")
    if finding not in {None, ""} and finding not in ALLOWED_FINDINGS:
        errors.append("invalid_finding_vocab")

    if row.get("state") not in ALLOWED_STATES:
        errors.append("invalid_state")
    if row.get("visual_dependency") not in ALLOWED_VISUAL_DEPENDENCY:
        errors.append("invalid_visual_dependency")

    image_path = first_non_empty(row, ["image_path"], "")
    if check_images and image_path:
        resolved = Path(str(image_path))
        if not resolved.is_absolute():
            resolved = data_root / resolved
        if not resolved.exists():
            errors.append("image_path_missing")

    report = row.get("report_text")
    evidence = row.get("evidence_span")
    evidence_source = row.get("evidence_source")
    state = row.get("state")
    if evidence_source == "report_substring" or evidence:
        if not report:
            errors.append("evidence_without_report")
        elif not evidence_is_substring(evidence, report):
            errors.append("evidence_not_substring")

    if answer_type in EVIDENCE_REQUIRED_TYPES and state not in {"null", "not_applicable"}:
        if not report:
            warnings.append("evidence_type_without_report")
        elif not evidence:
            warnings.append("evidence_type_missing_span")

    answer = normalize_text(row.get("answer"))
    if state in {"null", "not_applicable"} and any(re.search(pattern, answer) for pattern in ABSENT_CLAIM_PATTERNS):
        errors.append("null_as_absent_error")

    laterality = normalize_text(row.get("laterality") or row.get("location"))
    if laterality in LATERALITY_TERMS and not report_supports_any(report, LATERALITY_TERMS[laterality]):
        errors.append("laterality_without_report_support")

    severity = normalize_text(row.get("severity"))
    if severity in SEVERITY_TERMS and not report_supports_any(report, SEVERITY_TERMS[severity]):
        errors.append("severity_without_report_support")

    answer_len = len(str(row.get("answer") or "").split())
    if answer_len == 0:
        errors.append("empty_answer")
    elif answer_type != "fixed_json_schema" and answer_len > 80:
        errors.append("answer_too_long")

    question_len = len(str(row.get("question") or "").split())
    if question_len == 0:
        errors.append("empty_question")
    elif question_len > 120:
        warnings.append("question_long")

    if answer_type == "counterfactual_choice":
        question = str(row.get("question") or "")
        if len(re.findall(r"\b[A-D][\).]", question)) < 2:
            warnings.append("counterfactual_choice_without_explicit_options")
        if row.get("counterfactual_type") in {None, "", "none"}:
            warnings.append("counterfactual_type_missing")

    return errors, warnings


def audit_file(
    path: Path,
    label: str,
    check_images: bool,
    data_root: Path,
    rejected_dir: Path | None,
    validated_dir: Path | None,
) -> dict[str, Any]:
    rows = read_jsonl(path)
    seen: set[tuple[str, str, str, str]] = set()
    accepted = 0
    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    error_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    answer_type_counts: Counter[str] = Counter()
    finding_counts: Counter[str] = Counter()
    visual_counts: Counter[str] = Counter()

    for row in rows:
        normalized = normalize_row(row)
        key = (
            str(normalized.get("sample_id")),
            str(normalized.get("finding")),
            str(normalized.get("answer_type")),
            compact_key(normalized.get("question")),
        )
        errors, warnings = validate_row(normalized, check_images=check_images, data_root=data_root)
        if key in seen:
            errors.append("duplicate_sample_finding_type_question")
        seen.add(key)

        if errors:
            rejected = dict(normalized)
            rejected["validation_status"] = "rejected"
            rejected["reject_reason"] = ";".join(errors)
            rejected["validation_warnings"] = warnings
            rejected_rows.append(rejected)
            error_counts.update(errors)
        else:
            accepted += 1
            accepted_row = dict(normalized)
            accepted_row["validation_status"] = "auto_validated"
            accepted_row["validation_warnings"] = warnings
            accepted_rows.append(accepted_row)
            warning_counts.update(warnings)
            answer_type_counts[str(normalized.get("answer_type"))] += 1
            finding_counts[str(normalized.get("finding"))] += 1
            visual_counts[str(normalized.get("visual_dependency"))] += 1

    if rejected_dir is not None:
        rejected_path = rejected_dir / f"{label}_rejected.jsonl"
        write_jsonl(rejected_path, rejected_rows)
    else:
        rejected_path = None
    if validated_dir is not None:
        validated_path = validated_dir / f"{label}_validated.jsonl"
        write_jsonl(validated_path, accepted_rows)
    else:
        validated_path = None

    return {
        "label": label,
        "path": str(path),
        "input_records": len(rows),
        "accepted_records": accepted,
        "rejected_records": len(rejected_rows),
        "warning_records": sum(warning_counts.values()),
        "top_error": error_counts.most_common(1)[0][0] if error_counts else "",
        "top_warning": warning_counts.most_common(1)[0][0] if warning_counts else "",
        "error_counts": dict(error_counts),
        "warning_counts": dict(warning_counts),
        "answer_type_counts": dict(answer_type_counts),
        "finding_counts": dict(finding_counts),
        "visual_dependency_counts": dict(visual_counts),
        "rejected_path": str(rejected_path) if rejected_path is not None else "",
        "validated_path": str(validated_path) if validated_path is not None else "",
    }


def write_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label",
        "path",
        "input_records",
        "accepted_records",
        "rejected_records",
        "warning_records",
        "top_error",
        "top_warning",
        "rejected_path",
        "validated_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field, "") for field in fields})


def write_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Clinical Instruction Data Audit",
        "",
        "| Split | Input | Accepted | Rejected | Warnings | Top error | Top warning |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for item in summaries:
        lines.append(
            f"| {item['label']} | {item['input_records']} | {item['accepted_records']} | "
            f"{item['rejected_records']} | {item['warning_records']} | "
            f"{item['top_error'] or '-'} | {item['top_warning'] or '-'} |"
        )
    lines.extend(["", "## Per-Split Distributions", ""])
    for item in summaries:
        lines.extend([f"### {item['label']}", "", f"- Path: `{item['path']}`", ""])
        for title, key in [
            ("Errors", "error_counts"),
            ("Warnings", "warning_counts"),
            ("Answer Types", "answer_type_counts"),
            ("Visual Dependency", "visual_dependency_counts"),
        ]:
            lines.extend([f"#### {title}", "", "| Value | Count |", "| --- | ---: |"])
            counts = Counter(item[key])
            if not counts:
                lines.append("| - | 0 |")
            else:
                for value, count in counts.most_common():
                    lines.append(f"| {value} | {count} |")
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path, help="JSONL path; repeatable.")
    parser.add_argument("--label", action="append", help="Label for each --input; repeatable.")
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/final_tables/instruction_data_audit.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("outputs/final_tables/instruction_data_audit.md"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/final_tables/instruction_data_audit.json"))
    parser.add_argument("--rejected-dir", type=Path, default=Path("outputs/instruction_data/glm_rejected"))
    parser.add_argument("--validated-dir", type=Path, default=Path("outputs/instruction_data/glm_validated"))
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--no-image-check", action="store_true")
    parser.add_argument("--allow-rejections", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = args.label or []
    if labels and len(labels) != len(args.input):
        raise SystemExit("--label count must match --input count")
    summaries = []
    for idx, input_path in enumerate(args.input):
        label = labels[idx] if labels else input_path.stem
        summaries.append(
            audit_file(
                path=input_path,
                label=label,
                check_images=not args.no_image_check,
                data_root=args.data_root,
                rejected_dir=args.rejected_dir,
                validated_dir=args.validated_dir,
            )
        )

    write_csv(args.output_csv, summaries)
    write_markdown(args.output_md, summaries)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))

    if any(item["rejected_records"] for item in summaries) and not args.allow_rejections:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
