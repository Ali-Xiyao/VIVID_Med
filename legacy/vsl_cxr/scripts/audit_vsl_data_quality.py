"""Audit VSL-CXR JSONL data quality for the v5 visual-sufficiency schema."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs" / "final_tables"
LABELS = {"support", "contradict", "uncertain", "insufficient"}
REQUIRED_FIELDS = [
    "sample_id",
    "image_path",
    "question",
    "answer",
    "answer_type",
    "finding",
    "visual_dependency",
    "source",
    "validation_status",
]


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
        for line_no, line in enumerate(handle, start=1):
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


def resolve_image(path_value: Any, data_root: Path) -> Path:
    path = Path(str(path_value or ""))
    return path if path.is_absolute() else data_root / path


def validate_row(
    row: dict[str, Any],
    seen_ids: set[str],
    check_images: bool,
    data_root: Path,
    schema: str,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if row.get("_json_error"):
        return [f"json_decode_error:{row['_json_error']}"], warnings

    for field in REQUIRED_FIELDS:
        if row.get(field) in {None, ""}:
            errors.append(f"missing_{field}")

    label = str(row.get("sufficiency_label") or "")
    if schema == "vsl4" or label:
        if row.get("statement") in {None, ""}:
            errors.append("missing_statement")
        if label not in LABELS:
            errors.append("invalid_sufficiency_label")
        if str(row.get("answer_type") or "") != label:
            errors.append("answer_type_label_mismatch")
        if str(row.get("answer") or "") != label:
            errors.append("answer_label_mismatch")
    elif schema == "d9":
        if row.get("d9_component") in {None, ""}:
            errors.append("missing_d9_component")
        if row.get("source_dataset_id") in {None, ""}:
            errors.append("missing_source_dataset_id")

    instruction_id = str(row.get("instruction_id") or "")
    if instruction_id:
        if instruction_id in seen_ids:
            errors.append("duplicate_instruction_id")
        seen_ids.add(instruction_id)
    else:
        warnings.append("missing_instruction_id")

    statement = str(row.get("statement") or "").strip()
    if label and len(statement.split()) < 3:
        errors.append("statement_too_short")
    if label == "insufficient":
        marker = statement.casefold()
        if "insufficient" not in marker and "not visually answerable" not in marker:
            warnings.append("insufficient_statement_lacks_boundary_phrase")
    if label == "contradict" and not row.get("counterfactual_statement"):
        warnings.append("contradict_without_counterfactual_statement")
    if label in {"support", "contradict"} and row.get("state") in {None, "", "null", "not_applicable"}:
        warnings.append("support_or_contradict_without_concrete_state")
    if label == "uncertain" and row.get("state") not in {"uncertain", None, ""}:
        warnings.append("uncertain_label_state_not_uncertain")

    if check_images:
        image_path = resolve_image(row.get("image_path"), data_root)
        if not image_path.exists():
            errors.append("image_path_missing")
        neg = row.get("negative_image_path")
        if neg not in {None, ""} and not resolve_image(neg, data_root).exists():
            errors.append("negative_image_path_missing")

    return errors, warnings


def audit_file(
    path: Path,
    split: str,
    check_images: bool,
    data_root: Path,
    schema: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = read_jsonl(path)
    seen_ids: set[str] = set()
    label_counts: Counter[str] = Counter()
    component_counts: Counter[str] = Counter()
    finding_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    detail_rows: list[dict[str, Any]] = []
    accepted = 0

    for row in rows:
        label = str(row.get("sufficiency_label") or "")
        label_counts[label] += 1
        component_counts[str(row.get("d9_component") or "")] += 1
        finding_counts[str(row.get("finding") or "")] += 1
        errors, warnings = validate_row(row, seen_ids, check_images, data_root, schema)
        error_counts.update(errors)
        warning_counts.update(warnings)
        if not errors:
            accepted += 1
        if errors or warnings:
            detail_rows.append(
                {
                    "split": split,
                    "line_no": row.get("_line_no"),
                    "instruction_id": row.get("instruction_id"),
                    "sample_id": row.get("sample_id"),
                    "sufficiency_label": label,
                    "finding": row.get("finding"),
                    "errors": ";".join(errors),
                    "warnings": ";".join(warnings),
                    "statement": row.get("statement"),
                }
            )

    expected = [label_counts.get(label, 0) for label in sorted(LABELS)]
    nonzero = [value for value in expected if value > 0]
    balance_ratio = (min(nonzero) / max(nonzero)) if nonzero else 0.0
    summary = {
        "split": split,
        "artifact": repo_rel(path),
        "rows": len(rows),
        "accepted_rows": accepted,
        "error_rows": len(rows) - accepted,
        "warning_rows": len(detail_rows),
        "support": label_counts.get("support", 0),
        "contradict": label_counts.get("contradict", 0),
        "uncertain": label_counts.get("uncertain", 0),
        "insufficient": label_counts.get("insufficient", 0),
        "class_balance_ratio": f"{balance_ratio:.4f}",
        "top_findings": json.dumps(dict(finding_counts.most_common(8)), ensure_ascii=False),
        "components": json.dumps(dict(component_counts.most_common(12)), ensure_ascii=False, sort_keys=True),
        "errors": json.dumps(error_counts, ensure_ascii=False, sort_keys=True),
        "warnings": json.dumps(warning_counts, ensure_ascii=False, sort_keys=True),
        "status": "accepted" if accepted == len(rows) and (schema == "d9" or balance_ratio >= 0.45) else "needs_review",
    }
    return summary, detail_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "outputs/instruction_data/vsl_cxr/d6_vsl_4class_train.jsonl")
    parser.add_argument("--val", type=Path, default=ROOT / "outputs/instruction_data/vsl_cxr/d6_vsl_4class_val.jsonl")
    parser.add_argument("--summary-csv", type=Path, default=FINAL_DIR / "vsl_cxr_data_quality_summary.csv")
    parser.add_argument("--summary-md", type=Path, default=FINAL_DIR / "vsl_cxr_data_quality_summary.md")
    parser.add_argument("--detail-csv", type=Path, default=FINAL_DIR / "vsl_cxr_data_quality_detail.csv")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "dataset")
    parser.add_argument("--schema", choices=["vsl4", "d9"], default="vsl4")
    parser.add_argument("--check-images", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for split, path in [("train", root_path(args.train)), ("val", root_path(args.val))]:
        summary, detail = audit_file(path, split, args.check_images, root_path(args.data_root), args.schema)
        summaries.append(summary)
        detail_rows.extend(detail)

    summary_columns = [
        "split",
        "artifact",
        "rows",
        "accepted_rows",
        "error_rows",
        "warning_rows",
        "support",
        "contradict",
        "uncertain",
        "insufficient",
        "class_balance_ratio",
        "top_findings",
        "components",
        "errors",
        "warnings",
        "status",
    ]
    detail_columns = ["split", "line_no", "instruction_id", "sample_id", "sufficiency_label", "finding", "errors", "warnings", "statement"]
    write_csv(root_path(args.summary_csv), summaries, summary_columns)
    write_csv(root_path(args.detail_csv), detail_rows, detail_columns)
    root_path(args.summary_md).write_text(
        "# VSL-CXR Data Quality Summary\n\n"
        "This structural audit checks v5 four-class schema validity. It does not replace manual correctness review.\n\n"
        + md_table(summaries, summary_columns)
        + "\n",
        encoding="utf-8",
    )
    for summary in summaries:
        print(f"{summary['split']}: rows={summary['rows']} accepted={summary['accepted_rows']} status={summary['status']}")
    print(f"summary={repo_rel(root_path(args.summary_csv))}")
    print(f"detail={repo_rel(root_path(args.detail_csv))}")
    if args.fail_on_error and any(int(summary["error_rows"]) for summary in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
