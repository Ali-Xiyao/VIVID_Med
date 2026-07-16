"""Generate VSL-CXR four-class visual sufficiency labels.

The generator is conservative: it derives support/uncertain/insufficient rows
from structured UMS JSON supervision and support/contradict rows from validated
counterfactual A/B instructions. It does not claim manual correctness; it
creates auditable v5-format candidates and summary tables for the next gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "outputs" / "instruction_data" / "vsl_cxr"
FINAL_DIR = ROOT / "outputs" / "final_tables"
FINDINGS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]
VSL_LABELS = ("support", "contradict", "uncertain", "insufficient")


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
    if not path.exists():
        return rows
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


def parse_answer_json(row: dict[str, Any]) -> dict[str, Any] | None:
    answer = row.get("answer")
    if isinstance(answer, dict):
        return answer
    if not isinstance(answer, str) or not answer.strip().startswith("{"):
        return None
    try:
        return json.loads(answer)
    except json.JSONDecodeError:
        return None


def finding_text(finding: str) -> str:
    return str(finding).replace("_", " ").strip().lower()


def state_statement(finding: str, state: str | None, laterality: str | None = None, severity: str | None = None) -> str:
    desc = finding_text(finding)
    prefix = ""
    if severity:
        prefix += f"{severity} "
    if laterality:
        prefix += f"{laterality} "
    if finding == "Support Devices":
        if state == "absent":
            return "No support device is visible."
        if state == "uncertain":
            return "Support device visibility is uncertain."
        return "A support device is visible."
    if state == "absent":
        return f"There is no {desc}."
    if state == "uncertain":
        return f"There is possible {prefix}{desc}."
    return f"There is {prefix}{desc}."


def insufficient_statement(finding: str) -> str:
    return f"{finding} is not visually answerable from this chest X-ray."


def make_vsl_row(
    *,
    source_row: dict[str, Any],
    split: str,
    index: int,
    label: str,
    statement: str,
    finding: str,
    state: str | None,
    source_version: str,
    evidence_span: Any = None,
    counterfactual_statement: str | None = None,
    laterality: Any = None,
    severity: Any = None,
    negative_image_path: Any = None,
    negative_type: str | None = None,
    visual_dependency: str | None = None,
) -> dict[str, Any]:
    sample_id = str(source_row.get("sample_id") or source_row.get("instruction_id") or f"sample_{index}")
    row_id = f"d6_vsl4_{split}_{index:07d}"
    return {
        "sample_id": sample_id,
        "image_path": source_row.get("image_path"),
        "statement": statement,
        "question": "Does the chest X-ray provide sufficient visual evidence to support this clinical statement?",
        "answer": label,
        "answer_type": label,
        "finding": finding,
        "state": state,
        "laterality": laterality,
        "severity": severity,
        "evidence_span": evidence_span,
        "counterfactual_statement": counterfactual_statement,
        "sufficiency_label": label,
        "visual_dependency": visual_dependency or source_row.get("visual_dependency") or "medium",
        "negative_image_path": negative_image_path,
        "negative_type": negative_type,
        "source": "vsl_cxr_heuristic_from_validated_supervision",
        "generation_model": source_row.get("generation_model") or (source_row.get("metadata") or {}).get("fact_model"),
        "validation_status": "auto_vsl_candidate",
        "source_instruction_id": source_row.get("instruction_id"),
        "source_version": source_version,
        "vsl_dataset_id": "D6",
        "vsl_label_version": "vsl4_heuristic_v1",
        "quality_flags": sorted(set(list(source_row.get("quality_flags") or []) + ["vsl4_heuristic_v1", f"vsl_{label}"])),
        "report_text": source_row.get("report_text") or source_row.get("report") or "",
        "metadata": {
            "split": split,
            "source_answer_type": source_row.get("answer_type"),
            "source_state": source_row.get("state"),
            "source_certainty": source_row.get("certainty"),
        },
        "instruction_id": row_id,
    }


def rows_from_structured(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        payload = parse_answer_json(row)
        if not payload:
            continue
        findings = payload.get("findings") or {}
        answerability = payload.get("answerability") or {}
        uncertainty = payload.get("uncertainty") or {}
        for finding in FINDINGS:
            info = findings.get(finding) or {}
            state = info.get("state")
            is_answerable = bool(answerability.get(finding))
            uncertainty_state = uncertainty.get(finding)
            if is_answerable and state in {"present", "absent"}:
                label = "support"
                statement = state_statement(finding, state)
            elif is_answerable and (state == "uncertain" or uncertainty_state == "uncertain"):
                label = "uncertain"
                statement = state_statement(finding, "uncertain")
            elif not is_answerable or state in {None, "null"}:
                label = "insufficient"
                statement = insufficient_statement(finding)
                state = "null"
            else:
                continue
            out.append(
                make_vsl_row(
                    source_row=row,
                    split=split,
                    index=len(out),
                    label=label,
                    statement=statement,
                    finding=finding,
                    state=state,
                    source_version="structured_ums_fixed_json",
                    evidence_span=row.get("evidence_span"),
                    visual_dependency="medium" if label == "insufficient" else "high",
                )
            )
    return out


def rows_from_counterfactual(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("answer_type") != "counterfactual_choice":
            continue
        positive = str(row.get("positive_option") or row.get("answer_short") or row.get("answer") or "").upper()
        negative = str(row.get("negative_option") or ("B" if positive == "A" else "A")).upper()
        option_map = {
            "A": row.get("option_a"),
            "B": row.get("option_b"),
            "C": row.get("option_c"),
            "D": row.get("option_d"),
        }
        positive_statement = option_map.get(positive)
        negative_statement = option_map.get(negative)
        if positive_statement:
            label = "uncertain" if row.get("state") == "uncertain" or row.get("certainty") == "uncertain" else "support"
            out.append(
                make_vsl_row(
                    source_row=row,
                    split=split,
                    index=len(out),
                    label=label,
                    statement=str(positive_statement),
                    finding=str(row.get("finding") or "global"),
                    state="uncertain" if label == "uncertain" else row.get("state"),
                    source_version="counterfactual_positive_option",
                    evidence_span=row.get("evidence_span"),
                    counterfactual_statement=str(negative_statement) if negative_statement else None,
                    laterality=row.get("laterality"),
                    severity=row.get("severity"),
                    visual_dependency=row.get("visual_dependency") or "high",
                )
            )
        if negative_statement:
            out.append(
                make_vsl_row(
                    source_row=row,
                    split=split,
                    index=len(out),
                    label="contradict",
                    statement=str(negative_statement),
                    finding=str(row.get("finding") or "global"),
                    state=row.get("state"),
                    source_version="counterfactual_negative_option",
                    evidence_span=row.get("evidence_span"),
                    counterfactual_statement=str(positive_statement) if positive_statement else None,
                    laterality=row.get("laterality"),
                    severity=row.get("severity"),
                    negative_type=str(row.get("negative_option_source") or row.get("counterfactual_type") or "counterfactual_option"),
                    visual_dependency=row.get("visual_dependency") or "high",
                )
            )
    return out


def balanced_sample(rows: list[dict[str, Any]], max_per_class: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("sufficiency_label"))].append(row)
    selected: list[dict[str, Any]] = []
    for label in VSL_LABELS:
        bucket = grouped.get(label, [])
        rng.shuffle(bucket)
        selected.extend(bucket[:max_per_class])
    rng.shuffle(selected)
    for idx, row in enumerate(selected):
        row["instruction_id"] = f"d6_vsl4_balanced_{idx:07d}"
    return selected


def summarize(split: str, rows: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    label_counts = Counter(str(row.get("sufficiency_label")) for row in rows)
    answer_counts = Counter(str(row.get("answer_type")) for row in rows)
    finding_counts = Counter(str(row.get("finding")) for row in rows)
    summary = [
        {
            "split": split,
            "artifact": repo_rel(path),
            "rows": len(rows),
            "support": label_counts.get("support", 0),
            "contradict": label_counts.get("contradict", 0),
            "uncertain": label_counts.get("uncertain", 0),
            "insufficient": label_counts.get("insufficient", 0),
            "answer_types": json.dumps(answer_counts, ensure_ascii=False, sort_keys=True),
            "top_findings": json.dumps(dict(finding_counts.most_common(8)), ensure_ascii=False),
            "status": "materialized" if rows else "empty",
        }
    ]
    return summary


def write_manual_audit_template(path: Path, rows: list[dict[str, Any]], seed: int, n: int) -> None:
    rng = random.Random(seed)
    sample = list(rows)
    rng.shuffle(sample)
    sample = sample[: min(n, len(sample))]
    columns = [
        "audit_id",
        "dataset",
        "sample_id",
        "statement",
        "sufficiency_label",
        "image_path",
        "negative_image_path",
        "evidence_span",
        "finding",
        "leakage?",
        "correct?",
        "hard_neg_valid?",
        "note",
    ]
    audit_rows = []
    for idx, row in enumerate(sample, start=1):
        audit_rows.append(
            {
                "audit_id": idx,
                "dataset": "D6 VSL-4class",
                "sample_id": row.get("sample_id"),
                "statement": row.get("statement"),
                "sufficiency_label": row.get("sufficiency_label"),
                "image_path": row.get("image_path"),
                "negative_image_path": row.get("negative_image_path"),
                "evidence_span": row.get("evidence_span"),
                "finding": row.get("finding"),
                "leakage?": "",
                "correct?": "",
                "hard_neg_valid?": "",
                "note": "",
            }
        )
    write_csv(path, audit_rows, columns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structured-train", type=Path, default=ROOT / "outputs/instruction_data/glm_validated/d0_train_validated.jsonl")
    parser.add_argument("--structured-val", type=Path, default=ROOT / "outputs/instruction_data/glm_validated/d0_val_validated.jsonl")
    parser.add_argument("--cf-train", type=Path, default=ROOT / "outputs/instruction_data/glm_validated/d6_hard_cf_3k.jsonl")
    parser.add_argument("--cf-val", type=Path, default=ROOT / "outputs/instruction_data/glm_validated/d6_hard_cf_val200.jsonl")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-per-class-train", type=int, default=3000)
    parser.add_argument("--max-per-class-val", type=int, default=400)
    parser.add_argument("--manual-audit-n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    structured_train = read_jsonl(root_path(args.structured_train))
    structured_val = read_jsonl(root_path(args.structured_val))
    cf_train = read_jsonl(root_path(args.cf_train))
    cf_val = read_jsonl(root_path(args.cf_val))

    train_candidates = rows_from_structured(structured_train, "train") + rows_from_counterfactual(cf_train, "train")
    val_candidates = rows_from_structured(structured_val, "val") + rows_from_counterfactual(cf_val, "val")
    train_rows = balanced_sample(train_candidates, args.max_per_class_train, args.seed)
    val_rows = balanced_sample(val_candidates, args.max_per_class_val, args.seed + 1)

    train_path = root_path(args.out_dir) / "d6_vsl_4class_train.jsonl"
    val_path = root_path(args.out_dir) / "d6_vsl_4class_val.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    summary_rows = summarize("train", train_rows, train_path) + summarize("val", val_rows, val_path)
    summary_columns = [
        "split",
        "artifact",
        "rows",
        "support",
        "contradict",
        "uncertain",
        "insufficient",
        "answer_types",
        "top_findings",
        "status",
    ]
    summary_csv = FINAL_DIR / "vsl_cxr_d6_dataset_summary.csv"
    summary_md = FINAL_DIR / "vsl_cxr_d6_dataset_summary.md"
    write_csv(summary_csv, summary_rows, summary_columns)
    summary_md.write_text(
        "# VSL-CXR D6 VSL-4class Dataset Summary\n\n"
        "Generated from structured UMS fixed-json rows and validated counterfactual A/B rows. "
        "This is an auto-labeled candidate dataset and requires manual audit before formal claims.\n\n"
        + md_table(summary_rows, summary_columns)
        + "\n",
        encoding="utf-8",
    )
    write_manual_audit_template(FINAL_DIR / "vsl_cxr_d6_manual_audit_template.csv", train_rows + val_rows, args.seed, args.manual_audit_n)

    print(f"train_rows={len(train_rows)}")
    print(f"val_rows={len(val_rows)}")
    print(f"train={repo_rel(train_path)}")
    print(f"val={repo_rel(val_path)}")
    print(f"summary={repo_rel(summary_csv)}")


if __name__ == "__main__":
    main()
