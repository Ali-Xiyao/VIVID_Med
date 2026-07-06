"""Summarize answerability/null semantics from existing artifacts."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs" / "final_tables"
FAILURE_DIR = ROOT / "outputs" / "failure_cases"

HIGH_NULL_FIELDS = ["Fracture", "Lung Lesion", "Pneumonia", "Cardiomegaly"]


@dataclass(frozen=True)
class ObjectiveSpec:
    objective: str
    method_key: str
    output_dir: str
    semantics_preserved: str
    interpretation: str


OBJECTIVES = [
    ObjectiveSpec(
        objective="no mask / null kept",
        method_key="frozen_lm_ums",
        output_dir="lp_A_ums_12label",
        semantics_preserved="partial",
        interpretation="strong baseline; null serialized in fixed UMS objective",
    ),
    ObjectiveSpec(
        objective="answerability mask",
        method_key="answerability_mask",
        output_dir="lp_ums_ansmask_12label",
        semantics_preserved="yes",
        interpretation="missingness-faithful objective; lower AUC does not by itself invalidate semantics",
    ),
    ObjectiveSpec(
        objective="null-as-negative",
        method_key="null_as_negative",
        output_dir="lp_ums_null_as_negative_12label",
        semantics_preserved="questionable",
        interpretation="dense classification baseline; changes null semantics toward absent",
    ),
]


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def metrics_for(spec: ObjectiveSpec) -> dict[str, Any]:
    path = ROOT / "outputs" / spec.output_dir / "metrics_final.json"
    payload = read_json(path)
    if payload is None:
        return {}
    metrics = payload.get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def per_field_lookup() -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv_rows(FINAL_DIR / "per_field_results.csv")
    return {(row["method_key"], row["field"]): row for row in rows}


def mean_from_lookup(
    lookup: dict[tuple[str, str], dict[str, str]],
    method_key: str,
    fields: list[str],
    metric: str,
) -> float | None:
    values = [to_float(lookup.get((method_key, field), {}).get(metric)) for field in fields]
    clean = [value for value in values if value is not None]
    return mean(clean) if clean else None


def read_answerability(path: Path) -> dict[str, dict[str, str]]:
    return {row["label"]: row for row in read_csv_rows(path)}


def has_prediction_artifacts() -> bool:
    patterns = ["*pred*.csv", "*pred*.json", "*prob*.csv", "*logit*.csv", "*oof*.csv"]
    for pattern in patterns:
        if list((ROOT / "outputs").glob(f"**/{pattern}")):
            return True
    return False


def answerability_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    lookup = per_field_lookup()
    audit_summary = read_json(ROOT / "outputs" / "schema_manual_audit_chexpert_val_200_summary.json") or {}
    prediction_available = has_prediction_artifacts()
    rows: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []

    for spec in OBJECTIVES:
        metrics = metrics_for(spec)
        if not metrics:
            missing.append(
                {
                    "artifact": f"outputs/{spec.output_dir}/metrics_final.json",
                    "field": spec.objective,
                    "reason": "core metrics missing",
                }
            )
        high_null_auc = mean_from_lookup(lookup, spec.method_key, HIGH_NULL_FIELDS, "auc")
        high_null_f1 = mean_from_lookup(lookup, spec.method_key, HIGH_NULL_FIELDS, "f1")
        if high_null_auc is None:
            missing.append(
                {
                    "artifact": "outputs/final_tables/per_field_results.csv",
                    "field": spec.objective,
                    "reason": "high-null fields unavailable for this objective",
                }
            )
        rows.append(
            {
                "Objective": spec.objective,
                "AUC": fmt(to_float(metrics.get("macro_auc"))),
                "F1": fmt(to_float(metrics.get("macro_f1"))),
                "High-null AUC": fmt(high_null_auc),
                "High-null F1": fmt(high_null_f1),
                "Mean null fields per 200-sample audit": fmt(
                    to_float(audit_summary.get("mean_null_count"))
                ),
                "Mean answerable fields per 200-sample audit": fmt(
                    to_float(audit_summary.get("mean_answerable_count"))
                ),
                "Null calibration": "available" if prediction_available else "missing_sample_predictions",
                "Predicted absent rate on null fields": (
                    "available" if prediction_available else "missing_sample_predictions"
                ),
                "Semantics preserved?": spec.semantics_preserved,
                "Interpretation": spec.interpretation,
                "Evidence boundary": (
                    "classification metrics + schema prevalence only"
                    if not prediction_available
                    else "classification metrics + prediction artifacts"
                ),
            }
        )
    if not prediction_available:
        missing.append(
            {
                "artifact": "outputs/**/*pred*|*prob*|*logit*|*oof*",
                "field": "null calibration / predicted absent rate",
                "reason": "sample-level probabilities/logits were not found",
            }
        )
    return rows, missing


def null_calibration_rows() -> list[dict[str, str]]:
    train = read_answerability(ROOT / "outputs" / "schema_answerability_chexpert_train.csv")
    val = read_answerability(ROOT / "outputs" / "schema_answerability_chexpert_val.csv")
    lookup = per_field_lookup()
    rows: list[dict[str, str]] = []
    for field in HIGH_NULL_FIELDS:
        rows.append(
            {
                "field": field,
                "train_null_rate": fmt(to_float(train.get(field, {}).get("null_rate"))),
                "val_null_rate": fmt(to_float(val.get(field, {}).get("null_rate"))),
                "train_uncertain_rate": fmt(to_float(train.get(field, {}).get("uncertain_rate"))),
                "val_uncertain_rate": fmt(to_float(val.get(field, {}).get("uncertain_rate"))),
                "no_mask_auc": fmt(to_float(lookup.get(("frozen_lm_ums", field), {}).get("auc"))),
                "answerability_mask_auc": fmt(
                    to_float(lookup.get(("answerability_mask", field), {}).get("auc"))
                ),
                "null_as_negative_auc": fmt(
                    to_float(lookup.get(("null_as_negative", field), {}).get("auc"))
                ),
                "null_as_negative_minus_ansmask_auc": fmt(
                    diff(
                        lookup.get(("null_as_negative", field), {}).get("auc"),
                        lookup.get(("answerability_mask", field), {}).get("auc"),
                    )
                ),
                "predicted_absent_rate_on_null_fields": "missing_sample_predictions",
                "calibration_status": "not_available_from_current_artifacts",
            }
        )
    return rows


def diff(left: Any, right: Any) -> float | None:
    left_f = to_float(left)
    right_f = to_float(right)
    if left_f is None or right_f is None:
        return None
    return left_f - right_f


def write_null_candidate_csv() -> int:
    audit_rows = read_csv_rows(ROOT / "outputs" / "schema_manual_audit_chexpert_val_200.csv")
    FAILURE_DIR.mkdir(parents=True, exist_ok=True)
    columns = [
        "sample_id",
        "image_path",
        "field",
        "ums_state",
        "answerable",
        "predicted_absent_probability",
        "status",
    ]
    out_rows: list[dict[str, str]] = []
    for row in audit_rows:
        for field in HIGH_NULL_FIELDS:
            state = row.get(f"{field}__state", "")
            if state != "null":
                continue
            out_rows.append(
                {
                    "sample_id": row.get("sample_id", ""),
                    "image_path": row.get("absolute_image_path") or row.get("original_path", ""),
                    "field": field,
                    "ums_state": state,
                    "answerable": row.get(f"{field}__answerable", ""),
                    "predicted_absent_probability": "",
                    "status": "candidate_null_slot; sample-level predictions missing",
                }
            )
    write_csv(FAILURE_DIR / "null_as_negative_over_absent.csv", out_rows, columns)
    return len(out_rows)


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = [row.get(column, "").replace("\n", " ") for column in columns]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body]) + "\n"


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    answer_rows, missing = answerability_rows()
    calibration_rows = null_calibration_rows()
    null_candidates = write_null_candidate_csv()

    answer_columns = [
        "Objective",
        "AUC",
        "F1",
        "High-null AUC",
        "High-null F1",
        "Mean null fields per 200-sample audit",
        "Mean answerable fields per 200-sample audit",
        "Null calibration",
        "Predicted absent rate on null fields",
        "Semantics preserved?",
        "Interpretation",
        "Evidence boundary",
    ]
    calibration_columns = [
        "field",
        "train_null_rate",
        "val_null_rate",
        "train_uncertain_rate",
        "val_uncertain_rate",
        "no_mask_auc",
        "answerability_mask_auc",
        "null_as_negative_auc",
        "null_as_negative_minus_ansmask_auc",
        "predicted_absent_rate_on_null_fields",
        "calibration_status",
    ]
    missing_columns = ["artifact", "field", "reason"]

    write_csv(FINAL_DIR / "answerability_semantics.csv", answer_rows, answer_columns)
    (FINAL_DIR / "answerability_semantics.md").write_text(
        "# Answerability Semantics\n\n" + markdown_table(answer_rows, answer_columns),
        encoding="utf-8",
    )
    write_csv(FINAL_DIR / "null_field_calibration.csv", calibration_rows, calibration_columns)
    (FINAL_DIR / "null_field_calibration.md").write_text(
        "# Null Field Calibration\n\n"
        + markdown_table(calibration_rows, calibration_columns),
        encoding="utf-8",
    )
    (FINAL_DIR / "answerability_missing_artifacts.md").write_text(
        "# Answerability Missing Artifacts\n\n"
        + (markdown_table(missing, missing_columns) if missing else "No missing artifacts.\n"),
        encoding="utf-8",
    )
    print(f"Wrote {len(answer_rows)} answerability objective rows")
    print(f"Wrote {len(calibration_rows)} null-field calibration rows")
    print(f"Wrote {null_candidates} candidate null slots")
    print(f"Recorded {len(missing)} missing/boundary issues")


if __name__ == "__main__":
    main()
