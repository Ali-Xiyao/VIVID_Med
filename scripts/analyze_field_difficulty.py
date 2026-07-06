"""Analyze field difficulty groups from existing VIVID-Med metrics."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "final_tables"


@dataclass(frozen=True)
class MethodSpec:
    method: str
    key: str
    output_dir: str


METHODS = [
    MethodSpec("Data-matched BCE ViT-B", "bce", "baseline_vit_full14"),
    MethodSpec("Frozen-LM UMS / no-SPD", "frozen_lm_ums", "lp_A_ums_12label"),
    MethodSpec(
        "no-LM UMS state classifier",
        "no_lm_ums",
        "lp_ums_classifier_no_llm_12label_full",
    ),
    MethodSpec("Frozen-LM UMS + SPD default", "spd_default", "lp_A_ums_spd_12label"),
    MethodSpec("Frozen-LM UMS + SPD G=2", "spd_g2", "lp_spd_g2_12label"),
    MethodSpec("Frozen-LM free-text target", "free_text", "lp_A_freetext_12label"),
    MethodSpec("Random-mask proxy", "random_mask", "lp_random_mask_12label"),
    MethodSpec("BiomedCLIP baseline", "biomedclip", "lp_biomedclip_baseline_seed0"),
    MethodSpec(
        "Frozen-LM UMS + answerability mask",
        "answerability_mask",
        "lp_ums_ansmask_12label",
    ),
    MethodSpec(
        "Frozen-LM UMS + null-as-negative",
        "null_as_negative",
        "lp_ums_null_as_negative_12label",
    ),
    MethodSpec(
        "Random-LM same-architecture UMS",
        "random_lm",
        "lp_ums_random_lm_12label",
    ),
]


FIELD_GROUPS = {
    "common": ["Pleural Effusion", "Lung Opacity", "Support Devices"],
    "rare": ["Fracture", "Lung Lesion", "Pneumonia"],
    "uncertain-heavy": ["Atelectasis", "Consolidation", "Pneumonia"],
    "high-null": ["Fracture", "Lung Lesion", "Pneumonia", "Cardiomegaly"],
}


PRIMARY_KEYS = ["bce", "no_lm_ums", "frozen_lm_ums", "random_lm"]


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


def read_answerability(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["label"]: row for row in csv.DictReader(handle)}


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


def method_metrics(spec: MethodSpec) -> dict[str, dict[str, Any]]:
    path = ROOT / "outputs" / spec.output_dir / "metrics_final.json"
    payload = read_json(path)
    if payload is None:
        return {}
    metrics = payload.get("metrics", {})
    per_label = metrics.get("per_label", {})
    return per_label if isinstance(per_label, dict) else {}


def group_membership(field: str) -> str:
    groups = [group for group, fields in FIELD_GROUPS.items() if field in fields]
    return ";".join(groups)


def build_per_field_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train_stats = read_answerability(ROOT / "outputs" / "schema_answerability_chexpert_train.csv")
    val_stats = read_answerability(ROOT / "outputs" / "schema_answerability_chexpert_val.csv")
    rows: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []

    for spec in METHODS:
        per_label = method_metrics(spec)
        if not per_label:
            missing.append(
                {
                    "method": spec.method,
                    "artifact": f"outputs/{spec.output_dir}/metrics_final.json",
                    "field": "metrics.per_label",
                    "reason": "missing or unparsable per-label metrics",
                }
            )
            continue
        for field, metrics in per_label.items():
            if not isinstance(metrics, dict):
                continue
            train = train_stats.get(field, {})
            val = val_stats.get(field, {})
            row = {
                "method": spec.method,
                "method_key": spec.key,
                "field": field,
                "field_group": group_membership(field),
                "auc": fmt(to_float(metrics.get("auc"))),
                "f1": fmt(to_float(metrics.get("f1"))),
                "accuracy": fmt(to_float(metrics.get("accuracy"))),
                "support": str(metrics.get("support", "")),
                "train_present_rate": fmt(to_float(train.get("present_rate"))),
                "train_null_rate": fmt(to_float(train.get("null_rate"))),
                "train_uncertain_rate": fmt(to_float(train.get("uncertain_rate"))),
                "train_answerable_rate": fmt(to_float(train.get("answerable_rate"))),
                "val_present_rate": fmt(to_float(val.get("present_rate"))),
                "val_null_rate": fmt(to_float(val.get("null_rate"))),
                "val_uncertain_rate": fmt(to_float(val.get("uncertain_rate"))),
                "val_answerable_rate": fmt(to_float(val.get("answerable_rate"))),
                "metrics_path": rel(ROOT / "outputs" / spec.output_dir / "metrics_final.json"),
            }
            rows.append(row)
            if field in {item for fields in FIELD_GROUPS.values() for item in fields} and not train:
                missing.append(
                    {
                        "method": spec.method,
                        "artifact": "outputs/schema_answerability_chexpert_train.csv",
                        "field": field,
                        "reason": "field is in difficulty group but missing train answerability stats",
                    }
                )
    return rows, missing


def metric_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["method_key"], row["field"]): row for row in rows}


def mean_metric(
    lookup: dict[tuple[str, str], dict[str, str]],
    method_key: str,
    fields: list[str],
    metric: str,
) -> float | None:
    values = [
        to_float(lookup.get((method_key, field), {}).get(metric))
        for field in fields
    ]
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return mean(clean)


def build_group_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    lookup = metric_lookup(rows)
    grouped: list[dict[str, str]] = []
    for group, fields in FIELD_GROUPS.items():
        values = {
            key: mean_metric(lookup, key, fields, "auc")
            for key in PRIMARY_KEYS
        }
        f1_values = {
            key: mean_metric(lookup, key, fields, "f1")
            for key in PRIMARY_KEYS
        }
        frozen = values.get("frozen_lm_ums")
        no_lm = values.get("no_lm_ums")
        bce = values.get("bce")
        random_lm = values.get("random_lm")
        grouped.append(
            {
                "field_group": group,
                "fields": "; ".join(fields),
                "bce_auc": fmt(bce),
                "no_lm_auc": fmt(no_lm),
                "frozen_lm_auc": fmt(frozen),
                "random_lm_auc": fmt(random_lm),
                "frozen_minus_no_lm_auc": fmt(
                    frozen - no_lm if frozen is not None and no_lm is not None else None
                ),
                "frozen_minus_bce_auc": fmt(
                    frozen - bce if frozen is not None and bce is not None else None
                ),
                "frozen_minus_random_lm_auc": fmt(
                    frozen - random_lm
                    if frozen is not None and random_lm is not None
                    else None
                ),
                "bce_f1": fmt(f1_values.get("bce")),
                "no_lm_f1": fmt(f1_values.get("no_lm_ums")),
                "frozen_lm_f1": fmt(f1_values.get("frozen_lm_ums")),
                "random_lm_f1": fmt(f1_values.get("random_lm")),
                "interpretation": interpret_group(frozen, no_lm, bce, random_lm),
            }
        )
    return grouped


def interpret_group(
    frozen: float | None,
    no_lm: float | None,
    bce: float | None,
    random_lm: float | None,
) -> str:
    if frozen is None or no_lm is None:
        return "insufficient metrics"
    delta = frozen - no_lm
    if delta >= 0.02:
        return "frozen-LM clearly better than no-LM on this group"
    if delta >= 0.01:
        return "frozen-LM has a modest group-specific advantage"
    if delta > -0.005:
        return "frozen-LM and no-LM are close"
    if random_lm is not None and frozen - random_lm >= 0.02:
        return "no-LM leads frozen-LM, but pretrained LM remains above random-LM"
    return "no-LM leads or frozen-LM advantage is weak"


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


def write_summary(group_rows: list[dict[str, str]], missing: list[dict[str, str]]) -> None:
    text = "# Field Difficulty Summary\n\n"
    text += (
        "This analysis uses existing per-label CheXpert metrics and schema "
        "answerability statistics only. It does not use sample-level logits, so "
        "confidence-gap failure mining remains a separate P1 task.\n\n"
    )
    for row in group_rows:
        text += (
            f"- {row['field_group']}: frozen-LM AUC {row['frozen_lm_auc']}, "
            f"no-LM AUC {row['no_lm_auc']}, BCE AUC {row['bce_auc']}, "
            f"random-LM AUC {row['random_lm_auc']}; "
            f"frozen-noLM delta {row['frozen_minus_no_lm_auc']}. "
            f"{row['interpretation']}.\n"
        )
    text += "\n## Missing/Boundary Notes\n\n"
    if missing:
        for row in missing:
            text += f"- {row['method']} / {row['field']}: {row['reason']} ({row['artifact']}).\n"
    else:
        text += "- No required field/group metric artifacts were missing.\n"
    text += (
        "- `No Finding` and `Pleural Other` appear in model metrics but are not "
        "part of the 12-field UMS answerability schema used for difficulty groups.\n"
    )
    (OUTPUT_DIR / "field_difficulty_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_field_rows, missing = build_per_field_rows()
    group_rows = build_group_rows(per_field_rows)

    per_field_columns = [
        "method",
        "method_key",
        "field",
        "field_group",
        "auc",
        "f1",
        "accuracy",
        "support",
        "train_present_rate",
        "train_null_rate",
        "train_uncertain_rate",
        "train_answerable_rate",
        "val_present_rate",
        "val_null_rate",
        "val_uncertain_rate",
        "val_answerable_rate",
        "metrics_path",
    ]
    group_columns = [
        "field_group",
        "fields",
        "bce_auc",
        "no_lm_auc",
        "frozen_lm_auc",
        "random_lm_auc",
        "frozen_minus_no_lm_auc",
        "frozen_minus_bce_auc",
        "frozen_minus_random_lm_auc",
        "bce_f1",
        "no_lm_f1",
        "frozen_lm_f1",
        "random_lm_f1",
        "interpretation",
    ]
    missing_columns = ["method", "artifact", "field", "reason"]

    write_csv(OUTPUT_DIR / "per_field_results.csv", per_field_rows, per_field_columns)
    (OUTPUT_DIR / "per_field_results.md").write_text(
        "# Per-Field Results\n\n" + markdown_table(per_field_rows, per_field_columns),
        encoding="utf-8",
    )
    write_csv(OUTPUT_DIR / "grouped_field_results.csv", group_rows, group_columns)
    (OUTPUT_DIR / "grouped_field_results.md").write_text(
        "# Grouped Field Results\n\n" + markdown_table(group_rows, group_columns),
        encoding="utf-8",
    )
    write_csv(OUTPUT_DIR / "field_difficulty_missing_artifacts.csv", missing, missing_columns)
    (OUTPUT_DIR / "field_difficulty_missing_artifacts.md").write_text(
        "# Field Difficulty Missing Artifacts\n\n"
        + (
            markdown_table(missing, missing_columns)
            if missing
            else "No required field difficulty artifacts were missing.\n"
        ),
        encoding="utf-8",
    )
    write_summary(group_rows, missing)
    print(f"Wrote {len(per_field_rows)} per-field rows")
    print(f"Wrote {len(group_rows)} grouped field rows")
    print(f"Recorded {len(missing)} missing/boundary issues")


if __name__ == "__main__":
    main()
