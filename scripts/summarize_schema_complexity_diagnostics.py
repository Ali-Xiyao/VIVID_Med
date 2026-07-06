"""Consolidate schema-complexity diagnostics for the revision plan."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs" / "final_tables"


COLUMNS = [
    "evidence_id",
    "pathway",
    "schema_level",
    "artifact_type",
    "scope",
    "metric_target",
    "support",
    "primary_metric",
    "primary_value",
    "secondary_metrics",
    "interpretation",
    "claim_boundary",
    "evidence_paths",
]


def read_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def exists(rel_path: str) -> bool:
    return (ROOT / rel_path).exists()


def required(rel_path: str) -> str:
    if not exists(rel_path):
        raise FileNotFoundError(ROOT / rel_path)
    return rel_path


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    text = str(value)
    try:
        return f"{float(text):.6f}"
    except ValueError:
        return text


def step_extrema(rel_dir: str) -> tuple[dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in (ROOT / rel_dir).glob("metrics_step_*.json"):
        step = int(path.stem.rsplit("_", 1)[-1])
        payload = read_json(path)
        metrics = payload["metrics"]
        rows.append(
            {
                "step": step,
                "val_loss": payload["val_loss"],
                "macro_auc": metrics.get("macro_auc"),
                "macro_f1": metrics.get("macro_f1"),
                "micro_f1": metrics.get("micro_f1"),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No metrics_step_*.json files under {rel_dir}")
    best_loss = min(rows, key=lambda row: row["val_loss"])
    best_auc = max(rows, key=lambda row: row["macro_auc"] if row["macro_auc"] is not None else float("-inf"))
    return best_loss, best_auc


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(row.get(column, "") for column in columns) + " |")
    return "\n".join([header, divider, *body]) + "\n"


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_rows() -> list[dict[str, str]]:
    metrics_path = Path(required("outputs/final_tables/no_lm_schema_derivative_metrics.csv"))
    no_lm_metrics = {row["target"]: row for row in read_dicts(ROOT / metrics_path)}
    frozen_source_path = Path(required("outputs/final_tables/frozen_lm_source_training_summary.csv"))
    frozen_source = {row["run_id"]: row for row in read_dicts(ROOT / frozen_source_path)}

    s2_best = required("outputs/schema_sweep/frozen_lm_s2_state_answerability_seed900122/checkpoints/best.pt")
    s2_final = required("outputs/schema_sweep/frozen_lm_s2_state_answerability_seed900122/checkpoints/final.pt")
    s3_best = required("outputs/schema_sweep/frozen_lm_s3_state_uncertainty_seed900123/checkpoints/best.pt")
    s3_final = required("outputs/schema_sweep/frozen_lm_s3_state_uncertainty_seed900123/checkpoints/final.pt")
    no_lm_s2_metrics_path = required("outputs/schema_sweep/no_lm_s2_state_answerability_seed900124/metrics_final.json")
    no_lm_s2_best = required("outputs/schema_sweep/no_lm_s2_state_answerability_seed900124/best.pt")
    no_lm_s2_final = required("outputs/schema_sweep/no_lm_s2_state_answerability_seed900124/final.pt")
    no_lm_s3_metrics_path = required("outputs/schema_sweep/no_lm_s3_state_uncertainty_seed900125/metrics_final.json")
    no_lm_s3_best = required("outputs/schema_sweep/no_lm_s3_state_uncertainty_seed900125/best.pt")
    no_lm_s3_final = required("outputs/schema_sweep/no_lm_s3_state_uncertainty_seed900125/final.pt")
    no_lm_s1_formal_metrics_path = required("outputs/schema_sweep/no_lm_s1_state_only/metrics_final.json")
    no_lm_s1_formal_best = required("outputs/schema_sweep/no_lm_s1_state_only/best.pt")
    no_lm_s1_formal_final = required("outputs/schema_sweep/no_lm_s1_state_only/final.pt")
    no_lm_s1_formal_step = required("outputs/schema_sweep/no_lm_s1_state_only/step_10000.pt")
    no_lm_s1_formal_log = required("outputs/logs/schema_no_lm_s1_source_gpu0.log")
    no_lm_s2_formal_metrics_path = required("outputs/schema_sweep/no_lm_s2_state_answerability/metrics_final.json")
    no_lm_s2_formal_best = required("outputs/schema_sweep/no_lm_s2_state_answerability/best.pt")
    no_lm_s2_formal_final = required("outputs/schema_sweep/no_lm_s2_state_answerability/final.pt")
    no_lm_s2_formal_step = required("outputs/schema_sweep/no_lm_s2_state_answerability/step_10000.pt")
    no_lm_s2_formal_log = required("outputs/logs/schema_no_lm_s2_source_resume4_gpu0.log")
    no_lm_s3_formal_metrics_path = required("outputs/schema_sweep/no_lm_s3_state_uncertainty/metrics_final.json")
    no_lm_s3_formal_best = required("outputs/schema_sweep/no_lm_s3_state_uncertainty/best.pt")
    no_lm_s3_formal_final = required("outputs/schema_sweep/no_lm_s3_state_uncertainty/final.pt")
    no_lm_s3_formal_step = required("outputs/schema_sweep/no_lm_s3_state_uncertainty/step_10000.pt")
    no_lm_s3_formal_log = required("outputs/logs/schema_no_lm_s3_source_resume3_gpu0.log")
    no_lm_s1_lp_metrics_path = required("outputs/schema_sweep/lp_no_lm_s1_state_only/metrics_final.json")
    no_lm_s1_lp_best = required("outputs/schema_sweep/lp_no_lm_s1_state_only/best.pt")
    no_lm_s1_lp_final = required("outputs/schema_sweep/lp_no_lm_s1_state_only/final.pt")
    no_lm_s1_lp_step = required("outputs/schema_sweep/lp_no_lm_s1_state_only/step_3000.pt")
    no_lm_s1_lp_log = required("outputs/logs/schema_lp_no_lm_s1_gpu0.log")
    no_lm_s2_lp_metrics_path = required("outputs/schema_sweep/lp_no_lm_s2_state_answerability/metrics_final.json")
    no_lm_s2_lp_best = required("outputs/schema_sweep/lp_no_lm_s2_state_answerability/best.pt")
    no_lm_s2_lp_final = required("outputs/schema_sweep/lp_no_lm_s2_state_answerability/final.pt")
    no_lm_s2_lp_step = required("outputs/schema_sweep/lp_no_lm_s2_state_answerability/step_3000.pt")
    no_lm_s2_lp_log = required("outputs/logs/schema_lp_no_lm_s2_gpu0.log")
    no_lm_s3_lp_metrics_path = required("outputs/schema_sweep/lp_no_lm_s3_state_uncertainty/metrics_final.json")
    no_lm_s3_lp_best = required("outputs/schema_sweep/lp_no_lm_s3_state_uncertainty/best.pt")
    no_lm_s3_lp_final = required("outputs/schema_sweep/lp_no_lm_s3_state_uncertainty/final.pt")
    no_lm_s3_lp_step = required("outputs/schema_sweep/lp_no_lm_s3_state_uncertainty/step_3000.pt")
    no_lm_s3_lp_log = required("outputs/logs/schema_lp_no_lm_s3_gpu1.log")
    frozen_lm_s1_lp_metrics_path = required("outputs/schema_sweep/lp_frozen_lm_s1_state_only/metrics_final.json")
    frozen_lm_s1_lp_best = required("outputs/schema_sweep/lp_frozen_lm_s1_state_only/best.pt")
    frozen_lm_s1_lp_final = required("outputs/schema_sweep/lp_frozen_lm_s1_state_only/final.pt")
    frozen_lm_s1_lp_step = required("outputs/schema_sweep/lp_frozen_lm_s1_state_only/step_3000.pt")
    frozen_lm_s1_lp_log = required("outputs/logs/schema_lp_frozen_lm_s1_gpu0.log")
    frozen_lm_s2_lp_metrics_path = required("outputs/schema_sweep/lp_frozen_lm_s2_state_answerability/metrics_final.json")
    frozen_lm_s2_lp_best = required("outputs/schema_sweep/lp_frozen_lm_s2_state_answerability/best.pt")
    frozen_lm_s2_lp_final = required("outputs/schema_sweep/lp_frozen_lm_s2_state_answerability/final.pt")
    frozen_lm_s2_lp_step = required("outputs/schema_sweep/lp_frozen_lm_s2_state_answerability/step_3000.pt")
    frozen_lm_s2_lp_log = required("outputs/logs/schema_lp_frozen_lm_s2_gpu0.log")
    frozen_lm_s3_lp_metrics_path = required("outputs/schema_sweep/lp_frozen_lm_s3_state_uncertainty/metrics_final.json")
    frozen_lm_s3_lp_best = required("outputs/schema_sweep/lp_frozen_lm_s3_state_uncertainty/best.pt")
    frozen_lm_s3_lp_final = required("outputs/schema_sweep/lp_frozen_lm_s3_state_uncertainty/final.pt")
    frozen_lm_s3_lp_step = required("outputs/schema_sweep/lp_frozen_lm_s3_state_uncertainty/step_3000.pt")
    frozen_lm_s3_lp_log = required("outputs/logs/schema_lp_frozen_lm_s3_gpu0.log")

    answerability = no_lm_metrics["answerability"]
    uncertainty = no_lm_metrics["uncertainty"]
    no_lm_s2_payload = read_json(ROOT / no_lm_s2_metrics_path)
    no_lm_s2_metrics = no_lm_s2_payload["metrics"]
    no_lm_s3_payload = read_json(ROOT / no_lm_s3_metrics_path)
    no_lm_s3_metrics = no_lm_s3_payload["metrics"]
    no_lm_s1_formal_payload = read_json(ROOT / no_lm_s1_formal_metrics_path)
    no_lm_s1_formal_metrics = no_lm_s1_formal_payload["metrics"]
    no_lm_s1_formal_best_loss, no_lm_s1_formal_best_auc = step_extrema("outputs/schema_sweep/no_lm_s1_state_only")
    no_lm_s2_formal_payload = read_json(ROOT / no_lm_s2_formal_metrics_path)
    no_lm_s2_formal_metrics = no_lm_s2_formal_payload["metrics"]
    no_lm_s3_formal_payload = read_json(ROOT / no_lm_s3_formal_metrics_path)
    no_lm_s3_formal_metrics = no_lm_s3_formal_payload["metrics"]
    no_lm_s1_lp_payload = read_json(ROOT / no_lm_s1_lp_metrics_path)
    no_lm_s1_lp_metrics = no_lm_s1_lp_payload["metrics"]
    no_lm_s1_lp_best_loss, no_lm_s1_lp_best_auc = step_extrema("outputs/schema_sweep/lp_no_lm_s1_state_only")
    no_lm_s2_lp_payload = read_json(ROOT / no_lm_s2_lp_metrics_path)
    no_lm_s2_lp_metrics = no_lm_s2_lp_payload["metrics"]
    no_lm_s2_lp_best_loss, no_lm_s2_lp_best_auc = step_extrema("outputs/schema_sweep/lp_no_lm_s2_state_answerability")
    no_lm_s3_lp_payload = read_json(ROOT / no_lm_s3_lp_metrics_path)
    no_lm_s3_lp_metrics = no_lm_s3_lp_payload["metrics"]
    no_lm_s3_lp_best_loss, no_lm_s3_lp_best_auc = step_extrema("outputs/schema_sweep/lp_no_lm_s3_state_uncertainty")
    frozen_lm_s1_source = frozen_source["frozen_lm_s1_state_only_formal_source"]
    frozen_lm_s2_source = frozen_source["frozen_lm_s2_state_answerability_formal_source"]
    frozen_lm_s3_source = frozen_source["frozen_lm_s3_state_uncertainty_formal_source"]
    frozen_lm_s1_lp_payload = read_json(ROOT / frozen_lm_s1_lp_metrics_path)
    frozen_lm_s1_lp_metrics = frozen_lm_s1_lp_payload["metrics"]
    frozen_lm_s1_lp_best_loss, frozen_lm_s1_lp_best_auc = step_extrema(
        "outputs/schema_sweep/lp_frozen_lm_s1_state_only"
    )
    frozen_lm_s2_lp_payload = read_json(ROOT / frozen_lm_s2_lp_metrics_path)
    frozen_lm_s2_lp_metrics = frozen_lm_s2_lp_payload["metrics"]
    frozen_lm_s2_lp_best_loss, frozen_lm_s2_lp_best_auc = step_extrema(
        "outputs/schema_sweep/lp_frozen_lm_s2_state_answerability"
    )
    frozen_lm_s3_lp_payload = read_json(ROOT / frozen_lm_s3_lp_metrics_path)
    frozen_lm_s3_lp_metrics = frozen_lm_s3_lp_payload["metrics"]
    frozen_lm_s3_lp_best_loss, frozen_lm_s3_lp_best_auc = step_extrema(
        "outputs/schema_sweep/lp_frozen_lm_s3_state_uncertainty"
    )
    return [
        {
            "evidence_id": "no_lm_answerability_derivative",
            "pathway": "no-LM UMS 4-state classifier",
            "schema_level": "S2 state_answerability diagnostic",
            "artifact_type": "validation-only metric export",
            "scope": "CheXpert fixed val, 1000 samples, 12 labels",
            "metric_target": "answerability = state != null; p = 1 - p_null",
            "support": f"{answerability['support_positive']} / {answerability['support_total']} positive fields",
            "primary_metric": "macro_auc",
            "primary_value": fmt(answerability["macro_auc"]),
            "secondary_metrics": (
                f"macro_f1={fmt(answerability['macro_f1'])}; "
                f"accuracy={fmt(answerability['accuracy'])}; "
                f"ece={fmt(answerability['ece'])}"
            ),
            "interpretation": "no-LM has a weak-to-moderate derived answerability signal.",
            "claim_boundary": "Derived from 4-state logits; not explicit no-LM S2 schema supervision.",
            "evidence_paths": "outputs/final_tables/no_lm_schema_derivative_metrics.csv",
        },
        {
            "evidence_id": "no_lm_uncertainty_derivative",
            "pathway": "no-LM UMS 4-state classifier",
            "schema_level": "S3 state_uncertainty diagnostic",
            "artifact_type": "validation-only metric export",
            "scope": "CheXpert fixed val, 1000 samples, 12 labels",
            "metric_target": "uncertainty = state == uncertain; p = p_uncertain",
            "support": f"{uncertainty['support_positive']} / {uncertainty['support_total']} positive fields",
            "primary_metric": "macro_auc",
            "primary_value": fmt(uncertainty["macro_auc"]),
            "secondary_metrics": (
                f"macro_f1={fmt(uncertainty['macro_f1'])}; "
                f"pred_rate={fmt(uncertainty['pred_rate'])}; "
                f"prevalence={fmt(uncertainty['prevalence'])}"
            ),
            "interpretation": "AUC is modest, but uncertainty is rare and default-threshold F1 is very low.",
            "claim_boundary": "Derived from 4-state logits; not explicit no-LM S3 schema supervision.",
            "evidence_paths": "outputs/final_tables/no_lm_schema_derivative_metrics.csv",
        },
        {
            "evidence_id": "no_lm_s2_explicit_head_debug",
            "pathway": "no-LM UMS explicit auxiliary head",
            "schema_level": "S2 state_answerability",
            "artifact_type": "20-step debug runtime and metric export",
            "scope": "debug only; seed 900124; 200 train samples and 50 val samples",
            "metric_target": "answerability auxiliary head trained with BCE target from UMS answerability",
            "support": (
                f"{no_lm_s2_metrics['answerability_support_positive']} / "
                f"{50 * 12} positive fields in debug val"
            ),
            "primary_metric": "answerability_macro_auc",
            "primary_value": fmt(no_lm_s2_metrics["answerability_macro_auc"]),
            "secondary_metrics": (
                f"answerability_macro_f1={fmt(no_lm_s2_metrics['answerability_macro_f1'])}; "
                f"answerability_accuracy={fmt(no_lm_s2_metrics['answerability_accuracy'])}; "
                f"state_macro_auc={fmt(no_lm_s2_metrics['macro_auc'])}; "
                f"val_loss={fmt(no_lm_s2_payload['val_loss'])}"
            ),
            "interpretation": "no-LM S2 explicit answerability head builds, trains, validates, and exports auxiliary metrics.",
            "claim_boundary": "Debug entry only; not formal S2 schema-complexity performance evidence.",
            "evidence_paths": f"{no_lm_s2_metrics_path}; {no_lm_s2_best}; {no_lm_s2_final}",
        },
        {
            "evidence_id": "no_lm_s3_explicit_head_debug",
            "pathway": "no-LM UMS explicit auxiliary head",
            "schema_level": "S3 state_uncertainty",
            "artifact_type": "20-step debug runtime and metric export",
            "scope": "debug only; seed 900125; 200 train samples and 50 val samples",
            "metric_target": "uncertainty auxiliary head trained with BCE target labels == -1",
            "support": (
                f"{no_lm_s3_metrics['uncertainty_support_positive']} / "
                f"{50 * 12} positive fields in debug val"
            ),
            "primary_metric": "uncertainty_macro_auc",
            "primary_value": fmt(no_lm_s3_metrics["uncertainty_macro_auc"]),
            "secondary_metrics": (
                f"uncertainty_macro_f1={fmt(no_lm_s3_metrics['uncertainty_macro_f1'])}; "
                f"uncertainty_accuracy={fmt(no_lm_s3_metrics['uncertainty_accuracy'])}; "
                f"state_macro_auc={fmt(no_lm_s3_metrics['macro_auc'])}; "
                f"val_loss={fmt(no_lm_s3_payload['val_loss'])}"
            ),
            "interpretation": "no-LM S3 explicit uncertainty head builds, trains, validates, and exports auxiliary metrics.",
            "claim_boundary": "Debug entry only; not formal S3 schema-complexity performance evidence.",
            "evidence_paths": f"{no_lm_s3_metrics_path}; {no_lm_s3_best}; {no_lm_s3_final}",
        },
        {
            "evidence_id": "no_lm_s1_state_only_formal_source",
            "pathway": "no-LM UMS 4-state classifier",
            "schema_level": "S1 state_only",
            "artifact_type": "10000-step formal source run",
            "scope": "CheXpert fixed split; 29000 train records, 1000 val records, 12 labels",
            "metric_target": "4-state UMS source classification without auxiliary schema fields",
            "support": f"{len(no_lm_s1_formal_metrics['per_label'])} labels in fixed val",
            "primary_metric": "macro_auc",
            "primary_value": fmt(no_lm_s1_formal_metrics["macro_auc"]),
            "secondary_metrics": (
                f"macro_f1={fmt(no_lm_s1_formal_metrics['macro_f1'])}; "
                f"micro_f1={fmt(no_lm_s1_formal_metrics['micro_f1'])}; "
                f"state_accuracy_all_fields={fmt(no_lm_s1_formal_metrics['state_accuracy_all_fields'])}; "
                f"state_accuracy_answerable_fields={fmt(no_lm_s1_formal_metrics['state_accuracy_answerable_fields'])}; "
                f"val_loss={fmt(no_lm_s1_formal_payload['val_loss'])}; "
                f"best_loss_step={no_lm_s1_formal_best_loss['step']}:{fmt(no_lm_s1_formal_best_loss['macro_auc'])}; "
                f"best_auc_step={no_lm_s1_formal_best_auc['step']}:{fmt(no_lm_s1_formal_best_auc['macro_auc'])}"
            ),
            "interpretation": "formal no-LM S1 fixed-split source row completed; it provides the state-only baseline for the schema sweep.",
            "claim_boundary": "Formal no-LM S1 source evidence only; paired no-LM S1 LP and matched frozen-LM S1 source+LP rows are recorded separately.",
            "evidence_paths": (
                f"{no_lm_s1_formal_metrics_path}; {no_lm_s1_formal_best}; {no_lm_s1_formal_final}; "
                f"{no_lm_s1_formal_step}; {no_lm_s1_formal_log}"
            ),
        },
        {
            "evidence_id": "no_lm_s1_state_only_formal_lp",
            "pathway": "no-LM UMS 4-state classifier -> frozen ViT linear probe",
            "schema_level": "S1 state_only",
            "artifact_type": "3000-step formal LP run",
            "scope": "CheXpert fixed split; 29000 train records, 1000 val records, 14 binary LP labels",
            "metric_target": "binary LP performance from frozen S1 source ViT backbone",
            "support": f"{len(no_lm_s1_lp_metrics['per_label'])} labels; source checkpoint initialized from S1 best.pt",
            "primary_metric": "macro_auc",
            "primary_value": fmt(no_lm_s1_lp_metrics["macro_auc"]),
            "secondary_metrics": (
                f"macro_f1={fmt(no_lm_s1_lp_metrics['macro_f1'])}; "
                f"micro_f1={fmt(no_lm_s1_lp_metrics['micro_f1'])}; "
                f"val_loss={fmt(no_lm_s1_lp_payload['val_loss'])}; "
                f"best_loss_step={no_lm_s1_lp_best_loss['step']}:{fmt(no_lm_s1_lp_best_loss['macro_auc'])}; "
                f"best_auc_step={no_lm_s1_lp_best_auc['step']}:{fmt(no_lm_s1_lp_best_auc['macro_auc'])}"
            ),
            "interpretation": "formal S1 LP completed; downstream binary LP performance is strong, with best macro-AUC early in training.",
            "claim_boundary": "Formal no-LM S1 source+LP evidence; frozen-LM formal matched S1 rows are still missing.",
            "evidence_paths": (
                f"{no_lm_s1_lp_metrics_path}; {no_lm_s1_lp_best}; {no_lm_s1_lp_final}; "
                f"{no_lm_s1_lp_step}; {no_lm_s1_lp_log}"
            ),
        },
        {
            "evidence_id": "no_lm_s2_explicit_head_formal_source",
            "pathway": "no-LM UMS explicit auxiliary head",
            "schema_level": "S2 state_answerability",
            "artifact_type": "10000-step formal source run",
            "scope": "CheXpert fixed split; 29000 train records, 1000 val records, 12 labels",
            "metric_target": "answerability auxiliary head trained with BCE target from UMS answerability",
            "support": f"{no_lm_s2_formal_metrics['answerability_support_positive']} / {1000 * 12} positive fields in fixed val",
            "primary_metric": "answerability_macro_auc",
            "primary_value": fmt(no_lm_s2_formal_metrics["answerability_macro_auc"]),
            "secondary_metrics": (
                f"answerability_macro_f1={fmt(no_lm_s2_formal_metrics['answerability_macro_f1'])}; "
                f"answerability_accuracy={fmt(no_lm_s2_formal_metrics['answerability_accuracy'])}; "
                f"answerability_pred_rate={fmt(no_lm_s2_formal_metrics['answerability_pred_rate'])}; "
                f"state_macro_auc={fmt(no_lm_s2_formal_metrics['macro_auc'])}; "
                f"state_macro_f1={fmt(no_lm_s2_formal_metrics['macro_f1'])}; "
                f"val_loss={fmt(no_lm_s2_formal_payload['val_loss'])}"
            ),
            "interpretation": "formal no-LM S2 source run completed and shows explicit answerability signal above the legacy derived diagnostic.",
            "claim_boundary": "Formal no-LM S2 source evidence only; S2 LP and frozen-LM formal matched rows are still missing.",
            "evidence_paths": (
                f"{no_lm_s2_formal_metrics_path}; {no_lm_s2_formal_best}; {no_lm_s2_formal_final}; "
                f"{no_lm_s2_formal_step}; {no_lm_s2_formal_log}"
            ),
        },
        {
            "evidence_id": "no_lm_s3_explicit_head_formal_source",
            "pathway": "no-LM UMS explicit auxiliary head",
            "schema_level": "S3 state_uncertainty",
            "artifact_type": "10000-step formal source run",
            "scope": "CheXpert fixed split; 29000 train records, 1000 val records, 12 labels",
            "metric_target": "uncertainty auxiliary head trained with BCE target labels == -1",
            "support": f"{no_lm_s3_formal_metrics['uncertainty_support_positive']} / {1000 * 12} positive fields in fixed val",
            "primary_metric": "uncertainty_macro_auc",
            "primary_value": fmt(no_lm_s3_formal_metrics["uncertainty_macro_auc"]),
            "secondary_metrics": (
                f"uncertainty_macro_f1={fmt(no_lm_s3_formal_metrics['uncertainty_macro_f1'])}; "
                f"uncertainty_accuracy={fmt(no_lm_s3_formal_metrics['uncertainty_accuracy'])}; "
                f"uncertainty_pred_rate={fmt(no_lm_s3_formal_metrics['uncertainty_pred_rate'])}; "
                f"state_macro_auc={fmt(no_lm_s3_formal_metrics['macro_auc'])}; "
                f"state_macro_f1={fmt(no_lm_s3_formal_metrics['macro_f1'])}; "
                f"val_loss={fmt(no_lm_s3_formal_payload['val_loss'])}"
            ),
            "interpretation": "formal no-LM S3 source run completed; uncertainty ranking signal is present but default-threshold positives remain rare.",
            "claim_boundary": "Formal no-LM S3 source evidence only; S3 LP and matched frozen-LM S3 evidence are recorded separately.",
            "evidence_paths": (
                f"{no_lm_s3_formal_metrics_path}; {no_lm_s3_formal_best}; {no_lm_s3_formal_final}; "
                f"{no_lm_s3_formal_step}; {no_lm_s3_formal_log}"
            ),
        },
        {
            "evidence_id": "no_lm_s2_explicit_head_formal_lp",
            "pathway": "no-LM UMS explicit auxiliary head -> frozen ViT linear probe",
            "schema_level": "S2 state_answerability",
            "artifact_type": "3000-step formal LP run",
            "scope": "CheXpert fixed split; 29000 train records, 1000 val records, 14 binary LP labels",
            "metric_target": "binary LP performance from frozen S2 source ViT backbone",
            "support": f"{len(no_lm_s2_lp_metrics['per_label'])} labels; source checkpoint initialized from S2 best.pt",
            "primary_metric": "macro_auc",
            "primary_value": fmt(no_lm_s2_lp_metrics["macro_auc"]),
            "secondary_metrics": (
                f"macro_f1={fmt(no_lm_s2_lp_metrics['macro_f1'])}; "
                f"micro_f1={fmt(no_lm_s2_lp_metrics['micro_f1'])}; "
                f"val_loss={fmt(no_lm_s2_lp_payload['val_loss'])}; "
                f"best_loss_step={no_lm_s2_lp_best_loss['step']}:{fmt(no_lm_s2_lp_best_loss['macro_auc'])}; "
                f"best_auc_step={no_lm_s2_lp_best_auc['step']}:{fmt(no_lm_s2_lp_best_auc['macro_auc'])}"
            ),
            "interpretation": "formal S2 LP completed; downstream binary LP performance is strong, with best macro-AUC earlier than the final checkpoint.",
            "claim_boundary": "Formal no-LM S2 source+LP evidence; frozen-LM formal matched S2 rows are still missing.",
            "evidence_paths": (
                f"{no_lm_s2_lp_metrics_path}; {no_lm_s2_lp_best}; {no_lm_s2_lp_final}; "
                f"{no_lm_s2_lp_step}; {no_lm_s2_lp_log}"
            ),
        },
        {
            "evidence_id": "no_lm_s3_explicit_head_formal_lp",
            "pathway": "no-LM UMS explicit auxiliary head -> frozen ViT linear probe",
            "schema_level": "S3 state_uncertainty",
            "artifact_type": "3000-step formal LP run",
            "scope": "CheXpert fixed split; 29000 train records, 1000 val records, 14 binary LP labels",
            "metric_target": "binary LP performance from frozen S3 source ViT backbone",
            "support": f"{len(no_lm_s3_lp_metrics['per_label'])} labels; source checkpoint initialized from S3 best.pt",
            "primary_metric": "macro_auc",
            "primary_value": fmt(no_lm_s3_lp_metrics["macro_auc"]),
            "secondary_metrics": (
                f"macro_f1={fmt(no_lm_s3_lp_metrics['macro_f1'])}; "
                f"micro_f1={fmt(no_lm_s3_lp_metrics['micro_f1'])}; "
                f"val_loss={fmt(no_lm_s3_lp_payload['val_loss'])}; "
                f"best_loss_step={no_lm_s3_lp_best_loss['step']}:{fmt(no_lm_s3_lp_best_loss['macro_auc'])}; "
                f"best_auc_step={no_lm_s3_lp_best_auc['step']}:{fmt(no_lm_s3_lp_best_auc['macro_auc'])}"
            ),
            "interpretation": "formal S3 LP completed; downstream binary LP performance is strong, though below S2 at the final endpoint.",
            "claim_boundary": "Formal no-LM S3 source+LP evidence; matched frozen-LM S3 source+LP is recorded separately.",
            "evidence_paths": (
                f"{no_lm_s3_lp_metrics_path}; {no_lm_s3_lp_best}; {no_lm_s3_lp_final}; "
                f"{no_lm_s3_lp_step}; {no_lm_s3_lp_log}"
            ),
        },
        {
            "evidence_id": "frozen_lm_s1_state_only_formal_source",
            "pathway": "frozen-LM VIVID source",
            "schema_level": "S1 state_only",
            "artifact_type": "10000-step formal source run",
            "scope": "CheXpert fixed split; 29000 train records, 1000 val records, 12 labels",
            "metric_target": "source-token validation loss for state-only JSON serialization",
            "support": "250 fixed-val batches; final and step_10000 checkpoints confirm step 10000",
            "primary_metric": "checkpoint_best_val_loss",
            "primary_value": frozen_lm_s1_source["checkpoint_best_val_loss"],
            "secondary_metrics": (
                f"final_log_val_loss={frozen_lm_s1_source['final_val_loss_log']}; "
                f"checkpoint_best_step={frozen_lm_s1_source['checkpoint_best_step']}; "
                f"runtime={frozen_lm_s1_source['runtime']}; "
                f"exitcode={frozen_lm_s1_source['exitcode']}"
            ),
            "interpretation": "formal frozen-LM S1 source checkpoint completed under the fixed-split protocol.",
            "claim_boundary": "Source-loss/checkpoint evidence only; paired frozen-LM S1 LP is recorded separately.",
            "evidence_paths": frozen_lm_s1_source["evidence_paths"],
        },
        {
            "evidence_id": "frozen_lm_s1_state_only_formal_lp",
            "pathway": "frozen-LM VIVID source -> frozen ViT linear probe",
            "schema_level": "S1 state_only",
            "artifact_type": "3000-step formal LP run",
            "scope": "CheXpert fixed split; 29000 train records, 1000 val records, 14 binary LP labels",
            "metric_target": "binary LP performance from frozen-LM S1 source ViT backbone",
            "support": f"{len(frozen_lm_s1_lp_metrics['per_label'])} labels; source checkpoint initialized from frozen-LM S1 best.pt",
            "primary_metric": "macro_auc",
            "primary_value": fmt(frozen_lm_s1_lp_metrics["macro_auc"]),
            "secondary_metrics": (
                f"macro_f1={fmt(frozen_lm_s1_lp_metrics['macro_f1'])}; "
                f"micro_f1={fmt(frozen_lm_s1_lp_metrics['micro_f1'])}; "
                f"val_loss={fmt(frozen_lm_s1_lp_payload['val_loss'])}; "
                f"best_loss_step={frozen_lm_s1_lp_best_loss['step']}:{fmt(frozen_lm_s1_lp_best_loss['macro_auc'])}; "
                f"best_auc_step={frozen_lm_s1_lp_best_auc['step']}:{fmt(frozen_lm_s1_lp_best_auc['macro_auc'])}"
            ),
            "interpretation": "formal frozen-LM S1 LP completed; paired S1 source+LP evidence is now available under the fixed split.",
            "claim_boundary": "Formal frozen-LM S1 source+LP evidence; compare direction and magnitude from the consolidated table rather than assuming benefit.",
            "evidence_paths": (
                f"{frozen_lm_s1_lp_metrics_path}; {frozen_lm_s1_lp_best}; {frozen_lm_s1_lp_final}; "
                f"{frozen_lm_s1_lp_step}; {frozen_lm_s1_lp_log}"
            ),
        },
        {
            "evidence_id": "frozen_lm_s2_state_answerability_formal_source",
            "pathway": "frozen-LM VIVID source",
            "schema_level": "S2 state_answerability",
            "artifact_type": "10000-step formal source run",
            "scope": "CheXpert fixed split; 29000 train records, 1000 val records, 12 labels",
            "metric_target": "source-token validation loss for state+answerability JSON serialization",
            "support": "250 fixed-val batches; checkpoint metadata confirms step 10000",
            "primary_metric": "checkpoint_best_val_loss",
            "primary_value": frozen_lm_s2_source["checkpoint_best_val_loss"],
            "secondary_metrics": (
                f"final_log_val_loss={frozen_lm_s2_source['final_val_loss_log']}; "
                f"best_step_log={frozen_lm_s2_source['best_step_log']}; "
                f"runtime={frozen_lm_s2_source['runtime']}; "
                f"exitcode={frozen_lm_s2_source['exitcode']}"
            ),
            "interpretation": "formal frozen-LM S2 source checkpoint completed under fixed-split protocol.",
            "claim_boundary": "Source-loss/checkpoint evidence only; paired S2 LP is recorded separately.",
            "evidence_paths": frozen_lm_s2_source["evidence_paths"],
        },
        {
            "evidence_id": "frozen_lm_s2_state_answerability_formal_lp",
            "pathway": "frozen-LM VIVID source -> frozen ViT linear probe",
            "schema_level": "S2 state_answerability",
            "artifact_type": "3000-step formal LP run",
            "scope": "CheXpert fixed split; 29000 train records, 1000 val records, 14 binary LP labels",
            "metric_target": "binary LP performance from frozen-LM S2 source ViT backbone",
            "support": f"{len(frozen_lm_s2_lp_metrics['per_label'])} labels; source checkpoint initialized from frozen-LM S2 best.pt",
            "primary_metric": "macro_auc",
            "primary_value": fmt(frozen_lm_s2_lp_metrics["macro_auc"]),
            "secondary_metrics": (
                f"macro_f1={fmt(frozen_lm_s2_lp_metrics['macro_f1'])}; "
                f"micro_f1={fmt(frozen_lm_s2_lp_metrics['micro_f1'])}; "
                f"val_loss={fmt(frozen_lm_s2_lp_payload['val_loss'])}; "
                f"best_loss_step={frozen_lm_s2_lp_best_loss['step']}:{fmt(frozen_lm_s2_lp_best_loss['macro_auc'])}; "
                f"best_auc_step={frozen_lm_s2_lp_best_auc['step']}:{fmt(frozen_lm_s2_lp_best_auc['macro_auc'])}"
            ),
            "interpretation": "formal frozen-LM S2 LP completed; paired S2 source+LP evidence is now available under the fixed split.",
            "claim_boundary": "Formal frozen-LM S2 source+LP evidence; S1/S2/S3 frozen-LM source+LP rows are now complete.",
            "evidence_paths": (
                f"{frozen_lm_s2_lp_metrics_path}; {frozen_lm_s2_lp_best}; {frozen_lm_s2_lp_final}; "
                f"{frozen_lm_s2_lp_step}; {frozen_lm_s2_lp_log}"
            ),
        },
        {
            "evidence_id": "frozen_lm_s3_state_uncertainty_formal_source",
            "pathway": "frozen-LM VIVID source",
            "schema_level": "S3 state_uncertainty",
            "artifact_type": "10000-step formal source run",
            "scope": "CheXpert fixed split; 29000 train records, 1000 val records, 12 labels",
            "metric_target": "source-token validation loss for state+answerability+uncertainty JSON serialization",
            "support": "250 fixed-val batches; checkpoint metadata confirms step 10000 after resume from step 8000",
            "primary_metric": "checkpoint_best_val_loss",
            "primary_value": frozen_lm_s3_source["checkpoint_best_val_loss"],
            "secondary_metrics": (
                f"final_log_val_loss={frozen_lm_s3_source['final_val_loss_log']}; "
                f"best_step_log={frozen_lm_s3_source['best_step_log']}; "
                f"runtime={frozen_lm_s3_source['runtime']}; "
                f"exitcode={frozen_lm_s3_source['exitcode']}"
            ),
            "interpretation": "formal frozen-LM S3 source checkpoint completed after an external interruption was recovered by resume.",
            "claim_boundary": "Source-loss/checkpoint evidence only; paired S3 LP is recorded separately.",
            "evidence_paths": frozen_lm_s3_source["evidence_paths"],
        },
        {
            "evidence_id": "frozen_lm_s3_state_uncertainty_formal_lp",
            "pathway": "frozen-LM VIVID source -> frozen ViT linear probe",
            "schema_level": "S3 state_uncertainty",
            "artifact_type": "3000-step formal LP run",
            "scope": "CheXpert fixed split; 29000 train records, 1000 val records, 14 binary LP labels",
            "metric_target": "binary LP performance from frozen-LM S3 source ViT backbone",
            "support": f"{len(frozen_lm_s3_lp_metrics['per_label'])} labels; source checkpoint initialized from frozen-LM S3 best.pt",
            "primary_metric": "macro_auc",
            "primary_value": fmt(frozen_lm_s3_lp_metrics["macro_auc"]),
            "secondary_metrics": (
                f"macro_f1={fmt(frozen_lm_s3_lp_metrics['macro_f1'])}; "
                f"micro_f1={fmt(frozen_lm_s3_lp_metrics['micro_f1'])}; "
                f"val_loss={fmt(frozen_lm_s3_lp_payload['val_loss'])}; "
                f"best_loss_step={frozen_lm_s3_lp_best_loss['step']}:{fmt(frozen_lm_s3_lp_best_loss['macro_auc'])}; "
                f"best_auc_step={frozen_lm_s3_lp_best_auc['step']}:{fmt(frozen_lm_s3_lp_best_auc['macro_auc'])}"
            ),
            "interpretation": "formal frozen-LM S3 LP completed; paired S3 source+LP evidence is now available under the fixed split.",
            "claim_boundary": "Formal frozen-LM S3 source+LP evidence; S1/S2/S3 frozen-LM source+LP rows are now complete.",
            "evidence_paths": (
                f"{frozen_lm_s3_lp_metrics_path}; {frozen_lm_s3_lp_best}; {frozen_lm_s3_lp_final}; "
                f"{frozen_lm_s3_lp_step}; {frozen_lm_s3_lp_log}"
            ),
        },
        {
            "evidence_id": "frozen_lm_s2_serializer_debug",
            "pathway": "frozen-LM UMS JSON serialization",
            "schema_level": "S2 state_answerability",
            "artifact_type": "5-step debug runtime and serializer validation",
            "scope": "debug only; seed 900122",
            "metric_target": "target_json includes explicit answerable fields",
            "support": "debug checkpoint artifacts present",
            "primary_metric": "serializer/runtime",
            "primary_value": "passed",
            "secondary_metrics": "documented debug val_loss step2=0.7578; step4=0.7426",
            "interpretation": "frozen-LM path can accept and serialize answerability fields.",
            "claim_boundary": "Debug entry only; not formal schema-complexity performance evidence.",
            "evidence_paths": f"{s2_best}; {s2_final}",
        },
        {
            "evidence_id": "frozen_lm_s3_serializer_debug",
            "pathway": "frozen-LM UMS JSON serialization",
            "schema_level": "S3 state_uncertainty",
            "artifact_type": "5-step debug runtime and serializer validation",
            "scope": "debug only; seed 900123",
            "metric_target": "target_json includes explicit uncertain fields",
            "support": "debug checkpoint artifacts present",
            "primary_metric": "serializer/runtime",
            "primary_value": "passed",
            "secondary_metrics": "documented debug val_loss step2=0.7468; step4=0.7321",
            "interpretation": "frozen-LM path can accept and serialize uncertainty fields.",
            "claim_boundary": "Debug entry only; not formal schema-complexity performance evidence.",
            "evidence_paths": f"{s3_best}; {s3_final}",
        },
    ]


def main() -> None:
    rows = build_rows()
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(FINAL_DIR / "schema_complexity_diagnostic_summary.csv", rows, COLUMNS)
    text = "# Schema Complexity Diagnostic Summary\n\n"
    text += (
        "This table consolidates the current schema-complexity evidence. no-LM S1/S2/S3 "
        "formal source and LP rows are now complete; frozen-LM S1/S2/S3 formal source and LP rows are also complete. "
        "Frozen-LM debug rows here remain serializer/runtime "
        "debug evidence unless explicitly marked otherwise.\n\n"
    )
    text += markdown_table(rows, COLUMNS)
    text += "\n## Claim Boundary\n\n"
    text += (
        "- The current evidence now includes matched no-LM and frozen-LM S1/S2/S3 formal source+LP rows under the fixed split; interpret metric direction directly from this table.\n"
        "- Frozen-LM S2/S3 serializer support is validated at debug-entry level.\n"
        "- no-LM answerability/uncertainty remains measurable from 4-state logits as a legacy derived diagnostic.\n"
        "- no-LM S1 fixed-split source+LP is complete; no-LM S2/S3 explicit auxiliary heads pass debug entries and now also have formal source and LP endpoints.\n"
        "- frozen-LM S1/S2/S3 source rows and frozen-LM S1/S2/S3 LP rows are complete.\n"
    )
    (FINAL_DIR / "schema_complexity_diagnostic_summary.md").write_text(text, encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {FINAL_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
