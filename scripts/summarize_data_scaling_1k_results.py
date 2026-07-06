"""Summarize completed 1k data-scaling artifacts without importing torch."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs" / "final_tables"


@dataclass(frozen=True)
class RunSpec:
    method: str
    stage: str
    run_dir: Path
    comparable_group: str
    provenance_note: str


RUNS = [
    RunSpec(
        method="BCE ViT",
        stage="classifier",
        run_dir=ROOT / "outputs" / "data_scaling" / "bce_1k",
        comparable_group="classifier_reference",
        provenance_note="formal BCE 1k classifier baseline; no source-LP split",
    ),
    RunSpec(
        method="no-LM UMS",
        stage="source",
        run_dir=ROOT / "outputs" / "data_scaling" / "no_lm_ums_1k",
        comparable_group="source_not_lp",
        provenance_note="formal no-LM source/state-classifier metrics; not matched LP",
    ),
    RunSpec(
        method="no-LM UMS",
        stage="linear_probe",
        run_dir=ROOT / "outputs" / "data_scaling" / "lp_no_lm_ums_1k",
        comparable_group="matched_lp",
        provenance_note="formal matched LP from no-LM source best.pt",
    ),
    RunSpec(
        method="frozen-LM UMS",
        stage="linear_probe",
        run_dir=ROOT / "outputs" / "data_scaling" / "lp_frozen_lm_ums_1k",
        comparable_group="matched_lp",
        provenance_note="formal matched LP from frozen-LM source best.pt with documented early-stop provenance",
    ),
]


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metrics_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def row_from_payload(spec: RunSpec, policy: str, path: Path, payload: dict[str, Any]) -> dict[str, str]:
    metrics = metrics_from_payload(payload)
    per_label = metrics.get("per_label", {})
    n_labels = len(per_label) if isinstance(per_label, dict) else ""
    step = ""
    match = re.search(r"metrics_step_(\d+)\.json$", path.name)
    if match:
        step = match.group(1)
    elif path.name == "metrics_final.json":
        step = "final"

    return {
        "method": spec.method,
        "stage": spec.stage,
        "metric_policy": policy,
        "step": step,
        "val_loss": format_float(payload.get("val_loss")),
        "macro_auc": format_float(metrics.get("macro_auc")),
        "macro_f1": format_float(metrics.get("macro_f1")),
        "micro_f1": format_float(metrics.get("micro_f1")),
        "state_accuracy_all_fields": format_float(metrics.get("state_accuracy_all_fields")),
        "state_accuracy_answerable_fields": format_float(metrics.get("state_accuracy_answerable_fields")),
        "n_labels_with_metrics": str(n_labels),
        "metrics_path": rel(path),
        "comparable_group": spec.comparable_group,
        "provenance_note": spec.provenance_note,
    }


def format_float(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)):
        return f"{value:.6f}"
    return str(value)


def numeric(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    if not value:
        return None
    return float(value)


def best_step_row(spec: RunSpec) -> dict[str, str]:
    candidates = []
    for path in spec.run_dir.glob("metrics_step_*.json"):
        payload = load_json(path)
        val_loss = payload.get("val_loss")
        if isinstance(val_loss, (int, float)):
            candidates.append((float(val_loss), path, payload))
    if not candidates:
        raise FileNotFoundError(f"No metrics_step_*.json with val_loss under {rel(spec.run_dir)}")
    _, path, payload = min(candidates, key=lambda item: item[0])
    return row_from_payload(spec, "best_val_step", path, payload)


def final_row(spec: RunSpec) -> dict[str, str]:
    path = spec.run_dir / "metrics_final.json"
    if not path.exists():
        raise FileNotFoundError(rel(path))
    return row_from_payload(spec, "final_checkpoint", path, load_json(path))


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row.get(column, "") for column in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body]) + "\n"


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def find_row(rows: list[dict[str, str]], method: str, stage: str, policy: str) -> dict[str, str]:
    for row in rows:
        if row["method"] == method and row["stage"] == stage and row["metric_policy"] == policy:
            return row
    raise KeyError((method, stage, policy))


def delta_row(
    comparison: str,
    left: dict[str, str],
    right: dict[str, str],
    interpretation: str,
) -> dict[str, str]:
    row = {
        "comparison": comparison,
        "left": f"{left['method']} {left['stage']} {left['metric_policy']}",
        "right": f"{right['method']} {right['stage']} {right['metric_policy']}",
        "delta_macro_auc": "",
        "delta_macro_f1": "",
        "delta_micro_f1": "",
        "interpretation": interpretation,
    }
    for metric in ["macro_auc", "macro_f1", "micro_f1"]:
        left_value = numeric(left, metric)
        right_value = numeric(right, metric)
        if left_value is not None and right_value is not None:
            row[f"delta_{metric}"] = f"{left_value - right_value:+.6f}"
    return row


def main() -> None:
    rows: list[dict[str, str]] = []
    for spec in RUNS:
        rows.append(final_row(spec))
        rows.append(best_step_row(spec))

    bce_final = find_row(rows, "BCE ViT", "classifier", "final_checkpoint")
    no_lm_lp_final = find_row(rows, "no-LM UMS", "linear_probe", "final_checkpoint")
    frozen_lp_final = find_row(rows, "frozen-LM UMS", "linear_probe", "final_checkpoint")
    no_lm_lp_best = find_row(rows, "no-LM UMS", "linear_probe", "best_val_step")
    frozen_lp_best = find_row(rows, "frozen-LM UMS", "linear_probe", "best_val_step")

    deltas = [
        delta_row(
            "matched_lp_final_frozen_minus_no_lm",
            frozen_lp_final,
            no_lm_lp_final,
            "Matched LP final policy: frozen-LM is lower on macro-AUC, slightly higher on F1.",
        ),
        delta_row(
            "matched_lp_best_val_frozen_minus_no_lm",
            frozen_lp_best,
            no_lm_lp_best,
            "Matched LP best-val policy: frozen-LM remains lower on macro-AUC, higher on F1.",
        ),
        delta_row(
            "lp_final_frozen_minus_bce_final",
            frozen_lp_final,
            bce_final,
            "Final-policy reference only: frozen-LM LP is near BCE, with small macro-AUC gain.",
        ),
        delta_row(
            "lp_final_no_lm_minus_bce_final",
            no_lm_lp_final,
            bce_final,
            "Final-policy reference only: no-LM LP has larger macro-AUC gain over BCE.",
        ),
    ]

    summary_columns = [
        "method",
        "stage",
        "metric_policy",
        "step",
        "val_loss",
        "macro_auc",
        "macro_f1",
        "micro_f1",
        "state_accuracy_all_fields",
        "state_accuracy_answerable_fields",
        "n_labels_with_metrics",
        "metrics_path",
        "comparable_group",
        "provenance_note",
    ]
    delta_columns = [
        "comparison",
        "left",
        "right",
        "delta_macro_auc",
        "delta_macro_f1",
        "delta_micro_f1",
        "interpretation",
    ]

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(FINAL_DIR / "data_scaling_1k_matched_summary.csv", rows, summary_columns)
    write_csv(FINAL_DIR / "data_scaling_1k_matched_deltas.csv", deltas, delta_columns)

    summary_md = "# 1k Data-Scaling Matched Summary\n\n"
    summary_md += (
        "This table separates source metrics, classifier-reference metrics, matched LP final metrics, "
        "and matched LP best-validation-step metrics. The interrupted frozen-LM source run is used only "
        "through its documented early-stop `best.pt` provenance.\n\n"
    )
    summary_md += markdown_table(rows, summary_columns)
    summary_md += "\n## Deltas\n\n"
    summary_md += markdown_table(deltas, delta_columns)
    summary_md += "\n## Interpretation\n\n"
    summary_md += (
        "- Under matched LP final metrics, frozen-LM minus no-LM macro-AUC is "
        f"{deltas[0]['delta_macro_auc']}; this does not support a 1k low-data frozen-LM necessity claim.\n"
        "- Under matched LP best-validation-step metrics, frozen-LM minus no-LM macro-AUC is "
        f"{deltas[1]['delta_macro_auc']}; the negative sign remains.\n"
        "- Frozen-LM has small positive matched-LP F1 deltas, so the 1k evidence is mixed rather than "
        "a broad frozen-LM win.\n"
        "- no-LM source/state-classifier metrics are included for provenance but should not be compared "
        "directly to LP rows.\n"
    )
    (FINAL_DIR / "data_scaling_1k_matched_summary.md").write_text(summary_md, encoding="utf-8")
    (FINAL_DIR / "data_scaling_1k_matched_deltas.md").write_text(
        "# 1k Data-Scaling Matched Deltas\n\n" + markdown_table(deltas, delta_columns),
        encoding="utf-8",
    )
    print(f"Wrote 1k matched data-scaling summary to {FINAL_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
