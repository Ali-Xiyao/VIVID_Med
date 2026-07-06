"""Summarize completed 10k data-scaling rows."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs" / "final_tables"

RUNS = [
    {
        "run_id": "bce_10k",
        "method": "BCE ViT",
        "stage": "source_classifier",
        "sample_size": "10000",
        "run_dir": ROOT / "outputs" / "data_scaling" / "bce_10k",
        "claim_boundary": "source_control_only",
    },
    {
        "run_id": "no_lm_ums_10k",
        "method": "no-LM UMS",
        "stage": "source_classifier",
        "sample_size": "10000",
        "run_dir": ROOT / "outputs" / "data_scaling" / "no_lm_ums_10k",
        "claim_boundary": "source_checkpoint_for_lp",
    },
    {
        "run_id": "lp_no_lm_ums_10k",
        "method": "no-LM UMS",
        "stage": "linear_probe",
        "sample_size": "10000",
        "run_dir": ROOT / "outputs" / "data_scaling" / "lp_no_lm_ums_10k",
        "claim_boundary": "matched_no_lm_lp_complete",
    },
    {
        "run_id": "lp_frozen_lm_ums_10k",
        "method": "frozen-LM UMS",
        "stage": "linear_probe",
        "sample_size": "10000",
        "run_dir": ROOT / "outputs" / "data_scaling" / "lp_frozen_lm_ums_10k",
        "claim_boundary": "matched_frozen_lm_lp_complete",
    },
]


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_float(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)):
        return f"{value:.6f}"
    return str(value)


def row_from_path(run: dict[str, Any], path: Path, metric_policy: str) -> dict[str, str]:
    payload = load_json(path)
    metrics = payload.get("metrics", {})
    match = re.search(r"metrics_step_(\d+)\.json$", path.name)
    step = match.group(1) if match else "final"
    return {
        "run_id": str(run["run_id"]),
        "method": str(run["method"]),
        "stage": str(run["stage"]),
        "sample_size": str(run["sample_size"]),
        "metric_policy": metric_policy,
        "step": step,
        "val_loss": format_float(payload.get("val_loss")),
        "macro_auc": format_float(metrics.get("macro_auc")),
        "macro_f1": format_float(metrics.get("macro_f1")),
        "micro_f1": format_float(metrics.get("micro_f1")),
        "metrics_path": rel(path),
        "claim_boundary": str(run["claim_boundary"]),
    }


def numeric(row: dict[str, str], field: str) -> float:
    return float(row[field])


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


def collect_run(run: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    run_dir = Path(run["run_dir"])
    step_paths = sorted(
        run_dir.glob("metrics_step_*.json"),
        key=lambda path: int(path.stem.replace("metrics_step_", "")),
    )
    if not step_paths:
        raise FileNotFoundError(f"No metrics_step_*.json under {rel(run_dir)}")
    final_path = run_dir / "metrics_final.json"
    if not final_path.exists():
        raise FileNotFoundError(rel(final_path))

    trajectory = [row_from_path(run, path, "trajectory") for path in step_paths]
    final_row = row_from_path(run, final_path, "final_checkpoint")
    best_val = min(trajectory, key=lambda row: numeric(row, "val_loss"))
    best_auc = max(trajectory, key=lambda row: numeric(row, "macro_auc"))
    best_f1 = max(trajectory, key=lambda row: numeric(row, "macro_f1"))

    key_rows = [
        {**final_row, "selection_note": "final checkpoint"},
        {**best_val, "metric_policy": "best_val_loss", "selection_note": "lowest validation loss"},
        {**best_auc, "metric_policy": "best_macro_auc", "selection_note": "highest macro-AUC"},
        {**best_f1, "metric_policy": "best_macro_f1", "selection_note": "highest macro-F1"},
    ]
    return key_rows, trajectory


def main() -> None:
    key_rows: list[dict[str, str]] = []
    trajectory: list[dict[str, str]] = []
    for run in RUNS:
        run_key_rows, run_trajectory = collect_run(run)
        key_rows.extend(run_key_rows)
        trajectory.extend(run_trajectory)

    columns = [
        "run_id",
        "method",
        "stage",
        "sample_size",
        "metric_policy",
        "step",
        "val_loss",
        "macro_auc",
        "macro_f1",
        "micro_f1",
        "metrics_path",
        "claim_boundary",
        "selection_note",
    ]
    trajectory_columns = columns[:-1]
    write_csv(FINAL_DIR / "data_scaling_10k_progress.csv", key_rows, columns)
    write_csv(FINAL_DIR / "data_scaling_10k_trajectory.csv", trajectory, trajectory_columns)

    text = "# Data Scaling 10k Progress\n\n"
    text += (
        "This table summarizes completed 10k data-scaling rows. "
        "At this gate the BCE source-control row, no-LM UMS source+LP branch, and frozen-LM UMS LP row are complete; "
        "the frozen-LM source-loss trajectory is summarized separately in `data_scaling_frozen_source_progress.*`.\n\n"
    )
    text += "## Key Rows\n\n"
    text += markdown_table(key_rows, columns)
    text += "\n## Full Trajectory\n\n"
    text += markdown_table(trajectory, trajectory_columns)
    text += "\n## Claim Boundary\n\n"
    text += (
        "- `bce_10k` is a source-control row only.\n"
        "- `no_lm_ums_10k` is a source row and creates the checkpoint required by `lp_no_lm_ums_10k`.\n"
        "- `lp_no_lm_ums_10k` completes the no-LM 10k LP branch but does not complete matched frozen-LM evidence.\n"
        "- `lp_frozen_lm_ums_10k` completes the frozen-LM 10k LP branch from the completed frozen-LM source checkpoint.\n"
        "- Final checkpoints can overfit relative to best validation-loss or best macro-AUC checkpoints; preserve metric policies separately.\n"
    )
    (FINAL_DIR / "data_scaling_10k_progress.md").write_text(text, encoding="utf-8")
    print(f"Wrote 10k data-scaling summaries to {rel(FINAL_DIR)}")


if __name__ == "__main__":
    main()
