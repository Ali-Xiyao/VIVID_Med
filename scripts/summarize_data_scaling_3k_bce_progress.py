"""Summarize completed 3k data-scaling source runs."""

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
        "run_id": "bce_3k",
        "method": "BCE ViT",
        "stage": "source_classifier",
        "sample_size": "3000",
        "run_dir": ROOT / "outputs" / "data_scaling" / "bce_3k",
        "claim_boundary": "source_control_only",
    },
    {
        "run_id": "no_lm_ums_3k",
        "method": "no-LM UMS",
        "stage": "source_state_classifier",
        "sample_size": "3000",
        "run_dir": ROOT / "outputs" / "data_scaling" / "no_lm_ums_3k",
        "claim_boundary": "source_only_requires_matched_lp",
    },
    {
        "run_id": "frozen_lm_ums_3k",
        "method": "frozen-LM UMS",
        "stage": "source_state_classifier",
        "sample_size": "3000",
        "run_dir": ROOT / "outputs" / "data_scaling" / "frozen_lm_ums_3k",
        "log_path": ROOT / "outputs" / "logs" / "data_scaling_frozen_lm_ums_3k_source_gpu0.log",
        "claim_boundary": "source_loss_only_requires_matched_lp_for_auc",
    },
]

LP_RUNS = [
    {
        "run_id": "lp_no_lm_ums_3k",
        "method": "no-LM UMS",
        "stage": "linear_probe",
        "sample_size": "3000",
        "run_dir": ROOT / "outputs" / "data_scaling" / "lp_no_lm_ums_3k",
        "claim_boundary": "matched_3k_lp_comparator",
    },
    {
        "run_id": "lp_frozen_lm_ums_3k",
        "method": "frozen-LM UMS",
        "stage": "linear_probe",
        "sample_size": "3000",
        "run_dir": ROOT / "outputs" / "data_scaling" / "lp_frozen_lm_ums_3k",
        "claim_boundary": "matched_3k_lp_comparator",
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
    if match:
        step = match.group(1)
    else:
        step = "final"
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


def numeric(row: dict[str, str], field: str) -> float:
    return float(row[field])


def collect_run(run: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    run_dir = Path(run["run_dir"])
    step_paths = sorted(
        run_dir.glob("metrics_step_*.json"),
        key=lambda path: int(path.stem.replace("metrics_step_", "")),
    )
    if not step_paths and run.get("log_path"):
        return collect_source_log_run(run)
    if not step_paths:
        raise FileNotFoundError(f"No metrics_step_*.json under {rel(run_dir)}")
    final_path = run_dir / "metrics_final.json"
    if not final_path.exists():
        raise FileNotFoundError(rel(final_path))

    trajectory = [row_from_path(run, path, "trajectory") for path in step_paths]
    final_row = row_from_path(run, final_path, "final_checkpoint")
    best_val = min(trajectory, key=lambda row: numeric(row, "val_loss"))
    key_rows = [
        {**final_row, "selection_note": "final checkpoint"},
        {**best_val, "metric_policy": "best_val_loss", "selection_note": "lowest validation loss"},
    ]
    if all(row["macro_auc"] for row in trajectory):
        best_auc = max(trajectory, key=lambda row: numeric(row, "macro_auc"))
        key_rows.append({**best_auc, "metric_policy": "best_macro_auc", "selection_note": "highest macro-AUC"})
    if all(row["macro_f1"] for row in trajectory):
        best_f1 = max(trajectory, key=lambda row: numeric(row, "macro_f1"))
        key_rows.append({**best_f1, "metric_policy": "best_macro_f1", "selection_note": "highest macro-F1"})
    return key_rows, trajectory


def collect_source_log_run(run: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    log_path = Path(run["log_path"])
    if not log_path.exists():
        raise FileNotFoundError(rel(log_path))
    rows: list[dict[str, str]] = []
    pattern = re.compile(r"Step (\d+): val_loss = ([0-9.]+)")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        rows.append(
            {
                "run_id": str(run["run_id"]),
                "method": str(run["method"]),
                "stage": str(run["stage"]),
                "sample_size": str(run["sample_size"]),
                "metric_policy": "trajectory",
                "step": match.group(1),
                "val_loss": format_float(float(match.group(2))),
                "macro_auc": "",
                "macro_f1": "",
                "micro_f1": "",
                "metrics_path": rel(log_path),
                "claim_boundary": str(run["claim_boundary"]),
            }
        )
    if not rows:
        raise ValueError(f"No source val_loss rows found in {rel(log_path)}")
    final_row = max(rows, key=lambda row: int(row["step"]))
    best_val = min(rows, key=lambda row: numeric(row, "val_loss"))
    key_rows = [
        {**final_row, "metric_policy": "final_checkpoint", "selection_note": "final checkpoint from source log"},
        {**best_val, "metric_policy": "best_val_loss", "selection_note": "lowest validation loss from source log"},
    ]
    return key_rows, rows


def write_progress_outputs(
    prefix: str,
    title: str,
    description: str,
    key_rows: list[dict[str, str]],
    trajectory: list[dict[str, str]],
    claim_boundary: str,
) -> None:
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

    write_csv(FINAL_DIR / f"{prefix}_progress.csv", key_rows, columns)
    write_csv(FINAL_DIR / f"{prefix}_trajectory.csv", trajectory, trajectory_columns)

    text = f"# {title}\n\n"
    text += description + "\n\n"
    text += "## Key Rows\n\n"
    text += markdown_table(key_rows, columns)
    text += "\n## Full Trajectory\n\n"
    text += markdown_table(trajectory, trajectory_columns)
    text += "\n## Claim Boundary\n\n"
    text += claim_boundary
    (FINAL_DIR / f"{prefix}_progress.md").write_text(text, encoding="utf-8")


def main() -> None:
    source_key_rows: list[dict[str, str]] = []
    source_trajectory: list[dict[str, str]] = []
    by_run: dict[str, tuple[list[dict[str, str]], list[dict[str, str]]]] = {}

    for run in RUNS:
        key_rows, trajectory = collect_run(run)
        source_key_rows.extend(key_rows)
        source_trajectory.extend(trajectory)
        by_run[str(run["run_id"])] = (key_rows, trajectory)

    bce_key_rows, bce_trajectory = by_run["bce_3k"]
    write_progress_outputs(
        "data_scaling_3k_bce",
        "Data Scaling 3k BCE Progress",
        (
            "This table summarizes only the completed 3k BCE source-classifier row. "
            "It does not complete the matched 3k no-LM/frozen-LM matrix."
        ),
        bce_key_rows,
        bce_trajectory,
        (
            "- The 3k BCE source run completed with `EXITCODE 0`.\n"
            "- Best validation loss and best macro-AUC both occur at step 1000.\n"
            "- Final checkpoint has higher F1 but lower macro-AUC and much higher validation loss.\n"
            "- This is a BCE control row only; it does not establish frozen-LM necessity.\n"
        ),
    )

    write_progress_outputs(
        "data_scaling_3k_source",
        "Data Scaling 3k Source Progress",
        (
            "This table summarizes completed 3k source-classifier rows. "
            "It includes BCE, no-LM UMS, and frozen-LM UMS source rows; source rows do not replace matched LP metrics."
        ),
        source_key_rows,
        source_trajectory,
        (
            "- The 3k BCE source-control row is complete.\n"
            "- The 3k no-LM UMS source row is complete with `EXITCODE 0`; its matched LP is reported in separate LP/current-progress outputs.\n"
            "- The 3k frozen-LM UMS source row is complete with `EXITCODE 0`; its source table fields are val-loss-only because `train_cxr.py` writes source metrics to the log/checkpoints rather than `metrics_*.json`.\n"
            "- Do not compare source validation loss directly with LP macro-AUC/F1.\n"
        ),
    )

    lp_key_rows: list[dict[str, str]] = []
    lp_trajectory: list[dict[str, str]] = []
    for run in LP_RUNS:
        key_rows, trajectory = collect_run(run)
        lp_key_rows.extend(key_rows)
        lp_trajectory.extend(trajectory)

    write_progress_outputs(
        "data_scaling_3k_no_lm_lp",
        "Data Scaling 3k no-LM LP Progress",
        (
            "This table summarizes the completed 3k no-LM UMS linear-probe row. "
            "It is now comparable to the completed 3k frozen-LM LP row in the current-progress table."
        ),
        lp_key_rows,
        lp_trajectory,
        (
            "- The 3k no-LM source+LP branch is complete.\n"
            "- The best no-LM LP macro-AUC occurs at step 200.\n"
            "- Use `data_scaling_3k_current_progress.*` for the matched no-LM vs frozen-LM 3k LP comparison.\n"
        ),
    )

    write_progress_outputs(
        "data_scaling_3k_current",
        "Data Scaling 3k Current Progress",
        (
            "This table summarizes completed 3k source and LP endpoints currently available. "
            "It keeps source and LP rows in the `stage` column so metric families are not mixed."
        ),
        [*source_key_rows, *lp_key_rows],
        [*source_trajectory, *lp_trajectory],
        (
            "- 3k BCE source, no-LM UMS source, and no-LM UMS LP are complete.\n"
            "- 3k frozen-LM UMS source and LP are complete.\n"
            "- In matched 3k LP final metrics, frozen-LM macro-AUC is 0.744936 versus no-LM 0.734235, while no-LM has the higher best-AUC checkpoint.\n"
            "- Source rows and LP rows must be compared only within their own stage/metric family.\n"
        ),
    )
    print(f"Wrote 3k data-scaling summaries to {rel(FINAL_DIR)}")


if __name__ == "__main__":
    main()
