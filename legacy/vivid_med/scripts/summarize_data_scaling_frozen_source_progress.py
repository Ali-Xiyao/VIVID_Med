"""Summarize data-scaling frozen-LM source logs and checkpoints."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs" / "final_tables"

SUMMARY_COLUMNS = [
    "run_id",
    "method",
    "stage",
    "sample_size",
    "status",
    "latest_step",
    "latest_val_loss",
    "best_step",
    "best_val_loss",
    "exitcode",
    "checkpoint_best_exists",
    "checkpoint_final_exists",
    "checkpoint_step_10000_exists",
    "metrics_path",
    "claim_boundary",
]

TRACE_COLUMNS = [
    "run_id",
    "method",
    "stage",
    "sample_size",
    "step",
    "val_loss",
    "metrics_path",
    "claim_boundary",
]

RUNS = [
    {
        "run_id": "frozen_lm_ums_10k",
        "sample_size": "10000",
        "log": ROOT / "outputs" / "logs" / "data_scaling_frozen_lm_ums_10k_source_gpu0.log",
        "checkpoint_dir": ROOT / "outputs" / "data_scaling" / "frozen_lm_ums_10k" / "checkpoints",
    },
    {
        "run_id": "frozen_lm_ums_30k",
        "sample_size": "29000",
        "log": ROOT / "outputs" / "logs" / "data_scaling_frozen_lm_ums_30k_source_gpu1.log",
        "checkpoint_dir": ROOT / "outputs" / "data_scaling" / "frozen_lm_ums_30k" / "checkpoints",
    },
]


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.6f}"


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row.get(column, "") for column in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body]) + "\n"


def parse_run(run: dict[str, Any]) -> tuple[dict[str, str], list[dict[str, str]]]:
    log_path = Path(run["log"])
    if not log_path.exists():
        raise FileNotFoundError(rel(log_path))
    text = log_path.read_text(encoding="utf-8", errors="replace")
    losses = [
        (int(match.group(1)), float(match.group(2)))
        for match in re.finditer(r"Step\s+(\d+):\s+val_loss\s+=\s+([0-9.]+)", text)
    ]
    if not losses:
        raise RuntimeError(f"No validation losses found in {rel(log_path)}")

    latest_step, latest_loss = max(losses, key=lambda item: item[0])
    best_step, best_loss = min(losses, key=lambda item: (item[1], -item[0]))
    exitcodes = re.findall(r"EXITCODE\s+(\d+)", text)
    exitcode = exitcodes[-1] if exitcodes else ""

    checkpoint_dir = Path(run["checkpoint_dir"])
    best_exists = (checkpoint_dir / "best.pt").exists()
    final_exists = (checkpoint_dir / "final.pt").exists()
    step_10000_exists = (checkpoint_dir / "step_10000.pt").exists()
    status = "completed" if exitcode == "0" and final_exists and step_10000_exists else "in_progress"
    if exitcode and exitcode != "0":
        status = "failed"

    base = {
        "run_id": str(run["run_id"]),
        "method": "frozen-LM UMS",
        "stage": "source_loss_only",
        "sample_size": str(run["sample_size"]),
        "metrics_path": rel(log_path),
        "claim_boundary": "source_loss_only_not_lp_metrics",
    }
    summary = {
        **base,
        "status": status,
        "latest_step": str(latest_step),
        "latest_val_loss": fmt(latest_loss),
        "best_step": str(best_step),
        "best_val_loss": fmt(best_loss),
        "exitcode": exitcode,
        "checkpoint_best_exists": str(best_exists).lower(),
        "checkpoint_final_exists": str(final_exists).lower(),
        "checkpoint_step_10000_exists": str(step_10000_exists).lower(),
    }
    trace = [
        {
            **base,
            "step": str(step),
            "val_loss": fmt(loss),
        }
        for step, loss in losses
    ]
    return summary, trace


def main() -> None:
    summary_rows: list[dict[str, str]] = []
    trace_rows: list[dict[str, str]] = []
    for run in RUNS:
        if not Path(run["log"]).exists():
            continue
        summary, trace = parse_run(run)
        summary_rows.append(summary)
        trace_rows.extend(trace)

    write_csv(FINAL_DIR / "data_scaling_frozen_source_progress.csv", summary_rows, SUMMARY_COLUMNS)
    write_csv(FINAL_DIR / "data_scaling_frozen_source_trajectory.csv", trace_rows, TRACE_COLUMNS)
    (FINAL_DIR / "data_scaling_frozen_source_progress.md").write_text(
        "# Data Scaling Frozen-LM Source Progress\n\n"
        + "These rows summarize source-training validation loss/checkpoint evidence only; they are not LP macro-AUC/F1 metrics.\n\n"
        + "## Summary\n\n"
        + markdown_table(summary_rows, SUMMARY_COLUMNS)
        + "\n## Trajectory\n\n"
        + markdown_table(trace_rows, TRACE_COLUMNS),
        encoding="utf-8",
    )
    print(f"Wrote {len(summary_rows)} frozen source rows to {rel(FINAL_DIR)}")


if __name__ == "__main__":
    main()
