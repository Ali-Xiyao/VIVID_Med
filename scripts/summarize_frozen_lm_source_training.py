"""Summarize frozen-LM source training logs and checkpoints."""

from __future__ import annotations

import csv
import re
import warnings
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs" / "final_tables"


SUMMARY_COLUMNS = [
    "run_id",
    "schema_level",
    "status",
    "final_step",
    "final_val_loss_log",
    "best_step_log",
    "best_val_loss_log",
    "checkpoint_best_step",
    "checkpoint_best_val_loss",
    "runtime",
    "exitcode",
    "boundary",
    "evidence_paths",
]

TRACE_COLUMNS = ["run_id", "step", "val_loss"]


RUNS = [
    {
        "run_id": "frozen_lm_s1_state_only_formal_source",
        "schema_level": "S1 state_only",
        "logs": ["outputs/logs/schema_frozen_lm_s1_source_gpu0.log"],
        "checkpoint_dir": "outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints",
        "boundary": "source checkpoint only; final checkpoint completed at step 10000 while best checkpoint is step 9000; paired frozen-LM S1 LP is recorded separately",
    },
    {
        "run_id": "frozen_lm_s2_state_answerability_formal_source",
        "schema_level": "S2 state_answerability",
        "logs": ["outputs/logs/schema_frozen_lm_s2_source_gpu0.log"],
        "checkpoint_dir": "outputs/schema_sweep/frozen_lm_s2_state_answerability/checkpoints",
        "boundary": "source checkpoint only; frozen-LM LP rows are recorded separately",
    },
    {
        "run_id": "frozen_lm_s3_state_uncertainty_formal_source",
        "schema_level": "S3 state_uncertainty",
        "logs": [
            "outputs/logs/schema_frozen_lm_s3_source_gpu0.log",
            "outputs/logs/schema_frozen_lm_s3_source_resume8000_gpu0.log",
        ],
        "checkpoint_dir": "outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints",
        "boundary": "source checkpoint only; original log was externally interrupted at step 8128 and recovered by resume; paired frozen-LM S3 LP is recorded separately",
    },
]


def fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row.get(column, "")) for column in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body]) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_meta(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "", ""
    warnings.filterwarnings("ignore", message="You are using `torch.load`")
    state = torch.load(path, map_location="cpu")
    try:
        return str(state.get("global_step", "")), fmt(state.get("best_val_loss"))
    finally:
        del state


def parse_last_progress(text: str) -> str:
    matches = re.findall(r"Training:\s+\d+%.*?\|\s*(\d+/\d+)\s+\[([^\]]+)\]", text)
    if not matches:
        return ""
    step_text, runtime = matches[-1]
    return f"interrupted_at_{step_text} after {runtime}"


def parse_run(run: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    logs = run.get("logs") or [run["log"]]
    texts: list[tuple[str, str]] = []
    for rel_log in logs:
        log_path = ROOT / rel_log
        if not log_path.exists():
            raise FileNotFoundError(log_path)
        texts.append((rel_log, log_path.read_text(encoding="utf-8", errors="replace")))

    losses: list[dict[str, Any]] = []
    runtimes: list[str] = []
    exitcodes: list[str] = []
    for rel_log, text in texts:
        for match in re.finditer(r"Step\s+(\d+):\s+val_loss\s+=\s+([0-9.]+)", text):
            losses.append(
                {
                    "run_id": run["run_id"],
                    "step": int(match.group(1)),
                    "val_loss": fmt(match.group(2)),
                }
            )
        completion = re.findall(r"Training:\s+100%.*?\[([^\]]+)\]", text)
        if completion:
            runtimes.append(f"{rel_log}:{completion[-1]}")
        elif "^C" in text:
            interrupted = parse_last_progress(text)
            runtimes.append(f"{rel_log}:{interrupted or 'interrupted'}")
        exitcodes.extend(re.findall(r"EXITCODE\s+(\d+)", text))
    if not losses:
        raise RuntimeError(f"No val_loss entries found in {logs}")

    runtime = " | ".join(runtimes)
    exitcode = exitcodes[-1] if exitcodes else ""
    final = max(losses, key=lambda row: row["step"])
    best = min(losses, key=lambda row: (float(row["val_loss"]), -int(row["step"])))

    checkpoint_dir = ROOT / run["checkpoint_dir"]
    best_step, best_val = checkpoint_meta(checkpoint_dir / "best.pt")
    final_checkpoint_step, _ = checkpoint_meta(checkpoint_dir / "final.pt")
    step_checkpoint_step, _ = checkpoint_meta(checkpoint_dir / "step_10000.pt")
    paths = [
        *logs,
        f"{run['checkpoint_dir']}/best.pt",
        f"{run['checkpoint_dir']}/final.pt",
        f"{run['checkpoint_dir']}/step_10000.pt",
    ]
    row = {
        "run_id": run["run_id"],
        "schema_level": run["schema_level"],
        "status": (
            "completed"
            if exitcode == "0" and final_checkpoint_step == "10000" and step_checkpoint_step == "10000"
            else "failed_or_unknown"
        ),
        "final_step": final["step"],
        "final_val_loss_log": final["val_loss"],
        "best_step_log": best["step"],
        "best_val_loss_log": best["val_loss"],
        "checkpoint_best_step": best_step,
        "checkpoint_best_val_loss": best_val,
        "runtime": runtime,
        "exitcode": exitcode,
        "boundary": run["boundary"],
        "evidence_paths": "; ".join(paths),
    }
    return row, losses


def main() -> None:
    summary_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for run in RUNS:
        summary, trace = parse_run(run)
        summary_rows.append(summary)
        trace_rows.extend(trace)

    write_csv(FINAL_DIR / "frozen_lm_source_training_summary.csv", summary_rows, SUMMARY_COLUMNS)
    write_csv(FINAL_DIR / "frozen_lm_source_val_loss_trace.csv", trace_rows, TRACE_COLUMNS)
    (FINAL_DIR / "frozen_lm_source_training_summary.md").write_text(
        "# Frozen-LM Source Training Summary\n\n"
        + markdown_table(summary_rows, SUMMARY_COLUMNS)
        + "\n## Boundary\n\n"
        + "- These rows summarize source-training loss/checkpoint evidence only.\n"
        + "- They are not downstream LP/classification metrics.\n",
        encoding="utf-8",
    )
    (FINAL_DIR / "frozen_lm_source_val_loss_trace.md").write_text(
        "# Frozen-LM Source Validation-Loss Trace\n\n" + markdown_table(trace_rows, TRACE_COLUMNS),
        encoding="utf-8",
    )
    print(f"Wrote {len(summary_rows)} summary rows and {len(trace_rows)} trace rows to {FINAL_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
