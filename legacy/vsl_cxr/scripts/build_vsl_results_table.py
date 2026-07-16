"""Build VSL-CXR formal run result tables from run output directories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = Path("F:/Xiyao_Wang/021_260129VIVID_vsl_cxr_outputs/qwen3vl")
FINAL_DIR = ROOT / "outputs" / "final_tables"


def repo_rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def collect_eval_steps(run_dir: Path) -> tuple[str, str]:
    values: list[tuple[int, float]] = []
    for path in sorted(run_dir.glob("metrics_step_*.json")):
        payload = read_json(path)
        step = payload.get("global_step")
        val_loss = payload.get("val_loss")
        if isinstance(step, int) and isinstance(val_loss, (int, float)):
            values.append((step, float(val_loss)))
    if not values:
        return "", ""
    best_step, best_loss = min(values, key=lambda item: item[1])
    compact = "; ".join(f"{step}:{loss:.6f}" for step, loss in values)
    return compact, f"{best_step}:{best_loss:.6f}"


def collect_run(run_dir: Path) -> dict[str, Any]:
    config = read_json(run_dir / "config_snapshot.json")
    lp_config = read_json(run_dir / "resolved_config.json")
    final_metrics = read_json(run_dir / "metrics_final.json")
    runtime = read_json(run_dir / "runtime_summary.json")
    resolved = config.get("resolved_config") or lp_config or {}
    experiment = resolved.get("experiment") or {}
    data = resolved.get("data") or {}
    training = resolved.get("training") or {}
    model = resolved.get("model") or {}
    run_type = "linear_probe" if (run_dir / "final_probe.pt").exists() or experiment.get("id", "").endswith("_lp") else "instruction_training"
    checkpoint = run_dir / "final_probe.pt" if run_type == "linear_probe" else run_dir / "checkpoints" / "final.pt"
    eval_curve, best_eval_step = collect_eval_steps(run_dir)
    has_final = bool(final_metrics)
    has_checkpoint = checkpoint.exists()
    has_config = bool(config) or bool(lp_config)
    status = "completed" if has_final and has_checkpoint else "in_progress" if has_config else "missing"
    metrics = final_metrics.get("metrics") or {}
    return {
        "run_dir": run_dir.as_posix(),
        "run_id": experiment.get("run_id", run_dir.name),
        "experiment_id": experiment.get("id", run_dir.name),
        "route": experiment.get("route", ""),
        "data_version": experiment.get("data_version", ""),
        "run_type": run_type,
        "status": status,
        "global_step": final_metrics.get("global_step", ""),
        "best_val_loss": final_metrics.get("best_val_loss", ""),
        "final_val_loss": final_metrics.get("final_val_loss", ""),
        "macro_auc": metrics.get("macro_auc", ""),
        "macro_f1": metrics.get("macro_f1", ""),
        "micro_f1": metrics.get("micro_f1", ""),
        "eval_curve_step_val_loss": eval_curve,
        "best_eval_step_val_loss": best_eval_step,
        "elapsed_seconds": final_metrics.get("elapsed_seconds", runtime.get("elapsed_seconds", "")),
        "train_records": final_metrics.get("train_records", ""),
        "val_records": final_metrics.get("val_records", ""),
        "train_instruction_path": data.get("train_instruction_path", data.get("train_ums_path", "")),
        "val_instruction_path": data.get("val_instruction_path", data.get("val_ums_path", "")),
        "model_path": model.get("model_path", ""),
        "trainable_groups": ",".join(model.get("trainable_groups") or (["linear_probe_head"] if run_type == "linear_probe" else [])),
        "device": resolved.get("device", ""),
        "max_steps": training.get("max_steps", ""),
        "checkpoint": checkpoint.as_posix() if has_checkpoint else "",
        "metrics_final": (run_dir / "metrics_final.json").as_posix() if has_final else "",
        "config_snapshot": (run_dir / "config_snapshot.json").as_posix() if config else ((run_dir / "resolved_config.json").as_posix() if lp_config else ""),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-csv", type=Path, default=FINAL_DIR / "vsl_cxr_formal_run_results.csv")
    parser.add_argument("--out-md", type=Path, default=FINAL_DIR / "vsl_cxr_formal_run_results.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root
    rows = [
        collect_run(path)
        for path in sorted(run_root.glob("vsl_cxr_*"))
        if path.is_dir() and not path.name.endswith("_debug")
    ]
    rows.sort(key=lambda row: (str(row.get("status") != "completed"), str(row.get("run_id"))))
    columns = [
        "run_id",
        "experiment_id",
        "route",
        "data_version",
        "run_type",
        "status",
        "global_step",
        "best_val_loss",
        "final_val_loss",
        "macro_auc",
        "macro_f1",
        "micro_f1",
        "best_eval_step_val_loss",
        "elapsed_seconds",
        "train_records",
        "val_records",
        "device",
        "max_steps",
        "trainable_groups",
        "checkpoint",
        "metrics_final",
        "config_snapshot",
        "run_dir",
        "eval_curve_step_val_loss",
        "train_instruction_path",
        "val_instruction_path",
        "model_path",
    ]
    write_csv(args.out_csv, rows, columns)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(
        "# VSL-CXR Formal Run Results\n\n"
        "Generated from current run output directories. Instruction-training runs are marked completed only when "
        "`metrics_final.json` and `checkpoints/final.pt` both exist; linear-probe readout runs are marked completed "
        "only when `metrics_final.json` and `final_probe.pt` both exist.\n\n"
        + md_table(rows, columns)
        + "\n",
        encoding="utf-8",
    )
    print(f"rows={len(rows)}")
    print(f"completed={sum(1 for row in rows if row.get('status') == 'completed')}")
    print(f"csv={repo_rel(args.out_csv)}")
    print(f"md={repo_rel(args.out_md)}")


if __name__ == "__main__":
    main()
