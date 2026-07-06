"""Create seed-specific configs and a runnable multi-seed manifest."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from case_study_modules_common import FINAL_DIR, read_json, root_path, write_csv_rows, write_md_table


COLUMNS = [
    "run_id",
    "id",
    "base_id",
    "seed",
    "config",
    "output_dir",
    "train_config",
    "train_output_dir",
    "success_path",
    "val_instruction_path",
    "status",
    "latest_formal_eval_step",
    "latest_formal_val_loss",
    "latest_live_step",
    "command",
    "boundary",
]


DEFAULT_RUNS = ["shuf_3k", "shuf_tw_clinical", "sameq_shuf_3k", "shuf_k4", "shuf_k4_tw_visual"]
CVCP_TRAINING_MANIFEST = "outputs/final_tables/cvcp_ccsh_training_manifest.csv"


def load_configs() -> dict[str, dict[str, Any]]:
    payload = read_json("outputs/next_stage_manifests/config_manifest.json") or {"configs": []}
    rows = {str(row.get("id")): row for row in payload.get("configs", [])}
    rows["shuf_3k"] = {
        "run_id": "SHUF-3k",
        "id": "shuf_3k",
        "config": "configs/qwen3vl_instruction/shuf_3k_5k.yaml",
        "boundary": "historical_p4v2_config_reference",
    }
    cvcp_manifest = root_path(CVCP_TRAINING_MANIFEST)
    if cvcp_manifest.exists():
        with cvcp_manifest.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                run_key = str(row.get("id", ""))
                if not run_key:
                    continue
                rows[run_key] = {
                    "run_id": str(row.get("run_id", run_key)),
                    "id": run_key,
                    "config": str(row.get("train_config", "")),
                    "val_instruction_path": str(row.get("val_instruction_path", "")),
                    "boundary": "cvcp_v4_seed_sweep_manifest",
                }
    return rows


def make_seed_config(source: Path, output: Path, seed: int, run_id: str, run_key: str, output_dir: Path) -> str:
    if not source.exists():
        return "source_config_missing"
    cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    cfg["seed"] = seed
    cfg.setdefault("experiment", {})["id"] = f"{run_key}_seed{seed}"
    cfg.setdefault("experiment", {})["run_id"] = f"{run_id}-seed{seed}"
    cfg.setdefault("training", {})["output_dir"] = output_dir.as_posix()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return "planned"


def derive_seed_status(output_dir: Path) -> str:
    if (output_dir / "metrics_final.json").exists():
        return "completed_existing"
    if not output_dir.exists():
        return "queued"
    progress_markers = [
        output_dir / "training_log.txt",
        output_dir / "progress.json",
        output_dir / "config_snapshot.json",
        output_dir / "resolved_config.yaml",
    ]
    if any(path.exists() for path in progress_markers):
        return "active"
    if any(output_dir.glob("metrics_step_*.json")):
        return "active"
    if (output_dir / "checkpoints").exists():
        return "active"
    return "output_dir_created"


def latest_formal_eval(output_dir: Path) -> tuple[str, str]:
    latest_step = ""
    latest_loss = ""
    best_file: Path | None = None
    best_step = -1
    for path in output_dir.glob("metrics_step_*.json"):
        match = re.match(r"metrics_step_(\d+)\.json$", path.name)
        if not match:
            continue
        step = int(match.group(1))
        if step > best_step:
            best_step = step
            best_file = path
    if best_file is None:
        return latest_step, latest_loss
    latest_step = str(best_step)
    try:
        payload = json.loads(best_file.read_text(encoding="utf-8"))
        value = payload.get("val_loss", "")
        latest_loss = "" if value in (None, "") else str(value)
    except (OSError, json.JSONDecodeError):
        latest_loss = ""
    return latest_step, latest_loss


def latest_live_step(output_dir: Path) -> str:
    log_path = output_dir / "training_log.txt"
    if not log_path.exists():
        return ""
    best_step = -1
    pattern = re.compile(r'"global_step"\s*:\s*(\d+)')
    try:
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                match = pattern.search(line)
                if match:
                    best_step = max(best_step, int(match.group(1)))
    except OSError:
        return ""
    return "" if best_step < 0 else str(best_step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", dest="run_ids", default=[])
    parser.add_argument("--seed", action="append", type=int, dest="seeds", default=[])
    parser.add_argument("--config-dir", type=Path, default=Path("configs/qwen3vl_instruction/case_study_multiseed"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/qwen3vl_case_study_multiseed"))
    parser.add_argument("--launch", action="store_true", help="Run generated training commands. Default only writes a manifest.")
    parser.add_argument("--output-csv", type=Path, default=FINAL_DIR / "multiseed_run_manifest.csv")
    parser.add_argument("--output-md", type=Path, default=FINAL_DIR / "multiseed_run_manifest.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_ids = args.run_ids or DEFAULT_RUNS
    seeds = args.seeds or [0, 1, 2]
    configs = load_configs()
    rows: list[dict[str, Any]] = []
    for run_key in run_ids:
        spec = configs.get(run_key)
        if not spec:
            rows.append(
                {
                    "run_id": run_key,
                    "id": run_key,
                    "base_id": run_key,
                    "seed": "",
                    "config": "",
                    "output_dir": "",
                    "train_config": "",
                    "train_output_dir": "",
                    "success_path": "",
                    "val_instruction_path": "",
                    "status": "missing_spec",
                    "command": "",
                    "boundary": "no_source_config",
                }
            )
            continue
        for seed in seeds:
            seed_id = f"{run_key}_seed{seed}"
            config_out = root_path(args.config_dir) / f"{run_key}_seed{seed}.yaml"
            out_dir = root_path(args.output_root) / f"{run_key}_seed{seed}"
            status = make_seed_config(root_path(spec["config"]), config_out, seed, str(spec.get("run_id", run_key)), run_key, out_dir)
            command = f"python scripts/train_qwen3vl_clinical_instruction.py --config {config_out.as_posix()} --seed {seed}"
            if status == "planned":
                current_status = derive_seed_status(out_dir)
                if args.launch and current_status == "queued":
                    completed = subprocess.run(command.split(), cwd=root_path("."), check=False)
                    status = f"exit_{completed.returncode}"
                else:
                    status = current_status
            rows.append(
                {
                    "run_id": f"{spec.get('run_id', run_key)}-seed{seed}",
                    "id": seed_id,
                    "base_id": run_key,
                    "seed": seed,
                    "config": config_out.as_posix(),
                    "output_dir": out_dir.as_posix(),
                    "train_config": config_out.as_posix(),
                    "train_output_dir": out_dir.as_posix(),
                    "success_path": (out_dir / "metrics_final.json").as_posix(),
                    "val_instruction_path": spec.get("val_instruction_path", ""),
                    "status": status,
                    "latest_formal_eval_step": latest_formal_eval(out_dir)[0],
                    "latest_formal_val_loss": latest_formal_eval(out_dir)[1],
                    "latest_live_step": latest_live_step(out_dir),
                    "command": command,
                    "boundary": spec.get("boundary", "seed_config_generated"),
                }
            )
    note = "This manifest is executable. Rows are not counted as multi-seed evidence until their seed output directory contains metrics_final.json and downstream diagnostics."
    write_csv_rows(args.output_csv, rows, COLUMNS)
    write_md_table(args.output_md, "Multi-Seed Run Manifest", rows, COLUMNS, note)


if __name__ == "__main__":
    main()

