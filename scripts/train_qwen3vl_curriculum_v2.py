"""Wrapper for launching curriculum-v2 Qwen3-VL instruction training."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

import yaml

from case_study_modules_common import FINAL_DIR, root_path, write_csv_rows, write_md_table


COLUMNS = ["status", "config", "output_dir", "command", "boundary"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=Path("configs/qwen3vl_instruction/next_stage/prog_mix.yaml"))
    parser.add_argument("--train-jsonl", type=Path, default=Path("outputs/instruction_data/next_stage/cur_v2_progressive_replay_train.jsonl"))
    parser.add_argument("--val-jsonl", type=Path, default=Path("outputs/instruction_data/next_stage/storymix_qa8_val.jsonl"))
    parser.add_argument("--output-config", type=Path, default=Path("configs/qwen3vl_instruction/case_study_modules/cur_v2_progressive_replay.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen3vl_case_study_modules/cur_v2_progressive_replay"))
    parser.add_argument("--max-steps", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--manifest-csv", type=Path, default=FINAL_DIR / "curriculum_v2_training_manifest.csv")
    parser.add_argument("--manifest-md", type=Path, default=FINAL_DIR / "curriculum_v2_training_manifest.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = "planned"
    boundary = "generated_config_only"
    base = root_path(args.base_config)
    if not base.exists():
        status = "missing_base_config"
        command = ""
        output_dir = root_path(args.output_dir)
    else:
        cfg = yaml.safe_load(base.read_text(encoding="utf-8"))
        cfg["seed"] = args.seed
        cfg.setdefault("experiment", {})["id"] = "cur_v2_progressive_replay"
        cfg.setdefault("experiment", {})["run_id"] = "CUR-v2-progressive-replay"
        cfg.setdefault("data", {})["train_instruction_path"] = args.train_jsonl.as_posix()
        cfg.setdefault("data", {})["val_instruction_path"] = args.val_jsonl.as_posix()
        cfg.setdefault("data", {})["num_workers"] = 0
        cfg.setdefault("training", {})["output_dir"] = args.output_dir.as_posix()
        cfg.setdefault("training", {})["max_steps"] = args.max_steps
        out_cfg = root_path(args.output_config)
        out_cfg.parent.mkdir(parents=True, exist_ok=True)
        out_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        output_dir = root_path(args.output_dir)
        command = f"python scripts/train_qwen3vl_clinical_instruction.py --config {out_cfg.as_posix()} --seed {args.seed}"
        if (output_dir / "metrics_final.json").exists():
            status = "completed_existing"
            boundary = "metrics_final_exists"
        elif args.run:
            completed = subprocess.run(command.split(), cwd=root_path("."), check=False)
            status = f"exit_{completed.returncode}"
            boundary = "launched"
    row = {"status": status, "config": args.output_config.as_posix(), "output_dir": output_dir.as_posix(), "command": command, "boundary": boundary}
    write_csv_rows(args.manifest_csv, [row], COLUMNS)
    write_md_table(args.manifest_md, "Curriculum v2 Training Manifest", [row], COLUMNS)


if __name__ == "__main__":
    main()
