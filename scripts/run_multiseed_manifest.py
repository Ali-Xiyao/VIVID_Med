"""Create seed-specific configs and a runnable multi-seed manifest."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

import yaml

from case_study_modules_common import FINAL_DIR, read_json, root_path, write_csv_rows, write_md_table


COLUMNS = ["run_id", "id", "seed", "config", "output_dir", "status", "command", "boundary"]


DEFAULT_RUNS = ["shuf_3k", "shuf_tw_clinical", "sameq_shuf_3k", "shuf_k4", "shuf_k4_tw_visual"]


def load_configs() -> dict[str, dict[str, Any]]:
    payload = read_json("outputs/next_stage_manifests/config_manifest.json") or {"configs": []}
    rows = {str(row.get("id")): row for row in payload.get("configs", [])}
    rows["shuf_3k"] = {
        "run_id": "SHUF-3k",
        "id": "shuf_3k",
        "config": "configs/qwen3vl_instruction/shuf_3k_5k.yaml",
        "boundary": "historical_p4v2_config_reference",
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
            rows.append({"run_id": run_key, "id": run_key, "seed": "", "config": "", "output_dir": "", "status": "missing_spec", "command": "", "boundary": "no_source_config"})
            continue
        for seed in seeds:
            config_out = root_path(args.config_dir) / f"{run_key}_seed{seed}.yaml"
            out_dir = root_path(args.output_root) / f"{run_key}_seed{seed}"
            status = make_seed_config(root_path(spec["config"]), config_out, seed, str(spec.get("run_id", run_key)), run_key, out_dir)
            command = f"python scripts/train_qwen3vl_clinical_instruction.py --config {config_out.as_posix()} --seed {seed}"
            if status == "planned" and (out_dir / "metrics_final.json").exists():
                status = "completed_existing"
            elif args.launch and status == "planned":
                completed = subprocess.run(command.split(), cwd=root_path("."), check=False)
                status = f"exit_{completed.returncode}"
            rows.append(
                {
                    "run_id": spec.get("run_id", run_key),
                    "id": run_key,
                    "seed": seed,
                    "config": config_out.as_posix(),
                    "output_dir": out_dir.as_posix(),
                    "status": status,
                    "command": command,
                    "boundary": spec.get("boundary", "seed_config_generated"),
                }
            )
    note = "This manifest is executable. Rows are not counted as multi-seed evidence until their seed output directory contains metrics_final.json and downstream diagnostics."
    write_csv_rows(args.output_csv, rows, COLUMNS)
    write_md_table(args.output_md, "Multi-Seed Run Manifest", rows, COLUMNS, note)


if __name__ == "__main__":
    main()

