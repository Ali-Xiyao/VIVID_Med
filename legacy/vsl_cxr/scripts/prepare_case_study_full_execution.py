"""Prepare executable configs/manifests for real case-study stability runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs" / "final_tables"
MODEL_PATH = "H:/Xiyao_Wang/001_models/qwen3-vl-2b-thinking-new"

DEFAULT_RUNS = ["shuf_3k", "shuf_tw_clinical", "sameq_shuf_3k", "shuf_k4"]
COLUMNS = [
    "run_id",
    "id",
    "seed",
    "val_instruction_path",
    "train_config",
    "train_output_dir",
    "train_status",
    "train_command",
    "lp_config",
    "lp_output_dir",
    "nih_output_dir",
    "visual_output",
    "counterfactual_output",
    "ab_swap_input",
    "ab_swap_config",
    "ab_swap_output",
    "paraphrase_output",
    "boundary",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in COLUMNS})


def write_md(path: Path, title: str, rows: list[dict[str, Any]], note: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if note:
        lines.extend([note, ""])
    lines.append("| " + " | ".join(COLUMNS) + " |")
    lines.append("| " + " | ".join("---" for _ in COLUMNS) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")).replace("|", "\\|") for key in COLUMNS) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def root_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def load_run_specs() -> dict[str, dict[str, Any]]:
    manifest = read_json(ROOT / "outputs" / "next_stage_manifests" / "config_manifest.json")
    specs = {str(row.get("id")): dict(row) for row in manifest.get("configs", [])}
    specs["shuf_3k"] = {
        "run_id": "SHUF-3k",
        "id": "shuf_3k",
        "config": "configs/qwen3vl_instruction/shuf_3k_5k.yaml",
        "boundary": "historical_p4v2_config_reference",
    }
    return specs


def write_seed_config(spec: dict[str, Any], run_key: str, seed: int, output_dir: Path, config_dir: Path) -> Path:
    source = root_path(spec["config"])
    if not source.exists():
        raise FileNotFoundError(f"Missing source config for {run_key}: {source}")
    cfg = read_yaml(source)
    cfg["seed"] = seed
    cfg["device"] = "cuda:0"
    cfg.setdefault("experiment", {})["id"] = f"{run_key}_seed{seed}"
    cfg.setdefault("experiment", {})["run_id"] = f"{spec.get('run_id', run_key)}-seed{seed}"
    cfg.setdefault("data", {})["num_workers"] = 0
    cfg.setdefault("training", {})["output_dir"] = output_dir.as_posix()
    out_cfg = config_dir / f"{run_key}_seed{seed}.yaml"
    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    out_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out_cfg


def write_lp_config(train_config: Path, run_key: str, seed: int, train_output_dir: Path, lp_dir: Path, lp_output_root: Path) -> Path:
    train_cfg = read_yaml(train_config)
    output_dir = lp_output_root / f"{run_key}_seed{seed}_chexpert_1k"
    cfg = {
        "experiment": {
            "id": f"lp_case_study_{run_key}_seed{seed}_chexpert_1k",
            "route": "qwen3vl_case_study_vision_linear_probe",
            "source_run_id": train_cfg.get("experiment", {}).get("run_id", f"{run_key}-seed{seed}"),
        },
        "data": {
            "data_root": "H:/Xiyao_Wang/000_Public Dataset",
            "train_ums_path": "./data/splits/chexpert_train_1k.jsonl",
            "val_ums_path": "./data/splits/chexpert_val_fixed.jsonl",
            "use_common_labels_only": True,
            "max_train_samples": 1000,
            "max_val_samples": 1000,
            "num_workers": 0,
            "processor_prompt": "Classify the chest X-ray findings.",
        },
        "model": {
            "model_path": str(train_cfg.get("model", {}).get("model_path", MODEL_PATH)),
            "dtype": str(train_cfg.get("model", {}).get("dtype", "bf16")),
            "freeze_backbone": True,
            "vision_checkpoint": (train_output_dir / "checkpoints" / "best.pt").as_posix(),
        },
        "training": {
            "learning_rate": 1.0e-3,
            "weight_decay": 0.01,
            "max_steps": 1000,
            "batch_size": 2,
            "eval_batch_size": 2,
            "uncertain_policy": "ignore",
            "log_interval": 25,
            "eval_interval": 250,
            "save_interval": 500,
            "output_dir": output_dir.as_posix(),
        },
        "seed": seed,
        "device": "cuda:0",
    }
    path = lp_dir / f"lp_{run_key}_seed{seed}_chexpert_1k.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", dest="run_ids", default=[])
    parser.add_argument("--seed", action="append", type=int, dest="seeds", default=[])
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs/qwen3vl_instruction/case_study_multiseed")
    parser.add_argument("--lp-config-dir", type=Path, default=ROOT / "configs/qwen3vl_instruction/case_study_multiseed_lp")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/qwen3vl_case_study_multiseed")
    parser.add_argument("--lp-output-root", type=Path, default=ROOT / "outputs/qwen3vl_case_study_multiseed_lp")
    parser.add_argument("--transfer-root", type=Path, default=ROOT / "outputs/qwen3vl_case_study_multiseed_transfer")
    parser.add_argument("--diagnostic-root", type=Path, default=ROOT / "outputs/qwen3vl_case_study_multiseed_diagnostics")
    parser.add_argument("--output-csv", type=Path, default=FINAL_DIR / "case_study_full_execution_manifest.csv")
    parser.add_argument("--output-md", type=Path, default=FINAL_DIR / "case_study_full_execution_manifest.md")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs/next_stage_manifests/case_study_full_execution_manifest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_ids = args.run_ids or DEFAULT_RUNS
    seeds = args.seeds or [1, 2, 3]
    specs = load_run_specs()
    rows: list[dict[str, Any]] = []
    for run_key in run_ids:
        spec = specs.get(run_key)
        if not spec:
            raise KeyError(f"Unknown run id: {run_key}")
        for seed in seeds:
            train_output_dir = args.output_root / f"{run_key}_seed{seed}"
            train_config = write_seed_config(spec, run_key, seed, train_output_dir, args.config_dir)
            train_cfg = read_yaml(train_config)
            lp_config = write_lp_config(train_config, run_key, seed, train_output_dir, args.lp_config_dir, args.lp_output_root)
            run_name = f"{run_key}_seed{seed}"
            train_status = "completed_existing" if (train_output_dir / "metrics_final.json").exists() else "planned"
            rows.append(
                {
                    "run_id": f"{spec.get('run_id', run_key)}-seed{seed}",
                    "id": run_name,
                    "seed": seed,
                    "val_instruction_path": train_cfg.get("data", {}).get("val_instruction_path", ""),
                    "train_config": train_config.as_posix(),
                    "train_output_dir": train_output_dir.as_posix(),
                    "train_status": train_status,
                    "train_command": f"conda run -n vivid python scripts/train_qwen3vl_clinical_instruction.py --config {train_config.as_posix()} --seed {seed}",
                    "lp_config": lp_config.as_posix(),
                    "lp_output_dir": (args.lp_output_root / f"{run_key}_seed{seed}_chexpert_1k").as_posix(),
                    "nih_output_dir": (args.transfer_root / f"{run_key}_seed{seed}_nih_available").as_posix(),
                    "visual_output": (args.diagnostic_root / f"{run_key}_seed{seed}_visual_dependence.json").as_posix(),
                    "counterfactual_output": (args.diagnostic_root / f"{run_key}_seed{seed}_counterfactual.json").as_posix(),
                    "ab_swap_input": (args.diagnostic_root / f"{run_key}_seed{seed}_ab_swap.jsonl").as_posix(),
                    "ab_swap_config": (args.diagnostic_root / f"{run_key}_seed{seed}_ab_swap_config.yaml").as_posix(),
                    "ab_swap_output": (args.diagnostic_root / f"{run_key}_seed{seed}_ab_swap_counterfactual.json").as_posix(),
                    "paraphrase_output": (args.diagnostic_root / f"{run_key}_seed{seed}_paraphrase.json").as_posix(),
                    "boundary": spec.get("boundary", "real_seed_training_required"),
                }
            )
    write_csv(args.output_csv, rows)
    write_md(args.output_md, "Case Study Full Execution Manifest", rows, "Rows are complete only after train, LP, NIH available transfer, and diagnostics artifacts exist.")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"runs": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"runs": len(rows), "csv": str(args.output_csv), "json": str(args.output_json)}, indent=2))


if __name__ == "__main__":
    main()
