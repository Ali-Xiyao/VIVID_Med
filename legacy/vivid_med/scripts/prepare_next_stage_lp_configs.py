"""Generate CheXpert linear-probe configs for next-stage Qwen3-VL runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = "H:/Xiyao_Wang/001_models/qwen3-vl-2b-thinking-new"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def path_exists(raw: str) -> bool:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path.exists()


def lp_config(run: dict[str, Any], train_cfg: dict[str, Any], output_root: Path, device: str) -> dict[str, Any]:
    run_id = str(run["id"])
    instruction_output = Path(str(train_cfg["training"]["output_dir"]))
    checkpoint = instruction_output / "checkpoints" / "best.pt"
    if not checkpoint.is_absolute():
        checkpoint = Path(".") / checkpoint
    output_dir = output_root / f"{run_id}_chexpert_1k"
    return {
        "experiment": {
            "id": f"lp_next_stage_{run_id}_chexpert_1k",
            "route": "qwen3vl_next_stage_vision_linear_probe",
            "source_run_id": run.get("run_id", run_id),
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
            "vision_checkpoint": checkpoint.as_posix(),
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
            "output_dir": f"./{output_dir.relative_to(ROOT).as_posix()}",
        },
        "seed": int(train_cfg.get("seed", 42)),
        "device": device,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=ROOT / "outputs/next_stage_manifests/config_manifest.json",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default=ROOT / "configs/qwen3vl_instruction/next_stage_lp",
        type=Path,
    )
    parser.add_argument(
        "--lp-output-root",
        default=ROOT / "outputs/qwen3vl_next_stage_lp_runs",
        type=Path,
    )
    parser.add_argument(
        "--lp-manifest",
        default=ROOT / "outputs/next_stage_manifests/lp_config_manifest.json",
        type=Path,
    )
    parser.add_argument("--skip-missing-train-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    device_index = 0
    for run in manifest.get("configs", []):
        train_path = str(run.get("train") or "")
        if args.skip_missing_train_data and train_path and not path_exists(train_path):
            skipped.append({"id": run.get("id"), "reason": "missing_train_data", "train": train_path})
            continue
        train_cfg = read_yaml(Path(str(run["config"])))
        device = f"cuda:{device_index % 2}"
        cfg = lp_config(run, train_cfg, args.lp_output_root, device)
        path = args.output_dir / f"lp_{run['id']}_chexpert_1k.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        rows.append(
            {
                "run_id": run.get("run_id"),
                "id": run.get("id"),
                "config": str(path),
                "source_config": run.get("config"),
                "vision_checkpoint": cfg["model"]["vision_checkpoint"],
                "output_dir": cfg["training"]["output_dir"],
                "device": device,
            }
        )
        device_index += 1

    args.lp_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.lp_manifest.write_text(
        json.dumps({"configs": rows, "skipped": skipped}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"configs": len(rows), "skipped": len(skipped), "manifest": str(args.lp_manifest)}, indent=2))


if __name__ == "__main__":
    main()
