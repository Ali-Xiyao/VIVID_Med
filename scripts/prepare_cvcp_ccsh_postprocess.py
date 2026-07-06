"""Prepare CVCP/CCSH postprocess configs and queue manifest."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs" / "final_tables"
TRAIN_MANIFEST = FINAL_DIR / "cvcp_ccsh_training_manifest.csv"
LP_CONFIG_DIR = ROOT / "configs" / "qwen3vl_instruction" / "cvcp_ccsh_lp"
OUT_ROOT = Path("F:/Xiyao_Wang/021_260129VIVID_cvcp_ccsh_outputs")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def write_md_table(path: Path, title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        values = [str(row.get(key, "")).replace("|", "\\|").replace("\n", " ") for key in columns]
        lines.append("| " + " | ".join(values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def lp_config_for(row: dict[str, str]) -> dict[str, Any]:
    run_id = row["id"]
    checkpoint = Path(row["train_output_dir"]) / "checkpoints" / "final.pt"
    return {
        "experiment": {
            "id": f"lp_{run_id}_chexpert_1k",
            "route": "qwen3vl_cvcp_ccsh_vision_linear_probe",
            "source_run_id": row["run_id"],
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
            "model_path": "H:/Xiyao_Wang/001_models/qwen3-vl-2b-thinking-new",
            "dtype": "bf16",
            "freeze_backbone": True,
            "vision_checkpoint": checkpoint.as_posix(),
        },
        "training": {
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "max_steps": 1000,
            "batch_size": 2,
            "eval_batch_size": 2,
            "uncertain_policy": "ignore",
            "log_interval": 25,
            "eval_interval": 250,
            "save_interval": 1001,
            "output_dir": (OUT_ROOT / "lp" / f"{run_id}_chexpert_1k").as_posix(),
        },
        "seed": int(row.get("seed") or 42),
        "device": "cuda:0",
    }


def main() -> None:
    train_rows = read_csv(TRAIN_MANIFEST)
    LP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for row in train_rows:
        run_id = row["id"]
        lp_cfg = lp_config_for(row)
        lp_path = LP_CONFIG_DIR / f"lp_{run_id}_chexpert_1k.yaml"
        lp_path.write_text(yaml.safe_dump(lp_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        diag_root = OUT_ROOT / "diagnostics"
        ab_root = OUT_ROOT / "ab_swap"
        rows.append(
            {
                "id": run_id,
                "run_id": row["run_id"],
                "family": row["family"],
                "seed": row.get("seed", "42"),
                "steps": row.get("steps", ""),
                "train_records": row.get("train_records", ""),
                "train_config": row["train_config"],
                "train_output_dir": row["train_output_dir"],
                "checkpoint_path": (Path(row["train_output_dir"]) / "checkpoints" / "final.pt").as_posix(),
                "lp_config": lp_path.as_posix(),
                "lp_output_dir": lp_cfg["training"]["output_dir"],
                "nih_output_dir": (OUT_ROOT / "nih_appendix" / f"{run_id}_nih_1k").as_posix(),
                "nih_max_samples": 1000,
                "val_instruction_path": row.get("val_instruction_path", ""),
                "visual_output": (diag_root / f"{run_id}_visual_dependence.json").as_posix(),
                "counterfactual_output": (diag_root / f"{run_id}_counterfactual_diagnostics.json").as_posix(),
                "paraphrase_output": (diag_root / f"{run_id}_paraphrase_robustness.json").as_posix(),
                "ab_swap_input": (ab_root / f"{run_id}_ab_swap.jsonl").as_posix(),
                "ab_swap_config": (ab_root / f"{run_id}_ab_swap_config.yaml").as_posix(),
                "ab_swap_output": (diag_root / f"{run_id}_ab_swap_counterfactual_diagnostics.json").as_posix(),
            }
        )

    columns = [
        "id",
        "run_id",
        "family",
        "seed",
        "steps",
        "train_records",
        "train_config",
        "train_output_dir",
        "checkpoint_path",
        "lp_config",
        "lp_output_dir",
        "nih_output_dir",
        "nih_max_samples",
        "val_instruction_path",
        "visual_output",
        "counterfactual_output",
        "paraphrase_output",
        "ab_swap_input",
        "ab_swap_config",
        "ab_swap_output",
    ]
    out_csv = FINAL_DIR / "cvcp_ccsh_postprocess_manifest.csv"
    write_csv(out_csv, rows, columns)
    write_md_table(FINAL_DIR / "cvcp_ccsh_postprocess_manifest.md", "CVCP/CCSH Postprocess Manifest", rows, columns)
    print(f"wrote_rows={len(rows)} manifest={out_csv}")


if __name__ == "__main__":
    main()
