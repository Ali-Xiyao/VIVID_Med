"""Prepare configs and manifests for CVCP/CCSH formal training queues."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "qwen3vl_instruction" / "cvcp_ccsh"
FINAL_DIR = ROOT / "outputs" / "final_tables"
DEFAULT_OUTPUT_ROOT = Path("F:/Xiyao_Wang/021_260129VIVID_cvcp_ccsh_outputs/qwen3vl")
MODEL_PATH = "H:/Xiyao_Wang/001_models/qwen3-vl-2b-thinking-new"

COLUMNS = [
    "id",
    "run_id",
    "family",
    "seed",
    "train_config",
    "train_output_dir",
    "success_path",
    "steps",
    "train_records",
    "val_instruction_path",
    "notes",
]


def read_jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in COLUMNS})


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CVCP/CCSH Training Manifest",
        "",
        "This manifest contains formal Qwen3-VL backbone training rows prepared for the CVCP/CCSH plan. Outputs are placed on the configured output root to avoid filling the project drive.",
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "| " + " | ".join("---" for _ in COLUMNS) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")).replace("|", "\\|") for key in COLUMNS) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def margin(weight: float = 0.15) -> dict[str, Any]:
    return {"enabled": True, "weight": weight, "margin": 0.2}


def tw_visual() -> dict[str, Any]:
    return {
        "enabled": True,
        "default_weight": 1.0,
        "visual_dependency_weights": {"very_high": 1.6, "high": 1.4, "medium": 1.15, "low": 0.9, "unknown": 1.0},
        "answer_type_weights": {"same_question_different_answer": 1.35, "counterfactual_choice": 1.25},
        "normalize": True,
    }


def base_config(spec: dict[str, Any], output_root: Path) -> dict[str, Any]:
    train_path = spec["train"]
    cfg: dict[str, Any] = {
        "experiment": {
            "id": spec["id"],
            "run_id": spec["run_id"],
            "route": spec["family"],
            "data_version": spec["run_id"],
        },
        "data": {
            "data_root": ".",
            "train_instruction_path": train_path,
            "val_instruction_path": spec.get("val", "outputs/instruction_data/next_stage/storymix_qa8_val.jsonl"),
            "max_length": spec.get("max_length", 768),
            "max_val_samples": spec.get("max_val_samples", 1000),
            "num_workers": 0,
        },
        "model": {
            "model_path": spec.get("model_path", MODEL_PATH),
            "dtype": "bf16",
            "trainable_groups": spec.get("trainable_groups", ["vision_tower", "visual_connector"]),
        },
        "training": {
            "learning_rate": 2.0e-5,
            "vision_learning_rate": 1.0e-5,
            "connector_learning_rate": 2.0e-5,
            "weight_decay": 0.01,
            "max_steps": spec["steps"],
            "batch_size": spec.get("batch_size", 1),
            "eval_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "log_interval": 25,
            "eval_interval": spec.get("eval_interval", 500),
            "save_interval": spec["steps"] + 1,
            "save_checkpoints": False,
            "save_best_checkpoint": False,
            "save_final_checkpoint": True,
            "output_dir": (output_root / spec["id"]).as_posix(),
        },
        "seed": spec.get("seed", 42),
        "device": "cuda:0",
    }
    if spec.get("image_margin", True):
        cfg["training"]["image_shuffle_margin"] = margin(float(spec.get("image_margin_weight", 0.15)))
    if spec.get("answer_margin", False):
        cfg["training"]["answer_margin"] = margin(float(spec.get("answer_margin_weight", 0.15)))
    if spec.get("loss_weighting") == "tw_visual":
        cfg["training"]["loss_weighting"] = tw_visual()
    if spec.get("trainable_vision_last_n"):
        cfg["model"]["trainable_vision_last_n"] = spec["trainable_vision_last_n"]
    return cfg


def specs() -> list[dict[str, Any]]:
    ns = "outputs/instruction_data/cvcp_ccsh"
    old = "outputs/instruction_data/next_stage"
    return [
        {"id": "cvcp_v1_sameq_3k", "run_id": "CVCP-v1-SAMEQ-3k", "family": "CVCP-v1", "train": f"{ns}/cvcp_v1_sameq_3k_train.jsonl", "val": f"{old}/sameq_shuf_val.jsonl", "steps": 5000, "answer_margin": True},
        {"id": "cvcp_v1_sameq_10k", "run_id": "CVCP-v1-SAMEQ-10k", "family": "CVCP-v1", "train": f"{ns}/cvcp_v1_sameq_10k_train.jsonl", "val": f"{old}/sameq_shuf_val.jsonl", "steps": 8000, "answer_margin": True},
        {"id": "cvcp_v1_sameq_full", "run_id": "CVCP-v1-SAMEQ-full", "family": "CVCP-v1", "train": f"{ns}/cvcp_v1_sameq_full_train.jsonl", "val": f"{old}/sameq_shuf_val.jsonl", "steps": 12000, "answer_margin": True},
        {"id": "cvcp_v2_shuf_k2", "run_id": "CVCP-v2-SHUF-K2", "family": "CVCP-v2", "train": f"{ns}/cvcp_v2_shuf_k2_train.jsonl", "val": f"{old}/shuf_k2_val.jsonl", "steps": 5000},
        {"id": "cvcp_v2_shuf_k4", "run_id": "CVCP-v2-SHUF-K4", "family": "CVCP-v2", "train": f"{ns}/cvcp_v2_shuf_k4_train.jsonl", "val": f"{old}/shuf_k4_val.jsonl", "steps": 5000},
        {"id": "cvcp_v2_shuf_k8", "run_id": "CVCP-v2-SHUF-K8", "family": "CVCP-v2", "train": f"{ns}/cvcp_v2_shuf_k8_train.jsonl", "val": f"{old}/shuf_k4_val.jsonl", "steps": 8000},
        {"id": "cvcp_v3_prog_3k", "run_id": "CVCP-v3-prog-3k", "family": "CVCP-v3", "train": f"{ns}/cvcp_v3_prog_3k_train.jsonl", "steps": 8000, "answer_margin": True},
        {"id": "cvcp_v3_prog_10k", "run_id": "CVCP-v3-prog-10k", "family": "CVCP-v3", "train": f"{ns}/cvcp_v3_prog_10k_train.jsonl", "steps": 12000, "answer_margin": True},
        {"id": "cvcp_v3_prog_full", "run_id": "CVCP-v3-prog-full", "family": "CVCP-v3", "train": f"{ns}/cvcp_v3_prog_full_train.jsonl", "steps": 16000, "answer_margin": True},
        {"id": "cvcp_v4_replay_10k", "run_id": "CVCP-v4-replay-10k", "family": "CVCP-v4", "train": f"{ns}/cvcp_v4_replay_10k_train.jsonl", "steps": 12000, "answer_margin": True},
        {"id": "cvcp_v4_replay_full", "run_id": "CVCP-v4-replay-full", "family": "CVCP-v4", "train": f"{ns}/cvcp_v4_replay_full_train.jsonl", "steps": 16000, "answer_margin": True},
        {"id": "cvcp_v5_cdcs_field", "run_id": "CVCP-v5-CDCS-field", "family": "CVCP-v5", "train": f"{ns}/cvcp_v5_cdcs_field_train.jsonl", "steps": 12000, "answer_margin": True},
        {"id": "cvcp_v5_cdcs_hardneg", "run_id": "CVCP-v5-CDCS-hardneg", "family": "CVCP-v5", "train": f"{ns}/cvcp_v5_cdcs_hardneg_train.jsonl", "steps": 12000, "answer_margin": True},
        {"id": "cvcp_v5_cdcs_full", "run_id": "CVCP-v5-CDCS-full", "family": "CVCP-v5", "train": f"{ns}/cvcp_v5_cdcs_full_train.jsonl", "steps": 16000, "answer_margin": True},
        {"id": "sameq_cf_10", "run_id": "SAMEQ-CF-10", "family": "SAMEQ-CF", "train": f"{ns}/sameq_cf_10_train.jsonl", "val": f"{old}/sameq_shuf_val.jsonl", "steps": 5000, "answer_margin": True},
        {"id": "sameq_cf_20", "run_id": "SAMEQ-CF-20", "family": "SAMEQ-CF", "train": f"{ns}/sameq_cf_20_train.jsonl", "val": f"{old}/sameq_shuf_val.jsonl", "steps": 5000, "answer_margin": True},
        {"id": "sameq_cf_30", "run_id": "SAMEQ-CF-30", "family": "SAMEQ-CF", "train": f"{ns}/sameq_cf_30_train.jsonl", "val": f"{old}/sameq_shuf_val.jsonl", "steps": 5000, "answer_margin": True},
        {"id": "k4_cf_20", "run_id": "K4-CF-20", "family": "K4-CF", "train": f"{ns}/k4_cf_20_train.jsonl", "val": f"{old}/shuf_k4_val.jsonl", "steps": 5000, "answer_margin": True},
        {"id": "k4_cf_20_tw", "run_id": "K4-CF-20-TW", "family": "K4-CF", "train": f"{ns}/k4_cf_20_tw_train.jsonl", "val": f"{old}/shuf_k4_val.jsonl", "steps": 5000, "answer_margin": True, "loss_weighting": "tw_visual"},
        {"id": "k4_cf_30_tw", "run_id": "K4-CF-30-TW", "family": "K4-CF", "train": f"{ns}/k4_cf_30_tw_train.jsonl", "val": f"{old}/shuf_k4_val.jsonl", "steps": 6000, "answer_margin": True, "loss_weighting": "tw_visual"},
        {"id": "dual_light", "run_id": "Dual-light", "family": "DualMargin", "train": f"{ns}/k4_cf_20_train.jsonl", "val": f"{old}/shuf_k4_val.jsonl", "steps": 5000, "image_margin_weight": 0.1, "answer_margin": True, "answer_margin_weight": 0.1},
        {"id": "dual_img_heavy", "run_id": "Dual-img-heavy", "family": "DualMargin", "train": f"{ns}/k4_cf_20_train.jsonl", "val": f"{old}/shuf_k4_val.jsonl", "steps": 5000, "image_margin_weight": 0.3, "answer_margin": True, "answer_margin_weight": 0.1},
        {"id": "dual_answer_heavy", "run_id": "Dual-answer-heavy", "family": "DualMargin", "train": f"{ns}/k4_cf_20_train.jsonl", "val": f"{old}/shuf_k4_val.jsonl", "steps": 5000, "image_margin_weight": 0.1, "answer_margin": True, "answer_margin_weight": 0.3},
        {"id": "dual_balanced", "run_id": "Dual-balanced", "family": "DualMargin", "train": f"{ns}/k4_cf_20_train.jsonl", "val": f"{old}/shuf_k4_val.jsonl", "steps": 5000, "image_margin_weight": 0.2, "answer_margin": True, "answer_margin_weight": 0.2},
        {"id": "sameq_tw_visual", "run_id": "SAMEQ-TW-visual", "family": "TokenWeighting", "train": f"{ns}/cvcp_v1_sameq_10k_train.jsonl", "val": f"{old}/sameq_shuf_val.jsonl", "steps": 5000, "answer_margin": True, "loss_weighting": "tw_visual"},
        {"id": "k4_tw_visual", "run_id": "K4-TW-visual", "family": "TokenWeighting", "train": f"{ns}/cvcp_v2_shuf_k4_train.jsonl", "val": f"{old}/shuf_k4_val.jsonl", "steps": 5000, "loss_weighting": "tw_visual"},
        {"id": "cvcp_tw_visual", "run_id": "CVCP-TW-visual", "family": "TokenWeighting", "train": f"{ns}/cvcp_v4_replay_10k_train.jsonl", "steps": 12000, "answer_margin": True, "loss_weighting": "tw_visual"},
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--manifest-csv", type=Path, default=FINAL_DIR / "cvcp_ccsh_training_manifest.csv")
    parser.add_argument("--manifest-md", type=Path, default=FINAL_DIR / "cvcp_ccsh_training_manifest.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.config_dir.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in specs():
        cfg = base_config(spec, args.output_root)
        config_path = args.config_dir / f"{spec['id']}.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        output_dir = Path(cfg["training"]["output_dir"])
        rows.append(
            {
                "id": spec["id"],
                "run_id": spec["run_id"],
                "family": spec["family"],
                "seed": cfg["seed"],
                "train_config": config_path.as_posix(),
                "train_output_dir": output_dir.as_posix(),
                "success_path": (output_dir / "metrics_final.json").as_posix(),
                "steps": spec["steps"],
                "train_records": read_jsonl_count(ROOT / spec["train"]),
                "val_instruction_path": cfg["data"]["val_instruction_path"],
                "notes": "final checkpoint only; output_dir on non-project drive",
            }
        )
    write_csv(args.manifest_csv, rows)
    write_md(args.manifest_md, rows)
    print(f"wrote_rows={len(rows)}")
    print(f"manifest={args.manifest_csv.as_posix()}")


if __name__ == "__main__":
    main()
