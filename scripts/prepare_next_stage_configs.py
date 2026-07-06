"""Generate Qwen3-VL config YAMLs for next-stage experiment runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = "H:/Xiyao_Wang/001_models/qwen3-vl-2b-thinking-new"


LOSS_WEIGHTING = {
    "tw-role": {
        "enabled": True,
        "base_weight": 1.0,
        "answer_type_weights": {
            "counterfactual_choice": 1.5,
            "same_question_different_answer": 1.5,
            "laterality_location": 1.8,
            "finding_verification": 1.1,
            "uncertainty": 1.2,
            "answerability": 1.1,
        },
        "visual_dependency_weights": {"very_high": 1.5, "high": 1.25, "medium": 1.0, "low": 0.9},
        "quality_flag_weights": {"hard_image_shuffle": 1.5, "standardized_ab": 1.3},
    },
    "tw-visual": {
        "enabled": True,
        "base_weight": 1.0,
        "answer_type_weights": {
            "same_question_different_answer": 2.0,
            "counterfactual_choice": 1.8,
            "laterality_location": 2.0,
            "finding_verification": 1.2,
            "uncertainty": 1.5,
            "answerability": 1.2,
        },
        "visual_dependency_weights": {"very_high": 2.0, "high": 1.5, "medium": 1.0, "low": 0.7},
        "quality_flag_weights": {"hard_image_shuffle": 2.0, "multi_negative_k4": 1.2, "same_question": 1.3},
    },
    "tw-clinical": {
        "enabled": True,
        "base_weight": 1.0,
        "answer_type_weights": {
            "counterfactual_choice": 1.5,
            "same_question_different_answer": 1.6,
            "laterality_location": 1.8,
            "evidence_phrase": 1.3,
            "uncertainty": 1.4,
            "answerability": 1.2,
        },
        "visual_dependency_weights": {"very_high": 1.7, "high": 1.35, "medium": 1.0, "low": 0.85},
        "quality_flag_weights": {"hard_image_shuffle": 1.5, "rare_finding": 1.3},
    },
}


def base_config(spec: dict[str, Any]) -> dict[str, Any]:
    max_steps = int(spec.get("max_steps", 5000))
    cfg: dict[str, Any] = {
        "experiment": {
            "id": spec["id"],
            "run_id": spec["run_id"],
            "route": spec.get("route", "next_stage"),
            "data_version": spec.get("data_version", spec["run_id"]),
        },
        "data": {
            "data_root": spec.get("data_root", "."),
            "train_instruction_path": spec["train"],
            "val_instruction_path": spec.get("val", "./outputs/instruction_data/glm_validated/d6_hard_cf_val200.jsonl"),
            "max_length": int(spec.get("max_length", 768)),
            "max_val_samples": int(spec.get("max_val_samples", 1000)),
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
            "max_steps": max_steps,
            "batch_size": int(spec.get("batch_size", 1)),
            "eval_batch_size": int(spec.get("eval_batch_size", 1)),
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "log_interval": 25,
            "eval_interval": int(spec.get("eval_interval", 500)),
            "save_interval": int(spec.get("save_interval", 1000)),
            "save_checkpoints": True,
            "output_dir": f"./outputs/qwen3vl_instruction/next_stage/{spec['id']}",
        },
        "seed": int(spec.get("seed", 42)),
        "device": spec.get("device", "cuda:0"),
    }
    if spec.get("loss_masking"):
        cfg["training"]["loss_masking"] = spec["loss_masking"]
    if spec.get("trainable_vision_last_n"):
        cfg["model"]["trainable_vision_last_n"] = int(spec["trainable_vision_last_n"])
    if spec.get("loss_weighting"):
        cfg["training"]["loss_weighting"] = spec["loss_weighting"]
    if spec.get("image_shuffle_margin"):
        cfg["training"]["image_shuffle_margin"] = spec["image_shuffle_margin"]
    if spec.get("answer_margin"):
        cfg["training"]["answer_margin"] = spec["answer_margin"]
    if spec.get("in_batch_negative"):
        cfg["training"]["in_batch_negative"] = spec["in_batch_negative"]
    if spec.get("curriculum_schedule"):
        cfg["training"]["curriculum_schedule"] = spec["curriculum_schedule"]
    return cfg


def margin(weight: float = 0.1, value: float = 0.2) -> dict[str, Any]:
    return {"enabled": True, "weight": weight, "margin": value}


def specs() -> list[dict[str, Any]]:
    d0_train = "./outputs/instruction_data/glm_validated/d0_train_validated.jsonl"
    d0_val = "./outputs/instruction_data/glm_validated/d0_val_validated.jsonl"
    d6_train = "./outputs/instruction_data/glm_validated/d6_hard_cf_3k.jsonl"
    d6_val = "./outputs/instruction_data/glm_validated/d6_hard_cf_val200.jsonl"
    d7_train = "./outputs/instruction_data/glm_validated/d7_hard_shuffle_3k.jsonl"
    d7_val = "./outputs/instruction_data/glm_validated/d7_hard_shuffle_val200.jsonl"
    ns = "./outputs/instruction_data/next_stage"
    return [
        {"id": "p2_value_only", "run_id": "P2-value-only", "route": "json_mask", "train": d0_train, "val": d0_val, "max_steps": 3000, "data_root": "H:/Xiyao_Wang/000_Public Dataset", "loss_masking": {"mode": "json_value_only", "values": ["present", "absent", "uncertain", "null", "true", "false"]}},
        {"id": "p2_no_punct", "run_id": "P2-no-punct", "route": "json_mask", "train": d0_train, "val": d0_val, "max_steps": 3000, "data_root": "H:/Xiyao_Wang/000_Public Dataset", "loss_masking": {"mode": "json_no_punct"}},
        {"id": "p2_state_only_compact", "run_id": "P2-state-only-compact", "route": "json_mask", "train": f"{ns}/p2_state_only_compact_train.jsonl", "val": f"{ns}/p2_state_only_compact_val.jsonl", "max_steps": 3000, "data_root": "H:/Xiyao_Wang/000_Public Dataset"},
        {"id": "p2_field_query", "run_id": "P2-field-query", "route": "json_mask", "train": f"{ns}/p2_field_query_train.jsonl", "val": f"{ns}/p2_field_query_val.jsonl", "max_steps": 3000, "data_root": "H:/Xiyao_Wang/000_Public Dataset"},
        {"id": "balanced_mix_qa8", "run_id": "BalancedMix-QA8", "route": "rich_qa_mixture", "train": f"{ns}/balanced_mix_qa8_train.jsonl", "val": f"{ns}/balanced_mix_qa8_val.jsonl", "image_shuffle_margin": margin(), "answer_margin": margin()},
        {"id": "cf_heavy_qa8", "run_id": "CF-heavy-QA8", "route": "rich_qa_mixture", "train": f"{ns}/cf_heavy_qa8_train.jsonl", "val": f"{ns}/cf_heavy_qa8_val.jsonl", "image_shuffle_margin": margin(), "answer_margin": margin()},
        {"id": "shuf_heavy_qa8", "run_id": "SHUF-heavy-QA8", "route": "rich_qa_mixture", "train": f"{ns}/shuf_heavy_qa8_train.jsonl", "val": f"{ns}/shuf_heavy_qa8_val.jsonl", "image_shuffle_margin": margin(), "answer_margin": margin()},
        {"id": "clinical_rich_qa8", "run_id": "Clinical-rich-QA8", "route": "rich_qa_mixture", "train": f"{ns}/clinical_rich_qa8_train.jsonl", "val": f"{ns}/clinical_rich_qa8_val.jsonl", "image_shuffle_margin": margin(), "answer_margin": margin()},
        {"id": "storymix_qa5", "run_id": "StoryMix-QA5", "route": "rich_qa_mixture", "train": f"{ns}/storymix_qa5_train.jsonl", "val": f"{ns}/storymix_qa5_val.jsonl", "image_shuffle_margin": margin(), "answer_margin": margin()},
        {"id": "storymix_qa8", "run_id": "StoryMix-QA8", "route": "rich_qa_mixture", "train": f"{ns}/storymix_qa8_train.jsonl", "val": f"{ns}/storymix_qa8_val.jsonl", "image_shuffle_margin": margin(), "answer_margin": margin()},
        {"id": "storymix_qa10", "run_id": "StoryMix-QA10", "route": "rich_qa_mixture", "train": f"{ns}/storymix_qa10_train.jsonl", "val": f"{ns}/storymix_qa10_val.jsonl", "image_shuffle_margin": margin(), "answer_margin": margin()},
        {"id": "storymix_qa12", "run_id": "StoryMix-QA12", "route": "rich_qa_mixture", "train": f"{ns}/storymix_qa12_train.jsonl", "val": f"{ns}/storymix_qa12_val.jsonl", "image_shuffle_margin": margin(), "answer_margin": margin()},
        {"id": "sameq_shuf_3k", "run_id": "SAMEQ-SHUF-3k", "route": "shuf_plus_sameq", "train": f"{ns}/sameq_shuf_3k_train.jsonl", "val": f"{ns}/sameq_shuf_val.jsonl", "image_shuffle_margin": margin(0.15), "answer_margin": margin(0.15)},
        {"id": "shuf_tw_role", "run_id": "SHUF-TW-role", "route": "token_weighting", "train": d7_train, "val": d7_val, "image_shuffle_margin": margin(), "loss_weighting": LOSS_WEIGHTING["tw-role"]},
        {"id": "shuf_tw_visual", "run_id": "SHUF-TW-visual", "route": "token_weighting", "train": d7_train, "val": d7_val, "image_shuffle_margin": margin(), "loss_weighting": LOSS_WEIGHTING["tw-visual"]},
        {"id": "shuf_tw_clinical", "run_id": "SHUF-TW-clinical", "route": "token_weighting", "train": d7_train, "val": d7_val, "image_shuffle_margin": margin(), "loss_weighting": LOSS_WEIGHTING["tw-clinical"]},
        {"id": "train_conn", "run_id": "TRAIN-CONN", "route": "training_policy_connector_only", "train": d7_train, "val": d7_val, "max_steps": 5000, "trainable_groups": ["visual_connector"], "image_shuffle_margin": margin()},
        {"id": "train_last4", "run_id": "TRAIN-LAST4", "route": "training_policy_last4_vision", "train": d7_train, "val": d7_val, "max_steps": 5000, "trainable_groups": ["vision_tower", "visual_connector"], "trainable_vision_last_n": 4, "image_shuffle_margin": margin()},
        {"id": "train_fullvision", "run_id": "TRAIN-FULLVISION", "route": "training_policy_full_vision", "train": d7_train, "val": d7_val, "max_steps": 5000, "trainable_groups": ["vision_tower", "visual_connector"], "image_shuffle_margin": margin()},
        {"id": "mix_story_qa8", "run_id": "Mix-Story-QA8", "route": "workflow_single_stage", "train": f"{ns}/storymix_qa8_train.jsonl", "val": f"{ns}/storymix_qa8_val.jsonl", "max_steps": 8000, "image_shuffle_margin": margin(), "answer_margin": margin()},
        {"id": "cur_p3_shuf", "run_id": "CUR-P3-SHUF", "route": "workflow_curriculum", "train": f"{ns}/cur_p3_cf_shuf_train.jsonl", "val": f"{ns}/storymix_qa8_val.jsonl", "max_steps": 8000, "image_shuffle_margin": margin(), "curriculum_schedule": "./outputs/next_stage_manifests/cur-p3-cf-shuf_materialized_schedule.json"},
        {"id": "cur_cf_shuf", "run_id": "CUR-CF-SHUF", "route": "workflow_curriculum", "train": f"{ns}/cur_p3_cf_shuf_train.jsonl", "val": d7_val, "max_steps": 8000, "image_shuffle_margin": margin(), "answer_margin": margin(), "curriculum_schedule": "./outputs/next_stage_manifests/cur-p3-cf-shuf_materialized_schedule.json"},
        {"id": "cur_p3_cf_shuf", "run_id": "CUR-P3-CF-SHUF", "route": "workflow_curriculum", "train": f"{ns}/cur_p3_cf_shuf_train.jsonl", "val": f"{ns}/storymix_qa8_val.jsonl", "max_steps": 8000, "image_shuffle_margin": margin(), "answer_margin": margin(), "curriculum_schedule": "./outputs/next_stage_manifests/cur-p3-cf-shuf_materialized_schedule.json"},
        {"id": "prog_mix", "run_id": "PROG-Mix", "route": "workflow_progressive", "train": f"{ns}/prog_mix_train.jsonl", "val": f"{ns}/storymix_qa8_val.jsonl", "max_steps": 8000, "image_shuffle_margin": margin(), "answer_margin": margin(), "curriculum_schedule": "./outputs/next_stage_manifests/prog-mix_materialized_schedule.json"},
        {"id": "prog_mix_tw", "run_id": "PROG-Mix-TW", "route": "workflow_progressive_tw", "train": f"{ns}/prog_mix_train.jsonl", "val": f"{ns}/storymix_qa8_val.jsonl", "max_steps": 8000, "image_shuffle_margin": margin(), "answer_margin": margin(), "loss_weighting": LOSS_WEIGHTING["tw-visual"], "curriculum_schedule": "./outputs/next_stage_manifests/prog-mix_materialized_schedule.json"},
        {"id": "prog_mix_sameq", "run_id": "PROG-Mix-SAMEQ", "route": "workflow_progressive_sameq", "train": f"{ns}/prog_mix_sameq_train.jsonl", "val": f"{ns}/sameq_shuf_val.jsonl", "max_steps": 8000, "image_shuffle_margin": margin(0.15), "answer_margin": margin(), "curriculum_schedule": "./outputs/next_stage_manifests/prog-mix-sameq_materialized_schedule.json"},
        {"id": "prog_mix_dualmargin", "run_id": "PROG-Mix-DualMargin", "route": "workflow_progressive_dual_margin", "train": f"{ns}/prog_mix_train.jsonl", "val": f"{ns}/storymix_qa8_val.jsonl", "max_steps": 8000, "image_shuffle_margin": margin(0.15), "answer_margin": margin(0.15), "curriculum_schedule": "./outputs/next_stage_manifests/prog-mix_materialized_schedule.json"},
        {"id": "shuf_k2", "run_id": "SHUF-K2", "route": "shuf_plus_multi_negative", "train": f"{ns}/shuf_k2_train.jsonl", "val": f"{ns}/shuf_k2_val.jsonl", "image_shuffle_margin": margin(0.15)},
        {"id": "shuf_k4", "run_id": "SHUF-K4", "route": "shuf_plus_multi_negative", "train": f"{ns}/shuf_k4_train.jsonl", "val": f"{ns}/shuf_k4_val.jsonl", "image_shuffle_margin": margin(0.15)},
        {"id": "inbatch_shuf", "run_id": "InBatch-SHUF", "route": "shuf_plus_in_batch_negative", "train": d7_train, "val": d7_val, "image_shuffle_margin": margin(0.15), "in_batch_negative": {"enabled": True, "offset": 1}, "batch_size": 2, "eval_batch_size": 1, "max_length": 640},
        {"id": "mined_shuf", "run_id": "Mined-SHUF", "route": "shuf_plus_embedding_mined", "train": f"{ns}/mined_shuf_train.jsonl", "val": f"{ns}/mined_shuf_val.jsonl", "image_shuffle_margin": margin(0.15)},
        {"id": "selfhard_shuf", "run_id": "SelfHard-SHUF", "route": "shuf_plus_confidence_mined", "train": f"{ns}/selfhard_shuf_train.jsonl", "val": d7_val, "image_shuffle_margin": margin(0.15)},
        {"id": "dual_cf_shuf", "run_id": "DUAL-CF-SHUF", "route": "shuf_plus_dual_margin", "train": d7_train, "val": d7_val, "image_shuffle_margin": margin(0.15), "answer_margin": margin(0.15)},
        {"id": "progressive_hardneg", "run_id": "Progressive-HardNeg", "route": "shuf_plus_progressive_hardneg", "train": f"{ns}/progressive_hardneg_train.jsonl", "val": d7_val, "max_steps": 8000, "image_shuffle_margin": margin(0.15), "answer_margin": margin(0.15), "curriculum_schedule": "./outputs/next_stage_manifests/progressive-hardneg_schedule.json"},
        {"id": "shuf_k4_tw_visual", "run_id": "SHUF-K4-TW-visual", "route": "shuf_plus_multi_negative_tw", "train": f"{ns}/shuf_k4_train.jsonl", "val": f"{ns}/shuf_k4_val.jsonl", "image_shuffle_margin": margin(0.15), "loss_weighting": LOSS_WEIGHTING["tw-visual"]},
        {"id": "shuf_10k_8k", "run_id": "SHUF-10k-8k", "route": "scale_shuf_ums", "train": f"{ns}/shuf_10k_train.jsonl", "val": f"{ns}/shuf_10k_val.jsonl", "max_steps": 8000, "image_shuffle_margin": margin()},
        {"id": "storymix_10k_8k", "run_id": "StoryMix-10k-8k", "route": "scale_storymix_ums", "train": f"{ns}/storymix_10k_train.jsonl", "val": f"{ns}/storymix_10k_val.jsonl", "max_steps": 8000, "image_shuffle_margin": margin(), "answer_margin": margin()},
        {"id": "prog_mix_10k_8k", "run_id": "PROG-Mix-10k-8k", "route": "scale_progressive_ums", "train": f"{ns}/prog_mix_10k_train.jsonl", "val": f"{ns}/storymix_10k_val.jsonl", "max_steps": 8000, "image_shuffle_margin": margin(), "answer_margin": margin(), "curriculum_schedule": "./outputs/next_stage_manifests/prog-mix-10k_materialized_schedule.json"},
        {"id": "prog_mix_tw_10k", "run_id": "PROG-Mix-TW-10k", "route": "scale_progressive_tw_ums", "train": f"{ns}/prog_mix_10k_train.jsonl", "val": f"{ns}/storymix_10k_val.jsonl", "max_steps": 8000, "image_shuffle_margin": margin(), "answer_margin": margin(), "loss_weighting": LOSS_WEIGHTING["tw-visual"], "curriculum_schedule": "./outputs/next_stage_manifests/prog-mix-10k_materialized_schedule.json"},
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=ROOT / "configs/qwen3vl_instruction/next_stage", type=Path)
    parser.add_argument("--manifest", default=ROOT / "outputs/next_stage_manifests/config_manifest.json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for spec in specs():
        path = args.output_dir / f"{spec['id']}.yaml"
        config = base_config(spec)
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        manifest_rows.append({"run_id": spec["run_id"], "id": spec["id"], "config": str(path), "train": spec["train"], "val": spec.get("val", "")})
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({"configs": manifest_rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "configs": len(manifest_rows), "manifest": str(args.manifest)}, indent=2))


if __name__ == "__main__":
    main()
