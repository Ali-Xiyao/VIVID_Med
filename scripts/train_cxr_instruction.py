"""Train VIVID-Med on clinical instruction records."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import CXRInstructionDataset, instruction_collate_fn
from models import VIVIDModel
from training import VIVIDTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader | None]:
    data_cfg = config["data"]
    prompt_template = data_cfg.get("prompt_template", "Question: {question}\nAnswer: ")

    train_dataset = CXRInstructionDataset(
        data_root=data_cfg["data_root"],
        instruction_jsonl_path=data_cfg["train_instruction_path"],
        image_size=int(data_cfg.get("image_size", 224)),
        is_train=True,
        max_samples=data_cfg.get("max_train_samples"),
        prompt_template=prompt_template,
    )

    val_loader = None
    if data_cfg.get("val_instruction_path"):
        val_dataset = CXRInstructionDataset(
            data_root=data_cfg["data_root"],
            instruction_jsonl_path=data_cfg["val_instruction_path"],
            image_size=int(data_cfg.get("image_size", 224)),
            is_train=False,
            max_samples=data_cfg.get("max_val_samples"),
            prompt_template=prompt_template,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(config["training"].get("eval_batch_size", config["training"]["batch_size"])),
            shuffle=False,
            num_workers=int(data_cfg.get("num_workers", 0)),
            collate_fn=instruction_collate_fn,
            pin_memory=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=instruction_collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader, val_loader


def create_model(config: dict[str, Any], device: str) -> VIVIDModel:
    model_cfg = config["model"]
    spd_cfg = model_cfg.get("spd", {})
    model = VIVIDModel(
        vit_model_name=model_cfg.get("vit_model_name", "vit_base_patch16_224"),
        vit_pretrained=bool(model_cfg.get("vit_pretrained", True)),
        vit_output_type=model_cfg.get("vit_output_type", "all"),
        num_prefix_tokens=int(model_cfg.get("num_prefix_tokens", 4)),
        projector_dropout=float(model_cfg.get("projector_dropout", 0.1)),
        projector_mlp_hidden_dim=model_cfg.get("projector_mlp_hidden_dim"),
        llm_model_name=model_cfg["llm_model_name"],
        use_flash_attention=bool(model_cfg.get("use_flash_attention", False)),
        max_text_length=int(model_cfg.get("max_text_length", 256)),
        load_llm=True,
        llm_random_init=bool(model_cfg.get("llm_random_init", False)),
        spd_enabled=bool(spd_cfg.get("enabled", False)),
        spd_num_groups=int(spd_cfg.get("num_groups", 3)),
        spd_tokens_per_group=int(spd_cfg.get("tokens_per_group", 2)),
    )
    model = model.to(device)
    print(f"Model created on {device}")
    print(f"  Trainable parameters: {model.get_num_trainable_parameters():,}")
    print(f"  Frozen parameters: {model.get_num_frozen_parameters():,}")
    return model


def apply_debug_overrides(config: dict[str, Any]) -> None:
    data_cfg = config["data"]
    train_cfg = config["training"]
    data_cfg["max_train_samples"] = min(int(data_cfg.get("max_train_samples") or 16), 16)
    data_cfg["max_val_samples"] = min(int(data_cfg.get("max_val_samples") or 4), 4)
    data_cfg["num_workers"] = 0
    train_cfg["batch_size"] = 1
    train_cfg["eval_batch_size"] = 1
    train_cfg["gradient_accumulation_steps"] = 1
    train_cfg["max_steps"] = min(int(train_cfg.get("max_steps", 2)), 2)
    train_cfg["warmup_steps"] = 1
    train_cfg["log_interval"] = 1
    train_cfg["eval_interval"] = 1
    train_cfg["save_interval"] = 2
    train_cfg["bf16"] = False
    train_cfg["fp16"] = False
    config["model"]["use_flash_attention"] = False


def save_run_config(config: dict[str, Any], config_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.debug:
        print("Debug mode enabled")
        apply_debug_overrides(config)
    if args.seed is not None:
        config["seed"] = args.seed
        base = str(config["training"]["output_dir"]).rstrip("/")
        config["training"]["output_dir"] = f"{base}_seed{args.seed}"

    output_dir = Path(config["training"]["output_dir"])
    if (output_dir / "metrics_final.json").exists():
        raise SystemExit(f"{output_dir / 'metrics_final.json'} already exists; remove it manually to rerun.")

    set_seed(int(config.get("seed", 42)))
    requested_device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    else:
        device = str(requested_device)
    print(f"Using device: {device}")

    save_run_config(config, args.config, output_dir)
    train_loader, val_loader = create_dataloaders(config)
    print(f"Train batches: {len(train_loader)}")
    if val_loader:
        print(f"Val batches: {len(val_loader)}")

    model = create_model(config, device)
    train_cfg = config["training"]
    trainer = VIVIDTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        warmup_steps=int(train_cfg["warmup_steps"]),
        max_steps=int(train_cfg["max_steps"]),
        vit_learning_rate=train_cfg.get("vit_learning_rate"),
        projector_learning_rate=train_cfg.get("projector_learning_rate"),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        max_grad_norm=float(train_cfg["max_grad_norm"]),
        fp16=bool(train_cfg.get("fp16", False)),
        bf16=bool(train_cfg.get("bf16", True)),
        log_interval=int(train_cfg["log_interval"]),
        eval_interval=int(train_cfg["eval_interval"]),
        save_interval=int(train_cfg["save_interval"]),
        output_dir=str(output_dir),
        token_weighting=train_cfg.get("token_weighting"),
        prompt_template=config["data"].get("fallback_prompt_template", "Question: {question}\nAnswer: "),
        mask_ratio=float(config.get("model", {}).get("mask_ratio", 0.0)),
    )
    if args.resume:
        trainer.load_checkpoint(args.resume)

    started = time.time()
    trainer.train()
    elapsed = time.time() - started

    metrics = {
        "global_step": trainer.global_step,
        "best_val_loss": trainer.best_val_loss,
        "elapsed_seconds": elapsed,
        "train_records": len(train_loader.dataset),
        "val_records": len(val_loader.dataset) if val_loader is not None else 0,
    }
    (output_dir / "metrics_final.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "runtime_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
