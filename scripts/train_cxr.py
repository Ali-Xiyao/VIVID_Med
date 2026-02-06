"""
VIVID-Med CXR 训练脚本

用法:
    python train_cxr.py --config ../configs/cxr_chexpert.yaml

或使用默认配置:
    python train_cxr.py
"""

import os
import sys
import argparse
import random
from pathlib import Path

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import CheXpertUMSDataset, get_train_transforms, get_val_transforms
from data.chexpert_dataset import collate_fn
from models import VIVIDModel
from training import VIVIDTrainer


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: dict):
    """创建数据加载器"""
    data_cfg = config["data"]
    image_size = data_cfg["image_size"]
    data_root = data_cfg["data_root"]
    train_ums_path = data_cfg["train_ums_path"]
    val_ums_path = data_cfg.get("val_ums_path")
    use_common_labels_only = data_cfg.get("use_common_labels_only", False)
    max_train_samples = data_cfg.get("max_train_samples")
    max_val_samples = data_cfg.get("max_val_samples", 1000)
    json_include_all_labels = data_cfg.get("json_include_all_labels", False)
    json_missing_state = data_cfg.get("json_missing_state")
    json_null_state = data_cfg.get("json_null_state")

    # 训练数据集
    train_dataset = CheXpertUMSDataset(
        data_root=data_root,
        ums_jsonl_path=train_ums_path,
        transform=get_train_transforms(image_size),
        is_train=True,
        use_common_labels_only=use_common_labels_only,
        max_samples=max_train_samples,
        json_include_all_labels=json_include_all_labels,
        json_missing_state=json_missing_state,
        json_null_state=json_null_state,
    )

    # 验证数据集
    val_dataset = None
    if val_ums_path:
        val_dataset = CheXpertUMSDataset(
            data_root=data_root,
            ums_jsonl_path=val_ums_path,
            transform=get_val_transforms(image_size),
            is_train=False,
            use_common_labels_only=use_common_labels_only,
            max_samples=max_val_samples,
            json_include_all_labels=json_include_all_labels,
            json_missing_state=json_missing_state,
            json_null_state=json_null_state,
        )
    else:
        if max_val_samples and max_val_samples < len(train_dataset):
            generator = torch.Generator().manual_seed(config.get("seed", 42))
            indices = torch.randperm(len(train_dataset), generator=generator).tolist()
            val_indices = indices[:max_val_samples]
            train_indices = indices[max_val_samples:]

            train_dataset = Subset(train_dataset, train_indices)

            val_pool_max_samples = max_train_samples
            val_base_dataset = CheXpertUMSDataset(
                data_root=data_root,
                ums_jsonl_path=train_ums_path,
                transform=get_val_transforms(image_size),
                is_train=False,
                use_common_labels_only=use_common_labels_only,
                max_samples=val_pool_max_samples,
                json_include_all_labels=json_include_all_labels,
                json_missing_state=json_missing_state,
                json_null_state=json_null_state,
            )
            val_dataset = Subset(val_base_dataset, val_indices)

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=data_cfg.get("num_workers", 4),
            collate_fn=collate_fn,
            pin_memory=True,
        )

    return train_loader, val_loader


def create_model(config: dict, device: str):
    """创建模型"""
    model_cfg = config["model"]

    print("Creating VIVID model...")
    print(f"  ViT: {model_cfg['vit_model_name']}")
    print(f"  LLM: {model_cfg['llm_model_name']}")
    print(f"  Prefix tokens: {model_cfg['num_prefix_tokens']}")

    model = VIVIDModel(
        vit_model_name=model_cfg["vit_model_name"],
        vit_pretrained=model_cfg["vit_pretrained"],
        vit_output_type=model_cfg.get("vit_output_type", "cls"),
        num_prefix_tokens=model_cfg["num_prefix_tokens"],
        projector_dropout=model_cfg.get("projector_dropout", 0.1),
        llm_model_name=model_cfg["llm_model_name"],
        use_flash_attention=model_cfg.get("use_flash_attention", True),
        max_text_length=model_cfg.get("max_text_length", 512),
        load_llm=True,
    )

    model = model.to(device)

    print(f"Model created on {device}")
    print(f"  Trainable parameters: {model.get_num_trainable_parameters():,}")
    print(f"  Frozen parameters: {model.get_num_frozen_parameters():,}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train VIVID-Med CXR model")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent.parent / "configs" / "cxr_chexpert.yaml"),
        help="Path to config file",
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--debug", action="store_true", help="Debug mode (fewer samples)")
    args = parser.parse_args()

    # 加载配置
    print(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Debug 模式
    if args.debug:
        print("Debug mode enabled")
        config["data"]["max_train_samples"] = 20
        config["data"]["max_val_samples"] = 4
        config["data"]["num_workers"] = 0
        config["training"]["batch_size"] = 2
        config["training"]["gradient_accumulation_steps"] = 1
        config["training"]["max_steps"] = 5
        config["training"]["log_interval"] = 1
        config["training"]["eval_interval"] = 2
        config["training"]["save_interval"] = 5
        # 使用小模型以加速调试（避免下载大模型）
        config["model"]["llm_model_name"] = config["model"].get(
            "debug_llm_model_name", "sshleifer/tiny-gpt2"
        )
        config["model"]["use_flash_attention"] = False
        # 调试时禁用混合精度，避免 CPU 环境报错
        config["training"]["bf16"] = False
        config["training"]["fp16"] = False
        if "token_weighting" in config["training"]:
            config["training"]["token_weighting"]["enabled"] = False

    # 设置随机种子
    seed = config.get("seed", 42)
    set_seed(seed)
    print(f"Random seed: {seed}")

    # 设备
    requested_device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(requested_device, str) and requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = "cpu"
        else:
            device = requested_device
    else:
        device = requested_device
    print(f"Using device: {device}")

    # 创建数据加载器
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"  Train batches: {len(train_loader)}")
    if val_loader:
        print(f"  Val batches: {len(val_loader)}")

    # 创建模型
    print("\nCreating model...")
    model = create_model(config, device)

    # 创建训练器
    print("\nCreating trainer...")
    training_cfg = config["training"]
    prompt_cfg = config.get("prompt", {})
    wandb_cfg = config.get("wandb", {})

    trainer = VIVIDTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        learning_rate=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
        warmup_steps=training_cfg["warmup_steps"],
        max_steps=training_cfg["max_steps"],
        lambda_rank=training_cfg.get("lambda_rank", 0.0),
        lambda_vdep=training_cfg.get("lambda_vdep", 0.0),
        lambda_ans=training_cfg.get("lambda_ans", 0.0),
        token_weighting=training_cfg.get("token_weighting"),
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        max_grad_norm=training_cfg["max_grad_norm"],
        fp16=training_cfg.get("fp16", False),
        bf16=training_cfg.get("bf16", True),
        log_interval=training_cfg["log_interval"],
        eval_interval=training_cfg["eval_interval"],
        save_interval=training_cfg["save_interval"],
        output_dir=training_cfg["output_dir"],
        use_wandb=wandb_cfg.get("enabled", False),
        wandb_project=wandb_cfg.get("project", "vivid-med"),
        wandb_run_name=wandb_cfg.get("run_name"),
        prompt_template=prompt_cfg.get("template", "Generate a structured medical report:\n"),
    )

    # 恢复训练
    if args.resume:
        print(f"\nResuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # 开始训练
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50 + "\n")

    trainer.train()

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
