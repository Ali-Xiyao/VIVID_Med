"""
VIVID Trainer
训练流程管理
"""

import os
import json
import time
import contextlib
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .losses import VIVIDLoss


class VIVIDTrainer:
    """
    VIVID 训练器

    训练流程：
    1. 只跑 L_tok（确保结构化生成可学）
    2. 加 L_rank（学会拒绝"错但像"）
    3. 加 L_vdep（逼迫看图）
    4. 接入 sample policy（role 分流）
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        # 优化器配置
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_steps: int = 10000,
        # 损失配置
        lambda_rank: float = 0.0,  # v1.0 先设为 0
        lambda_vdep: float = 0.0,  # v1.0 先设为 0
        lambda_ans: float = 0.0,   # v1.0 先设为 0
        # 训练配置
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        fp16: bool = False,
        bf16: bool = True,
        # 日志配置
        log_interval: int = 10,
        eval_interval: int = 500,
        save_interval: int = 1000,
        output_dir: str = "./outputs",
        # wandb 配置
        use_wandb: bool = False,
        wandb_project: str = "vivid-med",
        wandb_run_name: Optional[str] = None,
        # Prompt 配置
        prompt_template: str = "Generate a structured medical report in JSON format for this chest X-ray image:\n",
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.fp16 = fp16
        self.bf16 = bf16

        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.prompt_template = prompt_template

        # 设备
        self.device = next(model.parameters()).device

        # 损失函数
        self.criterion = VIVIDLoss(
            lambda_rank=lambda_rank,
            lambda_vdep=lambda_vdep,
            lambda_ans=lambda_ans,
        )

        # 优化器（只优化可训练参数）
        trainable_params = model.get_trainable_parameters()
        self.optimizer = AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # 学习率调度器
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max_steps - warmup_steps,
            eta_min=learning_rate * 0.1,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        # 混合精度
        self.scaler = None
        if fp16 and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
        elif fp16:
            print("Warning: fp16 requested but CUDA not available; disabling GradScaler.")

        # wandb
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "warmup_steps": warmup_steps,
                    "max_steps": max_steps,
                    "lambda_rank": lambda_rank,
                    "lambda_vdep": lambda_vdep,
                    "lambda_ans": lambda_ans,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "fp16": fp16,
                    "bf16": bf16,
                },
            )

        # 训练状态
        self.global_step = 0
        self.best_val_loss = float("inf")

    def train(self):
        """主训练循环"""
        print(f"Starting training...")
        print(f"  Trainable parameters: {self.model.get_num_trainable_parameters():,}")
        print(f"  Frozen parameters: {self.model.get_num_frozen_parameters():,}")
        print(f"  Max steps: {self.max_steps}")
        print(f"  Output dir: {self.output_dir}")

        self.model.train()
        train_iter = iter(self.train_dataloader)

        progress_bar = tqdm(total=self.max_steps, desc="Training")

        accumulated_loss = 0.0
        num_accumulated = 0

        while self.global_step < self.max_steps:
            # 获取 batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)

            # 训练一步
            loss_dict = self._train_step(batch)

            accumulated_loss += loss_dict["total"].item()
            num_accumulated += 1

            # 梯度累积
            if num_accumulated >= self.gradient_accumulation_steps:
                # 梯度裁剪
                if self.max_grad_norm > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.get_trainable_parameters(),
                        self.max_grad_norm,
                    )

                # 优化器步骤
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1
                progress_bar.update(1)

                # 日志
                if self.global_step % self.log_interval == 0:
                    avg_loss = accumulated_loss / num_accumulated
                    lr = self.scheduler.get_last_lr()[0]

                    log_dict = {
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/step": self.global_step,
                    }

                    progress_bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

                    if self.use_wandb:
                        wandb.log(log_dict, step=self.global_step)

                accumulated_loss = 0.0
                num_accumulated = 0

                # 验证
                if self.val_dataloader is not None and self.global_step % self.eval_interval == 0:
                    val_loss = self._validate()
                    print(f"\nStep {self.global_step}: val_loss = {val_loss:.4f}")

                    if self.use_wandb:
                        wandb.log({"val/loss": val_loss}, step=self.global_step)

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint("best")

                    self.model.train()

                # 保存
                if self.global_step % self.save_interval == 0:
                    self._save_checkpoint(f"step_{self.global_step}")

        progress_bar.close()
        self._save_checkpoint("final")
        print("Training completed!")

    def _train_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """单步训练"""
        # 移动数据到设备
        images = batch["images"].to(self.device)
        target_jsons = batch["target_jsons"]

        # 混合精度上下文（仅 CUDA）
        use_amp = self.device.type == "cuda" and (self.bf16 or self.fp16)
        if use_amp:
            autocast_dtype = torch.bfloat16 if self.bf16 else torch.float16
            autocast_ctx = torch.cuda.amp.autocast(dtype=autocast_dtype)
        else:
            autocast_ctx = contextlib.nullcontext()

        with autocast_ctx:
            # 前向传播
            outputs = self.model(
                images=images,
                prompt_text=self.prompt_template,
                target_text=target_jsons,
            )

            # 计算损失
            loss_dict = {"total": outputs["loss"]}

        # 反向传播
        loss = loss_dict["total"] / self.gradient_accumulation_steps

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss_dict

    @torch.no_grad()
    def _validate(self) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_dataloader, desc="Validating", leave=False):
            images = batch["images"].to(self.device)
            target_jsons = batch["target_jsons"]

            outputs = self.model(
                images=images,
                prompt_text=self.prompt_template,
                target_text=target_jsons,
            )

            total_loss += outputs["loss"].item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _save_checkpoint(self, name: str):
        """保存检查点"""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        # 只保存可训练参数（ViT + Projector）
        state_dict = {
            "vit": self.model.vit.state_dict(),
            "projector": self.model.projector.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }

        if self.scaler is not None:
            state_dict["scaler"] = self.scaler.state_dict()

        save_path = checkpoint_dir / f"{name}.pt"
        torch.save(state_dict, save_path)
        print(f"Checkpoint saved: {save_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        state_dict = torch.load(checkpoint_path, map_location=self.device)

        self.model.vit.load_state_dict(state_dict["vit"])
        self.model.projector.load_state_dict(state_dict["projector"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.global_step = state_dict["global_step"]
        self.best_val_loss = state_dict.get("best_val_loss", float("inf"))

        if self.scaler is not None and "scaler" in state_dict:
            self.scaler.load_state_dict(state_dict["scaler"])

        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"  Resuming from step {self.global_step}")
