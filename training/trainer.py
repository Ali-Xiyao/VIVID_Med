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

from .losses import StructuredLoss


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
        token_weighting: Optional[Dict[str, Any]] = None,
        answerability_mask: Optional[Dict[str, Any]] = None,
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
        self.token_weighting = token_weighting or {}
        self.answerability_mask = self._build_answerability_mask_config(answerability_mask)
        self._answerability_span_cache: Dict[str, Dict[str, Any]] = {}
        self._num_visual_tokens = self._infer_num_visual_tokens()

        # 设备
        self.device = next(model.parameters()).device

        # 可选：state-aware token loss（V3 anti-collapse）
        self.token_loss = self._build_token_loss(self.token_weighting)
        self.base_token_loss = StructuredLoss(
            ignore_index=-100,
            label_smoothing=0.0,
            reduction="mean",
        )
        if self.answerability_mask["enabled"]:
            print("Using answerability-aware token mask:")
            print(f"  true_weight: {self.answerability_mask['true_weight']}")
            print(f"  false_weight: {self.answerability_mask['false_weight']}")
            print(f"  scope: {self.answerability_mask['scope']}")
            print(f"  keep_structural_tokens: {self.answerability_mask['keep_structural_tokens']}")

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

    def _infer_num_visual_tokens(self) -> int:
        projector = getattr(self.model, "projector", None)
        vit = getattr(self.model, "vit", None)
        if projector is not None and vit is not None and hasattr(vit, "get_num_tokens"):
            return int(projector.num_prefix_tokens + vit.get_num_tokens())
        return 0

    def _build_answerability_mask_config(self, mask_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        cfg = dict(mask_cfg or {})
        enabled = bool(cfg.get("enabled", False))
        true_weight = float(cfg.get("true_weight", 1.0))
        false_weight = float(cfg.get("false_weight", 0.0))
        scope = str(cfg.get("scope", "findings_only"))
        keep_structural_tokens = bool(cfg.get("keep_structural_tokens", True))

        if true_weight < 0.0:
            raise ValueError("answerability_mask.true_weight must be >= 0")
        if false_weight < 0.0:
            raise ValueError("answerability_mask.false_weight must be >= 0")
        if scope != "findings_only":
            raise ValueError("answerability_mask.scope currently only supports 'findings_only'")

        return {
            "enabled": enabled,
            "true_weight": true_weight,
            "false_weight": false_weight,
            "scope": scope,
            "keep_structural_tokens": keep_structural_tokens,
        }

    def _build_state_token_weights(
        self,
        state_weights: Dict[str, Any],
    ) -> (Dict[int, float], List[tuple[list[int], float]]):
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None:
            return {}, []

        token_id_weights: Dict[int, float] = {}
        token_sequence_weights: List[tuple[list[int], float]] = []

        for raw_state_name, state_weight in state_weights.items():
            state_name = "null" if raw_state_name is None else str(raw_state_name)
            try:
                weight = float(state_weight)
            except (TypeError, ValueError):
                continue

            variants = [
                state_name,
                f" {state_name}",
                f"\"{state_name}\"",
                f" \"{state_name}\"",
                f": {state_name}",
                f": \"{state_name}\"",
                f"state\": \"{state_name}\"",
            ]

            seen = set()
            for text in variants:
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                if not token_ids:
                    continue
                key = tuple(int(token_id) for token_id in token_ids)
                if key in seen:
                    continue
                seen.add(key)
                if len(key) == 1:
                    token_id = key[0]
                    prev = token_id_weights.get(token_id, 1.0)
                    token_id_weights[token_id] = max(prev, weight)
                else:
                    token_sequence_weights.append((list(key), weight))

        return token_id_weights, token_sequence_weights

    def _build_token_loss(self, token_weighting: Dict[str, Any]) -> Optional[StructuredLoss]:
        if not token_weighting or not token_weighting.get("enabled", False):
            return None

        state_weights = token_weighting.get("state_weights", {})
        token_id_weights, token_sequence_weights = self._build_state_token_weights(state_weights)
        if not token_id_weights and not token_sequence_weights:
            print("Warning: token weighting enabled but no state tokens were found in tokenizer.")
            return None

        default_weight = float(token_weighting.get("default_weight", 1.0))
        label_smoothing = float(token_weighting.get("label_smoothing", 0.0))
        print("Using state-aware token weighting:")
        print(f"  default weight: {default_weight}")
        print(f"  weighted token ids: {len(token_id_weights)}")
        print(f"  weighted token sequences: {len(token_sequence_weights)}")

        return StructuredLoss(
            ignore_index=-100,
            label_smoothing=label_smoothing,
            reduction="mean",
            token_id_weights=token_id_weights,
            token_sequence_weights=token_sequence_weights,
            default_token_weight=default_weight,
        )

    def _find_matching_brace(self, text: str, open_brace_index: int) -> int:
        if open_brace_index < 0 or open_brace_index >= len(text) or text[open_brace_index] != "{":
            return -1

        depth = 0
        in_string = False
        escaped = False

        for index in range(open_brace_index, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == "\"":
                    in_string = False
                continue

            if char == "\"":
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return index

        return -1

    def _find_finding_span(
        self,
        target_json: str,
        finding_name: str,
        keep_structural_tokens: bool,
    ) -> Optional[tuple[int, int]]:
        name_token = f"\"{finding_name}\""
        name_pos = target_json.find(name_token)
        if name_pos < 0:
            return None

        colon_pos = target_json.find(":", name_pos + len(name_token))
        if colon_pos < 0:
            return None

        object_start = target_json.find("{", colon_pos)
        if object_start < 0:
            return None

        object_end = self._find_matching_brace(target_json, object_start)
        if object_end < 0:
            return None

        if not keep_structural_tokens:
            return name_pos, object_end + 1

        state_key_pos = target_json.find("\"state\"", object_start, object_end + 1)
        if state_key_pos < 0:
            return None

        state_colon_pos = target_json.find(":", state_key_pos, object_end + 1)
        if state_colon_pos < 0:
            return None

        value_start = state_colon_pos + 1
        while value_start < object_end and target_json[value_start] in {" ", "\t", "\n", "\r"}:
            value_start += 1

        if value_start > object_end:
            return None

        if target_json[value_start] == "\"":
            value_end = target_json.find("\"", value_start + 1, object_end + 1)
            if value_end < 0:
                return None
            value_end += 1
        else:
            value_end = value_start
            while value_end <= object_end and target_json[value_end] not in {",", "}"}:
                value_end += 1

        return value_start, value_end

    def _get_or_build_answerability_cache_entry(self, target_json: str) -> Optional[Dict[str, Any]]:
        cache_key = f"{self.answerability_mask['keep_structural_tokens']}|{target_json}"
        if cache_key in self._answerability_span_cache:
            return self._answerability_span_cache[cache_key]

        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None:
            return None

        prompt_text = self.prompt_template
        full_text = f"{prompt_text}{target_json}"

        try:
            tokenized = tokenizer(
                full_text,
                padding=False,
                truncation=True,
                max_length=getattr(self.model, "max_text_length", 512),
                return_offsets_mapping=True,
                add_special_tokens=True,
            )
        except Exception:
            return None

        input_ids = tokenized.get("input_ids")
        offsets = tokenized.get("offset_mapping")
        if input_ids is None or offsets is None:
            return None

        try:
            findings = json.loads(target_json).get("findings", {})
            finding_names = list(findings.keys())
        except Exception:
            finding_names = []

        token_indices_by_finding: Dict[str, List[int]] = {}
        prompt_char_len = len(prompt_text)
        keep_structural_tokens = self.answerability_mask["keep_structural_tokens"]

        for finding_name in finding_names:
            span = self._find_finding_span(
                target_json=target_json,
                finding_name=finding_name,
                keep_structural_tokens=keep_structural_tokens,
            )
            if span is None:
                continue

            char_start, char_end = span
            full_char_start = prompt_char_len + char_start
            full_char_end = prompt_char_len + char_end

            matched_token_indices: List[int] = []
            for token_index, (token_start, token_end) in enumerate(offsets):
                if token_end <= token_start:
                    continue
                if token_end <= full_char_start or token_start >= full_char_end:
                    continue
                matched_token_indices.append(token_index)

            if matched_token_indices:
                token_indices_by_finding[finding_name] = matched_token_indices

        cache_entry = {
            "seq_len": len(input_ids),
            "finding_names": finding_names,
            "token_indices_by_finding": token_indices_by_finding,
        }
        self._answerability_span_cache[cache_key] = cache_entry
        return cache_entry

    def _build_answerability_token_weights(
        self,
        target_jsons: List[str],
        answerable: torch.Tensor,
        labels: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not self.answerability_mask["enabled"]:
            return None
        if labels is None or labels.dim() != 2:
            return None

        batch_size = labels.shape[0]
        if not isinstance(target_jsons, list) or len(target_jsons) != batch_size:
            return None

        if self._num_visual_tokens <= 0:
            return None

        text_seq_len = labels.shape[1] - self._num_visual_tokens
        if text_seq_len <= 0:
            return None

        true_weight = self.answerability_mask["true_weight"]
        false_weight = self.answerability_mask["false_weight"]
        token_weights = torch.full(
            labels.shape,
            fill_value=true_weight,
            dtype=torch.float32,
            device=labels.device,
        )

        for sample_index, target_json in enumerate(target_jsons):
            cache_entry = self._get_or_build_answerability_cache_entry(target_json)
            if cache_entry is None:
                continue

            sample_seq_len = int(cache_entry["seq_len"])
            pad_len = text_seq_len - sample_seq_len
            if pad_len < 0:
                continue

            answerable_row = answerable[sample_index]
            if answerable_row.device.type != "cpu":
                answerable_row = answerable_row.cpu()
            answerable_row = answerable_row.tolist()

            finding_names = cache_entry["finding_names"]
            token_indices_by_finding = cache_entry["token_indices_by_finding"]

            for finding_index, finding_name in enumerate(finding_names):
                if finding_index >= len(answerable_row):
                    break
                if bool(answerable_row[finding_index]):
                    continue

                token_indices = token_indices_by_finding.get(finding_name, [])
                for token_index in token_indices:
                    label_index = self._num_visual_tokens + pad_len + int(token_index)
                    if 0 <= label_index < labels.shape[1] and labels[sample_index, label_index] != -100:
                        token_weights[sample_index, label_index] = false_weight

        return token_weights

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
        answerable = batch.get("answerable")

        # 混合精度上下文（仅 CUDA）
        use_amp = self.device.type == "cuda" and (self.bf16 or self.fp16)
        if use_amp:
            autocast_dtype = torch.bfloat16 if self.bf16 else torch.float16
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=autocast_dtype)
            else:
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
            answerability_token_weights = None
            if outputs.get("labels") is not None and answerable is not None:
                answerability_token_weights = self._build_answerability_token_weights(
                    target_jsons=target_jsons,
                    answerable=answerable,
                    labels=outputs["labels"],
                )

            if outputs.get("labels") is not None and (
                self.token_loss is not None or answerability_token_weights is not None
            ):
                token_loss_fn = self.token_loss if self.token_loss is not None else self.base_token_loss
                total_loss = token_loss_fn(
                    outputs["logits"],
                    outputs["labels"],
                    extra_token_weights=answerability_token_weights,
                )
            else:
                total_loss = outputs["loss"]
            loss_dict = {"total": total_loss}

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
            answerable = batch.get("answerable")

            outputs = self.model(
                images=images,
                prompt_text=self.prompt_template,
                target_text=target_jsons,
            )

            answerability_token_weights = None
            if outputs.get("labels") is not None and answerable is not None:
                answerability_token_weights = self._build_answerability_token_weights(
                    target_jsons=target_jsons,
                    answerable=answerable,
                    labels=outputs["labels"],
                )

            if outputs.get("labels") is not None and (
                self.token_loss is not None or answerability_token_weights is not None
            ):
                token_loss_fn = self.token_loss if self.token_loss is not None else self.base_token_loss
                val_loss = token_loss_fn(
                    outputs["logits"],
                    outputs["labels"],
                    extra_token_weights=answerability_token_weights,
                )
            else:
                val_loss = outputs["loss"]
            total_loss += val_loss.item()
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
