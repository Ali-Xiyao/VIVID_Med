"""
训练损失函数

包含：
- L_tok: Next-token prediction (主监督)
- L_rank: Hard-negative ranking (可选)
- L_vdep: Visual-dependence (anti-shortcut, 可选)
- L_ans: Answerability prediction (可选)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class StructuredLoss(nn.Module):
    """
    结构化生成损失

    L_tok: 标准 cross-entropy，让模型生成 target JSON tokens
    对 answerable=false 的字段，target 固定为 null/uncertain
    """

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        """
        Args:
            ignore_index: 忽略的标签索引（不计算 loss）
            label_smoothing: 标签平滑
            reduction: 归约方式
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction=reduction,
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算 L_tok

        Args:
            logits: (B, L, V) 模型输出的 logits
            labels: (B, L) 目标 token ids
            attention_mask: (B, L) attention mask（可选）

        Returns:
            loss: 标量损失
        """
        # Shift logits and labels for next-token prediction
        # logits: predict token at position t
        # labels: actual token at position t+1
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten
        vocab_size = shift_logits.shape[-1]
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)

        # Compute loss
        loss = self.ce_loss(shift_logits, shift_labels)

        return loss


class AnswerabilityLoss(nn.Module):
    """
    Answerability 预测损失

    L_ans: 为每个字段预测可答性（二分类）
    推理时用于 gating：当预测不可答时，强制输出 null/uncertain
    """

    def __init__(self, num_fields: int = 14):
        """
        Args:
            num_fields: 字段数量（CheXpert 有 14 个 findings）
        """
        super().__init__()
        self.num_fields = num_fields
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        pred_answerability: torch.Tensor,
        target_answerability: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算 L_ans

        Args:
            pred_answerability: (B, num_fields) 预测的可答性 logits
            target_answerability: (B, num_fields) 目标可答性 (0/1)
            mask: (B, num_fields) 有效字段 mask

        Returns:
            loss: 标量损失
        """
        loss = self.bce_loss(pred_answerability, target_answerability.float())

        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss


class RankingLoss(nn.Module):
    """
    Hard-negative ranking 损失

    L_rank: 让模型学会区分正确答案和"错得很像真的"答案
    score(y_true) > score(y_neg) + margin
    """

    def __init__(self, margin: float = 1.0):
        """
        Args:
            margin: ranking margin
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 L_rank (margin ranking loss)

        Args:
            pos_scores: (B,) 正样本得分
            neg_scores: (B, K) 负样本得分，K 是每个样本的负样本数

        Returns:
            loss: 标量损失
        """
        # 扩展 pos_scores 以匹配 neg_scores
        if neg_scores.dim() == 2:
            pos_scores = pos_scores.unsqueeze(1).expand_as(neg_scores)

        # Margin ranking loss: max(0, margin - (pos - neg))
        loss = F.relu(self.margin - (pos_scores - neg_scores))

        return loss.mean()


class VisualDependenceLoss(nn.Module):
    """
    Visual-dependence 损失 (anti-shortcut)

    L_vdep: 确保模型真的在看图，而不是走语言捷径
    - prompt corruption: 语义不变，措辞变化 → 输出应一致
    - image swap: prompt 不变，图像变化 → 输出应变化
    """

    def __init__(
        self,
        consistency_weight: float = 1.0,
        diversity_weight: float = 1.0,
    ):
        """
        Args:
            consistency_weight: prompt corruption 一致性权重
            diversity_weight: image swap 多样性权重
        """
        super().__init__()
        self.consistency_weight = consistency_weight
        self.diversity_weight = diversity_weight

    def forward(
        self,
        original_logits: torch.Tensor,
        corrupted_logits: Optional[torch.Tensor] = None,
        swapped_logits: Optional[torch.Tensor] = None,
        original_labels: Optional[torch.Tensor] = None,
        swapped_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算 L_vdep

        Args:
            original_logits: (B, L, V) 原始输入的 logits
            corrupted_logits: (B, L, V) prompt corruption 后的 logits
            swapped_logits: (B, L, V) image swap 后的 logits
            original_labels: (B, L) 原始标签
            swapped_labels: (B, L) swap 后的标签

        Returns:
            包含各项损失的字典
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=original_logits.device)

        # 1. Prompt corruption consistency
        # 同语义不同措辞 → 输出分布应一致
        if corrupted_logits is not None:
            # KL divergence between original and corrupted
            original_probs = F.softmax(original_logits, dim=-1)
            corrupted_probs = F.softmax(corrupted_logits, dim=-1)

            consistency_loss = F.kl_div(
                corrupted_probs.log(),
                original_probs,
                reduction="batchmean",
            )
            losses["consistency"] = consistency_loss
            total_loss = total_loss + self.consistency_weight * consistency_loss

        # 2. Image swap diversity
        # 同 prompt 换图 → 输出应变化（尤其 findings/measurements）
        if swapped_logits is not None and original_labels is not None and swapped_labels is not None:
            # 如果标签不同，输出也应该不同
            # 使用负的 KL divergence 或其他多样性度量
            label_diff = (original_labels != swapped_labels).float().mean()

            if label_diff > 0:
                original_probs = F.softmax(original_logits, dim=-1)
                swapped_probs = F.softmax(swapped_logits, dim=-1)

                # 鼓励输出不同（当标签不同时）
                similarity = F.cosine_similarity(
                    original_probs.view(original_probs.shape[0], -1),
                    swapped_probs.view(swapped_probs.shape[0], -1),
                    dim=-1,
                )
                diversity_loss = similarity.mean() * label_diff
                losses["diversity"] = diversity_loss
                total_loss = total_loss + self.diversity_weight * diversity_loss

        losses["total"] = total_loss
        return losses


class VIVIDLoss(nn.Module):
    """
    VIVID 总损失

    L = L_tok + λ_rank * L_rank + λ_vdep * L_vdep + λ_ans * L_ans
    """

    def __init__(
        self,
        lambda_rank: float = 0.1,
        lambda_vdep: float = 0.1,
        lambda_ans: float = 0.1,
        num_fields: int = 14,
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            lambda_rank: L_rank 权重
            lambda_vdep: L_vdep 权重
            lambda_ans: L_ans 权重
            num_fields: 字段数量
            label_smoothing: 标签平滑
        """
        super().__init__()

        self.lambda_rank = lambda_rank
        self.lambda_vdep = lambda_vdep
        self.lambda_ans = lambda_ans

        self.l_tok = StructuredLoss(label_smoothing=label_smoothing)
        self.l_rank = RankingLoss()
        self.l_vdep = VisualDependenceLoss()
        self.l_ans = AnswerabilityLoss(num_fields=num_fields)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        # 可选输入
        pos_scores: Optional[torch.Tensor] = None,
        neg_scores: Optional[torch.Tensor] = None,
        corrupted_logits: Optional[torch.Tensor] = None,
        pred_answerability: Optional[torch.Tensor] = None,
        target_answerability: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算总损失

        Returns:
            包含各项损失的字典
        """
        losses = {}

        # L_tok (主损失)
        l_tok = self.l_tok(logits, labels)
        losses["l_tok"] = l_tok
        total = l_tok

        # L_rank (可选)
        if pos_scores is not None and neg_scores is not None:
            l_rank = self.l_rank(pos_scores, neg_scores)
            losses["l_rank"] = l_rank
            total = total + self.lambda_rank * l_rank

        # L_vdep (可选)
        if corrupted_logits is not None:
            vdep_losses = self.l_vdep(logits, corrupted_logits=corrupted_logits)
            losses["l_vdep"] = vdep_losses["total"]
            total = total + self.lambda_vdep * vdep_losses["total"]

        # L_ans (可选)
        if pred_answerability is not None and target_answerability is not None:
            l_ans = self.l_ans(pred_answerability, target_answerability)
            losses["l_ans"] = l_ans
            total = total + self.lambda_ans * l_ans

        losses["total"] = total
        return losses
