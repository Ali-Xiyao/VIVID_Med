"""
Frozen Semantic Anchor (FSA) - Augmentation Consistency Loss

核心思想：冻结 LLM 提供稳定的语义锚点空间。
对同一图像的两个增强视图，经过 ViT→Projector→LLM 后的 hidden states 应该一致。

与 ViTP 的区别：ViTP 训练 LLM（目标空间在漂移），
我们冻结 LLM（目标空间固定，consistency loss 有明确的优化目标）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AugmentationConsistencyLoss(nn.Module):
    """
    计算两个增强视图在 LLM hidden state 空间中的一致性 loss

    L_consistency = 1 - mean(cosine_similarity(h1, h2))
    """

    def __init__(self, pool_mode: str = "mean"):
        """
        Args:
            pool_mode: "mean" (mean pool over sequence) or "cls" (first token)
        """
        super().__init__()
        self.pool_mode = pool_mode

    def _pool(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pool hidden states to a single vector per sample"""
        if self.pool_mode == "cls":
            return hidden_states[:, 0, :]  # (B, D)
        else:  # "mean"
            return hidden_states.mean(dim=1)  # (B, D)

    def forward(
        self,
        hidden_states_1: torch.Tensor,
        hidden_states_2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states_1: (B, L, D) LLM hidden states for view 1
            hidden_states_2: (B, L, D) LLM hidden states for view 2

        Returns:
            loss: scalar consistency loss
        """
        h1 = self._pool(hidden_states_1)  # (B, D)
        h2 = self._pool(hidden_states_2)  # (B, D)
        return 1.0 - F.cosine_similarity(h1, h2, dim=-1).mean()
