"""
V10.1 SPD: Structured Prediction Decomposition Projector

核心思想：用多组可学习 query tokens 通过 cross-attention 从 ViT 特征中
提取不同方面的视觉信息（结构/发现/细节），正交性损失迫使各组关注不同区域，
让 ViT 学到解耦的、可迁移的表征。

与 VisionProjector 的区别：
- VisionProjector: 固定 prefix tokens + MLP 投影（prefix 不依赖图像内容）
- SPDProjector: cross-attention query tokens（依赖图像内容，自适应提取）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SPDProjector(nn.Module):
    """
    Structured Prediction Decomposition Projector

    多组 query tokens 通过 cross-attention 从 ViT 特征中提取不同方面信息。
    """

    def __init__(
        self,
        vit_embed_dim: int = 768,
        llm_embed_dim: int = 1536,
        num_groups: int = 3,
        tokens_per_group: int = 2,
        num_heads: int = 4,
        mlp_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vit_embed_dim = vit_embed_dim
        self.llm_embed_dim = llm_embed_dim
        self.num_groups = num_groups
        self.tokens_per_group = tokens_per_group
        if mlp_hidden_dim is None:
            mlp_hidden_dim = vit_embed_dim * 2

        # 每组独立的 query tokens (在 ViT 空间中)
        self.group_queries = nn.ParameterList([
            nn.Parameter(torch.randn(1, tokens_per_group, vit_embed_dim) * 0.02)
            for _ in range(num_groups)
        ])

        # 每组独立的 cross-attention: query tokens attend to ViT features
        self.cross_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                vit_embed_dim, num_heads=num_heads,
                dropout=dropout, batch_first=True,
            )
            for _ in range(num_groups)
        ])

        # 共享 MLP 投影: vit_dim → llm_dim
        self.mlp = nn.Sequential(
            nn.Linear(vit_embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, llm_embed_dim),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(llm_embed_dim)

        # 存储 attention weights 用于正交性损失
        self._attn_weights = []

        self._init_weights()

    @property
    def num_prefix_tokens(self):
        """兼容接口：SPD query tokens 等效于 prefix tokens"""
        return self.num_groups * self.tokens_per_group

    def _init_weights(self):
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, vit_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vit_features: (B, N, vit_embed_dim) ViT 输出特征

        Returns:
            (B, num_groups * tokens_per_group + N, llm_embed_dim)
        """
        B = vit_features.shape[0]
        if vit_features.dim() == 2:
            vit_features = vit_features.unsqueeze(1)

        self._attn_weights = []
        group_outputs = []

        for i in range(self.num_groups):
            queries = self.group_queries[i].expand(B, -1, -1)
            attended, attn_w = self.cross_attentions[i](
                queries, vit_features, vit_features,
                need_weights=True, average_attn_weights=True,
            )
            group_outputs.append(attended)  # (B, tokens_per_group, vit_dim)
            self._attn_weights.append(attn_w)  # (B, tokens_per_group, N)

        # Concat 所有组: (B, num_groups * tokens_per_group, vit_dim)
        spd_tokens = torch.cat(group_outputs, dim=1)

        # 投影到 LLM 空间
        spd_projected = self.layer_norm(self.mlp(spd_tokens))

        # ViT 特征也投影
        vit_projected = self.layer_norm(self.mlp(vit_features))

        # 输出: [SPD query tokens, ViT tokens]
        return torch.cat([spd_projected, vit_projected], dim=1)

    def get_orthogonality_loss(self) -> torch.Tensor:
        """
        计算各组 attention pattern 之间的正交性损失。
        鼓励不同组关注 ViT 特征的不同区域。
        """
        if len(self._attn_weights) < 2:
            device = self.group_queries[0].device
            return torch.tensor(0.0, device=device)

        # 每组的平均 attention: (B, N)
        group_attns = []
        for attn_w in self._attn_weights:
            avg_attn = attn_w.mean(dim=1)  # (B, N)
            group_attns.append(avg_attn)

        # 两两计算 cosine similarity，鼓励正交
        ortho_loss = torch.tensor(0.0, device=group_attns[0].device)
        count = 0
        for i in range(len(group_attns)):
            for j in range(i + 1, len(group_attns)):
                cos_sim = F.cosine_similarity(
                    group_attns[i], group_attns[j], dim=-1
                )
                ortho_loss = ortho_loss + cos_sim.abs().mean()
                count += 1

        return ortho_loss / max(count, 1)
