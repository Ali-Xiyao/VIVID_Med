"""
Vision Projector
将 ViT 特征映射到 LLM 的 embedding 空间

架构：
- VisionProjector: 2-layer MLP + learnable prefix tokens (V9)
"""

import torch
import torch.nn as nn
from typing import Optional


class VisionProjector(nn.Module):
    """
    视觉投影器：将 ViT 特征映射到 LLM embedding 空间

    包含：
    1. MLP 投影层（2层）
    2. 可学习的 prefix tokens
    """

    def __init__(
        self,
        vit_embed_dim: int = 768,      # ViT-B/16 的输出维度
        llm_embed_dim: int = 1536,     # Qwen3-1.7B 的 hidden_size
        num_prefix_tokens: int = 16,    # 可学习前缀 token 数量
        mlp_hidden_dim: Optional[int] = None,  # MLP 隐藏层维度
        dropout: float = 0.1,
    ):
        """
        Args:
            vit_embed_dim: ViT 输出的嵌入维度
            llm_embed_dim: LLM 的嵌入维度
            num_prefix_tokens: 可学习前缀 token 数量（v1.0 默认 16）
            mlp_hidden_dim: MLP 隐藏层维度，默认为 vit_embed_dim * 4
            dropout: Dropout 概率
        """
        super().__init__()

        self.vit_embed_dim = vit_embed_dim
        self.llm_embed_dim = llm_embed_dim
        self.num_prefix_tokens = num_prefix_tokens

        if mlp_hidden_dim is None:
            mlp_hidden_dim = vit_embed_dim * 4

        # 2-layer MLP 投影
        self.mlp = nn.Sequential(
            nn.Linear(vit_embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, llm_embed_dim),
            nn.Dropout(dropout),
        )

        # 可学习的 prefix tokens
        # 这些 tokens 会被拼接到视觉特征之前，作为 LLM 的输入前缀
        self.prefix_tokens = nn.Parameter(
            torch.randn(1, num_prefix_tokens, llm_embed_dim) * 0.02
        )

        # Layer Norm（可选，有助于稳定训练）
        self.layer_norm = nn.LayerNorm(llm_embed_dim)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        vit_features: torch.Tensor,
        return_prefix_only: bool = False,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            vit_features: ViT 输出特征
                - 如果是 (B, embed_dim)：单个 CLS token
                - 如果是 (B, N, embed_dim)：所有 tokens
            return_prefix_only: 是否只返回 prefix tokens（用于调试）

        Returns:
            projected: 投影后的特征 (B, num_tokens, llm_embed_dim)
                - num_tokens = num_prefix_tokens + num_vit_tokens
        """
        batch_size = vit_features.shape[0]

        # 确保输入是 3D
        if vit_features.dim() == 2:
            vit_features = vit_features.unsqueeze(1)  # (B, 1, embed_dim)

        # MLP 投影
        projected = self.mlp(vit_features)  # (B, N, llm_embed_dim)

        # Layer Norm
        projected = self.layer_norm(projected)

        # 扩展 prefix tokens 到 batch size
        prefix = self.prefix_tokens.expand(batch_size, -1, -1)  # (B, num_prefix, llm_embed_dim)

        if return_prefix_only:
            return prefix

        # 拼接 prefix tokens 和投影后的视觉特征
        # [prefix_tokens, visual_tokens]
        output = torch.cat([prefix, projected], dim=1)  # (B, num_prefix + N, llm_embed_dim)

        return output

    def get_num_visual_tokens(self, num_vit_tokens: int = 1) -> int:
        """获取输出的视觉 token 总数"""
        return self.num_prefix_tokens + num_vit_tokens


class SimpleProjector(nn.Module):
    """
    简化版投影器：只有 MLP，没有 prefix tokens
    用于消融实验
    """

    def __init__(
        self,
        vit_embed_dim: int = 768,
        llm_embed_dim: int = 1536,
        mlp_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if mlp_hidden_dim is None:
            mlp_hidden_dim = vit_embed_dim * 4

        self.mlp = nn.Sequential(
            nn.Linear(vit_embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, llm_embed_dim),
            nn.LayerNorm(llm_embed_dim),
        )

    def forward(self, vit_features: torch.Tensor) -> torch.Tensor:
        if vit_features.dim() == 2:
            vit_features = vit_features.unsqueeze(1)
        return self.mlp(vit_features)


