"""Audited SPD baseline with the historical 4x2 identity frozen."""

from __future__ import annotations

import itertools

import torch
from torch import nn
from torch.nn import functional as F


class SPDProjector(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        output_dim: int,
        *,
        num_groups: int = 4,
        tokens_per_group: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_groups != 4 or tokens_per_group != 2:
            raise ValueError("the controlled SPD baseline is frozen at 4 groups x 2 tokens")
        if vision_dim % num_heads:
            raise ValueError("vision_dim must be divisible by num_heads")
        self.num_groups = num_groups
        self.tokens_per_group = tokens_per_group
        self.queries = nn.Parameter(torch.empty(num_groups, tokens_per_group, vision_dim))
        nn.init.normal_(self.queries, std=0.02)
        self.attention = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    vision_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_groups)
            ]
        )
        self.projection = nn.Sequential(
            nn.Linear(vision_dim, vision_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(vision_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )
        self._attention_maps: list[torch.Tensor] = []

    @property
    def num_query_tokens(self) -> int:
        return self.num_groups * self.tokens_per_group

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        if patch_tokens.ndim != 3:
            raise ValueError("patch_tokens must have shape [batch, tokens, channels]")
        batch = patch_tokens.shape[0]
        outputs: list[torch.Tensor] = []
        self._attention_maps = []
        for index, layer in enumerate(self.attention):
            query = self.queries[index].unsqueeze(0).expand(batch, -1, -1)
            attended, weights = layer(
                query,
                patch_tokens,
                patch_tokens,
                need_weights=True,
                average_attn_weights=True,
            )
            outputs.append(attended)
            self._attention_maps.append(weights)
        return self.projection(torch.cat(outputs, dim=1))

    def decorrelation_loss(self) -> torch.Tensor:
        if len(self._attention_maps) < 2:
            return self.queries.sum() * 0.0
        means = [item.mean(dim=1) for item in self._attention_maps]
        losses = [
            F.cosine_similarity(left, right, dim=-1).abs().mean()
            for left, right in itertools.combinations(means, 2)
        ]
        return torch.stack(losses).mean()
