"""Field-anchored query projector with an SPD-matched token budget."""

from __future__ import annotations

import torch
from torch import nn


FIELD_NAMES = ("observation", "assertion", "anatomy", "global")


class FieldAnchoredProjector(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        output_dim: int,
        *,
        tokens_per_field: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if tokens_per_field != 2:
            raise ValueError("field-anchored comparison must retain two tokens per field")
        if vision_dim % num_heads:
            raise ValueError("vision_dim must be divisible by num_heads")
        self.field_names = FIELD_NAMES
        self.tokens_per_field = tokens_per_field
        field_embeddings = torch.zeros(len(FIELD_NAMES), 1, vision_dim)
        field_embeddings[:, 0, : len(FIELD_NAMES)] = torch.eye(len(FIELD_NAMES))
        self.register_buffer("field_embeddings", field_embeddings * 0.02)
        self.residual_queries = nn.Parameter(
            torch.empty(len(FIELD_NAMES), tokens_per_field, vision_dim)
        )
        nn.init.normal_(self.residual_queries, std=0.02)
        self.attention = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    vision_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in FIELD_NAMES
            ]
        )
        self.projection = nn.Sequential(
            nn.Linear(vision_dim, vision_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(vision_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    @property
    def num_query_tokens(self) -> int:
        return len(self.field_names) * self.tokens_per_field

    def forward(self, patch_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        if patch_tokens.ndim != 3:
            raise ValueError("patch_tokens must have shape [batch, tokens, channels]")
        batch = patch_tokens.shape[0]
        result: dict[str, torch.Tensor] = {}
        for index, name in enumerate(self.field_names):
            query = self.field_embeddings[index] + self.residual_queries[index]
            query = query.unsqueeze(0).expand(batch, -1, -1)
            attended, _ = self.attention[index](
                query, patch_tokens, patch_tokens, need_weights=False
            )
            result[name] = self.projection(attended)
        return result

    @staticmethod
    def flatten(fields: dict[str, torch.Tensor]) -> torch.Tensor:
        missing = [name for name in FIELD_NAMES if name not in fields]
        if missing:
            raise ValueError(f"missing field outputs: {missing}")
        return torch.cat([fields[name] for name in FIELD_NAMES], dim=1)
