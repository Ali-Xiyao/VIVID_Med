"""Clinical Consistency Scoring Head (CCSH)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ClinicalConsistencyHead(nn.Module):
    """Score image/statement pairs as support, contradict, or uncertain."""

    def __init__(self, image_dim: int, statement_dim: int | None = None, hidden_dim: int | None = None, num_classes: int = 3):
        super().__init__()
        statement_dim = statement_dim or image_dim
        hidden_dim = hidden_dim or image_dim
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.statement_proj = nn.Linear(statement_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, image_embedding: torch.Tensor, statement_embedding: torch.Tensor) -> torch.Tensor:
        image = self.image_proj(image_embedding)
        statement = self.statement_proj(statement_embedding)
        features = torch.cat([image, statement, torch.abs(image - statement), image * statement], dim=-1)
        return self.classifier(features)

