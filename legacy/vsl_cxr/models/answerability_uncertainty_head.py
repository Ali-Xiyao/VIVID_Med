"""Answerability-Uncertainty Calibration Head (AUCH)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnswerabilityUncertaintyHead(nn.Module):
    """Predict answerability, uncertainty, and state for each finding."""

    def __init__(self, embed_dim: int, hidden_dim: int | None = None, num_states: int = 3):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.trunk = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, hidden_dim), nn.GELU())
        self.answerability = nn.Linear(hidden_dim, 1)
        self.uncertainty = nn.Linear(hidden_dim, 1)
        self.state = nn.Linear(hidden_dim, num_states)

    def forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.trunk(embeddings)
        return {
            "answerability_logit": self.answerability(features).squeeze(-1),
            "uncertainty_logit": self.uncertainty(features).squeeze(-1),
            "state_logits": self.state(features),
        }

    def loss(
        self,
        outputs: dict[str, torch.Tensor],
        answerable: torch.Tensor | None = None,
        uncertain: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        losses = []
        if answerable is not None:
            losses.append(F.binary_cross_entropy_with_logits(outputs["answerability_logit"], answerable.float()))
        if uncertain is not None:
            losses.append(F.binary_cross_entropy_with_logits(outputs["uncertainty_logit"], uncertain.float()))
        if state is not None:
            logits = outputs["state_logits"]
            if state_mask is not None:
                logits = logits[state_mask]
                state = state[state_mask]
            if logits.numel() > 0:
                losses.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), state.reshape(-1).long()))
        if not losses:
            return outputs["state_logits"].sum() * 0.0
        return sum(losses)


def brier_score(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.mean((probabilities - targets.float()) ** 2)

