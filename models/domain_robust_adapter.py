"""Domain-Robust Adapter (DRA) utilities."""

from __future__ import annotations

import torch
import torch.nn as nn


class DomainRobustAdapter(nn.Module):
    """Residual adapter with optional domain-specific parameters."""

    def __init__(self, embed_dim: int, hidden_dim: int | None = None, domain_count: int = 1):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, embed_dim),
                )
                for _ in range(domain_count)
            ]
        )

    def forward(self, embeddings: torch.Tensor, domain_id: int = 0) -> torch.Tensor:
        adapter = self.adapters[min(max(domain_id, 0), len(self.adapters) - 1)]
        return embeddings + adapter(embeddings)


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source = source - source.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    cov_s = source.T @ source / max(source.shape[0] - 1, 1)
    cov_t = target.T @ target / max(target.shape[0] - 1, 1)
    return torch.mean((cov_s - cov_t) ** 2)


def mmd_rbf(source: torch.Tensor, target: torch.Tensor, gamma: float | None = None) -> torch.Tensor:
    if gamma is None:
        with torch.no_grad():
            joined = torch.cat([source[:512], target[:512]], dim=0)
            dist = torch.cdist(joined, joined) ** 2
            positive = dist[dist > 0]
            median = positive.median() if positive.numel() else torch.tensor(1.0, device=source.device)
            gamma = float(1.0 / torch.clamp(median, min=1e-6))
    kxx = torch.exp(-gamma * torch.cdist(source, source) ** 2).mean()
    kyy = torch.exp(-gamma * torch.cdist(target, target) ** 2).mean()
    kxy = torch.exp(-gamma * torch.cdist(source, target) ** 2).mean()
    return kxx + kyy - 2 * kxy


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, weight: float) -> torch.Tensor:  # type: ignore[override]
        ctx.weight = weight
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        return -ctx.weight * grad_output, None


def gradient_reverse(inputs: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    return GradientReverse.apply(inputs, weight)

