"""Differentiable evidence-set gates."""

from __future__ import annotations

import torch
import torch.nn as nn


class EvidenceGate(nn.Module):
    """Create a sparse patch gate without defining a state classifier."""

    def __init__(
        self,
        mode: str = "soft_topk",
        topk: int = 16,
        temperature: float = 0.5,
        hard_concrete_low: float = -0.1,
        hard_concrete_high: float = 1.1,
    ) -> None:
        super().__init__()
        if mode not in {"sigmoid", "soft_topk", "hard_concrete"}:
            raise ValueError(f"unsupported gate mode: {mode}")
        if topk <= 0 or temperature <= 0:
            raise ValueError("topk and temperature must be positive")
        self.mode = mode
        self.topk = int(topk)
        self.temperature = float(temperature)
        self.low = float(hard_concrete_low)
        self.high = float(hard_concrete_high)

    def forward(self, logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if logits.shape != valid_mask.shape:
            raise ValueError("gate logits and valid_mask must have shape [batch, patches]")
        valid_mask = valid_mask.bool()
        masked_logits = logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)

        if self.mode == "sigmoid":
            gate = torch.sigmoid(masked_logits / self.temperature)
        elif self.mode == "soft_topk":
            valid_counts = valid_mask.sum(dim=-1).clamp_min(1)
            k = min(self.topk, int(valid_counts.max().item()))
            threshold = torch.topk(masked_logits, k=k, dim=-1).values[..., -1:].detach()
            gate = torch.sigmoid((masked_logits - threshold) / self.temperature)
        else:
            if self.training:
                uniform = torch.rand_like(masked_logits).clamp_(1e-6, 1.0 - 1e-6)
                logistic = torch.log(uniform) - torch.log1p(-uniform)
                relaxed = torch.sigmoid((masked_logits + logistic) / self.temperature)
            else:
                relaxed = torch.sigmoid(masked_logits)
            gate = (relaxed * (self.high - self.low) + self.low).clamp(0.0, 1.0)

        return gate.masked_fill(~valid_mask, 0.0)
