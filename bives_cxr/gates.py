"""Differentiable evidence-set gates."""

from __future__ import annotations

import torch
import torch.nn as nn


def exact_topk_mask(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Return an exact-K binary mask for every sample.

    Formal BiVES batches must expose at least ``topk`` valid content patches.
    Silently shrinking K would break the equal-area intervention contract.
    """

    if scores.shape != valid_mask.shape:
        raise ValueError("scores and valid_mask must have shape [batch, patches]")
    valid_mask = valid_mask.bool()
    valid_counts = valid_mask.sum(dim=-1)
    if bool((valid_counts < topk).any()):
        bad = torch.where(valid_counts < topk)[0].tolist()
        raise ValueError(
            f"exact top-k requires at least {topk} valid patches per sample; "
            f"failed batch indices={bad}"
        )
    masked_scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
    indices = torch.topk(masked_scores, k=topk, dim=-1).indices
    hard = torch.zeros_like(scores)
    hard.scatter_(1, indices, 1.0)
    return hard.masked_fill(~valid_mask, 0.0)


class EvidenceGate(nn.Module):
    """Create a sparse patch gate without defining a state classifier."""

    def __init__(
        self,
        mode: str = "soft_topk",
        topk: int = 16,
        temperature: float = 0.5,
    ) -> None:
        super().__init__()
        if mode not in {"sigmoid", "soft_topk"}:
            raise ValueError(f"unsupported gate mode: {mode}")
        if topk <= 0 or temperature <= 0:
            raise ValueError("topk and temperature must be positive")
        self.mode = mode
        self.topk = int(topk)
        self.temperature = float(temperature)

    def forward(self, logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if logits.shape != valid_mask.shape:
            raise ValueError("gate logits and valid_mask must have shape [batch, patches]")
        valid_mask = valid_mask.bool()
        if self.mode == "soft_topk" and bool((valid_mask.sum(dim=-1) < self.topk).any()):
            raise ValueError(
                f"soft_topk requires at least {self.topk} valid patches for every sample"
            )
        masked_logits = logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)

        if self.mode == "sigmoid":
            relaxed = torch.sigmoid(masked_logits / self.temperature)
        elif self.mode == "soft_topk":
            threshold = torch.topk(masked_logits, k=self.topk, dim=-1).values[..., -1:].detach()
            relaxed = torch.sigmoid((masked_logits - threshold) / self.temperature)
        relaxed = relaxed.masked_fill(~valid_mask, 0.0)
        if self.mode == "sigmoid":
            return relaxed
        hard = exact_topk_mask(relaxed.detach(), valid_mask, self.topk)
        return hard + relaxed - relaxed.detach()
