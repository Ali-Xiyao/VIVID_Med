"""Patch-level multiple-instance dense verifier for ARISE-CXR development."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MILVerifierConfig:
    visual_dim: int
    num_statements: int
    temperature: float = 0.5
    max_pool_weight: float = 0.75

    def validate(self) -> None:
        if self.visual_dim <= 0 or self.num_statements <= 0:
            raise ValueError("MIL verifier dimensions must be positive")
        if self.temperature <= 0.0:
            raise ValueError("MIL verifier temperature must be positive")
        if not 0.0 <= self.max_pool_weight <= 1.0:
            raise ValueError("MIL verifier max_pool_weight must be in [0, 1]")

    def to_dict(self) -> dict[str, int | float]:
        self.validate()
        return asdict(self)


class PatchMILVerifier(nn.Module):
    """Finding-specific patch logits with smooth-max and mean aggregation."""

    has_flat_four_class_head = False

    def __init__(self, config: MILVerifierConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.weight = nn.Embedding(config.num_statements, config.visual_dim)
        self.bias = nn.Embedding(config.num_statements, 1)
        nn.init.normal_(self.weight.weight, std=0.02)
        nn.init.zeros_(self.bias.weight)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        statement_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if patch_tokens.ndim != 3 or valid_mask.shape != patch_tokens.shape[:2]:
            raise ValueError("patch_tokens/valid_mask must be [B,P,D]/[B,P]")
        if patch_tokens.shape[-1] != self.config.visual_dim:
            raise ValueError("MIL verifier visual dimension changed")
        if statement_indices.shape != (patch_tokens.shape[0],):
            raise ValueError("statement_indices must be [B]")
        valid = valid_mask.bool()
        if not bool(valid.any(dim=1).all()):
            raise ValueError("every MIL verifier sample needs a valid patch")

        normalized = F.layer_norm(patch_tokens.float(), (self.config.visual_dim,))
        weight = self.weight(statement_indices)
        bias = self.bias(statement_indices).squeeze(-1)
        patch_logits = torch.einsum("bpd,bd->bp", normalized, weight)
        patch_logits = patch_logits / (self.config.visual_dim**0.5) + bias.unsqueeze(1)
        masked = patch_logits.masked_fill(~valid, float("-inf"))
        temperature = self.config.temperature
        counts = valid.sum(dim=1).to(patch_logits.dtype)
        smooth_max = temperature * (
            torch.logsumexp(masked / temperature, dim=1) - torch.log(counts)
        )
        mean = (patch_logits * valid.to(patch_logits.dtype)).sum(dim=1) / counts
        alpha = self.config.max_pool_weight
        margin = alpha * smooth_max + (1.0 - alpha) * mean
        return {
            "margin": margin,
            "support_probability": torch.sigmoid(margin),
            "patch_logits": patch_logits,
            "valid_mask": valid,
        }


def mil_binary_loss(margin: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if margin.ndim != 1 or labels.shape != margin.shape:
        raise ValueError("MIL margin and labels must be [B]")
    return F.binary_cross_entropy_with_logits(margin, labels.to(margin.dtype))
