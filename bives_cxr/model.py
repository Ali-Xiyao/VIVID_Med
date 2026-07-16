"""Core BiVES-CXR bipolar evidence model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .decoder import EvidenceStateDecoder
from .gates import EvidenceGate
from .interventions import (
    build_equal_area_control_mask,
    delete_control,
    delete_evidence,
    retain_evidence,
)


@dataclass(frozen=True)
class BiVESModelConfig:
    visual_dim: int
    statement_dim: int
    fusion_dim: int = 512
    evidence_max: float = 8.0
    gate_mode: str = "soft_topk"
    topk: int = 16
    gate_temperature: float = 0.5
    tau_a: float = 1.0
    tau_d: float = 1.0
    tau_p: float = 1.0


class BiVESCXR(nn.Module):
    """Statement-conditioned bipolar spatial evidence field."""

    decoder_kind = "bipolar_closed_form"
    has_flat_state_head = False

    def __init__(self, config: BiVESModelConfig) -> None:
        super().__init__()
        if config.evidence_max <= 0:
            raise ValueError("evidence_max must be positive")
        self.config = config
        self.visual_projection = nn.Linear(config.visual_dim, config.fusion_dim)
        self.statement_projection = nn.Linear(config.statement_dim, config.fusion_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(config.fusion_dim * 4),
            nn.Linear(config.fusion_dim * 4, config.fusion_dim),
            nn.GELU(),
            nn.LayerNorm(config.fusion_dim),
        )
        self.evidence_head = nn.Linear(config.fusion_dim, 2)
        self.gate_head = nn.Linear(config.fusion_dim, 1)
        self.gate = EvidenceGate(
            mode=config.gate_mode,
            topk=config.topk,
            temperature=config.gate_temperature,
        )
        self.decoder = EvidenceStateDecoder(config.tau_a, config.tau_d, config.tau_p)
        self.mask_token = nn.Parameter(torch.zeros(config.visual_dim))

    def score_tokens(
        self,
        patch_tokens: torch.Tensor,
        statement_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if patch_tokens.ndim != 3:
            raise ValueError("patch_tokens must have shape [B,P,D]")
        if statement_embeddings.ndim != 2 or statement_embeddings.shape[0] != patch_tokens.shape[0]:
            raise ValueError("statement_embeddings must have shape [B,D]")
        if valid_mask.shape != patch_tokens.shape[:2]:
            raise ValueError("valid_mask must have shape [B,P]")

        visual = self.visual_projection(patch_tokens)
        statement = self.statement_projection(statement_embeddings).unsqueeze(1).expand_as(visual)
        fused = self.fusion(torch.cat((visual, statement, visual * statement, torch.abs(visual - statement)), dim=-1))
        evidence_pm = torch.sigmoid(self.evidence_head(fused)) * self.config.evidence_max
        gate_logits = self.gate_head(fused).squeeze(-1)
        gate = self.gate(gate_logits, valid_mask)
        evidence_maps = evidence_pm * gate.unsqueeze(-1)
        denominator = gate.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        aggregated = evidence_maps.sum(dim=1) / denominator
        decoder = self.decoder(aggregated[:, 0], aggregated[:, 1])
        return {
            "gate_logits": gate_logits,
            "gate": gate,
            "evidence_pm": evidence_pm,
            "evidence_maps": evidence_maps,
            "evidence_pos": aggregated[:, 0],
            "evidence_neg": aggregated[:, 1],
            **decoder,
        }

    def forward(
        self,
        patch_tokens: torch.Tensor,
        statement_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
        run_interventions: bool = True,
    ) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        original = self.score_tokens(patch_tokens, statement_embeddings, valid_mask)
        output: dict[str, dict[str, torch.Tensor] | torch.Tensor] = {"original": original}
        if not run_interventions:
            return output

        keep_tokens = retain_evidence(patch_tokens, original["gate"], self.mask_token)
        drop_tokens = delete_evidence(patch_tokens, original["gate"], self.mask_token)
        control_mask = build_equal_area_control_mask(original["gate"], valid_mask, self.config.topk)
        control_tokens = delete_control(patch_tokens, control_mask, self.mask_token)
        output["keep"] = self.score_tokens(keep_tokens, statement_embeddings, valid_mask)
        output["drop"] = self.score_tokens(drop_tokens, statement_embeddings, valid_mask)
        output["control"] = self.score_tokens(control_tokens, statement_embeddings, valid_mask)
        output["control_mask"] = control_mask
        return output
