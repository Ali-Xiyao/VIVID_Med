"""Core BiVES-CXR bipolar evidence model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .decoder import EvidenceStateDecoder
from .gates import EvidenceGate, exact_topk_mask
from .interventions import (
    build_matched_control_masks,
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
    num_controls: int = 4
    control_mode: str = "random_disjoint"
    contextual_layers: int = 1
    contextual_heads: int = 4
    contextual_dropout: float = 0.0


class BiVESCXR(nn.Module):
    """Statement-conditioned bipolar spatial evidence field."""

    decoder_kind = "bipolar_closed_form"
    has_flat_state_head = False

    def __init__(self, config: BiVESModelConfig) -> None:
        super().__init__()
        if config.evidence_max <= 0:
            raise ValueError("evidence_max must be positive")
        if config.contextual_layers <= 0:
            raise ValueError("contextual_layers must be positive")
        if config.contextual_heads <= 0 or config.fusion_dim % config.contextual_heads:
            raise ValueError("contextual_heads must divide fusion_dim")
        self.config = config
        self.visual_projection = nn.Linear(config.visual_dim, config.fusion_dim)
        self.statement_projection = nn.Linear(config.statement_dim, config.fusion_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(config.fusion_dim * 4),
            nn.Linear(config.fusion_dim * 4, config.fusion_dim),
            nn.GELU(),
            nn.LayerNorm(config.fusion_dim),
        )
        contextual_layer = nn.TransformerEncoderLayer(
            d_model=config.fusion_dim,
            nhead=config.contextual_heads,
            dim_feedforward=config.fusion_dim * 4,
            dropout=config.contextual_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.contextual_evidence = nn.TransformerEncoder(
            contextual_layer,
            num_layers=config.contextual_layers,
            enable_nested_tensor=False,
        )
        self.contextual_norm = nn.LayerNorm(config.fusion_dim)
        self.evidence_head = nn.Linear(config.fusion_dim, 2)
        self.gate_head = nn.Linear(config.fusion_dim, 1)
        self.gate = EvidenceGate(
            mode=config.gate_mode,
            topk=config.topk,
            temperature=config.gate_temperature,
        )
        self.decoder = EvidenceStateDecoder(config.tau_a, config.tau_d, config.tau_p)

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
        fused = self.fusion(
            torch.cat(
                (visual, statement, visual * statement, torch.abs(visual - statement)),
                dim=-1,
            )
        )
        valid_float = valid_mask.to(dtype=fused.dtype).unsqueeze(-1)
        fused = fused * valid_float
        contextual = self.contextual_evidence(
            fused,
            src_key_padding_mask=~valid_mask.bool(),
        )
        contextual = self.contextual_norm(contextual) * valid_float
        evidence_pm = torch.sigmoid(self.evidence_head(contextual)) * self.config.evidence_max
        gate_logits = self.gate_head(contextual).squeeze(-1)
        gate = self.gate(gate_logits, valid_mask)
        evidence_maps = evidence_pm * gate.unsqueeze(-1)
        denominator = gate.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        aggregated = evidence_maps.sum(dim=1) / denominator
        decoder = self.decoder(aggregated[:, 0], aggregated[:, 1])
        return {
            "gate_logits": gate_logits,
            "gate": gate,
            "contextual_tokens": contextual,
            "evidence_pm": evidence_pm,
            "evidence_maps": evidence_maps,
            "evidence_pos": aggregated[:, 0],
            "evidence_neg": aggregated[:, 1],
            "valid_mask": valid_mask,
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

        evidence_hard_mask = exact_topk_mask(
            original["gate"].detach(),
            valid_mask,
            self.config.topk,
        ).bool()
        keep_valid = valid_mask.bool() & evidence_hard_mask
        drop_valid = valid_mask.bool() & ~evidence_hard_mask
        # The intervention is applied before the shared contextual evidence
        # block. Forward values remain exact-K while the straight-through gate
        # supplies selector gradients.
        keep_tokens = retain_evidence(patch_tokens, original["gate"])
        drop_tokens = delete_evidence(patch_tokens, original["gate"])
        control_masks = build_matched_control_masks(
            evidence_hard_mask.to(original["gate"].dtype),
            valid_mask,
            self.config.topk,
            num_controls=self.config.num_controls,
            mode=self.config.control_mode,
        )
        output["keep"] = self.score_tokens(keep_tokens, statement_embeddings, keep_valid)
        output["drop"] = self.score_tokens(drop_tokens, statement_embeddings, drop_valid)

        controls: list[dict[str, torch.Tensor]] = []
        control_valid_masks: list[torch.Tensor] = []
        for control_index in range(control_masks.shape[1]):
            control_mask = control_masks[:, control_index]
            control_valid = valid_mask.bool() & ~(control_mask > 0.5)
            control_valid_masks.append(control_valid)
            control_tokens = delete_control(patch_tokens, control_mask)
            controls.append(self.score_tokens(control_tokens, statement_embeddings, control_valid))
        output["controls"] = controls
        output["control"] = {
            key: (
                torch.stack([branch[key] for branch in controls], dim=0).mean(dim=0)
                if controls[0][key].is_floating_point()
                else torch.stack([branch[key] for branch in controls], dim=0).any(dim=0)
            )
            for key in controls[0]
        }
        output["evidence_hard_mask"] = evidence_hard_mask
        output["control_masks"] = control_masks
        output["keep_valid_mask"] = keep_valid
        output["drop_valid_mask"] = drop_valid
        output["control_valid_masks"] = torch.stack(control_valid_masks, dim=1)
        return output
