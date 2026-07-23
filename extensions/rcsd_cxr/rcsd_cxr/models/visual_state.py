"""Equal-budget visual state models for the simplified post-G2 route."""

from __future__ import annotations

from pathlib import Path

import timm
import torch
from safetensors.torch import load_file
from torch import nn

from .field_anchored import FIELD_NAMES, FieldAnchoredProjector
from .spd import SPDProjector


class VisualStateModel(nn.Module):
    def __init__(
        self,
        variant: str,
        *,
        num_findings: int = 12,
        num_states: int = 3,
        output_dim: int = 2048,
        backbone_name: str = "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        backbone_weights: Path | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if variant not in {"spd", "field_anchor"}:
            raise ValueError("variant must be spd or field_anchor")
        self.variant = variant
        self.num_findings = num_findings
        self.num_states = num_states
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, num_classes=0
        )
        if backbone_weights is not None:
            state = load_file(str(backbone_weights))
            incompatible = self.backbone.load_state_dict(state, strict=False)
            allowed = {"head.weight", "head.bias"}
            if set(incompatible.unexpected_keys) - allowed:
                raise ValueError(
                    f"unexpected backbone keys: {incompatible.unexpected_keys}"
                )
            if incompatible.missing_keys:
                raise ValueError(
                    f"missing backbone keys: {incompatible.missing_keys}"
                )
        vision_dim = int(self.backbone.embed_dim)
        projector_type = (
            SPDProjector if variant == "spd" else FieldAnchoredProjector
        )
        self.projector = projector_type(
            vision_dim=vision_dim,
            output_dim=output_dim,
            dropout=dropout,
        )
        self.state_head = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
            nn.Linear(output_dim, num_findings * num_states),
        )

    def encode_fields(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_features(images)
        prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))
        patches = tokens[:, prefix_tokens:, :]
        projected = self.projector(patches)
        if self.variant == "spd":
            return projected.reshape(
                projected.shape[0], len(FIELD_NAMES), 2, projected.shape[-1]
            ).mean(dim=2)
        return torch.stack(
            [projected[name].mean(dim=1) for name in FIELD_NAMES], dim=1
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        fields = self.encode_fields(images)
        pooled = fields.mean(dim=1)
        logits = self.state_head(pooled).reshape(
            images.shape[0], self.num_findings, self.num_states
        )
        return {"logits": logits, "fields": fields}

    def trainable_counts(self) -> dict[str, int]:
        return {
            "backbone": sum(
                parameter.numel() for parameter in self.backbone.parameters()
            ),
            "projector": sum(
                parameter.numel() for parameter in self.projector.parameters()
            ),
            "state_head": sum(
                parameter.numel() for parameter in self.state_head.parameters()
            ),
        }
