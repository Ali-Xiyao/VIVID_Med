"""Small morphology-specific concept heads for the preregistered gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


EXPERT_TYPES = ("generic", "region", "boundary", "geometry", "distribution")
FINDING_TO_EXPERT = {
    "pneumothorax": "boundary",
    "consolidation": "region",
    "pleural_effusion": "region_boundary",
    "cardiomegaly": "geometry",
}


@dataclass(frozen=True)
class MorphologyExpertConfig:
    visual_dim: int
    num_statements: int
    expert_type: str
    temperature: float = 0.5

    def validate(self) -> None:
        if self.visual_dim <= 0 or self.num_statements <= 0:
            raise ValueError("MORPH expert dimensions must be positive")
        if self.expert_type not in EXPERT_TYPES:
            raise ValueError(f"unknown MORPH expert: {self.expert_type}")
        if self.temperature <= 0:
            raise ValueError("MORPH expert temperature must be positive")

    def to_dict(self) -> dict[str, int | float | str]:
        self.validate()
        return asdict(self)


def _grid_coordinates(
    batch: int, grid_hw: tuple[int, int], device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = map(int, grid_hw)
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return xx.reshape(1, -1).expand(batch, -1), yy.reshape(1, -1).expand(batch, -1)


def _local_mean(tokens: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    batch, patches, dim = tokens.shape
    height, width = map(int, grid_hw)
    if patches != height * width:
        raise ValueError("MORPH token/grid geometry changed")
    feature_map = tokens.transpose(1, 2).reshape(batch, dim, height, width)
    smoothed = F.avg_pool2d(feature_map, kernel_size=3, stride=1, padding=1)
    return smoothed.flatten(2).transpose(1, 2)


def _boundary_energy(tokens: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    batch, patches, dim = tokens.shape
    height, width = map(int, grid_hw)
    feature_map = tokens.reshape(batch, height, width, dim)
    dx = torch.zeros((batch, height, width), device=tokens.device, dtype=tokens.dtype)
    dy = torch.zeros_like(dx)
    dx[:, :, 1:] = (feature_map[:, :, 1:] - feature_map[:, :, :-1]).square().mean(-1)
    dy[:, 1:, :] = (feature_map[:, 1:] - feature_map[:, :-1]).square().mean(-1)
    energy = torch.sqrt((dx + dy).clamp_min(1e-12)).reshape(batch, patches)
    scale = energy.mean(1, keepdim=True).clamp_min(1e-6)
    return energy / scale


class MorphologyConceptExpert(nn.Module):
    """Concept-only head with expert-specific spatial evidence construction."""

    has_flat_four_class_head = False
    concept_names = (
        "presence",
        "mean_evidence",
        "extent",
        "boundary_strength",
        "horizontal_extent",
        "vertical_extent",
        "centrality",
    )

    def __init__(self, config: MorphologyExpertConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.patch_weight = nn.Embedding(config.num_statements, config.visual_dim)
        self.patch_bias = nn.Embedding(config.num_statements, 1)
        self.coordinate_weight = nn.Embedding(config.num_statements, 5)
        self.boundary_weight = nn.Embedding(config.num_statements, 1)
        self.concept_weight_raw = nn.Embedding(
            config.num_statements, len(self.concept_names)
        )
        self.concept_bias = nn.Embedding(config.num_statements, 1)
        nn.init.normal_(self.patch_weight.weight, std=0.02)
        nn.init.zeros_(self.patch_bias.weight)
        nn.init.zeros_(self.coordinate_weight.weight)
        nn.init.zeros_(self.boundary_weight.weight)
        nn.init.constant_(self.concept_weight_raw.weight, -1.0)
        nn.init.constant_(self.concept_bias.weight, -1.0)

    def _active_concepts(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mapping = {
            "generic": (1, 1, 1, 0, 0, 0, 0),
            "region": (1, 1, 1, 0, 1, 1, 0),
            "boundary": (1, 1, 1, 1, 0, 0, 0),
            "geometry": (1, 0, 1, 0, 1, 1, 1),
            "distribution": (1, 1, 1, 0, 1, 1, 1),
        }
        return torch.tensor(mapping[self.config.expert_type], device=device, dtype=dtype)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        statement_indices: torch.Tensor,
        *,
        grid_hw: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        if patch_tokens.ndim != 3 or valid_mask.shape != patch_tokens.shape[:2]:
            raise ValueError("MORPH patch tokens/mask must be [B,P,D]/[B,P]")
        if patch_tokens.shape[-1] != self.config.visual_dim:
            raise ValueError("MORPH visual dimension changed")
        if statement_indices.shape != (patch_tokens.shape[0],):
            raise ValueError("MORPH statement indices must be [B]")
        if patch_tokens.shape[1] != int(grid_hw[0]) * int(grid_hw[1]):
            raise ValueError("MORPH grid is inconsistent with patch count")
        valid = valid_mask.bool()
        if not bool(valid.any(1).all()):
            raise ValueError("every MORPH sample needs valid patches")

        normalized = F.layer_norm(patch_tokens.float(), (self.config.visual_dim,))
        expert_tokens = normalized
        if self.config.expert_type == "region":
            expert_tokens = _local_mean(normalized, grid_hw)
        boundary = _boundary_energy(normalized, grid_hw)
        weight = self.patch_weight(statement_indices)
        logits = torch.einsum("bpd,bd->bp", expert_tokens, weight)
        logits = logits / (self.config.visual_dim**0.5)
        logits = logits + self.patch_bias(statement_indices).squeeze(-1).unsqueeze(1)

        xx, yy = _grid_coordinates(
            patch_tokens.shape[0], grid_hw, patch_tokens.device, logits.dtype
        )
        if self.config.expert_type in ("geometry", "distribution"):
            coords = torch.stack((xx, yy, xx.square(), yy.square(), xx * yy), dim=-1)
            logits = logits + torch.einsum(
                "bpc,bc->bp", coords, self.coordinate_weight(statement_indices)
            )
        if self.config.expert_type == "boundary":
            scale = F.softplus(self.boundary_weight(statement_indices).squeeze(-1))
            logits = logits + scale.unsqueeze(1) * boundary

        masked_logits = logits.masked_fill(~valid, float("-inf"))
        attention = torch.sigmoid(logits / self.config.temperature) * valid.to(logits.dtype)
        mass = attention.sum(1).clamp_min(1e-6)
        counts = valid.sum(1).to(logits.dtype)
        peak = self.config.temperature * (
            torch.logsumexp(masked_logits / self.config.temperature, dim=1)
            - torch.log(counts)
        )
        mean = (logits * valid.to(logits.dtype)).sum(1) / counts
        cx = (attention * xx).sum(1) / mass
        cy = (attention * yy).sum(1) / mass
        sx = torch.sqrt(((xx - cx[:, None]).square() * attention).sum(1) / mass + 1e-6)
        sy = torch.sqrt(((yy - cy[:, None]).square() * attention).sum(1) / mass + 1e-6)
        concepts = torch.stack(
            (
                F.softplus(peak),
                F.softplus(mean),
                mass / counts,
                (attention * boundary).sum(1) / mass,
                sx,
                sy,
                (1.0 - 0.5 * (cx.abs() + cy.abs())).clamp(0.0, 1.0),
            ),
            dim=1,
        )
        active = self._active_concepts(concepts.device, concepts.dtype)
        weights = F.softplus(self.concept_weight_raw(statement_indices)) * active
        margin = (weights * concepts).sum(1) + self.concept_bias(statement_indices).squeeze(-1)
        return {
            "margin": margin,
            "support_probability": torch.sigmoid(margin),
            "patch_logits": logits,
            "attention": attention,
            "concepts": concepts,
            "concept_weights": weights,
            "valid_mask": valid,
            "boundary_energy": boundary,
            "centroid": torch.stack((cx, cy), dim=1),
            "spread": torch.stack((sx, sy), dim=1),
        }


def concept_monotonicity_deltas(
    concepts: torch.Tensor,
    concept_weights: torch.Tensor,
) -> torch.Tensor:
    """Margin decrease when each active concept is set to zero."""

    if concepts.shape != concept_weights.shape or concepts.ndim != 2:
        raise ValueError("MORPH concepts/weights must have equal [B,C] shape")
    return concepts * concept_weights
