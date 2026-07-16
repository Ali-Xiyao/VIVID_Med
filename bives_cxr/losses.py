"""BiVES-CXR state and interventional closure losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class BiVESLossConfig:
    lambda_ies: float = 0.5
    lambda_nec: float = 1.0
    lambda_ctrl: float = 0.5
    lambda_pair: float = 0.0
    pair_margin: float = 0.2
    lambda_i_mag: float = 0.1
    lambda_min: float = 1e-3
    lambda_tv: float = 1e-4
    eps: float = 1e-8


def nll_from_probs(probabilities: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    selected = probabilities.gather(-1, targets.long().unsqueeze(-1)).squeeze(-1)
    return -torch.log(selected.clamp_min(eps))


def jensen_shannon(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    left = left.clamp_min(eps)
    right = right.clamp_min(eps)
    middle = 0.5 * (left + right)
    return 0.5 * (
        (left * (left.log() - middle.log())).sum(dim=-1)
        + (right * (right.log() - middle.log())).sum(dim=-1)
    )


def total_variation(mask: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    height, width = grid_hw
    if height * width != mask.shape[-1]:
        raise ValueError("grid height * width must equal patch count")
    grid = mask.reshape(mask.shape[0], height, width)
    horizontal = torch.abs(grid[:, :, 1:] - grid[:, :, :-1]).sum(dim=(1, 2))
    vertical = torch.abs(grid[:, 1:, :] - grid[:, :-1, :]).sum(dim=(1, 2))
    return (horizontal + vertical).mean()


class BiVESLoss(nn.Module):
    """Unified objective from the BiVES-CXR proposal."""

    def __init__(self, config: BiVESLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or BiVESLossConfig()

    def forward(
        self,
        outputs: dict[str, dict[str, torch.Tensor] | torch.Tensor],
        targets: torch.Tensor,
        grid_hw: tuple[int, int],
        support_pair_indices: torch.Tensor | None = None,
        contradict_pair_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        original = outputs["original"]
        assert isinstance(original, dict)
        state_per_sample = nll_from_probs(original["state_probs"], targets, self.config.eps)
        state_loss = state_per_sample.mean()
        total = state_loss
        losses: dict[str, torch.Tensor] = {"state": state_loss}

        answerable = targets != 3
        if all(key in outputs for key in ("keep", "drop", "control")):
            keep = outputs["keep"]
            drop = outputs["drop"]
            control = outputs["control"]
            assert isinstance(keep, dict) and isinstance(drop, dict) and isinstance(control, dict)
            if bool(answerable.any()):
                sufficiency = nll_from_probs(keep["state_probs"][answerable], targets[answerable], self.config.eps).mean()
                insufficient_targets = torch.full_like(targets[answerable], 3)
                necessity = nll_from_probs(
                    drop["state_probs"][answerable],
                    insufficient_targets,
                    self.config.eps,
                ).mean()
            else:
                sufficiency = state_loss.new_zeros(())
                necessity = state_loss.new_zeros(())
            control_loss = jensen_shannon(original["state_probs"], control["state_probs"], self.config.eps).mean()
            ies = sufficiency + self.config.lambda_nec * necessity + self.config.lambda_ctrl * control_loss
            total = total + self.config.lambda_ies * ies
            losses.update({"sufficiency": sufficiency, "necessity": necessity, "control": control_loss, "ies": ies})

        insufficient = targets == 3
        i_magnitude = original["total_evidence"][insufficient].mean() if bool(insufficient.any()) else state_loss.new_zeros(())
        minimality = original["gate"].mean()
        tv = total_variation(original["gate"], grid_hw)
        total = total + self.config.lambda_i_mag * i_magnitude
        total = total + self.config.lambda_min * minimality + self.config.lambda_tv * tv
        losses.update({"insufficient_magnitude": i_magnitude, "minimality": minimality, "tv": tv})

        if support_pair_indices is not None and contradict_pair_indices is not None:
            rho = (original["evidence_pos"] - original["evidence_neg"]) / (
                original["total_evidence"] + self.config.eps
            )
            pair = torch.relu(
                self.config.pair_margin
                - rho[support_pair_indices.long()]
                + rho[contradict_pair_indices.long()]
            ).mean()
            total = total + self.config.lambda_pair * pair
            losses["pair"] = pair

        losses["total"] = total
        return losses
