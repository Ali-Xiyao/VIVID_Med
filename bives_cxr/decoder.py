"""Monotone closed-form four-state decoder for bipolar visual evidence."""

from __future__ import annotations

import torch
import torch.nn as nn


STATE_NAMES = ("support", "contradict", "uncertain", "insufficient")
DECODER_PARAMETER_NAMES = ("tau_a", "tau_p", "uncertainty_mass")


class EvidenceStateDecoder(nn.Module):
    """Derive S/C/U/I probabilities from positive and negative evidence.

    This module intentionally contains no trainable four-class weight matrix.
    """

    decoder_kind = "monotone_bipolar_conditional"
    has_flat_state_head = False

    def __init__(
        self,
        tau_a: float = 1.0,
        tau_p: float = 1.0,
        uncertainty_mass: float = 1.0,
    ) -> None:
        super().__init__()
        for name, value in (
            ("tau_a", tau_a),
            ("tau_p", tau_p),
            ("uncertainty_mass", uncertainty_mass),
        ):
            if float(value) <= 0:
                raise ValueError(f"{name} must be positive")
            self.register_buffer(name, torch.tensor(float(value)))

    def forward(self, evidence_pos: torch.Tensor, evidence_neg: torch.Tensor) -> dict[str, torch.Tensor]:
        if evidence_pos.shape != evidence_neg.shape:
            raise ValueError("positive and negative evidence must have identical shapes")
        total = evidence_pos + evidence_neg
        delta = evidence_pos - evidence_neg
        availability = 1.0 - torch.exp(-total / self.tau_a)

        signed_logit = delta / (2.0 * self.tau_p)
        uncertain_logit = torch.log(2.0 * self.uncertainty_mass).expand_as(signed_logit)
        conditional = torch.softmax(
            torch.stack((signed_logit, -signed_logit, uncertain_logit), dim=-1),
            dim=-1,
        )
        conditional_support, conditional_contradict, conditional_uncertain = conditional.unbind(-1)
        decisiveness = conditional_support + conditional_contradict
        polarity = conditional_support / decisiveness.clamp_min(torch.finfo(conditional.dtype).tiny)

        insufficient = 1.0 - availability
        uncertain = availability * conditional_uncertain
        support = availability * conditional_support
        contradict = availability * conditional_contradict
        probabilities = torch.stack((support, contradict, uncertain, insufficient), dim=-1)

        return {
            "state_probs": probabilities,
            "availability": availability,
            "decisiveness": decisiveness,
            "polarity": polarity,
            "total_evidence": total,
            "signed_evidence": delta,
        }
