"""Closed-form four-state decoder for bipolar visual evidence."""

from __future__ import annotations

import torch
import torch.nn as nn


STATE_NAMES = ("support", "contradict", "uncertain", "insufficient")


class EvidenceStateDecoder(nn.Module):
    """Derive S/C/U/I probabilities from positive and negative evidence.

    This module intentionally contains no trainable four-class weight matrix.
    """

    decoder_kind = "bipolar_closed_form"
    has_flat_state_head = False

    def __init__(self, tau_a: float = 1.0, tau_d: float = 1.0, tau_p: float = 1.0) -> None:
        super().__init__()
        for name, value in (("tau_a", tau_a), ("tau_d", tau_d), ("tau_p", tau_p)):
            if float(value) <= 0:
                raise ValueError(f"{name} must be positive")
            self.register_buffer(name, torch.tensor(float(value)))

    def forward(self, evidence_pos: torch.Tensor, evidence_neg: torch.Tensor) -> dict[str, torch.Tensor]:
        if evidence_pos.shape != evidence_neg.shape:
            raise ValueError("positive and negative evidence must have identical shapes")
        total = evidence_pos + evidence_neg
        availability = 1.0 - torch.exp(-total / self.tau_a)
        decisiveness = 1.0 - torch.exp(-torch.abs(evidence_pos - evidence_neg) / self.tau_d)
        polarity = torch.sigmoid((evidence_pos - evidence_neg) / self.tau_p)

        insufficient = 1.0 - availability
        uncertain = availability * (1.0 - decisiveness)
        support = availability * decisiveness * polarity
        contradict = availability * decisiveness * (1.0 - polarity)
        probabilities = torch.stack((support, contradict, uncertain, insufficient), dim=-1)

        return {
            "state_probs": probabilities,
            "availability": availability,
            "decisiveness": decisiveness,
            "polarity": polarity,
            "total_evidence": total,
        }
