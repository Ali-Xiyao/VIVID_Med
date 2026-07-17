"""Archived absolute-exponential BiVES decoder for ablation reproduction only.

This formula has a polarity-inverted support stationary point at
``delta=-asinh(1)`` when ``tau_d=tau_p=1``. It must not be imported by the
active BiVES implementation.
"""

from __future__ import annotations

import torch


def legacy_abs_exp_probabilities(
    evidence_pos: torch.Tensor,
    evidence_neg: torch.Tensor,
    *,
    tau_a: float = 1.0,
    tau_d: float = 1.0,
    tau_p: float = 1.0,
) -> torch.Tensor:
    total = evidence_pos + evidence_neg
    delta = evidence_pos - evidence_neg
    availability = 1.0 - torch.exp(-total / tau_a)
    decisiveness = 1.0 - torch.exp(-delta.abs() / tau_d)
    polarity = torch.sigmoid(delta / tau_p)
    return torch.stack(
        (
            availability * decisiveness * polarity,
            availability * decisiveness * (1.0 - polarity),
            availability * (1.0 - decisiveness),
            1.0 - availability,
        ),
        dim=-1,
    )
