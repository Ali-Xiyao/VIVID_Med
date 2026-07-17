"""Calibration utilities for the closed-form BiVES evidence decoder."""

from __future__ import annotations

import torch


def probabilities_from_evidence(
    evidence_pos: torch.Tensor,
    evidence_neg: torch.Tensor,
    tau_a: torch.Tensor,
    tau_p: torch.Tensor,
    uncertainty_mass: torch.Tensor,
) -> torch.Tensor:
    total = evidence_pos + evidence_neg
    delta = evidence_pos - evidence_neg
    availability = 1.0 - torch.exp(-total / tau_a)
    signed_logit = delta / (2.0 * tau_p)
    uncertain_logit = torch.log(2.0 * uncertainty_mass).expand_as(signed_logit)
    conditional = torch.softmax(
        torch.stack((signed_logit, -signed_logit, uncertain_logit), dim=-1),
        dim=-1,
    )
    return torch.stack(
        (
            availability * conditional[..., 0],
            availability * conditional[..., 1],
            availability * conditional[..., 2],
            1.0 - availability,
        ),
        dim=-1,
    )


def fit_decoder_parameters(
    evidence_pos: torch.Tensor,
    evidence_neg: torch.Tensor,
    targets: torch.Tensor,
    initial: tuple[float, float, float] = (1.0, 1.0, 1.0),
    max_iter: int = 100,
) -> dict[str, float]:
    """Fit positive monotone-decoder parameters on a locked calibration split."""

    if evidence_pos.shape != evidence_neg.shape or evidence_pos.ndim != 1:
        raise ValueError("evidence_pos and evidence_neg must be aligned one-dimensional tensors")
    if targets.shape != evidence_pos.shape:
        raise ValueError("targets must align with calibration evidence")
    if evidence_pos.numel() == 0:
        raise ValueError("decoder parameter fitting requires calibration samples")
    if any(value <= 0 for value in initial):
        raise ValueError("initial decoder parameters must be positive")

    positive = evidence_pos.detach().double().cpu()
    negative = evidence_neg.detach().double().cpu()
    labels = targets.detach().long().cpu()
    log_parameters = torch.tensor(
        [float(value) for value in initial],
        dtype=torch.float64,
    ).log().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [log_parameters],
        lr=0.25,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        parameters = log_parameters.exp().clamp(1e-4, 1e4)
        probabilities = probabilities_from_evidence(
            positive,
            negative,
            parameters[0],
            parameters[1],
            parameters[2],
        )
        selected = probabilities.gather(1, labels[:, None]).squeeze(1)
        loss = -selected.clamp_min(1e-12).log().mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    fitted = log_parameters.detach().exp().clamp(1e-4, 1e4)
    return {
        "tau_a": float(fitted[0]),
        "tau_p": float(fitted[1]),
        "uncertainty_mass": float(fitted[2]),
    }
