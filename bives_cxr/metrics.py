"""Mechanism-focused BiVES-CXR evaluation metrics."""

from __future__ import annotations

import torch


def intervention_metrics(
    outputs: dict[str, dict[str, torch.Tensor] | torch.Tensor],
    targets: torch.Tensor,
) -> dict[str, float]:
    original = outputs["original"]
    keep = outputs["keep"]
    drop = outputs["drop"]
    control = outputs["control"]
    assert isinstance(original, dict) and isinstance(keep, dict)
    assert isinstance(drop, dict) and isinstance(control, dict)

    answerable = targets != 3
    original_pred = original["state_probs"].argmax(dim=-1)
    keep_pred = keep["state_probs"].argmax(dim=-1)
    drop_pred = drop["state_probs"].argmax(dim=-1)
    control_pred = control["state_probs"].argmax(dim=-1)
    denominator = max(int(answerable.sum().item()), 1)
    eos = float(((keep_pred == targets) & answerable).sum().item() / denominator)
    eri = float(((drop_pred == 3) & answerable).sum().item() / denominator)
    iis = float((control_pred == original_pred).float().mean().item())
    target_change = (drop["state_probs"] - original["state_probs"]).abs().sum(dim=-1)
    control_change = (control["state_probs"] - original["state_probs"]).abs().sum(dim=-1)
    tcig = float((target_change - control_change).mean().item())
    return {"evidence_only_sufficiency": eos, "evidence_removal_insufficient": eri, "irrelevant_stability": iis, "target_control_gap": tcig}
