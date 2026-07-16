"""Mechanism-focused BiVES-CXR evaluation metrics."""

from __future__ import annotations

import torch


def intervention_metric_counts(
    outputs: dict[str, dict[str, torch.Tensor] | torch.Tensor],
    targets: torch.Tensor,
) -> dict[str, float]:
    """Return additive numerators/denominators for dataset-level aggregation."""

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
    original_correct = original_pred == targets
    eos_eligible = original_correct
    eri_eligible = original_correct & answerable
    target_change = (drop["state_probs"] - original["state_probs"]).abs().sum(dim=-1)
    control_change = (control["state_probs"] - original["state_probs"]).abs().sum(dim=-1)
    return {
        "eos_numerator": float(((keep_pred == targets) & eos_eligible).sum().item()),
        "eos_denominator": float(eos_eligible.sum().item()),
        "eri_numerator": float(((drop_pred == 3) & eri_eligible).sum().item()),
        "eri_denominator": float(eri_eligible.sum().item()),
        "iis_numerator": float((control_pred == original_pred).sum().item()),
        "iis_denominator": float(targets.numel()),
        "tcig_sum": float((target_change - control_change).sum().item()),
        "tcig_denominator": float(targets.numel()),
    }


def finalize_intervention_metrics(counts: dict[str, float]) -> dict[str, float]:
    def ratio(numerator: str, denominator: str) -> float:
        value = counts.get(denominator, 0.0)
        return float(counts.get(numerator, 0.0) / value) if value > 0 else float("nan")

    return {
        "evidence_only_sufficiency": ratio("eos_numerator", "eos_denominator"),
        "evidence_removal_insufficient": ratio("eri_numerator", "eri_denominator"),
        "irrelevant_stability": ratio("iis_numerator", "iis_denominator"),
        "target_control_gap": ratio("tcig_sum", "tcig_denominator"),
    }


def intervention_metrics(
    outputs: dict[str, dict[str, torch.Tensor] | torch.Tensor],
    targets: torch.Tensor,
) -> dict[str, float]:
    return finalize_intervention_metrics(intervention_metric_counts(outputs, targets))
