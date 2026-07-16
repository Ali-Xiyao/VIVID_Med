"""Feature-space interventions for BiVES-CXR evidence closure."""

from __future__ import annotations

import torch


def _expand_mask(mask: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3 or mask.ndim != 2 or mask.shape != tokens.shape[:2]:
        raise ValueError("tokens must be [B,P,D] and mask must be [B,P]")
    return mask.to(dtype=tokens.dtype).unsqueeze(-1)


def retain_evidence(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    replacement: torch.Tensor | None = None,
) -> torch.Tensor:
    expanded = _expand_mask(mask, tokens)
    if replacement is None:
        return expanded * tokens
    return expanded * tokens + (1.0 - expanded) * replacement.view(1, 1, -1)


def delete_evidence(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    replacement: torch.Tensor | None = None,
) -> torch.Tensor:
    expanded = _expand_mask(mask, tokens)
    if replacement is None:
        return (1.0 - expanded) * tokens
    return (1.0 - expanded) * tokens + expanded * replacement.view(1, 1, -1)


def delete_control(
    tokens: torch.Tensor,
    control_mask: torch.Tensor,
    replacement: torch.Tensor | None = None,
) -> torch.Tensor:
    return delete_evidence(tokens, control_mask, replacement)


def build_matched_control_masks(
    evidence_hard_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    topk: int,
    num_controls: int = 4,
    mode: str = "random_disjoint",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample exact-K controls that are disjoint from the evidence set."""

    if evidence_hard_mask.shape != valid_mask.shape:
        raise ValueError("evidence_hard_mask and valid_mask must share [B,P] shape")
    if mode != "random_disjoint":
        raise ValueError(f"unsupported matched-control mode: {mode}")
    if num_controls <= 0:
        raise ValueError("num_controls must be positive")
    valid_mask = valid_mask.bool()
    evidence = evidence_hard_mask.detach() > 0.5
    if bool(((evidence & valid_mask).sum(dim=-1) != topk).any()):
        raise ValueError("evidence_hard_mask must contain exactly topk valid patches")

    controls = evidence_hard_mask.new_zeros(
        (evidence_hard_mask.shape[0], num_controls, evidence_hard_mask.shape[1])
    )
    for batch_index in range(evidence_hard_mask.shape[0]):
        candidates = torch.where(valid_mask[batch_index] & ~evidence[batch_index])[0]
        if candidates.numel() < topk:
            raise ValueError(
                f"sample {batch_index} has only {candidates.numel()} disjoint control candidates "
                f"for topk={topk}"
            )
        for control_index in range(num_controls):
            order = torch.randperm(candidates.numel(), device=candidates.device, generator=generator)
            selected = candidates[order[:topk]]
            controls[batch_index, control_index, selected] = 1.0
    return controls


def build_equal_area_control_mask(
    evidence_gate: torch.Tensor,
    valid_mask: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Compatibility wrapper returning one random-disjoint control."""

    evidence_hard = evidence_gate > 0.5
    return build_matched_control_masks(
        evidence_hard.to(evidence_gate.dtype),
        valid_mask,
        topk,
        num_controls=1,
    )[:, 0]
