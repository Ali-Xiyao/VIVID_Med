"""Feature-space interventions for BiVES-CXR evidence closure."""

from __future__ import annotations

import torch


def _expand_mask(mask: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3 or mask.ndim != 2 or mask.shape != tokens.shape[:2]:
        raise ValueError("tokens must be [B,P,D] and mask must be [B,P]")
    return mask.to(dtype=tokens.dtype).unsqueeze(-1)


def retain_evidence(tokens: torch.Tensor, mask: torch.Tensor, mask_token: torch.Tensor) -> torch.Tensor:
    expanded = _expand_mask(mask, tokens)
    return expanded * tokens + (1.0 - expanded) * mask_token.view(1, 1, -1)


def delete_evidence(tokens: torch.Tensor, mask: torch.Tensor, mask_token: torch.Tensor) -> torch.Tensor:
    expanded = _expand_mask(mask, tokens)
    return (1.0 - expanded) * tokens + expanded * mask_token.view(1, 1, -1)


def delete_control(tokens: torch.Tensor, control_mask: torch.Tensor, mask_token: torch.Tensor) -> torch.Tensor:
    return delete_evidence(tokens, control_mask, mask_token)


def build_equal_area_control_mask(
    evidence_gate: torch.Tensor,
    valid_mask: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Select low-evidence valid patches as a disjoint equal-area control."""
    if evidence_gate.shape != valid_mask.shape:
        raise ValueError("evidence_gate and valid_mask must share [B,P] shape")
    valid_mask = valid_mask.bool()
    control = torch.zeros_like(evidence_gate)
    for batch_index in range(evidence_gate.shape[0]):
        valid_indices = torch.where(valid_mask[batch_index])[0]
        if valid_indices.numel() == 0:
            continue
        k = min(int(topk), int(valid_indices.numel() // 2 or 1))
        values = evidence_gate[batch_index, valid_indices].detach()
        evidence_indices = valid_indices[torch.topk(values, k=k).indices]
        candidate_mask = torch.ones(valid_indices.numel(), dtype=torch.bool, device=valid_indices.device)
        for selected in evidence_indices:
            candidate_mask &= valid_indices != selected
        candidates = valid_indices[candidate_mask]
        if candidates.numel() == 0:
            continue
        k_control = min(k, int(candidates.numel()))
        candidate_values = evidence_gate[batch_index, candidates].detach()
        selected_control = candidates[torch.topk(-candidate_values, k=k_control).indices]
        control[batch_index, selected_control] = 1.0
    return control
