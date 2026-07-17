"""Feature-space interventions for BiVES-CXR evidence closure."""

from __future__ import annotations

import hashlib

import torch


CONTROL_PROTOCOL_VERSION = "random_disjoint_v1"


def stable_control_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def bives_control_seed(
    *,
    split: str,
    sample_id: str,
    training_seed: int,
    evaluation_control_seed: int,
    epoch: int = 0,
    protocol_version: str = CONTROL_PROTOCOL_VERSION,
) -> int:
    """Separate train-time control randomness from locked evaluation controls."""

    if split == "train":
        return stable_control_seed(
            protocol_version,
            "train",
            int(training_seed),
            int(epoch),
            sample_id,
        )
    return stable_control_seed(
        protocol_version,
        split,
        int(evaluation_control_seed),
        sample_id,
    )


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
    sample_seeds: torch.Tensor | list[int] | tuple[int, ...] | None = None,
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
    if sample_seeds is not None:
        sample_seeds = [int(value) for value in torch.as_tensor(sample_seeds).cpu().tolist()]
        if len(sample_seeds) != evidence_hard_mask.shape[0]:
            raise ValueError("sample_seeds must contain one seed per batch row")
    for batch_index in range(evidence_hard_mask.shape[0]):
        candidates = torch.where(valid_mask[batch_index] & ~evidence[batch_index])[0]
        if candidates.numel() < topk:
            raise ValueError(
                f"sample {batch_index} has only {candidates.numel()} disjoint control candidates "
                f"for topk={topk}"
            )
        for control_index in range(num_controls):
            row_generator = generator
            if sample_seeds is not None:
                row_generator = torch.Generator(device=candidates.device)
                row_generator.manual_seed(
                    stable_control_seed(sample_seeds[batch_index], control_index)
                )
            order = torch.randperm(
                candidates.numel(),
                device=candidates.device,
                generator=row_generator,
            )
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
