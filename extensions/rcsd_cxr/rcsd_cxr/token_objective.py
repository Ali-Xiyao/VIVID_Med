"""Deterministic hard-UMS tokenization and D0/D1 loss contracts."""

from __future__ import annotations

import json
from typing import Mapping, Sequence

import torch
from torch.nn import functional as F


def finding_block_spans(target: str) -> dict[str, tuple[int, int]]:
    """Return exact character spans for each serialized finding member."""

    payload = json.loads(target)
    findings = payload.get("findings")
    if not isinstance(findings, dict):
        raise ValueError("hard UMS target requires a findings object")
    spans: dict[str, tuple[int, int]] = {}
    cursor = 0
    for finding, item in findings.items():
        needle = (
            f"{json.dumps(finding, ensure_ascii=False)}: "
            f"{json.dumps(item, ensure_ascii=False)}"
        )
        start = target.find(needle, cursor)
        if start < 0:
            raise ValueError(f"cannot locate serialized finding block: {finding}")
        end = start + len(needle)
        spans[finding] = (start, end)
        cursor = end
    return spans


def prepare_token_batch(
    tokenizer,
    *,
    prompt: str,
    targets: Sequence[str],
    finding_weights: Sequence[Mapping[str, float]],
    variant: str,
    max_length: int = 512,
) -> dict[str, torch.Tensor]:
    """Tokenize identical D0/D1 targets and attach D1 span weights."""

    if variant not in {"d0", "d1"}:
        raise ValueError("variant must be d0 or d1")
    if len(targets) != len(finding_weights):
        raise ValueError("targets and finding_weights must have equal length")
    full_texts = [prompt + target for target in targets]
    encoded = tokenizer(
        full_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")
    labels = encoded["input_ids"].clone()
    weights = torch.ones_like(labels, dtype=torch.float32)
    attention = encoded["attention_mask"]
    for row, (target, row_weights) in enumerate(
        zip(targets, finding_weights, strict=True)
    ):
        spans = finding_block_spans(target)
        for finding, weight in row_weights.items():
            if finding not in spans:
                raise ValueError(f"weight refers to absent finding: {finding}")
            if not 0.0 <= float(weight) <= 1.0:
                raise ValueError(f"invalid finding weight for {finding}: {weight}")
        for column, (start, end) in enumerate(offsets[row].tolist()):
            if attention[row, column] == 0 or end <= len(prompt):
                labels[row, column] = -100
                weights[row, column] = 0.0
                continue
            if start == end:
                labels[row, column] = -100
                weights[row, column] = 0.0
                continue
            if variant == "d1":
                target_start = start - len(prompt)
                target_end = end - len(prompt)
                for finding, (block_start, block_end) in spans.items():
                    if target_start < block_end and target_end > block_start:
                        weights[row, column] = float(row_weights[finding])
                        break
    if not (labels != -100).any():
        raise ValueError("token batch contains no target tokens")
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": attention,
        "labels": labels,
        "token_weights": weights,
    }


def token_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    token_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Next-token CE normalized by valid token weights."""

    shifted_logits = logits[:, :-1].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    losses = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(shifted_labels)
    valid = shifted_labels != -100
    if token_weights is None:
        return losses[valid].mean()
    shifted_weights = token_weights[:, 1:].to(losses)
    effective = shifted_weights * valid
    denominator = effective.sum()
    if not torch.isfinite(denominator) or denominator <= 0:
        raise ValueError("token weight denominator must be finite and positive")
    return (losses * effective).sum() / denominator


def token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    """Return correct and observed next-token counts."""

    predictions = logits[:, :-1].argmax(dim=-1)
    truth = labels[:, 1:]
    valid = truth != -100
    return int((predictions[valid] == truth[valid]).sum()), int(valid.sum())
