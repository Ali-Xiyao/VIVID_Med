"""Deterministic hard-UMS token objective."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn import functional as F


def prepare_token_batch(
    tokenizer,
    *,
    prompt: str,
    targets: Sequence[str],
    max_length: int = 512,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        [prompt + target for target in targets],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")
    labels = encoded["input_ids"].clone()
    attention = encoded["attention_mask"]
    for row in range(labels.shape[0]):
        for column, (start, end) in enumerate(offsets[row].tolist()):
            if attention[row, column] == 0 or end <= len(prompt) or start == end:
                labels[row, column] = -100
    if not (labels != -100).any():
        raise ValueError("token batch contains no target tokens")
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": attention,
        "labels": labels,
    }


def token_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    shifted_logits = logits[:, :-1].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
    )


def token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    predictions = logits[:, :-1].argmax(dim=-1)
    truth = labels[:, 1:]
    valid = truth != -100
    return int((predictions[valid] == truth[valid]).sum()), int(valid.sum())
