"""Hard-Negative Memory Bank (HNMB)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class MemoryEntry:
    sample_id: str
    embedding: torch.Tensor
    metadata: dict[str, Any]


class HardNegativeMemoryBank:
    """In-memory embedding bank for nearest opposite-state negatives."""

    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.entries: list[MemoryEntry] = []

    def clear(self) -> None:
        self.entries.clear()

    def add(self, sample_id: str, embedding: torch.Tensor, **metadata: Any) -> None:
        vector = embedding.detach().float().cpu().flatten()
        if self.normalize:
            vector = F.normalize(vector, dim=0)
        self.entries.append(MemoryEntry(str(sample_id), vector, dict(metadata)))

    def add_batch(self, sample_ids: list[str], embeddings: torch.Tensor, metadata: list[dict[str, Any]] | None = None) -> None:
        metadata = metadata or [{} for _ in sample_ids]
        for sample_id, embedding, meta in zip(sample_ids, embeddings, metadata):
            self.add(sample_id, embedding, **meta)

    def mine(self, embedding: torch.Tensor, sample_id: str | None = None, top_k: int = 4, **filters: Any) -> list[dict[str, Any]]:
        if not self.entries:
            return []
        query = embedding.detach().float().cpu().flatten()
        if self.normalize:
            query = F.normalize(query, dim=0)
        scored = []
        for entry in self.entries:
            if sample_id is not None and entry.sample_id == str(sample_id):
                continue
            if not self._compatible(entry.metadata, filters):
                continue
            score = float(torch.dot(query, entry.embedding))
            scored.append((score, entry))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {"sample_id": entry.sample_id, "score": score, **entry.metadata}
            for score, entry in scored[:top_k]
        ]

    @staticmethod
    def _compatible(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        finding = filters.get("finding")
        if finding and metadata.get("finding") and metadata.get("finding") != finding:
            return False
        state = filters.get("state")
        if state and metadata.get("state") and metadata.get("state") == state:
            return False
        answer = filters.get("answer")
        if answer and metadata.get("answer") and metadata.get("answer") == answer:
            return False
        laterality = filters.get("laterality")
        if laterality and metadata.get("laterality") and metadata.get("laterality") == laterality:
            return False
        return True


def margin_loss(pos_nll: torch.Tensor, neg_nll: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    return torch.relu(margin + pos_nll - neg_nll).mean()

