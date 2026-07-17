"""Cached-token bipolar polarity models for the expert S/C route."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .gates import EvidenceGate


@dataclass(frozen=True)
class PolarityModelConfig:
    visual_dim: int
    num_statements: int
    statement_dim: int = 128
    fusion_dim: int = 256
    evidence_max: float = 8.0
    mode: str = "dense"
    topk: int = 16
    gate_temperature: float = 0.5
    tau_p: float = 1.0
    contextual_layers: int = 1
    contextual_heads: int = 4
    contextual_dropout: float = 0.0


class BipolarPolarityModel(nn.Module):
    """Statement-conditioned bipolar field with dense or exact-K aggregation."""

    has_flat_binary_head = False

    def __init__(self, config: PolarityModelConfig) -> None:
        super().__init__()
        if config.mode not in {"dense", "sparse_exact_k"}:
            raise ValueError("polarity mode must be dense or sparse_exact_k")
        if config.tau_p <= 0 or config.evidence_max <= 0:
            raise ValueError("tau_p and evidence_max must be positive")
        if config.fusion_dim % config.contextual_heads:
            raise ValueError("contextual_heads must divide fusion_dim")
        self.config = config
        self.statement_table = nn.Embedding(config.num_statements, config.statement_dim)
        self.visual_projection = nn.Linear(config.visual_dim, config.fusion_dim)
        self.statement_projection = nn.Linear(config.statement_dim, config.fusion_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(config.fusion_dim * 4),
            nn.Linear(config.fusion_dim * 4, config.fusion_dim),
            nn.GELU(),
            nn.LayerNorm(config.fusion_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.fusion_dim,
            nhead=config.contextual_heads,
            dim_feedforward=config.fusion_dim * 4,
            dropout=config.contextual_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.contextual_evidence = nn.TransformerEncoder(
            layer,
            num_layers=config.contextual_layers,
            enable_nested_tensor=False,
        )
        self.contextual_norm = nn.LayerNorm(config.fusion_dim)
        self.evidence_head = nn.Linear(config.fusion_dim, 2)
        if config.mode == "sparse_exact_k":
            self.gate_head: nn.Linear | None = nn.Linear(config.fusion_dim, 1)
            self.gate: EvidenceGate | None = EvidenceGate(
                mode="soft_topk",
                topk=config.topk,
                temperature=config.gate_temperature,
            )
        else:
            self.gate_head = None
            self.gate = None

    def forward(
        self,
        patch_tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        statement_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if patch_tokens.ndim != 3 or valid_mask.shape != patch_tokens.shape[:2]:
            raise ValueError("patch_tokens/valid_mask must be [B,P,D]/[B,P]")
        visual = self.visual_projection(patch_tokens.float())
        statement = self.statement_projection(
            self.statement_table(statement_indices)
        ).unsqueeze(1).expand_as(visual)
        fused = self.fusion(
            torch.cat(
                (visual, statement, visual * statement, torch.abs(visual - statement)),
                dim=-1,
            )
        )
        valid_float = valid_mask.to(fused.dtype).unsqueeze(-1)
        fused = fused * valid_float
        contextual = self.contextual_evidence(
            fused,
            src_key_padding_mask=~valid_mask.bool(),
        )
        contextual = self.contextual_norm(contextual) * valid_float
        evidence_pm = torch.sigmoid(self.evidence_head(contextual)) * self.config.evidence_max
        if self.config.mode == "dense":
            gate_logits = torch.zeros_like(valid_mask, dtype=contextual.dtype)
            gate = valid_mask.to(contextual.dtype)
        else:
            assert self.gate_head is not None and self.gate is not None
            gate_logits = self.gate_head(contextual).squeeze(-1)
            gate = self.gate(gate_logits, valid_mask)
        denominator = gate.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        evidence = (evidence_pm * gate.unsqueeze(-1)).sum(dim=1) / denominator
        signed_evidence = evidence[:, 0] - evidence[:, 1]
        support_probability = torch.sigmoid(signed_evidence / self.config.tau_p)
        return {
            "evidence_pos": evidence[:, 0],
            "evidence_neg": evidence[:, 1],
            "signed_evidence": signed_evidence,
            "support_probability": support_probability,
            "evidence_pm": evidence_pm,
            "gate": gate,
            "gate_logits": gate_logits,
            "valid_mask": valid_mask,
        }


def polarity_loss(
    signed_evidence: torch.Tensor,
    binary_labels: torch.Tensor,
    tau_p: float = 1.0,
) -> torch.Tensor:
    if tau_p <= 0:
        raise ValueError("tau_p must be positive")
    signs = binary_labels.to(signed_evidence.dtype) * 2.0 - 1.0
    return F.softplus(-signs * signed_evidence / tau_p).mean()


class CachedSCDataset(Dataset):
    """Read a locked S/C index and its immutable per-image token cache."""

    def __init__(self, cache_dir: str | Path, split: str) -> None:
        self.cache_dir = Path(cache_dir)
        self.lock = json.loads((self.cache_dir / "cache_lock.json").read_text(encoding="utf-8"))
        if self.lock.get("status") != "complete":
            raise ValueError("patch-token cache is not complete")
        index_path = self.cache_dir / f"{split}_index.jsonl"
        self.rows = [
            json.loads(line)
            for line in index_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.rows:
            raise ValueError(f"cached S/C {split} index is empty")
        statements = sorted({str(row["canonical_statement_id"]) for row in self.rows})
        self.statement_to_index = {statement: index for index, statement in enumerate(statements)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        payload = torch.load(
            self.cache_dir / row["cache_file"],
            map_location="cpu",
            weights_only=False,
        )
        if payload.get("identity") != {
            key: self.lock[key]
            for key in (
                "format_version",
                "manifest_sha256",
                "model_snapshot_sha256",
                "processor_snapshot_sha256",
                "image_preprocess_version",
                "image_size",
                "dtype",
            )
        }:
            raise ValueError(f"cached token identity mismatch: {row['cache_file']}")
        if str(payload["image_sha256"]) != str(row["image_sha256"]):
            raise ValueError(f"cached image hash mismatch: {row['sample_id']}")
        return {
            "sample_id": str(row["sample_id"]),
            "unit_id": str(row["unit_id"]),
            "patient_id": row.get("patient_id"),
            "canonical_statement_id": str(row["canonical_statement_id"]),
            "statement_index": self.statement_to_index[str(row["canonical_statement_id"])],
            "binary_label": int(row["binary_label"]),
            "patch_tokens": payload["patch_tokens"].float(),
            "valid_mask": payload["valid_mask"].bool(),
            "grid_hw": tuple(payload["grid_hw"]),
        }


def collate_cached_sc(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_patches = max(item["patch_tokens"].shape[0] for item in batch)
    visual_dim = int(batch[0]["patch_tokens"].shape[1])
    tokens = torch.zeros((len(batch), max_patches, visual_dim), dtype=torch.float32)
    valid = torch.zeros((len(batch), max_patches), dtype=torch.bool)
    for index, item in enumerate(batch):
        patches = item["patch_tokens"].shape[0]
        tokens[index, :patches] = item["patch_tokens"]
        valid[index, :patches] = item["valid_mask"]
    return {
        "sample_ids": [item["sample_id"] for item in batch],
        "unit_ids": [item["unit_id"] for item in batch],
        "patient_ids": [item["patient_id"] for item in batch],
        "canonical_statement_ids": [item["canonical_statement_id"] for item in batch],
        "statement_indices": torch.tensor([item["statement_index"] for item in batch]),
        "binary_labels": torch.tensor([item["binary_label"] for item in batch]),
        "patch_tokens": tokens,
        "valid_mask": valid,
        "grid_hw": [item["grid_hw"] for item in batch],
    }
