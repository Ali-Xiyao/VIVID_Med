"""Clinical Evidence Query (CEQ) module.

CEQ maps image patch tokens to finding-specific evidence embeddings through
learned clinical queries and cross attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ClinicalEvidenceQuery(nn.Module):
    """Finding-specific cross-attention over visual patch tokens."""

    def __init__(self, num_findings: int, embed_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_findings = num_findings
        self.embed_dim = embed_dim
        self.clinical_queries = nn.Parameter(torch.randn(num_findings, embed_dim) * 0.02)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.output = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim))

    def forward(self, patch_tokens: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Return evidence embeddings and attention weights.

        Args:
            patch_tokens: Tensor shaped [batch, patches, dim].
            key_padding_mask: Optional bool mask shaped [batch, patches].
        """
        if patch_tokens.ndim != 3:
            raise ValueError("patch_tokens must have shape [batch, patches, dim]")
        batch = patch_tokens.shape[0]
        queries = self.clinical_queries.unsqueeze(0).expand(batch, -1, -1)
        attended, weights = self.attention(queries, patch_tokens, patch_tokens, key_padding_mask=key_padding_mask, need_weights=True)
        evidence = self.output(self.norm(attended + queries))
        return {"evidence": evidence, "attention": weights}


class ClinicalEvidenceClassifier(nn.Module):
    """Small per-finding state classifier on top of CEQ embeddings."""

    def __init__(self, embed_dim: int, num_states: int = 3):
        super().__init__()
        self.classifier = nn.Linear(embed_dim, num_states)

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        return self.classifier(evidence)

