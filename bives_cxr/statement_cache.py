"""Provenance-complete frozen statement embedding caches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .data import normalize_statement_text


CACHE_FORMAT_VERSION = 1
UNLOCKED_EXPECTATION_VALUES = {
    "",
    "LOCK_AFTER_BUILD",
    "TO_BE_FILLED_BY_BUILD_BIVES_STATEMENT_EMBEDDINGS",
}
REQUIRED_ENCODER_FIELDS = {
    "model_name_or_path",
    "revision",
    "tokenizer_revision",
    "tokenizer_class",
    "pooling",
    "normalize",
    "dtype",
}


def text_sha256(text: str) -> str:
    return hashlib.sha256(normalize_statement_text(text).encode("utf-8")).hexdigest()


def vocabulary_sha256(text_by_id: dict[str, str]) -> str:
    canonical = [
        [statement_id, normalize_statement_text(text_by_id[statement_id])]
        for statement_id in sorted(text_by_id)
    ]
    return hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_statement_cache_payload(
    embeddings: dict[str, torch.Tensor],
    text_by_id: dict[str, str],
    encoder: dict[str, Any],
    *,
    sources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    missing_encoder = REQUIRED_ENCODER_FIELDS - set(encoder)
    if missing_encoder:
        raise ValueError(f"encoder provenance is missing fields: {sorted(missing_encoder)}")
    if set(embeddings) != set(text_by_id):
        raise ValueError("embedding IDs and ontology text IDs must match exactly")
    normalized = {
        statement_id: normalize_statement_text(text)
        for statement_id, text in text_by_id.items()
    }
    dimensions = {
        int(torch.as_tensor(embedding).reshape(-1).numel())
        for embedding in embeddings.values()
    }
    if len(dimensions) != 1:
        raise ValueError(f"statement embedding dimensions are inconsistent: {sorted(dimensions)}")
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "encoder": dict(encoder),
        "ontology": {
            "text_by_id": normalized,
            "text_sha256_by_id": {
                statement_id: text_sha256(text)
                for statement_id, text in normalized.items()
            },
            "vocabulary_sha256": vocabulary_sha256(normalized),
        },
        "embedding_dim": next(iter(dimensions)),
        "embeddings": {
            statement_id: torch.as_tensor(embedding).detach().cpu().reshape(-1)
            for statement_id, embedding in embeddings.items()
        },
        "sources": dict(sources or {}),
    }


def validate_statement_cache(
    payload: Any,
    expected_text_by_id: dict[str, str],
) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict) or payload.get("format_version") != CACHE_FORMAT_VERSION:
        raise ValueError(
            f"statement cache must use format_version={CACHE_FORMAT_VERSION}"
        )
    encoder = payload.get("encoder")
    if not isinstance(encoder, dict):
        raise ValueError("statement cache is missing encoder provenance")
    missing_encoder = REQUIRED_ENCODER_FIELDS - set(encoder)
    if missing_encoder:
        raise ValueError(f"encoder provenance is missing fields: {sorted(missing_encoder)}")
    ontology = payload.get("ontology")
    embeddings = payload.get("embeddings")
    if not isinstance(ontology, dict) or not isinstance(embeddings, dict):
        raise ValueError("statement cache must contain ontology and embeddings mappings")
    expected = {
        statement_id: normalize_statement_text(text)
        for statement_id, text in expected_text_by_id.items()
    }
    cached_texts = ontology.get("text_by_id")
    cached_hashes = ontology.get("text_sha256_by_id")
    if cached_texts != expected:
        raise ValueError("statement cache ontology text does not match the locked manifests")
    expected_hashes = {
        statement_id: text_sha256(text) for statement_id, text in expected.items()
    }
    if cached_hashes != expected_hashes:
        raise ValueError("statement cache per-statement text hashes are invalid")
    if ontology.get("vocabulary_sha256") != vocabulary_sha256(expected):
        raise ValueError("statement cache vocabulary_sha256 is invalid")
    if set(embeddings) != set(expected):
        raise ValueError("statement cache embedding IDs do not match the locked vocabulary")
    rows = {
        statement_id: torch.as_tensor(embeddings[statement_id]).float().reshape(-1)
        for statement_id in expected
    }
    dimensions = {int(row.numel()) for row in rows.values()}
    if len(dimensions) != 1 or next(iter(dimensions)) != int(payload.get("embedding_dim", -1)):
        raise ValueError("statement cache embedding_dim is inconsistent")
    if any(not bool(torch.isfinite(row).all()) for row in rows.values()):
        raise ValueError("statement cache contains non-finite embeddings")
    if bool(encoder["normalize"]):
        for statement_id, row in rows.items():
            if not torch.isclose(row.norm(), torch.tensor(1.0), atol=1e-4, rtol=1e-4):
                raise ValueError(
                    f"normalized statement embedding {statement_id!r} is not unit norm"
                )
    return rows


def validate_ontology_subset(
    locked_text_by_id: dict[str, str],
    subset_text_by_id: dict[str, str],
) -> None:
    locked = {
        statement_id: normalize_statement_text(text)
        for statement_id, text in locked_text_by_id.items()
    }
    subset = {
        statement_id: normalize_statement_text(text)
        for statement_id, text in subset_text_by_id.items()
    }
    inconsistent = {
        statement_id: text
        for statement_id, text in subset.items()
        if locked.get(statement_id) != text
    }
    if inconsistent:
        raise ValueError(
            "statement subset contains unseen or inconsistent statements: "
            f"{list(inconsistent.items())[:5]}"
        )


def load_statement_embedding_matrix(
    path: str | Path,
    statement_to_index: dict[str, int],
    expected_text_by_id: dict[str, str],
    expected_cache: dict[str, Any] | None = None,
) -> torch.Tensor:
    cache_path = Path(path)
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"formal BiVES config requires frozen Qwen3.5 statement embeddings: {cache_path}"
        )
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if expected_cache is not None:
        required = {
            "expected_sha256",
            "expected_vocabulary_sha256",
            "expected_pooling",
        }
        missing = required - set(expected_cache)
        if missing:
            raise ValueError(
                f"statement cache expectations are missing: {sorted(missing)}"
            )
        for key in required:
            if str(expected_cache[key]).strip() in UNLOCKED_EXPECTATION_VALUES:
                raise ValueError(
                    f"statement cache expectation {key} is not locked; "
                    "rebuild with --lock-config before formal training"
                )
        actual_sha = file_sha256(cache_path)
        if actual_sha != str(expected_cache["expected_sha256"]).lower():
            raise ValueError("statement cache SHA256 does not match config")
        vocabulary = str(
            payload.get("ontology", {}).get("vocabulary_sha256", "")
        )
        if vocabulary != str(expected_cache["expected_vocabulary_sha256"]).lower():
            raise ValueError("statement cache vocabulary SHA256 does not match config")
        pooling = str(payload.get("encoder", {}).get("pooling", ""))
        if pooling != str(expected_cache["expected_pooling"]):
            raise ValueError("statement cache pooling does not match config")
    mapping = validate_statement_cache(payload, expected_text_by_id)
    return torch.stack(
        [
            mapping[statement_id]
            for statement_id, _ in sorted(
                statement_to_index.items(), key=lambda item: item[1]
            )
        ]
    )
