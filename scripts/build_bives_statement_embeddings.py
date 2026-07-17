"""Build a provenance-complete fixed Qwen3.5 statement embedding cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from safetensors import safe_open
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bives_cxr.backbones import validate_qwen35_model_path
from bives_cxr.data import read_manifest, statement_text_by_id
from bives_cxr.statement_cache import build_statement_cache_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="local")
    parser.add_argument("--tokenizer-revision", default="local")
    parser.add_argument("--pooling", choices=("input_embedding_mean",), default="input_embedding_mean")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--config-template", type=Path, help="Tracked config template to copy.")
    parser.add_argument("--output-config", type=Path, help="Ignored local locked config to write.")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_locked_config(template: Path, output: Path, cache: Path, payload: dict[str, Any], cache_sha256: str, pooling: str) -> None:
    """Write a local lock without mutating a tracked experiment template."""
    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    statement = config["model"]["statement_embeddings"]
    if str(statement.get("mode")) != "frozen_cached":
        raise ValueError(f"{template} does not use frozen_cached statements")
    statement["path"] = str(cache)
    statement["expected_sha256"] = cache_sha256
    statement["expected_vocabulary_sha256"] = payload["ontology"]["vocabulary_sha256"]
    statement["expected_pooling"] = pooling
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def load_input_embedding_weight(model_path: Path) -> tuple[torch.Tensor, str]:
    index_path = model_path / "model.safetensors.index.json"
    candidates: dict[str, str] = {}
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        candidates = {
            key: shard
            for key, shard in weight_map.items()
            if key.endswith("embed_tokens.weight")
        }
    else:
        single = model_path / "model.safetensors"
        if not single.is_file():
            raise FileNotFoundError("Qwen3.5 safetensors weights are required")
        with safe_open(single, framework="pt", device="cpu") as shard:
            candidates = {
                key: single.name
                for key in shard.keys()
                if key.endswith("embed_tokens.weight")
            }
    preferred = [
        key for key in candidates
        if "language" in key or key.startswith("model.embed_tokens")
    ]
    keys = preferred or list(candidates)
    if len(keys) != 1:
        raise ValueError(f"expected one Qwen3.5 input embedding tensor, found {keys}")
    key = keys[0]
    with safe_open(model_path / candidates[key], framework="pt", device="cpu") as shard:
        return shard.get_tensor(key), key


def main() -> None:
    args = parse_args()
    if bool(args.config_template) != bool(args.output_config):
        raise ValueError("--config-template and --output-config must be supplied together")
    validate_qwen35_model_path(args.model_path)
    all_rows: list[dict[str, Any]] = []
    for manifest in args.manifest:
        all_rows.extend(read_manifest(manifest))
    texts = statement_text_by_id(all_rows)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        local_files_only=True,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Qwen3.5 processor has no tokenizer")
    weight, weight_key = load_input_embedding_weight(args.model_path)
    embeddings: dict[str, torch.Tensor] = {}
    for statement_id in sorted(texts):
        tokenized = tokenizer(
            texts[statement_id],
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = tokenized["input_ids"][0]
        vector = weight[input_ids].float().mean(dim=0)
        if args.normalize:
            vector = F.normalize(vector, dim=0)
        embeddings[statement_id] = vector
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    embeddings = {
        statement_id: vector.to(dtype=dtype_map[args.dtype])
        for statement_id, vector in embeddings.items()
    }
    config_path = args.model_path / "config.json"
    payload = build_statement_cache_payload(
        embeddings,
        texts,
        {
            "model_name_or_path": str(args.model_path),
            "revision": args.revision,
            "tokenizer_revision": args.tokenizer_revision,
            "tokenizer_class": type(tokenizer).__name__,
            "pooling": args.pooling,
            "normalize": args.normalize,
            "dtype": args.dtype,
            "embedding_weight_key": weight_key,
            "config_sha256": file_sha256(config_path),
        },
        sources={
            "manifests": {
                str(path): file_sha256(path) for path in args.manifest
            }
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    cache_sha256 = file_sha256(args.output)
    if args.config_template:
        write_locked_config(args.config_template, args.output_config, args.output, payload, cache_sha256, args.pooling)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "statements": len(texts),
                "embedding_dim": payload["embedding_dim"],
                "cache_sha256": cache_sha256,
                "vocabulary_sha256": payload["ontology"]["vocabulary_sha256"],
                "output_config": str(args.output_config) if args.output_config else None,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
