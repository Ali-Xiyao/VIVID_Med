"""Produce the R0 parity and local-cache audit before D0/D1 training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from rcsd_cxr.d0_d1_contract import render_hard_ums_target
from rcsd_cxr.models.token_distillation import ExactSPDProjector
from rcsd_cxr.token_objective import prepare_token_batch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_legacy_spd(path: Path):
    spec = importlib.util.spec_from_file_location("legacy_spd", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SPDProjector


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-path", required=True, type=Path)
    parser.add_argument("--backbone-weights", required=True, type=Path)
    parser.add_argument("--legacy-spd-source", required=True, type=Path)
    parser.add_argument("--historical-checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    teacher_index = args.teacher_path / "model.safetensors.index.json"
    teacher_weight = args.teacher_path / "model.safetensors"
    if not teacher_weight.is_file() and not teacher_index.is_file():
        raise FileNotFoundError(
            f"no safetensors weight or index under {args.teacher_path}"
        )
    if not args.backbone_weights.is_file():
        raise FileNotFoundError(args.backbone_weights)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.teacher_path),
        local_files_only=True,
        trust_remote_code=True,
        padding_side="right",
    )
    config = AutoConfig.from_pretrained(
        str(args.teacher_path),
        local_files_only=True,
        trust_remote_code=True,
    )
    teacher_config = getattr(config, "text_config", config)
    teacher_hidden_size = int(teacher_config.hidden_size)
    legacy_type = load_legacy_spd(args.legacy_spd_source)
    torch.manual_seed(0)
    legacy = legacy_type(
        vit_embed_dim=768,
        llm_embed_dim=teacher_hidden_size,
        num_groups=4,
        tokens_per_group=2,
        num_heads=4,
        # The historical SPD hidden layer is twice the 768-d ViT width. The
        # teacher change affects only the output projection dimension.
        mlp_hidden_dim=1536,
        dropout=0.1,
    )
    torch.manual_seed(0)
    clean = ExactSPDProjector(
        vision_dim=768,
        output_dim=teacher_hidden_size,
        num_groups=4,
        tokens_per_group=2,
        num_heads=4,
        dropout=0.1,
    )
    legacy_parameters = sum(parameter.numel() for parameter in legacy.parameters())
    clean_parameters = sum(parameter.numel() for parameter in clean.parameters())
    sample = torch.randn(2, 197, 768)
    legacy_shape = list(legacy(sample).shape)
    clean_shape = list(clean(sample).shape)
    target = render_hard_ums_target(
        {
            "Cardiomegaly": "present",
            "Edema": "absent",
            "Pleural Effusion": "uncertain",
        }
    )
    weights = {
        "Cardiomegaly": 1.0,
        "Edema": 0.420619835714305,
        "Pleural Effusion": 0.0,
    }
    d0 = prepare_token_batch(
        tokenizer,
        prompt="Generate a structured medical report:\n",
        targets=[target],
        finding_weights=[weights],
        variant="d0",
    )
    d1 = prepare_token_batch(
        tokenizer,
        prompt="Generate a structured medical report:\n",
        targets=[target],
        finding_weights=[weights],
        variant="d1",
    )
    checkpoint_files = (
        sorted(
            str(path.relative_to(args.historical_checkpoint_dir))
            for path in args.historical_checkpoint_dir.rglob("*")
            if path.is_file()
        )
        if args.historical_checkpoint_dir.is_dir()
        else []
    )
    checks = {
        "teacher_is_qwen35": config.model_type == "qwen3_5",
        "teacher_hidden_size_positive": teacher_hidden_size > 0,
        "spd_parameter_count_parity": clean_parameters == legacy_parameters,
        "spd_output_shape_parity": (
            clean_shape
            == legacy_shape
            == [2, 205, teacher_hidden_size]
        ),
        "spd_query_layout": (
            clean.num_groups == 4
            and clean.tokens_per_group == 2
            and clean.num_query_tokens == 8
        ),
        "d0_d1_input_ids_identical": torch.equal(
            d0["input_ids"], d1["input_ids"]
        ),
        "d0_d1_labels_identical": torch.equal(d0["labels"], d1["labels"]),
        "d1_has_nonunit_span_weights": bool(
            (d1["token_weights"] < 1.0).any()
        ),
        "target_fits_512_tokens": int((d0["labels"] != -100).sum()) < 512,
        "historical_checkpoint_unavailable_recorded": len(checkpoint_files) == 0,
    }
    result = {
        "schema_version": 1,
        "artifact": "d0_d1_r0_parity",
        "pass": all(checks.values()),
        "checks": checks,
        "teacher": {
            "path": str(args.teacher_path),
            "hidden_size": teacher_hidden_size,
            "model_type": config.model_type,
        },
        "spd": {
            "legacy_parameters": legacy_parameters,
            "clean_parameters": clean_parameters,
            "legacy_output_shape": legacy_shape,
            "clean_output_shape": clean_shape,
        },
        "historical_checkpoint": {
            "directory": str(args.historical_checkpoint_dir),
            "files": checkpoint_files,
            "resolution": (
                "unavailable_provenance_only"
                if not checkpoint_files
                else "files_present_requires_hash_import"
            ),
        },
        "hashes": {
            "teacher_weight_authority": sha256_file(
                teacher_index if teacher_index.is_file() else teacher_weight
            ),
            "backbone_weights": sha256_file(args.backbone_weights),
            "legacy_spd_source": sha256_file(args.legacy_spd_source),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
