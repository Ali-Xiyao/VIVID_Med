"""Fail-fast artifact lineage locks for formal BiVES-CXR runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from .interventions import CONTROL_PROTOCOL_VERSION


RUN_LOCK_FORMAT_VERSION = 1


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolved_config_text(config: dict[str, Any]) -> str:
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


def resolved_config_sha256(config: dict[str, Any]) -> str:
    return hashlib.sha256(resolved_config_text(config).encode("utf-8")).hexdigest()


def base_model_lock_files(model_path: str | Path) -> tuple[Path, Path]:
    root = Path(model_path)
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"base model config is missing: {config_path}")
    candidates = (
        root / "model.safetensors.index.json",
        root / "model.safetensors",
    )
    weight_lock = next((path for path in candidates if path.is_file()), None)
    if weight_lock is None:
        raise FileNotFoundError(
            f"base model has no safetensors index or single weight file: {root}"
        )
    return config_path, weight_lock


def build_run_lock(
    config: dict[str, Any],
    *,
    git_commit: str,
) -> dict[str, Any]:
    if not str(git_commit).strip() or str(git_commit).strip().lower() == "unknown":
        raise ValueError(
            "formal run lock requires a known Git commit; set BIVES_GIT_COMMIT "
            "or deploy a .bives_source_commit marker"
        )
    data = config["data"]
    model = config["model"]
    statement = model["statement_embeddings"]
    cache_path = Path(statement["path"])
    if not cache_path.is_file():
        raise FileNotFoundError(f"statement cache is missing: {cache_path}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    vocabulary_sha = str(
        cache.get("ontology", {}).get("vocabulary_sha256", "")
    )
    if not vocabulary_sha:
        raise ValueError("statement cache is missing ontology vocabulary_sha256")
    base_config, base_weight_lock = base_model_lock_files(model["path"])
    manifests: dict[str, str] = {}
    for split, key in (
        ("train", "train_manifest"),
        ("val", "val_manifest"),
        ("calibration", "calibration_manifest"),
        ("test", "test_manifest"),
    ):
        path = Path(str(data.get(key, "")))
        if not path.is_file():
            raise FileNotFoundError(f"locked {split} manifest is missing: {path}")
        manifests[split] = file_sha256(path)
    control_seed = config.get("evaluation", {}).get("control_seed")
    if control_seed is None:
        raise ValueError("evaluation.control_seed is required for a formal run lock")
    return {
        "format_version": RUN_LOCK_FORMAT_VERSION,
        "git_commit": str(git_commit),
        "resolved_config_sha256": resolved_config_sha256(config),
        "base_model_config_sha256": file_sha256(base_config),
        "base_model_weight_index_file": base_weight_lock.name,
        "base_model_weight_index_sha256": file_sha256(base_weight_lock),
        "statement_cache_sha256": file_sha256(cache_path),
        "statement_vocabulary_sha256": vocabulary_sha,
        "manifest_sha256": manifests,
        "control_protocol_version": CONTROL_PROTOCOL_VERSION,
        "evaluation_control_seed": int(control_seed),
    }


def validate_checkpoint_run_lock(
    checkpoint: dict[str, Any],
    *,
    current_git_commit: str,
) -> dict[str, Any]:
    run_lock = checkpoint.get("run_lock")
    if not isinstance(run_lock, dict):
        raise ValueError("checkpoint is missing run_lock")
    if run_lock.get("format_version") != RUN_LOCK_FORMAT_VERSION:
        raise ValueError("checkpoint run_lock format_version is unsupported")
    expected_hash = canonical_json_sha256(run_lock)
    if checkpoint.get("run_lock_sha256") != expected_hash:
        raise ValueError("checkpoint run_lock_sha256 does not match its run_lock")
    if str(run_lock.get("git_commit")) != str(current_git_commit):
        raise ValueError(
            "current evaluator Git commit does not match checkpoint run_lock"
        )
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint is missing resolved config")
    if resolved_config_sha256(config) != run_lock.get("resolved_config_sha256"):
        raise ValueError("checkpoint config does not match resolved_config_sha256")
    model_config, model_weight_lock = base_model_lock_files(config["model"]["path"])
    if file_sha256(model_config) != run_lock.get("base_model_config_sha256"):
        raise ValueError("base model config does not match run_lock")
    if model_weight_lock.name != run_lock.get("base_model_weight_index_file"):
        raise ValueError("base model weight-index filename does not match run_lock")
    if file_sha256(model_weight_lock) != run_lock.get("base_model_weight_index_sha256"):
        raise ValueError("base model weight index does not match run_lock")
    evaluation_seed = config.get("evaluation", {}).get("control_seed")
    if int(evaluation_seed) != int(run_lock.get("evaluation_control_seed")):
        raise ValueError("evaluation control seed does not match run_lock")
    if run_lock.get("control_protocol_version") != CONTROL_PROTOCOL_VERSION:
        raise ValueError("control protocol version does not match active evaluator")
    return run_lock


def validate_calibrated_release_chain(
    *,
    checkpoint_path: str | Path,
    checkpoint: dict[str, Any],
    calibration: dict[str, Any],
    statement_cache_path: str | Path,
    test_manifest_path: str | Path,
    current_git_commit: str,
) -> dict[str, Any]:
    run_lock = validate_checkpoint_run_lock(
        checkpoint,
        current_git_commit=current_git_commit,
    )
    checks = {
        "base checkpoint": (
            file_sha256(checkpoint_path),
            calibration.get("base_checkpoint_sha256"),
        ),
        "run lock": (
            checkpoint.get("run_lock_sha256"),
            calibration.get("run_lock_sha256"),
        ),
        "statement cache": (
            file_sha256(statement_cache_path),
            run_lock.get("statement_cache_sha256"),
        ),
        "calibration statement cache": (
            calibration.get("statement_cache_sha256"),
            run_lock.get("statement_cache_sha256"),
        ),
        "statement vocabulary": (
            calibration.get("statement_vocabulary_sha256"),
            run_lock.get("statement_vocabulary_sha256"),
        ),
        "locked test manifest": (
            file_sha256(test_manifest_path),
            run_lock.get("manifest_sha256", {}).get("test"),
        ),
        "selected best step": (
            int(checkpoint.get("step", -1)),
            int(calibration.get("selected_best_step", -2)),
        ),
    }
    mismatches = [name for name, (actual, expected) in checks.items() if actual != expected]
    if mismatches:
        raise ValueError(
            "locked-test artifact chain mismatch: " + ", ".join(mismatches)
        )
    return run_lock
