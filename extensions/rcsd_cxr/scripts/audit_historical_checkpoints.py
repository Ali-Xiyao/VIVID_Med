"""Read-only metadata audit for the historical UMS and SPD checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_shape(value: object) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return [int(item) for item in shape] if shape is not None else None


def load_checkpoint_summary(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    state = torch.load(path, map_location="meta", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint is not a mapping: {path}")
    projector = state.get("projector")
    if not isinstance(projector, dict):
        raise TypeError(f"checkpoint has no projector state: {path}")
    group_queries = {
        key: tensor_shape(value)
        for key, value in projector.items()
        if "group_queries." in key
    }
    prefix_tokens = {
        key: tensor_shape(value)
        for key, value in projector.items()
        if key.endswith("prefix_tokens")
    }
    summary = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "top_level_keys": sorted(str(key) for key in state),
        "global_step": int(state["global_step"]),
        "best_val_loss": float(state["best_val_loss"]),
        "group_queries": group_queries,
        "prefix_tokens": prefix_tokens,
    }
    return summary, state


def run_audit(
    *,
    ums_checkpoint: Path,
    spd_checkpoint: Path,
    ums_config: Path,
    spd_config: Path,
    trainer_source: Path,
) -> dict[str, object]:
    for path in (
        ums_checkpoint,
        spd_checkpoint,
        ums_config,
        spd_config,
        trainer_source,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    ums, _ = load_checkpoint_summary(ums_checkpoint)
    spd, _ = load_checkpoint_summary(spd_checkpoint)
    ums_cfg = yaml.safe_load(ums_config.read_text(encoding="utf-8"))
    spd_cfg = yaml.safe_load(spd_config.read_text(encoding="utf-8"))
    trainer_text = trainer_source.read_text(encoding="utf-8")
    group_shapes = list(spd["group_queries"].values())
    config_spd = spd_cfg.get("model", {}).get("spd", {})
    checks = {
        "ums_spd_disabled": not ums_cfg.get("model", {}).get("spd", {}).get(
            "enabled", False
        ),
        "spd_config_4x2": (
            config_spd.get("enabled") is True
            and config_spd.get("num_groups") == 4
            and config_spd.get("tokens_per_group") == 2
        ),
        "spd_checkpoint_four_groups": len(group_shapes) == 4,
        "spd_checkpoint_two_tokens_per_group": bool(group_shapes)
        and all(shape is not None and len(shape) == 3 and shape[1] == 2 for shape in group_shapes),
        "checkpoint_keys_complete": all(
            required in set(ums["top_level_keys"]) & set(spd["top_level_keys"])
            for required in (
                "vit",
                "projector",
                "optimizer",
                "scheduler",
                "global_step",
                "best_val_loss",
            )
        ),
        "single_selection_rule_found": "if val_loss < self.best_val_loss:" in trainer_text,
    }
    return {
        "schema_version": 1,
        "gate": "G1_historical_checkpoint_static_audit",
        "pass": all(checks.values()),
        "checks": checks,
        "ums": ums,
        "spd": spd,
        "configs": {
            "ums": str(ums_config.resolve()),
            "ums_sha256": sha256_file(ums_config),
            "spd": str(spd_config.resolve()),
            "spd_sha256": sha256_file(spd_config),
        },
        "checkpoint_selection": {
            "source": str(trainer_source.resolve()),
            "source_sha256": sha256_file(trainer_source),
            "criterion": "strictly lower validation loss",
            "artifact": "best.pt",
        },
        "scope_note": (
            "This is read-only historical evidence. It does not establish a "
            "runnable RCSD trainer or authorize visual-model training."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ums-checkpoint", required=True, type=Path)
    parser.add_argument("--spd-checkpoint", required=True, type=Path)
    parser.add_argument("--ums-config", required=True, type=Path)
    parser.add_argument("--spd-config", required=True, type=Path)
    parser.add_argument("--trainer-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run_audit(
            ums_checkpoint=args.ums_checkpoint,
            spd_checkpoint=args.spd_checkpoint,
            ums_config=args.ums_config,
            spd_config=args.spd_config,
            trainer_source=args.trainer_source,
        )
    except Exception as error:
        result = {
            "schema_version": 1,
            "gate": "G1_historical_checkpoint_static_audit",
            "pass": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
