"""Extract trained Qwen3-VL vision-side weights from a trainable checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_qwen3vl_clinical_instruction import (
    DEFAULT_MODEL_PATH,
    apply_freeze_plan,
    choose_dtype,
    classify_parameter,
    count_parameters,
)


def load_model(model_path: str, dtype: torch.dtype) -> torch.nn.Module:
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    return model


def load_trainable_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("trainable_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {
        "global_step": checkpoint.get("global_step") if isinstance(checkpoint, dict) else None,
        "best_val_loss": checkpoint.get("best_val_loss") if isinstance(checkpoint, dict) else None,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }


def grouped_state_dict(model: torch.nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    grouped: dict[str, dict[str, torch.Tensor]] = {
        "vision_tower": {},
        "visual_connector": {},
    }
    for name, param in model.named_parameters():
        group = classify_parameter(name)
        if group in grouped:
            grouped[group][name] = param.detach().cpu()
    return grouped


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dtype", default="bf16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = choose_dtype(args.dtype)
    model = load_model(args.model_path, dtype=dtype)
    load_info = load_trainable_checkpoint(model, args.checkpoint)
    apply_freeze_plan(model, train_groups={"vision_tower", "visual_connector"})
    grouped = grouped_state_dict(model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vision_path = args.output_dir / "qwen3vl_vision_tower_state.pt"
    connector_path = args.output_dir / "qwen3vl_visual_connector_state.pt"
    combined_path = args.output_dir / "qwen3vl_vision_side_state.pt"
    torch.save(grouped["vision_tower"], vision_path)
    torch.save(grouped["visual_connector"], connector_path)
    torch.save({**grouped["vision_tower"], **grouped["visual_connector"]}, combined_path)

    manifest = {
        "model_path": str(args.model_path),
        "checkpoint": str(args.checkpoint),
        "vision_tower_state": str(vision_path),
        "visual_connector_state": str(connector_path),
        "combined_vision_side_state": str(combined_path),
        "load_info": load_info,
        "parameter_groups": count_parameters(model),
        "note": "Load with the same Qwen3-VL base model, then apply these state_dict entries with strict=False.",
    }
    write_manifest(args.output_dir / "manifest.json", manifest)
    write_manifest(args.output_dir / "vision_export_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
