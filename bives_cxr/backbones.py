"""Qwen3.5-only visual backbone adapter for BiVES-CXR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass
class PatchBatch:
    tokens: torch.Tensor
    valid_mask: torch.Tensor
    grid_hw: list[tuple[int, int]]


def validate_qwen35_model_path(model_path: str | Path) -> dict[str, Any]:
    import json

    path = Path(model_path)
    config_path = path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_5" or not config.get("vision_config"):
        raise ValueError(f"{path} is not a multimodal Qwen3.5 model")
    return config


def locate_visual_module(model: nn.Module) -> nn.Module:
    for dotted in ("visual", "model.visual"):
        module: Any = model
        for part in dotted.split("."):
            if not hasattr(module, part):
                break
            module = getattr(module, part)
        else:
            return module
    raise AttributeError("Could not locate Qwen3.5 visual module at visual or model.visual")


class Qwen35VisionAdapter(nn.Module):
    """Extract merger-preceding spatial patch tokens from Qwen3.5."""

    def __init__(self, model: nn.Module, spatial_merge_size: int) -> None:
        super().__init__()
        self.visual = locate_visual_module(model)
        self.spatial_merge_size = int(spatial_merge_size)

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> PatchBatch:
        if bool((image_grid_thw[:, 0] != 1).any()):
            raise ValueError("BiVES-CXR currently supports one static CXR frame per sample (T=1)")
        visual_output = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True)
        hidden = getattr(visual_output, "last_hidden_state", visual_output)
        if hidden.ndim != 2:
            raise ValueError(f"unexpected visual output shape: {tuple(hidden.shape)}")

        counts = image_grid_thw.prod(dim=1).long().tolist()
        if sum(counts) != int(hidden.shape[0]):
            raise ValueError(
                f"Qwen3.5 spatial-token mismatch: expected {sum(counts)} from grid, got {hidden.shape[0]}"
            )
        chunks = torch.split(hidden, counts, dim=0)
        max_patches = max(counts)
        padded = hidden.new_zeros((len(chunks), max_patches, hidden.shape[-1]))
        valid = torch.zeros((len(chunks), max_patches), dtype=torch.bool, device=hidden.device)
        grid_hw: list[tuple[int, int]] = []
        for index, (chunk, grid) in enumerate(zip(chunks, image_grid_thw)):
            padded[index, : chunk.shape[0]] = chunk
            valid[index, : chunk.shape[0]] = True
            grid_hw.append((int(grid[1]), int(grid[2])))
        return PatchBatch(padded, valid, grid_hw)


def load_qwen35_model_and_processor(
    model_path: str | Path,
    dtype: str = "bf16",
    device_map: str | dict[str, Any] | None = None,
) -> tuple[nn.Module, Any, dict[str, Any]]:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    config = validate_qwen35_model_path(model_path)
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if dtype not in dtype_map:
        raise ValueError(f"unsupported dtype: {dtype}")
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=dtype_map[dtype],
        device_map=device_map,
    )
    return model, processor, config
