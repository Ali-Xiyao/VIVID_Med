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
    if all(hasattr(model, name) for name in ("patch_embed", "blocks", "merger")):
        return model
    for dotted in ("visual", "model.visual"):
        module: Any = model
        for part in dotted.split("."):
            if not hasattr(module, part):
                break
            module = getattr(module, part)
        else:
            return module
    raise AttributeError("Could not locate Qwen3.5 visual module at visual or model.visual")


def restore_qwen35_row_major(
    chunk: torch.Tensor,
    height: int,
    width: int,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Invert Qwen3.5's merge-block-major pre-merger token ordering."""

    merge = int(spatial_merge_size)
    if chunk.ndim != 2 or chunk.shape[0] != height * width:
        raise ValueError("chunk must have shape [height * width, hidden]")
    if height % merge != 0 or width % merge != 0:
        raise ValueError(
            f"Qwen3.5 grid {(height, width)} is not divisible by spatial_merge_size={merge}"
        )
    hidden = chunk.shape[-1]
    return (
        chunk.view(height // merge, width // merge, merge, merge, hidden)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
        .view(height * width, hidden)
    )


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
            height, width = int(grid[1]), int(grid[2])
            chunk = restore_qwen35_row_major(
                chunk,
                height,
                width,
                self.spatial_merge_size,
            )
            padded[index, : chunk.shape[0]] = chunk
            valid[index, : chunk.shape[0]] = True
            grid_hw.append((height, width))
        return PatchBatch(padded, valid, grid_hw)


def load_qwen35_visual_and_processor(
    model_path: str | Path,
    dtype: str = "bf16",
    attention_implementation: str = "eager",
) -> tuple[nn.Module, Any, dict[str, Any]]:
    """Load the vision tower without retaining the unused language model."""

    import json

    from safetensors import safe_open
    from transformers import AutoConfig, AutoProcessor
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    config = validate_qwen35_model_path(model_path)
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if dtype not in dtype_map:
        raise ValueError(f"unsupported dtype: {dtype}")
    processor = AutoProcessor.from_pretrained(model_path)
    parent_config = AutoConfig.from_pretrained(model_path)
    if attention_implementation not in {"eager", "sdpa"}:
        raise ValueError(
            f"unsupported Qwen3.5 visual attention implementation: {attention_implementation}"
        )
    parent_config.vision_config._attn_implementation = attention_implementation
    parent_config.vision_config._attn_implementation_internal = attention_implementation
    visual = Qwen3_5VisionModel(parent_config.vision_config)
    rotary_inv_freq = visual.rotary_pos_emb.inv_freq.detach().clone()
    model_root = Path(model_path)
    index_path = model_root / "model.safetensors.index.json"
    state_dict: dict[str, torch.Tensor] = {}
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        visual_weight_map = {
            key: file_name
            for key, file_name in weight_map.items()
            if key.startswith("model.visual.")
        }
        if not visual_weight_map:
            raise ValueError(f"{index_path} contains no model.visual.* weights")
        for shard_name in sorted(set(visual_weight_map.values())):
            with safe_open(model_root / shard_name, framework="pt", device="cpu") as shard:
                for full_key, mapped_shard in visual_weight_map.items():
                    if mapped_shard == shard_name:
                        state_dict[full_key.removeprefix("model.visual.")] = shard.get_tensor(full_key)
    else:
        single_path = model_root / "model.safetensors"
        if not single_path.is_file():
            raise FileNotFoundError(
                f"Qwen3.5 vision-only loading requires {index_path.name} or {single_path.name}"
            )
        with safe_open(single_path, framework="pt", device="cpu") as shard:
            for full_key in shard.keys():
                if full_key.startswith("model.visual."):
                    state_dict[full_key.removeprefix("model.visual.")] = shard.get_tensor(full_key)
    incompatible = visual.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict Qwen3.5 visual load failed: {incompatible}")
    visual.to(dtype=dtype_map[dtype])
    # The official full Qwen3.5 model keeps the non-persistent rotary frequency
    # buffer in FP32 even when model weights are BF16/FP16. Module.to(dtype)
    # would silently downcast it and materially change visual outputs.
    visual.rotary_pos_emb.inv_freq = rotary_inv_freq
    return visual, processor, config
