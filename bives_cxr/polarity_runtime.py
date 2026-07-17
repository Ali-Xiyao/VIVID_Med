"""Runtime helpers for locked cached-token bipolar polarity checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .polarity import BipolarPolarityModel, PolarityModelConfig
from .qwen35_preprocessing import content_mask_for_grid, letterbox_image


def load_locked_polarity_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[BipolarPolarityModel, dict[str, int], dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("variant") not in {"B1_dense", "B2_sparse_exact_k"}:
        raise ValueError("checkpoint is not a locked B1/B2 polarity checkpoint")
    config = PolarityModelConfig(**payload["model_config"])
    model = BipolarPolarityModel(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    statement_to_index = {str(key): int(value) for key, value in payload["statement_to_index"].items()}
    if len(statement_to_index) != config.num_statements:
        raise ValueError("checkpoint statement vocabulary size mismatch")
    return model, statement_to_index, payload


@torch.no_grad()
def extract_qwen35_patch_batch(
    images: list[Image.Image],
    statement_texts: list[str],
    *,
    adapter: torch.nn.Module,
    visual: torch.nn.Module,
    processor: Any,
    device: torch.device,
    image_size: int = 448,
) -> list[dict[str, Any]]:
    if not images or len(images) != len(statement_texts):
        raise ValueError("images and statement_texts must be non-empty and aligned")
    prepared = [letterbox_image(image, image_size) for image in images]
    letterboxed = [item[0] for item in prepared]
    content_boxes = [item[1] for item in prepared]
    texts = []
    for image, statement_text in zip(letterboxed, statement_texts, strict=True):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": statement_text},
                ],
            }
        ]
        texts.append(
            processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        )
    batch = processor(text=texts, images=letterboxed, return_tensors="pt", padding=True)
    pixel_values = batch["pixel_values"].to(device=device, dtype=next(visual.parameters()).dtype)
    grid = batch["image_grid_thw"].to(device)
    patches = adapter(pixel_values, grid)
    result = []
    for index, content_box in enumerate(content_boxes):
        content_mask = content_mask_for_grid(
            batch["image_grid_thw"][index], content_box, image_size
        )
        valid_mask = patches.valid_mask[index].detach().cpu() & content_mask
        result.append(
            {
                "letterboxed_image": letterboxed[index],
                "content_box": content_box,
                "patch_tokens": patches.tokens[index].detach().float(),
                "valid_mask": valid_mask.to(device),
                "grid_hw": tuple(patches.grid_hw[index]),
            }
        )
    return result


@torch.no_grad()
def extract_qwen35_patches(
    image: Image.Image,
    statement_text: str,
    *,
    adapter: torch.nn.Module,
    visual: torch.nn.Module,
    processor: Any,
    device: torch.device,
    image_size: int = 448,
) -> dict[str, Any]:
    return extract_qwen35_patch_batch(
        [image],
        [statement_text],
        adapter=adapter,
        visual=visual,
        processor=processor,
        device=device,
        image_size=image_size,
    )[0]


@torch.no_grad()
def score_statements(
    model: BipolarPolarityModel,
    patch_tokens: torch.Tensor,
    valid_mask: torch.Tensor,
    statement_indices: list[int],
) -> dict[str, torch.Tensor]:
    count = len(statement_indices)
    if count <= 0:
        raise ValueError("statement_indices must be non-empty")
    tokens = patch_tokens.unsqueeze(0).expand(count, -1, -1)
    masks = valid_mask.unsqueeze(0).expand(count, -1)
    indices = torch.tensor(statement_indices, device=patch_tokens.device, dtype=torch.long)
    return model(tokens, masks, indices)
