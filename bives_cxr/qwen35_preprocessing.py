"""Shared deterministic image geometry for Qwen3.5 BiVES inputs."""

from __future__ import annotations

import torch
from PIL import Image


QWEN35_IMAGE_PREPROCESS_VERSION = "bives_qwen35_letterbox_v1"


def letterbox_image(
    image: Image.Image,
    image_size: int = 448,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    image = image.convert("RGB")
    scale = min(image_size / image.width, image_size / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.BICUBIC,
    )
    canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    left = (image_size - resized.width) // 2
    top = (image_size - resized.height) // 2
    canvas.paste(resized, (left, top))
    return canvas, (left, top, left + resized.width, top + resized.height)


def content_mask_for_grid(
    grid: torch.Tensor,
    content_box: tuple[int, int, int, int],
    image_size: int = 448,
) -> torch.Tensor:
    height, width = int(grid[1]), int(grid[2])
    left, top, right, bottom = content_box
    x_centers = (torch.arange(width, dtype=torch.float32) + 0.5) * image_size / width
    y_centers = (torch.arange(height, dtype=torch.float32) + 0.5) * image_size / height
    return (
        (y_centers[:, None] >= top)
        & (y_centers[:, None] < bottom)
        & (x_centers[None, :] >= left)
        & (x_centers[None, :] < right)
    ).reshape(-1)
