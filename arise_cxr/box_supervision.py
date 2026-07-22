"""VinDr-train expert-box supervision for the ARISE patch-MIL verifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from bives_cxr.pixel_interventions import transform_mask_to_letterbox, union_box_mask


def boxes_to_patch_mask(
    bounding_boxes: list[dict[str, Any]],
    *,
    native_width: int,
    native_height: int,
    grid_hw: tuple[int, int],
    valid_mask: torch.Tensor,
    image_size: int = 448,
) -> torch.Tensor:
    """Map native VinDr boxes to intersecting patches under frozen letterboxing.

    Patch-center sampling can silently discard a real but small expert box when
    the box falls between neighboring centers.  Supervision instead uses the
    conservative geometric contract "any expert-mask pixel intersects this
    patch cell" and still excludes padded/invalid visual tokens.
    """

    if not bounding_boxes:
        raise ValueError("positive box supervision requires a non-empty box list")
    if native_width <= 0 or native_height <= 0 or image_size <= 0:
        raise ValueError("box-supervision image dimensions must be positive")
    grid_h, grid_w = (int(grid_hw[0]), int(grid_hw[1]))
    if grid_h <= 0 or grid_w <= 0 or valid_mask.numel() != grid_h * grid_w:
        raise ValueError("box-supervision patch geometry changed")
    scale = min(image_size / native_width, image_size / native_height)
    resized_width = max(1, round(native_width * scale))
    resized_height = max(1, round(native_height * scale))
    left = (image_size - resized_width) // 2
    top = (image_size - resized_height) // 2
    content_box = (left, top, left + resized_width, top + resized_height)
    native = union_box_mask(native_width, native_height, bounding_boxes)
    letterboxed = transform_mask_to_letterbox(native, content_box, image_size)
    patch_overlap = np.zeros((grid_h, grid_w), dtype=np.bool_)
    for patch_y in range(grid_h):
        y_min = patch_y * image_size // grid_h
        y_max = (patch_y + 1) * image_size // grid_h
        for patch_x in range(grid_w):
            x_min = patch_x * image_size // grid_w
            x_max = (patch_x + 1) * image_size // grid_w
            patch_overlap[patch_y, patch_x] = bool(
                letterboxed[y_min:y_max, x_min:x_max].any()
            )
    patch = torch.from_numpy(patch_overlap.reshape(-1)).bool()
    patch &= valid_mask.bool().cpu()
    if not bool(patch.any()):
        raise ValueError("expert boxes contain no valid patch center")
    return patch


def box_ranking_loss(
    patch_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    box_mask: torch.Tensor,
    box_available: torch.Tensor,
    *,
    margin: float = 0.5,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Rank expert-box patch evidence above valid outside patches."""

    if patch_logits.ndim != 2 or valid_mask.shape != patch_logits.shape:
        raise ValueError("box ranking logits/valid mask must be [B,P]")
    if box_mask.shape != patch_logits.shape or box_available.shape != (patch_logits.shape[0],):
        raise ValueError("box ranking masks have incompatible shape")
    if margin < 0.0 or temperature <= 0.0:
        raise ValueError("box ranking margin/temperature is invalid")
    losses = []
    for index in range(patch_logits.shape[0]):
        if not bool(box_available[index]):
            continue
        inside = box_mask[index].bool() & valid_mask[index].bool()
        outside = ~box_mask[index].bool() & valid_mask[index].bool()
        if not bool(inside.any()) or not bool(outside.any()):
            raise ValueError("box ranking requires valid inside and outside patches")
        inside_score = temperature * torch.logsumexp(
            patch_logits[index, inside] / temperature,
            dim=0,
        )
        outside_score = temperature * torch.logsumexp(
            patch_logits[index, outside] / temperature,
            dim=0,
        )
        inside_score -= temperature * torch.log(inside.sum().to(patch_logits.dtype))
        outside_score -= temperature * torch.log(outside.sum().to(patch_logits.dtype))
        losses.append(F.softplus(outside_score - inside_score + margin))
    if not losses:
        return patch_logits.sum() * 0.0
    return torch.stack(losses).mean()


class VinDrBoxCachedDataset(Dataset):
    """Read immutable cached VinDr tokens plus score-free expert boxes."""

    def __init__(self, cache_dir: Path, manifest_path: Path, split: str) -> None:
        self.cache_dir = Path(cache_dir)
        self.lock = json.loads((self.cache_dir / "cache_lock.json").read_text(encoding="utf-8"))
        if self.lock.get("status") != "complete":
            raise ValueError("VinDr patch-token cache is not complete")
        self.rows = [
            json.loads(line)
            for line in Path(manifest_path).read_text(encoding="utf-8").splitlines()
            if line
        ]
        index_rows = [
            json.loads(line)
            for line in (self.cache_dir / f"{split}_index.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        index = {str(row["sample_id"]): row for row in index_rows}
        if set(index) != {str(row["sample_id"]) for row in self.rows}:
            raise ValueError("VinDr cache/manifest sample identities differ")
        self.index = index
        statements = sorted({str(row["canonical_statement_id"]) for row in self.rows})
        self.statement_to_index = {value: index for index, value in enumerate(statements)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        cached = self.index[str(row["sample_id"])]
        payload = torch.load(
            self.cache_dir / cached["cache_file"],
            map_location="cpu",
            weights_only=False,
        )
        expected_identity = {
            key: self.lock[key]
            for key in (
                "format_version",
                "manifest_sha256",
                "model_snapshot_sha256",
                "processor_snapshot_sha256",
                "image_preprocess_version",
                "image_size",
                "dtype",
            )
        }
        if payload.get("identity") != expected_identity:
            raise ValueError("VinDr cached token identity mismatch")
        if str(payload.get("image_sha256")) != str(cached.get("image_sha256")):
            raise ValueError("VinDr cached image SHA-256 mismatch")
        tokens = payload["patch_tokens"].float()
        valid = payload["valid_mask"].bool()
        grid_hw = tuple(int(value) for value in payload["grid_hw"])
        if tokens.shape[0] != valid.numel() or tokens.shape[0] != grid_hw[0] * grid_hw[1]:
            raise ValueError("VinDr cached patch geometry changed")
        is_support = int(row["binary_label"]) == 1
        box_mask = torch.zeros_like(valid)
        if is_support:
            box_mask = boxes_to_patch_mask(
                list(row["bounding_boxes"]),
                native_width=int(row["native_columns"]),
                native_height=int(row["native_rows"]),
                grid_hw=grid_hw,
                valid_mask=valid,
            )
        return {
            "sample_id": str(row["sample_id"]),
            "canonical_statement_id": str(row["canonical_statement_id"]),
            "statement_index": self.statement_to_index[str(row["canonical_statement_id"])],
            "binary_label": int(row["binary_label"]),
            "patch_tokens": tokens,
            "valid_mask": valid,
            "box_mask": box_mask,
            "box_available": is_support,
            "grid_hw": grid_hw,
        }


def collate_vindr_box(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_patches = max(item["patch_tokens"].shape[0] for item in batch)
    visual_dim = int(batch[0]["patch_tokens"].shape[1])
    tokens = torch.zeros((len(batch), max_patches, visual_dim), dtype=torch.float32)
    valid = torch.zeros((len(batch), max_patches), dtype=torch.bool)
    boxes = torch.zeros((len(batch), max_patches), dtype=torch.bool)
    for index, item in enumerate(batch):
        patches = item["patch_tokens"].shape[0]
        tokens[index, :patches] = item["patch_tokens"]
        valid[index, :patches] = item["valid_mask"]
        boxes[index, :patches] = item["box_mask"]
    return {
        "sample_ids": [item["sample_id"] for item in batch],
        "canonical_statement_ids": [item["canonical_statement_id"] for item in batch],
        "statement_indices": torch.tensor([item["statement_index"] for item in batch]),
        "binary_labels": torch.tensor([item["binary_label"] for item in batch]),
        "patch_tokens": tokens,
        "valid_mask": valid,
        "box_mask": boxes,
        "box_available": torch.tensor([item["box_available"] for item in batch]),
    }
