"""Immutable cached-token dataset and spatial targets for MORPH-CXR."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from arise_cxr.box_supervision import boxes_to_patch_mask
from morph_cxr.protocol import file_sha256, validate_manifest


def boundary_from_mask(mask: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    height, width = map(int, grid_hw)
    if mask.numel() != height * width:
        raise ValueError("MORPH boundary mask/grid mismatch")
    image = mask.bool().reshape(1, 1, height, width).float()
    eroded = -F.max_pool2d(-image, kernel_size=3, stride=1, padding=1)
    boundary = image.bool() & ~eroded.bool()
    result = boundary.reshape(-1)
    return result if bool(result.any()) else mask.bool()


def mask_moments(mask: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    height, width = map(int, grid_hw)
    weights = mask.float().reshape(-1)
    if not bool(weights.sum() > 0):
        raise ValueError("MORPH spatial moments require a non-empty mask")
    y = torch.linspace(-1.0, 1.0, height)
    x = torch.linspace(-1.0, 1.0, width)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    mass = weights.sum()
    cx = (weights * xx).sum() / mass
    cy = (weights * yy).sum() / mass
    sx = torch.sqrt((weights * (xx - cx).square()).sum() / mass + 1e-6)
    sy = torch.sqrt((weights * (yy - cy).square()).sum() / mass + 1e-6)
    return torch.stack((cx, cy, sx, sy))


class MorphCachedDataset(Dataset):
    def __init__(self, cache_dir: Path, manifest_path: Path, split: str) -> None:
        self.cache_dir = Path(cache_dir)
        all_rows = [
            json.loads(line)
            for line in Path(manifest_path).read_text(encoding="utf-8").splitlines()
            if line
        ]
        validate_manifest(all_rows)
        self.rows = [row for row in all_rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"MORPH split is empty: {split}")
        self.lock = json.loads((self.cache_dir / "cache_lock.json").read_text(encoding="utf-8"))
        if self.lock.get("status") != "complete":
            raise ValueError("MORPH token cache is incomplete")
        if self.lock.get("manifest_sha256") != file_sha256(manifest_path):
            raise ValueError("MORPH cache/manifest identity mismatch")
        index_rows = [
            json.loads(line)
            for line in (self.cache_dir / "index.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        self.index = {str(row["sample_id"]): row for row in index_rows}
        if set(self.index) != {str(row["sample_id"]) for row in all_rows}:
            raise ValueError("MORPH cache index identities changed")
        statements = sorted({str(row["canonical_statement_id"]) for row in all_rows})
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
        tokens = payload["patch_tokens"].float()
        valid = payload["valid_mask"].bool()
        grid_hw = tuple(map(int, payload["grid_hw"]))
        if tokens.shape[0] != valid.numel() or tokens.shape[0] != grid_hw[0] * grid_hw[1]:
            raise ValueError("MORPH cached token geometry changed")
        spatial = torch.zeros_like(valid)
        boundary = torch.zeros_like(valid)
        moments = torch.zeros(4, dtype=torch.float32)
        positive = int(row["binary_label"]) == 1
        if positive:
            spatial = boxes_to_patch_mask(
                list(row["bounding_boxes"]),
                native_width=int(row["native_columns"]),
                native_height=int(row["native_rows"]),
                grid_hw=grid_hw,
                valid_mask=valid,
            )
            boundary = boundary_from_mask(spatial, grid_hw) & valid
            moments = mask_moments(spatial, grid_hw)
        return {
            "sample_id": str(row["sample_id"]),
            "patient_sha256": str(row["patient_sha256"]),
            "finding": str(row["canonical_statement_id"]),
            "statement_index": self.statement_to_index[str(row["canonical_statement_id"])],
            "label": int(row["binary_label"]),
            "tokens": tokens,
            "valid": valid,
            "spatial": spatial,
            "boundary": boundary,
            "moments": moments,
            "spatial_available": positive,
            "grid_hw": grid_hw,
        }


def collate_morph(batch: list[dict[str, Any]]) -> dict[str, Any]:
    grid_hw = batch[0]["grid_hw"]
    if any(item["grid_hw"] != grid_hw for item in batch):
        raise ValueError("MORPH batches require one frozen patch grid")
    return {
        "sample_ids": [item["sample_id"] for item in batch],
        "patients": [item["patient_sha256"] for item in batch],
        "findings": [item["finding"] for item in batch],
        "statement_indices": torch.tensor([item["statement_index"] for item in batch]),
        "labels": torch.tensor([item["label"] for item in batch]),
        "tokens": torch.stack([item["tokens"] for item in batch]),
        "valid": torch.stack([item["valid"] for item in batch]),
        "spatial": torch.stack([item["spatial"] for item in batch]),
        "boundary": torch.stack([item["boundary"] for item in batch]),
        "moments": torch.stack([item["moments"] for item in batch]),
        "spatial_available": torch.tensor([item["spatial_available"] for item in batch]),
        "grid_hw": grid_hw,
    }
