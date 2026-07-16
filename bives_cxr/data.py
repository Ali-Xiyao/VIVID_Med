"""Manifest schema and fail-fast dataset utilities for BiVES-CXR."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from .decoder import STATE_NAMES


REQUIRED_FIELDS = {
    "sample_id",
    "patient_id",
    "image_path",
    "canonical_statement_id",
    "statement_text",
    "state",
}


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = REQUIRED_FIELDS - set(row)
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
            state = str(row["state"]).lower()
            if state not in STATE_NAMES:
                raise ValueError(f"{path}:{line_number} invalid state: {state}")
            row["state"] = state
            rows.append(row)
    if not rows:
        raise ValueError(f"empty BiVES manifest: {path}")
    return rows


class BiVESManifestDataset(Dataset):
    """Load images without converting IO failures into false insufficient samples."""

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path = ".",
        statement_to_index: dict[str, int] | None = None,
    ) -> None:
        self.rows = read_manifest(manifest_path)
        self.data_root = Path(data_root)
        if statement_to_index is None:
            statement_ids = sorted({str(row["canonical_statement_id"]) for row in self.rows})
            statement_to_index = {statement_id: index for index, statement_id in enumerate(statement_ids)}
        unknown = {
            str(row["canonical_statement_id"])
            for row in self.rows
            if str(row["canonical_statement_id"]) not in statement_to_index
        }
        if unknown:
            raise ValueError(f"manifest contains statement IDs absent from the training vocabulary: {sorted(unknown)[:5]}")
        self.statement_to_index = dict(statement_to_index)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = Path(str(row["image_path"]))
        if not image_path.is_absolute():
            image_path = self.data_root / image_path
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        image = Image.open(image_path).convert("RGB")
        return {
            **row,
            "image": image,
            "image_path": str(image_path),
            "statement_index": self.statement_to_index[str(row["canonical_statement_id"])],
            "state_index": STATE_NAMES.index(str(row["state"])),
        }
