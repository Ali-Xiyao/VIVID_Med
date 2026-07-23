"""Fail-closed image manifest dataset.

The manifest is deliberately image-format agnostic. Dataset-specific builders
must resolve patient/study/image identity before this loader is used.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image


@dataclass(frozen=True)
class ManifestRow:
    patient_id: str
    study_id: str
    image_id: str
    image_path: Path
    split: str


class ManifestImageDataset:
    REQUIRED_COLUMNS = {"patient_id", "study_id", "image_id", "image_path", "split"}

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        image_root: str | Path | None = None,
        transform: Callable | None = None,
        allowed_splits: set[str] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"manifest not found: {self.manifest_path}")
        self.image_root = Path(image_root) if image_root is not None else None
        self.transform = transform
        self.rows = self._read_rows(allowed_splits)
        if not self.rows:
            raise ValueError("manifest selection is empty")

    def _read_rows(self, allowed_splits: set[str] | None) -> list[ManifestRow]:
        rows: list[ManifestRow] = []
        seen_images: set[str] = set()
        with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            missing = self.REQUIRED_COLUMNS - fields
            if missing:
                raise ValueError(f"manifest is missing columns: {sorted(missing)}")
            for line_number, raw in enumerate(reader, start=2):
                if allowed_splits is not None and raw["split"] not in allowed_splits:
                    continue
                values = {key: (raw[key] or "").strip() for key in self.REQUIRED_COLUMNS}
                empty = [key for key, value in values.items() if not value]
                if empty:
                    raise ValueError(f"line {line_number} has empty fields: {empty}")
                if values["image_id"] in seen_images:
                    raise ValueError(f"duplicate image_id: {values['image_id']}")
                seen_images.add(values["image_id"])
                relative = Path(values["image_path"])
                image_path = self.image_root / relative if self.image_root else relative
                rows.append(
                    ManifestRow(
                        patient_id=values["patient_id"],
                        study_id=values["study_id"],
                        image_id=values["image_id"],
                        image_path=image_path,
                        split=values["split"],
                    )
                )
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        if not row.image_path.is_file():
            raise FileNotFoundError(f"image not found: {row.image_path}")
        try:
            with Image.open(row.image_path) as handle:
                image = handle.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"failed to decode image: {row.image_path}") from exc
        if self.transform is not None:
            image = self.transform(image)
        return {"image": image, "identity": row}
