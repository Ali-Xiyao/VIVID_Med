"""Score-free preparation helpers for the VinDr Qwen3.5 development gate."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from bives_cxr.dicom import load_cxr_dicom
from bives_cxr.pixel_interventions import transform_mask_to_letterbox, union_box_mask
from bives_cxr.qwen35_preprocessing import letterbox_image


IMAGE_SIZE = 448
FINDINGS = ("consolidation", "pleural_effusion")
AREA_QUARTILES = (1, 4)


def select_development_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Freeze four prior-exposed VinDr-train development positives."""

    candidates = [
        dict(row)
        for row in rows
        if row.get("rescue_split") == "protocol_design"
        and int(row.get("binary_label", 0)) == 1
        and row.get("canonical_statement_id") in FINDINGS
        and int(row.get("box_area_quartile", 0)) in AREA_QUARTILES
    ]
    selected: list[dict[str, Any]] = []
    used_images: set[str] = set()
    for finding in FINDINGS:
        for quartile in AREA_QUARTILES:
            subset = sorted(
                (
                    row
                    for row in candidates
                    if row["canonical_statement_id"] == finding
                    and int(row["box_area_quartile"]) == quartile
                ),
                key=lambda row: (
                    0 if row.get("reader_consensus") == "3_of_3" else 1,
                    str(row["sample_id"]),
                ),
            )
            chosen = next(
                (row for row in subset if str(row["image_id"]) not in used_images),
                None,
            )
            if chosen is None:
                raise ValueError(f"no unique VinDr row for {finding}|q{quartile}")
            used_images.add(str(chosen["image_id"]))
            selected.append(chosen)
    if len(selected) != 4 or len(used_images) != 4:
        raise ValueError("VinDr development selection must contain four unique images")
    return selected


def prepare_expert_masks(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Decode one hash-bound DICOM and map released boxes to 448 letterbox space."""

    image_path = Path(str(row["image_path"]))
    if file_sha256(image_path) != str(row["official_image_sha256"]):
        raise ValueError(f"VinDr source image hash changed: {row['sample_id']}")
    image, preprocess = load_cxr_dicom(image_path)
    if image.size != (int(row["native_columns"]), int(row["native_rows"])):
        raise ValueError(f"VinDr native geometry changed: {row['sample_id']}")
    _, content_box = letterbox_image(image, IMAGE_SIZE)
    left, top, right, bottom = content_box
    content = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    content[top:bottom, left:right] = True
    native = union_box_mask(image.width, image.height, row["bounding_boxes"])
    expert = transform_mask_to_letterbox(native, content_box, IMAGE_SIZE) & content
    if not expert.any() or bool((expert & ~content).any()):
        raise ValueError(f"VinDr expert mask is invalid: {row['sample_id']}")
    return expert, content, {
        "dicom_preprocess": preprocess.to_dict(),
        "content_box": list(content_box),
        "expert_area_pixels": int(expert.sum()),
    }


def shard_rows(rows: list[dict[str, Any]], shard_index: int, shard_count: int) -> list[dict[str, Any]]:
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid VinDr development shard")
    return [row for index, row in enumerate(rows) if index % shard_count == shard_index]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
