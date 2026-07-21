"""Score-free CheXlocalize validation binding and expert-mask helpers."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from bives_cxr.c6_intake import parse_chexlocalize_key, parse_chexpert_path
from bives_cxr.pixel_interventions import transform_mask_to_letterbox
from bives_cxr.qwen35_preprocessing import letterbox_image


IMAGE_SIZE = 448
TARGETS = {
    "Consolidation": ("consolidation", "Pulmonary consolidation is present."),
    "Pleural Effusion": ("pleural_effusion", "A pleural effusion is present."),
}


def identifier_sha256(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}:{value}".encode("utf-8")).hexdigest()


def read_chexpert_validation_rows(
    valid_csv: Path, chexpert_root: Path
) -> dict[str, dict[str, Any]]:
    with valid_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("CheXpert validation CSV is empty")
    path_column = next(
        (column for column in rows[0] if column.lower() in {"path", "image_path"}),
        None,
    )
    if path_column is None:
        raise ValueError("CheXpert validation CSV has no path column")
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        raw_path = str(row[path_column]).replace("\\", "/")
        identity = parse_chexpert_path(raw_path)
        relative = Path(raw_path)
        if relative.parts and relative.parts[0].lower() == chexpert_root.name.lower():
            relative = Path(*relative.parts[1:])
        image_path = (chexpert_root / relative).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if identity["image"] in output:
            raise ValueError(f"duplicate CheXpert validation image: {identity['image']}")
        output[identity["image"]] = {
            "identity": identity,
            "image_path": image_path,
        }
    return output


def bind_annotation_identities(
    annotations: dict[str, Any],
    validation_rows: dict[str, dict[str, Any]],
) -> list[tuple[str, dict[str, str], dict[str, Any]]]:
    bound = []
    for annotation_key, payload in annotations.items():
        identity = parse_chexlocalize_key(str(annotation_key))
        if identity["image"] not in validation_rows:
            raise ValueError("CheXlocalize annotation is absent from valid.csv")
        if not isinstance(payload, dict):
            raise ValueError("CheXlocalize annotation payload must be an object")
        bound.append((str(annotation_key), identity, payload))
    return sorted(bound, key=lambda row: row[1]["image"])


def rasterize_contours(
    contours: Any,
    image_size: Any,
    *,
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    if not isinstance(image_size, list) or len(image_size) != 2:
        raise ValueError("invalid CheXlocalize img_size")
    height, width = (int(image_size[0]), int(image_size[1]))
    if height <= 0 or width <= 0:
        raise ValueError("invalid CheXlocalize native dimensions")
    if not isinstance(contours, list) or not contours:
        raise ValueError("empty CheXlocalize expert contours")
    output_width, output_height = output_size or (width, height)
    if output_width <= 0 or output_height <= 0:
        raise ValueError("invalid CheXlocalize output dimensions")
    scale_x = output_width / width
    scale_y = output_height / height
    canvas = Image.new("1", (output_width, output_height), 0)
    draw = ImageDraw.Draw(canvas)
    for contour in contours:
        if not isinstance(contour, list) or len(contour) < 3:
            raise ValueError("invalid CheXlocalize expert contour")
        points = []
        for point in contour:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError("invalid CheXlocalize expert point")
            x, y = float(point[0]), float(point[1])
            if not (0 <= x <= width and 0 <= y <= height):
                raise ValueError("out-of-bounds CheXlocalize expert point")
            points.append((x * scale_x, y * scale_y))
        draw.polygon(points, fill=1)
    mask = np.asarray(canvas, dtype=bool)
    if not mask.any():
        raise ValueError("CheXlocalize expert mask is empty")
    return mask


def prepare_letterboxed_masks(
    image_path: Path, contours: Any, image_size: Any
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    annotation_height, annotation_width = int(image_size[0]), int(image_size[1])
    image_width, image_height = image.size
    annotation_ratio = annotation_width / annotation_height
    image_ratio = image_width / image_height
    aspect_ratio_delta = abs(annotation_ratio - image_ratio)
    if aspect_ratio_delta > 0.01:
        raise ValueError("CheXlocalize annotation/image aspect ratio mismatch")
    native_mask = rasterize_contours(
        contours,
        image_size,
        output_size=(image_width, image_height),
    )
    letterboxed, content_box = letterbox_image(image, IMAGE_SIZE)
    del letterboxed
    left, top, right, bottom = content_box
    content = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    content[top:bottom, left:right] = True
    expert = transform_mask_to_letterbox(native_mask, content_box, IMAGE_SIZE) & content
    if not expert.any() or bool((expert & ~content).any()):
        raise ValueError("invalid letterboxed CheXlocalize expert mask")
    return expert, content, {
        "annotation_size": [annotation_height, annotation_width],
        "local_image_size": [image_height, image_width],
        "annotation_to_local_scale_xy": [
            image_width / annotation_width,
            image_height / annotation_height,
        ],
        "aspect_ratio_delta": aspect_ratio_delta,
        "content_box": list(content_box),
        "expert_area_pixels": int(expert.sum()),
    }


def patient_disjoint_shard(
    rows: list[dict[str, Any]], shard_index: int, shard_count: int
) -> list[dict[str, Any]]:
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid CheXlocalize development shard")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        patient = str(row["patient_id_hash"])
        grouped.setdefault(patient, []).append(row)
    assignments: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    totals = [0] * shard_count
    for patient in sorted(grouped, key=lambda key: (-len(grouped[key]), key)):
        target = min(range(shard_count), key=lambda index: (totals[index], index))
        assignments[target].extend(grouped[patient])
        totals[target] += len(grouped[patient])
    return sorted(
        assignments[shard_index],
        key=lambda row: (str(row["patient_id_hash"]), str(row["sample_id"])),
    )
