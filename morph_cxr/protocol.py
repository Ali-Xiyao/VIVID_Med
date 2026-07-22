"""Fail-closed protocol helpers for the MORPH-CXR separability gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


MORPH_FINDINGS = (
    "pneumothorax",
    "consolidation",
    "pleural_effusion",
    "cardiomegaly",
)
MORPH_SPLITS = ("train", "validation")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_rank(seed: int, *parts: str) -> str:
    return hashlib.sha256(
        (str(seed) + "\0" + "\0".join(map(str, parts))).encode("utf-8")
    ).hexdigest()


def validate_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("MORPH manifest is empty")
    required = {
        "sample_id",
        "patient_sha256",
        "image_sha256",
        "image_path",
        "canonical_statement_id",
        "binary_label",
        "split",
        "bounding_boxes",
        "native_columns",
        "native_rows",
        "source_dataset",
        "source_split",
    }
    sample_ids: set[str] = set()
    image_ids: set[str] = set()
    patients_by_split = {split: set() for split in MORPH_SPLITS}
    counts: dict[str, dict[str, dict[int, int]]] = {
        split: {finding: {0: 0, 1: 0} for finding in MORPH_FINDINGS}
        for split in MORPH_SPLITS
    }
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"MORPH manifest row missing fields: {sorted(missing)}")
        if row["source_dataset"] != "Chest-ImaGenome-gold-v1.0.0":
            raise ValueError("MORPH gate accepts only the frozen Chest ImaGenome gold source")
        if row["source_split"] != "development_gold":
            raise ValueError("MORPH gate source split changed")
        split = str(row["split"])
        finding = str(row["canonical_statement_id"])
        label = int(row["binary_label"])
        if split not in MORPH_SPLITS or finding not in MORPH_FINDINGS or label not in (0, 1):
            raise ValueError("MORPH manifest split/finding/label changed")
        sample_id = str(row["sample_id"])
        image_hash = str(row["image_sha256"])
        patient_hash = str(row["patient_sha256"])
        if sample_id in sample_ids or image_hash in image_ids:
            raise ValueError("MORPH manifest identities must be globally image-disjoint")
        sample_ids.add(sample_id)
        image_ids.add(image_hash)
        patients_by_split[split].add(patient_hash)
        counts[split][finding][label] += 1
        boxes = list(row["bounding_boxes"])
        if label == 1 and not boxes:
            raise ValueError("MORPH positive rows require spatial concept boxes")
        if int(row["native_columns"]) <= 0 or int(row["native_rows"]) <= 0:
            raise ValueError("MORPH native dimensions must be positive")
        for box in boxes:
            if not (
                0 <= float(box["x_min"]) < float(box["x_max"]) <= int(row["native_columns"])
                and 0 <= float(box["y_min"]) < float(box["y_max"]) <= int(row["native_rows"])
            ):
                raise ValueError("MORPH concept box is out of bounds")
    overlap = patients_by_split["train"] & patients_by_split["validation"]
    if overlap:
        raise ValueError("MORPH train/validation patients overlap")
    for finding in MORPH_FINDINGS:
        expected = {"train": 3, "validation": 3}
        for split, per_label in expected.items():
            for label in (0, 1):
                if counts[split][finding][label] != per_label:
                    raise ValueError(
                        f"MORPH {split}/{finding}/{label} count changed: "
                        f"{counts[split][finding][label]} != {per_label}"
                    )
    return {
        "records": len(rows),
        "images": len(image_ids),
        "train_patients": len(patients_by_split["train"]),
        "validation_patients": len(patients_by_split["validation"]),
        "counts": counts,
        "patient_overlap": 0,
    }
