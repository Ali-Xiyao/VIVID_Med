#!/usr/bin/env python
"""Freeze a new patient-disjoint Chest ImaGenome gold development surface."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from morph_cxr.protocol import (  # noqa: E402
    MORPH_FINDINGS,
    canonical_sha256,
    file_sha256,
    stable_rank,
    validate_manifest,
)


DEFAULT_ARCHIVE = Path(r"E:\Xiyaowang\chest-imagenome-dataset-1.0.0.zip")
DEFAULT_IMAGES = Path(
    r"H:\xiyao\dataset\MIMIC-CXR\mimic-cxr\mimic-cxr\mimic-cxr-images"
)
DEFAULT_OUTPUT = ROOT / "local_runs/morph_cxr/separability_v0_data"
GOLD_MEMBER = (
    "chest-imagenome-dataset-1.0.0/gold_dataset/"
    "gold_object_attribute_with_coordinates.txt"
)
BBOX_MEMBER = (
    "chest-imagenome-dataset-1.0.0/gold_dataset/"
    "gold_bbox_coordinate_annotations_1000images.csv"
)
LICENSE_MEMBER = "chest-imagenome-dataset-1.0.0/LICENSE.txt"
LABELS = {
    "pneumothorax": "pneumothorax",
    "consolidation": "consolidation",
    "pleural_effusion": "pleural effusion",
    "cardiomegaly": "enlarged cardiac silhouette",
}
STATEMENTS = {
    "pneumothorax": "Pneumothorax is present.",
    "consolidation": "Pulmonary consolidation is present.",
    "pleural_effusion": "Pleural effusion is present.",
    "cardiomegaly": "Cardiomegaly is present.",
}
SELECTION_ORDER = (
    "pneumothorax",
    "consolidation",
    "pleural_effusion",
    "cardiomegaly",
)


def _member_sha256(archive: zipfile.ZipFile, member: str) -> str:
    digest = hashlib.sha256()
    with archive.open(member) as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _historical_patient_ids(root: Path, *, exclude_root: Path | None = None) -> set[str]:
    result: set[str] = set()
    suffixes = {".jsonl", ".json", ".csv", ".md", ".yaml", ".yml"}
    pattern = re.compile(r"\bp?(\d{8})\b")
    if not root.is_dir():
        return result
    excluded = exclude_root.resolve() if exclude_root is not None else None
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if excluded is not None and (path.resolve() == excluded or excluded in path.resolve().parents):
            continue
        if path.stat().st_size > 100_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        result.update(match.group(1) for match in pattern.finditer(text))
    return result


def _parse_box(value: str) -> dict[str, float]:
    coords = ast.literal_eval(value)
    if not isinstance(coords, list) or len(coords) != 4:
        raise ValueError("Chest ImaGenome coordinate format changed")
    x_min, y_min, x_max, y_max = map(float, coords)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Chest ImaGenome coordinate is invalid")
    return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}


def _select(
    candidates: list[dict[str, Any]],
    count: int,
    *,
    used_patients: set[str],
    seed: int,
    key: str,
) -> list[dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda row: (
            stable_rank(seed, key, str(row["patient_id"]), str(row["image_id"])),
            str(row["patient_id"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    for row in ordered:
        patient = str(row["patient_id"])
        if patient in used_patients:
            continue
        selected.append(row)
        used_patients.add(patient)
        if len(selected) == count:
            return selected
    raise ValueError(f"cannot select {count} patient-disjoint rows for {key}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--history-root", type=Path, default=ROOT / "local_runs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()
    if not args.archive.is_file() or not args.images_root.is_dir():
        raise FileNotFoundError("Chest ImaGenome archive or MIMIC image root is missing")

    historical = _historical_patient_ids(
        args.history_root,
        exclude_root=args.output_dir,
    )
    with zipfile.ZipFile(args.archive) as archive:
        required = {GOLD_MEMBER, BBOX_MEMBER, LICENSE_MEMBER}
        if not required.issubset(archive.namelist()):
            raise ValueError("Chest ImaGenome gold members changed")
        gold_bytes = archive.read(GOLD_MEMBER)
        bbox_bytes = archive.read(BBOX_MEMBER)
        gold_rows = list(
            csv.DictReader(
                io.StringIO(gold_bytes.decode("utf-8")),
                delimiter="\t",
            )
        )
        bbox_rows = list(csv.DictReader(io.StringIO(bbox_bytes.decode("utf-8"))))
        source_members = {
            GOLD_MEMBER: hashlib.sha256(gold_bytes).hexdigest(),
            BBOX_MEMBER: hashlib.sha256(bbox_bytes).hexdigest(),
            LICENSE_MEMBER: _member_sha256(archive, LICENSE_MEMBER),
        }

    contexts: dict[tuple[str, str], set[int]] = defaultdict(set)
    grouped: dict[tuple[str, str, int], dict[str, Any]] = {}
    invalid_images: set[str] = set()
    for row in gold_rows:
        normalized_label = str(row["label_name"]).strip().lower()
        finding = next((key for key, value in LABELS.items() if value == normalized_label), None)
        if finding is None:
            continue
        context = str(row["context"]).strip().lower()
        if context not in {"yes", "no"}:
            continue
        label = int(context == "yes")
        image_id = str(row["image_id"]).removesuffix(".dcm")
        patient_id = str(row["patient_id"])
        study_id = str(row["study_id"])
        contexts[(image_id, finding)].add(label)
        key = (image_id, finding, label)
        block = grouped.setdefault(
            key,
            {
                "image_id": image_id,
                "patient_id": patient_id,
                "study_id": study_id,
                "finding": finding,
                "binary_label": label,
                "bounding_boxes": [],
                "regions": set(),
            },
        )
        if block["patient_id"] != patient_id or block["study_id"] != study_id:
            raise ValueError("Chest ImaGenome image identity changed")
        try:
            box = _parse_box(str(row["coord224"]))
        except (ValueError, SyntaxError):
            invalid_images.add(image_id)
            continue
        block["bounding_boxes"].append(box)
        block["regions"].add(str(row["bbox"]))
    ambiguous = [key for key, values in contexts.items() if values == {0, 1}]
    ambiguous_images = {image_id for image_id, _finding in ambiguous}

    candidates: dict[str, dict[int, list[dict[str, Any]]]] = {
        finding: {0: [], 1: []} for finding in MORPH_FINDINGS
    }
    for block in grouped.values():
        if (
            str(block["image_id"]) in invalid_images
            or str(block["image_id"]) in ambiguous_images
            or not block["bounding_boxes"]
        ):
            continue
        patient_id = str(block["patient_id"])
        if patient_id in historical:
            continue
        image_id = str(block["image_id"])
        prefix = patient_id[:2]
        image_path = (
            args.images_root
            / "files"
            / f"p{prefix}"
            / f"p{patient_id}"
            / f"s{block['study_id']}"
            / f"{image_id}.jpg"
        )
        if not image_path.is_file():
            continue
        with Image.open(image_path) as image:
            native_width, native_height = map(int, image.size)
        if any(
            not (
                0 <= box["x_min"] < box["x_max"] <= native_width
                and 0 <= box["y_min"] < box["y_max"] <= native_height
            )
            for box in block["bounding_boxes"]
        ):
            continue
        block["image_path"] = str(image_path.resolve())
        block["native_columns"] = native_width
        block["native_rows"] = native_height
        block["regions"] = sorted(block["regions"])
        candidates[str(block["finding"])][int(block["binary_label"])].append(block)

    used_patients = set(historical)
    selected: list[dict[str, Any]] = []
    for finding in SELECTION_ORDER:
        for split, count in (("train", 3), ("validation", 3)):
            for label in (1, 0):
                rows = _select(
                    candidates[finding][label],
                    count,
                    used_patients=used_patients,
                    seed=args.seed,
                    key=f"{finding}:{split}:{label}",
                )
                for row in rows:
                    image_path = Path(row["image_path"])
                    image_hash = file_sha256(image_path)
                    patient_hash = hashlib.sha256(
                        f"morph-v0-patient:{row['patient_id']}".encode("utf-8")
                    ).hexdigest()
                    sample_key = (
                        f"{split}:{finding}:{label}:{patient_hash}:{image_hash}"
                    )
                    selected.append(
                        {
                            "sample_id": f"morph-v0::{hashlib.sha256(sample_key.encode()).hexdigest()}",
                            "patient_sha256": patient_hash,
                            "image_sha256": image_hash,
                            "image_path": str(image_path),
                            "canonical_statement_id": finding,
                            "statement_text": STATEMENTS[finding],
                            "binary_label": label,
                            "state": "support" if label else "contradict",
                            "split": split,
                            "bounding_boxes": row["bounding_boxes"] if label else [],
                            "positive_regions": row["regions"] if label else [],
                            "native_columns": row["native_columns"],
                            "native_rows": row["native_rows"],
                            "source_dataset": "Chest-ImaGenome-gold-v1.0.0",
                            "source_split": "development_gold",
                            "patient_disjoint": True,
                            "historical_patient_excluded": True,
                            "selection_seed": int(args.seed),
                        }
                    )

    selected.sort(key=lambda row: str(row["sample_id"]))
    summary = validate_manifest(selected)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "morph_separability_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
        newline="\n",
    )
    lock = {
        "schema_version": "morph-separability-data-lock-v1",
        "status": "complete",
        "formal_result": False,
        "confirmatory_evidence": False,
        "source_dataset": "Chest-ImaGenome-gold-v1.0.0",
        "source_role": "exposed_development_only",
        "archive_path": str(args.archive.resolve()),
        "archive_size": args.archive.stat().st_size,
        "source_member_sha256": source_members,
        "manifest_sha256": file_sha256(manifest_path),
        "selected_image_hashes_sha256": canonical_sha256(
            sorted(str(row["image_sha256"]) for row in selected)
        ),
        "historical_patient_ids_excluded": len(historical),
        "invalid_coordinate_images_excluded": len(invalid_images),
        "ambiguous_context_images_excluded": len(ambiguous_images),
        "selection_seed": int(args.seed),
        "summary": summary,
        "chexlocalize_test_opened": False,
        "vindr_test_opened": False,
    }
    lock["canonical_sha256"] = canonical_sha256(lock)
    (args.output_dir / "data_lock.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(lock, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
