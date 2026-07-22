"""Freeze new image-disjoint VinDr-train roles for VICER V0."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pydicom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_bives_vindr_rescue_dev import rectangle_union_area  # noqa: E402
from vicer_cxr.validity import (  # noqa: E402
    VALIDITY_FINDINGS,
    VALIDITY_ROLES,
    canonical_sha256,
    file_sha256,
    stable_rank,
    validate_v0_manifest,
)


DATASET_ROOT = Path(
    r"H:\Xiyao_Wang\000_Public Dataset\vindr-cxr-an-open-dataset-of-chest-x-rays-with-radiologists-annotations-1.0.0"
)
OUTPUT_DIR = ROOT / "local_runs/vicer_cxr/v0_vindr_validity"
SEED = 20260722
FINDING_SPECS = {
    "pneumothorax": ("Pneumothorax", "A pneumothorax is present."),
    "consolidation": ("Consolidation", "Pulmonary consolidation is present."),
    "pleural_effusion": ("Pleural effusion", "A pleural effusion is present."),
    "cardiomegaly": ("Cardiomegaly", "Cardiomegaly is present."),
}
ARISE_MANIFESTS = (
    ROOT / "local_runs/arise_cxr/vindr_box_sc_v1/vindr_box_sc_train.jsonl",
    ROOT / "local_runs/arise_cxr/vindr_box_sc_v1/vindr_box_sc_val.jsonl",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_official_hashes(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, relative = line.split(maxsplit=1)
            result[relative.strip().replace("\\", "/")] = digest.lower()
    return result


def normalized_boxes(
    boxes: list[dict[str, Any]], width: int, height: int
) -> list[dict[str, float | str]]:
    return [
        {
            "x_min": float(box["x_min"]) / width,
            "y_min": float(box["y_min"]) / height,
            "x_max": float(box["x_max"]) / width,
            "y_max": float(box["y_max"]) / height,
            "rad_id": str(box["rad_id"]),
        }
        for box in boxes
    ]


def denormalize_boxes(
    boxes: list[dict[str, Any]], width: int, height: int
) -> list[dict[str, float | str]]:
    return [
        {
            "x_min": float(box["x_min"]) * width,
            "y_min": float(box["y_min"]) * height,
            "x_max": float(box["x_max"]) * width,
            "y_max": float(box["y_max"]) * height,
            "rad_id": str(box["rad_id"]),
        }
        for box in boxes
    ]


def select_distinct(
    candidates: list[dict[str, Any]],
    count: int,
    *,
    used_images: set[str],
    seed: int,
    key: str,
) -> list[dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda row: (stable_rank(seed, key, str(row["image_id"])), str(row["image_id"])),
    )
    selected = []
    for row in ordered:
        image_id = str(row["image_id"])
        if image_id in used_images:
            continue
        selected.append(row)
        used_images.add(image_id)
        if len(selected) == count:
            return selected
    raise ValueError(f"cannot select {count} distinct rows for {key}; found {len(selected)}")


def select_eval_quartiles(
    candidates: list[dict[str, Any]],
    *,
    used_images: set[str],
    seed: int,
    finding: str,
) -> list[dict[str, Any]]:
    selected = []
    for quartile in range(1, 5):
        subset = [row for row in candidates if int(row["box_area_quartile"]) == quartile]
        selected.extend(
            select_distinct(
                subset,
                2,
                used_images=used_images,
                seed=seed,
                key=f"{finding}:validity_eval:q{quartile}",
            )
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--hash-workers", type=int, default=8)
    args = parser.parse_args()
    if args.hash_workers <= 0:
        raise ValueError("hash-workers must be positive")

    dataset_root = args.dataset_root.resolve()
    annotation_dir = dataset_root / "annotations"
    labels_path = annotation_dir / "image_labels_train.csv"
    boxes_path = annotation_dir / "annotations_train.csv"
    hashes_path = dataset_root / "SHA256SUMS.txt"
    official = read_official_hashes(hashes_path)
    for relative, path in (
        ("annotations/image_labels_train.csv", labels_path),
        ("annotations/annotations_train.csv", boxes_path),
    ):
        if official.get(relative) != file_sha256(path):
            raise ValueError(f"official VinDr metadata hash mismatch: {relative}")

    arise_used: set[str] = set()
    for path in ARISE_MANIFESTS:
        if not path.is_file():
            raise FileNotFoundError(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                arise_used.add(str(json.loads(line)["image_id"]))

    labels_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(labels_path):
        labels_by_image[str(row["image_id"])].append(row)
    if len(labels_by_image) != 15000:
        raise ValueError("VinDr train image count changed")

    class_to_finding = {spec[0]: finding for finding, spec in FINDING_SPECS.items()}
    boxes_by_key_reader: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in read_csv(boxes_path):
        finding = class_to_finding.get(str(row["class_name"]))
        if finding is None:
            continue
        box = {
            key: float(row[key]) for key in ("x_min", "y_min", "x_max", "y_max")
        }
        if box["x_max"] <= box["x_min"] or box["y_max"] <= box["y_min"]:
            raise ValueError("invalid VinDr expert box")
        box["rad_id"] = str(row["rad_id"])
        boxes_by_key_reader[(str(row["image_id"]), finding, str(row["rad_id"]))].append(box)

    metadata: dict[str, tuple[int, int]] = {}
    def image_dimensions(image_id: str) -> tuple[int, int]:
        if image_id not in metadata:
            image_path = dataset_root / "train" / f"{image_id}.dicom"
            header = pydicom.dcmread(
                image_path,
                stop_before_pixels=True,
                specific_tags=["Rows", "Columns"],
            )
            metadata[image_id] = (int(header.Columns), int(header.Rows))
        return metadata[image_id]

    positives: dict[str, list[dict[str, Any]]] = defaultdict(list)
    negatives: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image_id, readers in labels_by_image.items():
        if image_id in arise_used:
            continue
        if len(readers) != 3 or len({str(row["rad_id"]) for row in readers}) != 3:
            raise ValueError("VinDr train reader multiplicity changed")
        for finding, (column, statement) in FINDING_SPECS.items():
            votes = {str(row["rad_id"]): int(row[column]) for row in readers}
            vote_count = sum(votes.values())
            if vote_count == 0:
                if any(boxes_by_key_reader[(image_id, finding, reader)] for reader in votes):
                    raise ValueError("0-of-3 negative unexpectedly has a target box")
                negatives[finding].append(
                    {
                        "image_id": image_id,
                        "statement_text": statement,
                        "reader_votes": votes,
                        "positive_vote_count": 0,
                        "bounding_boxes": [],
                    }
                )
                continue
            positive_boxes = []
            for reader, vote in votes.items():
                if vote:
                    reader_boxes = boxes_by_key_reader[(image_id, finding, reader)]
                    if not reader_boxes:
                        raise ValueError("positive VinDr reader lacks a box")
                    positive_boxes.extend(reader_boxes)
            width, height = image_dimensions(image_id)
            area = rectangle_union_area(positive_boxes, width, height)
            positives[finding].append(
                {
                    "image_id": image_id,
                    "statement_text": statement,
                    "reader_votes": votes,
                    "positive_vote_count": vote_count,
                    "native_columns": width,
                    "native_rows": height,
                    "bounding_boxes": positive_boxes,
                    "box_area_fraction": float(area / (width * height)),
                }
            )

    for finding in VALIDITY_FINDINGS:
        ordered = sorted(
            positives[finding], key=lambda row: (row["box_area_fraction"], row["image_id"])
        )
        for rank, row in enumerate(ordered):
            row["box_area_quartile"] = min(4, (rank * 4) // len(ordered) + 1)

    selected: list[dict[str, Any]] = []
    used_images = set(arise_used)
    for finding in VALIDITY_FINDINGS:
        positive_by_role: dict[str, list[dict[str, Any]]] = {}
        positive_by_role["validity_eval"] = select_eval_quartiles(
            positives[finding], used_images=used_images, seed=args.seed, finding=finding
        )
        for role in ("critic_train", "critic_calibration", "verifier_train", "verifier_calibration"):
            positive_by_role[role] = select_distinct(
                positives[finding],
                VALIDITY_ROLES[role],
                used_images=used_images,
                seed=args.seed,
                key=f"{finding}:{role}:support",
            )
        for role in ("critic_train", "critic_calibration", "verifier_train", "verifier_calibration"):
            negative_rows = select_distinct(
                negatives[finding],
                VALIDITY_ROLES[role],
                used_images=used_images,
                seed=args.seed,
                key=f"{finding}:{role}:contradict",
            )
            templates = positive_by_role[role]
            for index, row in enumerate(negative_rows):
                width, height = image_dimensions(str(row["image_id"]))
                row["native_columns"] = width
                row["native_rows"] = height
                template = templates[index]
                row["bounding_boxes"] = denormalize_boxes(
                    normalized_boxes(
                        template["bounding_boxes"],
                        int(template["native_columns"]),
                        int(template["native_rows"]),
                    ),
                    int(row["native_columns"]),
                    int(row["native_rows"]),
                )
                row["roi_template_source_image_id"] = str(template["image_id"])
            positive_by_role[f"{role}:negative"] = negative_rows

        for role in VALIDITY_ROLES:
            for state, role_rows in (
                ("support", positive_by_role[role]),
                ("contradict", positive_by_role.get(f"{role}:negative", [])),
            ):
                for row in role_rows:
                    image_id = str(row["image_id"])
                    image_path = dataset_root / "train" / f"{image_id}.dicom"
                    selected.append(
                        {
                            "sample_id": f"vicer-v0::{role}::{image_id}::{finding}::{state}",
                            "unit_id": image_id,
                            "image_id": image_id,
                            "patient_id": None,
                            "patient_id_status": "not_provided_by_public_release",
                            "patient_level_claim": False,
                            "source_dataset": "VinDr-CXR-v1.0.0",
                            "source_split": "train",
                            "image_path": str(image_path),
                            "canonical_statement_id": finding,
                            "statement_text": row["statement_text"],
                            "state": state,
                            "binary_label": int(state == "support"),
                            "v0_role": role,
                            "reader_votes": row["reader_votes"],
                            "positive_vote_count": int(row["positive_vote_count"]),
                            "reader_consensus": f"{row['positive_vote_count']}_of_3",
                            "roi_boxes": row["bounding_boxes"],
                            "roi_template_source_image_id": row.get("roi_template_source_image_id"),
                            "box_area_fraction": row.get("box_area_fraction"),
                            "box_area_quartile": row.get("box_area_quartile"),
                            "native_columns": int(row["native_columns"]),
                            "native_rows": int(row["native_rows"]),
                            "arise_identity_excluded": True,
                            "selection_seed": int(args.seed),
                        }
                    )

    selected.sort(key=lambda row: str(row["sample_id"]))
    image_ids = sorted({str(row["image_id"]) for row in selected})

    def verify_image(image_id: str) -> tuple[str, str]:
        path = dataset_root / "train" / f"{image_id}.dicom"
        expected = official.get(f"train/{image_id}.dicom")
        if not expected:
            raise ValueError(f"official image hash missing: {image_id}")
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"official image hash mismatch: {image_id}")
        return image_id, actual

    with ThreadPoolExecutor(max_workers=args.hash_workers) as executor:
        actual_hashes = dict(executor.map(verify_image, image_ids))
    for row in selected:
        row["image_sha256"] = actual_hashes[str(row["image_id"])]
        row["actual_image_sha256_verified"] = True

    summary = validate_v0_manifest(selected)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "vicer_v0_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
        newline="\n",
    )
    lock = {
        "schema_version": "vicer-v0-vindr-data-lock-v1",
        "status": "complete",
        "formal_result": False,
        "source_split": "VinDr-CXR train only",
        "image_disjoint_only": True,
        "patient_level_claim": False,
        "arise_excluded_image_count": len(arise_used),
        "arise_excluded_image_set_sha256": canonical_sha256(sorted(arise_used)),
        "selection_seed": int(args.seed),
        "finding_order": list(VALIDITY_FINDINGS),
        "role_positive_counts": VALIDITY_ROLES,
        "manifest_sha256": file_sha256(manifest_path),
        "metadata_sha256": {
            "image_labels_train.csv": file_sha256(labels_path),
            "annotations_train.csv": file_sha256(boxes_path),
        },
        "summary": summary,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "vindr_test_opened": False,
        "chexlocalize_test_opened": False,
        "clinical_claim": "none",
        "single_reader_positive_allowed": True,
        "single_reader_boundary": (
            "Unused consolidation and pleural-effusion images contain only 1-of-3 positives; "
            "V0 is an exploratory intervention-validity development gate, not clinical evidence."
        ),
    }
    lock["canonical_sha256"] = canonical_sha256(lock)
    (args.output_dir / "vicer_v0_data_lock.json").write_text(
        json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(lock, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
