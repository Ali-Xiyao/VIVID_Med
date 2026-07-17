"""Prepare a fail-closed VinDr-CXR test consensus S/C intake manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PUBLIC_ROOT = Path(r"H:\Xiyao_Wang\000_Public Dataset")
DATASET_DIRNAME = "vindr-cxr-an-open-dataset-of-chest-x-rays-with-radiologists-annotations-1.0.0"
DEFAULT_DATASET_ROOT = PUBLIC_ROOT / DATASET_DIRNAME
DEFAULT_OUTPUT_DIR = Path("local_runs/bives_cxr/vindr_expert_sc_intake")
FINDINGS = {
    "pleural_effusion": {
        "column": "Pleural effusion",
        "statement_text": "A pleural effusion is present.",
    },
    "consolidation": {
        "column": "Consolidation",
        "statement_text": "Pulmonary consolidation is present.",
    },
    "pulmonary_edema": {
        "column": "Edema",
        "statement_text": "Pulmonary edema is present.",
    },
}
FORMAT_VERSION = "bives_vindr_expert_sc_intake_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-test-count", type=int, default=3000)
    parser.add_argument("--verify-all-image-sha256", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_official_hashes(path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        hashes[relative.strip().replace("\\", "/")] = digest.lower()
    return hashes


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _binary(value: str, *, image_id: str, column: str) -> int:
    if value not in {"0", "1"}:
        raise ValueError(f"non-binary VinDr test label {column}={value!r} for {image_id}")
    return int(value)


def prepare_intake(
    dataset_root: Path,
    output_dir: Path,
    *,
    expected_test_count: int = 3000,
    verify_all_image_sha256: bool = False,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    annotations_root = dataset_root / "annotations"
    label_path = annotations_root / "image_labels_test.csv"
    box_path = annotations_root / "annotations_test.csv"
    official_path = dataset_root / "SHA256SUMS.txt"
    marker_path = dataset_root / "_extraction_complete.json"
    for path in (label_path, box_path, official_path, marker_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    official_hashes = read_official_hashes(official_path)
    for relative in ("annotations/image_labels_test.csv", "annotations/annotations_test.csv"):
        if relative not in official_hashes:
            raise ValueError(f"official manifest is missing {relative}")
        actual = file_sha256(dataset_root / relative)
        if actual != official_hashes[relative]:
            raise ValueError(f"official metadata SHA-256 mismatch: {relative}")

    labels = read_csv(label_path)
    boxes = read_csv(box_path)
    if len(labels) != expected_test_count:
        raise ValueError(f"VinDr test labels have {len(labels)} rows; expected {expected_test_count}")
    image_ids = [str(row.get("image_id", "")) for row in labels]
    if "" in image_ids or len(image_ids) != len(set(image_ids)):
        raise ValueError("VinDr test image_id values must be non-empty and unique")

    boxes_by_image_finding: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    known_columns = {spec["column"] for spec in FINDINGS.values()}
    for row in boxes:
        image_id = str(row.get("image_id", ""))
        class_name = str(row.get("class_name", ""))
        if class_name not in known_columns:
            continue
        coordinates = {
            key: float(row[key]) for key in ("x_min", "y_min", "x_max", "y_max")
        }
        if not (
            coordinates["x_min"] < coordinates["x_max"]
            and coordinates["y_min"] < coordinates["y_max"]
        ):
            raise ValueError(f"invalid bounding box for {image_id}/{class_name}: {coordinates}")
        boxes_by_image_finding[(image_id, class_name)].append(coordinates)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    finding_summary: dict[str, dict[str, Any]] = {}
    verified_images = 0
    for canonical_id, spec in FINDINGS.items():
        column = spec["column"]
        values = [_binary(str(row[column]), image_id=str(row["image_id"]), column=column) for row in labels]
        positive = sum(values)
        negative = len(values) - positive
        eligible = positive > 0 and negative > 0
        positive_without_box = 0
        negative_with_box = 0
        if eligible:
            for source, value in zip(labels, values, strict=True):
                image_id = str(source["image_id"])
                finding_boxes = boxes_by_image_finding[(image_id, column)]
                if value == 1 and not finding_boxes:
                    positive_without_box += 1
                if value == 0 and finding_boxes:
                    negative_with_box += 1
            if positive_without_box or negative_with_box:
                raise ValueError(
                    f"label/box inconsistency for {canonical_id}: "
                    f"positive_without_box={positive_without_box}, negative_with_box={negative_with_box}"
                )
        finding_summary[canonical_id] = {
            "source_column": column,
            "positive": positive,
            "negative": negative,
            "eligible_sc": eligible,
            "ineligible_reason": None if eligible else "degenerate_binary_test_label",
            "positive_without_box": positive_without_box,
            "negative_with_box": negative_with_box,
        }
        if not eligible:
            continue
        for source, value in zip(labels, values, strict=True):
            image_id = str(source["image_id"])
            relative = f"test/{image_id}.dicom"
            image_path = dataset_root / relative
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            official_image_sha256 = official_hashes.get(relative)
            if not official_image_sha256:
                raise ValueError(f"official manifest is missing {relative}")
            actual_verified = False
            if verify_all_image_sha256:
                if file_sha256(image_path) != official_image_sha256:
                    raise ValueError(f"image SHA-256 mismatch: {relative}")
                actual_verified = True
                verified_images += 1
            rows.append(
                {
                    "sample_id": f"vindr-test::{image_id}::{canonical_id}",
                    "unit_id": image_id,
                    "patient_id": None,
                    "patient_id_status": "not_provided_by_public_release",
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "official_image_sha256": official_image_sha256,
                    "actual_image_sha256_verified": actual_verified,
                    "canonical_statement_id": canonical_id,
                    "statement_text": spec["statement_text"],
                    "state": "support" if value == 1 else "contradict",
                    "binary_label": value,
                    "bounding_boxes": boxes_by_image_finding[(image_id, column)],
                    "label_source": "vindr_cxr_test_consensus_5_radiologists",
                    "annotation_status": "expert_consensus",
                    "source_dataset": "VinDr-CXR-v1.0.0",
                    "source_split": "test",
                    "expert_axis": "statement_polarity_sc",
                    "contradict_definition": "consensus_binary_negative_for_same_finding",
                    "four_state_claim": False,
                    "formal_result": False,
                }
            )

    manifest_path = output_dir / "vindr_test_expert_sc.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    lock = {
        "format_version": FORMAT_VERSION,
        "status": "intake_ready",
        "formal_result": False,
        "four_state_claim": False,
        "patient_level_ci_ready": False,
        "patient_level_ci_blocker": "VinDr public release does not expose a patient identifier",
        "source_split": "test_consensus_only",
        "label_source": "consensus of five radiologists",
        "manifest_sha256": file_sha256(manifest_path),
        "official_manifest_sha256": file_sha256(official_path),
        "image_labels_test_sha256": file_sha256(label_path),
        "annotations_test_sha256": file_sha256(box_path),
        "verify_all_image_sha256": verify_all_image_sha256,
        "verified_image_hash_count": verified_images,
        "findings": finding_summary,
    }
    lock_path = output_dir / "vindr_test_expert_sc_lock.json"
    lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "status": "intake_ready",
        "format_version": FORMAT_VERSION,
        "formal_result": False,
        "records": len(rows),
        "source_images": len(labels),
        "eligible_findings": [key for key, value in finding_summary.items() if value["eligible_sc"]],
        "ineligible_findings": [key for key, value in finding_summary.items() if not value["eligible_sc"]],
        "findings": finding_summary,
        "manifest": str(manifest_path),
        "lock": str(lock_path),
        "patient_level_ci_ready": False,
        "patient_level_ci_blocker": lock["patient_level_ci_blocker"],
        "claim_boundary": "expert S/C test intake only; no U/I or four-state claim",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = prepare_intake(
        args.dataset_root,
        args.output_dir,
        expected_test_count=args.expected_test_count,
        verify_all_image_sha256=args.verify_all_image_sha256,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
