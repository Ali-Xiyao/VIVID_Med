"""Build the locked image-disjoint VinDr-train rescue development manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pydicom

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bives_cxr.rescue_protocol import (  # noqa: E402
    RESCUE_SPLIT_VERSION,
    deterministic_multilabel_half_split,
    stable_hash_int,
)


PUBLIC_ROOT = Path(r"H:\Xiyao_Wang\000_Public Dataset")
DATASET_DIRNAME = "vindr-cxr-an-open-dataset-of-chest-x-rays-with-radiologists-annotations-1.0.0"
DEFAULT_DATASET_ROOT = PUBLIC_ROOT / DATASET_DIRNAME
DEFAULT_OUTPUT_DIR = Path("local_runs/bives_cxr/vindr_rescue_dev")
FORMAT_VERSION = "bives_vindr_train_rescue_dev_v1"
SPLIT_SEED = 20260718
FINDINGS = {
    "consolidation": {
        "column": "Consolidation",
        "statement_text": "Pulmonary consolidation is present.",
    },
    "pleural_effusion": {
        "column": "Pleural effusion",
        "statement_text": "A pleural effusion is present.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hash-workers", type=int, default=8)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_official_hashes(path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        hashes[relative.strip().replace("\\", "/")] = digest.lower()
    return hashes


def rectangle_union_area(boxes: list[dict[str, Any]], width: int, height: int) -> float:
    clipped = []
    for box in boxes:
        x0 = max(0.0, min(float(width), float(box["x_min"])))
        y0 = max(0.0, min(float(height), float(box["y_min"])))
        x1 = max(0.0, min(float(width), float(box["x_max"])))
        y1 = max(0.0, min(float(height), float(box["y_max"])))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"invalid clipped expert box: {box}")
        clipped.append((x0, y0, x1, y1))
    xs = sorted({value for box in clipped for value in (box[0], box[2])})
    area = 0.0
    for left, right in zip(xs, xs[1:], strict=False):
        intervals = sorted(
            (y0, y1) for x0, y0, x1, y1 in clipped if x0 < right and x1 > left
        )
        if not intervals:
            continue
        covered = 0.0
        start, end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start > end:
                covered += end - start
                start, end = next_start, next_end
            else:
                end = max(end, next_end)
        covered += end - start
        area += (right - left) * covered
    return area


def assign_area_quartiles(positive_rows: dict[tuple[str, str], dict[str, Any]]) -> None:
    for finding in FINDINGS:
        subset = [row for (image_id, key), row in positive_rows.items() if key == finding]
        ordered = sorted(subset, key=lambda row: (row["box_area_fraction"], row["image_id"]))
        for rank, row in enumerate(ordered):
            row["box_area_quartile"] = min(4, (rank * 4) // len(ordered) + 1)


def source_record(path: Path, expected_sha256: str) -> dict[str, Any]:
    if path.parent.name != "train" or path.suffix != ".dicom":
        raise ValueError(f"rescue manifest attempted a non-train image path: {path}")
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"official DICOM SHA-256 mismatch: {path}")
    metadata = pydicom.dcmread(
        path,
        stop_before_pixels=True,
        specific_tags=["Rows", "Columns", "PatientID", "StudyInstanceUID", "SeriesInstanceUID"],
    )
    grouping = {
        "patient_id": str(getattr(metadata, "PatientID", "")).strip() or None,
        "study_instance_uid": str(getattr(metadata, "StudyInstanceUID", "")).strip() or None,
        "series_instance_uid": str(getattr(metadata, "SeriesInstanceUID", "")).strip() or None,
    }
    if any(value is not None for value in grouping.values()):
        raise ValueError(f"unexpected grouping identifier appeared in VinDr train: {path}")
    return {
        "image_id": path.stem,
        "actual_image_sha256": actual,
        "rows": int(metadata.Rows),
        "columns": int(metadata.Columns),
        **grouping,
    }


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_rescue_dev(
    dataset_root: Path,
    output_dir: Path,
    *,
    hash_workers: int = 8,
    split_seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    if hash_workers <= 0:
        raise ValueError("hash_workers must be positive")
    dataset_root = dataset_root.resolve()
    annotations_root = dataset_root / "annotations"
    label_path = annotations_root / "image_labels_train.csv"
    box_path = annotations_root / "annotations_train.csv"
    official_path = dataset_root / "SHA256SUMS.txt"
    for path in (label_path, box_path, official_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    official_hashes = read_official_hashes(official_path)
    for relative, path in (
        ("annotations/image_labels_train.csv", label_path),
        ("annotations/annotations_train.csv", box_path),
    ):
        if official_hashes.get(relative) != file_sha256(path):
            raise ValueError(f"official metadata SHA-256 mismatch: {relative}")

    label_rows = read_csv(label_path)
    box_rows = read_csv(box_path)
    labels_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in label_rows:
        labels_by_image[str(row["image_id"])].append(row)
    if len(labels_by_image) != 15000:
        raise ValueError(f"VinDr train image count is {len(labels_by_image)}; expected 15000")
    for image_id, rows in labels_by_image.items():
        readers = [str(row["rad_id"]) for row in rows]
        if len(rows) != 3 or len(set(readers)) != 3:
            raise ValueError(f"VinDr train must have exactly three unique readers: {image_id}")

    columns_to_finding = {spec["column"]: finding for finding, spec in FINDINGS.items()}
    boxes_by_key_reader: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in box_rows:
        finding = columns_to_finding.get(str(row["class_name"]))
        if finding is None:
            continue
        values = {key: str(row[key]).strip() for key in ("x_min", "y_min", "x_max", "y_max")}
        if any(not value for value in values.values()):
            raise ValueError(f"positive finding row has empty coordinates: {row}")
        box = {key: float(value) for key, value in values.items()}
        if box["x_max"] <= box["x_min"] or box["y_max"] <= box["y_min"]:
            raise ValueError(f"invalid VinDr train box: {row}")
        box["rad_id"] = str(row["rad_id"])
        boxes_by_key_reader[(str(row["image_id"]), finding, str(row["rad_id"]))].append(box)

    vote_records: dict[tuple[str, str], dict[str, Any]] = {}
    positive_rows: dict[tuple[str, str], dict[str, Any]] = {}
    disagreements: Counter[str] = Counter()
    negatives: dict[str, list[str]] = defaultdict(list)
    for image_id, readers in labels_by_image.items():
        image_path = dataset_root / "train" / f"{image_id}.dicom"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        metadata = pydicom.dcmread(
            image_path, stop_before_pixels=True, specific_tags=["Rows", "Columns"]
        )
        width, height = int(metadata.Columns), int(metadata.Rows)
        for finding, spec in FINDINGS.items():
            votes = {str(row["rad_id"]): int(row[spec["column"]]) for row in readers}
            if any(value not in {0, 1} for value in votes.values()):
                raise ValueError(f"non-binary train vote: {image_id}/{finding}")
            positive_voters = sorted(reader for reader, value in votes.items() if value == 1)
            vote_count = len(positive_voters)
            record = {"reader_votes": votes, "positive_vote_count": vote_count}
            vote_records[(image_id, finding)] = record
            if vote_count == 0:
                if any(boxes_by_key_reader[(image_id, finding, reader)] for reader in votes):
                    raise ValueError(f"0-of-3 negative has a finding box: {image_id}/{finding}")
                negatives[finding].append(image_id)
                continue
            if vote_count == 1:
                disagreements[finding] += 1
                continue
            boxes: list[dict[str, Any]] = []
            for reader in positive_voters:
                reader_boxes = boxes_by_key_reader[(image_id, finding, reader)]
                if not reader_boxes:
                    raise ValueError(f"positive reader lacks a box: {image_id}/{finding}/{reader}")
                boxes.extend(reader_boxes)
            area = rectangle_union_area(boxes, width, height)
            positive_rows[(image_id, finding)] = {
                "image_id": image_id,
                "canonical_statement_id": finding,
                "statement_text": spec["statement_text"],
                "reader_votes": votes,
                "positive_vote_count": vote_count,
                "reader_consensus": f"{vote_count}_of_3",
                "bounding_boxes": boxes,
                "box_area_fraction": float(area / (width * height)),
                "native_rows": height,
                "native_columns": width,
            }
    assign_area_quartiles(positive_rows)

    strata_by_image: dict[str, list[str]] = defaultdict(list)
    for (image_id, finding), row in positive_rows.items():
        strata_by_image[image_id].append(
            f"{finding}|{row['reader_consensus']}|q{row['box_area_quartile']}"
        )
    positive_assignment, split_audit = deterministic_multilabel_half_split(
        strata_by_image, seed=split_seed
    )

    positive_counts: dict[str, Counter[str]] = {finding: Counter() for finding in FINDINGS}
    for (image_id, finding), row in positive_rows.items():
        split = positive_assignment[image_id]
        row["rescue_split"] = split
        positive_counts[finding][split] += 1

    unit_assignment = dict(positive_assignment)
    selected_negative: dict[tuple[str, str], list[str]] = {}
    for finding in FINDINGS:
        for split in ("protocol_design", "rescue_confirm"):
            candidates = []
            for image_id in negatives[finding]:
                assigned = unit_assignment.get(image_id)
                if assigned is None:
                    assigned = (
                        "protocol_design"
                        if stable_hash_int(f"{split_seed}:negative-split:{image_id}") % 2 == 0
                        else "rescue_confirm"
                    )
                if assigned == split:
                    candidates.append(image_id)
            candidates.sort(
                key=lambda image_id: (
                    stable_hash_int(f"{split_seed}:negative:{finding}:{split}:{image_id}"),
                    image_id,
                )
            )
            required = positive_counts[finding][split]
            if len(candidates) < required:
                raise ValueError(f"not enough {finding}/{split} negatives")
            selected_negative[(finding, split)] = candidates[:required]
            for image_id in candidates[:required]:
                existing = unit_assignment.get(image_id)
                if existing is not None and existing != split:
                    raise AssertionError("negative sampling created split overlap")
                unit_assignment[image_id] = split

    selected_units = sorted(unit_assignment)
    official_selected: dict[str, str] = {}
    for image_id in selected_units:
        relative = f"train/{image_id}.dicom"
        if relative not in official_hashes:
            raise ValueError(f"official manifest is missing {relative}")
        official_selected[image_id] = official_hashes[relative]
    with ThreadPoolExecutor(max_workers=hash_workers) as pool:
        source_records = list(
            pool.map(
                lambda image_id: source_record(
                    dataset_root / "train" / f"{image_id}.dicom", official_selected[image_id]
                ),
                selected_units,
            )
        )
    source_by_image = {record["image_id"]: record for record in source_records}

    manifest_rows: list[dict[str, Any]] = []
    for (image_id, finding), row in positive_rows.items():
        source = source_by_image[image_id]
        manifest_rows.append(
            {
                "sample_id": f"vindr-train::{image_id}::{finding}",
                "unit_id": image_id,
                "patient_id": None,
                "patient_id_status": "not_provided_by_public_release",
                "image_id": image_id,
                "image_path": str(dataset_root / "train" / f"{image_id}.dicom"),
                "official_image_sha256": official_selected[image_id],
                "actual_image_sha256": source["actual_image_sha256"],
                "actual_image_sha256_verified": True,
                "native_rows": source["rows"],
                "native_columns": source["columns"],
                "canonical_statement_id": finding,
                "statement_text": row["statement_text"],
                "state": "support",
                "binary_label": 1,
                "bounding_boxes": row["bounding_boxes"],
                "reader_votes": row["reader_votes"],
                "positive_vote_count": row["positive_vote_count"],
                "reader_consensus": row["reader_consensus"],
                "box_area_fraction": row["box_area_fraction"],
                "box_area_quartile": row["box_area_quartile"],
                "rescue_split": row["rescue_split"],
                "source_dataset": "VinDr-CXR-v1.0.0",
                "source_split": "train",
                "split_version": RESCUE_SPLIT_VERSION,
                "image_disjoint_only": True,
                "patient_level_claim": False,
                "formal_result": False,
            }
        )
    for (finding, split), image_ids in selected_negative.items():
        spec = FINDINGS[finding]
        for image_id in image_ids:
            source = source_by_image[image_id]
            vote = vote_records[(image_id, finding)]
            manifest_rows.append(
                {
                    "sample_id": f"vindr-train::{image_id}::{finding}",
                    "unit_id": image_id,
                    "patient_id": None,
                    "patient_id_status": "not_provided_by_public_release",
                    "image_id": image_id,
                    "image_path": str(dataset_root / "train" / f"{image_id}.dicom"),
                    "official_image_sha256": official_selected[image_id],
                    "actual_image_sha256": source["actual_image_sha256"],
                    "actual_image_sha256_verified": True,
                    "native_rows": source["rows"],
                    "native_columns": source["columns"],
                    "canonical_statement_id": finding,
                    "statement_text": spec["statement_text"],
                    "state": "contradict",
                    "binary_label": 0,
                    "bounding_boxes": [],
                    "reader_votes": vote["reader_votes"],
                    "positive_vote_count": 0,
                    "reader_consensus": "0_of_3",
                    "box_area_fraction": None,
                    "box_area_quartile": None,
                    "rescue_split": split,
                    "source_dataset": "VinDr-CXR-v1.0.0",
                    "source_split": "train",
                    "split_version": RESCUE_SPLIT_VERSION,
                    "image_disjoint_only": True,
                    "patient_level_claim": False,
                    "formal_result": False,
                }
            )
    manifest_rows.sort(
        key=lambda row: (
            row["rescue_split"],
            row["canonical_statement_id"],
            -row["binary_label"],
            row["image_id"],
        )
    )
    units_by_split = {
        split: {row["unit_id"] for row in manifest_rows if row["rescue_split"] == split}
        for split in ("protocol_design", "rescue_confirm")
    }
    overlap = sorted(units_by_split["protocol_design"] & units_by_split["rescue_confirm"])
    if overlap:
        raise AssertionError(f"rescue split image overlap: {overlap[:5]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "vindr_train_rescue_dev.jsonl"
    temporary = manifest_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(manifest_path)

    row_summary: dict[str, Any] = {}
    for finding in FINDINGS:
        row_summary[finding] = {}
        for split in ("protocol_design", "rescue_confirm"):
            subset = [
                row
                for row in manifest_rows
                if row["canonical_statement_id"] == finding and row["rescue_split"] == split
            ]
            row_summary[finding][split] = {
                "support": sum(row["binary_label"] == 1 for row in subset),
                "contradict": sum(row["binary_label"] == 0 for row in subset),
                "unique_images": len({row["unit_id"] for row in subset}),
            }
    lock = {
        "format_version": FORMAT_VERSION,
        "status": "pass",
        "formal_result": False,
        "source_split": "train_only",
        "forbidden_test_path_accessed": False,
        "image_disjoint_only": True,
        "patient_level_claim": False,
        "patient_grouping_fields_present": False,
        "split_seed": int(split_seed),
        "split_version": RESCUE_SPLIT_VERSION,
        "git_head_before_rescue_implementation": git_head(),
        "builder_sha256": file_sha256(Path(__file__)),
        "protocol_module_sha256": file_sha256(ROOT / "bives_cxr" / "rescue_protocol.py"),
        "official_manifest_sha256": file_sha256(official_path),
        "image_labels_train_sha256": file_sha256(label_path),
        "annotations_train_sha256": file_sha256(box_path),
        "manifest_sha256": file_sha256(manifest_path),
        "selected_image_hash_count": len(selected_units),
        "selected_image_hashes_all_verified": True,
        "train_images": len(labels_by_image),
        "manifest_rows": len(manifest_rows),
        "manifest_unique_images": len({row["unit_id"] for row in manifest_rows}),
        "split_image_overlap": len(overlap),
        "positive_union_images": len(strata_by_image),
        "positive_finding_rows": len(positive_rows),
        "disagreement_excluded": dict(sorted(disagreements.items())),
        "split_audit": split_audit,
        "row_summary": row_summary,
    }
    lock_path = output_dir / "vindr_train_rescue_dev_lock.json"
    write_json(lock_path, lock)
    summary = {
        "status": "pass",
        "format_version": FORMAT_VERSION,
        "manifest": str(manifest_path),
        "lock": str(lock_path),
        "manifest_sha256": lock["manifest_sha256"],
        "manifest_rows": len(manifest_rows),
        "manifest_unique_images": lock["manifest_unique_images"],
        "selected_image_hash_count": len(selected_units),
        "split_image_overlap": 0,
        "image_disjoint_only": True,
        "patient_level_claim": False,
        "row_summary": row_summary,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    result = prepare_rescue_dev(
        args.dataset_root,
        args.output_dir,
        hash_workers=args.hash_workers,
        split_seed=args.split_seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
