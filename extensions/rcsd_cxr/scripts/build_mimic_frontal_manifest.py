"""Build a patient-aware canonical-frontal manifest from official MIMIC tables.

This script changes no pixels. It selects PA over AP and then the
lexicographically first DICOM ID within each study. The official test split is
deliberately rejected for the paper-one training pipeline.
"""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path


VIEW_PRIORITY = {"PA": 0, "AP": 1}


def read_split_table(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dicom_id", "subject_id", "study_id", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"split table missing columns: {sorted(missing)}")
        for row in reader:
            dicom_id = row["dicom_id"]
            if dicom_id in result:
                raise ValueError(f"duplicate dicom_id in split table: {dicom_id}")
            result[dicom_id] = row["split"]
    return result


def select_rows(
    metadata_path: Path,
    split_by_dicom: dict[str, str],
    allowed_splits: set[str],
) -> list[dict[str, str]]:
    candidates: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    with gzip.open(metadata_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dicom_id", "subject_id", "study_id", "ViewPosition"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"metadata table missing columns: {sorted(missing)}")
        for row in reader:
            view = (row["ViewPosition"] or "").upper()
            if view not in VIEW_PRIORITY:
                continue
            dicom_id = row["dicom_id"]
            split = split_by_dicom.get(dicom_id)
            if split is None:
                raise ValueError(f"metadata dicom_id absent from split table: {dicom_id}")
            if split not in allowed_splits:
                continue
            key = (row["subject_id"], row["study_id"], split)
            candidates.setdefault(key, []).append(row)

    selected: list[dict[str, str]] = []
    for (subject_id, study_id, split), rows in sorted(candidates.items()):
        row = min(rows, key=lambda item: (VIEW_PRIORITY[item["ViewPosition"].upper()], item["dicom_id"]))
        prefix = f"p{subject_id[:2]}"
        selected.append(
            {
                "patient_id": subject_id,
                "study_id": study_id,
                "image_id": row["dicom_id"],
                "view_position": row["ViewPosition"].upper(),
                "split": split,
                "image_path": f"files/{prefix}/p{subject_id}/s{study_id}/{row['dicom_id']}.jpg",
                "report_path": f"files/{prefix}/p{subject_id}/s{study_id}.txt",
            }
        )
    return selected


def validate_patient_disjoint(rows: list[dict[str, str]]) -> None:
    patient_splits: dict[str, set[str]] = {}
    for row in rows:
        patient_splits.setdefault(row["patient_id"], set()).add(row["split"])
    overlap = {patient: splits for patient, splits in patient_splits.items() if len(splits) > 1}
    if overlap:
        preview = list(overlap.items())[:5]
        raise ValueError(f"patient split overlap detected; examples: {preview}")


def validate_paths(rows: list[dict[str, str]], image_root: Path, report_root: Path) -> None:
    missing_images = [row["image_path"] for row in rows if not (image_root / row["image_path"]).is_file()]
    missing_reports = [row["report_path"] for row in rows if not (report_root / row["report_path"]).is_file()]
    if missing_images or missing_reports:
        raise FileNotFoundError(
            f"missing images={len(missing_images)} reports={len(missing_reports)}; "
            f"image examples={missing_images[:3]} report examples={missing_reports[:3]}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--split-table", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--report-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "validate"])
    args = parser.parse_args()
    allowed_splits = set(args.splits)
    if "test" in allowed_splits:
        raise ValueError("MIMIC test is not authorized in the paper-one manifest builder")
    split_by_dicom = read_split_table(args.split_table)
    rows = select_rows(args.metadata, split_by_dicom, allowed_splits)
    if not rows:
        raise ValueError("no canonical frontal rows matched the requested splits")
    validate_patient_disjoint(rows)
    validate_paths(rows, args.image_root, args.report_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} canonical frontal studies to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
