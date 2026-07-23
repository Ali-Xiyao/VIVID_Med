"""Fail-closed Gate-0 audit for a canonical MIMIC-CXR manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = {
    "patient_id",
    "study_id",
    "image_id",
    "view_position",
    "split",
    "image_path",
    "report_path",
}
ALLOWED_SPLITS = {"train", "validate"}
ALLOWED_VIEWS = {"PA", "AP"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_rows(
    rows: Iterable[dict[str, str]],
    *,
    image_root: Path,
    report_root: Path,
    check_paths: bool,
) -> dict[str, object]:
    patients: dict[str, str] = {}
    studies: set[str] = set()
    images: set[str] = set()
    split_counts: dict[str, int] = {}
    view_counts: dict[str, int] = {}
    missing_images: list[str] = []
    missing_reports: list[str] = []
    empty_files: list[str] = []
    total = 0

    for line_number, raw in enumerate(rows, start=2):
        total += 1
        values = {key: (raw.get(key) or "").strip() for key in REQUIRED_COLUMNS}
        empty = sorted(key for key, value in values.items() if not value)
        if empty:
            raise ValueError(f"line {line_number} has empty fields: {empty}")

        split = values["split"]
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"line {line_number} has forbidden split: {split}")
        view = values["view_position"].upper()
        if view not in ALLOWED_VIEWS:
            raise ValueError(f"line {line_number} has forbidden view: {view}")

        patient_id = values["patient_id"]
        prior_split = patients.setdefault(patient_id, split)
        if prior_split != split:
            raise ValueError(
                f"patient {patient_id} appears in both {prior_split} and {split}"
            )
        study_id = values["study_id"]
        if study_id in studies:
            raise ValueError(f"duplicate canonical study: {study_id}")
        studies.add(study_id)
        image_id = values["image_id"]
        if image_id in images:
            raise ValueError(f"duplicate canonical image: {image_id}")
        images.add(image_id)

        split_counts[split] = split_counts.get(split, 0) + 1
        view_counts[view] = view_counts.get(view, 0) + 1

        if check_paths:
            image = image_root / values["image_path"]
            report = report_root / values["report_path"]
            if not image.is_file():
                if len(missing_images) < 20:
                    missing_images.append(values["image_path"])
            elif image.stat().st_size <= 0:
                if len(empty_files) < 20:
                    empty_files.append(str(image))
            if not report.is_file():
                if len(missing_reports) < 20:
                    missing_reports.append(values["report_path"])
            elif report.stat().st_size <= 0:
                if len(empty_files) < 20:
                    empty_files.append(str(report))

    if total == 0:
        raise ValueError("manifest is empty")
    if check_paths and (missing_images or missing_reports or empty_files):
        raise FileNotFoundError(
            "Gate 0 path audit failed: "
            f"missing_images={missing_images[:3]}, "
            f"missing_reports={missing_reports[:3]}, empty={empty_files[:3]}"
        )
    return {
        "rows": total,
        "patients": len(patients),
        "studies": len(studies),
        "images": len(images),
        "split_counts": split_counts,
        "view_counts": view_counts,
        "test_rows": 0,
        "patient_split_overlap": 0,
        "duplicate_studies": 0,
        "duplicate_images": 0,
        "missing_images": 0,
        "missing_reports": 0,
        "empty_files": 0,
    }


def run_audit(
    manifest: Path,
    *,
    image_root: Path,
    report_root: Path,
    metadata: Path,
    split_table: Path,
    check_paths: bool = True,
) -> dict[str, object]:
    for path in (manifest, metadata, split_table):
        if not path.is_file():
            raise FileNotFoundError(path)
    for root in (image_root, report_root):
        if not root.is_dir():
            raise NotADirectoryError(root)
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest missing columns: {sorted(missing)}")
        summary = audit_rows(
            reader,
            image_root=image_root,
            report_root=report_root,
            check_paths=check_paths,
        )
    return {
        "schema_version": 1,
        "gate": "G0_mimic_manifest",
        "pass": True,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "metadata_sha256": sha256_file(metadata),
        "split_table_sha256": sha256_file(split_table),
        "image_root": str(image_root.resolve()),
        "report_root": str(report_root.resolve()),
        "full_path_check": check_paths,
        "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--report-root", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--split-table", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--skip-path-check", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run_audit(
            args.manifest,
            image_root=args.image_root,
            report_root=args.report_root,
            metadata=args.metadata,
            split_table=args.split_table,
            check_paths=not args.skip_path_check,
        )
    except Exception as error:
        result = {
            "schema_version": 1,
            "gate": "G0_mimic_manifest",
            "pass": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False))
        return 2
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

