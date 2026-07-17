"""Index paired MIMIC-CXR image/report studies for the BiVES-CXR P0 audit.

This is deliberately an intake tool, not a label generator.  It never reads
or emits report text, never infers BiVES states, and never creates a training
manifest.  Its JSONL output is an ignored local review input that records only
the paired source identifiers and paths required for a separately frozen
parser and blinded clinical review workflow.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument(
        "--max-studies",
        type=int,
        default=0,
        help="Maximum paired studies to write; 0 means no limit.",
    )
    parser.add_argument(
        "--skip-paired-studies",
        type=int,
        default=0,
        help="Number of eligible paired studies to skip before writing rows.",
    )
    parser.add_argument(
        "--max-images-per-study",
        type=int,
        default=0,
        help="Maximum images per study; 0 means retain all images for review.",
    )
    parser.add_argument("--source-dataset", default="MIMIC-CXR-JPG")
    return parser.parse_args()


def _files_root(root: Path) -> Path:
    return root / "files" if (root / "files").is_dir() else root


def iter_study_dirs(images_files_root: Path) -> Iterator[tuple[str, str, str, Path]]:
    """Yield the standard MIMIC pXX / patient / study hierarchy."""
    for prefix_dir in sorted(images_files_root.iterdir()):
        if not prefix_dir.is_dir() or not prefix_dir.name.startswith("p"):
            continue
        for patient_dir in sorted(prefix_dir.iterdir()):
            if not patient_dir.is_dir() or not patient_dir.name.startswith("p"):
                continue
            for study_dir in sorted(patient_dir.iterdir()):
                if study_dir.is_dir() and study_dir.name.startswith("s"):
                    yield prefix_dir.name, patient_dir.name, study_dir.name, study_dir


def candidate_rows(
    *,
    images_root: Path,
    reports_root: Path,
    max_studies: int,
    skip_paired_studies: int,
    max_images_per_study: int,
    source_dataset: str,
) -> tuple[Iterator[dict[str, Any]], dict[str, int]]:
    images_files = _files_root(images_root)
    reports_files = _files_root(reports_root)
    if not images_files.is_dir():
        raise FileNotFoundError(f"MIMIC image files root is missing: {images_files}")
    if not reports_files.is_dir():
        raise FileNotFoundError(f"MIMIC report files root is missing: {reports_files}")
    if max_studies < 0 or skip_paired_studies < 0 or max_images_per_study < 0:
        raise ValueError("study and image limits must be non-negative")

    summary = {
        "studies_scanned": 0,
        "paired_studies": 0,
        "skipped_paired_studies": 0,
        "missing_report_studies": 0,
        "empty_image_studies": 0,
        "rows_written": 0,
    }

    def generate() -> Iterator[dict[str, Any]]:
        written_studies = 0
        for prefix, patient_id, study_id, study_dir in iter_study_dirs(images_files):
            if max_studies and written_studies >= max_studies:
                break
            summary["studies_scanned"] += 1
            report_path = reports_files / prefix / patient_id / f"{study_id}.txt"
            if not report_path.is_file():
                summary["missing_report_studies"] += 1
                continue
            images = sorted(path for path in study_dir.iterdir() if path.suffix.lower() == ".jpg")
            if not images:
                summary["empty_image_studies"] += 1
                continue
            if summary["skipped_paired_studies"] < skip_paired_studies:
                summary["skipped_paired_studies"] += 1
                continue
            summary["paired_studies"] += 1
            selected = images if max_images_per_study == 0 else images[:max_images_per_study]
            for image_path in selected:
                row = {
                    "candidate_id": f"mimic_p0_{patient_id}_{study_id}_{image_path.stem}",
                    "source_dataset": source_dataset,
                    "patient_id": patient_id,
                    "study_id": study_id,
                    "image_id": image_path.stem,
                    "image_path": str(image_path),
                    "report_path": str(report_path),
                    "candidate_status": "unparsed",
                    "p0_role": "parser_and_blind_review_input",
                }
                summary["rows_written"] += 1
                yield row
            written_studies += 1

    return generate(), summary


def main() -> None:
    args = parse_args()
    rows, summary = candidate_rows(
        images_root=args.images_root,
        reports_root=args.reports_root,
        max_studies=args.max_studies,
        skip_paired_studies=args.skip_paired_studies,
        max_images_per_study=args.max_images_per_study,
        source_dataset=args.source_dataset,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary.update(
        {
            "status": "intake_only",
            "output": str(args.output),
            "images_root": str(args.images_root),
            "reports_root": str(args.reports_root),
            "labeling_claim": "none",
        }
    )
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
