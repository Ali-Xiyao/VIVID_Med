"""Prepare MIMIC-CXR image-report manifest rows for instruction generation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def normalize_report(text: str, max_chars: int) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].strip()
    return text


def study_parts(study_dir: Path) -> tuple[str, str, str] | None:
    try:
        study_id = study_dir.name
        subject_id = study_dir.parent.name
        prefix = study_dir.parent.parent.name
    except IndexError:
        return None
    if not study_id.startswith("s") or not subject_id.startswith("p") or not prefix.startswith("p"):
        return None
    return prefix, subject_id, study_id


def iter_study_dirs(images_files_root: Path):
    for prefix_dir in sorted(images_files_root.iterdir()):
        if not prefix_dir.is_dir():
            continue
        for subject_dir in sorted(prefix_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            for study_dir in sorted(subject_dir.iterdir()):
                if study_dir.is_dir():
                    yield study_dir


def make_row(
    image_path: Path,
    report_path: Path,
    report: str,
    subject_id: str,
    study_id: str,
    image_index: int,
    source_name: str,
) -> dict[str, Any]:
    image_id = image_path.stem
    sample_id = f"{source_name}_{subject_id}_{study_id}_{image_index:02d}_{image_id}"
    return {
        "anatomy": "chest",
        "findings": {},
        "answerability": {},
        "uncertainty": {},
        "study_view": None,
        "report": report,
        "extensions": {
            "sample_id": sample_id,
            "original_path": str(image_path),
            "subject_id": subject_id.removeprefix("p"),
            "study_id": study_id.removeprefix("s"),
            "image_id": image_id,
            "report_path": str(report_path),
            "source_dataset": source_name,
        },
        "provenance": {
            "dataset": source_name,
            "image_path": str(image_path),
            "report_path": str(report_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", required=True, type=Path)
    parser.add_argument("--reports-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-studies", type=int, default=1000)
    parser.add_argument("--skip-studies", type=int, default=0)
    parser.add_argument("--max-images-per-study", type=int, default=1)
    parser.add_argument("--min-report-chars", type=int, default=80)
    parser.add_argument("--max-report-chars", type=int, default=4000)
    parser.add_argument("--source-name", default="mimic_cxr")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_files_root = args.images_root / "files"
    reports_files_root = args.reports_root / "files"
    if not images_files_root.exists():
        raise SystemExit(f"Missing images files root: {images_files_root}")
    if not reports_files_root.exists():
        raise SystemExit(f"Missing reports files root: {reports_files_root}")

    rows: list[dict[str, Any]] = []
    skipped_available = 0
    seen_available = 0
    missing_report = 0
    short_report = 0
    no_images = 0

    for study_dir in iter_study_dirs(images_files_root):
        parts = study_parts(study_dir)
        if parts is None:
            continue
        prefix, subject_id, study_id = parts
        report_path = reports_files_root / prefix / subject_id / f"{study_id}.txt"
        if not report_path.exists():
            missing_report += 1
            continue
        images = sorted(study_dir.glob("*.jpg"))
        if not images:
            no_images += 1
            continue
        report = normalize_report(
            report_path.read_text(encoding="utf-8", errors="replace"),
            max_chars=args.max_report_chars,
        )
        if len(report) < args.min_report_chars:
            short_report += 1
            continue
        if skipped_available < args.skip_studies:
            skipped_available += 1
            continue
        seen_available += 1
        for image_index, image_path in enumerate(images[: args.max_images_per_study]):
            rows.append(
                make_row(
                    image_path=image_path,
                    report_path=report_path,
                    report=report,
                    subject_id=subject_id,
                    study_id=study_id,
                    image_index=image_index,
                    source_name=args.source_name,
                )
            )
        if args.max_studies and seen_available >= args.max_studies:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": len(rows),
                "studies": seen_available,
                "skipped_available": skipped_available,
                "missing_report": missing_report,
                "short_report": short_report,
                "no_images": no_images,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
