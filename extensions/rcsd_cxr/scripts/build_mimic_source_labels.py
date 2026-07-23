"""Join canonical MIMIC studies to official CheXpert and NegBio sources."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import TextIO


FINDINGS = (
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Fracture",
    "Support Devices",
)
ALLOWED_VALUES = {"", "-1.0", "0.0", "1.0", "-1", "0", "1"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def normalize_label(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if text.lower() == "nan":
        return ""
    if text not in ALLOWED_VALUES:
        raise ValueError(f"unsupported source label: {value}")
    if text in {"-1.0", "0.0", "1.0"}:
        return text[:-2]
    return text


def load_source(path: Path) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    with open_text(path) as handle:
        reader = csv.DictReader(handle)
        required = {"subject_id", "study_id", *FINDINGS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            study_id = (row.get("study_id") or "").strip()
            if not study_id:
                raise ValueError(f"{path}:{line_number} has empty study_id")
            if study_id in result:
                raise ValueError(f"{path} has duplicate study_id {study_id}")
            result[study_id] = tuple(normalize_label(row[name]) for name in FINDINGS)
    return result


def build_source_manifest(
    canonical_manifest: Path,
    chexpert_table: Path,
    negbio_table: Path,
    output: Path,
) -> dict[str, object]:
    chexpert = load_source(chexpert_table)
    negbio = load_source(negbio_table)
    missing_chexpert: list[str] = []
    missing_negbio: list[str] = []
    missing_chexpert_count = 0
    missing_negbio_count = 0
    counts = {
        source: {slug(name): {"present": 0, "absent": 0, "uncertain": 0, "missing": 0}
                 for name in FINDINGS}
        for source in ("chexpert", "negbio")
    }
    conflicts = {slug(name): 0 for name in FINDINGS}
    rows_written = 0
    all_missing_rows = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with canonical_manifest.open("r", encoding="utf-8", newline="") as source_handle:
        reader = csv.DictReader(source_handle)
        if "study_id" not in set(reader.fieldnames or []):
            raise ValueError("canonical manifest is missing study_id")
        source_fields = [
            f"{source}__{slug(name)}"
            for source in ("chexpert", "negbio")
            for name in FINDINGS
        ]
        with output.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(
                output_handle, fieldnames=[*(reader.fieldnames or []), *source_fields]
            )
            writer.writeheader()
            for row in reader:
                study_id = (row.get("study_id") or "").strip()
                chex_values = chexpert.get(study_id)
                negbio_values = negbio.get(study_id)
                if chex_values is None:
                    missing_chexpert_count += 1
                    if len(missing_chexpert) < 20:
                        missing_chexpert.append(study_id)
                    chex_values = tuple("" for _ in FINDINGS)
                if negbio_values is None:
                    missing_negbio_count += 1
                    if len(missing_negbio) < 20:
                        missing_negbio.append(study_id)
                    negbio_values = tuple("" for _ in FINDINGS)
                observed = 0
                for index, name in enumerate(FINDINGS):
                    key = slug(name)
                    left = chex_values[index]
                    right = negbio_values[index]
                    row[f"chexpert__{key}"] = left
                    row[f"negbio__{key}"] = right
                    for source, value in (("chexpert", left), ("negbio", right)):
                        state = {"1": "present", "0": "absent", "-1": "uncertain"}.get(
                            value, "missing"
                        )
                        counts[source][key][state] += 1
                    observed += int(bool(left)) + int(bool(right))
                    if left and right and left != right:
                        conflicts[key] += 1
                all_missing_rows += int(observed == 0)
                writer.writerow(row)
                rows_written += 1
    return {
        "schema_version": 1,
        "artifact": "mimic_official_source_label_manifest",
        "pass": True,
        "rows": rows_written,
        "source_rows_missing": {
            "chexpert": missing_chexpert_count,
            "negbio": missing_negbio_count,
        },
        "source_rows_missing_examples": {
            "chexpert": missing_chexpert,
            "negbio": missing_negbio,
        },
        "findings": list(FINDINGS),
        "all_missing_rows": all_missing_rows,
        "conflicts": conflicts,
        "counts": counts,
        "hashes": {
            "canonical_manifest": sha256_file(canonical_manifest),
            "chexpert_table": sha256_file(chexpert_table),
            "negbio_table": sha256_file(negbio_table),
            "output": sha256_file(output),
        },
        "policy": {
            "unmentioned": "missing",
            "missing_source_row": "all_fields_missing",
            "uncertain": "report_uncertain_state",
            "test_rows": "absent_by_canonical_manifest_contract",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-manifest", required=True, type=Path)
    parser.add_argument("--chexpert-table", required=True, type=Path)
    parser.add_argument("--negbio-table", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    args = parser.parse_args()
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = build_source_manifest(
            args.canonical_manifest,
            args.chexpert_table,
            args.negbio_table,
            args.output,
        )
    except Exception as error:
        result = {
            "schema_version": 1,
            "artifact": "mimic_official_source_label_manifest",
            "pass": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    args.audit_output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
