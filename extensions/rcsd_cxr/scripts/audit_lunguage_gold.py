"""Qualify LUNGUAGE as independent report gold without emitting report text."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


REQUIRED_GOLD_COLUMNS = {
    "subject_id",
    "study_id",
    "ent_idx",
    "section",
    "cat",
    "ent",
    "normed_ent",
    "dx_status",
    "dx_certainty",
}
REQUIRED_VOCAB_COLUMNS = {
    "category",
    "subcategory",
    "target_term",
    "normed_term",
    "UMLS (w code)",
}
EXPECTED_REPORTS = 1473
EXPECTED_PATIENTS = 230
EXPECTED_ENTITIES = 17949
EXPECTED_VOCAB_ROWS = 3827
COUNT_RELATIVE_TOLERANCE = 0.02


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_id(value: object) -> str:
    return str(value or "").strip().removeprefix("p").removeprefix("s")


def load_training_studies(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "study_id" not in set(reader.fieldnames or []):
            raise ValueError("training manifest is missing study_id")
        return {_normalize_id(row["study_id"]) for row in reader}


def audit_lunguage(
    gold_path: Path,
    vocab_path: Path,
    training_manifest: Path,
) -> dict[str, object]:
    training_studies = load_training_studies(training_manifest)
    patients: set[str] = set()
    studies: set[str] = set()
    status = Counter()
    certainty = Counter()
    categories = Counter()
    empty_ids = 0
    gold_rows = 0
    with gold_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_GOLD_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Lunguage.csv missing columns: {sorted(missing)}")
        for row in reader:
            gold_rows += 1
            patient_id = _normalize_id(row["subject_id"])
            study_id = _normalize_id(row["study_id"])
            empty_ids += int(not patient_id or not study_id)
            if patient_id:
                patients.add(patient_id)
            if study_id:
                studies.add(study_id)
            status[str(row["dx_status"] or "").strip().lower() or "missing"] += 1
            certainty[str(row["dx_certainty"] or "").strip().lower() or "missing"] += 1
            categories[str(row["cat"] or "").strip().lower() or "missing"] += 1

    vocab_rows = 0
    with vocab_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_VOCAB_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Lunguage_vocab.csv missing columns: {sorted(missing)}")
        for _ in reader:
            vocab_rows += 1

    overlap = sorted(studies & training_studies)
    expected_counts = {
        "patients": {"observed": len(patients), "expected": EXPECTED_PATIENTS},
        "reports": {"observed": len(studies), "expected": EXPECTED_REPORTS},
        "entities": {"observed": gold_rows, "expected": EXPECTED_ENTITIES},
        "vocab_rows": {"observed": vocab_rows, "expected": EXPECTED_VOCAB_ROWS},
    }
    exact_identity_counts = (
        len(patients) == EXPECTED_PATIENTS and len(studies) == EXPECTED_REPORTS
    )
    row_count_within_release_tolerance = all(
        abs(item["observed"] - item["expected"]) / item["expected"]
        <= COUNT_RELATIVE_TOLERANCE
        for key, item in expected_counts.items()
        if key in {"entities", "vocab_rows"}
    )
    published_count_mismatches = {
        key: item
        for key, item in expected_counts.items()
        if item["observed"] != item["expected"]
    }
    required_states = status.get("positive", 0) > 0 and status.get("negative", 0) > 0
    tentative_available = certainty.get("tentative", 0) > 0
    passed = (
        exact_identity_counts
        and row_count_within_release_tolerance
        and empty_ids == 0
        and not overlap
        and required_states
        and tentative_available
    )
    return {
        "schema_version": 1,
        "artifact": "lunguage_gold_qualification",
        "pass": passed,
        "expected_counts": expected_counts,
        "published_count_mismatches": published_count_mismatches,
        "row_count_relative_tolerance": COUNT_RELATIVE_TOLERANCE,
        "empty_identity_rows": empty_ids,
        "training_study_overlap_count": len(overlap),
        "training_study_overlap_examples": overlap[:20],
        "dx_status_counts": dict(sorted(status.items())),
        "dx_certainty_counts": dict(sorted(certainty.items())),
        "category_counts": dict(sorted(categories.items())),
        "three_state_contract": {
            "present": "dx_status=positive",
            "absent": "dx_status=negative",
            "uncertain": "dx_certainty=tentative takes precedence over dx_status",
            "note": "certainty is report uncertainty, never visual uncertainty",
        },
        "hashes": {
            "gold": sha256_file(gold_path),
            "vocab": sha256_file(vocab_path),
            "training_manifest": sha256_file(training_manifest),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--vocab", required=True, type=Path)
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = audit_lunguage(args.gold, args.vocab, args.training_manifest)
    except Exception as error:
        result = {
            "schema_version": 1,
            "artifact": "lunguage_gold_qualification",
            "pass": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
