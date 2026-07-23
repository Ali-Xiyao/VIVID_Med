"""Build a de-identified CheXpert/CheXpert-Plus lineage summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_image_path(value: object) -> str:
    parts = PurePosixPath(str(value).replace("\\", "/")).parts
    for split in ("train", "valid", "test"):
        if split in parts:
            index = parts.index(split)
            return "/".join(parts[index:])
    raise ValueError(f"path has no recognized split component: {value}")


def build_summary(plus_table: Path, valid_csv: Path) -> dict[str, object]:
    plus = pd.read_parquet(
        plus_table,
        columns=["path_to_image", "split", "deid_patient_id", "frontal_lateral"],
    )
    valid = pd.read_csv(valid_csv, usecols=["Path"])
    plus["split"] = plus["split"].astype(str).str.lower()
    plus["normalized_path"] = plus["path_to_image"].map(normalize_image_path)
    valid_paths = set(valid["Path"].map(normalize_image_path))
    plus_valid_paths = set(
        plus.loc[plus["split"].eq("valid"), "normalized_path"].astype(str)
    )
    patient_splits = plus.groupby("deid_patient_id", dropna=False)["split"].nunique()
    split_counts = {
        str(key): int(value) for key, value in plus["split"].value_counts().items()
    }
    patient_counts = {
        str(key): int(value)
        for key, value in plus.groupby("split")["deid_patient_id"].nunique().items()
    }
    return {
        "schema_version": 1,
        "artifact": "chexpert_lineage_manifest",
        "contains_patient_identifiers": False,
        "source": {
            "chexpert_plus_table": str(plus_table.resolve()),
            "chexpert_plus_sha256": sha256_file(plus_table),
            "chexpert_valid_csv": str(valid_csv.resolve()),
            "chexpert_valid_sha256": sha256_file(valid_csv),
        },
        "summary": {
            "rows": int(len(plus)),
            "patients": int(plus["deid_patient_id"].nunique()),
            "split_counts": split_counts,
            "patient_counts": patient_counts,
            "multi_split_patients": int((patient_splits > 1).sum()),
            "frontal_train_rows": int(
                (
                    plus["split"].eq("train")
                    & plus["frontal_lateral"].astype(str).str.lower().eq("frontal")
                ).sum()
            ),
            "official_valid_rows": int(len(valid_paths)),
            "plus_valid_rows": int(len(plus_valid_paths)),
            "valid_path_overlap": int(len(valid_paths & plus_valid_paths)),
            "valid_path_symmetric_difference": int(
                len(valid_paths ^ plus_valid_paths)
            ),
        },
        "decision": {
            "chexpert_plus_valid_is_official_chexpert_valid": (
                valid_paths == plus_valid_paths
            ),
            "track_b_training_policy": "train_split_only_and_frontal_only",
            "reserved_valid_policy": "exclude_all_valid_rows_and_patients",
            "chexlocalize_test_policy": "absent_and_sealed",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plus-table", required=True, type=Path)
    parser.add_argument("--valid-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build_summary(args.plus_table, args.valid_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
