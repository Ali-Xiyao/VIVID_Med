#!/usr/bin/env python
"""Freeze image-disjoint VinDr-train box-supervised ARISE manifests."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bives_cxr.provenance import canonical_json_sha256, file_sha256  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "local_runs/bives_cxr/vindr_rescue_dev",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "local_runs/arise_cxr/vindr_box_sc_v1",
    )
    args = parser.parse_args()
    source_manifest = args.source_dir / "vindr_train_rescue_dev.jsonl"
    source_lock_path = args.source_dir / "vindr_train_rescue_dev_lock.json"
    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    if source_lock.get("status") != "pass" or source_lock.get("source_split") != "train_only":
        raise ValueError("VinDr source lock is not train-only/pass")
    if source_lock.get("forbidden_test_path_accessed") is not False:
        raise ValueError("VinDr source lock test boundary changed")
    if file_sha256(source_manifest) != source_lock["manifest_sha256"]:
        raise ValueError("VinDr source manifest hash changed")
    rows = read_jsonl(source_manifest)
    if len(rows) != int(source_lock["manifest_rows"]):
        raise ValueError("VinDr source row count changed")
    if any(row.get("source_split") != "train" for row in rows):
        raise ValueError("non-train VinDr row encountered")
    if any(row.get("actual_image_sha256_verified") is not True for row in rows):
        raise ValueError("VinDr source image hash verification is incomplete")
    if any(
        str(row.get("actual_image_sha256")) != str(row.get("official_image_sha256"))
        for row in rows
    ):
        raise ValueError("VinDr actual/official image SHA-256 differs")
    rows = [
        {**row, "image_sha256": str(row["actual_image_sha256"])}
        for row in rows
    ]
    splits = {
        "train": [row for row in rows if row.get("rescue_split") == "protocol_design"],
        "val": [row for row in rows if row.get("rescue_split") == "rescue_confirm"],
    }
    train_units = {row["unit_id"] for row in splits["train"]}
    val_units = {row["unit_id"] for row in splits["val"]}
    if train_units & val_units:
        raise ValueError("VinDr ARISE train/val image leakage")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, split_rows in splits.items():
        path = args.output_dir / f"vindr_box_sc_{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in split_rows),
            encoding="utf-8",
        )
        paths[split] = path
    counts = {
        split: dict(
            sorted(
                Counter(
                    f"{row['canonical_statement_id']}|{row['state']}"
                    for row in split_rows
                ).items()
            )
        )
        for split, split_rows in splits.items()
    }
    expected = source_lock["row_summary"]
    for finding, split_block in expected.items():
        for source_split, expected_counts in split_block.items():
            split = "train" if source_split == "protocol_design" else "val"
            for state in ("support", "contradict"):
                if counts[split].get(f"{finding}|{state}") != int(expected_counts[state]):
                    raise ValueError("VinDr ARISE stratum count changed")
    lock = {
        "schema_version": "arise-vindr-box-sc-data-lock-v1",
        "status": "ready_for_token_cache",
        "formal_result": False,
        "confirmatory_evidence": False,
        "test_opened": False,
        "source_split": "VinDr-CXR train only",
        "patient_level_claim": False,
        "image_disjoint": True,
        "source_lock_sha256": file_sha256(source_lock_path),
        "source_manifest_sha256": file_sha256(source_manifest),
        "train_records": len(splits["train"]),
        "val_records": len(splits["val"]),
        "train_images": len(train_units),
        "val_images": len(val_units),
        "image_overlap": 0,
        "counts": counts,
        "train_manifest_sha256": file_sha256(paths["train"]),
        "val_manifest_sha256": file_sha256(paths["val"]),
    }
    lock["canonical_sha256"] = canonical_json_sha256(lock)
    (args.output_dir / "data_lock.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(lock, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
