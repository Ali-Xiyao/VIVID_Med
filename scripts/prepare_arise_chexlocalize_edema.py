#!/usr/bin/env python
"""Bind CheXlocalize validation Edema regions for ARISE development."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bives_cxr.chexlocalize_validation import (  # noqa: E402
    bind_annotation_identities,
    identifier_sha256,
    prepare_letterboxed_masks,
    read_chexpert_validation_rows,
)
from bives_cxr.provenance import canonical_json_sha256, file_sha256  # noqa: E402


PUBLIC_ROOT = Path(r"H:\Xiyao_Wang\000_Public Dataset")
RELEASE_NAME = "Edema"
FINDING_ID = "pulmonary_edema"
STATEMENT = "Pulmonary edema is present."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-root",
        type=Path,
        default=PUBLIC_ROOT / "CheXlocalize/redivis_v1_0/validation",
    )
    parser.add_argument(
        "--chexpert-root",
        type=Path,
        default=PUBLIC_ROOT / "CheXpert-v1.0-small",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "local_runs/arise_cxr/chexlocalize_edema_development",
    )
    args = parser.parse_args()
    annotation_path = args.release_root / "gt_annotations_val.json"
    valid_csv = args.chexpert_root / "valid.csv"
    if not annotation_path.is_file() or not valid_csv.is_file():
        raise FileNotFoundError("authorized CheXlocalize validation inputs are incomplete")
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    validation = read_chexpert_validation_rows(valid_csv, args.chexpert_root)
    bound = bind_annotation_identities(annotations, validation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.output_dir / "expert_masks"
    mask_dir.mkdir(exist_ok=True)
    rows = []
    for _, identity, payload in bound:
        if RELEASE_NAME not in payload:
            continue
        image_record = validation[identity["image"]]
        image_path = image_record["image_path"]
        sample_hash = identifier_sha256(
            "arise-chexlocalize-validation",
            f"{identity['image']}|{FINDING_ID}",
        )
        expert, content, geometry = prepare_letterboxed_masks(
            image_path,
            payload[RELEASE_NAME],
            payload.get("img_size"),
        )
        mask_name = f"{sample_hash[:24]}.npz"
        mask_path = mask_dir / mask_name
        np.savez_compressed(
            mask_path,
            expert_mask=expert.astype(np.uint8),
            content_mask=content.astype(np.uint8),
        )
        rows.append(
            {
                "sample_id": sample_hash,
                "dataset_id": "chexlocalize_validation_development_v1_0",
                "dataset_role": "development_prior_exposed",
                "source_split": "publisher_validation",
                "patient_id_hash": identifier_sha256("chexpert", identity["patient"]),
                "study_id_hash": identifier_sha256("chexpert", identity["study"]),
                "image_id_hash": identifier_sha256("chexpert", identity["image"]),
                "image_path": str(image_path),
                "official_image_sha256": file_sha256(image_path),
                "canonical_statement_id": FINDING_ID,
                "statement_text": STATEMENT,
                "released_finding_name": RELEASE_NAME,
                "mask_file": mask_name,
                "mask_sha256": file_sha256(mask_path),
                "score_free_audit": geometry,
            }
        )
    rows.sort(key=lambda row: (row["patient_id_hash"], row["image_id_hash"]))
    if len(rows) != 45:
        raise ValueError(f"CheXlocalize validation Edema count drift: {len(rows)} != 45")
    manifest = args.output_dir / "development_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    lock = {
        "schema_version": "arise-chexlocalize-edema-data-lock-v1",
        "status": "score_free_data_ready",
        "formal_result": False,
        "confirmatory_evidence": False,
        "test_opened": False,
        "dataset_role": "publisher_validation_prior_exposed_development",
        "finding": FINDING_ID,
        "pairs": len(rows),
        "images": len({row["image_id_hash"] for row in rows}),
        "patients": len({row["patient_id_hash"] for row in rows}),
        "source_hashes": {
            "gt_annotations_val.json": file_sha256(annotation_path),
            "CheXpert-v1.0-small/valid.csv": file_sha256(valid_csv),
        },
        "manifest_sha256": file_sha256(manifest),
        "raw_identifiers_emitted_to_tracked_files": False,
    }
    lock["canonical_sha256"] = canonical_json_sha256(lock)
    (args.output_dir / "development_data_lock.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(lock, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
