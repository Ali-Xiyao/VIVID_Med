#!/usr/bin/env python
"""Build the score-free CheXlocalize validation Qwen3.5 development lock."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bives_cxr.chexlocalize_validation import (  # noqa: E402
    TARGETS,
    bind_annotation_identities,
    identifier_sha256,
    prepare_letterboxed_masks,
    read_chexpert_validation_rows,
)
from bives_cxr.provenance import canonical_json_sha256, file_sha256  # noqa: E402


PUBLIC_ROOT = Path(r"H:\Xiyao_Wang\000_Public Dataset")
DEFAULT_RELEASE = PUBLIC_ROOT / "CheXlocalize/redivis_v1_0/validation"
DEFAULT_CHEXPERT = PUBLIC_ROOT / "CheXpert-v1.0-small"
DEFAULT_OUTPUT = ROOT / "local_runs/cxr_localization_causality/chexlocalize_qwen35_development"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--chexpert-root", type=Path, default=DEFAULT_CHEXPERT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation_path = args.release_root / "gt_annotations_val.json"
    segmentation_path = args.release_root / "gt_segmentations_val.json"
    gradcam_segmentation_path = args.release_root / "gradcam_segmentations_val.json"
    valid_csv = args.chexpert_root / "valid.csv"
    for path in (annotation_path, segmentation_path, gradcam_segmentation_path, valid_csv):
        if not path.is_file():
            raise FileNotFoundError(path)

    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    segmentations = json.loads(segmentation_path.read_text(encoding="utf-8"))
    gradcam_segmentations = json.loads(
        gradcam_segmentation_path.read_text(encoding="utf-8")
    )
    if not isinstance(annotations, dict) or not annotations:
        raise ValueError("CheXlocalize validation annotations are empty")
    if set(segmentations) != set(annotations):
        raise ValueError("CheXlocalize validation annotation/segmentation keys differ")
    if len(gradcam_segmentations) != 234:
        raise ValueError("CheXlocalize validation Grad-CAM identity count drift")

    validation_rows = read_chexpert_validation_rows(valid_csv, args.chexpert_root)
    if len(validation_rows) != 234:
        raise ValueError("CheXpert validation image count drift")
    validation_patients = {
        row["identity"]["patient"] for row in validation_rows.values()
    }
    if len(validation_patients) != 200:
        raise ValueError("CheXpert validation patient count drift")
    bound = bind_annotation_identities(annotations, validation_rows)
    if len(bound) != 187 or len({row[1]["patient"] for row in bound}) != 170:
        raise ValueError("CheXlocalize annotated identity count drift")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.output_dir / "expert_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for _, identity, payload in bound:
        for released_name, (finding_id, statement) in TARGETS.items():
            if released_name not in payload:
                continue
            image_record = validation_rows[identity["image"]]
            image_path = image_record["image_path"]
            sample_hash = identifier_sha256(
                "chexlocalize-validation", f"{identity['image']}|{finding_id}"
            )
            expert, content, geometry = prepare_letterboxed_masks(
                image_path, payload[released_name], payload.get("img_size")
            )
            mask_name = f"{sample_hash[:24]}.npz"
            mask_path = mask_dir / mask_name
            np.savez_compressed(
                mask_path,
                expert_mask=expert.astype(np.uint8),
                content_mask=content.astype(np.uint8),
            )
            manifest_rows.append(
                {
                    "sample_id": sample_hash,
                    "dataset_id": "chexlocalize_validation_development_v1_0",
                    "dataset_role": "development_prior_exposed",
                    "source_split": "publisher_validation",
                    "patient_id_hash": identifier_sha256(
                        "chexpert", identity["patient"]
                    ),
                    "study_id_hash": identifier_sha256("chexpert", identity["study"]),
                    "image_id_hash": identifier_sha256("chexpert", identity["image"]),
                    "image_path": str(image_path),
                    "official_image_sha256": file_sha256(image_path),
                    "canonical_statement_id": finding_id,
                    "statement_text": statement,
                    "released_finding_name": released_name,
                    "mask_file": mask_name,
                    "mask_sha256": file_sha256(mask_path),
                    "score_free_audit": geometry,
                }
            )

    manifest_rows.sort(
        key=lambda row: (row["patient_id_hash"], row["image_id_hash"], row["canonical_statement_id"])
    )
    finding_counts = Counter(row["canonical_statement_id"] for row in manifest_rows)
    if finding_counts != {"consolidation": 33, "pleural_effusion": 67}:
        raise ValueError(f"CheXlocalize target count drift: {dict(finding_counts)}")
    manifest_path = args.output_dir / "development_manifest.jsonl"
    manifest_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in manifest_rows
        ),
        encoding="utf-8",
    )
    lock = {
        "schema_version": "chexlocalize-qwen35-development-data-lock-v1",
        "status": "score_free_data_ready",
        "formal_result": False,
        "test_opened": False,
        "confirmatory_evidence": False,
        "patient_level_claim": True,
        "cluster_unit": "patient_id_hash",
        "release_reference": "aimi.chexlocalize:efx9:v1_0",
        "source_hashes": {
            "gt_annotations_val.json": file_sha256(annotation_path),
            "gt_segmentations_val.json": file_sha256(segmentation_path),
            "gradcam_segmentations_val.json": file_sha256(gradcam_segmentation_path),
            "CheXpert-v1.0-small/valid.csv": file_sha256(valid_csv),
        },
        "counts": {
            "validation_images": len(validation_rows),
            "validation_patients": len(validation_patients),
            "annotated_images": len(bound),
            "annotated_patients": len({row[1]["patient"] for row in bound}),
            "target_pairs": len(manifest_rows),
            "target_pair_findings": dict(sorted(finding_counts.items())),
            "target_pair_images": len({row["image_id_hash"] for row in manifest_rows}),
            "target_pair_patients": len({row["patient_id_hash"] for row in manifest_rows}),
        },
        "manifest_sha256": file_sha256(manifest_path),
        "raw_identifiers_emitted_to_tracked_files": False,
    }
    lock["canonical_sha256"] = canonical_json_sha256(lock)
    (args.output_dir / "development_data_lock.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
