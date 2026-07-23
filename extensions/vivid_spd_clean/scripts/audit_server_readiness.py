"""Audit strict-route manifests, model files, and protected boundaries."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vivid_spd_clean.io import (  # noqa: E402
    load_jsonl,
    sha256_file,
    teacher_weight_authority,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard-manifest", required=True, type=Path)
    parser.add_argument("--overfit-ids", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--teacher-path", required=True, type=Path)
    parser.add_argument("--backbone-weights", required=True, type=Path)
    parser.add_argument("--probe-train-manifest", required=True, type=Path)
    parser.add_argument("--expert-manifest", required=True, type=Path)
    parser.add_argument("--chexpert-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = load_jsonl(args.hard_manifest)
    row_ids = [str(row["row_id"]) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("duplicate hard-UMS row ids")
    splits = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "validate")
    }
    train_patients = {str(row["patient_id"]) for row in splits["train"]}
    validation_patients = {
        str(row["patient_id"]) for row in splits["validate"]
    }
    overlap = train_patients & validation_patients
    missing = [
        str(args.image_root / str(row["image_path"]))
        for row in rows
        if not (args.image_root / str(row["image_path"])).is_file()
    ]
    sample = random.Random(0).sample(rows, min(64, len(rows)))
    decoded = 0
    for row in sample:
        with Image.open(args.image_root / str(row["image_path"])) as image:
            image.convert("RGB").load()
        decoded += 1
    overfit = json.loads(args.overfit_ids.read_text(encoding="utf-8"))[
        "row_ids"
    ]
    train_ids = {str(row["row_id"]) for row in splits["train"]}
    chexpert_missing = []
    for manifest in (args.probe_train_manifest, args.expert_manifest):
        import csv

        with manifest.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                path = args.chexpert_root / row["image_path"]
                if not path.is_file():
                    chexpert_missing.append(str(path))
                    if len(chexpert_missing) >= 10:
                        break
        if chexpert_missing:
            break
    checks = {
        "hard_rows_present": bool(rows),
        "train_rows_present": bool(splits["train"]),
        "validation_rows_present": bool(splits["validate"]),
        "patient_overlap_zero": not overlap,
        "all_mimic_images_present": not missing,
        "decode_sample_64": decoded == min(64, len(rows)),
        "overfit_256": len(overfit) == 256,
        "overfit_subset_of_train": set(overfit) <= train_ids,
        "teacher_present": args.teacher_path.is_dir(),
        "backbone_present": args.backbone_weights.is_file(),
        "probe_manifests_present": (
            args.probe_train_manifest.is_file()
            and args.expert_manifest.is_file()
        ),
        "chexpert_sample_paths_present": not chexpert_missing,
    }
    result = {
        "schema_version": 1,
        "artifact": "strict_vivid_spd_server_readiness",
        "pass": all(checks.values()),
        "checks": checks,
        "rows": {
            "train": len(splits["train"]),
            "validate": len(splits["validate"]),
            "overfit": len(overfit),
        },
        "patient_overlap": len(overlap),
        "missing_images": len(missing),
        "hashes": {
            "hard_manifest": sha256_file(args.hard_manifest),
            "overfit_ids": sha256_file(args.overfit_ids),
            "backbone_weights": sha256_file(args.backbone_weights),
            "probe_train_manifest": sha256_file(args.probe_train_manifest),
            "expert_manifest": sha256_file(args.expert_manifest),
            "teacher_weights": teacher_weight_authority(args.teacher_path),
        },
        "protected_surfaces_opened": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result))
    return 0 if result["pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
