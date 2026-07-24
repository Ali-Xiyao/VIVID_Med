"""Audit the frozen Stage-A server inputs before launching GPU work."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vivid_gds.contracts import parse_ums_target  # noqa: E402
from vivid_gds.io import load_jsonl, sha256_file, teacher_weight_authority  # noqa: E402


EXPECTED_HARD_SHA256 = (
    "1da254ab25ab8f005536ff16ac7a1c40e33f15add2afa25277a8c6e06f6e30b4"
)


def csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


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
    parser.add_argument("--a2-summary", required=True, type=Path)
    parser.add_argument("--a2-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    try:
        hard_sha = sha256_file(args.hard_manifest)
        checks["hard_manifest_hash"] = hard_sha == EXPECTED_HARD_SHA256
        rows = load_jsonl(args.hard_manifest)
        required = {
            "row_id",
            "patient_id",
            "study_id",
            "split",
            "image_path",
            "target",
        }
        checks["manifest_schema"] = all(required <= set(row) for row in rows)
        split_counts = Counter(str(row["split"]) for row in rows)
        checks["frozen_20k_train"] = split_counts["train"] == 20000
        train_patients = {
            str(row["patient_id"]) for row in rows if row["split"] == "train"
        }
        validation_patients = {
            str(row["patient_id"])
            for row in rows
            if row["split"] == "validate"
        }
        checks["patient_disjoint"] = not (
            train_patients & validation_patients
        )
        for row in rows:
            parse_ums_target(str(row["target"]))
        checks["hard_targets_valid"] = True
        checks["images_present"] = all(
            (args.image_root / str(row["image_path"])).is_file() for row in rows
        )
        overfit = json.loads(args.overfit_ids.read_text(encoding="utf-8"))
        selected = set(overfit["row_ids"])
        train_ids = {
            str(row["row_id"]) for row in rows if row["split"] == "train"
        }
        checks["overfit_256"] = len(selected) == 256 and selected <= train_ids
        checks["teacher_present"] = bool(
            teacher_weight_authority(args.teacher_path)["files"]
        )
        checks["backbone_present"] = args.backbone_weights.is_file()
        checks["probe_inputs_present"] = (
            args.probe_train_manifest.is_file()
            and args.expert_manifest.is_file()
            and args.chexpert_root.is_dir()
        )
        a2 = json.loads(args.a2_summary.read_text(encoding="utf-8"))
        checks["a2_identity"] = (
            a2.get("arm") == "ums_prefix4"
            and a2.get("mode") == "pilot"
            and a2.get("pass") is True
            and a2["hashes"]["hard_manifest"] == EXPECTED_HARD_SHA256
            and a2["budget"]["max_steps"] == 3000
            and a2["budget"]["effective_batch_size"] == 32
            and a2["budget"]["seed"] == 0
            and args.a2_checkpoint.is_file()
        )
        protected_text = " ".join(
            str(value)
            for value in vars(args).values()
            if isinstance(value, Path)
        ).lower()
        checks["protected_test_not_referenced"] = (
            "chexlocalize" not in protected_text
            and "vindr" not in protected_text
        )
        details = {
            "rows": len(rows),
            "split_counts": dict(split_counts),
            "train_patients": len(train_patients),
            "validation_patients": len(validation_patients),
            "probe_train_rows": csv_rows(args.probe_train_manifest),
            "expert_development_rows": csv_rows(args.expert_manifest),
            "hashes": {
                "hard_manifest": hard_sha,
                "overfit_ids": sha256_file(args.overfit_ids),
                "backbone_weights": sha256_file(args.backbone_weights),
                "a2_checkpoint": sha256_file(args.a2_checkpoint),
            },
        }
    except Exception as error:
        checks["exception_free"] = False
        details["error_type"] = type(error).__name__
        details["error"] = str(error)
    result = {
        "schema_version": 1,
        "artifact": "vivid_gds_server_readiness",
        "pass": bool(checks) and all(checks.values()),
        "checks": checks,
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
