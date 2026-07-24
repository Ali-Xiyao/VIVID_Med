"""Fail-closed readiness audit for the fresh-development replication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ARMS = ["A0_direct", "A2_ums", "A3_gds"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_identity(
    path: Path,
) -> tuple[int, int, set[str], set[str], set[str]]:
    rows = 0
    patients: set[str] = set()
    paths: set[str] = set()
    splits: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            patients.add(str(row["patient_id"]))
            paths.add(str(row["image_path"]))
            splits.add(str(row["split"]))
    return rows, len(patients), patients, paths, splits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--split-audit", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--hard-manifest", required=True, type=Path)
    parser.add_argument("--mimic-image-root", required=True, type=Path)
    parser.add_argument("--chexpert-image-root", required=True, type=Path)
    parser.add_argument("--teacher-path", required=True, type=Path)
    parser.add_argument("--backbone-weights", required=True, type=Path)
    parser.add_argument("--training-script", required=True, type=Path)
    parser.add_argument("--probe-script", required=True, type=Path)
    parser.add_argument("--gate-script", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    split_audit = json.loads(args.split_audit.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    try:
        checks["split_audit_pass"] = split_audit.get("pass") is True
        checks["hard_manifest_hash"] = (
            sha256_file(args.hard_manifest)
            == lock["pretraining"]["hard_ums_manifest_sha256"]
        )
        checks["pretraining_identity"] = (
            lock["pretraining"]["arms"] == ARMS
            and lock["pretraining"]["seeds"] == [0, 1, 2]
            and lock["pretraining"]["teacher"] == "Qwen3.5-2B"
            and lock["pretraining"]["projector"] == "historical_prefix4"
            and lock["pretraining"]["pilot_steps"] == 3000
            and lock["pretraining"]["effective_batch_size"] == 32
            and lock["pretraining"]["lambda_schema"] == 0.5
            and lock["pretraining"]["lambda_ramp_steps"] == 500
        )
        checks["teacher_present"] = (
            args.teacher_path.is_dir()
            and any(args.teacher_path.glob("*.safetensors"))
        )
        checks["backbone_present"] = args.backbone_weights.is_file()
        checks["scripts_present"] = all(
            path.is_file()
            for path in (
                args.training_script,
                args.probe_script,
                args.gate_script,
            )
        )
        split_details: dict[str, object] = {}
        patient_sets: dict[str, set[str]] = {}
        path_sets: dict[str, set[str]] = {}
        for split in ("probe_train", "probe_validation", "fresh_development"):
            path = args.split_dir / f"{split}.csv"
            rows, patients, patient_set, path_set, split_literals = csv_identity(
                path
            )
            expected = lock["data"]["expected"][split]
            checks[f"{split}_identity"] = (
                rows == expected["rows"]
                and patients == expected["patients"]
                and split_literals == {split}
                and sha256_file(path)
                == split_audit["hashes"]["manifests"][split]
            )
            patient_sets[split] = patient_set
            path_sets[split] = path_set
            split_details[split] = {
                "rows": rows,
                "patients": patients,
                "sha256": sha256_file(path),
            }
        names = tuple(patient_sets)
        checks["patient_disjoint"] = all(
            not (patient_sets[left] & patient_sets[right])
            for index, left in enumerate(names)
            for right in names[index + 1 :]
        )
        checks["path_disjoint"] = all(
            not (path_sets[left] & path_sets[right])
            for index, left in enumerate(names)
            for right in names[index + 1 :]
        )
        checks["chexpert_images_present"] = all(
            (args.chexpert_image_root / path).is_file()
            for paths in path_sets.values()
            for path in paths
        )
        protected_text = " ".join(
            str(value).lower()
            for value in vars(args).values()
            if isinstance(value, Path)
        )
        checks["protected_tests_not_referenced"] = (
            "chexlocalize" not in protected_text
            and "vindr" not in protected_text
        )
        details = {
            "splits": split_details,
            "hashes": {
                "lock": sha256_file(args.lock),
                "split_audit": sha256_file(args.split_audit),
                "hard_manifest": sha256_file(args.hard_manifest),
                "backbone_weights": sha256_file(args.backbone_weights),
                "training_script": sha256_file(args.training_script),
                "probe_script": sha256_file(args.probe_script),
                "gate_script": sha256_file(args.gate_script),
            },
        }
    except Exception as error:
        checks["exception_free"] = False
        details = {"error_type": type(error).__name__, "error": str(error)}
    result = {
        "schema_version": 1,
        "artifact": "vivid_gds_fresh_replication_readiness",
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
