"""Freeze patient-disjoint CheXpert probe-train and expert-development rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


FINDINGS = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
)
PATIENT_PATTERN = re.compile(r"patient(\d+)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_path(value: str) -> str:
    path = value.replace("\\", "/")
    prefix = "CheXpert-v1.0-small/"
    if path.startswith(prefix):
        path = path[len(prefix):]
    return path


def patient_id(path: str) -> str:
    match = PATIENT_PATTERN.search(path)
    if match is None:
        raise ValueError(f"cannot parse CheXpert patient from {path}")
    return match.group(1)


def build(input_path: Path, output_path: Path, *, split: str) -> dict[str, int]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    kept = []
    for row in rows:
        if str(row.get("Frontal/Lateral") or "").strip() != "Frontal":
            continue
        path = normalize_path(row["Path"])
        kept.append(
            {
                "patient_id": patient_id(path),
                "image_path": path,
                "split": split,
                **{finding: str(row.get(finding) or "").strip()
                   for finding in FINDINGS},
            }
        )
    if not kept:
        raise ValueError(f"no frontal rows in {input_path}")
    kept.sort(key=lambda row: (row["patient_id"], row["image_path"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(kept[0]))
        writer.writeheader()
        writer.writerows(kept)
    return {
        "rows": len(kept),
        "patients": len({row["patient_id"] for row in kept}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--valid-csv", required=True, type=Path)
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--expert-output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    args = parser.parse_args()
    train = build(args.train_csv, args.train_output, split="probe_train")
    expert = build(args.valid_csv, args.expert_output, split="expert_dev")
    with args.train_output.open("r", encoding="utf-8", newline="") as handle:
        train_patients = {row["patient_id"] for row in csv.DictReader(handle)}
    with args.expert_output.open("r", encoding="utf-8", newline="") as handle:
        expert_patients = {row["patient_id"] for row in csv.DictReader(handle)}
    overlap = train_patients & expert_patients
    if overlap:
        raise ValueError(f"CheXpert patient overlap: {len(overlap)}")
    result = {
        "schema_version": 1,
        "artifact": "chexpert_probe_manifests",
        "pass": True,
        "findings": list(FINDINGS),
        "train": train,
        "expert_development": expert,
        "patient_overlap": 0,
        "expert_role": "exposed development gate only",
        "hashes": {
            "train_source": sha256_file(args.train_csv),
            "valid_source": sha256_file(args.valid_csv),
            "probe_train_manifest": sha256_file(args.train_output),
            "expert_dev_manifest": sha256_file(args.expert_output),
        },
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
