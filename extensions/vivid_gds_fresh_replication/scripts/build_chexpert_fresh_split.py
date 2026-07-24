"""Build the locked patient-level CheXpert fresh-development manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


PATIENT_RE = re.compile(r"patient(\d+)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patient_id(path: str) -> str:
    match = PATIENT_RE.search(path)
    if match is None:
        raise ValueError(f"cannot parse patient from {path!r}")
    return match.group(1)


def image_path(path: str) -> str:
    prefix = "CheXpert-v1.0-small/"
    if not path.startswith(prefix):
        raise ValueError(f"unexpected CheXpert path: {path!r}")
    return path[len(prefix) :]


def bucket(patient: str, salt: str, modulus: int) -> int:
    digest = hashlib.sha256(f"{salt}:{patient}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def read_patients_and_paths(path: Path) -> tuple[set[str], set[str]]:
    patients: set[str] = set()
    paths: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            patients.add(patient_id(str(row["Path"])))
            paths.add(image_path(str(row["Path"])))
    return patients, paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--protected-valid-csv", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    data_lock = lock["data"]
    checks: dict[str, bool] = {
        "source_hash": sha256_file(args.train_csv)
        == data_lock["source_sha256"],
        "protected_valid_hash": sha256_file(args.protected_valid_csv)
        == data_lock["protected_valid_sha256"],
    }
    findings = list(data_lock["findings"])
    fields = ["patient_id", "image_path", "split", *findings]
    buckets_by_split = {
        "probe_train": set(data_lock["probe_train_buckets"]),
        "probe_validation": set(data_lock["probe_validation_buckets"]),
        "fresh_development": set(data_lock["fresh_development_buckets"]),
    }
    all_buckets = set().union(*buckets_by_split.values())
    checks["bucket_partition"] = (
        all_buckets == set(range(int(data_lock["bucket_modulus"])))
        and sum(len(values) for values in buckets_by_split.values())
        == len(all_buckets)
    )

    rows_by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    patients_by_split: dict[str, set[str]] = defaultdict(set)
    paths_by_split: dict[str, set[str]] = defaultdict(set)
    label_counts: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    missing_images: list[str] = []
    with args.train_csv.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        for source in csv.DictReader(handle):
            if str(source["Frontal/Lateral"]) != data_lock["eligible_view"]:
                continue
            patient = patient_id(str(source["Path"]))
            relative_path = image_path(str(source["Path"]))
            value = bucket(
                patient,
                str(data_lock["patient_hash_salt"]),
                int(data_lock["bucket_modulus"]),
            )
            selected = [
                name
                for name, values in buckets_by_split.items()
                if value in values
            ]
            if len(selected) != 1:
                raise ValueError(f"bucket {value} has {len(selected)} owners")
            split = selected[0]
            row = {
                "patient_id": patient,
                "image_path": relative_path,
                "split": split,
            }
            for finding in findings:
                label = str(source[finding]).strip()
                row[finding] = label if label in {"0", "0.0", "1", "1.0"} else ""
                label_counts[split][
                    (finding, "positive" if row[finding].startswith("1") else
                     "negative" if row[finding].startswith("0") else "masked")
                ] += 1
            rows_by_split[split].append(row)
            patients_by_split[split].add(patient)
            paths_by_split[split].add(relative_path)
            if not (args.image_root / relative_path).is_file():
                missing_images.append(relative_path)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest_hashes: dict[str, str] = {}
    for split in ("probe_train", "probe_validation", "fresh_development"):
        path = args.output_dir / f"{split}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows_by_split[split])
        manifest_hashes[split] = sha256_file(path)

    protected_patients, protected_paths = read_patients_and_paths(
        args.protected_valid_csv
    )
    split_names = tuple(rows_by_split)
    patient_overlaps = {
        f"{left}__{right}": len(
            patients_by_split[left] & patients_by_split[right]
        )
        for index, left in enumerate(split_names)
        for right in split_names[index + 1 :]
    }
    path_overlaps = {
        f"{left}__{right}": len(paths_by_split[left] & paths_by_split[right])
        for index, left in enumerate(split_names)
        for right in split_names[index + 1 :]
    }
    protected_overlap = {
        split: {
            "patients": len(patients_by_split[split] & protected_patients),
            "paths": len(paths_by_split[split] & protected_paths),
        }
        for split in split_names
    }
    counts = {
        split: {
            "rows": len(rows_by_split[split]),
            "patients": len(patients_by_split[split]),
            "labels": {
                finding: {
                    state: label_counts[split][(finding, state)]
                    for state in ("positive", "negative", "masked")
                }
                for finding in findings
            },
        }
        for split in split_names
    }
    checks.update(
        expected_counts=all(
            counts[split]["rows"] == data_lock["expected"][split]["rows"]
            and counts[split]["patients"]
            == data_lock["expected"][split]["patients"]
            for split in data_lock["expected"]
        ),
        patient_disjoint=all(value == 0 for value in patient_overlaps.values()),
        path_disjoint=all(value == 0 for value in path_overlaps.values()),
        protected_surface_disjoint=all(
            item["patients"] == 0 and item["paths"] == 0
            for item in protected_overlap.values()
        ),
        images_present=not missing_images,
        findings_nondegenerate=all(
            label_counts["fresh_development"][(finding, "positive")] > 0
            and label_counts["fresh_development"][(finding, "negative")] > 0
            for finding in findings
        ),
    )
    result = {
        "schema_version": 1,
        "artifact": "vivid_gds_fresh_split_audit",
        "pass": all(checks.values()),
        "checks": checks,
        "counts": counts,
        "patient_overlaps": patient_overlaps,
        "path_overlaps": path_overlaps,
        "protected_overlap": protected_overlap,
        "hashes": {
            "source_train_csv": sha256_file(args.train_csv),
            "protected_valid_csv": sha256_file(args.protected_valid_csv),
            "lock": sha256_file(args.lock),
            "manifests": manifest_hashes,
        },
        "missing_image_count": len(missing_images),
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
