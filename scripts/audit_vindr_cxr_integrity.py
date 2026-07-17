"""Audit an extracted VinDr-CXR package against its official manifest."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bives_cxr.dicom import DICOM_PREPROCESS_VERSION, load_cxr_dicom  # noqa: E402


PUBLIC_ROOT = Path(r"H:\Xiyao_Wang\000_Public Dataset")
DATASET_DIRNAME = "vindr-cxr-an-open-dataset-of-chest-x-rays-with-radiologists-annotations-1.0.0"
DEFAULT_DATASET_ROOT = PUBLIC_ROOT / DATASET_DIRNAME
DEFAULT_JSON = ROOT / "outputs/final_tables/vindr_cxr_integrity_audit.json"
DEFAULT_MD = ROOT / "outputs/final_tables/vindr_cxr_integrity_audit.md"
DEFAULT_PROGRESS = ROOT / "local_runs/bives_cxr/vindr_formal_preflight_progress.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--dicom-sha-samples-per-split", type=int, default=32)
    parser.add_argument("--dicom-decode-samples-per-split", type=int, default=8)
    parser.add_argument("--decode-all-test", action="store_true")
    parser.add_argument("--decode-workers", type=int, default=1)
    parser.add_argument("--full-sha256", action="store_true")
    parser.add_argument("--progress-json", type=Path, default=DEFAULT_PROGRESS)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evenly_spaced(items: list[str], count: int) -> list[str]:
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return items
    indices = np.linspace(0, len(items) - 1, num=count, dtype=int)
    return [items[int(index)] for index in indices]


def read_official_hashes(path: Path) -> dict[str, str]:
    hashes = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        hashes[relative.strip().replace("\\", "/")] = digest.lower()
    return hashes


def decode_one(payload: tuple[str, str]) -> tuple[str, dict[str, Any] | None, str | None]:
    dataset_root, relative = payload
    try:
        image, record = load_cxr_dicom(Path(dataset_root) / relative)
        return (
            relative,
            {
                "path": relative,
                "shape": [image.height, image.width, 3],
                **record.to_dict(),
            },
            None,
        )
    except Exception as exc:  # noqa: BLE001 - preserve worker failure text.
        return relative, None, f"{type(exc).__name__}: {exc}"


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    marker = dataset_root / "_extraction_complete.json"
    if not marker.is_file():
        raise FileNotFoundError(f"Extraction completion marker is missing: {marker}")
    expected = read_official_hashes(dataset_root / "SHA256SUMS.txt")
    official_manifest_sha256 = sha256(dataset_root / "SHA256SUMS.txt")
    missing = sorted(relative for relative in expected if not (dataset_root / relative).is_file())

    train_dicoms = sorted(relative for relative in expected if relative.startswith("train/") and relative.endswith(".dicom"))
    test_dicoms = sorted(relative for relative in expected if relative.startswith("test/") and relative.endswith(".dicom"))
    metadata_files = sorted(relative for relative in expected if not relative.endswith(".dicom"))
    if args.full_sha256:
        hash_targets = sorted(expected)
    else:
        hash_targets = metadata_files
        hash_targets += evenly_spaced(train_dicoms, args.dicom_sha_samples_per_split)
        hash_targets += evenly_spaced(test_dicoms, args.dicom_sha_samples_per_split)
        hash_targets = sorted(set(hash_targets))

    progress: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "official_manifest_sha256": official_manifest_sha256,
        "full_sha256": bool(args.full_sha256),
        "decode_all_test": bool(args.decode_all_test),
        "hash_passed": [],
        "hash_mismatches": [],
        "decode_passed": {},
        "decode_failures": [],
    }
    if args.progress_json.is_file():
        try:
            candidate = json.loads(args.progress_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            candidate = {}
        identity = (
            candidate.get("dataset_root") == str(dataset_root)
            and candidate.get("official_manifest_sha256") == official_manifest_sha256
            and bool(candidate.get("full_sha256")) == bool(args.full_sha256)
            and bool(candidate.get("decode_all_test")) == bool(args.decode_all_test)
        )
        if identity:
            progress = candidate
    args.progress_json.parent.mkdir(parents=True, exist_ok=True)

    def save_progress() -> None:
        temporary = args.progress_json.with_suffix(args.progress_json.suffix + ".tmp")
        temporary.write_text(
            json.dumps(progress, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, args.progress_json)

    hash_passed = set(progress.get("hash_passed", []))
    hash_mismatches = list(progress.get("hash_mismatches", []))
    mismatch_paths = {row["path"] for row in hash_mismatches}
    for index, relative in enumerate(hash_targets, start=1):
        if relative in hash_passed or relative in mismatch_paths:
            continue
        actual = sha256(dataset_root / relative)
        if actual != expected[relative]:
            hash_mismatches.append({"path": relative, "expected": expected[relative], "actual": actual})
            mismatch_paths.add(relative)
        else:
            hash_passed.add(relative)
        if index % 25 == 0:
            progress["hash_passed"] = sorted(hash_passed)
            progress["hash_mismatches"] = hash_mismatches
            progress["phase"] = "sha256"
            progress["hash_completed"] = len(hash_passed) + len(hash_mismatches)
            progress["hash_total"] = len(hash_targets)
            save_progress()

    decode_targets = evenly_spaced(train_dicoms, args.dicom_decode_samples_per_split)
    decode_targets += (
        test_dicoms
        if args.decode_all_test
        else evenly_spaced(test_dicoms, args.dicom_decode_samples_per_split)
    )
    decode_failures = list(progress.get("decode_failures", []))
    decode_failure_paths = {row["path"] for row in decode_failures}
    decoded_by_path: dict[str, dict[str, Any]] = dict(progress.get("decode_passed", {}))
    pending_decode = [
        relative
        for relative in decode_targets
        if relative not in decoded_by_path and relative not in decode_failure_paths
    ]
    workers = max(1, int(args.decode_workers))
    if workers == 1:
        decoded_stream = map(decode_one, [(str(dataset_root), relative) for relative in pending_decode])
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        decoded_stream = executor.map(
            decode_one,
            [(str(dataset_root), relative) for relative in pending_decode],
            chunksize=1,
        )
    try:
        for index, (relative, record, error) in enumerate(decoded_stream, start=1):
            if error is None:
                assert record is not None
                decoded_by_path[relative] = record
            else:
                decode_failures.append({"path": relative, "error": error})
                decode_failure_paths.add(relative)
            if index % 10 != 0:
                continue
            progress["hash_passed"] = sorted(hash_passed)
            progress["hash_mismatches"] = hash_mismatches
            progress["decode_passed"] = decoded_by_path
            progress["decode_failures"] = decode_failures
            progress["phase"] = "decode"
            progress["decode_completed"] = len(decoded_by_path) + len(decode_failures)
            progress["decode_total"] = len(decode_targets)
            save_progress()
    finally:
        if workers != 1:
            executor.shutdown(wait=True)

    decoded = [decoded_by_path[path] for path in decode_targets if path in decoded_by_path]

    status = "pass"
    if missing or hash_mismatches or decode_failures or len(train_dicoms) != 15000 or len(test_dicoms) != 3000:
        status = "fail"
    payload = {
        "status": status,
        "dataset_root": str(dataset_root),
        "official_manifest_entries": len(expected),
        "train_dicoms": len(train_dicoms),
        "test_dicoms": len(test_dicoms),
        "missing_files": len(missing),
        "missing_examples": missing[:20],
        "sha256_mode": "full" if args.full_sha256 else "all metadata plus deterministic DICOM sample",
        "sha256_checked": len(hash_passed) + len(hash_mismatches),
        "sha256_mismatches": hash_mismatches,
        "dicom_decode_checked": len(decode_targets),
        "dicom_decode_mode": "all_test_plus_train_sample" if args.decode_all_test else "deterministic_samples",
        "dicom_preprocess_version": DICOM_PREPROCESS_VERSION,
        "dicom_decode_failures": decode_failures,
        "decoded_examples": decoded,
        "crc_evidence": json.loads(marker.read_text(encoding="utf-8")).get("zip_crc_note", ""),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_md.write_text(
        "# VinDr-CXR Integrity Audit\n\n"
        f"- Status: `{status}`\n"
        f"- Official manifest entries: {len(expected)}\n"
        f"- DICOMs: train={len(train_dicoms)}, test={len(test_dicoms)}\n"
        f"- Missing files: {len(missing)}\n"
        f"- SHA-256 mode/checks/mismatches: {payload['sha256_mode']} / {len(hash_targets)} / {len(hash_mismatches)}\n"
        f"- DICOM decode checks/failures: {len(decode_targets)} / {len(decode_failures)}\n"
        f"- ZIP extraction CRC evidence: {payload['crc_evidence']}\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    progress.update(
        {
            "phase": "complete",
            "status": status,
            "hash_passed": sorted(hash_passed),
            "hash_mismatches": hash_mismatches,
            "decode_passed": decoded_by_path,
            "decode_failures": decode_failures,
            "output_json": str(args.output_json),
        }
    )
    save_progress()
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
