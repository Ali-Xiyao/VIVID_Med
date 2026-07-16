"""Audit an extracted VinDr-CXR package against its official manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pydicom


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = Path(r"H:\Xiyao_Wang\000_Public Dataset")
DATASET_DIRNAME = "vindr-cxr-an-open-dataset-of-chest-x-rays-with-radiologists-annotations-1.0.0"
DEFAULT_DATASET_ROOT = PUBLIC_ROOT / DATASET_DIRNAME
DEFAULT_JSON = ROOT / "outputs/final_tables/vindr_cxr_integrity_audit.json"
DEFAULT_MD = ROOT / "outputs/final_tables/vindr_cxr_integrity_audit.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--dicom-sha-samples-per-split", type=int, default=32)
    parser.add_argument("--dicom-decode-samples-per-split", type=int, default=8)
    parser.add_argument("--full-sha256", action="store_true")
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


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    marker = dataset_root / "_extraction_complete.json"
    if not marker.is_file():
        raise FileNotFoundError(f"Extraction completion marker is missing: {marker}")
    expected = read_official_hashes(dataset_root / "SHA256SUMS.txt")
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

    hash_mismatches = []
    for relative in hash_targets:
        actual = sha256(dataset_root / relative)
        if actual != expected[relative]:
            hash_mismatches.append({"path": relative, "expected": expected[relative], "actual": actual})

    decode_targets = evenly_spaced(train_dicoms, args.dicom_decode_samples_per_split)
    decode_targets += evenly_spaced(test_dicoms, args.dicom_decode_samples_per_split)
    decode_failures = []
    decoded: list[dict[str, Any]] = []
    for relative in decode_targets:
        try:
            dicom = pydicom.dcmread(dataset_root / relative)
            pixels = dicom.pixel_array
            decoded.append(
                {
                    "path": relative,
                    "shape": list(pixels.shape),
                    "dtype": str(pixels.dtype),
                    "min": int(np.min(pixels)),
                    "max": int(np.max(pixels)),
                    "photometric_interpretation": str(dicom.PhotometricInterpretation),
                }
            )
        except Exception as exc:  # noqa: BLE001 - audit must preserve exact failure text.
            decode_failures.append({"path": relative, "error": f"{type(exc).__name__}: {exc}"})

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
        "sha256_checked": len(hash_targets),
        "sha256_mismatches": hash_mismatches,
        "dicom_decode_checked": len(decode_targets),
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
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
