"""Joint four-split dataset locks for formal BiVES-CXR releases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .audit import audit_manifests
from .data import read_manifest
from .provenance import canonical_json_sha256, file_sha256


DATASET_LOCK_FORMAT_VERSION = 1
REQUIRED_SPLITS = ("train", "val", "calibration", "test")


def _split_set_hashes(manifest_path: str | Path) -> dict[str, str]:
    rows = read_manifest(manifest_path)
    fields = {
        "patient_ids": sorted({str(row["patient_id"]) for row in rows}),
        "study_ids": sorted({str(row.get("study_id", "")).strip() for row in rows}),
        "image_hashes": sorted({str(row.get("image_sha256", "")).strip().lower() for row in rows}),
        "group_ids": sorted({str(row["group_id"]) for row in rows}),
    }
    if any(not value for value in fields["study_ids"] + fields["image_hashes"]):
        raise ValueError(f"dataset lock requires non-empty study_id and image_sha256: {manifest_path}")
    return {name: canonical_json_sha256(values) for name, values in fields.items()}


def build_dataset_lock(
    manifests: dict[str, str | Path],
    *,
    data_root: str | Path,
    audit_options: dict[str, Any],
) -> dict[str, Any]:
    missing = set(REQUIRED_SPLITS) - set(manifests)
    if missing:
        raise ValueError(f"dataset lock requires all four splits; missing={sorted(missing)}")
    normalized = {split: Path(manifests[split]) for split in REQUIRED_SPLITS}
    audit = audit_manifests(normalized, data_root=data_root, **audit_options)
    if audit["status"] != "pass":
        raise ValueError("joint four-split dataset audit failed: " + "; ".join(audit["errors"][:5]))
    return {
        "format_version": DATASET_LOCK_FORMAT_VERSION,
        "status": "pass",
        "manifest_sha256": {split: file_sha256(path) for split, path in normalized.items()},
        "joint_audit_report_sha256": canonical_json_sha256(audit),
        "split_set_sha256": {split: _split_set_hashes(path) for split, path in normalized.items()},
        "audit_options": dict(audit_options),
    }


def dataset_lock_sha256(lock: dict[str, Any]) -> str:
    return canonical_json_sha256(lock)


def validate_dataset_lock(
    lock_path: str | Path,
    manifests: dict[str, str | Path],
    *,
    data_root: str | Path,
    audit_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import json

    path = Path(lock_path)
    if not path.is_file():
        raise FileNotFoundError(f"formal dataset lock is missing: {path}")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("format_version") != DATASET_LOCK_FORMAT_VERSION or lock.get("status") != "pass":
        raise ValueError("dataset lock is not a passing supported lock")
    options = dict(audit_options if audit_options is not None else lock.get("audit_options", {}))
    rebuilt = build_dataset_lock(manifests, data_root=data_root, audit_options=options)
    for field in ("manifest_sha256", "joint_audit_report_sha256", "split_set_sha256", "audit_options"):
        if lock.get(field) != rebuilt.get(field):
            raise ValueError(f"dataset lock mismatch: {field}")
    return lock
