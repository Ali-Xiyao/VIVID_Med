#!/usr/bin/env python
"""Verify and freeze the completed validation-only Redivis download."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bives_cxr.chexlocalize_acquisition import select_validation_files  # noqa: E402
from bives_cxr.provenance import canonical_json_sha256, file_sha256  # noqa: E402


RELEASE = Path(
    r"H:\Xiyao_Wang\000_Public Dataset\CheXlocalize\redivis_v1_0\validation"
)


def md5_hex(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    inventory_path = RELEASE / "validation_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    selected = select_validation_files(inventory["files"])
    failures = []
    for row in selected:
        path = RELEASE / Path(str(row["path"]))
        if not path.is_file():
            failures.append({"path_sha256": hashlib.sha256(str(row["path"]).encode()).hexdigest(), "reason": "missing"})
            continue
        if path.stat().st_size != int(row["size"]):
            failures.append({"path_sha256": hashlib.sha256(str(row["path"]).encode()).hexdigest(), "reason": "size"})
            continue
        if md5_hex(path) != str(row["md5"]):
            failures.append({"path_sha256": hashlib.sha256(str(row["path"]).encode()).hexdigest(), "reason": "md5"})
    if failures:
        raise ValueError(f"CheXlocalize validation download verification failed: {len(failures)}")
    payload = {
        "schema_version": "chexlocalize-validation-download-lock-v1",
        "status": "validation_download_complete",
        "release_reference": inventory["dataset_reference"],
        "file_count": len(selected),
        "total_bytes": sum(int(row["size"]) for row in selected),
        "inventory_sha256": file_sha256(inventory_path),
        "test_opened": False,
        "test_files_present": False,
        "all_redivis_md5_verified": True,
    }
    payload["canonical_sha256"] = canonical_json_sha256(payload)
    lock_path = RELEASE / "validation_download_lock.json"
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
