#!/usr/bin/env python
"""Freeze the model/explanation/operator lock after validation download completion."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bives_cxr.provenance import canonical_json_sha256, file_sha256  # noqa: E402
from bives_cxr.qwen35_localization_audit import QWEN35_2B_SNAPSHOT_SHA256  # noqa: E402


INPUT = ROOT / "local_runs/cxr_localization_causality/chexlocalize_qwen35_development"
RELEASE = Path(r"H:\Xiyao_Wang\000_Public Dataset\CheXlocalize\redivis_v1_0\validation")
MODEL = Path(r"H:\Xiyao_Wang\001_models\Qwen3.5-2B")


def snapshot_model_files(model_root: Path) -> str:
    """Hash the frozen model payload using the established Qwen3.5 lock scope.

    Repository metadata, tokenizers, and documentation are deliberately outside
    this identity.  The opening locks the model configuration, safetensors
    index, and every local safetensors payload, matching the earlier VinDr and
    synthetic Qwen3.5 development gates.
    """

    names = {"config.json", "model.safetensors.index.json"}
    names.update(path.name for path in model_root.glob("*.safetensors"))
    files = {
        name: file_sha256(model_root / name)
        for name in sorted(names)
        if (model_root / name).is_file()
    }
    if "config.json" not in files or not any(
        name.endswith(".safetensors") for name in files
    ):
        raise ValueError("Qwen3.5 model payload is incomplete")
    return canonical_json_sha256(files)


def main() -> None:
    data_lock_path = INPUT / "development_data_lock.json"
    manifest_path = INPUT / "development_manifest.jsonl"
    download_lock_path = RELEASE / "validation_download_lock.json"
    data_lock = json.loads(data_lock_path.read_text(encoding="utf-8"))
    download_lock = json.loads(download_lock_path.read_text(encoding="utf-8"))
    if data_lock.get("status") != "score_free_data_ready" or data_lock.get("test_opened") is not False:
        raise ValueError("CheXlocalize score-free data lock is not ready")
    if download_lock.get("status") != "validation_download_complete" or download_lock.get("test_opened") is not False:
        raise ValueError("CheXlocalize validation download lock is not ready")
    if file_sha256(manifest_path) != data_lock.get("manifest_sha256"):
        raise ValueError("CheXlocalize development manifest changed")
    if snapshot_model_files(MODEL) != QWEN35_2B_SNAPSHOT_SHA256:
        raise ValueError("Qwen3.5-2B snapshot changed")
    source_paths = [
        ROOT / "audit/local_chexlocalize_qwen35_development_opening_20260719.json",
        ROOT / "bives_cxr/chexlocalize_validation.py",
        ROOT / "bives_cxr/localization_causality.py",
        ROOT / "bives_cxr/qwen35_localization_audit.py",
        ROOT / "scripts/run_chexlocalize_qwen35_development.py",
    ]
    payload = {
        "schema_version": "chexlocalize-qwen35-development-lock-v1",
        "status": "score_free_ready",
        "formal_result": False,
        "confirmatory_evidence": False,
        "test_opened": False,
        "patient_level_claim": True,
        "cluster_unit": "patient_id_hash",
        "samples": int(data_lock["counts"]["target_pairs"]),
        "patients": int(data_lock["counts"]["target_pair_patients"]),
        "findings": data_lock["counts"]["target_pair_findings"],
        "manifest_sha256": file_sha256(manifest_path),
        "data_lock_canonical_sha256": data_lock["canonical_sha256"],
        "download_lock_canonical_sha256": download_lock["canonical_sha256"],
        "model_path": MODEL.as_posix(),
        "model_snapshot_sha256": QWEN35_2B_SNAPSHOT_SHA256,
        "explanation": "local_mean_occlusion_4x4_stable_top1",
        "operators": ["local_mean_ring8", "masked_gaussian_blur_sigma8"],
        "shards": 2,
        "source_sha256": {str(path): file_sha256(path) for path in source_paths},
    }
    payload["canonical_sha256"] = canonical_json_sha256(payload)
    (INPUT / "development_lock.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
