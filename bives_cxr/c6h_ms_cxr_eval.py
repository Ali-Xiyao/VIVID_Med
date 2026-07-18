"""Fail-closed C6H lock and geometry contracts for one-time MS-CXR evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import ndimage

from .c6_ms_cxr_eval import EXPECTED_BOXES, EXPECTED_ROWS, validate_ms_cxr_manifest
from .provenance import canonical_json_sha256, file_sha256


LOCK_FORMAT_VERSION = "bives_c6h_ms_cxr_preopen_lock_v1"
C6G_LOCK_FORMAT_VERSION = "bives_c6g_ms_cxr_geometry_lock_v1"
C6G_CONTROL_VERSION = "bives_continuous_location_connected_control_v2"
EXPECTED_C6G_CANONICAL_SHA256 = (
    "6271ba51e8442baad92126473513b0b901619403a4e22c353e455395ec801752"
)
EXPECTED_STRICT_INTAKE_CANONICAL_SHA256 = (
    "0027358c2998773e73dbd19da02a37dac27c060150bf42e59469d218fb24b4ed"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def validate_c6h_protocol(config: dict[str, Any]) -> None:
    authorization = config.get("authorization", {})
    if (
        authorization.get("model_evaluation_authorized") is not True
        or authorization.get("one_time_execution_authorized") is not True
        or authorization.get("local_only") is not True
    ):
        raise ValueError("C6H requires explicit one-time local authorization")
    if config["model"]["family"] != "Qwen3.5" or config["model"]["scale"] != "2B":
        raise ValueError("C6H authorizes only Qwen3.5-2B")
    checkpoint = config["checkpoint"]
    if (
        checkpoint["variant"] != "B2_sparse_exact_k"
        or int(checkpoint["step"]) != 450
        or int(checkpoint["topk"]) != 16
    ):
        raise ValueError("C6H requires the frozen B2 step-450 exact-K=16 checkpoint")
    if config["geometry"]["control_version"] != C6G_CONTROL_VERSION:
        raise ValueError("C6H requires the frozen C6G v2 control")
    if config["geometry"]["lock_canonical_sha256"] != EXPECTED_C6G_CANONICAL_SHA256:
        raise ValueError("C6H C6G lock identity changed")
    intervention = config["intervention"]
    observed = (
        int(intervention["image_size"]),
        int(intervention["local_mean_ring_width"]),
        float(intervention["masked_gaussian_sigma"]),
        float(intervention["masked_gaussian_truncate"]),
        float(intervention["dilation_fraction"]),
    )
    if observed != (448, 8, 8.0, 3.0, 0.0):
        raise ValueError("C6H frozen intervention changed")
    evaluation = config["evaluation"]
    if (
        int(evaluation["bootstrap_replicates"]) != 2000
        or int(evaluation["bootstrap_seed"]) != 17
        or evaluation.get("allow_classification_metrics") is not False
        or evaluation.get("terminal_after_completion") is not True
    ):
        raise ValueError("C6H frozen evaluation boundary changed")
    if config["scale"].get("qwen35_4b_authorized") or config["scale"].get(
        "qwen35_9b_authorized"
    ):
        raise ValueError("C6H does not authorize Qwen3.5-4B/9B")


def validate_c6g_geometry_release(
    *,
    lock_path: Path,
    rows_path: Path,
    certificates_path: Path,
    manifest_path: Path,
    mask_dir: Path,
) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    declared = lock.get("canonical_sha256")
    summary_keys = (
        "format_version",
        "status",
        "rows",
        "eligible",
        "infeasible",
        "control_version",
        "image_size",
        "per_finding",
        "invariant_failures",
        "denominator_exclusions",
        "evaluation_gate_open_geometry",
        "model_evaluation_authorized",
        "gpu_authorized",
        "image_decode_authorized",
        "scores_accessed",
        "thresholds",
    )
    summary = {key: lock[key] for key in summary_keys}
    summary_canonical = canonical_json_sha256(summary)
    final_identity_payload = dict(lock)
    final_identity_payload["canonical_sha256"] = summary_canonical
    if declared != canonical_json_sha256(final_identity_payload):
        raise ValueError("C6G geometry lock canonical hash mismatch")
    if (
        lock.get("format_version") != C6G_LOCK_FORMAT_VERSION
        or lock.get("status") != "pass"
        or int(lock.get("rows", -1)) != 29
        or int(lock.get("eligible", -1)) != 29
        or int(lock.get("infeasible", -1)) != 0
        or int(lock.get("denominator_exclusions", -1)) != 0
        or int(lock.get("invariant_failures", -1)) != 0
        or lock.get("evaluation_gate_open_geometry") is not True
        or lock.get("model_evaluation_authorized") is not False
        or lock.get("gpu_authorized") is not False
        or lock.get("image_decode_authorized") is not False
        or lock.get("scores_accessed") is not False
        or lock.get("control_version") != C6G_CONTROL_VERSION
    ):
        raise ValueError("C6G geometry lock is not the frozen 29/29 score-free pass")
    expected_hashes = {
        "geometry_rows_sha256": file_sha256(rows_path),
        "candidate_certificates_sha256": file_sha256(certificates_path),
        "source_manifest_sha256": file_sha256(manifest_path),
    }
    mismatches = [key for key, value in expected_hashes.items() if lock.get(key) != value]
    if mismatches:
        raise ValueError("C6G geometry release mismatch: " + ", ".join(mismatches))
    rows = _read_jsonl(rows_path)
    certificates = _read_jsonl(certificates_path)
    if len(rows) != 29 or len(certificates) != 29:
        raise ValueError("C6G geometry release must contain 29 rows and certificates")
    row_ids = {str(row["sample_id"]) for row in rows}
    certificate_ids = {str(row["sample_id"]) for row in certificates}
    if len(row_ids) != 29 or row_ids != certificate_ids:
        raise ValueError("C6G row/certificate sample identities differ")
    for row in rows:
        sample_id = str(row["sample_id"])
        if (
            row.get("feasible") is not True
            or row.get("model_evaluation_authorized") is not False
            or row.get("gpu_authorized") is not False
            or row.get("image_decode_authorized") is not False
            or row.get("scores_accessed") is not False
        ):
            raise ValueError(f"C6G row gate changed: {sample_id}")
        mask_path = mask_dir / str(row["mask_file"])
        if not mask_path.is_file() or file_sha256(mask_path) != row["mask_sha256"]:
            raise ValueError(f"C6G mask identity changed: {sample_id}")
        with np.load(mask_path, allow_pickle=False) as payload:
            target = payload["target_mask"].astype(bool)
            control = payload["control_mask"].astype(bool)
            content = payload["content_mask"].astype(bool)
        if target.shape != (448, 448) or control.shape != target.shape or content.shape != target.shape:
            raise ValueError(f"C6G mask shape changed: {sample_id}")
        labels, component_count = ndimage.label(control, structure=ndimage.generate_binary_structure(2, 1))
        del labels
        if (
            int(target.sum()) != int(control.sum())
            or int(target.sum()) != int(row["target_area_pixels"])
            or int(control.sum()) != int(row["control_area_pixels"])
            or np.any(target & control)
            or np.any(target & ~content)
            or np.any(control & ~content)
            or int(component_count) != 1
        ):
            raise ValueError(f"C6G mask invariant changed: {sample_id}")
    return lock


def verify_c6f_immutability(c6g_lock: dict[str, Any], paths: dict[str, Path]) -> None:
    expected = c6g_lock.get("c6f_immutable_sha256", {})
    if set(expected) != set(paths):
        raise ValueError("C6F immutable path set changed")
    mismatches = [name for name, path in paths.items() if file_sha256(path) != expected[name]]
    if mismatches:
        raise ValueError("C6F immutable artifact changed: " + ", ".join(mismatches))


def build_c6h_lock(
    *,
    source_commit: str,
    authority_path: Path,
    config_path: Path,
    manifest_path: Path,
    strict_intake_path: Path,
    c6g_lock_path: Path,
    c6g_rows_path: Path,
    c6g_certificates_path: Path,
    c6g_mask_dir: Path,
    c6f_paths: dict[str, Path],
    source_paths: list[Path],
    artifact_paths: list[Path],
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_c6h_protocol(config)
    manifest_rows = _read_jsonl(manifest_path)
    counts = validate_ms_cxr_manifest(manifest_rows)
    if counts["per_finding"] != EXPECTED_ROWS or counts["per_finding_boxes"] != EXPECTED_BOXES:
        raise ValueError("C6H manifest counts changed")
    intake = json.loads(strict_intake_path.read_text(encoding="utf-8"))
    if (
        intake.get("canonical_artifact_sha256") != EXPECTED_STRICT_INTAKE_CANONICAL_SHA256
        or intake.get("license_gate_passed") is not True
        or intake.get("model_evaluation_authorized") is not False
    ):
        raise ValueError("strict C6E intake identity changed")
    c6g_lock = validate_c6g_geometry_release(
        lock_path=c6g_lock_path,
        rows_path=c6g_rows_path,
        certificates_path=c6g_certificates_path,
        manifest_path=manifest_path,
        mask_dir=c6g_mask_dir,
    )
    verify_c6f_immutability(c6g_lock, c6f_paths)
    payload: dict[str, Any] = {
        "format_version": LOCK_FORMAT_VERSION,
        "status": "pass_preopen",
        "evaluation_gate_open": True,
        "model_evaluation_authorized": True,
        "one_time_execution_authorized": True,
        "local_only": True,
        "positive_only": True,
        "classification_metrics_authorized": False,
        "qwen35_4b_authorized": False,
        "qwen35_9b_authorized": False,
        "source_commit": source_commit,
        "authority_sha256": file_sha256(authority_path),
        "config_sha256": file_sha256(config_path),
        "manifest_sha256": file_sha256(manifest_path),
        "strict_intake_sha256": file_sha256(strict_intake_path),
        "strict_intake_canonical_sha256": intake["canonical_artifact_sha256"],
        "c6g_lock_sha256": file_sha256(c6g_lock_path),
        "c6g_lock_canonical_sha256": c6g_lock["canonical_sha256"],
        "c6g_rows_sha256": file_sha256(c6g_rows_path),
        "c6g_certificates_sha256": file_sha256(c6g_certificates_path),
        "c6f_immutable_sha256": c6g_lock["c6f_immutable_sha256"],
        "counts": counts,
        "model_snapshot_sha256": config["model"]["snapshot_sha256"],
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "source_sha256": {str(path): file_sha256(path) for path in source_paths},
        "artifact_sha256": {str(path): file_sha256(path) for path in artifact_paths},
    }
    payload["canonical_artifact_sha256"] = canonical_json_sha256(payload)
    return payload


def validate_c6h_lock(
    lock_path: Path,
    *,
    authority_path: Path,
    config_path: Path,
    manifest_path: Path,
    strict_intake_path: Path,
    c6g_lock_path: Path,
    c6g_rows_path: Path,
    c6g_certificates_path: Path,
    c6g_mask_dir: Path,
) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    declared = lock.get("canonical_artifact_sha256")
    body = {key: value for key, value in lock.items() if key != "canonical_artifact_sha256"}
    if declared != canonical_json_sha256(body):
        raise ValueError("C6H pre-open lock canonical hash mismatch")
    required = {
        "authority_sha256": file_sha256(authority_path),
        "config_sha256": file_sha256(config_path),
        "manifest_sha256": file_sha256(manifest_path),
        "strict_intake_sha256": file_sha256(strict_intake_path),
        "c6g_lock_sha256": file_sha256(c6g_lock_path),
        "c6g_rows_sha256": file_sha256(c6g_rows_path),
        "c6g_certificates_sha256": file_sha256(c6g_certificates_path),
    }
    mismatches = [key for key, value in required.items() if lock.get(key) != value]
    if mismatches:
        raise ValueError("C6H pre-open lock mismatch: " + ", ".join(mismatches))
    if (
        lock.get("format_version") != LOCK_FORMAT_VERSION
        or lock.get("status") != "pass_preopen"
        or lock.get("evaluation_gate_open") is not True
        or lock.get("model_evaluation_authorized") is not True
        or lock.get("one_time_execution_authorized") is not True
        or lock.get("local_only") is not True
        or lock.get("positive_only") is not True
        or lock.get("classification_metrics_authorized") is not False
        or lock.get("qwen35_4b_authorized") is not False
        or lock.get("qwen35_9b_authorized") is not False
        or lock.get("counts", {}).get("rows") != 29
        or lock.get("c6g_lock_canonical_sha256") != EXPECTED_C6G_CANONICAL_SHA256
    ):
        raise ValueError("C6H pre-open gate is not open")
    validate_c6g_geometry_release(
        lock_path=c6g_lock_path,
        rows_path=c6g_rows_path,
        certificates_path=c6g_certificates_path,
        manifest_path=manifest_path,
        mask_dir=c6g_mask_dir,
    )
    for path_string, expected in {**lock["source_sha256"], **lock["artifact_sha256"]}.items():
        path = Path(path_string)
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"C6H locked artifact changed: {path}")
    return lock
