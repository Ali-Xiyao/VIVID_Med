"""C6I actual-input-space geometry and fail-closed evaluation contracts."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage

from .c6_ms_cxr_eval import (
    EXPECTED_BOXES,
    EXPECTED_ROWS,
    IMAGE_SIZE,
    _content_geometry,
    validate_ms_cxr_manifest,
)
from .c6g_geometry import (
    ControlSearchFailure,
    deterministic_continuous_location_connected_control_mask,
)
from .pixel_interventions import transform_mask_to_letterbox, union_box_mask
from .provenance import canonical_json_sha256, file_sha256


CONTROL_VERSION = "bives_actual_input_continuous_location_connected_control_v1"
COORDINATE_TRANSFORM_VERSION = "ms_cxr_native_box_to_bound_jpg_xy_scale_v1"
GEOMETRY_FORMAT_VERSION = "bives_c6i_ms_cxr_actual_input_geometry_v1"
GEOMETRY_LOCK_FORMAT_VERSION = "bives_c6i_ms_cxr_actual_input_geometry_lock_v1"
PREOPEN_LOCK_FORMAT_VERSION = "bives_c6i_ms_cxr_replacement_preopen_lock_v1"
EXPECTED_ACTUAL_SIZE = (224, 224)
EXPECTED_STRICT_INTAKE_CANONICAL_SHA256 = (
    "0027358c2998773e73dbd19da02a37dac27c060150bf42e59469d218fb24b4ed"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def scale_boxes_to_actual_input(
    boxes: list[dict[str, Any]],
    *,
    native_width: int,
    native_height: int,
    actual_width: int,
    actual_height: int,
) -> list[dict[str, float]]:
    """Map released native-image boxes to the exact bound JPG coordinate space."""

    if min(native_width, native_height, actual_width, actual_height) <= 0:
        raise ValueError("C6I image dimensions must be positive")
    scale_x = actual_width / float(native_width)
    scale_y = actual_height / float(native_height)
    transformed: list[dict[str, float]] = []
    for box in boxes:
        mapped = {
            "x_min": float(box["x_min"]) * scale_x,
            "y_min": float(box["y_min"]) * scale_y,
            "x_max": float(box["x_max"]) * scale_x,
            "y_max": float(box["y_max"]) * scale_y,
        }
        if (
            mapped["x_min"] < 0
            or mapped["y_min"] < 0
            or mapped["x_max"] > actual_width + 1e-9
            or mapped["y_max"] > actual_height + 1e-9
            or mapped["x_max"] <= mapped["x_min"]
            or mapped["y_max"] <= mapped["y_min"]
        ):
            raise ValueError("C6I transformed box is outside the bound JPG")
        transformed.append(mapped)
    if not transformed:
        raise ValueError("C6I requires at least one released box")
    return transformed


def _build_geometry_row(
    payload: tuple[dict[str, Any], str, dict[str, Any]],
) -> dict[str, Any]:
    row, mask_dir_string, thresholds = payload
    image_path = Path(str(row["image_path"]))
    if file_sha256(image_path) != row["official_image_sha256"]:
        raise ValueError(f"C6I image hash mismatch: {row['sample_id']}")
    with Image.open(image_path) as image:
        actual_width, actual_height = image.size
    if (actual_width, actual_height) != EXPECTED_ACTUAL_SIZE:
        raise ValueError(
            f"C6I bound JPG size changed for {row['sample_id']}: "
            f"{actual_width}x{actual_height}"
        )
    native_width = int(row["native_columns"])
    native_height = int(row["native_rows"])
    transformed_boxes = scale_boxes_to_actual_input(
        list(row["bounding_boxes"]),
        native_width=native_width,
        native_height=native_height,
        actual_width=actual_width,
        actual_height=actual_height,
    )
    content_box, content = _content_geometry(actual_width, actual_height)
    target = transform_mask_to_letterbox(
        union_box_mask(actual_width, actual_height, transformed_boxes),
        content_box,
        IMAGE_SIZE,
    ) & content
    base = {
        "format_version": GEOMETRY_FORMAT_VERSION,
        "sample_id": row["sample_id"],
        "canonical_statement_id": row["canonical_statement_id"],
        "coordinate_transform_version": COORDINATE_TRANSFORM_VERSION,
        "native_columns": native_width,
        "native_rows": native_height,
        "actual_columns": actual_width,
        "actual_rows": actual_height,
        "scale_x": actual_width / float(native_width),
        "scale_y": actual_height / float(native_height),
        "transformed_boxes": transformed_boxes,
        "target_area_pixels": int(target.sum()),
        "target_area_fraction": float(target.sum() / content.sum()),
        "image_header_accessed": True,
        "image_pixels_scored": False,
        "model_evaluation_authorized": False,
        "gpu_authorized": False,
        "scores_accessed": False,
    }
    try:
        control, audit = deterministic_continuous_location_connected_control_mask(
            target,
            content,
            seed_key=f"{row['sample_id']}:{CONTROL_VERSION}",
            thresholds=thresholds,
        )
    except ControlSearchFailure as error:
        return {
            **base,
            "control_area_pixels": 0,
            "mask_file": None,
            "mask_sha256": None,
            "control_audit": error.certificate,
            "feasible": False,
            "failure": str(error),
        }
    mask_name = hashlib.sha256(str(row["sample_id"]).encode("utf-8")).hexdigest() + ".npz"
    mask_path = Path(mask_dir_string) / mask_name
    temporary = mask_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        target_mask=target,
        control_mask=control,
        content_mask=content,
    )
    temporary.replace(mask_path)
    return {
        **base,
        "control_area_pixels": int(control.sum()),
        "mask_file": mask_name,
        "mask_sha256": file_sha256(mask_path),
        "control_audit": audit,
        "feasible": True,
        "failure": None,
    }


def build_c6i_geometry(
    rows: list[dict[str, Any]],
    *,
    mask_dir: Path,
    thresholds: dict[str, Any],
    workers: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_ms_cxr_manifest(rows)
    if workers <= 0:
        raise ValueError("C6I geometry workers must be positive")
    mask_dir.mkdir(parents=True, exist_ok=True)
    payloads = [(row, str(mask_dir), thresholds) for row in rows]
    if workers == 1:
        geometry_rows = [_build_geometry_row(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            geometry_rows = list(executor.map(_build_geometry_row, payloads))
    by_finding = {
        finding: sorted(
            (row for row in geometry_rows if row["canonical_statement_id"] == finding),
            key=lambda row: (float(row["target_area_fraction"]), str(row["sample_id"])),
        )
        for finding in EXPECTED_ROWS
    }
    for subset in by_finding.values():
        boundaries = np.quantile(
            [float(row["target_area_fraction"]) for row in subset], [0.25, 0.5, 0.75]
        )
        for row in subset:
            row["box_area_quartile"] = int(
                np.searchsorted(
                    boundaries, float(row["target_area_fraction"]), side="right"
                )
                + 1
            )
    geometry_rows.sort(key=lambda row: str(row["sample_id"]))
    summary = {
        "format_version": GEOMETRY_LOCK_FORMAT_VERSION,
        "status": "pass" if all(row["feasible"] for row in geometry_rows) else "fail_geometry",
        "rows": len(geometry_rows),
        "eligible": sum(bool(row["feasible"]) for row in geometry_rows),
        "infeasible": sum(not bool(row["feasible"]) for row in geometry_rows),
        "control_version": CONTROL_VERSION,
        "coordinate_transform_version": COORDINATE_TRANSFORM_VERSION,
        "model_input_size": IMAGE_SIZE,
        "actual_image_sizes": {"224x224": len(geometry_rows)},
        "per_finding": {
            finding: sum(row["canonical_statement_id"] == finding for row in geometry_rows)
            for finding in EXPECTED_ROWS
        },
        "invariant_failures": 0,
        "denominator_exclusions": 0,
        "evaluation_gate_open_geometry": all(row["feasible"] for row in geometry_rows),
        "model_evaluation_authorized": False,
        "gpu_authorized": False,
        "scores_accessed": False,
        "thresholds": thresholds,
    }
    if summary["rows"] != 29 or summary["per_finding"] != EXPECTED_ROWS:
        raise ValueError("C6I geometry denominator changed")
    return geometry_rows, summary


def write_c6i_geometry_artifacts(
    *,
    output_dir: Path,
    geometry_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    manifest_path: Path,
    authority_path: Path,
    threshold_path: Path,
    predecessor_paths: dict[str, Path],
    implementation_paths: dict[str, Path],
    source_commit: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "c6i_geometry_rows.jsonl"
    certificates_path = output_dir / "c6i_candidate_certificates.jsonl"
    lock_path = output_dir / "c6i_geometry_lock.json"
    write_jsonl(rows_path, geometry_rows)
    write_jsonl(
        certificates_path,
        [
            {
                "sample_id": row["sample_id"],
                "canonical_statement_id": row["canonical_statement_id"],
                "feasible": row["feasible"],
                "failure": row["failure"],
                "coordinate_transform_version": row["coordinate_transform_version"],
                "native_size": [row["native_columns"], row["native_rows"]],
                "actual_size": [row["actual_columns"], row["actual_rows"]],
                "scale_xy": [row["scale_x"], row["scale_y"]],
                "transformed_boxes": row["transformed_boxes"],
                "certificate": row["control_audit"],
            }
            for row in geometry_rows
        ],
    )
    lock = {
        **summary,
        "source_manifest_sha256": file_sha256(manifest_path),
        "authority_sha256": file_sha256(authority_path),
        "threshold_artifact_sha256": file_sha256(threshold_path),
        "geometry_rows_sha256": file_sha256(rows_path),
        "candidate_certificates_sha256": file_sha256(certificates_path),
        "predecessor_immutable_sha256": {
            name: file_sha256(path) for name, path in sorted(predecessor_paths.items())
        },
        "implementation_sha256": {
            name: file_sha256(path) for name, path in sorted(implementation_paths.items())
        },
        "source_commit": source_commit,
    }
    lock["canonical_sha256"] = canonical_json_sha256(lock)
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "geometry_rows": rows_path,
        "candidate_certificates": certificates_path,
        "geometry_lock": lock_path,
        "lock": lock,
    }


def validate_c6i_geometry_release(
    *,
    lock_path: Path,
    rows_path: Path,
    certificates_path: Path,
    manifest_path: Path,
    mask_dir: Path,
) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    declared = lock.get("canonical_sha256")
    body = {key: value for key, value in lock.items() if key != "canonical_sha256"}
    if declared != canonical_json_sha256(body):
        raise ValueError("C6I geometry lock canonical hash mismatch")
    if (
        lock.get("format_version") != GEOMETRY_LOCK_FORMAT_VERSION
        or lock.get("status") != "pass"
        or int(lock.get("rows", -1)) != 29
        or int(lock.get("eligible", -1)) != 29
        or int(lock.get("infeasible", -1)) != 0
        or int(lock.get("denominator_exclusions", -1)) != 0
        or int(lock.get("invariant_failures", -1)) != 0
        or lock.get("evaluation_gate_open_geometry") is not True
        or lock.get("model_evaluation_authorized") is not False
        or lock.get("gpu_authorized") is not False
        or lock.get("scores_accessed") is not False
        or lock.get("control_version") != CONTROL_VERSION
        or lock.get("coordinate_transform_version") != COORDINATE_TRANSFORM_VERSION
        or lock.get("actual_image_sizes") != {"224x224": 29}
    ):
        raise ValueError("C6I geometry lock is not the frozen 29/29 score-free pass")
    expected = {
        "geometry_rows_sha256": file_sha256(rows_path),
        "candidate_certificates_sha256": file_sha256(certificates_path),
        "source_manifest_sha256": file_sha256(manifest_path),
    }
    mismatches = [key for key, value in expected.items() if lock.get(key) != value]
    if mismatches:
        raise ValueError("C6I geometry release mismatch: " + ", ".join(mismatches))
    rows = read_jsonl(rows_path)
    certificates = read_jsonl(certificates_path)
    if len(rows) != 29 or len(certificates) != 29:
        raise ValueError("C6I release must contain 29 rows and certificates")
    if {str(row["sample_id"]) for row in rows} != {
        str(row["sample_id"]) for row in certificates
    }:
        raise ValueError("C6I row/certificate identities differ")
    for row in rows:
        sample_id = str(row["sample_id"])
        if (
            row.get("feasible") is not True
            or row.get("actual_columns") != 224
            or row.get("actual_rows") != 224
            or row.get("image_pixels_scored") is not False
            or row.get("model_evaluation_authorized") is not False
            or row.get("gpu_authorized") is not False
            or row.get("scores_accessed") is not False
        ):
            raise ValueError(f"C6I row gate changed: {sample_id}")
        mask_path = mask_dir / str(row["mask_file"])
        if not mask_path.is_file() or file_sha256(mask_path) != row["mask_sha256"]:
            raise ValueError(f"C6I mask identity changed: {sample_id}")
        with np.load(mask_path, allow_pickle=False) as payload:
            target = payload["target_mask"].astype(bool)
            control = payload["control_mask"].astype(bool)
            content = payload["content_mask"].astype(bool)
        _, component_count = ndimage.label(
            control, structure=ndimage.generate_binary_structure(2, 1)
        )
        if (
            target.shape != (448, 448)
            or control.shape != target.shape
            or content.shape != target.shape
            or not bool(content.all())
            or int(target.sum()) != int(row["target_area_pixels"])
            or int(control.sum()) != int(row["control_area_pixels"])
            or int(target.sum()) != int(control.sum())
            or np.any(target & control)
            or np.any(target & ~content)
            or np.any(control & ~content)
            or int(component_count) != 1
        ):
            raise ValueError(f"C6I mask invariant changed: {sample_id}")
    return lock


def validate_c6i_protocol(config: dict[str, Any]) -> None:
    authorization = config.get("authorization", {})
    if (
        authorization.get("model_evaluation_authorized") is not True
        or authorization.get("replacement_one_time_execution_authorized") is not True
        or authorization.get("local_only") is not True
    ):
        raise ValueError("C6I requires explicit replacement one-time local authorization")
    if config["model"]["family"] != "Qwen3.5" or config["model"]["scale"] != "2B":
        raise ValueError("C6I authorizes only Qwen3.5-2B")
    checkpoint = config["checkpoint"]
    if (
        checkpoint["variant"] != "B2_sparse_exact_k"
        or int(checkpoint["step"]) != 450
        or int(checkpoint["topk"]) != 16
    ):
        raise ValueError("C6I requires the frozen B2 step-450 exact-K=16 checkpoint")
    geometry = config["geometry"]
    if (
        geometry["control_version"] != CONTROL_VERSION
        or geometry["coordinate_transform_version"] != COORDINATE_TRANSFORM_VERSION
        or geometry.get("required_actual_size") != [224, 224]
    ):
        raise ValueError("C6I actual-input geometry contract changed")
    intervention = config["intervention"]
    observed = (
        int(intervention["image_size"]),
        int(intervention["local_mean_ring_width"]),
        float(intervention["masked_gaussian_sigma"]),
        float(intervention["masked_gaussian_truncate"]),
        float(intervention["dilation_fraction"]),
    )
    if observed != (448, 8, 8.0, 3.0, 0.0):
        raise ValueError("C6I frozen intervention changed")
    evaluation = config["evaluation"]
    if (
        int(evaluation["bootstrap_replicates"]) != 2000
        or int(evaluation["bootstrap_seed"]) != 17
        or evaluation.get("allow_classification_metrics") is not False
        or evaluation.get("terminal_after_completion") is not True
    ):
        raise ValueError("C6I frozen evaluation boundary changed")
    if config["scale"].get("qwen35_4b_authorized") or config["scale"].get(
        "qwen35_9b_authorized"
    ):
        raise ValueError("C6I does not authorize Qwen3.5-4B/9B")


def build_c6i_preopen_lock(
    *,
    source_commit: str,
    authority_path: Path,
    config_path: Path,
    manifest_path: Path,
    strict_intake_path: Path,
    geometry_lock_path: Path,
    geometry_rows_path: Path,
    geometry_certificates_path: Path,
    geometry_mask_dir: Path,
    predecessor_paths: dict[str, Path],
    source_paths: list[Path],
    artifact_paths: list[Path],
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_c6i_protocol(config)
    manifest_rows = read_jsonl(manifest_path)
    counts = validate_ms_cxr_manifest(manifest_rows)
    if counts["per_finding"] != EXPECTED_ROWS or counts["per_finding_boxes"] != EXPECTED_BOXES:
        raise ValueError("C6I manifest counts changed")
    intake = json.loads(strict_intake_path.read_text(encoding="utf-8"))
    if (
        intake.get("canonical_artifact_sha256") != EXPECTED_STRICT_INTAKE_CANONICAL_SHA256
        or intake.get("license_gate_passed") is not True
        or intake.get("model_evaluation_authorized") is not False
    ):
        raise ValueError("C6I strict intake identity changed")
    geometry_lock = validate_c6i_geometry_release(
        lock_path=geometry_lock_path,
        rows_path=geometry_rows_path,
        certificates_path=geometry_certificates_path,
        manifest_path=manifest_path,
        mask_dir=geometry_mask_dir,
    )
    predecessor_hashes = {
        name: file_sha256(path) for name, path in sorted(predecessor_paths.items())
    }
    if predecessor_hashes != geometry_lock["predecessor_immutable_sha256"]:
        raise ValueError("C6I predecessor immutability changed after geometry freeze")
    payload: dict[str, Any] = {
        "format_version": PREOPEN_LOCK_FORMAT_VERSION,
        "status": "pass_preopen",
        "evaluation_gate_open": True,
        "model_evaluation_authorized": True,
        "replacement_one_time_execution_authorized": True,
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
        "geometry_lock_sha256": file_sha256(geometry_lock_path),
        "geometry_lock_canonical_sha256": geometry_lock["canonical_sha256"],
        "geometry_rows_sha256": file_sha256(geometry_rows_path),
        "geometry_certificates_sha256": file_sha256(geometry_certificates_path),
        "predecessor_immutable_sha256": predecessor_hashes,
        "counts": counts,
        "model_snapshot_sha256": config["model"]["snapshot_sha256"],
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "source_sha256": {str(path): file_sha256(path) for path in source_paths},
        "artifact_sha256": {str(path): file_sha256(path) for path in artifact_paths},
    }
    payload["canonical_artifact_sha256"] = canonical_json_sha256(payload)
    return payload


def validate_c6i_preopen_lock(
    lock_path: Path,
    *,
    authority_path: Path,
    config_path: Path,
    manifest_path: Path,
    strict_intake_path: Path,
    geometry_lock_path: Path,
    geometry_rows_path: Path,
    geometry_certificates_path: Path,
    geometry_mask_dir: Path,
) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    declared = lock.get("canonical_artifact_sha256")
    body = {key: value for key, value in lock.items() if key != "canonical_artifact_sha256"}
    if declared != canonical_json_sha256(body):
        raise ValueError("C6I pre-open lock canonical hash mismatch")
    required = {
        "authority_sha256": file_sha256(authority_path),
        "config_sha256": file_sha256(config_path),
        "manifest_sha256": file_sha256(manifest_path),
        "strict_intake_sha256": file_sha256(strict_intake_path),
        "geometry_lock_sha256": file_sha256(geometry_lock_path),
        "geometry_rows_sha256": file_sha256(geometry_rows_path),
        "geometry_certificates_sha256": file_sha256(geometry_certificates_path),
    }
    mismatches = [key for key, value in required.items() if lock.get(key) != value]
    if mismatches:
        raise ValueError("C6I pre-open lock mismatch: " + ", ".join(mismatches))
    if (
        lock.get("format_version") != PREOPEN_LOCK_FORMAT_VERSION
        or lock.get("status") != "pass_preopen"
        or lock.get("evaluation_gate_open") is not True
        or lock.get("model_evaluation_authorized") is not True
        or lock.get("replacement_one_time_execution_authorized") is not True
        or lock.get("local_only") is not True
        or lock.get("positive_only") is not True
        or lock.get("classification_metrics_authorized") is not False
        or lock.get("qwen35_4b_authorized") is not False
        or lock.get("qwen35_9b_authorized") is not False
        or lock.get("counts", {}).get("rows") != 29
    ):
        raise ValueError("C6I pre-open gate is not open")
    geometry_lock = validate_c6i_geometry_release(
        lock_path=geometry_lock_path,
        rows_path=geometry_rows_path,
        certificates_path=geometry_certificates_path,
        manifest_path=manifest_path,
        mask_dir=geometry_mask_dir,
    )
    if lock.get("geometry_lock_canonical_sha256") != geometry_lock["canonical_sha256"]:
        raise ValueError("C6I geometry canonical identity changed")
    for path_string, expected in {**lock["source_sha256"], **lock["artifact_sha256"]}.items():
        path = Path(path_string)
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"C6I locked artifact changed: {path}")
    return lock
