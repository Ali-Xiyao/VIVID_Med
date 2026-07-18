"""Frozen patient-manifest, geometry, and metric contracts for C6F MS-CXR."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .pixel_interventions import transform_mask_to_letterbox, union_box_mask
from .provenance import canonical_json_sha256, file_sha256
from .rescue_protocol import (
    COORDINATE_ZONE_CONTROL_VERSION,
    deterministic_coordinate_zone_connected_control_mask,
)


FORMAT_VERSION = "bives_c6f_ms_cxr_manifest_v1"
LOCK_FORMAT_VERSION = "bives_c6f_ms_cxr_dataset_lock_v1"
GEOMETRY_FORMAT_VERSION = "bives_c6f_ms_cxr_geometry_v1"
IMAGE_SIZE = 448
FINDINGS = ("consolidation", "pleural_effusion")
OPERATORS = ("local_mean", "masked_gaussian_blur")
EXPECTED_ROWS = {"consolidation": 15, "pleural_effusion": 14}
EXPECTED_BOXES = {"consolidation": 25, "pleural_effusion": 20}
STATEMENTS = {
    "consolidation": "Focal air-space consolidation is present.",
    "pleural_effusion": "Pleural effusion is present.",
}
PUBLISHER_TO_CANONICAL = {
    "Consolidation": "consolidation",
    "Pleural Effusion": "pleural_effusion",
}


def _identifier_sha256(kind: str, value: str) -> str:
    return hashlib.sha256(f"mimic-cxr:{kind}:{value}".encode("utf-8")).hexdigest()


def _sample_id(patient: str, study: str, image: str, finding: str) -> str:
    payload = f"ms-cxr:1.1.0:test:{patient}:{study}:{image}:{finding}"
    return "ms_cxr_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_target_metadata(path: Path, dicom_ids: set[str]) -> dict[str, dict[str, str]]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    result: dict[str, dict[str, str]] = {}
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"dicom_id", "subject_id", "study_id"}.issubset(reader.fieldnames or []):
            raise ValueError("MIMIC metadata schema is incomplete")
        for row in reader:
            dicom = str(row["dicom_id"]).lower()
            if dicom not in dicom_ids:
                continue
            if dicom in result:
                raise ValueError(f"duplicate MIMIC metadata DICOM ID: {dicom}")
            patient = "p" + str(row["subject_id"]).removeprefix("p")
            study = "s" + str(row["study_id"]).removeprefix("s")
            result[dicom] = {"patient": patient, "study": study}
    missing = dicom_ids - set(result)
    if missing:
        raise ValueError(f"MIMIC metadata is missing {len(missing)} MS-CXR images")
    return result


def _content_geometry(width: int, height: int) -> tuple[tuple[int, int, int, int], np.ndarray]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    left = (IMAGE_SIZE - resized_width) // 2
    top = (IMAGE_SIZE - resized_height) // 2
    box = (left, top, left + resized_width, top + resized_height)
    content = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    content[top : top + resized_height, left : left + resized_width] = True
    return box, content


def build_ms_cxr_manifest(
    *,
    annotations_path: Path,
    mimic_metadata_path: Path,
    mimic_images_root: Path,
) -> list[dict[str, Any]]:
    """Build the ignored 29-row official-test positive manifest."""

    release = json.loads(annotations_path.read_text(encoding="utf-8"))
    categories = {int(row["id"]): str(row["name"]) for row in release["categories"]}
    images = {int(row["id"]): row for row in release["images"]}
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for annotation in release["annotations"]:
        publisher = categories[int(annotation["category_id"])]
        if str(annotation.get("split", "")).lower() != "test":
            continue
        if publisher not in PUBLISHER_TO_CANONICAL:
            continue
        label_text = str(annotation.get("label_text", "")).strip()
        if not label_text:
            raise ValueError("target MS-CXR annotation is missing label_text")
        grouped[(publisher, int(annotation["image_id"]), label_text)].append(annotation)
    dicom_ids = {Path(str(images[key[1]]["file_name"])).stem.lower() for key in grouped}
    metadata = _read_target_metadata(mimic_metadata_path, dicom_ids)
    rows: list[dict[str, Any]] = []
    image_root = mimic_images_root.resolve()
    for (publisher, image_id, label_text), annotations in sorted(grouped.items()):
        finding = PUBLISHER_TO_CANONICAL[publisher]
        image = images[image_id]
        dicom = Path(str(image["file_name"])).stem.lower()
        identity = metadata[dicom]
        expected_relative = Path(
            "files",
            identity["patient"][:3],
            identity["patient"],
            identity["study"],
            f"{dicom}.jpg",
        )
        if Path(str(image["path"])) != expected_relative:
            raise ValueError("MS-CXR release path disagrees with MIMIC metadata")
        image_path = (image_root / expected_relative).resolve()
        if image_root not in image_path.parents or not image_path.is_file():
            raise FileNotFoundError(image_path)
        boxes = []
        for annotation in sorted(annotations, key=lambda row: int(row["id"])):
            x, y, width, height = (float(value) for value in annotation["bbox"])
            box = {
                "x_min": x,
                "y_min": y,
                "x_max": x + width,
                "y_max": y + height,
            }
            if (
                width <= 0
                or height <= 0
                or x < 0
                or y < 0
                or box["x_max"] > float(image["width"])
                or box["y_max"] > float(image["height"])
            ):
                raise ValueError("MS-CXR box is outside native image geometry")
            boxes.append(box)
        patient_hash = _identifier_sha256("patient", identity["patient"])
        study_hash = _identifier_sha256("study", identity["study"])
        image_hash = _identifier_sha256("image", dicom)
        rows.append(
            {
                "format_version": FORMAT_VERSION,
                "sample_id": _sample_id(
                    identity["patient"], identity["study"], dicom, finding
                ),
                "unit_id": patient_hash,
                "patient_sha256": patient_hash,
                "study_sha256": study_hash,
                "image_id_sha256": image_hash,
                "image_path": str(image_path),
                "official_image_sha256": file_sha256(image_path),
                "canonical_statement_id": finding,
                "statement_text": STATEMENTS[finding],
                "publisher_label_text_sha256": hashlib.sha256(
                    label_text.encode("utf-8")
                ).hexdigest(),
                "native_columns": int(image["width"]),
                "native_rows": int(image["height"]),
                "bounding_boxes": boxes,
                "box_count": len(boxes),
                "source_split": "publisher_test",
                "positive_only": True,
            }
        )
    rows.sort(key=lambda row: str(row["sample_id"]))
    validate_ms_cxr_manifest(rows)
    return rows


def validate_ms_cxr_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != sum(EXPECTED_ROWS.values()):
        raise ValueError("MS-CXR manifest must contain exactly 29 rows")
    sample_ids = [str(row["sample_id"]) for row in rows]
    patients = [str(row["patient_sha256"]) for row in rows]
    studies = [str(row["study_sha256"]) for row in rows]
    images = [str(row["image_id_sha256"]) for row in rows]
    for name, values in (
        ("sample", sample_ids),
        ("patient", patients),
        ("study", studies),
        ("image", images),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"MS-CXR manifest has duplicate {name} identities")
    counts = {finding: 0 for finding in FINDINGS}
    boxes = {finding: 0 for finding in FINDINGS}
    for row in rows:
        finding = str(row["canonical_statement_id"])
        if finding not in FINDINGS or row.get("statement_text") != STATEMENTS[finding]:
            raise ValueError("MS-CXR manifest statement contract mismatch")
        if row.get("source_split") != "publisher_test" or row.get("positive_only") is not True:
            raise ValueError("MS-CXR manifest contains a non-test/non-positive row")
        if not Path(str(row["image_path"])).is_file():
            raise FileNotFoundError(row["image_path"])
        if file_sha256(row["image_path"]) != row["official_image_sha256"]:
            raise ValueError("MS-CXR image hash changed")
        counts[finding] += 1
        boxes[finding] += len(row["bounding_boxes"])
    if counts != EXPECTED_ROWS or boxes != EXPECTED_BOXES:
        raise ValueError(f"MS-CXR finding/box counts changed: {counts}, {boxes}")
    return {
        "rows": len(rows),
        "patients": len(set(patients)),
        "studies": len(set(studies)),
        "images": len(set(images)),
        "per_finding": counts,
        "per_finding_boxes": boxes,
    }


def build_ms_cxr_geometry(
    rows: list[dict[str, Any]], *, mask_dir: Path, workers: int = 1
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create the score-free exact-area connected-control lock for all rows."""

    validate_ms_cxr_manifest(rows)
    mask_dir.mkdir(parents=True, exist_ok=True)
    if workers <= 0:
        raise ValueError("geometry workers must be positive")
    payloads = [(row, str(mask_dir)) for row in rows]
    if workers == 1:
        geometry_rows = [_build_ms_cxr_geometry_row(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            geometry_rows = list(executor.map(_build_ms_cxr_geometry_row, payloads))
    by_finding = {
        finding: sorted(
            (row for row in geometry_rows if row["canonical_statement_id"] == finding),
            key=lambda row: (float(row["target_area_fraction"]), str(row["sample_id"])),
        )
        for finding in FINDINGS
    }
    quartiles: dict[str, int] = {}
    for finding, subset in by_finding.items():
        boundaries = np.quantile(
            [float(row["target_area_fraction"]) for row in subset], [0.25, 0.5, 0.75]
        )
        for row in subset:
            quartiles[str(row["sample_id"])] = int(
                np.searchsorted(boundaries, float(row["target_area_fraction"]), side="right") + 1
            )
    for row in geometry_rows:
        row["box_area_quartile"] = quartiles[str(row["sample_id"])]
    geometry_rows.sort(key=lambda row: str(row["sample_id"]))
    summary = {
        "format_version": GEOMETRY_FORMAT_VERSION,
        "status": "pass" if all(row["feasible"] for row in geometry_rows) else "fail_geometry",
        "rows": len(geometry_rows),
        "eligible": sum(bool(row["feasible"]) for row in geometry_rows),
        "infeasible": sum(not bool(row["feasible"]) for row in geometry_rows),
        "control_version": COORDINATE_ZONE_CONTROL_VERSION,
        "image_size": IMAGE_SIZE,
        "per_finding": {
            finding: sum(row["canonical_statement_id"] == finding for row in geometry_rows)
            for finding in FINDINGS
        },
    }
    if summary["rows"] != 29 or summary["per_finding"] != EXPECTED_ROWS:
        raise ValueError("MS-CXR geometry denominator changed")
    return geometry_rows, summary


def _build_ms_cxr_geometry_row(
    payload: tuple[dict[str, Any], str]
) -> dict[str, Any]:
    row, mask_dir_string = payload
    width = int(row["native_columns"])
    height = int(row["native_rows"])
    content_box, content = _content_geometry(width, height)
    target = transform_mask_to_letterbox(
        union_box_mask(width, height, row["bounding_boxes"]),
        content_box,
        IMAGE_SIZE,
    ) & content
    base = {
        "format_version": GEOMETRY_FORMAT_VERSION,
        "sample_id": row["sample_id"],
        "canonical_statement_id": row["canonical_statement_id"],
        "target_area_pixels": int(target.sum()),
        "target_area_fraction": float(target.sum() / content.sum()),
    }
    try:
        control, audit = deterministic_coordinate_zone_connected_control_mask(
            target,
            content,
            seed_key=f"{row['sample_id']}:{COORDINATE_ZONE_CONTROL_VERSION}",
        )
    except ValueError as error:
        return {
            **base,
            "control_area_pixels": 0,
            "mask_file": None,
            "mask_sha256": None,
            "control_audit": None,
            "feasible": False,
            "failure": str(error),
        }
    if int(target.sum()) != int(control.sum()) or bool((target & control).any()):
        raise AssertionError("MS-CXR target/control invariant failed")
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


def summarize_operator(
    rows: list[dict[str, Any]],
    operator: str,
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 17,
) -> dict[str, Any]:
    if operator not in OPERATORS or bootstrap_replicates <= 0:
        raise ValueError("invalid C6F operator/bootstrap configuration")
    rng = np.random.default_rng(bootstrap_seed)
    per_finding: dict[str, Any] = {}
    for finding in FINDINGS:
        subset = [row for row in rows if row["canonical_statement_id"] == finding]
        if len(subset) != EXPECTED_ROWS[finding]:
            raise ValueError(f"C6F result denominator changed for {finding}")
        values = np.asarray([float(row[operator]["tcig"]) for row in subset])
        units = sorted({str(row["unit_id"]) for row in subset})
        by_unit = {unit: [row for row in subset if str(row["unit_id"]) == unit] for unit in units}
        replicates = []
        for _ in range(bootstrap_replicates):
            sampled = rng.choice(units, size=len(units), replace=True)
            sampled_rows = [row for unit in sampled for row in by_unit[str(unit)]]
            replicates.append(float(np.mean([float(row[operator]["tcig"]) for row in sampled_rows])))
        highest = [row for row in subset if int(row["box_area_quartile"]) == 4]
        per_finding[finding] = {
            "records": len(subset),
            "patients": len(units),
            "mean_tcig": float(values.mean()),
            "bootstrap_95ci": {
                "lower": float(np.percentile(replicates, 2.5)),
                "upper": float(np.percentile(replicates, 97.5)),
            },
            "positive_patient_fraction": float((values > 0).mean()),
            "highest_area_quartile_records": len(highest),
            "highest_area_quartile_mean_tcig": float(
                np.mean([float(row[operator]["tcig"]) for row in highest])
            ),
            "mean_topk_localization_gain": float(
                np.mean([float(row["topk_localization_gain"]) for row in subset])
            ),
        }
    return {
        "operator": operator,
        "per_finding": per_finding,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "confidence_interval_unit": "patient",
    }


def evaluate_survival_gate(operator_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for operator in OPERATORS:
        for finding in FINDINGS:
            point = operator_results[operator]["per_finding"][finding]
            checks[f"{operator}|{finding}"] = {
                "mean_tcig_positive": float(point["mean_tcig"]) > 0.0,
                "positive_patient_fraction_at_least_0_60": (
                    float(point["positive_patient_fraction"]) >= 0.60
                ),
                "highest_area_quartile_nonnegative": (
                    float(point["highest_area_quartile_mean_tcig"]) >= 0.0
                ),
            }
    ci_checks = {
        finding: any(
            float(operator_results[operator]["per_finding"][finding]["bootstrap_95ci"]["lower"]) > 0.0
            for operator in OPERATORS
        )
        for finding in FINDINGS
    }
    passed = all(all(item.values()) for item in checks.values()) and all(ci_checks.values())
    return {
        "status": "pass" if passed else "fail_final_stop",
        "pass": passed,
        "per_operator_finding": checks,
        "at_least_one_operator_ci_lower_positive_per_finding": ci_checks,
    }


def build_dataset_lock(
    *,
    manifest_path: Path,
    geometry_rows_path: Path,
    geometry_summary: dict[str, Any],
    strict_intake_path: Path,
    authority_path: Path,
    config_path: Path,
    source_paths: list[Path],
    release_paths: list[Path],
) -> dict[str, Any]:
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    counts = validate_ms_cxr_manifest(rows)
    intake = json.loads(strict_intake_path.read_text(encoding="utf-8"))
    if intake.get("canonical_artifact_sha256") != (
        "0027358c2998773e73dbd19da02a37dac27c060150bf42e59469d218fb24b4ed"
    ):
        raise ValueError("strict C6E intake identity changed")
    if intake.get("license_gate_passed") is not True or intake.get("model_evaluation_authorized") is not False:
        raise ValueError("strict C6E intake contract changed")
    payload: dict[str, Any] = {
        "format_version": LOCK_FORMAT_VERSION,
        "status": "pass" if geometry_summary.get("status") == "pass" else "fail_geometry",
        "model_evaluation_authorized": True,
        "evaluation_gate_open": geometry_summary.get("status") == "pass",
        "source_split": "publisher_test_only",
        "positive_only": True,
        "classification_metrics_authorized": False,
        "strict_intake_file_sha256": file_sha256(strict_intake_path),
        "strict_intake_canonical_sha256": intake["canonical_artifact_sha256"],
        "manifest_sha256": file_sha256(manifest_path),
        "geometry_rows_sha256": file_sha256(geometry_rows_path),
        "geometry_summary": geometry_summary,
        "counts": counts,
        "identity_set_sha256": {
            kind: canonical_json_sha256(sorted({str(row[f"{kind}_sha256"]) for row in rows}))
            for kind in ("patient", "study")
        },
        "image_payload_sha256": canonical_json_sha256(
            sorted((str(row["image_id_sha256"]), str(row["official_image_sha256"])) for row in rows)
        ),
        "authority_sha256": file_sha256(authority_path),
        "config_sha256": file_sha256(config_path),
        "source_sha256": {str(path): file_sha256(path) for path in source_paths},
        "release_sha256": {str(path): file_sha256(path) for path in release_paths},
    }
    payload["canonical_artifact_sha256"] = canonical_json_sha256(payload)
    return payload


def validate_dataset_lock(
    lock_path: Path,
    *,
    manifest_path: Path,
    geometry_rows_path: Path,
    strict_intake_path: Path,
    authority_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    declared = lock.get("canonical_artifact_sha256")
    body = {key: value for key, value in lock.items() if key != "canonical_artifact_sha256"}
    if declared != canonical_json_sha256(body):
        raise ValueError("C6F dataset lock canonical hash mismatch")
    required = {
        "manifest_sha256": file_sha256(manifest_path),
        "geometry_rows_sha256": file_sha256(geometry_rows_path),
        "strict_intake_file_sha256": file_sha256(strict_intake_path),
        "authority_sha256": file_sha256(authority_path),
        "config_sha256": file_sha256(config_path),
    }
    mismatches = [key for key, value in required.items() if lock.get(key) != value]
    if mismatches:
        raise ValueError("C6F dataset lock mismatch: " + ", ".join(mismatches))
    if (
        lock.get("status") != "pass"
        or lock.get("model_evaluation_authorized") is not True
        or lock.get("evaluation_gate_open") is not True
        or lock.get("classification_metrics_authorized") is not False
        or lock.get("counts", {}).get("rows") != 29
        or lock.get("geometry_summary", {}).get("eligible") != 29
    ):
        raise ValueError("C6F dataset lock gate is not open")
    return lock
