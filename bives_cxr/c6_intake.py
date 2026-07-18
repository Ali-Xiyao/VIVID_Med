"""Fail-closed metadata intake helpers for the BiVES C6 final-data gate."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .provenance import canonical_json_sha256, file_sha256


FORMAT_VERSION = "bives_c6_chexlocalize_intake_v1"
REGISTRY_FORMAT_VERSION = "bives_c6_prior_access_registry_v1"
TARGET_FINDINGS = ("Consolidation", "Pleural Effusion")
CHEXPERT_KEY = re.compile(
    r"patient(?P<patient>\d+)[/\\]study(?P<study>\d+)[/\\]"
    r"(?P<view>view\d+_(?:frontal|lateral))\.(?:jpg|jpeg|png)$",
    flags=re.IGNORECASE,
)
CHEXLOCALIZE_KEY = re.compile(
    r"^patient(?P<patient>\d+)_study(?P<study>\d+)_"
    r"(?P<view>view\d+_(?:frontal|lateral))$",
    flags=re.IGNORECASE,
)


def _identifier_sha256(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}:{value}".encode("utf-8")).hexdigest()


def _identity(patient: str, study: str, view: str) -> dict[str, str]:
    patient_id = f"patient{patient}"
    study_id = f"{patient_id}_study{study}"
    image_id = f"{study_id}_{view.lower()}"
    return {"patient": patient_id, "study": study_id, "image": image_id}


def parse_chexpert_path(value: str) -> dict[str, str]:
    match = CHEXPERT_KEY.search(value.replace("\\", "/"))
    if match is None:
        raise ValueError(f"invalid CheXpert image path: {value!r}")
    return _identity(match["patient"], match["study"], match["view"])


def parse_chexlocalize_key(value: str) -> dict[str, str]:
    match = CHEXLOCALIZE_KEY.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid CheXlocalize annotation key: {value!r}")
    return _identity(match["patient"], match["study"], match["view"])


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _path_column(rows: list[dict[str, str]]) -> str:
    if not rows:
        raise ValueError("CheXpert CSV is empty")
    by_lower = {str(key).lower(): str(key) for key in rows[0]}
    for candidate in ("path", "image_path"):
        if candidate in by_lower:
            return by_lower[candidate]
    raise ValueError("CheXpert CSV must contain a Path or image_path column")


def _registry_payload(
    identities: list[dict[str, str]],
    *,
    source_split: str,
    source_sha256: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": REGISTRY_FORMAT_VERSION,
        "namespace": "chexpert",
        "source_split": source_split,
        "source_sha256": source_sha256,
        "counts": {
            key: len({row[key] for row in identities})
            for key in ("patient", "study", "image")
        },
        "identifier_sha256": {
            key: sorted(
                {_identifier_sha256("chexpert", row[key]) for row in identities}
            )
            for key in ("patient", "study", "image")
        },
        "contains_raw_identifiers": False,
    }
    payload["canonical_artifact_sha256"] = canonical_json_sha256(payload)
    return payload


def build_chexpert_prior_access_registry(
    valid_csv: Path,
    *,
    source_split: str = "validation",
) -> dict[str, Any]:
    rows = _read_csv(valid_csv)
    path_column = _path_column(rows)
    identities = [parse_chexpert_path(str(row[path_column])) for row in rows]
    return _registry_payload(
        identities,
        source_split=source_split,
        source_sha256=file_sha256(valid_csv),
    )


def validate_prior_access_registry(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != REGISTRY_FORMAT_VERSION:
        raise ValueError("unsupported prior-access registry format")
    if payload.get("namespace") != "chexpert":
        raise ValueError("prior-access registry namespace must be chexpert")
    if payload.get("contains_raw_identifiers") is not False:
        raise ValueError("prior-access registry must not contain raw identifiers")
    expected = payload.get("canonical_artifact_sha256")
    body = {key: value for key, value in payload.items() if key != "canonical_artifact_sha256"}
    if expected != canonical_json_sha256(body):
        raise ValueError("prior-access registry canonical hash mismatch")
    hashes = payload.get("identifier_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("prior-access registry is missing identifier hashes")
    for key in ("patient", "study", "image"):
        values = hashes.get(key)
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in values
        ):
            raise ValueError(f"invalid prior-access {key} hashes")
    return payload


def validate_license_record(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = (
        "dataset_name",
        "release_version",
        "source_url",
        "terms_url",
        "retrieved_at",
        "package_sha256",
    )
    missing = [key for key in required if not payload.get(key)]
    if missing:
        raise ValueError("license record missing fields: " + ", ".join(missing))
    if payload["dataset_name"] != "CheXlocalize":
        raise ValueError("license record dataset_name must be CheXlocalize")
    if payload.get("terms_accepted_by_user") is not True:
        raise ValueError("CheXlocalize terms must be accepted by the user")
    if payload.get("access_secret_not_persisted") is not True:
        raise ValueError("license record must attest that no SAS/access secret was persisted")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", str(payload["package_sha256"])):
        raise ValueError("license record package_sha256 must be a SHA-256 digest")
    return payload


def _validate_contours(
    image_key: str,
    finding: str,
    contours: Any,
    image_size: Any,
) -> int:
    if not isinstance(image_size, list) or len(image_size) != 2:
        raise ValueError(f"invalid img_size for {image_key}")
    height, width = image_size
    if not all(isinstance(value, (int, float)) and value > 0 for value in (height, width)):
        raise ValueError(f"invalid img_size for {image_key}")
    if not isinstance(contours, list) or not contours:
        raise ValueError(f"empty expert contours for {image_key}/{finding}")
    for contour in contours:
        if not isinstance(contour, list) or len(contour) < 3:
            raise ValueError(f"invalid expert contour for {image_key}/{finding}")
        for point in contour:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError(f"invalid expert point for {image_key}/{finding}")
            x, y = point
            if not all(isinstance(value, (int, float)) for value in (x, y)):
                raise ValueError(f"non-numeric expert point for {image_key}/{finding}")
            if not (0 <= x <= width and 0 <= y <= height):
                raise ValueError(f"out-of-bounds expert point for {image_key}/{finding}")
    return len(contours)


def audit_chexlocalize_test_release(
    dataset_root: Path,
    *,
    license_record: Path,
    prior_registry: Path,
    expected_test_images: int = 668,
    expected_test_patients: int = 500,
) -> dict[str, Any]:
    root = dataset_root.resolve()
    labels_path = root / "CheXpert" / "test_labels.csv"
    annotations_path = root / "CheXlocalize" / "gt_annotations_test.json"
    segmentations_path = root / "CheXlocalize" / "gt_segmentations_test.json"
    for path in (labels_path, annotations_path, segmentations_path, license_record, prior_registry):
        if not path.is_file():
            raise FileNotFoundError(path)

    license_payload = validate_license_record(license_record)
    prior = validate_prior_access_registry(prior_registry)
    label_rows = _read_csv(labels_path)
    path_column = _path_column(label_rows)
    identities: dict[str, dict[str, str]] = {}
    image_hash_pairs: list[tuple[str, str]] = []
    test_image_root = (root / "CheXpert" / "test").resolve()
    for row in label_rows:
        source_path = str(row[path_column]).replace("\\", "/")
        anchored_source = f"/{source_path.lstrip('/')}"
        marker_index = anchored_source.lower().find("/test/")
        if marker_index < 0:
            raise ValueError(f"non-test CheXlocalize label path is forbidden: {source_path!r}")
        identity = parse_chexpert_path(source_path)
        if identity["image"] in identities:
            raise ValueError(f"duplicate CheXlocalize test image: {identity['image']}")
        identities[identity["image"]] = identity
        suffix = anchored_source[marker_index + len("/test/") :]
        image_path = (test_image_root / Path(suffix)).resolve()
        if test_image_root not in image_path.parents:
            raise ValueError(f"CheXlocalize image path escapes test root: {source_path!r}")
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        image_hash_pairs.append(
            (_identifier_sha256("chexpert", identity["image"]), file_sha256(image_path))
        )

    patients = {row["patient"] for row in identities.values()}
    if len(identities) != expected_test_images:
        raise ValueError(
            f"CheXlocalize test has {len(identities)} images; expected {expected_test_images}"
        )
    if len(patients) != expected_test_patients:
        raise ValueError(
            f"CheXlocalize test has {len(patients)} patients; expected {expected_test_patients}"
        )

    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    if not isinstance(annotations, dict) or not annotations:
        raise ValueError("CheXlocalize test annotations must be a non-empty object")
    finding_images: dict[str, set[str]] = defaultdict(set)
    finding_patients: dict[str, set[str]] = defaultdict(set)
    finding_contours: dict[str, int] = defaultdict(int)
    for image_key, payload in annotations.items():
        identity = parse_chexlocalize_key(str(image_key))
        if identity["image"] not in identities:
            raise ValueError(f"annotation key is absent from test labels: {image_key}")
        if not isinstance(payload, dict):
            raise ValueError(f"annotation payload must be an object: {image_key}")
        for finding in TARGET_FINDINGS:
            if finding not in payload:
                continue
            finding_contours[finding] += _validate_contours(
                str(image_key), finding, payload[finding], payload.get("img_size")
            )
            finding_images[finding].add(identity["image"])
            finding_patients[finding].add(identity["patient"])
    missing_targets = [finding for finding in TARGET_FINDINGS if not finding_images[finding]]
    if missing_targets:
        raise ValueError("CheXlocalize test is missing target regions: " + ", ".join(missing_targets))

    prior_hashes = prior["identifier_sha256"]
    overlap = {
        key: len(
            {_identifier_sha256("chexpert", row[key]) for row in identities.values()}
            & set(prior_hashes[key])
        )
        for key in ("patient", "study", "image")
    }
    if any(overlap.values()):
        raise ValueError(f"CheXlocalize test overlaps prior access registry: {overlap}")

    identity_hashes = {
        key: canonical_json_sha256(
            sorted({_identifier_sha256("chexpert", row[key]) for row in identities.values()})
        )
        for key in ("patient", "study", "image")
    }
    payload = {
        "format_version": FORMAT_VERSION,
        "status": "metadata_intake_ready_no_model_authority",
        "formal_result": False,
        "model_evaluation_authorized": False,
        "source_split": "publisher_test_only",
        "dataset_root": str(root),
        "release": {
            "license_record_sha256": file_sha256(license_record),
            "package_sha256": str(license_payload["package_sha256"]).lower(),
            "test_labels_sha256": file_sha256(labels_path),
            "gt_annotations_test_sha256": file_sha256(annotations_path),
            "gt_segmentations_test_sha256": file_sha256(segmentations_path),
            "test_image_payload_sha256": canonical_json_sha256(sorted(image_hash_pairs)),
        },
        "counts": {
            "test_images": len(identities),
            "test_patients": len(patients),
            "annotated_images": len(annotations),
        },
        "targets": {
            finding: {
                "images": len(finding_images[finding]),
                "patients": len(finding_patients[finding]),
                "contours": finding_contours[finding],
            }
            for finding in TARGET_FINDINGS
        },
        "identity_set_sha256": identity_hashes,
        "prior_registry_sha256": file_sha256(prior_registry),
        "prior_overlap_counts": overlap,
        "raw_identifiers_emitted": False,
        "claim_boundary": "metadata-only intake; C5 remains final stop",
        "intake_module_sha256": file_sha256(Path(__file__)),
    }
    payload["canonical_artifact_sha256"] = canonical_json_sha256(payload)
    return payload
