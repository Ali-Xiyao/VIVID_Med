"""Fail-closed MS-CXR official-test intake for the BiVES C6 data gate."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .provenance import canonical_json_sha256, file_sha256


FORMAT_VERSION = "bives_c6_ms_cxr_intake_v1"
REGISTRY_FORMAT_VERSION = "bives_c6_mimic_prior_registry_v1"
TARGET_FINDINGS = ("Consolidation", "Pleural Effusion")
EXPECTED_TEST_PAIRS = {"Consolidation": 15, "Pleural Effusion": 14}
EXPECTED_TEST_SUBJECTS = {"Consolidation": 15, "Pleural Effusion": 14}
EXPECTED_PRIOR_PATIENTS = 1414
EXPECTED_PRIOR_STUDIES = 5008
EXPECTED_PRIOR_PATIENT_SET_SHA256 = (
    "106e13b9500ff5ad9c7e67a168861c04a0f2486a9786ebc8850bf5000e207950"
)
EXPECTED_PRIOR_STUDY_SET_SHA256 = (
    "76e8ae65bc0d740908d064fff5748ddec390eb121c456a8f75f42020c472cd86"
)
DICOM_ID = re.compile(r"^[0-9a-f]{8}(?:-[0-9a-f]{8}){4}$", re.IGNORECASE)
PATIENT_ID = re.compile(r"^p?(?P<value>\d+)$", re.IGNORECASE)
STUDY_ID = re.compile(r"^s?(?P<value>\d+)(?P<suffix>-proxy-i)?$", re.IGNORECASE)


def _identifier_sha256(kind: str, value: str) -> str:
    return hashlib.sha256(f"mimic-cxr:{kind}:{value}".encode("utf-8")).hexdigest()


def _set_sha256(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(set(values))) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_patient(value: object) -> str:
    match = PATIENT_ID.fullmatch(str(value).strip())
    if match is None:
        raise ValueError(f"invalid MIMIC patient identifier: {value!r}")
    return f"p{match['value']}"


def _normalize_study(value: object) -> str:
    match = STUDY_ID.fullmatch(str(value).strip())
    if match is None:
        raise ValueError(f"invalid MIMIC study identifier: {value!r}")
    return f"s{match['value']}{match['suffix'] or ''}"


def _read_jsonl_identities(paths: Iterable[Path]) -> tuple[set[str], set[str]]:
    patients: set[str] = set()
    studies: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                try:
                    patients.add(_normalize_patient(row["patient_id"]))
                    studies.add(_normalize_study(row["study_id"]))
                except KeyError as error:
                    raise ValueError(
                        f"{path}:{line_number} is missing {error.args[0]}"
                    ) from error
    return patients, studies


def build_mimic_prior_access_registry(
    manifest_paths: Iterable[Path],
    *,
    enforce_frozen_identity: bool = True,
) -> dict[str, Any]:
    paths = [Path(path).resolve() for path in manifest_paths]
    patients, studies = _read_jsonl_identities(paths)
    patient_set_sha256 = _set_sha256(patients)
    study_set_sha256 = _set_sha256(studies)
    if enforce_frozen_identity:
        observed = (
            len(patients),
            len(studies),
            patient_set_sha256,
            study_set_sha256,
        )
        expected = (
            EXPECTED_PRIOR_PATIENTS,
            EXPECTED_PRIOR_STUDIES,
            EXPECTED_PRIOR_PATIENT_SET_SHA256,
            EXPECTED_PRIOR_STUDY_SET_SHA256,
        )
        if observed != expected:
            raise ValueError(f"frozen MIMIC prior-use identity mismatch: {observed}")
    payload: dict[str, Any] = {
        "format_version": REGISTRY_FORMAT_VERSION,
        "namespace": "mimic-cxr",
        "source_manifest_sha256": [file_sha256(path) for path in paths],
        "counts": {"patient": len(patients), "study": len(studies)},
        "identifier_sha256": {
            "patient": sorted(_identifier_sha256("patient", value) for value in patients),
            "study": sorted(_identifier_sha256("study", value) for value in studies),
        },
        "identifier_set_sha256": {
            "patient": patient_set_sha256,
            "study": study_set_sha256,
        },
        "contains_raw_identifiers": False,
    }
    payload["canonical_artifact_sha256"] = canonical_json_sha256(payload)
    return payload


def validate_mimic_prior_access_registry(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != REGISTRY_FORMAT_VERSION:
        raise ValueError("unsupported MIMIC prior-access registry format")
    if payload.get("namespace") != "mimic-cxr":
        raise ValueError("MIMIC prior-access registry namespace mismatch")
    if payload.get("contains_raw_identifiers") is not False:
        raise ValueError("MIMIC prior-access registry must not contain raw identifiers")
    expected = payload.get("canonical_artifact_sha256")
    body = {key: value for key, value in payload.items() if key != "canonical_artifact_sha256"}
    if expected != canonical_json_sha256(body):
        raise ValueError("MIMIC prior-access registry canonical hash mismatch")
    hashes = payload.get("identifier_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("MIMIC prior-access registry is missing identifier hashes")
    for kind in ("patient", "study"):
        values = hashes.get(kind)
        if not isinstance(values, list) or any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in values
        ):
            raise ValueError(f"invalid MIMIC prior-access {kind} hashes")
    return payload


def _assert_frozen_prior_registry(payload: dict[str, Any]) -> None:
    counts = payload.get("counts", {})
    set_hashes = payload.get("identifier_set_sha256", {})
    identifier_hashes = payload.get("identifier_sha256", {})
    observed = (
        counts.get("patient"),
        counts.get("study"),
        set_hashes.get("patient"),
        set_hashes.get("study"),
        len(identifier_hashes.get("patient", [])),
        len(identifier_hashes.get("study", [])),
    )
    expected = (
        EXPECTED_PRIOR_PATIENTS,
        EXPECTED_PRIOR_STUDIES,
        EXPECTED_PRIOR_PATIENT_SET_SHA256,
        EXPECTED_PRIOR_STUDY_SET_SHA256,
        EXPECTED_PRIOR_PATIENTS,
        EXPECTED_PRIOR_STUDIES,
    )
    if observed != expected:
        raise ValueError(f"frozen MIMIC prior-access registry mismatch: {observed}")


def validate_ms_cxr_license_record(path: Path) -> dict[str, Any]:
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
        raise ValueError("MS-CXR license record missing fields: " + ", ".join(missing))
    if payload["dataset_name"] != "MS-CXR":
        raise ValueError("license record dataset_name must be MS-CXR")
    if payload["release_version"] != "1.1.0":
        raise ValueError("MS-CXR release_version must be 1.1.0")
    for key in (
        "credentialed_access_confirmed",
        "citi_training_confirmed",
        "dua_signed_by_user",
        "access_secret_not_persisted",
    ):
        if payload.get(key) is not True:
            raise ValueError(f"MS-CXR license record requires {key}=true")
    if re.fullmatch(r"[0-9a-fA-F]{64}", str(payload["package_sha256"])) is None:
        raise ValueError("MS-CXR package_sha256 must be a SHA-256 digest")
    return payload


def _read_mimic_metadata(
    path: Path, required_dicom_ids: set[str]
) -> dict[str, dict[str, str]]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    matched: dict[str, dict[str, str]] = {}
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dicom_id", "subject_id", "study_id"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("MIMIC metadata must contain dicom_id, subject_id, study_id")
        for row in reader:
            dicom_id = str(row["dicom_id"]).lower()
            if dicom_id not in required_dicom_ids:
                continue
            if dicom_id in matched:
                raise ValueError(f"duplicate DICOM ID in MIMIC metadata: {dicom_id}")
            matched[dicom_id] = {
                "patient": _normalize_patient(row["subject_id"]),
                "study": _normalize_study(row["study_id"]),
            }
    missing = sorted(required_dicom_ids - set(matched))
    if missing:
        raise ValueError(f"MS-CXR images are absent from MIMIC metadata: {len(missing)}")
    return matched


def _validate_ltwh_box(annotation: dict[str, Any], image: dict[str, Any]) -> None:
    bbox = annotation.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("MS-CXR annotation bbox must be LTWH with four values")
    if not all(isinstance(value, (int, float)) for value in bbox):
        raise ValueError("MS-CXR annotation bbox must be numeric")
    x, y, width, height = (float(value) for value in bbox)
    image_width = float(image.get("width", 0))
    image_height = float(image.get("height", 0))
    if width <= 0 or height <= 0 or image_width <= 0 or image_height <= 0:
        raise ValueError("MS-CXR annotation/image dimensions must be positive")
    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        raise ValueError("MS-CXR annotation bbox is out of image bounds")


def _inspect_ms_cxr_test_release(
    dataset_root: Path,
    *,
    mimic_metadata: Path,
    mimic_images_root: Path,
    license_record: Path | None,
    package_archive: Path | None,
    prior_registry: Path,
    expected_test_pairs: dict[str, int] | None = None,
    expected_test_subjects: dict[str, int] | None = None,
    enforce_frozen_prior_identity: bool = True,
) -> dict[str, Any]:
    expected_pairs = expected_test_pairs or EXPECTED_TEST_PAIRS
    expected_subjects = expected_test_subjects or EXPECTED_TEST_SUBJECTS
    root = dataset_root.resolve()
    annotations_path = root / "MS_CXR_Local_Alignment_v1.1.0.json"
    required_files = [annotations_path, mimic_metadata, prior_registry]
    if license_record is not None:
        required_files.append(license_record)
    if package_archive is not None:
        required_files.append(package_archive)
    if license_record is None and package_archive is None:
        raise ValueError(
            "MS-CXR inspection requires either a license record or package archive"
        )
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    image_root = mimic_images_root.resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)

    if license_record is not None:
        license_payload = validate_ms_cxr_license_record(license_record)
        package_sha256 = str(license_payload["package_sha256"]).lower()
        if package_archive is not None:
            actual_package_sha256 = file_sha256(package_archive)
            if actual_package_sha256 != package_sha256:
                raise ValueError(
                    "MS-CXR package SHA-256 disagrees with license record"
                )
        license_record_sha256: str | None = file_sha256(license_record)
        license_gate_passed = True
    else:
        assert package_archive is not None
        package_sha256 = file_sha256(package_archive)
        license_record_sha256 = None
        license_gate_passed = False
    prior = validate_mimic_prior_access_registry(prior_registry)
    if enforce_frozen_prior_identity:
        _assert_frozen_prior_registry(prior)
    release = json.loads(annotations_path.read_text(encoding="utf-8"))
    if not isinstance(release, dict):
        raise ValueError("MS-CXR release must be a COCO JSON object")
    if str(release.get("info", {}).get("version")) != "1.1.0":
        raise ValueError("MS-CXR COCO info.version must be 1.1.0")

    categories: dict[int, str] = {}
    for row in release.get("categories", []):
        category_id = int(row["id"])
        if category_id in categories:
            raise ValueError(f"duplicate MS-CXR category ID: {category_id}")
        categories[category_id] = str(row["name"])
    for finding in TARGET_FINDINGS:
        if finding not in categories.values():
            raise ValueError(f"MS-CXR release is missing category: {finding}")

    images: dict[int, dict[str, Any]] = {}
    dicom_by_image: dict[int, str] = {}
    for row in release.get("images", []):
        image_id = int(row["id"])
        if image_id in images:
            raise ValueError(f"duplicate MS-CXR image ID: {image_id}")
        file_name = Path(str(row.get("file_name", "")))
        if file_name.name != str(row.get("file_name", "")) or file_name.suffix.lower() != ".jpg":
            raise ValueError("MS-CXR image file_name must be a basename JPG")
        dicom_id = file_name.stem.lower()
        if DICOM_ID.fullmatch(dicom_id) is None:
            raise ValueError(f"invalid MS-CXR DICOM filename: {file_name}")
        images[image_id] = row
        dicom_by_image[image_id] = dicom_id

    target_annotations: list[tuple[str, int, str, dict[str, Any]]] = []
    annotation_ids: set[int] = set()
    for row in release.get("annotations", []):
        annotation_id = int(row["id"])
        if annotation_id in annotation_ids:
            raise ValueError(f"duplicate MS-CXR annotation ID: {annotation_id}")
        annotation_ids.add(annotation_id)
        split = str(row.get("split", "")).lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"invalid MS-CXR annotation split: {split!r}")
        image_id = int(row["image_id"])
        if image_id not in images:
            raise ValueError(f"MS-CXR annotation references unknown image: {image_id}")
        category_id = int(row["category_id"])
        if category_id not in categories:
            raise ValueError(f"MS-CXR annotation references unknown category: {category_id}")
        finding = categories[category_id]
        if finding in TARGET_FINDINGS and split == "test":
            label_text = str(row.get("label_text", "")).strip()
            if not label_text:
                raise ValueError("MS-CXR target test annotation is missing label_text")
            _validate_ltwh_box(row, images[image_id])
            target_annotations.append((finding, image_id, label_text, row))
    if not target_annotations:
        raise ValueError("MS-CXR release contains no target test annotations")

    pair_keys = {
        (finding, image_id, label_text)
        for finding, image_id, label_text, _ in target_annotations
    }
    required_dicom_ids = {
        dicom_by_image[image_id] for _, image_id, _ in pair_keys
    }
    metadata = _read_mimic_metadata(mimic_metadata, required_dicom_ids)
    finding_pairs: dict[str, int] = defaultdict(int)
    finding_patients: dict[str, set[str]] = defaultdict(set)
    finding_studies: dict[str, set[str]] = defaultdict(set)
    image_hash_pairs: list[tuple[str, str]] = []
    hashed_images: set[str] = set()
    identities: list[dict[str, str]] = []
    for finding, image_id, _ in sorted(pair_keys):
        dicom_id = dicom_by_image[image_id]
        identity = metadata[dicom_id]
        patient = identity["patient"]
        study = identity["study"]
        expected_relative = Path(
            "files", patient[:3], patient, study, f"{dicom_id}.jpg"
        )
        release_relative = Path(str(images[image_id].get("path", "")))
        if release_relative != expected_relative:
            raise ValueError(
                "MS-CXR image path does not match MIMIC metadata binding"
            )
        finding_pairs[finding] += 1
        finding_patients[finding].add(identity["patient"])
        finding_studies[finding].add(identity["study"])
        identities.append(identity)
        if dicom_id in hashed_images:
            continue
        image_path = (
            image_root / expected_relative
        ).resolve()
        if image_root not in image_path.parents:
            raise ValueError("resolved MIMIC image path escapes image root")
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        image_hash_pairs.append(
            (_identifier_sha256("image", dicom_id), file_sha256(image_path))
        )
        hashed_images.add(dicom_id)

    for finding in TARGET_FINDINGS:
        if finding_pairs[finding] != int(expected_pairs[finding]):
            raise ValueError(
                f"MS-CXR test {finding} has {finding_pairs[finding]} pairs; "
                f"expected {expected_pairs[finding]}"
            )
        if len(finding_patients[finding]) != int(expected_subjects[finding]):
            raise ValueError(
                f"MS-CXR test {finding} has {len(finding_patients[finding])} subjects; "
                f"expected {expected_subjects[finding]}"
            )

    prior_hashes = prior["identifier_sha256"]
    overlap = {
        kind: len(
            {_identifier_sha256(kind, row[kind]) for row in identities}
            & set(prior_hashes[kind])
        )
        for kind in ("patient", "study")
    }
    if any(overlap.values()):
        raise ValueError(f"MS-CXR official test overlaps prior MIMIC registry: {overlap}")

    identity_hashes = {
        kind: canonical_json_sha256(
            sorted({_identifier_sha256(kind, row[kind]) for row in identities})
        )
        for kind in ("patient", "study")
    }
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": (
            "metadata_intake_ready_no_model_authority"
            if license_gate_passed
            else "structure_preflight_passed_license_attestation_pending"
        ),
        "formal_result": False,
        "model_evaluation_authorized": False,
        "license_gate_passed": license_gate_passed,
        "source_split": "publisher_test_only",
        "dataset_root": str(root),
        "release": {
            "license_record_sha256": license_record_sha256,
            "package_sha256": package_sha256,
            "annotations_sha256": file_sha256(annotations_path),
            "mimic_metadata_sha256": file_sha256(mimic_metadata),
            "test_image_payload_sha256": canonical_json_sha256(sorted(image_hash_pairs)),
        },
        "counts": {
            "target_test_pairs": len(pair_keys),
            "target_test_boxes": len(target_annotations),
            "target_test_images": len(required_dicom_ids),
            "target_test_patients": len({row["patient"] for row in identities}),
            "target_test_studies": len({row["study"] for row in identities}),
        },
        "targets": {
            finding: {
                "pairs": finding_pairs[finding],
                "patients": len(finding_patients[finding]),
                "studies": len(finding_studies[finding]),
            }
            for finding in TARGET_FINDINGS
        },
        "identity_set_sha256": identity_hashes,
        "prior_registry_sha256": file_sha256(prior_registry),
        "prior_overlap_counts": overlap,
        "raw_identifiers_emitted": False,
        "claim_boundary": (
            "metadata-only intake; C5 remains final stop"
            if license_gate_passed
            else "structure-only preflight; credential/CITI/DUA attestation pending; "
            "C5 remains final stop"
        ),
        "intake_module_sha256": file_sha256(Path(__file__)),
    }
    payload["canonical_artifact_sha256"] = canonical_json_sha256(payload)
    return payload


def audit_ms_cxr_test_release(
    dataset_root: Path,
    *,
    mimic_metadata: Path,
    mimic_images_root: Path,
    license_record: Path,
    package_archive: Path | None = None,
    prior_registry: Path,
    expected_test_pairs: dict[str, int] | None = None,
    expected_test_subjects: dict[str, int] | None = None,
    enforce_frozen_prior_identity: bool = True,
) -> dict[str, Any]:
    """Run the strict intake audit after user access attestations exist."""

    return _inspect_ms_cxr_test_release(
        dataset_root,
        mimic_metadata=mimic_metadata,
        mimic_images_root=mimic_images_root,
        license_record=license_record,
        package_archive=package_archive,
        prior_registry=prior_registry,
        expected_test_pairs=expected_test_pairs,
        expected_test_subjects=expected_test_subjects,
        enforce_frozen_prior_identity=enforce_frozen_prior_identity,
    )


def preflight_ms_cxr_test_release(
    dataset_root: Path,
    *,
    mimic_metadata: Path,
    mimic_images_root: Path,
    package_archive: Path,
    prior_registry: Path,
    expected_test_pairs: dict[str, int] | None = None,
    expected_test_subjects: dict[str, int] | None = None,
    enforce_frozen_prior_identity: bool = True,
) -> dict[str, Any]:
    """Inspect acquired files without fabricating credential/CITI/DUA claims."""

    return _inspect_ms_cxr_test_release(
        dataset_root,
        mimic_metadata=mimic_metadata,
        mimic_images_root=mimic_images_root,
        license_record=None,
        package_archive=package_archive,
        prior_registry=prior_registry,
        expected_test_pairs=expected_test_pairs,
        expected_test_subjects=expected_test_subjects,
        enforce_frozen_prior_identity=enforce_frozen_prior_identity,
    )
