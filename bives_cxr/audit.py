"""Readiness audits for BiVES-CXR manifests."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from PIL import Image, ImageStat

from .data import read_manifest
from .decoder import STATE_NAMES


PROVENANCE_FIELDS = {
    "group_id",
    "image_sha256",
    "label_source",
    "annotation_status",
}


def _normalized_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def _resolve_image_path(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_manifests(
    manifests: dict[str, str | Path],
    data_root: str | Path = ".",
    check_images: bool = False,
    require_complete_statements: bool = False,
    check_decodable: bool = False,
    reject_constant_images: bool = False,
    require_provenance: bool = False,
    verify_image_sha256: bool = True,
    require_matching_protocol: bool = False,
) -> dict[str, Any]:
    """Audit split isolation, group semantics, provenance, and image readiness."""

    root = Path(data_root)
    report: dict[str, Any] = {
        "status": "pass",
        "errors": [],
        "warnings": [],
        "splits": {},
    }
    patients_by_split: dict[str, set[str]] = {}
    all_sample_ids: dict[str, str] = {}
    image_paths_by_split: dict[str, set[str]] = {}
    image_hashes_by_split: dict[str, set[str]] = {}
    studies_by_split: dict[str, set[str]] = {}
    groups_by_split: dict[str, set[str]] = {}
    statement_texts: dict[str, set[str]] = defaultdict(set)
    image_statement_states: dict[tuple[str, str], set[str]] = defaultdict(set)
    image_hash_statement_states: dict[tuple[str, str], set[str]] = defaultdict(set)
    study_statement_states: dict[tuple[str, str], set[str]] = defaultdict(set)
    image_hash_cache: dict[str, str] = {}
    actual_hash_by_sample: dict[str, str] = {}

    for split, manifest_path in manifests.items():
        rows = read_manifest(manifest_path)
        patients = {str(row["patient_id"]) for row in rows}
        patients_by_split[split] = patients
        state_counts = Counter(str(row["state"]) for row in rows)
        statement_states: dict[str, set[str]] = defaultdict(set)
        quartet_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        missing_images: list[str] = []
        unreadable_images: list[str] = []
        constant_images: list[str] = []
        split_paths: set[str] = set()
        split_hashes: set[str] = set()
        split_studies: set[str] = set()
        split_groups: set[str] = set()
        provenance_missing = 0
        insufficient_kinds: Counter[str] = Counter()
        source_counts: Counter[str] = Counter()
        view_counts: Counter[str] = Counter()
        finding_counts: Counter[str] = Counter()

        for row in rows:
            sample_id = str(row["sample_id"])
            if sample_id in all_sample_ids:
                report["errors"].append(
                    f"duplicate sample_id {sample_id!r} in {all_sample_ids[sample_id]} and {split}"
                )
            else:
                all_sample_ids[sample_id] = split

            statement_id = str(row["canonical_statement_id"])
            statement_states[statement_id].add(str(row["state"]))
            statement_texts[statement_id].add(_normalized_text(row["statement_text"]))

            image_path = _resolve_image_path(root, row["image_path"])
            normalized_path = os.path.normcase(str(image_path.resolve(strict=False)))
            split_paths.add(normalized_path)
            image_statement_states[(normalized_path, statement_id)].add(str(row["state"]))
            image_hash = str(row.get("image_sha256", "")).strip().lower()
            if image_hash:
                if require_provenance and re.fullmatch(r"[0-9a-f]{64}", image_hash) is None:
                    report["errors"].append(
                        f"{split} sample {sample_id!r} has invalid image_sha256={image_hash!r}"
                    )
                if not verify_image_sha256:
                    split_hashes.add(image_hash)
                    actual_hash_by_sample[sample_id] = image_hash
            study_id = str(row.get("study_id", "")).strip()
            if study_id:
                split_studies.add(study_id)
                study_statement_states[(study_id, statement_id)].add(str(row["state"]))
            group_id = str(row.get("group_id", "")).strip()
            if group_id:
                split_groups.add(group_id)
                quartet_rows[group_id].append(row)
            declared_split = str(row.get("split", "")).strip()
            if declared_split and declared_split != split:
                report["errors"].append(
                    f"{split} sample {sample_id!r} declares split={declared_split!r}"
                )

            if require_provenance:
                missing = {
                    field
                    for field in PROVENANCE_FIELDS
                    if field not in row or not str(row[field]).strip()
                }
                if missing:
                    provenance_missing += 1
                if str(row["state"]) == "insufficient":
                    kind = str(row.get("insufficient_kind", "")).strip().lower()
                    insufficient_kinds[kind] += 1
                    if kind not in {"natural", "synthetic"}:
                        report["errors"].append(
                            f"{split} sample {sample_id!r} has invalid insufficient_kind={kind!r}"
                        )

            source_counts[str(row.get("source", row.get("source_dataset", "unknown")))] += 1
            view_counts[str(row.get("view", "unknown"))] += 1
            finding_counts[str(row.get("finding", statement_id))] += 1

            if check_images:
                if not image_path.is_file():
                    missing_images.append(str(image_path))
                elif check_decodable:
                    try:
                        with Image.open(image_path) as image:
                            image.load()
                            if reject_constant_images:
                                extrema = ImageStat.Stat(image.convert("L")).extrema[0]
                                if extrema[0] == extrema[1]:
                                    constant_images.append(str(image_path))
                    except Exception:
                        unreadable_images.append(str(image_path))
            if verify_image_sha256 and image_path.is_file():
                actual_hash = image_hash_cache.get(normalized_path)
                if actual_hash is None:
                    actual_hash = file_sha256(image_path)
                    image_hash_cache[normalized_path] = actual_hash
                split_hashes.add(actual_hash)
                actual_hash_by_sample[sample_id] = actual_hash
                image_hash_statement_states[(actual_hash, statement_id)].add(
                    str(row["state"])
                )
                if image_hash and image_hash != actual_hash:
                    report["errors"].append(
                        f"{split} sample {sample_id!r} image_sha256 mismatch: "
                        f"declared={image_hash}, actual={actual_hash}"
                    )

        absent_states = [state for state in STATE_NAMES if state_counts[state] == 0]
        if absent_states:
            report["errors"].append(f"{split} is missing states: {absent_states}")

        incomplete = {
            statement_id: sorted(set(STATE_NAMES) - states)
            for statement_id, states in statement_states.items()
            if states != set(STATE_NAMES)
        }
        if incomplete:
            message = f"{split} has {len(incomplete)} canonical statements without all four states"
            if require_complete_statements:
                report["errors"].append(message)
            else:
                report["warnings"].append(message)

        if missing_images:
            report["errors"].append(
                f"{split} has {len(missing_images)} missing image files; "
                f"examples={missing_images[:3]}"
            )
        if unreadable_images:
            report["errors"].append(
                f"{split} has {len(unreadable_images)} unreadable image files; "
                f"examples={unreadable_images[:3]}"
            )
        if constant_images:
            report["errors"].append(
                f"{split} has {len(constant_images)} constant image files; "
                f"examples={constant_images[:3]}"
            )
        if provenance_missing:
            report["errors"].append(
                f"{split} has {provenance_missing} rows missing required provenance fields "
                f"{sorted(PROVENANCE_FIELDS)}"
            )
        if require_provenance and state_counts["insufficient"] > 0:
            if insufficient_kinds["natural"] == 0 or insufficient_kinds["synthetic"] == 0:
                report["errors"].append(
                    f"{split} must contain both natural and synthetic insufficient rows; "
                    f"counts={dict(insufficient_kinds)}"
                )

        invalid_quartets: dict[str, dict[str, object]] = {}
        for group_id, group_rows in quartet_rows.items():
            group_state_counts = Counter(str(row["state"]) for row in group_rows)
            statement_ids = {
                str(row["canonical_statement_id"]) for row in group_rows
            }
            normalized_texts = {
                _normalized_text(row["statement_text"]) for row in group_rows
            }
            protocols = {
                str(row.get("matching_protocol_version", "")).strip()
                for row in group_rows
                if str(row.get("matching_protocol_version", "")).strip()
            }
            strata = {
                json.dumps(
                    row.get("matching_stratum"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for row in group_rows
                if isinstance(row.get("matching_stratum"), dict)
                and row.get("matching_stratum")
            }
            group_hashes = {
                actual_hash_by_sample.get(str(row["sample_id"]), "")
                for row in group_rows
            } - {""}
            group_studies = {
                str(row.get("study_id", "")).strip() for row in group_rows
            } - {""}
            group_patients = {
                str(row.get("patient_id", "")).strip() for row in group_rows
            } - {""}
            if (
                len(group_rows) != len(STATE_NAMES)
                or any(group_state_counts[state] != 1 for state in STATE_NAMES)
                or set(group_state_counts) != set(STATE_NAMES)
                or len(statement_ids) != 1
                or len(normalized_texts) != 1
                or (
                    require_matching_protocol
                    and (
                        len(protocols) != 1
                        or len(strata) != 1
                        or len(group_hashes) != len(STATE_NAMES)
                        or len(group_studies) != len(STATE_NAMES)
                        or len(group_patients) != len(STATE_NAMES)
                    )
                )
            ):
                invalid_quartets[group_id] = {
                    "records": len(group_rows),
                    "state_counts": {
                        state: group_state_counts[state] for state in STATE_NAMES
                    },
                    "statement_ids": sorted(statement_ids),
                    "statement_texts": sorted(normalized_texts),
                    "matching_protocols": sorted(protocols),
                    "matching_strata": sorted(strata),
                    "unique_image_hashes": len(group_hashes),
                    "unique_studies": len(group_studies),
                    "unique_patients": len(group_patients),
                }
        if invalid_quartets and (require_provenance or require_matching_protocol):
            report["errors"].append(
                f"{split} has {len(invalid_quartets)} invalid group_id quartets; "
                f"examples={list(invalid_quartets.items())[:3]}"
            )

        report["splits"][split] = {
            "manifest": str(manifest_path),
            "records": len(rows),
            "patients": len(patients),
            "statements": len(statement_states),
            "state_counts": {state: state_counts[state] for state in STATE_NAMES},
            "incomplete_statement_count": len(incomplete),
            "missing_image_count": len(missing_images),
            "unreadable_image_count": len(unreadable_images),
            "constant_image_count": len(constant_images),
            "provenance_missing_count": provenance_missing,
            "invalid_group_count": len(invalid_quartets),
            "verified_image_hash_count": len(split_hashes),
            "insufficient_kind_counts": dict(insufficient_kinds),
            "source_counts": dict(source_counts),
            "view_counts": dict(view_counts),
            "finding_counts": dict(finding_counts),
        }
        image_paths_by_split[split] = split_paths
        image_hashes_by_split[split] = split_hashes
        studies_by_split[split] = split_studies
        groups_by_split[split] = split_groups

    split_names = list(patients_by_split)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            overlap = patients_by_split[left] & patients_by_split[right]
            if overlap:
                report["errors"].append(
                    f"patient leakage between {left} and {right}: "
                    f"{len(overlap)} patients; examples={sorted(overlap)[:5]}"
                )
            path_overlap = image_paths_by_split[left] & image_paths_by_split[right]
            if path_overlap:
                report["errors"].append(
                    f"image leakage between {left} and {right}: "
                    f"{len(path_overlap)} paths; examples={sorted(path_overlap)[:3]}"
                )
            hash_overlap = image_hashes_by_split[left] & image_hashes_by_split[right]
            if hash_overlap:
                report["errors"].append(
                    f"image hash leakage between {left} and {right}: "
                    f"{len(hash_overlap)} hashes; examples={sorted(hash_overlap)[:3]}"
                )
            study_overlap = studies_by_split[left] & studies_by_split[right]
            if study_overlap:
                report["errors"].append(
                    f"study leakage between {left} and {right}: "
                    f"{len(study_overlap)} studies; examples={sorted(study_overlap)[:3]}"
                )
            group_overlap = groups_by_split[left] & groups_by_split[right]
            if group_overlap:
                report["errors"].append(
                    f"group leakage between {left} and {right}: "
                    f"{len(group_overlap)} groups; examples={sorted(group_overlap)[:3]}"
                )

    semantic_mismatches = {
        statement_id: sorted(texts)
        for statement_id, texts in statement_texts.items()
        if len(texts) != 1
    }
    if semantic_mismatches:
        report["errors"].append(
            f"{len(semantic_mismatches)} canonical statement IDs map to inconsistent text; "
            f"examples={list(semantic_mismatches.items())[:3]}"
        )
    conflicts = {
        key: sorted(states)
        for key, states in image_statement_states.items()
        if len(states) != 1
    }
    if conflicts:
        report["errors"].append(
            f"{len(conflicts)} image-statement pairs have conflicting labels; "
            f"examples={list(conflicts.items())[:3]}"
        )
    hash_conflicts = {
        key: sorted(states)
        for key, states in image_hash_statement_states.items()
        if len(states) != 1
    }
    if hash_conflicts:
        report["errors"].append(
            f"{len(hash_conflicts)} image-hash-statement pairs have conflicting labels; "
            f"examples={list(hash_conflicts.items())[:3]}"
        )
    study_conflicts = {
        key: sorted(states)
        for key, states in study_statement_states.items()
        if len(states) != 1
    }
    if study_conflicts:
        report["errors"].append(
            f"{len(study_conflicts)} study-statement pairs have conflicting labels; "
            f"examples={list(study_conflicts.items())[:3]}"
        )

    if report["errors"]:
        report["status"] = "fail"
    elif report["warnings"]:
        report["status"] = "pass_with_warnings"
    return report
