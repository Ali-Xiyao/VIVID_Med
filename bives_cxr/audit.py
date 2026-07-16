"""Readiness audits for BiVES-CXR manifests."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .data import read_manifest
from .decoder import STATE_NAMES


def audit_manifests(
    manifests: dict[str, str | Path],
    data_root: str | Path = ".",
    check_images: bool = False,
    require_complete_statements: bool = False,
) -> dict[str, Any]:
    """Audit schema, split isolation, state coverage, and optional image paths."""

    root = Path(data_root)
    report: dict[str, Any] = {
        "status": "pass",
        "errors": [],
        "warnings": [],
        "splits": {},
    }
    patients_by_split: dict[str, set[str]] = {}
    all_sample_ids: dict[str, str] = {}

    for split, manifest_path in manifests.items():
        rows = read_manifest(manifest_path)
        patients = {str(row["patient_id"]) for row in rows}
        patients_by_split[split] = patients
        state_counts = Counter(str(row["state"]) for row in rows)
        statement_states: dict[str, set[str]] = defaultdict(set)
        missing_images: list[str] = []

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

            if check_images:
                image_path = Path(str(row["image_path"]))
                if not image_path.is_absolute():
                    image_path = root / image_path
                if not image_path.is_file():
                    missing_images.append(str(image_path))

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

        report["splits"][split] = {
            "manifest": str(manifest_path),
            "records": len(rows),
            "patients": len(patients),
            "statements": len(statement_states),
            "state_counts": {state: state_counts[state] for state in STATE_NAMES},
            "incomplete_statement_count": len(incomplete),
            "missing_image_count": len(missing_images),
        }

    split_names = list(patients_by_split)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            overlap = patients_by_split[left] & patients_by_split[right]
            if overlap:
                report["errors"].append(
                    f"patient leakage between {left} and {right}: "
                    f"{len(overlap)} patients; examples={sorted(overlap)[:5]}"
                )

    if report["errors"]:
        report["status"] = "fail"
    elif report["warnings"]:
        report["status"] = "pass_with_warnings"
    return report
