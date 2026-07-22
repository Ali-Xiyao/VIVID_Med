"""Aggregate, identifier-free failure diagnosis for ARISE oracle runs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np

from bives_cxr.localization_causality import validate_precomputed_rows
from bives_cxr.provenance import canonical_json_sha256


CASE_STUDY_SCHEMA_VERSION = "arise-cxr-oracle-case-study-v1"
SCORE_FIELDS = ("s0", "dX", "dCX", "CS_X", "dE", "dCE", "CS_E")


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def analyze_oracle_failure(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize mechanism failures without exposing patient/image identities."""

    records = validate_precomputed_rows(rows)
    if any(row.get("dataset_role") != "development" for row in records):
        raise ValueError("ARISE case study accepts development rows only")
    if any(row.get("test_opened") is not False for row in records):
        raise ValueError("ARISE case study cannot consume opened test rows")

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["pathology_id"]), str(row["operator_id"]))].append(row)

    cells: dict[str, Any] = {}
    pathology_operator_means: dict[str, dict[str, float]] = defaultdict(dict)
    for (pathology, operator), cell_rows in sorted(grouped.items()):
        values = {
            field: [float(row["scores"][field]) for row in cell_rows]
            for field in SCORE_FIELDS
        }
        means = {field: _mean(field_values) for field, field_values in values.items()}
        pathology_operator_means[pathology][operator] = means["CS_X"]
        diagnoses: list[str] = []
        response_scale = max(
            max(abs(value) for value in values["dX"]),
            max(abs(value) for value in values["dCX"]),
        )
        if response_scale < 1e-3:
            diagnoses.append("score_amplitude_collapsed")
        if abs(means["dX"]) < 0.1 * max(abs(means["dCX"]), 1e-12):
            diagnoses.append("expert_target_inert_relative_to_control")
        if means["dCX"] > means["dX"]:
            diagnoses.append("matched_control_effect_exceeds_expert_target")
        if means["dX"] <= 0.0:
            diagnoses.append("expert_target_mean_effect_nonpositive")
        cells[f"{pathology}|{operator}"] = {
            "records": len(cell_rows),
            "means": means,
            "median_CS_X": float(np.median(values["CS_X"])),
            "positive_CS_X_fraction": float(np.mean(np.asarray(values["CS_X"]) > 0.0)),
            "response_scale_max_abs_dX_dCX": response_scale,
            "diagnoses": diagnoses,
        }

    pathologies: dict[str, Any] = {}
    for pathology, operator_means in sorted(pathology_operator_means.items()):
        signs = {operator: value > 0.0 for operator, value in sorted(operator_means.items())}
        sign_agreement = len(set(signs.values())) == 1
        diagnoses = [] if sign_agreement else ["operator_sign_reversal"]
        pathologies[pathology] = {
            "operator_mean_CS_X": dict(sorted(operator_means.items())),
            "operator_positive_sign": signs,
            "operator_sign_agreement": sign_agreement,
            "diagnoses": diagnoses,
        }

    diagnosis_counts: dict[str, int] = defaultdict(int)
    for cell in cells.values():
        for diagnosis in cell["diagnoses"]:
            diagnosis_counts[diagnosis] += 1
    for pathology in pathologies.values():
        for diagnosis in pathology["diagnoses"]:
            diagnosis_counts[diagnosis] += 1

    result = {
        "schema_version": CASE_STUDY_SCHEMA_VERSION,
        "status": "complete_development_case_study",
        "formal_result": False,
        "confirmatory_evidence": False,
        "test_opened": False,
        "identifier_free": True,
        "rows": len(records),
        "patients": len({row["patient_id"] for row in records}),
        "cells": cells,
        "pathologies": pathologies,
        "diagnosis_counts": dict(sorted(diagnosis_counts.items())),
    }
    result["canonical_sha256"] = canonical_json_sha256(result)
    return result
