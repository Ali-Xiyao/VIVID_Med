"""Result-locked failure diagnosis for VICER-CXR V0."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np

from .validity import canonical_sha256, meets_minimum


CASE_STUDY_SCHEMA_VERSION = "vicer-v0-failure-case-study-v1"


def analyze_v0_failure(
    rows: Iterable[dict[str, Any]],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Attribute a completed V0 failure without changing its frozen thresholds."""
    records = list(rows)
    if result.get("schema_version") != "vicer-v0-validity-dose-response-v1":
        raise ValueError("unexpected VICER V0 result schema")
    if len(records) != int(result.get("records", -1)):
        raise ValueError("VICER V0 rows/result record count mismatch")
    if result.get("v0_pass") is not False or result.get("v1_authorized") is not False:
        raise ValueError("failure case study requires a terminal failed V0 result")

    thresholds = result["thresholds"]
    minimum_remove = float(result["run_identity"]["thresholds"]["minimum_q_remove"])
    minimum_preserve = float(thresholds["minimum_preservation"])
    minimum_realism = float(thresholds["minimum_realism"])
    minimum_rho = float(thresholds["minimum_monotonic_spearman"])
    minimum_valid_fraction = float(thresholds["minimum_valid_fraction"])

    cells: dict[str, Any] = {}
    aggregate_row_failures: Counter[str] = Counter()
    failed_cell_components: Counter[str] = Counter()
    for family, family_result in sorted(result["per_operator_family"].items()):
        for finding, summary in family_result["per_finding"].items():
            cell_rows = [
                row
                for row in records
                if row["operator_family"] == family
                and row["canonical_statement_id"] == finding
            ]
            row_failures: Counter[str] = Counter()
            for row in cell_rows:
                components = []
                if float(row["q_remove"]) < minimum_remove:
                    components.append("removal")
                if float(row["q_preserve"]) < minimum_preserve:
                    components.append("preservation")
                if float(row["q_realism"]) < minimum_realism:
                    components.append("realism")
                key = "+".join(components) if components else "valid"
                row_failures[key] += 1
                for component in components:
                    aggregate_row_failures[component] += 1

            failed_components = []
            if not meets_minimum(
                float(summary["q_remove_strength_spearman"]), minimum_rho
            ):
                failed_components.append("removal_monotonicity")
            if not meets_minimum(
                float(summary["median_q_preserve"][-1]), minimum_preserve
            ):
                failed_components.append("strongest_preservation")
            if not meets_minimum(
                float(summary["median_q_realism"][-1]), minimum_realism
            ):
                failed_components.append("strongest_realism")
            if not meets_minimum(
                float(summary["valid_fraction"]), minimum_valid_fraction
            ):
                failed_components.append("valid_fraction")
            gap = float(summary["mean_valid_target_control_gap"])
            if not np.isfinite(gap) or gap <= 0.0:
                failed_components.append("valid_target_control_gap")
            if not summary["pass"]:
                failed_cell_components.update(failed_components)

            cells[f"{family}|{finding}"] = {
                "pass": bool(summary["pass"]),
                "failed_components": failed_components,
                "row_failure_counts": dict(sorted(row_failures.items())),
                "q_remove_strength_spearman": float(summary["q_remove_strength_spearman"]),
                "strongest_median_q_preserve": float(summary["median_q_preserve"][-1]),
                "strongest_median_q_realism": float(summary["median_q_realism"][-1]),
                "valid_fraction": float(summary["valid_fraction"]),
                "mean_valid_target_control_gap": gap,
            }

    passing_cells = sum(cell["pass"] for cell in cells.values())
    gap_failure_cells = int(failed_cell_components["valid_target_control_gap"])
    diagnosis = {
        "engineering_execution_failure": False,
        "head_calibration_failure": not bool(result["head_gate_pass"]),
        "dominant_invalid_row_component": (
            aggregate_row_failures.most_common(1)[0][0]
            if aggregate_row_failures
            else None
        ),
        "all_cells_have_positive_mean_gap_when_valid": gap_failure_cells == 0,
        "interpretation": (
            "The frozen local critic does not verify sufficient finding removal "
            "for enough samples across all findings; preservation also fails for "
            "pneumothorax. Valid rows retain positive target-control effects, so "
            "the terminal failure is intervention validity/coverage rather than "
            "head calibration, execution, realism at strongest dose, or score orientation."
        ),
        "repair_decision": (
            "No code or threshold repair is justified. Freeze V0 as failed and keep "
            "V1 coverage-redundancy and V2 coalition learning locked."
        ),
    }
    case_study = {
        "schema_version": CASE_STUDY_SCHEMA_VERSION,
        "status": "terminal_v0_failure_diagnosed",
        "v0_result_canonical_sha256": result["canonical_sha256"],
        "v0_rows_sha256": result["rows_sha256"],
        "records": len(records),
        "cells_total": len(cells),
        "cells_passed": passing_cells,
        "cells_failed": len(cells) - passing_cells,
        "surviving_operator_families": list(result["surviving_operator_families"]),
        "aggregate_row_failure_components": dict(sorted(aggregate_row_failures.items())),
        "failed_cell_components": dict(sorted(failed_cell_components.items())),
        "cells": cells,
        "diagnosis": diagnosis,
        "boundaries": {
            "thresholds_changed": False,
            "model_rescored": False,
            "test_opened": False,
            "v1_authorized": False,
            "selector_started": False,
        },
    }
    case_study["canonical_sha256"] = canonical_sha256(case_study)
    return case_study
