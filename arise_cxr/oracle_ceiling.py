"""Fail-closed expert-mask oracle ceiling gate for ARISE-CXR development.

The gate deliberately reuses already-frozen development scores when their
identity and hashes match.  It does not load a model, create a new score, or
touch the reserved CheXlocalize test split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from bives_cxr.localization_causality import (
    summarize_audit_rows,
    validate_precomputed_rows,
)
from bives_cxr.provenance import canonical_json_sha256, file_sha256


ORACLE_CEILING_SCHEMA_VERSION = "arise-cxr-oracle-ceiling-development-v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("oracle ceiling input rows are empty")
    return rows


def load_locked_development_rows(
    *,
    rows_path: str | Path,
    merged_result_path: str | Path,
    expected_input_lock_sha256: str,
    expected_model_id: str,
    expected_explanation_id: str,
    expected_pathologies: Iterable[str],
    expected_operators: Iterable[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load Phase-H rows only when every frozen development identity matches."""

    rows_file = Path(rows_path)
    result_file = Path(merged_result_path)
    if not rows_file.is_file() or not result_file.is_file():
        raise FileNotFoundError("locked Phase-H rows or merged result are missing")

    result = json.loads(result_file.read_text(encoding="utf-8"))
    declared_canonical = result.get("canonical_sha256")
    canonical_payload = dict(result)
    canonical_payload.pop("canonical_sha256", None)
    if declared_canonical != canonical_json_sha256(canonical_payload):
        raise ValueError("Phase-H merged result canonical SHA256 changed")
    if result.get("rows_sha256") != file_sha256(rows_file):
        raise ValueError("Phase-H audit row SHA256 changed")
    if result.get("input_lock_canonical_sha256") != expected_input_lock_sha256:
        raise ValueError("Phase-H input lock identity changed")
    if result.get("status") != "complete_nonformal_development":
        raise ValueError("Phase-H merged result is not complete development evidence")
    if result.get("formal_result") is not False:
        raise ValueError("oracle ceiling cannot consume a formal result")
    if result.get("confirmatory_evidence") is not False:
        raise ValueError("oracle ceiling input cannot be confirmatory evidence")
    if result.get("test_opened") is not False:
        raise ValueError("oracle ceiling input opened the reserved test split")

    rows = validate_precomputed_rows(_read_jsonl(rows_file))
    if result.get("audit_rows") != len(rows):
        raise ValueError("Phase-H merged row count changed")
    if any(row.get("source_split") != "publisher_validation_prior_exposed_development" for row in rows):
        raise ValueError("oracle ceiling accepts only prior-exposed validation development rows")
    if {row["model_id"] for row in rows} != {expected_model_id}:
        raise ValueError("oracle ceiling model identity changed")
    if {row["explanation_id"] for row in rows} != {expected_explanation_id}:
        raise ValueError("oracle ceiling explanation identity changed")
    if {row["pathology_id"] for row in rows} != set(expected_pathologies):
        raise ValueError("oracle ceiling pathology identity changed")
    if {row["operator_id"] for row in rows} != set(expected_operators):
        raise ValueError("oracle ceiling operator identity changed")
    if any(row.get("patient_level_claim") is not True for row in rows):
        raise ValueError("oracle ceiling requires patient-aware development rows")
    return rows, result


def evaluate_oracle_ceiling(
    rows: Iterable[Mapping[str, Any]],
    *,
    required_pathologies: Iterable[str],
    required_operators: Iterable[str],
    minimum_passing_pathologies: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    ci_lower_threshold: float = 0.0,
) -> dict[str, Any]:
    """Evaluate whether expert masks establish a multi-operator causal ceiling.

    A pathology passes only when every required operator has a patient-cluster
    bootstrap lower confidence bound for expert-region ``CS_X`` strictly above
    ``ci_lower_threshold``.  The ARISE selector remains locked unless at least
    ``minimum_passing_pathologies`` pass.
    """

    required_pathology_set = set(required_pathologies)
    required_operator_set = set(required_operators)
    if minimum_passing_pathologies < 1:
        raise ValueError("minimum_passing_pathologies must be positive")
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    if not required_pathology_set or not required_operator_set:
        raise ValueError("oracle ceiling requires pathologies and operators")

    records = validate_precomputed_rows(rows)
    observed_pathologies = {row["pathology_id"] for row in records}
    observed_operators = {row["operator_id"] for row in records}
    if observed_pathologies != required_pathology_set:
        raise ValueError("oracle ceiling observed pathology set changed")
    if observed_operators != required_operator_set:
        raise ValueError("oracle ceiling observed operator set changed")

    summary = summarize_audit_rows(
        records,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    cells: dict[str, Any] = {}
    pathology_results: dict[str, Any] = {}
    for pathology in sorted(required_pathology_set):
        operator_passes: list[bool] = []
        operator_means: dict[str, float] = {}
        for operator in sorted(required_operator_set):
            matching = [
                value
                for key, value in summary["groups"].items()
                if key.endswith(f"|{pathology}|{operator}")
            ]
            if len(matching) != 1:
                raise ValueError(f"oracle ceiling cell is missing or duplicated: {pathology}/{operator}")
            group = matching[0]
            interval = group["patient_cluster_bootstrap_95ci"]["CS_X"]
            lower = float(interval["lower"])
            upper = float(interval["upper"])
            mean = float(group["mean_CS_X"])
            passed = lower > float(ci_lower_threshold)
            key = f"{pathology}|{operator}"
            cells[key] = {
                "records": int(group["records"]),
                "patients": int(group["patients"]),
                "mean_CS_X": mean,
                "patient_cluster_bootstrap_95ci": {"lower": lower, "upper": upper},
                "ci_lower_threshold": float(ci_lower_threshold),
                "pass": passed,
            }
            operator_means[operator] = mean
            operator_passes.append(passed)
        pathology_results[pathology] = {
            "operators": operator_means,
            "all_operator_means_positive": all(value > 0.0 for value in operator_means.values()),
            "operator_sign_agreement": len({value > 0.0 for value in operator_means.values()}) == 1,
            "worst_operator_mean_CS_X": min(operator_means.values()),
            "all_operator_ci_lower_bounds_above_threshold": all(operator_passes),
            "pass": all(operator_passes),
        }

    passing = sorted(name for name, value in pathology_results.items() if value["pass"])
    coverage_pass = len(required_pathology_set) >= minimum_passing_pathologies
    passing_count_pass = len(passing) >= minimum_passing_pathologies
    selector_unlocked = coverage_pass and passing_count_pass
    failure_reasons: list[str] = []
    if not coverage_pass:
        failure_reasons.append(
            f"only {len(required_pathology_set)} pathologies are available; "
            f"the method gate requires {minimum_passing_pathologies}"
        )
    failed_pathologies = sorted(set(required_pathology_set) - set(passing))
    if failed_pathologies:
        failure_reasons.append(
            "expert-mask oracle CI gate failed for: " + ", ".join(failed_pathologies)
        )
    if not passing_count_pass:
        failure_reasons.append(
            f"only {len(passing)} pathologies pass; "
            f"the method gate requires {minimum_passing_pathologies}"
        )

    result = {
        "schema_version": ORACLE_CEILING_SCHEMA_VERSION,
        "status": "pass_unlock_sc_selector" if selector_unlocked else "fail_stop_before_selector",
        "formal_result": False,
        "confirmatory_evidence": False,
        "test_opened": False,
        "new_model_scores_created": False,
        "source": "frozen_phase_h_chexlocalize_validation_development_rows",
        "rows": len(records),
        "patients": len({row["patient_id"] for row in records}),
        "required_pathologies": sorted(required_pathology_set),
        "required_operators": sorted(required_operator_set),
        "minimum_passing_pathologies": int(minimum_passing_pathologies),
        "ci_lower_threshold": float(ci_lower_threshold),
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed": int(bootstrap_seed),
        "cells": cells,
        "pathologies": pathology_results,
        "passing_pathologies": passing,
        "coverage_pass": coverage_pass,
        "passing_count_pass": passing_count_pass,
        "selector_training_unlocked": selector_unlocked,
        "failure_reasons": failure_reasons,
        "next_action": (
            "freeze an S/C-only ARISE selector development identity"
            if selector_unlocked
            else "diagnose dense verifier, intervention, and matched controls one factor at a time"
        ),
    }
    result["canonical_sha256"] = canonical_json_sha256(result)
    return result
