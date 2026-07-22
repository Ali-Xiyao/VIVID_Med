"""Read-only terminal diagnostics for the MORPH separability gate."""

from __future__ import annotations

import math
import statistics
from typing import Any

from .protocol import MORPH_FINDINGS, canonical_sha256


def _binary_auc(labels: list[int], scores: list[float]) -> float:
    positive = [score for label, score in zip(labels, scores, strict=True) if label == 1]
    negative = [score for label, score in zip(labels, scores, strict=True) if label == 0]
    if not positive or not negative:
        raise ValueError("AUROC requires both positive and negative rows")
    wins = 0.0
    for pos in positive:
        for neg in negative:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positive) * len(negative))


def _metric_for_expert(expert: str) -> str:
    return {
        "boundary": "boundary_hit",
        "region": "spatial_hit",
        "geometry": "geometry_iou",
        "distribution": "moment_score",
    }[expert]


def _seed_metrics(
    result: dict[str, Any], finding: str, prescribed: str
) -> dict[str, list[float]]:
    generic_runs = result["runs"]["generic"]
    generic_auroc = [float(run["metrics"][finding]["auroc"]) for run in generic_runs]
    if prescribed == "region_boundary":
        expert_auroc: list[float] = []
        expert_spatial: list[float] = []
        generic_spatial: list[float] = []
        for region, boundary, generic in zip(
            result["runs"]["region"],
            result["runs"]["boundary"],
            generic_runs,
            strict=True,
        ):
            region_rows = {
                row["sample_id"]: row
                for row in region["validation_rows"]
                if row["finding"] == finding
            }
            boundary_rows = {
                row["sample_id"]: row
                for row in boundary["validation_rows"]
                if row["finding"] == finding
            }
            if set(region_rows) != set(boundary_rows):
                raise ValueError("Region/boundary validation identities changed")
            keys = sorted(region_rows)
            labels = [int(region_rows[key]["label"]) for key in keys]
            scores = [
                0.5
                * (
                    float(region_rows[key]["margin"])
                    + float(boundary_rows[key]["margin"])
                )
                for key in keys
            ]
            expert_auroc.append(_binary_auc(labels, scores))
            expert_spatial.append(
                0.5
                * (
                    float(region["metrics"][finding]["spatial_hit"])
                    + float(boundary["metrics"][finding]["boundary_hit"])
                )
            )
            generic_spatial.append(
                0.5
                * (
                    float(generic["metrics"][finding]["spatial_hit"])
                    + float(generic["metrics"][finding]["boundary_hit"])
                )
            )
    else:
        metric = _metric_for_expert(prescribed)
        expert_runs = result["runs"][prescribed]
        expert_auroc = [
            float(run["metrics"][finding]["auroc"]) for run in expert_runs
        ]
        expert_spatial = [
            float(run["metrics"][finding][metric]) for run in expert_runs
        ]
        generic_spatial = [
            float(run["metrics"][finding][metric]) for run in generic_runs
        ]
    return {
        "generic_auroc": generic_auroc,
        "expert_auroc": expert_auroc,
        "generic_spatial": generic_spatial,
        "expert_spatial": expert_spatial,
    }


def _failure_class(discrimination_gain: float, spatial_gain: float) -> str:
    if discrimination_gain > 0 and spatial_gain > 0:
        return "joint_advantage"
    if discrimination_gain > 0 and spatial_gain <= 0:
        return "discrimination_only_without_spatial_advantage"
    if discrimination_gain <= 0 and spatial_gain > 0:
        return "spatial_only_without_discrimination_advantage"
    return "no_joint_advantage"


def analyze_morph_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("schema_version") != "morph-separability-result-v1":
        raise ValueError("Unsupported MORPH result schema")
    if set(result.get("per_finding", {})) != set(MORPH_FINDINGS):
        raise ValueError("MORPH result finding set changed")
    if result.get("chexlocalize_test_opened") or result.get("vindr_test_opened"):
        raise ValueError("Terminal case study cannot consume an opened test split")

    findings: dict[str, Any] = {}
    checkpoint_hashes: list[str] = []
    final_loss_lower_than_initial = 0
    total_runs = 0
    for expert, runs in result["runs"].items():
        for run in runs:
            total_runs += 1
            checkpoint_hashes.append(str(run["checkpoint_sha256"]))
            events = list(run["events"])
            if len(events) < 2:
                raise ValueError(f"{expert} run lacks frozen endpoint events")
            if float(events[-1]["loss"]) < float(events[0]["loss"]):
                final_loss_lower_than_initial += 1

    for finding in MORPH_FINDINGS:
        gate = result["per_finding"][finding]
        prescribed = str(gate["prescribed_expert"])
        seeds = _seed_metrics(result, finding, prescribed)
        discrimination_gains = [
            expert - generic
            for expert, generic in zip(
                seeds["expert_auroc"], seeds["generic_auroc"], strict=True
            )
        ]
        spatial_gains = [
            expert - generic
            for expert, generic in zip(
                seeds["expert_spatial"], seeds["generic_spatial"], strict=True
            )
        ]
        findings[finding] = {
            "prescribed_expert": prescribed,
            "pass": bool(gate["pass"]),
            "failure_class": _failure_class(
                float(gate["discrimination_gain"]), float(gate["spatial_gain"])
            ),
            "expert_auroc_by_seed": seeds["expert_auroc"],
            "generic_auroc_by_seed": seeds["generic_auroc"],
            "discrimination_gain_by_seed": discrimination_gains,
            "expert_spatial_by_seed": seeds["expert_spatial"],
            "generic_spatial_by_seed": seeds["generic_spatial"],
            "spatial_gain_by_seed": spatial_gains,
            "median_discrimination_gain": statistics.median(discrimination_gains),
            "median_spatial_gain": statistics.median(spatial_gains),
            "expert_auroc_range": max(seeds["expert_auroc"])
            - min(seeds["expert_auroc"]),
            "expert_spatial_range": max(seeds["expert_spatial"])
            - min(seeds["expert_spatial"]),
        }

    all_finite = all(
        math.isfinite(value)
        for finding in findings.values()
        for key, values in finding.items()
        if key.endswith("_by_seed")
        for value in values
    )
    analysis = {
        "schema_version": "morph-separability-case-study-v1",
        "source_result_canonical_sha256": result["canonical_sha256"],
        "source_status": result["status"],
        "gate_pass": bool(result["gate_pass"]),
        "passed_findings": int(result["passed_findings"]),
        "minimum_passing_findings": int(result["minimum_passing_findings"]),
        "findings": findings,
        "execution_integrity": {
            "completed_runs": total_runs,
            "expected_runs": 15,
            "unique_checkpoint_hashes": len(set(checkpoint_hashes)),
            "runs_with_lower_final_loss": final_loss_lower_than_initial,
            "all_reported_metrics_finite": all_finite,
            "monotonic_min_delta": float(result["monotonic_min_delta"]),
            "concept_direct_median_auroc_gap": float(
                result["concept_direct_median_auroc_gap"]
            ),
        },
        "diagnosis": (
            "The locked development surface does not support morphology-specific "
            "expert superiority over generic patch-MIL. Optimization and concept "
            "contracts completed, but only cardiomegaly achieved a joint "
            "discrimination-and-spatial advantage. This is a scientific survival-gate "
            "failure, not a threshold or execution repair target."
        ),
        "action": "stop_before_full_morph_no_result_driven_remap_or_rerun",
        "test_splits_opened": False,
    }
    analysis["canonical_sha256"] = canonical_sha256(analysis)
    return analysis
