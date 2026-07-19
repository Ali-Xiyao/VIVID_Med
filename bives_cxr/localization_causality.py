"""Model-agnostic contracts for a localization-causality audit.

The module consumes precomputed scores and deterministic masks.  It does not
load a model, open a test split, or define a test-time tuning surface.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from scipy import stats

from .rescue_protocol import (
    deterministic_coordinate_zone_connected_control_mask,
    mask_geometry,
)
from .terminal_audit import image_perturbation_metrics


AUDIT_SCHEMA_VERSION = "cxr_localization_causality_audit_v1"
DEVELOPMENT_DATASET_ROLES = frozenset({"synthetic_development", "development"})
IDENTITY_FIELDS = (
    "row_id",
    "patient_id",
    "image_id",
    "pathology_id",
    "model_id",
    "explanation_id",
    "operator_id",
    "dataset_role",
)
SCORE_FIELDS = ("s0", "sX", "sCX", "sE", "sCE")
STRENGTH_METRICS = (
    "masked_l1",
    "masked_rms",
    "ssim",
    "masked_edge_energy_change",
)


def _binary_mask(value: np.ndarray, *, name: str) -> np.ndarray:
    mask = np.asarray(value)
    if mask.ndim != 2:
        raise ValueError(f"{name} must be a 2D mask")
    if mask.dtype != np.bool_ and not np.isin(mask, (0, 1)).all():
        raise ValueError(f"{name} must be binary")
    return mask.astype(bool, copy=False)


def localization_metrics(
    expert_mask: np.ndarray,
    explanation_mask: np.ndarray,
    *,
    content_mask: np.ndarray,
    explanation_map: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    """Return spatial agreement metrics without selecting a test threshold."""

    expert = _binary_mask(expert_mask, name="expert_mask")
    explanation = _binary_mask(explanation_mask, name="explanation_mask")
    content = _binary_mask(content_mask, name="content_mask")
    if expert.shape != explanation.shape or expert.shape != content.shape:
        raise ValueError("expert/explanation/content masks must share geometry")
    if not expert.any() or not explanation.any():
        raise ValueError("expert and explanation regions must be non-empty")
    if bool(((expert | explanation) & ~content).any()):
        raise ValueError("expert and explanation regions must stay inside content")

    intersection = int((expert & explanation).sum())
    union = int((expert | explanation).sum())
    expert_area = int(expert.sum())
    explanation_area = int(explanation.sum())
    point_hit: int | None = None
    if explanation_map is not None:
        saliency = np.asarray(explanation_map, dtype=np.float64)
        if saliency.shape != expert.shape or not np.isfinite(saliency[content]).all():
            raise ValueError("explanation_map must be finite and match mask geometry")
        if not content.any():
            raise ValueError("content mask is empty")
        maximum = int(np.argmax(np.where(content, saliency, -np.inf)))
        point_hit = int(expert.reshape(-1)[maximum])

    return {
        "intersection_pixels": intersection,
        "union_pixels": union,
        "expert_area_pixels": expert_area,
        "explanation_area_pixels": explanation_area,
        "iou": float(intersection / union),
        "dice": float(2.0 * intersection / (expert_area + explanation_area)),
        "expert_coverage": float(intersection / expert_area),
        "explanation_precision": float(intersection / explanation_area),
        "point_hit": point_hit,
    }


def validate_target_control_pair(
    target_mask: np.ndarray,
    control_mask: np.ndarray,
    *,
    content_mask: np.ndarray,
) -> dict[str, Any]:
    """Fail closed unless one control is valid for exactly one target."""

    target = _binary_mask(target_mask, name="target_mask")
    control = _binary_mask(control_mask, name="control_mask")
    content = _binary_mask(content_mask, name="content_mask")
    if target.shape != control.shape or target.shape != content.shape:
        raise ValueError("target/control/content masks must share geometry")
    target_geometry = mask_geometry(target, content)
    control_geometry = mask_geometry(control, content)
    if bool((target & control).any()):
        raise ValueError("target and matched control must be disjoint")
    if target_geometry["area_pixels"] != control_geometry["area_pixels"]:
        raise ValueError("target and matched control must have exact equal area")
    if control_geometry["component_count"] != 1:
        raise ValueError("matched control must be one 4-connected component")

    diagonal = float(np.hypot(*target.shape))
    centroid_distance = float(
        np.hypot(
            target_geometry["centroid_x"] - control_geometry["centroid_x"],
            target_geometry["centroid_y"] - control_geometry["centroid_y"],
        )
        / max(diagonal, np.finfo(np.float64).tiny)
    )
    perimeter_ratio = float(
        abs(
            np.log(
                control_geometry["perimeter_edges"]
                / target_geometry["perimeter_edges"]
            )
        )
    )
    return {
        "pass": True,
        "exact_area": True,
        "disjoint": True,
        "control_connected": True,
        "normalized_centroid_distance": centroid_distance,
        "log_perimeter_ratio": perimeter_ratio,
        "target": target_geometry,
        "control": control_geometry,
    }


def build_target_specific_controls(
    expert_mask: np.ndarray,
    explanation_mask: np.ndarray,
    *,
    content_mask: np.ndarray,
    seed_key: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build separate controls and exclude the other relevant target.

    The controls may overlap each other because they are never used in the same
    contrast.  Each control is disjoint from both expert and explanation masks.
    """

    expert = _binary_mask(expert_mask, name="expert_mask")
    explanation = _binary_mask(explanation_mask, name="explanation_mask")
    content = _binary_mask(content_mask, name="content_mask")
    if expert.shape != explanation.shape or expert.shape != content.shape:
        raise ValueError("expert/explanation/content masks must share geometry")
    content_for_expert = content & ~(explanation & ~expert)
    content_for_explanation = content & ~(expert & ~explanation)
    expert_control, expert_certificate = (
        deterministic_coordinate_zone_connected_control_mask(
            expert,
            content_for_expert,
            seed_key=f"{seed_key}:C_X",
        )
    )
    explanation_control, explanation_certificate = (
        deterministic_coordinate_zone_connected_control_mask(
            explanation,
            content_for_explanation,
            seed_key=f"{seed_key}:C_E",
        )
    )
    if bool((expert_control & (expert | explanation)).any()):
        raise AssertionError("C_X overlaps an expert or explanation target")
    if bool((explanation_control & (expert | explanation)).any()):
        raise AssertionError("C_E overlaps an expert or explanation target")
    return (
        {"C_X": expert_control, "C_E": explanation_control},
        {"C_X": expert_certificate, "C_E": explanation_certificate},
    )


def score_contrasts(scores: Mapping[str, float]) -> dict[str, float]:
    """Compute target-specific causal contrasts under higher-is-support."""

    missing = [field for field in SCORE_FIELDS if field not in scores]
    if missing:
        raise ValueError(f"missing score fields: {missing}")
    values = {field: float(scores[field]) for field in SCORE_FIELDS}
    if not np.isfinite(list(values.values())).all():
        raise ValueError("audit scores must be finite")
    d_x = values["s0"] - values["sX"]
    d_cx = values["s0"] - values["sCX"]
    d_e = values["s0"] - values["sE"]
    d_ce = values["s0"] - values["sCE"]
    return {
        **values,
        "dX": float(d_x),
        "dCX": float(d_cx),
        "dE": float(d_e),
        "dCE": float(d_ce),
        "CS_X": float(d_x - d_cx),
        "CS_E": float(d_e - d_ce),
    }


def intervention_strength_metrics(
    original: np.ndarray,
    perturbed: np.ndarray,
    *,
    intervention_mask: np.ndarray,
    content_mask: np.ndarray,
) -> dict[str, float]:
    """Expose the existing frozen image-space metrics under the new audit."""

    return image_perturbation_metrics(
        original,
        perturbed,
        intervention_mask,
        content_mask,
    )


def strength_match_diagnostics(
    target_metrics: Mapping[str, float],
    control_metrics: Mapping[str, float],
    *,
    pair_geometry: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """Apply explicit development thresholds without outcome-driven tuning."""

    required_thresholds = {
        "max_normalized_centroid_distance",
        "max_log_perimeter_ratio",
        "max_masked_l1_difference",
        "max_masked_rms_difference",
        "max_ssim_difference",
        "max_edge_difference",
    }
    missing = sorted(required_thresholds - set(thresholds))
    if missing:
        raise ValueError(f"missing strength thresholds: {missing}")
    threshold_values = [float(thresholds[key]) for key in required_thresholds]
    if not np.isfinite(threshold_values).all() or any(
        value < 0.0 for value in threshold_values
    ):
        raise ValueError("strength thresholds must be finite and non-negative")
    for metric in STRENGTH_METRICS:
        if metric not in target_metrics or metric not in control_metrics:
            raise ValueError(f"missing target/control strength metric: {metric}")
        values = [float(target_metrics[metric]), float(control_metrics[metric])]
        if not np.isfinite(values).all():
            raise ValueError(f"strength metric must be finite: {metric}")

    differences = {
        "normalized_centroid_distance": float(
            pair_geometry["normalized_centroid_distance"]
        ),
        "log_perimeter_ratio": float(pair_geometry["log_perimeter_ratio"]),
        "masked_l1_difference": float(
            abs(target_metrics["masked_l1"] - control_metrics["masked_l1"])
        ),
        "masked_rms_difference": float(
            abs(target_metrics["masked_rms"] - control_metrics["masked_rms"])
        ),
        "ssim_difference": float(abs(target_metrics["ssim"] - control_metrics["ssim"])),
        "edge_difference": float(
            abs(
                target_metrics["masked_edge_energy_change"]
                - control_metrics["masked_edge_energy_change"]
            )
        ),
    }
    checks = {
        "centroid": differences["normalized_centroid_distance"]
        <= float(thresholds["max_normalized_centroid_distance"]),
        "perimeter": differences["log_perimeter_ratio"]
        <= float(thresholds["max_log_perimeter_ratio"]),
        "masked_l1": differences["masked_l1_difference"]
        <= float(thresholds["max_masked_l1_difference"]),
        "masked_rms": differences["masked_rms_difference"]
        <= float(thresholds["max_masked_rms_difference"]),
        "ssim": differences["ssim_difference"]
        <= float(thresholds["max_ssim_difference"]),
        "edge": differences["edge_difference"]
        <= float(thresholds["max_edge_difference"]),
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "differences": differences,
        "thresholds": {key: float(thresholds[key]) for key in sorted(thresholds)},
    }


def build_precomputed_audit_row(
    *,
    identity: Mapping[str, Any],
    scores: Mapping[str, float],
    expert_mask: np.ndarray,
    explanation_mask: np.ndarray,
    expert_control_mask: np.ndarray,
    explanation_control_mask: np.ndarray,
    content_mask: np.ndarray,
    strength_metrics: Mapping[str, Mapping[str, float]],
    strength_thresholds: Mapping[str, float],
    explanation_map: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build one fail-closed development row from precomputed model scores."""

    normalized_identity = _validate_identity(identity)
    expert = _binary_mask(expert_mask, name="expert_mask")
    explanation = _binary_mask(explanation_mask, name="explanation_mask")
    expert_control = _binary_mask(expert_control_mask, name="expert_control_mask")
    explanation_control = _binary_mask(
        explanation_control_mask, name="explanation_control_mask"
    )
    content = _binary_mask(content_mask, name="content_mask")
    shapes = {mask.shape for mask in (expert, explanation, expert_control, explanation_control, content)}
    if len(shapes) != 1:
        raise ValueError("all audit masks must share geometry")
    if bool((expert_control & (expert | explanation)).any()):
        raise ValueError("C_X must be disjoint from expert and explanation targets")
    if bool((explanation_control & (expert | explanation)).any()):
        raise ValueError("C_E must be disjoint from expert and explanation targets")

    pair_x = validate_target_control_pair(expert, expert_control, content_mask=content)
    pair_e = validate_target_control_pair(
        explanation, explanation_control, content_mask=content
    )
    required_roles = {"X", "C_X", "E", "C_E"}
    if required_roles - set(strength_metrics):
        raise ValueError(
            f"missing strength roles: {sorted(required_roles - set(strength_metrics))}"
        )
    strength_x = strength_match_diagnostics(
        strength_metrics["X"],
        strength_metrics["C_X"],
        pair_geometry=pair_x,
        thresholds=strength_thresholds,
    )
    strength_e = strength_match_diagnostics(
        strength_metrics["E"],
        strength_metrics["C_E"],
        pair_geometry=pair_e,
        thresholds=strength_thresholds,
    )
    if not strength_x["pass"] or not strength_e["pass"]:
        raise ValueError("target/control perturbation-strength gate failed")

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        **normalized_identity,
        "formal_result": False,
        "test_opened": False,
        "score_direction": "higher_is_more_support",
        "localization": localization_metrics(
            expert,
            explanation,
            content_mask=content,
            explanation_map=explanation_map,
        ),
        "scores": score_contrasts(scores),
        "geometry": {"X_vs_C_X": pair_x, "E_vs_C_E": pair_e},
        "strength": {"X_vs_C_X": strength_x, "E_vs_C_E": strength_e},
    }


def _validate_identity(identity: Mapping[str, Any]) -> dict[str, str]:
    missing = [field for field in IDENTITY_FIELDS if field not in identity]
    if missing:
        raise ValueError(f"missing audit identity fields: {missing}")
    normalized = {field: str(identity[field]).strip() for field in IDENTITY_FIELDS}
    if any(not value for value in normalized.values()):
        raise ValueError("audit identity fields must be non-empty")
    if any("|" in value or "\n" in value or "\r" in value for value in normalized.values()):
        raise ValueError("audit identity fields cannot contain delimiters or newlines")
    if normalized["dataset_role"] not in DEVELOPMENT_DATASET_ROLES:
        raise ValueError("only synthetic/development rows are accepted before test opening")
    return normalized


def validate_precomputed_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate a development JSONL surface and reject every test-like row."""

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in rows:
        row = dict(source)
        if row.get("schema_version") != AUDIT_SCHEMA_VERSION:
            raise ValueError("unexpected localization-causality audit schema")
        identity = _validate_identity(row)
        if row.get("formal_result") is not False or row.get("test_opened") is not False:
            raise ValueError("development rows must remain nonformal and test-closed")
        if identity["row_id"] in seen:
            raise ValueError(f"duplicate audit row_id: {identity['row_id']}")
        seen.add(identity["row_id"])
        if row.get("score_direction") != "higher_is_more_support":
            raise ValueError("audit score direction is not frozen")
        localization = row.get("localization", {})
        scores = row.get("scores", {})
        if not np.isfinite(
            [localization.get("iou", np.nan), scores.get("CS_X", np.nan), scores.get("CS_E", np.nan)]
        ).all():
            raise ValueError("audit primary metrics must be finite")
        for key in ("X_vs_C_X", "E_vs_C_E"):
            if row.get("geometry", {}).get(key, {}).get("pass") is not True:
                raise ValueError(f"geometry gate is not passed: {key}")
            if row.get("strength", {}).get(key, {}).get("pass") is not True:
                raise ValueError(f"strength gate is not passed: {key}")
        normalized.append(row)
    if not normalized:
        raise ValueError("audit rows are empty")
    return normalized


def summarize_audit_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int = 20260719,
) -> dict[str, Any]:
    """Summarize localization and causal endpoints with patient resampling."""

    records = validate_precomputed_rows(rows)
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        key = "|".join(
            [
                row["model_id"],
                row["explanation_id"],
                row["pathology_id"],
                row["operator_id"],
            ]
        )
        groups.setdefault(key, []).append(row)

    group_summaries: dict[str, Any] = {}
    for group_index, (key, group) in enumerate(sorted(groups.items())):
        iou = np.asarray([row["localization"]["iou"] for row in group], dtype=np.float64)
        cs_x = np.asarray([row["scores"]["CS_X"] for row in group], dtype=np.float64)
        cs_e = np.asarray([row["scores"]["CS_E"] for row in group], dtype=np.float64)
        point_values = [row["localization"].get("point_hit") for row in group]
        finite_point = [float(value) for value in point_values if value is not None]
        association = None
        if len(group) >= 3 and not np.all(iou == iou[0]) and not np.all(cs_e == cs_e[0]):
            association = float(stats.spearmanr(iou, cs_e).statistic)
        intervals = _patient_cluster_intervals(
            group,
            metrics=("localization_iou", "CS_X", "CS_E"),
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + group_index,
        )
        group_summaries[key] = {
            "records": len(group),
            "patients": len({row["patient_id"] for row in group}),
            "mean_localization_iou": float(iou.mean()),
            "mean_point_hit": float(np.mean(finite_point)) if finite_point else None,
            "mean_CS_X": float(cs_x.mean()),
            "mean_CS_E": float(cs_e.mean()),
            "CS_E_sign_counts": dict(
                sorted(Counter("positive" if value > 0 else "nonpositive" for value in cs_e).items())
            ),
            "localization_CS_E_spearman": association,
            "patient_cluster_bootstrap_95ci": intervals,
        }

    cross_operator: dict[str, Any] = {}
    base_keys = sorted(
        {
            "|".join([row["model_id"], row["explanation_id"], row["pathology_id"]])
            for row in records
        }
    )
    for base in base_keys:
        operator_values = {
            key.rsplit("|", 1)[-1]: summary["mean_CS_E"]
            for key, summary in group_summaries.items()
            if key.startswith(f"{base}|")
        }
        signs = {operator: value > 0 for operator, value in operator_values.items()}
        cross_operator[base] = {
            "operators": dict(sorted(operator_values.items())),
            "all_positive": bool(operator_values and all(signs.values())),
            "sign_agreement": bool(len(set(signs.values())) <= 1),
            "worst_mean_CS_E": float(min(operator_values.values())),
        }

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "synthetic_or_development_complete_nonformal",
        "formal_result": False,
        "test_opened": False,
        "rows": len(records),
        "patients": len({row["patient_id"] for row in records}),
        "groups": group_summaries,
        "cross_operator": cross_operator,
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed": int(bootstrap_seed),
    }


def _patient_cluster_intervals(
    rows: list[dict[str, Any]],
    *,
    metrics: tuple[str, ...],
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    patients = sorted({row["patient_id"] for row in rows})
    by_patient = {
        patient: [row for row in rows if row["patient_id"] == patient]
        for patient in patients
    }
    rng = np.random.default_rng(seed)
    values = {metric: [] for metric in metrics}
    for _ in range(replicates):
        sample = [
            row
            for patient in rng.choice(patients, size=len(patients), replace=True)
            for row in by_patient[str(patient)]
        ]
        for metric in metrics:
            if metric == "localization_iou":
                vector = [row["localization"]["iou"] for row in sample]
            else:
                vector = [row["scores"][metric] for row in sample]
            values[metric].append(float(np.mean(vector)))
    return {
        metric: {
            "lower": float(np.percentile(vector, 2.5)),
            "upper": float(np.percentile(vector, 97.5)),
        }
        for metric, vector in values.items()
    }
