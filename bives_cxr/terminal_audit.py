"""Read-only decomposition helpers for the terminal BiVES B2 evidence route.

This module operates only on already frozen score rows, geometry rows, masks,
and bound images.  It must never load a model or create a new experiment stage.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np
from scipy import ndimage, stats


OPERATORS = ("local_mean", "masked_gaussian_blur")


def classify_effect_pair(target_effect: float, control_effect: float) -> str:
    """Assign one of four sign/order categories without tuned thresholds."""

    target = float(target_effect)
    control = float(control_effect)
    if target < 0.0:
        return "target_sign_reversal"
    if target > 0.0 and target > control:
        return "target_dominant_positive"
    if control > 0.0:
        return "control_dominant_or_target_inert"
    return "both_nonpositive_or_tied"


def flatten_effect_rows(
    rows: Iterable[dict[str, Any]],
    *,
    source: str,
    geometry_by_sample: dict[str, dict[str, Any]] | None = None,
    manifest_by_sample: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Normalize frozen C5/C6I rows into an analysis-only long table."""

    geometry_by_sample = geometry_by_sample or {}
    manifest_by_sample = manifest_by_sample or {}
    flattened: list[dict[str, Any]] = []
    for row in rows:
        if source == "c5" and not (
            int(row.get("binary_label", 0)) == 1 and bool(row.get("mechanism_eligible"))
        ):
            continue
        sample_id = str(row["sample_id"])
        geometry = geometry_by_sample.get(sample_id, {})
        manifest = manifest_by_sample.get(sample_id, {})
        if source == "c6i":
            control_audit = geometry.get("control_audit", {})
            selected = control_audit.get("selected_candidate", {})
            target_fraction = geometry.get("target_area_fraction")
            location_distance = selected.get("location_distance")
            perimeter_ratio = selected.get("log_perimeter_ratio")
            control_objective = selected.get("objective")
        else:
            control_geometry = geometry.get("control_geometry", {})
            target_geometry = control_geometry.get("target", {})
            control_shape = control_geometry.get("control", {})
            target_fraction = None
            location_distance = _normalized_centroid_distance(
                target_geometry, control_shape
            )
            perimeter_ratio = _log_perimeter_ratio(target_geometry, control_shape)
            control_objective = control_geometry.get("geometry_objective")
        for operator in OPERATORS:
            values = row[operator]
            target_effect = float(values["target_effect"])
            control_effect = float(values["control_effect"])
            flattened.append(
                {
                    "source": source,
                    "sample_id": sample_id,
                    "unit_id": str(row["unit_id"]),
                    "canonical_statement_id": str(row["canonical_statement_id"]),
                    "operator": operator,
                    "target_effect": target_effect,
                    "control_effect": control_effect,
                    "tcig": float(values["tcig"]),
                    "failure_category": classify_effect_pair(
                        target_effect, control_effect
                    ),
                    "box_area_quartile": int(row["box_area_quartile"]),
                    "target_area_pixels": int(row["target_area_pixels"]),
                    "target_area_fraction": _optional_float(target_fraction),
                    "box_count": _optional_int(manifest.get("box_count")),
                    "control_location_distance": _optional_float(location_distance),
                    "control_log_perimeter_ratio": _optional_float(perimeter_ratio),
                    "control_geometry_objective": _optional_float(control_objective),
                    "original_support_probability": _original_support(row, source),
                    "topk_target_coverage": _optional_float(
                        row.get("topk_target_coverage")
                    ),
                    "random_target_coverage": _optional_float(
                        row.get("random_target_coverage")
                    ),
                    "topk_localization_gain": _optional_float(
                        row.get("topk_localization_gain")
                    ),
                }
            )
    return flattened


def summarize_effect_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return effect means, taxonomy counts, and descriptive associations."""

    if not rows:
        raise ValueError("terminal audit rows are empty")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = "|".join(
            [row["source"], row["operator"], row["canonical_statement_id"]]
        )
        groups.setdefault(key, []).append(row)
    return {
        "rows": len(rows),
        "groups": {
            key: {
                "records": len(group),
                "mean_target_effect": float(
                    np.mean([item["target_effect"] for item in group])
                ),
                "mean_control_effect": float(
                    np.mean([item["control_effect"] for item in group])
                ),
                "mean_tcig": float(np.mean([item["tcig"] for item in group])),
                "taxonomy": dict(
                    sorted(Counter(item["failure_category"] for item in group).items())
                ),
                "associations": descriptive_associations(group),
            }
            for key, group in sorted(groups.items())
        },
    }


def descriptive_associations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute descriptive correlations only; no p-values or causal claims."""

    features = (
        "box_area_quartile",
        "target_area_pixels",
        "target_area_fraction",
        "box_count",
        "control_location_distance",
        "control_log_perimeter_ratio",
        "control_geometry_objective",
        "original_support_probability",
        "topk_target_coverage",
        "topk_localization_gain",
    )
    result: dict[str, Any] = {}
    for feature in features:
        pairs = [
            (float(row[feature]), float(row["tcig"]))
            for row in rows
            if row.get(feature) is not None
        ]
        if len(pairs) < 3:
            continue
        x = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
        y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
        if np.all(x == x[0]) or np.all(y == y[0]):
            pearson = None
            spearman = None
        else:
            pearson = float(stats.pearsonr(x, y).statistic)
            spearman = float(stats.spearmanr(x, y).statistic)
        result[feature] = {
            "records": len(pairs),
            "pearson_r": pearson,
            "spearman_rho": spearman,
        }
    return result


def image_perturbation_metrics(
    original: np.ndarray,
    perturbed: np.ndarray,
    intervention_mask: np.ndarray,
    content_mask: np.ndarray,
) -> dict[str, float]:
    """Measure a frozen intervention in image space without model scores."""

    before = np.asarray(original, dtype=np.float64) / 255.0
    after = np.asarray(perturbed, dtype=np.float64) / 255.0
    mask = np.asarray(intervention_mask, dtype=bool)
    content = np.asarray(content_mask, dtype=bool)
    if before.shape != after.shape or before.ndim != 3 or before.shape[2] != 3:
        raise ValueError("image arrays must be shape-matched RGB arrays")
    if mask.shape != before.shape[:2] or content.shape != mask.shape:
        raise ValueError("image-space audit masks must match image geometry")
    if not mask.any() or bool((mask & ~content).any()):
        raise ValueError("intervention mask must be non-empty and within content")
    diff = after - before
    gray_before = _rgb_to_gray(before)
    gray_after = _rgb_to_gray(after)
    edge_before = np.hypot(
        ndimage.sobel(gray_before, axis=0), ndimage.sobel(gray_before, axis=1)
    )
    edge_after = np.hypot(
        ndimage.sobel(gray_after, axis=0), ndimage.sobel(gray_after, axis=1)
    )
    return {
        "mask_area_pixels": int(mask.sum()),
        "mask_area_fraction_of_content": float(mask.sum() / content.sum()),
        "global_l1": float(np.mean(np.abs(diff[content]))),
        "global_rms": float(np.sqrt(np.mean(np.square(diff[content])))),
        "masked_l1": float(np.mean(np.abs(diff[mask]))),
        "masked_rms": float(np.sqrt(np.mean(np.square(diff[mask])))),
        "ssim": float(_windowed_ssim(gray_before, gray_after, content)),
        "masked_edge_energy_change": float(
            np.mean(np.abs(edge_after[mask] - edge_before[mask]))
        ),
        "masked_edge_energy_signed_change": float(
            np.mean(edge_after[mask] - edge_before[mask])
        ),
        "masked_local_contrast_change": float(
            np.std(gray_after[mask]) - np.std(gray_before[mask])
        ),
    }


def summarize_image_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("image-space audit rows are empty")
    metrics = (
        "global_l1",
        "global_rms",
        "masked_l1",
        "masked_rms",
        "ssim",
        "masked_edge_energy_change",
        "masked_edge_energy_signed_change",
        "masked_local_contrast_change",
    )
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = "|".join(
            [row["operator"], row["canonical_statement_id"]]
        )
        groups.setdefault(key, []).append(row)
    output: dict[str, Any] = {}
    for key, group in sorted(groups.items()):
        item: dict[str, Any] = {"records": len(group)}
        for metric in metrics:
            target = np.asarray([row["target"][metric] for row in group])
            control = np.asarray([row["control"][metric] for row in group])
            item[metric] = {
                "target_mean": float(target.mean()),
                "control_mean": float(control.mean()),
                "target_minus_control_mean": float((target - control).mean()),
            }
        output[key] = item
    return {"rows": len(rows), "groups": output}


def _windowed_ssim(before: np.ndarray, after: np.ndarray, content: np.ndarray) -> float:
    sigma = 1.5
    truncate = 3.5
    mu_x = ndimage.gaussian_filter(before, sigma=sigma, truncate=truncate)
    mu_y = ndimage.gaussian_filter(after, sigma=sigma, truncate=truncate)
    sigma_x = ndimage.gaussian_filter(before * before, sigma=sigma, truncate=truncate) - mu_x**2
    sigma_y = ndimage.gaussian_filter(after * after, sigma=sigma, truncate=truncate) - mu_y**2
    sigma_xy = ndimage.gaussian_filter(before * after, sigma=sigma, truncate=truncate) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    values = numerator / np.maximum(denominator, np.finfo(np.float64).tiny)
    return float(np.mean(values[content]))


def _rgb_to_gray(array: np.ndarray) -> np.ndarray:
    return (
        0.2126 * array[:, :, 0]
        + 0.7152 * array[:, :, 1]
        + 0.0722 * array[:, :, 2]
    )


def _normalized_centroid_distance(
    target: dict[str, Any], control: dict[str, Any]
) -> float | None:
    required = ("centroid_x", "centroid_y")
    if not all(key in target and key in control for key in required):
        return None
    return float(
        np.hypot(
            (float(target["centroid_x"]) - float(control["centroid_x"])) / 448.0,
            (float(target["centroid_y"]) - float(control["centroid_y"])) / 448.0,
        )
    )


def _log_perimeter_ratio(
    target: dict[str, Any], control: dict[str, Any]
) -> float | None:
    if "perimeter_edges" not in target or "perimeter_edges" not in control:
        return None
    return float(
        abs(
            np.log(
                float(control["perimeter_edges"])
                / float(target["perimeter_edges"])
            )
        )
    )


def _original_support(row: dict[str, Any], source: str) -> float | None:
    if source == "c6i":
        return _optional_float(row.get("original", {}).get("support_probability"))
    return _optional_float(row.get("b2_support_probability"))


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
