from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REQUIRED_FIELDS = {
    "unit_id",
    "canonical_statement_id",
    "dilation_fraction",
    "original_score",
    "keep_score",
    "target_deletion_effect",
    "control_deletion_effect",
    "tcig",
    "target_area_pixels",
    "control_area_pixels",
    "topk_localization_gain",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_array(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite {key}")
    return values


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    if replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {"lower": float(lower), "upper": float(upper)}


def _slice_summary(
    tcig: np.ndarray,
    mask: np.ndarray,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    values = tcig[mask]
    if len(values) == 0:
        raise ValueError("empty diagnostic slice")
    return {
        "n": int(len(values)),
        "mean_tcig": float(values.mean()),
        "positive_fraction": float((values > 0).mean()),
        "bootstrap_mean_95ci": _bootstrap_mean_ci(values, replicates=replicates, rng=rng),
    }


def _standardized_ols(
    tcig: np.ndarray,
    *,
    localization_gain: np.ndarray,
    target_area: np.ndarray,
    original_score: np.ndarray,
) -> dict[str, Any]:
    raw = np.column_stack([localization_gain, target_area, original_score])
    std = raw.std(axis=0)
    if bool((std == 0).any()):
        return {"available": False, "reason": "constant predictor"}
    design = (raw - raw.mean(axis=0)) / std
    design = np.column_stack([np.ones(len(design)), design])
    coefficients = np.linalg.lstsq(design, tcig, rcond=None)[0]
    prediction = design @ coefficients
    denominator = float(((tcig - tcig.mean()) ** 2).sum())
    r_squared = None if denominator == 0 else 1.0 - float(((tcig - prediction) ** 2).sum()) / denominator
    return {
        "available": True,
        "intercept": float(coefficients[0]),
        "standardized_coefficients": {
            "topk_localization_gain": float(coefficients[1]),
            "target_area_pixels": float(coefficients[2]),
            "original_score": float(coefficients[3]),
        },
        "r_squared": r_squared,
    }


def _group_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    target = _as_array(rows, "target_deletion_effect")
    control = _as_array(rows, "control_deletion_effect")
    tcig = _as_array(rows, "tcig")
    target_area = _as_array(rows, "target_area_pixels")
    original = _as_array(rows, "original_score")
    keep = _as_array(rows, "keep_score")
    localization = _as_array(rows, "topk_localization_gain")

    ordered = np.sort(tcig)
    trim_count = int(len(ordered) * 0.1)
    trimmed = ordered[trim_count : len(ordered) - trim_count] if trim_count else ordered
    leave_one_out = (tcig.sum() - tcig) / (len(tcig) - 1) if len(tcig) > 1 else tcig.copy()

    localization_q25, localization_q75 = np.quantile(localization, [0.25, 0.75])
    area_q25, area_q75 = np.quantile(target_area, [0.25, 0.75])
    localization_slices = {
        "low": _slice_summary(
            tcig,
            localization <= localization_q25,
            replicates=bootstrap_replicates,
            rng=rng,
        ),
        "high": _slice_summary(
            tcig,
            localization >= localization_q75,
            replicates=bootstrap_replicates,
            rng=rng,
        ),
    }
    area_slices = {
        "low": _slice_summary(
            tcig, target_area <= area_q25, replicates=bootstrap_replicates, rng=rng
        ),
        "high": _slice_summary(
            tcig, target_area >= area_q75, replicates=bootstrap_replicates, rng=rng
        ),
    }
    control_area_correlation = _pearson(control, target_area)
    target_area_correlation = _pearson(target, target_area)
    return {
        "n": len(rows),
        "means": {
            "target_deletion_effect": float(target.mean()),
            "control_deletion_effect": float(control.mean()),
            "tcig": float(tcig.mean()),
            "original_score": float(original.mean()),
            "keep_score": float(keep.mean()),
            "topk_localization_gain": float(localization.mean()),
        },
        "medians": {
            "target_deletion_effect": float(np.median(target)),
            "control_deletion_effect": float(np.median(control)),
            "tcig": float(np.median(tcig)),
        },
        "fractions": {
            "target_effect_positive": float((target > 0).mean()),
            "control_effect_positive": float((control > 0).mean()),
            "tcig_positive": float((tcig > 0).mean()),
        },
        "tcig_quantiles": {
            key: float(value)
            for key, value in zip(
                ["min", "p10", "p25", "median", "p75", "p90", "max"],
                np.quantile(tcig, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]),
            )
        },
        "robustness": {
            "trimmed_mean_10pct": float(trimmed.mean()),
            "leave_one_out_mean_min": float(leave_one_out.min()),
            "leave_one_out_mean_max": float(leave_one_out.max()),
            "all_leave_one_out_means_negative": bool((leave_one_out < 0).all()),
        },
        "correlations": {
            "tcig_vs_target_area": _pearson(tcig, target_area),
            "tcig_vs_original_score": _pearson(tcig, original),
            "tcig_vs_localization_gain": _pearson(tcig, localization),
            "tcig_vs_original_minus_keep": _pearson(tcig, original - keep),
            "control_effect_vs_target_area": control_area_correlation,
            "target_effect_vs_target_area": target_area_correlation,
        },
        "quartile_slices": {
            "topk_localization_gain": localization_slices,
            "target_area_pixels": area_slices,
        },
        "standardized_ols": _standardized_ols(
            tcig,
            localization_gain=localization,
            target_area=target_area,
            original_score=original,
        ),
        "diagnosis_flags": {
            "negative_after_10pct_trim": bool(trimmed.mean() < 0),
            "high_localization_slice_better_than_low": bool(
                localization_slices["high"]["mean_tcig"]
                > localization_slices["low"]["mean_tcig"]
            ),
            "control_more_area_sensitive_than_target": bool(
                control_area_correlation is not None
                and target_area_correlation is not None
                and control_area_correlation > target_area_correlation
            ),
        },
    }


def summarize_intervention_failures(
    rows: Iterable[Mapping[str, Any]],
    *,
    primary_dilation: float = 0.0,
    bootstrap_replicates: int = 20_000,
    bootstrap_seed: int = 17,
) -> dict[str, Any]:
    normalized = [dict(row) for row in rows]
    if not normalized:
        raise ValueError("intervention rows are empty")
    seen: set[tuple[str, str, float]] = set()
    for index, row in enumerate(normalized):
        missing = sorted(REQUIRED_FIELDS - set(row))
        if missing:
            raise ValueError(f"row {index} missing fields: {missing}")
        key = (
            str(row["unit_id"]),
            str(row["canonical_statement_id"]),
            float(row["dilation_fraction"]),
        )
        if key in seen:
            raise ValueError(f"duplicate intervention task: {key}")
        seen.add(key)
        if int(row["target_area_pixels"]) != int(row["control_area_pixels"]):
            raise ValueError(f"target/control area mismatch: {key}")

    rng = np.random.default_rng(bootstrap_seed)
    dilations = sorted({float(row["dilation_fraction"]) for row in normalized})
    findings = sorted({str(row["canonical_statement_id"]) for row in normalized})
    if primary_dilation not in dilations:
        raise ValueError(f"primary dilation {primary_dilation} is absent")

    by_dilation: dict[str, Any] = {}
    for dilation in dilations:
        dilation_rows = [
            row for row in normalized if float(row["dilation_fraction"]) == dilation
        ]
        by_dilation[f"{dilation:g}"] = {
            finding: _group_summary(
                [
                    row
                    for row in dilation_rows
                    if str(row["canonical_statement_id"]) == finding
                ],
                bootstrap_replicates=bootstrap_replicates,
                rng=rng,
            )
            for finding in findings
        }

    paired: defaultdict[tuple[str, str], dict[float, Mapping[str, Any]]] = defaultdict(dict)
    for row in normalized:
        paired[(str(row["unit_id"]), str(row["canonical_statement_id"]))][
            float(row["dilation_fraction"])
        ] = row
    dilation_stability: dict[str, Any] = {}
    if len(dilations) == 2:
        first, second = dilations
        for finding in findings:
            pairs = [
                values
                for (unit_id, pair_finding), values in paired.items()
                if pair_finding == finding and first in values and second in values
            ]
            first_tcig = np.asarray([pair[first]["tcig"] for pair in pairs], dtype=np.float64)
            second_tcig = np.asarray([pair[second]["tcig"] for pair in pairs], dtype=np.float64)
            dilation_stability[finding] = {
                "n": len(pairs),
                "tcig_correlation": _pearson(first_tcig, second_tcig),
                "mean_tcig_change": float((second_tcig - first_tcig).mean()),
                "sign_agreement_fraction": float(
                    (np.sign(first_tcig) == np.sign(second_tcig)).mean()
                ),
            }

    return {
        "schema_version": "bives_vindr_intervention_failure_taxonomy_v1",
        "formal_result": False,
        "evaluation_only": True,
        "gate_override": False,
        "row_count": len(normalized),
        "findings": findings,
        "dilations": dilations,
        "primary_dilation": primary_dilation,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "results_by_dilation": by_dilation,
        "dilation_stability": dilation_stability,
    }


def render_markdown(summary: Mapping[str, Any], *, rows_sha256: str) -> str:
    primary = summary["results_by_dilation"][f"{summary['primary_dilation']:g}"]
    lines = [
        "# BiVES-CXR Post-Stop Intervention Failure Taxonomy",
        "",
        "This is a read-only diagnostic over the frozen seed-17 rows. It does not",
        "override the failed E8 gate or authorize new seeds, data, method changes,",
        "Qwen3.5-4B, or Qwen3.5-9B.",
        "",
        f"- Rows: `{summary['row_count']}`",
        f"- Rows SHA-256: `{rows_sha256}`",
        f"- Primary dilation: `{summary['primary_dilation']}`",
        "",
        "| Finding | N | Mean target | Mean control | Mean TCIG | 10% trimmed TCIG | TCIG > 0 | Control-area r | Target-area r |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for finding, result in primary.items():
        means = result["means"]
        robust = result["robustness"]
        correlations = result["correlations"]
        lines.append(
            f"| {finding} | {result['n']} | {means['target_deletion_effect']:.4f} | "
            f"{means['control_deletion_effect']:.4f} | {means['tcig']:.4f} | "
            f"{robust['trimmed_mean_10pct']:.4f} | {result['fractions']['tcig_positive']:.3f} | "
            f"{correlations['control_effect_vs_target_area']:.3f} | "
            f"{correlations['target_effect_vs_target_area']:.3f} |"
        )
    lines.extend(["", "## Quartile diagnosis", ""])
    for finding, result in primary.items():
        localization = result["quartile_slices"]["topk_localization_gain"]
        area = result["quartile_slices"]["target_area_pixels"]
        lines.extend(
            [
                f"### {finding}",
                "",
                f"- Low/high localization-gain quartile mean TCIG: "
                f"`{localization['low']['mean_tcig']:.4f}` / "
                f"`{localization['high']['mean_tcig']:.4f}`.",
                f"- Low/high target-area quartile mean TCIG: "
                f"`{area['low']['mean_tcig']:.4f}` / `{area['high']['mean_tcig']:.4f}`.",
                f"- Leave-one-out mean range: "
                f"`[{result['robustness']['leave_one_out_mean_min']:.4f}, "
                f"{result['robustness']['leave_one_out_mean_max']:.4f}]`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "The diagnostic separates outlier, localization, and intervention-area",
            "patterns. It is descriptive and post-stop; it is not a new model-selection",
            "or protocol-tuning surface.",
            "",
        ]
    )
    return "\n".join(lines)
