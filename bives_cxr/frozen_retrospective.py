"""Aggregate frozen VinDr C5 and MS-CXR C6I evidence without new scoring.

The legacy stages predate the active ``X/C_X/E/C_E`` audit schema.  This
module therefore emits a deliberately separate retrospective format.  It
never loads a model, decodes an image, regenerates a mask, or treats the
result as a new validation experiment.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Iterable

import numpy as np
from scipy import stats

from bives_cxr.terminal_audit import OPERATORS, classify_effect_pair


FORMAT_VERSION = "frozen_existing_data_retrospective_v1"

SOURCE_METADATA = {
    "c5": {
        "dataset": "VinDr-CXR",
        "stage": "C5",
        "dataset_role": "supplemental_prior_exposed",
        "claim_unit": "image",
        "patient_level_claim": False,
        "independent_primary_confirmation": False,
    },
    "c6i": {
        "dataset": "MS-CXR",
        "stage": "C6I",
        "dataset_role": "frozen_external_sensitivity",
        "claim_unit": "patient",
        "patient_level_claim": True,
        "independent_primary_confirmation": False,
    },
}


def normalize_frozen_rows(
    rows: Iterable[dict[str, Any]], *, source: str
) -> list[dict[str, Any]]:
    """Normalize legacy target/control rows while retaining their limitations."""

    if source not in SOURCE_METADATA:
        raise ValueError(f"unsupported frozen source: {source}")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if source == "c5" and not (
            int(row.get("binary_label", 0)) == 1
            and bool(row.get("mechanism_eligible"))
        ):
            continue
        sample_id = str(row["sample_id"])
        unit_id = str(row["unit_id"])
        finding = str(row["canonical_statement_id"])
        localization_gain = _localization_gain(row)
        for operator in OPERATORS:
            key = (sample_id, operator)
            if key in seen:
                raise ValueError(f"duplicate frozen sample/operator: {key}")
            seen.add(key)
            values = row.get(operator)
            if not isinstance(values, dict):
                raise ValueError(f"missing frozen operator {operator}: {sample_id}")
            target_effect = float(values["target_effect"])
            control_effect = float(values["control_effect"])
            tcig = float(values["tcig"])
            if not np.isclose(
                target_effect - control_effect, tcig, rtol=0.0, atol=1e-12
            ):
                raise ValueError(f"TCIG arithmetic mismatch: {sample_id}|{operator}")
            normalized.append(
                {
                    "source": source,
                    "sample_id": sample_id,
                    "unit_id": unit_id,
                    "canonical_statement_id": finding,
                    "operator": operator,
                    "localization_gain": localization_gain,
                    "target_effect": target_effect,
                    "control_effect": control_effect,
                    "tcig": tcig,
                    "effect_taxonomy": classify_effect_pair(
                        target_effect, control_effect
                    ),
                }
            )
    if not normalized:
        raise ValueError(f"no eligible frozen rows for {source}")
    return normalized


def aggregate_frozen_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create identifier-free descriptive cells for the frozen evidence matrix."""

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["source"]),
            str(row["operator"]),
            str(row["canonical_statement_id"]),
        )
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (source, operator, finding), group in sorted(groups.items()):
        metadata = SOURCE_METADATA[source]
        localization = np.asarray(
            [float(item["localization_gain"]) for item in group], dtype=np.float64
        )
        tcig = np.asarray([float(item["tcig"]) for item in group], dtype=np.float64)
        target = np.asarray(
            [float(item["target_effect"]) for item in group], dtype=np.float64
        )
        control = np.asarray(
            [float(item["control_effect"]) for item in group], dtype=np.float64
        )
        pearson, spearman = _descriptive_correlations(localization, tcig)
        output.append(
            {
                **metadata,
                "source": source,
                "operator": operator,
                "canonical_statement_id": finding,
                "records": len(group),
                "unique_units": len({str(item["unit_id"]) for item in group}),
                "mean_localization_gain": float(localization.mean()),
                "mean_target_effect": float(target.mean()),
                "mean_control_effect": float(control.mean()),
                "mean_tcig": float(tcig.mean()),
                "positive_tcig_fraction": float(np.mean(tcig > 0.0)),
                "localization_tcig_pearson_r": pearson,
                "localization_tcig_spearman_rho": spearman,
                "effect_taxonomy": dict(
                    sorted(Counter(item["effect_taxonomy"] for item in group).items())
                ),
            }
        )
    if not output:
        raise ValueError("frozen retrospective aggregation is empty")
    return output


def build_retrospective_summary(
    aggregate_rows: list[dict[str, Any]],
    *,
    source_sha256: dict[str, str],
    source_record_counts: dict[str, int],
) -> dict[str, Any]:
    """Build a deterministic, aggregate-only retrospective result."""

    if set(source_record_counts) != {"c5", "c6i"}:
        raise ValueError("frozen source record counts must cover c5 and c6i")
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "complete_frozen_descriptive_no_new_score",
        "formal_result": False,
        "model_loaded": False,
        "gpu_used": False,
        "scores_computed": False,
        "frozen_scores_reused": True,
        "test_opened": False,
        "training_performed": False,
        "new_experiment_stage_created": False,
        "source_sha256": dict(sorted(source_sha256.items())),
        "source_record_counts": dict(sorted(source_record_counts.items())),
        "aggregate_rows": aggregate_rows,
        "schema_boundary": {
            "compatible_with_primary_schema": False,
            "missing_roles": ["separate_explanation_region_E", "matched_control_C_E"],
            "reason": (
                "Frozen C5/C6I contain one legacy target/control contrast and cannot "
                "be represented as the active X/C_X/E/C_E primary audit without "
                "inventing unavailable interventions."
            ),
        },
        "interpretation_boundary": (
            "Descriptive reuse of immutable prior scores only. VinDr is supplemental "
            "and image-level; MS-CXR is a 29-patient positive-only frozen sensitivity "
            "set. Neither is CheXlocalize development or primary confirmation."
        ),
    }
    payload["canonical_artifact_sha256"] = canonical_sha256(payload)
    return payload


def canonical_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _localization_gain(row: dict[str, Any]) -> float:
    topk = float(row["topk_target_coverage"])
    random = float(row["random_target_coverage"])
    derived = topk - random
    recorded = row.get("topk_localization_gain")
    if recorded is not None and not np.isclose(
        float(recorded), derived, rtol=0.0, atol=1e-12
    ):
        raise ValueError("recorded localization gain does not match coverage contrast")
    return float(derived)


def _descriptive_correlations(
    x: np.ndarray, y: np.ndarray
) -> tuple[float | None, float | None]:
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return None, None
    return (
        float(stats.pearsonr(x, y).statistic),
        float(stats.spearmanr(x, y).statistic),
    )
