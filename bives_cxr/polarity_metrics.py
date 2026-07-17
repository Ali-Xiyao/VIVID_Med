"""Locked development metrics and thresholds for binary polarity runs."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def lock_threshold_at_specificity(
    labels: np.ndarray,
    scores: np.ndarray,
    target_specificity: float,
) -> float:
    if not 0.0 < target_specificity < 1.0:
        raise ValueError("target_specificity must be in (0, 1)")
    negative_scores = np.asarray(scores, dtype=np.float64)[np.asarray(labels) == 0]
    if negative_scores.size == 0:
        raise ValueError("threshold locking requires negative development examples")
    quantile = float(
        np.quantile(negative_scores, target_specificity, method="higher")
    )
    # Evaluation predicts support for score >= threshold, so move one floating
    # point above the selected negative score to make the specificity contract
    # inclusive and deterministic under ties.
    return float(np.nextafter(quantile, np.inf))


def polarity_metrics(
    rows: Iterable[dict[str, Any]],
    *,
    thresholds: dict[str, float] | None = None,
    target_specificity: float = 0.9,
) -> tuple[dict[str, Any], dict[str, float]]:
    materialized = list(rows)
    findings = sorted({str(row["canonical_statement_id"]) for row in materialized})
    per_finding: dict[str, Any] = {}
    locked = dict(thresholds or {})
    for finding in findings:
        subset = [row for row in materialized if row["canonical_statement_id"] == finding]
        labels = np.asarray([int(row["binary_label"]) for row in subset], dtype=np.int64)
        scores = np.asarray([float(row["support_probability"]) for row in subset])
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError(f"{finding} metrics require both labels")
        threshold = locked.get(finding)
        if threshold is None:
            threshold = lock_threshold_at_specificity(labels, scores, target_specificity)
            locked[finding] = threshold
        predictions = scores >= threshold
        positives = labels == 1
        negatives = labels == 0
        clipped = np.clip(scores, 1e-7, 1.0 - 1e-7)
        per_finding[finding] = {
            "records": len(subset),
            "prevalence": float(labels.mean()),
            "auroc": float(roc_auc_score(labels, scores)),
            "auprc": float(average_precision_score(labels, scores)),
            "nll": float(log_loss(labels, clipped, labels=[0, 1])),
            "brier": float(np.mean((clipped - labels) ** 2)),
            "support_probability_threshold": float(threshold),
            "target_specificity": float(target_specificity),
            "sensitivity_at_locked_threshold": float(predictions[positives].mean()),
            "specificity_at_locked_threshold": float((~predictions[negatives]).mean()),
        }
    macro = {
        key: float(np.mean([per_finding[finding][key] for finding in findings]))
        for key in ("auroc", "auprc", "nll", "brier")
    }
    return {"per_finding": per_finding, "macro": macro}, locked
