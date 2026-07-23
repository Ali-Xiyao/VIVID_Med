"""Reliability-calibrated structured report posterior primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


REPORT_STATES = ("present", "absent", "uncertain")


@dataclass(frozen=True)
class PosteriorResult:
    probabilities: np.ndarray
    reliability: float
    entropy: float
    source_count: int


def _as_probability_vector(values: Iterable[float]) -> np.ndarray:
    vector = np.asarray(tuple(values), dtype=np.float64)
    if vector.shape != (len(REPORT_STATES),):
        raise ValueError(f"expected {len(REPORT_STATES)} probabilities")
    if not np.all(np.isfinite(vector)) or np.any(vector < 0):
        raise ValueError("probabilities must be finite and non-negative")
    total = float(vector.sum())
    if total <= 0:
        raise ValueError("probabilities must have positive mass")
    return vector / total


def fuse_log_opinion_pool(
    source_probabilities: Iterable[Iterable[float] | None],
    source_weights: Iterable[float],
    *,
    temperature: float = 1.0,
    epsilon: float = 1e-8,
) -> PosteriorResult | None:
    """Fuse observed sources while leaving all-missing fields missing.

    `None` means a source did not observe the field. It is not converted to an
    absent label. Weights are renormalized over observed sources only.
    """

    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    probabilities = list(source_probabilities)
    weights = np.asarray(tuple(source_weights), dtype=np.float64)
    if len(probabilities) != len(weights):
        raise ValueError("source probabilities and weights must have equal length")
    if np.any(weights < 0) or not np.all(np.isfinite(weights)):
        raise ValueError("source weights must be finite and non-negative")

    observed: list[np.ndarray] = []
    observed_weights: list[float] = []
    for probability, weight in zip(probabilities, weights):
        if probability is None or weight == 0:
            continue
        observed.append(_as_probability_vector(probability))
        observed_weights.append(float(weight))
    if not observed:
        return None

    normalized_weights = np.asarray(observed_weights, dtype=np.float64)
    normalized_weights /= normalized_weights.sum()
    matrix = np.stack(observed, axis=0)
    logits = (normalized_weights[:, None] * np.log(matrix + epsilon)).sum(axis=0)
    logits /= temperature
    logits -= logits.max()
    posterior = np.exp(logits)
    posterior /= posterior.sum()

    entropy = float(-(posterior * np.log(posterior + epsilon)).sum())
    reliability = 1.0 - entropy / math.log(len(REPORT_STATES))
    return PosteriorResult(
        probabilities=posterior.astype(np.float32),
        reliability=float(np.clip(reliability, 0.0, 1.0)),
        entropy=entropy,
        source_count=len(observed),
    )
