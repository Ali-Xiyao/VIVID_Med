"""Pure CPU contracts for the review-gated D0 versus D1 diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Mapping


REPORT_STATES = ("present", "absent", "uncertain")


@dataclass(frozen=True)
class AgreementWeight:
    """Hard target plus entropy-derived agreement weight for one finding."""

    hard_target: str | None
    weight: float
    entropy: float | None
    observed_sources: int


def entropy_agreement_weight(
    source_states: Mapping[str, str | None],
    *,
    reference_source: str = "chexbert",
) -> AgreementWeight:
    """Compute the fixed D1 weight without changing the hard target.

    Missing sources are excluded from the unweighted mean distribution.
    A missing reference target remains masked with zero weight. When only the
    reference source is observed, the weight is one: missing corroboration is
    not silently treated as disagreement.
    """

    if reference_source not in source_states:
        raise ValueError(f"missing reference source: {reference_source}")

    normalized: dict[str, str | None] = {}
    for source, state in source_states.items():
        if state is not None and state not in REPORT_STATES:
            raise ValueError(f"invalid report state for {source}: {state}")
        normalized[source] = state

    hard_target = normalized[reference_source]
    observed = [state for state in normalized.values() if state is not None]
    if hard_target is None:
        return AgreementWeight(
            hard_target=None,
            weight=0.0,
            entropy=None,
            observed_sources=len(observed),
        )
    if not observed:
        raise AssertionError("reference target is observed but source list is empty")

    probabilities = [
        observed.count(state) / len(observed) for state in REPORT_STATES
    ]
    entropy = -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )
    weight = 1.0 - entropy / math.log(len(REPORT_STATES))
    return AgreementWeight(
        hard_target=hard_target,
        weight=min(max(weight, 0.0), 1.0),
        entropy=entropy,
        observed_sources=len(observed),
    )


def render_hard_ums_target(
    finding_states: Mapping[str, str | None],
    *,
    modality: str = "CXR",
    study_view: str | None = None,
) -> str:
    """Render the deterministic hard-UMS target shared by D0 and D1."""

    if modality != "CXR":
        raise ValueError("the bounded D0/D1 diagnostic is CXR-only")
    findings: dict[str, dict[str, str | None]] = {}
    for finding, state in finding_states.items():
        if state is None:
            continue
        if state not in REPORT_STATES:
            raise ValueError(f"invalid report state for {finding}: {state}")
        findings[finding] = {"state": state, "score": None}

    target = {
        "modality": modality,
        "findings": findings,
        "study_view": study_view,
    }
    return json.dumps(target, ensure_ascii=False)
