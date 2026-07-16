"""Case-Driven Curriculum Scheduler (CDCS)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CaseDrivenCurriculumScheduler:
    """Update sampling weights from failure-type and field statistics."""

    failure_type_weights: dict[str, float] = field(default_factory=lambda: {
        "laterality": 1.4,
        "false_hard_negative": 1.5,
        "label_mapping_or_noise": 1.2,
        "domain_shift": 1.3,
        "leakage": 0.6,
    })
    field_floor: float = 0.8
    field_boost: float = 1.25

    def summarize_failures(self, cases: list[dict[str, Any]]) -> dict[str, Counter]:
        failure_counts: Counter = Counter()
        field_counts: Counter = Counter()
        for case in cases:
            failure_counts[str(case.get("failure_type") or case.get("likely_cause") or "unknown")] += 1
            if case.get("finding") or case.get("nih_label"):
                field_counts[str(case.get("finding") or case.get("nih_label"))] += 1
        return {"failure_type": failure_counts, "field": field_counts}

    def sample_weight(self, sample: dict[str, Any], failure_stats: dict[str, Counter]) -> float:
        weight = 1.0
        failure_type = str(sample.get("failure_type") or sample.get("case_failure_type") or "")
        for key, value in self.failure_type_weights.items():
            if key in failure_type:
                weight *= value
        field = str(sample.get("finding") or sample.get("nih_label") or "")
        if field:
            field_counts = failure_stats.get("field", Counter())
            if field_counts and field_counts[field] >= max(field_counts.values()):
                weight *= self.field_boost
            else:
                weight *= self.field_floor
        return float(weight)

    def annotate(self, samples: list[dict[str, Any]], cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        stats = self.summarize_failures(cases)
        output = []
        for sample in samples:
            row = dict(sample)
            row["cdcs_sampling_weight"] = self.sample_weight(sample, stats)
            output.append(row)
        return output

