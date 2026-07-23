"""Validate the deidentified RCSD component-attribution decision."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _close(actual: float, expected: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)


def validate(payload: dict[str, object]) -> list[str]:
    errors: list[str] = []
    if payload.get("decision") != "FULL_RCSD_NO_GO_COMPONENT_AUDIT_OPEN":
        errors.append("unexpected top-level decision")
    if payload.get("test_sets_opened") != []:
        errors.append("test_sets_opened must remain empty")

    g2 = payload["g2"]
    g2_baseline = g2["baseline"]
    g2_candidate = g2["candidate"]
    g2_f1_gain = g2_candidate["macro_f1"] - g2_baseline["macro_f1"]
    g2_nll_change = g2_candidate["nll"] / g2_baseline["nll"] - 1.0
    g2_checks = {
        "macro_f1": g2_f1_gain
        >= g2["thresholds"]["macro_f1_gain_at_least"],
        "nll": g2_nll_change
        <= g2["thresholds"]["nll_relative_change_at_most"],
    }
    if g2["gate_pass"] != all(g2_checks.values()):
        errors.append("G2 gate_pass does not match recomputed checks")
    if sorted(g2["failed_checks"]) != sorted(
        key for key, passed in g2_checks.items() if not passed
    ):
        errors.append("G2 failed_checks do not match recomputed checks")

    g3 = payload["g3"]
    g3_baseline = g3["baseline"]
    g3_candidate = g3["candidate"]
    g3_nll_change = g3_candidate["nll"] / g3_baseline["nll"] - 1.0
    g3_f1_gain = g3_candidate["macro_f1"] - g3_baseline["macro_f1"]
    g3_ece_change = g3_candidate["ece"] - g3_baseline["ece"]
    g3_checks = {
        "nll": g3_nll_change
        <= g3["thresholds"]["nll_relative_change_at_most"],
        "macro_f1": g3_f1_gain
        >= g3["thresholds"]["macro_f1_gain_at_least"],
        "ece": g3_ece_change <= g3["thresholds"]["ece_change_at_most"],
    }
    if g3["gate_pass"] != all(g3_checks.values()):
        errors.append("G3 gate_pass does not match recomputed checks")
    expected_failed = sorted(
        key for key, passed in g3_checks.items() if not passed
    )
    if sorted(g3["failed_checks"]) != expected_failed:
        errors.append("G3 failed_checks do not match recomputed checks")

    if not _close(g2_nll_change, 0.06981170861255892):
        errors.append("G2 NLL delta drifted")
    if not _close(g3_nll_change, -0.00046054678742579735):
        errors.append("G3 NLL delta drifted")
    if not _close(g3_f1_gain, 0.0006478601223307567):
        errors.append("G3 macro-F1 delta drifted")

    arms = payload["arms"]
    if arms["D1"]["status"] != "UNTESTED":
        errors.append("D1 must remain untested until separately authorized")
    if arms["D4"]["status"] != "PROHIBITED":
        errors.append("D4 must remain prohibited")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--status",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "audit"
            / "rcsd_component_status.json"
        ),
    )
    args = parser.parse_args()
    payload = json.loads(args.status.read_text(encoding="utf-8"))
    errors = validate(payload)
    result = {
        "artifact": "rcsd_component_attribution_integrity",
        "audit_completed": True,
        "pass": not errors,
        "errors": errors,
        "scientific_decision": payload["decision"],
    }
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
