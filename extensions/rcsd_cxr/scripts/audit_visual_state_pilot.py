"""Compare the two frozen 20k pilot summaries and apply the survival gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare(spd: dict, anchored: dict) -> dict[str, object]:
    baseline = spd["best_validation"]
    method = anchored["best_validation"]
    nll_relative_change = method["nll"] / baseline["nll"] - 1.0
    macro_f1_gain = method["macro_f1"] - baseline["macro_f1"]
    ece_change = method["ece"] - baseline["ece"]
    per_finding_delta = {
        finding: (
            method["per_finding"][finding]["macro_f1"]
            - baseline["per_finding"][finding]["macro_f1"]
        )
        for finding in baseline["per_finding"]
    }
    large_drops = [
        finding for finding, delta in per_finding_delta.items() if delta < -0.02
    ]
    checks = {
        "nll": nll_relative_change <= -0.03,
        "macro_f1": macro_f1_gain >= 0.005,
        "ece": ece_change <= 0.01,
        "large_finding_drops": len(large_drops) <= 2,
        "equal_trainable_counts": spd["trainable_counts"]
        == anchored["trainable_counts"],
        "same_manifest": spd["hashes"]["manifest"]
        == anchored["hashes"]["manifest"],
        "same_backbone": spd["hashes"]["backbone_weights"]
        == anchored["hashes"]["backbone_weights"],
        "same_teacher": spd["hashes"]["prototypes"]
        == anchored["hashes"]["prototypes"],
    }
    return {
        "thresholds": {
            "nll_relative_change_at_most": -0.03,
            "macro_f1_gain_at_least": 0.005,
            "ece_change_at_most": 0.01,
            "findings_below_minus_0.02_at_most": 2,
        },
        "nll_relative_change": nll_relative_change,
        "macro_f1_gain": macro_f1_gain,
        "ece_change": ece_change,
        "per_finding_delta": per_finding_delta,
        "large_finding_drops": large_drops,
        "checks": checks,
        "g3_pass": all(checks.values()),
        "failure_action": (
            None
            if all(checks.values())
            else "stop field-anchor scale-up; retain as negative pilot"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spd", required=True, type=Path)
    parser.add_argument("--field-anchor", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    spd = json.loads(args.spd.read_text(encoding="utf-8"))
    anchored = json.loads(args.field_anchor.read_text(encoding="utf-8"))
    comparison = compare(spd, anchored)
    result = {
        "schema_version": 1,
        "artifact": "visual_state_20k_pilot_gate",
        "audit_completed": True,
        "pass": comparison["g3_pass"],
        "comparison": comparison,
        "hashes": {
            "spd_summary": sha256_file(args.spd),
            "field_anchor_summary": sha256_file(args.field_anchor),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
