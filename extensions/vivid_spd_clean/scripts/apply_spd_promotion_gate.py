"""Apply the frozen S3 prefix4-versus-SPD promotion gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-summary", required=True, type=Path)
    parser.add_argument("--spd-summary", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prefix = json.loads(args.prefix_summary.read_text(encoding="utf-8"))
    spd = json.loads(args.spd_summary.read_text(encoding="utf-8"))
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    p_metrics = prefix["expert_development"]
    s_metrics = spd["expert_development"]
    findings = lock["promotion"]["primary_findings"]
    per_finding = {}
    nonnegative = 0
    large_declines = 0
    for finding in findings:
        delta = (
            s_metrics["per_finding"][finding]["auroc"]
            - p_metrics["per_finding"][finding]["auroc"]
        )
        per_finding[finding] = {"auroc_delta": delta}
        nonnegative += int(delta >= 0.0)
        large_declines += int(
            delta < lock["promotion"]["large_decline_threshold"]
        )
    auroc_delta = s_metrics["macro_auroc"] - p_metrics["macro_auroc"]
    auprc_delta = s_metrics["macro_auprc"] - p_metrics["macro_auprc"]
    checks = {
        "macro_auroc": (
            auroc_delta >= lock["promotion"]["macro_auroc_delta_min"]
        ),
        "macro_auprc": (
            auprc_delta >= lock["promotion"]["macro_auprc_delta_min"]
        ),
        "nonnegative_findings": (
            nonnegative
            >= lock["promotion"]["nonnegative_finding_count_min"]
        ),
        "large_declines": (
            large_declines
            <= lock["promotion"]["large_decline_count_max"]
        ),
    }
    passed = all(checks.values())
    result = {
        "schema_version": 1,
        "artifact": "strict_vivid_spd_s3_verdict",
        "pass": passed,
        "verdict": (
            "STRICT_PASS"
            if passed
            else "STRICT_NO_GO_DIAGNOSTIC_OPEN"
        ),
        "deltas": {
            "macro_auroc": auroc_delta,
            "macro_auprc": auprc_delta,
            "nonnegative_findings": nonnegative,
            "large_declines": large_declines,
            "per_finding": per_finding,
        },
        "checks": checks,
        "thresholds": lock["promotion"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
