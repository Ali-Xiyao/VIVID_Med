"""Apply the three frozen VIVID-GDS Stage-A comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ARMS = ("A0_direct", "A1_freetext", "A2_ums", "A3_gds")


def comparison(
    left: dict[str, object],
    right: dict[str, object],
    thresholds: dict[str, object],
    findings: list[str],
) -> dict[str, object]:
    left_metrics = left["expert_development"]
    right_metrics = right["expert_development"]
    auroc = (
        float(left_metrics["macro_auroc"])
        - float(right_metrics["macro_auroc"])
    )
    auprc = (
        float(left_metrics["macro_auprc"])
        - float(right_metrics["macro_auprc"])
    )
    per_finding = {
        finding: (
            float(left_metrics["per_finding"][finding]["auroc"])
            - float(right_metrics["per_finding"][finding]["auroc"])
        )
        for finding in findings
    }
    nonnegative = sum(value >= 0 for value in per_finding.values())
    below = sum(value < -0.02 for value in per_finding.values())
    checks = {
        "macro_auroc": auroc
        >= float(thresholds["macro_auroc_at_least"]),
        "macro_auprc": auprc
        >= float(thresholds["macro_auprc_at_least"]),
    }
    if "nonnegative_findings_at_least" in thresholds:
        checks["nonnegative_findings"] = nonnegative >= int(
            thresholds["nonnegative_findings_at_least"]
        )
        checks["large_declines"] = below <= int(
            thresholds["findings_below_minus_0_02_at_most"]
        )
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "delta_macro_auroc": auroc,
        "delta_macro_auprc": auprc,
        "delta_per_finding_auroc": per_finding,
        "nonnegative_findings": nonnegative,
        "findings_below_minus_0_02": below,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for arm in ARMS:
        parser.add_argument(f"--{arm.replace('_', '-')}", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    payloads = {
        arm: json.loads(
            getattr(args, arm).read_text(encoding="utf-8")
        )
        for arm in ARMS
    }
    findings = lock["promotion"]["primary_findings"]
    results = {
        "A2_minus_A1": comparison(
            payloads["A2_ums"],
            payloads["A1_freetext"],
            lock["promotion"]["A2_minus_A1"],
            findings,
        ),
        "A2_minus_A0": comparison(
            payloads["A2_ums"],
            payloads["A0_direct"],
            lock["promotion"]["A2_minus_A0"],
            findings,
        ),
        "A3_minus_A2": comparison(
            payloads["A3_gds"],
            payloads["A2_ums"],
            lock["promotion"]["A3_minus_A2"],
            findings,
        ),
    }
    passed = all(row["pass"] for row in results.values())
    result = {
        "schema_version": 1,
        "artifact": "vivid_gds_stage_a_verdict",
        "pass": passed,
        "verdict": "STAGE_A_PASS" if passed else "STAGE_A_NO_GO",
        "comparisons": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result))
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
