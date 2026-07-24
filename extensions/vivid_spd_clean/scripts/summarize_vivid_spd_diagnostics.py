"""Freeze the bounded VIVID/SPD diagnostic verdict."""

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


def gate(candidate: dict, baseline: dict, promotion: dict) -> dict:
    candidate_metrics = candidate["expert_development"]
    baseline_metrics = baseline["expert_development"]
    per_finding: dict[str, dict[str, float]] = {}
    nonnegative = 0
    large_declines = 0
    for finding in promotion["primary_findings"]:
        delta = (
            candidate_metrics["per_finding"][finding]["auroc"]
            - baseline_metrics["per_finding"][finding]["auroc"]
        )
        per_finding[finding] = {"auroc_delta": delta}
        nonnegative += int(delta >= 0.0)
        large_declines += int(
            delta < promotion["large_decline_threshold"]
        )
    auroc_delta = (
        candidate_metrics["macro_auroc"] - baseline_metrics["macro_auroc"]
    )
    auprc_delta = (
        candidate_metrics["macro_auprc"] - baseline_metrics["macro_auprc"]
    )
    checks = {
        "macro_auroc": auroc_delta >= promotion["macro_auroc_delta_min"],
        "macro_auprc": auprc_delta >= promotion["macro_auprc_delta_min"],
        "nonnegative_findings": (
            nonnegative >= promotion["nonnegative_finding_count_min"]
        ),
        "large_declines": (
            large_declines <= promotion["large_decline_count_max"]
        ),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "deltas": {
            "macro_auroc": auroc_delta,
            "macro_auprc": auprc_delta,
            "nonnegative_findings": nonnegative,
            "large_declines": large_declines,
            "per_finding": per_finding,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-root", required=True, type=Path)
    parser.add_argument("--diagnostic-root", required=True, type=Path)
    parser.add_argument("--strict-lock", required=True, type=Path)
    parser.add_argument("--diagnostic-lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    strict_lock = json.loads(args.strict_lock.read_text(encoding="utf-8"))
    diagnostic_lock = json.loads(
        args.diagnostic_lock.read_text(encoding="utf-8")
    )
    authority = diagnostic_lock["strict_authority"]
    authority_paths = {
        "s3_verdict_sha256": args.strict_root / "s3_verdict.json",
        "prefix_s3_summary_sha256": (
            args.strict_root / "s3" / "ums_prefix4" / "summary.json"
        ),
        "spd_s3_summary_sha256": (
            args.strict_root / "s3" / "ums_spd4x2" / "summary.json"
        ),
        "prefix_s2_checkpoint_sha256": (
            args.strict_root / "s2" / "ums_prefix4" / "best.pt"
        ),
        "spd_s2_checkpoint_sha256": (
            args.strict_root / "s2" / "ums_spd4x2" / "best.pt"
        ),
    }
    observed_hashes = {
        key: sha256_file(path) for key, path in authority_paths.items()
    }
    if any(observed_hashes[key] != authority[key] for key in authority_paths):
        raise ValueError("strict authority hash mismatch")

    prefix = json.loads(
        authority_paths["prefix_s3_summary_sha256"].read_text(
            encoding="utf-8"
        )
    )
    historical_spd = json.loads(
        authority_paths["spd_s3_summary_sha256"].read_text(encoding="utf-8")
    )
    prefix8 = json.loads(
        (
            args.diagnostic_root
            / "s3"
            / "ums_prefix8"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )
    no_ortho = json.loads(
        (
            args.diagnostic_root
            / "s3"
            / "ums_spd4x2_no_ortho"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )
    promotion = strict_lock["promotion"]
    prefix8_gate = gate(prefix8, prefix, promotion)
    no_ortho_gate = gate(no_ortho, prefix, promotion)
    historical_gate = gate(historical_spd, prefix, promotion)
    no_ortho_macro = no_ortho["expert_development"]["macro_auroc"]
    historical_macro = historical_spd["expert_development"]["macro_auroc"]
    repair_checks = {
        "passes_original_s3_gate": no_ortho_gate["pass"],
        "exceeds_historical_spd_macro_auroc": (
            no_ortho_macro > historical_macro
        ),
        "nonnegative_count_not_reduced": (
            no_ortho_gate["deltas"]["nonnegative_findings"]
            >= historical_gate["deltas"]["nonnegative_findings"]
        ),
    }
    nominate = all(repair_checks.values())
    result = {
        "schema_version": 1,
        "artifact": "strict_vivid_spd_bounded_diagnostic_verdict",
        "strict_verdict": "STRICT_NO_GO_DIAGNOSTIC_OPEN",
        "verdict": "REPAIR_NOMINATED" if nominate else "TERMINAL_NO_GO",
        "repair_candidate": (
            "ums_spd4x2_no_ortho" if nominate else None
        ),
        "repair_checks": repair_checks,
        "historical_spd_gate": historical_gate,
        "prefix8_diagnostic": {
            "promotion_eligible": False,
            "gate_against_prefix4": prefix8_gate,
            "macro_auroc_vs_historical_spd": (
                prefix8["expert_development"]["macro_auroc"]
                - historical_macro
            ),
        },
        "no_ortho_diagnostic": {
            "gate_against_prefix4": no_ortho_gate,
            "macro_auroc_vs_historical_spd": (
                no_ortho_macro - historical_macro
            ),
        },
        "strict_authority_hashes": observed_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
