#!/usr/bin/env python
"""Rebuild a VICER V0 summary from a frozen, hash-locked row matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vicer_cxr.validity import canonical_sha256, file_sha256, summarize_v0_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--prior-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prior = json.loads(args.prior_result.read_text(encoding="utf-8"))
    rows_hash = file_sha256(args.rows)
    if rows_hash != prior.get("rows_sha256"):
        raise ValueError("VICER V0 row file hash does not match the prior result")
    rows = [
        json.loads(line)
        for line in args.rows.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    thresholds = prior["thresholds"]
    summary = summarize_v0_rows(
        rows,
        minimum_critic_auroc=float(thresholds["minimum_critic_auroc"]),
        minimum_verifier_auroc=float(thresholds["minimum_verifier_auroc"]),
        minimum_monotonic_spearman=float(thresholds["minimum_monotonic_spearman"]),
        minimum_preservation=float(thresholds["minimum_preservation"]),
        minimum_realism=float(thresholds["minimum_realism"]),
        minimum_valid_fraction=float(thresholds["minimum_valid_fraction"]),
    )
    for key in (
        "run_identity",
        "data_role",
        "single_reader_positive_allowed",
        "patient_level_claim",
        "chexlocalize_test_opened",
        "selector_started",
    ):
        summary[key] = prior[key]
    summary.update(
        {
            "rows_sha256": rows_hash,
            "summary_repair": {
                "reason": "Treat mathematically exact threshold equality as passing despite binary floating-point representation.",
                "prior_result_canonical_sha256": prior["canonical_sha256"],
                "model_rescored": False,
                "rows_changed": False,
                "thresholds_changed": False,
            },
        }
    )
    summary["canonical_sha256"] = canonical_sha256(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
