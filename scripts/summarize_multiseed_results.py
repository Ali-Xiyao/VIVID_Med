"""Summarize existing seed evidence and generated multi-seed manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from case_study_modules_common import FINAL_DIR, load_metric_rows, metric_lookup, read_csv_rows, write_csv_rows, write_md_sections, write_md_table


COLUMNS = ["run", "seed", "chexpert_auc", "nih_auc", "hard_shuffle_delta", "cf_acc", "ab_swap_acc", "leakage", "status", "source"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=FINAL_DIR / "multiseed_run_manifest.csv")
    parser.add_argument("--output-csv", type=Path, default=FINAL_DIR / "multiseed_stability.csv")
    parser.add_argument("--output-md", type=Path, default=FINAL_DIR / "multiseed_stability.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lookup = metric_lookup(load_metric_rows())
    manifest = read_csv_rows(args.manifest)
    rows: list[dict[str, Any]] = []
    for item in manifest:
        key = str(item.get("id", ""))
        metrics = lookup.get(key, {})
        seed = str(item.get("seed", ""))
        has_seed_reference = bool(metrics) and seed in {"0", "42"}
        status = item.get("status", "")
        if has_seed_reference and status == "planned":
            status = "existing_single_seed_reference"
        rows.append(
            {
                "run": item.get("run_id") or key,
                "seed": seed,
                "chexpert_auc": metrics.get("chexpert_auc", "") if has_seed_reference else "",
                "nih_auc": metrics.get("nih_auc", "") if has_seed_reference else "",
                "hard_shuffle_delta": metrics.get("hard_shuffle_delta", "") if has_seed_reference else "",
                "cf_acc": metrics.get("cf_acc", "") if has_seed_reference else "",
                "ab_swap_acc": metrics.get("ab_swap_acc", "") if has_seed_reference else "",
                "leakage": metrics.get("leakage_or_flag_pct", "") if has_seed_reference else "",
                "status": status,
                "source": metrics.get("source_table", "") if has_seed_reference else item.get("output_dir", ""),
            }
        )
    if not rows:
        for key in ["shuf_3k", "shuf_tw_clinical", "sameq_shuf_3k", "shuf_k4"]:
            metrics = lookup.get(key, {})
            rows.append(
                {
                    "run": metrics.get("run_id", key),
                    "seed": "single_existing",
                    "chexpert_auc": metrics.get("chexpert_auc", ""),
                    "nih_auc": metrics.get("nih_auc", ""),
                    "hard_shuffle_delta": metrics.get("hard_shuffle_delta", ""),
                    "cf_acc": metrics.get("cf_acc", ""),
                    "ab_swap_acc": metrics.get("ab_swap_acc", ""),
                    "leakage": metrics.get("leakage_or_flag_pct", ""),
                    "status": "existing_single_seed",
                    "source": metrics.get("source_table", ""),
                }
            )
    write_csv_rows(args.output_csv, rows, COLUMNS)
    note = (
        "Rows marked `existing_single_seed_reference` reuse current single-seed metrics as anchors. Rows with "
        "empty metrics are planned seed slots and should not be used as stability evidence until `metrics_final.json` "
        "and downstream diagnostics are generated for that seed."
    )
    write_md_table(args.output_md, "Multi-Seed Stability", rows, COLUMNS, note)
    write_md_sections(
        FINAL_DIR / "multiseed_stability_boundary.md",
        "Multi-Seed Stability Boundary",
        [
            (
                "Current Boundary",
                "Existing artifacts provide single-seed comparisons for the main candidate family. The generated manifest defines the seed3 execution contract; missing seed slots remain pending until their run packages exist.",
            )
        ],
    )


if __name__ == "__main__":
    main()
