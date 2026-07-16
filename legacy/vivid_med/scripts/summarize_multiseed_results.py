"""Summarize existing seed evidence and generated multi-seed manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

from case_study_modules_common import FINAL_DIR, load_metric_rows, metric_lookup, read_csv_rows, root_path, write_csv_rows, write_md_sections, write_md_table


COLUMNS = [
    "run",
    "seed",
    "chexpert_auc",
    "nih_auc",
    "hard_shuffle_delta",
    "cf_acc",
    "ab_swap_acc",
    "leakage",
    "status",
    "latest_formal_eval_step",
    "latest_formal_val_loss",
    "latest_live_step",
    "source",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=FINAL_DIR / "multiseed_run_manifest.csv")
    parser.add_argument("--output-csv", type=Path, default=FINAL_DIR / "multiseed_stability.csv")
    parser.add_argument("--output-md", type=Path, default=FINAL_DIR / "multiseed_stability.md")
    return parser.parse_args()


def latest_formal_eval_from_dir(output_dir: Path) -> tuple[str, str]:
    best_step = -1
    best_file: Path | None = None
    for path in output_dir.glob("metrics_step_*.json"):
        match = re.match(r"metrics_step_(\d+)\.json$", path.name)
        if not match:
            continue
        step = int(match.group(1))
        if step > best_step:
            best_step = step
            best_file = path
    if best_file is None:
        return "", ""
    try:
        payload = json.loads(best_file.read_text(encoding="utf-8"))
        loss = payload.get("val_loss", "")
    except (OSError, json.JSONDecodeError):
        loss = ""
    return str(best_step), "" if loss in (None, "") else str(loss)


def latest_live_step_from_dir(output_dir: Path) -> str:
    log_path = output_dir / "training_log.txt"
    if not log_path.exists():
        return ""
    best_step = -1
    pattern = re.compile(r'"global_step"\s*:\s*(\d+)')
    try:
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                match = pattern.search(line)
                if match:
                    best_step = max(best_step, int(match.group(1)))
    except OSError:
        return ""
    return "" if best_step < 0 else str(best_step)


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
        output_dir = root_path(item.get("output_dir", ""))
        latest_formal_eval_step = item.get("latest_formal_eval_step", "")
        latest_formal_val_loss = item.get("latest_formal_val_loss", "")
        latest_live_step = item.get("latest_live_step", "")
        if output_dir.exists():
            if not latest_formal_eval_step or not latest_formal_val_loss:
                latest_formal_eval_step, latest_formal_val_loss = latest_formal_eval_from_dir(output_dir)
            if not latest_live_step:
                latest_live_step = latest_live_step_from_dir(output_dir)
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
                "latest_formal_eval_step": latest_formal_eval_step,
                "latest_formal_val_loss": latest_formal_val_loss,
                "latest_live_step": latest_live_step,
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
        "empty metrics are planned, queued, or active seed slots and should not be used as stability evidence until "
        "`metrics_final.json` and downstream diagnostics are generated for that seed. `latest_formal_eval_step` / "
        "`latest_formal_val_loss` record the strongest currently landed intermediate eval artifact for active rows, "
        "while `latest_live_step` records the newest observed train-log step and is live progress only, not final evidence."
    )
    write_md_table(args.output_md, "Multi-Seed Stability", rows, COLUMNS, note)
    write_md_sections(
        FINAL_DIR / "multiseed_stability_boundary.md",
        "Multi-Seed Stability Boundary",
        [
            (
                "Current Boundary",
                "Existing artifacts provide single-seed comparisons for the main candidate family. The generated manifest defines the remaining queued and active seed execution contract; these rows do not become stability evidence until their run packages exist.",
            )
        ],
    )


if __name__ == "__main__":
    main()
