"""Paired bootstrap delta between two methods when sample predictions exist."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from case_study_modules_common import FINAL_DIR, load_metric_rows, metric_lookup, read_csv_rows, to_float, write_csv_rows, write_md_table


def load_pred(path: Path, sample_col: str, label_col: str, score_col: str) -> dict[str, tuple[int, float]]:
    rows = {}
    for row in read_csv_rows(path):
        y = to_float(row.get(label_col))
        score = to_float(row.get(score_col))
        sid = row.get(sample_col)
        if sid and y is not None and score is not None:
            rows[str(sid)] = (int(y), float(score))
    return rows


def paired_auc_delta(a: dict[str, tuple[int, float]], b: dict[str, tuple[int, float]], n_boot: int, seed: int) -> dict[str, Any]:
    keys = sorted(set(a) & set(b))
    y = np.asarray([a[k][0] for k in keys])
    score_a = np.asarray([a[k][1] for k in keys])
    score_b = np.asarray([b[k][1] for k in keys])
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(keys), len(keys))
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(float(roc_auc_score(y[idx], score_a[idx]) - roc_auc_score(y[idx], score_b[idx])))
    arr = np.asarray(deltas)
    return {
        "paired_n": len(keys),
        "delta_mean": float(arr.mean()) if len(arr) else "",
        "delta_ci_low": float(np.quantile(arr, 0.025)) if len(arr) else "",
        "delta_ci_high": float(np.quantile(arr, 0.975)) if len(arr) else "",
        "n_boot_valid": len(arr),
        "boundary": "paired_prediction_bootstrap",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pred", type=Path)
    parser.add_argument("--baseline-pred", type=Path)
    parser.add_argument("--candidate-id", default="shuf_tw_clinical")
    parser.add_argument("--baseline-id", default="shuf_3k")
    parser.add_argument("--metric-col", default="chexpert_auc")
    parser.add_argument("--sample-col", default="sample_id")
    parser.add_argument("--label-col", default="y_true")
    parser.add_argument("--score-col", default="y_score")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=Path, default=FINAL_DIR / "paired_bootstrap_method_delta.csv")
    parser.add_argument("--output-md", type=Path, default=FINAL_DIR / "paired_bootstrap_method_delta.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.candidate_pred and args.baseline_pred and args.candidate_pred.exists() and args.baseline_pred.exists():
        result = paired_auc_delta(
            load_pred(args.candidate_pred, args.sample_col, args.label_col, args.score_col),
            load_pred(args.baseline_pred, args.sample_col, args.label_col, args.score_col),
            args.n_boot,
            args.seed,
        )
    else:
        lookup = metric_lookup(load_metric_rows())
        cand = lookup.get(args.candidate_id, {})
        base = lookup.get(args.baseline_id, {})
        cand_value = to_float(cand.get(args.metric_col))
        base_value = to_float(base.get(args.metric_col))
        delta = (cand_value - base_value) if cand_value is not None and base_value is not None else ""
        result = {
            "paired_n": "",
            "delta_mean": delta,
            "delta_ci_low": "",
            "delta_ci_high": "",
            "n_boot_valid": "",
            "boundary": "summary_delta_only_no_paired_predictions",
        }
    rows = [{"candidate": args.candidate_id, "baseline": args.baseline_id, "metric": args.metric_col, **result}]
    columns = ["candidate", "baseline", "metric", "paired_n", "delta_mean", "delta_ci_low", "delta_ci_high", "n_boot_valid", "boundary"]
    write_csv_rows(args.output_csv, rows, columns)
    write_md_table(args.output_md, "Paired Bootstrap Method Delta", rows, columns)


if __name__ == "__main__":
    main()

