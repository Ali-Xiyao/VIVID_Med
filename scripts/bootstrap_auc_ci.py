"""Bootstrap confidence intervals for AUC/AUPRC/accuracy style metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from case_study_modules_common import FINAL_DIR, read_csv_rows, to_float, write_csv_rows, write_md_table


def bootstrap_metric(y_true: np.ndarray, y_score: np.ndarray, metric: str, n_boot: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2 and metric in {"auc", "auprc"}:
            continue
        if metric == "auc":
            values.append(float(roc_auc_score(y_true[idx], y_score[idx])))
        elif metric == "auprc":
            values.append(float(average_precision_score(y_true[idx], y_score[idx])))
        elif metric == "accuracy":
            values.append(float(((y_score[idx] >= 0.5).astype(int) == y_true[idx]).mean()))
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    if not values:
        return {"n_boot_valid": 0, "mean": "", "ci_low": "", "ci_high": ""}
    arr = np.asarray(values)
    return {
        "n_boot_valid": len(arr),
        "mean": float(arr.mean()),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="CSV with y_true and y_score columns.")
    parser.add_argument("--label-col", default="y_true")
    parser.add_argument("--score-col", default="y_score")
    parser.add_argument("--metric", choices=["auc", "auprc", "accuracy"], default="auc")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-value", type=float, help="Optional summary-only metric when predictions are unavailable.")
    parser.add_argument("--output-csv", type=Path, default=FINAL_DIR / "bootstrap_auc_ci.csv")
    parser.add_argument("--output-md", type=Path, default=FINAL_DIR / "bootstrap_auc_ci.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]]
    if args.input and args.input.exists():
        raw = read_csv_rows(args.input)
        y_true = np.asarray([to_float(row.get(args.label_col)) for row in raw], dtype=float)
        y_score = np.asarray([to_float(row.get(args.score_col)) for row in raw], dtype=float)
        mask = ~np.isnan(y_true) & ~np.isnan(y_score)
        result = bootstrap_metric(y_true[mask].astype(int), y_score[mask], args.metric, args.n_boot, args.seed)
        rows = [{"source": args.input.as_posix(), "metric": args.metric, "n": int(mask.sum()), **result, "boundary": "sample_level_bootstrap"}]
    else:
        rows = [
            {
                "source": str(args.input or ""),
                "metric": args.metric,
                "n": "",
                "n_boot_valid": "",
                "mean": args.summary_value if args.summary_value is not None else "",
                "ci_low": "",
                "ci_high": "",
                "boundary": "summary_only_no_prediction_csv",
            }
        ]
    columns = ["source", "metric", "n", "n_boot_valid", "mean", "ci_low", "ci_high", "boundary"]
    write_csv_rows(args.output_csv, rows, columns)
    write_md_table(args.output_md, "Bootstrap Metric CI", rows, columns)


if __name__ == "__main__":
    main()

