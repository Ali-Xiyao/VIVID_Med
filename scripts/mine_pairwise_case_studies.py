"""Mine SHUF-TW versus SHUF case-study rows from existing diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from case_study_modules_common import (
    FINAL_DIR,
    fmt,
    load_metric_rows,
    metric_lookup,
    read_first_diagnostic_csv,
    rel,
    summarize_metric_delta,
    to_float,
    truthy,
    write_csv_rows,
    write_md_table,
)


DEFAULT_COLUMNS = [
    "case_study",
    "case_id",
    "sample_id",
    "finding",
    "state",
    "candidate",
    "baseline",
    "win_loss",
    "candidate_score",
    "baseline_score",
    "evidence",
    "failure_type",
    "source",
    "manual_note",
]


def best_and_worst_rows(rows: list[dict[str, str]], limit: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    scored = []
    for row in rows:
        margin = to_float(row.get("best_negative_minus_correct_nll"))
        if margin is None:
            continue
        scored.append((margin, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    wins = [row for _, row in scored if truthy(row.get("correct_is_best"))][:limit]
    losses = [row for _, row in sorted(scored, key=lambda item: item[0]) if not truthy(row.get("correct_is_best"))][:limit]
    if not losses:
        losses = [row for _, row in sorted(scored, key=lambda item: item[0])[:limit]]
    return wins, losses


def row_to_case(
    row: dict[str, str],
    case_study: str,
    case_id: int,
    candidate: str,
    baseline: str,
    win_loss: str,
    source: Path | None,
) -> dict[str, Any]:
    margin = to_float(row.get("best_negative_minus_correct_nll"))
    failure_type = "visual_subtle_or_laterality" if (margin is not None and margin < 0.1) else "counterfactual_margin"
    if not truthy(row.get("correct_is_best")):
        failure_type = "model_wrong_option_or_shortcut"
    return {
        "case_study": case_study,
        "case_id": case_id,
        "sample_id": row.get("sample_id", ""),
        "finding": row.get("finding", ""),
        "state": row.get("state", ""),
        "candidate": candidate,
        "baseline": baseline,
        "win_loss": win_loss,
        "candidate_score": fmt(margin),
        "baseline_score": "",
        "evidence": f"correct_is_best={row.get('correct_is_best', '')}; cf_type={row.get('counterfactual_type', '')}",
        "failure_type": failure_type,
        "source": rel(source) if source else "",
        "manual_note": "Auto-mined from counterfactual row diagnostics; manual image/report review still required.",
    }


def run_level_cases(candidate: dict[str, Any], baseline: dict[str, Any], candidate_key: str, baseline_key: str) -> list[dict[str, Any]]:
    metrics = ["chexpert_auc", "nih_auc", "hard_shuffle_delta", "cf_acc", "ab_swap_acc", "leakage_or_flag_pct"]
    rows: list[dict[str, Any]] = []
    for index, metric in enumerate(metrics, start=1):
        delta = summarize_metric_delta(candidate, baseline, metric)
        cand = candidate.get(metric, "")
        base = baseline.get(metric, "")
        if not delta:
            continue
        rows.append(
            {
                "case_study": "run_level_delta",
                "case_id": index,
                "sample_id": "",
                "finding": metric,
                "state": "",
                "candidate": candidate_key,
                "baseline": baseline_key,
                "win_loss": "candidate_higher" if (to_float(delta) or 0) > 0 else "candidate_lower",
                "candidate_score": cand,
                "baseline_score": base,
                "evidence": f"{metric} delta={delta}",
                "failure_type": "summary_only_boundary",
                "source": f"{candidate.get('source_table', '')}; {baseline.get('source_table', '')}",
                "manual_note": "Run-level evidence only; paired sample predictions are required for strict per-case attribution.",
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="shuf_tw_clinical")
    parser.add_argument("--baseline", default="shuf_3k")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--output-csv", type=Path, default=FINAL_DIR / "case_study_shuf_tw_vs_shuf.csv")
    parser.add_argument("--output-md", type=Path, default=FINAL_DIR / "case_study_shuf_tw_vs_shuf.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lookup = metric_lookup(load_metric_rows())
    candidate = lookup.get(args.candidate)
    baseline = lookup.get(args.baseline)
    if not candidate or not baseline:
        raise SystemExit(f"Missing metric rows for candidate={args.candidate!r} baseline={args.baseline!r}")

    cases = run_level_cases(candidate, baseline, args.candidate, args.baseline)
    source, cf_rows = read_first_diagnostic_csv(args.candidate, "counterfactual_diagnostics_counterfactual_rows.csv")
    wins, losses = best_and_worst_rows(cf_rows, args.limit)
    for idx, row in enumerate(wins, start=1):
        cases.append(row_to_case(row, "CS1_SHUF_TW_wins_or_high_margin", idx, args.candidate, args.baseline, "candidate_win", source))
    for idx, row in enumerate(losses, start=1):
        cases.append(row_to_case(row, "CS2_SHUF_TW_losses_or_low_margin", idx, args.candidate, args.baseline, "candidate_loss", source))

    note = (
        "This casebook combines run-level deltas with available row-level counterfactual diagnostics. "
        "Strict paired attribution against SHUF-3k still requires shared sample-level prediction files for both methods."
    )
    write_csv_rows(args.output_csv, cases, DEFAULT_COLUMNS)
    write_md_table(args.output_md, "SHUF-TW-clinical vs SHUF-3k Case Study", cases, DEFAULT_COLUMNS, note)


if __name__ == "__main__":
    main()

