"""Summarize NIH transfer failures, per-label deltas, and likely causes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from case_study_modules_common import (
    FINAL_DIR,
    fmt,
    load_metric_rows,
    metric_lookup,
    nested,
    read_csv_rows,
    read_json,
    rel,
    root_path,
    to_float,
    write_csv_rows,
    write_md_sections,
    write_md_table,
)


CASE_COLUMNS = ["case_study", "case_id", "nih_label", "methods_disagree", "likely_cause", "evidence", "source"]
PER_LABEL_COLUMNS = ["run_id", "label", "auc", "auprc", "f1", "ece", "support", "source"]


def transfer_json_for(row: dict[str, Any]) -> Path | None:
    for key in ("nih_path", "path"):
        value = row.get(key)
        if value and Path(str(value)).suffix == ".json":
            candidate = root_path(str(value))
            if candidate.exists():
                return candidate
    run_id = row.get("id") or row.get("run_id")
    if not run_id:
        return None
    for base in (root_path("outputs/qwen3vl_next_stage_transfer"), root_path("outputs/qwen3vl_p4v2_transfer")):
        if base.exists():
            matches = sorted(base.rglob(f"*{run_id}*transfer_metrics.json"))
            if matches:
                return matches[0]
    return None


def collect_per_label(run_ids: list[str]) -> list[dict[str, Any]]:
    lp_rows = read_csv_rows(FINAL_DIR / "next_stage_lp_transfer_results.csv") + read_csv_rows(FINAL_DIR / "qwen3vl_p4v2_lp_results.csv")
    lookup = {}
    for row in lp_rows:
        for key in (row.get("id"), row.get("run_id")):
            if key:
                lookup[str(key).lower().replace("-", "_")] = row
    out: list[dict[str, Any]] = []
    for run_id in run_ids:
        row = lookup.get(run_id.lower().replace("-", "_"))
        path = transfer_json_for(row or {"id": run_id})
        payload = read_json(path) if path else None
        per_label = nested(payload, "metrics", "per_label") or {}
        for label, metrics in per_label.items():
            out.append(
                {
                    "run_id": run_id,
                    "label": label,
                    "auc": fmt(metrics.get("auc")),
                    "auprc": fmt(metrics.get("auprc")),
                    "f1": fmt(metrics.get("f1")),
                    "ece": fmt(metrics.get("ece")),
                    "support": fmt(metrics.get("support")),
                    "source": rel(path) if path else "",
                }
            )
    return out


def build_failure_cases(per_label: list[dict[str, Any]], candidate: str, baseline: str) -> list[dict[str, Any]]:
    by_label: dict[str, dict[str, dict[str, Any]]] = {}
    for row in per_label:
        by_label.setdefault(str(row["label"]), {})[str(row["run_id"])] = row
    rows: list[dict[str, Any]] = []
    for label, methods in sorted(by_label.items()):
        cand = methods.get(candidate)
        base = methods.get(baseline)
        if not cand or not base:
            continue
        cand_auc = to_float(cand.get("auc"))
        base_auc = to_float(base.get("auc"))
        if cand_auc is None or base_auc is None:
            continue
        delta = cand_auc - base_auc
        likely = "label_mapping_or_noise" if abs(delta) < 0.02 else ("candidate_transfer_gain" if delta > 0 else "candidate_transfer_loss")
        if label in {"Pneumonia", "No Finding", "Atelectasis"}:
            likely = "label_policy_or_prevalence"
        rows.append(
            {
                "case_study": "CS3_NIH_disagreement",
                "case_id": len(rows) + 1,
                "nih_label": label,
                "methods_disagree": f"{candidate} auc={fmt(cand_auc)} vs {baseline} auc={fmt(base_auc)}",
                "likely_cause": likely,
                "evidence": f"delta_auc={fmt(delta)}",
                "source": f"{cand.get('source', '')}; {base.get('source', '')}",
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="shuf_tw_clinical")
    parser.add_argument("--baseline", default="shuf_3k")
    parser.add_argument("--extra-run", action="append", default=["sameq_shuf_3k", "shuf_k4", "shuf_k4_tw_visual"])
    parser.add_argument("--per-label-csv", type=Path, default=FINAL_DIR / "nih_per_label_audit.csv")
    parser.add_argument("--case-csv", type=Path, default=FINAL_DIR / "case_study_nih_transfer.csv")
    parser.add_argument("--case-md", type=Path, default=FINAL_DIR / "case_study_nih_transfer.md")
    parser.add_argument("--report-md", type=Path, default=FINAL_DIR / "nih_domain_audit.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_ids = [args.baseline, args.candidate] + list(args.extra_run)
    per_label = collect_per_label(run_ids)
    cases = build_failure_cases(per_label, args.candidate, args.baseline)
    write_csv_rows(args.per_label_csv, per_label, PER_LABEL_COLUMNS)
    write_csv_rows(args.case_csv, cases, CASE_COLUMNS)
    write_md_table(args.case_md, "NIH Transfer Failure Case Study", cases, CASE_COLUMNS)

    mapping_rows = read_csv_rows(FINAL_DIR / "nih_label_mapping_audit.csv")
    sections = [
        (
            "Protocol Boundary",
            "Current verified NIH transfer artifacts use `nih_external_test_ums.jsonl` with 1000 evaluated records and 0 missing images for the inspected runs. NIH-full remains a separate execution boundary unless a larger manifest is supplied.",
        ),
        (
            "Label Mapping Audit",
            "\n".join(
                ["| chexpert_mimic_field | nih_label | confidence | risk |", "| --- | --- | --- | --- |"]
                + [
                    f"| {row.get('chexpert_mimic_field','')} | {row.get('nih_label','')} | {row.get('mapping_confidence','')} | {row.get('risk','')} |"
                    for row in mapping_rows
                ]
            ),
        ),
        (
            "Per-Label Transfer",
            "\n".join(
                ["| run_id | label | auc | auprc | f1 | ece |", "| --- | --- | ---: | ---: | ---: | ---: |"]
                + [
                    f"| {row.get('run_id','')} | {row.get('label','')} | {row.get('auc','')} | {row.get('auprc','')} | {row.get('f1','')} | {row.get('ece','')} |"
                    for row in per_label
                ]
            ),
        ),
        (
            "Interpretation",
            "Small NIH deltas should be treated as a label/protocol/domain diagnosis signal, not as strong external generalization evidence. Labels with larger disagreements should drive follow-up case review.",
        ),
    ]
    write_md_sections(args.report_md, "NIH Domain Audit", sections)


if __name__ == "__main__":
    main()

