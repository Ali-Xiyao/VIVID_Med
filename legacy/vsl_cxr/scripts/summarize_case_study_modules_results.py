"""Summarize case-study/module artifacts into final plan tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from case_study_modules_common import (
    FINAL_DIR,
    fmt,
    load_metric_rows,
    metric_lookup,
    read_csv_rows,
    root_path,
    to_float,
    write_csv_rows,
    write_md_sections,
    write_md_table,
)


MODULES = [
    ("CEQ", "models/clinical_evidence_query.py", "finding-specific evidence query", "attention maps/per-field AUC"),
    ("AUCH", "models/answerability_uncertainty_head.py", "answerability/uncertainty/calibration", "ECE/Brier/uncertain F1"),
    ("HNMB", "models/hard_negative_memory_bank.py", "hard-negative memory bank", "negative false rate/hard shuffle"),
    ("DRA", "models/domain_robust_adapter.py", "domain-robust adapter", "NIH/MMD/ECE external"),
    ("CCSH", "models/clinical_consistency_head.py", "clinical consistency scoring", "support/contradict/uncertain accuracy"),
    ("CDCS", "models/case_driven_curriculum_scheduler.py", "case-driven curriculum scheduler", "failure reduction"),
]


REQUIRED_ARTIFACTS = [
    "scripts/mine_pairwise_case_studies.py",
    "scripts/audit_nih_transfer_failure.py",
    "scripts/audit_hard_negative_quality.py",
    "scripts/audit_curriculum_leakage_cases.py",
    "scripts/build_casebook_markdown.py",
    "scripts/run_multiseed_manifest.py",
    "scripts/bootstrap_auc_ci.py",
    "scripts/paired_bootstrap_method_delta.py",
    "scripts/summarize_multiseed_results.py",
    "scripts/audit_label_mapping_nih.py",
    "scripts/run_nih_full_transfer.py",
    "scripts/compute_domain_shift_mmd.py",
    "scripts/plot_dataset_embedding_umap.py",
    "scripts/build_curriculum_v2_schedule.py",
    "scripts/generate_curriculum_v2_instructions.py",
    "scripts/train_qwen3vl_curriculum_v2.py",
    "outputs/final_tables/case_study_summary.md",
    "outputs/final_tables/multiseed_stability.md",
    "outputs/final_tables/nih_domain_audit.md",
    "outputs/final_tables/module_candidate_results.md",
    "outputs/final_tables/locked_final_comparison.md",
]


def choose(metrics: dict[str, dict[str, Any]], candidates: list[str], require_safety: bool = False) -> tuple[str, dict[str, Any]]:
    best_key = ""
    best_row: dict[str, Any] = {}
    best_score = -1e9
    for key in candidates:
        row = metrics.get(key, {})
        auc = to_float(row.get("chexpert_auc")) or 0.0
        hard = to_float(row.get("hard_shuffle_delta")) or 0.0
        leakage = to_float(row.get("leakage_or_flag_pct"))
        if require_safety and leakage is not None and leakage > 10:
            continue
        score = auc + 0.2 * hard
        if score > best_score:
            best_key, best_row, best_score = key, row, score
    return best_key, best_row


def module_rows() -> list[dict[str, Any]]:
    smoke = {row.get("module"): row for row in read_csv_rows(FINAL_DIR / "module_smoke_results.csv")}
    rows = []
    for module, path, purpose, metric in MODULES:
        exists = root_path(path).exists()
        smoke_status = smoke.get(module, {}).get("status", "not_run")
        rows.append(
            {
                "module": module,
                "source": path,
                "status": "implemented" if exists else "missing",
                "smoke_status": smoke_status,
                "purpose": purpose,
                "primary_readout": metric,
                "decision": "ready_for_formal_experiment" if exists and smoke_status == "passed" else "needs_fix_or_smoke",
            }
        )
    return rows


def locked_rows() -> list[dict[str, Any]]:
    metrics = metric_lookup(load_metric_rows())
    families = [
        ("Base", ["p2_value_only"], "baseline"),
        ("Direct SHUF", ["shuf_tw_clinical", "shuf_3k"], "candidate_vs_baseline"),
        ("SAMEQ", ["sameq_shuf_3k"], "grounding_family"),
        ("Multi-negative", ["shuf_k4_tw_visual", "shuf_k4"], "multi_negative_family"),
        ("Curriculum", ["prog_mix_tw_10k", "prog_mix_10k_8k", "prog_mix", "cur_p3_cf_shuf"], "curriculum_family"),
        ("Clinical module", [], "module_smoke_only_pending_training"),
        ("Domain robust", [], "module_smoke_only_pending_training"),
    ]
    rows = []
    for family, candidates, role in families:
        key, row = choose(metrics, candidates, require_safety=(family in {"Direct SHUF", "Multi-negative"}))
        rows.append(
            {
                "family": family,
                "finalist": row.get("run_id", key) if row else "",
                "seeds": "single_existing" if row else "",
                "chexpert_auc_mean_std": fmt(row.get("chexpert_auc")) if row else "",
                "nih_auc_mean_std": fmt(row.get("nih_auc")) if row else "",
                "hard_shuffle_mean_std": fmt(row.get("hard_shuffle_delta")) if row else "",
                "cf_mean_std": fmt(row.get("cf_acc")) if row else "",
                "auprc": "",
                "ece": "",
                "final_role": role,
                "boundary": "requires_seed3_before_final_claim" if row else "implemented_module_no_formal_run",
            }
        )
    return rows


def completion_rows() -> list[dict[str, Any]]:
    rows = []
    for path in REQUIRED_ARTIFACTS:
        rows.append({"artifact": path, "status": "completed" if root_path(path).exists() else "missing"})
    for module, path, _, _ in MODULES:
        rows.append({"artifact": path, "status": "completed" if root_path(path).exists() else "missing"})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-csv", type=Path, default=FINAL_DIR / "module_candidate_results.csv")
    parser.add_argument("--module-md", type=Path, default=FINAL_DIR / "module_candidate_results.md")
    parser.add_argument("--locked-csv", type=Path, default=FINAL_DIR / "locked_final_comparison.csv")
    parser.add_argument("--locked-md", type=Path, default=FINAL_DIR / "locked_final_comparison.md")
    parser.add_argument("--audit-csv", type=Path, default=FINAL_DIR / "case_study_modules_completion_audit.csv")
    parser.add_argument("--audit-md", type=Path, default=FINAL_DIR / "case_study_modules_completion_audit.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modules = module_rows()
    module_columns = ["module", "source", "status", "smoke_status", "purpose", "primary_readout", "decision"]
    write_csv_rows(args.module_csv, modules, module_columns)
    write_md_table(args.module_md, "Module Candidate Results", modules, module_columns)

    locked = locked_rows()
    locked_columns = ["family", "finalist", "seeds", "chexpert_auc_mean_std", "nih_auc_mean_std", "hard_shuffle_mean_std", "cf_mean_std", "auprc", "ece", "final_role", "boundary"]
    note = "Locked comparison table is deliberately conservative: single-seed rows remain candidates until seed3/CI artifacts exist."
    write_csv_rows(args.locked_csv, locked, locked_columns)
    write_md_table(args.locked_md, "Locked Final Comparison", locked, locked_columns, note)

    audit = completion_rows()
    write_csv_rows(args.audit_csv, audit, ["artifact", "status"])
    write_md_table(args.audit_md, "Case Study Modules Completion Audit", audit, ["artifact", "status"])

    sections = [
        ("Module Status", f"{sum(1 for row in modules if row['status'] == 'implemented')}/{len(modules)} modules implemented."),
        ("Locked Boundary", "No final-best claim is made here. The table encodes candidate family finalists and the remaining seed3/formal-module boundary."),
    ]
    write_md_sections(FINAL_DIR / "case_study_modules_summary.md", "Case Study Modules Summary", sections)


if __name__ == "__main__":
    main()
