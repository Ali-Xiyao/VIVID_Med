"""Summarize remaining gaps against the original VIVID-Med revision plan."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs" / "final_tables"


COLUMNS = [
    "requirement",
    "priority",
    "original_scope",
    "current_status",
    "evidence_strength",
    "evidence_paths",
    "gap",
    "decision",
    "next_safe_action",
    "stop_condition",
]


ROWS = [
    {
        "requirement": "P0_RESULT_CONSOLIDATION",
        "priority": "Priority 0 required",
        "original_scope": "Consolidate finished metrics, claim support, and missing artifacts.",
        "current_status": "completed",
        "evidence_strength": "strong",
        "evidence_paths": "outputs/final_tables/main_controlled_results.csv; outputs/final_tables/claim_support_matrix.md; outputs/final_tables/missing_artifacts.md",
        "gap": "none for current artifact inventory",
        "decision": "use for revision",
        "next_safe_action": "feed into Phase 4 claim/writing packet",
        "stop_condition": "stop if rows without checkpoint/config provenance are added",
    },
    {
        "requirement": "P0_COST_TABLE",
        "priority": "Priority 0 required",
        "original_scope": "Report training/deployment cost and distinguish deployment LLM-free from training-time frozen LM cost.",
        "current_status": "completed_with_missing_fields",
        "evidence_strength": "moderate",
        "evidence_paths": "outputs/final_tables/cost_table.csv; outputs/final_tables/cost_missing_artifacts.md",
        "gap": "peak memory and some parameter/runtime fields remain missing where historical logs do not prove them",
        "decision": "use with explicit missing-field caveat",
        "next_safe_action": "carry missing fields into final limitations instead of imputing",
        "stop_condition": "stop if resource-friendly language omits training-time LLM cost",
    },
    {
        "requirement": "P1_FIELD_DIFFICULTY_ANALYSIS",
        "priority": "Priority 0 required",
        "original_scope": "Find whether frozen-LM helps rare, uncertain, or high-null fields.",
        "current_status": "completed",
        "evidence_strength": "strong_for_aggregate_groups",
        "evidence_paths": "outputs/final_tables/grouped_field_results.csv; outputs/final_tables/per_field_results.csv; outputs/final_tables/field_difficulty_summary.md",
        "gap": "not sample-level failure mining",
        "decision": "use as frozen-LM use-case evidence",
        "next_safe_action": "connect to failure mining or final LLM-use-case table",
        "stop_condition": "stop if subgroup claim lacks per-field support",
    },
    {
        "requirement": "P1_DATA_SCALING",
        "priority": "Priority 0 required",
        "original_scope": "Run BCE/no-LM/frozen-LM over 1k/3k/10k/30k and summarize frozen-LM gain vs data size.",
        "current_status": "completed_data_scaling_matrix_with_documented_1k_frozen_source_early_stop",
        "evidence_strength": "strong_for_1k_3k_10k_30k_matched_lp_matrix_with_1k_source_provenance_caveat",
        "evidence_paths": "outputs/final_tables/data_scaling_1k_matched_summary.csv; outputs/final_tables/data_scaling_1k_matched_deltas.csv; outputs/data_scaling/lp_frozen_lm_ums_1k/metrics_final.json; outputs/data_scaling/lp_no_lm_ums_1k/metrics_final.json; outputs/final_tables/data_scaling_3k_bce_progress.csv; outputs/final_tables/data_scaling_3k_source_progress.csv; outputs/final_tables/data_scaling_3k_no_lm_lp_progress.csv; outputs/final_tables/data_scaling_3k_current_progress.csv; outputs/data_scaling/bce_3k/metrics_final.json; outputs/data_scaling/no_lm_ums_3k/metrics_final.json; outputs/data_scaling/lp_no_lm_ums_3k/metrics_final.json; outputs/logs/data_scaling_frozen_lm_ums_3k_source_gpu0.log; outputs/data_scaling/frozen_lm_ums_3k/checkpoints/best.pt; outputs/data_scaling/frozen_lm_ums_3k/checkpoints/final.pt; outputs/data_scaling/lp_frozen_lm_ums_3k/metrics_final.json; outputs/final_tables/data_scaling_10k_progress.csv; outputs/final_tables/data_scaling_frozen_source_progress.csv; outputs/data_scaling/bce_10k/metrics_final.json; outputs/data_scaling/no_lm_ums_10k/metrics_final.json; outputs/data_scaling/no_lm_ums_10k/best.pt; outputs/data_scaling/lp_no_lm_ums_10k/metrics_final.json; outputs/data_scaling/frozen_lm_ums_10k/checkpoints/final.pt; outputs/data_scaling/frozen_lm_ums_10k/checkpoints/best.pt; outputs/data_scaling/lp_frozen_lm_ums_10k/metrics_final.json; outputs/data_scaling/lp_frozen_lm_ums_10k/final.pt; outputs/final_tables/data_scaling_30k_progress.csv; outputs/data_scaling/bce_30k/metrics_final.json; outputs/data_scaling/no_lm_ums_30k/metrics_final.json; outputs/data_scaling/no_lm_ums_30k/best.pt; outputs/data_scaling/lp_no_lm_ums_30k/metrics_final.json; outputs/data_scaling/frozen_lm_ums_30k/checkpoints/final.pt; outputs/data_scaling/frozen_lm_ums_30k/checkpoints/best.pt; outputs/data_scaling/lp_frozen_lm_ums_30k/metrics_final.json; outputs/data_scaling/lp_frozen_lm_ums_30k/final.pt",
        "gap": "no remaining 1k/3k/10k/30k matched LP matrix gap; frozen-LM source 1k remains an explicitly documented early-stop provenance caveat because best.pt was accepted after external interruption",
        "decision": "do_not_claim_low_data_frozen_lm_necessity; do_not_expand_blindly",
        "next_safe_action": "use completed matrix with metric-policy qualifiers and preserve the 1k frozen-source early-stop caveat",
        "stop_condition": "stop if 1k result is reported as broad frozen-LM win, if the 3k final-AUC delta is generalized into broad necessity, if 10k no-LM LP is treated as matched frozen-LM evidence, or if fixed-split rows are mixed with historical P0 rows",
    },
    {
        "requirement": "P1_SCHEMA_COMPLEXITY_SWEEP",
        "priority": "Priority 0 required",
        "original_scope": "Formal no-LM vs frozen-LM comparison across S1/S2/S3 at minimum.",
        "current_status": "completed_formal_no_lm_and_frozen_lm_s1_s2_s3_source_lp_matrix",
        "evidence_strength": "strong_for_fixed_split_formal_schema_source_lp_matrix",
        "evidence_paths": "outputs/final_tables/schema_complexity_diagnostic_summary.csv; outputs/final_tables/schema_s1_coverage_audit.csv; outputs/final_tables/no_lm_schema_derivative_metrics.csv; outputs/final_tables/no_lm_schema_sweep_config_manifest.csv; outputs/final_tables/frozen_lm_source_training_summary.csv; outputs/final_tables/frozen_lm_source_val_loss_trace.csv; outputs/logs/schema_frozen_lm_s1_source_gpu0.log; outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/best.pt; outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/final.pt; outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/step_10000.pt; outputs/schema_sweep/lp_frozen_lm_s1_state_only/metrics_final.json; outputs/schema_sweep/lp_frozen_lm_s1_state_only/final.pt; outputs/schema_sweep/lp_frozen_lm_s1_state_only/best.pt; outputs/logs/schema_lp_frozen_lm_s1_gpu0.log; outputs/schema_sweep/frozen_lm_s2_state_answerability/checkpoints/best.pt; outputs/schema_sweep/frozen_lm_s2_state_answerability/checkpoints/final.pt; outputs/schema_sweep/frozen_lm_s2_state_answerability/checkpoints/step_10000.pt; outputs/logs/schema_frozen_lm_s3_source_gpu0.log; outputs/logs/schema_frozen_lm_s3_source_resume8000_gpu0.log; outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/best.pt; outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/final.pt; outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/step_10000.pt; outputs/schema_sweep/lp_frozen_lm_s2_state_answerability/metrics_final.json; outputs/schema_sweep/lp_frozen_lm_s2_state_answerability/final.pt; outputs/schema_sweep/lp_frozen_lm_s2_state_answerability/best.pt; outputs/schema_sweep/lp_frozen_lm_s3_state_uncertainty/metrics_final.json; outputs/schema_sweep/lp_frozen_lm_s3_state_uncertainty/final.pt; outputs/schema_sweep/lp_frozen_lm_s3_state_uncertainty/best.pt; outputs/logs/schema_lp_frozen_lm_s3_gpu0.log; outputs/schema_sweep/frozen_lm_s2_state_answerability_seed900122/checkpoints/final.pt; outputs/schema_sweep/frozen_lm_s3_state_uncertainty_seed900123/checkpoints/final.pt; outputs/schema_sweep/no_lm_s2_state_answerability_seed900124/metrics_final.json; outputs/schema_sweep/no_lm_s3_state_uncertainty_seed900125/metrics_final.json; outputs/schema_sweep/no_lm_s1_state_only/metrics_final.json; outputs/schema_sweep/no_lm_s1_state_only/final.pt; outputs/schema_sweep/no_lm_s1_state_only/best.pt; outputs/schema_sweep/lp_no_lm_s1_state_only/metrics_final.json; outputs/schema_sweep/lp_no_lm_s1_state_only/final.pt; outputs/schema_sweep/lp_no_lm_s1_state_only/best.pt; outputs/schema_sweep/no_lm_s2_state_answerability/metrics_final.json; outputs/schema_sweep/no_lm_s2_state_answerability/final.pt; outputs/schema_sweep/no_lm_s2_state_answerability/best.pt; outputs/schema_sweep/no_lm_s3_state_uncertainty/metrics_final.json; outputs/schema_sweep/no_lm_s3_state_uncertainty/final.pt; outputs/schema_sweep/no_lm_s3_state_uncertainty/best.pt; outputs/schema_sweep/lp_no_lm_s2_state_answerability/metrics_final.json; outputs/schema_sweep/lp_no_lm_s2_state_answerability/final.pt; outputs/schema_sweep/lp_no_lm_s2_state_answerability/best.pt; outputs/schema_sweep/lp_no_lm_s3_state_uncertainty/metrics_final.json; outputs/schema_sweep/lp_no_lm_s3_state_uncertainty/final.pt; outputs/schema_sweep/lp_no_lm_s3_state_uncertainty/best.pt",
        "gap": "none for the fixed-split formal S1/S2/S3 schema source+LP matrix; broader data-scaling scope remains separate",
        "decision": "use completed schema-complexity table with metric-family boundaries",
        "next_safe_action": "interpret schema-level trends from the consolidated table; do not compare source loss directly to LP macro-AUC",
        "stop_condition": "stop if source-loss rows are mixed with LP macro-AUC rows or debug serializer rows are used as formal performance evidence",
    },
    {
        "requirement": "P1_LLM_FAILURE_CASE_MINING",
        "priority": "Phase 1 listed; needed before Phase 2 modules",
        "original_scope": "Mine sample/label cases where frozen-LM, no-LM, BCE, and random-LM differ with probabilities and schema fields.",
        "current_status": "completed_eval_only",
        "evidence_strength": "strong_for_validation_split_binary_fields",
        "evidence_paths": "outputs/failure_cases/lp_failure_mining_predictions.csv; outputs/final_tables/failure_case_summary.csv; outputs/final_tables/failure_case_summary.md",
        "gap": "sample-level probabilities are now exported for val split; null/uncertain rows are retained but excluded from binary correctness",
        "decision": "use_for_phase4_module_deferral",
        "next_safe_action": "generate Phase 4 LLM necessity and module-candidate synthesis",
        "stop_condition": "stop if null/uncertain rows are treated as binary correctness",
    },
    {
        "requirement": "P1_EXTERNAL_CXR_TRANSFER",
        "priority": "Priority 2 optional",
        "original_scope": "Add external CXR dataset beyond CheXpert/NIH or strengthen NIH target LP.",
        "current_status": "deferred_optional",
        "evidence_strength": "existing_NIH_only",
        "evidence_paths": "outputs/final_tables/main_controlled_results.csv",
        "gap": "no MIMIC/PadChest/VinDr experiment in current execution; NIH rows already inform external dominance boundary",
        "decision": "defer unless revision requires another external dataset",
        "next_safe_action": "do not start external dataset work in this priority pass",
        "stop_condition": "stop if external dominance is claimed from NIH-mixed evidence",
    },
    {
        "requirement": "P3_ANSWERABILITY_SEMANTICS",
        "priority": "Priority 0 required",
        "original_scope": "Explain null-as-negative vs missingness semantics and calibration boundary.",
        "current_status": "completed_limited_by_oof",
        "evidence_strength": "moderate",
        "evidence_paths": "outputs/final_tables/answerability_semantics.csv; outputs/final_tables/null_field_calibration.csv; outputs/final_tables/answerability_missing_artifacts.md",
        "gap": "sample-level probability/OOD calibration remains missing",
        "decision": "use semantic/protocol claim; avoid calibration overclaim",
        "next_safe_action": "reuse future failure-mining probability export if available",
        "stop_condition": "stop if null calibration is claimed from aggregate metrics only",
    },
    {
        "requirement": "P3_SCHEMA_DEPENDENCY_WRITEUP",
        "priority": "Priority 0 required",
        "original_scope": "Quantify fixed-schema serialization dependence and write limitation.",
        "current_status": "completed",
        "evidence_strength": "strong_for_limitation",
        "evidence_paths": "outputs/final_tables/schema_dependency_diagnostics.csv; outputs/final_tables/schema_dependency_case_study.md",
        "gap": "mitigation not implemented; current scope only limitation/diagnostic",
        "decision": "use as transparent limitation",
        "next_safe_action": "include in Phase 4 limitation checklist",
        "stop_condition": "stop if schema-agnostic/paraphrase-robust language appears",
    },
    {
        "requirement": "Phase 2 module candidates",
        "priority": "Priority 1 select 2-3 modules",
        "original_scope": "Implement candidate modules only after Phase 1 failure cases motivate them.",
        "current_status": "deferred",
        "evidence_strength": "defer_decision_supported_by_failure_mining",
        "evidence_paths": "outputs/final_tables/failure_case_summary.csv; outputs/final_tables/schema_complexity_diagnostic_summary.csv; outputs/final_tables/data_scaling_1k_matched_summary.csv",
        "gap": "failure mining does not show frozen-LM dominance strong enough to justify new module training in this priority pass",
        "decision": "do_not_start_new_modules_now",
        "next_safe_action": "write module-candidate decision table with P2 deferred",
        "stop_condition": "stop if work drifts into SPD or unrelated new variants",
    },
    {
        "requirement": "Phase 4 paper tables and writing checklist",
        "priority": "required",
        "original_scope": "Produce final paper tables, llm necessity map, module candidates table, and writing/claim checklist.",
        "current_status": "completed_claim_synthesis",
        "evidence_strength": "strong_for_current_evidence_boundary",
        "evidence_paths": "outputs/final_tables/main_controlled_results.csv; outputs/final_tables/claim_support_matrix.md; outputs/final_tables/schema_complexity_diagnostic_summary.md; outputs/final_tables/data_scaling_1k_matched_summary.md; outputs/final_tables/failure_case_summary.csv; outputs/final_tables/llm_necessity.csv; outputs/final_tables/module_candidates.csv; outputs/final_tables/phase4_writing_claim_checklist.md",
        "gap": "3k BCE source, 3k matched no-LM/frozen-LM source+LP, 10k BCE source-control, 10k matched no-LM/frozen-LM source+LP, 30k matched no-LM/frozen-LM source+LP, and the formal schema S1/S2/S3 source+LP matrix are complete; frozen-LM 1k source keeps its documented early-stop provenance caveat",
        "decision": "use_for_paper_rewrite",
        "next_safe_action": "use completed data-scaling matrix with metric-policy qualifiers and the 1k source caveat",
        "stop_condition": "stop if final packet hides the 1k frozen-LM source early-stop caveat or mixes source-loss rows with LP metrics",
    },
    {
        "requirement": "Final required output set",
        "priority": "top-level checklist",
        "original_scope": "main_controlled_results, llm_necessity, module_candidates, answerability, schema dependency, cost, claim matrix.",
        "current_status": "completed_current_artifact_set",
        "evidence_strength": "strong_for_existing_tables",
        "evidence_paths": "outputs/final_tables/main_controlled_results.csv; outputs/final_tables/llm_necessity.csv; outputs/final_tables/module_candidates.csv; outputs/final_tables/answerability_semantics.csv; outputs/final_tables/schema_dependency_diagnostics.csv; outputs/final_tables/cost_table.csv; outputs/final_tables/claim_support_matrix.md",
        "gap": "current artifact set complete; original data-scaling matrix is complete under the documented 1k frozen-LM source early-stop caveat",
        "decision": "use_for_current_revision_boundary",
        "next_safe_action": "use for current revision boundary; preserve metric-policy qualifiers and the 1k early-stop caveat",
        "stop_condition": "stop if final outputs are declared complete before missing required files exist",
    },
]


def exists(rel_path: str) -> bool:
    return (ROOT / rel_path).exists()


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join([header, divider, *body]) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in ROWS:
        evidence_paths = [path.strip() for path in row["evidence_paths"].split(";")]
        present = [path for path in evidence_paths if path and exists(path)]
        row = dict(row)
        row["evidence_present_count"] = f"{len(present)} / {len([p for p in evidence_paths if p])}"
        rows.append(row)

    columns = [*COLUMNS, "evidence_present_count"]
    write_csv(FINAL_DIR / "revision_completion_gap_audit.csv", rows, columns)

    text = "# Revision Completion Gap Audit\n\n"
    text += (
        "This audit compares the original execution plan against current file evidence. "
        "It distinguishes completed evidence from diagnostic-only boundaries and missing requirements.\n\n"
    )
    text += markdown_table(rows, columns)
    text += "\n## Next Safe Action\n\n"
    text += (
        "The current non-training revision packet and the 1k/3k/10k/30k matched data-scaling LP matrix are complete. "
        "The remaining boundary is not a missing row but a provenance caveat: the 1k frozen-LM source was externally interrupted "
        "and accepted via best.pt for its downstream LP. Do not start P2 modules or broad new variants unless the paper explicitly requires more evidence.\n"
    )
    (FINAL_DIR / "revision_completion_gap_audit.md").write_text(text, encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {FINAL_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
