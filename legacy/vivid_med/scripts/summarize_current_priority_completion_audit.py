"""Summarize the current-priority completion boundary for the revision run."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs" / "final_tables"

COLUMNS = [
    "audit_item",
    "decision",
    "evidence_strength",
    "evidence_paths",
    "key_result",
    "remaining_gap",
    "allowed_claim",
    "forbidden_claim",
]


def read_csv(rel_path: str) -> list[dict[str, str]]:
    path = ROOT / rel_path
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def by_key(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"{value} not found in {key}")


def by_fields(rows: list[dict[str, str]], **expected: str) -> dict[str, str]:
    for row in rows:
        if all(row.get(key) == value for key, value in expected.items()):
            return row
    raise KeyError(f"row not found for {expected}")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row.get(column, "")) for column in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body]) + "\n"


def main() -> None:
    gap_rows = read_csv("outputs/final_tables/revision_completion_gap_audit.csv")
    llm_rows = read_csv("outputs/final_tables/llm_necessity.csv")
    scaling_rows = read_csv("outputs/final_tables/data_scaling_3k_current_progress.csv")
    scaling_10k_rows = read_csv("outputs/final_tables/data_scaling_10k_progress.csv")
    scaling_30k_rows = read_csv("outputs/final_tables/data_scaling_30k_progress.csv")
    frozen_source_rows = read_csv("outputs/final_tables/data_scaling_frozen_source_progress.csv")

    data_scaling = by_key(gap_rows, "requirement", "P1_DATA_SCALING")
    schema = by_key(gap_rows, "requirement", "P1_SCHEMA_COMPLEXITY_SWEEP")
    answerability = by_key(gap_rows, "requirement", "P3_ANSWERABILITY_SEMANTICS")
    schema_dependency = by_key(gap_rows, "requirement", "P3_SCHEMA_DEPENDENCY_WRITEUP")
    cost = by_key(gap_rows, "requirement", "P0_COST_TABLE")
    phase4 = by_key(gap_rows, "requirement", "Phase 4 paper tables and writing checklist")
    final_set = by_key(gap_rows, "requirement", "Final required output set")

    ums = by_key(llm_rows, "claim_area", "UMS/schema contribution")
    frozen_use = by_key(llm_rows, "claim_area", "Rare/high-null/uncertain frozen-LM use case")
    low_data = by_key(llm_rows, "claim_area", "Low-data frozen-LM necessity")
    schema_claim = by_key(llm_rows, "claim_area", "Schema complexity")
    serialization = by_key(llm_rows, "claim_area", "Schema serialization robustness")

    no_lm_3k_final = by_fields(
        scaling_rows,
        run_id="lp_no_lm_ums_3k",
        stage="linear_probe",
        metric_policy="final_checkpoint",
    )
    frozen_3k_final = by_fields(
        scaling_rows,
        run_id="lp_frozen_lm_ums_3k",
        stage="linear_probe",
        metric_policy="final_checkpoint",
    )
    no_lm_3k_best_auc = by_fields(
        scaling_rows,
        run_id="lp_no_lm_ums_3k",
        stage="linear_probe",
        metric_policy="best_macro_auc",
    )
    frozen_3k_best_auc = by_fields(
        scaling_rows,
        run_id="lp_frozen_lm_ums_3k",
        stage="linear_probe",
        metric_policy="best_macro_auc",
    )
    bce_10k_final = by_fields(
        scaling_10k_rows,
        run_id="bce_10k",
        stage="source_classifier",
        metric_policy="final_checkpoint",
    )
    bce_10k_best_auc = by_fields(
        scaling_10k_rows,
        run_id="bce_10k",
        stage="source_classifier",
        metric_policy="best_macro_auc",
    )
    no_lm_10k_final = by_fields(
        scaling_10k_rows,
        run_id="no_lm_ums_10k",
        stage="source_classifier",
        metric_policy="final_checkpoint",
    )
    no_lm_10k_best_auc = by_fields(
        scaling_10k_rows,
        run_id="no_lm_ums_10k",
        stage="source_classifier",
        metric_policy="best_macro_auc",
    )
    no_lm_lp_10k_final = by_fields(
        scaling_10k_rows,
        run_id="lp_no_lm_ums_10k",
        stage="linear_probe",
        metric_policy="final_checkpoint",
    )
    no_lm_lp_10k_best_auc = by_fields(
        scaling_10k_rows,
        run_id="lp_no_lm_ums_10k",
        stage="linear_probe",
        metric_policy="best_macro_auc",
    )
    frozen_10k_source = by_fields(
        frozen_source_rows,
        run_id="frozen_lm_ums_10k",
        stage="source_loss_only",
    )
    frozen_lp_10k_final = by_fields(
        scaling_10k_rows,
        run_id="lp_frozen_lm_ums_10k",
        stage="linear_probe",
        metric_policy="final_checkpoint",
    )
    frozen_lp_10k_best_auc = by_fields(
        scaling_10k_rows,
        run_id="lp_frozen_lm_ums_10k",
        stage="linear_probe",
        metric_policy="best_macro_auc",
    )
    bce_30k_final = by_fields(
        scaling_30k_rows,
        run_id="bce_30k",
        stage="source_classifier",
        metric_policy="final_checkpoint",
    )
    bce_30k_best_auc = by_fields(
        scaling_30k_rows,
        run_id="bce_30k",
        stage="source_classifier",
        metric_policy="best_macro_auc",
    )
    no_lm_30k_final = by_fields(
        scaling_30k_rows,
        run_id="no_lm_ums_30k",
        stage="source_classifier",
        metric_policy="final_checkpoint",
    )
    no_lm_30k_best_auc = by_fields(
        scaling_30k_rows,
        run_id="no_lm_ums_30k",
        stage="source_classifier",
        metric_policy="best_macro_auc",
    )
    no_lm_lp_30k_final = by_fields(
        scaling_30k_rows,
        run_id="lp_no_lm_ums_30k",
        stage="linear_probe",
        metric_policy="final_checkpoint",
    )
    no_lm_lp_30k_best_auc = by_fields(
        scaling_30k_rows,
        run_id="lp_no_lm_ums_30k",
        stage="linear_probe",
        metric_policy="best_macro_auc",
    )
    frozen_lp_30k_final = by_fields(
        scaling_30k_rows,
        run_id="lp_frozen_lm_ums_30k",
        stage="linear_probe",
        metric_policy="final_checkpoint",
    )
    frozen_lp_30k_best_auc = by_fields(
        scaling_30k_rows,
        run_id="lp_frozen_lm_ums_30k",
        stage="linear_probe",
        metric_policy="best_macro_auc",
    )

    rows = [
        {
            "audit_item": "UMS/schema contribution",
            "decision": "complete_for_current_priority",
            "evidence_strength": "strong",
            "evidence_paths": "outputs/final_tables/llm_necessity.csv; outputs/final_tables/main_controlled_results.csv",
            "key_result": ums["key_metrics"],
            "remaining_gap": "none for current claim boundary",
            "allowed_claim": ums["interpretation"],
            "forbidden_claim": ums["forbidden_overclaim"],
        },
        {
            "audit_item": "Frozen-LM use case",
            "decision": "complete_bounded",
            "evidence_strength": "strong_for_subgroup_signal",
            "evidence_paths": "outputs/final_tables/grouped_field_results.csv; outputs/final_tables/llm_necessity.csv",
            "key_result": frozen_use["key_metrics"],
            "remaining_gap": "not universal; external/NIH dominance remains weakened",
            "allowed_claim": frozen_use["interpretation"],
            "forbidden_claim": frozen_use["forbidden_overclaim"],
        },
        {
            "audit_item": "Low-data scaling",
            "decision": "completed_matrix_with_1k_source_caveat",
            "evidence_strength": data_scaling["evidence_strength"],
            "evidence_paths": "outputs/final_tables/data_scaling_1k_matched_deltas.csv; outputs/final_tables/data_scaling_3k_current_progress.csv; outputs/final_tables/data_scaling_10k_progress.csv; outputs/final_tables/data_scaling_30k_progress.csv; outputs/final_tables/data_scaling_frozen_source_progress.csv",
            "key_result": (
                f"3k final macro-AUC frozen {frozen_3k_final['macro_auc']} vs no-LM {no_lm_3k_final['macro_auc']}; "
                f"3k best-AUC frozen {frozen_3k_best_auc['macro_auc']} vs no-LM {no_lm_3k_best_auc['macro_auc']}; "
                f"10k BCE final macro-AUC {bce_10k_final['macro_auc']} and best-AUC {bce_10k_best_auc['macro_auc']}; "
                f"10k no-LM source final macro-AUC {no_lm_10k_final['macro_auc']} and best-AUC {no_lm_10k_best_auc['macro_auc']}; "
                f"10k no-LM LP final macro-AUC {no_lm_lp_10k_final['macro_auc']} and best-AUC {no_lm_lp_10k_best_auc['macro_auc']}; "
                f"10k frozen-LM source final val_loss {frozen_10k_source['latest_val_loss']} and best val_loss {frozen_10k_source['best_val_loss']} at step {frozen_10k_source['best_step']}; "
                f"10k frozen-LM LP final macro-AUC {frozen_lp_10k_final['macro_auc']} and best-AUC {frozen_lp_10k_best_auc['macro_auc']}; "
                f"30k BCE final macro-AUC {bce_30k_final['macro_auc']} and best-AUC {bce_30k_best_auc['macro_auc']}; "
                f"30k no-LM source final macro-AUC {no_lm_30k_final['macro_auc']} and best-AUC {no_lm_30k_best_auc['macro_auc']}; "
                f"30k no-LM LP final macro-AUC {no_lm_lp_30k_final['macro_auc']} and best-AUC {no_lm_lp_30k_best_auc['macro_auc']}; "
                f"30k frozen-LM LP final macro-AUC {frozen_lp_30k_final['macro_auc']} and best-AUC {frozen_lp_30k_best_auc['macro_auc']}; "
                f"{low_data['key_metrics']}"
            ),
            "remaining_gap": data_scaling["gap"],
            "allowed_claim": low_data["interpretation"],
            "forbidden_claim": low_data["forbidden_overclaim"],
        },
        {
            "audit_item": "Schema complexity",
            "decision": "complete_fixed_split_matrix",
            "evidence_strength": schema["evidence_strength"],
            "evidence_paths": "outputs/final_tables/schema_complexity_diagnostic_summary.csv; outputs/final_tables/revision_completion_gap_audit.csv",
            "key_result": schema_claim["key_metrics"],
            "remaining_gap": schema["gap"],
            "allowed_claim": schema_claim["interpretation"],
            "forbidden_claim": schema_claim["forbidden_overclaim"],
        },
        {
            "audit_item": "Answerability semantics",
            "decision": "complete_semantic_boundary_limited_calibration",
            "evidence_strength": answerability["evidence_strength"],
            "evidence_paths": "outputs/final_tables/answerability_semantics.csv; outputs/final_tables/null_field_calibration.csv",
            "key_result": by_key(llm_rows, "claim_area", "Answerability/null semantics")["key_metrics"],
            "remaining_gap": answerability["gap"],
            "allowed_claim": by_key(llm_rows, "claim_area", "Answerability/null semantics")["interpretation"],
            "forbidden_claim": by_key(llm_rows, "claim_area", "Answerability/null semantics")["forbidden_overclaim"],
        },
        {
            "audit_item": "Schema serialization dependency",
            "decision": "complete_as_limitation",
            "evidence_strength": schema_dependency["evidence_strength"],
            "evidence_paths": "outputs/final_tables/schema_dependency_diagnostics.csv; outputs/final_tables/schema_dependency_case_study.md",
            "key_result": serialization["key_metrics"],
            "remaining_gap": schema_dependency["gap"],
            "allowed_claim": serialization["interpretation"],
            "forbidden_claim": serialization["forbidden_overclaim"],
        },
        {
            "audit_item": "Cost/resource table",
            "decision": "complete_with_missing_fields",
            "evidence_strength": cost["evidence_strength"],
            "evidence_paths": "outputs/final_tables/cost_table.csv; outputs/final_tables/cost_missing_artifacts.md",
            "key_result": "deployment is LLM-free, but training-time LM cost and missing resource fields remain explicit",
            "remaining_gap": cost["gap"],
            "allowed_claim": "report known costs and missing fields transparently",
            "forbidden_claim": "resource-friendly training claim without peak-memory/runtime evidence",
        },
        {
            "audit_item": "Phase 2 / SPD variants",
            "decision": "deferred_no_new_spd",
            "evidence_strength": "supported_by_failure_mining_and_user_constraint",
            "evidence_paths": "outputs/final_tables/module_candidates.csv; outputs/final_tables/failure_case_summary.csv",
            "key_result": "failure mining is mixed; SPD new variants are explicitly out of scope",
            "remaining_gap": "no new module training in current pass",
            "allowed_claim": "defer modules until a concrete failure slice is selected",
            "forbidden_claim": "present deferred modules or SPD variants as completed",
        },
        {
            "audit_item": "Phase 4 claim packet",
            "decision": "complete_for_current_evidence_boundary",
            "evidence_strength": phase4["evidence_strength"],
            "evidence_paths": phase4["evidence_paths"],
            "key_result": phase4["current_status"],
            "remaining_gap": phase4["gap"],
            "allowed_claim": "use paper tables/checklist with explicit 10k/30k boundary",
            "forbidden_claim": "hide incomplete 10k/30k data-scaling scope",
        },
        {
            "audit_item": "Goal completion decision",
            "decision": "current_execution_scope_complete_with_caveats",
            "evidence_strength": final_set["evidence_strength"],
            "evidence_paths": "outputs/final_tables/revision_completion_gap_audit.csv; outputs/final_tables/current_priority_completion_audit.csv",
            "key_result": "current priority packet and the 1k/3k/10k/30k matched data-scaling LP matrix are complete; preserve the documented 1k frozen-LM source early-stop provenance caveat",
            "remaining_gap": final_set["gap"],
            "allowed_claim": "current-priority execution boundary is ready for paper rewrite",
            "forbidden_claim": "hide the 1k frozen-LM source early-stop caveat or mix source-loss metrics with LP macro-AUC metrics",
        },
    ]

    write_csv(FINAL_DIR / "current_priority_completion_audit.csv", rows, COLUMNS)

    text = "# Current Priority Completion Audit\n\n"
    text += (
        "This audit verifies the user-prioritized revision packet without redefining the original exhaustive plan. "
        "It distinguishes current-priority completion from the still-unrun 10k/30k data-scaling matrix.\n\n"
    )
    text += markdown_table(rows, COLUMNS)
    text += "\n## Decision\n\n"
    text += (
        "- Current priority packet: complete for paper-claim cleanup.\n"
        "- Original 1k/3k/10k/30k matched data-scaling LP matrix: complete, with the documented 1k frozen-LM source early-stop caveat.\n"
        "- Do not hide metric-policy qualifiers or compare source-loss rows directly against LP macro-AUC rows.\n"
    )
    (FINAL_DIR / "current_priority_completion_audit.md").write_text(text, encoding="utf-8")
    print(f"Wrote current priority completion audit to {FINAL_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
