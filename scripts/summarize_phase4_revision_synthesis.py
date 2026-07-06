"""Generate Phase 4 revision synthesis tables."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs" / "final_tables"


LLM_COLUMNS = [
    "claim_area",
    "status",
    "key_evidence",
    "key_metrics",
    "interpretation",
    "paper_action",
    "forbidden_overclaim",
]


MODULE_COLUMNS = [
    "candidate",
    "original_priority",
    "current_evidence",
    "decision",
    "reason",
    "next_if_required",
    "stop_condition",
]


CHECKLIST_COLUMNS = [
    "section",
    "action",
    "required_language",
    "avoid_language",
    "evidence_paths",
]


def read_csv(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def by_key(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"{value} not found in column {key}")


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


def build_llm_rows() -> list[dict[str, str]]:
    main = read_csv("outputs/final_tables/main_controlled_results.csv")
    field = read_csv("outputs/final_tables/grouped_field_results.csv")
    scaling_delta = read_csv("outputs/final_tables/data_scaling_1k_matched_deltas.csv")
    scaling_3k = read_csv("outputs/final_tables/data_scaling_3k_current_progress.csv")
    scaling_10k = read_csv("outputs/final_tables/data_scaling_10k_progress.csv")
    scaling_30k = read_csv("outputs/final_tables/data_scaling_30k_progress.csv")
    schema = read_csv("outputs/final_tables/schema_complexity_diagnostic_summary.csv")
    failure = read_csv("outputs/final_tables/failure_case_summary.csv")

    frozen = by_key(main, "Method", "Frozen-LM UMS / no-SPD")
    no_lm = by_key(main, "Method", "no-LM UMS state classifier")
    bce = by_key(main, "Method", "Data-matched BCE ViT-B")
    random_lm = by_key(main, "Method", "Random-LM same-architecture UMS")
    rare = by_key(field, "field_group", "rare")
    uncertain = by_key(field, "field_group", "uncertain-heavy")
    high_null = by_key(field, "field_group", "high-null")
    one_k = by_key(scaling_delta, "comparison", "matched_lp_final_frozen_minus_no_lm")
    no_lm_3k_final = by_fields(
        scaling_3k,
        run_id="lp_no_lm_ums_3k",
        stage="linear_probe",
        metric_policy="final_checkpoint",
    )
    no_lm_3k_best_auc = by_fields(
        scaling_3k,
        run_id="lp_no_lm_ums_3k",
        stage="linear_probe",
        metric_policy="best_macro_auc",
    )
    frozen_3k_final = by_fields(
        scaling_3k,
        run_id="lp_frozen_lm_ums_3k",
        stage="linear_probe",
        metric_policy="final_checkpoint",
    )
    frozen_3k_best_auc = by_fields(
        scaling_3k,
        run_id="lp_frozen_lm_ums_3k",
        stage="linear_probe",
        metric_policy="best_macro_auc",
    )
    no_lm_10k_final = by_fields(
        scaling_10k,
        run_id="lp_no_lm_ums_10k",
        stage="linear_probe",
        metric_policy="final_checkpoint",
    )
    no_lm_10k_best_auc = by_fields(
        scaling_10k,
        run_id="lp_no_lm_ums_10k",
        stage="linear_probe",
        metric_policy="best_macro_auc",
    )
    frozen_10k_final = by_fields(
        scaling_10k,
        run_id="lp_frozen_lm_ums_10k",
        stage="linear_probe",
        metric_policy="final_checkpoint",
    )
    frozen_10k_best_auc = by_fields(
        scaling_10k,
        run_id="lp_frozen_lm_ums_10k",
        stage="linear_probe",
        metric_policy="best_macro_auc",
    )
    no_lm_30k_final = by_fields(
        scaling_30k,
        run_id="lp_no_lm_ums_30k",
        stage="linear_probe",
        metric_policy="final_checkpoint",
    )
    no_lm_30k_best_auc = by_fields(
        scaling_30k,
        run_id="lp_no_lm_ums_30k",
        stage="linear_probe",
        metric_policy="best_macro_auc",
    )
    frozen_30k_final = by_fields(
        scaling_30k,
        run_id="lp_frozen_lm_ums_30k",
        stage="linear_probe",
        metric_policy="final_checkpoint",
    )
    frozen_30k_best_auc = by_fields(
        scaling_30k,
        run_id="lp_frozen_lm_ums_30k",
        stage="linear_probe",
        metric_policy="best_macro_auc",
    )
    frozen_better = by_key(failure, "failure_class", "frozen_better_than_no_lm")
    no_lm_better = by_key(failure, "failure_class", "no_lm_better_than_frozen")
    schema_no_lm_ans = by_key(schema, "evidence_id", "no_lm_answerability_derivative")
    schema_no_lm_s1_formal = by_key(schema, "evidence_id", "no_lm_s1_state_only_formal_source")
    schema_no_lm_s1_lp = by_key(schema, "evidence_id", "no_lm_s1_state_only_formal_lp")
    schema_no_lm_s2_formal = by_key(schema, "evidence_id", "no_lm_s2_explicit_head_formal_source")
    schema_no_lm_s3_formal = by_key(schema, "evidence_id", "no_lm_s3_explicit_head_formal_source")
    schema_no_lm_s2_lp = by_key(schema, "evidence_id", "no_lm_s2_explicit_head_formal_lp")
    schema_no_lm_s3_lp = by_key(schema, "evidence_id", "no_lm_s3_explicit_head_formal_lp")
    schema_frozen_s1_source = by_key(schema, "evidence_id", "frozen_lm_s1_state_only_formal_source")
    schema_frozen_s1_lp = by_key(schema, "evidence_id", "frozen_lm_s1_state_only_formal_lp")
    schema_frozen_s2_source = by_key(schema, "evidence_id", "frozen_lm_s2_state_answerability_formal_source")
    schema_frozen_s2_lp = by_key(schema, "evidence_id", "frozen_lm_s2_state_answerability_formal_lp")
    schema_frozen_s3_source = by_key(schema, "evidence_id", "frozen_lm_s3_state_uncertainty_formal_source")
    schema_frozen_s3_lp = by_key(schema, "evidence_id", "frozen_lm_s3_state_uncertainty_formal_lp")
    schema_frozen_s2 = by_key(schema, "evidence_id", "frozen_lm_s2_serializer_debug")
    s1_audit = read_csv("outputs/final_tables/schema_s1_coverage_audit.csv")
    s1_not_covered = any(row["coverage_decision"] != "covered_by_historical_formal_row" for row in s1_audit)

    return [
        {
            "claim_area": "UMS/schema contribution",
            "status": "supported",
            "key_evidence": "no-LM UMS beats BCE in main controlled CheXpert/NIH table.",
            "key_metrics": (
                f"no-LM CheXpert AUC {no_lm['CheXpert AUC']} vs BCE {bce['CheXpert AUC']}; "
                f"no-LM NIH AUC {no_lm['NIH AUC']}"
            ),
            "interpretation": "Structured UMS supervision is a stable contribution even without an LM.",
            "paper_action": "Make UMS/schema the main mechanism claim.",
            "forbidden_overclaim": "Do not frame all gains as coming from the frozen LM.",
        },
        {
            "claim_area": "Pretrained frozen-LM in-domain gain",
            "status": "limited_supported",
            "key_evidence": "Frozen-LM UMS is the strongest current CheXpert controlled row and far above random-LM.",
            "key_metrics": (
                f"frozen CheXpert AUC {frozen['CheXpert AUC']} vs no-LM {no_lm['CheXpert AUC']} "
                f"and random-LM {random_lm['CheXpert AUC']}"
            ),
            "interpretation": "Pretrained frozen-LM can help in-domain, but the gain is modest and not universal.",
            "paper_action": "Write as controlled analysis of where frozen-LM helps.",
            "forbidden_overclaim": "Do not claim pretrained frozen-LM is broadly necessary.",
        },
        {
            "claim_area": "External/NIH LLM dominance",
            "status": "weakened",
            "key_evidence": "NIH rows favor no-LM over frozen-LM in the main table.",
            "key_metrics": f"no-LM NIH AUC {no_lm['NIH AUC']} vs frozen NIH AUC {frozen['NIH AUC']}",
            "interpretation": "External transfer does not support LLM dominance.",
            "paper_action": "Narrow external claims; say UMS-family transfer is mixed.",
            "forbidden_overclaim": "Do not claim frozen-LM dominates external CXR transfer.",
        },
        {
            "claim_area": "Low-data frozen-LM necessity",
            "status": "mixed_no_broad_necessity",
            "key_evidence": "Matched 1k/3k/10k/30k LP matrix is complete; frozen-LM has small final-AUC gains at 3k/10k, but no-LM is stronger at 1k and 30k, and best-AUC policy often favors no-LM.",
            "key_metrics": (
                f"1k final frozen-minus-no-LM macro-AUC {one_k['delta_macro_auc']}; "
                f"3k final frozen {frozen_3k_final['macro_auc']} vs no-LM {no_lm_3k_final['macro_auc']}; "
                f"3k best-AUC frozen {frozen_3k_best_auc['macro_auc']} vs no-LM {no_lm_3k_best_auc['macro_auc']}; "
                f"10k final frozen {frozen_10k_final['macro_auc']} vs no-LM {no_lm_10k_final['macro_auc']}; "
                f"10k best-AUC frozen {frozen_10k_best_auc['macro_auc']} vs no-LM {no_lm_10k_best_auc['macro_auc']}; "
                f"30k final frozen {frozen_30k_final['macro_auc']} vs no-LM {no_lm_30k_final['macro_auc']}; "
                f"30k best-AUC frozen {frozen_30k_best_auc['macro_auc']} vs no-LM {no_lm_30k_best_auc['macro_auc']}"
            ),
            "interpretation": "Current matched evidence is mixed and does not support a broad low-data frozen-LM necessity claim.",
            "paper_action": "Report as metric-policy and sample-size dependent; preserve the 1k frozen-LM source early-stop provenance caveat.",
            "forbidden_overclaim": "Do not write a 500x or broad data-efficiency claim.",
        },
        {
            "claim_area": "Rare/high-null/uncertain frozen-LM use case",
            "status": "supported_as_subgroup_signal",
            "key_evidence": "Field difficulty analysis shows frozen-LM gains in selected difficult groups.",
            "key_metrics": (
                f"rare delta {rare['frozen_minus_no_lm_auc']}; "
                f"uncertain-heavy delta {uncertain['frozen_minus_no_lm_auc']}; "
                f"high-null delta {high_null['frozen_minus_no_lm_auc']}"
            ),
            "interpretation": "The best current frozen-LM use case is difficult field groups, not common findings.",
            "paper_action": "Use as the positive LLM-use-case evidence.",
            "forbidden_overclaim": "Do not generalize subgroup gains to all fields.",
        },
        {
            "claim_area": "Schema complexity",
            "status": "completed_fixed_split_formal_schema_source_lp_matrix",
            "key_evidence": "no-LM and frozen-LM S1/S2/S3 source+LP endpoints are complete under the fixed split; Frozen-LM S2/S3 serializers pass debug entries; no-LM derived diagnostics exist.",
            "key_metrics": (
                f"derived no-LM answerability AUC {schema_no_lm_ans['primary_value']}; "
                f"no-LM S1 source {schema_no_lm_s1_formal['primary_value']} / LP {schema_no_lm_s1_lp['primary_value']}; "
                f"no-LM S2 source {schema_no_lm_s2_formal['primary_value']} / LP {schema_no_lm_s2_lp['primary_value']}; "
                f"no-LM S3 source {schema_no_lm_s3_formal['primary_value']} / LP {schema_no_lm_s3_lp['primary_value']}; "
                f"frozen S1 source val_loss {schema_frozen_s1_source['primary_value']} / LP {schema_frozen_s1_lp['primary_value']}; "
                f"frozen S2 source val_loss {schema_frozen_s2_source['primary_value']} / LP {schema_frozen_s2_lp['primary_value']}; "
                f"frozen S3 source val_loss {schema_frozen_s3_source['primary_value']} / LP {schema_frozen_s3_lp['primary_value']}; "
                f"frozen S2 serializer {schema_frozen_s2['primary_value']}; "
                f"S1 historical coverage {'mismatch documented' if s1_not_covered else 'covered'}"
            ),
            "interpretation": "The fixed-split formal schema matrix is now complete; use it to discuss schema-level tradeoffs while keeping source-loss and LP macro-AUC metric families separate.",
            "paper_action": "Use as the formal schema-complexity table with explicit metric-family boundaries.",
            "forbidden_overclaim": "Do not infer frozen-LM dominance without checking per-level metric direction.",
        },
        {
            "claim_area": "Failure mining",
            "status": "completed_eval_only_mixed",
            "key_evidence": "Per-sample LP probabilities now compare frozen-LM and no-LM on binary val fields.",
            "key_metrics": f"frozen better {frozen_better['count']} vs no-LM better {no_lm_better['count']}",
            "interpretation": "Disagreement counts are balanced; not enough to justify immediate P2 module training.",
            "paper_action": "Use for qualitative cases and module deferral rationale.",
            "forbidden_overclaim": "Do not claim frozen-LM wins most hard cases.",
        },
        {
            "claim_area": "Answerability/null semantics",
            "status": "supported_as_semantic_boundary",
            "key_evidence": "Answerability analysis separates null-as-negative classification from missingness semantics.",
            "key_metrics": "sample-level null calibration remains limited; semantic distinction is documented.",
            "interpretation": "Null-as-negative can improve classification but changes the meaning of missingness.",
            "paper_action": "Write as protocol/semantics limitation, not as a pure AUC contest.",
            "forbidden_overclaim": "Do not say null means clinically absent.",
        },
        {
            "claim_area": "Schema serialization robustness",
            "status": "limitation_supported",
            "key_evidence": "Schema key/order/paraphrase diagnostics show strong fixed serialization dependence.",
            "key_metrics": "clinical paraphrase NLL gap and key/order variants all worsen relative to original.",
            "interpretation": "The method learns a fixed-schema interface, not schema-agnostic language understanding.",
            "paper_action": "State transparently as limitation.",
            "forbidden_overclaim": "Do not claim schema paraphrase robustness.",
        },
    ]


def build_module_rows() -> list[dict[str, str]]:
    return [
        {
            "candidate": "Adaptive LLM Gating",
            "original_priority": "Priority 1 recommended",
            "current_evidence": "Field groups show frozen-LM gains, but failure mining disagreement counts are balanced.",
            "decision": "defer",
            "reason": "No strong per-case frozen-LM dominance signal; new module would be speculative.",
            "next_if_required": "Only design a rare/high-null gate if reviewers demand a module beyond analysis.",
            "stop_condition": "Do not implement until a concrete failure slice is selected.",
        },
        {
            "candidate": "Hierarchical UMS Head",
            "original_priority": "Priority 1 recommended",
            "current_evidence": "no-LM and frozen-LM S1/S2/S3 source+LP endpoints are complete under the fixed split.",
            "decision": "completed_evidence_defer_new_module",
            "reason": "The formal schema matrix is now available for comparison; a new hierarchical head would be a new intervention rather than required evidence completion.",
            "next_if_required": "Only design a new module if the completed matrix reveals a specific failure mode that the paper needs to address.",
            "stop_condition": "Do not start module training before selecting a concrete failure slice.",
        },
        {
            "candidate": "Field/state-balanced loss",
            "original_priority": "Priority 1 recommended",
            "current_evidence": "All-methods-fail rows include rare/high-null cases, but no direct loss experiment exists.",
            "decision": "possible_future_appendix",
            "reason": "Plausible, but not motivated strongly enough to run before Phase 4 synthesis.",
            "next_if_required": "Use failure cases to choose field weights; run no-LM first before frozen-LM.",
            "stop_condition": "Do not claim balanced-loss benefit without new training evidence.",
        },
        {
            "candidate": "Field-query bottleneck",
            "original_priority": "Priority 1 backup",
            "current_evidence": "No current evidence that field-query architecture is needed.",
            "decision": "defer",
            "reason": "Would be a new architecture and not directly required by current reviewer-risk cleanup.",
            "next_if_required": "Prototype only after failure cases show field localization/schema binding failure.",
            "stop_condition": "Do not implement as an SPD replacement by inertia.",
        },
        {
            "candidate": "Counterfactual margin training",
            "original_priority": "Priority 1 backup",
            "current_evidence": "Existing counterfactual/schema diagnostics show dependence but no training objective yet.",
            "decision": "defer",
            "reason": "Could address schema grounding, but schema dependency is currently best written as limitation.",
            "next_if_required": "Run as appendix mitigation only after fixed-schema limitation is accepted.",
            "stop_condition": "Do not let diagnostic NLL improvements substitute for downstream AUC evidence.",
        },
        {
            "candidate": "Schema augmentation/canonicalization",
            "original_priority": "Priority 2 optional",
            "current_evidence": "Schema paraphrase/order dependence is strongly quantified.",
            "decision": "optional_future_mitigation",
            "reason": "Useful mitigation, but user priority is not to add new variants now.",
            "next_if_required": "Run order/alias augmentation only as mitigation appendix.",
            "stop_condition": "Do not claim robustness before mitigation is trained/evaluated.",
        },
        {
            "candidate": "New SPD variants",
            "original_priority": "explicitly out of scope",
            "current_evidence": "SPD historical variants are sensitivity/negative-baseline rows only.",
            "decision": "forbidden",
            "reason": "User explicitly said not to continue SPD new variants.",
            "next_if_required": "None.",
            "stop_condition": "Stop if work drifts into SPD.",
        },
    ]


def build_checklist_rows() -> list[dict[str, str]]:
    return [
        {
            "section": "Title / abstract",
            "action": "downgrade",
            "required_language": "Answerability-aware structured UMS supervision with controlled frozen-LM analysis.",
            "avoid_language": "Frozen-LM semantic teacher is broadly necessary or 500x data efficient.",
            "evidence_paths": "outputs/final_tables/llm_necessity.csv",
        },
        {
            "section": "Main results",
            "action": "center UMS",
            "required_language": "UMS/schema is stable and works without LM; frozen-LM is a modest in-domain addition.",
            "avoid_language": "All gains come from LLM semantics.",
            "evidence_paths": "outputs/final_tables/main_controlled_results.csv",
        },
        {
            "section": "Low-data analysis",
            "action": "report as mixed/negative",
            "required_language": "1k/3k/10k/30k matched LP matrix is complete and mixed: frozen-LM has small final-AUC gains at 3k/10k, while no-LM is stronger at 1k and 30k and often under best-AUC policy.",
            "avoid_language": "frozen-LM is better in low data.",
            "evidence_paths": "outputs/final_tables/data_scaling_1k_matched_deltas.csv; outputs/final_tables/data_scaling_3k_current_progress.csv; outputs/final_tables/data_scaling_10k_progress.csv; outputs/final_tables/data_scaling_30k_progress.csv",
        },
        {
            "section": "Schema complexity",
            "action": "label diagnostic",
            "required_language": "no-LM and frozen-LM S1/S2/S3 formal source+LP rows are available under the fixed split; source-loss and LP macro-AUC must be interpreted separately.",
            "avoid_language": "frozen-LM uniformly dominates no-LM across all schema levels.",
            "evidence_paths": "outputs/final_tables/schema_complexity_diagnostic_summary.csv; outputs/final_tables/schema_s1_coverage_audit.csv",
        },
        {
            "section": "Failure cases",
            "action": "use qualitatively",
            "required_language": "Frozen-LM/no-LM disagreements are balanced; use examples but defer new modules.",
            "avoid_language": "Frozen-LM wins most hard cases.",
            "evidence_paths": "outputs/final_tables/failure_case_summary.csv",
        },
        {
            "section": "Answerability",
            "action": "separate semantics from AUC",
            "required_language": "Null-as-negative is dense classification supervision and changes missingness semantics.",
            "avoid_language": "Answerability mask is invalid because null-as-negative has higher AUC.",
            "evidence_paths": "outputs/final_tables/answerability_semantics.csv",
        },
        {
            "section": "Limitations",
            "action": "state explicitly",
            "required_language": "Fixed schema serialization dependence, 1k frozen-LM source early-stop provenance caveat, and training-time LM cost.",
            "avoid_language": "schema-agnostic robustness or resource-friendly training.",
            "evidence_paths": "outputs/final_tables/schema_dependency_diagnostics.csv; outputs/final_tables/cost_table.csv",
        },
    ]


def main() -> None:
    required = [
        "outputs/final_tables/main_controlled_results.csv",
        "outputs/final_tables/grouped_field_results.csv",
        "outputs/final_tables/data_scaling_1k_matched_deltas.csv",
        "outputs/final_tables/data_scaling_10k_progress.csv",
        "outputs/final_tables/data_scaling_30k_progress.csv",
        "outputs/final_tables/schema_complexity_diagnostic_summary.csv",
        "outputs/final_tables/failure_case_summary.csv",
        "outputs/final_tables/answerability_semantics.csv",
        "outputs/final_tables/schema_dependency_diagnostics.csv",
        "outputs/final_tables/revision_completion_gap_audit.csv",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required synthesis inputs: {missing}")

    llm_rows = build_llm_rows()
    module_rows = build_module_rows()
    checklist_rows = build_checklist_rows()

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(FINAL_DIR / "llm_necessity.csv", llm_rows, LLM_COLUMNS)
    write_csv(FINAL_DIR / "module_candidates.csv", module_rows, MODULE_COLUMNS)

    llm_text = "# LLM Necessity Map\n\n"
    llm_text += markdown_table(llm_rows, LLM_COLUMNS)
    (FINAL_DIR / "llm_necessity.md").write_text(llm_text, encoding="utf-8")

    module_text = "# Module Candidates Decision Table\n\n"
    module_text += markdown_table(module_rows, MODULE_COLUMNS)
    (FINAL_DIR / "module_candidates.md").write_text(module_text, encoding="utf-8")

    checklist_text = "# Phase 4 Writing and Claim Checklist\n\n"
    checklist_text += markdown_table(checklist_rows, CHECKLIST_COLUMNS)
    checklist_text += "\n## Bottom Line\n\n"
    checklist_text += (
        "The strongest revision framing is UMS/schema-first with transparent frozen-LM boundary analysis. "
        "Do not start new SPD variants. Keep P2 modules deferred unless a later paper decision requires a focused mitigation experiment.\n"
    )
    (FINAL_DIR / "phase4_writing_claim_checklist.md").write_text(checklist_text, encoding="utf-8")
    print(f"Wrote Phase 4 synthesis tables to {FINAL_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
