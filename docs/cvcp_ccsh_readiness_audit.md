# VIVID-Med CVCP/CCSH Readiness Audit

Machine-readable CSV: `outputs/final_tables/cvcp_ccsh_readiness_audit.csv`

This audit turns the generated requirement ledger into execution categories. It does not mark CVCP/CCSH experiments complete unless the status explicitly says so.

## Status Counts

| Area | Status | Count |
| --- | --- | ---: |
| dataset | available_appendix | 2 |
| dataset | available_conditional | 2 |
| dataset | missing | 2 |
| dataset | open | 10 |
| dataset | partial_image_only | 2 |
| model | available | 5 |
| reusable_artifact | available | 1 |
| reusable_artifact | complete_embedding_level | 1 |
| reusable_artifact | complete_for_prior_protocol | 3 |
| target_output | existing | 7 |
| target_script | exact_exists | 21 |

## Audit Rows

| area | item | status | evidence | notes |
| --- | --- | --- | --- | --- |
| target_script | scripts/audit_false_hard_negatives.py | exact_exists | scripts/audit_false_hard_negatives.py | Exact target-plan entry point exists. |
| target_script | scripts/audit_instruction_leakage_v3.py | exact_exists | scripts/audit_instruction_leakage_v3.py | Exact target-plan entry point exists. |
| target_script | scripts/bootstrap_locked_comparison.py | exact_exists | scripts/bootstrap_locked_comparison.py | Exact target-plan entry point exists. |
| target_script | scripts/eval_ab_swap.py | exact_exists | scripts/eval_ab_swap.py | Exact target-plan entry point exists. |
| target_script | scripts/eval_calibration_auprc.py | exact_exists | scripts/eval_calibration_auprc.py | Exact target-plan entry point exists. |
| target_script | scripts/eval_ccsh_consistency.py | exact_exists | scripts/eval_ccsh_consistency.py | Exact target-plan entry point exists. |
| target_script | scripts/eval_ceq_attention.py | exact_exists | scripts/eval_ceq_attention.py | Exact target-plan entry point exists. |
| target_script | scripts/eval_external_dataset.py | exact_exists | scripts/eval_external_dataset.py | Exact target-plan entry point exists. |
| target_script | scripts/eval_hard_shuffle.py | exact_exists | scripts/eval_hard_shuffle.py | Exact target-plan entry point exists. |
| target_script | scripts/eval_locked_final_suite.py | exact_exists | scripts/eval_locked_final_suite.py | Exact target-plan entry point exists. |
| target_script | scripts/generate_ccsh_statements.py | exact_exists | scripts/generate_ccsh_statements.py | Exact target-plan entry point exists. |
| target_script | scripts/generate_ceq_targets.py | exact_exists | scripts/generate_ceq_targets.py | Exact target-plan entry point exists. |
| target_script | scripts/generate_cvcp_curriculum.py | exact_exists | scripts/generate_cvcp_curriculum.py | Exact target-plan entry point exists. |
| target_script | scripts/generate_sameq_cf_compatible.py | exact_exists | scripts/generate_sameq_cf_compatible.py | Exact target-plan entry point exists. |
| target_script | scripts/generate_shuf_k_cf_compatible.py | exact_exists | scripts/generate_shuf_k_cf_compatible.py | Exact target-plan entry point exists. |
| target_script | scripts/train_ceq_ccsh.py | exact_exists | scripts/train_ceq_ccsh.py | Exact target-plan entry point exists. |
| target_script | scripts/train_hnmb_ccsh.py | exact_exists | scripts/train_hnmb_ccsh.py | Exact target-plan entry point exists. |
| target_script | scripts/train_qwen3vl_cvcp.py | exact_exists | scripts/train_qwen3vl_cvcp.py | Exact target-plan entry point exists. |
| target_script | scripts/train_qwen3vl_sameq_ccsh.py | exact_exists | scripts/train_qwen3vl_sameq_ccsh.py | Exact target-plan entry point exists. |
| target_script | scripts/train_qwen3vl_shufk_ccsh.py | exact_exists | scripts/train_qwen3vl_shufk_ccsh.py | Exact target-plan entry point exists. |
| target_script | scripts/train_vlm_teacher_comparison.py | exact_exists | scripts/train_vlm_teacher_comparison.py | Exact target-plan entry point exists. |
| target_output | outputs/final_tables/casebook.md | existing | outputs/final_tables/casebook.md | Target-plan named final-table artifact. |
| target_output | outputs/final_tables/cost_table.md | existing | outputs/final_tables/cost_table.md | Target-plan named final-table artifact. |
| target_output | outputs/final_tables/cvcp_training_results.md | existing | outputs/final_tables/cvcp_training_results.md | Target-plan named final-table artifact. |
| target_output | outputs/final_tables/external_eval_results.md | existing | outputs/final_tables/external_eval_results.md | Target-plan named final-table artifact. |
| target_output | outputs/final_tables/locked_final_comparison.md | existing | outputs/final_tables/locked_final_comparison.md | Target-plan named final-table artifact. |
| target_output | outputs/final_tables/model_comparison_results.md | existing | outputs/final_tables/model_comparison_results.md | Target-plan named final-table artifact. |
| target_output | outputs/final_tables/module_combo_results.md | existing | outputs/final_tables/module_combo_results.md | Target-plan named final-table artifact. |
| dataset | VinDr-CXR | partial_image_only | data/dataset/vinbigdata_xhlulu_512png | VinBigData/VinDr-derived image package exists, but current audit found no label/bbox CSV in that package. |
| dataset | PadChest | missing |  | No local PadChest directory found in the first dataset audit. |
| dataset | MIMIC-CXR | available_conditional | H:/Xiyao_Wang/000_Public Dataset/mimic-cxr/mimic-cxr | MIMIC can be source or conditional external depending on train split usage. |
| dataset | NIH | available_appendix | data/dataset/NIH Chest X-rays | Local NIH exists, but the plan says NIH is appendix/stress-test, not main external. |
| dataset | VinDr-CXR | partial_image_only | data/dataset/vinbigdata_xhlulu_512png | VinBigData/VinDr-derived image package exists, but current audit found no label/bbox CSV in that package. |
| dataset | PadChest | missing |  | No local PadChest directory found in the first dataset audit. |
| dataset | MIMIC-CXR | available_conditional | H:/Xiyao_Wang/000_Public Dataset/mimic-cxr/mimic-cxr | MIMIC can be source or conditional external depending on train split usage. |
| dataset | NIH | available_appendix | data/dataset/NIH Chest X-rays | Local NIH exists, but the plan says NIH is appendix/stress-test, not main external. |
| dataset | SAMEQ-v2 | open |  | Availability must be checked by the phase V2 data audit. |
| dataset | SHUF-K4-v2 | open |  | Availability must be checked by the phase V2 data audit. |
| dataset | CVCP-progressive | open |  | Availability must be checked by the phase V2 data audit. |
| dataset | CCSH-statements | open |  | Availability must be checked by the phase V2 data audit. |
| dataset | CEQ-statements | open |  | Availability must be checked by the phase V2 data audit. |
| dataset | SAMEQ-CF-20 | open |  | Availability must be checked by the phase V2 data audit. |
| dataset | K4-CF-20-TW | open |  | Availability must be checked by the phase V2 data audit. |
| dataset | CVCP-v3-10k | open |  | Availability must be checked by the phase V2 data audit. |
| dataset | CVCP-v4-10k | open |  | Availability must be checked by the phase V2 data audit. |
| dataset | HNMB-static | open |  | Availability must be checked by the phase V2 data audit. |
| model | Qwen3VL current main | available | H:/Xiyao_Wang/001_models/qwen3-vl-2b-thinking-new; H:/Xiyao_Wang/001_models/Qwen3-VL-4B-Instruct; H:/Xiyao_Wang/001_models/Qwen3-VL-8B-Instruct | Local model-directory audit only; each model still needs a component/GPU smoke before formal comparison. |
| model | InternVL comparator | available | H:/Xiyao_Wang/001_models/InternVL2_5-1B; H:/Xiyao_Wang/001_models/InternVL3_5-1B; H:/Xiyao_Wang/001_models/InternVL3_5-2B; H:/Xiyao_Wang/001_models/InternVL3_5-4B; H:/Xiyao_Wang/001_models/InternVL3_5-8B | Local model-directory audit only; each model still needs a component/GPU smoke before formal comparison. |
| model | LLaVA/Llama vision comparator | available | H:/Xiyao_Wang/001_models/Llama-3.2-11B-Vision-Instruct | Local model-directory audit only; each model still needs a component/GPU smoke before formal comparison. |
| model | Qwen3.5 text scaffold | available | H:/Xiyao_Wang/001_models/Qwen3.5-2B; H:/Xiyao_Wang/001_models/Qwen3.5-4B; H:/Xiyao_Wang/001_models/Qwen3.5-9B | Local model-directory audit only; each model still needs a component/GPU smoke before formal comparison. |
| model | Qwen-Coder scaffold | available | H:/Xiyao_Wang/001_models/Qwen2.5-Coder-7B-Instruct | Local model-directory audit only; each model still needs a component/GPU smoke before formal comparison. |
| reusable_artifact | case-study multiseed stability/downstream | complete_for_prior_protocol | outputs/final_tables/case_study_full_execution_status.csv | 12/12 rows complete for families: SAMEQ-SHUF-3k, SHUF-3k, SHUF-K4, SHUF-TW-clinical. Reusable for baselines, not module-combo completion. |
| reusable_artifact | case-study extra curriculum/embedding/module queue | complete_for_prior_protocol | outputs/final_tables/case_study_extra_execution_status.csv | 13/13 rows complete; includes curriculum-v2 long training and embedding exports. |
| reusable_artifact | module ablation head evidence | complete_embedding_level | outputs/final_tables/module_ablation_results.csv | 6/6 CEQ/AUCH/HNMB/DRA/CCSH/CDCS rows complete; not full end-to-end module-combo training. |
| reusable_artifact | next-stage 39-run package audit | complete_for_prior_protocol | outputs/final_tables/next_stage_completion_audit.csv | completed=1049 |
| reusable_artifact | next-stage run package status | available | outputs/final_tables/next_stage_run_package_status.json | Use for old run package provenance while creating CVCP-specific summary tables. |
