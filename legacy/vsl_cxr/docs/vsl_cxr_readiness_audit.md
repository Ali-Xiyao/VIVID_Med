# VSL-CXR v5 Readiness Audit

Machine-readable CSV: `outputs/final_tables/vsl_cxr_readiness_audit.csv`

This audit is read-only. It classifies current scripts, data artifacts, final tables, models, and external-data buckets for the active VSL-CXR v5 plan.

## Status Counts

| Area | Status | Count |
| --- | --- | ---: |
| dataset | artifact_exists | 2 |
| dataset | candidate_qa_schema | 10 |
| dataset | candidate_vsl_schema | 4 |
| dataset | exists_schema_review_needed | 4 |
| external_data | exists_missing_label_manifest | 1 |
| external_data | exists_needs_label_audit | 3 |
| external_data | missing | 1 |
| final_table | historical_artifacts_exist | 8 |
| model | available_needs_smoke | 5 |
| script | exact_exists | 29 |

## Audit Rows

| area | item | status | evidence | notes |
| --- | --- | --- | --- | --- |
| script | scripts/extract_clinical_statements.py | exact_exists | scripts/extract_clinical_statements.py | Exact v5-named entry point exists. |
| script | scripts/generate_counterfactual_statements.py | exact_exists | scripts/generate_counterfactual_statements.py | Exact v5-named entry point exists. |
| script | scripts/generate_sameq_pairs.py | exact_exists | scripts/generate_sameq_pairs.py | Exact v5-named entry point exists. |
| script | scripts/generate_vsl_4class_labels.py | exact_exists | scripts/generate_vsl_4class_labels.py | Exact v5-named entry point exists. |
| script | scripts/generate_hard_negative_pairs.py | exact_exists | scripts/generate_hard_negative_pairs.py | Exact v5-named entry point exists. |
| script | scripts/mine_hard_negatives_memory_bank.py | exact_exists | scripts/mine_hard_negatives_memory_bank.py | Exact v5-named entry point exists. |
| script | scripts/audit_vsl_data_quality.py | exact_exists | scripts/audit_vsl_data_quality.py | Exact v5-named entry point exists. |
| script | scripts/audit_false_hard_negatives.py | exact_exists | scripts/audit_false_hard_negatives.py | Exact v5-named entry point exists. |
| script | scripts/train_vsl_cxr.py | exact_exists | scripts/train_vsl_cxr.py | Exact v5-named entry point exists. |
| script | scripts/train_vsl_ceq.py | exact_exists | scripts/train_vsl_ceq.py | Exact v5-named entry point exists. |
| script | scripts/train_vsl_ccsh.py | exact_exists | scripts/train_vsl_ccsh.py | Exact v5-named entry point exists. |
| script | scripts/train_vsl_hnmb.py | exact_exists | scripts/train_vsl_hnmb.py | Exact v5-named entry point exists. |
| script | scripts/train_vsl_full.py | exact_exists | scripts/train_vsl_full.py | Exact v5-named entry point exists. |
| script | scripts/train_vlm_teacher_comparison.py | exact_exists | scripts/train_vlm_teacher_comparison.py | Exact v5-named entry point exists. |
| script | scripts/eval_chexpert_lp.py | exact_exists | scripts/eval_chexpert_lp.py | Exact v5-named entry point exists. |
| script | scripts/eval_external_lp.py | exact_exists | scripts/eval_external_lp.py | Exact v5-named entry point exists. |
| script | scripts/eval_vsl_sufficiency.py | exact_exists | scripts/eval_vsl_sufficiency.py | Exact v5-named entry point exists. |
| script | scripts/eval_ccsh_consistency.py | exact_exists | scripts/eval_ccsh_consistency.py | Exact v5-named entry point exists. |
| script | scripts/eval_hard_shuffle.py | exact_exists | scripts/eval_hard_shuffle.py | Exact v5-named entry point exists. |
| script | scripts/eval_calibration.py | exact_exists | scripts/eval_calibration.py | Exact v5-named entry point exists. |
| script | scripts/eval_ceq_attention.py | exact_exists | scripts/eval_ceq_attention.py | Exact v5-named entry point exists. |
| script | scripts/eval_casebook.py | exact_exists | scripts/eval_casebook.py | Exact v5-named entry point exists. |
| script | scripts/eval_locked_final_comparison.py | exact_exists | scripts/eval_locked_final_comparison.py | Exact v5-named entry point exists. |
| script | scripts/build_vsl_results_table.py | exact_exists | scripts/build_vsl_results_table.py | Exact v5-named entry point exists. |
| script | scripts/build_external_results_table.py | exact_exists | scripts/build_external_results_table.py | Exact v5-named entry point exists. |
| script | scripts/build_module_results_table.py | exact_exists | scripts/build_module_results_table.py | Exact v5-named entry point exists. |
| script | scripts/build_case_study_markdown.py | exact_exists | scripts/build_case_study_markdown.py | Exact v5-named entry point exists. |
| script | scripts/build_paper_figures.py | exact_exists | scripts/build_paper_figures.py | Exact v5-named entry point exists. |
| script | scripts/build_cost_table.py | exact_exists | scripts/build_cost_table.py | Exact v5-named entry point exists. |
| dataset | D0 Basic-QA | candidate_qa_schema | outputs/instruction_data/glm_validated/d0_train_validated.jsonl | rows=1000; sample_keys=answer,answer_short,answer_type,answerability,counterfactual_type,evidence_phrase,evidence_source,evidence_span,finding,generation_model,image_path,instruction_id,laterality,location,metadata,quality_flags,question,reject_reason,report,report_text; answer_type=fixed_json_schema:200 |
| dataset | D1 CF-QA | candidate_qa_schema | outputs/instruction_data/glm_validated/d6_hard_cf_3k.jsonl | rows=14333; sample_keys=answer,answer_short,answer_type,certainty,counterfactual_type,evidence_span,finding,image_path,instruction_id,laterality,location,metadata,negative_option,negative_option_source,option_a,option_b,positive_option,quality_flags,question,report; answer_type=counterfactual_choice:186,finding_verification:9,uncertainty:5 |
| dataset | D1 CF-QA | candidate_qa_schema | outputs/instruction_data/next_stage/cf_10k_train.jsonl | rows=49594; sample_keys=answer,answer_short,answer_type,certainty,counterfactual_type,evidence_span,finding,image_path,instruction_id,laterality,location,metadata,negative_option,negative_option_source,option_a,option_b,positive_option,quality_flags,question,report; answer_type=counterfactual_choice:163,finding_verification:26,uncertainty:11 |
| dataset | D2 SAMEQ | candidate_qa_schema | outputs/instruction_data/next_stage/sameq_shuf_3k_train.jsonl | rows=9238; sample_keys=answer,answer_short,answer_type,certainty,evidence_span,finding,hard_negative_expected_answer,hard_negative_image_path,hard_negative_reason,hard_negative_sample_id,image_path,instruction_id,laterality,location,metadata,negative_answer,negative_fact_state,quality_flags,question,report; answer_type=same_question_different_answer:200 |
| dataset | D2 SAMEQ | candidate_qa_schema | outputs/instruction_data/cvcp_ccsh/cvcp_v1_sameq_full_train.jsonl | rows=9238; sample_keys=answer,answer_short,answer_type,certainty,cvcp_run_id,evidence_span,finding,hard_negative_expected_answer,hard_negative_image_path,hard_negative_reason,hard_negative_sample_id,image_path,instruction_id,laterality,location,metadata,negative_answer,negative_fact_state,quality_flags,question; answer_type=same_question_different_answer:200 |
| dataset | D3 SAMEQ-CF | candidate_qa_schema | outputs/instruction_data/cvcp_ccsh/sameq_cf_20_train.jsonl | rows=10000; sample_keys=answer,answer_short,answer_type,certainty,counterfactual_type,cvcp_run_id,evidence_span,finding,hard_negative_expected_answer,hard_negative_image_path,hard_negative_reason,hard_negative_sample_id,image_path,instruction_id,laterality,location,metadata,negative_answer,negative_fact_state,negative_option; answer_type=same_question_different_answer:145,counterfactual_choice:42,finding_verification:11,uncertainty:2 |
| dataset | D3 SAMEQ-CF | artifact_exists | outputs/final_tables/sameq_cf_compatible_manifest.csv |  |
| dataset | D4 SAMEQ-K | candidate_qa_schema | outputs/instruction_data/next_stage/shuf_k4_train.jsonl | rows=14333; sample_keys=answer,answer_short,answer_type,certainty,counterfactual_type,evidence_span,finding,hard_negative_expected_answer,hard_negative_image_path,hard_negative_image_paths,hard_negative_reason,hard_negative_reasons,hard_negative_sample_id,hard_negative_sample_ids,image_path,instruction_id,laterality,location,metadata,negative_option; answer_type=counterfactual_choice:186,finding_verification:9,uncertainty:5 |
| dataset | D4 SAMEQ-K | candidate_qa_schema | outputs/instruction_data/cvcp_ccsh/cvcp_v2_shuf_k4_train.jsonl | rows=5000; sample_keys=answer,answer_short,answer_type,certainty,counterfactual_type,cvcp_run_id,evidence_span,finding,hard_negative_expected_answer,hard_negative_image_path,hard_negative_image_paths,hard_negative_reason,hard_negative_reasons,hard_negative_sample_id,hard_negative_sample_ids,image_path,instruction_id,laterality,location,metadata; answer_type=counterfactual_choice:186,finding_verification:10,uncertainty:4 |
| dataset | D5 SAMEQ-HNMB | candidate_qa_schema | outputs/instruction_data/next_stage/mined_shuf_train.jsonl | rows=14333; sample_keys=answer,answer_short,answer_type,certainty,counterfactual_type,evidence_span,finding,hard_negative_expected_answer,hard_negative_image_path,hard_negative_reason,hard_negative_sample_id,image_path,instruction_id,laterality,location,metadata,negative_option,negative_option_source,option_a,option_b; answer_type=counterfactual_choice:185,finding_verification:10,uncertainty:5 |
| dataset | D5 SAMEQ-HNMB | candidate_qa_schema | outputs/instruction_data/next_stage/selfhard_shuf_train.jsonl | rows=21499; sample_keys=answer,answer_short,answer_type,certainty,counterfactual_type,evidence_span,finding,hard_negative_expected_answer,hard_negative_image_path,hard_negative_reason,hard_negative_sample_id,image_path,instruction_id,laterality,location,metadata,negative_option,negative_option_source,option_a,option_b; answer_type=counterfactual_choice:189,finding_verification:7,uncertainty:4 |
| dataset | D6 VSL-4class | candidate_vsl_schema | outputs/instruction_data/vsl_cxr/d6_vsl_4class_train.jsonl | rows=11149; sample_keys=answer,answer_type,counterfactual_statement,evidence_span,finding,generation_model,image_path,instruction_id,laterality,metadata,negative_image_path,negative_type,quality_flags,question,report_text,sample_id,severity,source,source_instruction_id,source_version; answer_type=support:58,insufficient:51,uncertain:46,contradict:45; sufficiency_label=support:58,insufficient:51,uncertain:46,contradict:45 |
| dataset | D6 VSL-4class | candidate_vsl_schema | outputs/instruction_data/vsl_cxr/d6_vsl_4class_val.jsonl | rows=1600; sample_keys=answer,answer_type,counterfactual_statement,evidence_span,finding,generation_model,image_path,instruction_id,laterality,metadata,negative_image_path,negative_type,quality_flags,question,report_text,sample_id,severity,source,source_instruction_id,source_version; answer_type=uncertain:61,support:53,insufficient:45,contradict:41; sufficiency_label=uncertain:61,support:53,insufficient:45,contradict:41 |
| dataset | D7 VSL-CEQ | exists_schema_review_needed | outputs/instruction_data/cvcp_ccsh/ceq_targets_train.jsonl | rows=14333; sample_keys=evidence_query,expected_region,finding,image_path,sample_id,source_instruction_id,source_version,state,target_id; answer_type=:200 |
| dataset | D8 VSL-CCSH | exists_schema_review_needed | outputs/instruction_data/cvcp_ccsh/ccsh_statements_train.jsonl | rows=28666; sample_keys=binary_label,finding,image_path,label,sample_id,source_instruction_id,source_version,state,statement,statement_id; answer_type=:200 |
| dataset | D9 VSL-full | candidate_vsl_schema | outputs/instruction_data/vsl_cxr/d9_vsl_full_train.jsonl | rows=18000; sample_keys=answer,answer_short,answer_type,certainty,counterfactual_statement,counterfactual_type,cvcp_run_id,d9_component,evidence_span,finding,generation_model,hard_negative_expected_answer,hard_negative_image_path,hard_negative_image_paths,hard_negative_reason,hard_negative_reasons,hard_negative_sample_id,hard_negative_sample_ids,image_path,instruction_id; answer_type=counterfactual_choice:77,same_question_different_answer:50,support:22,insufficient:20,contradict:12,uncertain:9; sufficiency_label=:137,support:22,insufficient:20,contradict:12,uncertain:9 |
| dataset | D9 VSL-full | candidate_vsl_schema | outputs/instruction_data/vsl_cxr/d9_vsl_full_val.jsonl | rows=2000; sample_keys=answer,answer_short,answer_type,certainty,counterfactual_statement,counterfactual_type,d9_component,evidence_span,finding,generation_model,hard_negative_expected_answer,hard_negative_image_path,hard_negative_image_paths,hard_negative_reason,hard_negative_reasons,hard_negative_sample_id,hard_negative_sample_ids,image_path,instruction_id,laterality; answer_type=counterfactual_choice:86,same_question_different_answer:55,insufficient:15,support:15,contradict:11,uncertain:10; sufficiency_label=:149,insufficient:15,support:15,contradict:11,uncertain:10 |
| dataset | D9 VSL-full | exists_schema_review_needed | outputs/instruction_data/vsl_cxr/d9_ceq_targets_train.jsonl | rows=13833; sample_keys=d9_companion_id,d9_component,evidence_query,expected_region,finding,image_path,sample_id,source_companion_id,source_dataset_id,source_instruction_id,source_version,state,target_id,vsl_dataset_id; answer_type=:200 |
| dataset | D9 VSL-full | exists_schema_review_needed | outputs/instruction_data/vsl_cxr/d9_ccsh_pairs_train.jsonl | rows=28166; sample_keys=binary_label,d9_companion_id,d9_component,finding,image_path,label,sample_id,source_companion_id,source_dataset_id,source_instruction_id,source_version,state,statement,statement_id,vsl_dataset_id; answer_type=:200 |
| dataset | D9 VSL-full | artifact_exists | outputs/final_tables/vsl_cxr_d9_full_dataset_manifest.csv |  |
| final_table | training/backbone results | historical_artifacts_exist | outputs/final_tables/cvcp_training_results.csv; outputs/final_tables/next_stage_training_results.csv; outputs/final_tables/sameq_v4_multiseed_training_results.csv | Must be remapped to v5 rows before reuse. |
| final_table | module/readout results | historical_artifacts_exist | outputs/final_tables/module_combo_results.csv; outputs/final_tables/module_ablation_results.csv; outputs/final_tables/ccsh_consistency_results.csv; outputs/final_tables/ceq_consistency_results.csv | Must be remapped to v5 rows before reuse. |
| final_table | external results | historical_artifacts_exist | outputs/final_tables/external_eval_results.csv; outputs/final_tables/sameq_v4_multiseed_external_eval_results.csv; outputs/final_tables/nih_available_transfer_status.csv | Must be remapped to v5 rows before reuse. |
| final_table | calibration | historical_artifacts_exist | outputs/final_tables/cvcp_calibration_auprc.csv; outputs/final_tables/next_stage_calibration_auprc.csv; outputs/final_tables/sameq_v4_multiseed_calibration_auprc.csv | Must be remapped to v5 rows before reuse. |
| final_table | hard shuffle | historical_artifacts_exist | outputs/final_tables/cvcp_hard_shuffle_results.csv; outputs/final_tables/sameq_v4_multiseed_hard_shuffle_results.csv; outputs/final_tables/visual_dependence_results.csv | Must be remapped to v5 rows before reuse. |
| final_table | casebook | historical_artifacts_exist | outputs/final_tables/casebook.md; outputs/final_tables/case_study_summary.md; outputs/final_tables/next_stage_qualitative_cases.md | Must be remapped to v5 rows before reuse. |
| final_table | cost | historical_artifacts_exist | outputs/final_tables/cost_table.csv; outputs/final_tables/qwen3vl_cost_table.csv; outputs/final_tables/sameq_v4_multiseed_cost_table.csv | Must be remapped to v5 rows before reuse. |
| final_table | locked comparison | historical_artifacts_exist | outputs/final_tables/locked_final_comparison.csv; outputs/final_tables/sameq_v4_multiseed_locked_final_comparison.csv; outputs/final_tables/bootstrap_locked_comparison.csv | Must be remapped to v5 rows before reuse. |
| model | Qwen3-VL | available_needs_smoke | H:/Xiyao_Wang/001_models/qwen3-vl-2b-thinking-new; H:/Xiyao_Wang/001_models/Qwen3-VL-4B-Instruct; H:/Xiyao_Wang/001_models/Qwen3-VL-8B-Instruct | Directory presence only; formal teacher rows require adapter/GPU smoke. |
| model | InternVL | available_needs_smoke | H:/Xiyao_Wang/001_models/InternVL3_5-1B; H:/Xiyao_Wang/001_models/InternVL3_5-2B; H:/Xiyao_Wang/001_models/InternVL3_5-4B; H:/Xiyao_Wang/001_models/InternVL3_5-8B | Directory presence only; formal teacher rows require adapter/GPU smoke. |
| model | LLaVA/Llama-based VLM | available_needs_smoke | H:/Xiyao_Wang/001_models/Llama-3.2-11B-Vision-Instruct | Directory presence only; formal teacher rows require adapter/GPU smoke. |
| model | Qwen3.5 text scaffold | available_needs_smoke | H:/Xiyao_Wang/001_models/Qwen3.5-0.8B; H:/Xiyao_Wang/001_models/Qwen3.5-2B; H:/Xiyao_Wang/001_models/Qwen3.5-4B; H:/Xiyao_Wang/001_models/Qwen3.5-9B | Directory presence only; formal teacher rows require adapter/GPU smoke. |
| model | Qwen-Coder scaffold | available_needs_smoke | H:/Xiyao_Wang/001_models/Qwen2.5-Coder-7B-Instruct | Directory presence only; formal teacher rows require adapter/GPU smoke. |
| external_data | VinBigData/VinDr derived images | exists_needs_label_audit | data/dataset/vinbigdata_xhlulu_512png; data/dataset/vinbigdata_xhlulu_512png/train_meta.csv | v5 main-external eligibility depends on labels/manifest and training-overlap audit. Current local VinBigData package exposes image files and train_meta.csv, but no class label CSV was found. |
| external_data | PadChest | missing | data/dataset/PadChest; H:/Xiyao_Wang/000_Public Dataset/PadChest | No local directory found at audited paths. |
| external_data | MIMIC-CXR | exists_missing_label_manifest | H:/Xiyao_Wang/000_Public Dataset/mimic-cxr | v5 main-external eligibility depends on labels/manifest and training-overlap audit. |
| external_data | NIH | exists_needs_label_audit | data/dataset/NIH Chest X-rays; H:/Xiyao_Wang/000_Public Dataset/NIH Chest X-rays; data/dataset/NIH Chest X-rays/Data_Entry_2017.csv; H:/Xiyao_Wang/000_Public Dataset/NIH Chest X-rays/Data_Entry_2017.csv | v5 main-external eligibility depends on labels/manifest and training-overlap audit. |
| external_data | CheXpert | exists_needs_label_audit | data/dataset/CheXpert-v1.0-small; H:/Xiyao_Wang/000_Public Dataset/CheXpert-v1.0-small; data/dataset/CheXpert-v1.0-small/valid.csv; H:/Xiyao_Wang/000_Public Dataset/CheXpert-v1.0-small/valid.csv | v5 main-external eligibility depends on labels/manifest and training-overlap audit. |
