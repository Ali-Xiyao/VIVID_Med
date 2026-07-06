# VIVID-Med Docs Index

This directory is the handoff entry point for active research state, requirement ledgers, and boundary notes. Generated experiment artifacts remain under `outputs/` and are intentionally ignored by Git.

## Active Entry Points

| Purpose | File |
| --- | --- |
| Current SAMEQ-CVCP paper-ready experiment plan and final write-back | `../vivid_med_sameq_cvcp_next_experiment_plan_v4.md` |
| Prior CVCP/CCSH full-plan closure reference | `../vivid_med_cvcp_ccsh_full_next_experiment_plan.md` |
| CVCP/CCSH requirement ledger | `cvcp_ccsh_requirement_ledger.md` |
| CVCP/CCSH readiness audit | `cvcp_ccsh_readiness_audit.md` |
| Current task checklist and execution boundaries | `../task_plan.md` |
| Durable findings and interpretation boundaries | `../findings.md` |
| Chronological progress log | `../progress.md` |

## Current Result Artifacts

| Purpose | Generated artifact |
| --- | --- |
| CVCP/CCSH training result table | `../outputs/final_tables/cvcp_training_results.md` |
| CVCP/CCSH postprocess status | `../outputs/final_tables/cvcp_ccsh_postprocess_status.md` |
| CVCP/CCSH module-combo result table | `../outputs/final_tables/module_combo_results.md` |
| CVCP/CCSH model comparison audit | `../outputs/final_tables/model_comparison_results.md` |
| CVCP/CCSH external-evaluation boundary/results | `../outputs/final_tables/external_eval_results.md` |
| CVCP/CCSH completion audit | `../outputs/final_tables/cvcp_ccsh_completion_audit.md` |
| SAMEQ v4 pure-seed live manifest | `../outputs/final_tables/sameq_v4_multiseed_manifest.md` |
| SAMEQ v4 pure-seed stability/status snapshot | `../outputs/final_tables/sameq_v4_multiseed_stability.md` |
| Prior case-study real-run status for seed stability + downstream diagnostics | `../outputs/final_tables/case_study_full_execution_status.md` |
| Prior NIH available/full transfer status with 1k/5k/all subset rows | `../outputs/final_tables/nih_available_transfer_status.md` |
| Prior extra queue status for curriculum-v2, embeddings, MMD/UMAP, and modules | `../outputs/final_tables/case_study_extra_execution_status.md` |
| Embedding-backed domain shift MMD | `../outputs/final_tables/domain_shift_mmd.md` |
| Embedding-backed dataset projection report | `../outputs/final_tables/dataset_embedding_projection.md` |
| Formal module ablation result table | `../outputs/final_tables/module_ablation_results.md` |
| Curriculum-v2 formal training package | `../outputs/qwen3vl_case_study_modules/cur_v2_progressive_replay/metrics_final.json` |
| Case-study summary and casebook | `../outputs/final_tables/case_study_summary.md` |
| Multi-seed manifest and legacy stability summary | `../outputs/final_tables/multiseed_stability.md` |
| NIH label/domain diagnosis | `../outputs/final_tables/nih_domain_audit.md` |
| TMI module smoke and experiment matrix | `../outputs/final_tables/module_candidate_results.md` |
| Conservative family-level comparison | `../outputs/final_tables/locked_final_comparison.md` |
| Completion audit | `../outputs/final_tables/case_study_modules_completion_audit.md` |

## Other Ledgers

| File | Scope |
| --- | --- |
| `next_stage_requirement_ledger.md` | Completed next-stage Qwen3-VL experiment suite and final audit. |
| `next_stage_external_model_availability.md` | Local model and external dataset availability boundaries. |
| `clinical_instruction_schema.md` | Clinical instruction JSONL schema and validation rules. |

## Structure Notes

- Keep medical datasets, generated instructions, model checkpoints, and experiment outputs under ignored paths such as `data/dataset/`, `data/instructions/`, `data/splits/`, `outputs/`, and `History/`.
- Keep source scripts, configs, module code, and concise docs in tracked project paths.
- Preserve failed or impossible runs as explicit case-study or boundary artifacts instead of silently deleting them.
