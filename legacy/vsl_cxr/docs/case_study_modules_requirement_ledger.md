# VIVID-Med Case Study + Module Plan Requirement Ledger

Source plan: `vivid_med_case_study_modules_next_experiment_plan.md`

This ledger tracks artifact-backed completion for the case-study/module plan. Previous next-stage results are reusable baseline evidence, but they do not close a row unless the artifact named here exists or the boundary is explicitly documented.

## Scope Boundary

The plan requires a shift from post-hoc best-row selection toward:

1. case study / failure mining,
2. multi-seed and paired uncertainty evidence,
3. NIH label/domain diagnosis,
4. fair curriculum-v2 retry,
5. TMI-friendly deployable modules,
6. family-level finalist selection and locked comparison.

## Required Source Artifacts

| Group | Artifact | Status | Notes |
|---|---|---|---|
| Case study | `scripts/mine_pairwise_case_studies.py` | completed | Generated SHUF-TW vs SHUF wins/losses from existing summary/prediction artifacts where available. |
| Case study | `scripts/audit_nih_transfer_failure.py` | completed | Per-label and method-disagreement NIH failure audit. |
| Case study | `scripts/audit_hard_negative_quality.py` | completed | False-negative / weak-negative audit from SHUF/SAMEQ/K data. |
| Case study | `scripts/audit_curriculum_leakage_cases.py` | completed | Curriculum leakage/stage-failure case mining from instruction audits. |
| Case study | `scripts/build_casebook_markdown.py` | completed | Consolidated CSV/JSON case rows into Markdown casebooks. |
| Stability | `scripts/run_multiseed_manifest.py` | completed | Produced executable seed configs/manifests; the reopened real-run queue has now completed 12 family/seed packages. |
| Stability | `scripts/bootstrap_auc_ci.py` | completed | Bootstrap CI utility implemented for prediction/summary inputs. |
| Stability | `scripts/paired_bootstrap_method_delta.py` | completed | Produced candidate-baseline summary delta with paired-prediction boundary. |
| Stability | `scripts/summarize_multiseed_results.py` | completed | Produced `multiseed_stability.md/csv` with single-seed reference and pending seed slots separated. |
| NIH/domain | `scripts/audit_label_mapping_nih.py` | completed | CheXpert/MIMIC-to-NIH mapping confidence/risk audit. |
| NIH/domain | `scripts/run_nih_full_transfer.py` | completed | Generated NIH transfer manifest; reopened real-run queue completed NIH `all_available=25596` transfer for all 12 seed rows. |
| NIH/domain | `scripts/compute_domain_shift_mmd.py` | completed | MMD/norm/variance script plus real embedding-backed MMD report. |
| NIH/domain | `scripts/plot_dataset_embedding_umap.py` | completed | Projection script plus real embedding-backed projection report and PNG. |
| Curriculum v2 | `scripts/build_curriculum_v2_schedule.py` | completed | Progressive replay and hard-negative schedule definitions. |
| Curriculum v2 | `scripts/generate_curriculum_v2_instructions.py` | completed | Materialized curriculum-v2 rows from existing next-stage JSONL. |
| Curriculum v2 | `scripts/train_qwen3vl_curriculum_v2.py` | completed | Thin wrapper around existing trainer with curriculum-v2 defaults and manifest generation. |
| Module | `models/clinical_evidence_query.py` | completed | CEQ module. |
| Module | `models/answerability_uncertainty_head.py` | completed | AUCH module. |
| Module | `models/hard_negative_memory_bank.py` | completed | HNMB module. |
| Module | `models/domain_robust_adapter.py` | completed | DRA module. |
| Module | `models/clinical_consistency_head.py` | completed | CCSH module. |
| Module | `models/case_driven_curriculum_scheduler.py` | completed | CDCS module. |

## Required Report Artifacts

| Artifact | Status | Required contents |
|---|---|---|
| `outputs/final_tables/case_study_shuf_tw_vs_shuf.md` | completed | CS1/CS2 pairwise wins/losses and failure taxonomy. |
| `outputs/final_tables/case_study_nih_transfer.md` | completed | NIH disagreement/failure cases and likely causes. |
| `outputs/final_tables/case_study_curriculum_failure.md` | completed | Curriculum leakage/stage-failure cases. |
| `outputs/final_tables/case_study_hard_negative_quality.md` | completed | False hard-negative / weak-negative audit. |
| `outputs/final_tables/case_study_summary.csv` | completed | Machine-readable combined case-study table. |
| `outputs/final_tables/case_study_summary.md` | completed | Reader-facing summary. |
| `outputs/final_tables/multiseed_stability.md` | completed | Seed/CI report with explicit no-new-seed boundary. |
| `outputs/final_tables/nih_domain_audit.md` | completed | Label mapping, per-label NIH metrics, domain shift, protocol sanity. |
| `outputs/final_tables/module_candidate_results.md` | completed | CEQ/AUCH/HNMB/DRA/CCSH/CDCS smoke evidence and experiment matrix. |
| `outputs/final_tables/case_study_full_execution_status.md` | completed | Real 12-row stability/downstream execution table. |
| `outputs/final_tables/case_study_extra_execution_status.md` | completed | Real curriculum/embedding/MMD/projection/module queue status table. |
| `outputs/final_tables/module_ablation_results.md` | completed | Formal CEQ/AUCH/HNMB/DRA/CCSH/CDCS ablation metrics. |
| `outputs/final_tables/locked_final_comparison.md` | completed | Family finalists, primary endpoints, safety gates, and final role. |

## Reusable Existing Evidence

| Existing artifact | Use in this plan | Boundary |
|---|---|---|
| `outputs/final_tables/next_stage_decision_summary.csv` | Main source for current candidate metrics. | Single-run result table; not seed stability. |
| `outputs/final_tables/next_stage_lp_transfer_results.csv` | CheXpert/NIH AUC/AUPRC/ECE/F1 for 39 runs. | NIH is 1k unless full/available transfer is rerun or bounded. |
| `outputs/final_tables/next_stage_visual_dependence.csv` | Image-shuffle and hard-shuffle deltas. | Diagnostic summary only; case scripts must mine examples separately where possible. |
| `outputs/final_tables/next_stage_counterfactual.csv` | CF accuracy and NLL deltas. | P2-style zero-row cases require explicit N/A boundary. |
| `outputs/final_tables/next_stage_ab_swap_counterfactual.csv` | A/B-swap accuracy. | Same boundary as CF. |
| `outputs/final_tables/next_stage_instruction_audit.csv` | Leakage/flag and A/B balance. | Supports curriculum/leakage mining. |
| `outputs/final_tables/next_stage_qualitative_cases.md` | Existing qualitative hard-negative assets. | Can seed casebook, but target casebooks need plan-specific taxonomy. |

## 2026-07-02 Real-run Refresh

The user reopened all rows that had previously been closed only by manifests, boundary reports, or smoke tests. Those rows now have real-run evidence:

- Multi-seed stability: 12/12 required family/seed rows completed 5000-step training and downstream diagnostics; see `outputs/final_tables/case_study_full_execution_status.md`.
- NIH transfer: all 12 rows completed NIH available transfer on 25,596 records with 1k/5k/all subset metrics.
- Domain MMD/projection: CheXpert/NIH embeddings were exported and the old no-embedding boundary reports were force-rerun from real embedding files.
- Curriculum v2: `cur_v2_progressive_replay` completed 12,000-step training and wrote `metrics_final.json`.
- Module ablation: CEQ/AUCH/HNMB/DRA/CCSH/CDCS completed formal 1000-step ablation training; see `outputs/final_tables/module_ablation_results.md`.

Remaining interpretation boundaries:

- NIH full means current UMS `all_available=25596`; a different raw-NIH full definition requires a larger manifest.
- SAMEQ CF/A-B option metrics are not applicable with the current same-question/different-answer data format.
- Module ablations are embedding-level module-head evidence, not yet full end-to-end final method evidence.

## Completion Checklist

- [x] All source artifacts are implemented.
- [x] All source artifacts pass syntax/import or targeted smoke tests.
- [x] All report artifacts are generated.
- [x] Requirement ledger statuses are refreshed from disk state.
- [x] `task_plan.md`, `findings.md`, and `progress.md` are updated after each phase.
- [x] `vivid_med_case_study_modules_next_experiment_plan.md` receives final results/status write-back.
- [x] Final verification includes script compilation, smoke tests, report existence, and a requirement audit.
