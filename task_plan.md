# VIVID-Med CVCP/CCSH Full Experiment Execution Plan

## Active Scope

Execute every required and optional experiment from `vivid_med_cvcp_ccsh_full_next_experiment_plan.md` using the document's formal framing: Clinical Visual Curriculum Pretraining (CVCP) with deployable consistency/evidence modules. Prior next-stage, case-study, and upload artifacts are reusable baseline evidence only when their run IDs, data, training protocol, diagnostics, and table semantics match this document. Empty planning tables in the target document are not completion evidence.

## Goal

Produce artifact-backed completion for the full CVCP/CCSH plan: data/external selection, leakage and false-negative audits, core baselines, multi-seed stability, paired bootstrap/CI, CVCP curriculum variants, CCSH/CEQ/HNMB/AUCH/DRA/CDCS module combinations, SHUF++/CF-compatible repairs, token weighting, VLM teacher comparisons, external evaluation, casebooks/visualizations, final waves, and a locked final comparison. Write final results and limitations back into `vivid_med_cvcp_ccsh_full_next_experiment_plan.md`.

## Current Phase

- [x] Phase V0: Skill pre-flight, memory lookup, session catchup, UTF-8 target-plan read, branch/GPU snapshot, and takeover of old planning state.
- [x] Phase V1: Build a requirement ledger from every table/run/script/output named in `vivid_med_cvcp_ccsh_full_next_experiment_plan.md`.
- [x] Phase V2: Audit current source scripts, configs, data availability, model availability, old artifacts, and exact gaps against the ledger.
- [x] Phase V3: Implement or adapt missing data generation and audit scripts:
  - [x] `scripts/generate_cvcp_curriculum.py`
  - [x] `scripts/generate_sameq_cf_compatible.py`
  - [x] `scripts/generate_shuf_k_cf_compatible.py`
  - [x] `scripts/generate_ccsh_statements.py`
  - [x] `scripts/generate_ceq_targets.py`
  - [x] `scripts/audit_instruction_leakage_v3.py`
  - [x] `scripts/audit_false_hard_negatives.py`
- [x] Phase V4: Implement or adapt formal training entry points and two-3090 queues for all document runs, including first wave, second wave, final wave, model-comparison, and optional module-stack experiments.
- [x] Phase V5: Run data preparation, leakage/false-negative audits, and external-dataset decision protocol. Preserve unavailable VinDr/PadChest/label gaps as explicit boundary rows rather than silently substituting NIH.
- [x] Phase V6: Run core baselines and seed stability: Base-Qwen3VL, SHUF-3k, SAMEQ-SHUF, SHUF-K4, SHUF-TW-clinical, P2-value-only, and P2-field-query, with required diagnostics and paired CI.
- [x] Phase V7: Run CVCP curriculum variants: v1 SAMEQ, v2 SHUF-K, v3 progressive, v4 replay, v5 case-driven/CDCS, plus full/scale variants where data exists.
- [x] Phase V8: Run module combination experiments: Base+CCSH, SHUF/SAMEQ/K4/CVCP+CCSH, CEQ variants, HNMB variants, AUCH variants, DRA conditional variants, and CEQ+HNMB+CCSH full module stack.
- [x] Phase V9: Run SHUF++/CF-compatible repair, dual-margin, and token-weighting experiments, including all table rows in Phases 4 and 5 of the target document.
- [x] Phase V10: Run VLM teacher comparison using the same data/curriculum/frozen policy where locally available, including Qwen3VL, InternVL/LLaVA or explicit unavailable boundaries, and text-only scaffold controls.
- [x] Phase V11: Run external evaluation, calibration/AUPRC, casebooks, attention/visualization outputs, cost table, and failure-mode taxonomy.
- [x] Phase V12: Build final summaries: CVCP training results, module combo results, model comparison, external eval, casebook, cost, data audit, seed/CI, and locked final comparison.
- [x] Phase V13: Write final closure and populated tables into `vivid_med_cvcp_ccsh_full_next_experiment_plan.md`; refresh docs/README and requirement ledgers.
- [x] Phase V14: Final verification after the last edit: ledger rows closed or explicitly bounded, output files current, two local 3090s/process state audited, and no prose-only completion claims.

## Execution Resources

- Local GPUs are approved by the user for this goal: GPU0 and GPU1 are both NVIDIA GeForce RTX 3090 24 GiB.
- Initial snapshot on 2026-07-04T14:20:10+08:00 showed both GPUs idle at `0 MiB` and `0%` utilization.
- Formal CVCP training launched at 2026-07-04T14:43:03+08:00:
  - GPU0 training queue PID `31416`, first run `cvcp_v1_sameq_3k`.
  - GPU1 training queue PID `11496`, first run `cvcp_v1_sameq_10k`.
- CVCP postprocess watchers launched at 2026-07-04T14:50:22+08:00:
  - GPU0 postprocess PID `2324`.
  - GPU1 postprocess PID `23840`.
- CVCP module-combo watchers launched at 2026-07-04T15:00:14+08:00 with conservative idle-window gating:
  - GPU0 module PID `31452`.
  - GPU1 module PID `22352`.
- Recovery snapshot at 2026-07-04T18:04:05+08:00: the first row `cvcp_v1_sameq_3k` is complete end-to-end; `cvcp_v1_sameq_10k` training is complete but LP postprocess was interrupted before `metrics_final.json`; training lanes for `cvcp_v1_sameq_full` and `cvcp_v2_shuf_k2` were externally interrupted with exit code `1073807364` and have no checkpoint because the new configs use final-only checkpointing.
- `scripts/run_cvcp_ccsh_training_queue.ps1` now archives any no-`metrics_final.json`/no-checkpoint training output directory into the F-drive `interrupted_runs/` evidence area before rerunning that run from scratch. Routine manual monitoring follows the user's latest request: after each status check, run a direct PowerShell sleep (`Start-Sleep -Seconds 7200`) and then check again. Do not set up automations or schedulers for this monitoring loop.
- Relaunch snapshot at 2026-07-04T18:05:37+08:00: training PIDs `22440`/`21372`, postprocess PIDs `11808`/`21488`, module-combo PIDs `20656`/`4480`. The rerun health check at 2026-07-04T18:09:30+08:00 showed `cvcp_v1_sameq_full` and `cvcp_v2_shuf_k2` actively logging new train steps and GPU memory below capacity.
- Current branch is `codex/vivid-med-p0-consolidation`; the worktree already contains substantial existing tracked and untracked changes from earlier project work. Do not revert unrelated changes.

## Evidence Rules

- A row is complete only when the exact run package, config/resolved config, metrics, logs, diagnostics, final tables, and summary/ledger rows exist or the row is explicitly marked impossible/unavailable with evidence.
- Old next-stage/case-study results can seed baselines, but cannot close new CVCP/CCSH rows unless the protocol and run identity match the current document.
- Script/config manifests alone are not completion for training or evaluation rows.
- External evaluation must follow the document's decision rules. NIH can remain appendix/stress-test; it cannot be silently promoted to the main external dataset if VinDr-CXR/PadChest remain unavailable.
- SAMEQ zero-row A/B option diagnostics must be recorded as not applicable, not missing or zero, when the format genuinely has no option-pairwise records.
- Two-GPU concurrency is allowed, but long training/postprocess queues need owner boundaries and memory guards to avoid duplicate writers or killing unrelated work.
- Medical-image data and generated outputs stay under ignored `data/`/`outputs/` paths unless the user explicitly asks to version artifacts.
- Final completion requires fresh post-edit verification using `verification-before-completion`.

## Errors Encountered

| Time | Error | Attempt | Resolution |
| --- | --- | --- | --- |
| 2026-07-04T14:18:00+08:00 | Existing active goal prevented a new `create_goal` call. | User invoked `/goal` with the CVCP/CCSH target. | `get_goal` confirmed the existing active goal already matches this request; execution continues under it. |
| 2026-07-04T14:20:10+08:00 | A first PowerShell one-liner for checking target-plan script existence failed with `An empty pipe element is not allowed`. | Tried to pipe a `foreach` statement directly to `Format-Table`. | Record the error and rerun the audit with a table variable before making script-existence claims. |
| 2026-07-04T14:24:00+08:00 | Broad `rg` over scripts/configs/docs/outputs timed out while looking for every possible CVCP/CCSH reference. | Ran a wide multi-pattern scan across many directories. | Switched to generated-ledger-driven targeted audits. |
| 2026-07-04T17:39:00+08:00 | Goal-continuation/tool interruption closed active CVCP process trees. `cvcp_v1_sameq_full`, `cvcp_v2_shuf_k2`, and `cvcp_v1_sameq_10k` LP exited with code `1073807364` before final artifacts. | Audited logs, output directories, and GPU state. `cvcp_v1_sameq_full` had progressed to about step 3450 and `cvcp_v2_shuf_k2` to step 500, but neither had a checkpoint because configs intentionally save final-only checkpoints. | Patched `scripts/run_cvcp_ccsh_training_queue.ps1` to archive no-metrics/no-checkpoint partial training directories under `interrupted_runs/` before clean rerun; LP can rerun in-place because `train_qwen3vl_vision_lp.py` only refuses existing `metrics_final.json`. |

## CVCP/CCSH Ledger Snapshot

| Area | Snapshot | Evidence |
| --- | --- | --- |
| Requirement rows | 239 rows generated from the target document. | `docs/cvcp_ccsh_requirement_ledger.md`, `outputs/final_tables/cvcp_ccsh_requirement_ledger.csv` |
| Target scripts | 21/21 target-plan script names are currently missing as exact files. | Requirement rows `CVCP-0001` through `CVCP-0021` |
| Target final tables | 2 existing, 5 missing by exact path. | `outputs/final_tables/locked_final_comparison.md`, `outputs/final_tables/cost_table.md`; missing CVCP/module/model/external/casebook tables |
| Data availability | NIH is local but appendix-only; MIMIC is conditionally available; PadChest is missing; VinBig/VinDr-derived image package is partial image-only. | Requirement rows `CVCP-0029` through `CVCP-0036` |
| Experiment rows | 36 have name-like candidate evidence that needs protocol audit; 143 remain open. | `outputs/final_tables/cvcp_ccsh_requirement_ledger.csv` |

## CVCP/CCSH Readiness Snapshot

| Area | Snapshot | Evidence |
| --- | --- | --- |
| Readiness audit | 56 rows generated; all 21 missing target scripts have reusable analogs but no exact target-named entry point. | `docs/cvcp_ccsh_readiness_audit.md`, `outputs/final_tables/cvcp_ccsh_readiness_audit.csv` |
| Model comparison | Qwen3VL, InternVL, Llama vision, Qwen3.5 text scaffold, and Qwen-Coder scaffold directories exist locally. | `H:/Xiyao_Wang/001_models/*`; each still needs model-specific component/GPU smoke before formal training. |
| Reusable baselines | 12/12 case-study multiseed rows are complete for prior protocol families: SHUF-3k, SHUF-TW-clinical, SAMEQ-SHUF-3k, SHUF-K4. | `outputs/final_tables/case_study_full_execution_status.csv` |
| Reusable modules | CEQ/AUCH/HNMB/DRA/CCSH/CDCS completed embedding-level 1000-step ablations. | `outputs/final_tables/module_ablation_results.csv`; not full end-to-end module-combo evidence. |
| Reusable next-stage suite | Previous 39-run next-stage audit is complete with `completed=1049`. | `outputs/final_tables/next_stage_completion_audit.csv`; old protocol only. |

## CVCP/CCSH V3 Implementation Snapshot

| Area | Snapshot | Evidence |
| --- | --- | --- |
| Target scripts | 21/21 exact target-plan script paths now exist and compile. | `scripts/cvcp_ccsh_driver.py` plus target wrappers under `scripts/` |
| Generated CVCP data | 14 CVCP curriculum datasets materialized. | `outputs/final_tables/cvcp_dataset_manifest.md` |
| CF-compatible data | 3 SAMEQ-CF datasets and 3 SHUF-K-CF datasets materialized. | `outputs/final_tables/sameq_cf_compatible_manifest.md`, `outputs/final_tables/shuf_k_cf_compatible_manifest.md` |
| Module data | CCSH statement data and CEQ target data materialized. | `outputs/final_tables/ccsh_statement_manifest.md`, `outputs/final_tables/ceq_target_manifest.md` |
| Data audits | Leakage v3 and false hard-negative audits ran on generated key datasets. | `outputs/final_tables/cvcp_instruction_leakage_v3.md`, `outputs/final_tables/cvcp_false_hard_negative_audit.md` |
| Formal target outputs | Target named casebook, CVCP training, module combo, model comparison, external eval, locked comparison, cost table outputs now exist. | `outputs/final_tables/{casebook,cvcp_training_results,module_combo_results,model_comparison_results,external_eval_results,locked_final_comparison,cost_table}.md` |
| Formal training queue | 27 Qwen3VL CVCP/CF-compatible/dual-margin/TW configs generated and queued across two 3090s. | `outputs/final_tables/cvcp_ccsh_training_manifest.md`, `scripts/run_cvcp_ccsh_training_queue.ps1` |
| Postprocess queue | CVCP-specific LP/NIH-appendix/visual/CF/A-B/paraphrase watcher generated and launched. | `outputs/final_tables/cvcp_ccsh_postprocess_manifest.md`, `scripts/run_cvcp_ccsh_postprocess_queue.ps1` |
| Module-combo queue | 18 Base/SAMEQ/SHUF-K4/CVCP + CCSH/CEQ/HNMB/AUCH/CDCS module-combo rows generated and queued with idle-GPU gating. | `outputs/final_tables/cvcp_ccsh_module_combo_manifest.md`, `scripts/run_cvcp_ccsh_module_combo_queue.ps1` |
| Teacher comparison audit | Local model compatibility audited; only Qwen3VL family is supported by the current formal trainer without architecture-specific adapters. | `outputs/final_tables/model_comparison_results.md` |

## CVCP/CCSH Final Completion Snapshot

| Area | Final state | Evidence |
| --- | --- | --- |
| Training/postprocess rows | `27/27` complete, with LP, NIH appendix, visual-dependence, counterfactual, A/B-swap, and paraphrase diagnostics refreshed. | `outputs/final_tables/cvcp_training_results.md`, `outputs/final_tables/cvcp_ccsh_postprocess_status.md` |
| Module-combo rows | `18/18` complete across CCSH, CEQ, HNMB, AUCH, CDCS, and full-stack combinations. | `outputs/final_tables/module_combo_results.md` |
| Locked comparison | Best empirical training row is `cvcp_v1_sameq_full`; strongest module readouts use `cvcp_v4_replay_10k` + CCSH/stack variants. | `outputs/final_tables/locked_final_comparison.md`, `outputs/final_tables/module_combo_results.md` |
| External/model boundaries | NIH is appendix/stress only; VinDr/VinBig is image-only without labels; PadChest missing; non-Qwen3VL model families are compatibility/boundary rows without trainer adapters. | `outputs/final_tables/external_eval_results.md`, `outputs/final_tables/model_comparison_results.md` |
| Source write-back | Final closure section appended with marker `CVCP_CCSH_FINAL_EXECUTION_CLOSURE_20260706`. | `vivid_med_cvcp_ccsh_full_next_experiment_plan.md` |
| Queue/GPU state | Both training lanes, postprocess lanes, and module lanes reached `QUEUE_DONE`; final GPU check showed GPU0/GPU1 `0 MiB`, `0%`. | `outputs/logs/cvcp_ccsh/*`, `outputs/logs/cvcp_ccsh_postprocess/*`, `outputs/logs/cvcp_ccsh_module_combos/*` |

---

# VIVID-Med Full Stability / NIH / Module Training Execution Plan

## Active Scope

Execute the boundary items from `vivid_med_case_study_modules_next_experiment_plan.md` as real runs, not only manifests: seed1/seed2/seed3 stability training and downstream diagnostics, NIH full/available transfer beyond the current 1k boundary where data exists, embedding export plus MMD/UMAP, and formal CEQ/AUCH/HNMB/DRA/CCSH/CDCS module ablation training.

## Goal

Convert the previous "implemented + manifest + boundary report" state into artifact-backed scientific evidence. A row is complete only when its run package, metrics, logs, downstream diagnostics, and refreshed summary tables exist. If current data/runtime makes a requested item impossible, preserve the failure as a case study with a precise missing-input or runtime boundary.

## Current Phase

- [x] Phase F0: Skill pre-flight, memory lookup, and previous boundary-state audit.
- [x] Phase F1: Re-open the requirement ledger and target markdown so seed/NIH/embedding/module items are active work, not completed claims.
- [x] Phase F2: Audit runnable environment, GPU/process state, existing checkpoints, configs, and exact training commands.
- [x] Phase F3: Launch true multi-seed training for the required family rows:
  - [x] `SHUF-3k` seeds 1/2/3 or documented equivalent seed slots.
  - [x] `SHUF-TW-clinical` seeds 1/2/3 or documented equivalent seed slots.
  - [x] `SAMEQ-SHUF-3k` seeds 1/2/3 or documented equivalent seed slots.
  - [x] `SHUF-K4` seeds 1/2/3 or documented equivalent seed slots.
- [x] Phase F4: For each completed seed run, run/package downstream diagnostics: vision export, CheXpert LP, NIH transfer, visual dependence, counterfactual/A-B where applicable, paraphrase where applicable, instruction audit, and cost/runtime markers.
- [x] Phase F5: Build or locate NIH full/available manifest and run transfer; clearly separate NIH 1k, NIH available, and NIH full.
- [x] Phase F6: Export train/validation/NIH embeddings for relevant finalists and compute MMD plus UMAP/projection outputs from real embedding files.
- [x] Phase F7: Launch formal module ablation training for CEQ, AUCH, HNMB, DRA, CCSH, and CDCS variants; record configs, metrics, and downstream diagnostics.
- [x] Phase F8: Refresh `multiseed_stability`, `nih_domain_audit`, `domain_shift_mmd`, `dataset_embedding_projection`, `module_candidate_results`, and `locked_final_comparison` from real artifacts.
- [x] Phase F9: Write final results and remaining limitations back into `vivid_med_case_study_modules_next_experiment_plan.md`, `docs/README.md`, `docs/case_study_modules_requirement_ledger.md`, `findings.md`, and `progress.md`.
- [x] Phase F10: Final verification: process/GPU audit, output existence, metric table consistency, and no prose-only completion.

## Evidence Rules

- `manifest/config only` is not completion for this active section.
- `NIH full` requires a larger/full manifest and completed transfer metrics. Current completed evidence is NIH `all_available=25596` from the available UMS manifest, not a claim about any unavailable raw-NIH rows.
- MMD/UMAP require actual embedding files; current `domain_shift_mmd` and `dataset_embedding_projection` are real embedding-backed outputs, not the old no-embedding boundary reports.
- Module rows require formal train/eval packages, not only import/smoke tests. Current CEQ/AUCH/HNMB/DRA/CCSH/CDCS rows all have `metrics_final.json`.
- Preserve failures and runtime/data blockers in `History/` or final tables rather than retrying silently.
- On this Windows host, prefer conservative GPU concurrency and `num_workers: 0` for resumed/long dataloaders unless current evidence proves another setting safe.

## Errors Encountered

| Time | Error | Attempt | Resolution |
| --- | --- | --- | --- |
| 2026-07-01T13:12:00+08:00 | Extra execution manifest initially treated old no-embedding `domain_shift_mmd` / `dataset_embedding_projection` boundary files as `completed_existing`. | Audited the extra queue before embeddings ran. | Added `force_run` support and regenerated the manifest so MMD/UMAP rerun after real embeddings exist; restarted only the waiting extra queue. |
| 2026-07-01T13:14:00+08:00 | The first full postprocess watcher omitted A/B-swap diagnostics even though the target stability table requires CF acc and A/B-swap acc separately. | Searched existing next-stage A/B-swap scripts and package semantics. | Added A/B-swap input/config/output columns to the full manifest and patched postprocess to generate swapped JSONL plus run the counterfactual evaluator; restarted only the waiting postprocess watchers. |
| 2026-07-01T14:40:00+08:00 | `conda run`/PowerShell reported exit 1 after a `torch_dtype` deprecation warning even though `shuf_3k_seed1/metrics_final.json` was written completely. | Inspected the per-run log and metrics JSON. | Kept the run as `ACCEPT_WITH_ARTIFACT`; preserve the warning as an execution-note artifact, not a scientific failure. |
| 2026-07-01T15:08:00+08:00 | NIH available transfer failed immediately with argparse `unrecognized arguments: Dataset`. | Manual two-sample NIH transfer succeeded, proving the evaluator and data path were valid; queued commands still used nested PowerShell double quotes around `H:/Xiyao_Wang/000_Public Dataset`. | Added single-quote argument wrapping in `run_case_study_postprocess_queue.ps1`, regenerated extra commands with single-quoted paths, parser/compile checked them, and restarted only postprocess/extra watchers while leaving training lanes running. |
| 2026-07-01T15:15:00+08:00 | `case_study_extra_execution_status` still displayed old MMD/UMAP boundary files as `completed` even after the manifest had `force_run=1`. | Refreshed the summary tables after regenerating the extra manifest. | Patched `summarize_case_study_full_execution.py` so force-run rows require a success file newer than the manifest; old files now show `rerun_required` until real embedding-backed recomputation finishes. |
| 2026-07-01T15:17:00+08:00 | NIH seed1/seed2 transfers were running but would have produced `transfer_metrics.json` without the computed NIH-1k/5k `subset_metrics`. | Inspected `evaluate_qwen3vl_lp_transfer.py` while the full pass was still early. | Patched the evaluator to write `subset_metrics` and a `progress.json` heartbeat, then intentionally stopped/restarted only the postprocess trees; the resulting `FAIL exit=-1/1` lines at 15:17 are controlled restarts, not model/data failures. |
| 2026-07-01T18:48:00+08:00 | Goal-continuation/tool interruption closed active training and postprocess process trees; seed1/seed2 visual diagnostics exited with code `1073807364` before writing JSON. | Audited GPU/process state, per-run visual logs, and available checkpoints. | Patched `run_case_study_training_queue.ps1` to auto-resume unfinished runs from the highest `step_*.pt` checkpoint or `best.pt`; verified TW seed2 checkpoint at step 1000 and TW seed3 checkpoint at step 500, then prepared to restart queues. |
| 2026-07-01T19:00:00+08:00 | Repeated a known PowerShell heredoc mistake while checking checkpoint metadata. | Tried `python - <<'PY'`. | Re-ran as `python -c`; keep using `python -c` or script files on this host. |
| 2026-07-01T19:10:00+08:00 | Misread postprocess command logs and changed lane1 child commands from logical `cuda:0` to `cuda:1`, which fails because `CUDA_VISIBLE_DEVICES=$Gpu` already remaps the selected physical GPU to logical device 0. | Patched `run_case_study_postprocess_queue.ps1`, restarted postprocess, and saw lane1 visual exit with code 1. | Reverted the child-device value to explicit `$device = "cuda:0"` with a comment documenting the remap; stopped stale visual/postprocess trees and relaunched clean watcher PIDs `20192` and `4820`. |
| 2026-07-02T06:38:00+08:00 | Embedding-backed MMD would likely allocate high-dimensional three-way difference tensors for 4k-row samples. | Audited the extra execution scripts before MMD/UMAP started. | Patched `scripts/compute_domain_shift_mmd.py` to use pairwise squared-distance matrix products and chunked RBF kernel means; `py_compile` passed for MMD/UMAP/module ablation scripts. |
| 2026-07-02T15:39:16+08:00 | First formal module ablation (`train_module_ceq`) failed before training with `ModuleNotFoundError: No module named 'evaluation'`. | Read the failing log and reproduced the CLI import path issue with the module ablation script. | Patched `scripts/train_case_study_module_ablation.py` to insert the repository root on `sys.path`, verified `--help` plus `py_compile`, and restarted the extra queue. CEQ/AUCH/HNMB/DRA/CCSH/CDCS then completed formal 1000-step training. |

## Final Completion Snapshot

| Area | Final state | Evidence |
| --- | --- | --- |
| Stability/downstream | 12/12 required family-seed rows completed true 5000-step training and downstream packages. | `outputs/final_tables/case_study_full_execution_status.md` |
| NIH available transfer | 12/12 rows completed `all_available` transfer on 25,596 NIH UMS records with subset metrics. | `outputs/final_tables/nih_available_transfer_status.md` and per-run `transfer_metrics.json` |
| Curriculum/embeddings/domain | Curriculum v2 completed 12,000 steps; train/val/Chexpert/NIH embeddings exported; MMD/projection recomputed from embeddings. | `outputs/final_tables/case_study_extra_execution_status.md`, `outputs/final_tables/domain_shift_mmd.md`, `outputs/final_tables/dataset_embedding_projection.md` |
| Formal modules | CEQ/AUCH/HNMB/DRA/CCSH/CDCS all completed 1000-step formal ablation training. | `outputs/final_tables/module_ablation_results.md` |
| Remaining interpretation boundary | Current results support mechanism diagnosis and next locked finalists; they do not make `SHUF-TW-clinical` a final-best method. | `vivid_med_case_study_modules_next_experiment_plan.md` Part 8 |

---

# VIVID-Med Case Study + TMI Module Execution Plan

## Active Scope

Execute `vivid_med_case_study_modules_next_experiment_plan.md` to artifact-backed completion, including all required case-study, stability, NIH/domain, curriculum-v2, TMI-module, and locked-comparison work. The previously completed `vivid_med_next_stage_comprehensive_experiment_plan.md` is baseline evidence only; it does not close this new plan unless the required scripts, outputs, reports, and module artifacts named by the new document exist and are verified.

## Goal

Move from a temporary `SHUF-TW-clinical` candidate toward a case-driven, statistically stable, modular, reproducible TMI-ready pipeline. The work must avoid post-hoc "综合最优" claims by producing case studies, multi-seed/CI evidence, NIH label/domain diagnostics, curriculum retry artifacts, deployable clinical modules, pre-registered family finalist rules, and final locked comparison tables.

## Current Phase

- [x] Phase C0: Skill pre-flight, memory lookup, session catchup, UTF-8 plan read, and existing planning-file takeover.
- [x] Phase C1: Build the requirement ledger and artifact manifest from `vivid_med_case_study_modules_next_experiment_plan.md`.
- [x] Phase C2: Audit existing outputs/configs/scripts to identify reusable evidence and exact gaps for this plan.
- [x] Phase C3: Implement and verify case-study scripts:
  - [x] `scripts/mine_pairwise_case_studies.py`
  - [x] `scripts/audit_nih_transfer_failure.py`
  - [x] `scripts/audit_hard_negative_quality.py`
  - [x] `scripts/audit_curriculum_leakage_cases.py`
  - [x] `scripts/build_casebook_markdown.py`
- [x] Phase C4: Implement and verify stability/statistics scripts:
  - [x] `scripts/run_multiseed_manifest.py`
  - [x] `scripts/bootstrap_auc_ci.py`
  - [x] `scripts/paired_bootstrap_method_delta.py`
  - [x] `scripts/summarize_multiseed_results.py`
- [x] Phase C5: Implement and verify NIH/domain scripts:
  - [x] `scripts/audit_label_mapping_nih.py`
  - [x] `scripts/run_nih_full_transfer.py`
  - [x] `scripts/compute_domain_shift_mmd.py`
  - [x] `scripts/plot_dataset_embedding_umap.py`
- [x] Phase C6: Implement and verify curriculum-v2 scripts/configs:
  - [x] `scripts/build_curriculum_v2_schedule.py`
  - [x] `scripts/generate_curriculum_v2_instructions.py`
  - [x] `scripts/train_qwen3vl_curriculum_v2.py`
  - [x] 10k and MIMIC/full boundaries or runnable configs.
- [x] Phase C7: Implement and verify module scaffolds:
  - [x] `models/clinical_evidence_query.py`
  - [x] `models/answerability_uncertainty_head.py`
  - [x] `models/hard_negative_memory_bank.py`
  - [x] `models/domain_robust_adapter.py`
  - [x] `models/clinical_consistency_head.py`
  - [x] `models/case_driven_curriculum_scheduler.py`
- [x] Phase C8: Generate no/low-training case study and audit outputs:
  - [x] `outputs/final_tables/case_study_shuf_tw_vs_shuf.md`
  - [x] `outputs/final_tables/case_study_nih_transfer.md`
  - [x] `outputs/final_tables/case_study_curriculum_failure.md`
  - [x] `outputs/final_tables/case_study_hard_negative_quality.md`
  - [x] `outputs/final_tables/case_study_summary.csv`
  - [x] `outputs/final_tables/nih_domain_audit.md`
  - [x] `outputs/final_tables/multiseed_stability.md`
- [x] Phase C9: Launch or queue required candidate/curriculum/module experiments where runnable; otherwise create explicit documented stop boundaries with evidence.
  - [x] Multi-seed configs/manifests generated, then reopened into real 12-row long-training/downstream execution.
  - [x] Curriculum-v2 progressive replay data/config/training manifest generated, then completed as a real 12000-step long run.
  - [x] NIH full/available wrapper generated the available-data boundary, then completed NIH `all_available=25596` transfer for all 12 rows.
  - [x] Domain-shift MMD/projection scripts generated initial boundary reports, then were force-rerun from real CheXpert/NIH embeddings.
- [x] Phase C10: Summarize module candidate results and locked comparison:
  - [x] `outputs/final_tables/module_candidate_results.md`
  - [x] `outputs/final_tables/locked_final_comparison.md`
  - [x] family finalist rules and Pareto/selection framework.
- [x] Phase C11: Write final status/results back into `vivid_med_case_study_modules_next_experiment_plan.md`.
- [x] Phase C12: Final completion audit against every explicit script, artifact, run, module, report table, and boundary before marking the goal complete.

## Evidence Rules

- Completion is based on current source files, configs, data manifests, generated casebooks, JSON/CSV/MD outputs, checkpoints/metrics where runs are launched, logs, process/GPU state, and final requirement audits.
- Planning prose and old result tables are context only, not completion proof for the new case-study/module plan.
- Preserve failed or impossible cases as case-study/boundary artifacts instead of silently dropping them.
- On this Windows host, prefer `num_workers: 0` for long-running/resumed loaders unless a current run proves another setting is safe.
- Keep medical-data-derived artifacts under ignored output/data directories. Source code, docs, and summary ledgers belong in tracked project paths.
- Do not claim `SHUF-TW-clinical` is the final method until multi-seed, NIH/domain, case-study, and locked-comparison evidence support the claim.

## Current Completion Snapshot

| Area | Completed evidence | Boundary preserved |
| --- | --- | --- |
| Prior next-stage suite | Previous audit completed 39 next-stage runs and those tables are now mapped into the case-study/module ledger. | Old runs remain baseline evidence, not automatic proof of new module performance. |
| Case studies | Plan-named scripts, casebooks, taxonomy, pairwise run-level deltas, NIH failures, curriculum leakage, and hard-negative quality reports exist. | Some pairwise rows are summary-only because paired prediction files are not available for every comparison. |
| Stability | Multi-seed manifest plus real 12-row 5000-step family/seed packages and downstream diagnostics exist. | SAMEQ CF/A-B option metrics are not applicable because the same-question/different-answer format has zero option-pairwise rows. |
| NIH/domain | NIH available transfer completed for all 12 rows; real CheXpert/NIH embeddings now back MMD/projection reports. | This is NIH `all_available=25596` from the current UMS manifest, not a claim about unavailable raw-NIH rows. |
| Curriculum v2 | Progressive-replay formal long training completed at 12000 steps. | Training loss alone is not a final-method selection claim. |
| TMI modules | CEQ, AUCH, HNMB, DRA, CCSH, and CDCS source files exist, smoke passed, and formal 1000-step ablation packages exist. | Current module results are embedding-level ablations, not yet full end-to-end backbone candidates. |
| Final reporting | `case_study_full_execution_status`, `case_study_extra_execution_status`, `module_ablation_results`, and the target plan Part 8 are refreshed from real artifacts. | Locked comparison remains conservative: mechanism diagnosis and finalist narrowing, not final-best selection. |

## Errors Encountered

| Time | Error | Attempt | Resolution |
| --- | --- | --- | --- |
| 2026-07-01 | Existing active goal prevented a new `create_goal` call. | User invoked `/goal`, so a new goal was attempted. | Used `get_goal`; the existing active goal already matched this request, so execution continues under it. |
| 2026-07-01 | Memory index pointed to two older rollout summary filenames that are not present in the current memory folder. | Tried direct `Get-Content` of the indexed paths. | Used `MEMORY.md` lines and current planning files instead; treat memory as guidance and revalidate current repo state. |
| 2026-07-01 | Plain PowerShell plan read initially showed mojibake Chinese. | Used default `Get-Content`. | Re-read the target markdown with `-Encoding UTF8`; future Chinese markdown reads should specify UTF-8. |

---

# Historical: VIVID-Med Next-Stage Comprehensive Execution Plan

## Active Scope

Execute `vivid_med_next_stage_comprehensive_experiment_plan.md` to artifact-backed completion, including the required Part A experiments and the expansion items in Part B. Earlier P4-v2 / SHUF-3k results are baseline evidence only; they do not prove completion for this broader next-stage plan unless the run ID, artifact set, and diagnostics match this plan.

## Goal

Turn the current P3/P4/CF/SHUF/QA8/P5 evidence into a complete Clinical Instruction Workflow experiment suite: JSON-loss diagnostics, rich QA mixtures, curriculum/progressive schedules, SHUF++ variants, token weighting, scale/external-transfer extensions, robustness/leakage/calibration/qualitative diagnostics, final method selection, and write-back into `vivid_med_next_stage_comprehensive_experiment_plan.md`.

## Current Phase

- [x] Phase N0: Skill pre-flight, memory lookup, session catchup, and UTF-8 plan read.
- [x] Phase N1: Confirm current repository evidence boundary: P4-v2/SHUF-3k completed and useful as baseline, but new next-stage G1 scripts are missing.
- [x] Phase N2: Build the plan requirement ledger and artifact manifest from Parts A-I.
- [x] Phase N3: Implement Phase 0 preparation artifacts:
  - [x] `scripts/generate_storymix_instructions.py`
  - [x] `scripts/generate_sameq_shuf_pairs.py`
  - [x] `scripts/generate_multi_negative_shuf.py`
  - [x] `scripts/audit_instruction_leakage_v2.py`
  - [x] `scripts/build_progressive_mixture_schedule.py`
  - [x] `scripts/build_token_weight_map.py`
  - [x] `scripts/mine_hard_negatives_from_embeddings.py`
  - [x] D6/D7-derived distribution tables and leakage audit 2.0 outputs for current next-stage JSONL files
  - [x] P2 value-only / no-punct / compact / field-query data and loss-mask support
- [x] Phase N4: Launch and verify Phase 1 quick diagnostics:
  - [x] P2-value-only
  - [x] P2-field-query
  - [x] StoryMix-QA8
  - [x] SHUF-heavy-QA8
  - [x] CF-heavy-QA8
  - [x] SAMEQ-SHUF-3k
  - [x] SHUF-TW-visual
- [x] Phase N5: Launch and verify Phase 2 workflow experiments:
  - [x] Mix-Story-QA8
  - [x] CUR-P3-SHUF
  - [x] CUR-CF-SHUF
  - [x] CUR-P3-CF-SHUF
  - [x] PROG-Mix
  - [x] PROG-Mix-TW
  - [x] PROG-Mix-SAMEQ
  - [x] PROG-Mix-DualMargin
- [x] Phase N6: Launch and verify Phase 3 SHUF++ extensions:
  - [x] SHUF-K2
  - [x] SHUF-K4
  - [x] InBatch-SHUF
  - [x] Mined-SHUF
  - [x] SelfHard-SHUF
  - [x] DUAL-CF-SHUF
  - [x] Progressive-HardNeg
- [x] Phase N7: Launch and verify Phase 4 scale/external items:
  - [x] SHUF-10k-8k
  - [x] StoryMix-10k-8k
  - [x] PROG-Mix-10k-8k
  - [x] PROG-Mix-TW-10k
  - [x] NIH 1k transfer for all 39 runs with local UMS support; MIMIC transfer boundary documented.
  - [x] Document explicit stop boundaries for PadChest/VinDr/larger VLM if local data/model is unavailable
- [x] Phase N8: Run Part B extended diagnostics and ablations:
  - [x] Training policy ablation
  - [x] Model scale/type controls where local models permit
  - [x] Calibration / threshold / AUPRC
  - [x] Prompt robustness / option bias
  - [x] Leakage audit 2.0 for all generated instruction sets
  - [x] Qualitative visualization or documented stop boundary if attention/Grad-CAM support is unavailable
- [x] Phase N9: Generate per-run required output package and consolidated final tables.
- [x] Phase N10: Apply Part F final-method selection rules and write final status/results back into `vivid_med_next_stage_comprehensive_experiment_plan.md`.
- [x] Phase N11: Completion audit against every explicit run, artifact, diagnostic, and boundary before marking the goal complete.

## Evidence Rules

- Completion is based on current data files, configs, checkpoints, `metrics_final.json`, diagnostic JSON/CSV/MD outputs, logs, GPU/process state, and final requirement audit tables.
- Planning prose and old result tables are not completion evidence unless the named artifact exists and matches this plan.
- Every run must preserve or generate the package requested in G3 where applicable: `config_snapshot.json`, `metrics_final.json`, `metrics_step_*.json`, `progress.json`, `training_log.txt`, `vision_export_manifest.json`, `lp_results.md`, `visual_dependence_results.md`, `counterfactual_results.md`, `ab_swap_results.md`, `paraphrase_results.md`, `instruction_audit.md`, and `cost_table.md`.
- Preserve API/network/GPU failures as case-study evidence; retry through resume/fill logic rather than deleting failure traces.
- On this Windows host, prefer `num_workers: 0` for long-running/resumed loaders unless a current run proves another setting is safe.
- Keep generated medical-data-derived instruction files under ignored output/data artifact directories, not committed source paths.
- Before final completion, verify Part F thresholds: CheXpert AUC, NIH AUC, hard shuffle delta, CF acc, leakage rate, A/B balance, cost, and interpretability.

## Current Artifact Gap Snapshot

| Area | Current evidence | Gap for next-stage plan |
| --- | --- | --- |
| Baseline SHUF-3k | P4-v2 tables show SHUF-3k CheXpert AUC 0.7267, NIH AUC 0.5680, hard shuffle delta 0.0807, CF acc 0.8707. | Treat as baseline only; still need new StoryMix, SAMEQ, TW, curriculum, SHUF++, scale, and extended diagnostics. |
| G1 scripts | Seven plan-named scripts plus P2/config helpers are implemented and `py_compile`-checked. | Use formal runs to prove the generated variants, not just script existence. |
| Training/evaluation trunk | Qwen3-VL instruction trainer now supports JSON value/no-punct masking, token weighting, image margin, answer margin, multi-negative hard images, and per-run progress/config snapshots. | Need full formal run packages and downstream diagnostics for the next-stage matrix. |
| Next-stage data/configs | P2 compact/field-query, Balanced/CF-heavy/SHUF-heavy/Clinical-rich, StoryMix QA5/8/10/12, SAMEQ-SHUF, SHUF-K2/K4, Mined-SHUF train/val, SelfHard-SHUF train, Progressive-HardNeg train/schedule, materialized curriculum/progressive, and 10k-scale data are generated; the training and LP manifests list all 39 configs. | Closed. Final completion audit reports 1049/1049 rows completed in `outputs/final_tables/next_stage_completion_audit.csv`. |
| Planning files | Previous plan says P4-v2 is complete. | This next-stage active section supersedes the old P4-v2 active scope. |

## Errors Encountered

| Time | Error | Attempt | Resolution |
| --- | --- | --- | --- |
| 2026-06-29 | First target-plan read showed mojibake Chinese in PowerShell output. | Plain `Get-Content` under default encoding. | Re-read with `[Console]::OutputEncoding=UTF8` and `-Encoding UTF8`; use UTF-8 for future Chinese markdown reads. |
| 2026-06-29 | Attempt to create a new goal failed because the thread already had the exact active goal. | Called `create_goal` after user invoked `/goal`. | Used `get_goal` and continued the existing active goal. |
| 2026-06-29 | Broad `rg --files ... outputs` scan timed out due many experiment artifacts. | Searched scripts/configs/data/models plus outputs in one command. | Switched to exact `Test-Path` checks and targeted output/config listings. |
| 2026-06-29 | Phase 1 postprocess commands can emit warning text to stderr even when the expected metrics/export artifact is written. | `p2_field_query` CheXpert LP wrote `metrics_final.json` but the wrapper recorded `EXITCODE 1` and stopped. | Patched postprocess `Run-Step` helpers to accept a step when its expected artifact path exists, then relaunched GPU1 postprocess from the next missing artifact. |
| 2026-06-30 | GPU0 three-lane postprocess/training window reached 23.9GB during `storymix_qa5` visual-dependence diagnostics. | Let `storymix_qa10` training, `mined_shuf` training, and `storymix_qa5` postprocess run concurrently after LP/NIH transfer succeeded. | Stopped only the `storymix_qa5` visual-dependence process tree to protect the two active training runs; preserve `EXITCODE -1` as a VRAM pause case and relaunch postprocess later, where completed export/LP/transfer steps will be skipped. |
| 2026-06-30 | GPU0 three-lane window reached ~24.24GB during `storymix_qa10` paraphrase postprocess while `storymix_qa12` and `selfhard_shuf` were training/evaluating. | Allowed the user-requested three-lane trial after `storymix_qa10` visual and counterfactual diagnostics had already published. | Stopped only the `storymix_qa10` isolated postprocess tree (`gpu0-storyqa10-20260630T055048`) to protect training; visual/counterfactual artifacts remain canonical, and a later rerun will skip completed steps and resume from paraphrase/package. |
| 2026-06-30 | GPU0 ready watcher restarted canonical `storymix_qa10` paraphrase after GPU0 memory dropped, while GPU1 isolated resume worker was already running the same missing paraphrase step. | `storymix_qa12` completed and freed memory, causing the watcher to pass its threshold. | Stopped only the later GPU0 canonical paraphrase PIDs and kept GPU1 isolated worker `gpu1-storyqa10-resume-20260630T065933` to avoid duplicate writes to the canonical paraphrase JSON. |
| 2026-06-30 | GPU1 ready watcher launched canonical `prog_mix` counterfactual while GPU0 isolated worker `gpu0-progmix-20260630T073805` was already running the same run's counterfactual diagnostics. | Old `ready_post_gpu1` watcher still included `prog_mix` in its queue and passed its memory gate after `shuf_tw_role`/`storymix_qa12` freed GPU1 lanes. | Stopped only the canonical `prog_mix_counterfactual_diagnostics.json` process tree and restarted the watcher as `ready_post_gpu1_rebased2` beginning at `prog_mix_tw`, excluding manually-owned `prog_mix` and `selfhard_shuf`. |
| 2026-06-30 | GPU1 three-lane training/postprocess window reached 24267 MiB while `prog_mix_tw`, `selfhard_shuf` A/B-swap, and manually added `prog_mix_dualmargin` were active. | Tried the user-approved third lane for `prog_mix_dualmargin`, chosen far enough downstream in the serial queue to reduce duplicate-output risk. | Stopped only the manual `prog_mix_dualmargin` process tree at about step 2500 to protect the original training/postprocess lanes; preserve its checkpoint for later resume. |
| 2026-06-30 | GPU0 three-lane training/postprocess window reached 24201 MiB during `dual_cf_shuf` A/B-swap counterfactual. | Ran `progressive_hardneg`, `train_last4`, and SHUF++ isolated postprocess concurrently after `dual_cf_shuf` LP/NIH/visual/primary counterfactual had already published. | `scripts/guard_gpu_memory_process_tree.ps1` stopped only the SHUF++ watcher tree, preserving training. Relaunched SHUF++ watcher as `ready_post_gpu0_shufpp_isolated_r2` with a stricter 18000 MiB gate so `dual_cf_shuf` can resume from the missing A/B-swap step. |
| 2026-06-30 | GPU0 SHUF++ watcher rebase cleanup matched its own PowerShell command line because the launcher text contained `ready_post_gpu0_shufpp_isolated_r2`. | Attempted a broad `CommandLine` match while moving `dual_cf_shuf` ownership from GPU0 to GPU1. | The cleanup stopped the intended r2 watcher/guard and the transient launcher shell before new watchers started. Verified no `dual_cf_shuf` postprocess remained, then relaunched GPU0 as `ready_post_gpu0_shufpp_isolated_r3` for `progressive_hardneg,shuf_k4_tw_visual` and launched `ready_post_gpu1_dualcf_isolated` with a separate guard. |
| 2026-06-30 | GPU1 reached 23831 MiB when a manual third lane for `shuf_10k_8k` was added beside `prog_mix_sameq` training and `prog_mix_dualmargin` visual-dependence diagnostics. | Used the user-approved three-lane policy and gave the new scale lane a stricter 23600 MiB guard. | The guard stopped only the manual `shuf_10k_8k` process tree; the original GPU1 training/postprocess lanes stayed alive. Preserve this as the conservative scale-run concurrency boundary. |
| 2026-06-30 | GPU1 later reached 24025 MiB when `shuf_10k_8k` was retried with a 24000 MiB guard beside `prog_mix_sameq` and `prog_mix_dualmargin` A/B-swap. | Retried the scale lane after the 23600 MiB threshold proved too conservative. | The guard again stopped only the manual `shuf_10k_8k` tree at about step 475. No checkpoint directory exists for this partial attempt, so the next formal run should restart from scratch when GPU1 has a larger free window. |

---

# Historical: VIVID-Med Qwen3-VL P4-v2 / Scale Execution Plan

## Active Scope

Execute `vivid_med_qwen3vl_p4v2_scale_experiment_plan.md` to artifact-backed completion. The prior Qwen3-VL v2 proposal objective remains historical evidence and must not be treated as completion proof for this new P4-v2 / scale plan unless the required outputs, run IDs, and diagnostics match this plan.

## Goal

Generate/audit P4-v2 hard counterfactual and hard image-shuffle instruction data, run the requested Qwen3-VL training/evaluation matrix as far as the plan requires, preserve API/GPU failure evidence, and write final results back into `vivid_med_qwen3vl_p4v2_scale_experiment_plan.md`.

## Current Phase

- [x] Phase A0: Skill pre-flight and previous-session catchup.
- [x] Phase A1: Read the new P4-v2 / scale plan and identify that prior planning files describe the older v2 objective.
- [x] Phase A2: Audit current API/generation/training processes, GPU state, scheduled tasks, and existing P4-v2 artifacts.
- [x] Phase A3: Map plan-required outputs to existing scripts/configs/artifacts; implement missing scripts/configs only where required.
- [x] Phase A4: Generate facts/D6/D7 data under the API concurrency limit and run instruction quality audits.
- [x] Phase A5: Run debug/main training and required diagnostics.
- [x] Phase A6: Summarize all results, mark explicit stop boundaries for optional/impossible items, write back to the plan document, and verify outputs.

## Evidence Rules

- Completion is based on current output files, metrics JSON/CSV/MD tables, logs, checkpoint contents, GPU/process state, and explicit final verification.
- Planning prose and old v2 proposal results are context only, not completion evidence for this P4-v2 / scale plan.
- If another API data-generation task is already running, let it finish unless it conflicts with this plan's required generation. If the artifact does not match this plan, generate this plan's data first, then allow the other queue to resume.
- Preserve network/API/model failures as evidence instead of silently deleting them.
- On this Windows host, prefer `num_workers: 0` for long-running or resumed data loaders unless a current run proves a different setting is safe.

## Errors Encountered

| Time | Error | Attempt | Resolution |
| --- | --- | --- | --- |
| 2026-06-28 | New P4-v2 / scale plan superseded the previously completed Qwen3-VL v2 proposal objective. | Existing `task_plan.md` still reported the older v2 objective as completed. | Added this active scope and will re-audit artifacts against `vivid_med_qwen3vl_p4v2_scale_experiment_plan.md`. |
| 2026-06-28 | PowerShell heredoc is invalid in this shell. | Tried a bash-style `python - <<'PY'` YAML check. | Re-ran as `python -c` and kept this reminder in the error log. |
| 2026-06-28 | `facts_1k` shard3 exited on `http.client.RemoteDisconnected`. | Initial fact generator caught `urllib.error.URLError` but not this low-level disconnect. | Patched `scripts/generate_p4v2_facts_with_glm.py` to record/recover from `RemoteDisconnected` and relaunched shard3 with `--resume`. |
| 2026-06-29 | Monitoring-only PowerShell command failed with `Sort-Object : The Name key is not valid`. | Combined `Sort-Object Name,@{Name='Lines';...}` while checking parse-error counts. | Re-ran the parse-error query with `Sort-Object Name` followed by `Select-Object`; API jobs were unaffected. |

## 2026-06-28 P4-v2 Live Milestones

- S-P4-3k training completed with `global_step=3000`, `best_val_loss=0.8201575026007666`, and language decoder trainable count 0.
- S-P4-3k CheXpert LP completed with `macro_auc=0.6811373789120363`, `macro_f1=0.8285535990106714`, and 1000 validation records.
- S-P4-5k training completed with `global_step=5000`, `best_val_loss=0.8109186573103456`, and language decoder trainable count 0.
- S-P4-8k old-P4 step-scaling was launched on GPU0 after changing its config device to `cuda:0` to avoid the active GPU1 LP/eval lane.
- P4-v2 `facts_1k` generation remains active in four API shards; old incompatible `v3_train_extra4k` generation remains paused until P4-v2 fact/D6/D7 generation no longer needs API priority.
- P4-v2 `facts_extra2k` completed as four 500-row shards and was combined with `facts_1k` into `facts_3k.jsonl` with 3000 rows and 0 duplicate drops.
- D6/D7 3k canonical files are ready: each has 14,333 validator-accepted records, 0 canonical rejections, 2,980 images, 4.8097 QA/image, 49.9773% answer A, and D7 hard-negative coverage 100%.
- Main training/evaluation matrix is complete, including S-P4 step scaling, CF-1k-3k, CF-3k-5k, SHUF-3k, CF-3k-8k, and QA8-3k. Summary tables were generated under `outputs/final_tables/qwen3vl_p4v2_*`, and Section 21 was appended to `vivid_med_qwen3vl_p4v2_scale_experiment_plan.md`.

---

# Historical: VIVID-Med Qwen3-VL Proposal v2 Execution Plan

## Active Scope

Continue `vivid_med_qwen3vl_proposal_v2_modification_plan.md` as the repository experiment objective. The old `vivid_med_clinical_instruction_proposal.md` MIMIC V1-V4 line is closed and must not receive new experiments except as baseline evidence for the Qwen3-VL comparison.

## Goal

Run the Qwen3-VL-coupled clinical instruction route to artifact-backed completion: local Qwen3-VL VLM, frozen language decoder, trainable vision tower plus visual connector, GLM/report-grounded clinical visual instruction data, extracted vision-side checkpoint, LP/transfer and visual-dependence diagnostics, and final requirement-by-requirement audit.

## Current Phase

- [x] Phase 0: Close old MIMIC V1-V4 document line and keep it as historical baseline evidence only.
- [x] Phase 1: Read the Qwen3-VL v2 modification plan and v2 proposal markdown.
- [x] Phase 2: Audit local Qwen3-VL model availability, environment support, API endpoint/key handling, and GPU state.
- [x] Phase 3: Create/update v2 proposal markdown and Qwen3-VL scaffold scripts/configs.
- [x] Phase 4: Run Qwen3-VL component audit and instruction-data validation.
- [x] Phase 5: Complete formal P2-P5 Qwen3-VL instruction training and CheXpert LP runs.
- [x] Phase 6: Refresh Qwen3-VL extraction manifests and consolidate instruction/LP/extraction tables.
- [x] Phase 7: Implement/run Qwen3-VL visual-dependence and counterfactual/paraphrase/template diagnostics.
- [x] Phase 8: Add transfer/subgroup/cost evidence where feasible and compare against old scaffold/no-LM controls.
- [x] Phase 9: Final requirement-by-requirement audit against the v2 plan; do not mark complete until every required item has current evidence or an explicit documented stop boundary.

## Required Outputs

| Output | Status | Notes |
| --- | --- | --- |
| `vivid_med_clinical_instruction_proposal_v2_qwen_vlm.md` | completed | Preserves old scaffold as baseline/negative control only. |
| `scripts/audit_qwen3vl_components.py` | completed | Loads local Qwen3-VL, finds vision/connector/LLM modules, counts params, and runs dummy forward loss. |
| `data/clinical_instruction_dataset.py` | completed | Qwen3-VL processor/message dataset for D0-D5 JSONL; masks labels to answer tokens. |
| `scripts/generate_clinical_instructions_with_glm.py` | completed | GLM Coding Plan wrapper; no raw key persisted. |
| `scripts/validate_clinical_instruction_jsonl.py` | completed | Enforces evidence span, null semantics, laterality/severity, duplicate checks; writes validated/rejected JSONL. |
| `scripts/train_qwen3vl_clinical_instruction.py` | completed | Freezes LLM, trains vision tower/connector, supports debug/resume/checkpoints. |
| `scripts/extract_qwen3vl_vision_backbone.py` | completed | Exports vision tower and connector state dicts from trainable checkpoints. |
| `scripts/train_qwen3vl_vision_lp.py` | completed | Generic LP for Qwen3-VL vision tower; CheXpert debug path verified. |
| `configs/qwen3vl_instruction/*.yaml` | completed | P2-P5 pilot configs plus LP debug config. |
| `outputs/final_tables/qwen3vl_component_audit.{md,json}` | completed | First executable evidence gate passed on local `qwen3-vl-2b-thinking-new`. |
| `outputs/final_tables/instruction_data_audit.{md,csv}` | completed | Data audit gate complete; rejected rows preserved under `outputs/instruction_data/glm_rejected`. |
| `outputs/final_tables/qwen3vl_pilot_matrix.md` | completed | Refreshed with P2-P5 training/LP/extraction evidence and remaining diagnostics boundary. |
| `outputs/final_tables/qwen3vl_empty_result_tables.md` | completed | Result table templates before runs. |
| `outputs/final_tables/qwen3vl_instruction_training_results.{csv,md}` | completed | Aggregates P2-P5 instruction metrics. |
| `outputs/final_tables/qwen3vl_vision_lp_results.{csv,md}` | completed | Aggregates base/P2-P5 LP metrics from nested `metrics` JSON fields. |
| `outputs/final_tables/qwen3vl_extraction_manifest.{csv,md}` | completed | Refreshed manifests show language decoder trainable count 0. |
| `outputs/final_tables/qwen3vl_cost_table.{csv,md}` | partial | Runtime-derived GPU-hours available; peak VRAM not captured. |
| `outputs/final_tables/qwen3vl_visual_dependence_results.{csv,md}` | completed | P2-P5 full val diagnostics complete; question-only deltas large, image-shuffle deltas small. |
| `outputs/final_tables/qwen3vl_counterfactual_results.{csv,md}` | completed | P4/P5 option-subset counterfactual pairwise diagnostics complete. |
| `outputs/final_tables/qwen3vl_paraphrase_robustness_results.{csv,md}` | completed | P2-P5 completed with EXITCODE 0; style rewrites are consistently harder than clinical rewrites. |
| `outputs/final_tables/qwen3vl_nih_transfer_results.{csv,md}` | completed | Base/P2/P3/P4/P5/P6 evaluated on NIH 1k with 0 missing images. |
| `outputs/qwen3vl_lp_runs/p6_data_only_no_lm_chexpert_1k/metrics_final.json` | completed | Data-only no-LM trainable Qwen3-VL vision tower + head control. |
| `outputs/final_tables/qwen3vl_subgroup_results.{csv,md}` | completed | CheXpert common/high-null/uncertain-derived subgroup summary; Support Devices not evaluated by common 8-label LP. |
| `outputs/final_tables/qwen3vl_final_requirement_audit.{csv,md}` | completed | Final completion proof with explicit stop boundaries. |
| `vivid_med_qwen3vl_proposal_v2_modification_plan.md` | completed | Final execution results, metrics, boundaries, and artifact index written back in Section 11. |

## Qwen3-VL Pilot Matrix

| ID | Route | Model | Data | Trainable | Status |
| --- | --- | --- | --- | --- | --- |
| P0 | old scaffold baseline | timm ViT + text-only Qwen/Qwen-Coder | D0 fixed JSON | ViT + new projector | existing old evidence only; do not extend unless needed |
| P1 | old scaffold control | timm ViT + text-only Qwen | D3 QA+CF | ViT + new projector | partially covered by old V3/V4; label as scaffold control |
| P2 | VLM-coupled | Qwen3-VL-2B | D0 fixed JSON | vision + connector | train + LP + extraction + visual-dependence + paraphrase complete |
| P3 | VLM-coupled | Qwen3-VL-2B | D2 report-grounded QA | vision + connector | train + LP + extraction + visual-dependence + paraphrase complete |
| P4 | VLM-coupled | Qwen3-VL-2B | D3 QA+CF | vision + connector | train + LP + extraction + visual-dependence + counterfactual + paraphrase complete |
| P5 | VLM-coupled | Qwen3-VL-2B | D4 QA+CF+token weighting | vision + connector | train + LP + extraction + visual-dependence + counterfactual + paraphrase complete |
| P6 | data-only no-LM | Qwen3-VL vision tower | CheXpert UMS label heads | vision + head | CheXpert LP + NIH transfer complete; not a GLM-instruction D3 head |
| P7 | optional upper bound | Qwen3-VL-2B | D3 QA+CF | vision + connector + LLM LoRA | optional after P4/P5 |

## API / Secret Boundary

- Coding Plan domestic endpoint: `https://open.bigmodel.cn/api/coding/paas/v4`.
- Use the user-provided key only through process environment variables.
- Never write the raw API key into repo files, configs, logs, generated prompts, or planning files.

## Evidence Rules

- Completion must be proved by current files, output artifacts, logs, process/GPU state, and metric JSON/CSV outputs.
- Planning prose is not completion evidence.
- Keep failed runs as failure-case evidence.
- On this Windows host, prefer `num_workers: 0` for long-running or resumed data loaders unless proven safe.

## 2026-07-03 Remote Upload Handoff

Goal: upload the useful runnable subset of local `021_260129VIVID` to the SUES HPC account described in `项目上传与远端运行交接说明_副本.md`, using a project folder directly under `~/projects/xiyaowang/`, then configure a project conda environment and run a GPU smoke test in the approved GPU resource.

| Phase | Status | Notes |
| --- | --- | --- |
| Skill/session preflight | complete | Used `superpowers:using-superpowers` and `planning-with-files`; catchup reported no unsynced context. |
| Read active handoff context | complete | Read `docs/README.md`, current planning files, `.gitignore`, and `项目上传与远端运行交接说明_副本.md`. |
| Revised user target | complete | User requested the project uploaded under `~/projects/xiyaowang/` instead of the split `code/data/outputs` layout. |
| Local upload-scope audit | complete | Full repository is about 1.06 TB: `outputs/` 896.635 GB, `data/` 148.712 GB, `vivid_env/` 5.938 GB, `.git/` 2.467 GB. No repo-local private key candidates found beyond package CA/private-module filenames in `vivid_env`. |
| SSH/auth connectivity | complete | Remote accepted dedicated key; `ssh sues-hpc` returns `SSH_OK`. |
| Remote directory creation | complete | Target folder exists: `~/projects/xiyaowang/021_260129VIVID`. |
| Scope refinement | complete | User asked to skip useless files and pure process records after seeing the 1.06 TB size. |
| Transfer method | complete | Local `rsync` is unavailable; build a filtered staging directory and stream it with tar over SSH. |
| Filtered upload | complete | Uploaded filtered package to `~/projects/xiyaowang/021_260129VIVID`. Excluded `outputs/`, raw data, `vivid_env/`, `.git/`, `History/`, `delete/`, large weights, caches, and process logs/planning records. |
| Remote verification | complete | Remote project is 457 MB, has 345 files at verification, and contains README/requirements/config/scripts/source trees. |
| Remote conda env | complete | Created `vivid_med310`; installed CUDA PyTorch `2.9.0+cu128`, torchvision `0.24.0+cu128`, and project requirements. |
| GPU smoke test | complete | Ran on approved `gpu` tmux session on `gpu02`; CUDA was available on A800, CUDA matmul succeeded, TMI module smoke passed all rows, and post-smoke GPU state was `0 MiB, 0%`. |
| Supplemental artifact/data audit | complete | User asked to prioritize "run first" data/docs. Replaced the partial remote `data/` tree with a tiny real-image runnable package, then verified data loading and GPU training smoke. |

## 2026-07-03 Minimal Runnable Remote Package

Goal: make the remote project runnable quickly without uploading the whole 1.07 TB tree.

| Bucket | Local size | Decision |
| --- | ---: | --- |
| Code/docs/config/scripts/source | 457 MB remote verified | Already uploaded. |
| Non-weight `outputs/` evidence | 3.0 GB remote verified | Already uploaded; includes result tables, instruction JSONL, logs, manifests, embeddings, and diagnostics without `.pt/.pth/.ckpt/.safetensors`. |
| `pretrained/` base weights | 979 MB remote verified | Already uploaded; base weights are useful for local model routes and are not repeated checkpoint clutter. |
| `data/dataset/processed` | 427 MB | Needed for CXR loaders/configs. |
| `data/instructions` + `data/splits` | 229 MB | Needed for instruction routes and CheXpert split-based configs. |
| `data/dataset/CheXpert-v1.0-small` | 10.684 GB | Primary real CXR dataset for quick first training/eval. |
| Small auxiliary data (`LIDC-IDRI-slices`, `organamnist_224.npz`) | 1.823 GB | Reasonable to include now; supports small CT/organ smoke routes. |
| Large raw data (`NIH Chest X-rays`, `AMOS22`, `KITS21`) | 135.565 GB | Defer until specifically needed. |
| Final-version weight candidates | 315.923 GB | Defer or select only specific final models when choosing the run; process checkpoints remain excluded. |

Actual first-run package now on remote:

| Remote item | Verified size/count | Evidence |
| --- | ---: | --- |
| Project root `~/projects/xiyaowang/021_260129VIVID` | 4.8 GB | Includes runnable code/docs/configs, non-weight outputs, pretrained base weights, and tiny real-image data. |
| `data/` | 844 MB | Replaced the interrupted 34 GB partial full-data upload. |
| Sampled CheXpert image files | 2,929 files | Enough for `--debug` CXR classifier smoke because the sampled images cover early processed/split/instruction rows. |
| `data/dataset/processed` | 13 files | Full processed metadata tree retained. |
| `data/instructions` | 41 files | Instruction metadata retained. |
| `data/splits` | 12 files | Split metadata retained. |
| Training smoke output | metrics only | `outputs/remote_smoke_ums_classifier/metrics_final.json` exists; smoke `.pt` files were removed after verification. |

Run verification:

- Data smoke passed in the remote `vivid_med310` environment with CUDA available and a batch tensor shape `(2, 3, 224, 224)`.
- `python scripts/train_ums_classifier.py --config configs/remote_smoke_ums_classifier.yaml --debug` completed on the approved `gpu` tmux session, wrote final metrics, and finished with `Training completed!`.
- Final GPU check in the `gpu` session reported `0 MiB, 0%`.

## 2026-07-03 Remote Project Model Upload

Goal: upload only the local base model needed by this VIVID project from `H:\Xiyao_Wang\001_models` into the remote project-local `model/` directory.

| Phase | Status | Notes |
| --- | --- | --- |
| Identify required model references | complete | Config/script/docs scan found `H:/Xiyao_Wang/001_models/qwen3-vl-2b-thinking-new` referenced 141 times; `Qwen2.5-Coder-7B-Instruct` appears only in legacy text-scaffold configs; BiomedCLIP uses the already-uploaded converted checkpoint under `pretrained/`. |
| Local model inventory | complete | `H:\Xiyao_Wang\001_models\qwen3-vl-2b-thinking-new` has 14 files and is 3.974 GB. |
| Remote model directory | complete | Uploaded to `~/projects/xiyaowang/021_260129VIVID/model/qwen3-vl-2b-thinking-new`; remote size is 4.0 GB with 14 files. |
| Remote compatibility path | complete | Added project-local symlink `H:/Xiyao_Wang/001_models/qwen3-vl-2b-thinking-new` -> `model/qwen3-vl-2b-thinking-new` so existing Windows-style configs work from the remote project root. |
| Remote load verification | complete | `AutoConfig`/`AutoProcessor` report `model_type=qwen3_vl`, `Qwen3VLForConditionalGeneration`, and `Qwen3VLProcessor`; GPU component audit forward pass succeeded with `forward_status=ok`. |
| Post-load GPU check | complete | Final GPU check after model audit reported `0 MiB, 0%`; project size is now 8.8 GB. |

## 2026-07-03 A800 Full Plan Remote Readiness

Goal: check whether `vivid_med_a800_full_next_experiment_plan.md` can be run on the remote project and upload the concrete already-existing artifacts needed for case-study/module continuation.

| Item | Status | Evidence / boundary |
| --- | --- | --- |
| A800 plan document | complete | Uploaded `vivid_med_a800_full_next_experiment_plan.md` to the remote project root. |
| Qwen3-VL base model | complete | `model/qwen3-vl-2b-thinking-new` is 4.0 GB and load/forward verified. |
| Existing instruction data | complete for existing generated data | Remote `outputs/instruction_data` has 147 files / 2.1 GB. |
| Existing next-stage/case-study configs | complete | Remote has 59 configs under the checked next-stage/case-study config roots. |
| Case-study final weights | complete | Uploaded and size-verified 62 expected final/best/probe-final weights; missing=0, mismatch=0. No `step_*.pt` or `probe_step_*.pt` middle checkpoints were uploaded. |
| Current remote project size | verified | Remote project is 29 GB after model and final-weight supplements. |
| Full CheXpert image data | not complete | Remote currently has 2,929 sampled CheXpert files for smoke/debug, not full CheXpert. |
| NIH raw data | missing | Remote check did not find raw NIH data under project `data/`. |
| MIMIC raw data | missing | Remote check did not find raw MIMIC under project `data/`; only instruction/manifest JSONL is present. |
| VinDr/PadChest external data | missing/not available locally in audited public-data root | Local public-data scan showed MIMIC and NIH but not VinDr/PadChest; A800 plan lists VinDr/PadChest as preferred main external datasets. |
| A800 plan named scripts | missing | The 23 script names listed in Section 11 of the A800 plan are not present in the current repo; existing older scripts/configs can cover prior next-stage/case-study routes, not the new A800 full plan as written. |

Remote project name decision: use `021_260129VIVID`; it matches the local project folder and contains no spaces or path separators.

## Previous Objective Closure

- Old MIMIC V1-V4 scaffold/instruction work reached artifact-backed completion for train, visual-dependence, LP, and counterfactual option diagnostics.
- Old image-shuffle deltas stayed near zero; do not use old results to claim strong visual grounding.
- No old objective processes were running when this v2 plan took over.

## Errors Encountered

| Time | Error | Attempt | Resolution |
| --- | --- | --- | --- |
| 2026-06-28 | User superseded the active proposal after old V4 diagnostics. | Continuing old paraphrase/NIH/cost gaps would no longer serve the active goal. | Stopped old-objective continuation and rewrote this plan for the Qwen3-VL v2 objective. |
| 2026-06-28 | PowerShell heredoc is not valid for inline Python in this shell. | Tried `python - <<'PY'` while checking Qwen3-VL environment. | Use `python -c` or script files for inline checks on this host. |
| 2026-06-28 | V2/V3 instruction validation found hard data-quality errors. | Strict audit rejected laterality/severity unsupported rows, null-as-absent rows, and invalid finding vocab rows. | Wrote auto-validated JSONL and rejected JSONL; P3-P5 configs now point at validated data. |
| 2026-06-28 | LP debug initially used black fallback images. | `data_root` was repo-local while CheXpert images live under `H:\Xiyao_Wang\000_Public Dataset`. | Updated LP config data root and reran debug successfully. |
| 2026-06-28 | First two-GPU queue scripts did not advance from P2/P4 to P3/P5 and log redirection only retained START lines. | Launched `scripts/run_qwen3vl_gpu0_queue.ps1` and `scripts/run_qwen3vl_gpu1_queue.ps1`; P2/P4 completed, but worker processes exited before writing second-run logs. | Preserve P2/P4 artifacts; relaunch P3/P5 directly with `cmd /c` and explicit `EXITCODE` log lines. |
| 2026-06-28 | Vision extraction manifest serialization failed. | `scripts/extract_qwen3vl_vision_backbone.py` returned raw checkpoint data in `load_info`, including Tensor values, then `json.dumps` failed. | Keep saved state files if present, patch manifest to record only scalar checkpoint metadata, then rerun extraction. |
| 2026-06-28 | Existing extraction manifests still show `language_decoder.trainable=1720574976`. | Manifest was inspected after the freeze-plan patch. | Re-run extraction P2-P5 so manifests become valid final evidence with language decoder trainable count 0. |
| 2026-06-28 | First LP aggregation read `macro_auc` at the top JSON level and produced zeros. | LP metrics are nested under `metrics.macro_auc`, `metrics.macro_f1`, and `metrics.micro_f1`. | Use nested metrics fields in final aggregation scripts/tables. |
| 2026-06-28 | Transfer queue wrappers did not reliably continue after a successful first process and some logs lacked exit-code lines. | Base/P4 and then P2/P5 JSONs were written, but wrappers exited before all queued runs. | Preserve existing artifacts, patch queue scripts to skip existing outputs, and run the remaining P3/P6 transfer commands directly with explicit `EXITCODE 0`. |
| 2026-06-28 | Final results were initially only in `outputs/final_tables` and planning files, not in the user-specified modification plan document. | User explicitly asked whether `vivid_med_qwen3vl_proposal_v2_modification_plan.md` itself had been updated. | Appended Section 11 to the modification plan with completion status, metrics, boundaries, and artifact index. |
| 2026-07-03 | Remote `python -c` model verification lost quotes through PowerShell/bash nesting. | Tried inline `python -c` over SSH. | Retried by piping a stdin Python script to remote `python -`; config and processor verification passed. |
| 2026-07-03 | `scripts/audit_qwen3vl_components.py --skip-forward` wrote JSON but failed Markdown generation with `KeyError: 'error'`. | Used skip-forward for a quick model-load-only audit. | Reran the audit without `--skip-forward`; model load and forward pass succeeded, and both JSON/Markdown audit files were written. |
| 2026-06-30 | GPU1 `shuf_10k_8k` third-lane scale starts exceeded the safe memory window while heavier postprocess was still active. | Tried root-isolated manual starts with 23600 MiB and 24000 MiB guards. | Guards stopped only the manual `shuf_10k_8k` trees; restarted after `prog_mix_dualmargin` freed its postprocess lane. |
| 2026-06-30 | GPU1 `storymix_10k_8k` third-lane pressure attempt exceeded the safe memory window during startup. | Launched a distinct-id manual `storymix_10k_8k` lane alongside `shuf_10k_8k` training and `prog_mix_sameq` NIH transfer. | The 24000 MiB root guard triggered at 24253 MiB and stopped only `storymix_10k_8k`; preserve as failure-case evidence and resume after the GPU1 postprocess lane is free. |
| 2026-06-30 | GPU1 three concurrent 10k-scale training lanes exceeded the safe memory window. | After `prog_mix_sameq` freed its postprocess lane, resumed `storymix_10k_8k` with `shuf_10k_8k`, then probed `prog_mix_10k_8k` as a third training lane with a lower 22600 MiB guard. | Total memory reached 24209 MiB; the probe guard stopped `prog_mix_10k_8k`, and the older `shuf_10k_8k` 24000 MiB guard also paused `shuf_10k_8k`. `storymix_10k_8k` stayed active; `shuf_10k_8k` was resumed from `step_2000.pt` with a lower guard. |
| 2026-06-30 | A second GPU1 three-training-lane probe for `prog_mix_10k_8k` again crossed the safe memory boundary and exposed Windows PID reuse in the tree guard. | Used a trigger-gated root with a lower 22400 MiB, 1-second guard so the third lane would be stopped before established lanes. | The guard stopped `prog_mix_10k_8k` at 22609 MiB before training progress, while `shuf_10k_8k` and `storymix_10k_8k` kept advancing. Patched `scripts/guard_gpu_memory_process_tree.ps1` to ignore stale children created before the guarded root, then restored the GPU0 token-scale guard. |

---

# 2026-07-03 A800 Available Data Upload

## Active Scope

Finish uploading every A800-plan-relevant dataset that is already present on the local workstation into the remote project at `/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/021_260129VIVID`, without waiting for datasets that are not locally available.

## Data Upload Checklist

| Dataset / bucket | Status | Local source | Remote target | Notes |
| --- | --- | --- | --- | --- |
| Full CheXpert small | in_progress | `H:\Xiyao_Wang\000_Public Dataset\CheXpert-v1.0-small` | `data/dataset/CheXpert-v1.0-small` | Resumed after pause; metadata/valid plus patient chunks 000-008 have done markers; chunk 009 local tar exists and is being verified/recovered. |
| NIH Chest X-rays | deferred_by_user | `H:\Xiyao_Wang\000_Public Dataset\NIH Chest X-rays` | not uploading now | User requested on 2026-07-03: "NIH先不用传了"; keep as explicit skip/defer marker, not completion. |
| MIMIC-CXR images/reports | pending | `H:\Xiyao_Wang\000_Public Dataset\mimic-cxr\mimic-cxr` | `data/dataset/mimic-cxr` | Needed for MIMIC-as-source/external fallback and existing report-grounded workflows. |
| MIMIC-CXR less | pending | `H:\Xiyao_Wang\000_Public Dataset\mimic-cxr_less` | `data/dataset/mimic-cxr_less` | Local auxiliary MIMIC subset/CSV bucket. |
| MIMIC supplementary files | pending | `H:\Xiyao_Wang\000_Public Dataset\mimic-cxr\mimic-cxr附加文件` | `data/dataset/mimic-cxr/mimic-cxr附加文件` | Upload if present; includes RadGraph/adjacency-style files. |
| VinDr-CXR / PadChest | unavailable_locally | not found in current public-data root | not uploaded | A800 plan prefers these as main external datasets, but they are not available locally in the audited data root. |
| Kaggle `xhlulu/vinbigdata` 512 PNG | downloaded_image_only | `data/dataset/vinbigdata_xhlulu_512png` | not uploaded | Same VinBigData/VinDr-CXR image family, but this Kaggle package contains only 15k train PNGs, 3k test PNGs, and `train_meta.csv`; no labels/bboxes/manifest, so it is not sufficient by itself for the A800 plan's VinDr-CXR external AUC/ECE/AUPRC evaluation. |

## Upload Rules

- Transfer in chunks and verify remote size/count after each dataset family.
- Preserve existing remote runnable package and final weights.
- Do not upload intermediate checkpoints unless separately requested.
- Record any interrupted chunk and resume with a different chunking strategy instead of retrying the identical failing stream.

## Errors Encountered

| Time | Error | Attempt | Resolution |
| --- | --- | --- | --- |
| 2026-07-03T15:00:00+08:00 | PowerShell `New-Item -LiteralPath` is unsupported in this shell. | Started the first local tar staging command for CheXpert metadata/valid. | Switch staging commands to `New-Item -Path` and keep chunked tar/scp/extract flow. |
| 2026-07-03T17:10:00+08:00 | Single full CheXpert train tar stalled at about 6.26 GB with no CPU progress. | Tried one 10 GB class tar file for all `train/`. | Stopped the stalled tar/helper queue before upload, kept verified metadata/valid, and switched CheXpert train to patient-range chunk tar files. |
| 2026-07-03T18:32:00+08:00 | SSH command layer repeatedly closed or hung during CheXpert chunk transfer/verification. | Retried chunked scp/extract and tested SSH debug/TTY behavior. | User requested pause; stopped A800 transfer processes, wrote `.remote_upload_tmp/PAUSED_BY_USER_20260703.txt`, and left chunk 009 for explicit verification before resuming. |

---

# 2026-07-04 Remote Shared Model Completion

## Active Scope

Upload every remaining local model under `H:\Xiyao_Wang\001_models` that is not already present in the remote shared model cache, then expose those models to the project through symlinks in `021_260129VIVID/model/`.

## Model Upload Checklist

| Bucket | Status | Local source | Remote target | Notes |
| --- | --- | --- | --- | --- |
| Existing Qwen models | complete | already remote | `/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/model` | Verified `Qwen3.5-4B` (`18G`), `Qwen3.5-9B` (`25G`), and `qwen3-vl-2b-thinking-new` (`4.0G`); project symlinks exist. |
| Remaining `001_models` directories | complete | `H:\Xiyao_Wang\001_models` | `/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/model` | `24/24` chunks complete. Normal `scp/sftp` stayed unreliable, so chunks `0018`-`0023` were completed through remote curl over an SSH reverse tunnel from remote `127.0.0.1:18766` to the local `.remote_upload_tmp` HTTP server on `127.0.0.1:18765`; chunk `0023` used Python PAX tar for the Unicode `新建文件夹` paths. |
| Project model links | complete | shared remote model cache | `/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/021_260129VIVID/model` | Project `model/` has 42 symlinks pointing to `../../model/<name>`, including the existing Qwen 2B/4B/9B models and all newly uploaded top-level model directories. |

## Transfer Rules

- Keep remote canonical weights in the shared singular directory `~/projects/xiyaowang/model/`; do not create a parallel `models/` copy.
- Transfer in resumable chunks and rely on local plus remote `.done` markers before skipping a chunk.
- Do not re-upload `Qwen3.5-4B`, `Qwen3.5-9B`, or `qwen3-vl-2b-thinking-new`.
- Keep local temporary tars under `.remote_upload_tmp/` and remove each chunk tar after successful remote extraction.
