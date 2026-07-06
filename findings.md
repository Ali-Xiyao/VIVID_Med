# VIVID-Med Execution Findings

## 2026-07-06 CVCP/CCSH Full Plan Completion Findings

- The active `vivid_med_cvcp_ccsh_full_next_experiment_plan.md` route reached artifact-backed completion for the local formal protocol: 27/27 Qwen3-VL CVCP/CF/token-weighting/dual-margin rows completed training plus LP, NIH appendix, visual-dependence, counterfactual, A/B-swap, and paraphrase diagnostics; 18/18 module-combo rows completed embedding export and module-stack training.
- The strongest empirical training row is `cvcp_v1_sameq_full` with CheXpert AUC `0.748633` and hard-shuffle delta `0.641917`. `sameq_cf_30` is close on CheXpert AUC (`0.744668`) with a strong repaired hard-shuffle signal (`0.481282`).
- The strongest deployable module readout is not the same as the best training row: replay-backed CCSH/stack rows on `cvcp_v4_replay_10k` reach best binary AUC `0.893317` and state accuracy `0.766000`, including `cvcp_replay_ccsh`, `ceq_hnmb_ccsh`, and `ceq_auch_ccsh`.
- The final story should stay two-part: SAMEQ-style CVCP gives the strongest image-specific training signal, while CCSH on replay/CVCP backbones gives the strongest deployable consistency readout. Do not collapse this into a single universal winner.
- External claims remain bounded: NIH is complete only as appendix/stress-test for all 27 rows; the VinDr/VinBig package is image-only without labels/bboxes; PadChest is missing locally. These boundaries are recorded in `outputs/final_tables/external_eval_results.md`.
- Model-comparison claims remain compatibility/audit claims for non-Qwen3VL families. Qwen3VL is the formal trainer route; InternVL/LLaVA/Llama/Qwen3.5 directories are present but require architecture-specific adapters before training-comparison claims.

## 2026-07-01 Case Study + Module Plan Takeover

- Active goal is now `vivid_med_case_study_modules_next_experiment_plan.md`; the user explicitly asked to complete the full document, including expansion/module content, before stopping.
- Skill pre-flight used `superpowers:using-superpowers`, `planning-with-files`, `executing-plans`, and `verification-before-completion`. `subagent-plan-decomposer` and `superpowers:subagent-driven-development` were inspected because the plan is large, but subagent spawning is not used unless explicitly needed and authorized.
- Session catchup produced no output. Existing `task_plan.md/findings.md/progress.md` were from the completed next-stage comprehensive objective, so this new plan needs a fresh active section rather than treating previous completion as sufficient.
- Memory for this repo emphasizes artifact-backed completion: output files, checkpoint contents, task states, logs, GPU/process checks, and failure-case preservation beat planning prose.
- The target plan explicitly says `SHUF-TW-clinical` is only a candidate. The required next step is not another post-hoc best-row selection; it is case study, seed stability, NIH/domain diagnosis, curriculum retry, moduleization, and locked comparison.
- Immediate required source additions named by the plan:
  - Case study scripts: `mine_pairwise_case_studies.py`, `audit_nih_transfer_failure.py`, `audit_hard_negative_quality.py`, `audit_curriculum_leakage_cases.py`, `build_casebook_markdown.py`.
  - Stability scripts: `run_multiseed_manifest.py`, `bootstrap_auc_ci.py`, `paired_bootstrap_method_delta.py`, `summarize_multiseed_results.py`.
  - NIH/domain scripts: `audit_label_mapping_nih.py`, `run_nih_full_transfer.py`, `compute_domain_shift_mmd.py`, `plot_dataset_embedding_umap.py`.
  - Curriculum v2 scripts: `build_curriculum_v2_schedule.py`, `generate_curriculum_v2_instructions.py`, `train_qwen3vl_curriculum_v2.py`.
  - Module files: `clinical_evidence_query.py`, `answerability_uncertainty_head.py`, `hard_negative_memory_bank.py`, `domain_robust_adapter.py`, `clinical_consistency_head.py`, `case_driven_curriculum_scheduler.py`.
- Required final reports named by the plan: `case_study_summary.md`, `multiseed_stability.md`, `nih_domain_audit.md`, `module_candidate_results.md`, and `locked_final_comparison.md` under `outputs/final_tables/`.

## 2026-06-29 Next-Stage Comprehensive Plan Takeover

- Active goal is now `vivid_med_next_stage_comprehensive_experiment_plan.md`; the user explicitly asked to complete the full document, including expansion items, before stopping.
- Skill pre-flight completed using `superpowers:using-superpowers` and `planning-with-files`. `subagent-plan-decomposer` and `superpowers:subagent-driven-development` were read because the plan may later benefit from independent implementation/review packages, but main-agent coordination remains necessary while the worktree is large and dirty.
- Session catchup produced no unsynced-context report output. Current planning files were from the P4-v2 objective, so they needed a new active section rather than being treated as completion evidence.
- Memory for this repo emphasizes that authoritative completion signals are current output files, checkpoints, task/queue states, logs, and GPU/process checks, not planning prose.
- UTF-8 reading is required for this Chinese plan; plain PowerShell output initially showed mojibake.
- Existing P4-v2/SHUF-3k evidence is strong baseline evidence only: the new plan cites SHUF-3k at CheXpert AUC 0.726709, NIH AUC 0.568045, random shuffle delta 0.0716429, hard shuffle delta 0.0806744, and CF acc 0.870748.
- Exact G1 script audit found the seven plan-named scripts are missing:
  - `scripts/generate_storymix_instructions.py`
  - `scripts/generate_sameq_shuf_pairs.py`
  - `scripts/generate_multi_negative_shuf.py`
  - `scripts/audit_instruction_leakage_v2.py`
  - `scripts/build_progressive_mixture_schedule.py`
  - `scripts/build_token_weight_map.py`
  - `scripts/mine_hard_negatives_from_embeddings.py`
- Existing training/evaluation trunk is reusable: `scripts/train_qwen3vl_clinical_instruction.py`, `scripts/train_qwen3vl_vision_lp.py`, visual-dependence, counterfactual, paraphrase, LP transfer, P4-v2 D6/D7 builder, and P4-v2 summarizer are present.
- Current `configs/qwen3vl_instruction/` covers P2/P3/P4/P5, P4-v2 CF/SHUF/QA8, LP, and transfer configs, but does not yet cover the new StoryMix, SAMEQ, P2 mask variants, curriculum/progressive, SHUF-K, token-weighting-on-SHUF, scale, or training-policy runs.
- Existing `outputs/final_tables/qwen3vl_p4v2_*` tables and P4-v2 D6/D7/QA8 validation outputs exist and should seed the next-stage baseline table, not close the new objective.
- The plan's required core work decomposes into: A1 JSON mask diagnostics, A2 rich QA mixture, A3 workflow/curriculum, A4 SHUF++, A5 token weighting, B1 scale, B2 training policy, B3 model/model-type controls, B4 external transfer, B5 calibration/AUPRC, B6 robustness/bias, B7 leakage audit 2.0, B8 qualitative visualization, G1 scripts, G2 training features, and G3 per-run packages.

## 2026-06-29 Next-Stage Preparation Artifacts

- Implemented the seven G1 plan-named scripts plus P2/config helper scripts: `generate_storymix_instructions.py`, `generate_sameq_shuf_pairs.py`, `generate_multi_negative_shuf.py`, `audit_instruction_leakage_v2.py`, `build_progressive_mixture_schedule.py`, `build_token_weight_map.py`, `mine_hard_negatives_from_embeddings.py`, `prepare_p2_loss_mask_variants.py`, and `prepare_next_stage_configs.py`.
- Extended the Qwen3-VL instruction collator/trainer for `loss_masking`, token weighting, image-shuffle margin, answer margin, multi-negative hard images, `config_snapshot.json`, progress events, step metrics, and `training_log.txt`.
- Generated next-stage instruction data under `outputs/instruction_data/next_stage/`: P2 compact train/val 1000/1000, P2 field-query 14000/14000, Balanced/CF-heavy/SHUF-heavy/Clinical-rich QA8 train sets around 23k rows, StoryMix QA5/8/10/12 train sets 14568/23269/29031/34877 rows, SAMEQ-SHUF train/val 9238/566, and SHUF-K2/K4 train/val 14333/953 each.
- Ran leakage audit 2.0 for the current next-stage JSONL files into `outputs/final_tables/next_stage_audits/`. P2 variants are 100% accepted; SAMEQ/SHUF-K are roughly 92-93% accepted with A/B balance near 50%; rich mixtures expose expected duplicate-question-per-image flags as QA/image increases.
- Generated 30 YAML configs under `configs/qwen3vl_instruction/next_stage/`, token-weight snippets under `configs/qwen3vl_instruction/next_stage_snippets/`, and schedule/config manifests under `outputs/next_stage_manifests/`. Only the two 10k placeholder configs intentionally point to missing `next_stage_missing` paths.
- Debug-smoked `P2-no-punct` and `SHUF-K4`; both wrote the required baseline per-run package including `config_snapshot.json`, `metrics_final.json`, `metrics_step_1.json`, `progress.json`, `resolved_config.yaml`, `runtime_summary.json`, and `training_log.txt`.
- Added `docs/next_stage_requirement_ledger.md` as the current requirement ledger. It marks A1/A2 preparation complete, A3/A4/A5 config/data support mostly ready, and Part B extension gaps still open for 10k scale, training policy, model scale, external transfer, calibration/AUPRC, prompt robustness/option bias, and qualitative visualization.

## 2026-06-29 Next-Stage Expansion Artifacts

- Phase 1 formal queues are active. `P2-value-only` has resumed cleanly from step 500 and reached step 1500+ eval; `P2-field-query` was relaunched with corrected 1k validation and reached step 1000 eval.
- Failure evidence is preserved under `outputs/failure_cases/next_stage/` for P2-field-query full-val eval and wrapper interruption. These are engineering cases, not silent retries.
- Training resume now restores checkpoint `global_step` and `best_val_loss`; queue scripts auto-resume from the newest checkpoint when `metrics_final.json` is missing.
- In-batch negative support is implemented and smoke-verified: a 2-row D7 batch produced 2 `negative_input_ids` rows from batch-rotated images.
- 10k scale data is now present and explicitly labeled as UMS-structured rather than GLM-report-derived: `ums_chexpert_10k_facts.jsonl`, `shuf_10k_train/val.jsonl`, `storymix_10k_train/val.jsonl`, with leakage audits for train/val.
- Config matrix now has 36 runnable configs and 36 LP configs. Newly covered items include `InBatch-SHUF`, TRAIN-CONN, TRAIN-LAST4, TRAIN-FULLVISION, SHUF-10k-8k, StoryMix-10k-8k, PROG-Mix-10k-8k, and PROG-Mix-TW-10k.
- B5 metrics support is expanded: LP/transfer metrics now include macro/per-label AUPRC, ECE, and Brier score.
- B6 support is expanded with A/B swap diagnostic JSONL generation and postprocess hooks for Phase 1 runs.
- B8 support is expanded with `next_stage_qualitative_cases.md` and 24 side-by-side hard-negative image assets. This is qualitative case review, not Grad-CAM/attention attribution.
- External/model audit is documented in `docs/next_stage_external_model_availability.md`: NIH external UMS is available; MIMIC reports/images exist but AUC transfer needs a UMS label manifest; PadChest/VinDr are unavailable in current repo/data paths; 4B/8B local models need separate VRAM audits before formal larger-VLM claims.
- Phase N4 P2 formal diagnostics are now complete: `p2_value_only` finished at 3000 steps with best validation loss 0.25470546504855157; `p2_field_query` finished at 3000 steps with best validation loss 0.1867514458609803. Both keep the language decoder frozen.
- Phase N4 queue behavior note: the GPU queue wrappers exited after the first completed run, so they were relaunched to skip completed P2 runs and continue with `storymix_qa8` on GPU0 and `shuf_heavy_qa8` on GPU1.
- A4 SHUF++ no longer has only a boundary for Mined/SelfHard: the repo now has scripts to export SHUF checkpoint embeddings, mine same-finding answer-mismatch negatives, score wrong-image NLL for confidence mining, oversample self-hard rows, and run a progressive-hardneg schedule. The actual mined JSONL still needs a GPU pass after a lane is free.
- Workflow/progressive configs previously carried `curriculum_schedule` as metadata only. The trainer now supports actual step-window sampling when rows include `curriculum_start_step` and `curriculum_end_step`; a small smoke materialized four stages and confirmed the active set changes by global step.
- Formal CUR/PROG configs now point to materialized JSONL with curriculum windows. Leakage audit flags are high for these files, mostly from duplicate-question-per-image introduced by stage resampling; preserve this as a cost/quality signal rather than treating curriculum data as clean by default.


## 2026-06-28 P4-v2 / Scale Plan Takeover

- Active goal is now `vivid_med_qwen3vl_p4v2_scale_experiment_plan.md`; prior Qwen3-VL v2 completion artifacts are useful context but do not prove the new scale/hard-counterfactual plan is complete.
- The new plan requires P4-v2 facts, D6 hard A/B counterfactual data, D7 hard image-shuffle data, instruction leakage/distribution/manual audit outputs, training/evaluation runs, and final write-back into the same markdown file.
- Memory and repo history indicate that authoritative completion signals in this repo are current output files, checkpoint contents, logs, process/GPU state, and generated tables, not plan prose.
- Initial process listing found multiple Python processes that may be lingering generation/evaluation jobs; command lines and GPU state still need to be audited before launching new API-heavy or GPU-heavy work.
- API key/endpoint details must remain environment-only; raw keys must not be written to code, configs, logs, or planning files.
- Resource audit found GPU0/GPU1 idle, no compute apps, and a present `ZHIPU_API_KEY` environment variable.
- Four active GLM API processes were generating older `v3_mimic_5k/train_extra4k` shards with `prompts/glm_instruction_generation_report_grounded_v3_counterfactual.txt`, not the new plan's standardized D6/D7 outputs.
- Pause point for old extra4k generation: shard0/shard1/shard2/shard3 API logs were roughly 277/226/259/323 successful API rows out of 500, and raw JSONL lines were 880/832/804/800. The jobs use `--resume`, so they can be resumed after P4-v2 data generation.
- Added P4-v2 support scripts: `scripts/generate_p4v2_facts_with_glm.py`, `scripts/build_p4v2_d6_d7.py`, `scripts/audit_p4v2_instruction_quality.py`, and `prompts/glm_p4v2_fact_extraction.txt`.
- Two-sample GLM fact smoke completed with 2 written, 0 errors. D6/D7 smoke built 10 records each; P4-v2 smoke audit reported 100% accepted, 0% leakage, and 50/50 A/B balance. Existing validator cross-check accepted 10/10 D6 smoke records with 0 rejects.
- Added hard-shuffle support to `scripts/evaluate_qwen3vl_visual_dependence.py` and configurable image-shuffle margin loss to the Qwen3-VL instruction trainer/dataset path for D7 runs.
- Core P4-v2 configs now exist for `CF-1k-3k`, `S-P4-3k`, `CF-3k-5k`, `CF-3k-8k`, and `SHUF-3k`; YAML parsing passed.
- Added `scripts/summarize_p4v2_results.py`, which writes the plan-requested `qwen3vl_p4v2_*` final tables with explicit `missing` statuses until artifacts exist.

## 2026-06-28 Qwen3-VL Resume

- Active goal is again `vivid_med_qwen3vl_proposal_v2_modification_plan.md`; old MIMIC V1-V4 document line remains closed and should only be used as baseline evidence.
- Official Zhipu API docs distinguish the general endpoint `https://open.bigmodel.cn/api/paas/v4` from the Coding Plan endpoint `https://open.bigmodel.cn/api/coding/paas/v4`; the latter is required for Coding Plan use, and the API key should be supplied as a Bearer token via environment variable rather than hard-coded.
- Formal Qwen3-VL P2/P3/P4/P5 instruction training artifacts exist with `global_step=1000` and frozen language decoder counts in their `metrics_final.json`.
- Formal CheXpert 1k LP artifacts exist for base/P2/P3/P4/P5, but metrics live under nested `metrics.*` fields; top-level `macro_auc` reads are invalid.
- Existing extraction manifests were created before the freeze-plan manifest patch was rerun; they still report the language decoder as trainable. Re-run extraction P2-P5 before treating extraction manifests as final evidence.
- Remaining v2 plan gaps include Qwen3-VL visual-dependence diagnostics, Qwen3-VL counterfactual diagnostics, paraphrase/template sensitivity, subgroup/cost/transfer evidence, and the final requirement audit.

## 2026-06-28 Old Document Closure

- User requested that the old document be closed and not continued afterward.
- Verified old MIMIC V1/V2/V3/V4 evidence tables are present: main results, V1-V4 comparison, visual-dependence results, counterfactual results, and answer-type diagnostics.
- Final old-document interpretation: V4 lowers teacher-forced validation loss relative to V3 and weakly improves option-subset counterfactual accuracy, but V2 remains best on LP macro-AUC and image-shuffle deltas remain near zero.
- Boundary: strong image-specific grounding is not validated; broader old-document gaps such as NIH transfer, paraphrase robustness, subgroup analysis, and cost tables are not pursued after this closure unless the user reopens the objective.
- Final process check found GPU0/GPU1 idle and no active Python experiment process; residual `cmd`/`powershell` processes are tool/plugin shells.

## 2026-06-28 Qwen3-VL Proposal v2 Takeover

- User superseded the previous `vivid_med_clinical_instruction_proposal.md` objective with `vivid_med_qwen3vl_proposal_v2_modification_plan.md`; do not continue old-document-only gaps.
- No old instruction training/evaluation/generation process was running at takeover, and both GPUs were idle.
- New proposal changes the main method from piecemeal `timm ViT + text-only Qwen/Qwen-Coder + new projector` to a pretrained VLM-coupled route.
- New main route: local Qwen3-VL VLM, frozen language decoder, trainable vision tower and visual connector, report-grounded GLM clinical instructions, then discard LLM and evaluate the vision tower.
- Required first executable gate is Qwen3-VL component audit, not immediate full training.
- Local model root `H:\Xiyao_Wang\001_models` does not contain a directory literally named `Qwen3-VL-2B-Instruct`, but does contain `qwen3-vl-2b-thinking-new`, `Qwen3-VL-4B-Instruct`, and `Qwen3-VL-8B-Instruct`.
- `H:\Xiyao_Wang\001_models\qwen3-vl-2b-thinking-new\config.json` has `model_type=qwen3_vl` and architecture `Qwen3VLForConditionalGeneration`; use it as the first local 2B Qwen3-VL candidate unless audit fails.
- In conda env `vivid`, `transformers 5.0.0` loads `Qwen3VLConfig` and `Qwen3VLProcessor`, and the processor has `apply_chat_template`; CUDA is available.
- Coding Plan domestic endpoint remains `https://open.bigmodel.cn/api/coding/paas/v4`; raw API key must stay out of repo files/logs.
- Qwen3-VL debug training works with `AutoModelForImageTextToText` and `Qwen3VLProcessor`; answer-only labels are generated through `apply_chat_template`.
- The validated freeze plan leaves `vision_tower` plus `visual_connector` trainable and keeps `language_decoder` frozen.
- V2/V3 report-grounded instruction data is usable after auto-validation, but hard rejects show real data-quality issues: unsupported laterality/severity claims, null-as-absent errors, and a few invalid finding names.
- Counterfactual-format warnings remain frequent, especially in V3; treat this as a data-quality limitation for P4/P5 rather than as a solved counterfactual benchmark.
- CheXpert relative image paths require `H:\Xiyao_Wang\000_Public Dataset` as `data_root`; repo-local `.` causes black-image fallbacks in Qwen3-VL LP.
- True D0 fixed-JSON Qwen3-VL data is now available and validated; P2 should use `outputs/instruction_data/glm_validated/d0_train_validated.jsonl` and `d0_val_validated.jsonl`, not the earlier V1 proxy.
- Formal P2/P4 runs completed with frozen language decoder and trainable vision tower/connector. P2's very low validation loss reflects the fixed-schema target and should not be interpreted as visual grounding by itself.
- Qwen3-VL P2-P5 visual-dependence diagnostics completed: question-only/black-image loss deltas are large, but image-shuffle deltas stay small, so image presence sensitivity is supported while strong image-specific grounding is not validated.
- Qwen3-VL P4/P5 counterfactual diagnostics completed on option-formatted subsets. P4/P5 prefer correct options on that subset, but most `counterfactual_choice` rows are not explicit A/B/C/D option records, so this remains a data-quality boundary.
- Qwen3-VL P2-P5 paraphrase/template diagnostics completed. Clinical rewrites cause small mean NLL increases; style rewrites are consistently harder, so paraphrase robustness is measured but not fully solved.
- NIH 1k transfer completed for Base/P2/P3/P4/P5/P6 with 0 missing images after resolving NIH `extensions.image_index` paths. P4 has the best NIH macro-AUC among these runs, but the margin over Base is small.
- P6 data-only no-LM control completed as a Qwen3-VL vision tower + linear head trained directly on CheXpert UMS labels. It does not use GLM D3 instruction data, so it is a useful no-LLM control but not a perfect D3 data-only match.
- The user-requested final write-back is now in `vivid_med_qwen3vl_proposal_v2_modification_plan.md` Section 11, including completion status, main metrics, visual-dependence/counterfactual/paraphrase boundaries, and artifact index.

## 2026-06-28 Initial Context

- `AGENTS.md` now requires a skill/superpower pre-flight before any task or command sequence.
- `planning-with-files` applies because the objective is a complex multi-step experiment execution workflow.
- Memory indicates that for this repo, authoritative completion signals are output files, checkpoint contents, queue/task state, log tails, and GPU/process checks rather than planning prose alone.
- Current repo has many pre-existing modified/untracked experiment files; do not revert or overwrite them casually.
- No raw GLM API key should be written into repo files.

## 2026-06-28 Proposal Requirements Extract

- The new repository-level objective is `Clinical Evidence Instruction Pretraining for Deployable CXR ViTs`.
- Core direction: convert CXR report + UMS schema into GLM-generated clinically grounded visual instructions, train deployable ViT/ViT+small head, and avoid deployment-time LLM.
- Required instruction types include finding verification, evidence phrase, laterality/location, severity, uncertainty, answerability, image-report consistency, counterfactual choice, temporal comparison, and device QA.
- Required data versions include V0 fixed JSON, V1 label-to-QA, V2 report-grounded QA, V3 report-grounded QA plus counterfactual, V4 token weighting, V5 counterfactual margin, V6 question-only/image-shuffled controls, optional V7 visual verifier, and V8 decoder controls.
- Required implementation tasks are G0 schema, G1 GLM generator, G2 filter, G3 stats, G4 manual audit sample, G5 instruction dataloader, G6 instruction trainer, G7 visual-dependence evaluator, and G8 downstream LP evaluation.
- Minimum runnable matrix: fixed JSON V0 1k, label-to-QA V1 1k, report-grounded V2 1k, V3 1k, no-LM schema 1k, question-only eval, image-shuffle eval, and visual token weighting.
- Final result package should include instruction dataset stats, main results, visual dependence results, ablations, counterfactual results, cost table, audit summary, failure cases, plus per-run configs/metrics/runtime/checkpoints/predictions.
- Success requires not only CheXpert AUC but also NIH transfer, counterfactual pairwise accuracy, image-shuffle drop, question-only degradation, paraphrase robustness, and rare/high-null/uncertain group behavior.

## 2026-06-28 Current Artifact Audit

# 2026-07-04 CVCP/CCSH Full Plan Takeover

- Formal CVCP training is now routed to `F:/Xiyao_Wang/021_260129VIVID_cvcp_ccsh_outputs/qwen3vl` because the project `H:` drive has only about 36-39 GB free. Configs remain in-repo; heavy metrics/checkpoints use the F-drive output root.
- The postprocess layer must use `checkpoints/final.pt`, not `best.pt`, for the new CVCP runs because the generated configs intentionally use final-only checkpointing to control storage growth.
- NIH is being treated as `appendix_stress_test` only. The postprocess queue uses a 1k NIH appendix subset for uniform stress evidence; VinDr/PadChest remain explicit main-external blockers unless labels/manifests are supplied.
- Current `model_comparison_results` is a compatibility audit, not a completed teacher-training comparison. Qwen3VL 2B is the formal active queue; Qwen3VL 4B/8B still need VRAM smoke before long runs; InternVL/Llama vision and text-only scaffolds require separate trainer/adapter implementations to be fair under the plan's same-protocol rule.
- Module-combo rows are now queued as embedding-backed module-head experiments after formal backbone checkpoints complete. This preserves Base+CCSH as a raw-Qwen3VL head-only baseline and avoids reusing old embedding-level module ablations as if they were new CVCP module-combo results.

- User started an active `/goal` for `vivid_med_cvcp_ccsh_full_next_experiment_plan.md` and explicitly requested completion of every required and optional experiment under the document's formal protocol.
- User approved use of both local RTX 3090 GPUs. Initial `nvidia-smi` snapshot showed GPU0 and GPU1 idle at `0 MiB` and `0%`.
- Skill preflight used `superpowers:using-superpowers`, `planning-with-files`, `subagent-plan-decomposer`, `subagent-driven-development` as conditionally applicable, `executing-plans` as fallback, and `verification-before-completion` for final claims.
- Session catchup produced no output. The prior planning files describe completed case-study/module and upload handoff work, so this new CVCP/CCSH goal needs a fresh active section rather than inheriting old completion claims.
- Memory guidance is useful only as a boundary pattern: write results back into the source markdown, refresh audits after final edits, keep GPU/process evidence fresh, and do not trust old row counts without revalidation.
- The target document frames the main story as Clinical Visual Curriculum Pretraining with deployable CCSH/CEQ-style modules, not as a final-best `SHUF-TW-clinical` claim.
- Current local dataset directories include CheXpert, NIH, MIMIC-derived processed data from earlier runs, AMOS22/KITS21/LIDC/OrganMNIST, and `vinbigdata_xhlulu_512png`. There is no current local `PadChest` directory in the first dataset-directory audit. The downloaded VinBigData package is image/meta-only per previous project findings and needs labels/manifest before it can satisfy a VinDr/VinBig external-evaluation row.
- Initial script-existence audit command failed before returning data; rerun before claiming which target-plan scripts exist or are missing.
- Generated `docs/cvcp_ccsh_requirement_ledger.md` and `outputs/final_tables/cvcp_ccsh_requirement_ledger.csv` with 239 rows. The conservative first-pass status distribution is: 21 missing scripts, 5 missing target final tables, 143 open experiment/metric rows, 36 experiment/metric rows with name-like candidate evidence requiring exact protocol audit, and 14 open qualitative/casebook/visualization rows.
- Exact target-plan script names are absent, but useful analogs exist: curriculum, SAMEQ, SHUF, bootstrap, case-study module, NIH/domain, and Qwen3VL training/eval scripts from prior plans. These should be adapted behind target-named entry points rather than counted complete as-is.
- Broad multi-directory `rg` for all CVCP/CCSH-related terms timed out, reinforcing that audits should be ledger-driven and targeted rather than whole-tree scans.
- Generated `docs/cvcp_ccsh_readiness_audit.md` and `outputs/final_tables/cvcp_ccsh_readiness_audit.csv` with 56 rows. It maps all 21 missing target scripts to existing analogs, confirms 5 local model-comparison buckets, records 5 missing/existing target output statuses, and separates reusable prior-protocol artifacts from new CVCP/CCSH completion.
- Local model availability: Qwen3VL current main (`qwen3-vl-2b-thinking-new`, 4B, 8B), InternVL (2.5/3.5 1B-8B), Llama-3.2-11B-Vision, Qwen3.5 2B/4B/9B, and Qwen2.5-Coder-7B directories exist under `H:/Xiyao_Wang/001_models`. Each needs a model-specific smoke before training claims.
- Reusable artifact boundary: case-study multiseed/downstream rows are complete for SHUF-3k, SHUF-TW-clinical, SAMEQ-SHUF-3k, and SHUF-K4; module ablations are complete at embedding-head level; next-stage 39-run audit is complete for the older protocol. None of these close new CVCP+CCSH, CEQ+CCSH, HNMB+CCSH, teacher-comparison, or external-main rows by themselves.
- Implemented `scripts/cvcp_ccsh_driver.py` plus 21 target-named wrappers. The exact script rows in the target document now exist and compile; readiness audit now reports `target_script exact_exists=21`.
- Generated formal CVCP/CCSH data artifacts: 14 CVCP curriculum datasets, 3 SAMEQ-CF-compatible datasets, 3 SHUF-K-CF-compatible datasets, 28,666 CCSH statement rows, and 14,333 CEQ target rows.
- Ran v3 data audits on generated key datasets. These audits are first-pass automatic gates; rows with flags require data filtering/rebalancing before final training claims if they affect selected candidates.
- Created target-named final-table outputs for casebook, CVCP training, module combo, model comparison, external eval, and locked comparison. Some are readiness/prior-protocol summaries until the new formal training rows finish.
- The 2026-07-04T17:39 goal-continuation/tool interruption is treated as an external interruption, not a model-quality result. Because CVCP configs use final-only checkpointing, interrupted training directories without `metrics_final.json` and without `checkpoints/final.pt` are now archived under the F-drive `interrupted_runs/` area before clean rerun. Official CVCP/CCSH tables should only count runs with final metrics/checkpoints or explicit data-unavailable boundaries.

---

- GPUs are currently idle: GPU0 and GPU1 both report 0 MiB memory use and 0% utilization.
- No VIVID training, instruction, or GLM generation process is currently running.
- Existing old-revision artifacts under `outputs/final_tables` and `outputs/failure_cases` are extensive but correspond to schema/frozen-LM/data-scaling work, not the new instruction-data proposal.
- No existing `data/instructions/*`, `scripts/train_cxr_instruction.py`, `data/cxr_instruction_dataset.py`, or GLM instruction generator was found by filename search.
- `data/splits/chexpert_train_{1k,3k,10k,30k}.jsonl` and `chexpert_val_fixed.jsonl` exist and contain UMS `findings`, `answerability`, `uncertainty`, and `extensions.original_path`.
- Current CheXpert small files include `train.csv`, `valid.csv`, and images, but not radiology report text files.
- First-pass conclusion: V1 label-to-QA/answerability can be generated from UMS now; V2/V3 report-grounded GLM generation needs a report text source or a fallback strategy clearly marked as non-report-grounded.

## 2026-06-28 GLM Coding Plan Setup

- Official docs identify the domestic BigModel/Zhipu Coding Plan OpenAI-compatible base URL as `https://open.bigmodel.cn/api/coding/paas/v4`.
- Official docs identify the international Z.AI Coding Plan OpenAI-compatible base URL as `https://api.z.ai/api/coding/paas/v4`.
- `scripts/generate_clinical_instructions.py` defaults to the domestic Coding Plan URL and reads the API key from `GLM_API_KEY`.
- Connectivity smoke test with the user-provided key, domestic Coding Plan URL, and `glm-5.2` returned a successful OpenAI-compatible chat-completions response.
- GLM generator smoke test produced 4 parsed records from 1 CheXpert sample with 0 errors at `data/instructions/raw/glm_v1_smoke/train.jsonl`.
- Secret check found no raw key written under `data/instructions`, `outputs/instruction_generation`, `scripts`, `docs`, `prompts`, or the planning files.

## 2026-06-28 Instruction V1 Debug Training

- Implemented `data/cxr_instruction_dataset.py` and `scripts/train_cxr_instruction.py`.
- Added local debug config `configs/debug_instruction_v1_qwen25_coder_7b.yaml`.
- `H:/Xiyao_Wang/001_models/Qwen3.5-0.8B` failed because the current `vivid` Transformers does not recognize `model_type=qwen3_5`.
- Switched to `H:/Xiyao_Wang/001_models/Qwen2.5-Coder-7B-Instruct`; debug training loaded 7.6B frozen parameters and trained 92.5M ViT/projector parameters.
- Two-step debug completed: global_step 2, best_val_loss 1.009125, train_records 16, val_records 4.
- Debug artifacts are under `outputs/instruction_runs/debug_v1_qwen25_coder_7b/` with `best.pt`, `step_2.pt`, `final.pt`, `config.yaml`, `resolved_config.yaml`, `metrics_final.json`, and `runtime_summary.json`.
- Both GPUs were idle after the debug run.
- Added `data/instructions/` to `.gitignore` because generated instruction data derives from medical data.

## 2026-06-28 Subagent Audits

- Report-source explorer checked CheXpert/NIH CSVs, split JSONL, processed UMS JSONL, instruction JSONL, schema/docs, and preprocessing scripts.
- No original CXR radiology report text was found in current repo data; UMS `findings` is structured label metadata, not narrative report text.
- Current instruction rows have schema fields such as `report` and `evidence_phrase`, but sampled V1 records leave them empty/null and carry `no_report_text`.
- Therefore V2/V3 cannot honestly run as report-grounded instruction generation until a licensed/de-identified report lookup manifest is added.
- Evaluation explorer confirmed old no-LM schema 1k and old LP comparator metrics already exist under `outputs/data_scaling/`, but these are older UMS-schema comparators, not new instruction-LP outputs.
- Required instruction-specific question-only/image-shuffle evaluation and instruction downstream LP were missing before this session; visual token weighting exists only as generic state-token weighting and is not meaningful for report-evidence V4 without V2/V3 report-backed rows.

## 2026-06-28 Instruction Visual-Dependence Smoke

- Added instruction teacher-forced loss evaluator for normal, question-only zero image, and image-shuffle modes.
- First image-shuffle implementation was misleading because V1 records are grouped by image; batch-local rolling can keep the same image. The evaluator now pairs each instruction row with an image from a different `sample_id`.
- On `step_250.pt`, 64 V1 validation rows: normal loss 1.628535, question-only loss 1.910276 (`+0.281741`), image-shuffle loss 1.629220 (`+0.000685`).
- Interpretation: the current V1 label-to-QA checkpoint is sensitive to blanking the image but has near-zero sensitivity to mismatched images in this small smoke. This supports keeping V1 claims narrow and treating stronger visual-dependence as unresolved until report-backed/counterfactual data exists.

## 2026-06-28 MIMIC Report Source

- `H:\Xiyao_Wang\000_Public Dataset\mimic-cxr\mimic-cxr\mimic-cxr-reports` contains report `.txt` files, with local count previously checked at 227835.
- `H:\Xiyao_Wang\000_Public Dataset\mimic-cxr\mimic-cxr\mimic-cxr-images` contains matching image study directories; image/report paths align by `pXX/p########/s########`.
- `mimic-cxr_less` CSVs have nonempty `text` and `text_augment`, but for V2/V3 we use the full image/report directory layout to keep image-study-report alignment explicit.
- Generated MIMIC manifest rows contain absolute image paths plus report text in local ignored JSONL; no report body was printed to the terminal.
- V2 MIMIC train generation produced 4000 raw rows from four shards with 0 duplicate drops; filtering kept 3993 rows across 1000 studies.
- V2 MIMIC val generation produced 800 raw rows from four shards with 0 duplicate drops; filtering kept 796 rows across 200 studies.
- V2 train answer types cover finding verification, evidence phrase, laterality/location, severity, uncertainty, answerability, report consistency, counterfactual choice, temporal comparison, and device position.
- V2 debug training completed on local Qwen2.5-Coder-7B with `global_step=2`, `best_val_loss=2.3839718103408813`, 16 train records, and 4 val records.
- V2 formal training completed on the MIMIC report-backed instruction set with `global_step=1000`, `best_val_loss=1.5185097228342563`, 3993 train records, and 796 val records.
- V2 visual-dependence eval on 796 val rows produced normal loss 1.5181669605586996, question-only delta +0.0026753411146265282, and image-shuffle delta +0.002212008115035191.
- V2 downstream LP completed with macro-AUC 0.6076272728674769, macro-F1 0.827882392711578, and micro-F1 0.8021211275467486.
- Interpretation boundary: V2 is a completed report-backed baseline, but the small perturbation deltas do not validate strong image-specific grounding.
- Dedicated V3 `v3_mimic_report_grounded_counterfactual` generation uses a stricter counterfactual prompt; the 2-sample smoke kept 8/8 rows and all were `counterfactual_choice`.
- V3 train generation completed after preserving and retrying one network timeout; filtered train keeps 3978/4000 rows across 1000 samples, including 3783 `counterfactual_choice` rows.
- V3 val generation completed after preserving and retrying one SSL EOF and one timeout; filtered val keeps 797/800 rows across 200 samples, including 747 `counterfactual_choice` rows.
- V3 formal training completed with `global_step=1000`, `best_val_loss=1.0704705653511417`, 3978 train records, and 797 val records.
- V3 visual-dependence eval on 797 val rows produced normal loss 1.0706396021434998, question-only delta +0.13083389097616682, and image-shuffle delta +0.0013180705187012531.
- V3 downstream LP completed with macro-AUC 0.5572995973253796, macro-F1 0.8365447527869113, and micro-F1 0.8057493720346078.
- Interpretation boundary: V3 strengthens question-only degradation relative to V2, but image-shuffle delta remains near zero; do not claim strong image-specific grounding yet.
- Implemented instruction-level token weighting in `training/trainer.py` for report-evidence/visual-dependency V4; weights are derived from batch `answer_types`, `visual_dependencies`, and `quality_flags`, and normalized over answer tokens.
- V4 debug training completed before formal launch, confirming the weighted instruction path runs on local Qwen2.5-Coder-7B.
- V4 formal training completed with `global_step=1000`, `best_val_loss=1.0197519861665882`, 3978 train records, and 797 val records.
- V4 visual-dependence eval on 797 val rows produced normal loss 1.0831622853711564, question-only delta +0.13674618338598155, and image-shuffle delta +0.000021105833174051014.
- V4 downstream LP completed with macro-AUC 0.5646456277698648, macro-F1 0.832328792498657, and micro-F1 0.8040747976555959.
- Interpretation boundary: V4 satisfies the visual token weighting matrix item and lowers teacher-forced validation loss compared with V3, but it still does not validate strong image-specific grounding because image-shuffle loss is effectively unchanged.
- Added instruction-specific counterfactual diagnostics for V3/V4. The diagnostic scores normal answer NLL by answer_type and scores A/B/C/D option-formatted counterfactual_choice records by correct-option vs best-negative-option NLL.
- V3 counterfactual diagnostics: 747 `counterfactual_choice` validation records, 231 explicit option-formatted records, 513 no-option records, 3 correct-letter failures, option-subset pairwise accuracy 0.4199134199134199, mean best-negative minus correct NLL -0.003256428250992896.
- V4 counterfactual diagnostics: 747 `counterfactual_choice` validation records, 231 explicit option-formatted records, 513 no-option records, 3 correct-letter failures, option-subset pairwise accuracy 0.45021645021645024, mean best-negative minus correct NLL 0.004010261470901575.
- Interpretation boundary: V4 weakly improves option-subset pairwise accuracy over V3, but the margin is near zero and most GLM-labeled counterfactual_choice rows are not actually multiple-choice; this is evidence of a data-quality gap, not a strong counterfactual-grounding win.

## 2026-06-30 Next-Stage Concurrency Findings

- GPU1 can sustain two concurrent 10k-scale Qwen3-VL next-stage trainings (`shuf_10k_8k` plus `storymix_10k_8k`) around the current 21-22 GiB window, but repeated third-training-lane probes with `prog_mix_10k_8k` crossed guarded thresholds before training progress. Treat a third GPU1 10k training as opportunistic only after another lane frees memory.
- Root-scoped memory guards preserve work better than broad cleanup: the 15:54 `prog_mix_10k_8k` probe was stopped at 22609 MiB while `shuf_10k_8k` and `storymix_10k_8k` kept advancing. A Windows PID-reuse edge case required creation-time filtering in `scripts/guard_gpu_memory_process_tree.ps1` so stale children created before the guarded root are ignored.
- A second guard robustness edge appeared at 18:16: guards wrote `GUARD_TRIGGER` for `prog_mix_tw_10k` and the waiting `storymix_10k_8k` watcher but did not write `STOPPED`; this indicates trigger-time process-tree construction can still fail before stop. The guard now catches timestamp-read failures per child and falls back to stopping the root if full tree construction fails. Reattach fixed guards after patching rather than trusting old already-running guard processes, because they retain the old script body.
- A third guard edge appeared at 20:17: after the fallback patch, the guard stopped the `prog_mix_tw_10k` root but child timestamp conversion failures caused children to be skipped, leaving the actual Python process alive. The safer policy is: if a root timestamp cannot be read, use `DateTime.MinValue`; if a child timestamp cannot be read, include that child in the stop tree instead of skipping. Old guards must be relaunched after this patch as well.
- A fourth refinement followed the 21:23 guard event: using `DateTime.MinValue` for a root whose `CreationDate` is unreadable can defeat PID-reuse filtering and over-include stale children. The guard now uses `Convert-ProcessCreationTime`: accept `[datetime]` values, try DMTF conversion for strings, and fall back to `Get-Process.StartTime`; only if root time is truly unavailable does it stop root-only. This is the current preferred implementation.
- A queue-ownership edge appeared at 22:29 after `train_fullvision` completed: the older GPU0 training queue still contained `prog_mix_tw_10k` and auto-resumed it from `step_6000.pt` while a manual GPU1 resume lane was already active. The duplicate GPU0 tree was stopped and logged. Future long queues should remove manually owned unfinished IDs or use a run-owner lock before auto-resuming a non-final output directory.
- A/B-swap diagnostics must be first-class package evidence. `StoryMix-10k-8k` initially packaged successfully without A/B-swap because the required `storymix_10k_val_ab_swap.jsonl` was missing and the previous marker set did not include `ab_swap_results.md`; the package, summary, and completion-audit scripts now include explicit A/B-swap evidence.
- Final next-stage completion is artifact-backed, not prose-backed: after refreshing per-run packages and final tables, `outputs/final_tables/next_stage_completion_audit.csv` contains 1049 rows and all are `completed`. This includes training packages, LP/NIH transfer metrics, visual-dependence, primary counterfactual, A/B-swap counterfactual, paraphrase, instruction audits, cost tables, and final consolidated CSV/MD tables for all 39 manifest runs.
- A/B-swap zero-row cases need an explicit not-applicable boundary. P2-style validation files have no A/B option rows; the final audit now treats existing zero-row A/B diagnostic inputs as `not_applicable_no_ab_rows` rather than false missing GPU diagnostics, while rowful A/B inputs still require canonical diagnostic JSON.
- After the user requested GPU0 be freed, all remaining next-stage work was migrated to `gpu1_only_remaining_after_free_gpu0_20260701T004917`. GPU0 was kept free from next-stage tasks; unrelated `outputs/runs/m1_dense/run_train*.py` processes repeatedly restarted on GPU0 and were stopped to honor the GPU0-free boundary during closeout.
- Final closeout verification refreshed all next-stage package/summary/audit outputs and confirmed no next-stage worker, watcher, or guard processes remained. The only post-closeout GPU0 pressure came from unrelated `m1_dense/opmem.eval` copy/reverse eval wrappers; after stopping those wrappers, a 20-second `nvidia-smi` observation showed both GPU0 and GPU1 at `0 MiB`.
- Because the unrelated `m1_dense/opmem.eval` wrappers restarted once more after the initial observation, final closeout used a synchronous 60-second narrow matcher and then a 12-second post-clean check. The final `nvidia-smi` state showed GPU0=`0 MiB`, GPU1=`0 MiB`, and no compute apps.
- The `m1_dense/opmem.eval` restarts came from an external Codex app-launched offset loop, not from next-stage workers. A hidden 20-minute narrow guard was started at 2026-07-01T02:12:58+08:00 to keep GPU0 free by stopping only `outputs\runs\m1_dense` / `opmem.eval checkpoint_last.pt` wrappers; it does not run GPU work itself.

## 2026-07-01/02 Case Study + Module Real Execution

- The active target `vivid_med_case_study_modules_next_experiment_plan.md` is now closed at the real-run artifact level, not only source/manifest/smoke level. The final entry points are `outputs/final_tables/case_study_full_execution_status.md`, `outputs/final_tables/case_study_extra_execution_status.md`, and `outputs/final_tables/module_ablation_results.md`.
- All 12 required stability/downstream rows completed true 5000-step long training plus CheXpert LP, NIH available transfer, visual-dependence, counterfactual/A-B or explicit not-applicable boundary, and paraphrase diagnostics. Every NIH available transfer evaluated 25,596 records and wrote `nih_1000`, `nih_5000`, and `all_available` subset metrics.
- Three-seed family means now change the interpretation: `SHUF-TW-clinical` remains a candidate but not a final-best winner. Mean CheXpert AUC is `SHUF-3k=0.686757`, `SHUF-TW-clinical=0.690742`, `SAMEQ-SHUF-3k=0.713825`, `SHUF-K4=0.709548`; mean hard-shuffle delta is `0.114476`, `0.046622`, `0.438040`, and `0.329075`, respectively.
- SAMEQ-SHUF-3k has strong/stable hard-shuffle signal but its CF/A-B option-pairwise accuracy is not applicable in the current same-question/different-answer format because the diagnostic JSONs have zero option records. Do not read those blanks as missing artifacts or zeros.
- NIH/domain evidence is now real available-transfer plus real embeddings. `domain_shift_mmd.md` reports RBF-MMD `0.139625` using CheXpert-val `n=1000` and NIH-available sampled `n=4000`; `dataset_embedding_projection.md` writes 5000 projected rows from actual embedding files.
- `cur_v2_progressive_replay` completed formal 12000-step training with `best_val_loss=0.237637`. This gives curriculum a fair long-training result, but training loss alone is not a final-method claim.
- CEQ, AUCH, HNMB, DRA, CCSH, and CDCS are implemented, smoke-passed, and now formal-ablation trained for 1000 steps each. The strongest current embedding-level module readout is CCSH (`state_accuracy=0.746065`, `binary_auc=0.894268`, `binary_auprc=0.847528`), but it should be promoted into a locked full-training candidate before being claimed as the method.
- The main paper direction should shift away from "SHUF-TW-clinical is综合最优" and toward mechanism-backed image-mismatch grounding plus deployable clinical consistency/evidence modules. Current evidence supports family narrowing and diagnosis, not final-best declaration.
- During real execution setup, two artifacts required queue-level protection against false completion: MMD/UMAP had to force rerun after real CheXpert/NIH embeddings existed because old boundary reports already occupied the target filenames, and multi-seed stability had to include A/B-swap diagnostics rather than only primary counterfactual diagnostics.
- For this Windows queue pattern, nested `powershell -Command $Command` does not preserve double-quoted path arguments reliably when the path contains spaces. The NIH evaluator was healthy in a manual 2-sample check, but queued NIH runs failed with a stray `Dataset` token until path arguments were single-quoted in the postprocess queue and extra execution manifest.
- CEQ formal ablation initially failed with `ModuleNotFoundError: No module named 'evaluation'`; the root cause was script-local import path resolution. `scripts/train_case_study_module_ablation.py` now inserts the repository root on `sys.path`, and the rerun completed all six module ablations.

## 2026-07-03 Remote Upload Handoff Findings

- Upload handoff target is `dqxy11@172.20.52.10`, with remote base `~/projects/xiyaowang` and revised target `~/projects/xiyaowang/021_260129VIVID`.
- User revised the upload scope to the whole repository folder rather than the document's split `code/data/outputs` layout.
- Before uploading the whole folder, check for unsafe credential/private-key material and transfer tooling; do not expose real passwords or private keys.
- Local full-repository size audit: `outputs/` 896.635 GB, `data/` 148.712 GB, `vivid_env/` 5.938 GB, `.git/` 2.467 GB, total roughly 1.06 TB plus small source/doc trees.
- Remote `/ipfs` has ample capacity (`1.2P` size, about `1.2P` available at audit time). Remote GNU tar is available.
- Local `rsync` is unavailable; after user refined scope, build a filtered staging directory and stream it to `~/projects/xiyaowang/021_260129VIVID`.
- Filtered upload should keep runnable source/config/docs (`configs`, `data` loaders and processed metadata, `models`, `training`, `evaluation`, `scripts`, `prompts`, `docs`, `profile`, root README/requirements/AGENTS and main proposal markdown), but exclude `outputs/`, raw dataset images/volumes, `vivid_env/`, `.git/`, `History/`, `delete/`, pretrained `.pt/.pth`, caches, and local process records (`progress.md`, `findings.md`, `task_plan.md`).
- Remote upload completed at `~/projects/xiyaowang/021_260129VIVID`; final remote project size is 457 MB with the filtered source/config/doc/data-processed subset.
- Remote environment `vivid_med310` uses Python 3.10, PyTorch `2.9.0+cu128`, torchvision `0.24.0+cu128`, transformers `5.12.1`, timm `1.0.27`, pandas `2.3.3`, and scikit-learn `1.7.2`.
- GPU smoke ran inside the approved `gpu` tmux session on `gpu02` with `CUDA_VISIBLE_DEVICES=0`; A800 CUDA availability and CUDA matmul passed, and `scripts/smoke_case_study_modules.py` passed CEQ/AUCH/HNMB/DRA/CCSH/CDCS.
- Post-smoke GPU check in the `gpu` session reported `0 MiB, 0%`; no smoke process remained on the GPU.
- For a quick runnable remote package, the useful data beyond code/docs is much smaller than the full repository: processed metadata is 427 MB, `data/instructions` + `data/splits` are 229 MB, CheXpert small is 10.684 GB, and small auxiliary LIDC/OrganMNIST is 1.823 GB.
- The huge local size is explained by deferred buckets: raw NIH/AMOS/KITS total 135.565 GB, output final-weight candidates are 315.923 GB, and output intermediate/process checkpoints dominate the remaining ~893.8 GB under `outputs/`.
- Remote full-data upload was stopped after it had created a partial 34 GB `data/`; this partial tree should be replaced by the minimal runnable data package before claiming data readiness.
- To satisfy the user's "run first" priority faster than the 13.2 GB minimal-complete transfer, the remote `data/` tree was replaced with a tiny real-image package: full data source code, full processed/instruction/split metadata, and 2,929 sampled CheXpert image files covering early debug rows. Remote `data/` verifies at 844 MB.
- Remote data smoke passed with CUDA available and a real image batch tensor `(2, 3, 224, 224)`.
- Remote training smoke passed in the approved `gpu` tmux session using `vivid_med310`: `scripts/train_ums_classifier.py --config configs/remote_smoke_ums_classifier.yaml --debug` completed 20 debug steps and wrote `outputs/remote_smoke_ums_classifier/metrics_final.json`.
- Smoke-generated `.pt` files under `outputs/remote_smoke_ums_classifier/` were removed after verification so process weights do not inflate the remote package; the final post-smoke GPU check reported `0 MiB, 0%`.
- Remote model upload scope should start with `qwen3-vl-2b-thinking-new`: it is referenced 141 times across configs/scripts/docs and is documented as the active audited route; local size is 3.974 GB. `Qwen2.5-Coder-7B-Instruct` appears in 8 legacy text-scaffold configs, and BiomedCLIP's raw folder is not needed for the current remote package because the converted `pretrained/biomedclip_vit_base.pt` is already present.
- `qwen3-vl-2b-thinking-new` is now uploaded to remote `~/projects/xiyaowang/021_260129VIVID/model/qwen3-vl-2b-thinking-new`, verifying as 14 files / 4.0 GB. A project-local compatibility symlink at `H:/Xiyao_Wang/001_models/qwen3-vl-2b-thinking-new` points to the same model so existing Windows-style configs can resolve from the remote project root.
- Remote model verification passed in `vivid_med310`: direct config/processor load reports `model_type=qwen3_vl`, `Qwen3VLForConditionalGeneration`, and `Qwen3VLProcessor`; GPU component audit wrote `outputs/remote_qwen3vl_model_audit.{json,md}` with `device=cuda`, `dtype=torch.bfloat16`, and `forward_status=ok`.
- After the remote Qwen3-VL model audit, the approved `gpu` tmux session returned to `0 MiB, 0%`; remote project size including the model is now 8.8 GB.
- For `vivid_med_case_study_modules_next_experiment_plan.md`, the remote now has all checked final/best/probe-final case-study/module weights required for artifact continuation: 62 expected files, 62 remote files, 0 missing, 0 size mismatches. Intermediate `step_*.pt` and `probe_step_*.pt` files remain excluded.
- For `vivid_med_a800_full_next_experiment_plan.md`, the remote is not fully data-ready as written. It has the plan document, Qwen3-VL 2B model, 2.1 GB of existing instruction data, 59 checked next-stage/case-study configs, and the 62 final case-study weights, but it lacks full CheXpert images, raw NIH, raw MIMIC, and VinDr/PadChest external data.
- The A800 full plan is partly aspirational relative to the current repo: the 23 Section 11 script names checked for data generation, A800-specific training, external evaluation, and plotting do not exist in the current repository. Existing older scripts/configs can run prior next-stage/case-study style routes, but the A800 full plan cannot be executed end-to-end exactly as written until those scripts and missing external/full datasets are added.

## 2026-07-03 A800 Available Data Upload Findings

- User requested "把有的先传齐" for A800 readiness. Current upload scope is all locally available A800-relevant datasets: full CheXpert small, NIH Chest X-rays, MIMIC-CXR images/reports, mimic-cxr_less, and MIMIC supplementary files if present.
- VinDr-CXR and PadChest remain unavailable in the audited local public-data root, so they cannot be uploaded until supplied separately.
- User later requested "NIH先不用传了"; NIH is now an explicit `deferred_by_user` data bucket with local and remote marker files, not an uploaded/completed dataset.
- Kaggle `xhlulu/vinbigdata` was downloaded to `data/dataset/vinbigdata_xhlulu_512png` and verified as a valid 1.94 GiB zip with 18,001 entries, extracting to 15,000 train PNGs, 3,000 test PNGs, and `train_meta.csv`. The images are 512x512 grayscale PNGs; `train_meta.csv` only has `image_id,dim0,dim1`, and the zip contains no annotation/label/bbox CSV. Treat this as an image-only VinBigData/VinDr-CXR-derived asset, not a complete external-evaluation dataset for `vivid_med_a800_full_next_experiment_plan.md` until official labels or a UMS/VinDr manifest are added.
