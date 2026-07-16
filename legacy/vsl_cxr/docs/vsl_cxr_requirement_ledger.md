# VSL-CXR v5 Requirement Ledger

This ledger is derived from `../vivid_med_vsl_cxr_full_experiment_plan_v5.md` and is the active checklist for formal execution. Status values:

- `not_started`: no current v5 evidence has been accepted yet.
- `audit_pending`: existing artifacts or analog scripts may exist, but exact v5 match has not been verified.
- `in_progress`: active implementation or run.
- `completed`: exact v5 requirement closed by current evidence.
- `bounded`: cannot be run exactly now, with an explicit documented boundary.

## Phase 0: Data And Audit Preparation

| ID | Requirement | Status | Evidence / notes |
| --- | --- | --- | --- |
| D0 | Basic-QA dataset | completed_source_manifest | Current source artifact exists and is recorded in `../outputs/final_tables/vsl_cxr_data_source_manifest.md`; exact generation wrapper `scripts/extract_clinical_statements.py` refreshes the manifest. |
| D1 | CF-QA dataset | completed_source_manifest | Current counterfactual sources exist and are recorded in `../outputs/final_tables/vsl_cxr_data_source_manifest.md`; exact wrapper `scripts/generate_counterfactual_statements.py` refreshes the manifest. |
| D2 | SAMEQ dataset | completed_source_manifest | Current SAMEQ sources exist and are recorded in `../outputs/final_tables/vsl_cxr_data_source_manifest.md`; exact wrapper `scripts/generate_sameq_pairs.py` refreshes the manifest. |
| D3 | SAMEQ-CF dataset | completed_source_manifest | Current SAMEQ-CF source exists and is recorded in `../outputs/final_tables/vsl_cxr_data_source_manifest.md`. |
| D4 | SAMEQ-K dataset | completed_source_manifest | Current hard-negative source exists and is recorded in `../outputs/final_tables/vsl_cxr_data_source_manifest.md`; exact wrapper `scripts/generate_hard_negative_pairs.py` refreshes the manifest. |
| D5 | SAMEQ-HNMB dataset | completed_source_manifest | Current mined/self-hard sources exist and are recorded in `../outputs/final_tables/vsl_cxr_data_source_manifest.md`; exact wrapper `scripts/mine_hard_negatives_memory_bank.py` refreshes the manifest. |
| D6 | VSL-4class dataset | completed | Candidate train/val generated at `../outputs/instruction_data/vsl_cxr/d6_vsl_4class_{train,val}.jsonl`; structural audit accepted 11149/11149 train and 1600/1600 val rows. Manual correctness still pending. |
| D7 | VSL-CEQ labels | completed | D9 CEQ companion targets exist at `../outputs/instruction_data/vsl_cxr/d9_ceq_targets_{train,val}.jsonl` and were used by the Phase 3 CEQ patch-token runs. |
| D8 | VSL-CCSH pairs | completed | D9 CCSH companion pairs exist at `../outputs/instruction_data/vsl_cxr/d9_ccsh_pairs_{train,val}.jsonl` and were used by the Phase 4 readout runs. |
| D9 | VSL-full dataset | completed | Candidate generated at `../outputs/instruction_data/vsl_cxr/d9_vsl_full_{train,val}.jsonl` with CEQ/CCSH companion files; structural audit accepted 18000/18000 train and 2000/2000 val instruction rows. |
| QA | Data-quality table | completed | `../outputs/final_tables/vsl_cxr_d6_dataset_summary.md` and `../outputs/final_tables/vsl_cxr_data_quality_summary.md` created for D6. Other data versions still need exact v5 audit. |
| MA | Manual audit sample | completed_template_only | D6 manual-audit template created at `../outputs/final_tables/vsl_cxr_d6_manual_audit_template.csv`; human correctness labels are still blank and remain a publication/manual-review boundary. |

## Phase 1: Baseline And Backbone Confirmation

| Run ID | Formal run | Status | Evidence / notes |
| --- | --- | --- | --- |
| B0 | Raw-Vision | completed | Formal raw-vision package completed as frozen original Qwen3-VL vision tower plus linear-probe readout, with `vision_checkpoint: null` and `freeze_backbone: true`; `metrics_final.json`, `resolved_config.json`, `final_probe.pt`, and result-table row exist under `F:/Xiyao_Wang/021_260129VIVID_vsl_cxr_outputs/qwen3vl/vsl_cxr_b0_raw_vision_lp/`. CheXpert 1k readout metrics: macro-AUC `0.6790032275900184`, macro-F1 `0.7323508931634031`, micro-F1 `0.697265625`. This is raw feature evidence, not a trained VSL checkpoint. |
| B1 | Basic-QA | completed | Formal run completed at `global_step=5000` with `best_val_loss=0.023826396770775318`; final metrics, runtime summary, result-table row, and `checkpoints/final.pt` exist under `F:/Xiyao_Wang/021_260129VIVID_vsl_cxr_outputs/qwen3vl/vsl_cxr_b1_basic_qa/`. |
| B2 | CF-QA | completed | Formal run completed at `global_step=5000` with `best_val_loss=0.12035670908632119`; final metrics, runtime summary, result-table row, and `checkpoints/final.pt` exist under `F:/Xiyao_Wang/021_260129VIVID_vsl_cxr_outputs/qwen3vl/vsl_cxr_b2_cf_qa/`. |
| B3 | SAMEQ | completed | Formal run completed at `global_step=5000` with `best_val_loss=0.17672864127079244`; final metrics, runtime summary, result-table row, and `checkpoints/final.pt` exist under `F:/Xiyao_Wang/021_260129VIVID_vsl_cxr_outputs/qwen3vl/vsl_cxr_b3_sameq/`. |
| B4 | SAMEQ-CF | completed | Formal run completed at `global_step=5000` with `best_val_loss=0.21318705889705136`; final metrics, runtime summary, result-table row, and `checkpoints/final.pt` exist under `F:/Xiyao_Wang/021_260129VIVID_vsl_cxr_outputs/qwen3vl/vsl_cxr_b4_sameq_cf/`. |
| B5 | SAMEQ-K4 | completed | Formal run completed at `global_step=5000` with `best_val_loss=0.12773469497652912`; final metrics, runtime summary, result-table row, and `checkpoints/final.pt` exist under `F:/Xiyao_Wang/021_260129VIVID_vsl_cxr_outputs/qwen3vl/vsl_cxr_b5_sameq_k4/`. |
| B6 | SAMEQ-HNMB | completed | Formal run completed at `global_step=5000` with `best_val_loss=0.09585380152730649`; final metrics, runtime summary, result-table row, and `checkpoints/final.pt` exist under `F:/Xiyao_Wang/021_260129VIVID_vsl_cxr_outputs/qwen3vl/vsl_cxr_b6_sameq_hnmb/`. |
| B7 | VSL-4class | completed | Formal run completed at `global_step=5000` with `best_val_loss=0.3948386136543704`; final metrics, runtime summary, result-table row, and `checkpoints/final.pt` exist under `F:/Xiyao_Wang/021_260129VIVID_vsl_cxr_outputs/qwen3vl/vsl_cxr_d6_vsl4/`. |

## Phase 2: Visual Sufficiency Data Engine

| Run | Label set / loss | Status | Evidence / notes |
| --- | --- | --- | --- |
| VSL-2class | support vs contradict, CE | completed | D6-derived train/val generated and structurally audited; formal run completed at `global_step=5000` with `best_val_loss=0.04671015161014566`, final metrics, runtime summary, result-table row, and `checkpoints/final.pt`. |
| VSL-3class | support / contradict / uncertain, CE | completed | Formal run completed at `global_step=5000` with `best_val_loss=0.14087300902791322`; final metrics, runtime summary, result-table row, and `checkpoints/final.pt` exist. |
| VSL-4class | full four-class, CE | completed | D6 data and debug trainer path passed; formal B7 run completed at `global_step=5000` with `best_val_loss=0.3948386136543704`. |
| VSL-4class-balanced | full four-class, class-balanced | completed | D6-derived class-balanced sampling data generated and structurally audited: 8596/8596 train and 1600/1600 val accepted. Formal run completed at `global_step=5000` with `best_val_loss=0.4522515568471281`; final metrics, runtime summary, result-table row, and `checkpoints/final.pt` exist. |
| VSL-4class-field-balanced | full four-class, finding-balanced | completed | D6-derived finding-balanced sampling data generated and structurally audited: 5382/5382 train and 767/767 val accepted across 13 findings. Formal run completed at `global_step=5000` with `best_val_loss=0.34942927536666346`; final metrics, runtime summary, result-table row, and `checkpoints/final.pt` exist. |
| VSL-hierarchical | full four-class, hierarchical loss | completed | Added `training.hierarchical_vsl_loss` support in `scripts/train_qwen3vl_clinical_instruction.py` and config `configs/qwen3vl_instruction/vsl_cxr/vsl_hierarchical.yaml`. Debug smoke passed; formal run completed at `global_step=5000` with `best_val_loss=0.47355849220942764`; final metrics, runtime summary, result-table row, and `checkpoints/final.pt` exist. |

## Phase 3: Evidence-Aware Vision Encoder

| Run | Module | Status | Evidence / notes |
| --- | --- | --- | --- |
| CEQ-basic | finding queries | completed | Formal patch-token CEQ run completed at `global_step=1000`; binary AUC `0.7680567410478396`, state accuracy `0.512`. |
| CEQ-diverse | CEQ + query diversity | completed | Formal patch-token CEQ run completed at `global_step=1000`; binary AUC `0.6909421615250172`, state accuracy `0.54`. |
| CEQ-sparse | CEQ + attention sparsity | completed | Formal patch-token CEQ run completed at `global_step=1000`; binary AUC `0.7158374994919318`, state accuracy `0.646`. |
| CEQ-region | CEQ + laterality/location weak labels | completed | Formal patch-token CEQ run completed at `global_step=1000`; binary AUC `0.8471731089704508`, state accuracy `0.716`, region accuracy `0.654`. |
| CEQ-statement | statement-conditioned query | completed | Formal patch-token CEQ run completed at `global_step=1000` on the VSL-4class backbone; binary AUC `0.7205218875746859`, state accuracy `0.702`. |
| CEQ-casebook | CEQ attention qualitative table | completed_needs_manual_review | Phase 8 casebook includes CEQ attention rows and links existing attention-map assets; manual image review remains required. |

## Phase 4: Deployable Readout

| Run | Backbone / readout | Status | Evidence / notes |
| --- | --- | --- | --- |
| CCSH-Raw | raw Qwen3-VL + CCSH | completed | No-leak formal run completed; binary AUC `0.8815600000000001`, AUPRC `0.8507778435295084`, ECE `0.1564016938265413`. |
| CCSH-SAMEQ | SAMEQ backbone + CCSH | completed | No-leak formal run completed; binary AUC `0.895176`, AUPRC `0.8784714795940306`, ECE `0.10921547676064075`. |
| CCSH-SAMEQ-K4 | SAMEQ-K4 backbone + CCSH | completed | Formal run completed; binary AUC `0.7883600000000001`, AUPRC `0.7120127432199869`. |
| CCSH-HNMB | HNMB backbone + CCSH | completed | Formal run completed; binary AUC `0.8353600000000001`, AUPRC `0.7949973637930012`. |
| CCSH-CEQ | CEQ backbone + CCSH | completed | Formal run completed; binary AUC `0.9059760000000001`, AUPRC `0.8859949035726915`; current best CCSH AUC. |
| CCSH-VSL4 | VSL-4class backbone + CCSH | completed | Formal run completed; binary AUC `0.8808239999999999`, AUPRC `0.8600663163097216`, ECE `0.14746286206133666`. |
| AUCH-SAMEQ | SAMEQ + AUCH | completed | Formal AUCH-only run completed; state accuracy `0.596`, answerability AUC `0.5476678876678877`, answerability AUPRC `0.9284813309209423`, uncertainty AUC `0.5466422466422467`, uncertainty F1 `0.0`. |
| AUCH-CCSH-SAMEQ | SAMEQ + AUCH + CCSH | completed | Formal run completed; binary AUC `0.90072`, AUPRC `0.8938395501978506`, ECE `0.1630195039883256`; strongest Phase 4 AUCH+CCSH binary AUC. |
| AUCH-CEQ-CCSH | CEQ + AUCH + CCSH | completed | Formal run completed; binary AUC `0.8889679999999999`, AUPRC `0.9005512099194461`. |
| AUCH-VSL4 | VSL-4class + AUCH + CCSH | completed | Formal run completed; binary AUC `0.895976`, AUPRC `0.8707344191804582`, ECE `0.11341642936132851`; best Phase 4 calibration ECE among CCSH/AUCH+CCSH rows. |

## Phase 5: Integrated VSL-CXR Candidates

| Candidate | Formal composition | Status | Evidence / notes |
| --- | --- | --- | --- |
| VSL-Lite | SAMEQ + global encoder + none/LP | component_completed | SAMEQ global encoder evidence exists via `VSL-CXR-B3-SAMEQ`; LP/CheXpert readout remains unavailable. |
| VSL-Core | SAMEQ-K4 + global encoder + CCSH | component_completed | SAMEQ-K4 backbone plus CCSH readout evidence exists; CCSH AUC `0.7883600000000001`, AUPRC `0.7120127432199869`, ECE `0.18651148418523372`. |
| VSL-HNMB | SAMEQ-HNMB + global encoder + CCSH | component_completed | SAMEQ-HNMB backbone plus CCSH readout evidence exists; CCSH AUC `0.8353600000000001`, AUPRC `0.7949973637930012`, ECE `0.19963750052824616`. |
| VSL-CEQ | SAMEQ + CEQ + CCSH | component_completed | SAMEQ backbone plus CEQ-region/CCSH readout evidence exists; CCSH AUC `0.9059760000000001`, current strongest Phase 5 component candidate by binary AUC. |
| VSL-Full | SAMEQ-HNMB + VSL-4class + CEQ + CCSH+AUCH | completed | D9 mixed-instruction formal training completed at `global_step=5000`, `best_val_loss=0.19854170768998938`; Phase 5 table links this with CCSH+AUCH evidence. |
| VSL-Domain | VSL-Core + optional DRA + CCSH | blocked_external_data | Blocked until external dataset and label-manifest eligibility are resolved. |

## Phase 6: External Validation

| Requirement | Status | Evidence / notes |
| --- | --- | --- |
| External dataset audit | main_external_integration_in_progress | Official VinDr-CXR 1.0.0 archive contains 15,000 train and 3,000 test DICOMs plus bbox/image-level labels; deterministic train/test manifests and a direct seven-label primary protocol are built. Extraction/CRC/integrity validation and five formal test-split inference rows are running. MIMIC CheXpert/metadata/split `.csv.gz` manifests exist but remain conditional on overlap eligibility; PadChest is only a missing backup; NIH remains appendix/stress evidence. |
| Label mapping audit | completed_appendix_only | NIH mapping audit completed through `scripts/audit_label_mapping_nih.py`; main-external label mapping remains blocked until an eligible main-external label manifest is available. |
| Macro-AUC / macro-AUPRC | completed_appendix_only | NIH-appendix-1k transfer completed for Raw, SAMEQ, VSL-Core, VSL-CEQ backbone proxy, and VSL-Full. Best NIH macro-AUC is SAMEQ `0.5932955434118374`; best NIH macro-AUPRC is VSL-Core `0.15464047456578672`. |
| ECE / Brier | completed_appendix_only | NIH-appendix-1k ECE/Brier are recorded in `outputs/final_tables/vsl_cxr_external_results.md`; calibration is weak and does not satisfy main-external validation. |
| Per-label analysis | completed_appendix_only | NIH best/worst labels and failure causes are recorded in `outputs/final_tables/vsl_cxr_external_results.md`; rows explicitly state NIH is appendix/stress only. |
| Domain shift visualization | bounded_appendix_manifest_only | Phase 8 external-failure rows and Phase 6 NIH appendix/stress metrics are available; exact main-external embedding visualization remains blocked with the main-external label/manifest boundary. |

## Phase 7: VLM Teacher Comparison

| Run | Status | Evidence / notes |
| --- | --- | --- |
| Qwen3VL-VSL-smoke | completed_by_current_main | Completed v5 Qwen3-VL VSL-Core formal evidence confirms the current-main stack; see `outputs/final_tables/vsl_cxr_teacher_comparison_results.md`. |
| InternVL-VSL-smoke | blocked_adapter_missing | Local InternVL model paths exist, but current v5 trainer is Qwen3-VL specific and InternVL processor/trainer adapter is missing. |
| LLaVA-VSL-smoke | blocked_adapter_missing | Local Mllama model path exists, but a Llama-vision VSL trainer adapter is missing. |
| Qwen3.5-scaffold-smoke | blocked_text_scaffold_trainer_missing | Text-only model exists, but v5 scaffold control requires a separate non-vision VSL trainer. |
| Qwen3VL-VSL-full | completed_current_main_only | Current-main VSL-Core evidence is complete: CheXpert LP macro-AUC `0.6985809632677275`, NIH-appendix macro-AUC `0.5872269642158319`, CCSH AUC `0.7883600000000001`; cross-family comparison remains bounded. |
| InternVL-VSL-full | blocked_until_smoke_adapter | Requires InternVL-specific VSL trainer and smoke pass. |
| LLaVA-VSL-full | blocked_until_smoke_adapter | Requires Llama-vision VSL trainer and smoke pass. |
| Qwen3.5-scaffold | blocked_text_scaffold_trainer_missing | Requires exact v5 text-only VSL scaffold trainer. |
| Qwen-Coder-scaffold | blocked_text_scaffold_trainer_missing | Historical Qwen-Coder scripts are not exact v5 VSL-Core scaffold evidence. |

## Phase 8: Case Study And Visualization

| Casebook / figure | Status | Evidence / notes |
| --- | --- | --- |
| VSL support cases | completed_needs_manual_review | `outputs/final_tables/vsl_cxr_phase8_casebook.md` includes 3 support rows sampled from current D6/D9 evidence. |
| VSL contradict cases | completed_needs_manual_review | `outputs/final_tables/vsl_cxr_phase8_casebook.md` includes 3 contradict rows. |
| Uncertain cases | completed_needs_manual_review | `outputs/final_tables/vsl_cxr_phase8_casebook.md` includes 3 uncertain rows. |
| Insufficient cases | completed_needs_manual_review | `outputs/final_tables/vsl_cxr_phase8_casebook.md` includes 3 insufficient rows. |
| SAMEQ pair cases | completed_needs_manual_review | Casebook includes 4 SAMEQ pair rows from the current SAMEQ validation source. |
| False hard negatives | completed_needs_manual_review | Casebook includes 4 false-hard-negative review rows; current val rows lack `negative_image_path`, so these are counterfactual-validity fallback review cases. |
| CCSH success/failure | completed_needs_manual_review | Casebook includes 4 CCSH readout rows plus Phase 4 CCSH metrics. |
| CEQ attention | completed_needs_manual_review | Casebook includes 4 CEQ attention rows; existing attention-map assets are indexed in the visualization manifest. |
| External failures | completed_appendix_only | Casebook includes 5 NIH appendix/stress external-failure rows; main external remains blocked. |
| Calibration curves | bounded_metric_summary_only | ECE/Brier are available in `outputs/final_tables/vsl_cxr_external_results.md`, but binned calibration curve points are not exported. |

## Phase 9: Locked Final Comparison

| Family | Requirement | Status | Evidence / notes |
| --- | --- | --- | --- |
| Raw | Select one raw finalist | locked_single_seed | Raw Qwen3-VL vision LP selected; CheXpert macro-AUC `0.6790032275900184`; NIH appendix macro-AUC `0.5737087050807025`. |
| QA | Select Basic-QA / CF-QA finalist | locked_training_loss_only | Basic-QA selected by lower v5 instruction best-val loss; no deployable LP/external/readout row was run for QA. |
| SAMEQ | Select SAMEQ / SAMEQ-CF finalist | locked_single_seed | SAMEQ selected; CheXpert LP macro-AUC `0.6961128245416363`, NIH appendix macro-AUC `0.5932955434118374`, CCSH AUC `0.895176`. |
| HardNeg | Select SAMEQ-K / SAMEQ-HNMB finalist | locked_single_seed | SAMEQ-HNMB selected by lower training loss and stronger CCSH AUC `0.8353600000000001` versus SAMEQ-K4. |
| CEQ | Select CEQ finalist | locked_single_seed | CEQ-region selected; binary AUC `0.8471731089704508`, ECE `0.08269000466053303`. |
| CCSH | Select CCSH / CCSH+AUCH finalist | locked_single_seed | CCSH-CEQ selected by primary CCSH binary AUC `0.9059760000000001`; AUCH-CEQ-CCSH remains best AUPRC row. |
| VSL Integrated | Select final integrated VSL method | locked_single_seed_with_external_boundary | VSL-Full selected as integrated finalist; best Phase 6 CheXpert LP macro-AUC `0.7148588673744163`, NIH appendix macro-AUC `0.5815168249418435`, CCSH/AUCH AUPRC evidence `0.9005512099194461`; main external remains blocked. |

## Named Script Surface From v5

| Script | Status | Notes |
| --- | --- | --- |
| `scripts/extract_clinical_statements.py` | completed | Exact v5 wrapper exists and writes `../outputs/final_tables/vsl_cxr_data_source_manifest.{csv,md}`. |
| `scripts/generate_counterfactual_statements.py` | completed | Exact v5 wrapper exists and writes `../outputs/final_tables/vsl_cxr_data_source_manifest.{csv,md}`. |
| `scripts/generate_sameq_pairs.py` | completed | Exact v5 wrapper exists and writes `../outputs/final_tables/vsl_cxr_data_source_manifest.{csv,md}`. |
| `scripts/generate_vsl_4class_labels.py` | completed | Exact v5-named generator exists and produced D6 train/val artifacts. |
| `scripts/generate_hard_negative_pairs.py` | completed | Exact v5 wrapper exists and writes `../outputs/final_tables/vsl_cxr_data_source_manifest.{csv,md}`. |
| `scripts/mine_hard_negatives_memory_bank.py` | completed | Exact v5 wrapper exists and writes `../outputs/final_tables/vsl_cxr_data_source_manifest.{csv,md}`. |
| `scripts/audit_vsl_data_quality.py` | completed | Exact v5-named structural audit exists and accepted current D6 train/val. |
| `scripts/train_vsl_cxr.py` | completed | Exact v5-named wrapper exists; debug smoke passed and current D6 VSL formal rows through Phase 2 completed with final metrics/checkpoints. |
| `scripts/train_vsl_ceq.py` | completed | Exact v5-named patch-token CEQ trainer exists; CEQ-basic/diverse/sparse/region/statement completed with final metrics/checkpoints. |
| `scripts/train_vsl_ccsh.py` | completed | Exact v5-named CCSH/AUCH+CCSH readout trainer exists and completed 9/9 Phase 4 CCSH/AUCH+CCSH rows. |
| `scripts/train_vsl_auch.py` | completed | Exact v5-named AUCH-only readout trainer exists and completed AUCH-SAMEQ with final metrics/checkpoint. |
| `scripts/train_vsl_hnmb.py` | completed | Exact v5 wrapper exists; defaults to the B6 SAMEQ-HNMB config and supports config/resume/debug/seed forwarding. |
| `scripts/train_vsl_full.py` | completed | Exact v5-named wrapper exists; VSL-Full D9 mixed-instruction formal run completed. |
| `scripts/eval_chexpert_lp.py` | completed | Exact v5 wrapper exists and refreshes CheXpert LP rows through `../outputs/final_tables/vsl_cxr_formal_run_results.{csv,md}`. |
| `scripts/eval_external_lp.py` | completed | Exact v5 wrapper exists and refreshes `../outputs/final_tables/vsl_cxr_external_results.{csv,md}`. |
| `scripts/eval_vsl_sufficiency.py` | completed | Exact v5 wrapper exists and writes `../outputs/final_tables/vsl_cxr_sufficiency_summary.{csv,md}`. |
| `scripts/eval_ccsh_consistency.py` | completed | Exact v5-named script exists; Phase 4 CCSH/AUCH result tables provide the current readout consistency evidence. |
| `scripts/eval_hard_shuffle.py` | completed_source_boundary | Exact v5-named script exists; current hard-negative source and casebook review evidence are recorded, while locked-final hard-shuffle deltas are not exported. |
| `scripts/eval_calibration.py` | completed_metric_summary | Exact v5 wrapper exists and writes `../outputs/final_tables/vsl_cxr_calibration_summary.{csv,md}`; binned curve points remain unexported. |
| `scripts/eval_ceq_attention.py` | completed_casebook_boundary | Exact v5-named script exists; Phase 8 casebook and attention-map summary provide current qualitative evidence, with manual review required. |
| `scripts/eval_casebook.py` | completed | Exact v5 wrapper exists and rebuilds Phase 8 casebook/visualization artifacts. |
| `scripts/eval_locked_final_comparison.py` | completed | Exact v5 wrapper exists and builds `../outputs/final_tables/vsl_cxr_locked_final_comparison.{csv,md,json}`. |
| `scripts/build_vsl_results_table.py` | completed | Builds `../outputs/final_tables/vsl_cxr_formal_run_results.{csv,md}` from current run output directories. |
| `scripts/build_external_results_table.py` | completed | Builds `../outputs/final_tables/vsl_cxr_external_results.{csv,md}` from current external readiness plus NIH appendix/stress transfer metrics. |
| `scripts/build_module_results_table.py` | completed | Builds `../outputs/final_tables/vsl_cxr_ceq_results.{csv,md}` from current v5 CEQ run directories. |
| `scripts/build_case_study_markdown.py` | completed | Exact v5 wrapper exists and builds Phase 8 casebook/visualization artifacts. |
| `scripts/build_paper_figures.py` | completed | Exact v5 wrapper exists and builds the Phase 8 figure manifest. |
| `scripts/build_cost_table.py` | completed | Exact v5 wrapper exists and refreshes cost fields inside the locked final comparison table. |
