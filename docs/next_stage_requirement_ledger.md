# VIVID-Med Next-Stage Requirement Ledger

Source plan: `vivid_med_next_stage_comprehensive_experiment_plan.md`  
Created: 2026-06-29

## Artifact Roots

| Purpose | Path |
| --- | --- |
| Next-stage instruction data | `outputs/instruction_data/next_stage/` |
| Leakage audit 2.0 tables | `outputs/final_tables/next_stage_audits/` |
| Mixture distribution tables | `outputs/final_tables/next_stage_*_distribution.csv` |
| Config matrix | `configs/qwen3vl_instruction/next_stage/` |
| Config snippets | `configs/qwen3vl_instruction/next_stage_snippets/` |
| Schedule/config manifests | `outputs/next_stage_manifests/` |
| Debug smoke runs | `outputs/qwen3vl_instruction/next_stage/*_debug/` |

## Part A Core Experiments

| Requirement | Current artifact/status | Evidence |
| --- | --- | --- |
| A1 P2 value-only / no-punct loss masks | Implemented in `Qwen3VLInstructionCollator.loss_masking`; configs generated. | `configs/qwen3vl_instruction/next_stage/p2_value_only.yaml`, `p2_no_punct.yaml`; `p2_no_punct_debug/metrics_final.json` |
| A1 P2 compact / field-query data | Generated train/val JSONL. | `outputs/instruction_data/next_stage/p2_state_only_compact_*.jsonl`, `p2_field_query_*.jsonl` |
| A2 mixture variants | Generated train/val JSONL for Balanced, CF-heavy, SHUF-heavy, Clinical-rich, StoryMix QA5/8/10/12. | `outputs/instruction_data/next_stage/*mix*_*.jsonl`; distribution/audit tables |
| A3 workflow/curriculum configs | Configs, materialized curriculum JSONL, schedule manifests, trainer step-window sampling, formal training, and postprocess packages are complete. | `configs/qwen3vl_instruction/next_stage/{mix_story_qa8,cur_*,prog_*}.yaml`; `outputs/instruction_data/next_stage/{cur_p3_cf_shuf,prog_mix,prog_mix_sameq,prog_mix_10k}_train.jsonl`; `outputs/next_stage_manifests/*materialized_schedule*`; `outputs/final_tables/next_stage_completion_audit.csv` |
| A4 SHUF++ data/configs | SAMEQ-SHUF, SHUF-K2, SHUF-K4, Mined-SHUF, SelfHard-SHUF, and Progressive-HardNeg data generated; in-batch support implemented and smoke-verified. SelfHard-SHUF used isolated score shards and a merge manifest. | `sameq_shuf_*.jsonl`, `shuf_k2_*.jsonl`, `shuf_k4_*.jsonl`, `mined_shuf_*.jsonl`, `selfhard_shuf_train.jsonl`, `progressive_hardneg_train.jsonl`, `progressive-hardneg_schedule.*`, `inbatch_shuf.yaml`, `mined_shuf.yaml`, `selfhard_shuf.yaml`, `progressive_hardneg.yaml`; `outputs/next_stage_manifests/hard_negative_mining/*`; `outputs/logs/qwen3vl_next_stage/shuf_3k_5k_*` |
| A5 token weighting | TW-role/TW-visual/TW-clinical snippets and SHUF-TW configs generated. | `configs/qwen3vl_instruction/next_stage_snippets/`; `shuf_tw_*.yaml` |

## Part B Extensions

| Requirement | Current artifact/status | Evidence/boundary |
| --- | --- | --- |
| B1 SHUF-10k / PROG-Mix-TW-10k | UMS-derived 10k scale facts/data, configs, training, LP/NIH transfer, diagnostics, and packages complete. | `ums_chexpert_10k_facts.jsonl`, `shuf_10k_*.jsonl`, `storymix_10k_*.jsonl`, `shuf_10k_8k.yaml`, `prog_mix_tw_10k.yaml`, `outputs/final_tables/next_stage_decision_summary.csv` |
| B2 training policy ablation | TRAIN-CONN, TRAIN-LAST4, and TRAIN-FULLVISION configs, training runs, diagnostics, and packages complete; LoRA variants remain explicit boundary unless PEFT is added. | `train_conn.yaml`, `train_last4.yaml`, `train_fullvision.yaml`, `outputs/final_tables/next_stage_training_results.csv` |
| B3 model scale/type ablation | Local 4B/8B model paths audited; larger model formal claims require separate VRAM/component smoke. | `docs/next_stage_external_model_availability.md` |
| B4 external transfer | NIH 1k transfer complete for all 39 manifest runs; MIMIC/PadChest/VinDr boundaries documented. | `outputs/final_tables/next_stage_lp_transfer_results.csv`; `docs/next_stage_external_model_availability.md` |
| B5 calibration / AUPRC | Shared LP/transfer metrics include macro/per-label AUPRC, ECE, and Brier; consolidated calibration table generated. | `evaluation/metrics.py`; `outputs/final_tables/next_stage_calibration_auprc.csv` |
| B6 prompt robustness / option bias | A/B swap diagnostic JSONL, canonical A/B diagnostic JSON, and paraphrase robustness complete; zero-row A/B cases are recorded as not applicable. | `*_ab_swap.jsonl`, `outputs/final_tables/next_stage_ab_swap_counterfactual.csv`, `outputs/final_tables/next_stage_paraphrase.csv` |
| B7 leakage audit 2.0 | Implemented and run for all current next-stage JSONL files. | `outputs/final_tables/next_stage_audits/` |
| B8 qualitative visualization | Hard-negative side-by-side casebook generated; Grad-CAM/attention attribution remains a boundary unless implemented later. | `outputs/final_tables/next_stage_qualitative_cases.md` |

## G1-G3 Execution Checklist

| Item | Status | Evidence |
| --- | --- | --- |
| G1 seven data-generation scripts | Complete. | `scripts/generate_storymix_instructions.py`, `generate_sameq_shuf_pairs.py`, `generate_multi_negative_shuf.py`, `audit_instruction_leakage_v2.py`, `build_progressive_mixture_schedule.py`, `build_token_weight_map.py`, `mine_hard_negatives_from_embeddings.py` |
| P2 variant generator | Complete. | `scripts/prepare_p2_loss_mask_variants.py` |
| Config matrix generator | Complete. | `scripts/prepare_next_stage_configs.py`, `outputs/next_stage_manifests/config_manifest.json` and `outputs/next_stage_manifests/lp_config_manifest.json` list all 39 runs. |
| G2 answer-only/value-only/token weighting/image margin/answer margin/multi-negative/in-batch support | Implemented in Qwen3-VL collator/trainer; debug-smoked for P2/SHUF-K4 and in-batch collator smoke. | `p2_no_punct_debug/metrics_final.json`, `shuf_k4_debug/metrics_final.json`, in-batch smoke output in progress log |
| G3 per-run package baseline | Complete for all 39 runs, including base training artifacts plus vision export, LP/transfer, visual-dependence, primary counterfactual, A/B-swap, paraphrase, instruction audit, and cost markers. | `outputs/final_tables/next_stage_completion_audit.csv`; run directories under `outputs/qwen3vl_instruction/next_stage/` |
| Final completion audit | Implemented as a read-only artifact checklist that can be run during interim states and at final closeout. | `scripts/audit_next_stage_completion.py`, `outputs/final_tables/next_stage_completion_audit.{csv,md}` |

## Next Verification Gate

Final verification gate is closed. The refreshed package, summary, and completion audit reports `1049` completed rows and no missing or pending rows in `outputs/final_tables/next_stage_completion_audit.csv`. Use the consolidated final tables under `outputs/final_tables/next_stage_*.{csv,md}` and the final status appendix in `vivid_med_next_stage_comprehensive_experiment_plan.md` as the active entry points.
