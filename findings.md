# VSL-CXR v5 Findings

## 2026-07-07 Project Organization Findings

- The active source plan is `vivid_med_vsl_cxr_full_experiment_plan_v5.md`, which reframes the project as Visual Sufficiency Learning for Chest X-rays rather than the previous SAMEQ-CVCP/CVCP-CCSH paper line.
- Existing root `task_plan.md`, `findings.md`, and `progress.md` were still focused on prior SAMEQ-CVCP v4, CVCP/CCSH, remote-upload, and A800 transfer tasks, so they were archived before starting the VSL-CXR execution state.
- Existing `docs/README.md` and `README.md` still pointed readers to SAMEQ-CVCP v4 as the active entry point; these must be updated before running new experiments so agents and humans start from the v5 plan.
- Root-level historical proposal/runbook markdowns were moved to `History/20260707_vsl_cxr_project_organization/` and remain recoverable context, but they are not active VSL-CXR completion evidence by themselves.
- Current v5 requires several named scripts that may not exist yet, including VSL data generation, VSL-specific training, sufficiency/readout/calibration evaluation, and VSL paper-table builders. The next audit must classify each as existing exact, analog available, missing, or historical-only.
- The v5 external-validation wording prefers VinDr-CXR/VinBigData with label/bbox CSV, then PadChest, then MIMIC if not used in training; NIH is appendix/stress rather than main. Prior memory indicates local VinBigData assets may be image/meta-only, so external-main readiness needs a fresh local audit before any claim.

## 2026-07-07 D6 VSL-4class Gate Findings

- Added and ran `scripts/audit_vsl_cxr_readiness.py`. Current readiness after D6 work is 63 rows: 8 exact v5 scripts, 21 missing exact scripts with analogs, 2 candidate VSL-schema artifacts, 1 remaining missing data candidate (`D9 VSL-full`), 8 historical final-table bundles needing v5 remap, 5 model families available but needing smoke, and external-data gaps for PadChest/MIMIC/NIH under the audited local paths.
- Added `scripts/generate_vsl_4class_labels.py` and generated D6 VSL-4class train/val artifacts at `outputs/instruction_data/vsl_cxr/d6_vsl_4class_train.jsonl` and `outputs/instruction_data/vsl_cxr/d6_vsl_4class_val.jsonl`.
- D6 train has 11149 rows: support 3000, contradict 3000, uncertain 2149, insufficient 3000. D6 val has 1600 rows, exactly 400 rows per label.
- Added `scripts/audit_vsl_data_quality.py`; structural audit accepted 11149/11149 train rows and 1600/1600 val rows with zero errors and zero warnings after normalizing uncertain-state rows.
- The D6 data is still an auto-labeled candidate. Manual correctness, leakage, and false-hard-negative review remain pending via `outputs/final_tables/vsl_cxr_d6_manual_audit_template.csv`.
- Added `scripts/train_vsl_cxr.py` and `configs/qwen3vl_instruction/vsl_cxr/debug_vsl_4class.yaml`; a debug Qwen3-VL trainer smoke completed on the D6 JSONL path with `global_step=1`, `best_val_loss=7.349977970123291`, 4 train records, 2 val records, and frozen language decoder / trainable vision tower plus visual connector.
- Post-smoke GPU check showed GPU1 free at `0 MiB` and `0%`. GPU0 had a non-VSL local `scripts/serve_local_hf_openai.py` service for `Qwen3.5-0.8B` using about `1785 MiB`; release or avoid GPU0 before formal dual-GPU VSL queues.

## 2026-07-07 D9 VSL-full and B7 Formal Launch Findings

- Added `scripts/generate_vsl_full_dataset.py` and generated D9 VSL-full as a composite package: trainable instruction rows plus CEQ/CCSH companion supervision.
- D9 instruction package paths: `outputs/instruction_data/vsl_cxr/d9_vsl_full_train.jsonl` with 18000 rows and `outputs/instruction_data/vsl_cxr/d9_vsl_full_val.jsonl` with 2000 rows.
- D9 companion paths: `d9_ceq_targets_train.jsonl` 13833 rows, `d9_ceq_targets_val.jsonl` 500 rows, `d9_ccsh_pairs_train.jsonl` 28166 rows, and `d9_ccsh_pairs_val.jsonl` 500 rows.
- D9 structural audit passed after normalizing missing source provenance in the HNMB component: train accepted 18000/18000 and val accepted 2000/2000 with zero errors and zero warnings.
- Added `scripts/train_vsl_full.py` and `configs/qwen3vl_instruction/vsl_cxr/debug_vsl_full.yaml`; D9 mixed-instruction debug smoke completed with `global_step=1`, `best_val_loss=9.204761981964111`, 4 train records, and 2 val records.
- Added formal B7 config `configs/qwen3vl_instruction/vsl_cxr/vsl_4class.yaml` using D6 VSL-4class data, frozen language decoder, trainable vision tower plus visual connector, 5000 steps, final-only checkpointing, GPU1, and F-drive output root.
- Launched formal B7 VSL-4class as PID `26240` at 2026-07-07 02:30:58 local time. Live evidence after launch: output root `F:/Xiyao_Wang/021_260129VIVID_vsl_cxr_outputs/qwen3vl/vsl_cxr_d6_vsl4` contains `config_snapshot.json`, `resolved_config.yaml`, `progress.json`, and `training_log.txt`.
- First dense handoff check closed cleanly: B7 wrote eval at `global_step=500` with `val_loss=0.6003475424525095`, then continued training through at least `global_step=825/5000`.
- Second dense handoff check closed cleanly: B7 wrote eval at `global_step=1000` with `val_loss=0.5235666893760499`, then continued training beyond `global_step=1200/5000`.
- Third dense handoff check closed cleanly: B7 wrote eval at `global_step=1500` with `val_loss=0.5967676737032307`, then continued training through at least `global_step=1900/5000`.
- Fourth dense handoff check closed cleanly: B7 wrote eval at `global_step=2000` with `val_loss=0.3948386136543704`, then continued training through at least `global_step=2125/5000`.
- Fifth dense handoff check closed cleanly: B7 wrote eval at `global_step=2500` with `val_loss=0.5731672507374606`, then continued training through at least `global_step=2975/5000`.
- Added `scripts/generate_vsl_label_variants.py` to derive Phase 2 label-ablation datasets from audited D6 without overwriting D6.
- VSL-2class data generated from D6 support/contradict rows: 6000 train and 800 val; structural audit accepted 6000/6000 train and 800/800 val.
- VSL-3class data generated from D6 support/contradict/uncertain rows: 8149 train and 1200 val; structural audit accepted 8149/8149 train and 1200/1200 val.
- Launched formal VSL-2class Phase 2 run on GPU0 as PID `22216`; initial log loaded 6000 train and 800 val records and reached at least `global_step=100/5000`.
- GPU state after dual-run launch: GPU0 runs VSL-2class plus a small unrelated Python service, with total memory around `9578 MiB`; GPU1 runs B7 PID `26240` at about `8853 MiB`.
- B7 VSL-4class formal run completed at `global_step=5000` with `best_val_loss=0.3948386136543704`, `train_records=11149`, `val_records=1000`, final checkpoint, final metrics, and runtime summary. The best eval was step 2000.
- VSL-2class formal run completed at `global_step=5000` with `best_val_loss=0.04671015161014566`, `train_records=6000`, `val_records=800`, final checkpoint, final metrics, and runtime summary. The best eval was step 2500.
- Added `scripts/build_vsl_results_table.py`; current formal result table has 3 rows: B7 and VSL-2class completed, VSL-3class in progress.
- Launched VSL-3class formal run on GPU0 as PID `9052`; initial log loaded 8149 train records and 1000 capped validation records.
- VSL-3class passed early eval handoff: step 500 `val_loss=0.2733266034766566`, step 1000 `val_loss=0.26886525302077646`, step 1500 `val_loss=0.6412859738743573`; training continued to at least step 1550/5000.
- After B7 and VSL-2class completed, GPU0 was occupied by VSL-3class at about `8799 MiB`; GPU1 later had an unrelated small Python service at about `1785 MiB`.
- VSL-3class continued through step 2500 eval with `val_loss=0.3237449594446225`, then resumed through at least step 2725/5000.
- Extended `scripts/generate_vsl_label_variants.py` with `vsl_4class_balanced`, an equal-class D6 sampling variant. It generated 8596 train rows with 2149 rows per class and 1600 val rows with 400 rows per class.
- VSL-4class-balanced structural audit accepted 8596/8596 train rows and 1600/1600 val rows.
- Launched VSL-4class-balanced formal run on GPU1 as PID `6384`; initial log loaded 8596 train and 1000 capped validation records and reached at least step 50/5000.
- VSL-4class-balanced continued through step 2000 eval with `val_loss=0.5415896703147446`, then resumed through at least step 2100/5000.
- Added `vsl_4class_field_balanced`, a finding-balanced D6 sampling variant covering 13 findings. It generated 5382 train rows and 767 val rows.
- VSL-4class-field-balanced structural audit accepted 5382/5382 train rows and 767/767 val rows.
- Launched VSL-4class-field-balanced formal run on GPU0 as PID `24488`; initial log loaded 5382 train and 767 val records and reached at least step 50/5000.
- VSL-3class completed at `global_step=5000` with `best_val_loss=0.14087300902791322`, final metrics, runtime summary, and final checkpoint. The best eval was step 5000.
- Refreshed `outputs/final_tables/vsl_cxr_formal_run_results.md`; current formal table has 5 rows, with B7/VSL-2class/VSL-3class completed and VSL-4class-balanced/VSL-4class-field-balanced in progress.
- VSL-4class-balanced reached step 2500 eval with `val_loss=0.6033141188775771`, then resumed training through at least step 2600/5000.
- VSL-4class-field-balanced reached step 500 eval with `val_loss=0.614734708572628` and step 1000 eval with `val_loss=0.5345917739312526`, then resumed training.
- Latest GPU process boundary after the dense check showed only VSL jobs on the GPUs: PID `24488` on GPU0 and PID `6384` on GPU1.
- VSL-4class-balanced completed at `global_step=5000` with `best_val_loss=0.4522515568471281`, final metrics, runtime summary, and final checkpoint. The best eval was step 3000.
- VSL-4class-field-balanced completed at `global_step=5000` with `best_val_loss=0.34942927536666346`, final metrics, runtime summary, and final checkpoint. The best eval was step 1500.
- After both balanced runs completed, `nvidia-smi --query-compute-apps` listed no active compute processes; VSL GPU training was closed cleanly at this boundary.
- Added an optional `training.hierarchical_vsl_loss` path to `scripts/train_qwen3vl_clinical_instruction.py`. It adds the v5 hierarchy on top of answer-token CE: answerable vs not, support vs contradict for answerable samples, and uncertain vs not.
- Added `configs/qwen3vl_instruction/vsl_cxr/vsl_hierarchical.yaml` for formal `VSL-CXR-D6-VSL4-HIERARCHICAL` on D6 four-class data.
- The VSL-hierarchical debug smoke completed at `global_step=1`, proving the tokenizer/model path can provide the required VSL label-token logits.
- Launched formal VSL-hierarchical on GPU1 as PID `14780`; it reached step 500 eval `val_loss=0.5881960963994497` and step 1000 eval `val_loss=0.7123697058961843`.
- Patched `scripts/build_vsl_results_table.py` to exclude `_debug` run directories; current formal table has 6 rows and all 6 are completed.
- VSL-hierarchical completed at `global_step=5000` with `best_val_loss=0.47355849220942764`, final metrics, runtime summary, result-table row, and final checkpoint. The best eval was step 2000.
- After VSL-hierarchical completed, `nvidia-smi --query-compute-apps` listed no active compute processes; VSL GPU training was closed cleanly at this boundary.
- Prepared v5-named Phase 1 baseline configs for B1-B6 under `configs/qwen3vl_instruction/vsl_cxr/phase1_baselines/`, all using Qwen3-VL, frozen language decoder, trainable vision tower plus visual connector, 5000 steps, and final-only checkpointing.
- B1 Basic-QA and B2 CF-QA launched as current formal v5 runs under the VSL-CXR F-drive output root. B1 PID `23488` and B2 PID `13184` both reached step 25 shortly after launch.
- B4 SAMEQ-CF has no standalone `sameq_cf_20_val.jsonl`; its formal config follows the prior SAMEQ-CF protocol and uses `outputs/instruction_data/next_stage/sameq_shuf_val.jsonl` for validation.
- B1 Basic-QA completed at `global_step=5000` with `best_val_loss=0.023826396770775318`, final metrics, runtime summary, result-table row, and final checkpoint. The best eval was step 3000.
- B3 SAMEQ launched on GPU0 as the next Phase 1 baseline after B1 released GPU0; it reached at least `global_step=175/5000` shortly after launch.
- B2 CF-QA completed at `global_step=5000` with `best_val_loss=0.12035670908632119`, final metrics, runtime summary, result-table row, and final checkpoint. The best eval was step 5000.
- B4 SAMEQ-CF launched on GPU1 as the next Phase 1 baseline after B2 released GPU1; it reached at least `global_step=175/5000` shortly after launch.
- B3 SAMEQ completed at `global_step=5000` with `best_val_loss=0.17672864127079244`, final metrics, runtime summary, result-table row, and final checkpoint. The best eval was step 3500.
- B5 SAMEQ-K4 launched on GPU0 as the next Phase 1 baseline after B3 released GPU0; it reached at least `global_step=225/5000` shortly after launch.
- B4 SAMEQ-CF completed at `global_step=5000` with `best_val_loss=0.21318705889705136`, final metrics, runtime summary, result-table row, and final checkpoint. The best eval was step 2000.
- B6 SAMEQ-HNMB launched on GPU1 as the final Phase 1 training baseline after B4 released GPU1; it reached at least `global_step=200/5000` shortly after launch.
- B5 SAMEQ-K4 completed at `global_step=5000` with `best_val_loss=0.12773469497652912`, final metrics, runtime summary, result-table row, and final checkpoint. The best eval was step 3000.
- B6 SAMEQ-HNMB completed at `global_step=5000` with `best_val_loss=0.09585380152730649`, final metrics, runtime summary, result-table row, and final checkpoint. The best eval was step 3000.
- After B6 completed, `nvidia-smi --query-compute-apps` listed no active compute processes; Phase 1 training rows closed cleanly at this boundary.
- B0 Raw-Vision is represented under the formal no-vision-training boundary as a frozen raw Qwen3-VL vision tower plus linear-probe readout. The completed B0 config has `vision_checkpoint: null` and `freeze_backbone: true`, so its metric can be used as raw feature evidence but not as a trained VSL checkpoint claim. Final metrics: macro-AUC `0.6790032275900184`, macro-F1 `0.7323508931634031`, micro-F1 `0.697265625`.

## 2026-07-16 VinDr-CXR Integration Findings

- The official VinDr-CXR archive now exists at `H:\Xiyao_Wang\000_Public Dataset\vindr-cxr-an-open-dataset-of-chest-x-rays-with-radiologist-annotations-1.0.0.zip` with size 151,581,123,766 bytes.
- Its central directory is readable and exposes 15,000 train plus 3,000 test DICOMs, bbox annotation CSVs, image-level label CSVs, and `SHA256SUMS.txt`; this supersedes the earlier claim that local VinDr data lacked labels.
- MIMIC-CXR CheXpert, NegBio, metadata, and split manifests exist as `.csv.gz` under `H:\Xiyao_Wang\000_Public Dataset\mimic_cxr_other`; the current readiness audit misses them because it checks different uncompressed paths.
- H: has 830.97 GiB free before extraction, exceeding the archive's approximately 205.97 GB uncompressed payload while leaving a substantial safety margin.
- Dataset/project separation is explicit: extracted medical data stays in the public dataset root; project-local work is limited to code, label maps, manifests, audit summaries, and documentation.
- The official image-level tables support a deterministic protocol: 45,000 train reader rows aggregate by 2-of-3 majority vote to 15,000 images; the test table has one row for each of 3,000 images; official train/test image IDs have zero overlap.
- The direct CheXpert/VSL mapping retains eight fields, but VinDr test has zero positive Edema rows. The primary external macro-AUC must therefore use seven directly comparable, non-degenerate labels; Edema stays visible in full metrics but is excluded from the primary denominator.
- Representative extracted DICOM decoding passed with `pydicom` and produces a valid RGB image after VOI LUT, percentile normalization, and MONOCHROME inversion handling. The shared LP loader now supports `.dicom`/`.dcm` paths.
- All five retained Phase 6 probe packages exist on F:, including `best_probe.pt` and `final_probe.pt`, so VinDr inference can run without retraining after extraction.
- The background suite PID 21928 waits for the extraction marker, then runs integrity audit, strict manifest regeneration, five full 3,000-image external evaluations across GPU0/GPU1, and final table/readiness refresh.

## Reuse Boundaries

## 2026-07-11 Storage Cleanup Findings

- Cleanup scope is the repository root `H:\Xiyao_Wang\021_260129VIVID`; referenced external datasets and external experiment roots are not implicitly authorized for deletion.
- Existing Git and active handoff state must be checked before deleting generated or historical material.
- A deletion manifest and resolved-path boundary validation are required before recursive removal.
- Live inventory found `outputs/` at 897.625 GiB, of which 1,352 `.pt` files account for 893.666 GiB. This makes completed-run weights the dominant safe reclamation target while preserving non-weight evidence.
- `.git/objects` is 124.305 GiB with 6,766 loose objects, but only 1,614 objects are reachable from current refs; Git garbage collection must be used instead of manual object deletion.
- `data/` is 152.594 GiB and is preserved as sensitive source/processed medical data. `pretrained/` is only 0.958 GiB and remains useful for reproducible initialization, so it is preserved.
- Post-cleanup state: `outputs/` 3.825 GiB, `.git` 1.070 GiB, `History/` 0.007 GiB, zero `.pt` files under `outputs/` or `History/`, and both `vivid_env/` and `.remote_upload_tmp/` absent.
- Approximate reclaimed space is 1,024.9 GiB (about 1.00 TiB). Git integrity passed with 1,614 reachable objects in one 1.07 GiB pack and zero loose or garbage objects.
- The VSL table rebuild chain still succeeds without the deleted weights: formal 33/33, CEQ 5/5, CCSH 9/9, AUCH 1/1, Phase 5 6 rows, external 8 rows with 5 appendix-completed, and locked final 8 rows with `VSL-Full` retained as integrated finalist.

## Reuse Boundaries (Existing Experiment Evidence)

- Prior CVCP/CCSH and SAMEQ artifacts can support baselines or interpretation only when the v5 ledger maps them to exact run definitions such as SAMEQ-full+CCSH or SAMEQ-K4+CCSH.
- VSL-4class, AUCH/insufficient labels, explicit support/contradict/uncertain/insufficient metrics, and v5 locked-final rows should be treated as not complete until current artifacts prove otherwise.
- Existing `outputs/final_tables/module_ablation_results.csv` remains useful historical embedding-level CEQ/AUCH/HNMB/DRA/CCSH evidence, but Phase 3 CEQ claims are now being regenerated with v5-named patch-token CEQ runs over D9 CEQ companion rows.
- Phase 3 CEQ quantitative runs are complete. CEQ-region is the strongest current CEQ variant on the D9 CEQ validation target: binary AUC `0.8471731089704508`, AUPRC `0.8014489081845343`, state accuracy `0.716`, ECE `0.08269000466053303`, and region accuracy `0.654`. This supports CEQ-region as the current CEQ finalist for later CCSH/AUCH/integrated rows, while attention-map casebooks remain pending.
- Phase 4 deployable readout runs are complete. Single-run leaders are CCSH-CEQ for binary AUC (`0.9059760000000001`), AUCH-CEQ-CCSH for AUPRC (`0.9005512099194461`), and AUCH-VSL4 for ECE (`0.11341642936132851`). AUCH-SAMEQ gives high answerability AUPRC (`0.9284813309209423`) but weak answerability AUC (`0.5476678876678877`) and uncertainty F1 `0.0`, so the AUCH-only result should be treated as calibration/answerability evidence, not as a strong uncertainty classifier.
- The post-Phase-4 readiness audit was superseded by the Phase 6 audit: VinBigData-derived images and NIH are locally present, MIMIC-CXR is locally present without the audited CheXpert label manifest, and PadChest remains missing.
- Phase 5 integrated candidate evidence is complete for the locally runnable candidates. VSL-CEQ is the strongest component-completed candidate by CCSH binary AUC (`0.9059760000000001`), while VSL-Full completed D9 mixed-instruction formal training with best val loss `0.19854170768998938` and inherits the current CCSH+AUCH evidence row (`AUCH-CEQ-CCSH`, AUPRC `0.9005512099194461`). VSL-Domain remains blocked by external dataset availability rather than by missing local training code.
- Phase 6 external evidence is complete only in the appendix/stress sense. NIH-appendix-1k transfer completed for Raw, SAMEQ, VSL-Core, VSL-CEQ backbone proxy, and VSL-Full. SAMEQ has the best NIH macro-AUC (`0.5932955434118374`), VSL-Core has the best NIH macro-AUPRC (`0.15464047456578672`), and all calibration numbers remain weak. This is useful domain-shift evidence but does not satisfy the v5 preferred main-external claim.
- The v5 main-external claim remains blocked by data eligibility, not by trainer/evaluator availability: the local VinBigData-derived image package has PNGs and `train_meta.csv` but no class label/bbox CSV, PadChest is absent, and local MIMIC-CXR lacks the audited CheXpert label manifest.
- Phase 7 teacher comparison is complete in a bounded audit sense. Qwen3-VL current-main smoke/full evidence is supported by the completed VSL-Core formal run plus CheXpert LP, NIH appendix/stress, and CCSH readout rows. InternVL and LLaVA/Mllama local model directories exist, but exact v5 comparison is blocked by missing family-specific VSL trainer adapters. Qwen3.5 and Qwen-Coder text-only controls require a separate non-vision VSL scaffold trainer; historical text-only scripts are not exact v5 evidence.
- Phase 8 case-study evidence is generated from current VSL-CXR artifacts rather than old mixed-plan casebooks. The new casebook has 33 rows covering VSL support, contradict, uncertain, insufficient, SAMEQ pairs, false-hard-negative review, CCSH, CEQ attention, and NIH external failures. These rows are publication candidates and still require manual image review.
- Phase 8 visualization evidence is organized in a 7-row manifest. CEQ attention-map assets exist, CCSH/external/calibration metrics are linked, and the calibration figure is bounded by missing binned curve points because current transfer outputs expose ECE/Brier/per-label metrics rather than per-sample probability bins.
- Phase 9 locked final comparison is complete with explicit boundaries. The integrated finalist is VSL-Full because it has the best Phase 6 CheXpert LP macro-AUC (`0.7148588673744163`) and completed D9 mixed-instruction training, but the NIH appendix macro-AUC still favors SAMEQ/Core and main external is blocked. Qwen3-VL 2B remains the teacher finalist because all cross-family rows lack exact v5 trainer adapters. Every locked row is single-seed, so the final claim should be written as current locked evidence rather than multi-seed statistical dominance.
- The v5 named-script surface has been closed at the readiness-audit level: after adding wrappers/manifests for D0-D5 sources, HNMB training, CheXpert/external LP summaries, VSL sufficiency, calibration, and casebook rebuilds, `scripts/audit_vsl_cxr_readiness.py` reports `script exact_exists=29` and no `missing_exact_analogs_exist` rows. This does not remove the real main-external, manual-review, binned-calibration, adapter, or multi-seed boundaries.
## 2026-07-16 VinDr Server Handoff

- The user explicitly moved execution away from the Windows workstation: the local VinDr formal suite must not continue.
- Local processes `21928`, `24956`, and `15084` were stopped before result generation; the incomplete local output directories are not formal evidence.
- Live SSH verification passed for `sues-hpc` (`mu01`). The established remote project root is `/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/021_260129VIVID`, and its existing `data/dataset` directory is writable.
- The extracted VinDr-CXR tree contains 18,008 files / 205,970,857,822 bytes (191.825 GiB). The remote project currently has no VinDr dataset path.
- The server handoff must include the five small final LP-probe packages and three source-backbone `final.pt` files, because the remote project did not expose those run directories during the live check.
- `scripts/run_vindr_external_suite.py` and `scripts/evaluate_qwen3vl_lp_transfer.py` now accept server path overrides instead of requiring Windows `H:`/`F:` paths.
- The remote VinDr transfer is not currently limited by network or global filesystem capacity. Its project destination reports `1.2P` free globally, but user-level quota enforcement rejects a 16 MiB write with `Disk quota exceeded`. The resumable state is preserved: logical ZIP parts `0000`/`0001` are done; `0002.sub.00` is done; `0002.sub.01` is safely retained at `1,849,688,064` bytes.
- A storage-minimal server path is scientifically sufficient for the already-trained external LP probes: `run_vindr_external_suite.py` consumes `vindr_cxr_external_test_ums.jsonl`, whose 3,000 entries resolve only under the official `test/` directory. Local audited sizes are `33,523,408,062` bytes for test (3,000 DICOMs) versus `172,434,631,116` bytes for train (15,000 DICOMs). A clearly labelled test-only remote package can therefore replace the full 18,000-image upload for external evaluation, provided the suite skips full-dataset extraction/audit preparation and performs a strict 3,000-image test-only manifest/path audit.
- Remote storage audit: the server portal's 1,024 GiB usage is consistent with the visible allocation. Major known consumers are `projects/xiyaowang/model` (366 GiB), `projects/xiyaowang/models/027_diffsionretrieval` (47 GiB), `projects/xiyaowang/035_Opmem` (196 GiB), `projects/xiyaowang/036_IndexMemory` (65 GiB), VIVID `outputs/` (71 GiB), the account-root `model/Qwen` (32 GiB), and `projects/xiyaowang/cache` (15 GiB: 12 GiB pip plus 2.8 GiB Hugging Face).
- VIVID `outputs/` contains 57 paired `best.pt`/`final.pt` checkpoint directories, 30.407 GiB per side (60.814 GiB total). Two SHA-256 samples prove `best.pt` and `final.pt` are not byte-identical, so neither side can be called a duplicate. The currently uploaded VinDr external-source artifacts total 2.275 GiB under `outputs/qwen3vl_external_sources` and must be retained. There are no `step_*.pt` / numeric checkpoint files in remote VIVID `outputs/`; cleanup there is a retention decision, not removal of obvious intermediates.

## 2026-07-16 BiVES-CXR Consolidation Findings

- `BiVES_CXR_MIA_TMI_ready_proposal.md` is now the user-designated final mainline. Its non-negotiable modeling object is a statement-conditioned bipolar spatial evidence set, not the existing VSL-CXR combination of VSL-4class, CEQ, CCSH, and AUCH.
- The default BiVES implementation must derive support, contradict, uncertain, and insufficient from evidence availability, decisiveness/conflict, and polarity. A separate flat four-class classifier would violate the proposal even if it improves loss.
- Existing SAMEQ/HNMB assets remain useful only for same-statement grouping, hard-negative sampling, and fair baselines. Existing CEQ/CCSH/AUCH code remains useful as legacy comparison code but must not be imported into the default BiVES package.
- The current worktree contains substantial uncommitted VSL-CXR/VinDr work from the previous phase, including new scripts/configs/docs and modifications to shared Qwen3-VL training/data code. These changes must be preserved and included deliberately rather than reset.
- Prior root proposal deletions already have recoverable copies under `History/20260707_vsl_cxr_project_organization/`; the current Git deletions are therefore intentional archive moves, not evidence loss.
- Local formal experiments remain out of scope. The correct local verification surface for this consolidation is synthetic CPU smoke/unit testing; formal training belongs on the server after the code is pushed.
- Local `H:\Xiyao_Wang\001_models` contains Qwen3.5-0.8B, 2B, 4B, and 9B. All four `config.json` files declare `Qwen3_5ForConditionalGeneration`, `model_type: qwen3_5`, image/video token IDs, and a non-empty `vision_config`; they are multimodal and can serve as the only active BiVES model family.
- The active Qwen3.5 policy is: 4B default, 2B debug/P0, 9B scale validation, 0.8B optional ultra-light smoke. Existing Qwen3-VL/Qwen2.5/other-model code and configs are legacy-only and must not appear in the active BiVES config tree.
- Independent architecture review recommends a new isolated `bives_cxr/` package with explicit tensor contracts, a bounded bipolar evidence field, a decoder containing no trainable four-class matrix, feature-space keep/drop/control rescoring, and synthetic CPU tests.
- Independent archive review recommends preserving old executable code under `legacy/vsl_cxr/` and old narrative/audit docs under `History/20260716_bives_pivot/`, while leaving large ignored outputs/data in place and referring to them from an archive manifest.
- Independent test review requires closed-form probability, S/C polarity swap, U/I semantics, intervention algebra, finite/backprop losses, mask regularizers, and an architecture contract forbidding a flat state head.
- Transformers 5.5.3 maps local Qwen3.5 configs to `Qwen3_5ForConditionalGeneration` and exposes `AutoModelForImageTextToText`. For Qwen3.5 vision output, `last_hidden_state` is the merger-preceding spatial token sequence with count `T*H*W` and dimension `vision_config.hidden_size`; `pooler_output` is merger-following and unsuitable as the default local evidence grid. BiVES uses `last_hidden_state`.
- The active repository surface is now intentionally narrow: `bives_cxr/`, three `configs/bives_cxr/qwen35_*.yaml` files, five active scripts, and two BiVES test modules. Pre-BiVES scripts/models/training/evaluation/loaders/profiles/prompts/evaluator tools are preserved under `legacy/vivid_med/`; VSL-specific assets remain under `legacy/vsl_cxr/`.
- The old VinDr UMS builder referenced an archived Qwen3-VL mapping and therefore cannot define BiVES labels. It is archived; only ZIP extraction and integrity audit remain active until a patient-disjoint four-state BiVES manifest passes the new readiness audit.
- Same-statement pair ranking is implemented in the loss but disabled in every active config until a locked group sampler and coverage audit exist. This prevents the 9B scale config from silently advertising a loss term the current loader does not supply.
- Current validation evidence: active Python compilation passed; 12/12 BiVES unit tests passed; the synthetic core smoke produced normalized four-state probabilities with no flat state head and finite gradients; a local Qwen3.5-0.8B processor-only smoke produced uniform `image_grid_thw=[[1,28,28],[1,28,28]]` from two differently shaped letterboxed CXRs without loading model weights.

## 2026-07-16 BiVES-CXR Code-Review Repair Findings

- The review was correct that the prior implementation was an architecture
  prototype rather than a P0-ready real-Qwen training path.
- Qwen3.5 merger-pre tokens are merge-block-major. The active adapter now
  restores row-major order before TV, content masks, controls, heatmaps, or
  pixel mapping.
- Both dtype boundaries are explicit: processor pixels are cast to the frozen
  vision-tower dtype, then merger-pre visual tokens are cast to the FP32 BiVES
  head dtype.
- Direct `Qwen3_5VisionModel.from_pretrained` is unsafe for the parent
  checkpoint because it does not strip `model.visual.` and initially produced
  missing/random visual weights. The final loader selectively reads
  `model.visual.*` safetensors, strips the prefix, and loads with `strict=True`.
  A direct tensor comparison against the source safetensor passed.
- Active training now requires complete same-statement S/C/U/I groups.
  Positive pair/U-polarity coefficients without aligned indices are hard
  errors, not silent no-ops.
- The evidence intervention forward path is exact-K with straight-through
  gradients. Keep/drop/control use zero replacement and branch-specific
  validity masks; controls are multiple random-disjoint exact-area masks.
- Letterbox padding is excluded by a row-major content-patch mask. A processor
  smoke over four aspect ratios produced content counts
  `[392, 392, 784, 392]` on a `28 x 28` grid and emitted no text tensors.
- Formal 4B/9B configs now use locked train/validation/calibration/test
  manifests and require frozen cached Qwen3.5 canonical statement embeddings.
  The 2B P0 config retains learned IDs as an explicitly bounded P0 ablation.
- Training now performs the strict manifest gate before visual weight loading,
  uses vision-only weights, saves resolved configuration, best/last/final
  checkpoints, resume/scheduler state, per-sample split outputs, manifest
  hashes, Git revision, and package/GPU metadata.
- Local validation after repair: Python compile passed, `20/20` BiVES tests
  passed, synthetic smoke passed, and the real Qwen3.5-0.8B vision-only smoke
  returned one finite `1 x 784 x 768` patch tensor with zero language
  parameters. No formal experiment was started.

## 2026-07-17 BiVES-CXR Round-2 Structural Review Findings

- The round-2 review identifies a real structural shortcut. The active
  `score_tokens()` path performs only per-patch fusion/heads, while keep and
  control preserve the original exact-K evidence patches and rerun selection.
  Without a post-mask cross-patch interaction, keep and disjoint-control
  probabilities are algebraically driven to match original probabilities.
- The repair must apply keep/drop/control validity masks before a shared
  statement-conditioned contextual token block. Branches may then share the
  same scorer and selector, but the evidence representation changes when
  context is removed, making sufficiency and specificity empirical rather
  than guaranteed.
- Primary validation/calibration/test must use full sequential row coverage.
  The same-statement sampler remains appropriate for training and a separate
  deterministic grouped mechanism evaluator, but cannot define the primary
  evaluation population.
- Fixed exact-K is a budget constraint, not adaptive minimality. Active P0
  documentation/configuration must say `K-budgeted evidence set` and set
  `lambda_min: 0`; adaptive minimal-set claims remain locked behind a future
  hard-concrete/L0 implementation.
- Feature-space closure remains a training/mechanism claim. Pixel-causal
  grounding requires later pixel-level keep/drop/control evaluation through
  the full vision tower and must not be inferred from the active feature-space
  intervention alone.
- The initial official-vs-selective Qwen3.5-2B mismatch was not a parameter
  loading error. The only divergent state was the non-persistent
  `visual.rotary_pos_emb.inv_freq` buffer: the selective loader downcast it to
  BF16 through `visual.to(dtype)`, while the official full model retained it in
  FP32. Preserving that buffer in FP32 after the selective move gives exact
  parameter and token-output equality.
- The bounded server integration gate now passes on the A800 with
  `331,416,576` selective visual parameters, `0` visual parameter mismatches,
  `0.0` max/mean alignment error, `0` retained language parameters, and peak
  alignment/training allocations of about `4.93` GiB / `1.16` GiB. Two
  optimization steps completed with losses `6.9683 -> 6.0655`, exact
  cardinality `16`, and nonzero keep/control effects.
- Code readiness does not yet equal experiment readiness. A fresh server check
  found none of the four P0 manifests, none of the four locked formal
  manifests, and no frozen
  `data/bives_cxr/statement_embeddings/qwen35_canonical.pt`; therefore P0 and
  formal 4B/9B dataset training remain fail-fast blocked.

## 2026-07-17 BiVES-CXR Round-3 Protocol Findings

- `canonical_statement_id` identifies the ontology item, while `group_id`
  identifies one matched S/C/U/I quartet. Sampling and pair alignment must use
  `group_id`; otherwise rows from distinct matched quartets can be mixed.
- Random-disjoint controls must be stochastic but reproducible during training
  and fully fixed during validation/calibration/test. Prediction artifacts
  need seeds, protocol version, evidence indices, control indices, and grid.
- Declared image hashes are not provenance unless the audit hashes the actual
  resolved files and uses those actual hashes for cross-split leakage checks.
- Formal statement embeddings represent a fixed canonical ontology, not
  open-vocabulary verification. The cache must fingerprint encoder settings,
  normalized statement text, vocabulary ordering, and format version.
- The locked test must be isolated from training and hyperparameter selection.
  Training may use train/validation/calibration only; final test access must be
  an explicit separate action with complete artifact hashes.
- Implemented the exact quartet contract: `group_id` is now the sampler and
  loss-alignment unit, with exactly one S/C/U/I row and one statement/text per
  group. Reused ontology statements can span multiple independent quartets.
- Evaluation controls are deterministic by split/sample/protocol seed.
  Prediction artifacts include control seed/protocol, exact evidence/control
  patch indices, and grid dimensions. Best-checkpoint selection defaults to
  original-state validation NLL.
- Manifest readiness now computes actual file SHA-256 with an 8 MiB streaming
  reader and resolved-path cache; actual hashes drive leakage checks.
- Frozen statement caches now require encoder/tokenizer/pooling/dtype
  provenance, normalized ontology text, per-text hashes, vocabulary hash, and
  finite dimension-consistent embeddings.
- Training no longer accesses the locked test. The explicit final evaluator
  requires `--run-locked-test` and records checkpoint, calibration, manifest,
  cache, code, and control-protocol provenance.
- Added eligible-conditioned specificity/gap metrics, fixed-four-class patient
  bootstrap bookkeeping, and compute-matched 4B/9B optimizer-step budgets.
# 2026-07-17 Round-4 Review Findings

- The latest independent review accepts the BiVES-CXR core architecture and
  explicitly says it should be frozen rather than restructured again.
- The remaining formal blockers are protocol-level: artifact lineage,
  full-ontology cache binding with legal test subsets, content-hash/study label
  conflict detection, and a cross-run evaluation control seed.
- The current final evaluator records artifact hashes but does not yet prove
  that its checkpoint, calibration artifact, statement cache, locked test,
  resolved config, and Git commit belong to one run.
- Exact cache validation against the test-only ontology incorrectly rejects a
  legal test subset; the full training ontology must instead travel with the
  checkpoint and remain the cache validation authority.
- Path-key conflict checks can be bypassed by byte-identical image copies under
  different paths; actual image SHA and study identity must be audited too.
- Evaluation controls are deterministic within one run but remain coupled to
  the model training seed, so multi-seed/model comparisons do not yet share a
  fixed control randomization protocol.
- The four formal blockers are now closed in code and regression tests. Active
  YAMLs intentionally retain `LOCK_AFTER_BUILD` cache fingerprints until the
  real statement cache is generated; this is a deliberate fail-fast readiness
  gate, not a runnable formal configuration value.
- Pixel-level causal evaluation remains a separate paper P1. Current results
  and code must continue to use the accurate term `feature-evidence closure`.
# 2026-07-17 Round-5 Intake

## Fifth-review evidence and decisions

- The active BiVES architecture is accepted; this round must not introduce legacy model paths or redesign the closed-form decoder.
- The current formal lock does not prove four-way data disjointness, full model weight/processor immutability, or a complete source snapshot. These are release-blocking P0 provenance gaps.
- Calibration needs a release-chain validator rather than a file-exists check. NaN temperatures must be rejected explicitly because ordinary `<= 0` checks do not reject NaN.
- Formal P0 sample caps are unsafe because statement-index mapping is created before the capped row set. Formal P0 configs will use frozen manifests without dynamic caps; debug selection will happen before ontology/mapping construction.

## Fifth-review repair outcome

- `scripts/lock_bives_dataset.py` now creates a joint audit-backed lock across all four splits. Training validates it before Qwen loading, binds its canonical SHA to `run_lock.json`, and the final evaluator revalidates it without computing test metrics first.
- Formal model locks now cover each indexed safetensors shard and local processor/tokenizer assets. A clean tracked source tree is hashed for Git runs; source-only deployments require a verified `.bives_source_manifest.json` inventory.
- Calibration artifacts now bind the calibration manifest, control protocol/seed, uncalibrated checkpoint temperatures, prediction SHA, pre/post NLL, algorithm version, and a canonical artifact SHA. Temperatures must be finite and within `[1e-4, 1e4]`.
- Active formal configs now declare dataset-lock locations and an independent bootstrap seed (`20260718`). Their P0 sample caps were removed; debug uses pre-vocabulary complete-quartet selection only.

## 2026-07-17 Round-6 Intake

- The sixth review freezes the accepted BiVES network and identifies three
  release-blocking protocol defects only: an undefined `config` in the final
  evaluator's dataset-lock branch, source-only snapshots that permit unlisted
  executable files, and calibration artifacts whose NLL claims are not
  recomputed from the locked prediction evidence.
- The user explicitly changed the immediate execution target to local. Active
  Qwen3.5 model directories are available at `H:\\Xiyao_Wang\\001_models`
  for 0.8B, 2B, 4B, and 9B. This does not waive the formal-data readiness
  locks; local formal training must still fail before model loading when
  locked manifests/cache artifacts are absent.

## 2026-07-17 Round-6 Repair Outcome

- `evaluate_bives_final.py` now receives all four split manifests explicitly,
  supports a no-model-load `--validate-release-chain-only` preflight, and no
  longer depends on checkpoint-internal paths for dataset-lock rebuilding.
- Source snapshots now enforce an exact active inventory across the BiVES
  package, active scripts/configs/tests/docs, and root startup/configuration
  hooks. This prevents ignored injected modules from evading a source-only
  release lock.
- Calibration artifacts now require a relative/absolute prediction evidence
  file plus hash. The release validator rebuilds closed-form probabilities from
  each evidence/target row and rejects any pre/post NLL claim that disagrees.
- All active configs now prefer local data and the workstation's Qwen3.5 model
  cache. This is a runtime-location change only; formal manifests, dataset
  lock, and canonical statement cache remain mandatory gates.

- User supplied a fifth review of public `main` and requested continued repair.
- The review must be treated as an issue report to verify against the current
  checked-out active BiVES surface; do not infer that it authorizes formal
  training or a new architecture.

## 2026-07-17 Round-7 Intake

- The seventh review correctly shifts the priority from server publication to
  local engineering readiness. Its public-main report predates the sixth-round
  release-chain fixes, but local inspection confirmed the separate P0 Dataset
  indentation regression: `__len__` and `__getitem__` are nested after a
  helper return instead of being class methods.
- The current active local gap is broader than a synthetic tensor smoke:
  Dataset item loading, a Windows-safe DataLoader gate, explicit local-debug
  versus local-formal protocol, local cache/lock preparation, and early GPU /
  BF16 / offline-model diagnostics.

## 2026-07-17 Round-7 Repair Result

- Restored `BiVESManifestDataset.__len__` and `__getitem__` to the class and
  added a real PNG plus `DataLoader(num_workers=0)` regression.
- Added explicit `local_debug` and `local_formal` modes. Debug is train/val
  only, selects complete groups, caps at two steps, creates no formal lock or
  calibration artifact, and records `formal_result: false`.
- Added repo-root path resolution, CUDA/BF16/GPU preflight, and
  `local_files_only=True` Qwen processor/config loads.
- Statement cache locking now copies a tracked template into a caller-selected
  ignored config, while the dataset-lock CLI validates the file it just wrote.

## 2026-07-17 Round-8 Local Mechanism Gate

- `--debug` is now refused for `local_formal`; only `local_debug` may use it.
  `local_debug`/`local_overfit` select validation rows only from the selected
  train ontology and reject patient overlap.
- BF16 validation now enters the requested CUDA device context before calling
  PyTorch's current-device API. `git_commit()` executes with `git -C` against
  the repository root.
- The first non-formal `local_overfit` run completed on CUDA 0 (RTX 3090):
  4 train rows, 4 patient-disjoint validation rows, one shared synthetic
  statement, 50/50 steps, 43.27 seconds. Its control feasibility report found
  644 valid content patches per row for `K=16`, satisfying `P_valid >= 2K`.
  The input uses transformed copies of one local CheXpert image with synthetic
  labels only; `metrics_final.json` records `formal_result: false`, so this is
  engineering evidence and not a medical or formal-result claim.

## 2026-07-17 Round-8 Mechanism Rescue Result

- The first stronger 100-step rescue was execution-green but did not pass the
  overfit learning gate. At the selected/final step, train and validation
  accuracy were both `0.75`; support was the only missed state. The train
  support row had `E+=2.8493`, `E-=3.7504`, so it remained in the contradict
  half-space despite perfect S-vs-C ranking AUROC.
- The learned mechanisms were directionally coherent in that run:
  pair-margin violation `0`, uncertain absolute polarity `0.0394`, train
  evidence-removal-to-insufficient `1.0`, irrelevant-control stability `1.0`,
  eligible target-control gap `1.3570`, and eligible control L1 effect
  `0.000258`. This isolates the failure to absolute state/polarity realization,
  not to a total absence of visual ranking or intervention response.
- Two bounded rescue candidates did not improve the gate. A 30-step
  state-only plus 30-step auxiliary ramp ended at train/val accuracy `0.50`.
  The proposal-grid candidate `lambda_IES=0.25`, run for the intended 150
  steps after correcting the old 100-step cap, selected step 110 at validation
  NLL `0.9381` and ended at train accuracy `0.25` / validation accuracy `0.50`.
- No architecture was changed and no formal result was produced. Further
  blind hyperparameter search is stopped; formal and mini-P0 execution remain
  blocked until a targeted mechanism repair can demonstrate all four states
  on the one-quartet gate.

## 2026-07-17 Support Polarity Root Cause

- The supplied analytic review matches the recorded run: with `tau_d=tau_p=1`,
  legacy support probability on the wrong half-axis has a stationary point at
  `delta=log(sqrt(2)-1)=-asinh(1)=-0.88137`; the observed support delta was
  `-0.9011`. The failure is therefore decoder geometry, not evidence that the
  image transform is intrinsically unlearnable or that IES alone is too large.
- The active migration surface includes the decoder/model config, all five
  active YAMLs, calibration fitting/reconstruction, train/final evaluator
  checkpoint temperature handling, release-chain provenance validation,
  tests, and implementation/proposal documentation. A decoder-only forward
  edit would leave the formal lock chain internally inconsistent.
- Authorized repair is the monotone bipolar conditional decoder: availability
  remains `1-exp(-T/tau_a)`; conditional S/C/U masses come from logits
  `[delta/(2*tau_p), -delta/(2*tau_p), log(2*uncertainty_mass)]`. The old
  absolute-exponential decisiveness formula is historical/ablation-only.

## 2026-07-17 Monotone Decoder Gate Result

- The decoder repair removes the support failure. The validation-NLL-selected
  step 50 has train/validation accuracy `1.0`; validation support rho is
  `+0.7957`, contradict rho is `-0.9160`, pair violation is `0`, insufficient
  total evidence is the lowest at `0.0065`, removal-to-insufficient is `1.0`,
  eligible target-control gap is `1.7962`, and eligible control L1 is
  `0.000733`.
- The only failed acceptance criterion is uncertain polarity generalization.
  Train uncertain rho is `+0.0046`, but validation uncertain rho is `+0.7130`.
  At step 30 validation uncertain rho was `+0.2644`, then drifted while train
  remained balanced; at step 100 it reached `+0.9439` and validation accuracy
  fell to `0.75`.
- The uncertain exact-K train/validation overlap is only `4/28` union patches
  (Jaccard `0.1429`) at the selected checkpoint, versus `10/22` (`0.4545`) for
  support. Capacity is sufficient to fit the train quartet, and the decoder
  monotonicity contracts pass, so the next diagnosis is the uncertain
  synthetic transform plus exact-K selection stability, not another decoder
  or loss-weight search. Mini-P0 and formal runs remain blocked.

## 2026-07-17 Uncertain Transform Replay And Repair

- The supplied stop decision is correct: support polarity is fixed, and the
  accepted monotone decoder should not be edited again in this phase. Loss
  weights, K, and Qwen3.5 capacity were also kept unchanged.
- Zero-training replay of the selected monotone-decoder checkpoint confirms
  one concrete validation-transform defect. The old U3 path
  `posterize -> contrast -> bicubic rotate` expands the uncertain cue from 8
  grayscale levels to 244 levels and has low U0 top-K overlap (Jaccard
  `0.1429`) plus low gate-logit rank agreement (Spearman `0.3729`).
- Repairing the synthetic validation order to `geometry -> state transform ->
  contrast` preserves the posterize cue (`train_uncertain.png` and
  `val_uncertain.png` both have 8 grayscale levels). However, the same
  100-step local gate still fails after this repair: train accuracy is `1.0`,
  validation accuracy is `0.75`, selected validation NLL is `0.4459805632`,
  and validation uncertain `abs(rho)` remains high at `0.8424913883`.
- Therefore the current blocker is no longer support polarity and is not fixed
  by transform order alone. The next mechanism target is exact-K
  selector/evidence-field train-to-validation stability for uncertain, using
  counterfactual replay and patch-level evidence diagnostics before any
  mini-P0 or formal run.

## 2026-07-17 Direct uncertain selector/evidence diagnosis

- `scripts/replay_bives_uncertain_selector.py` replaces approximation-only
  conclusions with an actual train/val uncertain-pair replay. It stores raw
  Qwen tokens, gates, evidence maps, masks, valid masks, grid, and affine
  transforms in `selector_evidence_arrays.pt`; JSON stores only the summary.
- A one-hot grid check establishes that the affine-grid direction used for
  `PIL.rotate(+1 degree)` is correct (`forward_mse=4.47e-10` versus inverse
  `7.51e-05`). The previously reported unaligned index Jaccard is no longer
  used as a final selector conclusion.
- On the repaired-posterize 100-step gate, aligned cross replay gave
  `rho_tt=-0.0010`, `rho_vv=0.8425`, `rho_vt=0.9355`, and
  `rho_tv=-0.8494`; all-valid validation pooling was `rho=0.5654`. Thus the
  validation polarity is already present in the field / synthetic input and
  cannot be attributed solely to exact-K selection.
- The single permitted repair was a local-only 2x2 spatial bipolar mixture:
  equal-area support-like regions at top-left/bottom-right and
  contradict-like regions at top-right/bottom-left. Train/validation masks are
  retained beside the ignored input images. The unchanged Qwen3.5-2B 100-step
  gate passed, and its direct pair replay reports
  `uncertain_failure_not_reproduced` with selected validation uncertain
  `abs(rho)=0.03850`. This validates the local engineering gate only, not a
  clinical uncertain-data claim or formal result.

## 2026-07-17 Formal P0 launch preflight

- The user authorized the next launch only after the synthetic Qwen3.5-2B
  mechanism gate passed. This does not satisfy the separate real-data
  readiness boundary.
- Exact local and remote checks both found the required four locked P0
  manifests, P0 dataset-lock JSON, and canonical Qwen3.5 statement embedding
  cache absent from the configured locations.
- The remote project also still reports source `3edb9f4`, so it does not yet
  contain the current BiVES implementation.
- Therefore no formal/mini-P0, calibration, or locked-test command was
  launched; the next legitimate step is construction and audit of the formal
  P0 assets followed by a fresh source sync.

## 2026-07-17 P0 data-source decision

- The final proposal resolves the source roles: MIMIC-CXR-JPG image/report
  pairs are the in-domain P0 candidate source; an internal expert-audited
  BiVES-CXR Audit Set is the later locked test; VinDr-CXR is an external
  availability/label/box and grounding candidate; CheXpert is secondary.
- Local storage confirms the MIMIC image and report directory trees and the
  complete VinDr train/test DICOM, image-label, and box-annotation surfaces.
  VinDr labels cannot replace the report-derived, patient-disjoint four-state
  P0 construction.
- P0 labels cannot be manufactured from report omission: the proposal requires
  explicit positive/negative auditing, a U/I blind-label pilot, same-statement
  coverage, and text-leakage checks before formal manifests are locked.
- The active source tree currently has audit, lock, and statement-cache tools,
  but no active MIMIC report parser or P0 manifest builder. The next active
  deliverable is therefore an auditable intake schema and preparation tool,
  not a formal training launch.
- `scripts/index_mimic_bives_p0_candidates.py` is now the active intake
  preparation tool. Its first local ignored shard contains `1,000` paired
  studies and `1,632` image candidates, all explicitly `unparsed` and with no
  report text or BiVES state emitted. This is suitable input to a frozen
  parser-plus-blinded-review workflow, but is not evidence that any P0 label
  is correct.

## 2026-07-17 P0 parser and blinded-review gate

- The frozen rule candidate run over the first intake shard produced `4,070`
  finding candidates from `1,632` image rows. It is explicitly nonclinical:
  `2,253` support-like, `1,608` contradict-like, `34` uncertain-like, and
  `175` internally conflicting heuristic candidates are review proposals, not
  labels.
- The stratified blinded packet has `433` rows. It includes local image/report
  paths and canonical statement text, but excludes parser state, parser cue,
  report hash, and report text.
- The packet validator was deliberately run before any review fields were
  filled and failed for all `433` rows. It requires distinct reviewer IDs,
  valid independent four-state selections, an adjudicator ID, and an
  adjudicated state. This is the current external-human-input boundary.
- No automatic process may turn these candidates into a P0 manifest, infer
  uncertain/insufficient adjudication, or launch Qwen3.5-2B P0.

## 2026-07-17 P0 clinical-review pause

- The user explicitly deferred the clinical blind review of uncertain versus
  insufficient cases and the explicit positive/negative audit because that
  qualified human input is not currently available.
- This is a pause, not a passed or failed clinical gate. The 4,070 parser
  candidates and 433-row blinded packet remain nonclinical preparation
  artifacts and are retained for a future restart.
- No weak-label substitution is authorized. Formal four-state manifests,
  dataset lock, canonical statement cache, and the Qwen3.5-2B P0 launch remain
  paused behind this non-bypassable dependency.

## 2026-07-17 local-only execution decision

- The user now requires every active BiVES-CXR experiment to run on the local
  workstation. Server synchronization, SSH execution, and Slurm submission are
  no longer part of the active experiment workflow.
- The active configs were already host-compatible: all Qwen3.5 model paths are
  under `H:/Xiyao_Wang/001_models/`, formal data roots are repository-local,
  and generated artifacts remain under ignored `outputs/` or `local_runs/`.
- This changes only the execution host. It does not convert parser candidates
  into reviewed labels or relax the formal manifest, dataset-lock, statement-
  cache, calibration, or locked-test gates. P0 remains paused because clinical
  review was separately deferred.

## 2026-07-17 permanent clinical-review removal

- The user explicitly removed qualified clinical review/adjudication from the
  executable plan because no such reviewer will be available.
- The scientifically honest replacement is a weak-label proxy P0, not a silent
  promotion of the frozen parser outputs to expert truth. Its results may test
  pipeline learnability and mechanism behavior, but cannot establish clinical
  U/I validity or close the proposal's expert-agreement claim.
- The existing 433-row packet and validator remain provenance artifacts only;
  they are no longer an execution dependency.

## 2026-07-17 weak-label proxy P0 result

- The real candidate inventory contains 4,070 rows, 1,515 unique images, and
  244 patients. Only atelectasis, consolidation, and pulmonary edema have
  enough patient-independent uncertain candidates for the deterministic proxy
  construction; pleural effusion, cardiomegaly, and pneumothorax were excluded.
- The proxy builder produced 24 train rows / 6 quartets and 12 validation rows
  / 3 quartets with global patient-disjointness, exact S/C/U/I grouping,
  source/report/image hashes, no report text/cue fields, and explicit synthetic
  evidence-removal provenance for I. The proxy lock is deliberately nonformal.
- The installed local package set differs from the older server lock. The
  active local lock now records torch 2.5.1, torchvision 0.20.1, transformers
  5.5.3, safetensors 0.7.0, numpy 1.26.4, Pillow 12.0.0, scikit-learn 1.7.2,
  and PyYAML 6.0.3.
- The first proxy diagnostic exposed a real provenance bug: the parser copied
  the image-level candidate ID into every finding row, producing cross-finding
  ID collisions. Parser v2 appends the finding, the builder now rejects global
  duplicates, and the original v1 lock/run are invalidated.
- The valid v2 Qwen3.5-2B run selected one atelectasis ontology with 8 train
  and 4 held-out validation rows. It completed 50 steps on local GPU1 in
  22.63s and selected step 30 by minimum NLL. Train S/C AUROC reached 1.0, but
  held-out S/C AUROC is 0.0; U/I AUROC is 1.0 on both. This is a weak-label/
  data-generalization failure, not evidence that the accepted closed-form
  decoder needs another redesign.
- Stop at this failed survival gate. Larger 4B/9B proxy runs and formal/locked
  claims are not justified; the next work is a read-only S/C data diagnostic.
