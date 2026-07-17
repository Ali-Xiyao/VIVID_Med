# VSL-CXR v5 Full Experiment Execution Plan

## Active Scope

Execute `vivid_med_vsl_cxr_full_experiment_plan_v5.md` under its formal VSL-CXR wording. The active method is:

```text
VSL-CXR: Visual Sufficiency Learning for Chest X-rays
```

All experiments must follow the v5 document's official definitions for visual sufficiency, SAMEQ/hard-negative data, evidence-aware encoder modules, deployable readouts, external validation, teacher comparison, case studies, and locked final comparison. Prior SAMEQ/CVCP/CCSH artifacts can only be reused when the active ledger records an exact protocol match or an explicit bounded limitation.

## Current Phase

- [x] Phase P0: Skill pre-flight, memory lookup, session catchup, current-state inventory, and project reorganization.
- [x] Phase P1: Build the VSL-CXR requirement ledger from the v5 source plan and map existing scripts/configs/artifacts to exact, partial, missing, or historical-only evidence.
- [x] Phase P2: Implement missing VSL data scripts, training entry points, evaluation scripts, and reporting scripts named in v5.
  - [x] `scripts/audit_vsl_cxr_readiness.py`
  - [x] `scripts/extract_clinical_statements.py`
  - [x] `scripts/generate_counterfactual_statements.py`
  - [x] `scripts/generate_sameq_pairs.py`
  - [x] `scripts/generate_vsl_4class_labels.py`
  - [x] `scripts/generate_hard_negative_pairs.py`
  - [x] `scripts/mine_hard_negatives_memory_bank.py`
  - [x] `scripts/generate_vsl_full_dataset.py`
  - [x] `scripts/audit_vsl_data_quality.py`
  - [x] `scripts/train_vsl_cxr.py`
  - [x] `scripts/train_vsl_hnmb.py`
  - [x] `scripts/train_vsl_full.py`
  - [x] `scripts/generate_vsl_label_variants.py`
  - [x] `scripts/eval_chexpert_lp.py`
  - [x] `scripts/eval_external_lp.py`
  - [x] `scripts/eval_vsl_sufficiency.py`
  - [x] `scripts/eval_calibration.py`
  - [x] `scripts/eval_casebook.py`
  - [x] `scripts/build_vsl_results_table.py`
  - [x] Final readiness audit now reports `script, exact_exists=29` and no `missing_exact_analogs_exist` rows.
- [x] Phase P3: Generate and audit D0-D9 data versions, including manual-audit samples and false-hard-negative/leakage gates.
  - [x] D6 VSL-4class candidate train/val generated.
  - [x] D6 structural quality audit passed.
  - [x] D6 manual correctness audit boundary recorded; manual image review remains a publication boundary, not an automated experiment blocker.
  - [x] D9 VSL-full candidate generated with instruction-mixture plus CEQ/CCSH companion files.
  - [x] D9 structural quality audit passed.
  - [x] D9 manual correctness / module-label audit boundary recorded; Phase 8 casebook rows require manual review before publication.
  - [x] D6-derived VSL-2class and VSL-3class label-ablation datasets generated and structurally audited.
  - [x] D6-derived VSL-4class-balanced sampling dataset generated and structurally audited.
- [x] Phase P4: Run Phase 1 baseline and backbone-confirmation experiments B0-B7.
  - [x] B0 Raw-Vision no-training evidence package completed as frozen raw Qwen3-VL vision tower plus linear-probe readout.
  - [x] B1 Basic-QA formal run completed on GPU0.
  - [x] B2 CF-QA formal run completed on GPU1.
  - [x] B3 SAMEQ formal run completed on GPU0.
  - [x] B4 SAMEQ-CF formal run completed on GPU1.
  - [x] B5 SAMEQ-K4 formal run completed on GPU0.
  - [x] B6 SAMEQ-HNMB formal run completed on GPU1.
  - [x] B7 VSL-4class debug trainer smoke passed.
  - [x] B7 formal run completed on GPU1.
- [x] Phase P5: Run Phase 2 VSL label/loss experiments.
  - [x] VSL-2class formal run completed on GPU0.
  - [x] VSL-3class formal run completed on GPU0.
  - [x] VSL-4class-balanced formal run completed on GPU1.
  - [x] VSL-4class-field-balanced formal run completed on GPU0.
  - [x] VSL-hierarchical loss formal run completed on GPU1.
- [x] Phase P6: Run Phase 3 CEQ evidence-aware encoder experiments.
  - [x] `scripts/train_vsl_ceq.py` implemented and debug-smoked on Qwen3-VL patch tokens.
  - [x] CEQ-basic formal run completed on GPU0.
  - [x] CEQ-diverse formal run completed on GPU1.
  - [x] CEQ-sparse formal run completed on GPU0.
  - [x] CEQ-region formal run completed on GPU1.
  - [x] CEQ-statement formal run completed on GPU0.
- [x] Phase P7: Run Phase 4 CCSH/AUCH deployable readout experiments.
  - [x] `scripts/train_vsl_ccsh.py` implemented and debug-smoked on D9 CCSH pairs.
  - [x] CCSH-Raw formal run completed.
  - [x] CCSH-SAMEQ formal run completed.
  - [x] CCSH-SAMEQ-K4 formal run completed.
  - [x] CCSH-HNMB formal run completed.
  - [x] CCSH-CEQ formal run completed.
  - [x] AUCH-CEQ-CCSH formal run completed.
  - [x] CCSH-VSL4 formal run completed.
  - [x] AUCH-SAMEQ formal row completed.
  - [x] AUCH-CCSH-SAMEQ formal row completed.
  - [x] AUCH-VSL4 formal row completed.
- [x] Phase P8: Run Phase 5 integrated VSL-CXR candidates.
- [x] Phase P9: Run Phase 6 external-validation audit and formal external evaluation.
  - [x] External-data readiness refreshed with real local paths: VinBigData image package exists without class labels, NIH exists with labels, MIMIC-CXR exists without audited CheXpert label manifest, PadChest missing.
  - [x] NIH appendix/stress manifest regenerated with 25596 official test rows.
  - [x] CheXpert LP readouts trained for SAMEQ, VSL-Core, VSL-CEQ backbone proxy, and VSL-Full; Raw LP reused from B0.
  - [x] NIH-appendix-1k transfer completed for Raw, SAMEQ, VSL-Core, VSL-CEQ backbone proxy, and VSL-Full.
  - [x] Main preferred external blocker recorded: VinDr/VinBigData labels or another eligible main-external label manifest are not available locally, so Phase 6 is closed as `completed_appendix_only`.
- [x] Phase P10: Run Phase 7 VLM teacher smoke/full comparisons where adapters and models are available.
  - [x] Qwen3-VL current-main smoke/full evidence recorded from completed v5 VSL-Core, CheXpert LP, NIH appendix/stress, and CCSH readout rows.
  - [x] Local teacher-model compatibility audit completed for Qwen3-VL, InternVL, LLaVA/Mllama, Qwen3.5 text-only, Qwen-Coder, and medical-VLM availability.
  - [x] InternVL/LLaVA/text-scaffold rows are bounded as blocked by missing VSL-specific trainer adapters or scaffold trainer, not by missing local model directories.
- [x] Phase P11: Build Phase 8 casebooks, visualizations, error taxonomy, attention maps, and calibration plots.
  - [x] VSL support/contradict/uncertain/insufficient, SAMEQ pair, false-hard-negative review, CCSH, CEQ attention, and external-failure casebooks generated.
  - [x] Figure manifest generated for Fig 1-Fig 7.
  - [x] Publication boundary recorded: casebook rows require manual visual review; calibration has ECE/Brier summaries but no exported binned curve points.
- [x] Phase P12: Build Phase 9 locked final comparison with one finalist per family, multi-seed evidence, cost, and final decision.
  - [x] One finalist selected per family from current v5 evidence.
  - [x] VSL-Full selected as the locked integrated finalist under single-seed evidence and main-external boundary.
  - [x] Boundary recorded: all locked rows are single-seed; main external remains blocked, and NIH values are appendix/stress evidence only.
- [x] Phase P13: Write final closure back into the v5 source markdown, refresh docs/ledger/final tables, and rerun post-edit verification.
  - [x] v5 source plan, README, docs index, requirement ledger, findings, progress, and task plan updated.
  - [x] All core tables regenerated after final edits.
  - [x] Final readiness audit and GPU checks recorded.

## Immediate Work Queue

| Priority | Task | Status | Evidence target |
| ---: | --- | --- | --- |
| 1 | Replace stale project navigation with VSL-CXR active docs | completed | `docs/README.md`, `README.md`, v5 source note |
| 2 | Build VSL-CXR requirement ledger | completed | `docs/vsl_cxr_requirement_ledger.md` |
| 3 | Audit existing v5-named scripts/configs/artifacts | completed | `docs/vsl_cxr_readiness_audit.md`, `outputs/final_tables/vsl_cxr_readiness_audit.csv` |
| 4 | Decide first runnable gate | completed | D6 VSL-4class generation, structural audit, and debug trainer smoke |
| 5 | Build D9 VSL-full candidate | completed | `outputs/final_tables/vsl_cxr_d9_full_dataset_manifest.md`, `outputs/final_tables/vsl_cxr_d9_data_quality_summary.md` |
| 6 | Complete formal B7 VSL-4class run | completed | `metrics_final.json`, `runtime_summary.json`, `checkpoints/final.pt`; `best_val_loss=0.3948386136543704` |
| 7 | Complete Phase 2 VSL-2class formal run | completed | `metrics_final.json`, `runtime_summary.json`, `checkpoints/final.pt`; `best_val_loss=0.04671015161014566` |
| 8 | Complete Phase 2 VSL-3class formal run | completed | `metrics_final.json`, `runtime_summary.json`, `checkpoints/final.pt`; `best_val_loss=0.14087300902791322` |
| 9 | Build formal run result table | completed | `outputs/final_tables/vsl_cxr_formal_run_results.md` |
| 10 | Complete Phase 2 VSL-4class-balanced formal run | completed | `metrics_final.json`, `runtime_summary.json`, `checkpoints/final.pt`; `best_val_loss=0.4522515568471281` |
| 11 | Complete Phase 2 VSL-4class-field-balanced formal run | completed | `metrics_final.json`, `runtime_summary.json`, `checkpoints/final.pt`; `best_val_loss=0.34942927536666346` |
| 12 | Refresh formal run result table | completed | Superseded by the current refreshed table: `outputs/final_tables/vsl_cxr_formal_run_results.md` has 33 rows and 33 completed runs |
| 13 | Implement and complete VSL-hierarchical loss row | completed | Optional `hierarchical_vsl_loss` implemented and smoke-tested; formal run completed at `global_step=5000` with `best_val_loss=0.47355849220942764` |
| 14 | Resume Phase 1 B0-B6 exact-baseline audit/runs | completed | B0-B6 completed; B0 raw no-vision-training LP evidence package has `metrics_final.json`, `resolved_config.json`, `final_probe.pt`, and result-table row |
| 15 | Implement and run Phase 3 CEQ variants | completed | Five CEQ variants completed; `outputs/final_tables/vsl_cxr_ceq_results.md` has 5 rows and 5 completed runs |
| 16 | Implement and run Phase 4 CCSH/AUCH variants | completed | Phase 4 readout closure completed: `outputs/final_tables/vsl_cxr_ccsh_results.md` has 9/9 completed CCSH/AUCH+CCSH rows and `outputs/final_tables/vsl_cxr_auch_results.md` has 1/1 completed AUCH-only row |
| 17 | Build and close Phase 5 integrated candidate evidence | completed | `VSL-Full` formal D9 mixed-instruction run completed at `global_step=5000`, `best_val_loss=0.19854170768998938`; `outputs/final_tables/vsl_cxr_phase5_candidate_results.md` has 4 component-completed candidates, 1 formal-training-completed candidate, and 1 external-data-blocked domain candidate |
| 18 | Run Phase 6 NIH appendix/stress external validation | completed_appendix_only | `outputs/final_tables/vsl_cxr_external_results.md` has 5 completed NIH-appendix-1k rows; best NIH macro-AUC is SAMEQ `0.5932955434118374`; main external is still blocked by missing VinDr/VinBigData class labels / eligible main-external label manifest |
| 19 | Run Phase 7 VLM teacher comparison audit | completed_bounded | `outputs/final_tables/vsl_cxr_teacher_comparison_results.md` has 9 Phase 7 rows: 2 current-main Qwen3-VL evidence rows and 7 bounded blocked rows for missing InternVL/LLaVA/text-scaffold VSL adapters |
| 20 | Build Phase 8 casebooks and visualization manifest | completed_needs_manual_review | `outputs/final_tables/vsl_cxr_phase8_casebook.md` has 33 rows across 9 required casebook groups; `outputs/final_tables/vsl_cxr_phase8_visualization_manifest.md` has Fig 1-Fig 7 status rows |
| 21 | Build Phase 9 locked final comparison | completed_with_boundaries | `outputs/final_tables/vsl_cxr_locked_final_comparison.md` has 8 finalist rows; integrated finalist is `VSL-Full`, teacher finalist is `Qwen3-VL 2B`, all rows are single-seed, and main external remains blocked |
| 22 | Close remaining exact v5 script entry-point gaps | completed | Added and smoke-verified wrappers/manifests for D0-D5 data sources, HNMB training entry, CheXpert/external LP summaries, VSL sufficiency, calibration, and casebook; readiness now has `exact_exists=29` |

## 2026-07-16 VinDr-CXR Main-External Data Integration

- [x] E0: Re-audit `E:\Xiyaowang`, `F:\Xiyao_Wang`, and `H:\Xiyao_Wang\000_Public Dataset` against the VSL-CXR story and v5 external-data rules.
- [x] E1: Confirm storage boundary and capacity: raw/extracted datasets stay under `H:\Xiyao_Wang\000_Public Dataset`; code, mappings, manifests, audits, and result tables stay in this repository; H: has 830.97 GiB free before extraction.
- [x] E2: Extract the official VinDr-CXR 1.0.0 package into the public-dataset root without copying DICOMs into the repository. Completed at 18,008 files / 191.825 GiB with the extraction marker present.
- [x] E3: Verify archive/extraction completeness, official checksums, DICOM counts, annotation rows, image-label coverage, and representative DICOM decoding. Passed for 15,000 train and 3,000 test DICOMs, zero missing files, zero sampled SHA mismatches, and zero sampled decode failures.
- [x] E4: Implement a deterministic VinDr-to-CheXpert/VSL label mapping and generate project-local train/test manifests without patient/image leakage. Official split has 15,000 train / 3,000 test images and zero image-ID overlap; strict final image-existence audit waits for extraction.
- [x] E5: Update readiness audit and external-result builders to discover the extracted VinDr package and MIMIC `.csv.gz` manifests rather than hard-coding stale blocker rows. External table now exposes five `pending_main_external` VinDr rows.
- [x] E6: Run manifest/data-quality audits, refresh the external/requirement/story documentation, and record the exact remaining experiment boundary.
- [ ] E7: Synchronize the completed VinDr dataset, project manifests/scripts/docs, required final LP probes, and source vision checkpoints to `sues-hpc` under the existing VIVID project directory. Project whitelist and 8/8 required probe/checkpoint artifacts are complete. Dataset upload is blocked by the remote account/project quota: logical parts `0000` and `0001` completed, `0002.sub.00` completed, and `0002.sub.01` safely paused at 1,849,688,064 bytes after the server returned `Disk quota exceeded` even for a 16 MiB write probe. Local formal inference was stopped at user request; no local formal result is claimed.

Storage rule: no medical image, DICOM, or full raw label dump is copied into Git-tracked project paths. The project stores only code, compact mappings, manifests, summaries, and reproducible provenance.

## Monitoring Policy

- While queues are stable, use direct foreground sleeps between checks, roughly every two hours.
- Increase cadence when a run approaches completion, postprocess handoff, partial-row closure, failure, memory pressure, or process anomaly.
- Do not create a scheduler/automation for this monitoring loop unless the user explicitly changes the policy.

## Evidence Rules

- A row is `completed` only when the run package, config/resolved config, metrics, diagnostics, logs, and required final-table row exist under the v5 protocol.
- A historical row can be `reusable_exact` only when data version, model/teacher, training policy, module stack, evaluation protocol, and target metric semantics match v5.
- A row can be `bounded` only when the current repo/data/model audit proves why it cannot be run exactly now.
- No final paper claim may be based only on prose, queue presence, or an old near-match artifact.
- After any source-document write-back, rerun the relevant audit before claiming completion.

## Current Repository Organization

## 2026-07-11 Storage Cleanup

- [x] C0: Load repository guidance, skills, memory context, and active handoff documents.
- [x] C1: Inventory disk usage, largest files, Git state, and references to checkpoint/model artifacts.
- [x] C2: Classify candidates into preserve, safely regenerable, and obsolete-after-result-writeback.
- [x] C3: Write a deletion manifest with resolved-path boundary checks.
- [x] C4: Delete approved in-scope candidates and record reclaimed bytes.
- [x] C5: Rebuild/check result tables and documentation references; record the retained artifact boundary.

Cleanup policy: preserve source code, configs, active documentation, final tables/metrics, sensitive datasets, and any unique final model needed for an unfinished or unreproducible claim. Delete caches, temporary uploads, duplicate/debug artifacts, superseded intermediate checkpoints, and completed-run weights only after confirming their scalar/result evidence is already durable and no active workflow requires those weights.

| Area | Current role |
| --- | --- |
| `vivid_med_vsl_cxr_full_experiment_plan_v5.md` | Active source-of-truth plan. |
| `docs/vsl_cxr_requirement_ledger.md` | Active formal requirement ledger. |
| `task_plan.md`, `findings.md`, `progress.md` | Active planning-with-files state for VSL-CXR. |
| `History/20260707_vsl_cxr_project_organization/` | Archived previous root-level plans and old planning state. |
| `outputs/` | Generated evidence artifacts; ignored by Git unless explicitly requested. |
| `configs/`, `scripts/`, `models/`, `training/`, `evaluation/`, `data/` | Active implementation and experiment surface. |

## Errors Encountered

| Error | Attempt | Resolution |
| --- | --- | --- |
| Transient `Get-Content` ProviderContentReadError while reading active training log | One monitor read during active writes | Switched to metrics-file checks and later reads; run continued normally. |
| Checkpoint directory absent for active balanced runs | Completion check before final handoff finished | Treated as expected in-progress state; rechecked after finalization and both `checkpoints/final.pt` files existed. |
| Formal result table counted hierarchical debug smoke | First table refresh after hierarchical smoke | Patched `scripts/build_vsl_results_table.py` to exclude `_debug` run directories; later refreshed table after hierarchical completion as 6 formal rows, 6 completed. |
| Bash-style heredoc failed in PowerShell | First Phase 1 config validation command | Re-ran validation with a PowerShell here-string piped to Python; config validation succeeded. |
| CEQ debug mixed bf16/fp32 tensors | First `train_vsl_ceq.py --debug` run | Cast visual patch tokens to float32 before CEQ attention. |
| CEQ debug attempted huge checkpoint JSON serialization | Second `train_vsl_ceq.py --debug` run | Store checkpoint metadata summary instead of tensor state. |
| CCSH readout label leakage | First CCSH-Raw/SAMEQ formal run embedded `label_type` derived from support/contradict target labels | Patched `scripts/train_vsl_ccsh.py` to remove `label_type` from statement embeddings, deleted the invalid generated Raw/SAMEQ output directories, and reran no-leak Raw/SAMEQ formal rows. |
| Recursive whole-repo post-cleanup size check timed out | Included the 624k-file medical-data tree in a verification pass | Re-scoped verification to directories changed by cleanup; `data/` was preserved and did not need a second traversal. |
| Initial Git gc tool wrapper timed out while work continued | Packing/pruning a 122 GiB loose-object store exceeded the wrapper yield | Detected the live `git pack-objects` process, stopped the competing read-only fsck, waited for the single gc chain, then verified zero loose/garbage objects. |
| Readiness refresh timed out while VinDr extraction was actively writing | First post-patch audit attempt shared the H: disk with the growing 206 GB extraction | Did not repeat the same short-timeout call; changed the VinDr check to marker/manifest-based constant-time logic and deferred the full audit refresh to the orchestrated post-extraction stage. |
| VinDr remote subpart could not append beyond 1,849,688,064 bytes | Remote `curl -C -` repeatedly returned write error while the server filesystem itself reported ample free capacity | Verified a 16 MiB remote write probe fails with `Disk quota exceeded`; stopped the uploader, retained completed markers/partial subpart, and require quota relief or an authorized alternate storage target before resuming. |

## 2026-07-16 BiVES-CXR Proposal Transition And Remote Storage Recovery

- [x] B0: Read `BiVES_CXR_MIA_TMI_ready_proposal.md` as the new proposal authority and map its reuse boundary.
- [x] B1: Audit remote storage and classify VIVID outputs. New BiVES work retains code/config/data/manifests, non-weight evidence, the optional VIVID initialization boundary, and current external-source artifacts; it does not require the historical 57 `best.pt`/`final.pt` pairs online.
- [ ] B2: Recover the complete remote VIVID `outputs/` tree to a local H: archive with file-level verification before any remote deletion.
- [ ] B3: Retain proposal-relevant cloud artifacts (`outputs/final_tables`, non-weight metadata/logs, and `outputs/qwen3vl_external_sources`), then delete only the recovered historical output weights from the remote project.
- [ ] B4: Recalculate remote writable space and replace the blocked full VinDr plan with the proposal-compatible test-only or later full-data plan.

## 2026-07-16 BiVES-CXR Mainline Consolidation

**Authoritative source:** `BiVES_CXR_MIA_TMI_ready_proposal.md`

**Goal:** Replace VSL-CXR as the active repository narrative with an executable BiVES-CXR codebase centered on one bipolar spatial evidence object, a closed-form four-state decoder, and keep/drop/control interventional closure. Preserve prior VSL-CXR work as pilot evidence and fair baselines without allowing CEQ/CCSH/AUCH or a flat four-class head to remain in the default BiVES model.

### Scope and boundaries

- Preserve all medical data, ignored outputs, result tables, checkpoints, and historical evidence.
- Preserve shared VIVID-Med encoders/data utilities when they are useful and disclose reuse.
- Archive superseded VSL-CXR plans, configs, wrappers, and narrative entry points instead of destroying them.
- Keep SAMEQ/HNMB as data strategies or baselines only.
- Keep CEQ/CCSH/AUCH as legacy baselines only; they must not be imported by the default BiVES model.
- All active BiVES-CXR model configurations must use the local Qwen3.5 multimodal family. Default main model: Qwen3.5-4B; P0/debug: Qwen3.5-2B; scale validation: Qwen3.5-9B; optional ultra-light smoke: Qwen3.5-0.8B. Older Qwen3-VL/Qwen2.5 and non-Qwen3.5 model families may remain only in legacy provenance.
- Do not run formal local experiments in this consolidation task. Only CPU/synthetic smoke tests are allowed locally.
- Commit and push source/config/documentation changes after validation.

### Execution phases

- [x] B0: Read repository guidance, final proposal, Git state, existing planning files, and current VSL implementation.
- [x] B1: Freeze the archive manifest and new package layout after independent architecture/archive/test audits.
- [x] B2: Archive superseded VSL-CXR/VIVID-Med active surfaces and create recoverable legacy indexes.
- [x] B3: Implement the Qwen3.5-only BiVES-CXR core package, losses, interventions, metrics, manifest audit, config schema, and training/smoke entry points.
- [x] B4: Update README/docs/AGENTS so BiVES-CXR is the only active mainline and all prior model families are explicitly legacy.
- [x] B5: Run compile checks, 12 CPU unit tests, the synthetic BiVES smoke, the Qwen3.5 processor/grid smoke, and active-path navigation audits.
- [x] B6: Reviewed the staged diff, committed the intended consolidation, pushed `main` to `origin`, verified commit equality, synchronized the source archive into the server project without touching data/outputs/pretrained/model, and reran server CPU validation.

## 2026-07-16 BiVES-CXR Code-Review Repair

**Authority:** `BiVES_CXR_MIA_TMI_ready_proposal.md` plus the user-supplied
`BiVES_CXR_code_review_2026-07-16` findings.

**Goal:** Close the gap between the correct BiVES architecture prototype and a
real Qwen3.5 P0-ready implementation. Do not start formal 4B/9B training in
this repair phase.

- [x] R0: Record the review findings and reproduce the current contract gaps.
- [x] R1: Fix BF16-to-FP32 head boundaries and restore Qwen3.5 merger-pre token
  order to row-major; add focused tests.
- [x] R2: Implement same-statement S/C/U/I group sampling, mandatory pair
  indices, and uncertain-polarity loss; enable them in active configs.
- [x] R3: Replace soft/learned-token interventions with exact-K straight-through
  evidence masks, branch-specific validity, zero replacement, and random
  disjoint equal-area controls.
- [x] R4: Add letterbox content masks, correct conditional/aggregate mechanism
  metrics, and restrict GPU batch movement to fields actually consumed.
- [x] R5: Make manifest readiness auditing a mandatory pre-model-load gate and
  strengthen split/group/semantic/image/provenance checks.
- [x] R6: Load and retain only the Qwen3.5 vision module for training, update
  locked 4B/9B data/config boundaries, and improve resolved-config/checkpoint
  metadata.
- [x] R7: Run compile, BiVES unit tests, synthetic smoke, and a bounded
  Qwen3.5 vision-only load/inference gate; then commit, push, and sync source
  changes to the server project.

R7 is complete. Local and server validation both passed compile, `20/20`
BiVES unit tests, and the synthetic smoke. The real Qwen3.5-0.8B vision-only
server smoke returned `784` finite row-major patches with `100,592,896`
visual parameters and `0` language parameters. Repair commit `eac1144` was
pushed to `origin/main` and its source archive was hash-verified and extracted
into the matching server project without touching `data/`, `outputs/`,
`pretrained/`, or shared `model/`.

### New errors / environment notes

| Error | Attempt | Resolution |
| --- | --- | --- |
| Repository guidance referenced `C:\Users\Admin\.codex\superpowers\skills\subagent-driven-development\SKILL.md`, but the file does not exist. | Skill preflight | Continue with `planning-with-files` plus bounded collaboration agents; final integration remains with the main agent. |
| `scripts/test_pipeline.py` is referenced by README/AGENTS but is absent from the active repository. | Existing smoke-entry inspection | Replace stale active documentation with a BiVES-specific CPU smoke entry point; do not restore the obsolete generic pipeline as the main entry. |
| Broad VSL import search timed out over the large repository. | Initial recursive `rg` across scripts/configs/data/models | Restrict later searches to explicit VSL/BiVES path patterns and source-only directories. |
| Initial S/C polarity-swap unit test failed at a `1.44e-8` floating-point tail difference. | First BiVES CPU unit-test run | Kept the exact symmetry check but set explicit `atol=1e-7`, `rtol=1e-6`; no model formula change was required. |
| The first `bives_cxr/model.py` add used `H:\Xiyaowang` instead of `H:\Xiyao_Wang`. | Initial core-package patch | Moved the only created file into the correct repository, verified the wrong tree contained no files, and removed the empty wrong-path directories. |
| Initial Qwen3.5 adapter divided `image_grid_thw` patch counts by `spatial_merge_size**2` while reading `last_hidden_state`. | Independent Qwen3.5 compatibility review | Corrected the adapter to use merger-preceding spatial tokens: count=`T*H*W`, grid=`H,W`, feature dim=`vision_config.hidden_size`. |
| A bash-style `python - <<'PY'` probe was accidentally issued in PowerShell during round-2 inspection. | One read-only dependency probe | No repository state changed; continue using PowerShell here-strings piped to Python. |
| A nested PowerShell/SSH quote corrupted a one-line remote PyYAML version probe. | First server lock-version check | Re-ran with a remote here-document; server PyYAML is `6.0.3`. |
| First Qwen3.5-2B integration-gate dispatch redirected into a missing `outputs/bives_cxr/` parent directory. | Initial tmux launch | No model process started; create the parent output directory before relaunching the bounded gate. |
| First real Qwen3.5-2B official-vs-selective visual comparison failed (`max_abs_error=5088`, `mean_abs_error=0.8527`) even though both visual towers had `331,416,576` parameters and the BiVES training object retained zero language parameters. | Round-2 integration gate | Keep S4 blocked. Compare full/selective state tensors and attention implementations, force a common backend, then rerun; do not accept the two-step smoke alone as a passed integration gate. |
| Replacing deprecated `np.trapz` directly with `np.trapezoid` broke the local NumPy 1.26 validation environment. | First cross-version metrics cleanup | Use `np.trapezoid` when available and fall back to `np.trapz`; formal server lock remains NumPy 2.2.6. |
| A read-only SSH monitor timed out while the eager-attention 2B integration gate was running inside the retained GPU tmux allocation. | First post-relaunch poll | Do not duplicate or restart the GPU task; wait for login SSH recovery and inspect the existing log/artifact. |
| A combined PowerShell/SSH archive command parsed the remote Python parentheses locally and stopped before upload. | First final source-sync attempt | Split archive creation, SCP, SHA256 verification, and extraction into separate commands. |
| `tmux send-keys` treated a multiword command as separate key arguments and removed spaces; a stale partial command then remained at the prompt. | First final-gate dispatch | Send `Ctrl-C`, load the complete command through a Base64 tmux buffer, paste it as one string, and then press Enter. No model process had started during the failed dispatches. |

## 2026-07-17 BiVES-CXR Round-2 Structural Repair

**Authority:** `BiVES_CXR_MIA_TMI_ready_proposal.md` plus the user-supplied
round-2 review dated 2026-07-16.

**Goal:** Remove structural intervention/evaluation shortcuts before any formal
paper run. Keep fixed-K as an explicitly budgeted P0 mechanism, not a learned
minimal-set claim. Do not unlock 4B/9B training in this phase.

- [x] S0: Read the round-2 review, active authority, handoff docs, planning
  files, Git state, and current server boundary.
- [x] S1: Make keep/drop/control nontrivial by applying branch masks before a
  shared statement-conditioned cross-patch contextual evidence block; add
  regression tests proving keep/control are not algebraic identities and the
  control objective has gradient.
- [x] S2: Separate full row-level validation/calibration/test coverage from
  deterministic grouped mechanism evaluation, with exact sample-ID coverage
  assertions.
- [x] S3: Close formal protocol gaps: fixed-K naming and zero minimality loss,
  best-checkpoint reload, ordered checkpoint metadata, fitted positive
  decoder-temperature calibration, answerable-only EOS, full classification
  and calibration metrics, and per-control mean/worst-case reporting.
- [x] S4: Pin the real Qwen3.5 integration dependencies and add a bounded
  server Qwen3.5-2B integration gate covering official-vs-vision-only alignment,
  a complete S/C/U/I forward/backward smoke, dtype/cardinality/memory evidence,
  and no language-model parameters.
- [x] S5: Run local and server regression validation, update docs/planning,
  commit, push `main`, and synchronize the source-only repair into the server
  project. Formal training remains blocked until locked manifests and frozen
  statement embeddings pass readiness.

**S4 evidence:** On the retained A800 allocation, selective Qwen3.5-2B visual
loading matched the official full model with `0` parameter mismatches and
exactly `0.0` max/mean token error. The BiVES training object retained `0`
language parameters, completed exactly two optimization steps, preserved
`K=16` on every S/C/U/I row, and produced nonzero keep/control changes.

## 2026-07-17 BiVES-CXR Round-3 Formal-Protocol Repair

**Authority:** `BiVES_CXR_MIA_TMI_ready_proposal.md` plus the user-supplied
round-3 review.

**Goal:** Close the remaining dataset grouping, reproducibility, provenance,
and locked-test protocol gates before any formal dataset run. Do not start
formal 4B/9B training in this phase.

- [x] T0: Read the round-3 review, active authority, planning files, Git state,
  and current server boundary.
- [x] T1: Make `group_id` the exact quartet sampling/alignment unit, enforce
  one S/C/U/I row and one normalized statement per formal group, and iterate
  every group in deterministic grouped evaluation.
- [x] T2: Make controls reproducible from per-sample protocol seeds; keep train
  variation epoch-aware, freeze eval controls, and export control/evidence
  indices plus seed/protocol metadata.
- [x] T3: Verify actual image SHA-256 with a resolved-path cache, use actual
  hashes for leakage auditing, and close image handles in the dataset loader.
- [x] T4: Add a provenance-complete canonical statement embedding cache
  builder and validate cache ontology/text/vocabulary fingerprints at load.
- [x] T5: Disable locked-test evaluation in training by default and add a
  separate explicit final-evaluation entry point with checkpoint/manifest/
  cache/code/calibration hashes.
- [x] T6: Add eligible-conditioned specificity metrics, fixed-four-class
  patient bootstrap accounting, compute-match the 9B config, and update the
  proposal wording from minimal to K-budgeted.
- [x] T7: Run local/server regression gates, commit/push `main`, and synchronize
  source-only changes to the server. Formal training remains blocked until
  manifests/cache are built and audited.
# 2026-07-17 Round-4 Formal Artifact Closure

## Objective

Freeze the accepted BiVES-CXR network architecture and close the four remaining
formal-run protocol blockers from the round-4 review before any 4B/9B formal
training or locked-test release.

## Checklist

| Gate | Status | Acceptance criterion |
| --- | --- | --- |
| P0-1 artifact lock chain | complete | `run_lock.json`, checkpoints, calibration artifacts, cache, config, manifests, Git commit, and final evaluator form a fail-fast SHA256 chain. |
| P0-2 ontology/cache binding | complete | Checkpoint stores the full locked ontology; test may be a consistent subset; the configured cache SHA, vocabulary SHA, and pooling must match exactly. |
| P0-3 content-level conflict audit | complete | Audit rejects conflicting states for the same actual image SHA + statement and for the same study + statement; path cache keys use `os.path.normcase`. |
| P0-4 independent evaluation control seed | complete | Validation/calibration/test controls use one locked evaluation seed independent of the training seed; the seed is recorded in the run lock. |
| Regression validation | complete | Compile, 37/37 active BiVES tests, and synthetic smoke pass locally; formal data/model training remains blocked. |
| Git/server handoff | complete | Commit `ca2130f` was pushed and synced by exact archive; 5/5 key SHA256 values matched, and server compile, 37/37 tests, and synthetic smoke passed without formal training. |

## Guardrails

- Do not redesign the accepted statement-conditioned bipolar evidence network.
- Do not reintroduce legacy model families or flat four-class heads.
- Do not access the locked test from the training loop; hashing it for the
  predeclared run lock is provenance locking, not evaluation.
- Do not launch formal 4B/9B training until all four P0 gates pass and the
  required manifests/cache exist.
# 2026-07-17 Round-5 Follow-up Repair

## Objective

Read the supplied round-5 review against current `main` (`d7864b7` claimed by
the review), identify only active BiVES-CXR defects, and repair/verify them
without changing the frozen evidence architecture or launching formal training.

## Initial rules

- `BiVES_CXR_MIA_TMI_ready_proposal.md` remains the sole research authority.
- Active code remains Qwen3.5-only and formal data/model runs remain behind
  manifest, statement-cache, and lock gates.
- Inspect the review and active source before assigning any P0/P1 status.

## Accepted fifth-review P0 repair scope

1. Add an explicit joint four-split dataset lock (`train`/`val`/`calibration`/`test`) and bind its canonical SHA to the formal run lock and final evaluator.
2. Replace the partial base-model lock with a full local model snapshot (weight shards plus required processor/tokenizer assets), and bind a clean source-tree snapshot to formal runs.
3. Make the calibrated-release chain strict: validate calibration provenance, finite bounded temperatures, checkpoint temperatures, prediction hashes, and canonical calibration-artifact integrity.
4. Remove dynamic sample limiting from formal P0 configurations. Keep any limit path debug-only and construct the statement ontology/mapping after debug selection.

## Completion

- [x] Joint four-split dataset lock builder/validator and formal config paths.
- [x] Full Qwen local snapshot (all safetensors shards plus relevant processor/tokenizer assets) and clean source-tree snapshot binding.
- [x] Strict calibrated-release validation including canonical artifact integrity, finite bounded decoder temperatures, checkpoint-temperature equality, protocol/seed/manifest binding, and prediction hash verification.
- [x] Formal P0 dynamic caps removed; debug selection happens before statement vocabulary construction.
- [x] Local regression gate: compile, 38/38 active BiVES tests, synthetic smoke, and Qwen3.5-only active-source scan.
- [x] Git/server handoff: commits `9a04245`, `03d2f0e`, and `3edb9f4` are pushed; the server source-only deployment passed compile, 38/38 tests, synthetic smoke, and archive-source snapshot verification.

## Deferred fifth-review work

- Matching-protocol registry, image-only visual prompt, minimum valid patch audit, pixel-space causal evidence, and external/VinDr formal execution remain P1 or later.
- No formal training or final test evaluation may start in this repair phase: the server remains missing the audited/frozen four-split manifests and canonical Qwen3.5-2B checkpoint.

# 2026-07-17 Round-6 Release-Chain Repair and Local Runtime

## Objective

Close the sixth-review release-chain defects without changing the frozen
BiVES-CXR architecture, and make the active configuration surface local-first
for the user's workstation. Formal training remains fail-fast until its locked
manifests and statement cache are present locally.

| Gate | Status | Acceptance criterion |
| --- | --- | --- |
| P0-1 final evaluator dataset lock | complete | The CLI now requires explicit train/val/calibration/test manifests; pass, missing config, missing split, manifest mismatch, and non-pass lock paths are integration-tested. |
| P0-2 source inventory closure | complete | Git and source-only snapshots reject unlisted files under active protected roots and root startup/config hooks; injected module and `sitecustomize.py` tests pass. |
| P0-3 calibration evidence closure | complete | Calibration predictions are required, hash-verified, and recompute both recorded pre/post NLL values from immutable evidence/target rows. |
| Local-first execution | complete | Active YAMLs use local `data` and `H:/Xiyao_Wang/001_models/Qwen3.5-*` paths; local synthetic and read-only real-weight smoke pass. |
| Regression and Git handoff | complete | Commit `b6e77d8` contains the repair; compile, 41/41 active tests, synthetic smoke, local path audit, active-path scan, 0.8B vision smoke, and a clean 896-file source snapshot all pass. |

# 2026-07-17 Round-7 Local Experiment Repair

## Objective

Close the verified local-experiment blockers without altering the frozen BiVES
network: prove real Dataset/DataLoader behavior, split local debug from local
formal protocol, make local cache/lock preparation reproducible without
dirtying tracked YAML, and add fail-fast local runtime diagnostics.

| Gate | Status | Acceptance criterion |
| --- | --- | --- |
| Dataset P0 verification | complete | Restored Dataset methods; temporary RGB image, item, and `DataLoader(num_workers=0)` regression pass. |
| Local debug/formal split | complete | Explicit debug/formal modes; debug caps complete matched groups/two steps and emits `formal_result: false`; formal keeps locks. |
| Local preparation workflow | complete | Cache builder writes a caller-selected local config; dataset-lock CLI revalidates the generated four-split lock. |
| Local runtime preflight | complete | Repo-root path resolution, Windows-safe workers, CUDA/BF16 diagnostics, and offline Qwen loading fail before costly work. |
| Regression and Git handoff | in_progress | 43 tests, synthetic smoke, py_compile, and diff check pass; commit/push remains. |

# 2026-07-17 Round-8 Local Mechanism Gate

## Objective

Close the remaining local-run identity and ontology defects, then run the
smallest non-formal Qwen3.5 local mechanism gate. The accepted BiVES network
remains frozen; formal locks, calibration, and test evaluation remain out of
scope.

| Gate | Status | Acceptance criterion |
| --- | --- | --- |
| Debug identity + ontology | complete | `local_formal --debug` is rejected; debug/overfit select validation quartets from the selected train ontology and enforce patient disjointness. |
| Device + provenance correctness | complete | BF16 is checked under the requested CUDA device context; Git lookup is rooted at the repository. |
| Local overfit mechanism gate | complete | Added one-quartet non-formal overfit template, actual-image transformation helper, and pre-optimizer `P_valid >= 2K` report. |
| Actual local run | complete | `cuda:0` RTX 3090 completed 50/50 steps on the generated non-clinical mechanism input; result is explicitly non-formal. |
| Regression and Git handoff | complete | Formal-debug rejection, 44 tests, compile, synthetic smoke, and diff check pass; repair is pushed as `4f5b2da`. |

## Round-8 mechanism rescue

| Gate | Status | Acceptance criterion |
| --- | --- | --- |
| Diagnose first run | complete | Separate engineering completion from learning survival; first run peaked at val accuracy 0.50 / NLL 1.4129 at step 5 and later collapsed. |
| Instrument train behavior | complete | Train-side primary metrics, all loss terms, auxiliary weight, and train predictions are emitted at each overfit checkpoint. |
| Stronger synthetic probe | complete | Ran separable non-clinical S/C/U/I transforms through the real frozen Qwen3.5-2B visual path; added the proposal-aligned optional state-only/auxiliary-ramp schedule and a 200-step non-formal safety ceiling. |
| Rescue decision | complete_failed_gate | The best bounded run reached train/val accuracy 0.75 with correct ranking/intervention directions but never learned absolute support polarity; ramped and `lambda_IES=0.25` candidates were worse. Formal/mini-P0 work stays blocked. |

# 2026-07-17 Monotone Decoder Repair

## Objective

Replace only the non-monotone conditional geometry of the closed-form BiVES
decoder, preserve the bipolar evidence field and no-flat-head contract, close
the calibration/provenance/config migration, and rerun exactly one unchanged
100-step local mechanism gate before considering mini-P0.

| Gate | Status | Acceptance criterion |
| --- | --- | --- |
| Root-cause verification | complete | Reproduce the legacy wrong-polarity stationary point near `-asinh(1)` and enumerate every active `tau_d`/decoder provenance dependency. |
| Monotone decoder implementation | complete | Active decoder uses conditional S/C/U softmax with positive `uncertainty_mass`; S rises strictly with delta, C falls strictly, U is symmetric/maximal at zero, and no trainable flat head is added. |
| Calibration/release migration | complete | `tau_d` is replaced by `uncertainty_mass` in calibration, checkpoint provenance, evaluator, all active configs, and method docs; run-lock/calibration format is v3. |
| Regression gate | complete | Compile, synthetic smoke, 48/48 tests, 1001-point monotonicity, all-half-axis gradient direction, legacy `-asinh(1)` trap, and release-chain tests pass. |
| Controlled GPU gate | complete_failed_uncertain_generalization | The complete 100-step run selected step 50 with train/val accuracy 1.0 and fixed S/C polarity, but val uncertain `abs(rho)=0.7130` while train uncertain is `0.0046`; all other intervention checks pass. |
| Git handoff | complete | Repair and explicit failed-one-criterion decision are committed; regression evidence is recorded and `main` is pushed. Formal/mini-P0 stays blocked until uncertain train-to-val stability is repaired. |

# 2026-07-17 Uncertain Transform Replay and Local Gate

## Objective

Follow the supplied support-polarity follow-up without touching the accepted
decoder, loss weights, K budget, or model capacity. The only allowed code repair
is the synthetic validation transform order; mini-P0 and formal execution remain
blocked unless the same local 100-step mechanism gate passes.

| Gate | Status | Acceptance criterion |
| --- | --- | --- |
| Zero-training transform replay | complete | Replayed the selected monotone-decoder checkpoint under U0-U5 uncertain transform variants and recorded state probabilities, signed evidence, top-K overlap, K sweep, gate-logit Spearman, and Qwen token similarity. |
| Synthetic transform-order repair | complete | Validation synthetic images now apply geometry before the state transform, use median fill, then apply photometric contrast last; uncertain validation keeps the posterize cue at 8 gray levels. |
| Controlled 100-step local gate | complete_failed_uncertain_stability | The unchanged Qwen3.5-2B/K=16/LR/loss 100-step local gate still fails: train accuracy `1.0`, validation accuracy `0.75`, selected/final validation NLL `0.4459805632`, and validation uncertain `abs(rho)=0.8424913883`. |
| Formal/mini-P0 boundary | blocked | No mini-P0, formal training, calibration, or locked-test evaluation was launched. The next target is selector/evidence-field stability rather than decoder geometry, loss weights, K, or model capacity. |

# 2026-07-17 Uncertain Selector/Evidence Diagnostic and Spatial Gate

## Objective

Use the supplied selector/evidence plan without changing the accepted decoder,
losses, exact-K budget, or Qwen3.5 capacity. First diagnose the real uncertain
train/validation pair with aligned cross-mask replay, then make at most one
data-side repair and rerun one non-formal 100-step mechanism gate.

| Gate | Status | Acceptance criterion |
| --- | --- | --- |
| Direct aligned pair replay | complete | `replay_bives_uncertain_selector.py` reads actual train/val uncertain images, verifies PIL rotation direction, saves patch arrays, and emits aligned Rtt/Rvv/Rvt/Rtv, soft/all/K sweep, and patch contribution evidence. |
| Root-cause decision | complete | Old posterized fixture is evidence-field/synthetic-definition dominant: Rvv `rho=0.8425`, Rvt `rho=0.9355`, and all-patch validation `rho=0.5654`; it is not a selector-only failure. |
| Bounded repair | complete | Replaced only the local synthetic uncertain engineering fixture with an equal-area 2x2 support/contradict spatial mixture and saved positive/negative masks. No decoder, loss, K, or capacity change. |
| Controlled 100-step local gate | complete_passed | Qwen3.5-2B selected step 80 at validation NLL `0.3711495355`, accuracy `1.0`, uncertain `abs(rho)=0.038496`; step 100 remains accuracy `1.0`, uncertain `abs(rho)=0.032554`. |
| Mini-P0/formal boundary | blocked_data_readiness | The synthetic mechanism gate is green, but no mini-P0/formal run was launched. Frozen real manifests and the canonical statement cache are still required before the separate readiness audit can unlock execution. |

# 2026-07-17 Formal P0 Launch Authorization Preflight

| Gate | Status | Evidence |
| --- | --- | --- |
| User launch authorization | complete | User authorized launch after the local synthetic uncertainty gate passed. |
| Local locked P0 assets | blocked | `p0_train.jsonl`, `p0_val.jsonl`, `p0_calibration.jsonl`, `p0_test.jsonl`, `p0_dataset_lock.json`, and `qwen35_canonical.pt` are absent from the exact configured paths. |
| Server locked P0 assets | blocked | The same six configured assets are absent under the remote project root. |
| Server source freshness | blocked | The remote `.bives_source_commit` remains at historical source `3edb9f4`, not the current local/pushed BiVES implementation. |
| Formal launch | blocked_data_readiness | No Qwen3.5 model load, GPU allocation, training, calibration, or locked-test job was submitted. Real-data manifest/cache construction and a fresh source sync are required first. |

# 2026-07-17 P0 Data-Source Decision and Intake Preparation

| Work item | Status | Decision / evidence |
| --- | --- | --- |
| P0 in-domain source | complete | Use local `H:\Xiyao_Wang\000_Public Dataset\mimic-cxr\mimic-cxr`: it contains separate MIMIC image and report trees arranged by `pXX/patient/study`. MIMIC reports are candidate generation only until P0-1/P0-2 audit evidence exists. |
| P0 external route | complete | Use VinDr-CXR only for P0-5 availability/label/box/permission audit and later external evaluation. Its train/test image-label CSVs and box annotation CSVs are present; it must not be mixed into the P0 in-domain train split. |
| CheXpert role | complete | Keep CheXpert as secondary external/linear-probe material, not a substitute for report-derived four-state P0 labels. |
| Active P0 intake chain | in_progress | Added an active path-only MIMIC intake indexer and generated the first ignored shard (`1,000` paired studies / `1,632` images). A frozen parser candidate table, blind review, adjudication, matching, four-split manifest construction, and cache build remain before training. |
| Source sync / formal launch | blocked | The remote project lacks the raw-data root and remains on historical source `3edb9f4`; synchronize only after local P0 audit artifacts pass. |

## P0 automated completion boundary

The user authorized completion of the P0 chain and formal launch. The active
work is limited to automatable, provenance-preserving preparation:

1. fixed-rule report parser candidates with no clinical-label claim; complete;
2. blinded reviewer packet generation and validator; complete and fail-closed;
3. post-review manifest/lock/cache build only when two reviewer labels and an
   adjudicated state are actually supplied.

No code path may fill reviewer fields, infer an adjudication, or submit a
Qwen3.5 P0 job from parser candidates alone.

| Gate | Status | Evidence |
| --- | --- | --- |
| Frozen parser candidate preparation | complete_nonclinical | The first 1,000-study intake shard produced 4,070 parser candidates under rules SHA256 `3340d89e...d55bee`; every row declares `labeling_claim: none`. |
| Blinded review packet | complete_waiting_for_human_review | A 433-row packet is ready under ignored `local_runs/bives_cxr/p0_intake/`; it omits parser state and report text. |
| Independent review/adjudication | deferred_by_user | On 2026-07-17 the user explicitly deferred clinical blind review and explicit positive/negative auditing because qualified review is not currently available. The unfilled 433-row packet is retained for a later restart. |
| Formal manifest/cache/2B P0 launch | paused_dependency | This remains paused behind the deferred, non-bypassable clinical-review gate. Parser candidates must not be substituted for reviewed labels. |

### P0 pause decision

- Do not request or fabricate reviewer/adjudicator decisions.
- Do not build formal four-state manifests, a dataset lock, or the canonical
  statement cache from the nonclinical parser candidates.
- Do not launch a Qwen3.5 P0 job while this dependency is paused.
- Preserve the ignored candidate table, blinded packet, and validator so work
  can resume from the same auditable boundary when qualified review becomes
  available.

# 2026-07-17 Local-Only Experiment Policy

The user replaced the previous mixed local/server execution policy. From this
point forward, every active BiVES-CXR experiment runs on this workstation.

| Work item | Status | Decision / evidence |
| --- | --- | --- |
| Active YAML host paths | complete | All five active configs use local `H:/Xiyao_Wang/001_models/Qwen3.5-*` model paths and local data/output roots. |
| Active docs and CLI wording | complete | Repository guidance, handoff docs, manifest audit, source manifest, and integration-gate messages now describe local-only execution. |
| Formal environment lock | complete | Renamed the active lock to `requirements-bives-local-lock.txt`; package versions are unchanged. |
| Regression guard | complete | `test_active_configs_are_qwen35_only` now also rejects remote markers and asserts local formal data/output roots. |
| Clinical/P0 readiness gate | superseded_by_proxy_policy | Local-only execution changes the host. The later weak-label authorization permanently removes clinical review from execution, while preserving the rule that proxy results are nonclinical and nonformal. |
| Remote experiment operations | retired | Do not sync active experiment assets or submit SSH/Slurm jobs. Historical remote records remain unchanged as provenance. |

## Local-only transition errors

| Error | Attempt | Resolution |
| --- | --- | --- |
| A combined read/search command returned exit code 1 after printing the requested files. | First active-surface inspection | The final `rg` searched tests for server-only wording and correctly found zero matches; reran targeted scans and treated zero hits as a pass rather than a code failure. |
| Direct `python -m unittest tests\\test_bives_proxy_p0.py -v` could not import the test module. | First proxy-builder unit-test invocation | This repository does not package `tests/`; use discovery with `python -m unittest discover -s tests -p "test_bives_proxy_p0.py" -v`. |

# 2026-07-17 Weak-Label Proxy P0 Authorization

The user permanently removed qualified clinical review/adjudication from the
executable workflow because no reviewer is available. This supersedes the
earlier `deferred_by_user` launch dependency, but it does not turn parser rules
into clinical ground truth.

## Claim boundary

- The next run is a **weak-label proxy P0 engineering experiment**, not an
  expert-audited clinical P0 and not evidence that U/I labels are clinically
  valid.
- Every generated row must preserve parser version/rules hash, source report
  hash, source study/image identity, and an explicit weak/synthetic label
  provenance field.
- Report omission must still not become contradiction or insufficiency.
- `insufficient` may be introduced only through an explicit reproducible
  synthetic evidence-removal transform with source-image provenance.
- No locked-test or publication-level clinical claim is authorized from this
  proxy run.

| Gate | Status | Acceptance criterion |
| --- | --- | --- |
| Candidate-state inventory | complete | Audited 4,070 candidates / 1,515 images / 244 patients. Retained atelectasis, consolidation, and pulmonary edema; excluded findings without enough independent uncertain patients. |
| Proxy manifest builder | complete | Built deterministic patient-disjoint manifests with 24 train rows / 6 quartets and 12 validation rows / 3 quartets. S/C/U are rule-derived; I is explicit synthetic evidence removal. |
| Proxy audit/lock | complete_nonclinical | Fixed cross-finding candidate-ID collisions, regenerated parser v2, and re-audited 4,070 globally unique IDs. Current proxy lock SHA256 is `2cb4f963acab7a66fbece3212c1307b66b71a58013e80719fbbc4b462acf4b19` and declares `formal_result=false`. |
| Local environment/model preflight | complete | Qwen3.5-2B vision smoke passed on local GPU1; package lock was corrected to the installed local environment; H: space and output path passed. |
| Local Qwen3.5-2B proxy P0 | complete_failed_proxy_polarity_generalization | The valid v2 bounded atelectasis run completed 50/50 steps in 22.63s and selected step 30. Train S/C AUROC is 1.0 but held-out proxy S/C AUROC is 0.0; U/I AUROC is 1.0. Stop at this survival gate and do not scale to 4B/9B. |

## Proxy P0 decision

- The run is execution-green but learning-red for held-out support versus
  contradiction polarity.
- The accepted decoder, loss weights, exact-K budget, and Qwen3.5 capacity are
  not reopened by this result. The synthetic mechanism gate already passed.
- The next permissible work is a read-only weak-label/data-split diagnostic of
  S/C cue noise, statement balance, and source leakage. No larger-model run is
  authorized from this proxy result.

## Proxy P0 execution errors

| Error | Attempt | Resolution |
| --- | --- | --- |
| Candidate inventory initially queried obsolete state-field names. | First parser-table count | Re-read the current candidate schema and counted `parser_state_candidate`. |
| `rg tests\test_bives_*.py` treated the wildcard as a literal Windows path. | First targeted test search | Used unittest discovery and repository-relative glob handling. |
| One large `apply_patch` contained an empty hunk. | Combined implementation patch | Split the change into focused patches; no file was partially modified. |
| The renamed package lock still contained server-era versions. | Local runtime preflight | Replaced them with the actually installed local versions and revalidated imports. |
| `candidate_id` omitted the finding and collided across six statements for each image. | First proxy-run provenance diagnostic | Parser v2 emits `<image-candidate>::<finding>` IDs, builder rejects global duplicates, artifacts were regenerated, and the v1 run/lock was invalidated. |
| Recursive cleanup of the ignored proxy directory was blocked by local command policy. | Pre-regeneration cleanup | Performed safe in-place overwrite; the regenerated manifests and lock enumerate only current files. |

# 2026-07-17 Weak-Label S/C Diagnostic

## Objective

Explain the corrected proxy-P0 held-out support/contradict failure without
changing the decoder, losses, exact-K budget, model capacity, or starting a new
training run. Diagnose the weak-label/data boundary first.

| Work item | Status | Acceptance criterion |
| --- | --- | --- |
| Failure-case collection | complete | Preserve selected-step train/validation evidence signs and confirm the failure is S/C-specific rather than a runtime or U/I failure. |
| Parser/sample audit | complete | Parser v3 scopes negation/uncertainty to the target mention and preserves cross-line context. It regenerated 4,201 unique candidate rows with rules hash `224cb4c4...530a`. |
| Frozen-feature audit | complete | Read-only local Qwen3.5-2B features show AUROC 0.750 for pleural effusion and 0.785 for pulmonary edema; atelectasis remains 0.360 with only five contradict patients. |
| Root-cause decision | complete | The failure combines a real parser scope bug, an unreliable atelectasis ontology, and a one-quartet validation split. The one justified repair is a larger patient-disjoint proxy restricted to the two feature-separable findings. |
| Training boundary | complete_mixed_proxy_result | The one permitted local Qwen3.5-2B run completed 50/50 steps. Aggregate held-out S/C AUROC is 0.8125, but pleural effusion is 0.5 versus pulmonary edema 1.0; no second run or 4B/9B scaling is authorized. |

## S/C diagnostic errors

| Error | Attempt | Resolution |
| --- | --- | --- |
| Foreground frozen-feature command exceeded the terminal's short yield timeout. | First real Qwen3.5 feature run | The process had already completed and wrote valid JSON/NPZ; verified GPU release and consumed the structured artifacts instead of relaunching. |
| One-off negation-scope probe referenced `s` instead of `sent`. | First contextual report audit | It failed before producing counts; corrected the local variable and reran the read-only probe. |
| `Start-Process` returned a PID but the diagnostic child exited before logging. | First parser-v3 feature launch | Relaunched the same read-only command as a bounded foreground job; it completed in 20.5 seconds and released GPU1. |
| `metrics_final.json` had an empty `split_metrics` map for nonformal runs. | Parser-v3 result collection | The step events and predictions were intact. Patched the trainer to re-evaluate the validation-selected checkpoint and write final `val` plus `train_proxy` metrics for future local runs. |
