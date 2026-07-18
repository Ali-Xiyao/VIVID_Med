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

# 2026-07-17 Expanded Patient-Disjoint Proxy Validation

## Objective

Test whether the parser-v3 mixed per-finding result is caused by the initial
1,000-study intake and four S/C validation examples per finding. Change only
the local MIMIC intake size; keep parser v3, Qwen3.5-2B, decoder, losses, K,
seed, and training budget frozen.

| Gate | Status | Acceptance criterion |
| --- | --- | --- |
| 5k intake index | complete | Indexed 5,000/5,000 paired studies and 8,220 images with zero missing-report or empty-image studies; raw images remained outside the repo. |
| Parser-v3 coverage | complete | Regenerated 20,204/20,204 unique candidates with unchanged rules hash. Pleural effusion, pulmonary edema, consolidation, and pneumothorax each have >=20 independent S and C patients; atelectasis/cardiomegaly do not. |
| Frozen-feature per-finding gate | complete | On 20 S + 20 C patients each, pleural effusion=0.7425, pulmonary edema=0.7775, and consolidation=0.8425 pass; pneumothorax=0.6050 is excluded. |
| Expanded proxy lock | complete | Built 48 train / 48 validation rows (12 quartets each) across the three passing findings; audit passed with lock `3473ad6a...ae9df`. |
| One bounded 2B rerun | complete_mixed_gate | The local 50-step Qwen3.5-2B run completed normally on 48/48 rows and selected step 50. Aggregate held-out S/C AUROC is 0.8056 and every retained finding is >=0.8125; U/I is 1.0. Absolute four-state argmax collapses all 48 validation rows to insufficient (accuracy 0.25), so ranking passes but the decision gate fails. No 4B/9B, formal calibration, locked test, or model/hyperparameter change is authorized. |
| Zero-training decoder-geometry attribution | complete_partial_explanation | Fit the existing three positive decoder parameters on train-proxy evidence only and evaluated the frozen validation evidence. NLL improves 1.3692 -> 1.1620 and accuracy 0.25 -> 0.5417, but only 1/12 contradict and 2/12 uncertain rows are recovered. This is diagnostic-only, not a locked calibration result. |
| Frozen evidence-distribution attribution | complete_root_cause | Validation insufficient has the lowest median total evidence (0.4356 versus 0.8518-0.9387), but median signed evidence remains positive for contradict (+0.0464) and uncertain (+0.0863). The model learns availability and relative ordering but not the absolute bipolar origin. |

## Expansion decision

- Stop after the single pre-authorized 2B run; do not reinterpret ranking AUROC
  as four-state classification readiness.
- Preserve the expanded patient-disjoint validation split and selected-step
  predictions as the fixed diagnostic surface.
- Probability geometry is a confirmed partial cause, and the frozen evidence
  distribution localizes the residual to absolute polarity centering. Do not
  change decoder/loss/K or start another model run inside this expansion cycle.

## Expansion errors

| Error | Attempt | Resolution |
| --- | --- | --- |
| A combined planning/progress patch used an outdated exact hash string as its context and did not apply. | First 5k planning writeback | Re-read the file tails and applied smaller, stable-anchor patches; no file was partially changed. |
| A combined six-file result patch used an exact experiment-log line break that did not match. | First decoder-diagnostic writeback | The patch was rejected before any hunk applied. Split it into per-file patches and corrected the canonical per-finding metric before writeback. |

# 2026-07-18 Optimization-Identifiability Gate

## Authority and frozen boundary

`BiVES_next_direction_without_local_clinical_review_2026-07-17.md` is the
execution authority for this cycle. Commit `c113b2a` is frozen as
**Proxy-P0-A**: 5,000-study candidate pool, 48/48 proxy rows, Qwen3.5-2B frozen
vision, 50 steps, relative S/C ranking signal, synthetic availability signal,
and failed absolute four-state fit. Do not overwrite or reinterpret it.

This cycle changes no parser pool, labels, clinical-claim boundary, model
family, exact-K budget, decoder, or evidence parameterization before the two
predeclared optimization diagnostics are evaluated.

| Gate | Status | Acceptance / stop rule |
| --- | --- | --- |
| Proxy-P0-A freeze record | complete | `docs/bives_cxr_proxy_p0_a_freeze.md` binds `c113b2a`, hashes the local config/manifests/metrics/checkpoint, and freezes the nonclinical failure boundary without publishing ignored data or weights. |
| Step0/step50 evidence audit | implementation_complete | `bives_cxr/optimization_audit.py` and the trainer save fixed-quartet E+/E-/T/delta/rho/probabilities/dense gate logits, signed-state direction, per-loss module gradient norms/cosines, and full train/val evidence summaries at steps 0/50/400. Runtime evidence begins with Run A. |
| Frozen-feature logistic probes | complete | Patient-group-disjoint global AUROC is `0.7889`; pleural effusion/pulmonary edema/consolidation are `0.8550/0.8075/0.8000`, with intercept, NLL, Brier, and ECE saved under ignored local outputs. |
| Run A: state-only overfit | complete_failed_survival_gate | Completed the fixed local 400-step run with final-step selection. Train accuracy is `0.7917`, below the required `1.0`; train S/C polarity is correct and insufficient has the lowest total evidence, but support and uncertain each recall only `8/12`. |
| Run B: full-objective overfit | not_run_by_gate | The authority permits Run B only after Run A fits all 48 train rows. Run A failed, so Run B is intentionally not launched. |
| Optimization-identifiability verdict | complete_hard_stop | `docs/bives_cxr_optimization_identifiability_verdict.md` records the fixed result. Stop on current optimization/readout, selector, or effective capacity; no auxiliary-conflict claim, parameterization refactor, sweep, extension, or 4B/9B scaling is authorized. |
| Public expert-data route | intake_complete_model_gate_closed | VinDr integrity passed (`18,006` entries, `0` missing, `70/70` sampled hashes, `16/16` decodes). The ignored test-consensus intake has 6,000 S/C rows for pleural effusion and consolidation; edema is ineligible with zero positives. Patient-level CI is blocked because the public release exposes no patient ID. CheXpert expert provenance and CheXlocalize remain unresolved. Do not run the failed current BiVES route on this intake. |

## Hard stops

- No Qwen3.5-4B/9B run.
- No parser U/I expansion or new rule-label search.
- No loss-weight sweep, decoder change, or magnitude-polarity refactor before
  the A/B verdict.
- No formal calibration, locked-test, or MIA/TMI-ready clinical claim.
- All execution remains local; no SSH/server/Slurm work.

# 2026-07-18 Expert Polarity + Interventional Evidence Route

## Authority and scope

`BiVES_995fb81_code_review_and_next_plan.md` governs the next route under the
final `BiVES_CXR_MIA_TMI_ready_proposal.md`. Commit `995fb81` remains the frozen
closeout of the parser-S/C/U + synthetic-I four-state route. The new primary
axes are public expert S/C polarity and pixel-level interventional evidence
sufficiency. U/I remain exploratory only.

| Phase | Status | Gate |
| --- | --- | --- |
| E0 review intake and closeout pin | complete | The tracked plan and handoff index preserve `995fb81` as closeout without reopening Run B, 4B, or 9B. |
| E1 engineering P0 fixes | complete | Invalid formal `--debug` removed; unused intervention branches skipped; all train quartets are audited; pre-clip norms/coefficient/fraction are recorded. Full suite passed 70/70 at closure. |
| E2 VinDr standard input path | complete | Deterministic DICOM preprocessing and four synthetic contracts pass; 16 real train/test samples cover MONOCHROME1/2 with unique deterministic hashes. |
| E3 independent Expert S/C interface | complete | Independent schema/evaluator has exact coverage, locked dev thresholds, per-finding metrics, and image-level clustered CI without quartet/U/I/patient requirements. |
| E4 VinDr formal data preflight | complete | Final integrity audit passed: all 18,006 official hashes match, all 3,000 test DICOMs plus 8 train samples decode under `bives_cxr_dicom_v1`, and no missing/hash/decode failure was observed. No model was loaded in this phase. |
| E5 weak S/C train/validation | complete | Locked 816 train / 274 val rows; 544/171 patients with zero overlap; both findings are S/C-balanced and only explicit cues are retained. Image existence passed; cache stage binds image SHA. |
| E6 frozen token cache | complete | Audited cache contains 1,046 unique images and 1,090 train/validation index rows. Every item file hash, source-image SHA, payload identity, model snapshot, processor snapshot, shared 448 geometry, content mask, and grid passed the full cache audit. |
| E7 B0/B1/B2 local 2B gate | complete_B2_promoted_to_external_gate | B0 pooled is frozen at macro AUROC/AUPRC `0.7857/0.7992`; B1 dense selected step 300 at `0.7713/0.7910`. B2 exact-K=16 selected step 450 at `0.8423/0.8240`, with both retained findings above B0/B1 in aggregate ranking; its 8/20 clipped eval points and maximum pre-clip norm `520.3` remain explicit limitations. |
| E8 expert polarity/intervention gate | complete_failed_stop | Corrected per-image-isolated seed-17 expert inference and 205-image paired intervention are complete. B2 fails the all-finding B0 comparison because consolidation AUPRC is `0.2338 < 0.2628`. Primary TCIG crosses zero for consolidation and is significantly negative for pleural effusion; localization gain alone cannot promote the route. Stop before more seeds or 4B/9B. |
| E9 CheXlocalize | not_started_by_E8_stop | No download/auth action was started. A new dataset or patient-level route requires a new reviewed authority rather than automatic continuation after the failed seed-17 causal gate. |
| E10 post-stop failure taxonomy | complete_failed_explained_no_rescue | Frozen 410-row analysis rules out a single-outlier explanation and localizes the failure to inconsistent selector localization plus disproportionate sensitivity to large arbitrary control deletions. Because VinDr test outcomes informed this diagnosis, it cannot authorize tuning or a same-test rerun. No model load, new experiment, method change, seed expansion, CheXlocalize action, or 4B/9B unlock occurred. |

Final closure validation passed on 2026-07-18: all `88/88` active BiVES
tests, the synthetic CPU smoke, Python compilation, `git diff --check`, and an
active `bives_cxr/` + `configs/bives_cxr/` + `scripts/` legacy-model-path scan.
The experimental stop verdict is therefore an evidence result, not a software
or packaging failure.

E10 closure validation then passed `90/90` active tests after adding the
read-only taxonomy tool and contracts. The synthetic smoke, Python compilation,
diff check, and active old-model-path scan also remain green.

## Route hard stops

- Do not run the old full-objective Run B.
- Do not run Qwen3.5-4B/9B before all 2B expert polarity and intervention gates pass.
- Do not use VinDr test for model selection, threshold selection, loss choice, or K choice.
- Do not call VinDr image-level clustered confidence intervals patient-level.
- Do not add a flat binary classification head; S/C uses bipolar signed evidence.
- Do not treat synthetic I as natural clinical insufficiency or parser U as expert uncertainty.
- Keep all execution local and raw public data outside the repository.

## Expert-route errors

| Error | Attempt | Resolution |
| --- | --- | --- |
| A combined PowerShell inspection command failed before execution because an embedded quoted `rg` pattern left an unterminated string. | First E1 source inspection | No files were read or changed by the failed command. Split the loss, trainer, README, and clipping inspections into separate commands. |
| `python -m unittest tests.test_bives_optimization_diagnostics -v` could not import the test module because `tests/` is not a Python package. | First E1 narrow test | Use the repository-compatible discovery form with `-s tests -p "test_bives_optimization_diagnostics.py"`. |
| A broad `rg` probe included nonexistent `pyproject.toml`, so `rg` returned exit 2 after the pydicom checks had already passed. | E2 dependency inspection | Re-ran only against existing repository paths; no implementation decision used the failed `rg` result. |
| Sorting parser inventory tuples containing both `None` and strings raised `TypeError`. | First E5 inventory summary | Normalize optional tuple fields to strings before sorting; candidate counts were recomputed successfully. |
| The first real weak-S/C build exceeded 120 seconds while hashing selected JPGs concurrently with the full VinDr disk audit. | E5 materialization | The process was terminated without artifacts. Rebuilt with image existence checks and `verify_images=false`; E6 is the single stage responsible for computing/binding every image SHA. |
| A cache progress monitor read the JSON while the writer was replacing its contents and briefly observed an incomplete document. | E6 progress monitoring | The cache payloads were unaffected. Progress writes now use an atomic temporary-file replace, and the completed cache passed a full item/index/hash audit. |
| The first compound B2 background-launch command was rejected by the local command safety policy before execution. | E7 B2 launch | Reissued a minimal launch command without cleanup operations; B2 started as local PID `42296` on GPU0. |
| The first full E1-E7 regression reached 82 passing tests but the active-config guard raised `KeyError: model` on the new cache-only B1/B2 YAMLs. | E7 regression | Added explicit Qwen3.5-2B model provenance and local-diagnostic experiment metadata to both cache-only configs; the trainer continues to consume the already-frozen audited cache. |
| The first expert S/C launch failed on its first sample because the evaluator referenced nonexistent `DicomPreprocessRecord.output_sha256`. | E8 expert inference | No prediction/progress artifact was written and GPU0 was released. Corrected the provenance field to the defined `rgb_sha256`, recompiled, and restarted from zero. |
| Two attempts to restart the VinDr decode audit with worker arguments failed before execution because `Start-Process` split the public-dataset path at spaces. | E4 parallel decode resume | No audit record was changed by either failed launch. Relaunched with a correctly quoted argument string; the resumed four-worker audit completed and the final audit passed. |
| The first batched expert evaluator saved only when the cumulative count happened to be divisible by 10, so batch size 32 produced sparse 160-unit checkpoints. | E8 expert inference resume | Preserved the valid 410-unit checkpoint, stopped only the owned watcher/child, changed progress to atomic save after every batch, and resumed from 410 with the same result identity. |
| Packed multi-image Qwen3.5 eager attention produced batch-dependent patch tokens and scores; one reconstructed sample differed by `0.0213` in support probability and one exact-K patch. | E8 expert/intervention validity | Located the upstream non-Flash rejoin on the head dimension, added a repository adapter guard that calls the official vision tower once per image, and proved on the real failing batch that patch/token/score/gate differences are exactly zero. Archived the packed expert/intervention outputs as invalid and restarted expert inference from zero. |
| A first E10 evidence inventory recursively listed hundreds of mask files and then requested nonexistent `metrics.json`. | E10 read-only diagnosis | No artifact was modified. Restrict subsequent reads to `metrics_final.json`, `intervention_rows.jsonl`, and explicit mask samples; do not use a broad recursive listing. |
| The first E10 planning writeback patch contained an empty `findings.md` hunk and was rejected by `apply_patch`. | E10 bookkeeping | No file changed. Split the writeback into a valid task-plan patch followed by anchored findings/progress append operations. |

# 2026-07-18 Candidate Selector/Intervention Rescue Authority

## Scope

Prepare a claim-driven, local-only candidate authority after the failed E8/E10
gate. This phase is planning and read-only data feasibility only. It does not
authorize a model load, training, a VinDr-test rerun, CheXlocalize download,
method mutation, extra seed, Qwen3.5-4B/9B, or server work.

| Phase | Status | Gate |
| --- | --- | --- |
| R0 freeze prior evidence | complete | E8 remains failed; E10 is descriptive and VinDr test is prohibited as a tuning surface. |
| R1 local development-data feasibility | complete_locked_image_disjoint_only | R001 locked 1,510 balanced S/C rows over 1,446 actually SHA-verified train DICOMs. Design/confirm image overlap is zero and all 16 finding/consensus/area strata are balanced within one sample. All DICOMs omit patient/study/series IDs, so no patient-level claim is allowed. R002 later failed its geometry gate. |
| R2 claim and anti-claim freeze | complete | Primary claim: target regions must beat topology-matched controls under distribution-preserving operators. Supporting claim: sparse localization must be consistent across development strata. Anti-claim: gains are generic deletion-area/topology sensitivity or model scale. Protocol repair precedes any model repair. |
| R3 compact experiment blocks | complete | `refine-logs/EXPERIMENT_PLAN.md` freezes five sequential blocks: development lock, topology control, operator robustness, selector audit/conditional single rescue, and one-time confirmation plus independent final boundary. Each block changes one mechanism and has an explicit stop rule. |
| R4 tracker and artifact manifest | complete | Timestamped/fixed plan and tracker files are byte-identical and registered in `refine-logs/MANIFEST.md` with SHA-256. Every run row is blocked pending review or an earlier dependency. |
| R5 review gate | complete_accepted_by_user | User replied `继续` after the draft handoff, explicitly authorizing execution under the frozen candidate scope. R001 starts first; downstream rows remain dependency-gated. |

## Rescue execution status

| Run | Status | Evidence |
| --- | --- | --- |
| R001 VinDr-train lock | complete_pass | Final replay lock SHA-256 `4251027b3069b21fb6fb5acd6bc02bf003206fbcfffb6d045abd2289ea2ac409`; manifest SHA-256 `bd84cd7ca5384afbcb6228c49331b028a9641dd3dd9011157c2cec75b1f6514f`, 1,510 rows, 1,446 unique actually verified train images, zero split overlap, exact S/C balance. |
| R002 topology geometry | complete_fail_hard_stop | Final replay lock SHA-256 `ce863abfa7a70db16aa5055f5e2038e9a0ae15b97cab4f8b84cf4370715534e6`; complete legal translation search reaches 89.39% overall and 88.89% for pleural effusion, below 90%; consolidation passes at 91.94%. Geometry rows SHA-256 `45115b0a8c3478f983b2408747cafb1afff7da05de67a9ca09cb7e79739eb9ee`. |
| R003/R004 frozen-model topology gate | not_run_R002_fail | Stop rule triggered before any Qwen3.5 load or GPU timing smoke. |

## Rescue-planning errors

| Error | Attempt | Resolution |
| --- | --- | --- |
| `experiment-plan` referenced three `skills/shared-references/output-*.md` files that are absent from the installed skill roots. | First output-protocol read | No project file changed. Follow the protocol stated in the skill body: timestamped artifact, fixed-name copy, and `refine-logs/MANIFEST.md`; record hashes after writing. |
| A sequential header-only scan of all 15,000 VinDr train DICOMs exceeded 184 seconds before emitting final grouping statistics. | R1 patient/study grouping audit | The process was terminated by timeout with no artifact. Do not repeat sequentially; use a bounded concurrent header reader with progress output, and fail R1 closed if it cannot establish a grouping key. |
| A combined R001 source-inventory command returned exit code 1 after printing the requested source because its final optional inventory probe had no matching output. | R001 implementation inventory | No file changed and the required source was present in stdout. Split later schema/geometry inspections into explicit commands whose empty optional results do not fail the whole inspection. |
| R002 geometry attempt 1 reached only `278/377 = 73.74%` feasibility; consolidation passed `56/62 = 90.32%`, but pleural effusion failed `222/315 = 70.48%`. | Conservative topology-control search | The implementation imposed an extra bounding-box-disjoint constraint not required by the accepted plan. Keep the failed artifact as diagnostic evidence, do not unlock R003, and test exact translated-mask disjointness with overlapping bounding boxes before deciding the gate. |
| The first R001 lock hashed `rescue_protocol.py` before R002 replaced the conservative geometry search with the complete legal translation search. | Final provenance audit | The split result is unaffected, but the lock's code hash no longer matches the final module. Archive the ignored pre-replay artifacts, rerun R001 with final code, then rerun R002 so the data-lock and geometry-lock chain binds the committed implementation. |
| A combined final-provenance writeback patch used a nonmatching sentence anchor in the execution log and was rejected atomically. | Final record update | No file changed. Replaced it with a smaller patch anchored on exact manifest/geometry hash lines, then updated the execution-log hash in the manifest. |
| `python -m unittest tests.test_bives_rescue_protocol -v` could not import the test because this repository's `tests/` directory is not a Python package. | C1 first narrow test invocation | Compilation had passed and no test body ran. Do not repeat module import; use the repository's `unittest discover -s tests -p "test_bives_rescue_protocol.py" -v` convention. |
| The first C2 process probe saw a PID file but no `Get-Process` row or log output and was initially interpreted as an immediate exit; a second foreground launch then failed before execution because the first process held the log files. | C2 full-audit launch monitoring | CIM inspection proved the original PID `39836` and eight workers were active; stdout then advanced to 20/377 with empty stderr. The second command never opened an audit process. Keep and monitor only PID `39836`; do not relaunch. |
| The first C3 warmup stopped before emitting any score because deterministic cuBLAS matmul requires `CUBLAS_WORKSPACE_CONFIG` to be set before CUDA initialization. | C3 first local timing/replay launch | No C3 row or lock artifact was produced. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` at module import before `torch`, preserve deterministic algorithms, and rerun the unchanged 16-image gate. |
| The first C5 process reached 307/756 rows and then Windows denied the atomic `progress.json.tmp -> progress.json` replace while the progress file was being monitored. | C5 one-time confirmation execution | No code, data, operator, metric, threshold, or result rule changed. Resumed the exact committed identity from its 307 completed-row checkpoint, stopped reading progress JSON during execution, and completed the single opening. The resume stderr was empty; the incident and corrected end-to-end compute accounting are frozen in the C5 execution log. |
| A recursive filename scan over the entire `H:\Xiyao_Wang\000_Public Dataset` tree exceeded 120 seconds before candidate matching completed. | C6 first local data inventory | The command was read-only and changed nothing. It established only the top-level dataset names. Do not repeat a root-wide recursive scan; inspect CheXpert, MIMIC, NIH, and ambiguous top-level folders with bounded shallow queries and dataset-specific metadata checks. |
| A combined PowerShell candidate-directory check placed a pipeline directly after a `foreach` block and failed with `An empty pipe element is not allowed`. | C6B resume preflight | The entire combined read-only call returned no usable result and changed no file. Replaced it with an explicit `$rows` array followed by a separate pipeline; the corrected check confirmed that neither CheXlocalize nor MS-CXR is present. |
| The first C6B code-style inspection requested nonexistent `scripts/audit_bives_vindr_integrity.py`. | C6B implementation orientation | No file changed. The handoff index names the actual script `scripts/audit_vindr_cxr_integrity.py`; use that exact path and do not repeat the guessed filename. |
| A second parallel style inspection still returned nonzero because at least one requested comparison path was absent, although the package inventory and hash references were returned. | C6B implementation orientation | No file changed. Stop issuing bundled guessed paths; derive exact filenames with `rg --files` and then read only confirmed files. |

# 2026-07-18 Coordinate-Zone Connected-Control Candidate

## Scope

Create a separately review-gated control-family authority after the immutable
R002 exact-shape translation failure. This phase is planning only. It does not
authorize implementation, data mutation, model loading, GPU work, confirmation
access, VinDr-test reuse, training, or Qwen3.5-4B/9B.

| Phase | Status | Gate |
| --- | --- | --- |
| C0.1 preserve stopped provenance | complete | R001 remains pass; R002 remains hard-stop fail; R003-R010 remain not run under the predecessor. |
| C0.2 freeze new control definition | complete_draft | Exact area, content containment, target disjointness, one 4-connected component, and same frozen vertical/horizontal coordinate-zone centroid. Exact target topology is explicitly not matched. |
| C0.3 freeze survival gates | complete_draft | Geometry requires >=95% overall/per finding and >=90% every finding-area quartile before any model load. Mechanism requires positive per-finding TCIG under local-mean and blur with prespecified CI/stratum gates. |
| C0.4 register fixed artifacts | complete | Timestamped/fixed plan and tracker pairs are byte-identical and registered in `refine-logs/MANIFEST.md`. |
| C0.5 review gate | complete_accepted_by_user | User replied `继续` after the draft handoff. C001 is complete-pass and C1 synthetic implementation is authorized. |
| C1 connected-control contracts | complete_pass | New API satisfies exact-area/disjoint/single-connected/zone/replay/fail-closed contracts; 98/98 active tests and synthetic smoke passed. Old translation contracts remain green. |
| C2 score-free geometry audit | complete_pass | 375/377 overall; every per-finding and finding-area gate passes; 0 invariant failures; full replay rows byte-identical at SHA-256 `b94b77bc...e039d9`. |
| C3 local timing/replay | complete_pass | 16 unique 8+8 protocol-design images; replay max diff 0, exact-K mismatches 0, estimated C4 0.2461 h against 4 h cap. |
| C4 connected-control mechanism gate | complete_pass | All 375 feasible protocol-design positives scored; both co-primary operators pass every finding-level mean/CI/high-area/positive-fraction gate; replay max diff 0. |
| C5 one-time internal confirmation | complete_fail_final_stop | Geometry passed 377/378 and the complete C4 mechanism gate reproduced, but consolidation B2 AUPRC `0.89381` fell below frozen B0 `0.91174`; no post-outcome changes, reruns, or 4B/9B scale-up. |
| C6 independent final-data authority | candidate_intake_passed_pending_new_research_authority | The original bounded local inventory had no eligible candidate, but the subsequently acquired MS-CXR official test now passes strict package/license/schema/image/zero-overlap intake. This does not reopen C5 or authorize model evaluation; a separately reviewed post-C5 authority is still required. |
| C6A official acquisition feasibility | complete_user_access_confirmed | The user explicitly confirmed approved credentialed access, required CITI training, and signed DUA. The acquired MS-CXR v1.1.0 ZIP matches its attested package hash; CheXlocalize validation remains excluded and no model action is authorized. |
| C6B local metadata-only intake tooling | complete_tooling_waiting_data | Added the fail-closed CheXlocalize test-only intake, hashed CheXpert-validation access registry, and seven contract tests. The ignored registry binds 234 images/200 patients/200 studies without raw IDs; 116/116 active tests and CPU smoke pass. Real intake remains blocked because the package is absent. |
| C6C MS-CXR official-test intake tooling | complete_strict_intake_no_model_authority | User access attestation and package hash binding pass. The strict ignored artifact locks 15/14 pairs, 25/20 boxes, 29 patients/studies/images, exact MIMIC paths, all image hashes, and zero prior patient/study overlap. It deliberately records `model_evaluation_authorized=false`; post-C5 research authority remains a separate gate. |

## Candidate artifact hashes

- Current plan: `647bf9f466d76553d3ba9a849c73f852227577f9214601776e0b31addd1fb12a`.
- Current tracker: `66688af2c3f009d3078412efdb44f834205e8edda04c9e9a852ef3d4ec3435b5`.
- C1/C2 execution log: `7e1e9c317d1560a5d369fefa034253cd0732f78eb4f82c0f111ff32bc47096c8`.
- C6A official acquisition plan: `4fd9234f2ba68a6535a5ef410790c2ec6204e1a48f8439b6b845a0cdb414bff3`.
- C6B metadata intake tooling log: `07c2fdaffd7874d4043ea204f633650b7b4feae61a0a41d098d8dacad9ffeffc`.
- C6C MS-CXR intake tooling log: `12cdc09dd9517ffbb4207b42f8e3eda2fcb23e9792d40f74a8bd469761182d6a`.
- C6D real-package structure preflight log: `f252ab3d9f46d6d88d5828fb4995c9b3b250a5472a2111906ce4fad684f2fb5d`.
- C6E strict intake log: `0cf2813122b2621908ec193c9b651a8619fcef11c72a4334775f7851afe5d360`.

## C6F independent MS-CXR post-C5 evaluation

| Phase | Status | Gate |
| --- | --- | --- |
| C6F.1 independent authority | complete_authorized_preopen | User explicitly authorized a separate `model_evaluation_authorized=true` record. Frozen C6E intake remains unchanged and false because it is not a model protocol. |
| C6F.2 patient manifest and lock | complete_fail_geometry | Ignored manifest passes 29 patients/studies/images, 15/14 rows, 25/20 boxes, and frozen release/overlap identities. Geometry is 28/29; one hashed Consolidation row has no legal connected control, so the dataset lock is `fail_geometry` and closed. |
| C6F.3 JPG evaluator and tests | complete_fail_closed | Implemented the local-only Qwen3.5-2B JPG evaluator using unchanged C4/C5 operators/gates. Its preflight was exercised and correctly rejected the failed geometry lock before model/GPU access. |
| C6F.4 pre-open commit | complete_ready_for_git_handoff | C6F tests 4/4, full active suite 133/133, CPU smoke, py_compile, diff check, and ignored-artifact boundary pass. Commit/push the authority, implementation, and truthful no-run handoff; do not start a model run. |
| C6F.5 one-time local evaluation | not_run_preopen_geometry_fail | The authorized 2B opening was never created. No JPG decode, model/checkpoint load, GPU, or score occurred; 4B/9B remain blocked. |

### C6F implementation error log

| Error | Attempt | Resolution |
| --- | --- | --- |
| The first bundled identity read guessed a nonexistent file named `C5_CONNECTED_CONTROL_CONFIRMATION_EXECUTION_LOG_20260718.md`. | C6F source orientation | The read-only command changed nothing. `rg --files refine-logs` located the actual authority record `CONNECTED_CONTROL_C5_EXECUTION_LOG_20260718.md`; all C5 identity facts were reread from that exact path. |
| The first real C6F score-free geometry build stopped on the first row whose expert-box mask had no exact-area connected control in the same coordinate zone. | C6F pre-open geometry | No model, image decode, GPU, or score was opened. Preserve the frozen control rule; changed only the audit reporting so all 29 rows are checked in parallel and a complete pass/fail geometry lock is written before the process exits nonzero. |
| The first full 133-test regression rejected the new config's descriptive mode name `local_one_time_evaluation`, which was outside the repository's frozen mode enum. | C6F full regression | No runtime behavior was exercised by the failed assertion. Changed only the config metadata to the existing legal `local_diagnostic` value and rebuilt the dataset lock so its config binding remains exact. |
| The second full regression correctly treated every `configs/bives_cxr/*.yaml` file as a training config and rejected the evaluation-only protocol because it intentionally has no training loss/sampler fields. | C6F config placement | Do not fabricate irrelevant training fields. Moved the frozen evaluation YAML to `refine-logs/C6F_MS_CXR_QWEN35_2B_EVAL_CONFIG_20260718.yaml`, updated both entrypoints, and rebuilt the exact data/config lock. |

## C6G MS-CXR score-free geometry protocol

| Step | Status | Evidence / boundary |
| --- | --- | --- |
| C6G.1 freeze C6F | complete | C6F remains `FAIL_PREOPEN_GEOMETRY_NO_MODEL_ACCESS`; its authority, config, log, manifest, geometry rows, geometry lock, and dataset lock hashes are frozen and may not be rewritten. |
| C6G.2 geometry-only authority | complete_authorized | `BiVES_C6G_MS_CXR_geometry_protocol_plan.md` and `refine-logs/C6G_MS_CXR_GEOMETRY_ONLY_AUTHORITY_20260718.md` authorize CPU geometry only. Model, GPU, JPG decode, score access, and opening markers remain forbidden. |
| C6G.3 frozen thresholds | complete | Derived only from 375 accepted C4 plus 377 accepted C5 controls: `max_location_distance=0.30062962749991123`, `max_log_perimeter_ratio=0.9737778227918367`. |
| C6G.4 candidate certificates | complete | All 29 rows record target/valid geometry, seed/candidate counts, nearest and selected candidates, threshold decisions, and no-model authorization flags. |
| C6G.5 v2 generator/tests | complete | Uniform exact-area, disjoint, within-content, one-component deterministic v2 implemented. New contracts 4/4 and full active suite 137/137 pass; CPU smoke passes. |
| C6G.6 29-row geometry build | complete_pass_geometry_only | Final committed-identity build passes 29/29 with zero exclusions/invariant failures. Rows/certificates/masks replay byte-identically; lock canonical SHA-256 is `6271ba51e8442baad92126473513b0b901619403a4e22c353e455395ec801752`. |
| C6H model evaluation | not_authorized | C6G cannot open Qwen or reuse the C6F model authority. A separate explicit C6H authority is required after a C6G pass. |

### C6G implementation error log

| Error | Attempt | Resolution |
| --- | --- | --- |
| The first formal command used a 10-second shell timeout, so the wrapper lost control while the worker pool continued in the background. | First C6G build launch | Did not treat that uncontrolled build as final. It later completed with byte-identical rows, certificates, and masks, providing a third deterministic replay; the controlled diagnostic and final runs used fresh ignored output directories with unchanged rules. |
| The first complete 29/29 lock bound authority, plan, thresholds, source manifest, candidate certificates, and frozen C6F hashes, but omitted the C6G module/entrypoint hashes and pre-open Git commit. | Post-build provenance audit | Treat the first complete build as a score-free diagnostic. Add identity fields only, commit the unchanged geometry implementation, then rebuild the final C6G lock. |

## C6H one-time local Qwen3.5-2B evaluation

| Step | Status | Evidence / boundary |
| --- | --- | --- |
| C6H.1 independent authority | complete_authorized | On 2026-07-18 the user explicitly approved C6H and instructed execution after preparation. Local-only Qwen3.5-2B; no training, tuning, 4B/9B, server, or C6F/C6G rewrite. |
| C6H.2 immutable input/config lock | implementation_complete_pending_clean_commit | Authority/config and fail-closed lock builder bind the final C6G lock, 29 masks/rows, C6E/C6F identities, Qwen3.5-2B snapshot, B2 step-450 checkpoint, canonical statements, operators, thresholds, and clean source commit. Real dry rehearsal passes. |
| C6H.3 evaluator/preflight/tests | complete_preopen | New C6H entrypoint/output namespace imports frozen C6F scoring/summary helpers read-only and consumes C6G v2 masks. New tests 4/4, full suite 141/141, CPU smoke, py_compile, and diff check pass. |
| C6H.4 pre-open commit | ready_for_commit | Commit/push the exact authority/config/implementation/tests and planning record. Generate the ignored lock only from that clean committed identity. |
| C6H.5 one-time local evaluation | complete_fail_pre_score_no_result | Pre-open commit/lock and GPU1 opening passed, but the evaluator stopped before the first forward/score because all 29 bound JPGs are 224x224 while C6G masks use declared native-resolution letterbox geometry. No progress/rows/metrics exist; GPU1 is idle. |
| C6I input-space geometry recovery | not_authorized | Requires a new score-free authority to regenerate all 29 masks in actual 224x224 input space, followed by a separately authorized replacement one-time model opening. Must not rewrite C6F/C6G/C6H. |

## C6I actual-input-space geometry recovery and replacement opening

| Step | Status | Evidence / boundary |
| --- | --- | --- |
| C6I.1 independent recovery authority | complete_authorized | On 2026-07-18 the user explicitly confirmed approved PhysioNet/CITI/DUA access and authorized the C6I actual-input-space geometry rebuild plus one replacement local Qwen3.5-2B opening after a 29/29 score-free pass. C6F/C6G/C6H remain immutable. |
| C6I.2 uniform pixel-coordinate transform | complete_implementation | For every row, independently scale released MS-CXR x coordinates by `actual_width/native_columns` and y coordinates by `actual_height/native_rows`, rasterize on the hash-bound actual JPG, then map that actual-image mask into the frozen 448x448 Qwen input canvas. No row-specific repair or score access is allowed. |
| C6I.3 score-free 29-row geometry gate | complete_pass | Formal and independent replay builds both pass 29/29. Rows/certificates/lock are byte-identical, 29/29 mask hashes match, actual sizes are uniformly 224x224, and lock canonical SHA-256 is `f6e6c8e6...31a0e`. No model/GPU/score access occurred. |
| C6I.4 implementation/tests/pre-open identity | complete | New C6I authority/config/module/entrypoints/tests only; frozen predecessor files remain byte-unchanged. New contracts pass 4/4, the full active suite 145/145, CPU smoke, py_compile, and diff check pass. Score-free evidence was committed before the identity-bound replacement lock/opening. |
| C6I.5 replacement one-time local evaluation | complete_fail_final_stop | The identity-bound 29/29 local Qwen3.5-2B evaluation completed normally. Pleural-effusion blur passes, but consolidation has no operator with positive CI lower bound and several mean/high-quartile/fraction gates fail. Overall survival `pass=false`; no rerun, tuning, C5 reopening, or 4B/9B scale-up. |

### C6H implementation error log

| Error | Attempt | Resolution |
| --- | --- | --- |
| Initial source inspection guessed nonexistent `scripts/evaluate_bives_ms_cxr_postc5.py`. | C6H source orientation | No model or file mutation occurred. Repository inventory identified the actual frozen evaluator as `scripts/evaluate_bives_c6_ms_cxr.py`; continue from that exact path. |
| A combined helper-import plus Qwen3.5-2B snapshot rehash exceeded the 30-second command window. | C6H identity preflight | No model was loaded and no artifact changed. Keep the frozen snapshot identity and rerun hashing alone with a larger timeout during C6H preflight. |
| The first real score-free C6H lock dry-run rejected the C6G canonical identity because it assumed a single-stage canonical hash. | C6H real 29-row lock rehearsal | C6G intentionally records a geometry-summary canonical identity before extending the final lock and hashing that extended payload. Reproduce and verify this two-stage frozen construction exactly; do not rewrite C6G. |
| The authorized opening loaded Qwen3.5-2B but stopped before the first forward/score because the bound JPG is 224x224 while the MS-CXR annotation metadata and C6G masks use 3056x2544 native geometry. | C6H one-time local evaluation | Zero progress rows and no metrics were written; GPU1 was released. Do not bypass the check or apply C6G masks to misaligned pixels. Freeze the pre-score failure, search locally for coordinate-matched full-resolution JPGs, and require a separately bound recovery identity before any new model load. |
| `python -m unittest tests.test_bives_c6i_input_geometry -v` could not import the new test because this repository's `tests/` directory is not a Python package. | C6I first contract-test invocation | No code/model/data action occurred. Use the repository's documented discovery form `python -m unittest discover -s tests -p "test_bives_c6i_input_geometry.py" -v`. |
| The first discovered C6I suite compared the full-boundary transformed coordinate with exact float equality and observed `223.99999999999997` rather than `224.0`. | C6I coordinate-transform unit test | The rasterized geometry is unchanged and the difference is IEEE-754 representation only. Replace exact object equality with per-coordinate `assertAlmostEqual`; do not round or alter the protocol coordinates. |
| A read-only C6I summary helper guessed `control_audit.selected`, but the frozen v2 certificate names the field `selected_candidate`. | C6I post-geometry summary | The formal build/replay and artifact hashes were already complete and unaffected. Inspect one certificate schema, then use `selected_candidate` for the compact execution-log summary. |
| The first read-only C6I result summary guessed a nested `patient_bootstrap_95ci` operator key, while the frozen evaluator stores each interval under `per_finding.<finding>.bootstrap_95ci`. | C6I result audit | The terminal metrics file was complete and unchanged. Inspect its schema and rerun only the read-only summary using the existing per-finding key; do not rerun scoring. |

## 2026-07-19 BiVES B2 terminal read-only audit

**Authority:** The reviewed C6I outcome is a valid terminal negative result. This
stage may read frozen C4/C5/C6I artifacts and decode the already bound C6I images
for image-space measurements, but it may not load a model, score a row, create a
C6J identity, tune an operator/control/threshold, reopen C5/C6I, train, or scale
to Qwen3.5-4B/9B.

| Step | Status | Evidence / boundary |
| --- | --- | --- |
| T1 freeze terminal identity | complete | Verified current Git/frozen rows/metrics and recorded C6I as the permanent end of the B2 rescue route. |
| T2 target-vs-control decomposition | complete | Produced 812 paired effect cells and a fixed threshold-free four-way taxonomy from existing frozen rows only. |
| T3 descriptive associations | complete | Reported Pearson/Spearman descriptions for frozen geometry/localization features without p-values or causal claims. |
| T4 image-space operator audit | complete | Reapplied the two frozen operators to 29 bound images/masks and reported 58 target/control L1/RMS/SSIM/edge/contrast rows; no model or score access. |
| T5 stage synthesis and proposal pivot | complete | Froze the C4/C5/C6I stage table and revised the active claim to a localization-causality audit; CheXlocalize remains a future separately authorized protocol. |
| T6 validation and Git handoff | complete | Audit replay is byte-deterministic; narrow tests 4/4, full active suite 149/149, CPU smoke, py_compile, and diff check pass. Commit/push only tracked code/docs/tests. |

### Terminal-audit implementation error log

| Error | Attempt | Resolution |
| --- | --- | --- |
| The first taxonomy contract classified `target_effect=0, control_effect<0` as `target_dominant_positive`. | First narrow unit test | Tighten the positive-target branch to require `target_effect > 0`; zero target effect now falls into the nonpositive/tied category. No frozen input or experiment artifact was changed. |
