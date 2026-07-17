# BiVES-CXR Handoff Index

BiVES-CXR is the only active paper and code mainline.

## Start here

| Purpose | File |
| --- | --- |
| Final research proposal | `../BiVES_CXR_MIA_TMI_ready_proposal.md` |
| Support-polarity root cause and decoder repair | `../BiVES_support_polarity_root_cause_and_repair.md` |
| Current no-clinical-review direction | `../BiVES_next_direction_without_local_clinical_review_2026-07-17.md` |
| Expert polarity + intervention execution plan | `../BiVES_995fb81_code_review_and_next_plan.md` |
| Frozen Proxy-P0-A diagnostic record | `bives_cxr_proxy_p0_a_freeze.md` |
| Repository overview | `../README.md` |
| Implementation contract | `bives_cxr_implementation.md` |
| Manifest schema | `bives_cxr_manifest_schema.md` |
| Migration/archive manifest | `bives_cxr_migration_manifest.md` |
| Persistent task plan | `../task_plan.md` |
| Durable findings | `../findings.md` |
| Chronological progress | `../progress.md` |

## Active code

| Surface | Path |
| --- | --- |
| Core package | `../bives_cxr/` |
| Qwen3.5 configs | `../configs/bives_cxr/` |
| Training entry | `../scripts/train_bives_cxr.py` |
| CPU smoke | `../scripts/smoke_bives_cxr.py` |
| Real-weight Qwen3.5 vision smoke | `../scripts/smoke_qwen35_vision.py` |
| Qwen3.5-to-BiVES local CUDA integration gate | `../scripts/smoke_qwen35_bives_integration.py` |
| Zero-training uncertain transform replay | `../scripts/replay_bives_uncertain_transform.py` |
| Direct uncertain selector/evidence replay | `../scripts/replay_bives_uncertain_selector.py` |
| Manifest audit | `../scripts/audit_bives_manifest.py` |
| P0 MIMIC intake index | `../scripts/index_mimic_bives_p0_candidates.py` |
| P0 parser candidate and blinded-review packet | `../scripts/prepare_bives_p0_report_review.py`, `../scripts/validate_bives_p0_review_packet.py` |
| Nonclinical weak-label proxy-P0 builder | `../scripts/build_bives_proxy_p0.py` |
| Frozen-feature proxy finding screen | `../scripts/diagnose_bives_proxy_sc.py` |
| Optimization-identifiability audit | `../bives_cxr/optimization_audit.py` |
| Optimization-identifiability verdict | `bives_cxr_optimization_identifiability_verdict.md` |
| Weak-label proxy-P0 experiment record | `bives_cxr_proxy_p0_experiment_log.md` |
| Joint four-split dataset lock | `../scripts/lock_bives_dataset.py` |
| Source-only deployment manifest | `../scripts/write_bives_source_manifest.py` |
| Statement cache builder | `../scripts/build_bives_statement_embeddings.py` |
| Explicit locked-test evaluator | `../scripts/evaluate_bives_final.py` |
| CPU tests | `../tests/test_bives_core.py`, `../tests/test_bives_readiness.py` |
| VinDr archive/integrity utilities | `../scripts/extract_vindr_cxr.py`, `../scripts/audit_vindr_cxr_integrity.py` |
| VinDr expert S/C intake | `../scripts/prepare_bives_vindr_expert_sc.py`, `bives_cxr_public_expert_sc_intake.md` |
| Deterministic CXR DICOM loader | `../bives_cxr/dicom.py` |
| Independent expert S/C dataset and evaluator | `../bives_cxr/expert_sc.py`, `../scripts/evaluate_bives_expert_sc.py` |
| Explicit weak MIMIC S/C builder | `../scripts/prepare_bives_weak_sc.py` |
| Frozen Qwen3.5 token cache | `../scripts/cache_qwen35_patch_tokens.py` |
| Cached-token B0/B1/B2 polarity route | `../scripts/run_bives_sc_b0_pooled.py`, `../scripts/train_bives_sc_cached.py` |
| Locked VinDr B0-vs-B2 expert polarity evaluation | `../scripts/evaluate_bives_vindr_sc.py` |
| Locked VinDr target/control/evidence-only evaluation | `../scripts/evaluate_bives_vindr_interventions.py`, `../bives_cxr/pixel_interventions.py` |
| Post-stop read-only intervention failure taxonomy | `../scripts/analyze_bives_vindr_intervention_failures.py`, `bives_cxr_post_stop_failure_taxonomy.md` |
| Stopped selector/intervention rescue authority | `../refine-logs/EXPERIMENT_PLAN.md`, `../refine-logs/EXPERIMENT_TRACKER.md`, `../refine-logs/MANIFEST.md` |
| VinDr-train rescue lock and geometry audit | `../scripts/prepare_bives_vindr_rescue_dev.py`, `../scripts/audit_bives_vindr_rescue_geometry.py`, `../refine-logs/R001_R002_EXECUTION_LOG_20260718.md` |
| Draft coordinate-zone connected-control rescue authority | `../refine-logs/CONNECTED_CONTROL_RESCUE_PLAN.md`, `../refine-logs/CONNECTED_CONTROL_RESCUE_TRACKER.md` |
| Connected-control C1/C2 execution record | `../scripts/audit_bives_vindr_connected_control_geometry.py`, `../refine-logs/CONNECTED_CONTROL_C1_C2_EXECUTION_LOG_20260718.md` |
| Connected-control C3 timing/replay gate | `../scripts/audit_bives_connected_control_timing_replay.py`, `../refine-logs/CONNECTED_CONTROL_C3_EXECUTION_LOG_20260718.md` |
| P0 data-source and audit boundary | `bives_cxr_p0_data_readiness.md` |

## Active model boundary

The active family is Qwen3.5 multimodal only:

- 2B for P0/debug;
- 4B for the main model;
- 9B for scale validation;
- 0.8B for optional ultra-light smoke.

No active config may reference Qwen3-VL, Qwen2.5, InternVL, LLaDA, Gemma, or
other prior model families.

## Legacy boundary

`../legacy/vsl_cxr/` preserves the prior VSL-CXR, CEQ, CCSH, AUCH, SAMEQ,
CVCP, case-study, Qwen3-VL, and old external-evaluation implementation.

`../legacy/vivid_med/` preserves pre-BiVES VIVID-Med scripts, models, training,
evaluation, loaders, configs, profiles, prompts, and external tools.

`../legacy/delete_archive/` preserves the small retired root-level cleanup
bundle that previously remained under `delete/`, including old Qwen2.5-era
configs and utility scripts.

`../legacy/bives_cxr/legacy_abs_exp_decoder.py` preserves the retired
absolute-difference decoder only for the documented failure ablation. Active
imports, configs, calibration, and release artifacts cannot select it.

Historical result files under ignored `outputs/` remain untouched. They are
pilot/provenance evidence only and cannot close a BiVES paper gate.

## Current execution status

The repository has an executable BiVES core, strict same-statement group
training, mask-before-contextual-block exact-K interventional closure, full-row
primary evaluation, a separate grouped mechanism evaluator, best-checkpoint
selection, monotone-decoder parameter calibration, a vision-only Qwen3.5 loader,
provenance-complete statement caches, deterministic intervention controls,
an isolated locked-test release entry, synthetic CPU smoke, and proposal-level
unit tests.

## Local-only execution

The active YAMLs now default to this workstation: `data_root: data` and local
Qwen3.5 paths under `H:/Xiyao_Wang/001_models/`. All active experiments,
including formal training, calibration, and final evaluation, run locally;
do not synchronize them to the server or submit SSH/Slurm jobs. The intended
local checks are:

```powershell
python scripts/smoke_bives_cxr.py
python -m unittest discover -s tests -p "test_bives_*.py" -v
python scripts/evaluate_bives_final.py --help
```

`--validate-release-chain-only` on the final-evaluator CLI validates the full
checkpoint/calibration/cache/four-split lock chain without loading a model or
emitting locked-test metrics. The normal evaluator still requires
`--run-locked-test`.

Expert review is unavailable and has been removed from the executable path.
The local proxy P0 is therefore explicitly weak-label and nonformal. The
current parser-v3 expansion over 5,000 paired MIMIC studies completed one
bounded Qwen3.5-2B run; the completed chain was:

1. build deterministic patient-disjoint proxy train/validation quartets with
   `scripts/build_bives_proxy_p0.py`;
2. require rule/report/image hashes and `weak_proxy_unreviewed` provenance;
3. audit all real and synthetic images plus exact S/C/U/I groups;
4. retain `proxy_dataset_lock.json` and `formal_result=false`;
5. screen findings with frozen Qwen3.5-2B visual features before training;
6. run one bounded 50-step Qwen3.5-2B proxy P0 locally on 48 train and 48
   patient-disjoint validation rows;
7. record aggregate/per-finding ranking and absolute four-state decision gates
   separately.

The expanded run passes the ranking gate: held-out S/C AUROC is `0.8056`, U/I
AUROC is `1.0`, and per-finding S/C AUROC is `0.875`, `0.8125`, and `1.0` for
consolidation, pleural effusion, and pulmonary edema. It fails the absolute
decision gate because all 48 validation argmax predictions are insufficient
(`accuracy=0.25`, `macro-F1=0.10`). Keep 4B/9B, calibration, locked-test, and
formal clinical claims closed. An exploratory decoder-parameter fit on the
train proxy improves held-out NLL from `1.3692` to `1.1620` and accuracy to
`0.5417`, but still recovers only 1/12 contradict and 2/12 uncertain rows. This
shows that uncalibrated geometry is material but not the whole failure; it is
not a locked calibration result and does not reopen scaling.

The 50-step run remains frozen as Proxy-P0-A. A patient-group-disjoint fixed
logistic probe on cached Qwen3.5-2B visual features obtains global S/C AUROC
`0.7889`; the three retained findings obtain `0.8550`, `0.8075`, and `0.8000`.
The subsequent 400-step state-only run learned correct absolute S/C polarity
but reached only `0.7917` train accuracy, failing the preregistered `1.0`
train-fit gate. The matched full-objective arm was not run. The weak-label
four-state route is stopped. The active bounded route is expert S/C polarity
plus pixel-level interventional evidence sufficiency. It uses balanced,
patient-disjoint explicit MIMIC S/C only for weak training and keeps VinDr
consensus test labels external to model, threshold, loss, and K selection.
B0/B1/B2 remain Qwen3.5-2B-only until the expert polarity and intervention
gates pass. The frozen-token gate is now complete for seed 17: B0 validation
macro AUROC/AUPRC is `0.7857/0.7992`, B1 dense is `0.7713/0.7910`, and B2
exact-K=16 is `0.8423/0.8240` at selected step 450. B2 is therefore the only
locked candidate that entered the external read-only gate. The corrected
per-image-isolated VinDr evaluation is now complete. B2 improves AUROC for both
findings and improves pleural-effusion AUPRC, but consolidation AUPRC is
`0.2338` versus B0 `0.2628`. Pixel intervention also fails: consolidation TCIG
is `0.0043` with CI crossing zero, while pleural-effusion TCIG is `-0.0390`
with a wholly negative CI. Top-K box overlap is better than random, but it does
not establish causal evidence sufficiency. The route stops at seed 17; no more
seeds or 4B/9B runs are authorized. See
[`bives_cxr_expert_polarity_intervention_verdict.md`](bives_cxr_expert_polarity_intervention_verdict.md).
The read-only post-stop diagnosis is recorded in
[`bives_cxr_post_stop_failure_taxonomy.md`](bives_cxr_post_stop_failure_taxonomy.md).
It attributes the failed causal gate to inconsistent selector localization
plus broad sensitivity to large arbitrary pixel deletions. Because that
diagnosis inspected VinDr test outcomes, it cannot be used to tune and retest
on the same evaluation surface.

A candidate protocol-first rescue is now documented under `refine-logs/`.
It was accepted and executed through its CPU-only survival gates. R001 passed,
but R002 stopped the route: exact same-band translated target-shape controls
are feasible for only 89.39% overall and 88.89% of pleural-effusion design
positives, below the locked 90% gate. R003/R004 were not run; no rescue model
or GPU was loaded. VinDr test remains excluded from rescue selection, and an
independent patient-grouped final evaluation remains unresolved.

## Current mechanism-gate status

Support polarity remains repaired by the monotone decoder. Direct replay of
the real train/validation uncertain pair showed that the remaining failure was
not solely exact-K selection: aligned train-mask replay and all-patch pooling
were also polarity-biased. The local-only uncertain fixture therefore uses an
equal-area spatial support/contradict mixture, with saved positive/negative
region masks, rather than treating posterization as a proxy for balanced
bipolar evidence. The single unchanged 100-step Qwen3.5-2B gate then passed
with selected validation NLL `0.37115`, validation accuracy `1.0`, and
uncertain `|rho|=0.03850`.

This is a non-formal synthetic mechanism result only. Mini-P0 and formal runs
remain behind the separate manifest, statement-cache, and readiness-audit
gates; the decoder, loss weights, K=16 budget, and model capacity were not
changed in this phase.
