# Repository Handoff Index

## Active strict VIVID/SPD route

| Purpose | File |
| --- | --- |
| Route overview | `../extensions/vivid_spd_clean/README.md` |
| Frozen scientific protocol | `../extensions/vivid_spd_clean/audit/VIVID_SPD_CLEAN_EXPERIMENT_PROTOCOL.md` |
| Machine lock | `../extensions/vivid_spd_clean/audit/vivid_spd_clean_lock.json` |
| Active task plan | `../extensions/vivid_spd_clean/task_plan.md` |
| Durable findings | `../extensions/vivid_spd_clean/findings.md` |
| Chronological progress | `../extensions/vivid_spd_clean/progress.md` |
| Experiment tracker | `../extensions/vivid_spd_clean/refine-logs/EXPERIMENT_TRACKER.md` |

The BiVES material below is retained history, not the active method on
`codex/vivid-spd-clean-extension`.

## Historical BiVES-CXR handoff

BiVES-CXR was the prior paper and code mainline.

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
| Connected-control C4 mechanism gate | `../scripts/evaluate_bives_connected_control_mechanism.py`, `../refine-logs/CONNECTED_CONTROL_C4_EXECUTION_LOG_20260718.md` |
| Connected-control C5 one-time confirmation and final stop | `../scripts/evaluate_bives_connected_control_confirmation.py`, `../refine-logs/CONNECTED_CONTROL_C5_EXECUTION_LOG_20260718.md` |
| C6 CheXlocalize metadata-only intake | `../scripts/audit_bives_c6_chexlocalize.py`, `../bives_cxr/c6_intake.py`, `../refine-logs/C6A_OFFICIAL_DATA_ACQUISITION_PLAN_20260718.md` |
| C6 MS-CXR official-test metadata-only intake | `bives_cxr_c6_ms_cxr_intake.md`, `../scripts/audit_bives_c6_ms_cxr.py`, `../bives_cxr/c6_ms_cxr.py`, `../refine-logs/C6C_MS_CXR_INTAKE_TOOLING_LOG_20260718.md`, `../refine-logs/C6D_MS_CXR_REAL_PACKAGE_PREFLIGHT_LOG_20260718.md`, `../refine-logs/C6E_MS_CXR_STRICT_INTAKE_LOG_20260718.md` |
| C6F MS-CXR post-C5 evaluation | `../refine-logs/C6F_MS_CXR_POST_C5_EVALUATION_AUTHORITY_20260718.md`, `../refine-logs/C6F_MS_CXR_QWEN35_2B_EVAL_CONFIG_20260718.yaml`, `../refine-logs/C6F_MS_CXR_PREOPEN_GEOMETRY_EXECUTION_LOG_20260718.md`, `../scripts/prepare_bives_c6_ms_cxr_evaluation.py`, `../scripts/evaluate_bives_c6_ms_cxr.py`, `../bives_cxr/c6_ms_cxr_eval.py` |
| B2 terminal read-only audit | `../scripts/audit_bives_b2_terminal.py`, `../bives_cxr/terminal_audit.py`, `../refine-logs/BIVES_B2_TERMINAL_READ_ONLY_AUDIT_RESULT_20260719.md` |
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

## C6G MS-CXR geometry-only status

C6F remains frozen as `FAIL_PREOPEN_GEOMETRY_NO_MODEL_ACCESS`. The separate
C6G authority changes only the geometry question: it uses frozen continuous
centroid/perimeter limits from 752 accepted C4/C5 controls and a uniform
deterministic connected-control v2 search. The final local CPU build passes
29/29 MS-CXR rows with zero exclusions or invariant failures. Its ignored lock
canonical SHA-256 is
`6271ba51e8442baad92126473513b0b901619403a4e22c353e455395ec801752`.
This C6G artifact is a geometry lock only: no JPG was decoded, no
Qwen/checkpoint or GPU was opened, no score was produced, and it does not by
itself authorize C6H. The separately authorized next phase is recorded below.

## C6H one-time local evaluation status

The user has separately authorized one C6H Qwen3.5-2B positive-only mechanism
evaluation after the C6G 29/29 pass. C6H has its own authority, config, lock,
entrypoint, output directory, and terminal stop rule. It freezes the same B2
step-450 exact-K=16 checkpoint, statements, operators, bootstrap, and survival
thresholds while replacing only the failed C6F v1 controls with the frozen C6G
v2 controls. It is not classification or clinical validation, and it does not
authorize training, tuning, reruns, or Qwen3.5-4B/9B.

The authorized opening subsequently failed before the first vision forward or
score: all 29 bound JPGs are 224x224, while C6G masks were built in the
MS-CXR-declared native-resolution letterbox space. No progress, evaluation row,
or metric exists, and GPU1 was released. C6H is frozen as
`FAIL_PRE_SCORE_PIXEL_ALIGNMENT_NO_RESULT`; applying the masks despite the
mismatch or relaunching under the same identity is forbidden.

## C6I actual-input recovery status

The user separately authorized C6I after confirming the applicable
PhysioNet/CITI/DUA access. C6I does not edit or rerun C6H. It maps every
released MS-CXR box from declared native coordinates into the exact hash-bound
224x224 JPG coordinates using independent x/y scale factors, then applies the
existing deterministic 448x448 Qwen input transform. All 29 rows must pass a
new score-free exact-area, target-disjoint, within-content, one-connected
control gate before the new Qwen3.5-2B replacement opening can exist. The
replacement remains positive-only and nonformal; training, tuning,
classification claims, server execution, and Qwen3.5-4B/9B are forbidden.

The score-free C6I build and independent replay now both pass 29/29. Their
geometry rows, candidate certificates, geometry lock, and all 29 mask files
are byte-identical; canonical geometry-lock SHA-256 is
`f6e6c8e6a4e7499376d8b316d588197fb1e57ae18a68b6c529dd31e60e531a0e`.
No model, GPU, or score was accessed while producing this lock.

The subsequently authorized one-time C6I replacement evaluation completed all
29 rows on local Qwen3.5-2B and terminated as `fail_final_stop`. Masked Gaussian
blur passes every pleural-effusion condition but has negative consolidation
TCIG with a wholly negative 95% CI; local mean also fails the complete frozen
per-finding/high-area gate. Positive localization gain is secondary and does
not rescue the paired intervention claim. C6I is terminal: no rerun, outcome-
driven tuning, C5 reopening, or Qwen3.5-4B/9B scale-up is authorized.

## B2 terminal interpretation

The 2026-07-19 read-only terminal audit freezes C6I as the permanent end of the
B2 rescue route. It decomposes existing C5/C6I target and control effects,
reports a fixed four-way sign/order taxonomy, and reapplies the two frozen
operators only for image-space L1/RMS/SSIM/edge/contrast measurements. It does
not load a model, compute a new score, create C6J, or tune any intervention.

The active manuscript direction is now a localization-causality audit rather
than a claim that BiVES B2 learned a valid causal evidence set. Positive top-K
localization gain did not imply matched target-versus-control necessity on the
independent C6I data. CheXlocalize remains a possible future validation/test
protocol and is not authorized by this terminal audit.
