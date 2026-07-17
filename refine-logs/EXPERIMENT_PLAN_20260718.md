# BiVES-CXR Selector/Intervention Rescue Experiment Plan

**Status:** DRAFT_REVIEW_REQUIRED
**Date:** 2026-07-18
**Base commit:** 7c3e7ed
**Parent authority:** BiVES_CXR_MIA_TMI_ready_proposal.md
**Failure evidence:** docs/bives_cxr_expert_polarity_intervention_verdict.md and docs/bives_cxr_post_stop_failure_taxonomy.md

## Problem anchor

The frozen Qwen3.5-2B B2 model has strong expert S/C ranking and positive
aggregate localization overlap, but it fails target-vs-control causal
intervention. The existing control is area-matched yet topologically
mismatched: a contiguous expert target is compared with randomly scattered
pixels, and both are zero-filled. Post-stop analysis also shows inconsistent
selector localization and stronger control sensitivity at large areas.

This plan asks whether a protocol-valid intervention and a consistently
localized sparse selector can be established without changing the decoder,
K budget, model family, or scale.

## Authority boundary

This document is a candidate subordinate authority. It does not supersede the
final proposal until explicitly accepted.

- All work is local to this workstation.
- Qwen3.5-2B is the only allowed model during the rescue gate.
- VinDr test is frozen diagnostic history and is prohibited for selection,
  thresholding, operator choice, loss choice, K choice, or same-test rerun.
- VinDr train may be used only as image-disjoint development. Its DICOMs expose
  no PatientID, StudyInstanceUID, or SeriesInstanceUID, so no patient-level
  claim is permitted.
- CheXlocalize is absent and no download is authorized by this draft.
- No Qwen3.5-4B/9B, server, SSH, Slurm, U/I clinical claim, flat head,
  decoder change, K sweep, or multi-variable rescue is permitted.
- Every experimental row remains BLOCKED_PENDING_REVIEW until this candidate
  authority is explicitly accepted.

## Claim map

| Claim | Why it matters | Minimum convincing evidence | Linked blocks |
|---|---|---|---|
| C1: target evidence has intervention-specific effect | Rules out the current area/topology confound | On locked development, expert target beats topology-matched control under two distribution-preserving operators for each finding; the confirmation split is used once after all choices are frozen | B1, B2, B5 |
| C2: sparse evidence localization is consistent, not an aggregate artifact | Rules out a small high-overlap subgroup carrying the result | Positive target-control effect across reader-consensus and lesion-area strata, with no reversal in the low-localization stratum; any model rescue improves this without harming polarity ranking | B3, B4, B5 |
| Anti-claim: the result is generic deletion sensitivity or capacity | Prevents an invalid 4B/9B scale story | Scattered controls are diagnostic only; topology-matched controls and non-black operators must pass before any model change, and 2B must pass before scale | B2, B3, B5 |

## Frozen local data facts

VinDr train contains 15,000 images and exactly three radiologist label rows per
image. Clean development eligibility is:

| Finding | 2-of-3 or 3-of-3 positive with box | 0-of-3 negative | Excluded 1-of-3 disagreement |
|---|---:|---:|---:|
| consolidation | 121 | 14,647 | 232 |
| pleural effusion | 634 | 13,962 | 404 |

All 15,000 DICOM headers were read successfully, but all lack patient/study/
series grouping keys. The development lock must therefore say
image_disjoint_only=true and patient_level_claim=false.

For intervention blocks, only positive images with boxes are eligible.
For polarity monitoring, a deterministic balanced negative sample may be
drawn from the 0-of-3 pool. The 1-of-3 disagreement pool is always excluded.

## Split and lock protocol

Create one deterministic lock before any model load:

- protocol_design: 50% of eligible positives per finding.
- rescue_confirm: remaining 50%, never read during control/operator/model
  design.
- Stratify by finding, 2-of-3 versus 3-of-3 consensus, and expert-box area
  quartile.
- Seed: 20260718.
- Bind annotation CSV hashes, DICOM streaming hashes, image IDs, reader votes,
  boxes, split assignment, preprocessing identity, and code commit.
- No unit may occur in both splits.
- Geometry exclusions must be outcome-independent and frozen before scoring.
- The confirmation split is opened once, only after the protocol and optional
  model rescue are frozen.

## Paper storyline

Main paper must eventually prove:

1. target regions have a stronger causal effect than spatially and
   distributionally matched controls; and
2. sparse statement-conditioned evidence is consistently localized.

Appendix may contain:

- the failed scattered-zero-control protocol;
- E10 failure taxonomy;
- operator sensitivity and geometry exclusions.

Experiments intentionally cut:

- additional teacher/model families;
- Qwen3.5-4B/9B before 2B gates;
- decoder, temperature, K, loss-weight, or learning-rate sweeps;
- U/I clinical claims;
- tuning on VinDr test;
- patient-level claims from VinDr.

## Experiment blocks

### B1: Development lock and geometry contracts

- Claim tested: the rescue has a non-test, leakage-controlled development
  surface and valid matched masks.
- Dataset: VinDr train only.
- Systems: no model.
- Checks:
  - three-reader vote reconstruction;
  - 2-of-3 support and 0-of-3 contradict definitions;
  - complete boxes for every positive;
  - deterministic design/confirm split;
  - exact annotation/image hashes;
  - image overlap equals zero;
  - explicit absence of patient grouping;
  - target/control equal area, disjointness, component count, perimeter,
    compactness, and bounding-box aspect-ratio checks.
- Success criterion: every lock and geometry contract passes with zero missing
  image, vote, box, or hash.
- Failure interpretation: stop; no model load.
- Target: data/protocol section and appendix audit.
- Priority: MUST-RUN.

### B2: Control-topology mechanism gate

- Claim tested: E8 failure is not created by comparing a contiguous target with
  scattered control pixels.
- Model: frozen existing Qwen3.5-2B B2 step-450 checkpoint; no training.
- Split: protocol_design positives only.
- Single changed factor: control topology.
- Compared controls:
  1. scattered exact-area control, retained only as historical diagnostic;
  2. deterministic contiguous translated target-shape control, disjoint from
     the target and matched on area, component topology, perimeter, and
     vertical image band.
- Operator: keep zero-fill fixed for this block.
- Metrics: per-finding target/control effect, TCIG with image bootstrap,
  fraction TCIG > 0, area-quartile TCIG, geometry-exclusion rate, and original
  score replay error.
- Go gate:
  - topology-matched control is feasible for at least 90% of eligible images;
  - mean TCIG is positive for both findings;
  - no finding has negative TCIG in its highest-area quartile;
  - original score replay stays within the frozen tolerance.
- Stop: if topology matching does not improve the control confound, do not
  change the model.
- Target: protocol figure/ablation.
- Priority: MUST-RUN.

### B3: Distribution-preserving operator gate

- Claim tested: target specificity survives operators that are less
  out-of-distribution than black zero-fill.
- Model/control: freeze B2 and the B2 topology-matched control.
- Split: protocol_design positives only.
- One operator changes per subrun; do not select the best after seeing
  rescue_confirm.
- Operators:
  1. deterministic local mean replacement from a fixed exterior ring;
  2. deterministic Gaussian blur restricted to the intervention mask.
- Zero-fill remains a diagnostic reference, not a candidate primary operator.
- Metrics: B2 metrics plus pixel-distribution shift summaries and agreement of
  TCIG direction across operators.
- Go gate:
  - both candidate operators give positive mean TCIG for both findings;
  - at least one candidate has bootstrap lower bound above zero per finding;
  - operator directions agree in every prespecified consensus/area stratum.
- Stop: inconsistent operator direction means intervention-specific evidence
  is not established; no selector rescue.
- Target: main intervention table and operator robustness appendix.
- Priority: MUST-RUN.

### B4: Selector-consistency audit and one allowed rescue

- Claim tested: exact-K evidence is consistently localized once the evaluation
  protocol is valid.
- Entry condition: B2 and B3 both pass.
- First action: frozen-model audit only; no training.
- Strata:
  - finding;
  - 2-of-3 versus 3-of-3 reader consensus;
  - expert-box area quartile;
  - low/high original score;
  - localization-gain quartile.
- Frozen-model go gate:
  - mean TCIG positive in every area quartile for both findings;
  - at least 60% of images have TCIG > 0 per finding;
  - low-localization quartile is not negative;
  - B2 polarity AUROC/AUPRC remains no lower than frozen B0 on the development
    polarity sample.
- Conditional rescue:
  - only if B2/B3 pass but selector consistency fails;
  - add exactly one dense-to-sparse preservation objective
    abs(delta_K - stop_gradient(delta_dense));
  - keep Qwen3.5-2B, exact K=16, decoder, preprocessing, optimizer budget, and
    all existing loss weights fixed;
  - one seed, one fixed step budget, final-step selection only;
  - train on the existing weak MIMIC S/C train lock and select only by its
    frozen validation surface plus protocol_design.
- Stop: no second rescue, no weight/LR/K sweep, and no 4B/9B.
- Target: selector ablation.
- Priority: CONDITIONAL MUST-RUN.

### B5: One-time confirmation and external boundary

- Claim tested: the fully frozen protocol/model direction survives an unseen
  local development split.
- Split: rescue_confirm, opened once.
- Compared systems:
  - frozen B0;
  - original frozen B2;
  - one rescued B2 only if B4 authorized it.
- Primary metrics:
  - per-finding polarity AUROC/AUPRC;
  - topology-matched TCIG and bootstrap CI under both candidate operators;
  - localization gain and area/consensus strata;
  - original-score replay and geometry exclusions.
- Internal confirmation gate:
  - no per-finding polarity metric below B0;
  - TCIG CI lower bound above zero for both findings under both operators;
  - positive TCIG in every area quartile;
  - at least 60% sample-level positive TCIG per finding;
  - no material result-identity or replay error.
- Interpretation:
  - pass means only an internal image-disjoint development success;
  - fail closes the rescue;
  - neither outcome authorizes a patient-level or final external claim.
- Final external requirement: a new patient-identified, expert-region
  evaluation surface must be approved and locked before any paper-ready claim.
- Priority: MUST-RUN once; external final is BLOCKED_DATA.

## Run order and milestones

| Milestone | Goal | Runs | Decision gate | Estimated cost | Main risk |
|---|---|---|---|---|---|
| M0 | Review candidate authority | no experiment | explicit acceptance | 0 GPU h | silent scope expansion |
| M1 | Freeze data and geometry | B1 | all locks/contracts pass | <1 CPU h | no patient grouping |
| M2 | Isolate topology confound | B2 | both findings positive, high-area nonnegative | 1-3 local GPU h after a 16-image timing smoke | translated control infeasible |
| M3 | Validate operators | B3 mean-fill then blur | cross-operator positive direction | 2-6 local GPU h | operator remains OOD |
| M4 | Audit selector | B4 frozen audit | every stratum survives | <1 CPU h after cached results | low-overlap subgroup failure |
| M5 | Optional single rescue | one preservation run only | fixed one-run gate | <=2 local GPU h | weak labels do not improve localization |
| M6 | One-time confirmation | B5 | all locked internal gates pass | 2-4 local GPU h | consolidation sample size |
| M7 | Independent final | not yet runnable | new data authority and patient-level lock | unknown | dataset unavailable |

Before M2, run a 16-image timing smoke and replace estimated time with measured
throughput. Do not reserve or interrupt unrelated GPU processes.

## Compute and data budget

- Mandatory planning/data work: CPU only.
- Mandatory frozen-model mechanism work after approval: capped at 9 local
  GPU-hours before confirmation.
- Conditional single model rescue: capped at 2 local GPU-hours.
- Confirmation: capped at 4 local GPU-hours.
- Maximum pre-external budget: 15 local GPU-hours.
- No server compute, no model download, and no Qwen3.5-4B/9B.
- Biggest bottleneck: absence of a patient-identified independent final set.

## Risks and mitigations

- Risk: topology-matched control is geometrically infeasible for large or
  bilateral boxes.
  - Mitigation: outcome-independent geometry preflight and prespecified
    exclusion ceiling; fail if feasibility is below 90%.
- Risk: local mean/blur still changes clinically relevant context.
  - Mitigation: require agreement across both operators and matched controls.
- Risk: VinDr train has no patient key.
  - Mitigation: label every result image-disjoint development only; prohibit
    patient-level claims.
- Risk: the same development set drives too many decisions.
  - Mitigation: protocol_design for all design work, rescue_confirm opened once.
- Risk: test leakage from E10.
  - Mitigation: VinDr test is prohibited from every rescue runtime and report.
- Risk: a model rescue hides an evaluation flaw.
  - Mitigation: model rescue is conditional on protocol gates B2/B3.

## Stop rules

Stop immediately if any of the following occurs:

1. data/geometry lock fails;
2. topology-matched control does not remove the high-area reversal;
3. candidate operators disagree in direction;
4. protocol-valid frozen B2 still fails both findings;
5. the one allowed preservation rescue fails;
6. confirmation fails any finding;
7. runtime attempts to read VinDr test;
8. an action would require patient-level claims without a patient key.

No failure unlocks 4B/9B, a decoder change, K sweep, additional seed, or a new
module stack.

## Final checklist

- [x] Dominant claim and anti-claim are explicit.
- [x] Protocol repair precedes model repair.
- [x] VinDr test is excluded from tuning.
- [x] Development and one-time confirmation are separated.
- [x] Qwen3.5-2B and exact K=16 stay fixed through the protocol gates.
- [x] One optional single-variable rescue is specified.
- [x] Must-run and conditional runs are separated.
- [x] Local compute cap and stop rules are explicit.
- [ ] Candidate authority explicitly accepted.
- [ ] Independent patient-level final dataset available.
