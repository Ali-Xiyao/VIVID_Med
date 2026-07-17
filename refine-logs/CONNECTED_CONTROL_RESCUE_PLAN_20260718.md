# BiVES-CXR Coordinate-Zone Connected-Control Rescue Plan

**Status:** ACCEPTED_C3_IN_PROGRESS
**Date:** 2026-07-18
**Base commit:** `98edc71`
**Parent authority:** `../BiVES_CXR_MIA_TMI_ready_proposal.md`
**Stopped predecessor:** `EXPERIMENT_PLAN.md` and
`R001_R002_EXECUTION_LOG_20260718.md`

## Decision anchor

R001 remains a valid VinDr-train image-disjoint development lock. R002 remains
a valid hard-stop: exact target-shape translation is feasible for only
`337/377 = 89.39%` of protocol-design positives and for `280/315 = 88.89%`
of pleural-effusion positives, below the preregistered 90% per-finding gate.
This plan does not lower that threshold, delete large lesions, reinterpret the
failure, or reopen R003-R010.

The new candidate changes the control family and therefore asks a different,
narrower question:

> Does a deterministic, exact-area, target-disjoint, connected nuisance region
> from the same coarse image-coordinate zone provide a feasible and less
> topologically out-of-distribution control for target intervention?

The term `coordinate-zone anatomy proxy` is deliberate. It is not a lung,
lobar, or radiologist-confirmed anatomy segmentation and must never be reported
as one.

## Authority boundary

- This subordinate authority was explicitly accepted when the user replied
  `继续` on 2026-07-18. Rows unlock strictly in tracker order; acceptance does
  not bypass any dependency or survival gate.
- All work is local to this workstation. No server, SSH, Slurm, or remote
  experiment action is permitted.
- Qwen3.5-2B is the only model allowed after geometry passes. Qwen3.5-4B/9B
  remain blocked.
- VinDr test remains frozen diagnostic history and may not be read, scored, or
  used for control design. `rescue_confirm` remains unopened until the entire
  control/operator protocol is frozen from `protocol_design`.
- Existing R001 artifacts and the stopped R001/R002 plan, tracker, locks, and
  execution log are immutable provenance. New artifacts use new names.
- No decoder, K budget, checkpoint, selector, loss, optimizer, calibration,
  state labels, or training data may change in this cycle.
- Geometry construction may use only expert target masks, deterministic image
  content bounds, image identifiers, and frozen constants. It may not use
  image intensities, model outputs, gradients, attention, scores, or outcomes.
- VinDr train lacks patient/study identifiers. Every result remains
  `image_disjoint_development_only=true` and cannot support a patient-level or
  paper-ready final claim.

## Frozen connected-control definition

Let `T` be the rasterized union of all expert boxes for one finding and image,
and let `C` be the deterministic valid image-content mask already used by the
rescue audit.

### Coarse coordinate zone

Normalize a mask centroid inside the bounding rectangle of `C`.

- Vertical class: upper, middle, or lower third.
- Horizontal class: left if normalized `x < 0.40`, central if
  `0.40 <= x <= 0.60`, and right if `x > 0.60`.
- The target zone is the pair of vertical and horizontal classes of the target
  centroid.

These thresholds are frozen before implementation. They are coordinate bins,
not learned anatomy.

### Candidate construction

Create a frozen seed set from a `17 x 17` normalized lattice over `C` plus the
centroid of every 4-connected component of `C \ T`. Map each point to its
nearest admissible pixel in `C \ T`, deduplicate, and order the seeds by
SHA-256 of `control-version:image-id:finding:y:x`. For every seed, grow one
4-neighbour connected region:

1. initialize the region with the seed;
2. maintain the 4-neighbour frontier inside `C \ T`;
3. pop frontier pixels by `(squared distance to seed, stable SHA-256 tie-break)`;
4. stop exactly when the region contains `|T|` pixels;
5. accept only if the final control centroid is in the target coordinate zone.

The selected control is the accepted candidate with the smallest frozen
geometry objective, followed by the stable hash tie-break:

`|dy_norm| + |dx_norm| + 0.10 * |log(perimeter_control / perimeter_target)|`.

The objective contains geometry only. It is not changed after feasibility or
model results are observed.

### Required invariants

Every emitted control must satisfy all of the following exactly:

- same pixel area as `T`;
- contained in `C`;
- zero pixel overlap with `T`;
- exactly one 4-connected component;
- centroid in the same frozen vertical/horizontal coordinate zone;
- deterministic replay produces the identical mask and metadata.

Target component count, target perimeter, compactness, and exact target shape
are diagnostics, not matching requirements. This is the scientific change
from failed exact-shape translation and must be disclosed, not hidden.

## Claim and anti-claim map

| Item | Required evidence |
| --- | --- |
| Geometry claim | The new control is feasible without area- or consensus-selective exclusions. |
| Mechanism claim | Under frozen Qwen3.5-2B, target intervention has a larger effect than the connected coordinate-zone control for both findings. |
| Robustness claim | The direction survives both local-mean replacement and masked Gaussian blur; zero-fill is diagnostic only. |
| Anti-claim | A positive result is not proof of true anatomic matching, causal lesion removal, clinical validity, or paper-ready external generalization. |

## Experiment blocks

### C0: Candidate authority review

- Action: review this document and tracker only.
- Success: explicit user acceptance.
- Failure/inaction: no implementation, data artifact, model load, or run.
- Compute: zero GPU hours.

### C1: Geometry implementation and contract tests

- Split accessible: none beyond synthetic masks.
- Implement the frozen control algorithm in a new module/API without changing
  the stopped translation implementation.
- Add tests for exact area, containment, disjointness, 4-connectivity, zone
  identity, deterministic replay, impossible cases, and score/intensity-free
  inputs.
- Gate: all new and active BiVES tests plus synthetic smoke pass.
- Stop: any contract ambiguity returns the plan to review; do not inspect
  VinDr outcomes to repair it.

### C2: Score-free geometry survival audit

- Split: R001 `protocol_design` positives only.
- Model: none; pixels, checkpoint, cached scores, and `rescue_confirm` forbidden.
- Report feasibility overall, per finding, reader-consensus stratum, and expert
  target-area quartile. Record every exclusion reason.
- Go gate:
  - feasibility at least 95% overall and for each finding;
  - feasibility at least 90% in every prespecified finding-by-area quartile;
  - every emitted mask passes every invariant;
  - exclusions are frozen before any score is read.
- Stop: any gate failure ends this control family. Do not lower thresholds,
  remove large lesions, or create another candidate in the same cycle.

### C3: Frozen local timing and replay gate

- Entry: C2 pass only.
- Model: frozen Qwen3.5-2B B2 step-450 checkpoint; no training.
- Data: 16 protocol-design images selected by the already frozen stable order.
- Action: measure local throughput and verify original-score replay before any
  full intervention run.
- Gate: checkpoint/config/cache identities match, replay is within the existing
  frozen tolerance, and estimated C4 compute is at most four local GPU hours.
- Stop: identity/replay mismatch or excess cost blocks C4.

### C4: Connected-control mechanism gate

- Split: `protocol_design` positives only.
- Frozen systems: original image, expert target, stopped scattered control as a
  historical diagnostic, and the new connected coordinate-zone control.
- Operators:
  1. deterministic local-mean replacement from the frozen exterior ring;
  2. deterministic masked Gaussian blur.
- Zero-fill is reported only as diagnostic history and cannot be selected as
  the primary operator.
- Primary metric: `TCIG = target effect - connected-control effect`, using the
  original-state probability effect already defined by the stopped protocol.
- Report per finding: mean TCIG, image-bootstrap 95% CI, positive-image
  fraction, consensus strata, target-area quartiles, localization quartiles,
  and geometry exclusions.
- Go gate:
  - mean TCIG is positive for both findings under both primary operators;
  - for each finding, at least one primary operator has a bootstrap CI lower
    bound above zero;
  - highest-area-quartile TCIG is nonnegative for both findings/operators;
  - at least 60% of eligible images have positive TCIG per finding/operator;
  - original-score replay remains within frozen tolerance.
- Stop: any gate failure closes the route. No model rescue or 4B/9B scale-up.

### C5: One-time internal confirmation

- Entry: C4 passes with all constants, code hashes, operator definitions, and
  reporting tables frozen.
- Split: open `rescue_confirm` once; no further choices or reruns.
- Gate: the complete C4 go gate must hold on confirmation, with no per-finding
  polarity AUROC/AUPRC below frozen B0.
- Interpretation: pass is only image-disjoint internal confirmation; fail is a
  final stop for this route.

### C6: Independent final boundary

- Still blocked: a new patient-identified, expert-region final dataset and
  separate authority are required for paper-ready claims.
- No C0-C5 result unlocks a patient-level claim by itself.

## Run order and compute cap

`C0 review -> C1 contracts -> C2 geometry -> C3 timing/replay -> C4 design ->
C5 one-time confirmation -> C6 independent final`.

- C0-C2 are CPU-only.
- C3-C5 are local workstation only and capped at six total GPU hours.
- No training, hyperparameter sweep, checkpoint selection, seed sweep, model
  download, or scale study is authorized.
- Stop at the first failed gate.

## Required reporting

Every execution record must include:

- fixed plan/tracker hashes and source commit;
- data-lock, annotation, DICOM, module, script, and config hashes;
- `model_loaded`, `scores_accessed`, split access, device, and wall time;
- geometry eligibility/exclusion counts and all prespecified strata;
- explicit `formal_result=false` and `patient_level_claim=false`;
- proof that VinDr test and `rescue_confirm` were not accessed early.

## Review checklist

- [x] R002 remains failed and immutable.
- [x] The new control family and weaker scientific claim are explicit.
- [x] Coordinate zones are not mislabeled as true anatomy.
- [x] Geometry is deterministic and score/intensity-free.
- [x] Large lesions cannot be silently excluded.
- [x] Two distribution-preserving operators are co-primary.
- [x] Qwen3.5-2B, decoder, K, selector, and checkpoint remain frozen.
- [x] VinDr test and confirmation isolation remain enforced.
- [x] Local-only compute and hard stop rules are explicit.
- [x] Explicit user acceptance recorded.

## Current verdict

`ACCEPTED_C3_IN_PROGRESS`. C1 passed 98/98 active tests and synthetic smoke.
C2 passed at 375/377 overall with every per-finding/quartile gate green and a
byte-identical full rows replay. C3's 16-image local Qwen3.5-2B timing/replay
gate is authorized; C4 remains blocked.
