# BiVES-CXR C6F MS-CXR post-C5 evaluation authority

**Date:** 2026-07-18

**Status:** `AUTHORIZED_PREOPEN_QWEN35_2B_ONLY`

**Execution surface:** local workstation only

## Independent authorization

The user explicitly authorized creation of a separate post-C5 model-evaluation
record and the corresponding local code on 2026-07-18. Therefore this file
records:

```text
model_evaluation_authorized=true
```

This authorization is independent of the frozen C6E intake JSON. The C6E
artifact must remain byte-identical and must continue to report
`model_evaluation_authorized=false`, because it is a metadata/license intake
record rather than a research protocol.

## Frozen question and anti-claims

MS-CXR v1.1.0 supplies 29 publisher-test positive image-text pairs (15
Consolidation and 14 Pleural Effusion) with 45 expert boxes. It does not supply
an authoritative negative set for these two questions. Consequently:

- the evaluation tests whether the frozen B2 exact-K evidence mechanism is
  sensitive to the expert-box region more than to an exact-area, disjoint,
  same-coordinate-zone connected control;
- AUROC, AUPRC, accuracy, and B2-versus-B0 classification claims are forbidden;
- the result is a small independent external mechanism evaluation, not a
  clinical validation claim;
- no training, tuning, threshold selection, checkpoint selection, statement
  rewrite, operator change, or model selection is allowed after opening.

## Frozen model identity

- Qwen3.5-2B snapshot SHA-256:
  `6b57c58ceb3c97d4199d4e8e32b54f6ff11d0cf3a8bde3a4e940d5775ae12120`.
- B2 sparse exact-K=16 step-450 checkpoint SHA-256:
  `09c2f77313027ca313f4b03c5553f90d3d7d57436e960888466d2712e9705480`.
- Training-cache lock SHA-256:
  `503ead96a0de948b56361b8097bd14cc1ba0942878b56cb3aa0ba2e39d3ec0f2`.
- Frozen training config SHA-256:
  `248d57d9a62e77acf36c6c0809428e17c1a3b37bb00741383b94db51ab4d395f`.
- Canonical statements remain exactly:
  `Focal air-space consolidation is present.` and
  `Pleural effusion is present.`

## Frozen data and geometry gates

Before any model load, the ignored patient manifest and dataset lock must bind:

- the strict C6E canonical intake identity
  `0027358c2998773e73dbd19da02a37dac27c060150bf42e59469d218fb24b4ed`;
- the exact release, license, prior-registry, metadata, and image hashes;
- 29 patient-disjoint publisher-test rows, 29 images/studies/patients, 45 boxes,
  15/14 finding counts, and zero prior patient/study overlap;
- one exact-area, target-disjoint, 4-connected, same-coordinate-zone control
  for every row. Any infeasible row is a hard pre-open failure; no denominator
  exclusion is allowed on this 29-row release.

Raw patient, study, and image identifiers remain only in ignored local
artifacts. Tracked files contain hashes and aggregate counts only.

## Frozen intervention and survival gate

The intervention code and thresholds are inherited unchanged from C4/C5:

- local-ring mean, exterior ring width 8 pixels;
- content-normalized masked Gaussian blur, sigma 8.0, truncate 3.0;
- exact-area coordinate-zone connected control;
- native released boxes with no dilation;
- patient-bootstrap 95% confidence intervals, 2,000 replicates, seed 17.

For both findings and both operators, the mean target-control intervention gain
(TCIG) must be positive, at least 60% of patients must have positive TCIG, and
the highest target-area quartile mean TCIG must be nonnegative. For each
finding, at least one of the two operators must have a bootstrap 95% CI lower
bound above zero. All checks are conjunctive. Top-K target coverage minus a
deterministic random exact-K baseline is reported as a secondary localization
diagnostic and is not a survival gate.

## Opening and scale rule

The model may be opened exactly once after the authority, implementation,
tests, patient manifest, dataset lock, geometry lock, and clean committed code
identity all pass. Only Qwen3.5-2B is authorized. Qwen3.5-4B and Qwen3.5-9B
remain blocked even if 2B passes; either scale requires a new explicit user
authorization. A failed 2B survival gate is a final stop for this protocol.
