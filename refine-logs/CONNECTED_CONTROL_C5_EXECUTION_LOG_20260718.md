# BiVES-CXR Connected-Control C5 Execution Log

**Date:** 2026-07-18
**Scope:** one-time local image-disjoint internal confirmation
**Result:** `FAIL_FINAL_STOP`

## Frozen opening and identity

- The pre-open authority was committed and pushed at `eb3678a` before any
  `rescue_confirm` row was read.
- Frozen plan SHA-256:
  `d04a0b3809a23828ce82eed669ed56aa18bc3e6c09c718bd708f6810b4380a70`.
- Frozen tracker SHA-256:
  `04887e6681e23dd8a7bd99120fa647ccd6194773cf565a4d1b7f6f62fc30e05f`.
- C4 mechanism rows and metrics matched SHA-256
  `268d2cc6f758d719ef7112399da38dd3ca60b1069ad12af7175afce93993dbdd`
  and `072128051b9266bb771f9c6c95a21dcbfd96ed324609b6f0758e850d3dab931c`.
- The frozen Qwen3.5-2B snapshot, B2 step-450 checkpoint, exact-K=16 config,
  training cache, both B0 models, control implementation, and operator code all
  matched their locked hashes before opening.
- `CONFIRMATION_OPENED.json` was written once at 08:25:52 local time. The run
  accessed VinDr-train `rescue_confirm` only, not VinDr test, and performed no
  training, tuning, model/checkpoint/threshold selection, or server action.

## Geometry gate

- All 378 frozen confirmation positives remained in the denominator.
- Eligible: 377/378 (99.74%); consolidation 59/59 and pleural effusion 318/319.
- The single exclusion was a prespecified pleural-effusion area-quartile-4
  sample for which no exact-area connected control had the target coordinate
  zone. It was retained as an outcome-independent geometry exclusion.
- Every overall, per-finding, and finding-area-quartile feasibility threshold
  passed; invariant failures were zero.

## Confirmation mechanism result

The complete C4 mechanism gate reproduced on all 377 eligible positives.

| Operator | Finding | N | Mean TCIG | Image-bootstrap 95% CI | Positive fraction | Highest-area-quartile TCIG |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| local mean | consolidation | 59 | 0.03872 | [0.02473, 0.05694] | 0.9153 | 0.04991 |
| local mean | pleural effusion | 318 | 0.02739 | [0.02207, 0.03310] | 0.7453 | 0.06506 |
| masked Gaussian blur | consolidation | 59 | 0.03181 | [0.01825, 0.05213] | 0.8644 | 0.02542 |
| masked Gaussian blur | pleural effusion | 318 | 0.00870 | [0.00537, 0.01317] | 0.6824 | 0.00844 |

Every mean and highest-area quartile is positive, all positive-image fractions
exceed 0.60, and all four CI lower bounds exceed zero.

## Frozen B2-versus-B0 polarity result

| Finding | Balanced rows | B0 AUROC | B2 AUROC | B0 AUPRC | B2 AUPRC | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| consolidation | 118 | 0.90003 | 0.93077 | 0.91174 | 0.89381 | **FAIL: B2 AUPRC below B0** |
| pleural effusion | 638 | 0.90545 | 0.97104 | 0.91762 | 0.97111 | pass |

The preregistered C5 rule requires both AUROC and AUPRC to be no lower than B0
for every finding. Consolidation AUPRC violates that rule despite its AUROC
gain and despite the complete mechanism-gate replication. Selectively ignoring
the failed metric is prohibited.

## Execution incident and compute accounting

- The first process reached 307/756 rows, then Windows denied an atomic
  `progress.json.tmp -> progress.json` replace while the progress file was being
  monitored. This was an IO checkpoint interruption, not a model or metric
  failure.
- The exact same committed identity resumed from the 307 completed rows. No
  code, data, operator, threshold, model, or result rule changed; no completed
  row was recomputed. Resume stderr was empty.
- The metrics file reports 0.7516 conservative hours using the resumed-process
  timer. A post-run wall-clock audit from the immutable opening marker to the
  final metrics timestamp gives 1,769.12 C5 seconds and a corrected conservative
  C3-C5 total of 1.0570 hours, still far below the six-hour cap.
- GPU1 returned to 13 MiB / 0% utilization after completion.

## Artifact identities and independent audit

- Confirmation rows SHA-256:
  `f6a010c7065ff292a839b0c8cac3b5680e0b604a4daa5c606df9c921686ff9ad`.
- Geometry rows SHA-256:
  `58b10019306e63b9ef1d08b94cd01e9540f3591155b3981762bc9657ccf0be41`.
- Metrics SHA-256:
  `b2baa4a588e6a9d020d0810b5f1d9eeec176cd46bed9382721b6ad841831a4fe`.
- A separate read-only audit reproduced all 756 unique rows, balanced labels,
  no test rows, geometry eligibility, B0/B2 AUROC/AUPRC, mechanism means, both
  row hashes, and the pass/fail split (`INDEPENDENT_C5_AUDIT_PASS`).

## Final decision

C5 is `FAIL_FINAL_STOP`. The connected-control route must not be modified and
rerun on this confirmation split. No 4B/9B scale-up, extra seed, alternate
operator, threshold adjustment, selector rescue, or VinDr-test use is
authorized. The positive mechanism replication remains descriptive internal
evidence only. C6 remains blocked on a separately authorized, patient-identified
expert-region final dataset and cannot be inferred from this image-disjoint run.
