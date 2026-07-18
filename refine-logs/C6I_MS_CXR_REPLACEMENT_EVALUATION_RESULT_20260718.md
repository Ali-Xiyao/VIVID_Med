# BiVES-CXR C6I replacement evaluation result

**Date:** 2026-07-18

**Status:** `FAIL_FINAL_STOP`

**Committed execution identity:** `1acf54325b36263bec81d48a19e767eb30a9c4f5`

## Execution boundary

The separately authorized replacement opening ran once on the local
workstation using Qwen3.5-2B, the frozen B2 sparse exact-K=16 step-450
checkpoint, and the 29/29 C6I actual-input geometry release. No training,
tuning, threshold selection, row exclusion, statement rewrite, operator
change, server action, or Qwen3.5-4B/9B action occurred.

The evaluator completed all 29 patients/rows in 20.63 scoring seconds, wrote an
identity-bound progress record after every row, exited normally, and produced
an empty stderr log. GPU1 returned to 13 MiB and 0% utilization.

## Frozen results

| Operator | Finding | Mean TCIG | 95% patient bootstrap CI | Positive-patient fraction | Highest-area-quartile mean TCIG | Mean top-K localization gain |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| local mean | consolidation | 0.004480 | [-0.005244, 0.013831] | 0.6000 | -0.009204 | 0.087617 |
| local mean | pleural effusion | -0.009099 | [-0.027365, 0.008178] | 0.3571 | -0.013114 | 0.096165 |
| masked Gaussian blur | consolidation | -0.015523 | [-0.027091, -0.004813] | 0.2667 | -0.021748 | 0.087617 |
| masked Gaussian blur | pleural effusion | 0.026064 | [0.003831, 0.063670] | 0.7857 | 0.068739 | 0.096165 |

Masked Gaussian blur passes every frozen pleural-effusion condition. It fails
consolidation with a wholly negative TCIG confidence interval, negative
highest-area-quartile effect, and low positive-patient fraction. Local mean
also fails the complete gate: consolidation has a CI crossing zero and a
negative highest-area-quartile effect, while pleural effusion has negative mean
TCIG, negative highest-area-quartile effect, and a low positive fraction.

The required condition that every finding has at least one operator with a
strictly positive CI lower bound is false for consolidation. The aggregate
survival gate is therefore `pass=false` and the route is `fail_final_stop`.

## Artifact identities

| Artifact | SHA-256 |
| --- | --- |
| C6I pre-open lock | `ae4dfa07652aab0fa9fa15c96bd08f0205a1caf203017e26a7ba9c8e755b0a6e` |
| opening marker | `34df18648aadcfea1541e4c7413e763c3f8eef1110808fd5d92432c6cce136e3` |
| final progress | `ad44c5b8c638cdf3d7a1b640a3e387718b48c2a1921020f820f6e7cb4241dfe4` |
| evaluation rows | `bd9ba71f67e3d6a5a6fdb97d66600ae21d01570dc2b5254fe537bf4b9b7a23ab` |
| metrics file | `eaa2e63e6a9e8d2dbb35e40dccc00007848bb3f2d6ada09215d1e65f388cd256` |
| metrics canonical artifact | `94e4e60f142e40aa98f70bb7ed82ef1acdc224cdeee53abae6b347e688b8136a` |
| stdout | `7b85a881f35789565c76feb1b6828559b4d7bfdb6c6a5f994d35f51cb0aa5d3e` |
| stderr | empty; `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

## Interpretation and terminal decision

C6I resolves the C6H pixel-alignment defect and yields a valid positive-only
external mechanism result. It does not rescue the frozen mechanism claim:
evidence localization is better than random on average, but target-region
intervention is not consistently more damaging than matched control
intervention across both findings and both operators.

This is nonformal and not clinical validation. The C6I route is terminal. Do
not rerun, tune masks/operators/thresholds on these outcomes, reopen C5, or
scale to Qwen3.5-4B/9B from this result.
