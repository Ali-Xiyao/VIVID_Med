# BiVES-CXR C6H MS-CXR one-time evaluation authority

**Date:** 2026-07-18

**Status:** `AUTHORIZED_ONE_TIME_LOCAL_QWEN35_2B`

**Execution surface:** local workstation only

## Independent authorization

After C6G completed its score-free 29/29 geometry gate, the user explicitly
approved the separate C6H one-time local Qwen3.5-2B model evaluation and asked
that it run immediately after preparation. Therefore this record states:

```text
model_evaluation_authorized=true
one_time_execution_authorized=true
```

This authority is new. It does not rewrite or reuse the model-opening identity
of C6F, and it does not change the C6G geometry authority or lock.

## Frozen input identities

- C6G geometry lock file SHA-256:
  `1b85ad241a0c8268bd71868945956c3a729fa966458308a31b3fdcb28259f8fc`.
- C6G geometry lock canonical SHA-256:
  `6271ba51e8442baad92126473513b0b901619403a4e22c353e455395ec801752`.
- C6G geometry rows SHA-256:
  `6b03e6854a3151f85e22a8f2078bec842df8de9c629b0452dd81a9005abaa50e`.
- C6G candidate certificates SHA-256:
  `26f694a990f83c255c6a9fa6382c1883663d4274b87e646a133ee0768d647a1c`.
- Frozen 29-row manifest SHA-256:
  `ba31d6e9e2cefe55effaef838a2f7cc8bf68d5c07021f22f2614782082f4f711`.
- Qwen3.5-2B snapshot SHA-256:
  `6b57c58ceb3c97d4199d4e8e32b54f6ff11d0cf3a8bde3a4e940d5775ae12120`.
- B2 sparse exact-K=16 step-450 checkpoint SHA-256:
  `09c2f77313027ca313f4b03c5553f90d3d7d57436e960888466d2712e9705480`.
- Training-cache lock SHA-256:
  `503ead96a0de948b56361b8097bd14cc1ba0942878b56cb3aa0ba2e39d3ec0f2`.
- Frozen training config SHA-256:
  `248d57d9a62e77acf36c6c0809428e17c1a3b37bb00741383b94db51ab4d395f`.

Every C6F immutable identity recorded inside the final C6G lock must still
match before opening. Any mismatch is a hard pre-open stop.

## Frozen question and intervention

C6H is a small independent external positive-only mechanism evaluation. It
tests whether intervention on the native expert-box region changes frozen B2
support more than intervention on the C6G exact-area, target-disjoint,
within-content, single-connected continuous-geometry control.

The following remain unchanged:

- canonical statements:
  `Focal air-space consolidation is present.` and
  `Pleural effusion is present.`;
- Qwen3.5-2B and the single B2 step-450 checkpoint;
- exact-K=16;
- native boxes with no dilation;
- local-ring mean with exterior-ring width 8;
- masked Gaussian blur with sigma 8.0 and truncate 3.0;
- patient bootstrap with 2,000 replicates and seed 17;
- the C6F/C4/C5 TCIG survival thresholds.

No training, tuning, threshold selection, checkpoint selection, statement
rewrite, operator change, row exclusion, model selection, or control repair is
allowed after opening.

## Claims and terminal rule

Classification metrics are forbidden because MS-CXR supplies no authoritative
negative set for these questions. C6H is not clinical validation, not a C5
failure reversal, and not authorization for Qwen3.5-4B/9B.

The evaluation may open once after the authority, configuration, implementation,
tests, pre-open lock, clean committed code identity, and live local-GPU check
all pass. A crash may resume only from an identity-matching atomic progress
record. After a terminal pass or fail, this rescue route stops and cannot be
rerun or tuned under C6H.
