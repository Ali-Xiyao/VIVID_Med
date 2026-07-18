# BiVES-CXR Connected-Control C4 Execution Log

**Date:** 2026-07-18
**Scope:** local-only frozen protocol-design mechanism gate
**Result:** `COMPLETE_PASS`

## Frozen identity and boundary

- Model: local multimodal Qwen3.5-2B, snapshot SHA-256
  `6b57c58ceb3c97d4199d4e8e32b54f6ff11d0cf3a8bde3a4e940d5775ae12120`.
- Polarity checkpoint: B2 sparse exact-K=16, step 450, SHA-256
  `09c2f77313027ca313f4b03c5553f90d3d7d57436e960888466d2712e9705480`.
- Training-cache lock SHA-256:
  `503ead96a0de948b56361b8097bd14cc1ba0942878b56cb3aa0ba2e39d3ec0f2`.
- Frozen config SHA-256:
  `248d57d9a62e77acf36c6c0809428e17c1a3b37bb00741383b94db51ab4d395f`.
- R001 manifest and C2 geometry rows matched SHA-256
  `bd84cd7ca5384afbcb6228c49331b028a9641dd3dd9011157c2cec75b1f6514f`
  and `b94b77bc528033ea44681fce33f868b87c2806d19450a66ff023628fc2e039d9`.
- Source split: VinDr-train `protocol_design` positives only. The run used 375
  geometry-feasible rows (62 consolidation, 313 pleural effusion). It accessed
  zero confirmation/test rows and performed no training, threshold selection,
  checkpoint selection, model selection, or server action.

## Frozen operators

- Local-mean replacement uses an 8-pixel exterior ring.
- Masked Gaussian blur uses sigma 8.0 and truncate 3.0, normalizes by blurred
  mask support, and changes only pixels inside the intervention mask.
- Both operators are co-primary. Zero-fill remained historical diagnostic only.

## Validation and execution

- New operator/mechanism contract tests passed.
- Full active BiVES suite: 106/106 passed.
- Synthetic CPU smoke passed with finite gradients and no flat state head.
- Original-score identity replay covered the frozen 16 C3 rows with maximum
  absolute difference 0 and zero exact-K mismatches.
- Geometry-mask generation took 532.83 seconds; five-forward mechanism scoring
  took 617.40 seconds on local GPU1. Peak allocated memory was 843,002,880
  bytes. The GPU was released after completion.
- An independent post-run audit reproduced row uniqueness, split isolation,
  exact target/control areas, source identities, all reported means, and every
  gate decision.

## Co-primary results

| Operator | Finding | N | Mean TCIG | Image-bootstrap 95% CI | Positive fraction | Highest-area-quartile TCIG |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| local mean | consolidation | 62 | 0.05647 | [0.03848, 0.07839] | 0.9194 | 0.05510 |
| local mean | pleural effusion | 313 | 0.03062 | [0.02070, 0.04155] | 0.7412 | 0.08309 |
| masked Gaussian blur | consolidation | 62 | 0.05063 | [0.03032, 0.07537] | 0.7903 | 0.02356 |
| masked Gaussian blur | pleural effusion | 313 | 0.01290 | [0.00502, 0.02126] | 0.7029 | 0.01246 |

Every finding/operator mean is positive, every positive-image fraction exceeds
0.60, and every highest-area-quartile TCIG is nonnegative. Both operators have
a CI lower bound above zero for both findings, exceeding the preregistered
requirement of at least one such operator per finding.

## Artifact identities

- Mechanism rows SHA-256:
  `268d2cc6f758d719ef7112399da38dd3ca60b1069ad12af7175afce93993dbdd`.
- Metrics SHA-256:
  `072128051b9266bb771f9c6c95a21dcbfd96ed324609b6f0758e850d3dab931c`.
- Geometry-mask cache lock SHA-256:
  `a6da0c0436ed35170f42cfa47ba2dad45b4f647c595a0624d59aa059806ada8f`.
- Successful run stderr was empty.

## Decision

C4 is complete-pass. The operator constants, code, identities, result rows, and
reporting definitions are now frozen. C5 is unlocked as one one-time opening of
the image-disjoint `rescue_confirm` split. C5 may only test whether the complete
C4 mechanism gate survives and whether per-finding B2 polarity AUROC/AUPRC is
not below frozen B0. It cannot tune, rerun after seeing confirmation outcomes,
access VinDr test, train, or scale to Qwen3.5-4B/9B.
