# BiVES-CXR Connected-Control C3 Execution Log

**Date:** 2026-07-18
**Scope:** local-only frozen timing and score replay gate
**Result:** `COMPLETE_PASS`

## Frozen identity

- Model: local multimodal Qwen3.5-2B, snapshot SHA-256
  `6b57c58ceb3c97d4199d4e8e32b54f6ff11d0cf3a8bde3a4e940d5775ae12120`.
- Polarity checkpoint: B2 sparse exact-K=16, step 450, SHA-256
  `09c2f77313027ca313f4b03c5553f90d3d7d57436e960888466d2712e9705480`.
- Training-cache lock SHA-256:
  `503ead96a0de948b56361b8097bd14cc1ba0942878b56cb3aa0ba2e39d3ec0f2`.
- Frozen config SHA-256:
  `248d57d9a62e77acf36c6c0809428e17c1a3b37bb00741383b94db51ab4d395f`.
- R001 manifest/data lock and C2 geometry rows all matched the accepted hashes.
- Source split was VinDr train `protocol_design` positives only. No VinDr test,
  `rescue_confirm`, training, threshold selection, or model selection occurred.

## Frozen 16-image selection

The gate took the first eight feasible consolidation rows by `sample_id`, then
the first eight feasible pleural-effusion rows by `sample_id` whose image unit
had not already been chosen. This produced 16 unique train images and an exact
8+8 finding balance without reading model scores.

## Validation and execution

- New C3 contract tests: 3/3 passed.
- Full active BiVES suite: 101/101 passed.
- Synthetic CPU smoke: passed with finite gradients and no flat state head.
- First launch stopped before any score because deterministic cuBLAS required
  `CUBLAS_WORKSPACE_CONFIG`; the script now sets `:4096:8` before importing
  PyTorch and retains deterministic-algorithm enforcement.
- Successful execution used local GPU1, BF16, eager attention, one unmeasured
  warmup, and two complete measured passes over the same 16 images.

## Gate result

- Pass times: `2.2064 s` and `2.3329 s`.
- Maximum replay absolute difference: `0.0` at tolerance `1e-6`.
- Exact-K top-index mismatches: `0`.
- Peak allocated CUDA memory: `823,914,496` bytes.
- C4 estimate: `0.2461` local GPU hours after counting 375 eligible rows times
  five full visual forwards, applying a fixed 1.25 timing multiplier, and adding
  the complete C2 geometry wall time.
- Compute cap: `4.0` local GPU hours.
- Rows SHA-256:
  `5d916fb9f86e45fc5fdee5e31fec1e1c4c849d861bfeb8f6d25c4ad4e6166da6`.
- Lock SHA-256:
  `e7047f605604def320079006d486f5f55645951de3566c9675fc8308510fd4d6`.

## Decision

C3 is complete-pass. C4 is unlocked on the same frozen Qwen3.5-2B/B2 identity
and the 375 feasible VinDr-train protocol-design positives. C4 may compare the
expert target with the accepted connected coordinate-zone control under the
two co-primary frozen operators (local mean and masked Gaussian blur). It may
not open `rescue_confirm`, access VinDr test, train, change K/decoder/selector,
or scale to Qwen3.5-4B/9B.
