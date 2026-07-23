# Strict VIVID/SPD Task Plan

## Objective

Reconstruct and evaluate the historical hard-UMS VIVID/SPD hypothesis with a
frozen Qwen3.5-2B teacher, a deployable ViT-B/16 encoder, paired data and
training budgets, and fail-closed promotion gates.

## Phases

- [x] S0a: create an isolated Git branch and recover the exact historical
  prefix-projector and SPD 4x2 identities.
- [x] S0b: freeze protocol, lock, data/model paths, hashes, and protected
  evaluation surfaces.
- [x] S0c: implement the paired model, trainer, linear-probe evaluator, queue,
  audits, and tests.
- [x] S1: run both 256-row overfit gates sequentially.
- [x] S2: run both 20k-study token-pretraining pilots sequentially.
- [x] S3: run the locked CheXpert development linear probe from each
  validation-NLL-selected ViT checkpoint and freeze PASS/NO-GO. The strict
  route is frozen as `STRICT_NO_GO_DIAGNOSTIC_OPEN`.
- [ ] D1: run only the preregistered `ums_prefix8` and
  `ums_spd4x2_no_ortho` bounded diagnostics from initialization, then freeze
  either `REPAIR_NOMINATED` or `TERMINAL_NO_GO`. (in progress)
- [ ] S4: only after PASS, run three seeds and then the full MIMIC scale track.
- [ ] S5: only after S4, freeze external mappings and run independent external
  evaluation.

## Current action

Launch and monitor the two-arm bounded diagnostic queue on retained allocation
3066. The strict result remains NO-GO; prefix8 is diagnostic-only, and
no-ortho is the sole repair candidate under the pre-frozen nomination rule.

## Stop conditions

- Stop before training on any identity/hash mismatch.
- Stop after one failed identity-preserving implementation repair.
- Do not promote a strict scientific NO-GO by changing thresholds or teacher
  size.
- Do not access CheXlocalize test or VinDr test.
