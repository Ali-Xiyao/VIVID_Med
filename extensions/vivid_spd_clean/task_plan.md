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
- [x] D1: run only the preregistered `ums_prefix8` and
  `ums_spd4x2_no_ortho` bounded diagnostics from initialization, then freeze
  either `REPAIR_NOMINATED` or `TERMINAL_NO_GO`. Result:
  `TERMINAL_NO_GO`.
- [x] S4: locked and not run because S3/diagnostics did not survive.
- [x] S5: locked and not run because S4 was not authorized.

## Current action

Terminal closure is complete. Preserve the strict and diagnostic artifacts;
do not launch further experiments under this route.

## Stop conditions

- Stop before training on any identity/hash mismatch.
- Stop after one failed identity-preserving implementation repair.
- Do not promote a strict scientific NO-GO by changing thresholds or teacher
  size.
- Do not access CheXlocalize test or VinDr test.
