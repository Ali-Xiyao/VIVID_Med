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
- [ ] S1: run both 256-row overfit gates sequentially.
- [ ] S2: run both 20k-study token-pretraining pilots sequentially.
- [ ] S3: run the locked CheXpert development linear probe from each
  validation-NLL-selected ViT checkpoint and freeze PASS/NO-GO.
- [ ] S4: only after PASS, run three seeds and then the full MIMIC scale track.
- [ ] S5: only after S4, freeze external mappings and run independent external
  evaluation.

## Current action

Submit the S1-S3 queue as an exclusive step behind the currently active,
unrelated step in allocation 3066.

## Stop conditions

- Stop before training on any identity/hash mismatch.
- Stop after one failed identity-preserving implementation repair.
- Do not promote a strict scientific NO-GO by changing thresholds or teacher
  size.
- Do not access CheXlocalize test or VinDr test.
