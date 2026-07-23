# RCSD D0/D1 automatic development protocol

**Recorded:** 2026-07-23

**User authorization:** continuously execute the bounded experiment loop,
perform development-only case studies after failures, propose and test
improvements, and stop when the original proposal is supported as far as the
frozen evidence permits or when the bounded route reaches a terminal NO-GO.
The user also approved server and cache use.

## Execution boundary amendment

On 2026-07-23 the user explicitly revoked the workstation-only boundary and
authorized this RCSD-CXR extension to run through retained allocation `3066`
on `gpu01`. Code and deidentified manifests may be synchronized to the
extension's existing remote project directory. Existing datasets and model
caches are referenced in place; they are not duplicated.

Every launch requires a fresh live allocation/GPU/process check. Unrelated
processes and jobs are never stopped or signalled. The authorization applies
only to the bounded RCSD-CXR state machine and does not reactivate any frozen
BiVES, ARISE, VICER, MORPH-CXR, or localization-causality route.

## Scientific boundary

This authorization does not reopen full RCSD. D2 posterior fusion and the
tested D3 field anchor remain NO-GO; D4, teacher scaling, institution mixing,
full-data scaling, multi-seed expansion, and external tests remain closed.

The only candidate is D1 against D0-CP:

- D0-CP reconstructs the original hard-UMS token objective with the user-
  amended frozen Qwen3.5-2B teacher, ViT-B/16, and SPD 4x2 on the locked 20k
  MIMIC surface.
- D1 changes only the normalized entropy agreement weight applied to finding
  token spans. Hard targets, target strings, data, initialization, optimizer,
  budget, and checkpoint rule remain identical.

## Automatic state machine

1. Produce and validate R0 implementation/provenance artifacts.
2. Produce and validate R1 data/reliability manifests.
3. Produce and validate R2 probe-train and exposed expert-development
   manifests.
4. Transition the machine lock to `OVERFIT_AUTHORIZED`.
5. Run D0-CP then D1 on the frozen 256-row overfit surface.
6. If the replacement Qwen3.5 R3 passes, transition to `PILOT_AUTHORIZED` and
   run D0-CP then D1 on the frozen 20k surface. The earlier Qwen2.5 overfit is
   provenance only; the 20k pair restarts from zero on allocation `3066`.
7. Apply the unchanged promotion gate.
8. If a gate fails, create a development-only case-study artifact before any
   repair. A repair may change one implementation or data-contract factor,
   must not change a threshold, target label, test surface, or scientific
   identity, and must rerun the failed gate from zero.
9. Stop at PASS or when all in-identity repairs are exhausted. Do not invent a
   successor module to force a pass.

### Qwen3.5-2B R3 execution record

The replacement `s2` queue passed on allocation 3066. D0 and D1 both reached
the 0.98 token-accuracy threshold at step 350 and reduced NLL by more than
97%. Their summary SHA-256 values are frozen in
`audit/rcsd_d0_d1_review_lock.json`. This result establishes learnability only
and authorizes R4; it is not a D1 efficacy result.

## Failure classes

| Class | Meaning | Allowed response |
| --- | --- | --- |
| Contract failure | hash, identity, serialization, parity, or leakage error | repair the exact failed contract before metrics |
| Implementation failure | crash, non-finite loss, invalid gradients, loader defect | fix one defect and rerun from zero |
| Learnability failure | overfit accuracy/loss gate fails | case study optimization and token-span diagnostics; one-factor repair |
| Scientific failure | valid 20k run fails the frozen promotion gate | case study only; close D1 unless a preregistered in-identity repair exists |

External-test inspection, threshold relaxation, result-dependent label
replacement, multiple simultaneous repairs, and selective reporting are never
valid repairs.
