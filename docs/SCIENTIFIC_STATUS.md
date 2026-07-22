# Scientific status

## Active scientific object

The repository audits when chest-X-ray localization agrees with causal model
reliance under independently specified matched interventions. The active
authority remains
[`../audit/CXR_localization_causality_audit_proposal.md`](../audit/CXR_localization_causality_audit_proposal.md).

## Frozen method routes

### BiVES B2

- status: terminal negative;
- frozen tag: `bives-b2-terminal-8bb1a94`;
- role: one audited intrinsic evidence method and a source of reusable
  geometry/intervention contracts;
- prohibited: C6J, same-test repair, 4B/9B scaling, or rewriting frozen
  C4/C5/C6I evidence.

### ARISE-v1

- status: oracle intervention development closed;
- final development gate: 3 of 4 finding/operator cells pass;
- unresolved cell: pleural-effusion Gaussian blur is positive but
  inconclusive (`0.01824`, 95% CI `[-0.05995, 0.09065]`);
- terminal result canonical SHA-256:
  `0f118a1ac7e534dfb91116dc2f85137e13c337463f079f2f4da493f0ed986f52`;
- selector, U/I, Qwen3.5-4B/9B, and same-validation tuning: not authorized;
- CheXlocalize test: absent and unopened.

The canonical ARISE report is
[`../audit/arise_cxr_method_development_result.md`](../audit/arise_cxr_method_development_result.md).

## VICER-CXR V0

VICER-CXR is an independent successor attempt, not an ARISE repair. Its V0
intervention-validity development gate is complete and terminal negative:

- 384/384 frozen rows completed over four findings, three families, and four
  strengths;
- all independent critic/verifier calibration heads passed;
- only 4/12 finding-family cells passed;
- no family passed all four findings;
- corrected result canonical SHA-256:
  `3c9ceb27c66abfe28c2a269e0566a2f2be08ea3a43dea8c18dc7ddaf5a330bb1`;
- V1 coverage-redundancy and V2 coalition selection remain locked.

The canonical report is
[`../audit/vicer_v0_intervention_validity_result.md`](../audit/vicer_v0_intervention_validity_result.md).

## Data and execution status

- local workstation execution only;
- medical images, identifiers, rows, caches, checkpoints, and model weights
  remain outside Git;
- CheXlocalize validation is prior-exposed development data;
- CheXlocalize test is reserved for one separately frozen final method;
- server and Slurm execution are closed.
