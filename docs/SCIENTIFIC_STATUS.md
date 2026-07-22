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

## Successor-method boundary

VICER-CXR is the proposed successor, not an ARISE repair. It requires a new
branch, proposal, development authority, data lock, and preregistered gates.
Its intended order is:

1. V0 independently validate intervention dose-response and collateral
   preservation on new development images;
2. V1 diagnose evidence coverage and redundancy using only V0-valid
   intervention families;
3. V2 train an adaptive evidence-coalition selector only if V0 and V1 pass.

This terminal snapshot does not itself authorize VICER scores or test use.

## Data and execution status

- local workstation execution only;
- medical images, identifiers, rows, caches, checkpoints, and model weights
  remain outside Git;
- CheXlocalize validation is prior-exposed development data;
- CheXlocalize test is reserved for one separately frozen final method;
- server and Slurm execution are closed.
