# S1 Prefix4 cuBLAS Implementation-Failure Case Study

Date: 2026-07-24  
Scope: development-only S1 implementation failure  
Scientific identity: unchanged

## Preserved failure

The strict queue acquired allocation `3066` after unrelated step
`3066.19219` completed. S0 readiness passed again, then the first S1 arm
(`ums_prefix4`) exited during its initial validation forward pass.

Preserved remote evidence:

- original run:
  `local_runs/strict_vivid_spd_qwen35_2b_20260723_s0_s3`;
- `queue_state.json`: `status=no_go`, `stage=S1`,
  `terminal_reason=ums_prefix4_S1_failed`;
- `s1/ums_prefix4.log`: deterministic cuBLAS runtime error before optimizer
  training;
- return code: `1`.

No S1 optimizer step completed, no checkpoint was selected, and no protected
evaluation surface was accessed.

## Root cause

`train_vivid_spd_token.py` correctly enables fail-closed deterministic
algorithms. On CUDA >=10.2, deterministic cuBLAS matrix multiplication also
requires `CUBLAS_WORKSPACE_CONFIG` to be set before CUDA initialization. The
server launcher did not provide that environment contract.

The failure is therefore an implementation/environment contract failure, not
an S1 scientific NO-GO and not evidence about prefix4 or SPD.

## Single identity-preserving repair

The one permitted S1 implementation repair is:

1. set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing PyTorch in
   `train_vivid_spd_token.py`;
2. export the same value in the allocation-3066 launcher.

This changes no model, teacher, target, data row, seed, optimizer, learning
rate, budget, threshold, checkpoint rule, or protected-surface boundary.

## Validation and rerun rule

Before relaunch:

- run the local unit/smoke suite;
- verify shell syntax;
- sync only the repaired code and this case study;
- archive the failed run without rewriting it.

Then rerun the complete strict queue from a new empty instance of the original
run-root identity. Both S1 arms must rerun from initialization. No partial
checkpoint or metric from the failed attempt may be reused.

Validation completed before sync:

- seven strict extension unit tests: PASS;
- Python bytecode compilation: PASS;
- launcher `bash -n`: PASS.

The immutable failed run was moved to:

`local_runs/history/strict_vivid_spd_qwen35_2b_20260723_s0_s3_failed_cublas_20260724T030859`

The repaired queue was launched from an empty original run root as Slurm step
`3066.19599`. It reruns S0 and both S1 arms from initialization.
