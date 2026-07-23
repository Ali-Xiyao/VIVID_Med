# Strict VIVID/SPD Progress

## 2026-07-23

- Created branch `codex/vivid-spd-clean-extension` from audited commit
  `515ae5a1922125d3456d1338e68c4322aa53bfa5`.
- Inspected the exact historical `VisionProjector`, `SPDProjector`, and paired
  A+UMS/A+UMS+SPD configurations.
- Confirmed the real SPD identity is four groups by two tokens with
  orthogonality weight `0.02`.
- Started an isolated extension surface. No new training or protected-test
  access has occurred yet.
- Implemented the exact paired projector/model, hard-UMS token trainer,
  CheXpert expert-development probe, readiness audit, promotion gate, and
  sequential queue.
- Local and server validation passed: seven unit tests, projector smoke, and
  machine-lock audit.
- Server S0 passed on 19,533 train, 1,679 validation, and 256 overfit rows;
  train/validation patient overlap is zero, selected MIMIC paths are complete,
  and protected test surfaces remain unopened.
- Allocation 3066 is currently occupied by an unrelated VPPM step. The strict
  queue will be submitted as an exclusive follow-on step rather than
  preempting or sharing that active process.
- Added a tracked allocation-3066 launcher. It requests an exclusive Slurm
  step, so the queue waits behind existing allocation work before acquiring
  the GPU and four CPUs.
- The first launcher attempt showed that Slurm `--exclusive` alone did not
  serialize job steps. Our S0-only step `3066.19335` was stopped before model
  loading, its empty/pretraining-free run root was removed, and the launcher
  now explicitly waits until allocation 3066 has no non-batch step.
- The corrected launcher is active as remote PID `2207535`. It is waiting
  behind unrelated step `3066.19219`; no strict-route GPU training has started
  yet. A ten-minute heartbeat monitor now owns the automatic handoff and
  bounded failure policy.
