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
