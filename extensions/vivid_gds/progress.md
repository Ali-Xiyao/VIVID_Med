# Progress

## 2026-07-24

- Read the supplied VIVID-Med+ proposal in UTF-8.
- Created branch `codex/vivid-gds-extension`.
- Froze the VIVID-GDS method identity, Stage-A arms, checkpoints, gates,
  protected surfaces, and stop rules.
- Confirmed the same frozen hard-UMS rows can authorize synchronized schema
  masks and deterministic free-text rendering.
- Implemented A0/A1/A2/A3 model identities, synchronized schema loss,
  deterministic free-text rendering, Stage-A training, readiness, queue, and
  frozen promotion-gate scripts.
- Passed six unit tests, the CPU smoke, the lock audit, Python compilation,
  CLI parsing, and `git diff --check`.
- Synced the code-only package to the isolated server root. Server unit tests,
  smoke, and lock audit passed.
- Preserved a prelaunch G0 failure caused by an incorrect documented row count.
  Applied the single allowed identity-preserving repair: the lock now records
  the actual immutable 19,533/1,679/21,212 split.
- Preserved a pre-step launcher failure caused by CRLF shell line endings.
  Added an LF repository contract; no Slurm GPU step or run root had started.
- Repaired G0 passed all data, model, checkpoint, patient, image, and protected
  surface checks.
- Launched the sequential Stage-A queue on allocation 3066 as Slurm step
  `3066.20183`; launcher PID is `632970`.
- G0 passed and G1 `A0_direct` began on GPU. At the first observed milestone,
  step 50 schema NLL was `0.842170` and schema accuracy was `0.701531`.
