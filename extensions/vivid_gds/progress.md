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
- G1 `A0_direct` passed at step 300: schema NLL fell from `1.093667` to
  `0.106180` (`90.29%` reduction), schema accuracy reached `0.993197`, and
  backbone/schema-head gradients were finite and nonzero.
- G1 `A1_freetext` is running. At step 300, token NLL was `0.358975` and token
  accuracy was `0.873562`; it had advanced beyond step 310 at the latest poll.
- The first G1 `A1_freetext` run stopped at step 500 with a scientific gate
  code but an implementation-level schedule mismatch: NLL reduction was
  `95.05%`, while accuracy was `0.964316` versus `0.98`. The 500-step run had
  no post-warmup interval.
- Preserved the failed run and froze the only schedule repair: generative
  overfit arms receive 1000 steps with the same 500-step warmup. G1 will restart
  from zero under run root suffix `_r1`.
- Validated and pushed the repair as commit `3836e8c`; local and server unit
  tests, lock audits, Python compilation, LF inspection, and launcher syntax
  checks passed.
- Started repaired run root
  `local_runs/vivid_gds_stage_a_qwen35_2b_20260724_r1` as Slurm step
  `3066.20209` (launcher PID `704958`). Repaired G0 passed and G1 restarted
  from zero with `A0_direct` currently using GPU 0.
- Repaired-run `A0_direct` reproduced its PASS exactly at step 300
  (schema accuracy `0.993197`, schema NLL reduction `90.29%`). `A1_freetext`
  then started its frozen 1000-step feasibility run and reached step 170 with
  finite training loss at the latest poll.
- Repaired-run `A1_freetext` passed at step 550: token accuracy `0.983383`,
  token NLL `0.064278`, NLL reduction `97.28%`, and finite nonzero backbone
  and projector gradients. This is the first post-warmup checkpoint and clears
  the fixed 0.98/80% gate without using the remaining budget.
- G1 `A3_gds` started automatically and reached step 150; token accuracy was
  `0.947603` and schema accuracy was `0.735544` at that milestone.
- G1 `A3_gds` passed at step 400 with token accuracy `0.989912`, schema
  accuracy `0.987245`, token NLL reduction `98.20%`, schema NLL reduction
  `92.12%`, and finite nonzero gradients for the backbone, projector, and
  schema head.
- All G1 arms are now PASS. The queue promoted automatically to G2 and started
  the frozen 3000-step `A0_direct` pilot.
- G2 `A0_direct` completed all 3000 steps: validation schema NLL `0.508074`,
  schema accuracy `0.816180`, and NLL reduction `53.82%`. The queue then
  started the paired `A1_freetext` pilot, which reached step 90 normally.
- G2 `A1_freetext` completed all 3000 steps: validation token NLL `0.480147`,
  token accuracy `0.819989`, and NLL reduction `79.73%`. G2 `A3_gds` started
  automatically and reached step 120 with both training losses decreasing.
- G2 `A3_gds` completed all 3000 steps and passed: validation token NLL
  `0.076986`, token accuracy `0.969969`, schema NLL `0.505107`, schema
  accuracy `0.821001`, and finite nonzero gradients for the backbone,
  projector, and schema head.
- The queue promoted to G3. `A0_direct` probe completed with expert-development
  macro AUROC `0.865671` and macro AUPRC `0.689420`; `A1_freetext` probe is
  now running automatically.
- G3 completed all probes and G4 applied the frozen comparisons. A3 was the
  strongest arm and passed A3-A2, but A2 failed both A2-A1 and A2-A0.
- Froze terminal `STAGE_A_NO_GO`. Completed the authorized bounded case study,
  nominated no repair, kept every external/protected surface closed, and
  recorded evidence hashes in the terminal report.
