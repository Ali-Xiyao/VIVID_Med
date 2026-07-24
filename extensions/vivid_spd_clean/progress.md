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
- At `2026-07-24T01:43+08:00`, unrelated step `3066.19219` remained healthy.
  Its MixLoRA phase had completed and the same preregistered `--full` wrapper
  had advanced to the plain-LoRA phase (`lora_piqa` at epoch 1, step
  5200/16113). The strict launcher remained alive and correctly blocked; its
  run root was still absent, so strict S0/S1 training had not started or used
  the GPU.
- At `2026-07-24T02:41+08:00`, the VPPM plain-LoRA phase had finished the
  large PIQA candidate and advanced to `lora_arc-e`. Its configuration trains
  three candidates one at a time (`train_lora_simultaneously_num=1`), so the
  strict route correctly remained queued behind step `3066.19219`. Launcher
  PID `2207535` was healthy and the strict run root was still absent.
- A PowerShell-to-SSH status command expanded remote shell variables locally
  and produced a read-only quoting error. The monitor command was changed to
  use literal remote arguments; no server state, experiment output, or queue
  state was modified.
- At `2026-07-24T03:07+08:00`, unrelated step `3066.19219` completed and the
  strict launcher acquired the allocation. S0 passed, but `ums_prefix4`
  failed before its first optimizer step because deterministic cuBLAS required
  `CUBLAS_WORKSPACE_CONFIG`. The failed run and logs are being preserved as an
  implementation case study. The single permitted identity-preserving repair
  sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`; both S1 arms will rerun from zero.
- The repair passed all seven strict extension unit tests, Python compilation,
  and launcher shell syntax validation before server synchronization.
- The original failure was archived unchanged under
  `local_runs/history/strict_vivid_spd_qwen35_2b_20260723_s0_s3_failed_cublas_20260724T030859`.
  The repaired queue launched from an empty original run root as step
  `3066.19599` at `2026-07-24T03:13:50+08:00`; its queue state was healthy and
  rerunning S0 before fresh paired S1 execution.
- S1 `ums_prefix4` passed after 400 steps: token accuracy `0.98907`, token NLL
  reduction `97.74%`, and finite nonzero gradients in both the backbone and
  projector. The queue advanced automatically to a fresh
  `ums_spd4x2` overfit run.
- S1 `ums_spd4x2` passed after 350 steps: token accuracy `0.98761`, token NLL
  reduction `98.15%`, and finite nonzero gradients in both the backbone and
  projector. The paired S1 gate is therefore PASS. The queue advanced to S2
  and began the 20k-study `ums_prefix4` pilot from initialization.
- S2 `ums_prefix4` passed at the locked 3,000-step budget. Validation token
  NLL fell from `1.40079` to `0.07827` (`94.41%` reduction), token accuracy
  reached `0.96953`, and the minimum validation NLL selected step 3000. The
  queue advanced automatically to the fresh S2 `ums_spd4x2` pilot.
- S2 `ums_spd4x2` passed at 3,000 steps. Validation token NLL fell from
  `1.85146` to `0.07681` (`95.85%` reduction), token accuracy reached
  `0.96988`, and step 3000 was selected.
- Both S3 expert-development probes completed. Prefix4 achieved macro AUROC
  `0.85921` and macro AUPRC `0.69087`; SPD4x2 achieved macro AUROC `0.86385`
  and macro AUPRC `0.69408`.
- The frozen promotion gate returned `STRICT_NO_GO_DIAGNOSTIC_OPEN`: macro
  AUROC delta `+0.004641` failed the `+0.005` minimum, and three of five
  findings were nonnegative versus the required four. The strict route will
  not scale or be relabeled as successful.
- Added a separate bounded-diagnostic protocol and lock. They authorize only
  `ums_prefix8` and `ums_spd4x2_no_ortho`, preserve the strict hashes and
  protected-test ban, and freeze the sole repair-nomination rule before any
  diagnostic run.
- Committed and pushed the diagnostic authority and runner as `7bc62c6`.
  Server-side shell syntax, Python compilation, and all 10 extension unit
  tests passed before launch.
- The bounded diagnostic queue started on allocation 3066 as Slurm step
  `3066.19836` at `2026-07-24T07:08:23+08:00`. It verified all five strict
  authority hashes before training and began fresh S1 `ums_prefix8` overfit.
  The first optimizer records are healthy; no protected evaluation surface was
  accessed.
- Diagnostic S1 `ums_prefix8` passed at step 400 with token accuracy
  `0.98466` and NLL reduction `97.22%`. The same queue advanced to fresh S1
  `ums_spd4x2_no_ortho`; no method identity, threshold, or data surface
  changed.
- Diagnostic S1 `ums_spd4x2_no_ortho` passed at step 350 with token accuracy
  `0.98769` and NLL reduction `98.16%`. Both bounded diagnostic arms are
  learnable, so the queue advanced to fresh S2 `ums_prefix8` at the unchanged
  3,000-step pilot budget.
- Diagnostic S2 `ums_prefix8` passed at step 3000. Validation NLL fell from
  `1.43329` to `0.07734` (`94.60%` reduction), token accuracy reached
  `0.96954`, and the minimum-NLL checkpoint was step 3000. The queue advanced
  to fresh S2 `ums_spd4x2_no_ortho`.
- Diagnostic S2 `ums_spd4x2_no_ortho` passed at step 3000. Validation NLL
  fell from `1.85146` to `0.07702` (`95.84%` reduction), token accuracy
  reached `0.97006`, and step 3000 was selected.
- S3 `ums_prefix8` completed with macro AUROC `0.85788` and macro AUPRC
  `0.68625`, both below the frozen prefix4 result. Prefix8 remains a
  diagnostic-only negative result. The queue advanced to the final
  `ums_spd4x2_no_ortho` S3 probe.
