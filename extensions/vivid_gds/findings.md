# Findings

## 2026-07-24 intake

- The strict SPD route is terminal NO-GO; prefix4 remains the supported
  generation baseline and is reusable as A2.
- The frozen hard-UMS manifest contains `row_id`, patient/study identity,
  split, image path, and a deterministic JSON target. Its target contains only
  selected answerable findings, so the same JSON can be the single authority
  for A0/A1/A2/A3 field masks.
- The canonical 20k source CSV also retains `report_path`, but raw reports are
  not needed for the proposal's historical free-text ablation. The historical
  VIVID free-text arm rendered finding states through sentence templates.
- A1 will therefore deterministically render the exact selected UMS fields
  into historical-style free text. This changes only representation format,
  not data identity or field coverage.
- A0 has no generation objective, so its pretraining checkpoint must be chosen
  by validation schema NLL. A1/A2/A3 remain selected by generation token NLL.
- Allocation 3066 is running on gpu01 and, at the latest audit, only its batch
  step is present. A new exclusive step may be queued without stopping or
  competing with unrelated work.
- The first server G0 prelaunch audit correctly failed because the proposal's
  “20k train” shorthand did not match the frozen manifest's exact counts:
  19,533 train, 1,679 validation, 21,212 total. Every identity hash and asset
  check passed. The failed `readiness_prelaunch.json` is preserved remotely.
- The one allowed identity-preserving implementation repair changes only the
  declared/checking counts to those exact observed values; it does not change
  the manifest, split, target, threshold, teacher, or checkpoint.
- The first launcher invocation then failed before creating a Slurm step
  because the Windows-exported shell file had CRLF line endings. The run root
  was never created and no GPU work occurred. The remote log is preserved.
- Shell files are now explicitly locked to LF through `.gitattributes`; this
  is a transport repair and does not alter the scientific identity.
- G1 A1 failed the original 500-step overfit gate despite a 95.05% NLL
  reduction because token accuracy reached 0.9643 rather than 0.98. The
  optimizer was still in its 500-step warmup for the entire feasibility run.
- The single A1 schedule repair gives generative overfit arms 500 additional
  post-warmup steps. It does not change the fixed thresholds, pilot budget,
  data, model, targets, or checkpoint rules. The original run is preserved.
- The repaired A1 crossed the unchanged gate at step 550: token accuracy
  `0.983383`, token NLL `0.064278`, and NLL reduction `97.28%`. This confirms
  the first failure was caused by the missing post-warmup interval rather than
  an unlearnable deterministic free-text target.
- A3 passed both unchanged feasibility gates together at step 400: token
  accuracy `0.989912`, schema accuracy `0.987245`, token NLL reduction
  `98.20%`, and schema NLL reduction `92.12%`. All three trainable modules had
  finite nonzero gradients, so the dual-path implementation is learnable.
- The G2 A0 pilot completed all 3000 steps on the frozen split. Validation
  schema NLL fell from `1.100093` to `0.508074` (`53.82%`) and schema accuracy
  rose from `0.316571` to `0.816180`; gradients remained finite and nonzero.
- The G2 A1 pilot completed all 3000 steps. Validation token NLL fell from
  `2.368876` to `0.480147` (`79.73%`) and token accuracy rose from `0.549314`
  to `0.819989`; the best checkpoint was the final checkpoint and gradients
  remained finite and nonzero.
- The G2 A3 pilot completed all 3000 steps and passed. Validation token NLL
  fell from `1.746747` to `0.076986` (`95.59%`) while token accuracy reached
  `0.969969`; schema NLL fell from `1.100093` to `0.505107` (`54.09%`) and
  schema accuracy reached `0.821001`. The final checkpoint was best under the
  frozen token-NLL rule, and all three trainable modules had finite nonzero
  gradients.
- G3 A0 produced a CheXpert expert-development macro AUROC of `0.865671` and
  macro AUPRC of `0.689420` from the internally selected epoch-23 probe. These
  values are development evidence only; A1, reused A2, and A3 must finish
  before the frozen three-comparison gate can be applied.
- All G3 probes completed. Macro AUROC/AUPRC were A0 `0.865671/0.689420`,
  A1 `0.860507/0.687679`, reused A2 `0.859209/0.690875`, and A3
  `0.869484/0.698273`.
- G4 froze `STAGE_A_NO_GO`: A2-A1 failed (`-0.001299` AUROC), A2-A0 failed
  (`-0.006462` AUROC), and A3-A2 passed (`+0.010275` AUROC and `+0.007398`
  AUPRC, 4/5 nonnegative findings, no decline below -0.02).
- The bounded case study shows that A3 learned and improved over A2, but the
  prerequisite claim that A2 improves over free text and direct schema did
  not hold. No identity-preserving repair can alter those baseline
  comparisons, so no repair or lambda sensitivity is nominated.
