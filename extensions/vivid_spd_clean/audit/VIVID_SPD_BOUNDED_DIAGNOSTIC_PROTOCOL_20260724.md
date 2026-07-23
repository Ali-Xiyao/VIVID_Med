# Strict VIVID/SPD Bounded Diagnostic Protocol

## Authority and frozen premise

The historical `ums_spd4x2` comparison completed S0-S3 and is permanently
recorded as `STRICT_NO_GO_DIAGNOSTIC_OPEN`. Its S3 verdict SHA-256 is
`426f46e54b7cc29f7647ba4bf62ca639de8f842498259737222fc6cd0004353a`.
Diagnostics cannot retroactively convert that strict result into a pass.

The data manifest, overfit rows, Qwen3.5-2B teacher, ViT-B/16 initialization,
hard-UMS target, seed, optimizer, training budget, checkpoint rule, expert
development surface, and S3 thresholds remain identical to the strict route.
CheXlocalize test and VinDr test remain prohibited.

## Diagnostic arms

Two and only two training diagnostics are authorized:

1. `ums_prefix8`: the historical prefix baseline with eight learned prefix
   tokens. This tests whether the strict result is explained by the four-token
   versus eight-token sequence budget. It is diagnostic-only and can never be
   nominated as the repaired SPD method.
2. `ums_spd4x2_no_ortho`: the historical four-by-two SPD projector with the
   orthogonality coefficient changed from `0.02` to `0.0`. This tests whether
   the historical orthogonality term dominates optimization.

Each arm reruns S1 overfit, S2 20k-study pilot, and S3 CheXpert expert-
development probe from fresh initialization. The arms run sequentially.
Independent per-finding deltas and SPD attention-group collapse diagnostics
are reported after training.

## Pre-frozen interpretation

`ums_prefix8` is compared with `ums_prefix4` to quantify sequence-budget
confounding and with historical `ums_spd4x2` to contextualize the strict
result. It is never a promotion candidate.

`ums_spd4x2_no_ortho` may be nominated as the single repaired identity only
when all of the following hold:

- it passes the unchanged S1 and S2 learnability gates;
- against the frozen `ums_prefix4` S3 summary, it passes every original S3
  promotion threshold;
- its macro AUROC is strictly greater than historical `ums_spd4x2`;
- its number of nonnegative per-finding AUROC deltas is not lower than the
  historical SPD result.

If those conditions hold, the outcome is `REPAIR_NOMINATED`, not
`REPAIRED_PASS`. A new repair lock and a fresh paired rerun are then required.
If they do not hold, the terminal outcome is `TERMINAL_NO_GO`.

No other architecture, loss, threshold, teacher size, target, split,
checkpoint selection, or evaluation surface may be tried in this diagnostic
window.
