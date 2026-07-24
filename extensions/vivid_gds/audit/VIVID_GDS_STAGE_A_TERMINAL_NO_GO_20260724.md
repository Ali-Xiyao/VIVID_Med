# VIVID-GDS Stage-A Terminal Result

## Verdict

**STAGE_A_NO_GO**

The frozen Stage-A rule required all three expert-development comparisons to
pass. Only `A3 - A2` passed. `A2 - A1` and `A2 - A0` failed, so Stage-B
expansion, protected-test opening, model scaling, and external evaluation
remain unauthorized.

This is a scientific result, not an implementation failure. G0, all repaired
G1 feasibility arms, all G2 pilots, and all G3 probes completed successfully.
The queue applied the preregistered G4 thresholds without modification.

## Frozen arms

| Arm | Expert-development macro AUROC | Macro AUPRC |
|---|---:|---:|
| A0 direct schema | 0.865671 | 0.689420 |
| A1 deterministic free text | 0.860507 | 0.687679 |
| A2 frozen UMS prefix4 | 0.859209 | 0.690875 |
| A3 VIVID-GDS | 0.869484 | 0.698273 |

The G2 A3 checkpoint was selected at the final 3000-step checkpoint by the
frozen token-NLL rule. Its validation token NLL fell by 95.59%, schema NLL
fell by 54.09%, token accuracy reached 0.969969, and schema accuracy reached
0.821001. Backbone, projector, and schema-head gradients were finite and
nonzero.

## Frozen comparison gate

| Comparison | Delta AUROC | Delta AUPRC | Nonnegative findings | Findings below -0.02 | Result |
|---|---:|---:|---:|---:|---|
| A2 - A1 | -0.001299 | +0.003196 | 2/5 | 1 | FAIL |
| A2 - A0 | -0.006462 | +0.001454 | 1/5 | 1 | FAIL |
| A3 - A2 | +0.010275 | +0.007398 | 4/5 | 0 | PASS |

For `A3 - A2`, the per-finding AUROC deltas were:

- Atelectasis: -0.006089;
- Cardiomegaly: +0.016043;
- Consolidation: +0.004596;
- Edema: +0.013839;
- Pleural Effusion: +0.022985.

## Bounded development-only case study

The preregistered learning-curve and per-finding diagnostics support three
conclusions:

1. The dual-path A3 objective is learnable and does improve the frozen
   structured-generation A2 representation. A3 passed the exact comparison
   intended to isolate the training-only schema bridge.
2. The broader proposed hierarchy does not hold on the frozen
   expert-development surface. A2 did not improve over either deterministic
   free text or the direct schema arm, and A0 remained stronger than A2 in
   macro AUROC.
3. The overall NO-GO is therefore caused by the predecessor structured-UMS
   premise, not by an A3 optimization collapse. Removing the two failed
   comparisons would change the preregistered claim and gate, so it is not an
   identity-preserving repair.

No repair is nominated. The preregistered `lambda_schema=0.25` sensitivity is
not run: A3 already passed its primary comparison, while changing its loss
weight cannot repair the failed `A2 - A1` and `A2 - A0` claims. Additional
case selection or same-surface tuning would be post-hoc.

The de-identified finding-level analysis and recommended fresh-split decision
gate are recorded in
`VIVID_GDS_CASE_STUDY_AND_NEXT_DIRECTION_20260724.md`.

## Evidence authority

Remote run root:

`/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/02101_vivid_gds/local_runs/vivid_gds_stage_a_qwen35_2b_20260724_r1`

The original failed run remains preserved at:

`/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/02101_vivid_gds/local_runs/vivid_gds_stage_a_qwen35_2b_20260724`

Key SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| repaired queue state | `f5e91156981f5134d55214550dc5d9f02807a9790f0e00b7dd4fac423d426891` |
| Stage-A verdict | `76cbf6b32fd81735e10d6feae891ff220f3132c3c9cf9a8ad6c326e97fd9e414` |
| G2 A0 summary | `b1f4a096ccdc9b70733510fda54b50a46d6d3665f562fe287465e23fcfdce2f5` |
| G2 A1 summary | `f562ecd8e34cc149eea643834b1cf320634730e3eb5675618255686ccf25e368` |
| G2 A3 summary | `405d0369844230cf69f08bb9e2c8784f97681941da4386f8d82c8a33487ad4b1` |
| G3 A0 summary | `bda11ab47aab4ab52650357d161f092bccb7e93eab7c3032cc734213e76d302c` |
| G3 A1 summary | `0204792eb3fa8f396cc878d54b3066d3859ede5ceb73ce732d29cd5ddd2d8931` |
| reused G3 A2 summary | `f707a740d877ce33bc181e365d8031d93a0641d9666e7caf830b8ee063450134` |
| G3 A3 summary | `94e960c97197b8f57018d94e09ae4f1dcbcc36eda4af6e4e60bf3ce0cf4a64b4` |

Generated predictions, checkpoints, logs, datasets, model weights, and patient
data remain outside Git.

## Closed boundaries

- Do not reinterpret the `A3 - A2` success as a Stage-A pass.
- Do not relax or remove the failed comparisons.
- Do not rerun the exposed expert-development surface for selection.
- Do not run `lambda_schema=0.25`, add a module, change the projector, or
  scale Qwen.
- Do not open CheXlocalize test, VinDr test, NIH, PadChest, CheXpert-Plus, or
  any other protected/external surface for rescue.
