# Strict VIVID/SPD Findings

## 2026-07-23 historical identity audit

- Historical A+UMS uses `VisionProjector`, a two-layer MLP, LayerNorm, and four
  learned prefix tokens; it also projects and appends all ViT tokens.
- Historical A+UMS+SPD uses four independent groups with two queries each,
  four-head cross-attention, a shared two-layer MLP, projected ViT tokens, and
  orthogonality weight `0.02`.
- The historical main checkpoint/config identity is 4x2, despite older prose
  describing 3x2.
- The strict comparison therefore has different learned prefix/query counts
  (4 versus 8). An 8-prefix matched-budget arm is diagnostic only and cannot
  replace the primary historical comparison.
- The user requested Qwen3.5-series teachers. Qwen3.5-2B is frozen as the
  initial teacher; larger sizes are scale studies, not rescue variables.
- Existing RCSD D0/D1 hard-UMS manifests and token code provide reusable
  engineering assets, but reliability weights and the RCSD conclusion are
  outside this method identity.
- The available CheXpert expert-development manifest has five expert findings,
  not twelve. The primary S3 gate is therefore five-label expert macro
  AUROC/AUPRC; a 12-label auto-label split is secondary only.
- Server S0 confirmed 19,533 MIMIC train rows, 1,679 patient-disjoint
  validation rows, 256 locked overfit rows, zero missing selected MIMIC images,
  available CheXpert probe/expert-development paths, and unchanged teacher,
  backbone, and manifest hashes.
- On this cluster, `srun --exclusive` can coexist with an existing step in the
  same retained allocation. Safe serialization therefore requires an explicit
  non-batch-step wait before `srun`; this is now part of the tracked launcher.
- The unrelated VPPM command is a two-stage `--full` wrapper: completing its
  MixLoRA phase does not free allocation step `3066.19219`; it immediately
  enters a plain-LoRA phase. Queue readiness must therefore be determined from
  the Slurm step and wrapper completion, not from a single phase marker.
- The VPPM plain-LoRA configuration contains PIQA, ARC-E, and BoolQ candidates
  with `train_lora_simultaneously_num=1`; its per-task epoch counter can reset
  after a candidate completes without indicating a stalled or restarted job.
- The first strict S1 launch exposed a missing deterministic CUDA runtime
  contract: `torch.use_deterministic_algorithms(True)` requires
  `CUBLAS_WORKSPACE_CONFIG` on this CUDA stack. The failure occurred in the
  initial validation forward pass before training, so it does not bear on
  either scientific arm. Setting the workspace contract before importing
  PyTorch is the one identity-preserving repair.
- The repaired historical prefix4 arm passed the locked S1 overfit gate
  (`0.98907` token accuracy; `97.74%` NLL reduction), confirming that the
  cuBLAS workspace repair restored deterministic execution without changing
  the scientific identity.
- Historical SPD 4x2 also passed S1 (`0.98761` token accuracy; `98.15%` NLL
  reduction). Both primary identities can fit the locked hard-UMS development
  surface, so the scientific comparison proceeds to paired S2 pilots.
- The historical prefix4 arm also passed the 20k-study S2 learnability gate
  (`94.41%` validation-NLL reduction; `0.96953` token accuracy). Its locked
  best checkpoint is step 3000 and will be used for the later S3 probe.
- Historical SPD 4x2 passed S2 with `95.85%` validation-NLL reduction and
  `0.96988` token accuracy; its minimum-NLL checkpoint was also step 3000.
- The strict S3 comparison is a scientific NO-GO under the frozen gate.
  SPD improved macro AUROC by `+0.004641` and macro AUPRC by `+0.003205`,
  but the AUROC gain missed the required `+0.005` and only three of five
  findings were nonnegative. Atelectasis declined by `-0.03349`; the other
  per-finding deltas were Cardiomegaly `+0.03732`, Consolidation `-0.00515`,
  Edema `+0.01637`, and Pleural Effusion `+0.00815`.
- The strict verdict authorizes only two bounded diagnostics. Prefix8 tests
  token-budget confounding and is never promotable. SPD4x2 without the
  orthogonality loss is the sole possible repaired identity, and it may only
  be nominated if it passes the unchanged S3 gate and improves on historical
  SPD under the pre-frozen rule.
- The prefix8 sequence-budget diagnostic does not improve the frozen prefix4
  expert-development result: macro AUROC is `0.85788` versus `0.85921`
  (`-0.00132`), and macro AUPRC is `0.68625` versus `0.69087` (`-0.00463`).
  Thus simply matching SPD's eight learned query/prefix tokens does not explain
  or rescue the strict comparison.
