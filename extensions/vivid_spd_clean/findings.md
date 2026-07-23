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
