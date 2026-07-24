# Fresh Replication Instructions

## Authority

This directory is a separate replication study. It may reuse the unchanged
VIVID-GDS A3 implementation and frozen A0/A2 controls, but it must not modify,
reinterpret, or relaunch the terminal Stage-A run in `../vivid_gds`.

No experiment is authorized until a protocol and machine-readable lock are
present here and the readiness audit passes.

## Scientific question

Does unchanged A3 add deployable representation value over both A0 direct
schema supervision and A2 structured generation on a genuinely fresh
development surface?

## Hard boundaries

- Three seeds are required.
- A3, A0, and A2 use identical data, ViT initialization, augmentation,
  optimizer budget, and probe protocol.
- No new module, projector, loss, teacher size, target, or threshold.
- The old CheXpert expert-development surface is historical evidence only.
- CheXlocalize test and VinDr test remain closed.
- Medical images, patient-level manifests, predictions, checkpoints, logs,
  models, and caches remain outside Git.
- Fail closed on any identity, provenance, split, overlap, or hash mismatch.

## Workflow

Use `task_plan.md`, `findings.md`, and `progress.md` as the active planning
surface. Audit data before writing launch code; validate before opening any
new score.

