# Strict VIVID/SPD Clean-Extension Instructions

## Active authority

`audit/VIVID_SPD_CLEAN_EXPERIMENT_PROTOCOL.md` and
`audit/vivid_spd_clean_lock.json` are the only scientific and execution
authorities for this directory.

This route is separate from RCSD, VSL, BiVES, ARISE, VICER, and MORPH. Do not
reuse their failed claims, posterior fusion, reliability weighting, field
anchors, four-state targets, selectors, or intervention results.

## Method identity

The primary comparison is paired and single-variable:

- `ums_prefix4`: frozen Qwen3.5-2B, deterministic hard UMS, ViT-B/16, the
  historical four learned prefix tokens, and the historical two-layer
  projector.
- `ums_spd4x2`: the identical teacher, hard UMS, ViT, data, optimizer, prompt,
  target, and checkpoint rule, replacing the prefix projector with historical
  four-by-two SPD and its fixed `0.02` orthogonality weight.

Qwen3.5-4B/9B and full-scale training remain locked until the Qwen3.5-2B
development gate passes. Teacher size is never a repair variable.

## Execution boundary

The user authorized server execution and cached models for this branch.
Execution may use the retained SUES allocation `3066` on `gpu01`, but:

- inspect allocation and step ownership before launching;
- do not stop, alter, or compete with unrelated jobs;
- run paired arms sequentially through one queue;
- keep all data, models, checkpoints, environments, and runtime outputs outside
  Git;
- never open CheXlocalize test or VinDr test in this route;
- never choose checkpoints from downstream or external metrics.

## Failure policy

Implementation failures preserve their logs and allow at most one
identity-preserving repair before rerunning the failed gate from zero.

Scientific failures trigger a development-only case study. Only diagnostics
already listed in the protocol may run. Thresholds, data identity, teacher
identity, hard targets, and protected evaluation surfaces never change. A
failed strict gate may end in NO-GO; success is not guaranteed or manufactured.

## Validation

```bash
python -m unittest discover -s extensions/vivid_spd_clean/tests -v
python extensions/vivid_spd_clean/scripts/smoke_vivid_spd_clean.py
python extensions/vivid_spd_clean/scripts/audit_vivid_spd_lock.py
```

## Coding and security

- Python: four spaces, `snake_case`, `CapWords`, `pathlib.Path`.
- Missing or corrupt images fail fast.
- All random splits are patient-aware and deterministic.
- Do not commit medical images, patient data, model weights, checkpoints,
  caches, credentials, or generated run outputs.
- Keep `task_plan.md`, `findings.md`, and `progress.md` current.
