# Strict VIVID/SPD Clean Extension

This directory rebuilds the last experimentally defensible VIVID-Med question:

> Does a historical four-by-two structured prediction decomposition (SPD)
> projector improve a deployable ViT encoder over the paired hard-UMS prefix
> projector when both are trained through the same frozen Qwen3.5 teacher?

The route intentionally excludes RCSD reliability weighting and every later
VSL/BiVES/MORPH claim. The initial experiment is a bounded Qwen3.5-2B,
MIMIC-CXR development gate. Full data, three seeds, external evaluation, and
larger teachers unlock only after the frozen promotion gate passes.

Start with:

- `audit/VIVID_SPD_CLEAN_EXPERIMENT_PROTOCOL.md`
- `audit/vivid_spd_clean_lock.json`
- `task_plan.md`
- `refine-logs/EXPERIMENT_PLAN.md`
- `refine-logs/EXPERIMENT_TRACKER.md`

Runtime outputs belong under the remote project run root and are ignored by
Git.
