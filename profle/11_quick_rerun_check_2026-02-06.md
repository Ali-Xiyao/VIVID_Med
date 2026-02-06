# 2026-02-06 Quick Rerun Check

## Goal
- Verify whether the core VIVID training idea is still intact.
- Remove the recent "all absent" collapse and re-check parsing + metrics.

## Core architecture check
- `models/vivid_model.py`: LLM is frozen (`requires_grad=False`).
- Trainable modules are still `ViT + Projector` (`get_trainable_parameters()`).
- This means the core method is unchanged: use LLM supervision space to train visual features.

## What was changed for this check
- Config used: `configs/cxr_chexpert_quick.yaml`
- Key settings:
  - `json_missing_state: null`
  - `json_null_state: null`
  - `json_include_all_labels: true`
  - Prompt schema default states set to `null` (not forced `absent`)
  - `num_workers: 0` (to avoid validation deadlock observed before)
  - fixed split: `chexpert_ums_train.jsonl` / `chexpert_ums_val.jsonl`

## Run artifacts
- Training output dir: `outputs/cxr_chexpert_quick_rerun`
- Checkpoints:
  - `outputs/cxr_chexpert_quick_rerun/checkpoints/best.pt`
  - `outputs/cxr_chexpert_quick_rerun/checkpoints/step_100.pt`
- Eval output:
  - `outputs/cxr_chexpert_quick_rerun/checkpoints/best.metrics.quick.json`
- Sample output file:
  - `outputs/cxr_chexpert_quick_rerun/checkpoints/sample_outputs.quick.jsonl`

## Quick eval result (200 val samples)
- `json_success_rate`: `1.0`
- `pred_nan_rate`: `0.8571`
- `macro_f1`: `0.1286`
- `micro_f1`: `0.4313`

## Observations
- JSON parsing is fixed and stable (`json_success_rate=1.0`).
- Model no longer outputs long chat garbage; outputs are mostly valid JSON.
- But prediction is still sparse (`null` too frequent), so classification quality remains weak.
- It is better than the prior collapsed VIVID-v2 run, but still far below ViT baseline.

## Next direction
- Keep JSON structure constraints, but reduce null collapse:
  - rebalance label/state targets in training data,
  - add stronger supervised signal on per-label states,
  - increase training steps after anti-collapse fixes,
  - run fixed-split baseline vs VIVID under same eval script.
