# Phase C synthetic development result

**Date:** 2026-07-19
**Status:** `COMPLETE_NONFORMAL_TEST_CLOSED`

## Scope

Phase C established the executable audit contracts without opening a real
dataset or the CheXlocalize test split. The user separately authorized local
model loading and both workstation GPUs. That opening was used only for a
Qwen3.5-2B synthetic-image interface and determinism gate.

## Implemented surface

- target-specific matched controls `C_X` and `C_E`;
- separate localization and causal-specificity metrics;
- image-space perturbation-strength diagnostics for local mean and masked
  Gaussian blur;
- patient-cluster bootstrap, operator-specific summaries, cross-operator sign
  agreement, and worst-case results;
- a fail-closed precomputed-row CLI that rejects test-like rows;
- a frozen Qwen3.5-2B first-token Yes/No statement scorer;
- deterministic 4x4 local-mean occlusion with row-major top-1 region selection.

## Local execution evidence

The model-free synthetic smoke generated 16 rows across 8 synthetic patients
and two operators. It includes both a positive localization/causality relation
and an inverse relation so the analysis is tested in both directions.

The real-model synthetic gate used Qwen3.5-2B snapshot
`6b57c58ceb3c97d4199d4e8e32b54f6ff11d0cf3a8bde3a4e940d5775ae12120`.
It ran once on each local NVIDIA RTX 3090 using BF16/eager attention. Both runs
completed with peak allocated CUDA memory `6,534,860,800` bytes. After removing
the device field and its dependent top-level hash, both outputs have identical
canonical SHA-256
`4c7931088e96a5a6fa1fb619d7366bc30130063960da58d475e0754a767c1cf4`.
The complete rows and occlusion explanation are equal across GPUs.

The complete active contract suite passes `157/157`. The preserved BiVES CPU
smoke retains finite gradients and no flat state head. Two independent
precomputed-row CLI replays also match at development-lock canonical SHA-256
`159015e8c0dedd3c6f1cbbd6b6116f8638f0f6161eaaeb6f735b238fc33c7deb`.

The synthetic image produced negative `CS_X` and `CS_E` under both operators.
That is an interface-smoke observation only; it is not a medical, performance,
or model-quality result.

## Execution commands

```powershell
python scripts/smoke_qwen35_localization_causality.py `
  --model-path H:/Xiyao_Wang/001_models/Qwen3.5-2B `
  --device cuda:0 --dtype bf16 --grid-size 4 `
  --output-dir local_runs/cxr_localization_causality/qwen35_synthetic_gpu0

python scripts/smoke_qwen35_localization_causality.py `
  --model-path H:/Xiyao_Wang/001_models/Qwen3.5-2B `
  --device cuda:1 --dtype bf16 --grid-size 4 `
  --output-dir local_runs/cxr_localization_causality/qwen35_synthetic_gpu1
```

Generated JSON stays under ignored `local_runs/`; no model weight, image,
patient identifier, credential, or runtime output is committed.

## Boundary and next gate

- `formal_result=false`;
- `test_opened=false`;
- no real patient image or annotation was read;
- no CheXlocalize package was downloaded or opened;
- no threshold was selected from a real outcome;
- no server, SSH, Slurm, C6 replay, or BiVES 4B/9B route was used.

The next real-data action is not an automatic launch. It requires an available
development dataset package plus a separately frozen data/model/explanation/
operator identity and a data opening that distinguishes development from the
one-time test lock.
