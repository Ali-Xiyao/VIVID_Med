# BiVES-CXR C6F MS-CXR pre-open geometry execution log

**Date:** 2026-07-18

**Authority:** `C6F_MS_CXR_POST_C5_EVALUATION_AUTHORITY_20260718.md`

**Result:** `FAIL_PREOPEN_GEOMETRY_NO_MODEL_ACCESS`

## Authorization boundary

The user explicitly authorized a separate post-C5 local Qwen3.5-2B model
evaluation. The new authority records `model_evaluation_authorized=true` and
does not modify the frozen C6E intake JSON, whose value remains false because
that artifact only certifies data/license intake.

MS-CXR is positive-only for the two locked findings. Therefore the frozen C6F
question is an expert-box target-versus-connected-control mechanism test. No
AUROC, AUPRC, accuracy, B0 comparison, clinical validation, training, tuning,
statement change, or 4B/9B scale-up is permitted.

## Implemented artifacts

- `bives_cxr/c6_ms_cxr_eval.py`: 29-row manifest, patient hashing, 45-box
  aggregation, geometry, dataset-lock, patient-bootstrap, and frozen survival
  gate contracts.
- `scripts/prepare_bives_c6_ms_cxr_evaluation.py`: local score-free manifest,
  geometry-mask, and dataset-lock builder.
- `scripts/evaluate_bives_c6_ms_cxr.py`: local JPG Qwen3.5-2B evaluator with a
  one-time marker, frozen artifact/model checks, no classification metrics, and
  a fail-closed 29/29 geometry prerequisite.
- `tests/test_bives_c6_ms_cxr_eval.py`: manifest/count/tamper/gate contracts.

## Score-free execution

```powershell
python scripts\prepare_bives_c6_ms_cxr_evaluation.py --geometry-workers 8
```

The manifest passes the complete official-test intake boundary:

- 29 rows, 29 patients, 29 studies, and 29 images;
- Consolidation 15 rows / 25 boxes;
- Pleural Effusion 14 rows / 20 boxes;
- canonical training statements only; released sentence text is represented
  by a hash and is not used as the query;
- strict C6E canonical intake identity and release/image hashes remain bound.

The frozen connected-control geometry reaches 28/29. One hashed Consolidation
row fails:

```text
sample_id=ms_cxr_338b3448b9149833dc960d6a34f7cedd64b0a4072d7a66a3f56aad8320308df7
target_area_pixels=25050
target_area_fraction=0.14990664496361547
box_area_quartile=3
failure=no exact-area connected control has the target coordinate zone
```

The predeclared authority requires all 29 rows and permits no geometry
exclusion. The dataset lock therefore reports `status=fail_geometry` and
`evaluation_gate_open=false`.

## Ignored artifact hashes

| Artifact | SHA-256 |
| --- | --- |
| patient manifest | `ba31d6e9e2cefe55effaef838a2f7cc8bf68d5c07021f22f2614782082f4f711` |
| geometry rows | `dbde2e628f7e67db05a815c9110b165b427cbe9c3d837741422febbb02ad2f84` |
| geometry lock | `c8ee9e4cb03bb8a8b068d08165b645591d1905a1675f89da246aa4b566301d91` |
| dataset lock | `9a3cc26b982536950b01eedfdde73f596f569338ca64ff706a83545a9758f073` |
| dataset-lock canonical identity | `59705861a3985053e271e1ab3b0a525c37b8e4670ee794fdce1d7b158e2e4db3` |

## Fail-closed launch proof

```powershell
python scripts\evaluate_bives_c6_ms_cxr.py
```

The evaluator stopped with `C6F requires 29/29 score-free geometry rows`.
It did so before the one-time opening marker, JPG decode, Qwen load, checkpoint
load, GPU use, score generation, or result-file creation.

The evaluation-only YAML is archived under `refine-logs/` rather than the
training-only `configs/bives_cxr/*.yaml` surface. Its SHA-256 is
`5bfd243c7c3a42a113e5e5b50b171d988cecb4a1d6a1426b4c94ad7a8e1ffb5a`.

## Validation

- C6F contract tests: 4/4 passed.
- Full active BiVES suite: 133/133 passed.
- Synthetic CPU smoke: finite gradients, normalized probabilities, and no flat
  four-state head.
- `py_compile` and `git diff --check`: passed.
- The ignored manifest, masks, and locks remain under `local_runs/`; no medical
  image, patient artifact, model weight, cache, or score is publishable.

## Decision

C6F is `FAIL_PREOPEN_GEOMETRY_NO_MODEL_ACCESS`. The authorized model evaluation
was not executed because its first prerequisite failed. The connected-control
definition and 29-row denominator must not be changed after this audit merely
to obtain a score. Qwen3.5-4B/9B remain blocked. A different research question
or control family requires a new explicit authority rather than an edit to this
failed protocol.
