# BiVES-CXR C6I actual-input recovery authority

**Date:** 2026-07-18

**Status:** `AUTHORIZED_SCORE_FREE_GEOMETRY_AND_ONE_REPLACEMENT_LOCAL_QWEN35_2B_OPENING`

**Execution surface:** local workstation only

## Independent authorization

The user explicitly confirmed approved PhysioNet credentials, required CITI
training, and the applicable DUA, then authorized continuation after the C6H
pre-score pixel-alignment stop. This separate authority permits:

```text
actual_input_geometry_rebuild_authorized=true
model_evaluation_authorized=true
replacement_one_time_execution_authorized=true
```

This is not a C6H rerun. C6F, C6G, and failed C6H remain immutable historical
evidence. C6I uses a new geometry lock, pre-open lock, opening marker, output
directory, and terminal identity.

## Frozen coordinate repair

For every one of the 29 released positive rows, and before any model access:

1. verify the exact bound JPG byte hash;
2. read its actual dimensions, required to remain 224x224;
3. map every released x coordinate by `actual_width/native_columns` and every
   y coordinate by `actual_height/native_rows`;
4. rasterize the transformed boxes in the actual JPG coordinate space;
5. map that actual-image mask through the existing deterministic 448x448 Qwen
   input transform;
6. regenerate the exact-area, target-disjoint, within-content, one-connected
   continuous-location control uniformly with the frozen C4/C5 thresholds.

No row-specific repair, exclusion, box edit, threshold change, control-family
change, statement rewrite, operator change, score access, or outcome inspection
is allowed during this geometry phase. The model gate opens only if all 29 rows
pass and the artifacts replay deterministically.

## Replacement evaluation boundary

After the score-free 29/29 gate and a clean committed pre-open identity, C6I
may open the frozen Qwen3.5-2B model and B2 step-450 exact-K=16 checkpoint once.
The canonical statements, local-ring mean, masked Gaussian blur, patient
bootstrap, and C4/C5 survival thresholds remain unchanged from C6H.

This remains a positive-only nonformal mechanism evaluation. It is not
classification, clinical validation, training, tuning, C5 reversal, or
authorization for Qwen3.5-4B/9B. A terminal C6I pass or fail ends this route.
