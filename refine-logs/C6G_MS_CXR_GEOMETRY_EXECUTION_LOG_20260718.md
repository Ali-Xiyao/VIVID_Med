# BiVES-CXR C6G MS-CXR geometry execution log

**Date:** 2026-07-18

**Authority:** `C6G_MS_CXR_GEOMETRY_ONLY_AUTHORITY_20260718.md`

**Result:** `COMPLETE_PASS_GEOMETRY_ONLY_NO_MODEL_AUTHORITY`

## Frozen pre-open identity

- Protocol plan SHA-256:
  `e8819e97e7fb11e1d7b3d689f73b536fe466db55a773171cb95ab136128f34f2`.
- Geometry-only authority SHA-256:
  `17305e6887491b179a672d6f67ea1186f06acb8236feef5e0a4f3f6cf62355e0`.
- Frozen threshold artifact SHA-256:
  `69a3bedeb43b65065eab41d28fdefe4870214babebbfa7872f5c5e8146ecb5ab`.
- Final pre-open source commit:
  `db3c033b32c1531503770a684c6900ab56501fe3`.
- C6G geometry module SHA-256:
  `675a30a0a6b7263683f4bedeaa3171b519371ebb467b22b2c9a7b70e15436501`.
- C6G entrypoint SHA-256:
  `f9d4ebf47b6ae1e15a94a61048b9b60cfc3f32b0f52780ea730ab18560c695ef`.

All frozen C6F authority/config/log/manifest/geometry/data-lock hashes were
reverified before and after C6G and remain byte-identical.

## Frozen threshold source

The limits were derived before MS-CXR model access from 752 accepted controls:

- C4 protocol-design: 375 accepted rows, rows SHA-256
  `b94b77bc528033ea44681fce33f868b87c2806d19450a66ff023628fc2e039d9`;
- C5 confirmation: 377 accepted rows, rows SHA-256
  `58b10019306e63b9ef1d08b94cd01e9540f3591155b3981762bc9657ccf0be41`.

Frozen maxima:

```text
max_location_distance=0.30062962749991123
max_log_perimeter_ratio=0.9737778227918367
log_perimeter_ratio_weight=0.10
```

## Score-free execution

```powershell
python scripts\prepare_bives_c6g_ms_cxr_geometry.py `
  --geometry-workers 8 `
  --output-dir local_runs\bives_cxr\c6g_ms_cxr_geometry_final
```

The final committed-identity build completed in 299.7 seconds:

```text
rows=29
eligible=29
infeasible=0
invariant_failures=0
denominator_exclusions=0
evaluation_gate_open_geometry=true
model_evaluation_authorized=false
gpu_authorized=false
image_decode_authorized=false
scores_accessed=false
```

Selected candidate families:

| Family | Rows |
| --- | ---: |
| target-boundary connected growth | 23 |
| 33x33 lattice growth | 5 |
| 17x17 lattice growth | 1 |

The maximum selected location distance is `0.11570502283445364`; the maximum
selected log-perimeter mismatch is `0.9295359586241757`. Both are below their
frozen limits.

The former C6F failure row
`ms_cxr_338b3448b9149833dc960d6a34f7cedd64b0a4072d7a66a3f56aad8320308df7`
selects a target-boundary candidate under
`target_distance_then_l1`, with:

```text
location_distance=0.11570502283445364
log_perimeter_ratio=0.6512731080888864
objective=0.18083233364334228
perimeter_edges=1216
```

No sample-specific rule or denominator exclusion was used.

## Ignored final artifact identities

| Artifact | SHA-256 |
| --- | --- |
| geometry rows | `6b03e6854a3151f85e22a8f2078bec842df8de9c629b0452dd81a9005abaa50e` |
| candidate certificates | `26f694a990f83c255c6a9fa6382c1883663d4274b87e646a133ee0768d647a1c` |
| geometry lock file | `1b85ad241a0c8268bd71868945956c3a729fa966458308a31b3fdcb28259f8fc` |
| geometry lock canonical identity | `6271ba51e8442baad92126473513b0b901619403a4e22c353e455395ec801752` |

The timed-out-wrapper build, the controlled diagnostic build, and the final
committed-identity build have byte-identical geometry rows, candidate
certificates, and all 29 mask files. The wrapper-timed-out build is retained
only as replay evidence because its worker pool continued without foreground
control and its old lock lacks final provenance identities. A separate
single-process replay of the former C6F failure row is also identical to the
8-worker result.

## Validation

- New C6G contracts: 4/4 passed.
- Full active BiVES suite: 137/137 passed.
- Synthetic CPU smoke: finite gradients, normalized probabilities, and no flat
  four-state head.
- Every final mask independently passes exact-area, target-disjoint,
  within-content, and one 4-connected-component checks.
- `py_compile` and `git diff --check`: passed.

## Incidents

1. The first launch used a 10-second shell timeout. The wrapper exited, but its
   child worker pool continued and later completed 29/29. It was not interpreted
   as the final lock because foreground control had been lost and the old lock
   lacked the final implementation/source identities. Its rows, certificates,
   and masks are byte-identical to both controlled builds.
2. The first complete lock omitted C6G implementation/source-commit identities.
   This was treated as a score-free diagnostic only. Identity fields were added
   without changing geometry, committed, and the final build was rerun.
3. A final audit command initially asserted a manually mistyped expansion of
   the short Git SHA. Reading `git rev-parse HEAD` and the lock directly showed
   an exact match; the corrected audit passed without changing artifacts.

## Decision

C6G is complete-pass as a geometry-only protocol. It freezes 29/29 masks and
the geometry lock, but it does not authorize image decoding, Qwen/checkpoint
loading, GPU use, score generation, or a C6H run. A separate explicit C6H
one-time model-evaluation authority is required before any model action.
