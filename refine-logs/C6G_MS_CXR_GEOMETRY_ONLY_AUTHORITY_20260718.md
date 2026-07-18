# BiVES-CXR C6G MS-CXR geometry-only authority

**Date:** 2026-07-18

**Status:** `AUTHORIZED_SCORE_FREE_GEOMETRY_ONLY`

**Execution surface:** local workstation, CPU geometry only

## Frozen boundary

This is a new control-family question. It does not edit, rerun, or reopen C6F.
C6F remains permanently frozen as `FAIL_PREOPEN_GEOMETRY_NO_MODEL_ACCESS`.
Under C6G:

```text
model_evaluation_authorized=false
gpu_authorized=false
image_decode_authorized=false
score_access_authorized=false
```

No Qwen/checkpoint load, CUDA access, JPG decode, model score, intervention
score, or evaluation opening marker is permitted. A C6G pass creates geometry
artifacts only. Model evaluation would require a separate C6H authority.

## Immutable input denominator

- MS-CXR v1.1.0 publisher test only;
- 29 rows, patients, studies, and images;
- 15 Consolidation rows / 25 boxes;
- 14 Pleural Effusion rows / 20 boxes;
- native expert boxes, no dilation;
- zero row exclusions;
- byte-identical C6E and C6F authority/config/log/locks.

Frozen C6F identities:

| Artifact | SHA-256 |
| --- | --- |
| C6F authority | `6599212f1c9b4177379196435a65deffd278440e94f4574f3057d4b107bc207c` |
| C6F config | `5bfd243c7c3a42a113e5e5b50b171d988cecb4a1d6a1426b4c94ad7a8e1ffb5a` |
| C6F execution log | `acbe78ac76d08b7ef9d4acd62d1c4d861299ab094631afc77a17985552bb82cd` |
| C6F manifest | `ba31d6e9e2cefe55effaef838a2f7cc8bf68d5c07021f22f2614782082f4f711` |
| C6F geometry rows | `dbde2e628f7e67db05a815c9110b165b427cbe9c3d837741422febbb02ad2f84` |
| C6F geometry lock | `c8ee9e4cb03bb8a8b068d08165b645591d1905a1675f89da246aa4b566301d91` |
| C6F dataset lock | `9a3cc26b982536950b01eedfdde73f596f569338ca64ff706a83545a9758f073` |

## Frozen v2 control

```text
control_version=bives_continuous_location_connected_control_v2
```

Every row uses the same deterministic candidate family. A selected control
must satisfy exact area, target disjointness, content containment, one
4-connected component, geometry-only selection, and zero denominator
exclusions.

The old hard equality between target/control categorical zones is removed.
For target/control content-normalized centroids, define:

```text
d_loc = abs(x_control - x_target) + abs(y_control - y_target)
perimeter_mismatch = abs(log(perimeter_control / perimeter_target))
J = d_loc + 0.10 * perimeter_mismatch
```

The selected candidate is the minimum-J candidate that also passes the two
frozen mismatch limits below.

## Frozen mismatch thresholds

Thresholds come only from the 752 accepted C4/C5 controls, before any MS-CXR
model score exists:

| Source | Accepted rows | Rows SHA-256 |
| --- | ---: | --- |
| frozen C4 protocol-design geometry | 375 | `b94b77bc528033ea44681fce33f868b87c2806d19450a66ff023628fc2e039d9` |
| frozen C5 confirmation geometry | 377 | `58b10019306e63b9ef1d08b94cd01e9540f3591155b3981762bc9657ccf0be41` |

The preregistered limits are the maxima over those accepted rows:

```text
max_location_distance=0.30062962749991123
max_log_perimeter_ratio=0.9737778227918367
centroid_l1_weight=1.0
log_perimeter_ratio_weight=0.10
```

They may not be changed after the MS-CXR geometry build.

## Deterministic candidate family

The uniform v2 search includes 17x17 and 33x33 lattice seeds, valid-component
centroids, distance-transform local maxima, exact translated-target candidates
when connected, and target-boundary connected growth under three frozen
frontier orders. All candidates are geometry-only. Candidate identity,
frontier order, counts, objective, target/control geometry, and selected mask
SHA are recorded.

## Gate and stop rule

```text
geometry_gate=29_of_29
invariant_failures=0
denominator_exclusions=0
```

If any row lacks a qualifying control, C6G stops the MS-CXR route without a
model opening. If all 29 pass, C6G freezes masks, geometry rows, a candidate
certificate, and a geometry/data lock. That pass still does not authorize C6H
or any Qwen3.5 run.
