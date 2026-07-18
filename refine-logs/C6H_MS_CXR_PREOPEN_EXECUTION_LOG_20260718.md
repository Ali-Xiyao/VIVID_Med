# BiVES-CXR C6H MS-CXR pre-open execution log

**Date:** 2026-07-18

**Authority:** `C6H_MS_CXR_ONE_TIME_EVALUATION_AUTHORITY_20260718.md`

**Status:** `READY_FOR_PREOPEN_COMMIT_NO_MODEL_OPENED`

## Purpose

Prepare a separate, one-time, local Qwen3.5-2B evaluation after the frozen C6G
29/29 score-free geometry pass. This phase does not modify or rerun C6F/C6G and
does not authorize Qwen3.5-4B/9B.

## Frozen inputs

| Artifact | SHA-256 |
| --- | --- |
| C6G geometry lock file | `1b85ad241a0c8268bd71868945956c3a729fa966458308a31b3fdcb28259f8fc` |
| C6G geometry lock canonical identity | `6271ba51e8442baad92126473513b0b901619403a4e22c353e455395ec801752` |
| C6G geometry rows | `6b03e6854a3151f85e22a8f2078bec842df8de9c629b0452dd81a9005abaa50e` |
| C6G candidate certificates | `26f694a990f83c255c6a9fa6382c1883663d4274b87e646a133ee0768d647a1c` |
| 29-row MS-CXR manifest | `ba31d6e9e2cefe55effaef838a2f7cc8bf68d5c07021f22f2614782082f4f711` |
| Qwen3.5-2B snapshot | `6b57c58ceb3c97d4199d4e8e32b54f6ff11d0cf3a8bde3a4e940d5775ae12120` |
| B2 step-450 checkpoint | `09c2f77313027ca313f4b03c5553f90d3d7d57436e960888466d2712e9705480` |
| Training-cache lock | `503ead96a0de948b56361b8097bd14cc1ba0942878b56cb3aa0ba2e39d3ec0f2` |
| Frozen B2 training config | `248d57d9a62e77acf36c6c0809428e17c1a3b37bb00741383b94db51ab4d395f` |

The model snapshot was freshly rehashed locally before the pre-open commit and
matches the frozen identity. Every C6F immutable identity embedded in C6G also
matches its current file.

## New C6H implementation identities

| Artifact | SHA-256 |
| --- | --- |
| Authority | `1eb5088ff4f71802e112c2579d086ed57eb739406c10a18c5c8c3f21e06c5302` |
| Config | `25e298a5b975ee0cba2c8549401511038b9b11d4a50db17008c3e914ac8a9449` |
| Lock/contracts module | `bfa0f74529156fd08588f1fa1cd5c8a22284a2ef1643d36e2b99d0cbb6532ddc` |
| Pre-open lock entrypoint | `7fdb319f06d88efe662c6d0c8a6b3409abdc4fb28ee9d9cd72e31be17fb9e3c1` |
| One-time evaluator | `5d6e2bdadfba15d829fc4b9165075645a7c0cd90b7749e49c02f8c87633e19dd` |
| Contract tests | `f9a1be9ea8200cf1a9ff6ae015abf935ec4e258ba45c8c0ac730ee9d15127207` |

The frozen C6F evaluator/module are imported read-only and were not edited.

## Score-free validation

- New C6H contracts: 4/4 passed.
- Full active BiVES suite: 141/141 passed.
- Synthetic CPU smoke: finite gradients, normalized probabilities, and no flat
  four-state head.
- `py_compile` and `git diff --check`: passed.
- Real 29-row dry lock rehearsal: passed with 29 patients, 29 studies, 29
  images, 15/14 findings, 25/20 boxes, all 29 C6G masks and C6F immutable
  hashes verified.
- Qwen3.5 snapshot hashing was read-only; no model/checkpoint was loaded.

## Opening rule

After this exact implementation is committed, the ignored C6H pre-open lock
must bind the clean commit and revalidate all sources/artifacts. GPU1 must be
rechecked and have at least 20 GiB free. Only then may the evaluator create its
one-time opening marker and load Qwen3.5-2B. A terminal pass or fail ends this
rescue route; no tuning, rerun, control edit, or scale-up follows.

## Incidents

1. Initial inspection guessed a nonexistent evaluator filename; repository
   inventory located the frozen `scripts/evaluate_bives_c6_ms_cxr.py` entry.
2. A combined snapshot hash command exceeded a short 30-second wrapper; the
   isolated long-timeout rehash completed and matched the frozen identity.
3. The first real dry lock assumed a single-stage C6G canonical hash. The C6G
   writer intentionally hashes the geometry summary first and then the extended
   provenance lock. C6H now reproduces that frozen two-stage identity exactly.
