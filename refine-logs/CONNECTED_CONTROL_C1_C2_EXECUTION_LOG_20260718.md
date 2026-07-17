# BiVES-CXR Connected-Control Execution Log: C1-C2

**Date:** 2026-07-18
**Authority:** `CONNECTED_CONTROL_RESCUE_PLAN.md`
**Execution:** local workstation, CPU only
**Verdict:** C1_PASS_C2_PASS_C3_UNLOCKED
**Formal result:** false
**Patient-level claim:** false

## Purpose

Implement the accepted geometry-only coordinate-zone connected control, prove
its synthetic contracts, and audit its feasibility on the immutable R001
VinDr-train `protocol_design` positives before any model or score access.

## C1: implementation and contracts

- Added `deterministic_coordinate_zone_connected_control_mask` without
  changing the stopped exact-translation implementation.
- Frozen seed set: 17-by-17 normalized lattice plus valid-space component
  centroids.
- Frozen construction: exact-area deterministic 4-neighbour best-first growth,
  target disjointness, content containment, same vertical/horizontal
  coordinate-zone centroid, and geometry-only objective.
- The emitted metadata explicitly sets `true_anatomy_segmentation=false`.
- Added synthetic tests for determinism, input immutability, exact area,
  containment, disjointness, one component, zone identity, target/control
  topology non-equivalence, fixed zone thresholds, and fail-closed insufficient
  content.
- Validation: 98/98 active `test_bives_*.py` tests pass; synthetic BiVES smoke
  passes with finite gradients and no flat state head.

## C2: score-free geometry audit

### Frozen inputs

- R001 manifest SHA-256:
  `bd84cd7ca5384afbcb6228c49331b028a9641dd3dd9011157c2cec75b1f6514f`.
- R001 data-lock SHA-256:
  `4251027b3069b21fb6fb5acd6bc02bf003206fbcfffb6d045abd2289ea2ac409`.
- C2 runtime accepted-plan SHA-256:
  `7bfc817863d2b0734fa6a536af7bc0a6e8fd4bf56cb0a2a59359e983adb1ebc5`.
- C2 runtime tracker SHA-256:
  `88d6b14bb247137b7db074d539965d947800ef5031d6cd5b2f9b3052156ec998`.
- Protocol module SHA-256:
  `dba8e2d35586355a4e1cdec568af74fc7ea5010ed7d7ad937b1620ef98d04beb`.
- Audit script SHA-256:
  `c85e3b8fe24528067cad9f710a2eb0fde3175dec183b4284b7fd2a2b8d428601`.
- Source base commit: `35146b313a3118efb87ecf069bbaaf5c431714a7`.

### First full pass

- Total: 377 protocol-design positives.
- Eligible: 375; excluded: 2.
- Overall feasibility: `375/377 = 99.47%` (gate: at least 95%).
- Consolidation: `62/62 = 100%` (gate: at least 95%).
- Pleural effusion: `313/315 = 99.37%` (gate: at least 95%).
- Lowest finding-area stratum: pleural-effusion quartile 4,
  `76/78 = 97.44%` (gate: at least 90%).
- Invariant failures: 0.
- Both exclusions remain in pleural-effusion quartile 4; one is 2-of-3 and one
  is 3-of-3 consensus. Both use the frozen reason
  `no exact-area connected control has the target coordinate zone`.
- Geometry rows SHA-256:
  `b94b77bc528033ea44681fce33f868b87c2806d19450a66ff023628fc2e039d9`.
- Canonical first-pass lock SHA-256:
  `91bc558f85cae38d143562ca9b08b9d71c1821569df5cbb35d98eb24f68af71b`.
- Wall time: 544.15 seconds with eight CPU workers.
- Stderr: empty.

### Full deterministic replay

The entire 377-row audit was regenerated into a separate output directory
with identical code, locks, plan, tracker, seed, and workers.

- Replay rows SHA-256:
  `b94b77bc528033ea44681fce33f868b87c2806d19450a66ff023628fc2e039d9`.
- First/replay rows are byte-identical.
- Every gate count, stratum table, exclusion reason, source hash, and code hash
  is identical.
- Replay lock SHA-256:
  `659de5a7d3633bb53fe0ea4dc03cc9a2905e75ba7d0390a83b94c3cf5a20c34b`;
  it differs from the first lock only in runtime-dependent wall time.
- Replay wall time: 528.70 seconds. Stderr: empty.

## Access and claim boundary

- `model_loaded=false`
- `scores_accessed=false`
- `image_pixels_accessed=false`
- `forbidden_test_path_accessed=false`
- `rescue_confirm_rows_used=0`
- `formal_result=false`
- `patient_level_claim=false`

## Decision

C1 and C2 pass exactly as preregistered. C3 is unlocked only as a 16-image
local Qwen3.5-2B timing and original-score replay gate. No full intervention,
training, confirmation access, Qwen3.5-4B/9B, or server work is unlocked.
