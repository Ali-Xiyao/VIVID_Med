# VICER-CXR V0 Experiment Plan

**Status:** PREOPEN_CODE_VALIDATION
**Date:** 2026-07-22
**Authority:** `../vicer_cxr/VICER_CXR_method_proposal.md`

## Claim map

| Claim | Minimum convincing evidence | Blocks |
| --- | --- | --- |
| C1: intervention validity is measurable independently of the audited target effect | Four-finding dose-response with disjoint local critic/global verifier/evaluation identities; at least one family passes all frozen validity and positive-gap criteria | V000-V003 |
| C2: single-region failure may reflect coverage/redundancy rather than engineering failure | Only after C1, compare frozen expert box, dilation, multi-box union, and anatomy envelope without changing the verifier/operator family | V100-V102 |

## Run blocks

| ID | Priority | Action | Entry gate | Pass/stop |
| --- | --- | --- | --- | --- |
| V000 | MUST | Freeze proposal, source, data roles, thresholds | ARISE terminal tag exists | source/tests/locks pass |
| V001 | MUST | Build new VinDr-train manifest and score-free geometry | V000 | exact counts, hashes, global role disjointness, 32/32 control masks |
| V002 | MUST | Cache Qwen3.5-2B originals and fit independent heads | V001 | every critic/verifier calibration AUROC >=0.60 |
| V003 | MUST | Run three-family four-strength V0 matrix | V002 | at least one family passes every finding; otherwise stop |
| V100 | CONDITIONAL | Freeze coverage-redundancy V1 | V003 pass | no new score before new lock |
| V200 | LOCKED | Train evidence coalition | V1 pass | remains unauthorized now |

## Frozen boundaries

All runs are local and use Qwen3.5-2B only. VinDr test, CheXlocalize test,
server, Slurm, selector, U/I, 4B/9B, and predecessor rewrites are prohibited.
VinDr supports only an image-level development claim. Any calibration-head
failure stops before intervention model scoring. Any V0 failure locks V1/V2.
