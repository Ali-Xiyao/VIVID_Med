# BiVES-CXR Candidate Rescue Planning Manifest

**Status:** STOPPED_R002_GEOMETRY_FAIL
**Created:** 2026-07-18
**Base commit:** `7c3e7ed`
**Parent authority:** `../BiVES_CXR_MIA_TMI_ready_proposal.md`

This manifest registers the accepted candidate authority and its CPU-only
R001/R002 execution. R002 failed its geometry survival gate, so it does not
authorize model loading, training, VinDr-test reuse, Qwen3.5-4B/9B scaling,
server execution, or a new clinical claim.

| Artifact | Versioned file | Fixed alias | SHA-256 | Verification |
| --- | --- | --- | --- | --- |
| Experiment plan and verdict | `EXPERIMENT_PLAN_20260718.md` | `EXPERIMENT_PLAN.md` | `b7202dd1ab08d6b4109d779e452b935b19646cfbb0f3b6479cbadb766fdb6a0a` | versioned and fixed files are byte-identical |
| Experiment tracker and stop state | `EXPERIMENT_TRACKER_20260718.md` | `EXPERIMENT_TRACKER.md` | `a4685ec4ed9c7fa716dbd9b318395fe678003c1c343b2c630eb434fa413e12e8` | versioned and fixed files are byte-identical |
| R001/R002 execution log | `R001_R002_EXECUTION_LOG_20260718.md` | none | `8f9c8dcee8418f332ad7094e9dfed5a56c08b2544c809f91310277f64b799922` | R001 pass; R002 hard-stop fail; final provenance replay recorded |

## Review gate

- R001 is complete-pass; R002 is complete-fail-hard-stop; R003 onward did not
  run or remain dependency/data blocked.
- VinDr test remains frozen as a diagnostic-only surface after E10.
- VinDr train may support image-disjoint development only; the public DICOMs
  expose no patient, study, or series identifier.
- An independent patient-grouped final dataset remains unavailable.
- A new continuation requires a separately reviewed control-family authority;
  it cannot lower the 90% threshold or reuse VinDr test for selection.

## Accepted connected-control candidate

The stopped R001/R002 package above remains immutable. A separate candidate
authority now defines an exact-area, target-disjoint, 4-connected control from
the same coarse content-coordinate zone. It explicitly weakens exact
target-shape matching and does not call the coordinate bins true anatomy.

| Artifact | Versioned file | Fixed alias | SHA-256 | Status |
| --- | --- | --- | --- | --- |
| Connected-control plan | `CONNECTED_CONTROL_RESCUE_PLAN_20260718.md` | `CONNECTED_CONTROL_RESCUE_PLAN.md` | `c5100a32d8c5b9f9858d37c73538695849413f80fa37ee45c471703c49ee844e` | byte-identical; stopped at C5 confirmation polarity gate |
| Connected-control tracker | `CONNECTED_CONTROL_RESCUE_TRACKER_20260718.md` | `CONNECTED_CONTROL_RESCUE_TRACKER.md` | `66688af2c3f009d3078412efdb44f834205e8edda04c9e9a852ef3d4ec3435b5` | byte-identical; C001-C005 pass; C006 final-stop fail; C007 blocked with no eligible local candidate |
| C1/C2 execution log | `CONNECTED_CONTROL_C1_C2_EXECUTION_LOG_20260718.md` | none | `7e1e9c317d1560a5d369fefa034253cd0732f78eb4f82c0f111ff32bc47096c8` | 98/98 tests; 375/377 geometry pass; full rows replay identical |
| C3 execution log | `CONNECTED_CONTROL_C3_EXECUTION_LOG_20260718.md` | none | `ffc59f9872b65e4345dbf05e073b317047c86cd646f69a079ce680b19170bbca` | 101/101 tests; zero replay error; C4 estimate 0.2461 h |
| C4 execution log | `CONNECTED_CONTROL_C4_EXECUTION_LOG_20260718.md` | none | `4a4dfc4ebf2861ae179ac481ffb578412a61d8185aecdf8934cb62280592d67f` | 106/106 tests; both co-primary operators pass every C4 gate |
| C5 execution log | `CONNECTED_CONTROL_C5_EXECUTION_LOG_20260718.md` | none | `b1f8d9ca51c822dd674dc6c66bdbb9142962161847650a70873f905c252e6804` | mechanism replicates; consolidation AUPRC below B0; final stop |
| C6 data-authority inventory | `CONNECTED_CONTROL_C6_DATA_AUTHORITY_INVENTORY_20260718.md` | none | `2453216d654290f6bb32cfc3b32edacc15c78040aa55bf0376295a654768651f` | read-only metadata audit; no eligible local patient-identified two-finding expert-region candidate; no run authorized |

The user accepted this candidate by replying `继续` on 2026-07-18. C001-C005
are complete-pass. C006/C5 opened the image-disjoint VinDr-train
`rescue_confirm` surface exactly once and failed its frozen per-finding polarity
gate because consolidation B2 AUPRC fell below B0. This route is final-stopped;
VinDr test, training, result-driven tuning/reruns, scale-up, and later rows
remain blocked.

C007/C6 was subsequently inspected as a read-only local data-authority gate.
NIH supplies patient-linked Effusion boxes but no Consolidation boxes;
MIMIC-CXR and CheXpert supply patient keys and both finding labels but no local
expert-region annotations. VinDr remains excluded. C007 therefore stays
`BLOCKED_DATA_NO_ELIGIBLE_LOCAL_CANDIDATE`, and this bookkeeping does not
reopen the C5 final stop.
