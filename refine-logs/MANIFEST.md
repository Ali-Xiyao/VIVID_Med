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
| Connected-control plan | `CONNECTED_CONTROL_RESCUE_PLAN_20260718.md` | `CONNECTED_CONTROL_RESCUE_PLAN.md` | `a9b29fff247b47381aae09148b70385ec32ff973d297dc4d5b04aa03889dd30c` | byte-identical; C1-C3 pass; C4 in progress |
| Connected-control tracker | `CONNECTED_CONTROL_RESCUE_TRACKER_20260718.md` | `CONNECTED_CONTROL_RESCUE_TRACKER.md` | `88eb184e5a251e839e5729b89f9c560a7f79f222c9bffd1ed800abde17db90b7` | byte-identical; C001-C004 pass; C005 in progress |
| C1/C2 execution log | `CONNECTED_CONTROL_C1_C2_EXECUTION_LOG_20260718.md` | none | `7e1e9c317d1560a5d369fefa034253cd0732f78eb4f82c0f111ff32bc47096c8` | 98/98 tests; 375/377 geometry pass; full rows replay identical |
| C3 execution log | `CONNECTED_CONTROL_C3_EXECUTION_LOG_20260718.md` | none | `ffc59f9872b65e4345dbf05e073b317047c86cd646f69a079ce680b19170bbca` | 101/101 tests; zero replay error; C4 estimate 0.2461 h |

The user accepted this candidate by replying `继续` on 2026-07-18. C001-C004
are complete-pass. C005/C4 is authorized only on the frozen Qwen3.5-2B B2 and
VinDr-train protocol-design surface; confirmation, test, training, scale-up,
and later rows remain blocked.
