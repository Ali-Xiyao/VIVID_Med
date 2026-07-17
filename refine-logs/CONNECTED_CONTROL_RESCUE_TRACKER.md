# BiVES-CXR Coordinate-Zone Connected-Control Rescue Tracker

**Status:** C3_IN_PROGRESS
**Authority:** `CONNECTED_CONTROL_RESCUE_PLAN.md`
**Predecessor status:** `EXPERIMENT_TRACKER.md` remains
`STOPPED_R002_GEOMETRY_FAIL`

| ID | Milestone | Question | Model / split | Locked gate | Priority | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C001 | C0 | Is the new control-family authority explicitly accepted? | no model / no data | explicit user acceptance | MUST | COMPLETE_PASS | user replied `继续` on 2026-07-18; old R002 remains failed |
| C002 | C1 | Does the frozen connected-control implementation satisfy every synthetic contract? | no model / synthetic masks | exact area, containment, target-disjoint, one 4-connected component, same coordinate zone, deterministic replay; full active tests/smoke pass | MUST | COMPLETE_PASS | 98/98 active tests and synthetic smoke passed; stopped translation code remains intact |
| C003 | C2 | Is geometry feasible without selective exclusions? | no model / protocol_design positives | >=95% overall and per finding; >=90% every finding-area quartile; zero invariant failures | MUST | COMPLETE_PASS | 375/377 overall; 62/62 consolidation; 313/315 effusion; lowest quartile 76/78; rows replay byte-identical |
| C004 | C3 | Can the frozen 2B checkpoint be replayed locally within the compute cap? | frozen Qwen3.5-2B B2 / 16 protocol_design images | identity chain and replay tolerance pass; estimated C4 <=4 local GPU h | MUST | IN_PROGRESS | no training or selection |
| C005 | C4 | Does target effect exceed connected-control effect under both primary operators? | frozen Qwen3.5-2B B2 / protocol_design positives | positive mean TCIG both findings/operators; >=1 positive CI per finding; high-area nonnegative; >=60% positive-image TCIG | MUST | BLOCKED_C004 | local-mean and blur co-primary; zero-fill diagnostic only |
| C006 | C5 | Does the frozen result survive one-time internal confirmation? | frozen system / rescue_confirm once | full C4 gate plus no per-finding polarity below B0 | MUST | BLOCKED_C005 | no post-confirmation changes or reruns |
| C007 | C6 | Is an independent patient-grouped expert-region final set authorized? | frozen system / new final dataset | separate data authority and patient-level lock | MUST FOR PAPER | BLOCKED_DATA | no current dataset authority |

## Immutable boundaries

- C001-C007 are new row IDs and do not overwrite R001-R011.
- R001 remains complete-pass; R002 remains complete-fail-hard-stop; R003-R010
  remain not run under the stopped predecessor.
- VinDr test remains prohibited and `rescue_confirm` remains unopened through
  C004.
- No active row permits Qwen3.5-4B/9B, training, decoder/K/selector/loss
  changes, server work, or a patient-level claim.
