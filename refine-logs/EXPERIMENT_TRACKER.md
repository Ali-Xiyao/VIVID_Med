# BiVES-CXR Selector/Intervention Rescue Tracker

**Status:** STOPPED_R002_GEOMETRY_FAIL
**Date:** 2026-07-18
**Plan:** refine-logs/EXPERIMENT_PLAN.md

| Run ID | Milestone | Purpose | System / variant | Split | Decisive metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M1 | Freeze votes, boxes, hashes, and image-disjoint split | no model | VinDr train | lock/hash/overlap/coverage audit | MUST | COMPLETE_PASS | 1,510 rows; 1,446 SHA-verified train images; zero overlap; exact S/C balance; manifest `bd84cd7c...b1f6514f` |
| R002 | M1 | Validate topology-matched control geometry | no model | protocol_design geometry | area/disjointness/components/perimeter/band match; feasibility >=90% | MUST | COMPLETE_FAIL_HARD_STOP | complete search: overall 89.39%; consolidation 91.94%; pleural effusion 88.89% |
| R003 | M2 | Historical scattered-control reference | frozen Qwen3.5-2B B2 | protocol_design | TCIG, area strata, replay | DIAGNOSTIC | NOT_RUN_R002_FAIL | zero-fill fixed |
| R004 | M2 | Isolate control topology | frozen Qwen3.5-2B B2 + contiguous translated control | protocol_design | per-finding TCIG; high-area TCIG | MUST | NOT_RUN_R002_FAIL | only control topology changes |
| R005 | M3 | Mean-fill robustness | frozen R004 system + local mean fill | protocol_design | per-finding TCIG CI; stratum agreement | MUST | NOT_RUN_R002_FAIL | operator only |
| R006 | M3 | Blur robustness | frozen R004 system + masked Gaussian blur | protocol_design | per-finding TCIG CI; agreement with R005 | MUST | NOT_RUN_R002_FAIL | operator only |
| R007 | M4 | Selector consistency audit | frozen B2 under locked protocol | protocol_design | TCIG by area/consensus/localization; positive fraction | MUST | NOT_RUN_R002_FAIL | no training |
| R008 | M5 | Dense teacher provenance/cache audit | frozen dense B1 teacher | weak MIMIC train/val | cache and delta identity | CONDITIONAL | NOT_RUN_R002_FAIL | only if protocol passes and selector fails |
| R009 | M5 | One dense-to-sparse preservation rescue | Qwen3.5-2B B2 + L_preserve only | weak MIMIC train/val + protocol_design | polarity plus selector gates | CONDITIONAL | NOT_RUN_R002_FAIL | one seed, no sweep |
| R010 | M6 | One-time internal confirmation | frozen B0/B2 and optional R009 | rescue_confirm | all per-finding polarity/TCIG/stratum gates | MUST | NOT_RUN_R002_FAIL | confirmation remained unopened |
| R011 | M7 | Independent patient-level final evaluation | frozen winning system | new external final | patient-level polarity/intervention/localization | MUST FOR PAPER | BLOCKED_DATA | no dataset authority yet |

## Global launch conditions

- Explicit acceptance of the candidate authority.
- Local-only execution.
- VinDr test path guard enabled and tested.
- No unrelated GPU process interrupted.
- Each predecessor row marked complete with hashes and gate decision.
