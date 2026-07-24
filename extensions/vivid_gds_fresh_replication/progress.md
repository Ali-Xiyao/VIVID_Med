# Progress

## 2026-07-24

- User approved continuing with the recommended fresh-split replication.
- Created a separate project root so the terminal VIVID-GDS Stage-A authority
  remains unchanged.
- Began a data-eligibility audit before implementing or launching experiments.
- Confirmed the server-side CheXpert image, label, CheXpert-Plus report, and
  historical probe-manifest identities.
- Confirmed that official CheXpert validation and CheXlocalize validation are
  the same exposed 234-image surface and excluded them from the new route.
- Selected CheXpert `train` as the only eligible source for a new,
  patient-disjoint development partition; no model predictions have been
  opened on the candidate partition.
- Added and froze the prospective protocol and machine-readable lock.
- Implemented the deterministic split builder, explicit three-way linear
  probe, paired patient-bootstrap gate, fail-closed readiness audit,
  sequential three-seed queue, and allocation-3066 launcher.
- Generated private manifests on SUES and passed the score-free split audit
  with zero patient/path/protected-surface overlap.
- Passed the independent server prelaunch readiness audit. No
  fresh-development prediction has been produced.
- Validated shell syntax, Python entrypoints, the six inherited VIVID-GDS
  contract tests, and the fresh replication lock tests.
- Committed and pushed the locked replication at Git commit `2882789`.
- Synchronized that source state to the isolated SUES project and launched the
  sequential queue through retained allocation 3066 on `gpu01`.
- Queue R0 passed and R1 began with `seed0_A0_direct`; the first ten GPU
  optimization steps completed normally. The remaining A0/A2/A3 seed pairs
  and all probes remain queued.
- R1 `seed0_A0_direct` completed and passed: validation schema NLL fell from
  `1.10009` to `0.50807` (53.82% reduction), with the locked best checkpoint
  at step 3000. The queue advanced to `seed0_A2_ums`.
