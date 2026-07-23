# Experiment Plan

Authority: `../audit/VIVID_SPD_CLEAN_EXPERIMENT_PROTOCOL.md`

| Stage | Runs | Input identity | Output authority | Unlock |
| --- | --- | --- | --- | --- |
| S0 | audits and CPU smoke | frozen lock and manifests | `readiness.json` | S1 |
| S1 | prefix4, then SPD4x2 | 256 identical rows | per-arm `summary.json` | S2 if both pass |
| S2 | prefix4, then SPD4x2 | 20k train + frozen val | best vision checkpoint and summary | S3 if both learn |
| S3 | paired linear probes | same CheXpert train/expert-dev rows | paired metrics and verdict | S4 or diagnostic |
| Diagnostic | prefix8 and/or SPD-no-ortho only as authorized | development only | case-study report and new lock | one repaired rerun or terminal |
| S4 | 3 seeds, then full MIMIC | unchanged primary identity | multi-seed paired evidence | S5 |
| S5 | frozen external evaluation | preregistered mapping | external result package | manuscript |

Every stage is sequential and fail-closed.
