# Experiment Tracker

| ID | Stage | Arm | Seed | Status | Gate | Evidence |
| --- | --- | --- | ---: | --- | --- | --- |
| VSC-S0 | S0 | shared | 0 | passed | readiness | remote `local_runs/s0_20260723/readiness.json` |
| VSC-S1-P0 | S1 | ums_prefix4 | 0 | pending | overfit | pending |
| VSC-S1-P1 | S1 | ums_spd4x2 | 0 | pending | overfit | pending |
| VSC-S2-P0 | S2 | ums_prefix4 | 0 | locked | pilot | S1 |
| VSC-S2-P1 | S2 | ums_spd4x2 | 0 | locked | pilot | S1 |
| VSC-S3-P0 | S3 | ums_prefix4 | 0 | locked | probe | S2 |
| VSC-S3-P1 | S3 | ums_spd4x2 | 0 | locked | probe | S2 |

Statuses are updated only from durable run artifacts.
