# Experiment Tracker

| ID | Stage | Arm | Seed | Status | Gate | Evidence |
| --- | --- | --- | ---: | --- | --- | --- |
| VSC-S0 | S0 | shared | 0 | passed | readiness | strict run `readiness.json` |
| VSC-S1-P0 | S1 | ums_prefix4 | 0 | passed | overfit | accuracy 0.98907 |
| VSC-S1-P1 | S1 | ums_spd4x2 | 0 | passed | overfit | accuracy 0.98761 |
| VSC-S2-P0 | S2 | ums_prefix4 | 0 | passed | pilot | NLL reduction 94.41% |
| VSC-S2-P1 | S2 | ums_spd4x2 | 0 | passed | pilot | NLL reduction 95.85% |
| VSC-S3-P0 | S3 | ums_prefix4 | 0 | reference | probe | macro AUROC 0.859209 |
| VSC-S3-P1 | S3 | ums_spd4x2 | 0 | strict_no_go | probe | delta +0.004641; 3/5 nonnegative |
| VSC-D1-P8 | diagnostic | ums_prefix8 | 0 | negative | sequence budget | macro AUROC 0.857884 |
| VSC-D1-NO | diagnostic | ums_spd4x2_no_ortho | 0 | negative | ortho removal | macro AUROC 0.858864 |
| VSC-D1-ATTN | diagnostic | SPD attention groups | 0 | complete | collapse | cosine 0.0000718 vs 0.992841 |
| VSC-TERM | terminal | route | 0 | terminal_no_go | final | `audit/VIVID_SPD_CLEAN_TERMINAL_RESULT_20260724.md` |

Statuses are updated only from durable run artifacts.
