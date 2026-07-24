# Experiment Tracker

| ID | Stage | Arm | Status | Promotion |
|---|---|---|---|---|
| G0 | contract | all | PASS | 21,212 rows; hashes/assets/patient split pass |
| G1-A0 | overfit | direct schema | PASS | repaired run reproduced acc 0.9932; NLL reduction 90.29% |
| G1-A1 | overfit | free-text Qwen | PASS | step 550; acc 0.9834; NLL reduction 97.28% |
| G1-A3 | overfit | VIVID-GDS | RUNNING | step 150; token acc 0.9476; schema acc 0.7355 |
| G2-A0 | 20k | direct schema | LOCKED | G1 pass |
| G2-A1 | 20k | free-text Qwen | LOCKED | G1 pass |
| G2-A2 | 20k | frozen UMS prefix4 | REUSE_CANDIDATE | hash parity |
| G2-A3 | 20k | VIVID-GDS | LOCKED | G1 pass |
| G3 | probe | A0–A3 | LOCKED | all G2 ready |
| G4 | verdict | three comparisons | LOCKED | frozen thresholds |
