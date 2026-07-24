# Experiment Tracker

| ID | Stage | Arm | Status | Promotion |
|---|---|---|---|---|
| G0 | contract | all | PASS | 21,212 rows; hashes/assets/patient split pass |
| G1-A0 | overfit | direct schema | PASS | repaired run reproduced acc 0.9932; NLL reduction 90.29% |
| G1-A1 | overfit | free-text Qwen | PASS | step 550; acc 0.9834; NLL reduction 97.28% |
| G1-A3 | overfit | VIVID-GDS | PASS | step 400; token acc 0.9899; schema acc 0.9872 |
| G2-A0 | 20k | direct schema | PASS | 3000 steps; schema NLL 0.5081; acc 0.8162 |
| G2-A1 | 20k | free-text Qwen | PASS | 3000 steps; token NLL 0.4801; acc 0.8200 |
| G2-A2 | 20k | frozen UMS prefix4 | REUSE_CANDIDATE | hash parity |
| G2-A3 | 20k | VIVID-GDS | PASS | 3000 steps; token NLL 0.0770; schema acc 0.8210 |
| G3-A0 | probe | direct schema | PASS | macro AUROC 0.8657; macro AUPRC 0.6894 |
| G3-A1 | probe | free-text Qwen | PASS | macro AUROC 0.8605; macro AUPRC 0.6877 |
| G3-A2 | probe | frozen UMS prefix4 | PASS | reused macro AUROC 0.8592; macro AUPRC 0.6909 |
| G3-A3 | probe | VIVID-GDS | PASS | macro AUROC 0.8695; macro AUPRC 0.6983 |
| G4 | verdict | three comparisons | TERMINAL_NO_GO | A2-A1 fail; A2-A0 fail; A3-A2 pass |
