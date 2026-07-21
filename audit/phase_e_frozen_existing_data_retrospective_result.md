# Phase E frozen existing-data retrospective result

## Verdict

Phase E completed a deterministic, aggregate-only reuse of two immutable prior
evidence sources:

- VinDr-CXR C5 as prior-exposed supplemental, image-level evidence;
- MS-CXR C6I as 29-patient, positive-only frozen external sensitivity
  evidence.

This is not a new model evaluation. No model was loaded, no GPU was used, no
image or mask was regenerated, no score was computed, and no CheXlocalize
validation or test identity was opened.

## Aggregate result

| Dataset | Role | Unit | Operator | Finding | N | Mean localization gain | Mean TCIG | Spearman rho |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| VinDr-CXR | supplemental, prior exposed | image | local mean | consolidation | 59 | 0.199711 | 0.038723 | 0.165868 |
| VinDr-CXR | supplemental, prior exposed | image | local mean | pleural effusion | 318 | 0.211369 | 0.027395 | -0.128227 |
| VinDr-CXR | supplemental, prior exposed | image | masked Gaussian blur | consolidation | 59 | 0.199711 | 0.031815 | 0.308416 |
| VinDr-CXR | supplemental, prior exposed | image | masked Gaussian blur | pleural effusion | 318 | 0.211369 | 0.008700 | 0.097608 |
| MS-CXR | frozen external sensitivity | patient | local mean | consolidation | 15 | 0.087617 | 0.004480 | 0.157143 |
| MS-CXR | frozen external sensitivity | patient | local mean | pleural effusion | 14 | 0.096165 | -0.009099 | -0.019780 |
| MS-CXR | frozen external sensitivity | patient | masked Gaussian blur | consolidation | 15 | 0.087617 | -0.015523 | 0.117857 |
| MS-CXR | frozen external sensitivity | patient | masked Gaussian blur | pleural effusion | 14 | 0.096165 | 0.026064 | 0.195604 |

The association direction is not stable across the eight dataset/operator/
finding cells. Better frozen top-K localization gain therefore does not
uniformly predict stronger target-versus-control causal specificity in these
existing data. This is descriptive evidence only; no cell is an unbiased
estimate for the planned primary matrix.

## Identity and replay

- normalized frozen operator rows: `812`;
- aggregate identifier-free cells: `8`;
- canonical aggregate artifact SHA-256:
  `b5b92f3213760de69c49a7e379fa431fc590ec9d44cc9bc92edfebcea73cab64`;
- aggregate rows file SHA-256:
  `c8b070975d2f149f5c333afd0edc6b7bfa39c1620b1eaf7ae7e7119bb628f021`;
- summary file SHA-256:
  `77dd794465ffa30f0fdf28c4f9becbca3c88d4ae7e8ef30f3c1ca7dc6bb12d0b`.

Two independent executions produced byte-identical aggregate rows, summary,
and Markdown result.

## Non-substitution boundary

The frozen C5/C6I rows predate the active three-region protocol. They contain
one expert-target/control contrast and do not contain separately intervened
expert and explanation regions with distinct controls (`X/C_X` and `E/C_E`).
They are therefore serialized as `frozen_existing_data_retrospective_v1`, not
as `cxr_localization_causality_audit_v1`.

CheXpert, NIH, and general MIMIC-CXR data may support later preprocessing or
cohort robustness work, but they do not replace an expert localization target.
Non-CXR local datasets remain outside this audit. CheXlocalize approval is not
needed for this completed retrospective, but its unopened test remains the
planned one-time primary evaluation surface.
