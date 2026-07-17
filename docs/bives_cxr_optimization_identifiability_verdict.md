# BiVES-CXR Optimization-Identifiability Verdict

## Decision

The fixed 400-step Qwen3.5-2B state-only diagnostic failed its preregistered
train-fit survival gate. It reached train accuracy `0.7917`, not the required
`1.0`. The matched full-objective Run B was therefore **not started**.

This is a hard stop for the current weak-label four-state route. It does not
authorize a longer run, learning-rate or loss-weight sweep, Qwen3.5-4B/9B
scaling, decoder replacement, or a magnitude-polarity refactor.

## Frozen inputs and execution

- Authority: `BiVES_next_direction_without_local_clinical_review_2026-07-17.md`
- Proxy freeze: `docs/bives_cxr_proxy_p0_a_freeze.md`
- Model: local Qwen3.5-2B frozen vision encoder
- Data: the same patient-disjoint 48 train / 48 validation proxy rows
- Seed / exact-K / steps: `17` / `16` / `400`
- Objective: state NLL only; all auxiliary loss weights are zero
- Selection: final step only; validation was not used for checkpoint selection
- Runtime: local workstation only; no server, SSH, Slurm, or remote GPU work
- Output: ignored local run
  `local_runs/bives_cxr/qwen35_2b_optimization_state_only_400`

## Frozen-feature readout check

The patient-group-disjoint logistic probe reused the frozen Qwen3.5-2B feature
cache and loaded no model weights. Global S/C AUROC is `0.7889`. Pleural
effusion, pulmonary edema, and consolidation obtain `0.8550`, `0.8075`, and
`0.8000`. These results show usable relative S/C information in the frozen
representation, but they do not establish four-state calibration or clinical
ground truth.

## Run A results

| Metric | Train | Validation |
| --- | ---: | ---: |
| Accuracy | `0.7917` | `0.5000` |
| Macro-F1 | `0.7952` | `0.5139` |
| NLL | `0.7563` | `1.0348` |
| S/C AUROC | `0.9861` | `0.8333` |
| U/I AUROC | `1.0000` | `1.0000` |

Final train recalls are support `8/12`, contradict `10/12`, uncertain `8/12`,
and insufficient `12/12`. The model therefore learned much of the fixed set
but did not satisfy the required exact train-fit gate.

The fixed train trajectory was:

| Step | Accuracy | NLL |
| ---: | ---: | ---: |
| 0 | `0.2708` | `2.7761` |
| 50 | `0.4792` | `1.1270` |
| 100 | `0.5833` | `1.3022` |
| 150 | `0.5000` | `1.0115` |
| 200 | `0.5625` | `0.8764` |
| 250 | `0.6875` | `0.8074` |
| 300 | `0.7708` | `0.7726` |
| 350 | `0.7917` | `0.7582` |
| 400 | `0.7917` | `0.7563` |

## Evidence and gradient diagnosis

The state-only model did learn the requested absolute polarity direction:

- support median signed evidence delta: `+1.9700`;
- contradict median signed evidence delta: `-2.8745`;
- uncertain median signed evidence delta: `+0.7307`;
- insufficient median total evidence: `0.8951`, the lowest state median.

The decoder-direction audit is also correct: increasing signed evidence lowers
support NLL and raises contradict NLL. The result therefore does **not** support
the hypothesis that the current failure requires magnitude-polarity
factorization.

At step 400, the weighted state-gradient norm is dominated by the gate head
(`264.79` of total norm `265.69`), followed by context (`20.99`), fusion
(`5.79`), evidence head (`2.08`), and statement table (`0.039`). With auxiliary
weights fixed at zero, no state-versus-auxiliary conflict can be inferred from
Run A. The remaining failure is bounded to the current optimization/readout,
selector, or effective-capacity path, with the largest residual errors in
support and uncertain.

## Step-0 provenance repair

The first resume attempt reran the step-0 audit after restoring the step-50
checkpoint and overwrote the original step-0 files. The trainer now guards the
initial audit with `step == 0`. A separate initialization-only replay performed
zero optimizer steps and exactly reproduced the original observed step-0
metrics and audit. Its four recovered files were copied into the main run only
after SHA-256 equality was verified against the replay outputs.

Recovered step-0 evidence includes train accuracy `0.2708`, train NLL `2.7761`,
validation accuracy `0.2292`, validation NLL `2.7679`, and total state-gradient
norm `960.72`. This recovery changes no trained weights and does not alter the
Run A verdict.

## Route boundary

- Run B: `not_run_by_gate`.
- Qwen3.5-4B/9B: closed.
- Decoder, exact-K, loss weights, and parameterization: frozen.
- Parser-derived U and synthetic I: mechanism proxies only, not clinical truth.
- Four-state weak-label route: stopped at the failed train-fit gate.
- Allowed next work: prepare a separate public expert S/C evaluation route and
  intervention-defined availability audit without converting absent labels
  into expert contradiction claims.

## Public-data intake status

The local VinDr-CXR 1.0.0 package passed its bounded integrity audit: all
18,006 official entries are present, sampled hashes and DICOM decodes pass,
and the test consensus intake contains 6,000 S/C rows over pleural effusion
and consolidation. See `docs/bives_cxr_public_expert_sc_intake.md`. This data
readiness is separate from the failed model gate and does not authorize the
current checkpoint for expert evaluation.

`CheXpert-v1.0-small/valid.csv` is present, but this file must not be called the
special radiologist expert set until its provenance is explicitly verified.
CheXlocalize was not found in the checked public-dataset locations. Neither
condition is silently filled by parser labels.
