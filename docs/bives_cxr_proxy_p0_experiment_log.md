# BiVES-CXR weak-label proxy P0 experiment log

## Status

The expanded parser-v3 proxy cycle completed locally on 2026-07-17. The frozen
Qwen3.5-2B representation separates support from contradict across all three
retained findings, but the closed-form four-state argmax remains collapsed to
`insufficient`. This is a nonclinical engineering proxy and must not be
reported as an expert-audited or formal BiVES-CXR result.

## Authorization and claim boundary

Qualified clinical review/adjudication was permanently removed from the
executable workflow because no reviewer is available. The replacement keeps
the following hard boundary:

- support, contradict, and uncertain are frozen rule-parser candidates;
- insufficient is an explicit reproducible synthetic evidence-removal image;
- report omission is never used as contradiction or insufficient;
- every row retains parser, report, source-image, and transform provenance;
- the proxy lock declares `formal_result=false` and
  `clinical_ground_truth=false`;
- no clinical U/I validity, locked-test, or publication claim follows from the
  run.

## Data diagnosis and repair

The earlier parser-v2 audit found target-scope errors: a negation applying to a
different finding could flip atelectasis, `clear of consolidation` was not
recognized, and newline splitting could detach a finding from its negation.
Parser v3 now uses target-local scope, preserves cross-line context, recognizes
explicit absence phrases, and includes every scope rule in its provenance
hash.

The original 1,000-study parser-v3 table contained 4,201 globally unique
candidates. The single-variable expansion indexed 5,000 paired MIMIC studies
and 8,220 image rows, then regenerated 20,204/20,204 unique parser-v3
candidates under the unchanged rules SHA256
`224cb4c4194ce0384c35a74f9fcb9cbbd3137a8bac60b288f12b13bdc39a530a`.
A read-only patient-balanced Qwen3.5-2B frozen-feature audit on 20 support and
20 contradict patients per coverage-eligible finding produced:

| Finding | S/C patients | LOO centroid AUROC |
| --- | ---: | ---: |
| Pleural effusion | 20 / 20 | 0.7425 |
| Pulmonary edema | 20 / 20 | 0.7775 |
| Consolidation | 20 / 20 | 0.8425 |
| Pneumothorax | 20 / 20 | 0.6050 |

The predeclared `>=0.65` feature gate retained pleural effusion, pulmonary
edema, and consolidation, and excluded pneumothorax. Atelectasis and
cardiomegaly were coverage-ineligible because they had fewer than 20 independent
contradict patients. The ignored expanded proxy input has 48 train rows / 12
quartets and 48 validation rows / 12 quartets, with zero cross-split patient or
image-hash overlap. Proxy dataset lock SHA256:
`3473ad6aab7350029e593b3c9e9f1e65b4433fdcdd058e8f813bfe9cd00ae9df`.

## Runtime

```text
model: H:/Xiyao_Wang/001_models/Qwen3.5-2B
device: cuda:1 (NVIDIA GeForce RTX 3090)
mode: local_proxy
steps: 50/50
elapsed: 68.8994 seconds
selected step: 50 by minimum validation NLL
formal_result: false
```

The read-only vision smoke passed before launch. Training completed normally,
wrote all step metrics/predictions, and released GPU1 after exit.

## Selected-step result

| Metric | Train proxy | Held-out proxy validation |
| --- | ---: | ---: |
| Accuracy | 0.2500 | 0.2500 |
| Macro F1 | 0.1000 | 0.1000 |
| S vs C AUROC | 0.8819 | 0.8056 |
| U vs I AUROC | 1.0000 | 1.0000 |
| NLL | 1.3675 | 1.3692 |
| Evidence-only sufficiency | not eligible | not eligible |
| Evidence-removal insufficient | not eligible | not eligible |
| Irrelevant stability | 1.0000 | 1.0000 |

Per-finding held-out S/C AUROC is `0.875` for consolidation, `0.8125` for
pleural effusion, and `1.0` for pulmonary edema (eight S/C examples per
finding). Per-finding U/I AUROC is `1.0` for all three. Thus the ranking signal
is not supplied by a single finding. However, every validation argmax is
`insufficient`; the four-state confusion matrix has only its last column
populated. The result therefore passes the ranking gate but fails the absolute
decision/calibration gate.

## Zero-training decoder-geometry diagnostic

To distinguish representation failure from uncalibrated probability geometry,
the existing monotone-decoder fitter was applied to the 48 train-proxy evidence
pairs only, then evaluated once on the frozen 48-row validation proxy. This is
an exploratory diagnosis, not a locked calibration split or release artifact.

| Metric | Uncalibrated validation | Train-proxy-fitted validation |
| --- | ---: | ---: |
| Accuracy | 0.2500 | 0.5417 |
| Macro F1 | 0.1000 | 0.4786 |
| NLL | 1.3692 | 1.1620 |
| ECE | 0.4003 | 0.1925 |
| S vs C AUROC | 0.8056 | 0.8056 |
| U vs I AUROC | 1.0000 | 1.0000 |

The fitted positive parameters are `tau_a=0.4469`, `tau_p=0.1104`, and
`uncertainty_mass=0.5917`. They recover all 12 insufficient cases and 11/12
support cases, but only 1/12 contradict and 2/12 uncertain cases. Calibration
therefore explains much of the all-insufficient collapse but does not solve the
four-state decision problem.

The frozen evidence distribution localizes the remaining error. Median total
evidence is `0.9387`, `0.8518`, `0.8627`, and `0.4356` for S/C/U/I, so the
availability axis separates synthetic insufficient. Median signed evidence
`E+ - E-` is `+0.1309`, `+0.0464`, `+0.0863`, and `+0.0215`: contradict never
crosses the negative polarity origin and uncertain is not centered near zero.
The proxy model therefore learns relative S/C ordering without learning an
absolute bipolar origin. This is the direct reason parameter fitting restores
I/S more readily than C/U.

## Historical invalid/failed attempts

- Parser v1 reused image-level candidate IDs across findings and is invalid
  evidence.
- Parser v2 fixed candidate lineage, but its atelectasis-only 8/4 run selected
  step 30 and obtained held-out S/C AUROC `0.0` despite train AUROC `1.0`.
- The 16/16 parser-v3 run improved aggregate S/C AUROC to `0.8125`, but its
  per-finding result was mixed (pulmonary edema `1.0`, pleural effusion `0.5`).
- The 48/48 expanded parser-v3 run is the current decision artifact.

## Decision

The 5k expansion resolves the prior per-finding ranking instability for the
three retained statements. It does not establish a usable four-state decision
rule: uncalibrated argmax is collapsed to insufficient. Do not scale this
proxy to Qwen3.5-4B/9B, and do not reopen the accepted decoder, losses, exact-K,
or capacity from this result. The train-proxy parameter fit confirms that
absolute probability geometry is material but insufficient. The next
permissible work is zero-training evidence-distribution and leakage diagnosis
on the frozen predictions. The distribution audit now identifies an absolute
polarity-origin failure, but no loss/decoder change or second model run is
authorized inside this completed cycle. Clinical and formal claims remain
unavailable.
