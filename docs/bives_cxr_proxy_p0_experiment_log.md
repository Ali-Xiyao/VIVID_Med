# BiVES-CXR weak-label proxy P0 experiment log

## Status

The parser-v3 data repair and one bounded Qwen3.5-2B rerun completed locally on
2026-07-17. The aggregate support/contradict survival gate now passes, but the
per-finding result remains mixed. This is a nonclinical engineering proxy and
must not be reported as an expert-audited or formal BiVES-CXR result.

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

The regenerated parser table contains 4,201 globally unique candidates. A
read-only patient-balanced Qwen3.5-2B frozen-feature audit produced:

| Finding | S/C patients | LOO centroid AUROC |
| --- | ---: | ---: |
| Pleural effusion | 10 / 10 | 0.750 |
| Pulmonary edema | 10 / 10 | 0.785 |
| Consolidation | 10 / 10 | 0.640 |
| Pneumothorax | 10 / 10 | 0.590 |
| Cardiomegaly | 4 / 4 | 0.375 |
| Atelectasis | 5 / 5 | 0.360 |

The one authorized data-side repair therefore removed the unreliable
atelectasis ontology and retained pleural effusion plus pulmonary edema. The
new ignored proxy input has 16 train rows / four quartets and 16 validation
rows / four quartets, with zero patient or image-hash overlap. Proxy dataset
lock SHA256: `7ba18607e835154796378b6b79871b6367031877b7f7f7fa0ebb67bfec583753`.

## Runtime

```text
model: H:/Xiyao_Wang/001_models/Qwen3.5-2B
device: cuda:1 (NVIDIA GeForce RTX 3090)
mode: local_proxy
steps: 50/50
elapsed: 29.4116 seconds
selected step: 40 by minimum validation NLL
formal_result: false
```

The read-only vision smoke passed before launch. Training completed normally,
wrote all step metrics/predictions, and released GPU1 after exit.

## Selected-step result

| Metric | Train proxy | Held-out proxy validation |
| --- | ---: | ---: |
| Accuracy | 0.9375 | 0.7500 |
| Macro F1 | 0.9365 | 0.7401 |
| S vs C AUROC | 1.0000 | 0.8125 |
| U vs I AUROC | 1.0000 | 1.0000 |
| NLL | 0.5884 | 0.8241 |
| Evidence-only sufficiency | 0.9091 | 1.0000 |
| Evidence-removal insufficient | 0.3636 | 0.5000 |
| Irrelevant stability | 1.0000 | 1.0000 |

Per-finding held-out S/C AUROC is `1.0` for pulmonary edema but only `0.5`
for pleural effusion (four S/C examples per finding). Thus the aggregate S/C
ranking repair is real, but the evidence is too small and heterogeneous to
claim stable multi-finding generalization.

## Historical invalid/failed attempts

- Parser v1 reused image-level candidate IDs across findings and is invalid
  evidence.
- Parser v2 fixed candidate lineage, but its atelectasis-only 8/4 run selected
  step 30 and obtained held-out S/C AUROC `0.0` despite train AUROC `1.0`.
- The parser-v3 rerun is the only result used for the current decision.

## Decision

The parser/data repair changes the proxy result from aggregate S/C
learning-red to aggregate-green, while exposing pleural-effusion instability.
Do not tune the closed-form decoder, losses, exact-K, or capacity, and do not
scale this proxy to Qwen3.5-4B/9B. The next data task is to enlarge
patient-disjoint same-statement validation and retain only findings that pass
per-finding S/C gates. Clinical and formal claims remain unavailable.
