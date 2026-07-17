# BiVES-CXR weak-label proxy P0 experiment log

## Status

Completed locally on 2026-07-17. This is a nonclinical engineering proxy and
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

## Proxy data

- Candidate input: 4,070 rows, 1,515 unique images, 244 patients.
- Retained findings: atelectasis, consolidation, pulmonary edema.
- Built manifest pool: 24 train rows / 6 quartets and 12 validation rows / 3
  quartets, patient-disjoint across splits.
- Proxy dataset-lock SHA256:
  `2cb4f963acab7a66fbece3212c1307b66b71a58013e80719fbbc4b462acf4b19`.
- The bounded local-overfit runner selected the atelectasis ontology: two
  training quartets (8 rows) and one held-out validation quartet (4 rows).

The ignored source artifacts are under
`local_runs/bives_cxr/proxy_p0_input/`. They are not published to Git because
they contain local dataset paths and derived images.

## Runtime

```text
model: H:/Xiyao_Wang/001_models/Qwen3.5-2B
device: cuda:1 (NVIDIA GeForce RTX 3090)
mode: local_overfit
steps: 50/50
elapsed: 22.6316 seconds
selected step: 30 by minimum validation NLL
formal_result: false
```

The read-only Qwen3.5 vision smoke passed before launch. Training exited with
empty stderr, wrote `metrics_final.json`, and released GPU1 after completion.

## Selected-step result

| Metric | Train proxy | Held-out proxy validation |
| --- | ---: | ---: |
| Accuracy | 0.625 | 0.500 |
| Macro F1 | 0.5595 | 0.375 |
| S vs C AUROC | 1.000 | 0.000 |
| U vs I AUROC | 1.000 | 1.000 |
| NLL | 0.7539 | 0.9868 |
| Evidence-only sufficiency | 1.000 | 1.000 |
| Evidence-removal insufficient | 1.000 | 1.000 |
| Irrelevant stability | 1.000 | 1.000 |

Validation predicts support, contradict, and uncertain examples as uncertain,
while correctly predicting the synthetic insufficient example. The mechanism
can memorize the training S/C ordering but reverses its held-out S/C ranking.

An earlier v1 run is invalid evidence: the parser candidate ID omitted the
finding, so candidates for different statements on the same image collided.
Parser v2 now emits globally unique `<image-candidate>::<finding>` IDs, the
builder rejects duplicate IDs, and the reported run above uses only the
regenerated v2 manifests and lock.

## Decision

The run is execution-green but fails the first learning survival gate. Do not
scale this proxy to Qwen3.5-4B or Qwen3.5-9B and do not tune decoder geometry,
loss weights, exact-K, or capacity. The next bounded step is a read-only audit
of weak-label S/C cue noise, per-statement balance, source/report overlap, and
split construction. Any later rerun must change only a justified data-side
factor and remain explicitly nonclinical.
