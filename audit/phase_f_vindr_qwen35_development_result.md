# Phase F VinDr Qwen3.5 local development result

## Verdict

The locked local development gate completed on both workstation GPUs with no
exclusion. Four prior-exposed VinDr-train images produced eight audit rows
across two findings and two operators.

This is a nonformal, supplemental, image-level development result. VinDr does
not provide patient identifiers, so no patient-level confidence or independent
primary claim is permitted.

## Frozen identity

- model: Qwen3.5-2B snapshot
  `6b57c58ceb3c97d4199d4e8e32b54f6ff11d0cf3a8bde3a4e940d5775ae12120`;
- data: VinDr-CXR train, prior-exposed `protocol_design` positives only;
- four image units: consolidation and pleural effusion, area quartiles 1 and 4;
- explanation: fixed 4x4 local-mean occlusion map, deterministic top-1 cell;
- target roles: expert region `X` and explanation region `E`, each with its
  own exact-area connected control `C_X` or `C_E`;
- operators: local mean ring 8 and masked Gaussian blur sigma 8;
- score-free development lock canonical SHA-256:
  `11a02e1d57f85d970e2eafb9e7217f2ee58f1b47a15945d3b8f156c819c05f45`.

## Result cells

| Finding | Operator | Image units | Mean IoU | Point hit | Mean CS_X | Mean CS_E | CS_E signs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Consolidation | local mean | 2 | 0.171605 | 0.50 | 0.354685 | 0.016723 | 2 positive |
| Consolidation | Gaussian blur | 2 | 0.171605 | 0.50 | 0.381618 | 0.009670 | 1 positive, 1 nonpositive |
| Pleural effusion | local mean | 2 | 0.156858 | 0.50 | 0.282055 | 0.067614 | 2 positive |
| Pleural effusion | Gaussian blur | 2 | 0.156858 | 0.50 | 0.255996 | -0.047302 | 1 positive, 1 nonpositive |

Expert-region specificity is positive in all four small development cells.
Explanation-region specificity is operator-sensitive and changes sign for
pleural effusion. The result is consistent with the audit premise that spatial
localization and causal specificity must be reported separately.

## Execution and provenance

- GPU0 shard canonical SHA-256:
  `2e4f10f36616dddea1ba4b9a4f367461c9c3a857fd9420a7d8812e851ffc6de1`;
- GPU1 shard canonical SHA-256:
  `179fb78c9349bdb88ddcd3b4d1080ca47e671e6ba14425db7974e37023c5774e`;
- merged rows SHA-256:
  `df3faf9032a2e2cff6478d45215cf0bbddd21c22f6a0da732d2402f818825a65`;
- merged result canonical SHA-256:
  `261a02fac2ee60ba8ac24b9a840901da80cf7d6531a6526373a393d40d04ce5f`;
- per-shard peak model CUDA allocation: `6,526,417,920` bytes;
- exclusions: zero;
- VinDr test opened: false;
- CheXlocalize opened: false;
- BiVES checkpoint loaded: false.

Both GPUs were released after their shard completed. Unrelated pre-existing GPU
jobs were not stopped or modified.

## Boundary and next gate

With only two image units per finding/operator cell, correlations are not
estimated and bootstrap intervals are development diagnostics only. This run
validates the real-data Qwen3.5 audit path; it does not set a threshold, choose
an operator, or authorize promotion to a larger/result-bearing matrix.

The next admissible step is a separately frozen, larger VinDr-train
development sample with image-level reporting, or CheXlocalize validation once
approval is obtained. CheXlocalize test remains sealed.
