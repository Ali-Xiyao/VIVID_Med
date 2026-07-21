# Phase H CheXlocalize Qwen3.5 development result

## Verdict

The approved validation-only development run completed under the frozen
Qwen3.5-2B, explanation, control, and operator identity. Of 100 score-free
image-finding pairs, 99 produced 198 operator rows over 70 patients. One
pleural-effusion pair was excluded before scoring because no exact-area,
connected control occupied the required coordinate zone.

The principal development finding is a separation between localization and
causal specificity. Mean explanation-region specificity (`CS_E`) is positive
for both findings and both operators, with patient-cluster bootstrap lower
bounds above zero. Localization is modest, however, and within-cell
localization--`CS_E` Spearman correlations are small or negative. Better
spatial overlap therefore does not predict stronger causal reliance in this
development cohort.

This is prior-exposed CheXlocalize validation protocol development. It is not
independent, confirmatory, or test evidence, and it does not select a threshold
or open the reserved CheXlocalize test split.

## Frozen identity

- source: authenticated Redivis release `aimi.chexlocalize:efx9:v1_0`,
  validation allowlist only;
- downloaded payload: 2,343 files / 3,849,154,259 bytes, all Redivis MD5
  values verified;
- download-lock canonical SHA-256:
  `c1d4ccf0cff7493b064574d5c3dd7c85fc0a6b994ba962bee11534cf7d164aea`;
- score-free data-lock canonical SHA-256:
  `a36d5c6f8e98095ec318fa7c7c09347c28f6681d91dbbff06264616ee7fe4a41`;
- development manifest SHA-256:
  `5e5eff8f93a06ab512a57674805bb7955283db5931f59166440c5fc99c54042c`;
- experiment-lock canonical SHA-256:
  `6b199600b6b7d0235f5e19e0503512b361a66421f8a4ed60a02c05e00733f163`;
- model: Qwen3.5-2B snapshot
  `6b57c58ceb3c97d4199d4e8e32b54f6ff11d0cf3a8bde3a4e940d5775ae12120`;
- findings: 33 Consolidation pairs and 67 Pleural Effusion pairs before the
  single fail-closed exclusion;
- explanation: fixed 4x4 local-mean occlusion map with deterministic top-1
  cell;
- target roles: expert region `X` and explanation region `E`, with distinct
  exact-area connected controls `C_X` and `C_E`;
- operators: local mean ring 8 and masked Gaussian blur sigma 8;
- resampling: 2,000 deterministic patient-cluster bootstrap replicates with
  seed `20260719`.

## Result cells

| Finding | Operator | Pairs | Patients | Mean IoU (95% patient CI) | Point hit | Mean CS_X (95% patient CI) | Mean CS_E (95% patient CI) | IoU--CS_E Spearman | CS_E signs |
| --- | --- | ---: | ---: | --- | ---: | --- | --- | ---: | --- |
| Consolidation | local mean | 33 | 32 | 0.1651 (0.1131, 0.2272) | 0.3030 | 0.1413 (0.0769, 0.2158) | 0.0615 (0.0289, 0.0934) | 0.1189 | 28 positive, 5 nonpositive |
| Consolidation | Gaussian blur | 33 | 32 | 0.1651 (0.1112, 0.2284) | 0.3030 | 0.1790 (0.1135, 0.2520) | 0.0426 (0.0003, 0.0874) | -0.2071 | 24 positive, 9 nonpositive |
| Pleural effusion | local mean | 66 | 63 | 0.0844 (0.0565, 0.1151) | 0.0455 | 0.0003 (-0.0367, 0.0388) | 0.1119 (0.0693, 0.1582) | -0.0186 | 54 positive, 12 nonpositive |
| Pleural effusion | Gaussian blur | 66 | 63 | 0.0844 (0.0572, 0.1142) | 0.0455 | -0.0185 (-0.0641, 0.0293) | 0.0728 (0.0305, 0.1219) | -0.1323 | 38 positive, 28 nonpositive |

Cross-operator `CS_E` signs agree for both findings. The worst operator mean is
`0.0426` for consolidation and `0.0728` for pleural effusion. This does not
make localization a causal proxy: the localization--`CS_E` association is
near zero or negative in three of four cells, while pleural-effusion expert
specificity (`CS_X`) is itself centered near zero under both operators.

All 198 accepted rows pass the frozen target/control geometry and intervention
strength checks for both `X/C_X` and `E/C_E`.

## Execution and provenance

- shard 0: 50 pairs / 34 patients / 100 rows / zero exclusions; canonical
  SHA-256 `6d88b7d50b0e657b77c456d2c6a578a2df6d780d382ada9a6b57a5d02444c4bf`;
- shard 1: 50 pairs / 36 patients / 98 rows / one score-free geometry
  exclusion; canonical SHA-256
  `1da7f754ca342beff87b160d80fe1cff55d6f9222ef09bc42ba4ee79b556a255`;
- merged rows SHA-256:
  `66dfa63556ab53657cef67341f43626ffadb234ef23cf0ebf8526b98af7970ed`;
- merged result canonical SHA-256:
  `49e29e887def84bbc3b855b2d58e8a99093782ca4f3b8151559199fa565175ca`;
- peak CUDA allocation: 6,526,417,920 bytes per shard;
- execution: both patient-disjoint shards ran sequentially on local GPU0
  because GPU1 was occupied by an unrelated pre-existing task, which was not
  stopped or modified;
- test files present: false;
- CheXlocalize test opened: false;
- server or Slurm execution: none;
- BiVES checkpoint loaded or repaired: false.

## Interpretation boundary

This result supports the audit motivation, not a new successful localization
method. It shows that a frozen model may have positive causal specificity for
an explanation region even when expert overlap is low, and that variation in
overlap does not reliably track variation in causal specificity. Because this
validation split was previously exposed to the repository and used for
protocol development, it cannot be promoted to unbiased evidence. The
CheXlocalize test remains reserved for a separately frozen one-time evaluation.
