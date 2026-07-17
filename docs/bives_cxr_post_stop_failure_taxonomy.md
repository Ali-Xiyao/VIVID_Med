# BiVES-CXR Post-Stop Failure Taxonomy

**Date:** 2026-07-18

**Authority:** `BiVES_CXR_MIA_TMI_ready_proposal.md` and
`BiVES_995fb81_code_review_and_next_plan.md`

**Scope:** read-only analysis of the frozen, corrected Qwen3.5-2B seed-17
VinDr intervention rows. This report does not override E8, authorize model
selection, or open another experiment.

## Frozen evidence

- Intervention rows: `410` rows, two dilations over the same `205` positives.
- Rows SHA-256:
  `14a18901ff0c5bd513f8146aa111ae9f9c0a318c376569901991da42e78ac1e3`.
- Failure-taxonomy JSON SHA-256:
  `d0d3d6bec133bba7471f02400c7d14e44b6adc5c1a475c1d25ff37a38036a0de`.
- The taxonomy sets `formal_result=false`, `evaluation_only=true`, and
  `gate_override=false`.
- No Qwen model was loaded. No score, mask, checkpoint, manifest, or source
  image was changed.

The ignored local diagnostic is reproducible with:

```powershell
python scripts/analyze_bives_vindr_intervention_failures.py `
  --rows local_runs/bives_cxr/qwen35_2b_vindr_interventions_seed17/intervention_rows.jsonl `
  --output-json local_runs/bives_cxr/qwen35_2b_vindr_interventions_seed17/failure_taxonomy.json `
  --output-md local_runs/bives_cxr/qwen35_2b_vindr_interventions_seed17/failure_taxonomy.md `
  --bootstrap-replicates 20000 `
  --bootstrap-seed 17
```

## Primary dilation result

| Finding | N | Mean target | Mean control | Mean TCIG | 10% trimmed TCIG | TCIG > 0 |
|---|---:|---:|---:|---:|---:|---:|
| consolidation | 94 | 0.0518 | 0.0475 | 0.0043 | 0.0104 | 0.543 |
| pleural effusion | 111 | 0.0217 | 0.0606 | -0.0390 | -0.0223 | 0.279 |

Pleural-effusion failure is not caused by one or two extreme cases. Every
leave-one-out mean remains negative, spanning only `[-0.0443, -0.0334]`.
Consolidation is less stable: its leave-one-out mean spans
`[-0.0014, 0.0113]`, which is consistent with a near-zero rather than a
reliable positive causal effect.

## Selector localization stratification

| Finding | Low localization-gain quartile TCIG | 95% CI | High localization-gain quartile TCIG | 95% CI |
|---|---:|---:|---:|---:|
| consolidation | -0.0569 | [-0.1438, 0.0209] | 0.0514 | [0.0117, 0.0951] |
| pleural effusion | -0.0924 | [-0.1573, -0.0397] | 0.0234 | [-0.0071, 0.0697] |

The aggregate positive localization gain therefore does not mean that the
selector is consistently correct. Images with stronger expert-box overlap
have materially better TCIG, while low-overlap cases drive negative causal
behavior. This is a selector-consistency limitation, not evidence that the
closed-form decoder should be changed.

## Deletion-area stratification

| Finding | Low target-area quartile TCIG | 95% CI | High target-area quartile TCIG | 95% CI |
|---|---:|---:|---:|---:|
| consolidation | 0.0369 | [0.0046, 0.0732] | -0.0042 | [-0.0992, 0.0843] |
| pleural effusion | 0.0023 | [-0.0279, 0.0492] | -0.1165 | [-0.1880, -0.0563] |

Control deletion effect is more correlated with target/control area than
target deletion effect:

| Finding | control effect vs area | target effect vs area |
|---|---:|---:|
| consolidation | 0.417 | 0.087 |
| pleural effusion | 0.588 | 0.340 |

The model is therefore sensitive to arbitrary large pixel deletion,
especially for pleural effusion. Exact area matching prevents a trivial area
imbalance, but it does not guarantee that a random-disjoint deletion is a
clinically neutral or distribution-preserving intervention.

## Joint descriptive model

A standardized descriptive regression of TCIG on localization gain, target
area, and original score explains only part of the variance:

| Finding | localization coefficient | area coefficient | original-score coefficient | R2 |
|---|---:|---:|---:|---:|
| consolidation | +0.0219 | -0.0319 | -0.0522 | 0.220 |
| pleural effusion | +0.0273 | -0.0411 | -0.0137 | 0.252 |

The coefficient directions agree with the stratified analysis, but the low R2
prevents a claim that these factors fully explain each individual failure.

## Dilation stability

| Finding | Paired N | TCIG correlation | Mean change at 0.1 dilation | Sign agreement |
|---|---:|---:|---:|---:|
| consolidation | 94 | 0.645 | -0.0126 | 0.766 |
| pleural effusion | 111 | 0.432 | +0.0002 | 0.820 |

The pleural-effusion negative mean is not repaired by the prespecified 0.1
dilation. Consolidation becomes slightly worse on average. Dilation choice is
therefore not a justified rescue.

## Final diagnosis

The E8 failure is best described as a combination of:

1. **inconsistent sparse-selector localization across images**, despite a
   positive aggregate localization gain; and
2. **broad sensitivity to arbitrary large pixel deletion**, which makes the
   equal-area disjoint control more damaging than the expert target on many
   cases.

It is not explained by a single outlier, the already-repaired packed-vision
bug, or an obvious decoder failure. The current evidence does not justify
changing the decoder, losses, K, model capacity, or dilation after observing
VinDr test results.

## Protocol consequence

This post-stop analysis uses VinDr test outcomes. It may explain the frozen
failure, but it must not become a tuning surface followed by another claim on
the same test set. Any future rescue requires a new reviewed authority and:

1. a separate development set for selector/intervention choices;
2. distribution-preserving intervention operators and anatomy-aware controls
   evaluated without VinDr-test tuning; and
3. a new independent final evaluation surface.

Until those conditions exist, E8 remains failed, CheXlocalize remains
unstarted, and Qwen3.5-4B/9B remain locked.
