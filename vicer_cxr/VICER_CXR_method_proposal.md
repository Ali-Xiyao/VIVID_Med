# VICER-CXR V0 Method Proposal

**Status:** candidate method; V0 only
**Date:** 2026-07-22
**Predecessor:** frozen ARISE-v1 tag `arise-oracle-v1-terminal-20260722`

## Method question

VICER-CXR asks whether a pixel intervention has independently verified
pathology-removal semantics before that intervention is used to train or audit
causal evidence. V0 does not learn an evidence selector. It calibrates a bank
of interventions on new VinDr-train images using three image-disjoint roles:

1. a local finding critic trained from expert-box Qwen3.5-2B features;
2. a separate global finding verifier trained from different image identities;
3. a third, untouched V0 dose-response evaluation role.

The validity critic never consumes the global verifier's target-control effect.
The shared frozen Qwen3.5-2B visual tower is a representation source only; the
two heads have separate training identities and endpoints.

## V0 data boundary

- Dataset: VinDr-CXR train only.
- Findings: pneumothorax, consolidation, pleural effusion, cardiomegaly.
- Every image used by ARISE VinDr train/validation is excluded.
- All roles are globally image-disjoint.
- Public VinDr DICOMs expose no patient/study/series IDs; all uncertainty is
  image-clustered and no patient-level claim is permitted.
- The remaining unused consolidation and pleural-effusion positives are
  1-of-3-reader findings. V0 explicitly permits these expert-box examples only
  as exploratory intervention-validity development, not clinical evidence.
- VinDr test and CheXlocalize test remain unopened.

Per finding, the fixed positive/negative roles are:

| Role | Support | Contradict | Purpose |
| --- | ---: | ---: | --- |
| critic train | 10 | 10 | local validity critic |
| critic calibration | 5 | 5 | independent critic AUROC gate |
| verifier train | 10 | 10 | global target-control effect scorer |
| verifier calibration | 6 | 6 | independent verifier AUROC gate |
| validity evaluation | 8 | 0 | expert-mask dose-response only |

## Intervention bank

The score-free bank contains three prespecified families:

- masked Gaussian blur with sigma 2, 4, 8, and 16;
- local-ring-mean replacement with alpha 0.25, 0.50, 0.75, and 1.00;
- low-frequency replacement with sigma 24 and the same four alpha levels.

Every evaluation expert mask has an exact-shape translated, target-disjoint,
same-vertical-third control. This V0 control is geometry matched but is not
claimed to be true anatomy or local-statistics matched; those stricter controls
remain a V1 refinement after V0 survival.

For each target intervention:

- `q_remove` is the decrease in the independent local critic probability;
- `q_preserve` is mean outside-target Qwen patch-token cosine similarity;
- `q_realism` is distance calibration against original, non-evaluation Qwen
  image features;
- the target-control gap is computed by the separate global verifier.

An individual intervention is valid only when:

```text
q_remove >= 0.02
q_preserve >= 0.98
q_realism >= 0.50
```

## Survival gate

Before any V0 model score, every finding-specific critic and verifier must have
calibration AUROC at least 0.60. A family survives only if, for all four
findings:

1. median `q_remove` has Spearman correlation at least 0.80 with strength;
2. strongest-dose median preservation is at least 0.98;
3. strongest-dose median realism is at least 0.50;
4. at least half of its sample-dose rows meet the independent validity rule;
5. mean target-control gap among valid rows is positive.

V0 passes only if at least one complete family survives. Otherwise the result
is a terminal V0 development failure and V1/V2 remain locked. Passing V0 opens
only the V1 coverage-redundancy ladder on development data.

## Explicit exclusions

V0 does not authorize a selector, evidence coalition, U/I labels, abstention
calibration, Qwen3.5-4B/9B, server execution, Slurm, CheXlocalize test access,
or any change to frozen ARISE/BiVES results.
