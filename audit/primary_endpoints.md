# Primary endpoints and analysis contract

**Status:** protocol definition; values and thresholds remain test-blind

## 1. Notation

For image `i`, pathology `p`, model `m`, explanation `e`, and operator `o`:

- `s0`: original model support score;
- `sX`, `sE`: score after expert- and explanation-region intervention;
- `sCX`, `sCE`: score after the target-specific expert/explanation controls;
- `dX = s0 - sX`, `dCX = s0 - sCX`;
- `dE = s0 - sE`, `dCE = s0 - sCE`;
- `CS_X = dX - dCX`: expert-region causal specificity;
- `CS_E = dE - dCE`: explanation-region causal specificity;
- `L`: localization quality of the continuous/thresholded explanation against
  the expert annotation.

Score orientation is frozen per model so that larger `s` consistently means
more support for the target pathology/statement.

## 2. Co-primary endpoint families

### 2.1 Localization

- continuous-map or thresholded-region mIoU, per the frozen explanation rule;
- hit rate / point localization as a complementary metric;
- patient-level pathology/model/explanation estimates with 95% confidence
  intervals.

### 2.2 Causal specificity

- `CS_E` for the explanation region versus its matched control;
- `CS_X` for the expert region versus its matched control;
- operator-specific patient-level mean effects and 95% confidence intervals;
- cross-operator sign agreement and the prespecified worst-operator effect.

No single composite score replaces these endpoint families.

### 2.3 Localization-causality relation

- patient-level association of continuous `L` with `CS_E`;
- prespecified interaction terms for pathology, model family, explanation
  method, operator, region area/shape, and confidence;
- four-cell high/low localization × positive/nonpositive causal summaries only
  after continuous estimates are reported.

The primary association model, covariates, random effects, transformations, and
high/low visualization thresholds are frozen during development.

## 3. Perturbation-strength diagnostics

For each target/control pair (`X`/`C_X` and `E`/`C_E`), report:

- area, centroid distance, perimeter ratio, connected components;
- normalized pixel L1/RMS change;
- SSIM change;
- edge-energy change;
- content containment and overlap/disjointness.

A row that fails a preregistered hard geometry/strength invariant is invalid;
it is not repaired from its outcome score. Residual strength differences are
reported and may enter a prespecified sensitivity model, not an outcome-driven
matching loop.

## 4. Statistical unit and uncertainty

- Resample and split at patient level.
- Use paired contrasts within the same image/model/operator whenever possible.
- Report pathology-stratified estimates before pooled estimates.
- Use patient-cluster bootstrap confidence intervals or a prespecified
  hierarchical/mixed model; do not mix methods after seeing test results.
- Freeze multiplicity control across co-primary cells before test.
- Report effect sizes and intervals even when a binary gate is used.

## 5. Secondary endpoints

- original model discrimination/calibration on the eligible cohort;
- explanation/expert region overlap and area/shape summaries;
- target/expert/explanation intervention rank concordance;
- control-failure taxonomy and operator-specific failure modes;
- runtime, memory, missing-map, and invalid-geometry rates.

These cannot rescue a failed co-primary endpoint.

## 6. Go/no-go and terminal interpretation

### Pre-test no-go

Stop before test if any model score orientation, explanation extraction,
patient identity, annotation geometry, matched control, strength audit,
deterministic replay, data identity, or source/weight license gate fails.

### Test interpretation

- **Joint support:** localization and causal specificity endpoint families meet
  their frozen criteria without material operator contradiction.
- **Localized-not-causal:** localization passes while `CS_E` is nonpositive or
  operator-fragile.
- **Causal-not-localized:** `CS_E` is positive while localization is weak.
- **Joint failure:** neither endpoint family is supported.

All outcomes, including null and negative outcomes, are terminal for the locked
test. No post-test retuning is authorized.
