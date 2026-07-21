# CheXlocalize development and one-time test protocol

**Status:** validation/development acquisition authorized; test remains unopened

**Primary rule:** development exposure and test evaluation are strictly separated

## 1. Split roles

The official dataset page describes 234 validation images from 200 patients
and 668 test images from 500 patients. It also distinguishes ground-truth
annotations from benchmark-radiologist annotations
([Stanford AIMI](https://aimi.stanford.edu/datasets/chexlocalize)).

This repository has prior CheXpert/CheXlocalize validation exposure. Therefore:

- the official validation split is a **development set**, not independent
  validation evidence;
- all choices made with it are disclosed as protocol development;
- no performance on it is described as locked, unbiased, or confirmatory;
- the official test split is the sole candidate for one-time primary evidence.

On 2026-07-19 the user confirmed approved Redivis access and explicitly
authorized local download/use of the development data. The frozen acquisition
opening is `local_chexlocalize_validation_acquisition_opening_20260719.json`.
It permits only the official `*_val` assets and `gradcam_maps_val/`; every
`*_test` asset remains excluded from acquisition and inspection.

## 2. Development-allowed decisions

Only the development split may be used to establish:

- eligible pathologies and minimum positive-patient count;
- model output-to-pathology mappings and score validity checks;
- explanation extraction layers/baselines/steps/patch sizes;
- deterministic continuous-map normalization;
- region thresholding or exact-area rules;
- expert/explanation geometry plus separate `C_X`/`C_E` control construction;
- operator definitions and perturbation-strength tolerances;
- handling of multiple expert polygons/boxes and multiple annotations;
- missing/ineligible-row rules;
- runtime/memory feasibility and deterministic replay;
- the complete statistical analysis implementation.

Choices are versioned and hashed. Development results may reject an infeasible
cell, but may not be used to advertise confirmatory performance.

## 3. Test-prohibited actions before freeze

Before the final lock, do not:

- decode or visualize test images/annotations for method selection;
- compute test saliency, overlap, scores, interventions, or summaries;
- inspect pathology-specific test counts beyond publisher metadata needed for
  prespecified feasibility;
- select a model, layer, threshold, operator, control, or exclusion based on a
  test result;
- generate partial test outputs as a “smoke.”

Pre-open validation may check only package hashes, paths, schema, patient
uniqueness, and encrypted/opaque expected identities without exposing outcomes.

## 4. Required final lock

The one-time test lock binds:

- Git commit and clean tracked-diff state;
- dataset/package digest and annotation-file digests;
- patient/image/pathology manifest and canonical digest;
- model repository/checkpoint/config/tokenizer/processor hashes;
- preprocessing and score mapping;
- explanation method, layer/baseline/steps, and map normalization;
- region construction and exact control algorithm;
- operator implementation and all geometry/strength thresholds;
- primary/secondary endpoint code and multiplicity plan;
- eligible cells, exclusions, stop rules, output schema, and run authorization;
- local GPU/device identity and append-only opening/progress markers.

Any mismatch closes the gate without loading a model.

## 5. Row construction

The analysis unit is an eligible image-pathology-model-explanation row with a
patient identifier. Each row records:

- expert annotation source and union/consensus rule;
- continuous explanation map plus deterministic region;
- separate disjoint matched controls `C_X` and `C_E`;
- original, expert-region, expert-control, explanation-region, and
  explanation-control model scores;
- localization metrics and all intervention-strength diagnostics;
- exact source/model/operator identities and row-level failure status.

Patient identity, not image identity, defines bootstrap/resampling and split
independence.

## 6. One-time execution

After explicit authorization, the test opens once on the local workstation.
The runner writes an opening marker before the first test decode/forward,
append-only row progress, and a terminal result even when a fail-closed gate is
triggered. There is no same-test repair, rerun, threshold update, operator
replacement, or selective cell omission.

## 7. Reporting

- Report all locked cells and all exclusions with denominators.
- Separate development findings from test findings in every table/figure.
- Present continuous localization and causal endpoints separately.
- Report operator-specific estimates, sign agreement, and worst-case results.
- Label prior MS-CXR/VinDr evidence as frozen/supplemental rather than new
  independent confirmation.
