# MORPH-CXR morphology separability gate

## Status and scope

This is a bounded, development-only survival gate. It is not a full MORPH-CXR
proposal and cannot support a clinical, causal, or confirmatory claim. ARISE-v1
and VICER V0 remain terminal records; their pixel-removal results are not tuned,
replayed, or relabeled here.

The gate asks one question: with one frozen Qwen3.5-2B visual backbone and a
new patient-disjoint development surface, does a preregistered morphology
expert beat generic patch-MIL on both finding discrimination and the matching
spatial or measurement endpoint?

## Frozen mapping

| Finding | Primary expert | Spatial endpoint |
| --- | --- | --- |
| Pneumothorax | boundary | boundary pointing hit |
| Consolidation | region | region pointing hit |
| Pleural effusion | region + boundary | mean region/boundary pointing hit |
| Cardiomegaly | geometry | predicted-region bounding-box IoU |

The four expert types are region, boundary, geometry, and distribution. The
distribution expert is implemented and trained as a contract/control expert,
but it is not selected post hoc as a primary expert for these four findings.

## Development data

The source is the Chest ImaGenome v1.0.0 gold object-attribute table joined to
local MIMIC-CXR-JPG pixels. This gold surface is permanently treated as exposed
development data after this gate. Patients found in prior local experiment
artifacts are excluded before selection. The final manifest contains 48 unique
patients/images: for every finding, three positive and three negative training
patients plus three positive and three negative validation patients. Train and
validation patients are globally disjoint across findings.

The local MIMIC derivative is 224 x 224, so spatial targets are bound to the
publisher's `coord224` field rather than `coord_original`; every selected box
must remain exactly in bounds and is never clipped.

The smallest positive class is pneumothorax: only six patients remain after
historical-exposure, conflicting-label, and invalid-geometry exclusions. It
determines the sample budget; the other findings are downsampled to the same
fixed budget before any model score is opened. This is a low-power feasibility
gate, not a paper-level effect estimate.

## Model and optimization

- frozen local `Qwen3.5-2B` visual encoder;
- cached 28 x 28 patch tokens under the frozen letterbox preprocessing;
- generic patch-MIL direct classifier baseline;
- concept-only morphology experts with a nonnegative concept combiner;
- fixed 200 steps, three fixed seeds, no early stopping or validation-based
  model selection;
- image-level binary loss plus the preregistered spatial loss;
- no LLM training, pixel counterfactual, operator family, or 4B/9B model.

## Survival rule

For each finding, compare medians across the three fixed seeds. A finding passes
only if the primary expert has a strictly positive AUROC gain and a strictly
positive matching spatial/measurement gain over generic patch-MIL. The gate
survives only if:

1. at least three of four findings pass both comparisons;
2. the median prescribed concept-only AUROC is no more than 0.05 below the
   generic direct-classifier median;
3. setting any active concept to zero never increases the support margin.

Failure stops before a full MORPH-CXR proposal, larger data, additional
findings, CheXlocalize test, VinDr test, LLM training, and 4B/9B scaling. A
post-run case study may diagnose failure but may not remap findings to experts,
relax thresholds, or turn the same validation result into a pass.
