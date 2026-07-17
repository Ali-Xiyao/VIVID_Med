# BiVES-CXR Expert Polarity and Intervention Verdict

**Date:** 2026-07-18
**Authority:** `BiVES_CXR_MIA_TMI_ready_proposal.md` and
`BiVES_995fb81_code_review_and_next_plan.md`
**Frozen starting point:** `995fb81`
**Execution:** local Qwen3.5-2B only, seed 17

## Verdict

The seed-17 expert-polarity and pixel-intervention gate **fails**. The sparse
exact-K model has strong expert S/C ranking, especially for pleural effusion,
and its selected patches overlap expert boxes more than a matched random mask.
It does not satisfy the frozen promotion rule because:

1. consolidation AUPRC is below the frozen pooled baseline; and
2. target-box deletion is not stronger than an equal-area disjoint control.

The failure triggers the declared stop rule. Do not run more seeds, Qwen3.5-4B,
Qwen3.5-9B, dense-to-sparse preservation, a loss/K/decoder sweep, or a new
weak-label route from this result.

## Locked expert S/C evaluation

VinDr test consensus remains evaluation-only. It was not used to choose the
checkpoint, K, thresholds, loss, or model. The corrected evaluator covers all
3,000 images and reports each finding separately.

| Finding | Prevalence | B0 AUROC | B2 AUROC | B0 AUPRC | B2 AUPRC | B2 >= B0 on both |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| consolidation | 0.032 | 0.8855 | 0.9197 | 0.2628 | 0.2338 | no |
| pleural effusion | 0.037 | 0.9004 | 0.9693 | 0.5171 | 0.7062 | yes |

The strict all-finding expert polarity gate is therefore false. Accuracy is
not a primary metric because both findings are highly imbalanced.

## Pixel intervention evaluation

Only expert-positive images with boxes are eligible. Exact-area disjoint
controls are mathematically impossible when the target union occupies more
than half of the letterboxed content region. A model-output-independent
geometry preflight therefore freezes the same 205-image cohort for dilation
0 and 0.1 and records two excluded consolidation samples. No outcome was used
to select the cohort.

Primary dilation 0 results:

| Finding | N | Target effect | Control effect | TCIG | TCIG 95% CI | Top-K minus random localization | Localization 95% CI |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| consolidation | 94 | 0.0518 | 0.0475 | 0.0043 | [-0.0280, 0.0342] | 0.2368 | [0.2108, 0.2635] |
| pleural effusion | 111 | 0.0217 | 0.0606 | -0.0390 | [-0.0637, -0.0167] | 0.2377 | [0.2094, 0.2693] |

The top-K localization signal is positive for both findings, but the causal
target-vs-control deletion gate fails. For pleural effusion it fails in the
opposite direction: deleting the disjoint control changes the score more than
deleting the expert target. Dilation 0.1 does not repair the result
(consolidation TCIG -0.0082; pleural-effusion TCIG -0.0387).

## Packed-attention invalidation and repair

An earlier expert run used packed multi-image Qwen3.5 eager vision attention.
During intervention reproduction, a real sample differed by 0.0213053 in
support probability between packed and singleton inference. Exact
reconstruction showed patch-token maximum absolute difference 2496 and only
15/16 exact-K patches in common.

The installed Transformers non-Flash vision path computes image chunks
separately but rejoins their attention outputs on the head dimension before a
global reshape. The repository adapter now calls the official vision tower
once per image and only then stacks outputs. On the same reconstructed batch,
patch tokens, score, and gate match singleton inference exactly. All packed
expert/intervention outputs were archived as invalid diagnostics and are not
used above. The corrected intervention run's maximum original-score replay
difference is 2.38e-7.

## Artifact bindings

Generated evidence remains ignored and local:

- expert metrics SHA-256:
  `0468014223af2f934e4ed11708956a82b77ffc412ba2259a11f942f779933135`
- expert predictions SHA-256:
  `342ed59e6bc797b7b3c54162490f5b6efcf4de654b8bef253e14639724322584`
- intervention metrics SHA-256:
  `5d609535083e979851eded69fc12bd28afa1c6286cca5a99989172cf92ae381d`
- intervention rows SHA-256:
  `14a18901ff0c5bd513f8146aa111ae9f9c0a318c376569901991da42e78ac1e3`

These are nonformal, image-level external results. VinDr exposes no patient ID,
so its bootstrap intervals must not be called patient-level intervals.

## Frozen decision

- Keep B0/B1/B2 seed-17 artifacts and the corrected expert/intervention outputs
  as diagnosis evidence.
- Keep the Qwen3.5 per-image isolation guard and its regression test.
- Keep 4B/9B and three-seed expansion closed.
- Do not reinterpret positive box overlap as causal evidence sufficiency.
- Any new method, dataset, or causal-control design requires a new reviewed
  authority rather than an automatic continuation of this gate.
