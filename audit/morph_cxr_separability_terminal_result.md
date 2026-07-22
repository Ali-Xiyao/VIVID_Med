# MORPH-CXR morphology separability terminal result

## Decision

**Terminal development NO-GO.** The preregistered morphology separability gate
passed for `1/4` findings, below the required `3/4`. A full MORPH-CXR proposal,
larger development study, selector, multi-source training, test opening, or
Qwen3.5-4B/9B scale-up is not authorized by this result.

This is a low-power survival test on exposed development data. It is not a
clinical, confirmatory, or population-level estimate. It is sufficient for its
prospective purpose: avoid a third large implementation bet when the defining
morphology-specific advantage is not present on the locked surface.

## Frozen identity

| Item | Frozen value |
| --- | --- |
| Source commit used for scoring | `dc3914afac9ae358bae7b79ce256f1170e89e0b1` |
| Opening commit | `0d68d22` |
| Opening canonical SHA-256 | `e544ec0c8d6ac6d4b5774ad9da6b2f1762d8776079a285d70965be2fc4179e37` |
| Data-lock canonical SHA-256 | `5401238e792711dbd49917fe136fa5e02851a74fdb048cf2050f41ff55bd3227` |
| Manifest SHA-256 | `5b14b5a449e5ebc0079df1025d62b433123fe590c9ac5cae78dab487d0360b96` |
| Selected-image lock SHA-256 | `d6f761b291945710143af606179d34ccde9480350e02c13b46dc1f5011b05555` |
| Qwen3.5-2B snapshot SHA-256 | `6b57c58ceb3c97d4199d4e8e32b54f6ff11d0cf3a8bde3a4e940d5775ae12120` |
| Cache-lock canonical SHA-256 | `a62f456b882b7722de562b7f9f1d262dd52a464d2e72605f1e3cb924de8f55d4` |
| Result canonical SHA-256 | `8cf57c5fc6d15d71911501ba052996c32df38e88a0848a04f78a8699b0d0fc55` |
| Case-study canonical SHA-256 | `357edde2ee2fc407cfda39f9c867e1f3de783fc1cd0936a1262bfc643ca1d250` |

The frozen Chest ImaGenome gold / MIMIC-CXR-JPG manifest contains 48 unique
patients and images: 24 train and 24 validation, with zero patient overlap.
Each finding has exactly three positive and three negative patients per split.
All local patient rows, images, cached tokens, checkpoints, and detailed
runtime output remain ignored and are not published.

## Preregistered endpoint result

Medians are over the three fixed seeds (`20260722`, `20260723`, `20260724`). A
finding required strictly positive gains over generic patch-MIL on both AUROC
and its prescribed spatial endpoint.

| Finding | Prescribed expert | Generic AUROC | Expert AUROC | AUROC gain | Generic spatial | Expert spatial | Spatial gain | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Pneumothorax | boundary | 0.1111 | 0.4444 | +0.3333 | 0.0000 | 0.0000 | 0.0000 | fail |
| Consolidation | region | 1.0000 | 0.8889 | -0.1111 | 0.6667 | 0.6667 | 0.0000 | fail |
| Pleural effusion | region + boundary | 0.3333 | 1.0000 | +0.6667 | 0.5000 | 0.5000 | 0.0000 | fail |
| Cardiomegaly | geometry | 0.6667 | 0.7778 | +0.1111 | 0.1423 | 0.1730 | +0.0307 | pass |

The two auxiliary contracts passed:

- prescribed concept-only versus generic median AUROC gap: `+0.3333`, above
  the frozen `-0.05` non-inferiority boundary;
- minimum concept-removal margin delta: `0.0`, satisfying the nonnegative
  monotonicity rule.

These auxiliary passes cannot override the primary `1/4` survival result.

## Automatic case study

The identifier-free case study found no incomplete execution or silent
optimization failure:

- `15/15` expert-by-seed runs completed;
- all 15 checkpoints have distinct SHA-256 values;
- all 15 final training losses are lower than their step-1 losses;
- all reported gate metrics are finite;
- every concept monotonicity check is nonnegative.

The failure is structured rather than random:

1. **Pneumothorax:** the boundary endpoint is `0.0` for both generic and
   boundary experts in every seed. Discrimination improves, but the proposed
   boundary representation does not localize the locked pleural-line target.
2. **Consolidation:** the region expert ties the generic spatial endpoint in
   every seed and loses AUROC in every seed. Generic patch-MIL is already the
   stronger representation on this surface.
3. **Pleural effusion:** the region-boundary ensemble strongly improves AUROC
   in every seed but ties the generic spatial endpoint in every seed. This is
   classification improvement, not evidence for morphology-specific spatial
   separability.
4. **Cardiomegaly:** the geometry expert is the sole surviving morphology
   hypothesis. Its small positive AUROC and IoU gains are stable across seeds,
   but one finding cannot justify the proposed multi-expert method.

The appropriate diagnosis is therefore **morphology-specific expert
superiority not supported on the locked development surface**, not a code bug,
threshold bug, or reason to remap experts after viewing results.

## Boundary and next action

- Do not rerun this exposed 48-patient surface with new seeds, thresholds,
  mappings, losses, or expert definitions.
- Do not build full MORPH-CXR from this result.
- Do not open CheXlocalize test or VinDr test.
- Do not resume pixel-removal ARISE/VICER, LLM training, or 4B/9B scaling.
- Preserve cardiomegaly geometry as a bounded observation for future research
  design, not as a post-hoc replacement mainline.

Any successor method requires a new question, new score-blind development
surface, new source identity, and a new prospective opening. This terminal
result itself authorizes no successor experiment.

## Reproduction commands

```powershell
python -m unittest discover -s tests -p "test_morph_*.py" -v
python scripts/analyze_morph_separability.py `
  --result local_runs/morph_cxr/separability_v0_result/result.json `
  --output local_runs/morph_cxr/separability_v0_result/case_study.json
```

The model-scoring command is retained in the local execution record but is not
a rerun recommendation. Runtime artifacts remain local and ignored.
