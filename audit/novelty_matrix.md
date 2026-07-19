# Novelty and related-work matrix

**Status:** living protocol document; refresh before submission

**Rule:** do not claim “first” from this matrix

| Work | Data / task | Main object | Spatial comparison | Causal intervention | Relation to this audit |
| --- | --- | --- | --- | --- | --- |
| CheXlocalize | CheXpert localization, 10 pathologies | Saliency localization benchmark across methods/models | explanation vs radiologist region | No matched target-vs-control causal endpoint | Supplies expert localization benchmark and reveals pathology/shape effects; does not establish localization as causal evidence. |
| Image-use causal audit ([arXiv:2606.17710](https://arxiv.org/abs/2606.17710)) | Medical VLM image use | Relevant/irrelevant occlusion and same-label swaps | task-relevant vs irrelevant image evidence | Yes, at image/region-use level | Establishes that image use is not guaranteed; the proposed audit focuses on the continuous relationship between expert localization, explanation regions, and matched controls. |
| SHOVIR ([arXiv:2606.30201](https://arxiv.org/abs/2606.30201)) | Spatial chest-X-ray datasets / VLMs | Direct/context shortcut reliance | region-occlusion structure | Yes | Closest spatial causal neighbor; this audit adds explicit explanation-method localization quality, expert/explanation/control triads, strength matching, and joint endpoint analysis. |
| C-Score ([arXiv:2604.08502](https://arxiv.org/abs/2604.08502)) | CAMs across CXR patients | Explanation consistency across patients | CAM-to-CAM consistency | No matched target/control intervention | Complementary reliability axis; consistency does not guarantee expert localization or causal specificity. |
| Frozen BiVES C4/C5/C6I | VinDr/MS-CXR positive cases | Top-K statement-conditioned evidence under target/control perturbation | BiVES region vs expert box/control | Yes | Provides the motivating terminal counterexample and one audited model case; cannot establish a general method claim. |
| Proposed localization-causality audit | CheXlocalize locked test plus frozen/supplemental evidence | When localization predicts causal specificity | expert vs explanation vs matched control | Yes, operator- and strength-audited | Integrates separate localization and causal endpoint families across models, explanations, pathologies, operators, and geometry/strength conditions. |

## Claimable distinction if the protocol is completed

The intended contribution is not a new saliency map or a new perturbation
operator. It is a locked, multi-factor audit of the *relationship* between
localization validity and causal specificity, with:

- separate expert, explanation, and target-specific matched-control regions;
- separate localization and causal endpoint families;
- patient-level interaction analysis across model/explanation/pathology/operator;
- perturbation-strength matching and diagnostics;
- a development-only split followed by one-time locked test execution;
- frozen negative BiVES evidence included without post-stop repair.

Before submission, repeat the literature search across PubMed, arXiv, Google
Scholar, IEEE Xplore, and major medical-imaging proceedings, then narrow every
novelty statement to what remains supported.
