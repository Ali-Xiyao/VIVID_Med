# CXR localization-causality audit proposal

> **Working title:** Beyond Localization: Auditing Causal Faithfulness of
> Chest-X-ray Explanations
>
> **Preferred venue:** Medical Image Analysis
>
> **Status:** active research authority; protocol design only
>
> **Execution:** local workstation only; no experiment is opened by this file

## 1. Formal pivot

The project no longer claims that BiVES B2 learned a clinically valid causal
evidence set. The frozen terminal result shows that positive expert-region
overlap can coexist with failed matched target-versus-control intervention
specificity. The research object is therefore changed from a new successful
evidence method to a systematic audit of when localization and causal model
reliance agree or disagree.

BiVES B2 is one preregistered audited case. It is not a finalist, a route to be
rescued, or the sole basis of the paper.

## 2. Research questions

- **RQ1 — Predictive validity:** Does better expert localization predict
  stronger causal specificity under perturbation-strength-matched controls?
- **RQ2 — Moderation:** How does the relation vary by model family,
  explanation method, pathology, operator, region size/shape, and confidence?
- **RQ3 — Joint validity:** Which model/explanation combinations satisfy both
  localization quality and causal specificity without relying on a composite
  score that hides failure in either dimension?

## 3. Three-region audit

For every eligible image/pathology/model/explanation row, construct three
separately identified regions:

1. **Expert region (`X`)** — the released pathology localization annotation.
2. **Explanation region (`E`)** — a preregistered, deterministic region derived
   from the model explanation without using test labels to tune its threshold.
3. **Matched control role (`C_T`)** — a disjoint in-content region matched to
   the current intervention target on area and prespecified geometry/strength
   constraints. Expert and explanation targets receive separate controls:
   `C_X` for `X` and `C_E` for `E`.

The audit evaluates the same frozen model score before and after intervention
on `X`, `C_X`, `E`, and `C_E`. The conceptual design still has three region
roles (expert, explanation, control), but it never reuses one mask as the
comparator for differently shaped targets. It never treats overlap alone as
causal evidence.

## 4. Localization-causality matrix

| Localization | Causal specificity | Interpretation |
| --- | --- | --- |
| high | high | jointly supported explanation |
| high | low/negative | localized but not causally specific |
| low | high | causally sensitive but spatially discordant with the annotation |
| low | low/negative | neither localization nor causal criterion supported |

Thresholds are frozen on the development protocol before the test identity is
opened. Continuous endpoint estimates remain primary; the four cells are an
interpretive summary, not a model-selection device.

## 5. Planned study matrix

### 5.1 Data roles

| Dataset | Role | Boundary |
| --- | --- | --- |
| CheXlocalize validation | protocol development only | Previously exposed in this repository; cannot serve as unbiased validation evidence. |
| CheXlocalize test | one-time locked primary evaluation | Unopened until the protocol, data identity, models, explanations, operators, and hypotheses are frozen. |
| MS-CXR | frozen external evidence / sensitivity analysis | Reuse only according to its frozen prior identities; no B2 tuning or post-stop selection. |
| VinDr-CXR | supplemental robustness | Prior C4/C5 use is disclosed; not an independent primary confirmation set. |

CheXlocalize provides radiologist localization annotations for 10 pathologies
and official validation/test cohorts. The original benchmark evaluated seven
saliency methods across three CNN architectures using localization metrics such
as mIoU and hit rate. These facts motivate, but do not themselves establish, a
causal audit ([CheXlocalize paper](https://www.nature.com/articles/s42256-022-00536-x),
[Stanford AIMI dataset page](https://aimi.stanford.edu/datasets/chexlocalize)).

### 5.2 Audited model families

The final matrix should cover at least:

- a conventional supervised chest-X-ray classifier;
- a vision/foundation encoder with a frozen or preregistered linear readout;
- a statement-conditioned multimodal Qwen3.5 case;
- frozen BiVES B2 as a terminal counterexample/audit case where compatible.

Exact checkpoints, licenses, output scores, preprocessing, and calibration are
frozen before test. Legacy model families are not reactivated inside the BiVES
implementation path merely to fill the matrix.

### 5.3 Explanation families

- gradient/CAM family (for example Grad-CAM under a fixed layer rule);
- integrated gradients under a fixed baseline and step count;
- occlusion sensitivity under a fixed patch/stride rule;
- attention rollout only when the model exposes a stable, documented tensor;
- BiVES top-K region as its own native explanation, without retuning K.

Every explanation must yield a deterministic continuous map and a test-blind
region-construction rule. Missing or unsupported explanations are recorded as
ineligible, not silently substituted.

### 5.4 Operator family

At minimum, audit a prespecified local-mean/constant replacement and masked
Gaussian blur. Any inpainting operator is admitted only after a separate
development-only identity, artifact, and failure analysis are frozen. Operator
results are reported separately, together with cross-operator sign agreement
and worst-case causal specificity.

## 6. Perturbation-strength matching

Target and control interventions must be audited on:

- pixel area;
- centroid/coordinate-zone displacement;
- perimeter and connectedness;
- normalized pixel L1 and RMS change;
- SSIM change;
- edge-energy change;
- content containment and target/control disjointness.

These are diagnostics and matching constraints, not outcome-adjusted tuning
variables. A causal contrast is invalid for primary analysis if its prespecified
hard geometry or strength gate fails.

## 7. Claims and non-claims

Allowed claims are conditional and empirical:

- localization quality and causal specificity can agree or diverge;
- their association and disagreement patterns can vary across prespecified
  models, explanations, pathologies, and operators;
- some explanation methods may satisfy both endpoint families under the locked
  audit while others do not.

Forbidden claims:

- expert overlap proves causal necessity or sufficiency;
- BiVES B2 is a successful causal-evidence method;
- a perturbation effect identifies the true clinical lesion mechanism;
- a post-test threshold/operator selected from CheXlocalize test is valid;
- a single composite score can compensate for failure of localization or
  causal specificity;
- this is the first causal chest-X-ray explanation study.

## 8. Novelty position

The audit is adjacent to, and must explicitly distinguish itself from:

- CheXlocalize localization benchmarking;
- recent relevant/irrelevant occlusion and same-label-swap audits of image use
  ([arXiv:2606.17710](https://arxiv.org/abs/2606.17710));
- SHOVIR spatial region-occlusion audits of shortcut reliance
  ([arXiv:2606.30201](https://arxiv.org/abs/2606.30201));
- C-Score cross-patient CAM consistency
  ([arXiv:2604.08502](https://arxiv.org/abs/2604.08502)).

The proposed distinction is the locked relationship analysis across expert,
explanation, and matched-control regions with separate localization and causal
endpoint families, strength-matched operators, and interaction analysis. This
position remains provisional until a submission-date literature search.

## 9. Phase gates

### Phase A — complete predecessor freeze

- tag the exact terminal commit;
- archive the old method proposal and negative result;
- prohibit C6J, B2 repair, and 4B/9B BiVES scaling.

### Phase B — protocol design (complete)

- freeze research questions, data roles, study matrix, endpoints, and no-claims;
- document prior CheXlocalize validation exposure;
- perform no model/data experiment.

### Phase C — development-only implementation (current)

- verify data rights and local package identity;
- implement deterministic annotations, explanation maps, controls, operators,
  and tests on the development split;
- freeze all model/checkpoint/explanation/operator identities and feasibility
  exclusions before any test opening.

The user's 2026-07-19 opening authorizes local Qwen3.5-2B synthetic model/GPU
interface and cross-device determinism gates. It does not authorize a
CheXlocalize download, real-patient development result, or test opening.

### Phase D — one-time locked test (future authority required)

- verify a clean Git identity and all source/data/model locks;
- execute once locally with append-only progress and fail-closed checks;
- report every preregistered pathology/model/explanation/operator cell;
- prohibit rerun, threshold change, operator change, or selective omission.

## 10. Go/no-go rules

Proceed to the locked test only if:

- the development pipeline is deterministic and all contracts pass;
- all primary cells meet minimum patient/sample feasibility;
- model scores and explanation maps are valid under frozen preprocessing;
- target/control geometry and strength gates pass at the prespecified rate;
- the test package has not been inspected for tuning;
- hypotheses, exclusions, multiplicity, and analysis code are frozen.

Stop before test if any identity, independence, data-rights, geometry,
explanation-availability, or reproducibility gate fails. After test, a null or
negative result is terminal evidence, not a new tuning opportunity.
