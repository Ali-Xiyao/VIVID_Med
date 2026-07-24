# VIVID-GDS Case Study and Next-Direction Recommendation

## 1. Cloud status

The durable code, protocol, planning records, and terminal report are on
GitHub branch `codex/vivid-gds-extension` at commit
`66d411d40c86395661a727793fc388442f691bd2`. The GitHub branch head and local
head were rechecked on 2026-07-24 and matched exactly.

The complete runtime evidence remains on the SUES server:

`/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/02101_vivid_gds/local_runs/vivid_gds_stage_a_qwen35_2b_20260724_r1`

The repaired run occupies about 2.0 GB; the isolated VIVID-GDS server project
occupies about 2.7 GB. Checkpoints, predictions, medical-image references,
logs, datasets, and patient-level files are deliberately not on GitHub.

## 2. What the experiment actually established

The result is more specific than “VIVID-GDS failed.”

| Arm | Macro AUROC | Macro AUPRC | Interpretation |
|---|---:|---:|---|
| A0 direct schema | 0.865671 | 0.689420 | strongest non-LLM control |
| A1 deterministic free text | 0.860507 | 0.687679 | unstructured generation control |
| A2 frozen UMS prefix4 | 0.859209 | 0.690875 | original structured-generation route |
| A3 VIVID-GDS | **0.869484** | **0.698273** | dual-path method |

A3 was the best single arm. Relative to A2, it gained 1.03 AUROC points and
0.74 AUPRC points and passed every preregistered A3-versus-A2 gate. Relative
to A0, however, its descriptive gain was only 0.38 AUROC points, although its
AUPRC was 0.89 points higher.

The terminal NO-GO came from a different part of the proposed causal chain:

- A2 did not beat A1;
- A2 did not beat A0;
- therefore the premise “structured UMS generation is already a stronger
  representation teacher” was not supported;
- A3 improved that weak A2 route, but could not retroactively validate the
  failed predecessor premise.

## 3. Finding-level case study

### 3.1 Where the schema bridge helped

| Finding | A2 - A1 AUROC | A2 - A0 AUROC | A3 - A2 AUROC |
|---|---:|---:|---:|
| Atelectasis | -0.0089 | +0.0073 | -0.0061 |
| Cardiomegaly | -0.0219 | -0.0108 | +0.0160 |
| Consolidation | +0.0261 | -0.0048 | +0.0046 |
| Edema | +0.0018 | -0.0003 | +0.0138 |
| Pleural Effusion | -0.0035 | -0.0238 | +0.0230 |

The clearest pattern is recovery rather than universal improvement:

- A2 lost substantial pleural-effusion discrimination relative to A0; A3
  recovered almost exactly that loss.
- A2 was weak on cardiomegaly relative to both A0 and A1; A3 recovered much
  of that deficit.
- Consolidation was the one finding where structured generation clearly
  helped over free text, but it still did not beat direct schema supervision.
- Atelectasis moved in the opposite direction: A3 was slightly worse than A2.

This pattern is consistent with the schema head acting as a training-time
regularizer that reconnects some clinically named fields to the deployed CLS
representation. It is not consistent with a uniform UMS-generation advantage.

### 3.2 De-identified disagreement examples

The following examples were selected only from the already exposed
expert-development predictions. Scores are uncalibrated model outputs and the
0.5 split is used only to make the disagreement direction readable. No
patient ID, image path, or mapping key is retained in Git.

| Example | Label | A0 | A2 | A3 | Observation |
|---|---:|---:|---:|---:|---|
| Pleural-effusion case 1 | 0 | 0.5325 | 0.9379 | 0.3983 | A3 removes a strong A2 false-positive tendency |
| Edema case 1 | 1 | 0.4470 | 0.4025 | 0.6691 | A3 moves the positive case above both controls |
| Atelectasis case 1 | 1 | 0.5925 | 0.6951 | 0.3119 | A3 creates a new miss despite both controls ranking it higher |
| Consolidation case 1 | 0 | 0.3551 | 0.2542 | 0.6377 | A3 creates a new false-positive tendency |

The case study explains why the macro result is positive but modest: A3 makes
large corrections in both directions. Its gains on cardiomegaly, edema, and
pleural effusion outweigh failures on a smaller set of atelectasis and
consolidation cases. A single global schema-loss weight is therefore not a
guarantee of uniformly improved finding representations.

## 4. Root-cause judgment

The failure is not caused by:

- inability to optimize the model;
- missing gradients;
- insufficient feasibility steps;
- a broken Qwen or projector path;
- failure of A3 relative to its immediate A2 control.

The main scientific problem is that three different claims were chained
together:

1. UMS generation should beat free-text generation;
2. frozen-Qwen UMS generation should beat direct schema supervision;
3. adding a deployable schema bridge should improve UMS generation.

Only claim 3 survived. Requiring all three was appropriate for the original
paper story, but it also shows that the new module is not the failed component.

## 5. Recommended next direction

### Recommendation: one fresh-split bridge replication, then a hard fork

Do not tune the frozen Stage-A run and do not add another module. If the user
wants one final method-oriented attempt, create a separate protocol and
evaluate the unchanged A3 identity on a genuinely fresh development surface.
The new scientific question should be:

> Does a training-only structured schema bridge add deployable representation
> value beyond both direct schema supervision and structured generation?

The old CheXpert expert-development set must not be used for selection again.
A suitable new development source would be a patient-disjoint subset of
CheXpert-Plus training data after explicit removal of every CheXpert expert
validation/test and CheXlocalize overlap. NIH and historically exposed VinDr
should not be presented as blind evidence.

The minimal new gate should use only A0, A2, and unchanged A3:

1. identical data, initialization, budget, augmentation, and probe protocol;
2. seeds 0, 1, and 2;
3. A3 must exceed both A0 and A2 by at least 0.005 macro AUROC on average;
4. all three seeds must have the same positive direction;
5. paired patient-bootstrap confidence intervals must exclude zero for the
   primary A3-versus-A0 comparison;
6. macro AUPRC must be noninferior and no finding may decline by more than
   0.02 without a preregistered explanation.

This is intentionally strict. The current descriptive A3-versus-A0 gain is
only 0.0038, so the next experiment has a real chance to stop the method
cleanly.

### Decision after that one gate

- If A3 beats both A0 and A2: continue as a narrow “training-only semantic
  bridge” paper. Treat A1 and the old A2 hierarchy as background, not as
  required causal steps. Then unlock full-data, low-data, calibration, and
  one untouched external evaluation in that order.
- If A3 beats A2 but not A0: stop the LLM-method claim. The evidence would say
  that Qwen generation is unnecessary relative to direct structured
  supervision.
- If A3 fails both: end the method route and convert the accumulated evidence
  into a systematic objective-to-representation mismatch audit.

## 6. Safer publication alternative

The lower-risk paper direction is an audit rather than another method:

> Strong generation or schema-training metrics do not guarantee a stronger
> deployable medical image representation.

VIVID/SPD, RCSD, and VIVID-GDS already provide controlled examples of this
disconnect. To make that paper complete, the audit would need a fresh
evaluation surface, multiple objective families, at least two backbones or
teachers, and a preregistered analysis of training-proxy versus downstream
transfer. It should not reuse the current expert-development surface to select
the conclusion.

## 7. Final recommendation

The best immediate next step is not another proposal module. It is a
document-first, fresh-split, three-seed A3-versus-A0/A2 replication. This
directly tests whether the frozen LLM contributes anything beyond the strong
direct-schema baseline. Run it once under a new lock; if it fails, close the
method direction and write the audit paper.

