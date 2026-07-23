# RCSD D0-versus-D1 review protocol

**Status:** prerequisites in progress; user execution approval recorded

**Active parent verdict:** `RCSD_P0_NO_GO_VERDICT.md`

## Decision this protocol is allowed to make

This protocol may determine whether the single selective-reliability component
is worth one paired development run. It does not reopen full RCSD, posterior
fusion, field anchoring, external testing, teacher scaling, multi-seed
training, or a new method proposal.

Training is authorized only through the state machine in
`RCSD_D0_D1_AUTO_DEV_PROTOCOL.md`. The machine lock remains non-executable
until every R0-R2 prerequisite has a frozen artifact hash or an explicit
fail-closed unavailability record.

## Three identities that must not be conflated

| Identity | Meaning | Use |
| --- | --- | --- |
| D0-H | Immutable historical CheXpert VIVID/SPD artifact | Provenance only |
| D0-CP | Original VIVID objective reconstructed on the common locked 20k MIMIC surface | Paired comparator |
| D1 | D0-CP plus one entropy-derived agreement weight | Only candidate |

D1 must be compared with D0-CP. It is invalid to compare a new MIMIC D1 result
directly against D0-H, because the data, seed, and diagnostic budget differ.

## D0-H historical contract

The accepted historical identity is:

- hard UMS JSON targets;
- frozen `Qwen/Qwen2.5-1.5B-Instruct`;
- ImageNet-initialized ViT-B/16 at 224 pixels;
- SPD with four groups and two tokens per group;
- next-token cross-entropy over the serialized UMS JSON;
- training-only SPD attention orthogonality loss with weight 0.02;
- 10,000 optimizer steps, seed 42, effective batch size 32;
- ViT learning rate 2e-5 and projector learning rate 1e-4;
- best checkpoint selected by strictly lower validation token loss.

The exact retained source identities are frozen in
`tables/rcsd_d0_source_contract.csv`. The legacy black-image fallback and
row-random fallback split are not scientific method components. D0-CP must use
the clean fail-closed loader and the existing patient lock.

Historical validation did not add the orthogonality penalty to validation
loss. D0-CP and D1 must preserve that checkpointing rule.

## Common controlled surface

Both arms must use:

- MIMIC canonical manifest SHA-256
  `00fde375c608017d5e5700f946a15f32097d44ceecec885ebae41dfc58578133`;
- current 20k-plus-validation row lock SHA-256
  `5e9e05552712a7c6298ff63731c4250c0bdc6d5e3d2a28e9e4476b7c7c242ae2`;
- the identical hard CheXbert state for every supervised finding;
- the identical deterministic UMS JSON serialization;
- the identical image bytes, augmentation, initialization, seed, batch,
  optimizer, precision, 3,000-step budget, and checkpoint rule;
- one sequential GPU allocation if a later approval is granted.

The hard-UMS manifest does not yet exist. Its hash must remain `null` until it
is built without opening an external test surface.

## D1 is one change only

For finding \(c\) in study \(i\), each available source emits a one-hot report
state over:

\[
\mathcal{S}=\{\text{present},\text{absent},\text{uncertain}\}.
\]

Missing sources are excluded. The unweighted mean is:

\[
\bar q_{ic}=\frac{1}{K_{ic}}\sum_{k \in O_{ic}}q_{ikc}.
\]

The frozen CheXbert state remains the hard target. Its agreement weight is:

\[
w_{ic}=m_{ic}
\left(1-\frac{H(\bar q_{ic})}{\log 3}\right),
\]

\(m_{ic}=0\) only when the CheXbert hard target is missing; otherwise it is
one. If only CheXbert is observed, \(w_{ic}=1\): missing corroboration is not
treated as disagreement.

For D0-CP every valid token has weight one. For D1, tokens belonging to a
finding block receive \(w_{ic}\); structural tokens retain weight one. The
weighted token mean is normalized by the sum of applied token weights, so D1
does not alter the nominal loss scale:

\[
L_{\mathrm{tok}}^{D1}=
\frac{\sum_{it}w_{it}\,\mathrm{CE}_{it}}
     {\sum_{it}w_{it}}.
\]

Both arms add the same training-only \(0.02L_{\mathrm{ortho}}\).

D1 may not:

- replace a hard state;
- learn source weights or a label model;
- use the failed calibrated posterior;
- use field anchors or Qwen3.5 field prototypes;
- change the teacher, query layout, prompt, data, augmentation, optimizer,
  checkpoint rule, or evaluation surface.

## Required pre-run gates

### R0: source and implementation parity

All must pass:

- import the D0-H checkpoint SHA and historical audit SHA, or record that the
  retained local checkpoint surface is empty and treat D0-H as unavailable
  provenance only;
- implement the clean frozen-teacher token objective;
- verify deterministic UMS serialization against retained historical examples;
- verify SPD output, parameter count, 4x2 token layout, and orthogonality loss;
- verify validation NLL excludes the orthogonality term;
- verify D0 and D1 render byte-identical hard target strings.

### R1: data and reliability lock

All must pass:

- zero patient overlap and zero image-identity conflict;
- hard-UMS manifest hash frozen;
- three-source input manifest hash frozen;
- D1 reliability manifest hash frozen;
- every weight is finite and in \([0,1]\);
- missing CheXbert targets have zero weight;
- no hard target differs between D0 and D1;
- reliability coverage and quartiles reported before visual training.

### R2: expert-development lock

Freeze one identical linear-probe training manifest and the CheXpert expert
validation manifest. This is an already exposed development surface, not final
evidence. CheXpert test and CheXlocalize test remain sealed.

### R3: 256-row paired overfit

Only after R0-R2 and separate execution approval:

- D0 and D1 token accuracy at least 0.98;
- loss reduction at least 80%;
- no NaN or Inf;
- nonzero finite gradients in ViT and SPD;
- no target or row difference between arms.

Failure stops the route before the 20k pilot.

## One allowed paired pilot after approval

| Item | Frozen value |
| --- | --- |
| Train surface | patient-locked 20k MIMIC studies |
| Validation surface | locked MIMIC validation rows |
| Seed | 0 |
| Steps | 3,000 |
| Effective batch | 32 |
| Precision | BF16 |
| Teacher | Qwen3.5-2B, frozen; other sizes conditional after the primary gate |
| Checkpoint | strictly lower unweighted validation token NLL |
| Arm order | D0-CP, then D1 |
| Restarts | none except an implementation failure before any metric is viewed |

## Development endpoints

The primary mechanism endpoint is unweighted validation token NLL. The
downstream endpoint is a matched linear probe evaluated on the five-label
CheXpert expert validation surface:

- macro AUROC;
- macro AUPRC;
- 15-bin classwise ECE;
- per-finding AUROC and AUPRC;
- patient-paired bootstrap confidence intervals.

The same frozen source checkpoint must produce all endpoints. Missing AUROC or
AUPRC values remain `null`; they are not reconstructed from F1.

## Promotion gate

D1 passes only if every condition holds relative to D0-CP:

| Condition | Frozen threshold |
| --- | ---: |
| Validation token NLL relative change | at most -3% |
| Expert-dev macro AUROC | at least +0.5 percentage points |
| Expert-dev ECE change | at most +0.01 |
| Findings below -2 AUROC points | at most 2 |
| Positive prevalence tiers | at least 2 of 3 |
| Findings with positive AUROC delta | at least 3 of 5 |

Macro AUPRC and paired confidence intervals are mandatory reported secondary
endpoints. They cannot override a failed primary gate. Thresholds may not be
changed after a result is observed.

## Required diagnostic tables

Every unavailable value remains `null` until its frozen producer has run.

### Table A: label and reliability layer

| Supervision | Status | Coverage | Macro-F1 | NLL | ECE | Reliability AUROC | High-low accuracy gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CheXbert hard target | frozen G2 evidence | 5,730 labels / 1,455 studies | 0.800504 | 0.371156 | 0.005906 | 0.800146 | 0.296089 |
| D1 entropy agreement | not run; same hard target | null | 0.800504 | 0.371156 | 0.005906 | null | null |
| D2 calibrated posterior | terminal G2 NO-GO | 5,730 labels / 1,455 studies | 0.813270 | 0.397067 | 0.015316 | 0.748079 | 0.261173 |

Table A must distinguish label correctness from visual utility. D1 does not
change the CheXbert hard prediction, so its key label-layer endpoint is whether
the weight ranks correctness, not a post-hoc relabelled F1.

### Table B: paired visual pilot

| Arm | Status | Unweighted val token NLL | Expert-dev macro AUROC | Macro AUPRC | ECE | Worst finding delta | Train stability |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| D0-CP | not run | null | null | null | null | null | null |
| D1 | not run | null | null | null | null | null | null |

### Table C: per-finding paired changes

| Finding | D0 AUROC | D1 AUROC delta | D0 AUPRC | D1 AUPRC delta | D1 coverage | Prevalence tier |
| --- | ---: | ---: | ---: | ---: | ---: | --- |

All five expert-development findings must be present, including negative or
undefined results.

### Table D: reliability strata

| Reliability quartile | Studies | Fields | Label accuracy | D0 token NLL | D1 token NLL | D0 AUROC | D1 AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

Quartile cut points must be computed on the locked training rows and reused
unchanged for validation. The table must also report finding and prevalence
composition so that “high reliability” is not silently equivalent to “common
easy finding.”

## Unique terminal decisions

- **D1 passes:** permit a new proposal review for selective
  reliability-weighted structured distillation. Do not automatically start
  full-data or multi-seed experiments.
- **D1 fails:** close RCSD reliability and field-anchor method development.
  The surviving paper-one option is the clean historical VIVID/SPD extension
  with modern controlled validation, not another rescue module.
- **Implementation/data gate fails:** repair only the failed contract before
  metrics exist; do not reinterpret it as a scientific result.

## Current blockers

The machine lock intentionally records these as incomplete:

- historical checkpoint and audit hashes have not been imported into Git;
- the clean D0 frozen-teacher token objective is not implemented;
- D0 serialization/SPD parity tests are incomplete;
- hard-UMS and D1 reliability manifests are not frozen;
- expert-development manifests are not frozen;
- no paired launcher has passed review;
- explicit execution approval is recorded, but all other implementation,
  manifest, and launcher prerequisites remain incomplete.

Therefore the current operational decision is:

> **PREREQUISITES IN PROGRESS - ZERO TRAINING JOBS AUTHORIZED UNTIL R0-R2
> PASS.** This sentence supersedes the stale review-only marker below.

> **REVIEW PACKAGE ONLY — ZERO TRAINING JOBS AUTHORIZED.**
