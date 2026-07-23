# Strict VIVID/SPD Clean-Extension Protocol

Version: 1.0  
Frozen date: 2026-07-23  
Status: development execution authorized; scale and external stages locked

## 1. Scientific question

Under identical hard structured medical supervision, image data, frozen
Qwen3.5-2B teacher, ViT-B/16 initialization, optimizer, prompt, target
serialization, training budget, and checkpoint selection, does historical SPD
produce a more transferable visual encoder than the historical four-prefix
projector?

The deployed artifact is the ViT encoder only. The teacher and projector are
training-time components.

## 2. Primary paired arms

### P0: `ums_prefix4`

- deterministic hard UMS target;
- Qwen3.5-2B frozen;
- ViT-B/16, 224x224, all tokens;
- historical two-layer MLP with four learned prefix tokens;
- prefix tokens are concatenated with projected ViT tokens;
- no reliability weighting and no orthogonality loss.

### P1: `ums_spd4x2`

- every P0 item remains identical;
- replace the prefix projector by four groups x two learned query tokens;
- independent four-head cross-attention per group;
- shared two-layer projection and projected ViT tokens;
- fixed orthogonality coefficient `0.02`.

The main comparison is deliberately historical, not parameter- or
sequence-length matched. `ums_prefix8` may run only as a diagnostic after a
scientific failure and may not substitute for P0 in the primary claim.

## 3. Data and target identity

- Initial source: the already-frozen MIMIC-CXR hard-UMS development manifest.
- Pilot train size: 20,000 studies.
- Validation: the frozen patient-disjoint internal validation subset.
- Overfit: the frozen 256-row train-only identity.
- Target: deterministic JSON with modality, observed finding states, and view.
- `missing` fields are omitted; they are never converted to absent.
- No RCSD posterior, source-agreement weight, field anchor, or soft target.

All manifests, row locks, source weights, teacher weights, backbone weights,
and scripts must be SHA-256 recorded before execution.

## 4. Training identity

- teacher: local Qwen3.5-2B, frozen;
- backbone: ViT-B/16, fixed initialization;
- image transform: 224x224; train-only mild crop, +/-5 degree rotation, mild
  brightness/contrast; deterministic resize for validation/overfit;
- optimizer: AdamW;
- backbone LR: `2e-5`;
- projector LR: `1e-4`;
- weight decay: `0.01`;
- gradient clip: `1.0`;
- BF16;
- effective batch size: 32 unless the locked hardware requires a smaller
  microbatch with the same accumulation-equivalent batch;
- warmup: 500 optimizer steps;
- pilot budget: 3,000 optimizer steps;
- seed: 0 for S1-S3; seeds 0/1/2 only after promotion;
- checkpoint: strictly minimum unweighted internal validation token NLL.

No downstream metric may select a checkpoint.

## 5. Gates

### S0: identity and readiness

All must pass:

- zero train/validation patient overlap;
- all selected images present and decodable;
- teacher and backbone files present and hashed;
- exact P0/P1 architecture contract passes;
- hard targets identical between arms;
- protected surfaces remain unopened;
- allocation ownership and free GPU capacity verified.

### S1: paired 256-row overfit

Each arm must independently achieve:

- token accuracy >= 0.98;
- token NLL reduction from step 0 >= 80%;
- finite nonzero gradients in backbone and projector;
- no NaN/Inf;
- no arm-specific data, prompt, or target change.

An implementation failure permits one identity-preserving repair and a fresh
rerun of both arms. Otherwise stop.

### S2: paired 20k token pilot

Each arm must independently achieve:

- validation token NLL reduction from step 0 >= 20%;
- finite training and validation metrics;
- best checkpoint selected by unweighted validation token NLL;
- complete hashes, logs, summaries, and vision-only checkpoint.

S2 establishes learnability, not the SPD contribution.

### S3: frozen CheXpert expert-development linear probe

Train the identical linear-probe protocol on the frozen CheXpert training
manifest and evaluate once on the expert development manifest. Use the ViT
checkpoint selected in S2.

The primary endpoint uses the five findings actually annotated on the
CheXpert expert validation surface: Atelectasis, Cardiomegaly, Consolidation,
Edema, and Pleural Effusion. A patient-disjoint 12-finding auto-label
development split may be reported only as secondary evidence.

Primary promotion requirements for P1 versus P0:

- macro AUROC delta >= +0.005;
- macro AUPRC delta >= -0.005;
- at least 4 of 5 expert findings have nonnegative AUROC delta;
- no more than one expert finding declines by more than 0.02 AUROC;
- all metrics derive from the single locked checkpoint per arm.

If all hold, promote. Otherwise freeze a strict NO-GO before scale-up.

### S4: three-seed and full-data promotion

Only after S3 PASS:

- repeat the paired experiment at seeds 0, 1, 2;
- SPD direction positive in at least 2/3 seeds;
- paired patient-bootstrap 95% CI reported;
- then run full MIMIC only if the three-seed development result remains
  positive.

### S5: external evaluation

Only after method and checkpoint policy freeze:

- preregister ontology mappings;
- use NIH and an unexposed external CXR surface;
- do not use VinDr test or CheXlocalize test;
- external results cannot select the method.

## 6. Failure case study and bounded repairs

After an S3 scientific NO-GO, development-only diagnostics may run:

1. `ums_prefix8` to test sequence-budget confounding;
2. `ums_spd4x2_no_ortho` to test whether the historical orthogonality term
   dominates optimization;
3. attention-group cosine/collapse and per-finding probe deltas;
4. high/low prevalence and view-position strata.

Only one method repair may be nominated, and only when a listed diagnostic
directly supports it. The repaired identity must be written to a new lock and
rerun from initialization. It cannot retroactively convert the strict P1
result into a pass. No threshold, teacher, target, split, or protected test
surface may change.

## 7. Terminal outcomes

- `STRICT_PASS`: S0-S3 pass; scale-up unlocks.
- `STRICT_NO_GO_DIAGNOSTIC_OPEN`: S3 fails; bounded diagnostics may run.
- `REPAIRED_PASS`: one preregistered repair passes a fresh gate; report it as a
  distinct method.
- `TERMINAL_NO_GO`: strict route and the single supported repair fail, or no
  diagnostic supports a repair.
- `BLOCKED`: user authority, missing data/model license, allocation ownership,
  or unrepairable infrastructure prevents valid execution.

The project stops at a valid terminal outcome. It does not run until a
favorable number appears.
