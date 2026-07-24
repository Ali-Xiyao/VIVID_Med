# VIVID-GDS Stage-A Experiment Protocol

Date frozen: 2026-07-24

## Primary claims

1. Under the same 20k MIMIC identity, hard-UMS generation through pretrained
   frozen Qwen3.5-2B provides a stronger deployed ViT than deterministic
   free-text generation and direct schema classification.
2. Adding the synchronized training-only UMS schema readout to the retained
   prefix4 generation path improves the deployed ViT over UMS generation alone.

These are survival claims, not publication claims. No external test is opened.

## Frozen identities

All arms use the hard-UMS manifest and patient split with SHA-256
`1da254ab25ab8f005536ff16ac7a1c40e33f15add2afa25277a8c6e06f6e30b4`.
The exact frozen population is 19,533 train rows and 1,679 validation rows
(21,212 total); “20k” is only the historical pilot shorthand.
All trainable arms start from the same ViT-B/16 safetensors authority and use
seed 0, 3000 optimizer steps, effective batch 32, AdamW, identical augmentation,
and the same deterministic CUDA contract.

- A0 direct: ViT plus the UMS schema head; no Qwen or projector.
- A1 free text: prefix4 plus frozen Qwen3.5-2B. Each frozen UMS field-state
  pair is rendered into one deterministic natural-language sentence.
- A2 UMS: the frozen strict prefix4 checkpoint and summary.
- A3 GDS: A2 generation path plus synchronized schema head.

A1 uses the same selected fields as A2; missing fields remain absent from both
targets. A3 schema masks are parsed from the exact target string passed to the
generation path.

## Checkpoint rules

- A0: strictly lowest internal validation schema NLL.
- A1/A2/A3: strictly lowest internal validation generation token NLL.
- Downstream probes never select a pretraining checkpoint.
- Every reported downstream metric for an arm comes from its one selected
  checkpoint.

The A0 exception is unavoidable because no generation path exists; it is
declared before training and must not be changed.

## Stage M0: contract and overfit gate

Before 20k pilots:

- audit all paths, hashes, row counts, patient disjointness, field/state schema,
  teacher weights, ViT weights, and protected-test non-use;
- A0, A1, and A3 must pass the 256-row overfit gate;
- A0 schema accuracy >= 0.98 and schema NLL reduction >= 80%;
- A1 token accuracy >= 0.98 and token NLL reduction >= 80%;
- A3 must satisfy both A1 generation and A0 schema criteria;
- backbone, projector where present, and schema head where present must each
  have finite nonzero gradients.

An implementation failure permits at most one identity-preserving repair and a
restart of the failed arm from zero.

## Stage M1: frozen 21,212-row (“20k”) pilots

Run A0, A1, and A3 for 3000 optimizer steps. Reuse A2 only after its manifest,
teacher, ViT initialization, optimizer budget, and selected checkpoint hashes
match this lock. Each new arm must reduce its primary validation NLL by at least
20% from step zero before downstream probing.

## Stage M2: expert-development probes

Use the frozen CheXpert probe train and five-finding expert-development
manifests from the strict route. Probe training, checkpointing, and metrics are
identical for all arms.

Primary gates:

| Comparison | AUROC | AUPRC | Per-finding |
|---|---:|---:|---|
| A2 - A1 | >= +0.005 | >= -0.005 | >=4/5 nonnegative; <=1 below -0.02 |
| A2 - A0 | >= +0.003 | >= -0.005 | report all five |
| A3 - A2 | >= +0.005 | >= -0.005 | >=4/5 nonnegative; <=1 below -0.02 |

All three comparisons must pass for Stage-B expansion. A failure freezes
NO-GO for the failed claim. No threshold relaxation, teacher scaling, external
test rescue, or new method module is permitted.

## Authorized bounded diagnostics after scientific NO-GO

Only development-only case studies are allowed:

- generation and schema learning curves;
- per-finding schema coverage and loss;
- gradient norm ratio between generation and schema branches;
- CLS representation drift from initialization;
- disagreement cases between A0, A2, and A3 on expert-development.

At most one supported repair may be nominated, and it must preserve the method
identity, data identity, teacher, targets, threshold, and protected-test locks.
The preregistered `lambda_schema=0.25` sensitivity is not a repair until A3 has
already passed the primary gate.

## Protected surfaces

CheXlocalize test, VinDr test, test-derived thresholds, Qwen3.5-4B/9B, full
MIMIC, CheXpert-Plus, NIH, and PadChest remain locked until Stage A passes.
