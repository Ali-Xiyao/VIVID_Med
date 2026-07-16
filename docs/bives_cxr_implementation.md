# BiVES-CXR Implementation Contract

## Core tensor contract

```text
patch_tokens         [B, P, Dv]
statement_embedding  [B, Ds]
valid_mask           [B, P]
gate                 [B, P]
evidence_pm          [B, P, 2]
evidence_maps        [B, P, 2]
evidence_pos/neg     [B]
state_probs          [B, 4]  # support, contradict, uncertain, insufficient
```

## Active mathematical path

1. Project patch and statement embeddings.
2. Fuse `[z, h, z*h, |z-h|]`.
3. Run a shared statement-conditioned cross-patch contextual evidence block.
4. Predict bounded positive/negative patch evidence.
5. Select an exact-K evidence set with a straight-through differentiable gate.
6. Aggregate `E+` and `E-`.
7. Derive four-state probabilities from availability, decisiveness, and
   polarity using the closed-form decoder.
8. Apply evidence-retained, evidence-deleted, and multiple random-disjoint
   equal-area masks before the shared contextual block, then re-score with
   branch-specific validity masks.

The active model must not define `state_head`, `four_class_head`, or an
independent trainable four-class classifier.

## Qwen3.5 adapter

BiVES uses merger-preceding Qwen3.5 spatial tokens:

- token count: `T * H * W` from `image_grid_thw`;
- feature dimension: `vision_config.hidden_size`;
- spatial grid for CXR: `H * W` with `T=1`.

The merger-following `pooler_output` is not the default evidence grid.
Merger-pre tokens are restored from Qwen3.5 merge-block-major order to ordinary
row-major `H * W` order before TV, masks, heatmaps, or pixel mapping.

Only `model.visual.*` safetensors are loaded. The language model, token
embeddings, and LM head are never retained or moved to the training GPU.

## Required batch contract

Each active batch contains one or more complete same-statement groups in the
fixed order support, contradict, uncertain, insufficient. The collator emits
aligned S/C pair indices and uncertain indices. A positive `lambda_pair` or
`lambda_u_pol` without these indices is a hard error.

Primary validation/calibration/test does not use the group sampler. It uses a
sequential full-row loader and asserts exact sample-ID coverage. A separate
deterministic grouped evaluator reports S/C pair-margin violation and uncertain
absolute polarity.

Letterbox padding is excluded by a row-major content-patch mask. Keep/drop and
control branches cannot reselect deleted or padded positions.

The 2B P0 config may use a learned canonical ID table. Formal 4B/9B configs
require `data/bives_cxr/statement_embeddings/qwen35_canonical.pt`, stored as a
`canonical_statement_id -> 1D tensor` mapping produced by a frozen Qwen3.5
text encoder. Missing or dimension-inconsistent caches fail before visual
weights load.

Fixed exact-K is explicitly a `K-budgeted evidence set`. It can learn which K
patches are selected, not per-example cardinality. Therefore active fixed-K
configs set `lambda_min: 0`. The adaptive/minimal evidence-set claim remains a
future hard-concrete/L0 gate.

After training, the validation-selected `best.pt` checkpoint is reloaded before
calibration or test. Positive `tau_a`, `tau_d`, and `tau_p` are fitted only on
the locked calibration split, and test artifacts report both pre- and
post-calibration classification, ECE/Brier/AURC, and intervention metrics.

Feature-space keep/drop/control is a mechanism-training contract, not by itself
proof of pixel-causal grounding. Paper-level causal claims require separate
pixel keep/drop/equal-area controls through the full frozen vision tower.

## Initial training order

1. Pass the strict manifest audit before loading Qwen3.5 weights.
2. Qwen3.5-2B, frozen vision tower, complete S/C/U/I groups.
3. Verify state, pair, uncertain-polarity, keep/drop/control, and content-mask
   behavior on a real single group.
4. Run the 2B mini mechanism experiment.
5. Move to Qwen3.5-4B only after the P0 go/no-go.
6. Use Qwen3.5-9B only for the locked scale study.

## Local verification boundary

Local checks are synthetic CPU tests only. Formal experiments run on the
server using the checked-in Qwen3.5 configs.
