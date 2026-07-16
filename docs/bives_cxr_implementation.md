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
3. Predict bounded positive/negative patch evidence.
4. Select the evidence set with a differentiable gate.
5. Aggregate `E+` and `E-`.
6. Derive four-state probabilities from availability, decisiveness, and
   polarity using the closed-form decoder.
7. Re-score evidence-retained, evidence-deleted, and equal-area control
   interventions.

The active model must not define `state_head`, `four_class_head`, or an
independent trainable four-class classifier.

## Qwen3.5 adapter

BiVES uses merger-preceding Qwen3.5 spatial tokens:

- token count: `T * H * W` from `image_grid_thw`;
- feature dimension: `vision_config.hidden_size`;
- spatial grid for CXR: `H * W` with `T=1`.

The merger-following `pooler_output` is not the default evidence grid.

## Initial training order

1. Qwen3.5-2B, frozen vision tower, state loss only.
2. Ramp in keep/drop/control closure.
3. Add minimality and TV regularization.
4. Add same-statement S/C pair ranking after group coverage is audited.
5. Switch from soft-topK to hard-concrete only after mask collapse audits pass.
6. Move to Qwen3.5-4B after P0 go/no-go.
7. Use Qwen3.5-9B only for the locked scale study.

## Local verification boundary

Local checks are synthetic CPU tests only. Formal experiments run on the
server using the checked-in Qwen3.5 configs.
