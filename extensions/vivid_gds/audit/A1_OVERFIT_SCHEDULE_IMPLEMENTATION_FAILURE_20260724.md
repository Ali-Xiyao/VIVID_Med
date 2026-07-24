# A1 overfit schedule implementation failure

## Frozen first-run result

`A1_freetext` ran all 500 original overfit steps and returned code 4:

- initial token NLL: `2.367224`;
- final token NLL: `0.117243`;
- NLL reduction: `95.0473%` (passes the 80% requirement);
- final token accuracy: `0.964316` (fails the fixed 0.98 requirement);
- backbone and projector gradients were finite and nonzero;
- the learning curve was still improving through step 500.

## Cause

The optimizer schedule used a 500-step linear warmup while the overfit
feasibility budget was also exactly 500 steps. The generative arm therefore
received no post-warmup optimization interval. This schedule mismatch is not
evidence that deterministic free-text targets are unlearnable.

## One identity-preserving repair

The generative overfit feasibility budget is increased to 1000 steps: the same
500-step warmup plus 500 post-warmup steps. The accuracy and NLL thresholds,
seed, 256 row IDs, hard UMS authority, deterministic free-text rendering,
Qwen3.5-2B, ViT weights, prefix4, optimizer, learning rates, and 3000-step
pilot remain unchanged.

The original run root and logs remain immutable. G1 is restarted from zero in
a new run root. No further overfit schedule repair is allowed.
