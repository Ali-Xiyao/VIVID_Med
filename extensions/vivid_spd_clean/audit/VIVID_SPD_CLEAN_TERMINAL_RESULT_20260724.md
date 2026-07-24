# Strict VIVID/SPD Clean Extension — Terminal Result

## Verdict

**`TERMINAL_NO_GO`**

The historical hard-UMS SPD 4x2 route failed the frozen S3 promotion gate.
Both preregistered bounded diagnostics were then completed. Neither the
eight-prefix sequence-budget control nor removal of the historical
orthogonality term supported a repaired SPD identity. S4 multi-seed/full-data
training, larger Qwen3.5 teachers, and external evaluation remain locked.

This terminal result does not invalidate the broader observation that
structured UMS supervision can learn a useful CXR encoder. It specifically
rejects promotion of SPD 4x2 as a stable improvement over the paired prefix4
baseline under this frozen Qwen3.5-2B, hard-UMS, 20k-study protocol.

## Frozen strict comparison

| Arm | Macro AUROC | Macro AUPRC | Delta AUROC vs prefix4 | Gate |
| --- | ---: | ---: | ---: | --- |
| `ums_prefix4` | 0.859209 | 0.690875 | — | reference |
| `ums_spd4x2` | 0.863850 | 0.694079 | +0.004641 | NO-GO |

The required AUROC delta was `+0.005`; only three of five findings were
nonnegative versus the required four.

Historical SPD per-finding AUROC deltas:

| Finding | Delta |
| --- | ---: |
| Atelectasis | -0.033491 |
| Cardiomegaly | +0.037322 |
| Consolidation | -0.005147 |
| Edema | +0.016369 |
| Pleural Effusion | +0.008152 |

## Bounded diagnostics

| Arm | Macro AUROC | Macro AUPRC | Delta AUROC vs prefix4 | Delta vs historical SPD | Nonnegative findings | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `ums_prefix8` | 0.857884 | 0.686250 | -0.001325 | -0.005966 | 2/5 | diagnostic negative |
| `ums_spd4x2_no_ortho` | 0.858864 | 0.703288 | -0.000345 | -0.004986 | 3/5 | repair not supported |

No-ortho improved macro AUPRC, but it failed the unchanged primary macro
AUROC and per-finding gate and was worse than historical SPD. It therefore
cannot be nominated as the single repaired identity.

## Attention-group mechanism diagnostic

The descriptive analysis used the lexicographically first 128 rows of the
frozen MIMIC `validate` split and had no promotion effect.

| Arm | Mean pairwise group-attention cosine | Maximum cosine | Group entropy range |
| --- | ---: | ---: | ---: |
| historical SPD, ortho 0.02 | 0.0000718 | 0.007122 | 0.430–0.586 |
| SPD no-ortho | 0.992841 | 0.998974 | 0.988–0.994 |

The historical orthogonality term strongly prevents group-attention collapse.
Removing it causes the four groups to become almost identical, high-entropy
attention maps. Because no-ortho also reduces macro AUROC, the evidence does
not support removing the term as a repair.

## Evidence hashes

- strict S3 verdict:
  `426f46e54b7cc29f7647ba4bf62ca639de8f842498259737222fc6cd0004353a`
- strict prefix4 S3 summary:
  `f707a740d877ce33bc181e365d8031d93a0641d9666e7caf830b8ee063450134`
- strict SPD S3 summary:
  `b318adc981b5e9e025f6bc458ddaebcb1a62e69c200285e4ac1a36db0f85ae79`
- bounded diagnostic verdict:
  `b6f13c5c15f8c638105703a97034b4157cf8c51813c5c3506e7ad1c3cfae44cd`
- prefix8 S3 summary:
  `0a50d6a6e6666b4b94d3fb98050798ef17bd5bad44d5b4380bf612c09f77ff38`
- no-ortho S3 summary:
  `30e5f4a5bd9d3305e614dca84af3f5b64342aa6be91b90067627232ea3aab57a`
- historical SPD attention diagnostic:
  `2c7e19c4562e2b07859098e04c8c3d1fa62287804b4da8a72bae1ab6cc0abe87`
- no-ortho attention diagnostic:
  `7e09c03278e2916f611f02669043de321e0da7d7d65082091b6421c8df9f372f`

## Implementation incidents

Three implementation incidents were preserved and repaired without changing
scientific identity:

1. deterministic cuBLAS workspace configuration before strict training;
2. Python `False` literal in the post-training verdict writer;
3. `validate` split literal in the final read-only attention diagnostic.

No failed run was silently overwritten. No external, CheXlocalize test, or
VinDr test surface was opened.

## Frozen next-step boundary

Do not:

- run S4 or S5;
- add seeds or full MIMIC scale;
- scale to Qwen3.5-4B/9B;
- change the hard-UMS target, thresholds, split, or checkpoint rule;
- introduce a second repair;
- open CheXlocalize test or VinDr test.

Any future VIVID expansion must start as a new scientific identity and new
lock, not as continuation or rescue of this terminal strict SPD route.
