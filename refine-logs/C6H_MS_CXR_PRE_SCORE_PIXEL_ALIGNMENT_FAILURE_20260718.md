# BiVES-CXR C6H pre-score pixel-alignment failure

**Date:** 2026-07-18

**Status:** `FAIL_PRE_SCORE_PIXEL_ALIGNMENT_NO_RESULT`

**Pre-open commit:** `a9f431c15f08c8424fdb72293dafe77708a4fc09`

## What opened

The separate user-authorized C6H evaluator passed its committed-source,
authority, config, C6E/C6F/C6G, checkpoint/cache, model-snapshot, and live-GPU
gates. It created its one-time opening marker and loaded the frozen
Qwen3.5-2B/B2 step-450 checkpoint on local GPU1.

## Fail-closed point

The evaluator stopped on the first bound image before `score_original`, any
vision forward, or any score. The bound JPG is 224x224, while its MS-CXR
annotation geometry declares 3056 columns by 2544 rows. The same audit over
the complete denominator found:

```text
actual JPG sizes: 29/29 are 224x224
declared native sizes: 10 distinct dimensions
actual-versus-declared mismatches: 29/29
```

C6G generated target/control/content masks in the declared native-resolution
letterbox coordinate system. Applying those masks to a square 224x224 image
would silently misregister both expert and control interventions. Removing the
size check or stretching only at evaluation time is therefore forbidden.

## Artifact evidence

| Artifact | SHA-256 / state |
| --- | --- |
| C6H pre-open lock file | `427ffa9d649f572f1f0301a26cf7fef4f1cfc47fbe44f4dcd413e394a477d0d2` |
| C6H pre-open lock canonical | `0925a7151fe5f3578c772daee64229c7791d4f02e72e6a58c14f78b1c3aa7786` |
| Opening marker | `18faf86b4e0a792d9d146d80c2adca683b4e1db67261812845d6d77eb9342dab` |
| stdout | empty; SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| stderr | 435 bytes; SHA-256 `3f6481bc958b02061b88a974c13149af6f5d8baf9164663da3439965117857e7` |
| progress | absent |
| evaluation rows | absent |
| final metrics | absent |

GPU1 returned to 13 MiB used and 0% utilization after the fail-closed exit.

## Local source search

The exact first image exists in both the bound local MIMIC tree and the public
dataset MIMIC tree, but both copies are the same 224x224 payload and SHA-256.
No coordinate-matched full-resolution copy was found in the local public-data
root. The auxiliary `mimic-cxr_less` tree does not contain this image.

## Decision

C6H has no result and cannot be interpreted as pass or mechanism fail. The
one-time opening identity is consumed by a pre-score implementation/data gate.
No relaunch is allowed under the same C6H identity.

The only scientifically valid local recovery is a new score-free input-space
geometry protocol that scales released boxes into the actual 224x224 pixel
space, uniformly regenerates all target/control/content masks, and reruns the
29/29 geometry gate before any further model access. Because this changes the
frozen control/mask identity after an opening, it requires a separate explicit
recovery authority and a new one-time model-opening identity. It may not rewrite
C6F, C6G, or this failed C6H record.
