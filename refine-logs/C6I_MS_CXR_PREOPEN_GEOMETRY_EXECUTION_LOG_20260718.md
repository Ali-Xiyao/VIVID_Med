# BiVES-CXR C6I pre-open actual-input geometry execution log

**Date:** 2026-07-18

**Status:** `PASS_SCORE_FREE_29_OF_29_READY_FOR_REPLACEMENT_PREOPEN`

**Source commit:** `611ab1dbd22edcbc952ad89c38a30a642615aee6`

## Authorized question

C6I repairs only the input coordinate surface exposed by the failed C6H
opening. It maps every released MS-CXR box from declared native coordinates to
the exact hash-bound 224x224 JPG using independent x/y scale factors, then
applies the existing deterministic 448x448 Qwen input transform. C6F, C6G, and
failed C6H were not edited or rerun.

## Score-free execution

The formal 8-worker build completed in 406.0 seconds. An independent 8-worker
replay in a separate ignored directory completed in 407.3 seconds. Both runs
reported:

```text
rows=29
eligible=29
infeasible=0
denominator_exclusions=0
actual_image_sizes={224x224: 29}
evaluation_gate_open_geometry=true
model_evaluation_authorized=false
gpu_authorized=false
scores_accessed=false
```

The 29 source rows contain 10 distinct declared native sizes, but all exact
bound JPG bytes remain 224x224. The transformed target masks range from 4,340
to 80,400 pixels on the 448x448 input canvas. Selected controls comprise 25
target-boundary-growth, 3 lattice33, and 1 lattice17 candidates. The maximum
selected location distance is `0.08364753868580396`; the maximum selected
absolute log-perimeter ratio is `0.8917496229580375`, both below the frozen
C4/C5 limits.

## Deterministic replay and hashes

| Artifact | SHA-256 | Replay |
| --- | --- | --- |
| geometry rows | `c3a24252837e1c08482bc598004abc2dc6174e2ba3f328ddbf649b4017ac1cdf` | byte-identical |
| candidate certificates | `e2ac8acc88d50c1c8c935e64bb7f1e5d773315e710d9226bcde2f2475b56034a` | byte-identical |
| geometry lock file | `6f064985ef8b89813e4a69c45a691ca2e92acbaf8df9586018881859176bb40b` | byte-identical |
| geometry lock canonical | `f6e6c8e6a4e7499376d8b316d588197fb1e57ae18a68b6c529dd31e60e531a0e` | identical |
| 29 compressed mask files | individually hash-bound | 29/29 identical; 0 mismatches |

Independent validation rechecked every mask as 448x448, full-square content,
non-empty target, exact target/control area, target disjointness, content
containment, and exactly one 4-connected control component.

## Code validation

- new C6I contract tests: 4/4 pass;
- complete active BiVES suite: 145/145 pass;
- CPU smoke: finite gradients, normalized probabilities, no flat state head;
- `py_compile`: pass;
- `git diff --check`: pass;
- frozen C6F/C6G/C6H tracked paths: no diff.

No Qwen/checkpoint was loaded, no GPU was used, and no score was accessed in
this phase. The next legal action is to commit this score-free evidence, build
the ignored C6I pre-open lock from the clean committed identity, recheck the
local GPU, and then consume the separately authorized replacement opening once.
