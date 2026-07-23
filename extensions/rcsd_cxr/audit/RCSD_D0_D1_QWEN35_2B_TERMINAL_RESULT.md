# RCSD D0/D1 Qwen3.5-2B terminal result

**Recorded:** 2026-07-23

**Decision:** terminal NO-GO for D1 selective agreement weighting

**Run:** `d0_d1_token_pilot_qwen35_2b_20260723_s2`

**Test sets opened:** none

## Outcome

Both frozen arms completed successfully on allocation `3066`, but D1 failed
the first required scientific promotion condition.

| Arm | Best step | Validation token NLL | Token accuracy | Elapsed |
| --- | ---: | ---: | ---: | ---: |
| D0-CP | 3,000 | 0.0768119148 | 0.9698801932 | 5,915.03 s |
| D1 | 3,000 | 0.0772889282 | 0.9697445184 | 8,030.78 s |

D1 changed validation token NLL by **+0.6210%**, meaning it was worse than
D0. The frozen gate required a relative change of at most **-3%**. Because all
promotion conditions were required, this first failure closes the gate.
Expert-development probing cannot rescue it and was not run.

Queue-level `pass: true` means both commands completed with return code zero.
It is not a scientific pass.

## Matched validation trajectory

| Step | D0 NLL | D1 NLL | Relative D1 change | Direction |
| ---: | ---: | ---: | ---: | --- |
| 500 | 0.0868966725 | 0.0883252861 | +1.6440% | favors D0 |
| 1,000 | 0.0820955950 | 0.0827696097 | +0.8210% | favors D0 |
| 1,500 | 0.0805892760 | 0.0813829365 | +0.9848% | favors D0 |
| 2,000 | 0.0785858087 | 0.0790323267 | +0.5682% | favors D0 |
| 2,500 | 0.0774704031 | 0.0780412085 | +0.7368% | favors D0 |
| 3,000 | 0.0768119148 | 0.0772889282 | +0.6210% | favors D0 |

The failure is not caused by an unlucky checkpoint choice: D1 was worse at
every frozen matched validation point, and both arms selected step 3,000 under
the same strictly-lower unweighted-NLL rule.

## Development-only case study

The scalar agreement mechanism had limited support:

| Diagnostic | Train | Validation |
| --- | ---: | ---: |
| Rows with any downweighted finding | 4,011 / 19,533 (20.53%) | 330 / 1,679 (19.65%) |
| Downweighted observed finding fields | 4,791 / 89,622 (5.35%) | 385 / 7,676 (5.02%) |
| Mean finding weight | 0.9687 | 0.9704 |

After frozen Qwen3.5 tokenization, only 6,853 of 169,523 validation target
tokens (4.04%) received a weight below one. Mean target-token weight remained
0.9762. The most concentrated effect was Pneumonia: 19.75% of its observed
training fields were downweighted and its mean field weight was 0.8851.

This supports a bounded interpretation:

1. D1 is trainable; both the 256-row overfit gate and finite-gradient audits
   passed.
2. D1 adds no new target information. It only suppresses a small, unevenly
   distributed subset of the same hard CheXbert finding blocks.
3. Most target tokens remain identical to D0 in both target and weight. The
   mechanism therefore has little opportunity to improve the unweighted
   validation objective.
4. Report-source disagreement is not equivalent to an incorrect CheXbert
   hard target. Selectively suppressing disagreement can remove useful signal,
   which is consistent with the small but persistent D1 deficit.
5. These diagnostics explain the observed failure but do not justify a new
   weight transformation. Changing the formula, floor, scope, labels, teacher,
   or threshold would be a new post-hoc factor.

D1 also took 35.77% longer than D0 in this shared environment. This is an
execution diagnostic, not a scientific endpoint, and cannot alter the gate.

## Validity checks

- Same 19,533-train / 1,679-validation patient-locked surface.
- Same seed, hard targets, Qwen3.5-2B teacher, ViT-B initialization, SPD 4x2,
  optimizer, 3,000-step budget, and checkpoint rule.
- Both arms completed with return code zero.
- ViT and projector gradients were finite and nonzero.
- No external, CheXlocalize, VinDr test, or confirmatory surface was opened.
- The first promotion failure was evaluated without changing its threshold.

## Evidence identities

| Artifact | SHA-256 |
| --- | --- |
| D0 summary | `6d39051eecb6fb00af58da2b14269fe2f7eb2f4262f85dd393f43b9e9c6b52dd` |
| D0 checkpoint | `37ab60d2782043d0cffc7e8925b385143264731f4283112e5666744384e14895` |
| D1 summary | `200d5c3cc541ccee2634b2781f71eb9bf03f26cacab00c9b3291c243831178ab` |
| D1 checkpoint | `f08357394e4c96ef1c4d2d42b92add9eb2e8313a45e5a0cee616fb3216b3023e` |
| Final queue state | `c7c8c26a8910497b3df237fc37d79f373b1dc0beaf2824fd19cdde167d1a1548` |
| Hard-UMS manifest | `1da254ab25ab8f005536ff16ac7a1c40e33f15add2afa25277a8c6e06f6e30b4` |
| D1 reliability manifest | `a06959dfb98a2074a638dfb032c39e540dd3bb9bd2f3b2eb66f9e352b914f692` |

The remote summaries and checkpoints remain outside Git. The machine-readable
deidentified result is
`audit/rcsd_d0_d1_qwen35_2b_terminal_result.json`.

## Binding decision

There is no preregistered in-identity repair for a valid D1 scientific
failure. Under the reviewed protocol:

- D1 selective agreement weighting is terminal NO-GO.
- RCSD posterior fusion and field anchoring remain terminal NO-GO.
- No D1 repair, expert-development probe, teacher-size sweep, full-data run,
  multi-seed run, institution mixing, or external test is authorized.
- Qwen3.5-0.8B/4B/9B are not run because the primary 2B gate failed.
- The surviving paper-one route is the strict historical VIVID/SPD extension
  with modern controlled validation, not another RCSD rescue component.
