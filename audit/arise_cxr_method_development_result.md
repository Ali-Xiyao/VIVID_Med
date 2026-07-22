# ARISE-CXR local method-development result

## Verdict

The bounded local ARISE-CXR development ladder is complete and fails closed
before selector training. The work produced a real mechanism improvement, but
it did not satisfy the preregistered survival gate.

The strongest admissible candidate is a Qwen3.5-2B patch-MIL verifier trained
with VinDr-train box supervision and evaluated with full pixel intervention,
complete visual re-encoding, and result-blind statistics-matched controls.
Three of four finding/operator cells have patient-cluster confidence intervals
above zero. Pleural-effusion Gaussian blur remains positive on average but its
95% interval crosses zero. Only two findings are available in the locked
oracle surface, while the gate requires three passing findings.

Therefore:

- the ARISE S/C selector remains locked;
- U/I and four-state training remain locked;
- CheXlocalize test remains absent and unopened;
- no Qwen3.5-4B/9B scaling is justified;
- no additional same-validation tuning is admissible.

This is nonconfirmatory, prior-exposed validation development evidence. It is
not a successful final method claim.

## Development ladder

| Stage | Main change | Result | Decision |
| --- | --- | --- | --- |
| Frozen zero-shot oracle | Replay the Phase-H expert-mask rows | Consolidation passes; pleural effusion fails both operators | Train a proposition verifier before any selector |
| Dense B1 head | Frozen visual features plus the historical dense head | Intervention margins collapse near `1e-6` | Retain only as a negative scale control |
| Pooled logistic head | One-factor score-scale repair | Order-one margins return; pleural target remains weaker than control | Score scale alone is not the root cause |
| Weak-S/C patch MIL | Patch-level smooth-max/mean verifier | Macro AUROC/AUPRC `0.8361/0.8349`; all means positive, three intervals cross zero | Add result-blind spatial supervision |
| VinDr box-supervised MIL | Conservative box-cell overlap and pointing/ranking loss | Macro AUROC `0.95052`; pointing hit `0.74186` (`+0.09944`) | Localization gate passes; run full-reencoding oracle |
| Frozen geometric controls | Same locked Phase-H expert masks and operators | Consolidation blur passes; three cells fail and pleural blur is negative | Trigger the prepared result-blind control diagnostic |
| Statistics-matched controls v2 | Exact-area, connected, same-zone controls matched on original-image statistics and perimeter | Three cells pass; all four means are positive; pleural blur interval still crosses zero | Final fail-stop before selector |

## Final locked result

The final matrix contains 99 image-finding pairs, 198 operator rows, and 70
patients. It was executed locally in four patient-disjoint shards and merged
only after row identity, hash, shard, and patient-overlap checks passed.

| Finding | Operator | Pairs | Patients | Mean `CS_X` | 95% patient-cluster CI | Gate |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Consolidation | local mean ring 8 | 33 | 32 | `0.16660` | `[0.02242, 0.30115]` | pass |
| Consolidation | Gaussian blur sigma 8 | 33 | 32 | `0.30928` | `[0.18602, 0.44791]` | pass |
| Pleural effusion | local mean ring 8 | 66 | 63 | `0.09316` | `[0.00467, 0.19371]` | pass |
| Pleural effusion | Gaussian blur sigma 8 | 66 | 63 | `0.01824` | `[-0.05995, 0.09065]` | fail |

Compared with the frozen geometric-control run, statistics matching increased
mean `CS_X` by `+0.07935`, `+0.04511`, `+0.01285`, and `+0.03611` in the four
cells respectively. It moved consolidation/local-mean and pleural/local-mean
intervals above zero and changed pleural/blur from a negative to a positive
mean. The remaining failure is consequently not dismissed as a control-family
implementation artifact.

## Case-study decision

The identifier-free final case study reports:

- no score-amplitude collapse;
- no target inertness;
- no mean matched-control excess;
- no operator sign reversal;
- positive operator means for both findings.

Pleural-effusion blur has a positive fraction of `0.5455` and a small median
`CS_X=0.01196`. Its residual failure is weak, heterogeneous image-level causal
reliance, not a diagnosed code, geometry, orientation, or control-strength
fault. The automatic repair ladder therefore ends here under the fail-closed
rule rather than tuning the same exposed validation result.

## Frozen provenance

- final result canonical SHA-256:
  `0f118a1ac7e534dfb91116dc2f85137e13c337463f079f2f4da493f0ed986f52`;
- final rows SHA-256:
  `4d340d309b989d6e73ff513337c7bc03d0db7581de901ec06672dbc0701bf0b4`;
- final case-study canonical SHA-256:
  `2d656aafd5bbe24527a2eaabdeb41fe8dfe86d357652ef3bc0be3916ddb85782`;
- box-supervised training result canonical SHA-256:
  `6f783fd4ba5c4da9b29604ea152b4a55c3494e3700dbf748d86a21be62a3870e`;
- box-supervised checkpoint SHA-256:
  `c4cbd7de800d0bfddb88147c36e4e44bb1fe8109eb2156067414f3378cdd9b5a`;
- statistics-matched control source SHA-256:
  `5ca744c7662332496f1c309aac51fc6c886754d0ed4fd1bd23a654c0e5db9444`;
- CheXlocalize Phase-H rows SHA-256:
  `66dfa63556ab53657cef67341f43626ffadb234ef23cf0ebf8526b98af7970ed`;
- model family and scale: Qwen3.5-2B only;
- test opened: false;
- server or Slurm execution: none.

Generated rows, checkpoints, caches, and patient-bearing local artifacts stay
under ignored `local_runs/` paths and are not publishable repository content.

## Final validation

- ARISE contract tests: `23/23` pass;
- frozen BiVES/audit regression tests: `174/174` pass;
- BiVES CPU smoke: pass;
- localization-causality synthetic smoke: pass with `test_opened=false`;
- ARISE/script compilation: pass;
- result, row, and case-study lock assertions: pass;
- Markdown result entrypoint targets: present;
- `git diff --check`: pass (line-ending warnings only);
- active ARISE experiment processes after closure: `0`.

## Scientific boundary

The experiments support three bounded conclusions:

1. Full visual re-encoding plus explicit spatial supervision is materially
   better than the historical dense/feature-closure route.
2. Control construction is a causal-estimation factor: result-blind image-
   statistics matching changes both magnitude and confidence intervals.
3. The current two-finding candidate still lacks operator-robust pleural-
   effusion evidence and minimum three-finding coverage, so it cannot support
   the proposed ARISE selector or a general method claim.

The next scientifically admissible step is not another repair on this exposed
validation split. It is a new, independently frozen development surface or a
new proposal that changes the method claim and data role before any further
model score is created.
