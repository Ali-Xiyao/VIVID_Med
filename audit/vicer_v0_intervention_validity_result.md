# VICER-CXR V0 Intervention-Validity Result

## Decision

`TERMINAL_FAIL_STOP_BEFORE_V1`

VICER-CXR V0 completed its frozen local Qwen3.5-2B development matrix, but no
operator family passed all four findings. V1 coverage-redundancy and V2
coalition-selector work therefore remain locked. This is a method-development
negative result, not a clinical claim and not a test-set result.

## Frozen execution identity

- data: VinDr-CXR train only, image-level development;
- records: 280 globally role-disjoint images;
- evaluation: 32 new positive expert-box images, 8 per finding;
- ARISE exclusions: all 1,446 prior VinDr identities;
- model: local frozen Qwen3.5-2B visual features;
- operators: masked Gaussian blur, local-ring mean, and low-frequency
  replacement, each at four frozen strengths;
- row matrix: 384 complete dose-response rows;
- CheXlocalize test and VinDr test: unopened;
- server/Slurm: unused.

The first score-free geometry attempt exposed one central cardiomegaly target
for which no disjoint same-band exact-shape translation exists. Before any
model or score was opened, source commit `1359d8e` and opening v2 froze a
deterministic score-blind fallback. The completed geometry contains 30
same-band exact-shape translations and 2 exact-area, disjoint, connected,
original-statistics-matched fallback controls. All 32 mask hashes and geometry
contracts pass.

## Independent head gate

All local critics and global verifiers passed the frozen calibration AUROC
minimum of 0.60 on identities disjoint from their training data and from V0
evaluation.

| Head | Pneumothorax | Consolidation | Pleural effusion | Cardiomegaly | Minimum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Local critic | 1.0000 | 1.0000 | 1.0000 | 0.9200 | 0.9200 |
| Global verifier | 1.0000 | 1.0000 | 0.7778 | 0.7500 | 0.7500 |

The V0 failure is therefore downstream of the head calibration gate.

## Frozen V0 result

| Operator family | Finding | Pass | Removal rho | Strongest preservation | Valid fraction | Mean valid target-control gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Local-ring mean | Pneumothorax | no | 0.20 | 0.9785 | 0.1875 | 0.1680 |
| Local-ring mean | Consolidation | no | 1.00 | 0.9867 | 0.4688 | 0.0534 |
| Local-ring mean | Pleural effusion | yes | 1.00 | 0.9943 | 0.7812 | 0.2578 |
| Local-ring mean | Cardiomegaly | no | 1.00 | 0.9832 | 0.3438 | 1.3788 |
| Low-frequency replacement | Pneumothorax | no | 0.40 | 0.9782 | 0.2500 | 0.8995 |
| Low-frequency replacement | Consolidation | no | 1.00 | 0.9873 | 0.4688 | 0.1897 |
| Low-frequency replacement | Pleural effusion | yes | 1.00 | 0.9928 | 0.8125 | 0.0765 |
| Low-frequency replacement | Cardiomegaly | no | 0.80 | 0.9840 | 0.2812 | 1.7216 |
| Masked Gaussian blur | Pneumothorax | no | 0.80 | 0.9789 | 0.1250 | 0.7009 |
| Masked Gaussian blur | Consolidation | yes | 1.00 | 0.9875 | 0.5000 | 0.3995 |
| Masked Gaussian blur | Pleural effusion | yes | 1.00 | 0.9943 | 0.6250 | 0.1936 |
| Masked Gaussian blur | Cardiomegaly | no | 1.00 | 0.9853 | 0.4062 | 1.1764 |

Only 4 of 12 finding-family cells pass. Pleural effusion passes all three
families and consolidation passes blur, but every family fails pneumothorax
and cardiomegaly. Consequently there are zero surviving complete operator
families and `v0_pass=false`.

## Automatic case study

The aggregate-only case study reads the frozen 384-row file and does not load
the model. It attributes invalid rows as follows:

- removal below the frozen minimum: 185 rows;
- collateral preservation below the frozen minimum: 43 rows;
- realism below the frozen minimum: 15 rows.

All 12 cells have a positive mean target-control gap when restricted to rows
that pass intervention validity. No cell fails because its valid-row gap is
nonpositive. The dominant failure is therefore insufficient and inconsistent
finding removal according to the independent local critic, with an additional
strongest-dose preservation failure for pneumothorax. This is consistent with
single-box coverage/redundancy limitations, but V0 does not authorize testing
that explanation in V1 after a failed validity gate.

## Numeric-boundary repair

The first summary used a raw floating comparison. SciPy represented a
mathematically exact Spearman value of 0.8 as `0.7999999999999999`, producing
two spurious monotonicity failure labels. Commit `631c9c8` introduced a fixed
`1e-12` comparison tolerance, and the separate summary-repair opening
authorized an offline recomputation from the identical frozen rows. No score,
row, threshold, or final verdict changed. The original summary is preserved
locally as `v0_result_pre_float_tolerance.json`.

## Artifact locks

| Artifact | SHA-256 |
| --- | --- |
| Manifest | `3a531b813ec2c0f377ab664051fd740c2cd61b07aa2a88c0003c8e159c141b7a` |
| Data lock canonical | `b4b54949faffa7419cfdca07fd7c9125d9e257078d546fbe2c33d9566dc2ea09` |
| Geometry lock canonical | `28102abbfc9e47993544ea16f77f866de4bdccad88b457e041d152aa256ae1ce` |
| Cache lock canonical | `f058c2b9ff79f756f5f58337b0ead391afd370d80111d5aeff24f26cce5a2fb4` |
| Head lock canonical | `058c82352aa3de21497a8977c4bab83580a7c1324fb0c015671c5e9792c5ffff` |
| V0 rows file | `fba536afab1fd993fe2537f694484c7a2b7d70cedafe45b8bf411d3386dbe98d` |
| Corrected V0 result canonical | `3c9ceb27c66abfe28c2a269e0566a2f2be08ea3a43dea8c18dc7ddaf5a330bb1` |
| Failure case-study canonical | `48b9c78e415e4c6136d893b60a3e2e66cd6a3c363d87f74a730b40e82b324a93` |

## Scientific boundary

V0 supports one narrow conclusion: under the frozen validity definition,
none of the three tested intervention families is independently valid across
all four development findings. It does not show that causal evidence is
impossible, that pleural-effusion results generalize clinically, or that the
test set should be opened. It does not authorize threshold relaxation,
operator selection from these results, V1/V2, U/I, Qwen3.5-4B/9B, or a server
run.
