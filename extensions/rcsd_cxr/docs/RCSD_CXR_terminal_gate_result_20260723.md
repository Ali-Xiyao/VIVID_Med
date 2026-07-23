# RCSD-CXR terminal gate result

**Date:** 2026-07-23  
**Status:** terminal NO-GO for RCSD-CXR as a new method  
**Active authority:** this document

## Decision

RCSD-CXR stopped at the first failed survival gates, as required by the
reviewed protocol.

1. The multi-source posterior failed G2 because its NLL was 6.98% worse than
   the best single source, CheXbert. The fusion contribution was dropped.
2. The surviving equal-budget field-anchor comparison failed G3. Its NLL
   improved by only 0.046%, below the required 3%, and macro-F1 improved by
   only 0.0648 percentage points, below the required 0.5 percentage points.

No full-MIMIC run, external evaluation, multi-seed expansion, multi-institution
training, or Qwen3.5 teacher-size sweep is authorized. There is no active
successor method experiment.

## G2: report-supervision gate

The independent report-gold surface contained 5,730 mapped study-finding
labels from 1,455 LUNGUAGE studies. Five patient-held-out outer folds and
disjoint calibration folds were frozen before scoring.

| System | Macro-F1 | NLL | ECE |
| --- | ---: | ---: | ---: |
| CheXbert, best single source | 0.800504 | 0.371156 | 0.005907 |
| CheXpert + NegBio + CheXbert fusion | 0.813270 | 0.397067 | 0.015316 |

The fusion improved macro-F1 by 1.2766 percentage points but degraded NLL by
6.981%. Because G2 required both discrimination and likelihood improvement,
the result is a formal NO-GO for multi-source posterior fusion.

Binding consequence:

- CheXbert became the frozen single source for one bounded visual
  learnability test.
- Missing labels remained masked and were never relabeled absent.
- Report uncertainty was not interpreted as visual uncertainty.
- No RadGraph/fourth-source rescue or teacher-size selection was allowed.

## G1 diagnostic: trainability

Both equal-budget visual variants passed the 256-row diagnostic:

| Variant | Best step | Observed-target accuracy | Loss reduction |
| --- | ---: | ---: | ---: |
| Unanchored SPD 4x2 | 200 | 98.554% | 95.662% |
| Field-anchored 4x2 | 100 | 98.129% | 95.219% |

Both models used:

- ViT-B backbone trainable parameters: 85,798,656;
- projector trainable parameters: 13,788,672;
- state-head trainable parameters: 77,860.

This rules out a basic inability to fit the simplified objective. It does not
establish downstream value.

## G3: paired 20k development pilot

The deterministic surface began with 20,000 training studies and all 1,733
canonical validation studies. Samples with all 12 CheXbert fields missing
were excluded at dataset initialization, without relabeling: 467 train and 54
validation samples. The final surface contained 19,533 train and 1,679
validation studies with 7,676 observed validation targets.

Both variants used the same:

- seed 0;
- manifest;
- ImageNet ViT-B initialization;
- Qwen3.5-2B frozen field prototypes;
- effective batch size 64;
- 1,000-step budget;
- checkpoint-selection rule.

| Variant | Best step | NLL | Macro-F1 | Accuracy | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unanchored SPD 4x2 | 1,000 | 0.478478 | 0.701950 | 0.819177 | 0.010646 |
| Field-anchored 4x2 | 1,000 | 0.478258 | 0.702598 | 0.819046 | 0.007876 |

Frozen G3 comparison:

| Check | Observed | Required | Result |
| --- | ---: | ---: | --- |
| Relative NLL change | -0.046% | at most -3.0% | fail |
| Macro-F1 gain | +0.0648 pp | at least +0.5 pp | fail |
| ECE change | -0.002769 | at most +0.01 | pass |
| Findings below -2 pp | 0 | at most 2 | pass |
| Equal parameter counts | yes | yes | pass |
| Same manifest/backbone/teacher | yes | yes | pass |

The tiny numerical differences are not sufficient evidence for the
field-anchor contribution. The route is therefore terminal NO-GO.

## Evidence identity

- G3 gate artifact:
  `local_runs/gate3_simplified_20260723_r2/pilot_gate.json`
- G3 gate artifact SHA-256 after schema correction:
  `159768072505e163ccc3e224567452714f22b40cdac28f65be8b00ab77535c41`
- SPD summary SHA-256:
  `91bb8eb0e895f8a949aa1a322498fd7589973c2df6376b0173293ddb5023eb08`
- Field-anchor summary SHA-256:
  `0a1464f8efafe6df1c18b3d70b120c1d54175ba03feeb65af2214c0e8e357397`
- ViT-B initialization SHA-256:
  `32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b`
- Qwen3.5-2B prototype SHA-256:
  `90185158299230d2d970f183d9255bb68c6afab82415ba72bd8e9a888b01b190`

The first G3 launch stopped before training because some studies had no
observed target. That attempt is preserved. The second launch restarted both
variants from zero after adding fail-closed all-missing exclusion.

## Stop actions

- G4 full-MIMIC seed 0: cancelled.
- G5 three-seed/downstream matrix: cancelled.
- External tests: unopened by this route.
- CheXlocalize test: remains sealed.
- Track-B MIMIC + CheXpert-Plus scaling: cancelled.
- Qwen3.5 0.8B/4B/9B sensitivity: cancelled.
- Further source, loss, query, teacher, or threshold rescue: prohibited.

The negative result is specific to the proposed RCSD additions. It does not
invalidate the earlier VIVID-Med UMS/SPD representation-learning evidence,
which remains a separate historical line requiring its own clean
paper-code-checkpoint audit.

## Final acceptance

- Local unit tests: 45/45 passed.
- Remote unit tests in `vivid_med310`: 45/45 passed.
- The server source tree was synchronized through the verified raw-PTY SFTP
  path using `provenance/server_upload_manifest_20260723.csv`.
- No RCSD screen or RCSD training process remained after closeout.
- Slurm allocation 3066 was not cancelled; it is retained shared
  infrastructure outside this terminal method decision.
