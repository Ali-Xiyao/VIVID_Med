# RCSD P0 deidentified NO-GO verdict

**Base implementation commit:** `bc1105f880116e97e06da023110b7080debc28a4`
**Audit branch:** `codex/rcsd-no-go-audit`
**Decision:** NO-GO for full RCSD; bounded component attribution remains open
**Patient-level material in Git:** none

## Machine-readable verdict

```yaml
commit: bc1105f880116e97e06da023110b7080debc28a4
decision: FULL_RCSD_NO_GO_COMPONENT_AUDIT_OPEN
gates:
  G2:
    decision: NO_GO
    evidence_sha256: 73d3dd216a5820dfe6a10adbeaf2d604fc286246810c013aa5e5b03e23d52980
    gold_samples: 5730
    gold_studies: 1455
    baseline:
      name: chexbert
      macro_f1: 0.8005042079250303
      val_nll: 0.37115624626034804
      ece: 0.005906493244007121
    candidate:
      name: chexpert_negbio_chexbert_fusion
      macro_f1: 0.8132700930713
      val_nll: 0.39706729797400664
      ece: 0.015316305349233814
    delta:
      macro_f1_pp: 1.2765885146269751
      val_nll_percent: 6.981170861255892
      ece: 0.009409812105226693
    failed_conditions:
      - nll_relative_change_at_most_minus_5_percent
  G3:
    decision: NO_GO
    evidence_sha256: 159768072505e163ccc3e224567452714f22b40cdac28f65be8b00ab77535c41
    dataset_lock_sha256: 5e9e05552712a7c6298ff63731c4250c0bdc6d5e3d2a28e9e4476b7c7c242ae2
    canonical_g0_manifest_sha256: 00fde375c608017d5e5700f946a15f32097d44ceecec885ebae41dfc58578133
    config_sha256: 348a902f6963ea09fd7c57ccdb872a2523ed8dc0b4be2471abe6df82085c0b98
    training_seed: 0
    checkpoint_rule: strictly_lower_validation_structured_nll
    baseline:
      name: chexbert_unanchored_spd_4x2
      val_nll: 0.4784779897995328
      macro_f1: 0.7019501519688692
      macro_auroc: null
      macro_auprc: null
      ece: 0.010645743861305276
    candidate:
      name: chexbert_field_anchor_4x2
      val_nll: 0.4782576282984767
      macro_f1: 0.7025980120912
      macro_auroc: null
      macro_auprc: null
      ece: 0.007876329822829781
    delta:
      val_nll_percent: -0.046054678742579735
      macro_f1_pp: 0.06478601223307567
      macro_auroc_pp: null
      macro_auprc_pp: null
      ece: -0.002769414038475495
    failed_conditions:
      - validation_nll_reduction_below_3_percent
      - macro_f1_gain_below_0_5_percentage_points
test_sets_opened: []
```

`macro_auroc` and `macro_auprc` are `null` because this pilot evaluated
structured report-state prediction, not an expert-labelled downstream
classification probe. They must not be reconstructed from macro-F1 or treated
as implicitly passed.

## G2: posterior-fusion attribution

The formal five-fold report-gold comparison used patient-held-out outer folds,
disjoint calibration folds, 5,730 mapped labels, and 1,455 LUNGUAGE studies.

| Supervision | Macro-F1 | NLL | ECE | Reliability AUROC | High-low accuracy gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| CheXbert, best single source | 0.800504 | 0.371156 | 0.005906 | 0.800146 | 0.296089 |
| Three-source calibrated fusion | 0.813270 | 0.397067 | 0.015316 | 0.748079 | 0.261173 |

The candidate passed the frozen macro-F1 threshold but failed the frozen NLL
threshold: NLL was 6.981% worse rather than at least 5% better. This directly
rejects the multi-source posterior contribution. It does not reject the
historical VIVID/SPD line.

The historical G2 JSON uses top-level `pass: true` to mean artifact production
completed. Its scientific field is `gate.g2_pass: false`. This audit
normalizes those two concepts as `audit_completed` and `gate_pass` and does
not reinterpret the result.

## G3: field-anchor attribution

The paired pilot used the same dataset lock, seed, ViT-B initialization,
Qwen3.5-2B field prototypes, parameter count, token budget, effective batch
size, 1,000-step budget, and validation-NLL checkpoint rule.

| Variant | Val NLL | Macro-F1 | Accuracy | ECE | Best step |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unanchored SPD 4x2 | 0.478478 | 0.701950 | 0.819177 | 0.010646 | 1,000 |
| Field-anchored 4x2 | 0.478258 | 0.702598 | 0.819046 | 0.007876 | 1,000 |

The field anchor improved NLL by only 0.046% against the frozen 3% threshold
and macro-F1 by only 0.0648 percentage points against the 0.5-point threshold.
It therefore failed the prospective G3 survival gate. ECE improved and no
finding fell by more than two points, but those secondary checks cannot rescue
the two failed primary checks.

## Exact evidence identities

- G2 gate artifact:
  `73d3dd216a5820dfe6a10adbeaf2d604fc286246810c013aa5e5b03e23d52980`
- G3 gate artifact:
  `159768072505e163ccc3e224567452714f22b40cdac28f65be8b00ab77535c41`
- G3 SPD summary:
  `91bb8eb0e895f8a949aa1a322498fd7589973c2df6376b0173293ddb5023eb08`
- G3 field-anchor summary:
  `0a1464f8efafe6df1c18b3d70b120c1d54175ba03feeb65af2214c0e8e357397`
- 20k+validation dataset lock:
  `5e9e05552712a7c6298ff63731c4250c0bdc6d5e3d2a28e9e4476b7c7c242ae2`
- ImageNet ViT-B initialization:
  `32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b`
- Qwen3.5-2B field prototypes:
  `90185158299230d2d970f183d9255bb68c6afab82415ba72bd8e9a888b01b190`
- G3 launcher/config identity:
  `348a902f6963ea09fd7c57ccdb872a2523ed8dc0b4be2471abe6df82085c0b98`

## Binding decision

1. Full RCSD remains NO-GO.
2. Posterior fusion is rejected as a contribution under the frozen G2
   protocol.
3. Field anchoring is rejected under the frozen G3 structured-state pilot.
4. The overall VIVID journal extension is not declared failed because the
   exact original VIVID/SPD D0 reconstruction and D1 selective-reliability arm
   have not both been evaluated under the proposed component-audit protocol.
5. No new method is promoted. The only open question is whether one
   prospectively specified D1 arm is worth a separately reviewed run.
6. External tests, full-data scaling, multi-seed expansion, teacher scaling,
   and D4 remain locked.
