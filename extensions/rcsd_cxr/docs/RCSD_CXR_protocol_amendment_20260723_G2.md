# RCSD-CXR protocol amendment: G2 posterior NO-GO

**Date:** 2026-07-23  
**Authority affected:** `docs/RCSD_CXR_active_protocol.md`  
**Evidence:** `local_runs/gate2_20260723/lunguage_three_source_g2_gate.json`

## Frozen result

The report-gold surface contains 5,730 mapped study-finding labels from 1,455
LUNGUAGE studies. Patient-hash outer folds, fixed disjoint calibration folds,
the entity mapping, source outputs, and thresholds were frozen before the
formal three-source score.

CheXbert was the best single source:

- macro-F1: 0.80050;
- NLL: 0.37116;
- ECE: 0.00591.

CheXpert + NegBio + CheXbert fusion produced:

- macro-F1: 0.81327, a gain of 0.01277;
- NLL: 0.39707, a relative degradation of 6.98%;
- ECE: 0.01532.

The fusion passed the discrimination threshold but failed the required NLL
improvement. G2 is therefore a formal **NO-GO for posterior fusion**.

## Binding decision

1. Drop the multi-source posterior and reliability-fusion contribution.
2. Freeze CheXbert as the single structured report-label source for the
   bounded visual learnability experiment.
3. Preserve blank as missing and mask it; never convert it to absent.
4. Preserve uncertain as a report assertion state only.
5. Do not change the LUNGUAGE gold mapping, temperatures, folds, thresholds,
   or source set to rescue fusion.
6. Do not add RadGraph or another parser as a post-hoc G2 rescue.
7. Do not run Qwen3.5 teacher-size scaling under the failed fusion claim.

## Simplified surviving method question

The only surviving new-method question is:

> With the same frozen CheXbert report targets and the same 4-by-2 token
> budget, does explicit field anchoring improve the deployable visual encoder
> over unanchored SPD?

The next bounded stages are:

1. produce frozen CheXbert labels for a deterministic development subset;
2. pass 256-row overfit for unanchored SPD and field-anchored SPD;
3. run a 20k-study, one-seed paired pilot;
4. stop if the anchored variant does not improve the preregistered
   development endpoints.

No full-data, multi-seed, external-test, or teacher-size experiment is
unlocked by this amendment.

