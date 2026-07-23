# RCSD component-attribution terminal diagnostic

## Goal

Determine which, if any, single RCSD component deserves one bounded
prospective evaluation. This is an attribution audit, not a rescue of the
full RCSD combination.

## Evidence inventory

| Arm | Intended definition | Existing evidence | Status |
| --- | --- | --- | --- |
| D0 | original VIVID hard structured labels + original SPD 4x2 + original objective | historical checkpoint/config audit exists; the current 20k comparator instead uses frozen CheXbert targets and Qwen field prototypes | not equivalent; exact D0 missing |
| D1 | agreement/confidence weighting + original SPD; no posterior or field anchor | source disagreement counts and label-gold confidence statistics exist; no visual arm was trained | untested |
| D2 | calibrated posterior + original SPD; no field anchor | label-level G2 exists and failed NLL against CheXbert | stopped before visual training |
| D3 | hard single-source targets + field-anchor 4x2; no posterior weighting | paired CheXbert 20k visual pilot exists | failed current G3 |
| D4 | posterior + weighting + anchor + decorrelation | no valid full combination after G2 | prohibited |

The current unanchored SPD comparator is **D0-like**, not an exact
reconstruction of original VIVID D0. It uses CheXbert report-state targets and
the same Qwen field-prototype semantic loss as the field-anchor arm. It is
valid for the frozen G3 pair, but it must not be relabelled as the historical
VIVID objective.

## What is already decided

- D2 posterior fusion is NO-GO at the independent report-gold gate.
- The tested D3 field anchor is NO-GO at the structured-state 20k gate.
- D4 is locked because its prerequisites did not survive independently.
- The G2/G3 thresholds, folds, source set, mappings, dataset, seed, teacher,
  optimizer, and checkpoint rules cannot be changed.

## Missing evidence

### D0

Missing:

- a verified runnable reconstruction of the original VIVID objective under
  the common 20k surface;
- a proof that its label semantics match the proposed D1 comparator;
- expert-development AUROC/AUPRC under one predeclared downstream protocol.

The historical checkpoint audit alone is insufficient to claim that D0 has
already been run in the new common protocol.

### D1

Missing:

- a frozen mathematical definition of agreement/confidence weighting;
- coverage and effective-weight distribution by finding/prevalence tier;
- one equal-budget visual comparison against exact D0;
- expert-development AUROC/AUPRC and reliability-quartile analysis.

### D2

No additional visual run is justified. It failed the prerequisite label-layer
NLL gate.

### D3

The existing pilot supplies NLL, macro-F1, ECE, stability, and per-finding
macro-F1. It does not supply expert-development AUROC/AUPRC. Because the
prospective G3 gate already failed, missing downstream endpoints may be
reported as unavailable but cannot be used to reopen or rescue D3.

## Required diagnostic tables

### Table A: label layer

Already available:

- best single-source and calibrated-posterior F1/NLL/ECE;
- reliability AUROC;
- high-low confidence accuracy gap.

Missing:

- a predeclared naive-majority baseline;
- D1 coverage and reliability stratification.

### Table B: visual pilot

Available for the D0-like comparator and D3:

- validation structured NLL;
- macro-F1;
- ECE;
- train stability.

Unavailable:

- expert-development macro AUROC;
- macro AUPRC.

D1 has not been run. D2 is blocked before visual training.

### Table C: per finding

`tables/rcsd_g3_per_finding_macro_f1.csv` records all currently valid
per-finding comparisons. These are macro-F1 values, not AUROC/AUPRC.

### Table D: reliability quartiles

Label-gold confidence statistics exist for G2, but a visual
reliability-quartile table has not been produced. It is required before D1 can
be promoted.

## Only permitted next decision

No experiment is authorized by this document.

After review, there are two valid choices:

1. close RCSD method development and return to a strict VIVID extension and
   multi-institution validation paper; or
2. authorize exactly one new diagnostic wave containing exact D0 and D1,
   with one common 20k dataset/seed/budget/checkpoint rule and no external
   test access.

If D1 is authorized, it must be fixed before launch as:

- original SPD 4x2;
- missing masked;
- nonmissing source agreement used only as a scalar weight;
- no inferred replacement label;
- no posterior fusion;
- no field anchor/anatomy/decorrelation;
- no new module, teacher scaling, full-data scaling, or threshold change.

The D1 promotion gate must be reviewed before execution. A failure closes the
RCSD additions and leaves only the strict VIVID extension route.
