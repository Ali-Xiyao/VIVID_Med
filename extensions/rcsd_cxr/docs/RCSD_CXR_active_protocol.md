# RCSD-CXR active execution protocol

**Status:** superseded by terminal G3 NO-GO  
**Date:** 2026-07-22  
**Source proposal:** `provenance/RCSD_CXR_full_proposal_20260722.original.md`

The terminal scientific authority is
`docs/RCSD_CXR_terminal_gate_result_20260723.md`. The protocol below is
retained to show the prospectively frozen gate order; it no longer authorizes
any experiment.

The binding G2 amendment is
`docs/RCSD_CXR_protocol_amendment_20260723_G2.md`. It drops posterior fusion,
freezes CheXbert as the best single source, and permits only the bounded
field-anchor learnability question described there.

## One claim

Estimate uncertainty-aware structured report supervision from multiple
imperfect sources, then distill it through a field-anchored 4x2 query interface
into a visual encoder that is deployed without the text teacher.

This project does not claim that report uncertainty is visual uncertainty, or
that query attention is causal evidence.

## Method objects

1. **Original controlled baseline:** UMS plus SPD with four groups and two
   tokens per group.
2. **Label contribution:** a calibrated posterior over present, absent, and
   uncertain report states. Unmentioned fields remain missing and are masked.
3. **Architecture contribution:** four named query groups: observation,
   assertion, anatomy, and global context, with the same total token budget as
   the baseline.
4. **Deployment object:** the visual encoder only.

## Pre-training gates

Training remains locked until all items below have machine-readable evidence:

- Dataset identity, version, license/DUA, patient IDs, split roles, and parent
  lineage are recorded.
- CheXpert/CheXpert-Plus/CheXlocalize and
  MIMIC/MS-CXR/Chest-ImaGenome overlaps are explicitly resolved.
- CheXlocalize test is absent from all paper-one manifests.
- Historical A+UMS and A+UMS+SPD checkpoint metadata confirm SPD 4x2 and a
  single checkpoint-selection rule.
- The report-label gold set is independent of image downstream tests.
- A posterior-quality gate is passed before any visual-model comparison.

## Stage order and stop rules

1. **G0 data identity:** zero forbidden patient/image overlap; all paths and
   hashes traceable. Failure stops all training.
2. **G1 baseline reconstruction:** unit tests, 256-row overfit, and a three-seed
   historical-direction audit. If SPD is not stable, it becomes only a
   comparator.
3. **G2 posterior validity:** formal NO-GO. Fusion failed NLL relative to
   CheXbert. CheXbert is frozen as the single source and the fusion claim is
   dropped.
4. **G3 20k pilot:** terminal NO-GO. Under equal data, parameters, teacher,
   backbone, seed, and compute, field anchoring improved NLL by only 0.046%
   and macro-F1 by only 0.0648 pp, below the frozen 3% and 0.5 pp thresholds.
5. **G4 full MIMIC single seed:** require expert-labelled and at least one
   external positive signal before multi-seed expansion.
6. **G5 three seeds:** require directionally consistent improvement and
   external non-inferiority. The simplest surviving model is final.
7. **G6 multi-institution scale:** MIMIC plus CheXpert-Plus is optional and can
   only test scaling after the controlled MIMIC result survives.

No failed gate may be rescued by opening a test set, adding a new module,
scaling the teacher, or switching to CT/pathology data.

G4-G6 are cancelled because G3 failed. CheXlocalize test remains sealed.

## Dataset roles

- Track A train/development: MIMIC-CXR-JPG/report family, patient-disjoint.
- Optional Track B train: CheXpert-Plus joined to non-reserved CheXpert images.
- External classification: NIH; VinDr only as previously inspected
  retrospective replication; PadChest only after acquisition and a frozen
  ontology mapping.
- Structured supervision: public labelers, RadGraph/Chest ImaGenome where
  lineage permits; these are not independent external cohorts.
- Excluded from paper one: CheXlocalize test, CT datasets, pathology datasets.

## Primary evaluation

- Primary endpoint: macro AUROC under a frozen label mapping.
- Key secondary: macro AUPRC, NLL, ECE, Brier score, low-data curves, and
  patient-level paired bootstrap confidence intervals.
- Every reported metric for a model comes from one preselected checkpoint.
- Per-finding results use false-discovery-rate control; subgroup results are
  descriptive, not causal fairness claims.

## Required main tables

1. Dataset/lineage/test-exposure ledger.
2. Supervision-source and posterior quality.
3. Controlled MIMIC-only main results.
4. External generalization with label-count denominators.
5. Low-data results.
6. Single-factor method ablation.
7. Optional multi-institution scaling decomposition.
8. Calibration and subgroup performance.

The full proposal contains detailed candidate thresholds. Those thresholds are
project-management gates, not journal requirements, and may be changed only
before the corresponding stage begins and with a dated protocol amendment.
