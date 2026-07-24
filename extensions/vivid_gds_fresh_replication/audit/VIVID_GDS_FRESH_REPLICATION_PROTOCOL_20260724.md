# VIVID-GDS Fresh-Development Replication Protocol

## 1. Question

This is one separately locked replication of the unchanged VIVID-GDS bridge.
It tests:

> Does the training-only schema bridge (A3) improve the deployed ViT
> representation beyond both direct schema supervision (A0) and structured
> UMS generation (A2)?

The terminal Stage-A experiment remains a scientific NO-GO. This replication
does not revise that verdict and does not reuse its expert-development scores
for selection.

## 2. Fixed method identities

All arms use the same ViT-B/16 initialization, MIMIC-CXR hard-UMS manifest,
augmentations, optimizer, 3,000-step pilot budget, effective batch size 32,
and Qwen3.5-2B teacher where applicable.

| Arm | Training objective | Checkpoint rule |
|---|---|---|
| A0 direct | masked 12-field schema cross-entropy | minimum MIMIC validation schema NLL |
| A2 UMS | prefix4 structured token generation | minimum MIMIC validation token NLL |
| A3 GDS | A2 plus unchanged schema bridge, `lambda_schema=0.5` with 500-step ramp | minimum MIMIC validation token NLL |

Seeds are exactly `0, 1, 2`. A fresh checkpoint is trained for every
arm/seed pair. No Stage-A checkpoint or probe head is reused.

## 3. Fresh development construction

Source labels and images come only from
`CheXpert-v1.0-small/train.csv`. Only `Frontal` rows are eligible. Patient ID
is parsed from the official path. The deterministic bucket is:

```text
sha256("vivid-gds-fresh-v1:" + patient_id)[:8] mod 20
```

- bucket 0: fresh development;
- bucket 1: probe internal validation;
- buckets 2-19: probe training.

The locked expected counts are:

| Split | Images | Patients |
|---|---:|---:|
| probe train | 172,478 | 58,230 |
| probe validation | 9,144 | 3,186 |
| fresh development | 9,405 | 3,118 |

The official 234-image CheXpert validation / CheXlocalize-validation surface
is excluded by source split and checked again by patient and path. The
CheXlocalize test and VinDr test are not referenced or opened.

Evaluation findings are Atelectasis, Cardiomegaly, Consolidation, Edema, and
Pleural Effusion. Only literal `0` and `1` labels are observed. Missing and
uncertain values remain masked, matching the historical probe policy.
CheXbert/CheXpert-Plus labels do not replace these targets.

This is a fresh development replication, not an untouched external test.
CheXpert train was historically used for other probe heads; every head in
this replication is retrained from zero and excludes all fresh-development
patients.

## 4. Probe protocol

The ViT is frozen. For each arm/seed checkpoint, train one five-label linear
head:

- AdamW, learning rate `1e-3`, weight decay `1e-4`;
- 50 epochs;
- matched arm/seed initialization;
- positive weights estimated only from probe training;
- select exactly one head by minimum probe-validation masked NLL;
- score fresh development once after selection.

All three arms within a seed use the same patient rows and head seed.

## 5. Frozen survival gate

All conditions are conjunctive:

1. mean(A3 - A0) macro AUROC across seeds is at least `+0.005`;
2. mean(A3 - A2) macro AUROC across seeds is at least `+0.005`;
3. both AUROC differences are positive in every seed;
4. the patient-cluster paired-bootstrap 95% lower bound for the mean
   A3-versus-A0 macro-AUROC difference is greater than zero;
5. mean macro-AUPRC differences for A3 versus A0 and A2 are each at least
   `-0.005`;
6. no five-finding mean AUROC difference for A3 versus either comparator is
   below `-0.020`.

Bootstrap uses 10,000 deterministic replicates, samples patients with
replacement, keeps every image for each sampled patient, pairs all arms and
seeds within each replicate, and reports percentile confidence intervals.

## 6. Decisions

- `REPLICATION_PASS`: all six conditions pass. Unlock only a document-first
  plan for full data, low-data, calibration, and one untouched external
  evaluation.
- `A3_BEATS_A2_NOT_A0`: stop the LLM-method claim; direct schema supervision
  remains sufficient.
- `TERMINAL_NO_GO`: otherwise close the method route and retain the evidence
  for an objective-to-representation mismatch audit.

There is no same-surface repair. A failed gate may receive a score-frozen
implementation audit, but no threshold, identity, loss, seed, or data repair.

## 7. Publication and privacy boundary

Git may contain code, this protocol, the lock, aggregate counts, hashes, and
the terminal verdict. Patient-level manifests, predictions, images,
checkpoints, and runtime logs stay on the SUES server.

