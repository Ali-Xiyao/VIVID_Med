# Findings

## 2026-07-24 intake

- Terminal Stage A established that unchanged A3 beat A2, while A2 failed
  against A0 and A1.
- The new replication is not a repair or rerun of that gate. It asks whether
  A3 can beat both A0 and A2 on a genuinely fresh development surface.
- Dataset presence is not authorization. Provenance, patient/image overlap,
  label ontology, split role, hashes, and prior exposure must be audited
  before scoring.
- Raw datasets remain in their existing dataset roots; Git contains only code,
  manifests without patient data, aggregate audits, and protocols.

## 2026-07-24 data eligibility audit

- The server has the ordinary CheXpert-v1.0-small image tree and label tables:
  `train.csv` has 223,414 rows and `valid.csv` has 234 rows. The image tree
  occupies about 11.45 GB.
- The server also has CheXpert-Plus metadata and reports: 223,462 image rows,
  64,725 patients, and 187,711 studies. The CheXpert-Plus directory contains
  no duplicate image payload; its image paths align to the ordinary CheXpert
  tree.
- CheXpert-Plus provenance is the previously audited Redivis
  `aimi.chexpert_plus:5yyj:v1_0` acquisition. Presence and provenance do not by
  themselves authorize a new score.
- The old probe used 191,027 frontal CheXpert-train images and the 234-image
  official CheXpert validation surface as expert development. That official
  validation surface is already exposed and cannot be reused as fresh
  evidence.
- CheXlocalize validation is the same 234-image official CheXpert validation
  surface (200 patients). CheXlocalize test remains unopened and absent from
  this replication.
- A legitimate fresh development surface can therefore only be carved
  deterministically from `CheXpert-v1.0-small/train.csv`, at patient level,
  before any new model score is produced. New probe heads must be retrained
  while excluding that holdout.
- To retain direct comparability with the terminal Stage-A probe, the new
  manifests will use the original CheXpert labels and its existing missing /
  uncertain masking policy for Atelectasis, Cardiomegaly, Consolidation,
  Edema, and Pleural Effusion. CheXbert-derived CheXpert-Plus labels will not
  replace the evaluation targets.
- This is a fresh development replication, not an untouched external test:
  CheXpert train was previously used to fit historical probe heads. The new
  model identities and thresholds must be locked before scoring, and the new
  heads must exclude the fresh-development patients.

## 2026-07-24 frozen split result

- The score-free server build reproduced the locked counts exactly:
  172,478 / 9,144 / 9,405 images and 58,230 / 3,186 / 3,118 patients.
- Patient overlap, image-path overlap, and overlap with official CheXpert
  validation were all exactly zero.
- All 191,027 referenced frontal images were present.
- Every primary finding had both positive and negative observed labels in the
  fresh-development split. Atelectasis has only 55 observed negatives, so its
  interval may be wider; the threshold and finding floor remain unchanged.
- The independent prelaunch readiness audit passed all 13 checks, including
  manifest hashes, method identity, Qwen3.5-2B and backbone presence, and the
  absence of CheXlocalize/VinDr references.
