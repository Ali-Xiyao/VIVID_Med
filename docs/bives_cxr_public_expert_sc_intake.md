# BiVES-CXR Public Expert S/C Intake

## Status

VinDr-CXR v1.0.0 test consensus is locally intake-ready for a bounded expert
support/contradict evaluation surface. This is data preparation only. The
current BiVES weak-label four-state model is **not** authorized to run on it
because the state-only 48-row train-fit survival gate failed.

## Source and label semantics

The official VinDr-CXR release states that each of the 3,000 test images was
resolved by a consensus of five radiologists. Its binary image-label table
encodes positive findings as `1` and negative findings as `0`. The test CSV
therefore supports a same-finding expert S/C interpretation:

- `1` -> support for the fixed non-negated canonical statement;
- `0` -> contradict for that same statement;
- no U or I label is inferred;
- no report omission or parser output participates in this intake.

Primary source: [VinDr-CXR v1.0.0 on PhysioNet](https://physionet.org/content/vindr-cxr/1.0.0/),
DOI `10.13026/3akn-b287`.

## Local integrity evidence

`scripts/audit_vindr_cxr_integrity.py` passed on the extracted public-dataset
tree:

- official manifest entries: `18,006`;
- train/test DICOMs: `15,000 / 3,000`;
- missing files: `0`;
- SHA-256 checks/mismatches: `70 / 0`;
- DICOM decode checks/failures: `16 / 0`;
- archive extraction marker reports member-by-member CRC checking.

The audit output remains ignored under
`outputs/final_tables/vindr_cxr_integrity_audit.{json,md}`.

## Intake result

`scripts/prepare_bives_vindr_expert_sc.py` verifies the official annotation
CSV hashes, unique test image IDs, binary labels, image existence, and
label/bounding-box consistency. It writes ignored artifacts under
`local_runs/bives_cxr/vindr_expert_sc_intake/`.

| Canonical finding | Positive | Negative | Positive with box | Eligible S/C |
| --- | ---: | ---: | ---: | --- |
| pleural effusion | 111 | 2,889 | 111 | yes |
| consolidation | 96 | 2,904 | 96 | yes |
| pulmonary edema | 0 | 3,000 | 0 | no: degenerate |

The resulting intake has `6,000` rows over `3,000` images and exactly two
eligible findings. It is explicitly marked `formal_result=false` and
`four_state_claim=false`.

## Fail-closed boundaries

- The public release exposes image IDs but no patient identifier. Patient-level
  confidence intervals are not ready and must not be claimed from this intake.
- Full DICOM SHA-256 verification is optional and was not performed during this
  preparation pass. Rows carry the official manifest digest and an explicit
  `actual_image_sha256_verified` flag.
- Pulmonary edema is excluded rather than assigned a fabricated positive set.
- Bounding boxes are used only for positive localization/intervention intake;
  a zero image-level label is not converted into a localization target.
- CheXpert-small `valid.csv` is not treated as the special expert set without
  provenance verification. CheXlocalize is not present in the checked local
  data roots.
- This intake does not reopen Run B, 4B/9B scaling, parser U/I expansion,
  decoder changes, calibration, or locked-test work.
