# BiVES-CXR P0 Data Readiness

This document fixes the active P0 data boundary. It does not authorize model
loading or training.

## Source decision

| Role | Dataset | Allowed use in this phase | Not allowed |
| --- | --- | --- | --- |
| In-domain candidate source | MIMIC-CXR-JPG image/report pairs | P0-1 parser candidates, P0-2 blind review, patient-disjoint four-state manifest construction | Treating unreviewed report text or report omission as a BiVES state label |
| Future locked test | BiVES-CXR Audit Set | Separate expert-audited patient-level test after training and calibration are locked | Training, threshold selection, or cache vocabulary expansion |
| External P0-5 candidate | VinDr-CXR | Label, bounding-box, image-integrity, licence, and mapping audit | Mixing VinDr rows into MIMIC P0 train/validation/calibration data |
| Secondary external | CheXpert | Later external statement-verification or linear-probe analysis | Replacing the report-derived four-state P0 source |

The local raw roots are outside the repository and must remain outside Git:

```text
H:\Xiyao_Wang\000_Public Dataset\mimic-cxr\mimic-cxr\mimic-cxr-images
H:\Xiyao_Wang\000_Public Dataset\mimic-cxr\mimic-cxr\mimic-cxr-reports
H:\Xiyao_Wang\000_Public Dataset\vindr-cxr-an-open-dataset-of-chest-x-rays-with-radiologists-annotations-1.0.0
```

## P0 intake chain

1. Build a MIMIC image/report **intake index**. This index contains paths and
   source identifiers only; it is not a label manifest.
2. Run one frozen parser version over indexed reports. Store parser version,
   rule/prompt/model identity, input report hash, extracted finding, polarity,
   uncertainty, laterality, anatomy, severity, view requirement, and quality
   flags in an ignored candidate table.
3. Sample parser candidates for P0-1 explicit-positive/negative audit and
   P0-2 blinded U/I review. Do not promote rows with unresolved disagreement.
4. Freeze the approved 4–6 initial findings. The proposal candidates are
   pleural effusion, pneumothorax, cardiomegaly, pulmonary edema, atelectasis,
   focal opacity/consolidation, and support-device position; a finding enters
   only after its positive, negative, U/I, and external-map evidence passes.
5. Split by patient **before** quartet construction. Construct one exact
   S/C/U/I quartet per `group_id` with four distinct patients, studies, and
   image hashes in one matching stratum.
6. Generate four immutable manifests (`p0_train`, `p0_val`,
   `p0_calibration`, `p0_test`), run `audit_bives_manifest.py`, and create the
   joint `p0_dataset_lock.json`.
7. Build `qwen35_canonical.pt` from the audited train/validation/calibration
   ontology, lock its hashes into an ignored P0 config, create the clean local
   source snapshot, and only then launch Qwen3.5-2B P0 on this workstation.

## Intake index command

Write this output under ignored `local_runs/`; it does not read report text or
create labels.

```powershell
python scripts/index_mimic_bives_p0_candidates.py `
  --images-root 'H:\Xiyao_Wang\000_Public Dataset\mimic-cxr\mimic-cxr\mimic-cxr-images' `
  --reports-root 'H:\Xiyao_Wang\000_Public Dataset\mimic-cxr\mimic-cxr\mimic-cxr-reports' `
  --output local_runs/bives_cxr/p0_intake/mimic_candidates_shard0.jsonl `
  --summary local_runs/bives_cxr/p0_intake/mimic_candidates_shard0_summary.json `
  --max-studies 1000
```

Each candidate row has `candidate_id`, source/patient/study/image identifiers,
an image path, a report path, `candidate_status: unparsed`, and
`p0_role: parser_and_blind_review_input`. It intentionally contains neither
report text nor a state label.

## Frozen parser candidates and blinded review

`scripts/prepare_bives_p0_report_review.py` reads an intake index, applies a
versioned conservative six-finding rule set, writes an ignored parser-candidate
JSONL, and creates a blinded CSV packet. The parser output includes only a
matched cue phrase and report SHA256; it does not export report text. The
reviewer packet excludes parser state, cue, and report hash.

```powershell
python scripts/prepare_bives_p0_report_review.py `
  --candidates local_runs/bives_cxr/p0_intake/mimic_candidates_shard0.jsonl `
  --parsed-output local_runs/bives_cxr/p0_intake/mimic_parser_candidates_shard0.jsonl `
  --review-packet local_runs/bives_cxr/p0_intake/mimic_blinded_review_shard0.csv `
  --summary local_runs/bives_cxr/p0_intake/mimic_parser_review_summary_shard0.json
```

Two qualified reviewers must independently fill the state fields, then an
adjudicator must complete the final state. Verify this before any manifest
construction:

```powershell
python scripts/validate_bives_p0_review_packet.py `
  --review-packet local_runs/bives_cxr/p0_intake/mimic_blinded_review_shard0.csv `
  --output local_runs/bives_cxr/p0_intake/mimic_blinded_review_validation.json
```

The validator fails on blank reviewer/adjudicator fields, invalid states, or
the same reviewer recorded twice. It deliberately does not infer missing
expert labels.

## Formal stop rules

- Report omission is never a contradict label.
- Missing, corrupt, or unreadable images are data errors, never insufficient.
- Natural and synthetic insufficient examples are both required in every
  formal split.
- No P0 manifest, cache, GPU job, calibration, or locked-test evaluation may
  be created from unreviewed parser candidates.
