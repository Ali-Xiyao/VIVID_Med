# VinDr-CXR Main-External Integration

## Storage Boundary

- Raw archive and extracted DICOM/annotation files: `H:\Xiyao_Wang\000_Public Dataset`.
- Project code, mappings, derived manifests, audits, logs, and result tables: `H:\Xiyao_Wang\021_260129VIVID`.
- Medical images are not copied into Git-tracked project paths.

## Source Package

- Dataset: VinDr-CXR 1.0.0.
- Archive: `H:\Xiyao_Wang\000_Public Dataset\vindr-cxr-an-open-dataset-of-chest-x-rays-with-radiologist-annotations-1.0.0.zip`.
- Archive inventory: 15,000 train DICOMs, 3,000 test DICOMs, bbox annotations, image-level labels, and the official `SHA256SUMS.txt`.
- Extracted root: `H:\Xiyao_Wang\000_Public Dataset\vindr-cxr-an-open-dataset-of-chest-x-rays-with-radiologists-annotations-1.0.0`.

## Fixed External Protocol

- Split: official VinDr-CXR test split only for the main external evaluation.
- Training table audit: three radiologist rows per train image; the derived train manifest uses a deterministic 2-of-3 majority vote.
- Test table audit: one official image-level label row per test image.
- Split leakage: train/test image-ID intersection is zero.
- Primary comparable labels: No Finding, Atelectasis, Cardiomegaly, Consolidation, Pleural Effusion, Pneumonia, and Pneumothorax.
- Edema remains in the full mapped manifest but is excluded from the primary macro-AUC because the official test table has zero positive Edema rows.
- Ambiguous hierarchy/synonym mappings such as Infiltration→Lung Opacity and Nodule/Mass→Lung Lesion are deliberately excluded from the primary protocol.

## Project Artifacts

- Label mapping: `../configs/qwen3vl_instruction/vsl_cxr/phase6_external/vindr_chexpert_label_mapping.json`.
- Resumable extractor: `../scripts/extract_vindr_cxr.py`.
- Manifest builder: `../scripts/prepare_vindr_cxr.py`.
- Integrity audit: `../scripts/audit_vindr_cxr_integrity.py`.
- Two-GPU evaluation driver: `../scripts/run_vindr_external_suite.py`.
- Derived manifests: `../data/dataset/processed/vindr_cxr_external_{train_majority,test}_ums.jsonl`.
- Data-quality audit: `../outputs/final_tables/vindr_cxr_data_quality_audit.{json,md}`.
- Integrity audit: `../outputs/final_tables/vindr_cxr_integrity_audit.{json,md}`.
- External results: `../outputs/final_tables/vsl_cxr_external_results.{csv,md}`.

## Current State

As of 2026-07-16, extraction/CRC validation is running in the background. The mapping and deterministic manifests are built, DICOM decoding is supported by the shared LP loader, all five retained probe packages are available, and the external table lists Raw, SAMEQ, VSL-Core, VSL-CEQ-backbone, and VSL-Full as `pending_main_external`. No VinDr performance claim is allowed until the full extraction, integrity audit, and five 3,000-image inference rows complete.
