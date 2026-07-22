# Local `H:\xiyao` asset inventory

**Observed:** 2026-07-22
**Scope:** read-only local inventory for documentation. This is not a data
lock, an experiment opening, or an authorization to copy, upload, score, or
publish any asset.

`H:\xiyao` (also addressable as `H:\Xiyao` on this Windows volume) contains a
separate local asset root with `dataset/` and `model/` directories. It is not
part of this repository's tracked source tree.

## Dataset directories observed

| Local path | Observed contents | Audit relevance | Current status |
| --- | --- | --- | --- |
| `H:\xiyao\dataset\CheXpert-Plus` | `chexpert_plus_reports_by_study.parquet`, `df_chexpert_plus_240401.parquet`, and a download manifest; a `chexbert_labels/` subdirectory | Potential report/label metadata source. It does not by itself establish expert localization or matched-control eligibility. | Inventory only; no active use. |
| `H:\xiyao\dataset\IU-Xray-OpenI` | `raw/` plus `xml_integrity_check.json` | Potential report--image resource for future protocol development, subject to provenance, image identity, and license review. | Inventory only; no active use. |
| `H:\xiyao\dataset\MIMIC-CXR` | `mimic-cxr/`, `mimic-cxr_less/`, `mimic_cxr_other/`, and `source_manifests/` (including `SHA256SUMS.txt`) | Potential local source for future report/image work. It is not a substitute for the currently defined CheXlocalize expert-region endpoint. | Inventory only; no active use. |
| `H:\xiyao\dataset\027_diffsionretrieval` | `external/`, `index/`, and `processed/` | Unclassified auxiliary corpus/index material. Provenance and task relevance must be reviewed before any audit use. | Out of scope pending review. |

## Model directory boundary

`H:\xiyao\model` contains separate local model/cache folders, including
Qwen3.5, CheXbert, RadGraph, BiomedCLIP, and other legacy families. Their
presence does **not** activate them for this audit. Any future model family
must be explicitly frozen with checkpoint, license, preprocessing, score, and
explanation-interface identities before a new development or locked-test use.

## Guardrails

- The active primary protocol remains the local CheXlocalize audit; its test
  split stays sealed.
- This inventory does not alter any existing data/model/explanation/operator
  lock or any completed Phase-H result.
- Do not synchronize these assets to a server, add them to Git, or infer their
  contents from directory names alone.
- Before any future use, create a separate approved opening and a fail-closed
  provenance, identity, geometry, and split-status review.
