# Next-Stage External and Model Availability Audit

Created: 2026-06-29

## Model Availability

| Item | Local evidence | Execution decision |
| --- | --- | --- |
| Qwen3VL-2B-SHUF | `H:\Xiyao_Wang\001_models\qwen3-vl-2b-thinking-new` exists and is the active audited route. | Main route; all current next-stage configs use it unless explicitly marked otherwise. |
| Qwen3VL-2B-SHUF-TW | Same local 2B model plus TW configs. | Runnable via `SHUF-TW-*`, `PROG-Mix-TW`, `PROG-Mix-TW-10k`. |
| Qwen3-VL-4B | `H:\Xiyao_Wang\001_models\Qwen3-VL-4B-Instruct` exists. | Candidate upper-bound model; needs separate component/VRAM audit before formal training. |
| Qwen3-VL-8B | `H:\Xiyao_Wang\001_models\qwen3-vl-8b` and `Qwen3-VL-8B-Instruct` exist. | Boundary item on dual RTX 3090 host unless a smoke load proves memory fit. |
| Text-only scaffold controls | Existing historical scaffold/Qwen-Coder results are present in prior final tables. | Use as control context only; not a VLM main-method route. |

## External Dataset Availability

| Dataset | Current evidence | Decision |
| --- | --- | --- |
| NIH external 1k | `data/dataset/processed/nih_external_test_ums.jsonl` exists and previous Qwen3-VL transfer code verifies images. | Required external transfer route for all completed next-stage LP heads. |
| NIH full/larger | NIH image root exists at `H:\Xiyao_Wang\000_Public Dataset\NIH Chest X-rays`; current repo has only the prepared external UMS JSONL. | Larger subset requires creating a larger UMS manifest first. |
| CheXpert different split | `data/splits/chexpert_train_10k.jsonl`, `chexpert_val_fixed.jsonl`, and processed CheXpert UMS files exist. | Used for 10k scale data generation and can support split-sensitivity controls. |
| MIMIC-CXR | Images/reports exist locally; `mimic_cxr_aug_validate.csv` has image/view/text fields but no disease label columns. | Report-backed instruction generation is possible, but AUC transfer needs a UMS label manifest not currently present. |
| PadChest | No repo-local UMS manifest found; broad public-dataset path search did not return a usable PadChest root before timeout. | Explicit unavailable-data boundary unless a dataset path is added. |
| VinDr-CXR | No repo-local UMS manifest found; broad public-dataset path search did not return a usable VinDr root before timeout. | Explicit unavailable-data boundary unless a dataset path is added. |

## Implemented Support

- `evaluation.metrics.compute_classification_metrics` now reports macro/per-label AUPRC, ECE, and Brier score when probabilities are available.
- `scripts/evaluate_qwen3vl_lp_transfer.py` remains the authoritative NIH transfer evaluator and verifies image paths when `--verify-images` is set.
- `scripts/generate_ums_facts.py` creates structured-label scale facts from UMS splits with an explicit `ums_structured_fact_extraction` boundary.

## Boundaries

- UMS-derived 10k facts are structured-label supervision and do not include GLM report evidence spans.
- PadChest/VinDr are not claimed as completed external AUC evaluations without local UMS manifests and image roots.
- Larger VLM results are not claimed until model-specific component loading and VRAM audits succeed.
