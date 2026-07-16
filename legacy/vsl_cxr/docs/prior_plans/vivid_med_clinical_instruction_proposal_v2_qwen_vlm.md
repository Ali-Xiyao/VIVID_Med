# VIVID-Med Proposal v2: Qwen3-VL-Coupled Clinical Instruction Pretraining

## Summary

This v2 proposal replaces the previous piecemeal `timm ViT + text-only Qwen/Qwen-Coder + newly initialized projector` main route with a pretrained VLM-coupled route.

The main method initializes from a local Qwen3-VL model, freezes the language decoder, trains the vision tower and visual connector on report-grounded clinical visual instructions, then discards the LLM and evaluates the trained vision tower through LP and transfer tasks.

## Why The Piecemeal Scaffold Is Not The Main Method

The previous implementation used a standalone ViT and a text-only frozen LLM connected by a newly initialized projector. This design is useful as a controlled baseline, but it does not match the VLM-coupled nature of visual instruction pretraining.

The old scaffold has four limits:

1. The vision tower and language decoder were not jointly aligned before VIVID-Med training.
2. The text-only decoder has no native visual token interface, so the new projector must learn the alignment from scratch.
3. Fixed JSON or schema-like answers can encourage template learning.
4. Weak image-shuffle sensitivity in the previous MIMIC runs shows that loss improvements alone do not prove image-specific grounding.

Therefore the old scaffold is kept only as baseline or negative control. It must not be described as the main VLM-coupled method.

## Main Method

We initialize from a pretrained Qwen3-VL model. During clinical instruction pretraining, the language decoder is frozen, while the vision tower and visual connector are trainable. The training data are report-grounded clinical visual instructions generated from chest X-ray reports and UMS schemas. After pretraining, we discard the language decoder and evaluate the trained vision tower using linear probing and transfer tasks.

| Component | Source | Train? | Notes |
| --- | --- | --- | --- |
| Vision tower | Qwen3-VL local model | yes | Main representation learning target; exported for LP/transfer. |
| Visual connector / merger / projector | Qwen3-VL local model | yes | Adapts visual tokens to the frozen decoder during instruction pretraining. |
| Language decoder | Qwen3-VL local model | no by default | Frozen in the first pass; optional LoRA only as an upper bound. |
| Processor / tokenizer | Qwen3-VL local model | no | Use the native VLM processor and chat template. |
| LP head | downstream only | yes | Evaluates the exported visual representation without deploying the LLM. |

## Model Choices

| Model | Role | Is VLM? | Use |
| --- | --- | --- | --- |
| `qwen3-vl-2b-thinking-new` local path | main local Qwen3-VL 2B candidate | yes | First candidate because local config is `model_type=qwen3_vl`. |
| `Qwen3-VL-4B-Instruct` local path | fallback / stronger VLM | yes | Use if the 2B candidate fails quality or interface audit. |
| `Qwen3-VL-8B-Instruct` local path | optional upper bound | yes | Use only if memory and time allow. |
| `Qwen3.5-2B` | text-only scaffold control | no | Baseline/control only, not the main method. |
| `Qwen2.5-Coder-7B-Instruct` | legacy text scaffold | no | Existing old-run baseline only. |

Do not describe Qwen3.5 text-only models as VLM-coupled methods.

## Clinical Instruction Data

The data versions are:

| Data version | Name | Description | Role |
| --- | --- | --- | --- |
| D0 | Fixed JSON schema | Old UMS JSON target | old baseline |
| D1 | Label-to-QA | UMS label converted into simple QA | language control |
| D2 | Report-grounded evidence QA | QA generated from report evidence, location, uncertainty | main data |
| D3 | Report-grounded QA + counterfactual | D2 plus hard clinical choices and consistency checks | main strongest data |
| D4 | D3 + visual-token answer weighting | Weight high visual-dependency answers/tokens | strongest training objective |
| D5 | D3 + image/report shuffle pairs | Wrong image-report pairs for visual-dependence diagnostics or optional training | diagnostic |

Instruction types:

| Type | Goal |
| --- | --- |
| `finding_verification` | Determine whether a finding is supported. |
| `evidence_phrase` | Identify a short report evidence phrase. |
| `laterality_location` | Ask about side or location. |
| `severity` | Ask about mild/moderate/severe or small/large severity. |
| `uncertainty` | Preserve possible/probable/uncertain semantics. |
| `answerability` | Preserve null semantics: unmentioned is not absent. |
| `image_report_consistency` | Test support for image/report statements. |
| `counterfactual_choice` | Choose correct statement against hard negatives. |
| `hard_negative_laterality` | Left/right flip. |
| `hard_negative_state` | Present/absent/uncertain flip. |

## JSONL Schema

Each instruction record should support these fields:

```json
{
  "sample_id": "string",
  "image_path": "string",
  "study_id": "string_or_null",
  "report_text": "report text or section",
  "question": "clinical visual instruction",
  "answer": "short answer grounded in report",
  "answer_short": "minimal answer for high-weight tokens",
  "finding": "pleural_effusion | pneumothorax | cardiomegaly | ...",
  "state": "present | absent | uncertain | null | not_applicable",
  "answer_type": "finding_verification | evidence_phrase | laterality_location | severity | uncertainty | answerability | image_report_consistency | counterfactual_choice",
  "evidence_span": "exact phrase copied from report if available",
  "location": "left | right | bilateral | basilar | apical | diffuse | null",
  "severity": "mild | moderate | severe | null",
  "visual_dependency": "high | medium | low",
  "counterfactual_type": "none | state_flip | laterality_flip | uncertainty_flip | image_report_mismatch",
  "source": "glm_generated",
  "generation_model": "glm_api_model_name",
  "validation_status": "raw | auto_validated | rejected | manually_audited",
  "reject_reason": null
}
```

Legacy fields from the previous instruction pipeline are accepted as aliases during migration:

| v2 field | legacy alias |
| --- | --- |
| `report_text` | `report` |
| `evidence_span` | `evidence_phrase` |
| `image_report_consistency` | `report_consistency` |

## GLM Generation And Validation

GLM generation uses the Coding Plan endpoint through an environment variable key. The raw key must never be written to repo files, configs, or logs.

Validation rejects:

| Check | Reject if |
| --- | --- |
| evidence span | required evidence is not found in the report |
| null semantics | null/unmentioned is converted to absent |
| laterality | left/right is generated without support |
| severity | severity is invented without support |
| answer length | answer is too long or unsupported |
| finding vocabulary | finding is outside the allowed list |
| counterfactual anchor | counterfactual has no factual anchor |
| duplicate | near-duplicate question for same sample/finding/type |

## Code Tasks

| Task | Output |
| --- | --- |
| Qwen3-VL component audit | `scripts/audit_qwen3vl_components.py` and `outputs/final_tables/qwen3vl_component_audit.{md,json}` |
| Clinical instruction dataset | `data/clinical_instruction_dataset.py` |
| GLM generation wrapper | `scripts/generate_clinical_instructions_with_glm.py` |
| JSONL validation | `scripts/validate_clinical_instruction_jsonl.py` |
| Qwen3-VL trainer | `scripts/train_qwen3vl_clinical_instruction.py` |
| Vision extraction | `scripts/extract_qwen3vl_vision_backbone.py` |
| Qwen3-VL vision LP | `scripts/train_qwen3vl_vision_lp.py` |
| Pilot configs | `configs/qwen3vl_instruction/*.yaml` |

## Pilot Matrix

| ID | Route | Model | Data | Trainable | Purpose |
| --- | --- | --- | --- | --- | --- |
| P0 | old scaffold | timm ViT + text-only Qwen/Qwen-Coder | D0 fixed JSON | ViT + new projector | old baseline |
| P1 | old scaffold | timm ViT + text-only Qwen | D3 QA+CF | ViT + new projector | data change under old scaffold |
| P2 | VLM-coupled | Qwen3-VL-2B | D0 fixed JSON | vision + connector | model coupling only |
| P3 | VLM-coupled | Qwen3-VL-2B | D2 report-grounded QA | vision + connector | main data effect |
| P4 | VLM-coupled | Qwen3-VL-2B | D3 QA+CF | vision + connector | main strongest |
| P5 | VLM-coupled | Qwen3-VL-2B | D4 QA+CF+token weighting | vision + connector | visual-dependence weighting |
| P6 | data-only no-LM | ViT-B / Qwen3-VL vision tower | D3 labels converted to heads | vision + heads | data-only control |
| P7 | optional LoRA | Qwen3-VL-2B | D3 QA+CF | vision + connector + LLM LoRA | upper bound |

## Evaluation

Downstream metrics:

| Run | CheXpert macro-AUC | CheXpert macro-F1 | NIH macro-AUC | NIH macro-F1 | Best step | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| P0 |  |  |  |  |  |  |
| P1 |  |  |  |  |  |  |
| P2 |  |  |  |  |  |  |
| P3 |  |  |  |  |  |  |
| P4 |  |  |  |  |  |  |
| P5 |  |  |  |  |  |  |
| P6 |  |  |  |  |  |  |

Visual-dependence diagnostics:

| Run | Question-only score | Image-shuffle drop | Counterfactual pairwise acc | Paraphrase robustness | Template sensitivity | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| P0 |  |  |  |  |  |  |
| P1 |  |  |  |  |  |  |
| P2 |  |  |  |  |  |  |
| P3 |  |  |  |  |  |  |
| P4 |  |  |  |  |  |  |
| P5 |  |  |  |  |  |  |

## Success And Failure Rules

P4/P5 are successful if at least two of the following hold:

1. CheXpert or NIH LP macro-AUC beats old scaffold P0/P1 by at least `+0.01`.
2. Counterfactual pairwise accuracy improves by at least `+0.05`.
3. Image-shuffle drop is larger than old scaffold.
4. Question-only score is lower than old scaffold.
5. Template/paraphrase robustness improves.
6. Rare/high-null/uncertain-heavy subgroup AUC improves by at least `+0.01`.

If VLM-coupled runs do not beat the old scaffold or no-LM controls, the paper story should honestly pivot back toward schema-aware/no-LM representation learning rather than overclaiming LLM benefits.
