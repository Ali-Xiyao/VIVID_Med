# Clinical Instruction Schema

This schema supports the `vivid_med_clinical_instruction_proposal.md` execution line: convert CXR labels/reports into evidence-aware visual instructions and train deployable ViT representations.

## Record Grain

Each JSONL row is one instruction record, not one image. Multiple rows can share the same `sample_id` and `image_path`.

## Required Fields

| Field | Meaning |
| --- | --- |
| `instruction_id` | Stable unique instruction id. |
| `sample_id` | Source CXR sample id from UMS `extensions.sample_id` when available. |
| `image_path` | Original image path relative to `data/dataset`. |
| `report` | Original report text when available; `null` for label-only V1. |
| `question` | Clinical instruction/question. |
| `answer` | Short target answer. |
| `finding` | Target finding or `global` for report-level tasks. |
| `state` | `present`, `absent`, `uncertain`, or `null`. |
| `answerability` | `answerable` or `not_answerable`. |
| `uncertainty` | `definite`, `uncertain`, or `null`. |
| `laterality`, `location`, `severity` | Report-grounded attributes when available. |
| `evidence_phrase` | Exact report substring when report text exists; otherwise `null`. |
| `evidence_source` | `report_substring`, `normalized_report`, `llm_inferred`, `structured_label`, `none`, or `null`. |
| `answer_type` | One of the proposal-defined QA types. |
| `visual_dependency` | `high`, `medium`, or `low`. |
| `counterfactual_type` | Counterfactual label when applicable. |
| `quality_flags` | Explicit caveats such as `no_report_text` or `v1_label_to_qa`. |

## Version Boundaries

- `v1_label_to_qa`: derived from structured UMS labels only. It can test QA formatting, answerability handling, and instruction training plumbing, but it is not report-grounded evidence.
- `v2_report_grounded`: requires original report text. Do not claim V2 completion without report-backed `evidence_phrase` fields.
- `v3_report_grounded_cf`: requires V2 plus counterfactual choices whose correct option is report-supported.

## GLM Coding Plan Endpoint

- Domestic BigModel / Zhipu Coding Plan OpenAI-compatible base URL: `https://open.bigmodel.cn/api/coding/paas/v4`
- International Z.AI Coding Plan OpenAI-compatible base URL: `https://api.z.ai/api/coding/paas/v4`
- This repo uses `GLM_API_BASE` to override the endpoint and `GLM_API_KEY` for the key. Do not hard-code API keys in tracked files.
- The generator default is the domestic BigModel Coding Plan endpoint because the active key was provided for the Zhipu/BigModel flow.

## Validation Rules

- `state = null` must not be verbalized as absent.
- Location, laterality, device position, and severity must come from report text for V2/V3.
- Records without report text must include a `no_report_text` quality flag.
- Deployment-time LLM use is not allowed in final model claims; GLM is only a data generator unless a later experiment explicitly says otherwise.
