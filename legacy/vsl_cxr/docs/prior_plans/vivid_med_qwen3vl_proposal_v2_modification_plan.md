# VIVID-Med Proposal v2 修改方案：从“散装 Text-LLM Scaffold”改成“Qwen3-VL 一套 VLM + Frozen LLM 的 Clinical Instruction Pretraining”

> 目标：把现有 proposal 从 `timm ViT + text-only Qwen + 新 projector + frozen LLM` 的散装路线，修改为 **pretrained VLM-coupled** 路线：  
> **使用一套已经视觉-语言对齐的 Qwen3-VL-2B VLM，冻结语言模型，训练视觉塔与视觉连接层，用 GLM/Qwen API 生成的 CXR clinical evidence instruction 进行继续预训练，最后抽出训练后的 vision tower 做 LP / transfer。**

---

## 0. 核心修改结论

### 0.1 原方案的问题

当前旧方案是：

```text
image -> timm vit_base_patch16_224 -> newly initialized projector -> frozen text-only Qwen / Qwen-Coder -> answer/schema loss
```

这个方案的问题：

1. **模型是散装拼接**：ViT 和 Qwen text LLM 不是一起预训练好的 VLM。
2. **Qwen text LLM 没有原生视觉 token 对齐能力**：projector 要从零学习如何把视觉 token 变成 LLM 能理解的 prefix。
3. **Qwen-Coder / text LLM 容易学模板**：尤其固定 JSON / schema / key order / structured text。
4. **training signal 可能不够视觉依赖**：模型可能优化的是固定格式文本，而不是 CXR 视觉证据。
5. **和 ViTP 的味道不一致**：ViTP 的关键是从一套 pretrained VLM 出发，再做 domain instruction continual pretraining。

因此：  
**旧方案不能再作为主方法，只能作为 baseline / negative control。**

---

### 0.2 新主方案

新主方案是：

```text
pretrained Qwen3-VL-2B-Instruct
  ├── vision tower / visual encoder        train
  ├── visual projector / merger / connector train
  └── language model decoder               freeze

image + clinical question -> short clinical answer loss
pretraining ends -> discard LLM -> extract trained vision tower -> LP / transfer
```

核心思想：

> 不是随便拼一个 ViT 和 text LLM，而是从一套已经视觉-语言对齐好的 VLM 初始化。  
> 冻结 LLM，训练视觉侧，让医学 instruction loss 反向更新 vision tower 和 visual connector。  
> 最后部署/评估只使用 vision tower，不依赖 LLM。

---

## 1. 文档层面的修改任务

Codex 需要把原文档 `vivid_med_clinical_instruction_proposal.md` 修改为 v2，建议另存为：

```text
vivid_med_clinical_instruction_proposal_v2_qwen_vlm.md
```

不要覆盖原文档，除非用户明确要求。

---

## 2. 文档需要新增 / 重写的章节

### 2.1 新增章节：Why the Current Piecemeal Scaffold Is Not the Main Method

需要写清楚：

```text
The previous implementation used a standalone ViT and a text-only frozen LLM connected by a newly initialized projector. This design is useful as a controlled baseline but does not match the VLM-coupled nature of visual instruction pretraining. The main proposal is therefore revised to use a pretrained Qwen3-VL model, whose vision tower, visual connector, and language decoder have already been jointly aligned.
```

中文解释：

```text
旧方法是“散装式”的 ViT + text LLM scaffold，不是完整 VLM。它可以回答 text-only frozen decoder 能不能作为监督空间，但不适合作为主方法。新版主方法必须用一套已经视觉-语言对齐的 VLM。
```

---

### 2.2 新增章节：Main Method — Qwen3-VL-Coupled Clinical Instruction Pretraining

写成：

```text
We initialize from Qwen3-VL-2B-Instruct. During clinical instruction pretraining, the language decoder is frozen, while the vision tower and visual connector are trainable. The training data are report-grounded clinical visual instructions generated from CXR reports and UMS schemas. After pretraining, we discard the language decoder and evaluate the trained vision tower using linear probing and transfer tasks.
```

组件策略表：

| Component | Source | Train? | Notes |
|---|---|---|---|
| Vision tower | Qwen3-VL-2B-Instruct | yes | 主学习对象；最终抽出来做 LP |
| Visual connector / merger / projector | Qwen3-VL-2B-Instruct | yes | 必须训练，适配 CXR instruction |
| LLM decoder | Qwen3-VL-2B-Instruct | freeze first | 第一轮冻结；后续可试 LoRA |
| Tokenizer / processor | Qwen3-VL-2B-Instruct | no | 使用 VLM 原生 processor |
| Standalone LP head | downstream only | yes | 评估 vision tower 表征 |

---

### 2.3 新增章节：Model Choices

明确区分 Qwen3.5 和 Qwen3-VL：

| Model | Role | Is VLM? | Use in proposal |
|---|---|---|---|
| `Qwen/Qwen3-VL-2B-Instruct` | main VLM-coupled branch | yes | 主方法 |
| `Qwen/Qwen3-VL-2B-Instruct-FP8` | memory-saving optional variant | yes | only if BF16 memory fails |
| `Qwen/Qwen3.5-2B` | text-only scaffold control | no | baseline / ablation |
| `Qwen/Qwen3.5-2B-Base` | text-only base control | no | optional, not main |
| previous `Qwen2.5-Coder-7B` | legacy text scaffold | no | old baseline only |

关键警告：

```text
Do not describe Qwen3.5-2B as a VLM-coupled method. It is a text-only LLM control. The main ViTP-like route must use Qwen3-VL-2B-Instruct or another pretrained VLM with an aligned vision tower and language decoder.
```

---

## 3. 数据侧修改：GLM 生成 Clinical Visual Instruction

### 3.1 保留并升级原来的数据版本

| Data version | Name | Description | Role |
|---|---|---|---|
| D0 | Fixed JSON schema | 旧 UMS JSON target | old baseline |
| D1 | Label-to-QA | 从 UMS label 直接改成简单 QA | simple language control |
| D2 | Report-grounded evidence QA | 从报告抽 evidence/location/uncertainty 生成 QA | main data |
| D3 | Report-grounded QA + counterfactual | 在 D2 上加反事实选择/一致性判断 | main strongest data |
| D4 | D3 + visual-token answer weighting | answer 中视觉相关 token 加权 | strongest training objective |
| D5 | D3 + image/report shuffle pairs | 生成错误 image-report pair 做 visual-dependence 检测 | diagnostic / optional training |

---

### 3.2 每张图生成的 instruction 类型

Codex 需要把文档中的 instruction 类型改成以下 10 类：

| Type | Question goal | Example |
|---|---|---|
| `finding_verification` | 判断某 finding 是否有证据 | Is there visual evidence of pleural effusion? |
| `evidence_phrase` | 要求给出短证据 | What report evidence supports cardiomegaly? |
| `laterality_location` | 问左右/位置 | Which side shows pleural effusion? |
| `severity` | 问程度 | Is the edema mild, moderate, or severe? |
| `uncertainty` | 问确定/可能/不确定 | Is this finding definite or uncertain? |
| `answerability` | 问是否可回答 | Is pneumothorax answerable from this image-report pair? |
| `image_report_consistency` | 判断图像-报告陈述是否一致 | Does the image-report pair support this statement? |
| `counterfactual_choice` | 正确陈述 vs 反事实陈述选择 | Which statement is better supported? |
| `hard_negative_laterality` | left/right flip | left effusion vs right effusion |
| `hard_negative_state` | present/absent/uncertain flip | present pneumothorax vs no pneumothorax |

---

### 3.3 GLM API 生成格式

每条 instruction 必须输出 JSONL，schema 如下：

```json
{
  "sample_id": "string",
  "image_path": "string",
  "study_id": "string_or_null",
  "report_text": "original report or report section",
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

---

### 3.4 GLM prompt 模板

文档中加入完整 prompt：

```text
You are a radiology instruction-data generator.

Input:
1. A chest X-ray report.
2. A structured finding schema with states: present, absent, uncertain, null.
3. A fixed list of findings.

Task:
Generate clinically faithful visual instruction examples for training a CXR vision-language pretraining model.

Rules:
- Do not invent findings not supported by the report.
- If a finding is not mentioned, do not label it as absent.
- Prefer questions that require visual evidence, such as laterality, location, severity, uncertainty, or image-report consistency.
- Extract evidence_span as an exact phrase from the report whenever possible.
- For null / unmentioned findings, only generate answerability questions, not absent claims.
- Generate hard counterfactuals only by flipping a known supported fact:
  present <-> absent,
  left <-> right,
  definite <-> uncertain.
- Keep answers short.
- Output JSONL records only.
- Each record must include:
  sample_id, question, answer, answer_short, finding, state, answer_type,
  evidence_span, location, severity, visual_dependency, counterfactual_type.
```

---

### 3.5 自动过滤规则

生成后必须自动过滤：

| Check | Reject if |
|---|---|
| evidence span check | evidence_span not found in report and answer_type requires evidence |
| null semantics check | null finding converted to absent |
| laterality check | left/right generated without report phrase support |
| severity check | severity invented without mild/moderate/severe/tiny/small/large phrase |
| answer length check | answer too long or contains unsupported reasoning |
| finding vocabulary check | finding not in allowed finding list |
| counterfactual check | counterfactual has no known factual anchor |
| duplicate check | near-duplicate question for same sample/finding/type |

输出：

```text
outputs/instruction_data/glm_raw/*.jsonl
outputs/instruction_data/glm_validated/*.jsonl
outputs/instruction_data/glm_rejected/*.jsonl
outputs/final_tables/instruction_data_audit.csv
outputs/final_tables/instruction_data_audit.md
```

---

## 4. 模型/代码实现任务

### 4.1 Task A：Qwen3-VL component audit

Codex 先实现一个只读脚本：

```text
scripts/audit_qwen3vl_components.py
```

功能：

1. 加载 `Qwen/Qwen3-VL-2B-Instruct`。
2. 打印 model class。
3. 打印 processor/tokenizer class。
4. 自动查找并列出：
   - vision tower / visual encoder
   - visual connector / merger / projector
   - language model / decoder
5. 打印各部分参数量。
6. 提供 freeze plan：
   - trainable vision tower
   - trainable visual connector
   - frozen LLM
7. 用一张 dummy image + dummy question 做 forward loss check。
8. 不启动训练。

输出：

```text
outputs/final_tables/qwen3vl_component_audit.md
outputs/final_tables/qwen3vl_component_audit.json
```

停止条件：

- model 无法加载；
- processor 无法处理 image+text；
- 找不到 vision / language submodules；
- forward loss 无法计算。

---

### 4.2 Task B：Clinical instruction dataset

新增：

```text
data/clinical_instruction_dataset.py
```

功能：

- 读取 D0-D5 JSONL。
- 加载 CXR image。
- 构造 Qwen3-VL messages。
- 支持 answer-only loss mask。
- 支持 visual-dependent token weighting metadata。
- 支持 debug truncation。
- 支持 image/report shuffle pair。

输出 batch 字段建议：

```python
{
  "pixel_values": ...,
  "input_ids": ...,
  "attention_mask": ...,
  "labels": ...,
  "loss_weights": ...,
  "metadata": {
    "sample_id": ...,
    "finding": ...,
    "answer_type": ...,
    "visual_dependency": ...
  }
}
```

---

### 4.3 Task C：Qwen3-VL clinical instruction trainer

新增：

```text
scripts/train_qwen3vl_clinical_instruction.py
```

核心要求：

- 支持 config YAML。
- 支持 `--debug`。
- 支持 `--resume`。
- 支持 WMI/detached-friendly logging。
- 冻结 LLM decoder。
- 训练 vision tower + visual connector。
- 可选训练 last N vision blocks。
- 可选 vision LoRA。
- 可选 LLM LoRA，但第一轮默认关闭。
- 保存：
  - full checkpoint
  - extracted vision checkpoint
  - trainable-only checkpoint
  - config snapshot
  - progress.json
  - metrics_step_*.json
  - metrics_final.json

训练 loss：

1. answer-only cross entropy；
2. optional token-weighted CE；
3. optional counterfactual margin loss；
4. optional image-report shuffle contrastive loss。

---

### 4.4 Task D：Extract vision tower checkpoint

新增：

```text
scripts/extract_qwen3vl_vision_backbone.py
```

功能：

- 从 full VLM checkpoint 中抽取：
  - vision tower state_dict
  - visual connector state_dict
  - config metadata
- 输出：

```text
outputs/<run>/vision_backbone.pt
outputs/<run>/vision_connector.pt
outputs/<run>/vision_export_manifest.json
```

---

### 4.5 Task E：Generic LP for Qwen3-VL vision tower

新增：

```text
scripts/train_qwen3vl_vision_lp.py
```

原因：Qwen3-VL 的 vision tower 不一定是 timm ViT-B，不能直接用旧 LP 脚本。

要求：

- 加载 extracted vision tower。
- 支持 pooling：
  - cls
  - mean
  - spatial mean
  - last hidden mean
  - connector output mean
- 冻结 vision tower。
- 训练 linear head。
- 输出 CheXpert / NIH / fixed split metrics。

---

## 5. 实验矩阵

### 5.1 Pilot matrix：先跑 8 个

| ID | Model route | Model | Data | Trainable | Purpose | Done? |
|---|---|---|---|---|---|---|
| P0 | old scaffold | timm ViT + Qwen3.5-2B / old Qwen-Coder | D0 fixed JSON | ViT + new projector | old baseline |  |
| P1 | old scaffold | timm ViT + Qwen3.5-2B | D3 QA+CF | ViT + new projector | data change under old scaffold |  |
| P2 | VLM-coupled | Qwen3-VL-2B | D0 fixed JSON | vision + connector | model coupling only |  |
| P3 | VLM-coupled | Qwen3-VL-2B | D2 report-grounded QA | vision + connector | main data effect |  |
| P4 | VLM-coupled | Qwen3-VL-2B | D3 QA+CF | vision + connector | main strongest |  |
| P5 | VLM-coupled | Qwen3-VL-2B | D4 QA+CF+token weighting | vision + connector | visual-dependence weighting |  |
| P6 | data-only no-LM | ViT-B / Qwen3-VL vision tower | D3 labels converted to heads | vision + heads | data-only control |  |
| P7 | optional LoRA | Qwen3-VL-2B | D3 QA+CF | vision + connector + LLM LoRA | upper bound |  |

最关键比较：

| Comparison | Interpretation |
|---|---|
| P0 vs P2 | 是否“一套 VLM 初始化”比散装模型更好 |
| P1 vs P4 | 同样 clinical instruction 下，VLM-coupled 是否优于散装 scaffold |
| P2 vs P4 | clinical QA+CF 是否比 fixed JSON 更好 |
| P4 vs P6 | 收益来自 VLM decoder 还是 GLM-generated data |
| P4 vs P5 | token weighting 是否增强 visual dependence |
| P4 vs P7 | LLM LoRA 是否值得 |

---

### 5.2 Scale matrix

跑完 pilot 后再扩展。

| Scale | Train samples | P0 | P1 | P2 | P3 | P4 | P5 | P6 | P7 |
|---|---:|---|---|---|---|---|---|---|---|
| debug | 20-200 | yes | yes | yes | yes | yes | yes | yes | optional |
| 1k | 1000 | yes | yes | yes | yes | yes | yes | yes | optional |
| 3k | 3000 | optional | optional | optional | yes | yes | yes | yes | optional |
| 10k | 10000 | no | optional | optional | yes | yes | yes | optional | optional |
| full | all | no | optional | optional | yes | yes | yes | optional | optional |

---

## 6. 评估指标

### 6.1 Downstream LP

| Run | CheXpert macro-AUC | CheXpert macro-F1 | NIH macro-AUC | NIH macro-F1 | Best step | Notes |
|---|---:|---:|---:|---:|---:|---|
| P0 |  |  |  |  |  |  |
| P1 |  |  |  |  |  |  |
| P2 |  |  |  |  |  |  |
| P3 |  |  |  |  |  |  |
| P4 |  |  |  |  |  |  |
| P5 |  |  |  |  |  |  |
| P6 |  |  |  |  |  |  |
| P7 |  |  |  |  |  |  |

---

### 6.2 Visual-dependence diagnostics

| Run | Question-only score | Image-shuffle drop | Counterfactual pairwise acc | Paraphrase robustness | Template sensitivity | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| P0 |  |  |  |  |  |  |
| P1 |  |  |  |  |  |  |
| P2 |  |  |  |  |  |  |
| P3 |  |  |  |  |  |  |
| P4 |  |  |  |  |  |  |
| P5 |  |  |  |  |  |  |

Definitions:

- `question-only score`: remove image, keep question. High score means model may rely on text prior.
- `image-shuffle drop`: pair question/report with wrong image. Large drop means visual dependency.
- `counterfactual pairwise acc`: correct statement NLL < counterfactual statement NLL.
- `paraphrase robustness`: same medical meaning with different wording.
- `template sensitivity`: fixed JSON/key/order perturbation sensitivity.

---

### 6.3 Clinical subgroup metrics

| Run | Common AUC | Rare AUC | High-null AUC | Uncertain-heavy AUC | Support-device AUC | Notes |
|---|---:|---:|---:|---:|---:|---|
| P0 |  |  |  |  |  |  |
| P1 |  |  |  |  |  |  |
| P2 |  |  |  |  |  |  |
| P3 |  |  |  |  |  |  |
| P4 |  |  |  |  |  |  |
| P5 |  |  |  |  |  |  |

---

### 6.4 Cost table

| Run | Model | Trainable params | Frozen params | Peak VRAM | GPU-hours | Steps/sec | Deployment model | Deployment LLM? |
|---|---|---:|---:|---:|---:|---:|---|---|
| P0 |  |  |  |  |  |  | ViT | no |
| P2 |  |  |  |  |  |  | Qwen3-VL vision tower | no |
| P4 |  |  |  |  |  |  | Qwen3-VL vision tower | no |
| P5 |  |  |  |  |  |  | Qwen3-VL vision tower | no |

---

## 7. 成功/失败判断

### 7.1 主方法成功条件

P4 / P5 被认为成功，如果满足至少两条：

1. CheXpert or NIH LP macro-AUC > P0/P1 old scaffold by at least `+0.01`.
2. Counterfactual pairwise accuracy improves by at least `+0.05`.
3. Image-shuffle drop is larger than old scaffold, showing stronger image dependence.
4. Question-only score is lower than old scaffold, showing less text-prior shortcut.
5. Template/paraphrase robustness improves.
6. Rare/high-null/uncertain-heavy subgroup improves by at least `+0.01`.

---

### 7.2 重要分支解释

| Result pattern | Interpretation | Paper claim |
|---|---|---|
| P4 > P1 | VLM-coupled initialization matters | 支持“散装拼接是瓶颈” |
| P1 ≈ P4 | 数据改写比 VLM coupling 更重要 | LLM decoder 不一定是核心 |
| P4 > P2 | clinical instruction 比 fixed JSON 更视觉依赖 | 支持新数据设计 |
| P5 > P4 | token weighting 增强视觉监督 | 支持视觉相关 token loss |
| P6 ≈ P4 | GLM-generated data 是主因，decoder 不必要 | 降级 VLM claim |
| P7 > P4 | 轻量 LLM adaptation 有用 | 可作为 upper bound |
| P4 not better than P0 | VLM route 不成立，回到 schema-aware no-LM 主线 | 不硬吹 LLM |

---

## 8. Codex 顶层执行 Prompt

Codex 可以直接使用下面这段作为目标模式 prompt。

```text
Project: VIVID-Med Proposal v2 — Qwen3-VL-Coupled Clinical Instruction Pretraining

Goal:
  Revise the existing proposal and prepare code/config scaffolds so the main method is no longer a piecemeal timm ViT + text-only Qwen scaffold.
  The new main method must use a pretrained Qwen3-VL-2B VLM as one coupled model:
    - initialize from Qwen/Qwen3-VL-2B-Instruct
    - freeze the language decoder
    - train the vision tower and visual connector
    - train on GLM-generated report-grounded clinical visual instructions
    - discard the LLM after pretraining
    - evaluate the trained vision tower by LP / transfer

Model policy:
  - Qwen/Qwen3-VL-2B-Instruct is the main VLM-coupled model.
  - Qwen/Qwen3.5-2B is text-only and may only be used as a scaffold/control baseline, not as the main VLM method.
  - The legacy timm ViT + Qwen-Coder setup must be clearly labeled as old scaffold baseline.
  - Do not describe text-only Qwen3.5 as a VLM.
  - Do not train SPD variants.

Document tasks:
  1. Create or update vivid_med_clinical_instruction_proposal_v2_qwen_vlm.md.
  2. Add sections:
      - Why the piecemeal scaffold is insufficient
      - Main method: Qwen3-VL-coupled frozen-LLM clinical instruction pretraining
      - Model choices: Qwen3-VL-2B vs Qwen3.5-2B controls
      - GLM instruction generation
      - Training/freezing policy
      - Experiment matrix P0-P7
      - Evaluation metrics and empty tables
      - Success/failure decision rules
  3. Preserve the old scaffold only as baseline / negative control.

Code scaffolding tasks:
  1. Add scripts/audit_qwen3vl_components.py.
  2. Add data/clinical_instruction_dataset.py.
  3. Add scripts/generate_clinical_instructions_with_glm.py.
  4. Add scripts/validate_clinical_instruction_jsonl.py.
  5. Add scripts/train_qwen3vl_clinical_instruction.py.
  6. Add scripts/extract_qwen3vl_vision_backbone.py.
  7. Add scripts/train_qwen3vl_vision_lp.py.
  8. Add configs/qwen3vl_instruction/*.yaml for P2-P5 pilot runs.
  9. Add configs/text_scaffold_controls/*.yaml for P0-P1 only if needed.

Implementation constraints:
  - Do not launch full training automatically.
  - First run only component audit and debug data validation.
  - Every script must support --debug.
  - Every training script must write config snapshot, progress.json, checkpoints, final metrics, and failure logs.
  - Every generated instruction must preserve null semantics: unmentioned is not absent.
  - Reject hallucinated evidence spans.
  - Use answer-only loss by default.
  - Add optional visual-dependent token weighting but keep it off in basic P3/P4 unless enabled by config.
  - No SPD variants.

First outputs:
  docs/proposal or root markdown:
    vivid_med_clinical_instruction_proposal_v2_qwen_vlm.md
  outputs/final_tables:
    qwen3vl_component_audit.md
    instruction_data_audit.md
    qwen3vl_pilot_matrix.md
    qwen3vl_empty_result_tables.md
```

---

## 9. 文档中的新 paper story

建议把 abstract / proposal summary 改成：

```text
Existing LLM-guided CXR pretraining can collapse into template learning when the model is trained to predict fixed schemas with a text-only frozen decoder. We propose Clinical Evidence Instruction Pretraining, a VLM-coupled framework that starts from a pretrained Qwen3-VL model, freezes the language decoder, and updates the vision tower and visual connector using report-grounded clinical instructions. These instructions ask about findings, evidence, laterality, uncertainty, answerability, and image-report consistency, forcing the visual backbone to encode clinically meaningful evidence rather than fixed JSON surface forms. After pretraining, the language decoder is discarded and the trained vision tower is evaluated as a deployable CXR representation through linear probing and transfer.
```

中文版本：

```text
固定 schema 预测容易退化成模板学习。我们提出 Clinical Evidence Instruction Pretraining：从已经视觉-语言对齐的 Qwen3-VL 出发，冻结语言 decoder，只训练视觉塔和视觉连接层；训练数据不是固定 JSON，而是由 GLM 从胸片报告生成的临床证据型 instruction，包括 finding、证据、左右位置、不确定性、answerability 和 image-report consistency。训练结束后丢弃 LLM，只评估视觉塔的 CXR 表征能力。
```

---

## 10. 最终注意事项

1. **主方法必须是 Qwen3-VL 一套 VLM，不是 Qwen3.5 text LLM。**
2. **Qwen3.5-2B 只能作为 text-only scaffold control。**
3. **冻结的是 language decoder，不是整个 VLM。vision tower 必须训练。**
4. **只换模型不够，fixed JSON 也必须换成 clinical instruction。**
5. **deployment 仍然只用 vision tower，不部署 LLM。**
6. **评价不能只看 AUC，还要看 visual-dependence diagnostics。**
7. **如果 VLM-coupled route 不赢，要诚实改回 schema-aware/no-LM 主线，不硬吹 LLM。**

---

## 11. 执行结果回写（2026-06-28）

本节是对上述方案的实际执行结果回写。前文保留为设计目标；本节记录已经跑完的产物、指标、边界和最终解释。

### 11.1 完成状态

| Requirement | Status | Evidence | Boundary |
|---|---|---|---|
| Preserve old scaffold as baseline/control only | Completed | `outputs/final_tables/qwen3vl_pilot_matrix.md` | 旧 MIMIC/text-scaffold 线按用户要求收尾后没有继续扩展。 |
| Qwen3-VL v2 proposal document | Completed | `vivid_med_clinical_instruction_proposal_v2_qwen_vlm.md` | 本 plan 文件也已写回最终结果。 |
| Qwen3-VL component audit and freeze plan | Completed | `outputs/final_tables/qwen3vl_component_audit.{json,md}` | Local model: `H:/Xiyao_Wang/001_models/qwen3-vl-2b-thinking-new`。 |
| D0-D4 instruction data generation/validation | Completed | `outputs/final_tables/instruction_data_audit.{csv,md}`; `outputs/instruction_data/glm_validated/` | rejected rows 保留为 data-quality evidence。 |
| P2-P5 Qwen3-VL instruction training | Completed | `outputs/final_tables/qwen3vl_instruction_training_results.{csv,md}` | 四个 run 都到 1000 steps；`language_decoder_trainable=0`。 |
| Extract trained vision-side checkpoints | Completed | `outputs/final_tables/qwen3vl_extraction_manifest.{csv,md}` | 导出 vision tower、visual connector 和 combined vision-side states。 |
| CheXpert LP for Base/P2-P6 | Completed | `outputs/final_tables/qwen3vl_vision_lp_results.{csv,md}` | P6 是 no-LM label-head control，不是完全等价的 GLM D3 instruction-head。 |
| NIH transfer | Completed | `outputs/final_tables/qwen3vl_nih_transfer_results.{csv,md}` | NIH external 1k subset，image audit 缺图数为 0；不是 full NIH external test。 |
| Visual-dependence diagnostics | Completed | `outputs/final_tables/qwen3vl_visual_dependence_results.{csv,md}` | question-only delta 大，但 image-shuffle delta 仍小。 |
| Counterfactual diagnostics | Completed | `outputs/final_tables/qwen3vl_counterfactual_results*.{csv,md}` | 只对 P4/P5 的 option-formatted subset 严格成立；很多 `counterfactual_choice` 不是显式 A/B/C/D。 |
| Paraphrase/template diagnostics | Completed | `outputs/final_tables/qwen3vl_paraphrase_robustness_results*.{csv,md}` | style rewrite 更难，说明 robustness 被测量到但没有完全解决。 |
| Clinical subgroup summary | Completed | `outputs/final_tables/qwen3vl_subgroup_results.{csv,md}` | active LP configs 使用 common 8-label subset，因此没有评估 `Support Devices`。 |
| Cost/runtime summary | Completed | `outputs/final_tables/qwen3vl_cost_table.{csv,md}` | 已记录 GPU-hours 和 steps/sec；peak VRAM 没有被训练 metrics 捕获。 |
| P7 LoRA upper bound | Optional skipped | `outputs/final_tables/qwen3vl_pilot_matrix.md` | P7 是 optional，不作为 P2-P6 完成门槛。 |

### 11.2 主训练结果

| Pilot | Data / route | Step | Best val loss | Trainable side | Language decoder trainable |
|---|---|---:|---:|---|---:|
| P2 | D0 fixed JSON schema | 1000 | 0.026330 | vision tower + visual connector | 0 |
| P3 | D2 report-grounded QA | 1000 | 1.114388 | vision tower + visual connector | 0 |
| P4 | D3 report-grounded QA + counterfactual | 1000 | 0.886641 | vision tower + visual connector | 0 |
| P5 | D4 counterfactual weighted | 1000 | 0.877448 | vision tower + visual connector | 0 |

### 11.3 CheXpert LP 结果

| Pilot | CheXpert macro-AUC | Macro-F1 | Micro-F1 | Notes |
|---|---:|---:|---:|---|
| Base | 0.679003 | 0.732351 | 0.697266 | Base Qwen3-VL vision tower。 |
| P2 | 0.666206 | 0.778519 | 0.724609 | Fixed JSON target；低 instruction loss 不等于 visual grounding。 |
| P3 | 0.685911 | 0.822713 | 0.703613 | Report-grounded QA 相比 P2 提升 macro-AUC。 |
| P4 | 0.689561 | 0.804594 | 0.727051 | instruction runs 中 CheXpert macro-AUC 最好。 |
| P5 | 0.681452 | 0.841326 | 0.707031 | macro-F1 最好，但 macro-AUC 不是最好。 |
| P6 | 0.663100 | 0.788543 | 0.761230 | Data-only no-LM control：Qwen3-VL vision tower + label head。 |

### 11.4 NIH Transfer 结果

| Pilot | NIH macro-AUC | Macro-F1 | Micro-F1 | Evaluated records | Missing images |
|---|---:|---:|---:|---:|---:|
| Base | 0.564267 | 0.192239 | 0.386625 | 1000 | 0 |
| P2 | 0.555443 | 0.190606 | 0.372250 | 1000 | 0 |
| P3 | 0.562041 | 0.186603 | 0.469125 | 1000 | 0 |
| P4 | 0.568298 | 0.192138 | 0.352375 | 1000 | 0 |
| P5 | 0.564290 | 0.182663 | 0.293125 | 1000 | 0 |
| P6 | 0.526401 | 0.120954 | 0.527625 | 1000 | 0 |

P4 是 Qwen3-VL instruction runs 中 NIH macro-AUC 最好的一组，但相对 Base 的优势很小；因此这只支持 modest transfer signal，不支持强 transfer claim。

### 11.5 Visual Dependence / Counterfactual / Robustness

| Pilot | Question-only delta | Image-shuffle delta | Counterfactual pairwise acc | Paraphrase readout |
|---|---:|---:|---:|---|
| P2 | +0.571060 | +0.008150 | N/A | style rewrite mean delta +0.008295 |
| P3 | +1.836000 | +0.017665 | N/A | style rewrite mean delta +0.016387 |
| P4 | +1.777182 | +0.008729 | 0.789954 | style rewrite mean delta +0.012060 |
| P5 | +1.781887 | +0.007566 | 0.767123 | style rewrite mean delta +0.016090 |

解释边界：
- blank/question-only 会显著提高 teacher-forced loss，说明模型对 image presence 敏感。
- image-shuffle delta 仍然很小，所以不能声称强 image-specific grounding 已验证。
- P4/P5 counterfactual accuracy 只适用于 option-formatted subset；多数 `counterfactual_choice` 记录并不是显式 A/B/C/D。
- style rewrite 比 clinical rewrite 更难，template sensitivity 仍是 limitation。

### 11.6 Subgroup 和 Cost

Subgroup 结果见 `outputs/final_tables/qwen3vl_subgroup_results.md`。当前 LP configs 使用 common 8-label subset，因此 final subgroup table 没有 `Support Devices`。

Cost 结果见 `outputs/final_tables/qwen3vl_cost_table.md`。已记录 runtime-derived GPU-hours 和 steps/sec；peak VRAM 未被训练 metrics 捕获。

### 11.7 最终科学结论边界

可以谨慎支持的结论：

> Qwen3-VL-coupled clinical instruction pretraining 可以在本 repo 端到端跑通：冻结 language decoder，训练 vision tower 和 visual connector，训练后抽取 vision-side checkpoint，并完成 CheXpert LP、NIH transfer、visual-dependence、counterfactual、paraphrase、subgroup 和 cost/runtime 评估。P4 是整体最强的 instruction run。

不能支持的更强结论：

> 不能声称已经验证 strong image-specific grounding，因为 P2-P5 的 image-shuffle loss delta 都仍然很小。

### 11.8 最终产物索引

| Artifact | Path |
|---|---|
| Final requirement audit | `outputs/final_tables/qwen3vl_final_requirement_audit.md` |
| Pilot matrix | `outputs/final_tables/qwen3vl_pilot_matrix.md` |
| Instruction training table | `outputs/final_tables/qwen3vl_instruction_training_results.md` |
| Extraction manifest table | `outputs/final_tables/qwen3vl_extraction_manifest.md` |
| CheXpert LP table | `outputs/final_tables/qwen3vl_vision_lp_results.md` |
| NIH transfer table | `outputs/final_tables/qwen3vl_nih_transfer_results.md` |
| Visual-dependence table | `outputs/final_tables/qwen3vl_visual_dependence_results.md` |
| Counterfactual table | `outputs/final_tables/qwen3vl_counterfactual_results.md` |
| Paraphrase robustness table | `outputs/final_tables/qwen3vl_paraphrase_robustness_results.md` |
| Subgroup table | `outputs/final_tables/qwen3vl_subgroup_results.md` |
| Cost table | `outputs/final_tables/qwen3vl_cost_table.md` |
