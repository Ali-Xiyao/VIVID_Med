# VIVID-Med 新 Proposal：Clinical Evidence Instruction Pretraining for Deployable CXR ViTs

> 版本：2026-06-27  
> 目标：把当前 **fixed JSON schema prediction** 升级成更像 ViTP 但仍然医疗相关、不会直接撞车的 **临床证据型视觉 instruction pretraining**。  
> 核心问题：**什么时候语言监督真的在教 ViT 看图，而不是只教它背固定医学表格？**

---

## 0. 一句话总方向

当前旧方案：

```text
CXR image -> ViT visual prefix -> frozen LLM -> fixed UMS JSON
```

问题：fixed JSON 太像多标签分类，LLM 很容易学字段顺序、key、模板和状态词，不一定真的逼 ViT 学视觉证据。

新方案：

```text
CXR report + UMS schema --GLM offline--> clinically grounded visual instructions
CXR image + clinical question -> short evidence-aware answer
最终只部署 ViT / ViT + 小 head，不部署 LLM
```

新的主线不是“让 LLM 预测 JSON”，而是：

> **用 GLM 把胸片报告改写成必须依赖医学影像证据才能回答的 instruction，训练 ViT 学 clinical visual evidence。**

---

## 1. 为什么要改：旧方案和 ViTP 的“味道”差在哪里

### 1.1 ViTP 的味道

ViTP 的训练不是单纯让模型填一个固定表格，而是用 image + instruction 让模型输出视觉相关答案。它的任务里有 VQA、visual grounding、caption、classification 等多种形式，且强调 instruction data 要和下游能力对齐：如果下游要检测/定位，预训练就要包含 grounding / fine-grained VQA。

ViTP 还有两个重要启发：

1. **任务必须迫使视觉 token 有用**：模型要回答问题，ViT token 必须带有图像区域、物体、关系、语义信息。
2. **LLM 不能太强到替 ViT 答题**：ViTP 论文里的 LLM size ablation 显示，LLM 越大下游性能反而可能下降，因为强 LLM 会“补偿”差的视觉特征，降低 ViT 学视觉的压力。

### 1.2 当前 VIVID-Med 的问题

当前 VIVID-Med 旧目标多半是：

```json
{
  "Cardiomegaly": {"state": "present"},
  "Pleural Effusion": {"state": "absent"},
  "Pneumothorax": {"state": "null"}
}
```

这个目标有医学结构，但视觉压力不够强：

| 问题 | 具体表现 | 后果 |
|---|---|---|
| 太像分类 | finding -> state | no-LM head 就能学得很强 |
| 模板 token 太多 | JSON key、字段顺序、括号、固定词 | LLM loss 很多在学格式 |
| 缺少视觉证据 | 没问 where / why / which side / evidence | ViT 不一定学局部医学证据 |
| null/absent 容易混 | 未提及不等于阴性 | 需要 answerability semantics |
| paraphrase 不稳 | 换 key/order/paraphrase NLL 变差 | 说明更像 fixed serialization |

所以新 proposal 要把“语言监督”改造成“视觉依赖监督”。

---

## 2. 新 Proposal 名字候选

### 最推荐标题

```text
Clinical Evidence Instruction Pretraining for Deployable Chest X-ray ViTs
```

中文：

```text
面向可部署胸片 ViT 的临床证据型视觉指令预训练
```

### 稍微更稳的标题

```text
Report-Grounded Clinical Instruction Supervision for Chest X-ray Representation Learning
```

### 如果想保留 VIVID-Med

```text
VIVID-Med: Clinical Evidence Instructions for Deployable Chest X-ray Vision Transformers
```

### 一句话 abstract 版本

```text
We convert chest X-ray reports into clinically grounded visual instructions that ask about findings, evidence, laterality, uncertainty, answerability, and report-image consistency. This turns language supervision from fixed schema prediction into visual-dependence supervision, enabling a deployable ViT to learn clinically meaningful CXR representations without requiring an LLM at deployment.
```

---

## 3. 核心贡献设计

### Contribution 1：Clinical Evidence Instruction Data

把每个 CXR report / UMS schema 转成多条 instruction：

```text
image + question -> short clinically grounded answer
```

不是 report generation，也不是 fixed JSON，而是多类型医学问题。

### Contribution 2：Visual-Dependency-Aware Supervision

给每条 instruction 标注：

- finding
- state
- answerability
- uncertainty
- laterality
- location
- severity
- evidence phrase
- visual_dependency: high / medium / low
- answer_type

训练时对真正需要看图的 token / answer type 加高权重。

### Contribution 3：Image-Report Counterfactual Grounding

构造 hard negative：

- present/absent flip
- left/right flip
- uncertain/definite flip
- wrong finding flip
- wrong image-report pair

用来测试/训练模型是否真的依赖图像和医学语义。

### Contribution 4：Controlled LLM Necessity Analysis

比较：

- fixed JSON schema
- label-to-QA
- report-grounded QA
- QA + counterfactual
- no-LM schema heads
- frozen-LM / small LLM / random-LM / GLM-generated data

目标不是硬说 LLM 一定赢，而是证明：

> 只有当 instruction 真的视觉依赖时，语言监督才会更像 ViTP 那样教 ViT 学视觉。

---

## 4. 数据改写总流程

### 4.1 输入

每个样本建议包含：

```json
{
  "sample_id": "...",
  "image_path": "...",
  "report": "...",
  "findings_section": "...",
  "impression_section": "...",
  "ums_schema": {
    "Cardiomegaly": {"state": "present", "answerable": true, "uncertain": false},
    "Pneumothorax": {"state": "absent", "answerable": true, "uncertain": false},
    "Fracture": {"state": "null", "answerable": false, "uncertain": null}
  },
  "chexpert_labels": {...}
}
```

如果当前没有原始 report，只能先做 V1 label-to-QA；如果有 report，主线一定要做 V2/V3。

### 4.2 输出

每张图生成多条 instruction record：

```json
{
  "instruction_id": "sample123_q004",
  "sample_id": "sample123",
  "image_path": "...",
  "question": "Is there visual evidence of cardiomegaly?",
  "answer": "Yes. The report describes an enlarged cardiac silhouette.",
  "finding": "Cardiomegaly",
  "state": "present",
  "answerability": "answerable",
  "uncertainty": "definite",
  "laterality": null,
  "location": null,
  "severity": null,
  "evidence_phrase": "enlarged cardiac silhouette",
  "evidence_source": "report_substring",
  "answer_type": "finding_verification",
  "visual_dependency": "medium",
  "counterfactual_type": null,
  "quality_flags": []
}
```

---

## 5. Instruction 类型设计

### Type A：Finding Verification QA

目标：从 fixed state 变成自然医学问答，但仍保留结构。

#### 例子

```text
Q: Is there visual evidence of cardiomegaly in this chest X-ray?
A: Yes. The report describes an enlarged cardiac silhouette.
```

```text
Q: Is pneumothorax definitely present, absent, uncertain, or not mentioned?
A: Pneumothorax is absent. The report explicitly states no pneumothorax.
```

#### 适用 finding

- Cardiomegaly
- Pleural Effusion
- Pneumothorax
- Edema
- Atelectasis
- Consolidation
- Lung Opacity
- Support Devices

#### 生成条件

| 条件 | 是否生成 |
|---|---|
| report 明确 positive | 是 |
| report 明确 negative | 是 |
| report 明确 uncertain | 是 |
| schema 是 null / unmentioned | 谨慎，只生成 answerability 问题，不生成 absent |

#### 风险

它仍然比较像分类，所以不能作为唯一主版本。

---

### Type B：Evidence Phrase QA

目标：让答案包含报告里的医学证据短语，减少纯标签化。

#### 例子

```text
Q: What report evidence supports pleural effusion?
A: The report mentions a small left pleural effusion.
```

```text
Q: What observation supports pulmonary edema?
A: The report describes bilateral interstitial opacities compatible with edema.
```

#### 生成条件

- report 中能提取 evidence phrase。
- evidence phrase 最好是原文 substring。
- 不能让 GLM 自己编一个影像征象。

#### 字段

| 字段 | 说明 |
|---|---|
| evidence_phrase | 必须尽量是 report 原文片段 |
| evidence_source | report_substring / normalized_report / llm_inferred |
| visual_dependency | medium/high |

---

### Type C：Laterality / Location QA

目标：让任务更视觉化，学习 left/right/bilateral/anatomy。

#### 例子

```text
Q: Which side has pleural effusion?
A: The pleural effusion is on the left side.
```

```text
Q: Where is the opacity described?
A: The opacity is described in the right lower lung zone.
```

```text
Q: Is the support device located on the left, right, or midline?
A: The support device is described in a midline position.
```

#### 允许位置词表

| 类别 | 词表 |
|---|---|
| laterality | left, right, bilateral, unilateral, midline |
| lung zone | upper, mid, lower, apical, basilar, perihilar, retrocardiac |
| anatomy | heart, mediastinum, pleura, lung base, hemithorax, diaphragm |
| device | endotracheal tube, central venous catheter, chest tube, pacemaker, enteric tube |

#### 生成条件

- report 中明确出现 laterality/location。
- 不从 CheXpert label 硬造 location。
- 没有 location 就不生成 Type C。

---

### Type D：Severity QA

目标：学习 mild/moderate/severe/small/large 等临床程度。

#### 例子

```text
Q: How severe is the pleural effusion?
A: It is described as small.
```

```text
Q: Is the cardiomegaly mild, moderate, or severe?
A: The report describes mild cardiomegaly.
```

#### severity 词表

| finding | severity candidates |
|---|---|
| Pleural Effusion | trace, small, moderate, large |
| Pneumothorax | trace, small, moderate, large |
| Edema | mild, moderate, severe |
| Cardiomegaly | mild, moderate, severe, enlarged |
| Atelectasis | mild, subsegmental, bibasilar, extensive |

#### 生成条件

- report 明确写 severity。
- 不要让 GLM 推断 severity。

---

### Type E：Uncertainty QA

目标：保留 radiology uncertainty，不把 uncertain 当 negative。

#### 例子

```text
Q: Is the pleural effusion definite or uncertain?
A: It is uncertain. The report says possible small pleural effusion.
```

```text
Q: Is pneumonia definitely present, definitely absent, or uncertain?
A: It is uncertain because the report describes possible pneumonia.
```

#### uncertainty marker

| marker | 例子 |
|---|---|
| possible | possible pneumonia |
| may represent | opacity may represent atelectasis |
| cannot exclude | cannot exclude small pneumothorax |
| likely | likely atelectasis |
| questionable | questionable edema |
| suspicious for | suspicious for consolidation |

---

### Type F：Answerability QA

目标：明确 “unmentioned/null 不等于 absent”。

#### 例子

```text
Q: Is fracture answerable from the report for this image?
A: Not answerable. The report does not mention fracture.
```

```text
Q: Does the report explicitly answer whether pneumothorax is present?
A: Yes. It explicitly states there is no pneumothorax.
```

#### 注意

这类问题是你的医学特色，不是普通 ViTP。

| schema state | answerability QA 答案 |
|---|---|
| present | answerable |
| absent with explicit negation | answerable |
| uncertain | answerable but uncertain |
| null/unmentioned | not answerable / unmentioned |

---

### Type G：Image-Report Consistency QA

目标：让模型学习图像和报告陈述是否一致。

如果只有 text-only GLM，先生成 report-grounded consistency；如果有 GLM-V / Qwen-VL，可进一步视觉验证。

#### 例子

```text
Q: Does the report support the statement: "There is a left pleural effusion"?
A: Supported. The report mentions a small left pleural effusion.
```

```text
Q: Does the report support the statement: "There is a right pleural effusion"?
A: Not supported. The report mentions the effusion on the left side, not the right side.
```

#### 扩展成 image consistency

```text
Q: Does this image-report pair support the statement: "There is a left pleural effusion"?
A: Supported by the report and expected to be visually grounded in the left pleural region.
```

更严格版本需要 VLM 或人工 audit。

---

### Type H：Counterfactual Choice QA

目标：固定模板不变，只改变医学事实，逼模型区分具体视觉/医学内容。

#### 例子

```text
Q: Which statement is better supported by the report?
A. There is a left pleural effusion.
B. There is a right pleural effusion.
C. There is a large pneumothorax.
D. This finding is not mentioned.
A: A. There is a left pleural effusion.
```

```text
Q: Which statement is most consistent with the report?
A. Pneumothorax is present.
B. Pneumothorax is absent.
C. Pneumothorax is uncertain.
D. Pneumothorax is not mentioned.
A: B. Pneumothorax is absent.
```

#### Counterfactual 类型

| 类型 | 正例 | 反事实 |
|---|---|---|
| state_flip | present | absent |
| uncertainty_flip | uncertain | definite |
| laterality_flip | left | right |
| severity_flip | small | large |
| finding_swap | effusion | pneumothorax |
| answerability_flip | mentioned | not mentioned |
| image_swap | image A + report A | image B + report A |

---

### Type I：Comparison / Temporal QA

如果 report 有 prior comparison，可以生成。

#### 例子

```text
Q: Has the pleural effusion improved, worsened, or remained unchanged compared with the prior study?
A: It is unchanged compared with the prior study.
```

#### 注意

只有 report 明确出现：

- unchanged
- improved
- worsened
- interval increase
- stable
- resolved

才生成。

---

### Type J：Device QA

Support device 是 CXR 里很视觉化，且容易定位。

#### 例子

```text
Q: What support device is visible or described?
A: An endotracheal tube is present.
```

```text
Q: Is the enteric tube position satisfactory?
A: The report states the enteric tube tip projects below the diaphragm.
```

#### 支持 device types

- endotracheal tube
- enteric tube
- central venous catheter
- chest tube
- pacemaker / ICD
- Swan-Ganz catheter
- tracheostomy tube

---

## 6. GLM 改写 Prompt 设计

### 6.1 主生成 Prompt

```text
You are a radiology instruction-data generator for chest X-ray representation learning.

You will receive:
1. A chest X-ray report.
2. A structured finding schema with states: present, absent, uncertain, null.
3. A fixed list of target findings.

Your task:
Generate clinically faithful visual instruction examples for training a CXR vision model.

Important rules:
- Do not invent findings, locations, severity, or devices not supported by the report.
- If a finding is not mentioned, do NOT label it as absent.
- For unmentioned findings, only generate answerability questions.
- Prefer questions that require medical visual evidence: finding state, laterality, location, severity, uncertainty, support devices, or report-statement consistency.
- Extract evidence_phrase from the report whenever possible.
- Keep answers short and structured.
- Avoid generic report generation.
- Avoid long explanations.
- Output JSON only.

Allowed answer_type:
- finding_verification
- evidence_phrase
- laterality_location
- severity
- uncertainty
- answerability
- report_consistency
- counterfactual_choice
- device_position
- temporal_comparison

Allowed visual_dependency:
- high: location/laterality/device/severity/counterfactual that should depend strongly on visual evidence
- medium: finding state or uncertainty with report evidence
- low: report-only answerability or administrative content

Output format:
{
  "sample_id": "...",
  "instructions": [
    {
      "question": "...",
      "answer": "...",
      "finding": "...",
      "state": "present|absent|uncertain|null",
      "answerability": "answerable|not_answerable",
      "uncertainty": "definite|uncertain|null",
      "laterality": "left|right|bilateral|midline|null",
      "location": "... or null",
      "severity": "... or null",
      "evidence_phrase": "exact report substring if possible, else null",
      "evidence_source": "report_substring|normalized_report|llm_inferred|null",
      "answer_type": "...",
      "visual_dependency": "high|medium|low",
      "quality_flags": []
    }
  ]
}
```

### 6.2 Counterfactual 生成 Prompt

```text
You are generating hard counterfactual medical QA pairs for chest X-ray representation learning.

Input:
- A report-grounded instruction record.
- Its correct answer.
- The original report.

Task:
Create one or more counterfactual choices by minimally changing one clinical fact.

Allowed counterfactuals:
- present <-> absent
- definite <-> uncertain
- left <-> right
- small <-> large
- finding swap within CXR findings
- mentioned <-> not mentioned

Rules:
- Do not create medically impossible or nonsensical choices.
- Do not change more than one clinical factor at a time.
- The correct option must remain directly supported by the report.
- Output multiple-choice QA.

Output JSON:
{
  "question": "Which statement is best supported by the report?",
  "options": {
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "..."
  },
  "correct_option": "A|B|C|D",
  "counterfactual_type": "state_flip|laterality_flip|severity_flip|uncertainty_flip|finding_swap|answerability_flip",
  "rationale": "short report-grounded reason"
}
```

### 6.3 GLM 自检 Prompt

```text
You are a strict radiology QA validator.

Given:
- the original report
- one generated instruction record

Check:
1. Is the answer supported by the report?
2. Is evidence_phrase an exact substring or faithful normalization?
3. Does the instruction hallucinate location/severity/device?
4. If the finding is unmentioned, does it avoid calling it absent?
5. Is the question clinically meaningful for CXR?
6. Is visual_dependency correctly assigned?

Return JSON:
{
  "valid": true/false,
  "errors": ["..."],
  "corrected_record": {... or null},
  "confidence": 0.0-1.0
}
```

---

## 7. 生成数据的质量控制

### 7.1 自动过滤规则

| 过滤项 | 规则 | 处理 |
|---|---|---|
| evidence substring | evidence_phrase 必须能在 report 中找到，或标为 normalized_report | 找不到且非 normalized 则丢弃 |
| null-as-negative | schema null 不能生成 absent | 丢弃或改为 answerability QA |
| location hallucination | report 没 left/right，不允许生成 laterality | 丢弃 |
| severity hallucination | report 没 mild/small/large，不允许生成 severity | 丢弃 |
| answer length | answer 5-40 tokens | 太长截断或丢弃 |
| question length | question 5-40 tokens | 太长改写 |
| forbidden generic | “Describe the image” 占比不能过高 | 限制比例 |
| duplicate QA | 同一样本同 finding 同 answer_type 去重 | 去重 |

### 7.2 数据平衡

| 维度 | 目标 |
|---|---|
| per image QA 数 | 3-8 条 |
| positive/negative/uncertain/null | 不让 absent/null 占绝对多数 |
| answer_type | finding verification 不超过 40% |
| visual_dependency | high+medium 至少 70% |
| rare finding | 过采样 rare/high-null fields |
| uncertainty | 过采样 uncertain examples |
| location/severity | 有则优先生成 |

### 7.3 人工 audit

每个 GLM prompt 版本抽样检查：

| audit 项 | 样本数 | 通过标准 |
|---|---:|---|
| report support | 200 | >= 95% |
| no hallucinated location | 100 with location | >= 95% |
| null not absent | 100 null examples | >= 98% |
| counterfactual correctness | 200 | >= 90% |
| clinical meaningfulness | 200 | >= 90% |

---

## 8. 训练数据版本设计

### V0：Fixed JSON Schema Baseline

旧方法。

```text
image + fixed prompt -> UMS JSON
```

目的：证明新 instruction 不是只靠 schema。

---

### V1：Label-to-QA Paraphrase

只用 UMS label 改写成 QA，不用 report evidence。

```text
Q: Is pleural effusion present?
A: Pleural effusion is present.
```

目的：测试“只把 JSON 换成自然语言”是否足够。预计提升有限。

---

### V2：Report-Grounded Evidence QA

用 GLM 从 report 抽 evidence/location/severity/uncertainty。

```text
Q: What evidence supports pleural effusion?
A: The report mentions a small left pleural effusion.
```

主版本。

---

### V3：Report-Grounded QA + Counterfactual

在 V2 基础上加入反事实选择题。

```text
Q: Which statement is best supported?
A. left pleural effusion
B. right pleural effusion
C. large pneumothorax
D. not mentioned
A: A
```

目的：强制 visual/text semantic discrimination。

---

### V4：Visual-Dependent Token Weighting

对答案 token 加权：

| token 类型 | 权重 |
|---|---:|
| present/absent/uncertain/not answerable | 2.0 |
| left/right/bilateral/location | 3.0 |
| severity | 2.5 |
| finding name | 2.0 |
| generic phrase | 0.5 |
| punctuation/template | 0.1 |

目的：避免 loss 被模板 token 吃掉。

---

### V5：Image-Report Counterfactual Margin

训练时加入：

```text
L = NLL(correct answer) + λ * max(0, margin + NLL(correct) - NLL(counterfactual))
```

或者：

```text
score(image, correct_QA) > score(image, wrong_QA)
```

目的：让模型真的区分医学事实。

---

### V6：Question-Only / Image-Shuffled Controls

诊断模型有没有偷懒。

| control | 说明 | 如果性能高说明什么 |
|---|---|---|
| question-only | 不给图，只给 question | 语言 prior 太强 |
| report-only | 不给图，只给 report/schema | 训练目标不视觉依赖 |
| image-shuffled | 图像和 QA 错配 | 模型没依赖图像 |
| no-image prefix | visual prefix zero | LLM 可自己猜答案 |

---

### V7：GLM-V / Qwen-VL Visual Verifier Optional

如果有视觉模型 API，可以做二阶段过滤：

```text
report-grounded QA -> VLM 看图判断是否 visually plausible -> filter or reweight
```

但这不是第一优先级，因为会增加 API 成本和噪声。

---

### V8：Small Decoder / Frozen Decoder / Trainable Adapter

测试 ViTP 里“decoder 太强反而不好”的现象。

| decoder | 目的 |
|---|---|
| Qwen 1.5B frozen | 当前路线 |
| small Qwen / tiny decoder | 提高视觉压力 |
| random same architecture | architecture control |
| trainable adapter / LoRA | 让 teacher 适配医学 QA |
| no-LM head | schema supervision lower bound |

---

## 9. 模型结构候选

### Model A：Current Frozen-LM Visual Prefix

```text
image -> ViT -> visual prefix -> frozen LLM + question -> answer
```

优点：延续现有代码。  
缺点：LLM 可能靠文本 prior。

---

### Model B：Frozen-LM + Visual-Token Drop

仿照 VRL，但不要照搬。

```text
image tokens -> randomly drop 50/75/90% -> LLM answer
```

实验：

| drop ratio | 说明 |
|---:|---|
| 0.0 | no drop |
| 0.25 | mild |
| 0.50 | medium |
| 0.75 | ViTP-inspired |
| 0.90 | stress test |

成功标准：

- counterfactual accuracy 提升；
- AUC 不明显下降；
- query paraphrase robustness 提升。

---

### Model C：Dual Loss

```text
L = L_instruction + α L_state_head + β L_answerability_head + γ L_uncertainty_head
```

优点：保留 no-LM schema 的稳定性，同时引入 instruction。

---

### Model D：Instruction Encoder + Classification Head

不用 LLM decoder 每步训练，而是：

```text
image -> ViT
instruction text -> text encoder
contrast image-text / QA matching
```

适合低成本版本。

---

### Model E：Prototype Semantic Space

```text
finding-state text -> GLM/Qwen embedding -> prototypes
image field embedding -> match prototype
```

优点：保留语言语义，不用大 LLM 每步生成。

---

## 10. 实验总路线

### Phase A：Instruction 数据构建

目标：证明 GLM 生成的数据干净、医学可信。

| Task ID | 任务 | 输入 | 输出 | 成功标准 |
|---|---|---|---|---|
| A1 | GLM prompt v0 试跑 100 reports | reports + UMS | qa_raw_v0.jsonl | JSON parse >= 95% |
| A2 | 自动过滤器 | qa_raw_v0 | qa_filtered_v0 | invalid <= 10% |
| A3 | 人工 audit 200 条 | qa_filtered_v0 | audit.csv | report-supported >= 90% |
| A4 | prompt v1 修正 | audit errors | qa_filtered_v1 | report-supported >= 95% |
| A5 | 全量生成 | all reports | instruction_full.jsonl | 每图 3-8 QA |
| A6 | 数据统计 | instruction_full | instruction_stats.md | 分布可解释 |

---

### Phase B：最小训练闭环

先别全量烧卡，跑 1k/3k。

| Task ID | 方法 | 数据 | 训练步数 | 输出 | 目的 |
|---|---|---:|---:|---|---|
| B0 | fixed JSON V0 | 1k | 5k/10k | baseline | 旧方法 |
| B1 | label-to-QA V1 | 1k | 5k/10k | QA baseline | 看自然语言化是否有效 |
| B2 | evidence QA V2 | 1k | 5k/10k | main | 看 evidence 是否有效 |
| B3 | evidence QA + CF V3 | 1k | 5k/10k | main+CF | 看视觉依赖是否增强 |
| B4 | V2 + token weighting | 1k | 5k/10k | ablation | 防止模板 loss |
| B5 | V3 + margin | 1k | 5k/10k | ablation | 反事实 grounding |

---

### Phase C：正式主实验

选择 Phase B 最强 2-3 个方法跑 full / 30k。

| Task ID | 方法 | 数据 | 对照 | 指标 |
|---|---|---:|---|---|
| C1 | best instruction | 30k | BCE / UMS / free-text | CheXpert/NIH |
| C2 | best instruction + CF | 30k | best instruction | visual dependence |
| C3 | no-LM schema head | 30k | best instruction | LLM necessity |
| C4 | random-LM decoder | 30k | frozen-LM | pretrained control |
| C5 | small decoder | 30k | Qwen 1.5B | decoder capacity |

---

### Phase D：诊断实验

| Task ID | 诊断 | 目的 |
|---|---|
| D1 question-only | 证明模型不能只靠问题/语言 prior |
| D2 image-shuffle | 证明模型依赖正确图像 |
| D3 counterfactual pairwise acc | 证明医学事实敏感 |
| D4 paraphrase robustness | 证明不只背固定模板 |
| D5 visual token drop | 测试 ViTP-style 视觉鲁棒学习 |
| D6 rare/high-null/uncertain groups | 看医学难例是否提升 |
| D7 manual failure audit | 找方法真正改进点 |

---

## 11. 训练/评估指标

### 11.1 下游分类指标

| metric | 说明 |
|---|---|
| CheXpert macro-AUC | 主指标 |
| CheXpert macro-F1 | 辅助 |
| NIH macro-AUC | 外部迁移 |
| per-field AUC | 看具体 finding |
| group AUC | rare/common/high-null/uncertain-heavy |

### 11.2 Instruction 指标

| metric | 说明 |
|---|---|
| state accuracy | present/absent/uncertain/null |
| answerability AUC/F1 | 是否 answerable |
| uncertainty AUC/F1 | 是否 uncertain |
| laterality accuracy | left/right/bilateral |
| severity accuracy | mild/moderate/severe/small/large |
| evidence phrase exact/soft match | answer 是否保留 evidence |
| counterfactual choice accuracy | 选择题正确率 |

### 11.3 视觉依赖指标

| metric | 说明 | 成功标准建议 |
|---|---|---|
| image-shuffle drop | 正确图 vs 错图性能差 | drop >= 10% |
| no-image degradation | 去掉图后性能下降 | drop >= 10% |
| counterfactual NLL gap | 正确答案 NLL 更低 | gap > 0 |
| pairwise counterfactual acc | correct > wrong | >= 70% 起步 |
| question-only baseline | 只给问题的性能 | 应显著低于 image+question |
| paraphrase robustness | 问法改写后性能保持 | drop <= 5% |

### 11.4 成本指标

| metric | 说明 |
|---|---|
| GPU-hours | 训练成本 |
| peak memory | 显存 |
| throughput | images/sec or steps/sec |
| trainable params | 训练参数 |
| frozen params | frozen LLM 参数 |
| deployment params | 部署模型大小 |
| deployment LLM? | 必须为 no |

---

## 12. 空实验表格

### 12.1 GLM Prompt 版本表

| Prompt ID | Date | GLM model | Input fields | QA types enabled | Counterfactual? | Validation prompt? | JSON parse rate | Filter pass rate | Manual support rate | Notes |
|---|---|---|---|---|---|---|---:|---:|---:|---|
| P0 |  |  | report + UMS | A/F | no | no |  |  |  |  |
| P1 |  |  | report + UMS | A/B/E/F | no | yes |  |  |  |  |
| P2 |  |  | report + UMS | A/B/C/D/E/F | no | yes |  |  |  |  |
| P3 |  |  | report + UMS | A/B/C/D/E/F/H | yes | yes |  |  |  |  |

### 12.2 Instruction 数据统计表

| Dataset version | #images | #QA | QA/image | finding_verification % | evidence % | location % | severity % | uncertainty % | answerability % | counterfactual % | high visual dep % | medium visual dep % | low visual dep % | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| V1 label-to-QA |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| V2 report-grounded |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| V3 report-grounded+CF |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

### 12.3 数据质量人工 Audit 表

| Sample size | Dataset version | Report-supported % | Hallucinated finding % | Hallucinated location % | Null-as-absent error % | Counterfactual correct % | Clinically meaningful % | Major errors | Decision |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 100 | V1 |  |  |  |  |  |  |  | keep/drop/fix |
| 200 | V2 |  |  |  |  |  |  |  | keep/drop/fix |
| 200 | V3 |  |  |  |  |  |  |  | keep/drop/fix |

### 12.4 训练 Run 表

| Run ID | Method | Dataset version | Model | Decoder | Frozen LLM? | Token weighting | Counterfactual loss | Token drop | Data size | Steps | Output dir | Status | Notes |
|---|---|---|---|---|---|---|---|---:|---:|---:|---|---|---|
| R0 | fixed JSON | V0 | ViT-B | Qwen 1.5B | yes | no | no | 0 | 1k |  |  |  |  |
| R1 | label-to-QA | V1 | ViT-B | Qwen 1.5B | yes | no | no | 0 | 1k |  |  |  |
| R2 | evidence QA | V2 | ViT-B | Qwen 1.5B | yes | no | no | 0 | 1k |  |  |  |
| R3 | evidence QA + CF | V3 | ViT-B | Qwen 1.5B | yes | no | yes | 0 | 1k |  |  |  |
| R4 | V2 + token weighting | V2 | ViT-B | Qwen 1.5B | yes | yes | no | 0 | 1k |  |  |  |
| R5 | V3 + weighting + margin | V3 | ViT-B | Qwen 1.5B | yes | yes | yes | 0 | 1k |  |  |  |
| R6 | no-LM schema head | UMS | ViT-B | none | no | n/a | no | 0 | 1k |  |  |  |
| R7 | random-LM QA | V2 | ViT-B | random Qwen | yes | yes | no | 0 | 1k |  |  |  |
| R8 | small decoder QA | V2 | ViT-B | small Qwen | yes | yes | no | 0 | 1k |  |  |  |

### 12.5 主结果表

| Method | Data | Supervision | Decoder | CheXpert AUC | CheXpert F1 | NIH AUC | NIH F1 | Rare AUC | High-null AUC | Uncertain-heavy AUC | Visual-dep acc | Cost GPU-h | Decision |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BCE |  | flat label | none |  |  |  |  |  |  |  |  |  | baseline |
| fixed JSON |  | UMS JSON | Qwen frozen |  |  |  |  |  |  |  |  |  | baseline |
| label-to-QA |  | label QA | Qwen frozen |  |  |  |  |  |  |  |  |  | keep/drop |
| report-grounded QA |  | evidence QA | Qwen frozen |  |  |  |  |  |  |  |  |  | keep/drop |
| QA + counterfactual |  | evidence+CF | Qwen frozen |  |  |  |  |  |  |  |  |  | keep/drop |
| QA + token weighting |  | weighted evidence QA | Qwen frozen |  |  |  |  |  |  |  |  |  | keep/drop |
| no-LM schema head |  | schema heads | none |  |  |  |  |  |  |  |  |  | comparator |

### 12.6 Visual-Dependence 诊断表

| Method | Question-only AUC/Acc | No-image AUC/Acc | Correct image acc | Image-shuffled acc | Shuffle drop | Counterfactual pair acc | NLL gap | Paraphrase drop | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fixed JSON |  |  |  |  |  |  |  |  | template-like? |
| label-to-QA |  |  |  |  |  |  |  |  |  |
| evidence QA |  |  |  |  |  |  |  |  |  |
| QA+CF |  |  |  |  |  |  |  |  |  |
| QA+CF+weighting |  |  |  |  |  |  |  |  |  |

### 12.7 QA 类型消融表

| QA types included | AUC | F1 | Counterfactual acc | Location acc | Answerability AUC | Uncertainty AUC | Paraphrase drop | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A only |  |  |  |  |  |  |  |  |
| A+B |  |  |  |  |  |  |  |  |
| A+B+C |  |  |  |  |  |  |  |  |
| A+B+C+D |  |  |  |  |  |  |  |  |
| A+B+C+D+E+F |  |  |  |  |  |  |  |  |
| A+B+C+D+E+F+H |  |  |  |  |  |  |  |  |

### 12.8 Token Weighting 消融表

| Weighting scheme | Clinical token weight | Template token weight | AUC | F1 | Counterfactual acc | NLL gap | Paraphrase drop | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| uniform | 1.0 | 1.0 |  |  |  |  |  |  |
| mild | 2.0 | 0.5 |  |  |  |  |  |  |
| strong | 3.0 | 0.1 |  |  |  |  |  |  |
| answer-only | 1.0 | 0.0 |  |  |  |  |  |  |

### 12.9 Counterfactual 类型表

| CF type | #QA | Pair acc | NLL gap | Most failed findings | Helps AUC? | Interpretation |
|---|---:|---:|---:|---|---|---|
| state_flip |  |  |  |  |  |  |
| laterality_flip |  |  |  |  |  |  |
| uncertainty_flip |  |  |  |  |  |  |
| severity_flip |  |  |  |  |  |  |
| finding_swap |  |  |  |  |  |  |
| image_swap |  |  |  |  |  |  |

### 12.10 Decoder / LLM 对照表

| Decoder | Params | Frozen? | Trainable adapter? | AUC | F1 | Counterfactual acc | Question-only gap | GPU-h | Interpretation |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| no decoder / no-LM | 0 | n/a | n/a |  |  |  |  |  |  |
| random-LM |  | yes | no |  |  |  |  |  | architecture control |
| Qwen 0.5B |  | yes | no |  |  |  |  |  | small teacher |
| Qwen 1.5B |  | yes | no |  |  |  |  |  | current teacher |
| Qwen 1.5B + LoRA |  | partial | yes |  |  |  |  |  | adapter |
| GLM / other |  | yes | no |  |  |  |  |  | model family |

### 12.11 Field Group 表

| Field group | BCE | fixed JSON | evidence QA | QA+CF | no-LM schema | Best method | Interpretation |
|---|---:|---:|---:|---:|---:|---|---|
| common |  |  |  |  |  |  |  |
| rare |  |  |  |  |  |  |  |
| high-null |  |  |  |  |  |  |  |
| uncertain-heavy |  |  |  |  |  |  |  |
| device-related |  |  |  |  |  |  |  |
| location-rich |  |  |  |  |  |  |  |

### 12.12 成本表

| Method | Training LLM? | Frozen params | Trainable params | Peak memory | GPU-hours | Throughput | Deployment model | Deployment LLM? |
|---|---|---:|---:|---:|---:|---:|---|---|
| BCE | no | 0 |  |  |  |  | ViT | no |
| no-LM schema | no | 0 |  |  |  |  | ViT | no |
| fixed JSON frozen-LM | yes |  |  |  |  |  | ViT | no |
| evidence QA frozen-LM | yes |  |  |  |  |  | ViT | no |
| QA+CF+weighting | yes |  |  |  |  |  | ViT | no |
| prototype | no/optional |  |  |  |  |  | ViT | no |

---

## 13. 成功/失败判断标准

### 13.1 新 instruction 方法成功

至少满足以下之一：

| 成功类型 | 判据 |
|---|---|
| 下游提升 | CheXpert/NIH AUC 比 fixed JSON 或 no-LM schema 高 >= 0.005 |
| 难例提升 | rare/high-null/uncertain group AUC 高 >= 0.01 |
| 视觉依赖增强 | image-shuffle drop >= 10% 且 question-only 明显低 |
| 反事实增强 | counterfactual pairwise acc 提升 >= 5% |
| 鲁棒性增强 | paraphrase robustness 明显好于 fixed JSON |
| 成本可接受 | 不比旧 frozen-LM 高太多，部署仍无 LLM |

### 13.2 新 instruction 方法失败

满足以下任意一项：

| 失败类型 | 判据 |
|---|---|
| 只学语言 prior | question-only / no-image 接近 image+question |
| 只学模板 | paraphrase 或 counterfactual 崩 |
| GLM 数据噪声大 | manual support rate < 90% |
| AUC 明显下降 | 下降 > 0.01 且没有诊断收益 |
| 成本过高 | GPU-hours 大幅增加但性能/诊断无收益 |
| null 语义破坏 | null 被大量变成 absent |

---

## 14. 代码实现任务清单

### G0：数据格式定义

```text
Task ID: G0_DEFINE_INSTRUCTION_SCHEMA

Goal:
  Define JSONL schema for clinical evidence instructions.

Outputs:
  data/instructions/schema/clinical_instruction_schema.json
  docs/clinical_instruction_schema.md

Fields:
  instruction_id, sample_id, image_path, report, question, answer,
  finding, state, answerability, uncertainty, laterality, location, severity,
  evidence_phrase, evidence_source, answer_type, visual_dependency,
  counterfactual_type, quality_flags.
```

### G1：GLM 生成器

```text
Task ID: G1_GLM_GENERATE_INSTRUCTIONS

Goal:
  Use GLM API to generate instruction candidates from report + UMS.

Inputs:
  data/dataset/processed/*_ums.jsonl
  report field or raw report lookup
  prompts/glm_instruction_generation_v*.txt

Outputs:
  data/instructions/raw/glm_v*/train.jsonl
  data/instructions/raw/glm_v*/val.jsonl
  outputs/instruction_generation/glm_v*/api_log.jsonl
  outputs/instruction_generation/glm_v*/parse_errors.jsonl

Success:
  JSON parse success >= 95%.
  No silent skip.
```

### G2：自动过滤器

```text
Task ID: G2_FILTER_INSTRUCTIONS

Goal:
  Filter hallucinated or low-quality instruction records.

Rules:
  - evidence substring check
  - allowed finding/state/location/severity vocabulary
  - no null-as-negative
  - answer length limits
  - duplicate removal
  - visual_dependency consistency

Outputs:
  data/instructions/filtered/glm_v*/train.jsonl
  data/instructions/filtered/glm_v*/val.jsonl
  outputs/instruction_generation/glm_v*/filter_report.md
```

### G3：数据统计

```text
Task ID: G3_INSTRUCTION_STATS

Goal:
  Compute instruction data statistics.

Metrics:
  - #QA/image
  - answer_type distribution
  - finding distribution
  - state distribution
  - visual_dependency distribution
  - evidence substring rate
  - location/severity coverage
  - counterfactual distribution

Outputs:
  outputs/final_tables/instruction_dataset_stats.csv
  outputs/final_tables/instruction_dataset_stats.md
```

### G4：人工 audit 表生成

```text
Task ID: G4_PREPARE_MANUAL_AUDIT

Goal:
  Sample instruction records for manual audit.

Outputs:
  outputs/audit/instruction_audit_sample_200.csv
  outputs/audit/instruction_audit_sample_500.csv

Columns:
  instruction_id, report, question, answer, finding, state,
  evidence_phrase, answer_type, visual_dependency,
  valid_report_supported, hallucinated_location, null_as_absent_error,
  clinically_meaningful, notes.
```

### G5：Instruction Dataset Loader

```text
Task ID: G5_INSTRUCTION_DATALOADER

Goal:
  Add dataset class for image + question -> answer instruction training.

Inputs:
  data/instructions/filtered/glm_v*/train.jsonl

Outputs:
  data/cxr_instruction_dataset.py

Requirements:
  - image loading compatible with current CXR pipeline
  - multiple QA per image
  - balanced sampler by answer_type/finding
  - batch fields: image, question, answer, token_weight_mask, metadata
```

### G6：Instruction Trainer

```text
Task ID: G6_INSTRUCTION_TRAINER

Goal:
  Train ViT visual prefix with frozen/small LLM decoder on instruction QA.

Features:
  - answer-only loss
  - optional clinical token weighting
  - optional visual token drop
  - optional counterfactual margin
  - logs token-level NLL by token type

Outputs:
  scripts/train_cxr_instruction.py
  outputs/instruction_runs/<run_id>/
```

### G7：Visual Dependence Evaluator

```text
Task ID: G7_VISUAL_DEPENDENCE_EVAL

Goal:
  Evaluate whether model depends on image.

Diagnostics:
  - correct image vs image-shuffled
  - image+question vs question-only
  - correct answer vs counterfactual answer NLL
  - paraphrased question robustness
  - per-answer_type accuracy

Outputs:
  outputs/final_tables/visual_dependence_results.csv
  outputs/final_tables/visual_dependence_results.md
```

### G8：Downstream LP Evaluation

```text
Task ID: G8_DOWNSTREAM_LP_EVAL

Goal:
  Evaluate learned ViT representations on CheXpert/NIH.

Outputs:
  outputs/final_tables/instruction_main_results.csv
  outputs/final_tables/instruction_main_results.md
```

---

## 15. 推荐执行顺序

### Round 1：最快验证，不烧太多卡

1. 选 500-1000 张 report。
2. 用 GLM 生成 V1/V2/V3。
3. 自动过滤 + 200 条人工 audit。
4. 跑 1k debug/full：
   - V0 fixed JSON
   - V1 label-to-QA
   - V2 report-grounded QA
   - V3 report-grounded QA + counterfactual
5. 做 visual dependence eval。

目标：证明新 QA 不是纯模板。

### Round 2：主结果筛选

从 Round 1 选两个最强版本：

- evidence QA
- evidence QA + counterfactual / token weighting

跑 3k/10k。

### Round 3：正式结果

跑 30k：

- BCE
- no-LM schema
- fixed JSON
- best instruction
- best instruction + CF
- random-LM or small decoder control

### Round 4：论文诊断

- question-only
- image-shuffle
- counterfactual acc
- paraphrase robustness
- high-null/rare/uncertain field analysis
- manual failure cases
- cost table

---

## 16. 最小可跑实验矩阵

如果时间紧，至少跑这 8 个：

| Priority | Run | 数据 | 目的 |
|---:|---|---:|---|
| 1 | fixed JSON V0 | 1k | 旧 baseline |
| 2 | label-to-QA V1 | 1k | 证明自然语言化是否有效 |
| 3 | report-grounded V2 | 1k | 主方向 |
| 4 | report-grounded+CF V3 | 1k | 视觉依赖 |
| 5 | no-LM schema | 1k | LLM necessity |
| 6 | question-only eval | 1k | 检查语言 prior |
| 7 | image-shuffle eval | 1k | 检查图像依赖 |
| 8 | visual token weighting | 1k | 防模板 loss |

如果这 8 个里 V2/V3 有明显 visual-dependence 增强，就值得全量。

---

## 17. 论文故事怎么写

### 新故事

```text
Existing CXR representation learning often compresses radiology reports into flat disease labels or fixed schemas. However, such targets do not necessarily ensure that language supervision is visually grounded. We propose Clinical Evidence Instruction Pretraining, which converts radiology reports into evidence-aware visual instructions covering findings, answerability, uncertainty, laterality, severity, and image-report counterfactuals. This design increases visual dependence and reduces fixed-template learning while preserving deployable ViT-only inference.
```

### 不要写

```text
LLM semantic manifold is the dominant reason.
```

### 可以写

```text
Language supervision teaches visual representations only when the instruction target is clinically grounded and visually dependent.
```

### 和 ViTP 的区别

| ViTP | 新 VIVID-Med |
|---|---|
| general/domain visual instruction foundation model | CXR clinical evidence instruction supervision |
| 多任务 VQA/VG/caption/CLS | finding/evidence/answerability/uncertainty/laterality/counterfactual |
| 目标是 domain foundation model SOTA | 目标是 clinically faithful deployable CXR ViT |
| VRL token drop 是核心模块 | visual-dependent instruction + token weighting/counterfactual 是核心 |
| 医学只是其中一个 benchmark | 医疗报告监督语义是核心问题 |

---

## 18. 可能结果和对应结论

### 情况 A：V2/V3 AUC 和诊断都提升

结论：

> Clinical evidence instructions successfully turn language supervision into visual supervision.

主打新方法。

### 情况 B：AUC 不升，但 counterfactual/paraphrase 明显提升

结论：

> 新方法不一定提高普通分类 AUC，但显著提升 visual grounding / robustness。可作为 clinically faithful representation learning。

论文要强调诊断价值和临床语义。

### 情况 C：V1/V2/V3 都不如 no-LM

结论：

> 对当前 CXR classification，schema decomposition 比 LLM instruction 更重要。论文转成 controlled negative study。

仍有价值，但主线改弱。

### 情况 D：question-only 接近 image+question

说明：

> 数据或任务仍然太文本化，必须加强 counterfactual/image-shuffle/location/evidence。

### 情况 E：GLM 生成噪声太大

解决：

- 加自检 prompt；
- evidence 必须 substring；
- 降低生成类型；
- 只保留 high-confidence QA；
- 引入 VLM verifier 或人工小样本校正。

---

## 19. 争议点清单，后续讨论用

| 争议点 | 选择 A | 选择 B | 我的建议 |
|---|---|---|---|
| LLM 每步训练还是只离线生成数据 | 每步 frozen LLM | GLM 离线生成 + no-LM/小 decoder 训练 | 先离线生成，再比较 |
| 是否上 GLM-V/Qwen-VL | 文本 GLM only | VLM verifier | 第二阶段再加 |
| 是否继续 JSON | fixed JSON | QA/instruction | JSON 只作 metadata，不作主 target |
| 是否用 counterfactual | 不用 | 用 | 一定要用，至少诊断用 |
| 是否做 token weighting | 不用 | 用 | 强烈建议 |
| 是否做 location/severity | 不做 | 只从 report 明确抽取 | 有则做，不硬造 |
| 是否保留 no-LM schema | 不保留 | 保留 | 必须保留，防止 LLM claim 不稳 |
| 是否保留 SPD | 保留 | 不保留 | 不保留，最多 appendix |
| 是否全量先跑 | 是 | 先 1k/3k | 先 1k/3k |

---

## 20. 最终推荐路线

最推荐实验路线：

```text
Step 1: GLM 生成 V1/V2/V3 instruction 数据
Step 2: 自动过滤 + 人工 audit
Step 3: 1k 跑 fixed JSON / V1 / V2 / V3 / no-LM
Step 4: 做 question-only / image-shuffle / counterfactual acc
Step 5: 如果 V2/V3 视觉依赖明显更强，跑 3k/10k/30k
Step 6: 最终论文写成 Clinical Evidence Instruction Pretraining
```

最推荐主方法：

```text
Report-grounded QA + Counterfactual + Visual-dependent token weighting
```

最重要的判断指标：

```text
不是只看 CheXpert AUC，而是同时看：
1. NIH transfer
2. counterfactual pairwise accuracy
3. image-shuffle drop
4. question-only degradation
5. paraphrase robustness
6. rare/high-null/uncertain groups
```

---

## 21. 你跑完后发给我的结果包格式

建议你最终把这些文件发回来：

```text
outputs/final_tables/instruction_dataset_stats.md
outputs/final_tables/instruction_main_results.md
outputs/final_tables/visual_dependence_results.md
outputs/final_tables/instruction_ablation_results.md
outputs/final_tables/counterfactual_results.md
outputs/final_tables/cost_table.md
outputs/audit/instruction_audit_summary.md
outputs/failure_cases/*.csv
```

以及每个 run 的：

```text
config.yaml
metrics_final.json
metrics_step_*.json
runtime_summary.json
checkpoint path
sample predictions jsonl/csv
```

---

# Appendix A：Example GLM Input/Output

## 输入示例

```json
{
  "sample_id": "cxr_001",
  "report": "Portable chest radiograph demonstrates mild enlargement of the cardiac silhouette. Small left pleural effusion is present. No pneumothorax. Mild bibasilar atelectatic opacity.",
  "ums_schema": {
    "Cardiomegaly": {"state": "present", "answerable": true, "uncertain": false},
    "Pleural Effusion": {"state": "present", "answerable": true, "uncertain": false},
    "Pneumothorax": {"state": "absent", "answerable": true, "uncertain": false},
    "Atelectasis": {"state": "present", "answerable": true, "uncertain": false},
    "Fracture": {"state": "null", "answerable": false, "uncertain": null}
  }
}
```

## 输出示例

```json
{
  "sample_id": "cxr_001",
  "instructions": [
    {
      "question": "Is there visual evidence of cardiomegaly?",
      "answer": "Yes. The report describes mild enlargement of the cardiac silhouette.",
      "finding": "Cardiomegaly",
      "state": "present",
      "answerability": "answerable",
      "uncertainty": "definite",
      "laterality": null,
      "location": "cardiac silhouette",
      "severity": "mild",
      "evidence_phrase": "mild enlargement of the cardiac silhouette",
      "evidence_source": "report_substring",
      "answer_type": "evidence_phrase",
      "visual_dependency": "medium",
      "quality_flags": []
    },
    {
      "question": "Which side has pleural effusion?",
      "answer": "The pleural effusion is on the left side.",
      "finding": "Pleural Effusion",
      "state": "present",
      "answerability": "answerable",
      "uncertainty": "definite",
      "laterality": "left",
      "location": "pleural space",
      "severity": "small",
      "evidence_phrase": "Small left pleural effusion",
      "evidence_source": "report_substring",
      "answer_type": "laterality_location",
      "visual_dependency": "high",
      "quality_flags": []
    },
    {
      "question": "Is pneumothorax present, absent, uncertain, or not mentioned?",
      "answer": "Pneumothorax is absent. The report explicitly states no pneumothorax.",
      "finding": "Pneumothorax",
      "state": "absent",
      "answerability": "answerable",
      "uncertainty": "definite",
      "laterality": null,
      "location": null,
      "severity": null,
      "evidence_phrase": "No pneumothorax",
      "evidence_source": "report_substring",
      "answer_type": "finding_verification",
      "visual_dependency": "medium",
      "quality_flags": []
    },
    {
      "question": "Is fracture answerable from the report?",
      "answer": "Not answerable. The report does not mention fracture.",
      "finding": "Fracture",
      "state": "null",
      "answerability": "not_answerable",
      "uncertainty": null,
      "laterality": null,
      "location": null,
      "severity": null,
      "evidence_phrase": null,
      "evidence_source": null,
      "answer_type": "answerability",
      "visual_dependency": "low",
      "quality_flags": []
    },
    {
      "question": "Which statement is best supported by the report? A. There is a right pleural effusion. B. There is a left pleural effusion. C. There is a large pneumothorax. D. Fracture is clearly present.",
      "answer": "B. There is a left pleural effusion.",
      "finding": "Pleural Effusion",
      "state": "present",
      "answerability": "answerable",
      "uncertainty": "definite",
      "laterality": "left",
      "location": "pleural space",
      "severity": "small",
      "evidence_phrase": "Small left pleural effusion",
      "evidence_source": "report_substring",
      "answer_type": "counterfactual_choice",
      "visual_dependency": "high",
      "counterfactual_type": "laterality_flip+finding_swap",
      "quality_flags": []
    }
  ]
}
```

---

# Appendix B：最终论文 Claim 对照

| Claim | 当前是否建议写 | 写法 |
|---|---|---|
| LLM semantic manifold 是主要收益来源 | 不建议 | 改成 language supervision needs visual-dependent design |
| fixed JSON 已经足够 | 不建议 | fixed JSON 是 baseline，容易 template learning |
| GLM 改写报告能解决一切 | 不建议 | GLM 是 data generator，需要过滤和 audit |
| answerability 是医疗核心 | 建议 | unmentioned is not absent |
| counterfactual 是核心诊断 | 强烈建议 | 检查是否真的依赖图像/医学事实 |
| 最终部署无 LLM | 强烈建议 | LLM training-time only |
| 和 ViTP 完全一样 | 不建议 | 借鉴 instruction pretraining，但聚焦 clinical evidence faithfulness |

