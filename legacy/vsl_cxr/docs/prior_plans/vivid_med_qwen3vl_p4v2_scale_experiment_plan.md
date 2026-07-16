# VIVID-Med Qwen3-VL P4-v2 / Scale 实验执行文档

> 目的：在当前 Qwen3-VL 一套 VLM 路线已经跑通的基础上，系统验证：
>
> 1. 之前 1000-step pilot 是否训练量不足；
> 2. P4「report-grounded QA + counterfactual」为什么最好；
> 3. 如何把语言监督真正变成视觉监督；
> 4. 扩大数据量、训练步数、hard counterfactual、hard image shuffle 是否能显著提升 CXR vision tower 表征；
> 5. 最终能否形成一条稳的论文主线：**Clinical Evidence Instruction Pretraining for Deployable Chest X-ray Vision Encoders**。

---

## 0. 当前已知结果总结

### 0.1 目前 Qwen3-VL pilot 的结论

当前已经跑完的 pilot 大致说明：

| Run | 内容 | CheXpert macro-AUC | NIH macro-AUC | 当前解释 |
|---|---|---:|---:|---|
| Base | 原始 Qwen3-VL vision tower | 0.6790 | 0.5643 | 起点 |
| P2 | fixed JSON schema | 0.6662 | 0.5554 | 固定 JSON 不好，可能学模板 |
| P3 | report-grounded QA | 0.6859 | 0.5620 | 比 fixed JSON 好 |
| P4 | report-grounded QA + counterfactual | **0.6896** | **0.5683** | 当前最好 |
| P5 | QA + counterfactual + token weighting | 0.6815 | 0.5643 | F1 高，但 AUC 不如 P4 |
| P6 | data-only no-LM control | 0.6631 | 0.5264 | 目前不能解释掉 P4 |

最重要的初步结论：

```text
P4 是当前最好的 instruction route。
fixed JSON 不适合作为主方法。
Qwen3-VL-coupled route 可行，但提升还比较 modest。
```

---

### 0.2 当前最大问题

Visual-dependence diagnostics 显示：

| Run | Question-only delta | Image-shuffle delta |
|---|---:|---:|
| P2 | +0.5711 | +0.0082 |
| P3 | +1.8360 | +0.0177 |
| P4 | +1.7772 | +0.0087 |
| P5 | +1.7819 | +0.0076 |

大白话解释：

- **Question-only delta 大**：把图像拿掉，loss 变差，说明模型知道「应该有图像」。
- **Image-shuffle delta 小**：把图像换成另一张错图，loss 几乎不变，说明模型还不够在乎「是不是正确那张图」。

所以目前不能强 claim：

```text
The model learns strong image-specific clinical grounding.
```

更稳的说法是：

```text
The current Qwen3-VL clinical instruction pipeline is feasible and P4 is the best pilot, but stronger image-specific grounding requires harder counterfactual and image-mismatch supervision.
```

---

## 1. 本轮实验的核心问题

本轮实验不是简单多跑，而是要回答 6 个问题。

| Question ID | 科学问题 | 对应实验 |
|---|---|---|
| Q1 | 当前 1000 steps 是否训练量不够？ | P4 step scaling: 1k / 3k / 5k / 8k |
| Q2 | 数据量是否不够？ | 1k / 3k / 10k / full image scaling |
| Q3 | 每张图 instruction 数量是否不够？ | QA per image: 2 / 5 / 8 |
| Q4 | 现有 counterfactual 是否太软？ | P4-v2 hard A/B counterfactual |
| Q5 | 模型是否仍然靠文本 shortcut？ | hard image shuffle / question-only / report leakage audit |
| Q6 | VLM-coupled 是否真的优于散装 scaffold / no-LM data-only？ | P4-v2 vs P1 scaffold vs P6 data-only |

---

## 2. 总体实验路线

### 2.1 不要直接盲目扩大旧 P4

旧 P4 已经比 P2/P3 好，但 image-shuffle delta 小。  
如果直接把旧 P4 训练到 8k steps，可能出现两种情况：

| 情况 | 解释 |
|---|---|
| AUC 涨、image-shuffle delta 也涨 | 训练量确实不够，继续扩大有价值 |
| AUC 涨、image-shuffle delta 不涨 | 表征可能变好，但还没真正 image-specific |
| loss 降、AUC 不涨 | 学了更多文本模板 |
| loss 降、image-shuffle delta 不涨 | 进一步确认 shortcut 问题 |

所以本轮不能只做 long training，必须同时设计 **P4-v2 hard counterfactual 数据**。

---

### 2.2 本轮主线

本轮主线是：

```text
P4-v2 = Qwen3-VL + report-grounded QA + standardized hard counterfactual + no report leakage + hard image-shuffle diagnostics
```

推荐第一阶段优先跑：

```text
P4-v2-3k-5k
= 3k images
= 每图 4-6 条 instruction
= 5000 training steps
= Qwen3-VL language decoder frozen
= train vision tower + visual connector
```

---

## 3. 数据生成方案

### 3.1 数据版本定义

本轮建议扩展为 D0-D8。

| Data ID | Name | Description | 是否优先 |
|---|---|---|---|
| D0 | fixed JSON schema | 旧 UMS JSON | baseline |
| D1 | label-to-QA | 简单 label 改写成 QA | optional |
| D2 | report-grounded QA | 从 report 抽 evidence/location/uncertainty 的 QA | 保留 |
| D3 | soft counterfactual QA | 当前 P4 类数据 | 已有 |
| D4 | token-weighted D3 | 当前 P5 类数据 | appendix |
| D6 | **standardized hard A/B counterfactual** | 统一 A/B 选择题，答案只输出 A/B/Yes/No | **主推** |
| D7 | hard image-report mismatch | 构造同问题不同图 / 同图不同报告 | **主推** |
| D8 | no-leakage clinical QA | 删除 question 中所有 report leakage | **主推** |
| D9 | anatomy/location focused QA | 专门问 left/right/bilateral/basilar/apical | optional |
| D10 | uncertainty/answerability focused QA | 专门问 uncertain / not answerable | optional |

---

### 3.2 P4-v2 的数据目标

每张图建议生成 4-6 条 instruction：

| Slot | Instruction type | 比例 | 例子 |
|---|---|---:|---|
| 1 | finding verification | 20% | Does this CXR support cardiomegaly? |
| 2 | location/laterality | 20% | Which side better supports pleural effusion? |
| 3 | uncertainty / answerability | 20% | Is the edema definite, uncertain, or not answerable? |
| 4 | standardized A/B counterfactual | 30% | A: left effusion. B: right effusion. Which is supported? |
| 5 | image-report consistency | 10% | Does the image support this statement? |

推荐比例：

```text
positive finding QA        20%
explicit negative QA       15%
location/laterality QA     20%
uncertainty/answerability  20%
hard counterfactual        25%
```

---

## 4. GLM/API 生成任务

### 4.1 生成分两步，不要一步到位

#### Step 1：抽取 clinical facts

GLM 先从 report 抽结构化事实，不直接生成 QA。

输出：

```json
{
  "sample_id": "...",
  "facts": [
    {
      "finding": "pleural_effusion",
      "state": "present",
      "evidence_span": "small left pleural effusion",
      "location": "left",
      "severity": "small",
      "certainty": "definite",
      "visual_dependency": "high"
    }
  ]
}
```

#### Step 2：根据 facts 生成 QA / counterfactual

这样做的好处：

- 更容易过滤 hallucination；
- 更容易构造 hard negative；
- 更容易做 evidence_span check；
- 更容易保证 null 不被改成 absent。

---

### 4.2 Step 1 Prompt：clinical fact extraction

```text
You are a radiology information extraction assistant.

Input:
- A chest X-ray report.
- A fixed list of findings.

Task:
Extract only clinically supported facts from the report.

Rules:
1. Do not invent findings.
2. Copy evidence_span exactly from the report whenever possible.
3. If a finding is not mentioned, do not mark it as absent.
4. Only mark absent if the report explicitly negates it, e.g. "no pneumothorax".
5. Extract laterality/location only if explicitly stated.
6. Extract severity only if explicitly stated.
7. Extract uncertainty if terms such as "possible", "may represent", "cannot exclude", "questionable" appear.
8. Output JSON only.

Allowed finding list:
[LIST_OF_FINDINGS]

Output schema:
{
  "facts": [
    {
      "finding": "...",
      "state": "present | absent | uncertain",
      "evidence_span": "...",
      "location": "left | right | bilateral | basilar | apical | diffuse | null",
      "severity": "tiny | small | mild | moderate | large | severe | null",
      "certainty": "definite | uncertain",
      "visual_dependency": "high | medium | low"
    }
  ],
  "unmentioned_findings": [...]
}
```

---

### 4.3 Step 2 Prompt：QA generation

```text
You are a radiology visual instruction generator.

Input:
- Extracted clinical facts from a chest X-ray report.
- The original report.
- A fixed list of allowed answer types.

Task:
Generate image-grounded clinical visual instructions.

Rules:
1. The model will see only the image and the question during training.
2. Do not include the original report text in the question.
3. Do not leak the answer in the question.
4. Prefer short answers.
5. For counterfactual questions, use a strict A/B format.
6. Every counterfactual must be anchored to a true extracted fact.
7. For laterality counterfactuals, only use left/right flips when the true fact has laterality.
8. For state counterfactuals, flip present/absent only when one side is explicitly supported.
9. Do not ask an absent question for unmentioned findings.
10. Output JSONL records only.

Required answer formats:
- yes_no: "Yes" or "No"
- choice: "A" or "B"
- state: "present", "absent", "uncertain", "not answerable"
- short_evidence: <= 12 words

Output fields:
sample_id, question, answer, answer_short, finding, state, answer_type,
evidence_span, location, severity, visual_dependency, counterfactual_type,
negative_option_source, validation_status
```

---

## 5. 关键：防止 report leakage

### 5.1 什么叫 leakage？

坏例子：

```text
Question:
The report mentions a small left pleural effusion. Which side has effusion?

Answer:
Left.
```

这不需要看图，问题里已经泄露答案。

---

### 5.2 好例子

```text
Question:
Which statement is better supported by the chest X-ray?
A. There is a left pleural effusion.
B. There is a right pleural effusion.

Answer:
A
```

或者：

```text
Question:
Does this chest X-ray support the statement: "There is a pneumothorax"?

Answer:
No.
```

---

### 5.3 自动检测 leakage

需要实现一个 script：

```text
scripts/audit_instruction_leakage.py
```

检查项：

| Check | Reject / Flag |
|---|---|
| question contains evidence_span | reject |
| question contains exact answer | reject |
| question contains "report says" | reject for train, maybe keep diagnostic |
| question contains location while answer asks location | flag |
| question contains severity while answer asks severity | flag |
| answer can be inferred from question alone | flag by heuristic |
| A/B option lengths extremely imbalanced | flag |
| correct option always A | reject / rebalance |

输出：

```text
outputs/final_tables/instruction_leakage_audit.csv
outputs/final_tables/instruction_leakage_audit.md
```

---

## 6. Hard counterfactual 设计

### 6.1 统一 A/B 格式

所有 counterfactual_choice 必须统一：

```json
{
  "question": "Which statement is better supported by the chest X-ray?\nA. There is a left pleural effusion.\nB. There is a right pleural effusion.",
  "answer": "A",
  "answer_short": "A",
  "finding": "pleural_effusion",
  "counterfactual_type": "laterality_flip",
  "positive_option": "A",
  "negative_option": "B"
}
```

---

### 6.2 反事实类型

| Type | 正例 | 反例 | 是否优先 |
|---|---|---|---|
| `laterality_flip` | left effusion | right effusion | 高 |
| `state_flip_present_absent` | pneumothorax present | no pneumothorax | 高 |
| `state_flip_absent_present` | no pneumothorax | pneumothorax present | 高 |
| `uncertainty_flip` | possible edema | definite edema | 中 |
| `severity_flip` | small effusion | large effusion | 中 |
| `location_flip` | basilar opacity | apical opacity | 中 |
| `wrong_finding` | pleural effusion | pneumothorax | 低，容易过简单 |

---

### 6.3 A/B 平衡

必须保证：

| Item | Target |
|---|---|
| Correct answer A | 50% |
| Correct answer B | 50% |
| laterality left/right | balanced if possible |
| present/absent | balanced if possible |
| finding distribution | not dominated by effusion/pneumothorax |

---

## 7. Hard image-shuffle 设计

### 7.1 不要只做 random shuffle

当前 random image-shuffle delta 小，可能是因为 shuffle 太粗或模型依赖 question prior。  
本轮要加 hard shuffle。

---

### 7.2 Shuffle 类型

| Shuffle ID | Name | Description | 重要性 |
|---|---|---|---|
| S0 | random image shuffle | 随机换图 | baseline |
| S1 | same-finding opposite-state | 同 finding，但 state 相反 | 高 |
| S2 | same-finding opposite-laterality | 同 finding，但 left/right 相反 | 高 |
| S3 | same-question different-answer | 同问题模板，答案不同 | 高 |
| S4 | same-label-distribution shuffle | 标签分布相似但具体答案不同 | 中 |
| S5 | same-patient different-study | 如果有，换同患者不同检查 | optional |
| S6 | report-image mismatch | 图和报告不匹配 | 高 |

---

### 7.3 Hard shuffle table

每条 instruction 需要可选地记录 hard negative image：

```json
{
  "sample_id": "...",
  "image_path": "...",
  "question": "...",
  "answer": "A",
  "hard_negative_image_path": "...",
  "hard_negative_reason": "same_finding_opposite_laterality",
  "hard_negative_expected_answer": "B"
}
```

---

## 8. 训练目标

### 8.1 基础目标：answer-only loss

默认只对 answer token 计算 loss，不对 question token 计算 loss。

| Token part | Loss? |
|---|---|
| system prompt | no |
| question | no |
| answer | yes |
| padding | no |

---

### 8.2 Optional：visual-dependent token weighting

先不要过强。建议从轻量权重开始。

| Token type | Weight |
|---|---:|
| A / B / Yes / No | 2.0 |
| present / absent / uncertain | 2.0 |
| left / right / bilateral | 2.0 |
| mild / moderate / severe | 1.5 |
| finding name | 1.5 |
| generic words | 1.0 |
| template words | 0.5 |

P5 旧结果显示 weighting 不一定提升 AUC，所以本轮不要只靠 weighting。  
建议 weighting 只作为 P4-v2-w 版本。

---

### 8.3 Optional：counterfactual margin loss

对 A/B pair 计算：

```text
loss = CE(correct_answer)
     + lambda_margin * max(0, margin + NLL(correct) - NLL(counterfactual))
```

建议初始：

```yaml
lambda_margin: 0.1
margin: 0.2
```

---

### 8.4 Optional：image-shuffle contrastive loss

对正图和 hard negative 图：

```text
NLL(answer | correct image, question)
<
NLL(answer | wrong image, question)
```

loss：

```text
loss_shuffle = max(0, margin + NLL(correct_image) - NLL(wrong_image))
```

建议先作为 R6 做，不要一开始混到所有 run。

---

## 9. 模型训练配置

### 9.1 固定主模型

```text
Qwen3-VL-2B-Instruct
```

训练策略：

| Component | Train? | Notes |
|---|---|---|
| language decoder | no | frozen |
| vision tower | yes | 主学习对象 |
| visual connector / merger | yes | 适配 CXR instruction |
| processor/tokenizer | no | fixed |
| LLM LoRA | no first | optional P7 |
| vision LoRA | optional | if full vision training too costly |

---

### 9.2 训练规模建议

| Scale | Images | QA per image | Total QA approx | Steps | 用途 |
|---|---:|---:|---:|---:|---|
| debug | 20-200 | 3-5 | 100-1000 | 20-100 | code check |
| pilot-1k | 1000 | 4-6 | 4000-6000 | 3000 | 快速比较 |
| main-3k | 3000 | 4-6 | 12000-18000 | 5000 | 主推荐 |
| extended-3k | 3000 | 4-6 | 12000-18000 | 8000 | 看 step saturation |
| main-10k | 10000 | 4-6 | 40000-60000 | 8000 | 若 3k 有效再跑 |
| full | all | 3-5 | large | 8k-12k | 最后版本 |

---

## 10. 实验矩阵：第一阶段

### 10.1 Step scaling：训练步数是否不够

| Run ID | Base data | Images | QA/img | Steps | Model | Train LLM? | Output dir | Done? |
|---|---|---:|---:|---:|---|---|---|---|
| S-P4-1k | old P4 | current | current | 1000 | Qwen3-VL | no |  | 已有 |
| S-P4-3k | old P4 | current | current | 3000 | Qwen3-VL | no |  |  |
| S-P4-5k | old P4 | current | current | 5000 | Qwen3-VL | no |  |  |
| S-P4-8k | old P4 | current | current | 8000 | Qwen3-VL | no |  |  |

空结果表：

| Run ID | Best train loss | Best val loss | CheXpert AUC | NIH AUC | Image-shuffle delta | CF acc | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| S-P4-1k |  |  |  |  |  |  |  |
| S-P4-3k |  |  |  |  |  |  |  |
| S-P4-5k |  |  |  |  |  |  |  |
| S-P4-8k |  |  |  |  |  |  |  |

---

### 10.2 P4-v2 hard counterfactual：数据是否更好

| Run ID | Data | Images | QA/img | Steps | Loss | Purpose | Done? |
|---|---|---:|---:|---:|---|---|---|
| CF-1k-3k | D6 hard CF | 1k | 4-6 | 3000 | CE | hard CF small |
| CF-3k-5k | D6 hard CF | 3k | 4-6 | 5000 | CE | main |
| CF-3k-8k | D6 hard CF | 3k | 4-6 | 8000 | CE | step saturation |
| CF-10k-8k | D6 hard CF | 10k | 4-6 | 8000 | CE | scale |
| CF-full | D6 hard CF | full | 3-5 | 8k-12k | CE | final optional |

空结果表：

| Run ID | CheXpert AUC | NIH AUC | CF acc | Image-shuffle delta | Question-only delta | Paraphrase delta | Cost GPU-h | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| CF-1k-3k |  |  |  |  |  |  |  |  |
| CF-3k-5k |  |  |  |  |  |  |  |  |
| CF-3k-8k |  |  |  |  |  |  |  |  |
| CF-10k-8k |  |  |  |  |  |  |  |  |
| CF-full |  |  |  |  |  |  |  |  |

---

### 10.3 Hard image-shuffle training

| Run ID | Data | Images | QA/img | Steps | Extra loss | Purpose | Done? |
|---|---|---:|---:|---:|---|---|---|
| SHUF-1k | D7 hard shuffle | 1k | 4-6 | 3000 | shuffle margin | quick check |
| SHUF-3k | D7 hard shuffle | 3k | 4-6 | 5000 | shuffle margin | main |
| SHUF-3k-w | D7 + token weight | 3k | 4-6 | 5000 | weighted + margin | optional |

空结果表：

| Run ID | CheXpert AUC | NIH AUC | Random shuffle delta | Hard shuffle delta | CF acc | Question-only delta | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| SHUF-1k |  |  |  |  |  |  |  |
| SHUF-3k |  |  |  |  |  |  |  |
| SHUF-3k-w |  |  |  |  |  |  |  |

---

## 11. 实验矩阵：第二阶段拓展

### 11.1 QA per image ablation

| Run ID | Images | QA/img | Steps | Data | Purpose |
|---|---:|---:|---:|---|---|
| QA2-3k | 3k | 2 | 5000 | D6 | 少量 QA |
| QA5-3k | 3k | 5 | 5000 | D6 | 推荐 |
| QA8-3k | 3k | 8 | 5000 | D6 | 看噪声是否增大 |

空表：

| Run ID | CheXpert AUC | NIH AUC | CF acc | Hard shuffle delta | Instruction audit pass rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| QA2-3k |  |  |  |  |  |  |
| QA5-3k |  |  |  |  |  |  |
| QA8-3k |  |  |  |  |  |  |

---

### 11.2 Training policy ablation

| Run ID | Vision tower | Connector | LLM | Data | Steps | Purpose |
|---|---|---|---|---|---:|---|
| TRAIN-CONN | frozen | train | frozen | D6 | 5000 | connector-only |
| TRAIN-LAST4 | last 4 blocks | train | frozen | D6 | 5000 | cheaper vision adaptation |
| TRAIN-FULLVISION | full train | train | frozen | D6 | 5000 | main |
| TRAIN-VISION-LORA | LoRA | train | frozen | D6 | 5000 | memory efficient |
| TRAIN-LLM-LORA | full vision | train | LoRA | D6 | 5000 | optional upper bound |

空表：

| Run ID | Trainable params | Peak VRAM | GPU-hours | CheXpert AUC | NIH AUC | Hard shuffle delta | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| TRAIN-CONN |  |  |  |  |  |  |  |
| TRAIN-LAST4 |  |  |  |  |  |  |  |
| TRAIN-FULLVISION |  |  |  |  |  |  |  |
| TRAIN-VISION-LORA |  |  |  |  |  |  |  |
| TRAIN-LLM-LORA |  |  |  |  |  |  |  |

---

### 11.3 Model control

| Run ID | Model | Data | Steps | Purpose |
|---|---|---|---:|---|
| VLM-QWEN3VL-2B | Qwen3-VL-2B | D6 | 5000 | main |
| TXT-QWEN35-2B | Qwen3.5-2B text scaffold | D6 | 5000 | text-only scaffold control |
| OLD-QWEN-CODER | Qwen-Coder scaffold | D6 | 5000 | old baseline |
| NO-LM-DATA | no-LM ViT / Qwen vision + heads | D6 labels | 5000 | data-only control |

空表：

| Run ID | CheXpert AUC | NIH AUC | CF acc | Hard shuffle delta | Question-only delta | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| VLM-QWEN3VL-2B |  |  |  |  |  |  |
| TXT-QWEN35-2B |  |  |  |  |  |  |
| OLD-QWEN-CODER |  |  |  |  |  |  |
| NO-LM-DATA |  |  |  |  |  |  |

---

## 12. 评估协议

### 12.1 LP evaluation

每个 source run 必须抽出 vision checkpoint，然后做 LP。

| Run ID | Vision checkpoint | Pooling | CheXpert AUC | CheXpert F1 | NIH AUC | NIH F1 | Notes |
|---|---|---|---:|---:|---:|---:|---|
|  |  | cls |  |  |  |  |  |
|  |  | mean |  |  |  |  |  |
|  |  | connector_mean |  |  |  |  |  |

建议每个 run 至少试：

- mean pooling
- connector output mean
- cls / special token if available

---

### 12.2 Visual-dependence diagnostics

每个 run 都必须输出：

| Run ID | Blank image loss | Correct image loss | Random shuffle loss | Hard shuffle loss | Question-only delta | Random shuffle delta | Hard shuffle delta |
|---|---:|---:|---:|---:|---:|---:|---:|
|  |  |  |  |  |  |  |  |

解释：

```text
Question-only delta = blank image loss - correct image loss
Random shuffle delta = random wrong image loss - correct image loss
Hard shuffle delta = hard wrong image loss - correct image loss
```

成功标准：

```text
Hard shuffle delta > 0.05: 有初步 image-specific grounding
Hard shuffle delta > 0.10: 比较强
Hard shuffle delta <= 0.02: 仍然弱
```

---

### 12.3 Counterfactual diagnostics

| Run ID | Total CF examples | Valid A/B examples | CF acc | Laterality CF acc | State CF acc | Uncertainty CF acc | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
|  |  |  |  |  |  |  |  |

指标定义：

```text
CF acc = P(NLL(correct option) < NLL(counterfactual option))
```

---

### 12.4 Paraphrase robustness

| Run ID | Original loss | Clinical paraphrase loss | Style paraphrase loss | Original-clinical delta | Original-style delta | Robustness conclusion |
|---|---:|---:|---:|---:|---:|---|
|  |  |  |  |  |  |  |

---

### 12.5 Subgroup metrics

| Run ID | Common AUC | Rare AUC | High-null AUC | Uncertain-heavy AUC | Location-related AUC | Support devices AUC | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
|  |  |  |  |  |  |  |  |

---

### 12.6 Instruction quality audit

| Dataset | Images | Instructions | Avg QA/img | Accepted % | Rejected % | Leakage % | Evidence-span pass % | Laterality pass % | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D6-1k |  |  |  |  |  |  |  |  |  |
| D6-3k |  |  |  |  |  |  |  |  |  |
| D7-3k |  |  |  |  |  |  |  |  |  |

---

## 13. 手工 audit 表格

至少抽 200 条 instruction 人工看。

| audit_id | sample_id | question | answer | evidence_span | finding | answer_type | visual_dependency | Correct? | Leakage? | Hallucination? | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |  | yes/no | yes/no | yes/no |  |
| 2 |  |  |  |  |  |  |  | yes/no | yes/no | yes/no |  |

通过标准：

```text
Correct >= 90%
Leakage <= 10%
Hallucination <= 5%
```

如果达不到，不要启动 full training。

---

## 14. 推荐执行顺序

### Phase 0：数据生成和质量审计

| Task | Output | Done? |
|---|---|---|
| Generate fact extraction for 1k reports | `facts_1k.jsonl` |  |
| Generate D6 hard CF 1k | `d6_hard_cf_1k.jsonl` |  |
| Validate D6 1k | audit tables |  |
| Manual audit 200 rows | manual audit CSV |  |
| Generate D6 3k | `d6_hard_cf_3k.jsonl` |  |
| Generate D7 hard shuffle 3k | `d7_hard_shuffle_3k.jsonl` |  |

---

### Phase 1：最小训练验证

| Task | Run | Done? |
|---|---|---|
| Debug train D6 1k | debug 20-100 steps |  |
| Train CF-1k-3k | 1k / 3000 steps |  |
| Evaluate LP + diagnostics | CF-1k-3k |  |
| Train current-P4-3k | old P4 / 3000 steps |  |
| Compare old P4 vs P4-v2 | result table |  |

---

### Phase 2：主实验

| Task | Run | Done? |
|---|---|---|
| Train CF-3k-5k | D6 / 3k / 5000 |  |
| Train SHUF-3k | D7 / 3k / 5000 |  |
| Evaluate all diagnostics | LP / NIH / CF / shuffle |  |
| Select best between CF and SHUF | decision table |  |

---

### Phase 3：扩展

| Task | Run | Done? |
|---|---|---|
| Train CF-3k-8k | step scaling |  |
| Train CF-10k-8k | image scaling |  |
| Train best policy ablation | last4 / full / LoRA |  |
| Add text-only scaffold control | if needed |  |
| Add no-LM data-only control | if needed |  |

---

## 15. 成功/失败判断

### 15.1 Strong success

满足至少 3 条：

| Criterion | Threshold |
|---|---|
| CheXpert AUC improves over P4 old | +0.01 |
| NIH AUC improves over P4 old | +0.01 |
| Hard shuffle delta | > 0.05 |
| CF acc | > 0.85 |
| Question-only delta stays high | > 1.0 |
| Paraphrase delta decreases | > 20% relative reduction |
| Rare/high-null/uncertain subgroup improves | +0.01 |

---

### 15.2 Partial success

满足：

```text
AUC improves, but hard shuffle still weak.
```

论文写法：

```text
Clinical instruction improves deployable CXR representations, but image-specific grounding remains incomplete.
```

---

### 15.3 Negative result

出现：

```text
AUC not better than old P4 or Base,
and hard shuffle delta still <= 0.02.
```

论文写法：

```text
Scaling instruction volume alone does not ensure visual grounding; stronger localization/evidence supervision is needed.
```

---

## 16. Codex 执行任务清单

### 16.1 数据生成

```text
Task ID: GEN_FACTS_WITH_GLM

Goal:
  Use GLM API to extract clinical facts from reports before generating QA.

Inputs:
  data reports / UMS records
  finding list

Outputs:
  outputs/instruction_data/facts/facts_1k.jsonl
  outputs/instruction_data/facts/facts_3k.jsonl
  outputs/final_tables/fact_extraction_audit.csv

Rules:
  - Do not invent facts.
  - Evidence span must be copied from report if available.
  - Null/unmentioned is not absent.
```

```text
Task ID: GEN_D6_HARD_CF

Goal:
  Generate standardized hard A/B counterfactual QA from extracted facts.

Outputs:
  outputs/instruction_data/glm_validated/d6_hard_cf_1k.jsonl
  outputs/instruction_data/glm_validated/d6_hard_cf_3k.jsonl
  outputs/final_tables/d6_instruction_audit.csv

Rules:
  - All counterfactual questions must be A/B format.
  - Correct answer A/B must be balanced.
  - No report leakage in question.
```

```text
Task ID: GEN_D7_HARD_SHUFFLE

Goal:
  Attach hard negative image paths for image-specific grounding diagnostics/training.

Outputs:
  outputs/instruction_data/glm_validated/d7_hard_shuffle_3k.jsonl
  outputs/final_tables/d7_shuffle_pair_audit.csv

Hard negative types:
  - same finding opposite state
  - same finding opposite laterality
  - same question different answer
  - report-image mismatch
```

---

### 16.2 审计

```text
Task ID: AUDIT_INSTRUCTION_QUALITY

Outputs:
  outputs/final_tables/instruction_leakage_audit.csv
  outputs/final_tables/instruction_distribution.csv
  outputs/final_tables/manual_audit_template.csv

Success:
  - leakage rate <= 10%
  - hallucination rate <= 5%
  - valid A/B CF >= 90%
```

---

### 16.3 训练

```text
Task ID: TRAIN_P4V2_CF_1K_3K

Config:
  configs/qwen3vl_instruction/cf_1k_3k.yaml

Train:
  Qwen3-VL language decoder frozen
  vision tower trainable
  visual connector trainable
  answer-only CE loss

Outputs:
  outputs/qwen3vl_instruction/cf_1k_3k/
```

```text
Task ID: TRAIN_P4V2_CF_3K_5K

Config:
  configs/qwen3vl_instruction/cf_3k_5k.yaml

Outputs:
  outputs/qwen3vl_instruction/cf_3k_5k/
```

```text
Task ID: TRAIN_SHUF_3K_5K

Config:
  configs/qwen3vl_instruction/shuf_3k_5k.yaml

Loss:
  answer-only CE + image-shuffle margin

Outputs:
  outputs/qwen3vl_instruction/shuf_3k_5k/
```

---

### 16.4 评估

```text
Task ID: EVAL_QWEN3VL_VISION_LP

For each source run:
  - extract vision checkpoint
  - run CheXpert LP
  - run NIH transfer
  - run subgroup metrics

Outputs:
  outputs/final_tables/qwen3vl_p4v2_lp_results.csv
```

```text
Task ID: EVAL_VISUAL_DEPENDENCE

For each source run:
  - blank image
  - question-only
  - random shuffle
  - hard shuffle
  - report-image mismatch

Outputs:
  outputs/final_tables/qwen3vl_p4v2_visual_dependence.csv
```

```text
Task ID: EVAL_COUNTERFACTUAL

For each source run:
  - compute NLL(correct option)
  - compute NLL(counterfactual option)
  - pairwise accuracy

Outputs:
  outputs/final_tables/qwen3vl_p4v2_counterfactual.csv
```

---

## 17. 最终论文写法模板

如果 P4-v2 成功：

```text
We found that fixed-schema generation improves teacher-forced loss but does not reliably improve deployable CXR representations. In contrast, standardized report-grounded counterfactual instructions yield stronger downstream linear-probe performance and substantially larger hard image-shuffle loss gaps, indicating improved image-specific clinical grounding. These results suggest that language supervision becomes visual supervision only when the instruction data are designed to require image-specific evidence.
```

如果 P4-v2 只小幅成功：

```text
Qwen3-VL-coupled clinical instruction pretraining provides modest gains over the base vision tower and fixed-schema generation. However, hard image-shuffle diagnostics reveal that image-specific grounding remains incomplete. This suggests that report-derived instruction data must be carefully designed and audited; scaling instruction volume alone is insufficient.
```

如果失败：

```text
Scaling report-grounded instruction data did not improve image-specific grounding beyond the initial pilot. This negative result suggests that CXR report-derived supervision may be too weak without explicit localization, segmentation, or stronger image-report alignment signals.
```

---

## 18. 本轮最推荐先跑的 6 个 run

最终建议：

| Priority | Run | Why |
|---:|---|---|
| 1 | CF-1k-3k | 快速看 P4-v2 hard CF 有没有用 |
| 2 | S-P4-3k | 看旧 P4 只是训练不够还是数据不够 |
| 3 | CF-3k-5k | 主实验 |
| 4 | SHUF-3k | 专门解决 image-shuffle delta 小 |
| 5 | CF-3k-8k | 看 step saturation |
| 6 | QA5-vs-QA8 | 看每图 instruction 数量是否过多/过少 |

---

## 19. 实验完成后请回传的文件

做实验的同学跑完后，请至少回传：

```text
outputs/final_tables/instruction_data_audit.md
outputs/final_tables/instruction_leakage_audit.md
outputs/final_tables/qwen3vl_p4v2_lp_results.md
outputs/final_tables/qwen3vl_p4v2_visual_dependence.md
outputs/final_tables/qwen3vl_p4v2_counterfactual.md
outputs/final_tables/qwen3vl_p4v2_paraphrase.md
outputs/final_tables/qwen3vl_p4v2_subgroup.md
outputs/final_tables/qwen3vl_p4v2_cost_table.md
outputs/final_tables/qwen3vl_p4v2_decision_summary.md
```

以及每个 run 的：

```text
config_snapshot.json
metrics_final.json
metrics_step_*.json
progress.json
failure_log.txt if failed
vision_export_manifest.json
```

---

## 20. 一句话总结

本轮实验不要只回答：

```text
多训练有没有用？
```

而要回答：

```text
更多、更难、更少泄露的 clinical counterfactual instruction，能不能让 Qwen3-VL 的 language supervision 真正变成 image-specific clinical visual supervision？
```

如果答案是 yes，论文主线就很清楚：

```text
Clinical Evidence Instruction Pretraining works when the instruction data require image-specific evidence.
```

如果答案是 no，也能形成有价值的结论：

```text
CXR report-derived instruction alone is not enough; explicit localization or stronger image-report grounding is needed.
```

---

## 21. 2026-06-29 执行结果写回

### 21.1 完成状态

本轮已完成第 18 节优先推荐矩阵，并补齐 QA5-vs-QA8 对照：

| Run | 状态 | 说明 |
|---|---|---|
| CF-1k-3k | completed | D6 hard CF 1k，3000 steps，完整 LP/NIH/visual/CF/paraphrase。 |
| S-P4-3k | completed | old P4，3000 steps，完整 LP/NIH/visual/CF/paraphrase。 |
| CF-3k-5k | completed | D6 hard CF 3k，5000 steps，完整 LP/NIH/visual/CF/paraphrase。 |
| SHUF-3k | completed | D7 hard shuffle 3k，5000 steps，shuffle margin，完整 LP/NIH/visual/CF/paraphrase。 |
| CF-3k-8k | completed | D6 hard CF 3k，8000 steps，完整 LP/NIH/visual/CF/paraphrase。 |
| QA8-3k | completed | D6 hard CF 3k / QA8，5000 steps，完整 LP/NIH/visual/CF/paraphrase；QA5 对照使用 CF-3k-5k。 |

### 21.2 数据生成与质量审计

| Dataset | Facts/images | Instructions | Images | QA/image | Accepted | Rejected/leakage heuristic | A answer | Hard negative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D6-1k | 1000 | 4796 | 1000 | 4.796 | 96.3511% | 3.6489% | 50.0% | 0% |
| D7-1k | 1000 | 4796 | 1000 | 4.796 | 96.3511% | 3.6489% | 50.0% | 100% |
| D6-3k | 3000 | 14333 | 2980 | 4.8097 | 96.4557% | 3.5443% | 49.9773% | 0% |
| D7-3k | 3000 | 14333 | 2980 | 4.8097 | 96.4557% | 3.5443% | 49.9773% | 100% |
| D6-QA8-3k | 3000 | 22613 | 2983 | 7.5806 | 96.7497% | 3.2503% | 49.9649% | 0% |

关键产物：

```text
outputs/instruction_data/facts/facts_1k.jsonl
outputs/instruction_data/facts/facts_extra2k.jsonl
outputs/instruction_data/facts/facts_3k.jsonl
outputs/instruction_data/glm_validated/d6_hard_cf_1k.jsonl
outputs/instruction_data/glm_validated/d7_hard_shuffle_1k.jsonl
outputs/instruction_data/glm_validated/d6_hard_cf_3k.jsonl
outputs/instruction_data/glm_validated/d7_hard_shuffle_3k.jsonl
outputs/instruction_data/glm_validated/d6_hard_cf_3k_qa8.jsonl
outputs/final_tables/p4v2_d6_d7_3k_canonical_validation.md
outputs/final_tables/p4v2_d6_qa8_3k_canonical_validation.md
```

API 并发处理：旧的 `v3_mimic_5k/train_extra4k` 生成任务在 P4-v2 造数期间保持暂停；P4-v2 `facts_extra2k` 四个 shard 先各处理 500 输入样本，因网络/解析错误短缺成功行后，使用 `--target-output-count 500` 精确补齐到 500/500/500/500。失败记录保留在 `outputs/instruction_generation/p4v2_facts/*_parse_errors.jsonl`。

旧造数任务恢复边界：P4-v2 完成后已检查旧 `v3_mimic_5k/train_extra4k` 的恢复点；当前只找到 `outputs/instruction_generation/v3_mimic_5k/train_extra4k_shard*.jsonl` API/progress/error 日志，未找到对应 raw instruction 输出 JSONL。因为 `scripts/generate_clinical_instructions.py --resume` 依赖 raw 输出中的 `sample_id` 去跳过已完成样本，若在 raw 路径缺失时直接重启会重新消耗 API 生成已成功样本，所以本轮没有盲目重启旧 extra4k。

### 21.3 主结果

| Run | CheXpert AUC | NIH AUC | Question-only delta | Random shuffle delta | Hard shuffle delta | CF acc | Clinical paraphrase delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| S-P4-1k | 0.689561 | 0.568298 | 1.77718 | 0.00872866 |  | 0.789954 | 0.0040401 |
| S-P4-3k | 0.681137 | 0.552140 | 1.77571 | 0.0130892 |  | 0.808219 | 0.0040401 |
| S-P4-5k | 0.693899 | 0.550347 | 1.75329 | 0.0164483 |  | 0.771689 | 0.00433018 |
| S-P4-8k | 0.708535 | 0.568478 | 1.80037 | 0.0176604 |  | 0.794521 | 0.00118312 |
| CF-1k-3k | 0.678611 | 0.544189 | 4.89754 | 0.0391953 |  | 0.895692 | 0.0220581 |
| CF-3k-5k | 0.699136 | 0.563358 | 4.93978 | 0.0600665 |  | 0.893424 | 0.0265334 |
| SHUF-3k | **0.726709** | 0.568045 | 5.21473 | 0.0716429 | **0.0806744** | 0.870748 | 0.0199110 |
| CF-3k-8k | 0.691562 | 0.555334 | 4.93286 | 0.0691310 |  | **0.899093** | 0.0294459 |
| QA8-3k | 0.706873 | 0.566925 | **5.31226** | **0.0832804** |  | 0.860544 | 0.0254821 |

完整表格：

```text
outputs/final_tables/qwen3vl_p4v2_training_results.md
outputs/final_tables/qwen3vl_p4v2_lp_results.md
outputs/final_tables/qwen3vl_p4v2_visual_dependence.md
outputs/final_tables/qwen3vl_p4v2_counterfactual.md
outputs/final_tables/qwen3vl_p4v2_paraphrase.md
outputs/final_tables/qwen3vl_p4v2_subgroup.md
outputs/final_tables/qwen3vl_p4v2_cost_table.md
outputs/final_tables/qwen3vl_p4v2_decision_summary.md
```

### 21.4 结论

本轮最强 run 是 `SHUF-3k`：

- CheXpert macro-AUC 0.726709，高于旧 P4 1k 的 0.689561，也高于旧 P4 8k 的 0.708535。
- NIH macro-AUC 0.568045，基本持平旧 P4 1k / 8k。
- random shuffle delta 0.0716429，hard shuffle delta 0.0806744，超过本计划 0.05 的初步 image-specific grounding 阈值。
- CF acc 0.870748，高于 0.85 阈值，但低于 CF-3k-8k 的 0.899093。

更稳妥的论文级表述：

```text
Standardized hard counterfactual supervision greatly improves counterfactual option preference, and hard image-shuffle training produces the clearest gain in deployable CXR representation quality and image-mismatch sensitivity. The evidence supports initial image-specific clinical grounding, but NIH transfer remains modest, so the claim should be framed as partial-to-strong success rather than final solved grounding.
```

### 21.5 QA5-vs-QA8

`QA5` 对照为 `CF-3k-5k`，`QA8` 为 `QA8-3k`：

| Comparison | CheXpert AUC | NIH AUC | Random shuffle delta | CF acc | Interpretation |
|---|---:|---:|---:|---:|---|
| QA5 / CF-3k-5k | 0.699136 | 0.563358 | 0.0600665 | 0.893424 | 更好的 CF acc。 |
| QA8-3k | 0.706873 | 0.566925 | 0.0832804 | 0.860544 | 更多 QA 提升 image-shuffle delta 和 AUC，但牺牲 CF acc。 |

结论：QA8 有效增加视觉扰动敏感性和轻微 AUC，但不是单调全面更好；若主线追求 CheXpert/image-shuffle，QA8 值得保留；若主线追求 CF acc，QA5/CF-3k-5k 更稳。

### 21.6 扩展项边界

本轮已完成优先 6 项及 QA8 对照。以下属于第 11-14 节中的条件扩展项，未在本轮继续启动：

| Item | Boundary |
|---|---|
| CF-10k-8k / CF-full | 需要额外大规模 GLM facts 生成；在 API 并发受限下未继续扩展。当前 3k 已能区分 CF vs SHUF 主线。 |
| TRAIN-CONN / TRAIN-LAST4 / TRAIN-VISION-LORA / TRAIN-LLM-LORA | 属于训练策略 ablation；本轮先固定 full vision + connector，避免把数据效果和训练策略混在一起。 |
| TXT-QWEN35-2B / OLD-QWEN-CODER | 旧 scaffold/no-LM 证据已有历史表；本轮主问题是 Qwen3-VL-coupled P4-v2 数据设计。 |
| CF-full final version | 需要先根据 SHUF-3k 结果决定是否把 D7 hard shuffle 作为最终数据主线。 |

当前推荐下一步：若继续扩展，优先做 `SHUF-10k-8k` 或 `SHUF-3k` 的训练策略 ablation，而不是继续单纯扩 D6 CF。
