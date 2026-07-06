# VIVID-Med 下一阶段实验计划：从 SHUF-3k 到完整 Clinical Instruction Workflow

> 版本：2026-06-29  
> 目标：把当前分散的 P3 / P4 / CF / SHUF / QA8 / P5 实验，整理成一套可以系统推进的实验路线。  
> 核心问题：**怎样把 Qwen3-VL 的语言监督真正变成 image-specific clinical visual supervision？**

---

## 0. 当前阶段判断

当前最强已知结果是：

```text
SHUF-3k:
CheXpert AUC = 0.726709
NIH AUC = 0.568045
Random shuffle delta = 0.0716429
Hard shuffle delta = 0.0806744
CF acc = 0.870748
```

这说明：

1. **直接 fixed JSON 不够**。模型容易学模板，不一定学视觉证据。
2. **P3/P4/CF 有用，但仍不够强**。基础 QA 和 counterfactual 能提高医学问答/反事实能力，但 image-specific grounding 仍有限。
3. **SHUF 是当前最关键组件**。它直接训练模型区分正确图和错误图，因此最能提高 hard shuffle delta。
4. **QA8 有价值**。更多 QA 提升 AUC 和图像扰动敏感性，但会稀释 CF acc。
5. **P5 / token weighting 不能单独当主方法，但可以作为 SHUF/curriculum 后面的增强项。**
6. 下一阶段不应该只做一个更大的 SHUF，而应该系统探索：
   - JSON loss masking；
   - rich QA mixture；
   - curriculum / progressive workflow；
   - SHUF++；
   - token weighting；
   - multi-scale；
   - model training policy；
   - external transfer and diagnostics。

---

# Part A. 本轮必须做的核心实验

下面这些是当前已经确定要做的，建议优先排入实验队列。

---

## A1. P2-value-only：fixed JSON 的 loss mask 诊断

### A1.1 为什么要做？

旧 P2 fixed JSON 效果差：

```text
Base CheXpert AUC = 0.6790
P2 fixed JSON CheXpert AUC = 0.6662
```

这说明 fixed JSON 不仅没有提升，反而可能伤害 vision tower 表征。

但我们还不知道 P2 差的原因到底是：

1. **JSON 这种任务本身太像模板**；
2. **loss 被大量 JSON key / punctuation / field-name token 污染**；
3. **字段顺序和格式 shortcut 太强**；
4. **状态 token 本身其实有用，但被模板 loss 淹没了**。

所以需要做 P2-value-only。

---

### A1.2 不要做什么？

不要用 attention mask 把字段名遮掉。

错误做法：

```text
模型看不到 "Cardiomegaly" / "Pleural Effusion" 等字段名。
```

这样模型不知道自己在预测哪个 finding，只能靠位置猜。

这会退化成：

```text
第一个位置大概率 present，第二个位置大概率 absent...
```

不是真正的医学视觉监督。

---

### A1.3 正确做法：loss mask

保留完整 JSON 输入/输出形式：

```json
{
  "Cardiomegaly": "present",
  "Pleural Effusion": "absent",
  "Pneumothorax": "null"
}
```

但是 loss 只算医学 value token：

```text
present / absent / uncertain / null
```

不算：

```text
{ } "Cardiomegaly" "Pleural Effusion" : , "state" 等 JSON key / punctuation / field-name token
```

也就是说：

```text
字段名仍然可见，模型知道自己在回答哪个 finding。
但字段名和 JSON 格式本身不参与 loss。
```

---

### A1.4 必跑实验

| Run ID | 数据 | Loss 策略 | 目的 | 优先级 |
|---|---|---|---|---|
| P2-full-json | D0 fixed JSON | 所有 answer token 都算 loss | 旧 baseline | 已有/可复现 |
| P2-value-only | D0 fixed JSON | 只算 present/absent/uncertain/null | 看模板 loss 是否有害 | 必跑 |
| P2-no-punct | D0 fixed JSON | 算 field name + value，不算 punctuation | 区分 punctuation vs field-name 影响 | 推荐 |
| P2-state-only-compact | compact schema | 只输出 field_id:value | 看精简 serialization 是否更好 | 推荐 |
| P2-field-query | 每次问一个字段，只答状态 | 去掉完整 JSON 顺序 | 必跑 |

---

### A1.5 P2-field-query 设计

不再让模型输出完整 JSON，而是每次问一个 finding。

例子：

```text
Q: What is the status of Cardiomegaly?
A: present
```

或者：

```text
Q: Is Pleural Effusion present, absent, uncertain, or not answerable?
A: absent
```

优点：

1. 没有 JSON 括号和字段顺序；
2. 字段名仍然明确；
3. answer 很短；
4. 更接近 QA；
5. 比 P2-value-only 更干净。

---

### A1.6 结果解释

| 结果模式 | 解释 |
|---|---|
| P2-value-only > P2-full-json | JSON template loss 有害 |
| P2-value-only ≈ P2-full-json | 问题不只是 template token，而是 fixed schema 本身弱 |
| P2-field-query > P2-value-only | 字段级 QA 比完整 JSON 更好 |
| P2-field-query 仍远低于 SHUF | 说明只做字段状态分类仍不够，需要 image-specific hard negatives |

---

### A1.7 空结果表

| Run ID | CheXpert AUC | NIH AUC | Random shuffle delta | CF acc | Template NLL gap | Notes |
|---|---:|---:|---:|---:|---:|---|
| P2-full-json |  |  |  |  |  |  |
| P2-value-only |  |  |  |  |  |  |
| P2-no-punct |  |  |  |  |  |  |
| P2-state-only-compact |  |  |  |  |  |  |
| P2-field-query |  |  |  |  |  |  |

---

## A2. Rich QA mixture：不要只做单一 QA 类型

### A2.1 为什么要做？

现在已有结果显示：

```text
QA8 比 QA5 在 AUC / shuffle sensitivity 上更好，但 CF acc 下降。
```

这说明问题不是“QA 越多越好”，而是：

```text
QA 类型比例怎么配最重要。
```

下一轮要把 QA 设计成正式的 instruction mixture，而不是随便多生成几个问题。

---

### A2.2 QA 类型池

| Type ID | QA 类型 | 教模型什么 | 视觉依赖强度 | 示例 |
|---|---|---|---|---|
| Q1 | basic finding QA | 基础疾病词汇和 yes/no | medium | Does this CXR support cardiomegaly? |
| Q2 | state QA | present/absent/uncertain/null | medium | What is the status of pneumothorax? |
| Q3 | location/laterality QA | 左右、局部证据 | high | Which side shows pleural effusion? |
| Q4 | severity QA | 程度 | high-medium | Is the effusion small or large? |
| Q5 | uncertainty QA | definite vs possible | medium | Is edema definite or uncertain? |
| Q6 | answerability QA | 未提及/不可回答 | low-medium | Is fracture answerable from this report-image pair? |
| Q7 | evidence phrase QA | 报告证据短语 | medium | What visual observation supports cardiomegaly? |
| Q8 | standardized A/B CF | 正确陈述 vs 错误陈述 | high | A left effusion / B right effusion |
| Q9 | image-report consistency | 图像是否支持陈述 | high | Does this CXR support the statement? |
| Q10 | SHUF QA | 正确图 vs 错图 | very high | same question, different images |
| Q11 | same-question different-answer | 相同问题不同图不同答案 | very high | same A/B question, image decides answer |
| Q12 | contradiction QA | 支持/反驳/不可判断 | high | Is the statement supported or contradicted? |
| Q13 | disease-pair discrimination | 区分相似 finding | high | edema vs consolidation |
| Q14 | anatomy-aware QA | anatomy region | high | Is the abnormality basilar or apical? |
| Q15 | temporal/change QA | 如果有 prior | high | Is the finding improved/worsened? |

---

### A2.3 推荐 mixture 版本

#### Mix-1：BalancedMix-QA8

```text
basic QA: 20%
location/laterality: 20%
uncertainty/answerability: 20%
hard CF: 25%
SHUF/image consistency: 15%
```

目的：

> 均衡训练，保留医学语义覆盖。

---

#### Mix-2：CF-heavy-QA8

```text
basic QA: 10%
location/laterality: 15%
uncertainty/answerability: 15%
hard CF: 45%
SHUF/image consistency: 15%
```

目的：

> 追求最高 CF acc。

预期：

- CF acc 高；
- AUC 不一定最高；
- hard shuffle delta 中等。

---

#### Mix-3：SHUF-heavy-QA8

```text
basic QA: 10%
location/laterality: 15%
uncertainty/answerability: 10%
hard CF: 25%
SHUF/image consistency: 40%
```

目的：

> 追求 image-specific grounding。

预期：

- hard shuffle delta 高；
- CheXpert AUC 可能高；
- CF acc 可能略降。

---

#### Mix-4：Clinical-rich-QA8

```text
basic QA: 15%
location/laterality: 25%
uncertainty/answerability: 25%
hard CF: 20%
SHUF/image consistency: 15%
```

目的：

> 医疗语义最强，突出 location、uncertainty、answerability。

预期：

- 如果 NIH / external transfer 上升，这个版本很有价值；
- CF acc 可能不最高。

---

#### Mix-5：StoryMix-QA8

```text
basic QA: 10%
location/laterality: 20%
uncertainty/answerability: 15%
hard CF: 30%
SHUF/image consistency: 25%
```

目的：

> 最适合作为最终主方法候选。

理由：

- 不让 basic QA 占太多；
- 保留医学语义；
- 大部分 budget 给 visual-dependent tasks；
- 论文故事最好讲。

---

### A2.4 QA 数量 ablation

固定 StoryMix 比例，尝试不同 QA/image：

| Run ID | QA/image | 目的 |
|---|---:|---|
| StoryMix-QA5 | 5 | 和当前 QA5/CF-3k-5k 接近 |
| StoryMix-QA8 | 8 | 主推荐 |
| StoryMix-QA10 | 10 | 看更多 QA 是否继续有益 |
| StoryMix-QA12 | 12 | 极限密集 QA，可能噪声变大 |

---

### A2.5 空结果表

| Run ID | QA/image | basic | loc | uncertain/ans | CF | SHUF | CheXpert AUC | NIH AUC | Hard shuffle delta | CF acc | Leakage % | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BalancedMix-QA8 | 8 | 20 | 20 | 20 | 25 | 15 |  |  |  |  |  |  |
| CF-heavy-QA8 | 8 | 10 | 15 | 15 | 45 | 15 |  |  |  |  |  |  |
| SHUF-heavy-QA8 | 8 | 10 | 15 | 10 | 25 | 40 |  |  |  |  |  |  |
| Clinical-rich-QA8 | 8 | 15 | 25 | 25 | 20 | 15 |  |  |  |  |  |  |
| StoryMix-QA5 | 5 | 10 | 20 | 15 | 30 | 25 |  |  |  |  |  |  |
| StoryMix-QA8 | 8 | 10 | 20 | 15 | 30 | 25 |  |  |  |  |  |  |
| StoryMix-QA10 | 10 | 10 | 20 | 15 | 30 | 25 |  |  |  |  |  |  |
| StoryMix-QA12 | 12 | 10 | 20 | 15 | 30 | 25 |  |  |  |  |  |  |

---

## A3. Workflow / Curriculum：把分散实验串成完整训练流程

### A3.1 为什么要做？

目前 P3/P4/CF/SHUF/QA8/P5 主要是单独实验。

单独实验回答的是：

```text
这个因素有没有用？
```

但最终论文主方法不能像“堆 tricks”。

最终主方法应该是一个 workflow：

```text
基础医学 QA
→ 反事实医学陈述
→ hard image-shuffle
→ optional token weighting
```

也就是：

```text
由易到难，逐步增加视觉依赖。
```

---

### A3.2 三种连接方式

#### 方式 A：Single-stage mixture

一开始就混合不同 QA：

```text
StoryMix-QA8:
basic QA 10%
location 20%
uncertainty 15%
CF 30%
SHUF 25%
```

优点：

- 简单；
- 好复现；
- 不需要阶段切换；
- 论文比较容易写。

缺点：

- 模型一开始就遇到 hard SHUF，可能比较难。

---

#### 方式 B：Hard curriculum

严格分阶段：

```text
Stage 1: P3-rich QA
Stage 2: standardized CF
Stage 3: SHUF
Stage 4: optional TW
```

优点：

- 最符合“教学”逻辑；
- 先基础，后困难；
- 论文故事非常清楚。

缺点：

- 阶段切换可能导致遗忘；
- 训练时间更长；
- 每个 stage 步数要调。

---

#### 方式 C：Progressive mixture

不是硬切，而是比例逐渐变化。

推荐。

总共 8000 steps 示例：

| Step range | basic QA | location | uncertainty | CF | SHUF |
|---|---:|---:|---:|---:|---:|
| 0-2000 | 35% | 25% | 20% | 20% | 0% |
| 2000-5000 | 20% | 20% | 15% | 35% | 10% |
| 5000-8000 | 10% | 15% | 10% | 30% | 35% |

优点：

- 先易后难；
- 不完全丢掉前面 QA；
- 比 hard curriculum 更稳；
- 比 single-stage 更有机制故事。

---

### A3.3 必跑 workflow 实验

| Run ID | 训练方式 | Steps | 目的 |
|---|---|---:|---|
| Direct-SHUF-3k | 直接 SHUF | 已有 | 当前强 baseline |
| Mix-Story-QA8 | 单阶段 StoryMix | 5000/8000 | 看简单混合是否够 |
| CUR-P3-SHUF | P3-rich → SHUF | 8000 | 看基础 QA warmup 是否有用 |
| CUR-CF-SHUF | CF → SHUF | 8000 | 看 CF warmup 是否有用 |
| CUR-P3-CF-SHUF | P3-rich → CF → SHUF | 8000 | 最完整 curriculum |
| PROG-Mix | progressive mixture | 8000 | 最推荐主方法候选 |
| PROG-Mix-TW | progressive mixture + TW-visual | 8000 | 最强候选 |

---

### A3.4 推荐 stage 配置

#### CUR-P3-CF-SHUF

| Stage | Steps | Data | Loss |
|---|---:|---|---|
| Stage 1 | 1600 | P3-rich QA | answer-only CE |
| Stage 2 | 2400 | D6 hard CF | answer-only CE + answer margin |
| Stage 3 | 4000 | D7 SHUF | answer-only CE + image-shuffle margin |

#### PROG-Mix

| Stage | Steps | Sampling |
|---|---:|---|
| Stage 1 | 0-2000 | basic/location/uncertainty-heavy |
| Stage 2 | 2000-5000 | CF-heavy |
| Stage 3 | 5000-8000 | SHUF-heavy |

#### PROG-Mix-TW

同 PROG-Mix，但加入 TW-visual：

```text
A/B/Yes/No token weight = 1.5-2.0
left/right/location token weight = 2.0
present/absent/uncertain token weight = 1.5
template token weight = 0.5
punctuation weight = 0.1-0.2
```

---

### A3.5 空结果表

| Run ID | Workflow | Steps | CheXpert AUC | NIH AUC | CF acc | Random shuffle delta | Hard shuffle delta | Paraphrase delta | Cost | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Direct-SHUF-3k | direct | 5000 | 0.7267 | 0.5680 | 0.8707 | 0.0716 | 0.0807 |  |  | baseline |
| Mix-Story-QA8 | single-stage |  |  |  |  |  |  |  |  |  |
| CUR-P3-SHUF | 2-stage |  |  |  |  |  |  |  |  |  |
| CUR-CF-SHUF | 2-stage |  |  |  |  |  |  |  |  |  |
| CUR-P3-CF-SHUF | 3-stage |  |  |  |  |  |  |  |  |  |
| PROG-Mix | progressive |  |  |  |  |  |  |  |  |  |
| PROG-Mix-TW | progressive + TW |  |  |  |  |  |  |  |  |  |

---

## A4. SHUF++：比当前 SHUF 更强的图像约束

当前 SHUF-3k 很强，但还可以继续加强。

---

### A4.1 Multi-negative SHUF

当前 SHUF 大概率是：

```text
1 positive image + 1 hard negative image
```

可以增强为：

```text
1 positive image + K hard negative images
```

K 可以试：

| Run ID | Negative 数量 |
|---|---:|
| SHUF-K1 | 1，当前 |
| SHUF-K2 | 2 |
| SHUF-K4 | 4 |
| SHUF-K8 | 8，较贵 |

推荐先试 K=2 / K=4。

训练目标：

```text
loss(correct image, question, answer)
<
loss(each negative image, question, answer)
```

空表：

| Run ID | K | CheXpert AUC | NIH AUC | Hard shuffle delta | CF acc | Cost | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| SHUF-K1 | 1 |  |  |  |  |  |  |
| SHUF-K2 | 2 |  |  |  |  |  |  |
| SHUF-K4 | 4 |  |  |  |  |  |  |
| SHUF-K8 | 8 |  |  |  |  |  |  |

---

### A4.2 In-batch SHUF

一个 batch 内其他图像当负样本。

对于第 i 个问题：

```text
positive image = image_i
negative candidates = image_j in same batch
```

只选答案不同/属性相反的作为 hard negatives。

优点：

- 不需要提前为每条样本存很多 hard negative；
- 动态负样本更丰富。

缺点：

- batch 内未必有合适负样本；
- 需要采样逻辑；
- batch size 太小可能效果有限。

---

### A4.3 Same-question different-answer SHUF

这是非常重要的干净测试。

完全相同的问题，不同图像答案不同。

例子：

```text
Question:
Which statement is better supported?
A. There is a left pleural effusion.
B. There is a right pleural effusion.

Image 1 answer: A
Image 2 answer: B
```

这个最大限度排除文本 shortcut，因为 question 完全一样。

Run：

| Run ID | 数据 | 目的 |
|---|---|---|
| SAMEQ-SHUF-3k | 同问题不同图不同答案 | 最干净 image-specific grounding |
| SAMEQ-SHUF-10k | 扩大规模 | 看是否能提升 NIH |
| SAMEQ-SHUF-K4 | 多负样本 | 最强版本 |

---

### A4.4 Mined-SHUF

步骤：

1. 先训练 SHUF-3k；
2. 用 SHUF-3k 抽 training image embeddings；
3. 对每张图找 embedding 最相似但答案不同的图；
4. 用这些模型最容易混淆的负图继续训练。

这叫：

```text
Mined-SHUF
```

优点：

- 负样本不是人工规则 hard，而是模型真的混淆；
- 更像 self-hard negative mining。

缺点：

- 要多一步 embedding extraction；
- 需要避免 false negative；
- 训练更复杂。

---

### A4.5 Confidence-based SHUF / Self-hard SHUF

先用当前模型计算：

```text
loss(wrong image, question, correct answer)
```

如果 wrong image loss 也很低，说明模型被错图骗了。

把这些样本加权采样。

这叫：

```text
Self-hard SHUF
```

优点：

- 针对模型真正弱点训练；
- 可能比固定规则负样本更有效。

---

### A4.6 Dual-CF-SHUF

四元组：

```text
I+ = correct image
I- = hard negative image
A+ = correct answer
A- = counterfactual answer
Q = same question
```

约束：

```text
loss(I+, Q, A+) 最低
loss(I+, Q, A-) 更高
loss(I-, Q, A+) 更高
```

这同时约束：

1. 图像要对；
2. 答案要对；
3. 图像和答案要匹配。

Run：

| Run ID | Loss |
|---|---|
| DUAL-CF-SHUF-light | CE + answer margin |
| DUAL-CF-SHUF-image | CE + image margin |
| DUAL-CF-SHUF-full | CE + answer margin + image margin |

---

### A4.7 Progressive hard-negative schedule

负样本难度逐步提高：

| Stage | Negative difficulty |
|---|---|
| 0-20% | random negative |
| 20-50% | same finding opposite state |
| 50-80% | same finding opposite laterality |
| 80-100% | mined hard negative / same-question different-answer |

这可以和 PROG-Mix 合并成最终强方法。

---

### A4.8 SHUF++ 空结果表

| Run ID | Key idea | CheXpert AUC | NIH AUC | CF acc | Random shuffle delta | Hard shuffle delta | Cost | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| SHUF-K2 | multi-negative K=2 |  |  |  |  |  |  |  |
| SHUF-K4 | multi-negative K=4 |  |  |  |  |  |  |  |
| InBatch-SHUF | batch negative |  |  |  |  |  |  |  |
| SAMEQ-SHUF | same question, different answer |  |  |  |  |  |  |  |
| Mined-SHUF | embedding-mined negative |  |  |  |  |  |  |  |
| SelfHard-SHUF | confidence-mined negative |  |  |  |  |  |  |  |
| DUAL-CF-SHUF | image + answer margin |  |  |  |  |  |  |  |
| Progressive-HardNeg | negative difficulty schedule |  |  |  |  |  |  |  |

---

## A5. Token weighting：P5 的升级版本

### A5.1 为什么还要做？

旧 P5 mixed：

```text
AUC 不如 P4，但 F1 更高。
```

所以它不能单独当主方法。

但它可以作为 SHUF / curriculum 后的增强：

```text
SHUF = 逼模型看对图
Token weighting = 让 loss 聚焦医学关键答案 token
```

两者不冲突。

---

### A5.2 三类权重设计

#### TW-role：按 token 功能

| Token 类型 | 例子 | 权重 |
|---|---|---:|
| 选项 token | A / B / Yes / No | 1.5 |
| 状态 token | present / absent / uncertain | 1.5 |
| 位置 token | left / right / bilateral | 1.8 |
| finding token | pneumothorax / effusion | 1.2 |
| 模板词 | there / is / shows | 0.8 |
| 标点 | . , : | 0.2 |

安全、稳，适合第一版。

---

#### TW-visual：按视觉依赖

| Token / QA 类型 | 权重 |
|---|---:|
| SHUF answer token | 2.0 |
| CF answer token | 1.8 |
| laterality/location token | 2.0 |
| state token | 1.5 |
| finding token | 1.2 |
| template words | 0.5 |

最推荐，因为和论文主线一致：

```text
视觉依赖越强，loss 权重越高。
```

---

#### TW-clinical-balanced：医学 + 稀有度

```text
final_weight = token_role_weight × field_balance_weight
field_balance_weight = min(2.0, 1 / sqrt(field_frequency_normalized))
```

适合作为 ablation。

谨慎点：不要把主方法建立在主观 disease-priority 上，否则容易被问“权重凭什么”。

---

### A5.3 必跑实验

| Run ID | Base | Weighting | 目的 |
|---|---|---|---|
| SHUF-3k | D7 | none | 当前 baseline |
| SHUF-TW-role | D7 | TW-role | 看轻量 token weighting 是否稳 |
| SHUF-TW-visual | D7 | TW-visual | 最推荐 |
| SHUF-TW-clinical | D7 | TW-clinical-balanced | 医学/稀有度先验 |
| PROG-Mix-TW-role | PROG-Mix | TW-role | curriculum + 轻量 |
| PROG-Mix-TW-visual | PROG-Mix | TW-visual | 最强候选 |
| SHUF-K4-TW-visual | SHUF++ | TW-visual | 多负样本 + 权重 |

---

### A5.4 判断标准

Token weighting 能进主方法的条件：

| 指标 | 要求 |
|---|---|
| CheXpert AUC | 不低于 SHUF-3k，最好 +0.005 |
| NIH AUC | 不明显下降 |
| Hard shuffle delta | 不低于 SHUF-3k |
| CF acc | 不低于 0.85 |
| Macro-F1 | 上升是加分 |
| Calibration | 上升是加分 |

如果出现：

```text
F1 上升，但 AUC / hard shuffle delta 下降
```

则只能放 ablation。

---

# Part B. 其他仍然值得拓展的方向

下面这些不是你文件里必须做的，但我认为有价值，尤其如果你愿意花时间多尝试。

---

## B1. SHUF-10k-8k：扩大最强路线

### 为什么做？

SHUF-3k 已经是当前最强，但还没有 10k / full scale。

建议优先扩：

```text
SHUF-10k-8k
```

而不是继续只扩 D6 CF。

### 实验表

| Run ID | Images | QA/image | Steps | 数据 | 目的 |
|---|---:|---:|---:|---|---|
| SHUF-3k-5k | 3k | 4-6 | 5000 | D7 | 已有 |
| SHUF-10k-8k | 10k | 4-6 | 8000 | D7 | 主扩展 |
| SHUF-10k-QA8 | 10k | 8 | 8000 | StoryMix/D7 | 看更多 QA |
| SHUF-full | full | 3-5 | 8k-12k | D7 | 最终版本 |

空结果表：

| Run ID | CheXpert AUC | NIH AUC | Hard shuffle delta | CF acc | Cost | Decision |
|---|---:|---:|---:|---:|---:|---|
| SHUF-3k-5k | 0.7267 | 0.5680 | 0.0807 | 0.8707 |  | baseline |
| SHUF-10k-8k |  |  |  |  |  |  |
| SHUF-10k-QA8 |  |  |  |  |  |  |
| SHUF-full |  |  |  |  |  |  |

---

## B2. Training policy ablation

现在默认是：

```text
train vision tower + connector
freeze LLM
```

但可以试更省/更强的版本。

| Run ID | Vision tower | Connector | LLM | 目的 |
|---|---|---|---|---|
| TRAIN-CONN | frozen | train | frozen | 只训 connector 是否够 |
| TRAIN-LAST4 | last 4 blocks train | train | frozen | 低成本视觉适配 |
| TRAIN-FULLVISION | full train | train | frozen | 当前主线 |
| TRAIN-VISION-LORA | LoRA | train | frozen | 省显存 |
| TRAIN-LLM-LORA | full train | train | LoRA | 上限，不优先 |

空表：

| Run ID | Trainable params | Peak VRAM | GPU hours | CheXpert AUC | NIH AUC | Hard shuffle delta | CF acc | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| TRAIN-CONN |  |  |  |  |  |  |  |  |
| TRAIN-LAST4 |  |  |  |  |  |  |  |  |
| TRAIN-FULLVISION |  |  |  |  |  |  |  |  |
| TRAIN-VISION-LORA |  |  |  |  |  |  |  |  |
| TRAIN-LLM-LORA |  |  |  |  |  |  |  |  |

---

## B3. Model scale / model type ablation

如果资源允许，比较不同 VLM 或 LLM route。

| Run ID | Model | Route | 目的 |
|---|---|---|---|
| Qwen3VL-2B-SHUF | Qwen3-VL-2B | VLM-coupled | 当前主线 |
| Qwen3VL-2B-SHUF-TW | Qwen3-VL-2B | VLM-coupled + TW | 主候选 |
| Qwen3.5-2B-text-scaffold | text-only | old scaffold control | 证明一套 VLM 是否必要 |
| old Qwen-Coder scaffold | text-only coder | old baseline | 验证 coder/template bias |
| larger VLM if available | VLM-coupled | scale upper bound | 看更大是否更好或更差 |

注意：

> Qwen3.5 text-only 不能写成 VLM 主方法，只能作为 control。

---

## B4. External transfer

当前 NIH 基本持平，没有大幅提升。

如果要增强论文，需要更多外部验证。

优先级：

| Dataset | 作用 |
|---|---|
| NIH full / larger subset | 先扩大当前 external |
| MIMIC-CXR | 最推荐外部 CXR |
| PadChest | 外部泛化 |
| VinDr-CXR | 临床标注质量较高 |
| CheXpert different split | 检查 split sensitivity |

表：

| Run ID | CheXpert AUC | NIH AUC | MIMIC AUC | PadChest AUC | VinDr AUC | Decision |
|---|---:|---:|---:|---:|---:|---|
| Base |  |  |  |  |  |  |
| SHUF-3k |  |  |  |  |  |  |
| StoryMix-QA8 |  |  |  |  |  |  |
| PROG-Mix-TW |  |  |  |  |  |  |
| SHUF-10k |  |  |  |  |  |  |

---

## B5. Calibration / threshold / AUPRC

AUC 不够全面，尤其医学数据长尾严重。

建议补：

| Metric | 目的 |
|---|---|
| AUPRC | rare positives 更敏感 |
| ECE | 校准 |
| Brier score | 概率质量 |
| per-field threshold F1 | 看阈值敏感性 |
| high-null calibration | answerability/null 语义 |
| subgroup AUC | rare/high-null/uncertain |

表：

| Run ID | Macro-AUC | Macro-AUPRC | ECE | Brier | Rare AUPRC | High-null ECE | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Base |  |  |  |  |  |  |  |
| SHUF-3k |  |  |  |  |  |  |  |
| PROG-Mix |  |  |  |  |  |  |  |
| PROG-Mix-TW |  |  |  |  |  |  |  |

---

## B6. Prompt robustness / option bias

必须防止模型只学 A/B 选项 bias。

检查：

| Diagnostic | 目的 |
|---|---|
| swap A/B order | 看是不是偏 A |
| paraphrase question | 看是不是背模板 |
| remove disease name | 看视觉证据是否仍有用 |
| style rewrite | 看语言稳健性 |
| same question different image | 看 image-specific |
| same image different question | 看 finding-specific |

表：

| Run ID | A/B swap acc | Paraphrase acc | SameQ diff-image acc | Same-image diff-question acc | Template sensitivity | Decision |
|---|---:|---:|---:|---:|---:|---|
| SHUF-3k |  |  |  |  |  |  |
| StoryMix |  |  |  |  |  |  |
| PROG-Mix |  |  |  |  |  |  |

---

## B7. Leakage audit 2.0

所有 QA / SHUF 数据都要持续做 leakage audit。

自动规则：

| Check | Reject / Flag |
|---|---|
| question contains evidence_span | reject |
| question contains exact answer | reject |
| question contains “report says” | reject for training |
| question asks location and includes location | flag/reject |
| question asks severity and includes severity | flag/reject |
| A/B correct always A | reject/rebalance |
| A/B option length imbalance | flag |
| duplicate question for same image | downsample |
| answer inferable from question alone | flag |

空表：

| Dataset | Instructions | Accepted % | Leakage % | A/B balance | Evidence-span pass | Duplicate % | Manual correct % | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| D6-3k |  |  |  |  |  |  |  |  |
| D7-3k |  |  |  |  |  |  |  |  |
| StoryMix-QA8 |  |  |  |  |  |  |  |  |
| PROG-Mix |  |  |  |  |  |  |  |  |

---

## B8. Qualitative visualization

为了论文更有说服力，可以加可视化。

| Method | 目的 |
|---|---|
| attention map | 看模型是否关注相关区域 |
| Grad-CAM / token attribution | 看 finding 对应图像区域 |
| positive vs hard negative saliency | 看 SHUF 是否改变关注点 |
| failure cases | 展示模型什么时候还错 |

尤其适合 laterality / pneumothorax / effusion / cardiomegaly。

表：

| Run ID | Sample | Question | Expected region | Model attention matches? | Notes |
|---|---|---|---|---|---|
| SHUF-3k |  |  |  |  |  |
| PROG-Mix |  |  |  |  |  |
| PROG-Mix-TW |  |  |  |  |  |

---

# Part C. 推荐执行顺序

下面是我建议真正执行的顺序。

---

## Phase 0：数据和诊断准备

| Priority | Task | 输出 | Done |
|---:|---|---|---|
| 0 | Leakage audit 2.0 | audit table |  |
| 0 | D6/D7 数据分布统计 | distribution table |  |
| 0 | P2-value-only data mask 实现 | dataset/loss mask |  |
| 0 | StoryMix 数据生成器 | StoryMix jsonl |  |
| 0 | SameQ-SHUF 生成器 | same-question jsonl |  |
| 0 | SHUF-K negative sampler | K-neg jsonl |  |

---

## Phase 1：快速诊断实验

先跑这些小而关键的：

| Run ID | 目的 |
|---|---|
| P2-value-only | 判断 JSON template loss 是否有害 |
| P2-field-query | 判断字段级 QA 是否优于完整 JSON |
| StoryMix-QA8 | 主 mixture 初测 |
| SHUF-heavy-QA8 | 追求 grounding |
| CF-heavy-QA8 | 追求 CF acc |
| SAMEQ-SHUF-3k | 最干净 image-specific test |
| SHUF-TW-visual | 看 TW 是否能增强 SHUF |

---

## Phase 2：workflow 实验

| Run ID | 目的 |
|---|---|
| Mix-Story-QA8 | 单阶段混合 |
| CUR-P3-CF-SHUF | hard curriculum |
| PROG-Mix | progressive curriculum |
| PROG-Mix-TW | 最强候选 |
| PROG-Mix-SAMEQ | progressive + same-question |
| PROG-Mix-DualMargin | progressive + image/answer dual margin |

---

## Phase 3：SHUF++ 扩展

| Run ID | 目的 |
|---|---|
| SHUF-K2 | 多负样本初测 |
| SHUF-K4 | 强多负样本 |
| InBatch-SHUF | 动态负样本 |
| Mined-SHUF | embedding hard mining |
| SelfHard-SHUF | confidence hard mining |
| DUAL-CF-SHUF | 图像+答案双 margin |

---

## Phase 4：scale 和 external

| Run ID | 目的 |
|---|---|
| SHUF-10k-8k | 扩大当前最强路线 |
| StoryMix-10k-8k | 扩大 mixture |
| PROG-Mix-10k-8k | 最终 workflow scale |
| PROG-Mix-TW-10k | 最强最终候选 |
| NIH full / MIMIC external | 外部泛化 |

---

# Part D. 最推荐的下一批 12 个实验

如果你说“时间不怕久，我想多试”，我建议下一批先做这 12 个。

| Priority | Run ID | 为什么做 |
|---:|---|---|
| 1 | P2-value-only | 解释 fixed JSON 差是不是模板 loss |
| 2 | P2-field-query | 去掉 JSON 顺序，保留字段级监督 |
| 3 | StoryMix-QA8 | 最适合论文故事的 mixture |
| 4 | SHUF-heavy-QA8 | 看更偏 SHUF 是否继续提升 grounding |
| 5 | CF-heavy-QA8 | 保住最高 CF acc |
| 6 | SAMEQ-SHUF-3k | 最干净验证 image-specific |
| 7 | SHUF-TW-visual | 看 token weighting 能否增强当前最强 SHUF |
| 8 | CUR-P3-CF-SHUF | 验证先易后难 curriculum |
| 9 | PROG-Mix | 最推荐主方法候选 |
| 10 | PROG-Mix-TW | 最强主方法候选 |
| 11 | SHUF-K4 | 多负样本是否继续提升 |
| 12 | SHUF-10k-8k | 扩大当前最强路线 |

---

# Part E. 最终方法候选

最终主方法应该从下面几个里面选，而不是提前确定。

---

## Candidate 1：Direct SHUF

```text
Qwen3-VL + D7 hard image-shuffle
```

优点：

- 当前最好；
- 简单；
- 证据最强。

缺点：

- 不够像完整课程；
- QA mixture 设计不够丰富。

---

## Candidate 2：StoryMix-QA8

```text
Qwen3-VL + rich QA mixture
```

优点：

- 论文故事好；
- 医疗相关性强；
- 不只是 hard negative。

缺点：

- 可能不如 SHUF 强。

---

## Candidate 3：Progressive-Mix

```text
Stage 1: basic clinical QA
Stage 2: hard A/B CF
Stage 3: SHUF
```

优点：

- 最像“从语言理解到视觉 grounding”的过程；
- 写作非常自然；
- 不是 trick 堆叠。

缺点：

- 训练复杂；
- 需要证明比 direct SHUF 更好。

---

## Candidate 4：Progressive-Mix + TW-visual

```text
Progressive-Mix
+ visual-token weighting
```

优点：

- 机制最完整；
- 可能最强。

缺点：

- 如果没超过 SHUF，会显得复杂。

---

## Candidate 5：SHUF++

```text
SHUF + multi-negative / same-question / mined negative / dual margin
```

优点：

- 最强 image-specific grounding 方向；
- 很适合解释 previous image-shuffle weakness。

缺点：

- 成本高；
- 实现复杂；
- 要控制 negative 质量。

---

# Part F. 选择最终方法的规则

最终主方法不是看单个指标。

建议使用硬规则：

| 指标 | 最低要求 |
|---|---|
| CheXpert AUC | ≥ SHUF-3k - 0.003 |
| NIH AUC | 不低于 SHUF-3k 明显超过 0.005 |
| Hard shuffle delta | ≥ 0.08 或至少 > 0.05 |
| CF acc | ≥ 0.85 |
| Leakage rate | ≤ 5%-10% |
| A/B balance | 45%-55% |
| Cost | 可解释，不爆炸 |
| 论文可解释性 | 必须比 direct SHUF 更清楚或更强 |

如果一个方法只提高 F1，不能进主方法。

如果一个方法只提高 CF acc，但 AUC 和 shuffle 下降，只放 ablation。

如果一个方法同时提高：

```text
CheXpert AUC
hard shuffle delta
CF acc 或 NIH AUC
```

就可以作为主方法候选。

---

# Part G. Codex / 实验同学执行清单

## G1. 数据生成脚本

需要新增或扩展：

```text
scripts/generate_storymix_instructions.py
scripts/generate_sameq_shuf_pairs.py
scripts/generate_multi_negative_shuf.py
scripts/audit_instruction_leakage_v2.py
scripts/build_progressive_mixture_schedule.py
scripts/build_token_weight_map.py
scripts/mine_hard_negatives_from_embeddings.py
```

---

## G2. 训练脚本需要支持

| 功能 | 说明 |
|---|---|
| dataset mixture ratio | 每类 QA 按比例采样 |
| curriculum schedule | stage-wise / progressive sampling |
| answer-only loss | 只算 answer |
| value-only JSON loss | fixed JSON 只算 value token |
| token weighting | 按 token role / visual dependency |
| image-shuffle margin | 正图 vs 错图 |
| answer margin | 正确答案 vs 错误答案 |
| multi-negative loss | 一个正图多个负图 |
| in-batch negative | batch 内负样本 |
| checkpoint provenance | 记录每个 stage / mixture |
| diagnostics export | shuffle/CF/paraphrase |

---

## G3. 每个 run 必须输出

```text
config_snapshot.json
metrics_final.json
metrics_step_*.json
progress.json
training_log.txt
vision_export_manifest.json
lp_results.md
visual_dependence_results.md
counterfactual_results.md
paraphrase_results.md
instruction_audit.md
cost_table.md
```

---

# Part H. 论文写作主线

最终如果结果支持，可以这样写：

```text
We find that not all language supervision improves deployable CXR vision encoders. Fixed schema generation can collapse into template learning, and simple report-grounded QA provides only limited image-specific grounding. We therefore organize report-derived clinical instructions by their visual dependence: basic clinical QA, standardized counterfactual choices, and hard image-shuffle pairs. This progression turns language supervision into image-specific clinical visual supervision. Among these, hard image-shuffle supervision provides the strongest gains in both CheXpert representation quality and image-mismatch sensitivity.
```

中文：

```text
我们发现，并不是所有语言监督都会提升胸片视觉表征。固定 schema 生成容易退化成模板学习，普通 report QA 也不一定真正依赖具体图像。因此我们按照“视觉依赖强度”组织 clinical instruction：基础医学 QA、标准化反事实选择题、hard image-shuffle 图像匹配题。这个逐步增强的设计让语言监督真正变成图像特异的临床视觉监督。其中 hard image-shuffle 对 CheXpert 表征质量和 image-mismatch sensitivity 提升最明显。
```

---

# Part I. 最终下一步建议

我的建议是：

## 先做 4 个最短闭环

```text
P2-value-only
P2-field-query
StoryMix-QA8
SAMEQ-SHUF-3k
```

这 4 个能快速回答：

1. JSON 差是不是模板 loss；
2. 字段级 QA 是否比完整 JSON 好；
3. rich QA mixture 是否优于单独 SHUF/CF；
4. same-question different-image 是否是更干净的 grounding 方式。

## 然后做 4 个主方法候选

```text
SHUF-TW-visual
CUR-P3-CF-SHUF
PROG-Mix
PROG-Mix-TW
```

这 4 个能决定最终 workflow。

## 最后扩两个大规模版本

```text
SHUF-10k-8k
PROG-Mix-TW-10k
```

这两个用于最终主表。

---

# Appendix A. 最终总结表

本表为 reader-facing 摘要；完整 provenance 以 `outputs/final_tables/*.csv`、per-run package markers、以及 `docs/next_stage_requirement_ledger.md` 为准。`Random shuffle delta` 对应 final visual-dependence 表中的 `image_shuffle_delta`，`Paraphrase robustness` 对应 clinical paraphrase delta，`Cost` 表示该 run 的 package 内已生成 `cost_table.md` marker。

| Group | Run ID | Status | CheXpert AUC | NIH AUC | CF acc | Random shuffle delta | Hard shuffle delta | Paraphrase robustness | Leakage % | Cost | Final role |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| JSON mask | P2-value-only | completed | 0.704 | 0.587493 |  | 0.022972 |  | 0.001759 | 0 | cost_table.md | pending_or_ablation |
| JSON mask | P2-field-query | completed | 0.670238 | 0.583546 |  | 0.018829 |  | 0.007309 | 0 | cost_table.md | pending_or_ablation |
| Mixture | StoryMix-QA8 | completed | 0.708764 | 0.577581 | 0.863636 | 0.045861 | -0.139533 | 0.018317 | 26.43 | cost_table.md | pending_or_ablation |
| Mixture | SHUF-heavy-QA8 | completed | 0.694846 | 0.566605 | 0.86067 | 0.079567 | -0.024161 | 0.024849 | 26.9283 | cost_table.md | pending_or_ablation |
| Mixture | CF-heavy-QA8 | completed | 0.702444 | 0.578461 | 0.852837 | 0.084438 | 0.016794 | 0.02033 | 26.1782 | cost_table.md | pending_or_ablation |
| Workflow | CUR-P3-CF-SHUF | completed | 0.689972 | 0.571986 | 0.857708 | 0.043706 | -0.044076 | 0.026571 | 51.275 | cost_table.md | pending_or_ablation |
| Workflow | PROG-Mix | completed | 0.692702 | 0.575148 | 0.847826 | 0.034479 | -0.131945 | 0.032519 | 56.2833 | cost_table.md | pending_or_ablation |
| Workflow | PROG-Mix-TW | completed | 0.686085 | 0.557968 | 0.873518 | 0.029744 | -0.155055 | 0.032883 | 56.2833 | cost_table.md | pending_or_ablation |
| SHUF++ | SAMEQ-SHUF-3k | completed | 0.741002 | 0.578871 |  | 0.391468 | 0.439227 | -0.002994 | 8.3243 | cost_table.md | pending_or_ablation |
| SHUF++ | SHUF-K4 | completed | 0.729614 | 0.579815 | 0.831066 | 0.236316 | 0.36384 | 0.024644 | 6.5095 | cost_table.md | pending_or_ablation |
| SHUF++ | DUAL-CF-SHUF | completed | 0.682445 | 0.56671 | 0.888889 | 0.076603 | 0.052539 | 0.017036 | 6.342 | cost_table.md | pending_or_ablation |
| Scale | SHUF-10k-8k | completed | 0.683235 | 0.560986 | 0.863636 | 0.055839 | 0.213363 | 0.008335 | 8.2228 | cost_table.md | pending_or_ablation |
| Scale | PROG-Mix-TW-10k | completed | 0.70922 | 0.57286 | 0.884127 | 0.069395 | 0.490834 | 0.010158 | 46.8875 | cost_table.md | pending_or_ablation |

---

# Appendix B. 最终优先级一句话

如果只能按一个路线推进：

```text
先做 StoryMix-QA8 和 SAMEQ-SHUF-3k，
再做 PROG-Mix-TW，
最后把最强版本扩到 10k。
```

如果目标是最快提升主结果：

```text
优先做 SHUF-10k-8k。
```

如果目标是最强论文故事：

```text
优先做 PROG-Mix-TW 和 SAMEQ-SHUF。
```

如果目标是解释 P2 为什么差：

```text
优先做 P2-value-only 和 P2-field-query。
```

---

# Final Execution Closure (2026-07-01)

本计划已按 artifact-backed 标准完成，不再以计划文字或历史表格作为完成证据。最终全量打包、汇总与审计已经刷新：

- Completion audit: `outputs/final_tables/next_stage_completion_audit.csv`
- Completion audit result: `1049 / 1049` rows completed, `0` missing, `0` pending
- Training matrix: `outputs/final_tables/next_stage_training_results.csv`
- LP / NIH transfer: `outputs/final_tables/next_stage_lp_transfer_results.csv`
- Visual-dependence: `outputs/final_tables/next_stage_visual_dependence.csv`
- Primary counterfactual: `outputs/final_tables/next_stage_counterfactual.csv`
- A/B-swap counterfactual: `outputs/final_tables/next_stage_ab_swap_counterfactual.csv`
- Paraphrase robustness: `outputs/final_tables/next_stage_paraphrase.csv`
- Leakage / instruction audit: `outputs/final_tables/next_stage_instruction_audit.csv`
- Calibration / AUPRC: `outputs/final_tables/next_stage_calibration_auprc.csv`
- Final decision table: `outputs/final_tables/next_stage_decision_summary.csv`
- Qualitative casebook: `outputs/final_tables/next_stage_qualitative_cases.md`
- Requirement ledger: `docs/next_stage_requirement_ledger.md`

## Final Gate Result

The strict final-method gate selects `SHUF-TW-clinical` as the only current `candidate`:

| Run | CheXpert AUC | NIH AUC | Hard shuffle delta | CF acc | A/B-swap acc | Leakage/flag % | A% | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SHUF-TW-clinical | 0.735671 | 0.570359 | 0.139729 | 0.879819 | 0.879819 | 6.342 | 49.9773 | candidate |

Strong ablation / diagnostic runs that support the paper story but do not pass every strict gate:

| Run | CheXpert AUC | NIH AUC | Hard shuffle delta | CF acc | A/B-swap acc | Boundary |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SAMEQ-SHUF-3k | 0.741002 | 0.578871 | 0.439227 |  |  | Strong grounding/shuffle evidence; no CF/A-B option diagnostic gate for this route. |
| SHUF-K4 | 0.729614 | 0.579815 | 0.363840 | 0.831066 | 0.841270 | Strong shuffle/transfer evidence; CF and A/B-swap below strict 0.85 gate. |
| SHUF-K4-TW-visual | 0.727916 | 0.580787 | 0.321406 | 0.817460 | 0.828798 | Strong visual-dependence/transfer evidence; CF and A/B-swap below strict gate. |
| PROG-Mix-10k-8k | 0.720443 | 0.576992 | 0.458131 | 0.879365 | 0.847000 | Strong hard-shuffle and CF; A/B-swap and leakage/flag boundary prevent main-method claim. |
| PROG-Mix-TW-10k | 0.709220 | 0.572860 | 0.490834 | 0.884127 | 0.849000 | Strong hard-shuffle and CF; A/B-swap just below strict gate and leakage/flag boundary prevent main-method claim. |

## Execution Notes

- All 39 manifest runs have training artifacts, LP/NIH transfer metrics, visual-dependence diagnostics, primary counterfactual diagnostics where applicable, A/B-swap diagnostics or explicit zero-row not-applicable evidence, paraphrase diagnostics, instruction audits, and per-run package markers.
- A/B-swap diagnostics are now first-class package and audit evidence. Runs with no A/B-option rows are recorded as `not_applicable_no_ab_rows` rather than false missing diagnostics.
- After the user requested GPU0 be freed, all remaining next-stage work was migrated to GPU1-only worker `gpu1_only_remaining_after_free_gpu0_20260701T004917`; unrelated `outputs/runs/m1_dense/run_train*.py` and `opmem.eval` GPU0 processes that restarted during closeout were stopped to preserve the GPU0-free boundary.
- External-data boundaries remain: NIH 1k transfer is complete for all 39 runs; MIMIC/PadChest/VinDr and larger-VLM formal claims require additional local manifests/model smoke tests beyond the current artifact set.
