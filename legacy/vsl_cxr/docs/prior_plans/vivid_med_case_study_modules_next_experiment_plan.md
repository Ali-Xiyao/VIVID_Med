# VIVID-Med 下一步实验与模块计划：从“综合最优”走向可解释、可复现、可投稿 TMI 的主线

> 版本：2026-07-01  
> 目标：针对当前实验中暴露的 6 个核心问题，重新组织下一阶段实验。  
> 当前态度：**不急着把 SHUF-TW-clinical 定为最终方法**。它现在只是一个候选。下一步先做 case study、seed 稳定性、NIH 跨域诊断、curriculum 训练充分性和模块化增强，再决定主线。  
> 核心原则：**不要让论文看起来像“做了一堆实验，挑一个综合最优”。必须形成一个有临床动机、机制清楚、模块明确、统计稳定、外部验证充分的 pipeline。**

---

## 0. 当前问题重新定调

### 0.1 当前结果给我们的信息

当前实验里，`SHUF-TW-clinical` 被 strict gate 暂时选为 candidate：

| Run | CheXpert AUC | NIH AUC | Hard shuffle delta | CF acc | A/B-swap acc | Leakage/flag % | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| SHUF-TW-clinical | 0.735671 | 0.570359 | 0.139729 | 0.879819 | 0.879819 | 6.342 | candidate |

但是这不是最终结论。主要原因：

1. 它相比 `SHUF-3k` 的 CheXpert AUC 只提升约 `+0.009`，可能被随机种子波动抹平。
2. NIH 上所有方法差距都不大，跨域泛化还没有拉开。
3. Curriculum 方向现在结果不好，但可能是数据量、训练 schedule、leakage 或 stage 设置没做好。
4. 如果论文只说“我们综合选了 SHUF-TW-clinical”，审稿人会觉得像随机试很多然后挑一个。
5. 需要 case study 先找问题，再决定下一步主线。
6. 面向 TMI，最好不只是一个训练 recipe，还要有可部署模块、临床语义模块、跨域鲁棒模块或可解释模块。

---

## 1. 六个问题逐条分析

## 1.1 问题一：SHUF-TW-clinical 相比 SHUF-3k 提升太小

### 现象

| Run | CheXpert AUC | NIH AUC | Hard shuffle delta | CF acc |
|---|---:|---:|---:|---:|
| SHUF-3k | 0.726709 | 0.568045 | 0.080674 | 0.870748 |
| SHUF-TW-clinical | 0.735671 | 0.570359 | 0.139729 | 0.879819 |
| Delta | +0.008962 | +0.002314 | +0.059055 | +0.009071 |

### 大白话解释

CheXpert AUC 确实涨了，但只涨了不到 1 个点。  
这个提升是否可靠，必须看随机种子方差。

如果一个 seed 换一下就能让 AUC 浮动 `±0.008` 或 `±0.010`，那这个提升就不能作为主 claim。  
但 hard shuffle delta 的提升是 `+0.059`，这个比 AUC 更大，说明 `SHUF-TW-clinical` 至少在 image-mismatch sensitivity 上确实有较强信号。

### 结论

现在不能说：

```text
SHUF-TW-clinical 显著优于 SHUF-3k。
```

只能说：

```text
SHUF-TW-clinical 是一个候选方法，它在 CheXpert AUC 和 hard-shuffle sensitivity 上同时优于 SHUF-3k，但需要 multi-seed 和 paired statistical testing 验证。
```

### 必做实验：multi-seed stability

| Run ID | Seeds | 目的 |
|---|---:|---|
| SHUF-3k-seed3 | 3 seeds | 估计 baseline 方差 |
| SHUF-TW-clinical-seed3 | 3 seeds | 估计 candidate 方差 |
| SAMEQ-SHUF-3k-seed3 | 3 seeds | 验证最强 grounding diagnostic 是否稳定 |
| SHUF-K4-seed3 | 3 seeds | 验证 multi-negative 方向是否稳定 |

如果资源允许，seed 数量建议为 5；如果资源有限，至少 3。

### 统计检验

每个方法必须报告：

| Metric | 必须报告 |
|---|---|
| CheXpert AUC | mean ± std |
| NIH AUC | mean ± std |
| Hard shuffle delta | mean ± std |
| CF acc | mean ± std |
| A/B-swap acc | mean ± std |
| Paired bootstrap CI | candidate - baseline |
| Significance / uncertainty | 不一定 p-value，但要有置信区间 |

### 空表

| Run | Seed | CheXpert AUC | NIH AUC | Hard shuffle delta | CF acc | A/B-swap acc | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| SHUF-3k | 0 |  |  |  |  |  |  |
| SHUF-3k | 1 |  |  |  |  |  |  |
| SHUF-3k | 2 |  |  |  |  |  |  |
| SHUF-TW-clinical | 0 |  |  |  |  |  |  |
| SHUF-TW-clinical | 1 |  |  |  |  |  |  |
| SHUF-TW-clinical | 2 |  |  |  |  |  |  |
| SAMEQ-SHUF-3k | 0 |  |  |  |  |  |  |
| SAMEQ-SHUF-3k | 1 |  |  |  |  |  |  |
| SAMEQ-SHUF-3k | 2 |  |  |  |  |  |  |

### 决策规则

| 结果模式 | 决策 |
|---|---|
| SHUF-TW-clinical mean AUC > SHUF-3k by > 1 std | 可以保留为 final candidate |
| AUC 差距落在 seed 方差内，但 hard shuffle delta 稳定更高 | 作为 grounding-enhanced variant，不主打 AUC superiority |
| AUC / hard shuffle 都不稳定 | 降级为 ablation |
| SAMEQ 或 SHUF-K4 多 seed 更稳 | 重新考虑主方法 |

---

## 1.2 问题二：NIH 跨域普遍拉不开差距

### 现象

当前 NIH AUC 大多集中在 `0.56-0.58` 区间，方法之间差距很小。

这可能有几类原因：

1. **NIH 数据本身和 CheXpert/MIMIC 标签空间不完全一致。**
2. **NIH label 噪声或 label mapping 导致方法差异被压缩。**
3. **现在只用了 NIH 1k subset，统计方差较大。**
4. **LP protocol 不够敏感，例如 sample 数、标签映射、pooling 方式、linear head 训练步数。**
5. **胸片域差异主要是 acquisition / label policy，而不是 representation 能轻易解决的视觉差异。**
6. **当前 instruction 主要围绕 CheXpert/MIMIC report 语义，未必对 NIH 的 label definition 最优。**

### 必做：NIH 跨域诊断

#### A. 扩大 NIH 评估规模

| Run | NIH samples | 目的 |
|---|---:|---|
| NIH-1k | 1000 | 当前 baseline |
| NIH-5k | 5000 | 降低方差 |
| NIH-full | full available | 最终外部评估 |

#### B. NIH per-label 分析

| Label | Base AUC | SHUF-3k | SHUF-TW-clinical | SAMEQ | SHUF-K4 | Delta best-base | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Atelectasis |  |  |  |  |  |  |  |
| Cardiomegaly |  |  |  |  |  |  |  |
| Effusion |  |  |  |  |  |  |  |
| Infiltration / Opacity |  |  |  |  |  |  |  |
| Pneumothorax |  |  |  |  |  |  |  |
| Mass/Nodule |  |  |  |  |  |  |  |
| Consolidation |  |  |  |  |  |  |  |
| Edema |  |  |  |  |  |  |  |

#### C. Label mapping audit

需要明确：

| CheXpert/MIMIC field | NIH label | Mapping confidence | Risk |
|---|---|---|---|
| Lung Opacity | Infiltration / opacity? | medium/low | definition mismatch |
| Pleural Effusion | Effusion | high | ok |
| Pneumothorax | Pneumothorax | high | ok |
| Cardiomegaly | Cardiomegaly | high | ok |
| Edema | Edema | medium | label noise |
| Consolidation | Consolidation | medium | overlap with opacity |

#### D. Feature-domain shift analysis

抽取 vision embeddings，分析：

| Analysis | 目的 |
|---|---|
| t-SNE / UMAP by dataset | 看 CheXpert vs NIH 是否分开 |
| MMD distance | 量化 domain gap |
| embedding norm / variance | 看不同数据集特征分布 |
| label-conditioned feature shift | 看同一 label 下两个数据集是否分布不同 |
| LP calibration difference | 看模型是否概率失准 |

#### E. NIH protocol sanity check

| Check | Why |
|---|---|
| same image preprocessing | 避免 resizing/windowing 差异 |
| same label subset | 避免 label mismatch |
| same LP head capacity | 避免 head too weak |
| multiple pooling | Qwen3-VL vision tower pooling 可能影响 |
| multiple seeds | NIH 1k 高方差 |
| no patient leakage | 外部数据通常无泄漏，但仍检查 |

### 空表：NIH 诊断总表

| Run | NIH subset | Pooling | Seed | Macro-AUC | Macro-AUPRC | Macro-F1 | ECE | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Base | 1k | mean | 0 |  |  |  |  |  |
| SHUF-3k | 1k | mean | 0 |  |  |  |  |  |
| SHUF-TW-clinical | 1k | mean | 0 |  |  |  |  |  |
| SAMEQ-SHUF-3k | 1k | mean | 0 |  |  |  |  |  |
| SHUF-K4 | 1k | mean | 0 |  |  |  |  |  |
| Base | full | mean | 0 |  |  |  |  |  |
| SHUF-3k | full | mean | 0 |  |  |  |  |  |
| SHUF-TW-clinical | full | mean | 0 |  |  |  |  |  |

### 可能结论

| 诊断结果 | 解释 |
|---|---|
| NIH full 仍拉不开 | NIH label/protocol 对当前方法不敏感；需要换外部数据 |
| 只有某些 label 拉开 | 写 per-label transfer，不写 overall transfer |
| MIMIC/PadChest 能拉开，NIH 拉不开 | NIH 是 dataset-specific issue |
| 所有外部都拉不开 | 方法主要提升 in-domain + grounding diagnostics，不要写强 external generalization |
| SHUF/SAMEQ 在 external grounding diagnostic 上好，但 label AUC 不好 | representation 学会 matching，但 label transfer 受 mapping/noise 限制 |

---

## 1.3 问题三：Curriculum 方向现在不能放弃，可能是数据量/训练不够

### 现象

当前 curriculum 小规模结果不好，且 leakage/flag 很高。

但这不代表 curriculum 没价值。它可能失败在：

1. 数据量不够；
2. stage 训练步数不合理；
3. stage 切换导致遗忘；
4. 混入的 QA leakage 太高；
5. final stage SHUF 比例不足；
6. 没有用足 MIMIC / CheXpert full report data；
7. LP evaluation 没选到 best checkpoint；
8. curriculum 应该是 progressive，而不是 hard stage。

### 不要现在下结论

不能写：

```text
Curriculum does not work.
```

更合理：

```text
Naive small-scale curriculum did not outperform direct SHUF, but curriculum remains under-optimized and requires leakage-controlled full-data evaluation.
```

### 下一步：Curriculum 重新设计

#### Data scale

| Curriculum Scale | Source data | Images | QA/image | Steps | 用途 |
|---|---|---:|---:|---:|---|
| CUR-3k-clean | cleaned CheXpert/MIMIC subset | 3k | 5-8 | 8k | debug |
| CUR-10k-clean | cleaned subset | 10k | 5-8 | 8k-12k | main |
| CUR-MIMIC-full | all available MIMIC report-image pairs | full | 3-6 | 12k-20k | upper bound |
| CUR-MIX-full | MIMIC + CheXpert | full | 3-6 | 12k-20k | strongest |

#### Curriculum 类型

| Run | Stage design | 目的 |
|---|---|---|
| CUR-hard-stage | P3 → CF → SHUF | 经典先易后难 |
| CUR-progressive | QA/CF/SHUF 比例渐变 | 避免 stage discontinuity |
| CUR-replay | 后 stage 保留 20% 前 stage QA | 防止遗忘 |
| CUR-SHUF-final | 前 40% QA+CF，后 60% SHUF | 强最终 grounding |
| CUR-SAMEQ-final | 后期加入 SAMEQ | 更干净 grounding |
| CUR-mined-final | 后期加入 mined hard negatives | hardest |

### 推荐的 refined curriculum

#### CUR-v2-progressive-replay

| Step range | Basic QA | Location | Uncertainty | CF | SHUF | SAMEQ | Replay |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0-20% | 35 | 25 | 20 | 20 | 0 | 0 | - |
| 20-50% | 20 | 20 | 15 | 35 | 10 | 0 | 20% from stage 1 |
| 50-80% | 10 | 15 | 10 | 30 | 30 | 5 | 15% previous |
| 80-100% | 5 | 10 | 5 | 20 | 35 | 25 | 10% previous |

#### CUR-v2-hardneg

| Step range | Negative difficulty |
|---|---|
| 0-20% | none / random |
| 20-50% | CF state flip |
| 50-80% | same-finding hard shuffle |
| 80-100% | SAMEQ / mined negatives |

### 空表

| Run | Data scale | Steps | Leakage % | CheXpert AUC | NIH AUC | Hard shuffle delta | CF acc | A/B-swap | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CUR-3k-clean | 3k | 8k |  |  |  |  |  |  |  |
| CUR-10k-clean | 10k | 12k |  |  |  |  |  |  |  |
| CUR-MIMIC-full | full | 12k-20k |  |  |  |  |  |  |  |
| CUR-progressive-replay | 10k/full | 12k |  |  |  |  |  |  |  |
| CUR-SAMEQ-final | 10k/full | 12k |  |  |  |  |  |  |  |

### 决策规则

| 结果 | 结论 |
|---|---|
| CUR full 仍不如 SHUF | direct hard grounding 是最有效路线 |
| CUR 提高 NIH 但 CheXpert 不最高 | curriculum 对 external transfer 有意义 |
| CUR 提高 hard shuffle 但 leakage 高 | 数据生成要继续清洗 |
| CUR 同时提高 AUC + hard shuffle + NIH | 可成为主方法 |
| CUR 只提高 CF | 放 ablation，不做主线 |

---

## 1.4 问题四：不能像“综合最优”，要有 pipeline 和 pre-registered selection

### 审稿人风险

如果论文写成：

> 我们做了 39 个实验，最后 SHUF-TW-clinical 综合最优。

审稿人可能会认为：

1. 没有明确假设；
2. 随机试很多，然后挑了一个；
3. 方法没有逻辑；
4. 不是 pipeline，只是 hyperparameter search；
5. 没有统计稳定性；
6. 不是各方面最优。

### 解决办法

需要把实验流程变成：

```text
Case study -> Hypothesis -> Module design -> Candidate families -> Dev selection -> Locked final evaluation
```

不是：

```text
Run 39 variants -> pick one best-looking row
```

### 预注册式 selection framework

#### Step 1：定义 primary endpoints

不要所有指标都当 primary。

建议：

| Level | Metric | Role |
|---|---|---|
| Primary 1 | CheXpert macro-AUC | deployable representation |
| Primary 2 | Hard shuffle delta | image-specific grounding |
| Safety gate | leakage <= 10%, A/B balance 45-55 | data validity |
| Secondary | NIH AUC | external transfer |
| Secondary | CF acc, A/B-swap acc | counterfactual robustness |
| Secondary | AUPRC/ECE | clinical calibration |

#### Step 2：定义候选方法 family

不要把 39 个方法平铺。分成 4 个 family：

| Family | 代表 | 科学假设 |
|---|---|---|
| JSON cleanup | P2-value-only | fixed JSON 差因模板 loss |
| Instruction mixture | StoryMix / CF-heavy / SHUF-heavy | QA 类型比例决定能力 |
| Hard grounding | SHUF / SAMEQ / K4 / mined | image-specific negatives 是核心 |
| Clinical module | CEQ / UAC / domain adapter / evidence memory | 模块化结构提升医学可解释性和泛化 |

#### Step 3：每个 family 只选 1 个 finalist

| Family | Finalist selection rule |
|---|---|
| JSON cleanup | highest AUC without leakage |
| Mixture | best AUC + CF acc >0.85 + leakage <10 |
| Hard grounding | best hard shuffle delta with AUC not worse than SHUF-3k |
| Module | highest primary score with NIH not worse |

#### Step 4：locked comparison

只把每个 family 的 finalist 放到 locked evaluation：

| Finalist | CheXpert AUC | NIH full | Hard shuffle | CF | A/B swap | Calibration | Seed mean ± std |
|---|---:|---:|---:|---:|---:|---:|---|
| Base |  |  |  |  |  |  |  |
| SHUF-3k |  |  |  |  |  |  |  |
| JSON-clean finalist |  |  |  |  |  |  |  |
| Mixture finalist |  |  |  |  |  |  |  |
| Hard-grounding finalist |  |  |  |  |  |  |  |
| Module finalist |  |  |  |  |  |  |  |

### Pareto frontier

如果没有一个方法所有指标都最优，不能强行叫最优。  
应该画 Pareto：

| Axis | Meaning |
|---|---|
| x-axis | CheXpert AUC |
| y-axis | hard shuffle delta |
| marker color | NIH AUC |
| marker shape | method family |
| marker size | cost |

这比“综合最优”更合理。

### 最终写法

不要写：

```text
We select SHUF-TW-clinical because it has the best overall tradeoff.
```

写成：

```text
We predefine two primary objectives: deployable classification performance and image-specific grounding. Methods are first grouped by mechanism and selected within each family on a development split. The final comparison is conducted only among family finalists on a locked evaluation suite. This avoids selecting a method post hoc from a large grid of variants.
```

---

## 1.5 问题五：先做 case study，再决定下一步主线

这是对的。下一步第一件事应该是 **case study / failure mining**，而不是继续盲目训练。

### Case study 要回答的问题

| Case study | 目的 |
|---|---|
| SHUF-TW-clinical vs SHUF-3k 成功/失败样本 | 看 0.9 AUC 提升来自哪里 |
| NIH failure cases | 看跨域拉不开是 label/noise/domain 还是模型问题 |
| SAMEQ 成功样本 | 看为什么 SAMEQ grounding 强 |
| SHUF-K4 失败样本 | 看 CF/A-B 为什么没过 gate |
| Curriculum 失败样本 | 看 leakage/stage forgetting/undertraining |
| P2-value-only 成功/失败样本 | 看 template-loss 诊断是否成立 |
| High-leakage QA 样本 | 查 mixture/curriculum 为什么 leakage 高 |
| Wrong hard negatives | 查 hard negative 是否 false negative |

### Case study 数据表

每个 case 需要字段：

| Field | Description |
|---|---|
| sample_id | 样本 |
| image_path | 图像 |
| question | 问题 |
| answer | 正确答案 |
| wrong_answer | 反事实答案 |
| hard_negative_image | 错图 |
| finding | 病种 |
| state | present/absent/uncertain |
| laterality | left/right/bilateral |
| model_a_score | 方法 A |
| model_b_score | 方法 B |
| win/loss | 谁对谁错 |
| report evidence | 报告证据 |
| manual note | 人工判断 |
| failure type | label noise / leakage / visual subtle / false negative / shortcut |

### Case taxonomy

| Failure type | 解释 |
|---|---|
| report leakage | 问题泄露答案 |
| label mismatch | CheXpert/NIH 标签定义不一致 |
| false hard negative | hard negative 实际也可能支持答案 |
| visual subtle | 图像证据很弱 |
| view/quality issue | AP/PA/portable/low quality |
| left-right ambiguity | laterality 难判断 |
| uncertainty mismatch | 报告 uncertain，但图像 definite 或相反 |
| answerability issue | 未提及不等于 absent |
| overfit to option pattern | A/B 位置偏置 |
| domain shift | NIH 风格/标签/图像分布不同 |

### 必做 case study

#### CS1. SHUF-TW-clinical wins over SHUF-3k

| Case | sample | finding | why SHUF-TW wins | category |
|---|---|---|---|---|
| 1 |  |  |  |  |
| 2 |  |  |  |  |

#### CS2. SHUF-3k wins over SHUF-TW-clinical

| Case | sample | finding | why SHUF-TW loses | category |
|---|---|---|---|---|
| 1 |  |  |  |  |
| 2 |  |  |  |  |

#### CS3. NIH disagreement cases

| Case | NIH image | label | methods disagree | likely cause |
|---|---|---|---|---|
| 1 |  |  |  | label mapping / domain / noise |
| 2 |  |  |  |  |

#### CS4. Curriculum leakage cases

| Case | question | answer | leakage pattern | fix |
|---|---|---|---|---|
| 1 |  |  |  |  |

#### CS5. False hard negatives

| Case | positive image | hard negative | question | issue |
|---|---|---|---|---|
| 1 |  |  |  | hard negative also positive |

### 输出文件

```text
outputs/final_tables/case_study_shuf_tw_vs_shuf.md
outputs/final_tables/case_study_nih_transfer.md
outputs/final_tables/case_study_curriculum_failure.md
outputs/final_tables/case_study_hard_negative_quality.md
outputs/final_tables/case_study_summary.csv
```

### Case study 之后的决策

| Case study finding | 下一步 |
|---|---|
| SHUF-TW wins on rare/high-null/laterality | 强化 TW-clinical / field-balanced |
| SHUF-TW wins mainly noise | 降级，不主打 |
| NIH fails due label mapping | 换 MIMIC/PadChest/VinDr 外部 |
| NIH fails due domain shift | 加 domain-robust module |
| curriculum fails due leakage | 先清洗数据再重训 |
| curriculum fails due undertraining | full MIMIC / longer schedule |
| hard negatives false-negative 多 | 加 hard-negative verifier/filter |
| SAMEQ successes明显 | 把 SAMEQ 纳入主方法候选 |

---

## 1.6 问题六：TMI 需要模块，不只是训练方式

用户担心得对。  
如果只是：

```text
我们设计了不同 instruction 训练 ViT。
```

对 TMI 可能显得像 training recipe，工程性强，方法模块不足。

下一阶段建议准备几个真正的模块一起测试。  
这些模块要满足：

1. deployable；
2. 医学相关；
3. 不依赖 LLM 部署；
4. 能解释或提升跨域/grounding；
5. 能和 SHUF/curriculum 配合；
6. 最好有明确数学定义或结构图。

---

# Part 2. 建议新增的 TMI 级模块

## Module M1：Clinical Evidence Query Module，简称 CEQ

### 2.1.1 核心想法

不用只靠全局 image embedding。  
给每个 finding 一个 learnable clinical query，让它去图像 patch token 里找证据。

类似：

```text
image patch tokens
    ↓ cross attention
finding-specific queries
    ↓
field evidence embeddings
```

比如：

```text
query_cardiomegaly
query_pneumothorax
query_effusion
query_edema
```

每个 query 负责一个医学 finding。

### 为什么有价值？

普通 vision tower 输出一个全局 embedding，可能丢掉局部病灶。  
胸片很多病灶是局部的，比如 pneumothorax、effusion、opacity。

CEQ 可以让模型学：

```text
每个 finding 对应一个 evidence embedding
```

这比纯训练 recipe 更像一个模块。

### 操作方式

输入：

```text
ViT patch tokens: [B, N, D]
clinical queries: [F, D]
```

cross attention：

```text
E_f = CrossAttention(query_f, patch_tokens)
```

输出：

```text
field evidence embeddings: [B, F, D]
```

用于：

1. downstream LP；
2. finding-specific classifier；
3. SHUF / CF answer scoring；
4. attention visualization。

### Loss

| Loss | 作用 |
|---|---|
| UMS state loss | 每个 finding 的 state |
| SHUF margin loss | field evidence 对错图敏感 |
| query diversity loss optional | 防止所有 query collapse |
| attention sparsity optional | 提高可解释性 |

### 实验矩阵

| Run | Base | Module | Data | 目的 |
|---|---|---|---|---|
| SHUF-3k | Qwen3-VL | none | D7 | baseline |
| SHUF+CEQ | Qwen3-VL | CEQ | D7 | 看 field queries 是否提升 |
| SHUF-TW+CEQ | Qwen3-VL | CEQ | D7+TW | 主候选 |
| SAMEQ+CEQ | Qwen3-VL | CEQ | SAMEQ | 看最干净 grounding |
| CUR+CEQ | Qwen3-VL | CEQ | curriculum | 看 workflow |

### 空表

| Run | CheXpert AUC | NIH AUC | Hard shuffle delta | CF acc | Per-field AUC | Attention quality | Decision |
|---|---:|---:|---:|---:|---:|---|---|
| SHUF+CEQ |  |  |  |  |  |  |  |
| SHUF-TW+CEQ |  |  |  |  |  |  |  |
| SAMEQ+CEQ |  |  |  |  |  |  |  |

### 预期收益

- CheXpert AUC 可能提升；
- hard shuffle delta 可能提升；
- per-field performance 更可解释；
- 可视化更适合 TMI。

---

## Module M2：Hard-Negative Memory Bank，简称 HNMB

### 2.2.1 核心想法

当前 SHUF 的 hard negative 是预先规则构造的。  
但模型真正容易混淆的负样本可能不是规则负样本。

建立一个 memory bank：

```text
image embedding memory
label/finding/state memory
question-answer metadata memory
```

训练中动态找 hard negative。

### 操作方式

1. 每隔 N steps 抽取训练集 image embeddings；
2. 对每个样本找相似 embedding；
3. 过滤掉同答案样本；
4. 选最像但答案不同的作为 hard negative；
5. 用 image-shuffle margin 训练。

### 负样本类型

| Type | 说明 |
|---|---|
| nearest opposite state | embedding 最相似但 present/absent 相反 |
| nearest opposite laterality | left/right 相反 |
| nearest uncertain mismatch | definite/uncertain 不同 |
| high-confusion negative | wrong image loss 很低 |
| mixed memory negative | 综合 |

### Loss

```text
L = CE(correct answer | positive image)
  + λ * max(0, margin + NLL(pos) - NLL(hard_neg))
```

### 实验矩阵

| Run | Negative source | Update | 目的 |
|---|---|---|---|
| SHUF-K1 | fixed negative | static | baseline |
| SHUF-K4 | fixed K negatives | static | multi-negative |
| HNMB-static | embedding-mined once | static | 看 mined 是否更好 |
| HNMB-online | update every epoch | dynamic | 最强 |
| HNMB+CEQ | memory bank + evidence query | dynamic | 结构化最强候选 |

### 空表

| Run | CheXpert AUC | NIH AUC | Hard shuffle delta | CF acc | Negative false rate | Cost | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| HNMB-static |  |  |  |  |  |  |  |
| HNMB-online |  |  |  |  |  |  |  |
| HNMB+CEQ |  |  |  |  |  |  |  |

### 预期收益

- hard shuffle delta 大幅提升；
- fewer random/weak negatives；
- 更像一个正式算法模块。

---

## Module M3：Answerability-Uncertainty Calibration Head，简称 AUCH

### 2.3.1 核心想法

医学报告里很重要的一点是：

```text
unmentioned ≠ absent
uncertain ≠ negative
```

现在 instruction training 学到了表征，但部署时只做 LP。  
可以在 vision tower 后加一个轻量临床校准头：

```text
answerability head
uncertainty head
state head
```

这不是 LLM，而是 deployable medical head。

### 操作方式

从 vision tower 提取 embedding，训练三个 head：

```text
h_answerable(f)
h_uncertain(f)
h_state(f)
```

对于每个 finding：

```text
p_answerable
p_uncertain
p_present
```

### Loss

```text
L = BCE(answerability)
  + CE(state | answerable)
  + BCE(uncertainty)
  + calibration loss
```

可加：

```text
Brier / ECE regularization
```

### 实验矩阵

| Run | Backbone | Head | 目的 |
|---|---|---|---|
| SHUF LP | SHUF | linear | baseline |
| SHUF+AUCH | SHUF | answerability/uncertainty/state | 看医学语义 |
| SHUF-TW+AUCH | SHUF-TW | AUCH | 主候选 |
| CEQ+AUCH | CEQ | AUCH | 结构化最强 |
| AUCH+calibration | AUCH + temperature / ECE | 临床校准 |

### 空表

| Run | Macro-AUC | Macro-AUPRC | ECE | Brier | High-null AUC | Uncertain F1 | NIH AUC | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| SHUF+AUCH |  |  |  |  |  |  |  |  |
| SHUF-TW+AUCH |  |  |  |  |  |  |  |  |
| CEQ+AUCH |  |  |  |  |  |  |  |  |

### 预期收益

- TMI 友好；
- 解释 missingness / uncertainty；
- 可能提升 rare/high-null；
- 提供临床可信度指标。

---

## Module M4：Domain-Robust Adapter，简称 DRA

### 2.4.1 为什么需要？

NIH 拉不开差距，可能是 domain shift。

所以要加一个专门针对跨域的模块，而不是继续只改 instruction。

### 模块选择

#### Option 1：Domain-specific BatchNorm / LayerNorm Adapter

训练时给不同域一个小 adapter：

```text
source domain adapter
target/domain-agnostic adapter
```

部署时可用：

```text
domain-agnostic adapter
```

#### Option 2：CORAL / MMD feature alignment

在 CheXpert/MIMIC/NIH unlabeled images 上对齐 feature distribution：

```text
L_domain = ||Cov(source) - Cov(target)||^2
```

或者 MMD。

#### Option 3：Domain adversarial head

加 domain classifier，gradient reversal，让特征难以区分 CheXpert vs NIH/MIMIC。

### 实验矩阵

| Run | Data | Domain module | 目的 |
|---|---|---|---|
| SHUF | CheXpert/MIMIC | none | baseline |
| SHUF+CORAL | + unlabeled NIH | CORAL | feature alignment |
| SHUF+DANN | + unlabeled NIH | adversarial | domain-invariant |
| SHUF+Adapter | source/target adapter | adapter | domain robustness |
| CEQ+DRA | CEQ + domain module | combined | TMI strongest |

### 空表

| Run | CheXpert AUC | NIH AUC | MIMIC AUC | Domain gap MMD | ECE external | Decision |
|---|---:|---:|---:|---:|---:|---|
| SHUF+CORAL |  |  |  |  |  |  |
| SHUF+DANN |  |  |  |  |  |  |
| SHUF+Adapter |  |  |  |  |  |  |

### 预期收益

- NIH/MIMIC/PadChest 上可能拉开差距；
- 能回应“为什么 NIH 不涨”；
- TMI 审稿人更容易接受临床跨域模块。

---

## Module M5：Clinical Consistency Scoring Head，简称 CCSH

### 2.5.1 核心想法

现在 SHUF 是通过 LLM NLL 做 image-question-answer matching。  
可以在 vision tower 上加一个轻量 consistency head：

```text
score(image, clinical statement)
```

部署时可以不使用 LLM，但保留一个 clinical consistency score 模块。

### 操作方式

输入：

```text
image embedding
statement embedding
```

输出：

```text
support / contradict / uncertain
```

statement 可以来自固定 finding-state templates，不需要 LLM 在线生成。

例子：

```text
image + "There is a left pleural effusion." -> support
image + "There is a right pleural effusion." -> contradict
```

### Loss

```text
CE(support/contradict/uncertain)
contrastive margin
```

### 实验矩阵

| Run | Module | Data |
|---|---|---|
| SHUF | none | D7 |
| SHUF+CCSH | consistency head | D7 |
| SAMEQ+CCSH | consistency head | SAMEQ |
| CEQ+CCSH | evidence queries + scoring | D7/SAMEQ |

### 空表

| Run | CheXpert AUC | NIH AUC | Consistency acc | Hard shuffle delta | CF acc | Decision |
|---|---:|---:|---:|---:|---:|---|
| SHUF+CCSH |  |  |  |  |  |  |
| SAMEQ+CCSH |  |  |  |  |  |  |
| CEQ+CCSH |  |  |  |  |  |  |

### 预期收益

- 把 SHUF 从纯训练技巧变成 deployable consistency module；
- 对 TMI 友好；
- 可解释为 image-report consistency learning。

---

## Module M6：Case-Driven Curriculum Scheduler，简称 CDCS

### 2.6.1 核心想法

根据 case study / failure mining 动态调整训练样本分布。

比如发现模型在 laterality 错误多，就提高 laterality QA/SHUF 比例。  
发现 NIH 失败在某些 label，就提高对应 finding 的 hard negatives。

### 操作方式

每 N steps：

1. 评估 dev set；
2. 按 failure type 统计；
3. 更新 sampling weights。

Sampling weights：

```text
w(sample) = base_weight
          × failure_type_weight
          × field_rarity_weight
          × visual_dependency_weight
```

### 实验矩阵

| Run | Scheduler |
|---|---|
| PROG-Mix | fixed schedule |
| CDCS-field | 按 field failure 动态采样 |
| CDCS-hardneg | 按 hard-negative failure 动态采样 |
| CDCS-domain | 按 external/NIH failure 动态采样 |
| CDCS-full | 综合 |

### 空表

| Run | CheXpert AUC | NIH AUC | Hard shuffle delta | CF acc | Failure reduction | Decision |
|---|---:|---:|---:|---:|---:|---|
| CDCS-field |  |  |  |  |  |  |
| CDCS-hardneg |  |  |  |  |  |  |
| CDCS-full |  |  |  |  |  |  |

### 预期收益

- 不再是固定 curriculum；
- 由 case study 驱动；
- 方法逻辑强，避免像随机试验。

---

# Part 3. 下一步总执行路线

## Phase 0：先做 Case Study 和统计稳定性

| Priority | Task | 必要性 |
|---:|---|---|
| 1 | SHUF-TW vs SHUF-3k seed stability | 解决 0.9 AUC 是否可靠 |
| 2 | NIH full + label mapping audit | 解释跨域拉不开 |
| 3 | case study: SHUF-TW vs SHUF | 找 0.9 提升来自哪里 |
| 4 | case study: NIH failure | 找外部失败原因 |
| 5 | case study: curriculum leakage | 判断 curriculum 是否数据问题 |
| 6 | case study: hard negative false negative | 判断 SHUF 数据质量 |

---

## Phase 1：补强 candidate，不急定主方法

| Priority | Run | 目的 |
|---:|---|---|
| 1 | SHUF-TW-clinical seed3 | 看稳定性 |
| 2 | SHUF-3k seed3 | baseline 方差 |
| 3 | SAMEQ-SHUF + CF-compatible rows | 让 SAMEQ 也能过 CF gate |
| 4 | SHUF-K4-TW-clinical | 让 K4 也尝试过 CF/A-B gate |
| 5 | CUR-v2-progressive-replay 10k | 给 curriculum 公平机会 |
| 6 | CUR-v2-MIMIC-full | 验证是不是数据量不够 |

---

## Phase 2：TMI 模块测试

优先做 3 个最有价值模块：

| Priority | Module | 为什么 |
|---:|---|---|
| 1 | CEQ | 增加 finding-specific evidence module，可解释 |
| 2 | AUCH | 医学 answerability/uncertainty/calibration，TMI 友好 |
| 3 | HNMB | 把 SHUF 变成正式 hard-negative mining algorithm |
| 4 | DRA | 针对 NIH 跨域问题 |
| 5 | CCSH | 把 image-report consistency 变成 deployable head |
| 6 | CDCS | case-driven curriculum，避免随机试验感 |

---

## Phase 3：最终 locked comparison

每个 family 只选一个 finalist：

| Family | Finalist candidate |
|---|---|
| Direct hard grounding | SHUF-TW-clinical or SHUF-3k |
| SAMEQ grounding | SAMEQ-CF-compatible |
| Multi-negative grounding | SHUF-K4-TW-clinical or HNMB |
| Curriculum | CUR-v2-progressive-replay |
| Clinical module | CEQ+AUCH or CEQ+CCSH |
| Domain robust | DRA variant |

然后统一跑：

```text
multi-seed
NIH full
MIMIC / external if available
AUPRC / ECE
case study
cost
```

---

# Part 4. 最推荐下一批实验清单

## 4.1 第一批：不训练或少训练，先查问题

| Run/Task | Output |
|---|---|
| Seed variance bootstrap for existing outputs | seed/CI report |
| NIH full label mapping audit | NIH diagnosis report |
| SHUF-TW vs SHUF case study | casebook |
| Curriculum leakage/failure case study | casebook |
| Hard negative false-negative audit | data quality report |

---

## 4.2 第二批：最关键训练

| Run | 目的 |
|---|---|
| SHUF-TW-clinical-seed3 | 稳定性 |
| SHUF-3k-seed3 | baseline 方差 |
| SAMEQ-CF-compatible | 看 SAMEQ 能否变 final |
| SHUF-K4-TW-clinical | 看 K4 能否过 CF/A-B gate |
| CUR-v2-progressive-replay-10k | 公平评估 curriculum |
| CUR-v2-progressive-replay-MIMIC-full | 验证数据量是否核心 |

---

## 4.3 第三批：模块实验

| Run | 目的 |
|---|---|
| SHUF+CEQ | evidence query module |
| SHUF-TW+CEQ | 当前 candidate + CEQ |
| SHUF+AUCH | clinical uncertainty/calibration |
| SHUF+HNMB | dynamic hard negative |
| SHUF+DRA | external transfer |
| CEQ+AUCH+SHUF | TMI 主候选 |
| CEQ+HNMB+SHUF | grounding 主候选 |
| CEQ+DRA+SHUF | external 主候选 |

---

# Part 5. 空总表

## 5.1 Seed stability

| Run | Seed | CheXpert AUC | NIH AUC | Hard shuffle delta | CF acc | A/B-swap | Leakage | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SHUF-3k | 0 |  |  |  |  |  |  |  |
| SHUF-3k | 1 |  |  |  |  |  |  |  |
| SHUF-3k | 2 |  |  |  |  |  |  |  |
| SHUF-TW-clinical | 0 |  |  |  |  |  |  |  |
| SHUF-TW-clinical | 1 |  |  |  |  |  |  |  |
| SHUF-TW-clinical | 2 |  |  |  |  |  |  |  |

---

## 5.2 NIH diagnosis

| Run | NIH subset | Labels | Macro-AUC | Macro-AUPRC | ECE | Label mapping issue | Notes |
|---|---|---|---:|---:|---:|---|---|
| Base | 1k |  |  |  |  |  |  |
| SHUF-3k | 1k |  |  |  |  |  |  |
| SHUF-TW-clinical | 1k |  |  |  |  |  |  |
| Base | full |  |  |  |  |  |  |
| SHUF-3k | full |  |  |  |  |  |  |
| SHUF-TW-clinical | full |  |  |  |  |  |  |

---

## 5.3 Curriculum fair retry

| Run | Data | Steps | Leakage | CheXpert AUC | NIH AUC | Hard shuffle | CF acc | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| CUR-v1 | 3k |  |  |  |  |  |  | failed baseline |
| CUR-v2-progressive-replay | 10k |  |  |  |  |  |  |  |
| CUR-v2-SAMEQ-final | 10k |  |  |  |  |  |  |  |
| CUR-v2-MIMIC-full | full |  |  |  |  |  |  |  |

---

## 5.4 Module comparison

| Module | Backbone/data | CheXpert AUC | NIH AUC | Hard shuffle | CF acc | Calibration | Interpretability | Decision |
|---|---|---:|---:|---:|---:|---:|---|---|
| CEQ | SHUF |  |  |  |  |  | attention maps |  |
| AUCH | SHUF |  |  |  |  |  | ECE/Brier |  |
| HNMB | SHUF |  |  |  |  |  | hard-neg cases |  |
| DRA | SHUF |  |  |  |  |  | domain shift |  |
| CCSH | SHUF |  |  |  |  |  | consistency score |  |
| CDCS | curriculum |  |  |  |  |  | failure-driven |  |

---

## 5.5 Final locked comparison

| Family | Finalist | Seeds | CheXpert AUC mean±std | NIH AUC mean±std | Hard shuffle mean±std | CF mean±std | AUPRC | ECE | Final role |
|---|---|---:|---|---|---|---|---:|---:|---|
| Base | Base |  |  |  |  |  |  |  | baseline |
| Direct SHUF |  |  |  |  |  |  |  |  |  |
| SAMEQ |  |  |  |  |  |  |  |  |  |
| Multi-negative |  |  |  |  |  |  |  |  |  |
| Curriculum |  |  |  |  |  |  |  |  |  |
| Clinical module |  |  |  |  |  |  |  |  |  |
| Domain robust |  |  |  |  |  |  |  |  |  |

---

# Part 6. 暂定论文主线

在下一步验证前，暂时不要写：

```text
SHUF-TW-clinical is the final best method.
```

现在更合理的论文定位是：

```text
We identify a key failure mode of language-supervised CXR pretraining: improvements in text-side instruction loss or QA accuracy do not necessarily translate into image-specific visual grounding. Through case-driven analysis, we develop a family of image-mismatch and clinical evidence modules. The final method is selected under a locked multi-objective evaluation that jointly measures deployable classification, hard image-shuffle sensitivity, counterfactual robustness, leakage, calibration, and external transfer.
```

中文：

```text
我们发现语言监督训练胸片视觉模型的关键问题是：文本侧任务学得好，不等于模型真的看懂具体图像。通过 case study，我们设计一系列 image-mismatch 和 clinical evidence 模块。最终方法不是从一堆实验里随便挑，而是在固定的多目标评估协议下选择，需要同时满足分类性能、hard image-shuffle、反事实稳健性、泄露控制、校准和外部迁移。
```

---

# Part 7. 立即执行建议

最推荐下一步不是直接训练大模型，而是：

```text
Step 1: case study + seed stability + NIH audit
Step 2: 修 SAMEQ/K4，让它们也能过 final gate
Step 3: 公平重做 curriculum v2，最好用 10k 或 full MIMIC
Step 4: 加 CEQ / AUCH / HNMB 三个 TMI 友好模块
Step 5: locked final comparison
```

如果只选最先做的 8 个：

| Priority | Task |
|---:|---|
| 1 | SHUF-TW vs SHUF-3k case study |
| 2 | NIH full + label mapping audit |
| 3 | SHUF-3k seed3 |
| 4 | SHUF-TW-clinical seed3 |
| 5 | SAMEQ-CF-compatible |
| 6 | SHUF-K4-TW-clinical |
| 7 | CUR-v2-progressive-replay-10k |
| 8 | CEQ module on SHUF |

---

# Appendix. Codex / 实验同学任务清单

## A. Case study scripts

```text
scripts/mine_pairwise_case_studies.py
scripts/audit_nih_transfer_failure.py
scripts/audit_hard_negative_quality.py
scripts/audit_curriculum_leakage_cases.py
scripts/build_casebook_markdown.py
```

## B. Stability scripts

```text
scripts/run_multiseed_manifest.py
scripts/bootstrap_auc_ci.py
scripts/paired_bootstrap_method_delta.py
scripts/summarize_multiseed_results.py
```

## C. NIH/domain scripts

```text
scripts/audit_label_mapping_nih.py
scripts/run_nih_full_transfer.py
scripts/compute_domain_shift_mmd.py
scripts/plot_dataset_embedding_umap.py
```

## D. Curriculum v2 scripts

```text
scripts/build_curriculum_v2_schedule.py
scripts/generate_curriculum_v2_instructions.py
scripts/train_qwen3vl_curriculum_v2.py
```

## E. Module scripts

```text
models/clinical_evidence_query.py
models/answerability_uncertainty_head.py
models/hard_negative_memory_bank.py
models/domain_robust_adapter.py
models/clinical_consistency_head.py
models/case_driven_curriculum_scheduler.py
```

## F. Final reporting

```text
outputs/final_tables/case_study_summary.md
outputs/final_tables/multiseed_stability.md
outputs/final_tables/nih_domain_audit.md
outputs/final_tables/module_candidate_results.md
outputs/final_tables/locked_final_comparison.md
```

---

# Part 8. 2026-07-02 Codex 真实执行结果回写

## 8.1 完成状态

本轮已把 2026-07-01 仍属于边界/manifest/smoke 的项目全部推进为 artifact-backed evidence：

| Area | Status | Evidence |
|---|---|---|
| Multi-seed stability | completed | 12 个 family/seed run 均完成 `5000` step long training，并完成 CheXpert LP、NIH all-available transfer、visual dependence、CF/A-B 或 not-applicable 诊断、paraphrase。 |
| NIH available transfer | completed | 每个 seed 均评估 `25596` 条 NIH available records；`transfer_metrics.json` 含 `nih_1000`、`nih_5000`、`all_available` subset metrics。 |
| Embedding-backed MMD/Projection | completed | 已导出 instruction train/val、CheXpert-val、NIH-available embeddings；`domain_shift_mmd.md` 与 `dataset_embedding_projection.md` 由真实 embedding 重算。 |
| Curriculum v2 | completed | `cur_v2_progressive_replay` 完成 `12000` step formal training，`best_val_loss=0.237637`。 |
| Formal module ablation | completed | CEQ/AUCH/HNMB/DRA/CCSH/CDCS 全部完成 `1000` step formal embedding-level training，并写入 `metrics_final.json`。 |
| Summary tables | completed | `case_study_full_execution_status.*`、`case_study_extra_execution_status.*`、`module_ablation_results.*` 已刷新。 |

完成审计和入口表：

```text
outputs/final_tables/case_study_full_execution_status.csv
outputs/final_tables/case_study_full_execution_status.md
outputs/final_tables/case_study_extra_execution_status.csv
outputs/final_tables/case_study_extra_execution_status.md
outputs/final_tables/module_ablation_results.csv
outputs/final_tables/module_ablation_results.md
```

## 8.2 Multi-seed 稳定性结果

下表来自 `outputs/final_tables/case_study_full_execution_status.csv`，所有 NIH 数字均为 `all_available`，`n=25596`。

| Family | CheXpert AUC mean +/- std | NIH AUC mean +/- std | Hard shuffle delta mean +/- std | CF acc mean +/- std | A/B-swap mean +/- std | Boundary |
|---|---:|---:|---:|---:|---:|---|
| SHUF-3k | 0.686757 +/- 0.051151 | 0.570194 +/- 0.013412 | 0.114476 +/- 0.069407 | 0.870748 +/- 0.025755 | 0.869993 +/- 0.016048 | complete |
| SHUF-TW-clinical | 0.690742 +/- 0.031593 | 0.569828 +/- 0.005610 | 0.046622 +/- 0.026961 | 0.885865 +/- 0.009508 | 0.892668 +/- 0.002360 | complete |
| SAMEQ-SHUF-3k | 0.713825 +/- 0.009142 | 0.583359 +/- 0.007227 | 0.438040 +/- 0.018546 | N/A | N/A | CF/A-B option-pairwise N/A because same-question/different-answer rows have zero option records. |
| SHUF-K4 | 0.709548 +/- 0.026408 | 0.581098 +/- 0.009345 | 0.329075 +/- 0.046650 | 0.806123 +/- 0.014826 | 0.811413 +/- 0.021282 | complete |

逐 run 结果见 `outputs/final_tables/case_study_full_execution_status.md`。当前读数说明：`SHUF-TW-clinical` 不再能基于 mean AUC 明显压过 `SHUF-3k`；`SAMEQ-SHUF-3k` 和 `SHUF-K4` 的 hard-shuffle grounding 信号更稳定，但 SAMEQ 的 CF/A-B option-pairwise 指标在当前数据格式下不可直接比较。

## 8.3 NIH / Domain 诊断

本轮不再停留在 NIH 1k boundary。12 个 run 均完成 NIH available transfer：

```text
outputs/qwen3vl_case_study_multiseed_transfer/*_nih_available/transfer_metrics.json
outputs/final_tables/nih_available_transfer_status.md
```

Embedding-backed domain shift 产物：

```text
outputs/case_study_module_embeddings/shuf_tw_clinical_instruction_train.npz
outputs/case_study_module_embeddings/shuf_tw_clinical_instruction_val.npz
outputs/case_study_module_embeddings/shuf_tw_clinical_chexpert_val.npz
outputs/case_study_module_embeddings/shuf_tw_clinical_nih_available.npz
outputs/final_tables/domain_shift_mmd.md
outputs/final_tables/dataset_embedding_projection.md
outputs/final_tables/dataset_embedding_projection.png
```

`domain_shift_mmd.md` 当前 RBF-MMD 为 `0.139625`，使用 CheXpert-val `n=1000` 与 NIH-available sampled `n=4000` embeddings；projection 报告写出 `5000` projected rows，当前降维方法为 `pca`。

## 8.4 Curriculum v2

`CUR-v2-progressive-replay` 已从 runnable config 变成正式长训结果：

| Run | Steps | best_val_loss | Evidence |
|---|---:|---:|---|
| cur_v2_progressive_replay | 12000 | 0.237637 | `outputs/qwen3vl_case_study_modules/cur_v2_progressive_replay/metrics_final.json` |

这个结果只证明 refined curriculum 已获得公平长训机会；是否成为论文主线仍需与 locked family finalists 结合 downstream 指标和 case study 判断，不能单靠 training loss 做 final-best claim。

## 8.5 Formal module ablation

六个 TMI-friendly modules 不再只是 implemented + smoke-ready。正式训练结果如下：

| Module | Steps | Val loss | State acc | Binary AUC | Binary AUPRC | Binary F1 | Binary ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| CEQ | 1000 | 0.740293 | 0.689402 | 0.822737 | 0.753232 | 0.658423 | 0.074109 |
| AUCH | 1000 | 0.731392 | 0.677859 | 0.826266 | 0.749298 | 0.672811 | 0.091024 |
| HNMB | 1000 | 0.731610 | 0.697796 | 0.827386 | 0.750214 | 0.672868 | 0.062489 |
| DRA | 1000 | 0.741210 | 0.678909 | 0.817569 | 0.730411 | 0.564460 | 0.082049 |
| CCSH | 1000 | 0.601051 | 0.746065 | 0.894268 | 0.847528 | 0.763689 | 0.063504 |
| CDCS | 1000 | 0.731102 | 0.683106 | 0.825703 | 0.749776 | 0.675969 | 0.067031 |

Evidence:

```text
outputs/qwen3vl_case_study_module_ablation/*/metrics_final.json
outputs/final_tables/module_ablation_results.md
```

当前最强 embedding-level module 读数是 `CCSH`，但它仍是 head-level formal ablation，不等同于 full Qwen3-VL end-to-end final method。下一步若要写成 TMI 主方法，需要把 `CCSH` 或 `CEQ+CCSH` 放入 locked final training/evaluation suite。

## 8.6 保留边界

1. NIH full：当前完成的是本地可用 NIH UMS manifest 的 `all_available=25596`；如果论文定义的 NIH full 必须覆盖 NIH 原始全库，需要另建完整 UMS manifest 后再跑。
2. SAMEQ CF/A-B：SAMEQ rows 生成了 counterfactual/A-B artifacts，但当前 same-question/different-answer 格式没有 option-pairwise rows，因此 `cf_acc` 与 `ab_swap_acc` 是 not-applicable，不是 0 或缺失。
3. Module ablation：CEQ/AUCH/HNMB/DRA/CCSH/CDCS 已完成 formal training；但这些是 embedding-level module head ablations，不应直接声称替代 full instruction-tuned backbone。
4. Final method：当前证据支持 family-level narrowing 与 mechanism diagnosis；仍不应把 `SHUF-TW-clinical` 写成 final-best。

## 8.7 当前推荐论文方向

更稳的写法是：

```text
SHUF-TW-clinical is not a statistically decisive final winner after three-seed reruns.
SAMEQ-SHUF-3k and SHUF-K4 show stronger and more stable image-mismatch grounding signals.
NIH available transfer remains tightly clustered, suggesting label/domain limits rather than a simple method ranking.
CCSH is the strongest current formal TMI module head and should be promoted into the next locked full-training candidate.
```

中文结论：

```text
当前最有投稿价值的主线不是“SHUF-TW-clinical 综合最优”，而是：
通过 seed stability、NIH available transfer、embedding domain shift 和 formal module ablation 证明，单一 recipe 的小幅 AUC 提升并不可靠；更稳的贡献应转向 image-mismatch grounding + deployable clinical consistency/evidence module 的机制化 pipeline。
```
