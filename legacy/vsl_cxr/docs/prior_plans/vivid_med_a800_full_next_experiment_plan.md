# VIVID-Med 下一阶段完整实验文档：A800-80GB 多路线并行验证版

> 版本：2026-07-03  
> 设备假设：单卡 NVIDIA A800 80GB。  
> 当前定位：不要急着把 `SHUF-TW-clinical` 写成最终方法；下一阶段要围绕一个更强的故事做系统验证：  
>
> **VLM 老师不是简单“教 ViT”，而是按临床视觉难度设计课程：基础医学问答 → 反事实医学陈述 → 同题换图 / 多负图匹配 → 可部署临床一致性读出。**
>
> 推荐新主线名：
>
> **CVCP-Med: Clinical Visual Curriculum Pretraining with VLM Teachers for Deployable Chest X-ray Vision Encoders**
>
> 或者更强调一致性：
>
> **CVC-Med: Clinical Visual Consistency Learning with VLM Teachers for Chest X-rays**

---

# 0. 当前证据和下一步目标

## 0.1 当前已经知道什么？

上一阶段 case study / multi-seed / module ablation 说明了几个关键事实：

1. `SHUF-TW-clinical` 不应该再被直接当成最终方法。  
   它在单 seed 下看起来综合好，但 multi-seed 后相对 `SHUF-3k` 的 CheXpert AUC 优势很小，hard shuffle 也不占优。

2. `SAMEQ-SHUF-3k` 是当前最稳定的 image-specific grounding 证据。  
   它的核心是：**同一个问题，不同图像，答案不同**。这能最大限度排除文本 shortcut。

3. `SHUF-K4` 很有潜力。  
   它通过一正多负图像训练强化 visual matching，但 CF / A-B swap 稳健性还需要补。

4. `CCSH` 是目前最强的 TMI-friendly 模块。  
   它把 “图像是否支持临床陈述” 变成一个可部署 consistency head。  
   但 CCSH 还需要和 SAMEQ / SHUF-K4 / CEQ 组合，证明它不是单独 head 自己强，而是读出了 VLM teacher 训练后的视觉能力。

5. NIH 不适合作为主跨域验证。  
   后续主 external 建议换成 **VinDr-CXR** 或 **PadChest**。NIH 可以降级为 appendix 或不主打。

---

## 0.2 下一阶段要回答的核心问题

| QID | 问题 | 为什么重要 |
|---|---|---|
| Q1 | Curriculum 是否能超过 direct SAMEQ / SHUF-K4？ | 决定是否主打 “Clinical Visual Curriculum” |
| Q2 | CCSH 是否能把 grounding 信号变成 deployable module？ | 决定是否够 TMI 方法高度 |
| Q3 | CEQ + CCSH 是否能提供可解释 finding-specific evidence？ | 决定是否能加入可视化 / 临床解释 |
| Q4 | SAMEQ / SHUF-K4 加 CF-compatible rows 后能否过 final gate？ | 修当前强 grounding 方法的短板 |
| Q5 | HNMB / mined hard negatives 是否能进一步提升 grounding？ | 把 SHUF 从固定负样本升级成算法模块 |
| Q6 | 一个外部 CXR 数据集上是否能看到更清楚泛化？ | 解决 NIH 拉不开问题 |
| Q7 | 不同 VLM teacher 是否都能教 vision tower？ | 避免只适配 Qwen3-VL，被说不可泛化 |
| Q8 | Text-only scaffold 是否明显弱于 coupled VLM？ | 证明“一套 VLM 初始化”确实必要 |

---

# 1. A800 80GB 资源使用策略

## 1.1 基本原则

A800 80GB 比 RTX 3090 24GB 宽裕很多，可以更快迭代，但仍然建议分任务类型调度。

不要一上来同时跑多个 full VLM training，先做显存 probe。

---

## 1.2 任务分层

| Lane | 任务类型 | 预计显存 | 是否可并行 | 说明 |
|---|---|---:|---|---|
| L0-heavy | Qwen3-VL full vision tower + connector training | 35-70GB | 1 个为主；probe 后可 2 个小 run | 主训练 |
| L1-mid | Qwen3-VL LoRA / last-block / connector-only | 15-35GB | 可 2-3 个 | 训练策略 ablation |
| L2-light | LP / CCSH / AUCH / CEQ head training | 2-12GB | 可多开 | 模块和评估 |
| L3-eval | visual dependence / CF / A-B swap / paraphrase | 2-20GB | 可和 L0 错峰 | 评估诊断 |
| L4-data | GLM/API 数据生成、audit、hard negative mining | CPU/API | 可并行 | 不占 GPU 或少量 GPU |

---

## 1.3 推荐并行策略

### 方案 A：稳妥模式

```text
1 个 L0-heavy 主训练
+ 2-4 个 L2-light 评估/模块训练
+ CPU/API 数据生成
```

适合正式 run。

### 方案 B：快速筛选模式

```text
2 个 L1-mid / 小规模 Qwen3-VL run
+ 2 个 LP/eval
```

适合 debug / pilot。

### 方案 C：显存压榨模式

只有在 peak VRAM probe 后使用：

```text
2 个 full Qwen3-VL 2B run
```

条件：

```text
单个 run peak VRAM < 38GB
无 OOM
吞吐下降可接受
```

如果单个 run peak VRAM > 45GB，不建议双开 full training。

---

## 1.4 A800 preflight 必跑

每种训练类型先跑 200-step probe：

| Probe ID | Run | 目的 |
|---|---|---|
| MEM-SAMEQ | SAMEQ direct 200 steps | 测 full training 显存 |
| MEM-SHUF-K4 | SHUF-K4 200 steps | 测 multi-negative 显存 |
| MEM-CCSH | CCSH head 200 steps | 测 module 显存 |
| MEM-CEQ-CCSH | CEQ+CCSH 200 steps | 测模块组合显存 |
| MEM-CVCP | progressive curriculum 200 steps | 测数据调度和显存 |

空表：

| Probe | Peak VRAM | Steps/sec | Batch size | Grad accum | Can co-run? | Notes |
|---|---:|---:|---:|---:|---|---|
| MEM-SAMEQ |  |  |  |  |  |  |
| MEM-SHUF-K4 |  |  |  |  |  |  |
| MEM-CCSH |  |  |  |  |  |  |
| MEM-CEQ-CCSH |  |  |  |  |  |  |
| MEM-CVCP |  |  |  |  |  |  |

---

## 1.5 A800 推荐训练配置起点

| Config | RTX3090 旧设置 | A800 起点 |
|---|---:|---:|
| Micro batch | 1-4 | 4-8 |
| Grad accumulation | 8 | 2-4 |
| BF16 | yes | yes |
| Gradient checkpointing | yes | yes for full vision; optional for head-only |
| FlashAttention | if available | strongly recommended |
| Num workers | Windows 曾设 0 | Linux/A800 可试 4-8 |
| Save interval | 500/1000 | 1000 or 2000 |
| Eval interval | 500/1000 | 1000 |
| Main steps | 5k-12k | 8k-20k depending data |

---

# 2. 新故事框架：CVCP-Med

## 2.1 主线一句话

> **我们把 VLM 老师设计成一个临床视觉考试官。它不是简单给 ViT 标签，而是按难度构造课程：基础医学问答、反事实陈述、同题换图、多负图匹配。ViT 在这个课程中逐渐从“会答医学词”变成“能判断这张图是否支持某个临床陈述”。训练后丢掉 LLM，用 vision tower + clinical consistency module 部署。**

---

## 2.2 为什么比“LLM 教 ViT”更高级？

朴素故事：

```text
LLM 教 ViT，所以 ViT 变强。
```

新故事：

```text
LLM/VLM 本身不是关键，关键是老师如何出题。
只有当 instruction 被设计成必须依赖具体图像时，语言监督才会变成视觉监督。
```

这能解释你所有结果：

| 现象 | 新故事解释 |
|---|---|
| fixed JSON 差 | 模板 token shortcut |
| 普通 QA 不够强 | 视觉依赖不足 |
| CF 提高反事实能力 | 学会真假临床陈述 |
| SHUF / SAMEQ 强 | 必须区分具体图像 |
| SHUF-K4 强 | 多负样本加强 image matching |
| CCSH 强 | 一致性能力可以被部署读出 |
| curriculum 需重做 | 老师出题顺序和难度很关键 |
| text-only scaffold 需对比 | 没有视觉语言对齐的老师可能不够好 |

---

## 2.3 方法结构图文字版

```text
Radiology report
   ↓
Clinical statement extraction
   ↓
Counterfactual statement generation
   ↓
Clinical visual curriculum:
   Stage 1: Basic QA
   Stage 2: Counterfactual statement exam
   Stage 3: Same-question image exam
   Stage 4: Multi-negative image exam
   ↓
Pretrained VLM teacher:
   freeze LLM
   train vision tower + connector
   ↓
Deployable student:
   vision tower
   + CCSH / CEQ+CCSH / classifier
```

---

# 3. 主要候选方法 family

下一阶段不再把所有 run 平铺。分成 5 个 family，每个 family 后面只选最强 finalist。

---

## Family A：Direct Image-Specific Grounding

### 目标

验证不做复杂课程，直接用最强 image-specific 任务是否已经足够。

### 必跑

| Run ID | Data | Module | Steps | 目的 |
|---|---|---|---:|---|
| A1-Direct-SAMEQ | SAMEQ | none | 8k | 干净 image-specific baseline |
| A2-Direct-SHUF-K4 | SHUF-K4 | none | 8k | multi-negative baseline |
| A3-SAMEQ-K4-Hybrid | SAMEQ + K4 | none | 8k | 同题换图 + 多负图 |
| A4-SAMEQ-CF-compatible | SAMEQ + CF rows | none | 8k | 让 SAMEQ 有 CF/A-B gate |
| A5-SHUF-K4-CF-compatible | K4 + CF rows | none | 8k | 修 K4 的 CF/A-B 短板 |

空表：

| Run | CheXpert AUC | External AUC | Hard shuffle | CF acc | A/B swap | Leakage | Cost | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A1-Direct-SAMEQ |  |  |  |  |  |  |  |  |
| A2-Direct-SHUF-K4 |  |  |  |  |  |  |  |  |
| A3-SAMEQ-K4-Hybrid |  |  |  |  |  |  |  |  |
| A4-SAMEQ-CF-compatible |  |  |  |  |  |  |  |  |
| A5-SHUF-K4-CF-compatible |  |  |  |  |  |  |  |  |

---

## Family B：Clinical Visual Curriculum Pretraining, CVCP

### 目标

验证 “VLM 老师按难度出题” 是否优于 direct SAMEQ / SHUF-K4。

### 课程任务类型

| Stage | 任务 | 作用 |
|---|---|---|
| Stage 1 | Basic clinical QA | 学 finding、状态、基本问答 |
| Stage 2 | Counterfactual statement | 学真假医学陈述 |
| Stage 3 | Same-question image exam | 同题换图，逼模型看图 |
| Stage 4 | Multi-negative SHUF | 多负图，加强 image matching |
| Stage 5 | Optional CCSH / CEQ | 部署读出和可解释性 |

---

### B1. Hard-stage curriculum

```text
0-20% steps: Basic QA
20-45% steps: CF
45-75% steps: SAMEQ
75-100% steps: SHUF-K4
```

### B2. Progressive curriculum

每个阶段不是硬切，而是比例逐渐变。

| Step range | Basic QA | CF | SAMEQ | SHUF-K4 |
|---|---:|---:|---:|---:|
| 0-20% | 50 | 30 | 20 | 0 |
| 20-50% | 25 | 35 | 25 | 15 |
| 50-80% | 10 | 25 | 35 | 30 |
| 80-100% | 5 | 15 | 40 | 40 |

### B3. Replay curriculum

后期保留一定比例前期任务，防止遗忘：

```text
final stage:
70% hard tasks
30% replay from basic/CF
```

### B4. Case-driven curriculum

根据 dev failure 动态调采样比例：

```text
laterality 错多 -> 增加 laterality SAMEQ
state flip 错多 -> 增加 present/absent CF
hard shuffle 错多 -> 增加 K4 / mined negatives
```

### 必跑

| Run ID | Design | Data scale | Steps | 目的 |
|---|---|---:|---:|---|
| B1-CVCP-HardStage-10k | hard stage | 10k | 12k | 基础 curriculum |
| B2-CVCP-Progressive-10k | progressive | 10k | 12k | 推荐候选 |
| B3-CVCP-Replay-10k | progressive + replay | 10k | 12k | 防遗忘 |
| B4-CVCP-CaseDriven-10k | dynamic sampling | 10k | 12k | case-driven |
| B5-CVCP-Progressive-Full | progressive | full available | 20k | upper bound |
| B6-CVCP-SAMEQFinal | progressive, last stage SAMEQ-heavy | 10k/full | 12k/20k | 最干净 grounding |
| B7-CVCP-K4Final | progressive, last stage K4-heavy | 10k/full | 12k/20k | 多负样本 final |

空表：

| Run | Data | Steps | CheXpert AUC | External AUC | Hard shuffle | CF acc | A/B swap | Leakage | Cost | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B1-CVCP-HardStage-10k |  |  |  |  |  |  |  |  |  |  |
| B2-CVCP-Progressive-10k |  |  |  |  |  |  |  |  |  |  |
| B3-CVCP-Replay-10k |  |  |  |  |  |  |  |  |  |  |
| B4-CVCP-CaseDriven-10k |  |  |  |  |  |  |  |  |  |  |
| B5-CVCP-Progressive-Full |  |  |  |  |  |  |  |  |  |  |

---

## Family C：Deployable Clinical Consistency Module

### 目标

把训练 recipe 升级成 TMI-friendly deployable module。

---

## C1. CCSH：Clinical Consistency Scoring Head

### 作用

判断：

```text
image 是否支持 clinical statement
```

输出：

```text
support / contradict / uncertain
```

### 必做 ablation

| Run ID | Vision tower | CCSH? | 目的 |
|---|---|---|---|
| C1-Base-CCSH | raw Qwen3-VL | yes | 看 CCSH 自己多强 |
| C2-SAMEQ-CCSH | SAMEQ-trained | yes | 看 SAMEQ 是否增强 consistency |
| C3-K4-CCSH | SHUF-K4-trained | yes | 看 K4 是否增强 consistency |
| C4-CVCP-CCSH | curriculum-trained | yes | 看课程是否增强 consistency |
| C5-RandomStatement-CCSH | trained vision | random statements | 负控制 |

---

## C2. CEQ：Clinical Evidence Query

### 作用

每个 finding 一个 query，在 patch tokens 里找证据。

```text
patch tokens -> clinical queries -> field evidence embeddings
```

### 必做组合

| Run ID | Module |
|---|---|
| C6-SAMEQ-CEQ | SAMEQ + CEQ |
| C7-SAMEQ-CEQ-CCSH | SAMEQ + CEQ + CCSH |
| C8-K4-CEQ-CCSH | K4 + CEQ + CCSH |
| C9-CVCP-CEQ-CCSH | curriculum + CEQ + CCSH |

---

## C3. AUCH：Answerability-Uncertainty Calibration Head

### 作用

部署时输出：

```text
answerability
uncertainty
state
```

用于医学语义和校准。

| Run ID | Module |
|---|---|
| C10-SAMEQ-AUCH | SAMEQ + AUCH |
| C11-CVCP-AUCH | CVCP + AUCH |
| C12-CEQ-CCSH-AUCH | CEQ+CCSH+AUCH |

---

### Family C 空表

| Run | Backbone | Module | CheXpert AUC | External AUC | Consistency AUC | AUPRC | ECE | Hard shuffle | Interpretability | Decision |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| C1-Base-CCSH |  | CCSH |  |  |  |  |  |  |  |  |
| C2-SAMEQ-CCSH |  | CCSH |  |  |  |  |  |  |  |  |
| C3-K4-CCSH |  | CCSH |  |  |  |  |  |  |  |  |
| C4-CVCP-CCSH |  | CCSH |  |  |  |  |  |  |  |  |
| C7-SAMEQ-CEQ-CCSH |  | CEQ+CCSH |  |  |  |  |  | attention maps |  |
| C9-CVCP-CEQ-CCSH |  | CEQ+CCSH |  |  |  |  |  | attention maps |  |

---

## Family D：Hard Negative Mining / Memory Bank

### 目标

把 SHUF 从静态规则负样本升级为算法模块。

---

## D1. HNMB-static

先训练一个 baseline，抽 embedding，离线挖最相似但答案不同的负样本。

---

## D2. HNMB-online

每 N steps 更新 memory bank，动态替换 hard negatives。

---

## D3. False-negative verifier

用 report label / statement / optionally VLM verifier 过滤可疑负样本，防止 hard negative 其实也支持答案。

---

### 必跑

| Run ID | Design | Data | Steps | 目的 |
|---|---|---|---:|---|
| D1-HNMB-static-SAMEQ | static mined negatives | SAMEQ | 8k | mined vs rule negative |
| D2-HNMB-static-K4 | static mined negatives | K4 | 8k | K4 加 mined |
| D3-HNMB-online | online memory | mixed | 12k | dynamic mining |
| D4-HNMB-CCSH | HNMB + CCSH | mixed | 12k | 模块组合 |
| D5-HNMB-CEQ-CCSH | HNMB + CEQ + CCSH | mixed | 12k | 最强候选 |

空表：

| Run | Negative type | CheXpert AUC | External AUC | Hard shuffle | CF acc | False-negative rate | Cost | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| D1-HNMB-static-SAMEQ |  |  |  |  |  |  |  |  |
| D2-HNMB-static-K4 |  |  |  |  |  |  |  |  |
| D3-HNMB-online |  |  |  |  |  |  |  |  |
| D4-HNMB-CCSH |  |  |  |  |  |  |  |  |

---

## Family E：Teacher Model Comparison

### 目标

证明方法不是 Qwen3-VL 特例；同时证明 coupled VLM 比 text-only scaffold 更合理。

---

## E1. 当前主 teacher

```text
Qwen3-VL-2B
```

---

## E2. 后续对比 teacher

根据本地可用模型选择：

| Model | Type | Role |
|---|---|---|
| Qwen3-VL-2B | VLM | current main |
| InternVL | VLM | strong VLM comparator |
| LLaVA / Llama-based VLM | VLM | architecture comparator |
| Qwen3.5-2B text-only | text-only | negative scaffold |
| Qwen-Coder text-only | text-only | template/coder bias control |
| raw vision tower | vision only | baseline |

---

## E3. 统一实验设置

统一用一个中等成本 recipe：

```text
CVCP-Progressive-10k
or SAMEQ+CCSH
or SHUF-K4+CCSH
```

不要每个 teacher 都跑所有实验。

### 必跑

| Run ID | Teacher | Route | 目的 |
|---|---|---|---|
| E1-Qwen3VL-CVCP | Qwen3-VL | VLM-coupled | main |
| E2-InternVL-CVCP | InternVL | VLM-coupled | VLM comparator |
| E3-LlamaVLM-CVCP | Llama-based VLM | VLM-coupled | architecture comparator |
| E4-Qwen35-text-scaffold | Qwen3.5 text | text scaffold | negative control |
| E5-QwenCoder-text-scaffold | Qwen-Coder | text scaffold | template bias |
| E6-RawVision | raw vision | no teacher | baseline |

空表：

| Run | Teacher | Type | CheXpert AUC | External AUC | Hard shuffle | CCSH AUC | Cost | Decision |
|---|---|---|---:|---:|---:|---:|---:|---|
| E1-Qwen3VL | Qwen3-VL | VLM |  |  |  |  |  |  |
| E2-InternVL | InternVL | VLM |  |  |  |  |  |  |
| E3-LlamaVLM | Llama-VLM | VLM |  |  |  |  |  |  |
| E4-Qwen35-text | Qwen3.5 | text-only |  |  |  |  |  |  |
| E5-QwenCoder-text | Qwen-Coder | text-only |  |  |  |  |  |  |
| E6-RawVision | none | vision |  |  |  |  |  |  |

---

# 4. 外部验证设计

## 4.1 NIH 降级

NIH 不再作为主 external。  
保留方式：

```text
Appendix: noisy label-mapping stress test
```

或者在主文只简单提一句：

```text
NIH is used as an auxiliary stress test due to known label mapping ambiguity.
```

---

## 4.2 主 external 选择

优先顺序：

| Priority | Dataset | 理由 |
|---:|---|---|
| 1 | VinDr-CXR | 胸片、临床标注更强、有位置/lesion 信息 |
| 2 | PadChest | 大规模、外部性强 |
| 3 | MIMIC-CXR | 如果未用于训练，可做 external；如果用于训练，只能 source-domain |
| 4 | NIH | appendix/stress |

---

## 4.3 主 external 评估

| Metric | 目的 |
|---|---|
| Macro-AUC | 分类 |
| Macro-AUPRC | 长尾 |
| ECE / Brier | 校准 |
| Per-label AUC | label mapping |
| Consistency score | CCSH 模块 |
| Case study | 临床解释 |
| Subgroup | rare/high-null/uncertain |

空表：

| Run | External dataset | Macro-AUC | Macro-AUPRC | ECE | Brier | Per-label notes | Decision |
|---|---|---:|---:|---:|---:|---|---|
| Base | VinDr/PadChest |  |  |  |  |  |  |
| SAMEQ | VinDr/PadChest |  |  |  |  |  |  |
| SHUF-K4 | VinDr/PadChest |  |  |  |  |  |  |
| CVCP | VinDr/PadChest |  |  |  |  |  |  |
| CVCP+CCSH | VinDr/PadChest |  |  |  |  |  |  |
| CEQ+CCSH | VinDr/PadChest |  |  |  |  |  |  |

---

# 5. 数据生成与清洗

## 5.1 Clinical statement extraction

从报告抽：

```text
finding
state
laterality
location
severity
uncertainty
answerability
evidence span
```

输出：

```json
{
  "finding": "pleural_effusion",
  "state": "present",
  "laterality": "left",
  "severity": "small",
  "uncertainty": "definite",
  "evidence_span": "small left pleural effusion"
}
```

---

## 5.2 Counterfactual generation

基于真实事实生成：

| Flip | Example |
|---|---|
| state flip | present ↔ absent |
| laterality flip | left ↔ right |
| uncertainty flip | definite ↔ uncertain |
| severity flip | small ↔ large |
| answerability flip | answerable ↔ not answerable |

---

## 5.3 SAMEQ generation

确保：

```text
same question
different images
different answer
```

必须避免：

- question 泄露答案；
- A/B 永远正确；
- hard negative 其实也支持答案。

---

## 5.4 K-negative generation

每个正样本配 K 个负图：

| K | 说明 |
|---:|---|
| 1 | baseline |
| 2 | moderate |
| 4 | recommended |
| 8 | expensive |

---

## 5.5 Leakage audit

必须检查：

| Check | Reject if |
|---|---|
| question contains evidence span | yes |
| question contains answer | yes |
| question says “report says” | yes |
| option A always correct | rebalance |
| A/B length imbalance | flag |
| false hard negative | flag/reject |
| duplicate question | downsample |

空表：

| Dataset | Rows | Accepted % | Leakage % | A/B balance | False negative % | Duplicate % | Manual pass % |
|---|---:|---:|---:|---:|---:|---:|---:|
| SAMEQ-10k |  |  |  |  |  |  |  |
| SHUF-K4-10k |  |  |  |  |  |  |  |
| CVCP-10k |  |  |  |  |  |  |  |
| HNMB-10k |  |  |  |  |  |  |  |

---

# 6. 训练目标设计

## 6.1 基础 answer-only loss

```text
L_answer = CE(answer tokens)
```

只算 answer，不算 question。

---

## 6.2 Image-shuffle margin

```text
L_img = max(0, m + NLL(pos_image, Q, A) - NLL(neg_image, Q, A))
```

---

## 6.3 Answer margin

```text
L_ans_margin = max(0, m + NLL(I, Q, A_correct) - NLL(I, Q, A_wrong))
```

---

## 6.4 CCSH loss

```text
L_ccsh = CE(support / contradict / uncertain)
```

---

## 6.5 CEQ auxiliary loss

```text
L_ceq = sum_f CE(state_f | evidence_query_f)
```

---

## 6.6 AUCH calibration loss

```text
L_auch = BCE(answerability) + BCE(uncertainty) + CE(state)
```

Optional:

```text
Brier / ECE proxy
```

---

## 6.7 总 loss 候选

### Direct SAMEQ / SHUF

```text
L = L_answer + λ_img L_img + λ_ans L_ans_margin
```

### CVCP

```text
L = L_answer + λ_img(t) L_img + λ_ans(t) L_ans_margin
```

其中 λ 随训练阶段逐步变大。

### CVCP + CCSH

```text
L = L_answer + λ_img L_img + λ_ans L_ans_margin + λ_ccsh L_ccsh
```

### CEQ + CCSH

```text
L = L_answer + λ_img L_img + λ_ccsh L_ccsh + λ_ceq L_ceq
```

空表：

| Run | λ_img | λ_ans | λ_ccsh | λ_ceq | λ_auch | Notes |
|---|---:|---:|---:|---:|---:|---|
| SAMEQ |  |  |  |  |  |  |
| K4 |  |  |  |  |  |  |
| CVCP |  |  |  |  |  |  |
| CVCP+CCSH |  |  |  |  |  |  |
| CEQ+CCSH |  |  |  |  |  |  |

---

# 7. 实验执行计划

## Phase 0：A800 环境与数据准备

| Task | Output | Priority |
|---|---|---|
| A800 memory probes | memory table | must |
| External dataset selection | VinDr/PadChest manifest | must |
| Leakage audit v3 | audit table | must |
| SAMEQ-CF-compatible generation | jsonl | must |
| SHUF-K4-CF-compatible generation | jsonl | must |
| CCSH statement dataset | jsonl | must |
| CEQ labels/field queries | manifest | recommended |
| HNMB mining pipeline | embeddings + negatives | recommended |

---

## Phase 1：主线骨架快速筛选

| Run | Steps | Seeds | Priority |
|---|---:|---:|---|
| A1-Direct-SAMEQ | 8k | 1-3 | must |
| A2-Direct-SHUF-K4 | 8k | 1-3 | must |
| A3-SAMEQ-K4-Hybrid | 8k | 1 | must |
| B2-CVCP-Progressive-10k | 12k | 1 | must |
| C2-SAMEQ-CCSH | 8k + head | 1 | must |
| C3-K4-CCSH | 8k + head | 1 | must |
| C7-SAMEQ-CEQ-CCSH | 8k + head | 1 | recommended |

---

## Phase 2：模块组合

| Run | Steps | Priority |
|---|---:|---|
| C4-CVCP-CCSH | 12k | must |
| C9-CVCP-CEQ-CCSH | 12k | recommended |
| D1-HNMB-static-SAMEQ | 8k | recommended |
| D2-HNMB-static-K4 | 8k | recommended |
| D4-HNMB-CCSH | 12k | recommended |
| C12-CEQ-CCSH-AUCH | 12k | optional |
| DRA on best candidate | 8k + adapter | external-dependent |

---

## Phase 3：scale

| Run | Data | Steps | Priority |
|---|---|---:|---|
| CVCP-Progressive-Full | full | 20k | if Phase1 positive |
| SAMEQ-10k/Full | 10k/full | 12k-20k | if SAMEQ still best |
| SHUF-K4-10k/Full | 10k/full | 12k-20k | if K4 still best |
| Best+CCSH full | full | 20k | final candidate |

---

## Phase 4：teacher comparison

只拿 Phase 1/2 最强 recipe 跑。

| Teacher | Run | Priority |
|---|---|---|
| Qwen3-VL | main | already |
| InternVL | same recipe | must if available |
| LLaVA/Llama-VLM | same recipe | recommended |
| Qwen3.5 text-only | scaffold control | must |
| Qwen-Coder | old scaffold | optional |
| raw vision | no teacher | must |

---

## Phase 5：locked final evaluation

每个 family 只选一个 finalist。

| Family | Finalist |
|---|---|
| Direct grounding |  |
| Curriculum |  |
| Consistency module |  |
| Evidence module |  |
| Hard-negative mining |  |
| Teacher comparison |  |

最终评估：

```text
3 seeds
CheXpert
VinDr/PadChest
hard shuffle
CF / A-B swap
AUPRC / ECE
case study
cost
```

---

# 8. 任务并行排程建议

## Batch 1：数据 + probe +轻量并行

同时运行：

```text
CPU/API:
- generate SAMEQ-CF-compatible
- generate SHUF-K4-CF-compatible
- generate CCSH statement data
- leakage audit

GPU:
- MEM-SAMEQ
- MEM-SHUF-K4
- MEM-CCSH
```

---

## Batch 2：主骨架训练

如果显存允许：

```text
Run 1: A1-Direct-SAMEQ
Run 2: A2-Direct-SHUF-K4
```

但默认建议：

```text
先单跑 A1
再单跑 A2
评估并行跑
```

---

## Batch 3：模块 head 并行

可并行：

```text
Base+CCSH
SAMEQ+CCSH
K4+CCSH
SAMEQ+CEQ
K4+CEQ
AUCH
```

---

## Batch 4：Curriculum 大 run

```text
B2-CVCP-Progressive-10k
B3-CVCP-Replay-10k
```

如果显存允许，可一主一小并行；不建议两个 12k full run 同时抢 I/O。

---

## Batch 5：Finalists multi-seed

选 3 个 finalist：

```text
Finalist 1: Direct SAMEQ/K4
Finalist 2: CVCP
Finalist 3: Module method
```

各跑 3 seeds。

---

# 9. 决策规则

## 9.1 进入 final candidate 的最低条件

| Metric | Requirement |
|---|---|
| CheXpert AUC | 不低于 direct SAMEQ/K4 baseline 超过 0.003 |
| External AUC | 不明显下降，最好提升 |
| Hard shuffle delta | > 0.10，最好 > 0.20 |
| CF acc | if applicable > 0.85 |
| A/B swap | if applicable > 0.85 |
| Leakage | < 10% |
| AUPRC/ECE | 不能明显变差 |
| Cost | 可解释 |
| Module story | 必须讲得通 |

---

## 9.2 如果没有方法所有指标最优

不要强说 best。  
用 Pareto frontier：

| Axis | Meaning |
|---|---|
| x | CheXpert AUC |
| y | hard shuffle delta |
| color | external AUC |
| shape | method family |
| size | cost |

最终选：

```text
best Pareto-balanced method
```

并明确：

> classification-oriented best 和 grounding-oriented best 可能不同。

---

## 9.3 对 CCSH 的特殊判断

| 结果 | 解释 |
|---|---|
| Base+CCSH 很强，训练后+CCSH 没更好 | CCSH 是 head-only，不支持 LLM teacher 主线 |
| SAMEQ+CCSH > Base+CCSH | LLM/VLM curriculum improves vision representation |
| CEQ+CCSH > CCSH | evidence queries 提供额外价值 |
| CCSH 提升 consistency 但不提升 classification | 做 secondary module，不做主方法 |
| CCSH 提升 external/calibration | 强 TMI 点 |

---

# 10. 最终论文故事可能走向

## Story A：Curriculum wins

标题：

```text
Clinical Visual Curriculum Pretraining with VLM Teachers for Chest X-ray Vision Encoders
```

主线：

```text
VLM teacher 按视觉依赖难度出题，逐步训练 ViT。
```

---

## Story B：SAMEQ / SHUF-K wins

标题：

```text
Image-Specific Clinical Consistency Pretraining for Chest X-ray Vision Encoders
```

主线：

```text
关键不是普通课程，而是同题换图和多负图，让模型必须看具体图像。
```

---

## Story C：CCSH / CEQ+CCSH wins

标题：

```text
Deployable Clinical Consistency Learning for Chest X-ray Vision Encoders
```

主线：

```text
用 VLM teacher 训练 vision tower，再用 clinical consistency module 部署。
```

---

## Story D：Teacher comparison is strongest

标题：

```text
When Can VLM Teachers Improve Medical Vision Encoders?
```

主线：

```text
不同 VLM teacher、text-only control、curriculum 难度决定是否有效。
```

---

# 11. Codex / 实验同学执行清单

## 11.1 数据脚本

```text
scripts/generate_clinical_statements.py
scripts/generate_counterfactual_statements.py
scripts/generate_sameq_cf_compatible.py
scripts/generate_shuf_k_cf_compatible.py
scripts/generate_ccsh_statement_pairs.py
scripts/generate_ceq_field_queries.py
scripts/audit_leakage_v3.py
scripts/audit_false_hard_negatives.py
```

---

## 11.2 训练脚本

```text
scripts/train_qwen3vl_cvcp.py
scripts/train_qwen3vl_sameq.py
scripts/train_qwen3vl_shuf_k.py
scripts/train_ccsh_head.py
scripts/train_ceq_ccsh.py
scripts/train_hnmb.py
scripts/train_teacher_comparison.py
```

---

## 11.3 评估脚本

```text
scripts/eval_lp_chexpert.py
scripts/eval_external_vindr_padchest.py
scripts/eval_visual_dependence.py
scripts/eval_cf_ab_swap.py
scripts/eval_ccsh_consistency.py
scripts/eval_calibration_auprc.py
scripts/eval_case_study.py
scripts/plot_pareto_frontier.py
```

---

## 11.4 每个 run 必须输出

```text
config_snapshot.json
training_log.txt
progress.json
metrics_final.json
metrics_step_*.json
vision_export_manifest.json
lp_chexpert_results.md
external_results.md
visual_dependence_results.md
cf_ab_results.md
calibration_auprc.md
cost_table.md
case_study_samples.md
```

---

# 12. 第一批推荐实际运行列表

考虑 A800 80GB，第一批建议这样排：

## 12.1 必跑 8 个

| Priority | Run |
|---:|---|
| 1 | A1-Direct-SAMEQ |
| 2 | A2-Direct-SHUF-K4 |
| 3 | A4-SAMEQ-CF-compatible |
| 4 | A5-SHUF-K4-CF-compatible |
| 5 | C1-Base-CCSH |
| 6 | C2-SAMEQ-CCSH |
| 7 | C3-K4-CCSH |
| 8 | B2-CVCP-Progressive-10k |

---

## 12.2 推荐再跑 8 个

| Priority | Run |
|---:|---|
| 9 | C7-SAMEQ-CEQ-CCSH |
| 10 | C8-K4-CEQ-CCSH |
| 11 | B3-CVCP-Replay-10k |
| 12 | B6-CVCP-SAMEQFinal |
| 13 | D1-HNMB-static-SAMEQ |
| 14 | D2-HNMB-static-K4 |
| 15 | E2-InternVL-CVCP |
| 16 | E4-Qwen35-text-scaffold |

---

## 12.3 Finalist 后再跑

| Run |
|---|
| Best direct grounding seed3 |
| Best curriculum seed3 |
| Best module seed3 |
| Best external dataset full |
| Best teacher comparison subset |

---

# 13. 空总表

| Family | Run | Status | Seed | CheXpert AUC | External AUC | Hard shuffle | CF acc | A/B swap | CCSH AUC | ECE | Cost | Final role |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Direct | A1-Direct-SAMEQ |  |  |  |  |  |  |  |  |  |  |  |
| Direct | A2-Direct-SHUF-K4 |  |  |  |  |  |  |  |  |  |  |  |
| Direct | A3-SAMEQ-K4-Hybrid |  |  |  |  |  |  |  |  |  |  |  |
| Curriculum | B2-CVCP-Progressive-10k |  |  |  |  |  |  |  |  |  |  |  |
| Curriculum | B3-CVCP-Replay-10k |  |  |  |  |  |  |  |  |  |  |  |
| Module | C1-Base-CCSH |  |  |  |  |  |  |  |  |  |  |  |
| Module | C2-SAMEQ-CCSH |  |  |  |  |  |  |  |  |  |  |  |
| Module | C3-K4-CCSH |  |  |  |  |  |  |  |  |  |  |  |
| Module | C7-SAMEQ-CEQ-CCSH |  |  |  |  |  |  |  |  |  |  |  |
| Mining | D1-HNMB-static-SAMEQ |  |  |  |  |  |  |  |  |  |  |  |
| Teacher | E2-InternVL-CVCP |  |  |  |  |  |  |  |  |  |  |  |
| Teacher | E4-Qwen35-text-scaffold |  |  |  |  |  |  |  |  |  |  |  |

---

# 14. 最后建议

最重要的是，不要再问：

```text
哪个单独 run 看起来综合最好？
```

下一阶段要问：

```text
哪条机制 family 真的成立？
```

优先级：

```text
Direct SAMEQ / SHUF-K4
→ CVCP curriculum
→ CCSH / CEQ+CCSH module
→ HNMB hard-negative mining
→ teacher model comparison
→ locked external evaluation
```

如果 `CVCP+CCSH` 或 `CEQ+CCSH` 能在 CheXpert、external、hard shuffle、CCSH consistency、case study 上同时站住，这篇论文就会比单纯 “LLM 教 ViT” 或 “SHUF 最优” 高一个层级。
