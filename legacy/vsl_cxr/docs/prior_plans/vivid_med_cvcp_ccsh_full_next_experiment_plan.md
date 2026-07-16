# VIVID-Med 下一阶段完整实验文档 v3  
## Clinical Visual Curriculum + Deployable Consistency Modules

> 版本：2026-07-04  
> 目标：把现在的结果从“很多 ablation”重新整理成一套有故事、有模块、有统计验证、有外部数据、有 locked comparison 的 TMI 级实验路线。  
> 当前核心判断：**不要急着把 SHUF-TW-clinical 定为最终方法；下一阶段主线应围绕“VLM teacher 设计临床视觉课程，让 ViT 学会 image-specific clinical consistency”展开。**

---

# 0. 总览：下一阶段要解决什么？

当前已有实验说明：

1. **单纯 fixed JSON / 普通 QA 不够。**  
   它们可能让模型学格式、模板、标签先验，而不是学具体图像证据。

2. **SHUF / SAMEQ / multi-negative 方向更接近真正的 visual supervision。**  
   尤其 SAMEQ：同一个问题、不同图像、不同答案，可以最大限度减少文本 shortcut。

3. **SHUF-TW-clinical 不是最终方法。**  
   它曾经在单 seed 下是综合 candidate，但多 seed 之后，对 SHUF-3k 的 AUC 优势不稳定。

4. **CCSH 是目前最强的模块信号，但不能单独当主线。**  
   它应该作为 deployable readout module，证明训练后的 vision tower 能做 image-clinical statement consistency。

5. **Curriculum 方向仍然值得做。**  
   现在更强显卡可用，应该重新设计成真正的 clinical visual curriculum，而不是简单 P3/CF/SHUF 拼接。

6. **NIH 不适合作为主外部验证。**  
   NIH 可以降级到 appendix 或取消主文地位，主外部验证建议换成一个更清晰的 CXR 数据集，例如 VinDr-CXR 或 PadChest。

7. **后续要做模型对比。**  
   当前先用 Qwen3-VL，但最终要对比 InternVL、LLaVA/Llama-based VLM、Qwen3.5 text-only scaffold 等，证明不是某个模型偶然有效。

---

# 1. 新主线设计

## 1.1 不推荐的主线

不要写成：

```text
LLM teaches ViT, so ViT becomes better.
```

这个太朴素，也容易和 ViTP 撞。

也不要写成：

```text
We tried many variants and SHUF-TW-clinical is best overall.
```

这个像随机试实验后挑结果。

---

## 1.2 推荐主线

推荐主线：

```text
Clinical Visual Curriculum Pretraining with VLM Teachers
```

中文：

```text
用 VLM 老师构造临床视觉课程，训练可部署胸片视觉编码器。
```

核心思想：

> VLM/LLM 不是简单输出答案，而是作为“临床视觉考试官”。  
> 它从报告中构造一组越来越难的临床视觉任务：基础 QA、反事实陈述、同题换图、多负样本、图像-陈述一致性。  
> ViT 通过这些任务学习图像特异的临床证据。  
> 训练后丢掉 LLM，只部署 vision tower + 可选 CCSH/CEQ 模块。

---

## 1.3 论文故事结构

可以这样讲：

```text
1. Fixed schema generation can collapse into template learning.
2. Basic report-grounded QA improves language alignment but is not sufficiently image-specific.
3. Counterfactual questions teach statement-level discrimination.
4. Same-question/different-image and multi-negative SHUF force the model to use image-specific evidence.
5. A clinical visual curriculum organizes these tasks from easy to hard.
6. Deployable modules such as CCSH and CEQ read out the learned image-statement consistency without using the LLM at deployment.
```

中文：

```text
固定 schema 容易学模板；普通 QA 虽然有医学语义，但不一定依赖具体图像。我们把临床 instruction 按视觉依赖难度组织成课程：基础问答、反事实陈述、同题换图、多负样本和图像-陈述一致性。这个课程让 VLM 老师真正训练 ViT 看图。训练后部署不需要 LLM，只使用视觉编码器和轻量临床一致性模块。
```

---

# 2. 总体实验路线图

## Phase 0：数据和外部验证准备

目标：

- 选择新的主外部数据集；
- 准备 label mapping；
- 建立 leakage audit 和 false-negative audit；
- 准备训练/评估 manifest。

## Phase 1：锁定核心 baselines

目标：

- 重新跑或整理 Direct-SAMEQ、SHUF-K4、SHUF-3k、SHUF-TW-clinical；
- 建立 multi-seed 和 bootstrap CI；
- 明确哪一个 family 最稳定。

## Phase 2：Clinical Visual Curriculum 实验

目标：

- 从 direct training 变成 curriculum training；
- 试 single-stage mixture、hard stage、progressive schedule、case-driven schedule；
- 让 curriculum 有公平机会，不因为数据量/训练不够被过早否定。

## Phase 3：模块组合实验

目标：

- 测 CCSH、CEQ、HNMB、AUCH、DRA、CDCS；
- 重点验证 SAMEQ+CCSH、SHUF-K4+CCSH、CEQ+CCSH；
- 形成 TMI 级别的模块贡献。

## Phase 4：Hard negative / SHUF++ 实验

目标：

- 修 SAMEQ 和 SHUF-K4 的 gate 问题；
- 加 CF-compatible rows；
- 做 K-negative、mined negative、in-batch negative、dual-margin。

## Phase 5：Token weighting 与 loss 设计

目标：

- 不把 TW 当主方法；
- 只作为 curriculum / SHUF / CCSH 后面的增强；
- 比较 TW-light、TW-visual、TW-clinical-balanced。

## Phase 6：VLM teacher 对比

目标：

- 用相同数据和训练协议对比 Qwen3-VL、InternVL、LLaVA/Llama-based VLM、Qwen3.5 text-only scaffold；
- 证明一套 VLM 初始化比散装 text scaffold 更合理；
- 证明你的方法不是只对 Qwen3-VL 有效。

## Phase 7：Locked final comparison

目标：

- 每个 family 只选一个 finalist；
- 在固定评估协议下统一多 seed、外部验证、AUPRC、ECE、case study、cost；
- 避免“从一堆实验里挑最优”。

---

# 3. Phase 0：外部数据与数据质量准备

## 3.1 外部数据集选择

当前 NIH 不作为主 external。建议：

| Dataset | 推荐级别 | 用途 | 优点 | 风险 |
|---|---|---|---|---|
| VinDr-CXR | 首选 | 主 external | 临床标注质量较高，有 finding/位置相关信息 | 数据获取/标签映射需要确认 |
| PadChest | 备选 | 主 external | 数据量大，标签丰富 | 标签体系复杂，mapping 需要审计 |
| MIMIC-CXR | 条件使用 | source 或 external | report-rich，适合 instruction generation | 如果用于训练，就不能作为真正 external |
| NIH | appendix/取消主文 | stress test | 已有管线 | label mapping/noise 导致差距压缩 |

### 外部数据决策表

| External candidate | Available? | Images | Labels usable | Label mapping confidence | Can be main external? | Notes |
|---|---|---:|---|---|---|---|
| VinDr-CXR |  |  |  |  |  |  |
| PadChest |  |  |  |  |  |  |
| MIMIC-CXR |  |  |  |  |  |  |
| NIH | yes |  |  | low/medium | no/main appendix |  |

---

## 3.2 Label mapping audit

必须做 label mapping 表，不能直接硬对齐。

| Source finding | External label | Mapping level | Confidence | Risk | Use in macro-AUC? |
|---|---|---|---|---|---|
| Pleural Effusion |  | exact / close / weak |  |  |  |
| Pneumothorax |  | exact / close / weak |  |  |  |
| Cardiomegaly |  | exact / close / weak |  |  |  |
| Edema |  | exact / close / weak |  |  |  |
| Consolidation |  | exact / close / weak |  |  |  |
| Atelectasis |  | exact / close / weak |  |  |  |
| Lung Opacity |  | exact / close / weak |  |  |  |
| Fracture |  | exact / close / weak |  |  |  |
| Lung Lesion |  | exact / close / weak |  |  |  |

### 决策规则

| 情况 | 决策 |
|---|---|
| exact / close labels >= 6 | 可以做主 external macro-AUC |
| exact labels < 4 | 不适合作主 external |
| 有位置标注 | 可做 grounding/location secondary |
| 标签噪声高 | 主写 AUPRC / per-label，不强调 macro-AUC |

---

## 3.3 Leakage audit v3

所有生成 instruction 都必须经过新版 audit。

### 自动规则

| Check | Reject / Flag |
|---|---|
| question contains evidence_span | reject |
| question contains exact answer | reject |
| question contains “report says” | reject for training |
| question asks laterality and includes laterality | reject |
| question asks severity and includes severity | reject |
| A/B correct answer imbalance | rebalance |
| A/B option length imbalance | flag |
| duplicate question same image/finding | downsample |
| answer inferable from question alone | flag |
| hard negative same answer | reject |
| false hard negative suspected | manual audit pool |

### 数据质量表

| Dataset | N images | N instructions | Accepted % | Leakage % | A/B balance | False-negative rate | Manual pass % | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| SAMEQ-v2 |  |  |  |  |  |  |  |  |
| SHUF-K4-v2 |  |  |  |  |  |  |  |  |
| CVCP-progressive |  |  |  |  |  |  |  |  |
| CCSH-statements |  |  |  |  |  |  |  |  |
| CEQ-statements |  |  |  |  |  |  |  |  |

---

# 4. Phase 1：核心 baseline 与 seed 稳定性

## 4.1 必须锁定的 baseline

| Run ID | Description | Why |
|---|---|---|
| Base-Qwen3VL | 原始 Qwen3-VL vision tower | 起点 |
| SHUF-3k | 旧 direct SHUF baseline | 当前基本 baseline |
| SAMEQ-SHUF-3k | strongest image-specific diagnostic | 当前强 grounding |
| SHUF-K4 | multi-negative baseline | 当前强 grounding/NIH |
| SHUF-TW-clinical | 旧 gate candidate | 需要降级/验证 |
| P2-value-only | JSON template loss diagnostic | 解释 fixed JSON |
| P2-field-query | field-level schema QA | 解释 schema QA |

---

## 4.2 Seed stability

至少 3 seeds，重要候选 5 seeds。

| Run | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean | Std | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| SHUF-3k CheXpert AUC |  |  |  |  |  |  |  |  |
| SAMEQ-SHUF CheXpert AUC |  |  |  |  |  |  |  |  |
| SHUF-K4 CheXpert AUC |  |  |  |  |  |  |  |  |
| SHUF-TW-clinical CheXpert AUC |  |  |  |  |  |  |  |  |

### 多 seed 总表

| Run | Seeds | CheXpert AUC mean±std | External AUC mean±std | Hard shuffle mean±std | CF acc mean±std | A/B-swap mean±std | Selected? |
|---|---:|---|---|---|---|---|---|
| Base-Qwen3VL |  |  |  |  |  |  |  |
| SHUF-3k |  |  |  |  |  |  |  |
| SAMEQ-SHUF |  |  |  |  |  |  |  |
| SHUF-K4 |  |  |  |  |  |  |  |
| SHUF-TW-clinical |  |  |  |  |  |  |  |

---

## 4.3 Paired bootstrap / CI

每个候选相对 baseline 都要做 paired CI。

| Comparison | Metric | Delta mean | 95% CI low | 95% CI high | Stable? |
|---|---|---:|---:|---:|---|
| SAMEQ - SHUF-3k | CheXpert AUC |  |  |  |  |
| SAMEQ - SHUF-3k | Hard shuffle |  |  |  |  |
| SHUF-K4 - SHUF-3k | CheXpert AUC |  |  |  |  |
| SHUF-K4 - SHUF-3k | External AUC |  |  |  |  |
| SHUF-TW - SHUF-3k | CheXpert AUC |  |  |  |  |

### 决策规则

| Result | Interpretation |
|---|---|
| CI entirely > 0 | Candidate reliably improves |
| CI crosses 0 but hard shuffle strong | Mechanism evidence, not AUC superiority |
| CI crosses 0 for all primary metrics | Candidate downgraded |
| External improves but CheXpert not | external/domain route |
| Hard shuffle improves but AUC drops | grounding-only diagnostic |

---

# 5. Phase 2：Clinical Visual Curriculum Pretraining

## 5.1 Curriculum 总思想

将 instruction 按视觉依赖难度排序：

| Stage | Task type | Goal | Difficulty |
|---|---|---|---|
| Stage 1 | Basic clinical QA | 学 finding / state / 医学词汇 | easy |
| Stage 2 | Counterfactual statement QA | 学真假医学陈述 | medium |
| Stage 3 | SAMEQ | 同一个问题，不同图像答案不同 | hard |
| Stage 4 | SHUF-K / multi-negative | 一个正图多个难负图 | very hard |
| Stage 5 | CCSH / CEQ readout | 部署式临床一致性判断 | readout |

---

## 5.2 Curriculum version list

### CVCP-v1：Direct SAMEQ

```text
只用 SAMEQ 训练
```

| Run | Data | Steps | Purpose |
|---|---|---:|---|
| CVCP-v1-SAMEQ-3k | SAMEQ | 5k | same-question baseline |
| CVCP-v1-SAMEQ-10k | SAMEQ | 8k | scale |
| CVCP-v1-SAMEQ-full | SAMEQ | 12k | upper bound |

---

### CVCP-v2：Direct SHUF-K

```text
只用 multi-negative SHUF 训练
```

| Run | K | Steps | Purpose |
|---|---:|---:|---|
| CVCP-v2-SHUF-K2 | 2 | 5k | K ablation |
| CVCP-v2-SHUF-K4 | 4 | 5k | current strong baseline |
| CVCP-v2-SHUF-K8 | 8 | 5k/8k | expensive upper |

---

### CVCP-v3：Progressive curriculum

```text
Basic QA -> CF -> SAMEQ -> SHUF-K
```

| Stage | Step range | Basic QA | CF | SAMEQ | SHUF-K |
|---|---|---:|---:|---:|---:|
| 1 | 0-20% | 60 | 30 | 10 | 0 |
| 2 | 20-50% | 30 | 40 | 20 | 10 |
| 3 | 50-80% | 10 | 25 | 40 | 25 |
| 4 | 80-100% | 5 | 15 | 40 | 40 |

| Run | Data scale | Steps | Purpose |
|---|---:|---:|---|
| CVCP-v3-prog-3k | 3k | 8k | debug |
| CVCP-v3-prog-10k | 10k | 12k | main |
| CVCP-v3-prog-full | full | 12k-20k | upper |

---

### CVCP-v4：Progressive + replay

为防止 stage 遗忘，每个后续阶段保留一部分早期 QA。

| Stage | Main tasks | Replay |
|---|---|---:|
| 1 | basic QA + CF | 0 |
| 2 | CF + SAMEQ | 20% stage1 |
| 3 | SAMEQ + SHUF-K | 15% previous |
| 4 | SHUF-K + SAMEQ | 10% previous |

| Run | Data scale | Steps | Purpose |
|---|---:|---:|---|
| CVCP-v4-replay-10k | 10k | 12k | primary curriculum candidate |
| CVCP-v4-replay-full | full | 16k | upper |

---

### CVCP-v5：Case-driven curriculum

每 N steps 根据 dev failures 调整采样。

| Failure type | Sampling increase |
|---|---|
| laterality errors | more left/right SAMEQ |
| state flip errors | more present/absent CF |
| high false hard negative | filter/rebuild negatives |
| uncertain errors | more uncertainty QA |
| rare finding errors | rare-field oversampling |
| external label errors | external-aware sampling |

| Run | Scheduler | Data | Steps |
|---|---|---:|---:|
| CVCP-v5-CDCS-field | field failure | 10k | 12k |
| CVCP-v5-CDCS-hardneg | hard negative failure | 10k | 12k |
| CVCP-v5-CDCS-full | field + hardneg + external | full | 16k |

---

## 5.3 Curriculum result table

| Run | Data scale | Steps | CheXpert AUC | External AUC | Hard shuffle | CF acc | A/B-swap | Leakage | Cost | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CVCP-v1-SAMEQ-3k |  |  |  |  |  |  |  |  |  |  |
| CVCP-v2-SHUF-K4 |  |  |  |  |  |  |  |  |  |  |
| CVCP-v3-prog-10k |  |  |  |  |  |  |  |  |  |  |
| CVCP-v4-replay-10k |  |  |  |  |  |  |  |  |  |  |
| CVCP-v5-CDCS-full |  |  |  |  |  |  |  |  |  |  |

---

## 5.4 Curriculum 决策规则

| Result | Decision |
|---|---|
| Progressive beats direct SAMEQ/K4 in AUC and hard shuffle | curriculum becomes main story |
| Progressive improves external only | write as external robustness route |
| Direct SAMEQ/K4 still best | curriculum becomes ablation |
| Case-driven improves over fixed progressive | CDCS becomes module contribution |
| Curriculum high leakage | fix data, do not claim failure |

---

# 6. Phase 3：模块组合实验

## 6.1 模块说明总表

| Module | Meaning | Role in story | Priority |
|---|---|---|---|
| CCSH | Clinical Consistency Scoring Head | deployable readout | highest |
| CEQ | Clinical Evidence Query | finding-specific visual evidence | high |
| HNMB | Hard Negative Memory Bank | dynamic hard negatives | high |
| AUCH | Answerability-Uncertainty Calibration Head | clinical semantics/calibration | medium |
| DRA | Domain-Robust Adapter | external/domain robustness | conditional |
| CDCS | Case-Driven Curriculum Scheduler | failure-driven curriculum | high if curriculum route |

---

## 6.2 CCSH experiments

### Why CCSH?

CCSH 把训练时的图像-陈述一致性任务变成部署时可用的轻量模块。

### Required ablations

| Run | Backbone | Training data | CCSH? | Purpose |
|---|---|---|---|---|
| Base+CCSH | raw Qwen3-VL | none | yes | head-only baseline |
| SHUF-3k+CCSH | SHUF-3k | D7 | yes | does SHUF improve CCSH |
| SAMEQ+CCSH | SAMEQ | SAMEQ | yes | main candidate |
| SHUF-K4+CCSH | SHUF-K4 | K4 | yes | main candidate |
| CVCP-prog+CCSH | curriculum | CVCP | yes | curriculum candidate |
| CCSH-random-statement | any | random statements | yes | negative control |

### CCSH result table

| Run | Vision pretrain | Binary AUC | Binary AUPRC | Consistency F1 | CheXpert AUC | External AUC | Hard shuffle | ECE | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Base+CCSH | none |  |  |  |  |  |  |  |  |
| SHUF+CCSH | SHUF |  |  |  |  |  |  |  |  |
| SAMEQ+CCSH | SAMEQ |  |  |  |  |  |  |  |  |
| SHUF-K4+CCSH | K4 |  |  |  |  |  |  |  |  |
| CVCP+CCSH | curriculum |  |  |  |  |  |  |  |  |

### CCSH decision

| Result | Interpretation |
|---|---|
| SAMEQ+CCSH > Base+CCSH | VLM training improves deployable consistency |
| Base+CCSH ~= SAMEQ+CCSH | CCSH head mostly drives gain, weakens VLM-training claim |
| CCSH improves consistency but not LP | use as auxiliary deployable module |
| CCSH improves LP + consistency | main module candidate |

---

## 6.3 CEQ experiments

### CEQ idea

每个 finding 一个 clinical evidence query，cross-attend 到 image patch tokens。

### Experiments

| Run | Base data | CEQ? | CCSH? | Purpose |
|---|---|---|---|---|
| SAMEQ+CEQ | SAMEQ | yes | no | evidence query alone |
| SAMEQ+CEQ+CCSH | SAMEQ | yes | yes | explainable consistency |
| SHUF-K4+CEQ | K4 | yes | no | multi-negative evidence |
| SHUF-K4+CEQ+CCSH | K4 | yes | yes | main TMI architecture |
| CVCP+CEQ+CCSH | curriculum | yes | yes | full candidate |

### CEQ result table

| Run | CheXpert AUC | External AUC | Hard shuffle | CF acc | Consistency AUC | Attention quality | Decision |
|---|---:|---:|---:|---:|---:|---|---|
| SAMEQ+CEQ |  |  |  |  |  |  |  |
| SAMEQ+CEQ+CCSH |  |  |  |  |  |  |  |
| SHUF-K4+CEQ+CCSH |  |  |  |  |  |  |  |
| CVCP+CEQ+CCSH |  |  |  |  |  |  |  |

### Attention / explanation table

| Run | Finding | Sample | Expected region | CEQ attention correct? | Notes |
|---|---|---|---|---|---|
| CEQ+CCSH | Pleural Effusion |  | costophrenic angle |  |  |
| CEQ+CCSH | Cardiomegaly |  | cardiac silhouette |  |  |
| CEQ+CCSH | Pneumothorax |  | lung apex/pleural line |  |  |
| CEQ+CCSH | Edema |  | bilateral lungs |  |  |

---

## 6.4 HNMB experiments

### HNMB idea

动态挖模型真正容易混淆的 hard negatives。

### Experiments

| Run | Negative mining | Update | CCSH? | Purpose |
|---|---|---|---|---|
| HNMB-static | offline embedding mining | once | no | cheap mined negatives |
| HNMB-online | periodic mining | every epoch | no | stronger |
| SAMEQ+HNMB | SAMEQ + memory negatives | periodic | no | main hard negative |
| HNMB+CCSH | memory + consistency head | periodic | yes | module candidate |
| CEQ+HNMB+CCSH | evidence + memory + consistency | periodic | yes | full module |

### HNMB table

| Run | CheXpert AUC | External AUC | Hard shuffle | CF acc | False negative rate | Mining cost | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| HNMB-static |  |  |  |  |  |  |  |
| HNMB-online |  |  |  |  |  |  |  |
| HNMB+CCSH |  |  |  |  |  |  |  |
| CEQ+HNMB+CCSH |  |  |  |  |  |  |  |

---

## 6.5 AUCH experiments

### AUCH idea

部署时显式预测 answerability 和 uncertainty，增强临床校准。

### Experiments

| Run | Backbone | AUCH? | CCSH? | Purpose |
|---|---|---|---|---|
| SAMEQ+AUCH | SAMEQ | yes | no | semantics |
| SAMEQ+AUCH+CCSH | SAMEQ | yes | yes | consistency + uncertainty |
| CEQ+AUCH+CCSH | CEQ/SAMEQ | yes | yes | TMI clinical module |

### AUCH table

| Run | Macro-AUC | AUPRC | ECE | Brier | Answerability AUC | Uncertainty F1 | High-null calibration | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| SAMEQ+AUCH |  |  |  |  |  |  |  |  |
| SAMEQ+AUCH+CCSH |  |  |  |  |  |  |  |  |
| CEQ+AUCH+CCSH |  |  |  |  |  |  |  |  |

---

## 6.6 DRA experiments

DRA 只在选定 external 后做。

| Run | External | Domain loss | CheXpert AUC | External AUC | Domain MMD | ECE external | Decision |
|---|---|---|---:|---:|---:|---:|---|
| SAMEQ+DRA |  | CORAL |  |  |  |  |  |
| SAMEQ+DANN |  | adversarial |  |  |  |  |  |
| CEQ+CCSH+DRA |  | CORAL/DANN |  |  |  |  |  |

---

# 7. Phase 4：SHUF++ and CF-compatible repairs

## 7.1 SAMEQ-CF-compatible

Problem:

```text
SAMEQ grounding is strong, but CF/A-B option-pairwise diagnostics are not directly applicable.
```

Fix:

```text
Add a controlled subset of explicit A/B option rows to SAMEQ data.
```

### Experiments

| Run | SAMEQ % | CF-compatible % | Purpose |
|---|---:|---:|---|
| SAMEQ-CF-10 | 90 | 10 | minimal diagnostic |
| SAMEQ-CF-20 | 80 | 20 | balanced |
| SAMEQ-CF-30 | 70 | 30 | stronger CF |

| Run | CheXpert AUC | External AUC | Hard shuffle | CF acc | A/B-swap | Decision |
|---|---:|---:|---:|---:|---:|---|
| SAMEQ-CF-10 |  |  |  |  |  |  |
| SAMEQ-CF-20 |  |  |  |  |  |  |
| SAMEQ-CF-30 |  |  |  |  |  |  |

---

## 7.2 SHUF-K4-CF-compatible

Problem:

```text
SHUF-K4 has strong hard shuffle and external signal but CF/A-B below 0.85.
```

Fix:

- Add explicit counterfactual rows;
- Rebalance A/B;
- Add TW-visual only on option/status/location tokens.

| Run | K | CF-compatible % | TW | Purpose |
|---|---:|---:|---|---|
| K4-CF-20 | 4 | 20 | none | repair CF |
| K4-CF-20-TW | 4 | 20 | visual | repair CF + token focus |
| K4-CF-30-TW | 4 | 30 | visual | stronger repair |

| Run | CheXpert AUC | External AUC | Hard shuffle | CF acc | A/B-swap | Decision |
|---|---:|---:|---:|---:|---:|---|
| K4-CF-20 |  |  |  |  |  |  |
| K4-CF-20-TW |  |  |  |  |  |  |
| K4-CF-30-TW |  |  |  |  |  |  |

---

## 7.3 Dual-margin training

Define:

```text
I+ = correct image
I- = hard negative image
A+ = correct answer
A- = counterfactual answer
Q = same question
```

Loss:

```text
L = CE(I+, Q, A+)
  + λ_img * max(0, m + NLL(I+,Q,A+) - NLL(I-,Q,A+))
  + λ_ans * max(0, m + NLL(I+,Q,A+) - NLL(I+,Q,A-))
```

| Run | λ_img | λ_ans | Purpose |
|---|---:|---:|---|
| Dual-light | 0.1 | 0.1 | safe |
| Dual-img-heavy | 0.3 | 0.1 | more visual |
| Dual-answer-heavy | 0.1 | 0.3 | more CF |
| Dual-balanced | 0.2 | 0.2 | main |

| Run | CheXpert AUC | External AUC | Hard shuffle | CF acc | Loss stability | Decision |
|---|---:|---:|---:|---:|---:|---|
| Dual-light |  |  |  |  |  |  |
| Dual-img-heavy |  |  |  |  |  |  |
| Dual-answer-heavy |  |  |  |  |  |  |
| Dual-balanced |  |  |  |  |  |  |

---

# 8. Phase 5：Token weighting

## 8.1 Weighting versions

| Weighting | Logic | Default? |
|---|---|---|
| TW-light | token role | safe baseline |
| TW-visual | visual dependency | recommended |
| TW-clinical-balanced | rarity + clinical role | ablation |
| TW-adaptive | failure-driven weights | advanced |

## 8.2 Recommended combinations

| Run | Base | Weighting | Purpose |
|---|---|---|---|
| SAMEQ-TW-visual | SAMEQ | TW-visual | token focus for SAMEQ |
| K4-TW-visual | SHUF-K4 | TW-visual | repair CF/A-B |
| CVCP-TW-visual | curriculum | TW-visual | main curriculum |
| CEQ+CCSH+TW | module | TW-visual | module candidate |

## 8.3 Table

| Run | Weighting | CheXpert AUC | External AUC | Hard shuffle | CF acc | A/B-swap | Calibration | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| SAMEQ-TW-visual | visual |  |  |  |  |  |  |  |
| K4-TW-visual | visual |  |  |  |  |  |  |  |
| CVCP-TW-visual | visual |  |  |  |  |  |  |  |
| CEQ+CCSH+TW | visual |  |  |  |  |  |  |  |

---

# 9. Phase 6：VLM teacher model comparison

## 9.1 Model families

| Model | Type | Role |
|---|---|---|
| Qwen3-VL-2B | VLM | current main |
| InternVL | VLM | ViTP-style strong comparator |
| LLaVA / Llama-based VLM | VLM | different LLM family |
| Medical VLM if available | VLM | domain-specific comparator |
| Qwen3.5-2B text-only | text-only | scaffold control |
| Qwen-Coder | text-only coder | template bias control |
| Raw vision tower | vision-only | base representation |

## 9.2 Fair protocol

All VLM teacher comparisons must use:

```text
same instruction data
same curriculum
same frozen LLM policy
same trainable vision/connector policy
same LP/external evaluation
same seeds if possible
```

## 9.3 Model comparison runs

| Run | Model | Data | Train policy | Purpose |
|---|---|---|---|---|
| Qwen3VL-CVCP | Qwen3-VL | CVCP | freeze LLM, train vision+connector | main |
| InternVL-CVCP | InternVL | CVCP | freeze LLM, train vision+connector | VLM comparison |
| LLaVA-CVCP | LLaVA/Llama | CVCP | freeze LLM, train vision+connector | VLM comparison |
| Qwen3.5-scaffold | text-only | CVCP | train ViT+new projector | no VLM control |
| Qwen-Coder-scaffold | text-only coder | CVCP | train ViT+new projector | template bias |
| Raw-Qwen3VL | Qwen3-VL vision | none | no train | base |

### Table

| Run | Model | CheXpert AUC | External AUC | Hard shuffle | CCSH AUC | Cost | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| Qwen3VL-CVCP |  |  |  |  |  |  |  |
| InternVL-CVCP |  |  |  |  |  |  |  |
| LLaVA-CVCP |  |  |  |  |  |  |  |
| Qwen3.5-scaffold |  |  |  |  |  |  |  |
| Qwen-Coder-scaffold |  |  |  |  |  |  |  |

## 9.4 Decision

| Result | Claim |
|---|---|
| all VLMs > text scaffold | VLM-coupled teacher matters |
| Qwen3VL only works | model-specific, claim narrower |
| text scaffold comparable | VLM coupling less important; data/curriculum drives gains |
| larger VLM worse | supports lightweight teacher pressure hypothesis |
| medical VLM best | domain teacher matters |

---

# 10. Phase 7：External evaluation

## 10.1 External protocol

One main external dataset only.

| Step | Required |
|---|---|
| label mapping audit | yes |
| per-label table | yes |
| macro-AUC | yes |
| macro-AUPRC | yes |
| ECE/Brier | yes |
| case study | yes |
| external failure taxonomy | yes |
| no NIH as main | yes |

## 10.2 External result table

| Run | External dataset | Macro-AUC | Macro-AUPRC | ECE | Brier | Per-label wins | Notes |
|---|---|---:|---:|---:|---:|---|---|
| Raw-Qwen3VL |  |  |  |  |  |  |  |
| SHUF-3k |  |  |  |  |  |  |  |
| SAMEQ |  |  |  |  |  |  |  |
| SHUF-K4 |  |  |  |  |  |  |  |
| CVCP |  |  |  |  |  |  |  |
| CVCP+CCSH |  |  |  |  |  |  |  |
| CEQ+CCSH |  |  |  |  |  |  |  |

---

# 11. Phase 8：Locked final comparison

## 11.1 Family finalists

Each family can contribute only one finalist.

| Family | Candidate pool | Selected finalist | Selection rule |
|---|---|---|---|
| Direct SHUF | SHUF-3k, SHUF-TW |  | best stable direct |
| SAMEQ | SAMEQ, SAMEQ-CF |  | best image-specific |
| Multi-negative | K2/K4/K8/HNMB |  | best hard-neg |
| Curriculum | CVCP-v3/v4/v5 |  | best curriculum |
| Module | CCSH/CEQ+CCSH/AUCH |  | best deployable module |
| Model comparison | Qwen/InternVL/LLaVA |  | best teacher protocol |

## 11.2 Locked metrics

| Metric | Role |
|---|---|
| CheXpert macro-AUC | primary deployable representation |
| Main external macro-AUC | primary external |
| Hard shuffle delta | primary image-specific grounding |
| CCSH binary AUC/AUPRC | primary consistency if module route |
| CF acc | secondary counterfactual |
| A/B-swap | secondary option robustness |
| AUPRC | clinical rare-label |
| ECE/Brier | calibration |
| leakage % | safety gate |
| cost | deployment/training feasibility |

## 11.3 Locked comparison table

| Family | Finalist | Seeds | CheXpert AUC mean±std | External AUC mean±std | Hard shuffle mean±std | CCSH AUC | CF acc | ECE | Cost | Final role |
|---|---|---:|---|---|---|---:|---:|---:|---:|---|
| Base | Raw Qwen3VL |  |  |  |  |  |  |  |  | baseline |
| Direct SHUF |  |  |  |  |  |  |  |  |  |  |
| SAMEQ |  |  |  |  |  |  |  |  |  |  |
| Multi-negative |  |  |  |  |  |  |  |  |  |  |
| Curriculum |  |  |  |  |  |  |  |  |  |  |
| Module |  |  |  |  |  |  |  |  |  |  |
| Teacher model |  |  |  |  |  |  |  |  |  |  |

---

# 12. Case study and visualization

## 12.1 Required casebooks

| Casebook | Purpose |
|---|---|
| SAMEQ successes | why same-question works |
| SAMEQ failures | where image-specific still fails |
| SHUF-K4 failures | why CF/A-B weak |
| CCSH successes | clinical consistency examples |
| CCSH failures | consistency head limitations |
| CEQ attention maps | evidence query interpretability |
| External failures | label/domain mismatch |
| False hard negatives | data quality |

## 12.2 Casebook template

| Case ID | Dataset | Image | Question/statement | Correct answer | Model prediction | Failure type | Manual note |
|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |

## 12.3 Visualization

| Visualization | Purpose |
|---|---|
| CEQ attention map | finding-specific evidence |
| CCSH support/contradict examples | deployable consistency |
| SAMEQ paired images | same question, different answer |
| SHUF-K4 positive + negatives | multi-negative explanation |
| UMAP external/domain | domain shift |
| calibration curves | clinical reliability |

---

# 13. Recommended experiment queue

## 13.1 First 12 experiments

| Priority | Run | Why |
|---:|---|---|
| 1 | Base+CCSH | test if CCSH alone explains gains |
| 2 | SAMEQ+CCSH | strongest grounding + strongest module |
| 3 | SHUF-K4+CCSH | multi-negative + consistency |
| 4 | CEQ+CCSH | interpretable TMI module |
| 5 | SAMEQ-CF-20 | make SAMEQ CF-compatible |
| 6 | K4-CF-20-TW | fix K4 CF/A-B weakness |
| 7 | CVCP-v3-prog-10k | progressive curriculum |
| 8 | CVCP-v4-replay-10k | curriculum with replay |
| 9 | HNMB-static+CCSH | mined negatives + consistency |
| 10 | HNMB-online+CCSH | dynamic hard negative |
| 11 | External dataset manifest + base eval | choose external |
| 12 | Qwen3VL vs InternVL smoke | prepare model comparison |

---

## 13.2 Second wave

| Priority | Run | Why |
|---:|---|---|
| 1 | CVCP-v5-CDCS | case-driven curriculum |
| 2 | CEQ+HNMB+CCSH | full TMI module stack |
| 3 | SAMEQ-K4 hybrid | combine strongest mechanisms |
| 4 | Dual-balanced | image + answer margin |
| 5 | CVCP-TW-visual | token weighting in curriculum |
| 6 | DRA on selected external | if domain gap persists |
| 7 | AUCH+CCSH | calibration and uncertainty |
| 8 | InternVL-CVCP | model comparison |
| 9 | LLaVA-CVCP | model comparison |
| 10 | Qwen3.5 text scaffold | non-VLM control |

---

## 13.3 Final wave

| Run | Purpose |
|---|---|
| Best candidate seed5 | final stability |
| Best candidate external full | final external |
| Best candidate AUPRC/ECE | clinical reliability |
| Best candidate casebook | qualitative |
| Best candidate cost | training/deployment |
| Best candidate vs ViTP-style recipe | novelty comparison if possible |

---

# 14. Detailed experimental tables

## 14.1 Candidate tracking table

| Run ID | Family | Data | Module | Model | Seeds | Status | CheXpert AUC | External AUC | Hard shuffle | CCSH AUC | CF acc | Notes |
|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| Base+CCSH | Module | statements | CCSH | Qwen3VL |  |  |  |  |  |  |  |  |
| SAMEQ+CCSH | Module | SAMEQ | CCSH | Qwen3VL |  |  |  |  |  |  |  |  |
| SHUF-K4+CCSH | Module | K4 | CCSH | Qwen3VL |  |  |  |  |  |  |  |  |
| CEQ+CCSH | Module | SAMEQ/K4 | CEQ+CCSH | Qwen3VL |  |  |  |  |  |  |  |  |
| CVCP-v3 | Curriculum | mixed | none | Qwen3VL |  |  |  |  |  |  |  |  |
| CVCP-v4 | Curriculum | mixed+replay | none | Qwen3VL |  |  |  |  |  |  |  |  |
| HNMB+CCSH | Hard negative | mined | HNMB+CCSH | Qwen3VL |  |  |  |  |  |  |  |  |
| InternVL-CVCP | Model | mixed | none | InternVL |  |  |  |  |  |  |  |  |

---

## 14.2 Data audit table

| Dataset | N images | N instructions | QA/img | Leakage % | A/B A% | False negative % | Manual pass % | Accepted? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| SAMEQ-CF-20 |  |  |  |  |  |  |  |  |
| K4-CF-20-TW |  |  |  |  |  |  |  |  |
| CVCP-v3-10k |  |  |  |  |  |  |  |  |
| CVCP-v4-10k |  |  |  |  |  |  |  |  |
| HNMB-static |  |  |  |  |  |  |  |  |

---

## 14.3 Training cost table

| Run | Model | Trainable params | Frozen params | GPU | Hours | Peak VRAM | Steps | Cost acceptable? |
|---|---|---:|---:|---|---:|---:|---:|---|
| SAMEQ+CCSH |  |  |  |  |  |  |  |  |
| SHUF-K4+CCSH |  |  |  |  |  |  |  |  |
| CEQ+CCSH |  |  |  |  |  |  |  |  |
| CVCP-v4 |  |  |  |  |  |  |  |  |
| InternVL-CVCP |  |  |  |  |  |  |  |  |

---

## 14.4 Failure mode table

| Run | Failure type | Count | Example | Next action |
|---|---|---:|---|---|
| SAMEQ+CCSH | false hard negative |  |  | verifier/filter |
| SHUF-K4+CCSH | A/B option bias |  |  | rebalance/TW |
| CEQ+CCSH | attention wrong region |  |  | query diversity |
| CVCP-v4 | stage forgetting |  |  | replay schedule |
| External | label mismatch |  |  | label mapping/filter |

---

# 15. Decision trees

## 15.1 If SAMEQ+CCSH wins

Main story:

```text
Same-question clinical exams are the cleanest VLM teaching signal. CCSH makes this deployable.
```

Next:

- seed5;
- external full;
- CEQ visualization optional.

## 15.2 If SHUF-K4+CCSH wins

Main story:

```text
Multi-negative clinical image exams provide stronger visual discrimination. CCSH reads out consistency.
```

Next:

- K ablation;
- false negative audit;
- K4 vs K8.

## 15.3 If CEQ+CCSH wins

Main story:

```text
Finding-specific evidence queries plus consistency scoring create an interpretable CXR visual encoder.
```

Next:

- attention maps;
- per-field analysis;
- TMI main architecture.

## 15.4 If curriculum wins

Main story:

```text
VLM teachers should teach CXR vision through a clinical curriculum, not a single QA format.
```

Next:

- curriculum schedule ablation;
- CDCS;
- model comparison.

## 15.5 If no module beats SAMEQ alone

Main story:

```text
Data construction is the core contribution; modules are secondary.
```

Next:

- strengthen data/teacher comparison;
- avoid overclaiming modules.

## 15.6 If external still fails

Interpretation:

```text
Model improves image-specific grounding but external classification is limited by label mapping/domain shift.
```

Next:

- present external as limitation;
- use external consistency/qualitative eval;
- add DRA only if needed.

---

# 16. Codex task checklist

## 16.1 Data generation

```text
scripts/generate_cvcp_curriculum.py
scripts/generate_sameq_cf_compatible.py
scripts/generate_shuf_k_cf_compatible.py
scripts/generate_ccsh_statements.py
scripts/generate_ceq_targets.py
scripts/audit_instruction_leakage_v3.py
scripts/audit_false_hard_negatives.py
```

## 16.2 Training

```text
scripts/train_qwen3vl_cvcp.py
scripts/train_qwen3vl_sameq_ccsh.py
scripts/train_qwen3vl_shufk_ccsh.py
scripts/train_ceq_ccsh.py
scripts/train_hnmb_ccsh.py
scripts/train_vlm_teacher_comparison.py
```

## 16.3 Evaluation

```text
scripts/eval_locked_final_suite.py
scripts/eval_external_dataset.py
scripts/eval_ccsh_consistency.py
scripts/eval_ceq_attention.py
scripts/eval_ab_swap.py
scripts/eval_hard_shuffle.py
scripts/eval_calibration_auprc.py
scripts/bootstrap_locked_comparison.py
```

## 16.4 Reporting

```text
outputs/final_tables/cvcp_training_results.md
outputs/final_tables/module_combo_results.md
outputs/final_tables/model_comparison_results.md
outputs/final_tables/external_eval_results.md
outputs/final_tables/locked_final_comparison.md
outputs/final_tables/casebook.md
outputs/final_tables/cost_table.md
```

---

# 17. Final recommendation

## 17.1 Highest priority

Run these first:

```text
Base+CCSH
SAMEQ+CCSH
SHUF-K4+CCSH
CEQ+CCSH
SAMEQ-CF-20
K4-CF-20-TW
CVCP-v4-replay-10k
```

These answer:

1. Is CCSH useful beyond head-only?
2. Does SAMEQ become a deployable method with CCSH?
3. Can SHUF-K4 be repaired?
4. Does CEQ make the method TMI-level and interpretable?
5. Can curriculum beat direct grounding?

## 17.2 Most likely final stories

### Story A

```text
Clinical Visual Curriculum Pretraining with CCSH
```

### Story B

```text
Same-Question Clinical Exam + Consistency Head
```

### Story C

```text
Clinical Evidence Query + Consistency Scoring
```

### Story D

```text
Model-Agnostic VLM Teacher Curriculum
```

## 17.3 Current best strategic bet

The strongest next bet is:

```text
SAMEQ + CCSH
```

because:

- SAMEQ is stable image-specific grounding;
- CCSH is strongest deployable module;
- the story is simple but not low-level;
- it preserves the VLM teacher idea;
- it avoids overclaiming SHUF-TW-clinical;
- it can be extended with CEQ for TMI interpretability.

Final candidate to beat:

```text
SAMEQ+CCSH
vs
SHUF-K4+CCSH
vs
CVCP-v4-replay
vs
CEQ+CCSH
```

---

# 18. Final Execution Closure (2026-07-06)

Closure marker: `CVCP_CCSH_FINAL_EXECUTION_CLOSURE_20260706`

This section records the artifact-backed completion of the full CVCP/CCSH experiment plan on the two local RTX 3090 GPUs. All required and optional runnable experiment families in this document were executed under the formal local protocol created for this plan. External datasets or model-family comparisons that were not runnable under the local repository/trainer constraints are recorded as explicit bounded evidence rather than silent substitutions.

## 18.1 Execution Completeness

| Area | Final status | Evidence |
| --- | --- | --- |
| CVCP/CF/token-weighting/dual-margin training rows | `27/27 complete` | `outputs/final_tables/cvcp_training_results.{csv,md}` |
| LP, NIH appendix, visual dependence, counterfactual, A/B-swap, paraphrase postprocess | `27/27 complete` | `outputs/final_tables/cvcp_ccsh_postprocess_status.{csv,md}` |
| Module-combo experiments | `18/18 complete` | `outputs/final_tables/module_combo_results.{csv,md}` |
| Target-plan scripts | `21/21 exact target entry points exist` | `docs/cvcp_ccsh_readiness_audit.md` |
| VLM/model comparison availability | audited with supported/bounded rows | `outputs/final_tables/model_comparison_results.{csv,md,json}` |
| External evaluation decision | VinDr image-only and PadChest missing are bounded; NIH remains appendix/stress-test only | `outputs/final_tables/external_eval_results.{csv,md}` |
| Casebook/reporting/cost/locked comparison | generated | `outputs/final_tables/casebook.md`, `cost_table.md`, `locked_final_comparison.md` |
| Queue/process/GPU closure | both training lanes, postprocess lanes, and module lanes reached `QUEUE_DONE`; GPU0/GPU1 returned to `0 MiB` | `outputs/logs/cvcp_ccsh/*`, `outputs/logs/cvcp_ccsh_postprocess/*`, `outputs/logs/cvcp_ccsh_module_combos/*`, final audit |

## 18.2 Locked Final Comparison

| Family | Finalist | CheXpert AUC | NIH appendix AUC | Hard-shuffle delta | CF acc | Final role |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| CVCP-v1 | `cvcp_v1_sameq_full` | 0.748633 | 0.576656 | 0.641917 |  | strongest CheXpert and hard-shuffle signal |
| SAMEQ-CF | `sameq_cf_30` | 0.744668 | 0.578584 | 0.481282 |  | strong SAMEQ repair/counterfactual-compatible row |
| TokenWeighting | `k4_tw_visual` | 0.736355 | 0.569783 | 0.278978 | 0.808390 | best token-weighting row |
| K4-CF | `k4_cf_20_tw` | 0.726790 | 0.589986 | 0.417582 | 0.807256 | strongest NIH appendix among K4-CF rows |
| CVCP-v4 | `cvcp_v4_replay_10k` | 0.723687 | 0.587297 | 0.151935 | 0.849802 | best replay finalist and module backbone |
| CVCP-v5 | `cvcp_v5_cdcs_field` | 0.737263 | 0.589017 | -0.276647 | 0.804348 | CDCS evidence row; strong NIH appendix, weak hard-shuffle |
| DualMargin | `dual_img_heavy` | 0.715921 | 0.574559 | 0.328933 | 0.802721 | dual-margin comparison row |

Full family table: `outputs/final_tables/locked_final_comparison.md`.

## 18.3 Module-Combo Results

| Module combo | Backbone | Best binary AUC | Best state accuracy | Interpretation |
| --- | --- | ---: | ---: | --- |
| `cvcp_replay_ccsh` | `cvcp_v4_replay_10k` | 0.893317 | 0.766000 | strongest CCSH readout on replay backbone |
| `ceq_hnmb_ccsh` | `cvcp_v4_replay_10k` | 0.893317 | 0.766000 | full CEQ+HNMB+CCSH stack; no gain beyond CCSH readout but complete |
| `ceq_auch_ccsh` | `cvcp_v4_replay_10k` | 0.893317 | 0.766000 | CEQ+AUCH+CCSH stack complete |
| `shufk_ccsh` | `cvcp_v2_shuf_k4` | 0.883185 | 0.764953 | best SHUF-K4 consistency module route |
| `cvcp_prog_ccsh` | `cvcp_v3_prog_10k` | 0.883392 | 0.759000 | progressive curriculum module route |
| `cvcp_cdcs_ccsh` | `cvcp_v5_cdcs_full` | 0.876974 | 0.725000 | CDCS+CCSH module stack complete |

The deployable module conclusion is conservative: CCSH is the strongest readout family, while CEQ/HNMB/AUCH/CDCS are completed stack variants and interpretation aids rather than automatic replacements for the best CCSH backbone.

## 18.4 External and Model Boundaries

| Item | Final decision |
| --- | --- |
| NIH | Completed for all 27 rows as `NIH-appendix-1k`; kept as appendix/stress-test only, not promoted to main external. |
| VinDr-CXR/VinBigData | Local image package exists, but no label/bbox CSV was available in the audited package; bounded as image-only, not valid for main AUC/ECE/AUPRC. |
| PadChest | Missing locally; bounded explicitly. |
| Qwen3-VL | Qwen3-VL local family is the supported formal trainer route. |
| InternVL/LLaVA/Llama/Qwen3.5 scaffolds | Model directories were audited; architecture-specific adapters are not implemented in the current formal trainer, so these remain compatibility/boundary rows instead of claimed trained comparisons. |

## 18.5 Final Recommendation

The post-execution recommendation changes from a speculative `SAMEQ+CCSH` bet to a two-part result:

1. **Primary empirical finalist:** `CVCP-v1 / cvcp_v1_sameq_full`, because it achieved the strongest CheXpert AUC and largest hard-shuffle delta in the formal 27-row queue.
2. **Deployable module finalist:** `CVCP-v4-replay-10k + CCSH`, because replay-backed CCSH and its CEQ/HNMB/AUCH stack variants reached the strongest module-combo AUC/state-accuracy profile.

For paper framing, avoid claiming a single universal winner. The supported story is: SAMEQ-style visual curricula give the strongest image-specific training signal, while CCSH on a replay/CVCP backbone gives the strongest deployable consistency readout. NIH remains appendix-only, and the unavailable VinDr/PadChest boundaries must be stated in the limitations.

## 18.6 Final Artifact Index

| Artifact | Path |
| --- | --- |
| Training results | `outputs/final_tables/cvcp_training_results.md` |
| Postprocess status | `outputs/final_tables/cvcp_ccsh_postprocess_status.md` |
| Module combo results | `outputs/final_tables/module_combo_results.md` |
| Model comparison audit | `outputs/final_tables/model_comparison_results.md` |
| External evaluation boundary/results | `outputs/final_tables/external_eval_results.md` |
| Locked comparison | `outputs/final_tables/locked_final_comparison.md` |
| Cost table | `outputs/final_tables/cost_table.md` |
| Casebook | `outputs/final_tables/casebook.md` |
| Requirement ledger | `docs/cvcp_ccsh_requirement_ledger.md` |
| Readiness audit | `docs/cvcp_ccsh_readiness_audit.md` |
| Final completion audit | `outputs/final_tables/cvcp_ccsh_completion_audit.md` |
