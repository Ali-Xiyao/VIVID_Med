# VIVID-Med 下一阶段完整实验计划 v4  
## 主线：SAMEQ-CVCP — Same-Question Clinical Visual Curriculum Pretraining

> 版本：2026-07-06  
> 目标：把后续实验收敛成一个清晰的主方法，而不是“很多 tricks 随机试”。  
> 核心主线：**SAMEQ-CVCP：同一个临床问题，配给不同胸片，答案由图像决定。**  
> 如果桥接实验成功，最终方法升级为：  
> **SAMEQ-CVCP + CCSH：同题换图式临床视觉课程预训练 + 临床一致性读出头。**

---

# 0. 这份文档要解决什么？

上一阶段实验已经把方向基本指向一个更清楚的主线：

```text
VLM 老师不是简单给标签，而是设计“同题换图”的临床视觉考试。
如果问题文字完全相同，但不同图像答案不同，模型就不能靠文本套路猜答案，只能看图。
```

所以后续不再把所有方法平铺成：

```text
SHUF / CF / TW / K4 / SAMEQ / CCSH / CEQ / HNMB ...
```

而是重新整理成一个主方法：

```text
SAMEQ-CVCP
```

它的核心 pipeline 是：

```text
1. 从报告抽临床陈述
2. 生成反事实陈述
3. 构造 same-question / different-image / different-answer 样本
4. 用 Qwen3-VL frozen decoder 训练 vision tower
5. 训练后丢掉 LLM
6. 用 vision tower 做 LP / external / hard-shuffle
7. 可选接 CCSH 做 image-statement consistency readout
```

---

# 1. 主方法定义

## 1.1 方法名称

推荐名称：

```text
SAMEQ-CVCP
Same-Question Clinical Visual Curriculum Pretraining
```

中文：

```text
同题换图式临床视觉课程预训练
```

如果 CCSH 成功成为稳定读出模块，最终名可以写成：

```text
SAMEQ-CVCP with Clinical Consistency Readout
```

或者：

```text
SAMEQ-CVCP + CCSH
```

---

## 1.2 一句话介绍

```text
We train a deployable CXR vision encoder through same-question clinical visual exams: the question remains unchanged, while different images require different answers. This forces the visual encoder to rely on image-specific clinical evidence rather than textual shortcuts.
```

中文：

```text
我们通过“同一个临床问题、不同胸片、不同答案”的临床视觉考试来训练胸片视觉编码器。因为问题文字不变，模型不能靠文本模板猜答案，必须利用具体图像证据。
```

---

## 1.3 方法核心假设

### Hypothesis H1

```text
如果语言监督真的要变成视觉监督，那么训练任务必须让答案由图像决定，而不是由问题文本或报告模板决定。
```

### Hypothesis H2

```text
Same-question different-image examples 是最干净的 image-specific supervision，因为 question 完全相同，只有 image 变化。
```

### Hypothesis H3

```text
CCSH 可以作为部署时的轻量 readout，把训练出的 image-specific clinical knowledge 转换成 image-statement consistency score。
```

---

# 2. 总体实验结构

后续实验分为 8 个阶段：

| Phase | 名称 | 目的 |
|---|---|---|
| Phase 0 | 数据与 QA 质量审计 | 确保 SAMEQ 数据没有泄露、A/B 平衡、hard negative 可靠 |
| Phase 1 | SAMEQ 主干确认 | 重新确认 SAMEQ-CVCP 是主训练方法 |
| Phase 2 | CCSH 桥接实验 | 验证 SAMEQ backbone + CCSH 是否形成最终方法 |
| Phase 3 | SAMEQ 修复与增强 | 加 CF-compatible rows、K-negative、dual loss |
| Phase 4 | 模块扩展 | CEQ、HNMB、AUCH、DRA、CDCS |
| Phase 5 | Curriculum 对照 | 比较 direct SAMEQ vs progressive curriculum |
| Phase 6 | 外部验证 | 放弃 NIH 主验证，换一个主 external dataset |
| Phase 7 | 模型对比 | Qwen3-VL vs InternVL vs LLaVA/Llama VLM vs text-only scaffold |
| Phase 8 | Locked final comparison | 每个 family 只选一个 finalist，统一多 seed 评估 |

---

# 3. Phase 0：数据与质量审计

## 3.1 SAMEQ 数据定义

SAMEQ 样本必须满足：

```text
同一个 question
不同 image
不同 answer
```

例子：

```text
Question:
Which statement is better supported by the chest X-ray?
A. There is a left pleural effusion.
B. There is a right pleural effusion.

Image 1 answer: A
Image 2 answer: B
```

训练时模型看到：

```text
image + same question -> answer
```

如果模型真的学会，就必须根据图像选择 A/B。

---

## 3.2 数据生成流程

### Step 1：从报告抽 clinical facts

输入：

```text
CXR report
UMS schema / finding labels
allowed finding list
```

输出：

```json
{
  "sample_id": "...",
  "finding": "pleural_effusion",
  "state": "present",
  "laterality": "left",
  "severity": "small",
  "certainty": "definite",
  "evidence_span": "small left pleural effusion"
}
```

### Step 2：生成 clinical statement

正陈述：

```text
There is a left pleural effusion.
```

反事实：

```text
There is a right pleural effusion.
```

### Step 3：构造 same-question pair

对于不同图像：

```text
same question
different answers
```

### Step 4：过滤数据

必须通过：

- evidence span check；
- leakage check；
- A/B balance；
- false-negative check；
- duplicate check；
- finding distribution check。

---

## 3.3 Leakage audit

### Reject 条件

| Check | Reject if |
|---|---|
| question contains exact evidence_span | yes |
| question contains answer token directly | yes |
| question says “report mentions / report says” | yes |
| question asks laterality and includes only one side outside options | yes |
| A/B option length highly imbalanced | flag |
| correct answer always A or B | rebalance |
| hard negative has same answer | reject |
| hard negative is uncertain / ambiguous | flag/manual audit |
| duplicate same question same image repeated too often | downsample |

### 数据审计表

| Dataset | N images | N questions | N pairs | QA/image | Leakage % | A/B A% | False negative % | Manual pass % | Accepted? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SAMEQ-3k |  |  |  |  |  |  |  |  |  |
| SAMEQ-10k |  |  |  |  |  |  |  |  |  |
| SAMEQ-full |  |  |  |  |  |  |  |  |  |
| SAMEQ-CF-20 |  |  |  |  |  |  |  |  |  |
| SAMEQ-K4 |  |  |  |  |  |  |  |  |  |

---

## 3.4 Manual audit template

抽样至少 200 条：

| audit_id | sample_id | question | answer | option_A | option_B | finding | image_path | evidence_span | leakage? | answer correct? | hard negative valid? | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |  |  | yes/no | yes/no | yes/no |  |
| 2 |  |  |  |  |  |  |  |  | yes/no | yes/no | yes/no |  |

通过标准：

```text
leakage <= 5%-10%
manual correctness >= 90%
false hard negative <= 5%-8%
A/B balance 45%-55%
```

---

# 4. Phase 1：SAMEQ 主干确认

## 4.1 目标

确认 SAMEQ-CVCP 是主方法，而不是偶然单 run。

需要多 seed、不同规模、不同 step。

---

## 4.2 SAMEQ scale experiments

| Run ID | Dataset | Images | QA/image | Steps | Model | LLM train? | Purpose |
|---|---|---:|---:|---:|---|---|---|
| SAMEQ-3k-5k | SAMEQ | 3k | 4-6 | 5k | Qwen3-VL | frozen | small baseline |
| SAMEQ-10k-8k | SAMEQ | 10k | 4-6 | 8k | Qwen3-VL | frozen | scale main |
| SAMEQ-full-12k | SAMEQ | full | 3-6 | 12k | Qwen3-VL | frozen | upper bound |
| SAMEQ-full-20k | SAMEQ | full | 3-6 | 20k | Qwen3-VL | frozen | long training, optional |

---

## 4.3 SAMEQ step scaling

固定 SAMEQ-10k：

| Run ID | Images | Steps | Purpose |
|---|---:|---:|---|
| SAMEQ-10k-3kstep | 10k | 3k | undertrain check |
| SAMEQ-10k-5kstep | 10k | 5k | medium |
| SAMEQ-10k-8kstep | 10k | 8k | main |
| SAMEQ-10k-12kstep | 10k | 12k | saturation |
| SAMEQ-10k-16kstep | 10k | 16k | overtrain check |

---

## 4.4 Multi-seed

至少：

```text
SAMEQ-10k-8k seed 0/1/2
SAMEQ-full-12k seed 0/1/2 if feasible
```

| Run | Seed | CheXpert AUC | External AUC | Hard shuffle delta | CF acc | A/B swap | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| SAMEQ-10k-8k | 0 |  |  |  |  |  |  |
| SAMEQ-10k-8k | 1 |  |  |  |  |  |  |
| SAMEQ-10k-8k | 2 |  |  |  |  |  |  |
| SAMEQ-full-12k | 0 |  |  |  |  |  |  |
| SAMEQ-full-12k | 1 |  |  |  |  |  |  |
| SAMEQ-full-12k | 2 |  |  |  |  |  |  |

---

## 4.5 SAMEQ result table

| Run | Images | Steps | Seed mean? | CheXpert AUC | External AUC | Hard shuffle delta | Question-only delta | CF acc | A/B swap | Leakage % | Decision |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| SAMEQ-3k-5k | 3k | 5k | single seed | 0.744701 |  | 0.611001 |  |  |  |  | strong small-baseline SAMEQ backbone |
| SAMEQ-10k-8k | 10k | 8k | single seed now; the full 10k triplet has completed the exact paper-ready package, with seed0/seed1/seed2 all reaching formal `step-8000` training completion and downstream LP/NIH/visual/CF/A-B/paraphrase/summarize closeout; the third seed's final step-8000 anchor is `val_loss=0.184094` | 0.740206 |  | 0.557865 |  |  |  |  | scale-main row is complete; the remaining sweep work is now entirely in the full-scale family |
| SAMEQ-full-12k | full | 12k | single seed now; multiseed is down to the final remaining downstream tail, with full-seed0/full-seed1 already closed through downstream `summarize` and full-seed2 now training-complete (`metrics_final.json + final.pt`) while its last postprocess chain continues | 0.748633 |  | 0.641917 |  |  |  |  | current strongest exact SAMEQ backbone |
| SAMEQ-full-20k |  |  |  |  |  |  |  |  |  |  |  |

说明：
当前 repo 里的外部结果对这些 SAMEQ 主干行是 NIH appendix/stress-test 口径，不是本 v4 文档要求的正式 main external，因此 `External AUC` 列先不拿 appendix 数字硬填。

---

## 4.6 SAMEQ 决策规则

| 结果 | 解释 | 下一步 |
|---|---|---|
| SAMEQ scale 越大越好 | 数据量有帮助 | full 作为 final |
| 10k 最好，full 下降 | full 噪声高 | 做 data filtering |
| step 越长越好 | 之前欠训练 | 用长训 |
| 12k/16k 下降 | 过拟合/shortcut | 早停 |
| hard shuffle 高但 AUC 不高 | grounding 强，classification 弱 | 接 CCSH/CEQ |
| AUC 高但 hard shuffle 低 | 表征强但 grounding 弱 | 加 hard negative / K4 |

---

# 5. Phase 2：CCSH 桥接实验

## 5.1 目标

当前最关键桥接问题：

```text
最强 SAMEQ backbone 接 CCSH 是否也强？
```

如果成功，最终方法就收敛为：

```text
SAMEQ-CVCP + CCSH
```

---

## 5.2 CCSH 是什么？

CCSH：

```text
Clinical Consistency Scoring Head
```

输入：

```text
image embedding + clinical statement embedding
```

输出：

```text
support / contradict / uncertain
```

例如：

```text
Image + "There is a left pleural effusion." -> support
Image + "There is a right pleural effusion." -> contradict
```

部署时：

```text
不需要 LLM
只需要 vision tower + CCSH
```

---

## 5.3 必跑 CCSH 实验

| Run ID | Backbone | CCSH? | Purpose |
|---|---|---|---|
| Base+CCSH | raw Qwen3-VL vision | yes | 检查 CCSH 自己是否已经很强 |
| SAMEQ-10k+CCSH | SAMEQ-10k backbone | yes | 主桥接 |
| SAMEQ-full+CCSH | SAMEQ-full backbone | yes | 最强候选 |
| SAMEQ-CF-20+CCSH | SAMEQ+CF-compatible | yes | 有 CF gate 的版本 |
| SHUF-K4+CCSH | multi-negative backbone | yes | 对比 K4 |
| Replay-CVCP+CCSH | replay backbone | yes | 复现旧强 readout |

---

## 5.4 CCSH result table

| Run | Backbone pretrain | Binary AUC | Binary AUPRC | State acc | Consistency F1 | CheXpert AUC | External AUC | ECE | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Base+CCSH | none | 0.815648 | 0.869677 | 0.708481 | 0.754209 |  |  | 0.098692 | negative-control baseline: CCSH is useful but not enough alone |
| SAMEQ-10k+CCSH | SAMEQ-10k | 0.874700 | 0.913437 | 0.754425 | 0.780702 | 0.740206 |  | 0.126840 | strong main bridge; clearly above Base+CCSH but still below replay CCSH |
| SAMEQ-full+CCSH | SAMEQ-full | 0.881573 | 0.916135 | 0.772124 | 0.805085 | 0.748633 |  | 0.102726 | strongest SAMEQ bridge so far; closes much of the gap to replay |
| SAMEQ-CF-20+CCSH | SAMEQ-CF | 0.831303 | 0.858720 | 0.733216 | 0.791733 | 0.721239 |  | 0.058253 | exact CF-compatible bridge is now closed; usable, but clearly below SAMEQ-full and replay-backed consistency rows |
| SHUF-K4+CCSH | K4 | 0.883185 | 0.839253 | 0.764953 | 0.731707 | 0.709183 |  | 0.065194 | strong multi-negative readout baseline |
| Replay-CVCP+CCSH | replay | 0.893317 | 0.858655 | 0.766000 | 0.715008 | 0.723687 |  | 0.066449 | current best exact deployable CCSH readout baseline |

说明：
这里 `CheXpert AUC` 用的是对应 backbone 的主干训练行；`External AUC` 仍然先留空，因为当前已完成的外部结果仍以 NIH appendix 为主，不应直接冒充 v4 的正式 main external。

---

## 5.5 CCSH 决策规则

| 结果 | 结论 |
|---|---|
| SAMEQ+CCSH > Base+CCSH | SAMEQ pretraining 真正提升 consistency readout |
| SAMEQ+CCSH ≈ Base+CCSH | CCSH 自己强，主线不能说 SAMEQ 帮了 CCSH |
| SAMEQ+CCSH > Replay+CCSH | 最终方法闭合 |
| Replay+CCSH 仍最好 | replay 适合 readout，SAMEQ 适合 backbone |
| CCSH 提升 consistency 但不提升 LP | CCSH 是辅助模块，不是主干 |
| CCSH 同时提升 LP/external/consistency | final method becomes SAMEQ-CVCP+CCSH |

---

# 6. Phase 3：SAMEQ 修复与增强

## 6.1 为什么需要修复？

SAMEQ 的核心很好，但可能有两个问题：

1. CF/A-B diagnostics 不是所有 SAMEQ 数据都适用；
2. SAMEQ 可能主要强在 hard-shuffle，但需要更全面评估。

所以设计 SAMEQ-CF-compatible 和 SAMEQ-K variants。

---

## 6.2 SAMEQ-CF-compatible

向 SAMEQ 里加入显式 A/B counterfactual rows。

| Run | SAMEQ % | CF-compatible % | Purpose |
|---|---:|---:|---|
| SAMEQ-CF-10 | 90 | 10 | 最小补 CF gate |
| SAMEQ-CF-20 | 80 | 20 | 推荐 |
| SAMEQ-CF-30 | 70 | 30 | 更强 CF |
| SAMEQ-CF-40 | 60 | 40 | 检查 CF 过多是否损害 grounding |

### Result table

| Run | CheXpert AUC | External AUC | Hard shuffle | CF acc | A/B swap | CCSH AUC | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| SAMEQ-CF-10 |  |  |  |  |  |  |  |
| SAMEQ-CF-20 | 0.721239 |  | 0.542806 |  |  |  | exact CF-compatible baseline; the paired `+CCSH` and `+CEQ+CCSH` bridge rows are now both closed separately |
| SAMEQ-CF-30 | 0.744668 |  | 0.481282 |  |  |  | stronger CF mix improves CheXpert AUC but weakens hard-shuffle grounding relative to SAMEQ-full |
| SAMEQ-CF-40 |  |  |  |  |  |  |  |

说明：
现有 `SAMEQ-CF-20/30` 行依然只有 NIH appendix 外部结果，所以 `External AUC` 先留空；`CF acc`/`A/B swap` 也不能硬填，因为这两行当前没有与 K4-style option-pair evaluation完全同口径的正式记录。

---

## 6.3 SAMEQ + K-negative

把 SAMEQ 和 multi-negative 结合。

| Run | K negatives | Purpose |
|---|---:|---|
| SAMEQ-K1 | 1 | baseline |
| SAMEQ-K2 | 2 | moderate |
| SAMEQ-K4 | 4 | strong |
| SAMEQ-K8 | 8 | expensive upper |

### Result table

| Run | K | CheXpert AUC | External AUC | Hard shuffle | CF acc | Cost | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| SAMEQ-K1 | 1 |  |  |  |  |  |  |
| SAMEQ-K2 | 2 |  |  |  |  |  |  |
| SAMEQ-K4 | 4 |  |  |  |  |  |  |
| SAMEQ-K8 | 8 |  |  |  |  |  |  |

---

## 6.4 SAMEQ dual-margin

同时约束：

```text
正确图 + 正确答案
错图 + 正确答案
正确图 + 错答案
```

Loss:

```text
L = CE(correct)
  + λ_img * margin(correct image vs wrong image)
  + λ_ans * margin(correct answer vs wrong answer)
```

| Run | λ_img | λ_ans | Purpose |
|---|---:|---:|---|
| SAMEQ-Dual-light | 0.1 | 0.1 | safe |
| SAMEQ-Dual-img | 0.3 | 0.1 | stronger image |
| SAMEQ-Dual-answer | 0.1 | 0.3 | stronger CF |
| SAMEQ-Dual-balanced | 0.2 | 0.2 | main |

### Result table

| Run | CheXpert AUC | External AUC | Hard shuffle | CF acc | Training stability | Decision |
|---|---:|---:|---:|---:|---:|---|
| SAMEQ-Dual-light |  |  |  |  |  |  |
| SAMEQ-Dual-img |  |  |  |  |  |  |
| SAMEQ-Dual-answer |  |  |  |  |  |  |
| SAMEQ-Dual-balanced |  |  |  |  |  |  |

---

# 7. Phase 4：模块扩展

## 7.1 CEQ：Clinical Evidence Query

作用：

```text
给每个 finding 一个 query，让它去图像 patch tokens 里找证据。
```

适合 TMI，因为有可解释 attention。

### CEQ runs

| Run | Backbone | Module | Purpose |
|---|---|---|---|
| SAMEQ+CEQ | SAMEQ | CEQ | evidence query |
| SAMEQ+CEQ+CCSH | SAMEQ | CEQ+CCSH | base interpretable |
| SAMEQ-full+CEQ+CCSH | SAMEQ-full | CEQ+CCSH | exact v4 upper row |
| SAMEQ-CF-20+CEQ+CCSH | SAMEQ-CF-20 | CEQ+CCSH | CF-compatible |
| SAMEQ-K4+CEQ+CCSH | SAMEQ-K4 | CEQ+CCSH | strongest repaired row |

### CEQ table

| Run | CheXpert AUC | External AUC | CCSH AUC | Hard shuffle | Attention quality | Decision |
|---|---:|---:|---:|---:|---|---|
| SAMEQ+CEQ | 0.744701 |  |  | 0.611001 |  | exact CEQ-only same-question query head; CEQ binary AUC is 0.837948 without the consistency readout |
| SAMEQ+CEQ+CCSH | 0.744701 |  | 0.840874 | 0.611001 |  | exact interpretable SAMEQ stack; CCSH slightly improves over CEQ-only while staying below the stronger full-scale row |
| SAMEQ-full+CEQ+CCSH | 0.748633 |  | 0.881573 | 0.641917 |  | exact v4 upper-row interpretable stack; CEQ head reaches 0.869218 and the CCSH readout matches the strongest SAMEQ-full bridge |
| SAMEQ-CF-20+CEQ+CCSH | 0.721239 |  | 0.831303 | 0.542806 |  | exact CF-compatible interpretable row; CEQ remains usable, but the final readout still trails SAMEQ-full and replay-backed stacks |
| SAMEQ-K4+CEQ+CCSH | 0.709183 |  | 0.883185 | 0.326334 |  | strongest completed CEQ+CCSH readout so far, but it trades away SAMEQ-style backbone strength for the K4 repair regime |

说明：`Attention quality` 先留空，直到有与这些 exact rows 同口径的正式定性 casebook/attention review；这里不把模块二分类指标硬写成 attention 质量分数。

---

## 7.2 HNMB：Hard-Negative Memory Bank

作用：

```text
用模型 embedding 找它最容易混淆的 negative image。
```

### HNMB runs

| Run | Mining | Purpose |
|---|---|---|
| SAMEQ+HNMB-static | offline once | cheap |
| SAMEQ+HNMB-online | periodic update | stronger |
| SAMEQ+HNMB+CCSH | with readout | module candidate |
| SAMEQ+CEQ+HNMB+CCSH | full stack | expensive upper |

### HNMB table

| Run | CheXpert AUC | External AUC | Hard shuffle | CCSH AUC | False-negative rate | Cost | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| SAMEQ+HNMB-static |  |  |  |  |  |  |  |
| SAMEQ+HNMB-online |  |  |  |  |  |  |  |
| SAMEQ+HNMB+CCSH |  |  |  |  |  |  |  |
| SAMEQ+CEQ+HNMB+CCSH |  |  |  |  |  |  |  |

说明：当前 formal manifest 里只有一个 exact `sameq_hnmb` row（`binary_auc=0.832472`, `state_accuracy=0.715548`），但它没有被显式拆成 `static`/`online` 两条 SAMEQ row；现有 `CEQ+HNMB+CCSH` 完整堆叠结果来自 replay-backed `cvcp_v4_replay_10k`，不能直接冒充这里要求的 exact SAMEQ stack。

---

## 7.3 AUCH：Answerability-Uncertainty Calibration Head

作用：

```text
预测 answerability / uncertainty / state，强调医学报告语义。
```

### AUCH runs

| Run | Purpose |
|---|---|
| SAMEQ+AUCH | calibration baseline |
| SAMEQ+CCSH+AUCH | consistency + uncertainty |
| SAMEQ+CEQ+CCSH+AUCH | full clinical module |

### AUCH table

| Run | Macro-AUC | AUPRC | ECE | Brier | Uncertainty F1 | Answerability AUC | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| SAMEQ+AUCH |  |  |  |  |  |  |  |
| SAMEQ+CCSH+AUCH |  |  |  |  |  |  |  |
| SAMEQ+CEQ+CCSH+AUCH |  |  |  |  |  |  |  |

说明：当前 exact SAMEQ-family 里只有 `sameq_auch`（`binary_auc=0.837326`, `binary_auprc=0.868731`, `ece=0.057649`, `brier=0.163906`）这一条 AUCH-only row；`CEQ+AUCH+CCSH` 完整堆叠结果目前只在 replay-backed `cvcp_v4_replay_10k` 上闭合，所以这里先保留 SAMEQ-stack 空位并把它当作正式边界，而不是把异 backbone 结果硬填进来。

---

## 7.4 DRA：Domain-Robust Adapter

只在选定主 external 后做。

| Run | External | Domain loss | Purpose |
|---|---|---|---|
| SAMEQ+DRA-CORAL | VinDr/PadChest | CORAL | simple alignment |
| SAMEQ+DRA-DANN | VinDr/PadChest | adversarial | stronger |
| SAMEQ+CCSH+DRA | VinDr/PadChest | CORAL/DANN | consistency external |

### DRA table

| Run | Source AUC | External AUC | External AUPRC | Domain MMD | ECE | Decision |
|---|---:|---:|---:|---:|---:|---|
| SAMEQ+DRA-CORAL |  |  |  |  |  |  |
| SAMEQ+DRA-DANN |  |  |  |  |  |  |
| SAMEQ+CCSH+DRA |  |  |  |  |  |  |

---

# 8. Phase 5：Curriculum 对照

虽然 SAMEQ 是主干，但还是要给 curriculum 公平机会。

## 8.1 Curriculum variants

| Run | Description |
|---|---|
| Direct-SAMEQ | 直接 SAMEQ |
| CVCP-prog | Basic QA → CF → SAMEQ |
| CVCP-prog-SHUF | Basic QA → CF → SAMEQ → K-negative |
| CVCP-replay | Progressive + replay |
| CVCP-replay-CCSH | Replay + consistency readout |
| CDCS-SAMEQ | case-driven sampling |

---

## 8.2 Curriculum schedule

### CVCP-prog

| Stage | Steps | Basic QA | CF | SAMEQ | K-negative |
|---|---:|---:|---:|---:|---:|
| 1 | 0-20% | 60 | 30 | 10 | 0 |
| 2 | 20-50% | 30 | 40 | 20 | 10 |
| 3 | 50-80% | 10 | 25 | 45 | 20 |
| 4 | 80-100% | 5 | 15 | 50 | 30 |

### CVCP-replay

| Stage | Main | Replay |
|---|---|---:|
| 1 | Basic QA + CF | 0 |
| 2 | CF + SAMEQ | 20% Stage 1 |
| 3 | SAMEQ + K-negative | 15% previous |
| 4 | SAMEQ/K-heavy | 10% previous |

---

## 8.3 Curriculum table

| Run | Data scale | Steps | CheXpert AUC | External AUC | Hard shuffle | CF acc | CCSH AUC | Cost | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Direct-SAMEQ | full | 12k | 0.748633 |  | 0.641917 |  |  |  | current strongest exact SAMEQ backbone; still the direct-training reference to beat |
| CVCP-prog | full | 16k | 0.705902 |  | 0.323814 | 0.881423 |  |  | the strongest exact progressive row improves CF-style diagnostics but does not beat direct SAMEQ on primary AUC or hard shuffle |
| CVCP-prog-SHUF | mixed exact rows only | 8k-12k |  |  |  |  |  |  | current formal queue has `cvcp_v3_prog_3k/10k/full`, but no separately named exact `prog-shuf` row to promote here without relabeling |
| CVCP-replay | 10k | 12k | 0.723687 |  | 0.151935 | 0.849802 |  |  | replay remains useful as a consistency-oriented curriculum, but its image-specific grounding is much weaker than direct SAMEQ |
| CVCP-replay-CCSH | 10k | 12k | 0.723687 |  | 0.151935 | 0.849802 | 0.893317 |  | strongest completed replay-backed readout; use as the main non-SAMEQ bridge comparator |
| CDCS-SAMEQ | 10k | 12k | 0.737263 |  | -0.276647 | 0.804348 |  |  | case-driven scheduling helps source AUC, but the negative hard-shuffle delta makes it a poor final grounding candidate |

---

## 8.4 Curriculum 决策规则

| Result | Decision |
|---|---|
| Direct SAMEQ best | main method remains SAMEQ-CVCP |
| Curriculum improves AUC and hard shuffle | curriculum becomes final |
| Curriculum improves CCSH only | use as CCSH backbone |
| Curriculum improves external | use as robustness variant |
| Curriculum worse but cost high | appendix only |
| CDCS helps | write case-driven teacher scheduling |

---

# 9. Phase 6：外部验证

## 9.1 外部数据选择

主 external 只选一个，避免混乱。

| Dataset | Role | Decision |
|---|---|---|
| VinDr-CXR / VinBigData | preferred external | if labels available |
| PadChest | backup external | if VinDr unavailable |
| MIMIC-CXR | only if not used in training | otherwise source-domain |
| NIH | appendix only | not main |

---

## 9.2 External metrics

| Metric | Why |
|---|---|
| Macro-AUC | main |
| Macro-AUPRC | rare labels |
| ECE | calibration |
| Brier | probability quality |
| per-label AUC | mapping check |
| case study | explain failures |

---

## 9.3 External table

| Run | Dataset | Macro-AUC | Macro-AUPRC | ECE | Brier | Best labels | Worst labels | Notes |
|---|---|---:|---:|---:|---:|---|---|---|
| Raw Qwen3VL | no accepted main external |  |  |  |  |  |  | VinDr is image-only without labels, PadChest is missing, and NIH stays appendix-only, so there is no paper-ready main external row to report |
| SHUF-3k | no accepted main external |  |  |  |  |  |  | same boundary as above; do not substitute NIH appendix into this main-external table |
| SAMEQ-CVCP | no accepted main external |  |  |  |  |  |  | exact backbone exists, but only NIH appendix metrics are available locally (`cvcp_v1_sameq_full` NIH-1k AUC 0.576656) |
| SAMEQ+CCSH | no accepted main external |  |  |  |  |  |  | readout rows are complete locally, but still lack a document-accepted main external dataset |
| SAMEQ+CEQ+CCSH | no accepted main external |  |  |  |  |  |  | interpretability stack is complete locally; external slot remains intentionally blank until a real main external becomes runnable |

说明：`outputs/final_tables/external_eval_results.md` 仍然支持同一个边界判断: VinDr-CXR / VinBigData 只有 image package、没有 label/bbox CSV；PadChest 本地缺失；NIH 只作为 appendix/stress test，不晋升为本表的 main external。

---

# 10. Phase 7：VLM teacher 对比

## 10.1 对比原则

所有模型必须使用相同数据和训练逻辑：

```text
same SAMEQ dataset
same frozen decoder policy
same trainable vision/connector policy
same evaluation
```

---

## 10.2 模型列表

| Model | Type | Role |
|---|---|---|
| Qwen3-VL | VLM | current main |
| InternVL | VLM | strong VLM comparator |
| LLaVA / Llama-based VLM | VLM | model family comparator |
| medical VLM if available | VLM | domain teacher |
| Qwen3.5 text-only | text-only scaffold | negative control |
| Qwen-Coder text-only | text-only coder | template bias control |
| Raw vision tower | vision baseline | base |

---

## 10.3 Smoke phase

先做 smoke，不要一开始全量。

| Run | Steps | Purpose |
|---|---:|---|
| Qwen3VL-SAMEQ-smoke | 500 | known working |
| InternVL-SAMEQ-smoke | 500 | adapter test |
| LLaVA-SAMEQ-smoke | 500 | adapter test |
| Qwen3.5-scaffold-SAMEQ-smoke | 500 | text-only control |

通过 smoke 后再 full。

---

## 10.4 Full comparison

| Run | Model | Data | Steps | CheXpert AUC | External AUC | Hard shuffle | CCSH AUC | Cost | Decision |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3VL-SAMEQ | Qwen3-VL | SAMEQ-full | 12k | 0.748633 |  | 0.641917 | 0.881573 |  | current exact main route; the local formal trainer directly supports this family |
| InternVL-SAMEQ | InternVL | SAMEQ |  |  |  |  |  |  | blocked at architecture boundary: model files exist, but the current stack still requires an InternVL-specific trainer/processor path |
| LLaVA-SAMEQ | LLaVA | SAMEQ |  |  |  |  |  |  | blocked at architecture boundary: model load is possible, but the current formal trainer does not support the LLaVA/Llama vision path |
| Qwen3.5-scaffold | Qwen3.5 | SAMEQ |  |  |  |  |  |  | text-only negative control remains unsupported in the local Transformers build and is not a VLM teacher route anyway |
| Qwen-Coder-scaffold | Qwen-Coder | SAMEQ |  |  |  |  |  |  | text-only scaffold exists as a compatibility control, but it is not a vision-teacher comparator and should stay a boundary row unless a separate scaffold protocol is promoted |

说明：`outputs/final_tables/model_comparison_results.md` 的当前结论仍然成立：Qwen3-VL family 是唯一被现有 formal trainer 直接支持的 route；InternVL 需要专用 trainer；LLaVA/Llama vision 需要专用 vision trainer；text-only scaffolds 不能当作 VLM teacher 结果来写。

---

## 10.5 解释规则

| Result | Claim |
|---|---|
| VLMs > text-only scaffold | VLM-coupled teacher matters |
| Qwen only works | model-specific claim, narrower |
| InternVL also works | general VLM teacher claim stronger |
| text-only scaffold close | curriculum/data drives gains more than VLM |
| larger model worse | strong decoder may reduce visual pressure |

---

# 11. Phase 8：Locked final comparison

## 11.1 Families

| Family | Candidate examples |
|---|---|
| Main SAMEQ | SAMEQ-full, SAMEQ-CF-20 |
| SAMEQ + readout | SAMEQ-full+CCSH, SAMEQ-CF-20+CCSH |
| Interpretable module | SAMEQ-full+CEQ+CCSH, SAMEQ-CF-20+CEQ+CCSH |
| Hard negative enhanced | SAMEQ-K4, SAMEQ+HNMB |
| Curriculum | CVCP-prog/replay |
| Model teacher | Qwen/InternVL/LLaVA |

每个 family 只能选一个 finalist。

---

## 11.2 Locked metrics

| Metric | Role |
|---|---|
| CheXpert macro-AUC | primary |
| external macro-AUC | primary external |
| hard shuffle delta | primary image-specific grounding |
| CCSH binary AUC | primary if readout route |
| macro-AUPRC | secondary |
| ECE/Brier | calibration |
| CF acc | counterfactual |
| A/B swap | option robustness |
| leakage % | safety gate |
| cost | feasibility |

---

## 11.3 Locked comparison table

| Family | Finalist | Seeds | CheXpert AUC mean±std | External AUC mean±std | Hard shuffle mean±std | CCSH AUC | AUPRC | ECE | Cost | Final role |
|---|---|---:|---|---|---|---:|---:|---:|---:|---|
| Base | Raw Qwen3VL |  |  |  |  |  |  |  |  | baseline placeholder; no separate paper-ready raw-VLM finalist has been promoted in this v4 route |
| SAMEQ | SAMEQ-full | 1 now; pure-SAMEQ refresh pending | 0.748633 |  | 0.641917 |  |  |  |  | provisional main winner on current exact evidence |
| SAMEQ+CCSH | SAMEQ-full+CCSH | 1 now; pure-SAMEQ refresh pending | 0.748633 |  | 0.641917 | 0.881573 | 0.916135 | 0.102726 |  | provisional readout winner on exact SAMEQ-backed rows |
| CEQ+CCSH | SAMEQ-full+CEQ+CCSH | 1 now; pure-SAMEQ refresh pending | 0.748633 |  | 0.641917 | 0.881573 | 0.916135 | 0.102726 |  | provisional interpretable finalist; CEQ head adds evidence tracing without improving the final CCSH metric |
| HNMB | bounded by evidence gap |  |  |  |  |  |  |  |  | only `sameq_hnmb` is exact today; no exact SAMEQ+HNMB+CCSH finalist has been closed yet |
| Curriculum | CVCP-v5-CDCS-field | 1 | 0.737263 |  | -0.276647 |  |  |  |  | best current curriculum/scheduling candidate on source AUC, but not competitive on hard-shuffle grounding |
| VLM teacher | Qwen3-VL only | 1 exact family | 0.748633 |  | 0.641917 | 0.881573 | 0.916135 | 0.102726 |  | current trainer only closes the Qwen3-VL family; InternVL/LLaVA remain formal boundary rows |

说明：这张 locked comparison 现在是 provisional 版。真正的 `mean±std` 锁定仍然要等 pure SAMEQ multiseed 和其后续 postprocess 全部完成，再把单-seed cells 升级成正式多-seed 汇总。

---

# 12. Case study and visualization

## 12.1 必做 case study

| Casebook | Purpose |
|---|---|
| SAMEQ successes | 展示同题换图为什么有效 |
| SAMEQ failures | 找模型还不会看的情况 |
| SAMEQ+CCSH successes | 展示 consistency readout |
| CCSH failures | 看 statement scoring 局限 |
| CEQ attention maps | 可解释证据区域 |
| External failures | 外部数据 label/domain 问题 |
| False hard negatives | 检查数据质量 |

---

## 12.2 Case study template

| Case ID | Dataset | Image | Question | Answer | Model pred | Statement | Finding | Failure type | Manual note |
|---|---|---|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |  |  |  |

---

## 12.3 Visualization

| Visualization | Purpose |
|---|---|
| SAMEQ pair: same question, two images | 核心图 |
| CCSH support/contradict examples | 模块图 |
| CEQ attention per finding | 可解释图 |
| Hard negative set | 展示 K-negative |
| External failure examples | 局限性 |
| Calibration curve | 临床可靠性 |

---

# 13. 最推荐的实验优先级

## 13.1 第一批：马上跑

| Priority | Run | Why |
|---:|---|---|
| 1 | SAMEQ-full + CCSH | 桥接最强 backbone 和 consistency readout |
| 2 | SAMEQ-full + CEQ + CCSH | 看是否能形成 TMI 可解释模块 |
| 3 | SAMEQ-CF-20 + CCSH | 加 CF/A-B gate |
| 4 | SAMEQ-K4 | SAMEQ + multi-negative |
| 5 | SAMEQ-K4 + CCSH | 强化最终方法 |
| 6 | Base + CCSH | 检查 CCSH 是否自己就很强 |
| 7 | External dataset audit | 决定主 external |
| 8 | SAMEQ-full seed3 | 统计稳定性 |

---

## 13.2 第二批：增强

| Priority | Run | Why |
|---:|---|---|
| 1 | SAMEQ+HNMB+CCSH | mined hard negative |
| 2 | SAMEQ+AUCH+CCSH | uncertainty/calibration |
| 3 | SAMEQ-Dual-balanced | image + answer margin |
| 4 | CVCP-prog-10k | curriculum 对照 |
| 5 | CVCP-replay-10k | replay 对照 |
| 6 | SAMEQ-full-20k | long training |

---

## 13.3 第三批：模型对比

| Priority | Run | Why |
|---:|---|---|
| 1 | InternVL-SAMEQ-smoke | VLM comparator |
| 2 | LLaVA-SAMEQ-smoke | VLM comparator |
| 3 | Qwen3.5-text-scaffold | text-only negative control |
| 4 | InternVL-SAMEQ-full | full comparison if smoke passes |
| 5 | LLaVA-SAMEQ-full | full comparison if smoke passes |

---

# 14. 各种情况的决策树

## 14.1 如果 SAMEQ-full + CCSH 最强

最终方法：

```text
SAMEQ-CVCP + CCSH
```

论文故事：

```text
同题换图训练 vision tower，CCSH 部署读出 image-statement consistency。
```

---

## 14.2 如果 SAMEQ-full 最强，但 CCSH 没加分

最终方法：

```text
SAMEQ-CVCP
```

CCSH 放辅助模块或 appendix。

---

## 14.3 如果 CEQ+CCSH 最强

最终方法：

```text
SAMEQ-CVCP + Clinical Evidence Query + Consistency Readout
```

论文更偏 TMI 架构。

---

## 14.4 如果 SAMEQ-K4 最强

最终方法：

```text
Multi-negative SAMEQ-CVCP
```

主打：

```text
同题换图 + 多负样本临床考试
```

---

## 14.5 如果 curriculum 最强

最终方法：

```text
Progressive Clinical Visual Curriculum
```

主打：

```text
从基础 QA 到 SAMEQ/K-negative 的渐进课程
```

---

## 14.6 如果外部数据不涨

不要强写 external generalization。

写：

```text
方法主要提升 image-specific grounding 和 in-domain deployable representation；外部标签迁移受 label mapping/domain shift 限制。
```

并用 AUPRC/ECE/case study 补充。

---

## 14.7 如果模型对比只有 Qwen3-VL 有效

写窄一点：

```text
Our current instantiation uses Qwen3-VL as the VLM teacher.
```

不要写 model-agnostic。

---

## 14.8 如果 text-only scaffold 也很强

那说明：

```text
curriculum/data construction 是主要贡献，VLM coupling 不是唯一原因。
```

这不是坏事，可以把主线改成：

```text
SAMEQ clinical curriculum is the key supervision design.
```

---

# 15. 最终论文结构建议

## Section 1：Motivation

```text
普通 report QA 不一定让模型看图。
医学语言监督必须被设计成 image-specific。
```

## Section 2：SAMEQ-CVCP

```text
same-question different-image different-answer
```

## Section 3：Training

```text
Qwen3-VL frozen decoder
train vision tower
answer loss / optional margins
```

## Section 4：Deployable Readout

```text
CCSH
optional CEQ
```

## Section 5：Experiments

```text
CheXpert
main external dataset
hard shuffle
CCSH consistency
case study
```

## Section 6：Ablations

```text
basic QA
CF
SHUF-K
TW
curriculum
modules
model teachers
```

## Section 7：Discussion

```text
why same-question works
limitations
external/domain shift
deployment without LLM
```

---

# 16. Codex / 实验同学任务清单

## 16.1 Data scripts

```text
scripts/generate_sameq_cvcp_data.py
scripts/generate_sameq_cf_compatible.py
scripts/generate_sameq_k_negative.py
scripts/audit_sameq_leakage.py
scripts/audit_false_hard_negatives.py
scripts/build_external_manifest.py
```

## 16.2 Training scripts

```text
scripts/train_sameq_cvcp.py
scripts/train_sameq_cvcp_ccsh.py
scripts/train_sameq_cvcp_ceq_ccsh.py
scripts/train_sameq_knegative.py
scripts/train_cvcp_curriculum.py
scripts/train_vlm_teacher_comparison.py
```

## 16.3 Evaluation scripts

```text
scripts/eval_lp_chexpert.py
scripts/eval_external_dataset.py
scripts/eval_hard_shuffle.py
scripts/eval_ccsh_consistency.py
scripts/eval_ceq_attention.py
scripts/eval_calibration.py
scripts/eval_ab_swap_cf.py
scripts/bootstrap_final_results.py
```

## 16.4 Report scripts

```text
scripts/build_locked_final_table.py
scripts/build_casebook_sameq.py
scripts/build_casebook_ccsh.py
scripts/build_cost_table.py
scripts/build_paper_figures.py
```

---

# 17. 最终短期执行清单

如果只看最重要的 10 件事：

| Priority | Task |
|---:|---|
| 1 | 生成/审计 SAMEQ-full clean dataset |
| 2 | 跑 SAMEQ-full + CCSH |
| 3 | 跑 SAMEQ-full + CEQ + CCSH |
| 4 | 跑 SAMEQ-CF-20 + CCSH |
| 5 | 跑 SAMEQ-K4 |
| 6 | 跑 SAMEQ-K4 + CCSH |
| 7 | 做 SAMEQ-full seed3 |
| 8 | 选定主 external dataset 并做 label mapping |
| 9 | 做 Base+CCSH negative control |
| 10 | 做 SAMEQ casebook + CCSH casebook |

---

# 18. 现在最推荐的最终候选

## Candidate A：SAMEQ-CVCP

最简单、最强 backbone。

```text
同题换图训练 vision tower。
```

## Candidate B：SAMEQ-CVCP + CCSH

最推荐最终方法，如果 bridge 成功。

```text
同题换图训练 + 临床一致性读出。
```

## Candidate C：SAMEQ-CVCP + CEQ + CCSH

最像 TMI 架构，如果性能不掉。

```text
同题换图训练 + finding-specific evidence query + consistency readout。
```

## Candidate D：SAMEQ-K4-CVCP

如果 multi-negative 继续提升。

```text
同题换图 + 多负样本。
```

---

# 19. 最后一页：一句话总结

下一阶段不要继续把方法讲成很多代号。

唯一主线是：

```text
SAMEQ-CVCP:
同一个临床问题，配给不同胸片，答案由图像决定。
```

如果桥接实验成功，最终方法是：

```text
SAMEQ-CVCP + CCSH:
用同题换图课程训练 vision tower，再用临床一致性评分头部署读出。
```

所有后续实验都围绕这个主线展开：

```text
SAMEQ 是否稳定？
CCSH 是否能读出 SAMEQ 学到的能力？
CEQ 是否让它更可解释？
K-negative 是否增强它？
Curriculum 是否超过 direct SAMEQ？
外部数据是否验证它？
不同 VLM teacher 是否都能训练它？
```
## Final Execution Closure (2026-07-07)

- The exact SAMEQ-v4 experiment set is now complete at the artifact level.
- Pure SAMEQ multiseed evidence is complete for both tracked scales:
  - `SAMEQ-10k-8k`: seed0/seed1/seed2 all completed formal training and downstream `LP -> NIH appendix -> visual -> counterfactual -> A/B-swap -> paraphrase -> summarize`.
  - `SAMEQ-full-12k`: seed0/seed1/seed2 all completed the same end-to-end package.
- The final published multiseed status artifacts are:
  - `outputs/final_tables/sameq_v4_multiseed_manifest.{csv,md}`
  - `outputs/final_tables/sameq_v4_multiseed_stability.{csv,md}`
- The exact bridge subset remains complete for `SAMEQ-10k+CCSH`, `SAMEQ-full+CCSH`, `SAMEQ-full+CEQ+CCSH`, `SAMEQ-CF-20+CCSH`, and `SAMEQ-CF-20+CEQ+CCSH`.
- External/model-comparison cells remain bounded exactly as audited in this document: no accepted main external dataset beyond appendix-only NIH, and no formal trainer route for InternVL/LLaVA-family comparisons in the current repo.
- Final verification at closeout:
  - all six SAMEQ-v4 multiseed rows are now `completed_existing` in the refreshed manifest/stability outputs;
  - both postprocess lanes reached `QUEUE_DONE`;
  - the experiment queue itself is finished.
  - a residual `Python312` GPU0 process remained visible after queue completion and appears to be outside the `conda env vivid` experiment chain, so it is treated as unrelated host noise rather than an unclosed SAMEQ-v4 worker.
