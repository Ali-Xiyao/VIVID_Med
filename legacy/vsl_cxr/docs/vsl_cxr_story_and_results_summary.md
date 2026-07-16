# VSL-CXR 实验结论、论文故事线与当前结果总览

生成时间：2026-07-07  
对应源计划：`vivid_med_vsl_cxr_full_experiment_plan_v5.md`  
当前证据入口：

- `outputs/final_tables/vsl_cxr_formal_run_results.md`
- `outputs/final_tables/vsl_cxr_ceq_results.md`
- `outputs/final_tables/vsl_cxr_ccsh_results.md`
- `outputs/final_tables/vsl_cxr_auch_results.md`
- `outputs/final_tables/vsl_cxr_phase5_candidate_results.md`
- `outputs/final_tables/vsl_cxr_external_results.md`
- `outputs/final_tables/vsl_cxr_teacher_comparison_results.md`
- `outputs/final_tables/vsl_cxr_phase8_casebook.md`
- `outputs/final_tables/vsl_cxr_locked_final_comparison.md`

## 1. 一句话结论

这轮实验把 VIVID-Med 的主线从“报告问答 / SAMEQ / CVCP / CCSH 等多个看似分散的模块”收束成一个更清楚的论文故事：

> 胸片报告中的语言监督并不天然等于视觉监督。VSL-CXR 将报告派生的临床陈述重新表述为“视觉充分性学习”：给定一张胸片和一个临床陈述，模型不仅学习陈述是否为真，还学习这张图像是否提供了足够视觉证据来支持、反驳、无法判断或证据不足。这个框架可以训练一个可部署的 CXR 视觉编码器，并通过 CCSH/AUCH 等轻量读出模块在部署时减少对 LLM 的依赖。

当前最适合写进论文的核心结论是：

1. SAMEQ 类视觉对比监督确实比单纯报告 QA 更接近视觉表征学习。
2. CEQ-region 是当前最强的 evidence-aware encoder 变体，CEQ 与 CCSH 组合在二分类读出上最强。
3. CCSH-CEQ 是当前最强 deployable readout，AUCH-CEQ-CCSH 的 AUPRC 最好，AUCH-VSL4 的 ECE 最好。
4. VSL-Full 是当前 locked integrated finalist，因为它完成了 D9 mixed-instruction 训练，并在 Phase 6 CheXpert LP 中达到最高 macro-AUC。
5. 但外部泛化证据仍必须保守书写：NIH 只能作为 appendix/stress；VinDr-CXR 主 external 原始包与标签已经到位，当前正在完成解压、完整性审计和五个正式推理 row，在结果落盘前仍不能给出主 external 性能 claim。
6. 当前 locked final table 是 single-seed evidence，不应写成多 seed 统计显著优势。

## 2. 论文想讲的故事

### 2.1 问题动机

医学影像中的报告文本非常丰富，但报告文本并不总是直接、唯一、充分地绑定到图像证据。例如：

- 报告可能包含历史信息、比较信息或临床背景；
- 某些句子可由图像直接支持；
- 某些句子被图像反驳；
- 某些句子在图像上不确定；
- 某些问题本身并不适合从当前图像判断。

如果我们直接把报告派生 QA 当作视觉监督，模型可能学到语言模板、标签先验、报告分布，甚至数据集特定的表达方式，而不是可迁移的影像证据。

VSL-CXR 的核心动机就是把这个监督目标重新定义为：

> 不是问“报告说了什么”，而是问“这张胸片是否给出了足够视觉证据来支持这个临床陈述”。

这让论文从传统 image-report alignment 转向更有辨识度的 visual sufficiency learning。

### 2.2 方法主线

VSL-CXR 可以拆成四层：

1. Data engine：从报告中构造 clinical statement、counterfactual statement、SAMEQ pair、hard negative、support/contradict/uncertain/insufficient 四类 sufficiency label。
2. Encoder learning：用 Qwen3-VL teacher/vision tower 学习视觉充分性表征，包含 SAMEQ、SAMEQ-K、SAMEQ-HNMB、VSL-4class、hierarchical loss 等变体。
3. Evidence-aware modules：用 CEQ 学习“在哪里找证据”，用 CCSH 学习可部署的一致性读出，用 AUCH 学习 answerability/uncertainty。
4. Final evaluation：用 CheXpert LP、NIH appendix/stress、casebook、teacher comparison、locked final table 组织证据。

这个结构的好处是：论文不需要把 SAMEQ、CEQ、CCSH、AUCH 写成几个散乱的小技巧，而可以把它们讲成 VSL-CXR 的逐层实现：

- SAMEQ 解决“同题不同图”的视觉对比问题；
- hard negative 解决“容易被语言模板骗过”的负例质量问题；
- CEQ 解决“证据区域在哪里”的问题；
- CCSH 解决“部署时不用 LLM 也能读出一致性”的问题；
- AUCH 解决“不确定/不可回答”的校准问题；
- locked final comparison 负责把每个 family 的 strongest evidence 收成最终故事。

## 3. 当前实验全景

### 3.1 已完成的主实验块

当前 v5 formal run table 为：

- `rows=33`
- `completed=33`

这 33 个 formal rows 覆盖：

- B0 Raw-Vision LP；
- B1-B6 Basic-QA / CF-QA / SAMEQ / SAMEQ-CF / SAMEQ-K4 / SAMEQ-HNMB；
- B7 VSL-4class；
- Phase 2 的 VSL-2class、VSL-3class、VSL-4class-balanced、VSL-4class-field-balanced、VSL-hierarchical；
- Phase 3 的 CEQ variants；
- Phase 4 的 CCSH/AUCH readouts；
- Phase 5 的 VSL-Full；
- Phase 6 的 CheXpert LP readouts。

此外：

- Phase 5 candidate table 有 6 行；
- Phase 6 external table 有 8 行；
- Phase 7 teacher comparison table 有 9 行；
- Phase 8 casebook 有 33 行；
- Phase 9 locked final table 有 8 行；
- readiness audit 中 v5 script surface 已达到 `script exact_exists=29`。

### 3.2 仍需谨慎处理的边界

这些不是可以忽略的小问题，而是论文写作时必须主动说明的边界：

1. 主 external 正在接入：官方 VinDr-CXR 1.0.0 的 18,000 DICOM、bbox 和 image-level labels 已到位；直接七标签映射与 deterministic manifests 已生成，解压/完整性审计和五组 3,000-image 推理尚在运行。
2. NIH 只能作为 appendix/stress：所有 NIH external 数字都不能写成主 external claim。
3. locked final 是 single-seed：不能写多 seed mean±std 或显著性结论。
4. Phase 8 casebook 需要人工视觉复核：当前是可用候选 casebook，不是最终论文图注。
5. calibration 有 ECE/Brier summary，但没有 binned curve points。
6. teacher comparison 的 InternVL/LLaVA/text scaffold 是 adapter/trainer blocked，不是模型失败。

## 4. 分阶段实验结论

## 4.1 Phase 1：baseline 与主干确认

Phase 1 主要回答：报告派生 QA、counterfactual、SAMEQ、hard negative、VSL-4class 这些监督信号中，哪些能形成更好的视觉表征。

关键结果：

| Run | 结论性指标 | 当前解释 |
|---|---:|---|
| B0 Raw-Vision LP | CheXpert macro-AUC 0.679003 | 原始 Qwen3-VL vision tower 已有一定 CXR 表征能力，但只是 raw feature baseline |
| B1 Basic-QA | best val loss 0.023826 | 文本 QA 很容易拟合，不能直接证明视觉充分性 |
| B2 CF-QA | best val loss 0.120357 | counterfactual 比 Basic-QA 更难，但仍主要是 QA-style supervision |
| B3 SAMEQ | best val loss 0.176729 | SAMEQ 成为核心视觉对比信号 |
| B4 SAMEQ-CF | best val loss 0.213187 | SAMEQ-CF 更难，但没有明显成为最终最强主线 |
| B5 SAMEQ-K4 | best val loss 0.127735 | K-negative 变体成为 VSL-Core 的基础 |
| B6 SAMEQ-HNMB | best val loss 0.095854 | HNMB 在训练 loss 和 CCSH 读出上有优势 |
| B7 VSL-4class | best val loss 0.394839 | 四类 visual sufficiency 监督跑通，是论文定义的核心任务 |

论文解释：

- Basic-QA 的低 loss 不能被过度解读，因为它可能包含大量语言/标签模板拟合。
- SAMEQ 和 hard negative 系列更贴近视觉判别，因为它们强制模型比较“同一问题在不同图像上的答案变化”。
- HNMB 的训练 loss 最低，说明 mined hard negatives 可能让监督更集中；但 Phase 6 LP 没有专门给 HNMB 一行，因此最终不能只靠 HNMB 作为 integrated winner。

## 4.2 Phase 2：Visual Sufficiency Data Engine

Phase 2 主要回答：support / contradict / uncertain / insufficient 这些 label 设计是否能跑通，以及哪种 label/loss 形式更稳定。

结果：

| Run | best val loss | 解释 |
|---|---:|---|
| VSL-2class | 0.046710 | support vs contradict 最容易，说明二分类 visual sufficiency 监督信号清晰 |
| VSL-3class | 0.140873 | 加入 uncertain 后难度上升，但仍稳定 |
| VSL-4class | 0.394839 | 完整四类任务更难，符合预期 |
| VSL-4class-balanced | 0.452252 | 类均衡后任务更难，说明原始分布中存在类别/难度偏置 |
| VSL-4class-field-balanced | 0.349429 | finding-balanced 反而优于 class-balanced，提示 disease/finding distribution 比 label count 更关键 |
| VSL-hierarchical | 0.473558 | hierarchical loss 跑通，但当前 single-run 不优于 field-balanced |

论文解释：

- VSL-2class / 3class / 4class 的 loss 梯度说明：任务难度随着 answerability 和 insufficiency 维度加入而增加。
- field-balanced 好于 class-balanced，说明 CXR 的视觉充分性监督不能只平衡 label，还要平衡 finding。
- hierarchical loss 的方向合理，但当前不是 strongest row；可以作为 future improvement 或 ablation 讨论。

## 4.3 Phase 3：CEQ evidence-aware encoder

Phase 3 主要回答：显式 evidence query 是否能改善模型对“证据区域/证据状态”的表征。

结果：

| CEQ variant | State accuracy | Binary AUC | AUPRC | ECE | 结论 |
|---|---:|---:|---:|---:|---|
| CEQ-basic | 0.512 | 0.768057 | 0.656821 |  | baseline CEQ 可用 |
| CEQ-diverse | 0.540 | 0.690942 | 0.514169 |  | query diversity 当前没有带来更强 AUC |
| CEQ-sparse | 0.646 | 0.715837 | 0.663323 |  | sparsity 提升 state accuracy，但 AUC 不最高 |
| CEQ-region | 0.716 | 0.847173 | 0.801449 | 0.082690 | 当前最强 CEQ finalist |
| CEQ-statement | 0.702 | 0.720522 | 0.716206 |  | statement-conditioned query 有效但不如 region |

论文解释：

- CEQ-region 是最值得写成主结果的 CEQ 变体。
- region weak labels 对 evidence-aware learning 很重要，因为它将“是什么 finding”推进到“证据大概在哪里”。
- CEQ 的价值不是直接提高所有 external 指标，而是在 CCSH readout 中提供更强、更可解释的中间表征。

## 4.4 Phase 4：CCSH/AUCH deployable readout

Phase 4 是当前最有论文说服力的一组结果，因为它回答部署问题：

> 能否在不用 LLM 的情况下，从训练后的视觉表征中读出 statement-image consistency？

关键结果：

| Readout | Binary AUC | AUPRC | ECE | 解释 |
|---|---:|---:|---:|---|
| CCSH-Raw | 0.881560 | 0.850778 | 0.156402 | 原始视觉 tower 已有基础能力 |
| CCSH-SAMEQ | 0.895176 | 0.878471 | 0.109215 | SAMEQ 改善 readout |
| CCSH-SAMEQ-K4 | 0.788360 | 0.712013 | 0.186511 | K4 对 CCSH AUC 不稳定 |
| CCSH-HNMB | 0.835360 | 0.794997 | 0.199638 | HNMB 比 K4 好，但不如 SAMEQ/CEQ |
| CCSH-CEQ | 0.905976 | 0.885995 | 0.119305 | 最强 binary AUC |
| CCSH-VSL4 | 0.880824 | 0.860066 | 0.147463 | VSL4 可用，但非最强 |
| AUCH-CEQ-CCSH | 0.888968 | 0.900551 | 0.133392 | 最强 AUPRC |
| AUCH-CCSH-SAMEQ | 0.900720 | 0.893840 | 0.163020 | AUC 很强，但校准不如 AUCH-VSL4 |
| AUCH-VSL4 | 0.895976 | 0.870734 | 0.113416 | 最好 ECE |
| AUCH-SAMEQ | answerability AUC 0.547668 | answerability AUPRC 0.928481 | uncertainty F1 0.0 | 高 AUPRC 但不应解释为强 uncertainty classifier |

论文解释：

- CCSH-CEQ 是 readout family 的 locked finalist，因为 primary metric 是 CCSH binary AUC。
- AUCH-CEQ-CCSH 可以作为 AUPRC 最强补充结果。
- AUCH-VSL4 可以作为 calibration 最强补充结果。
- AUCH-SAMEQ 的 uncertainty F1 为 0.0，说明 answerability/uncertainty 仍有建模难度，不能过度包装。

## 4.5 Phase 5：Integrated VSL-CXR candidates

Phase 5 把各个模块组合成候选方法。

| Candidate | Composition | Status | 关键证据 |
|---|---|---|---|
| VSL-Lite | SAMEQ + global + none/LP | component_completed | SAMEQ backbone 完成，但 LP/CheXpert readout 不完整 |
| VSL-Core | SAMEQ-K4 + global + CCSH | component_completed | CCSH AUC 0.788360 |
| VSL-HNMB | SAMEQ-HNMB + global + CCSH | component_completed | CCSH AUC 0.835360 |
| VSL-CEQ | SAMEQ + CEQ + CCSH | component_completed | CCSH AUC 0.905976，当前 component strongest |
| VSL-Full | SAMEQ-HNMB + VSL-4class + CEQ + CCSH+AUCH | formal_training_completed | D9 mixed-instruction 正式训练完成；AUPRC evidence 0.900551 |
| VSL-Domain | VSL-Core + optional DRA + CCSH | blocked_external_data | 受 external data/label manifest 阻塞 |

论文解释：

- 如果只看 CCSH binary AUC，VSL-CEQ 是最强 component candidate。
- 如果看 integrated method 的完整性，VSL-Full 是最终方法候选，因为它真正完成了 D9 mixed-instruction training，并整合 SAMEQ-HNMB、VSL-4class、CEQ、CCSH+AUCH。
- VSL-Domain 不能写成失败，只能写成 external data boundary。

## 4.6 Phase 6：External validation

Phase 6 是目前最需要保守写作的部分。

主 external 状态：

| Dataset | 状态 | 说明 |
|---|---|---|
| VinDr-CXR | integration_in_progress | 官方 15,000 train + 3,000 test DICOM 与 bbox/image-level labels 已到位；七标签主协议和 manifests 已生成，正式推理 pending |
| PadChest | missing | 本地未找到 |
| MIMIC-CXR | exists_label_manifest_overlap_audit_pending | CheXpert/metadata/split `.csv.gz` 已存在；因训练重叠资格仍只作 conditional external |
| NIH | completed_appendix_stress | 只能作为 appendix/stress，不是主 external |

NIH appendix/stress 结果：

| Run | NIH macro-AUC | Macro-AUPRC | ECE | Brier | 解释 |
|---|---:|---:|---:|---:|---|
| Raw | 0.573709 | 0.148959 | 0.691411 | 0.691808 | raw baseline |
| SAMEQ | 0.593296 | 0.148653 | 0.545161 | 0.530681 | NIH macro-AUC 最好 |
| VSL-Core | 0.587227 | 0.154640 | 0.655611 | 0.626613 | NIH macro-AUPRC 最好 |
| VSL-CEQ backbone proxy | 0.574289 | 0.147348 | 0.745252 | 0.727351 | proxy row，不是 CEQ classifier |
| VSL-Full | 0.581517 | 0.141877 | 0.715381 | 0.690700 | CheXpert 最强但 NIH 不最强 |

论文解释：

- NIH stress 说明跨数据集泛化仍弱，尤其 calibration 很弱。
- SAMEQ 在 NIH macro-AUC 上最好，VSL-Core 在 NIH macro-AUPRC 上最好，这提示更复杂的 VSL-Full 并不天然带来外部迁移优势。
- VSL-Full 仍可作为 integrated finalist，但 external claim 必须写弱：当前不能声称 VSL-Full 在外部泛化上全面胜出。

建议写法：

> On an NIH appendix/stress transfer set, SAMEQ and VSL-Core show the strongest macro-AUC/macro-AUPRC respectively, while VSL-Full remains strongest on the in-domain CheXpert LP. This suggests that richer visual-sufficiency composition improves deployable in-domain representation but does not by itself solve cross-dataset calibration and label-mapping shift.

## 4.7 Phase 7：VLM teacher comparison

Phase 7 当前不是“模型家族完整对比完成”，而是 bounded audit 完成。

| Run | Status | 解释 |
|---|---|---|
| Qwen3VL-VSL-smoke | completed_by_current_main | 当前 Qwen3-VL VSL-Core 正式证据已超过 500-step smoke |
| Qwen3VL-VSL-full | completed_current_main_only | CheXpert AUC 0.698581，NIH appendix AUC 0.587227，CCSH AUC 0.788360 |
| InternVL-VSL-smoke/full | blocked_adapter_missing | 本地模型存在，但缺 InternVL-specific VSL trainer adapter |
| LLaVA/Mllama-VSL-smoke/full | blocked_adapter_missing | 本地模型存在，但缺 Llama-vision VSL trainer adapter |
| Qwen3.5 / Qwen-Coder scaffold | blocked_text_scaffold_trainer_missing | 需要 exact text-only VSL scaffold trainer |

论文解释：

- 当前不能写“Qwen3-VL 优于 InternVL/LLaVA”，因为后两者没有 exact adapter。
- 可以写成 method boundary：当前实现以 Qwen3-VL 为 teacher，跨 teacher family 的 exact comparison 留作后续。
- 这个边界比强行跑不等价实验更干净。

## 4.8 Phase 8：case study and visualization

Phase 8 已生成当前可用 casebook：

- 总计 33 rows；
- 覆盖 9 类：
  - VSL support；
  - VSL contradict；
  - VSL uncertain；
  - VSL insufficient；
  - SAMEQ pair；
  - false-hard-negative review；
  - CCSH success/failure；
  - CEQ attention；
  - external failure。

同时生成 7-row figure manifest：

| Figure | 当前状态 |
|---|---|
| Fig 1 VSL-CXR framework | figure_spec_ready |
| Fig 2 SAMEQ examples | casebook_ready |
| Fig 3 support vs contradict | casebook_ready |
| Fig 4 CEQ attention maps | attention_assets_available |
| Fig 5 CCSH consistency readout | metrics_and_cases_ready |
| Fig 6 external failure examples | appendix_stress_cases_ready |
| Fig 7 calibration curves | metric_table_ready_curve_points_missing |

论文解释：

- Casebook 是论文 qualitative section 的素材库。
- 这些 rows 还需要人工视觉复核，尤其是 false-hard-negative review 和 CEQ attention。
- calibration figure 目前只能从 ECE/Brier summary 讲，不应画成没有原始 bin points 的 curve。

## 4.9 Phase 9：locked final comparison

当前 locked finalists：

| Family | Finalist | 当前理由 |
|---|---|---|
| Raw | Raw Qwen3-VL vision LP | baseline；CheXpert macro-AUC 0.679003 |
| QA | Basic-QA | training-loss-only baseline |
| SAMEQ | SAMEQ | CheXpert LP 0.696113，NIH appendix macro-AUC 0.593296 |
| Hard Negative | SAMEQ-HNMB | lower training loss + stronger CCSH than SAMEQ-K4 |
| CEQ | CEQ-region | CEQ binary AUC 0.847173，ECE 0.082690 |
| CCSH | CCSH-CEQ | CCSH binary AUC 0.905976 |
| VSL Integrated | VSL-Full | CheXpert LP macro-AUC 0.714859，full-stack training complete |
| Teacher model | Qwen3-VL 2B | only current exact teacher stack |

最终 integrated decision：

> VSL-Full 是当前 integrated finalist，但这个结论应限定为 current single-seed, in-domain/deployable evidence finalist，而不是 external/multiseed winner。

更细的解释：

- VSL-Full 在 CheXpert LP 上最高：0.714859。
- VSL-Full 完成了 D9 mixed-instruction formal training。
- VSL-Full 有 CCSH+AUCH evidence，AUPRC 证据来自 AUCH-CEQ-CCSH 的 0.900551。
- 但 NIH appendix macro-AUC 不是 VSL-Full 最好，SAMEQ/VSL-Core 更强。
- 所以论文中应该把 VSL-Full 写成“综合方法 finalist”，而不是“所有指标绝对最优”。

## 5. 建议论文结构

### 5.1 Title 方向

可以考虑以下方向：

1. Visual Sufficiency Learning for Report-Guided Chest X-ray Representation
2. Learning Visually Sufficient Clinical Evidence from Radiology Reports
3. VSL-CXR: Report-Derived Visual Sufficiency Learning for Deployable Chest X-ray Encoders

我更推荐第 3 个：它保留了方法名，也把 deployable encoder 这个卖点放进去。

### 5.2 Abstract 该讲什么

摘要建议按以下逻辑：

1. Problem：report-derived supervision is rich but not always visually grounded.
2. Method：formulate CXR representation learning as visual sufficiency learning with support/contradict/uncertain/insufficient labels.
3. Modules：SAMEQ/hard negatives construct visual contrast; CEQ learns evidence-aware queries; CCSH/AUCH provide deployable readout and uncertainty.
4. Results：formal v5 suite shows VSL-Full strongest on CheXpert LP, CEQ/CCSH strongest for deployable consistency, NIH appendix reveals domain-shift/calibration limitations.
5. Claim：VSL-CXR provides a coherent route from report language to deployable visual evidence encoders, while external generalization remains an explicit limitation.

### 5.3 Introduction 该讲什么

Introduction 可以分成四段：

1. Radiology reports are tempting supervision, but report language is not always image-grounded.
2. Existing report-supervised VLM/QA approaches often conflate text plausibility with visual evidence.
3. We propose visual sufficiency learning: image-statement pairs are labeled as support, contradict, uncertain, or insufficient.
4. We instantiate VSL-CXR with SAMEQ, hard negatives, CEQ, CCSH, and AUCH, and evaluate with in-domain LP, deployable readout, external stress, teacher-family audit, casebooks, and locked final comparison.

### 5.4 Method 该讲什么

Method 部分建议结构：

1. Clinical statement construction：
   - D0 Basic-QA；
   - D1 CF-QA；
   - D2 SAMEQ；
   - D3 SAMEQ-CF；
   - D4 SAMEQ-K；
   - D5 SAMEQ-HNMB。
2. Visual sufficiency labels：
   - support；
   - contradict；
   - uncertain；
   - insufficient。
3. VSL training objective：
   - 2class / 3class / 4class；
   - balanced / field-balanced；
   - hierarchical variant。
4. Evidence-aware encoder：
   - CEQ；
   - CEQ-region as strongest variant。
5. Deployable readout：
   - CCSH；
   - AUCH。
6. Integrated candidate：
   - VSL-Full。

### 5.5 Experiments 该讲什么

实验部分建议按“问题 -> 表”的方式写：

1. Does visual-sufficiency supervision run and scale?
   - Phase 1/2 formal table。
2. Does evidence-aware querying help?
   - CEQ table。
3. Can we deploy without LLM?
   - CCSH/AUCH table。
4. Which integrated method should be locked?
   - Phase 5 + Phase 9 locked table。
5. Does it generalize externally?
   - NIH appendix/stress + explicit main external blocker。
6. Does teacher family matter?
   - Phase 7 bounded teacher audit。
7. What qualitative evidence supports the mechanism?
   - Phase 8 casebook/figures。

## 6. 当前最强 claim 与不能 claim 的东西

### 6.1 可以 claim

可以比较稳地写：

- VSL-CXR provides a coherent formulation for converting report-derived supervision into visual sufficiency learning.
- SAMEQ-style supervision improves deployable readout compared with raw or plain QA-style supervision.
- CEQ-region is the strongest evidence-aware CEQ variant in the current suite.
- CCSH-CEQ is the strongest deployable readout by binary AUC.
- VSL-Full is the current integrated finalist by in-domain CheXpert LP and full-stack completion.
- NIH appendix/stress shows meaningful but limited cross-dataset transfer and exposes calibration/domain-shift issues.
- The current implementation is Qwen3-VL based; cross-teacher comparison is blocked by adapter availability, not by negative results.

### 6.2 不应该 claim

不要写：

- VSL-Full 在主 external 上超过所有方法。
- VSL-CXR 已经完成多 seed statistical dominance。
- InternVL/LLaVA 比 Qwen3-VL 差。
- AUCH 已经解决 uncertainty classification。
- NIH 是主 external validation。
- 当前 casebook 不需要人工复核。
- calibration curve 已经完整绘制。

### 6.3 推荐最终主结论措辞

推荐这样写：

> VSL-CXR reframes report-derived supervision as visual sufficiency learning and yields a deployable CXR encoder/readout pipeline. In the current v5 experiment suite, CEQ improves evidence-aware consistency readout, CCSH-CEQ achieves the strongest deployable binary consistency AUC, and VSL-Full is selected as the integrated finalist based on completed full-stack training and the strongest CheXpert LP result. However, NIH appendix transfer reveals remaining domain-shift and calibration weaknesses, and main-external/multi-seed validation remains a necessary next step.

## 7. 结果摘要表

### 7.1 最关键数字

| Topic | Best / selected row | Metric |
|---|---|---:|
| Raw baseline | Raw Qwen3-VL LP | CheXpert macro-AUC 0.679003 |
| SAMEQ core | SAMEQ | CheXpert LP macro-AUC 0.696113 |
| Hard negative | SAMEQ-HNMB | best val loss 0.095854；CCSH AUC 0.835360 |
| CEQ | CEQ-region | binary AUC 0.847173；ECE 0.082690 |
| CCSH readout | CCSH-CEQ | binary AUC 0.905976 |
| AUCH/readout AUPRC | AUCH-CEQ-CCSH | AUPRC 0.900551 |
| Calibration among CCSH/AUCH | AUCH-VSL4 | ECE 0.113416 |
| Integrated finalist | VSL-Full | CheXpert LP macro-AUC 0.714859 |
| NIH appendix AUC | SAMEQ | macro-AUC 0.593296 |
| NIH appendix AUPRC | VSL-Core | macro-AUPRC 0.154640 |
| Teacher finalist | Qwen3-VL 2B | only current exact v5 teacher stack |

### 7.2 最终 locked finalists

| Family | Finalist | Status |
|---|---|---|
| Raw | Raw Qwen3-VL vision LP | locked_single_seed |
| QA | Basic-QA | locked_training_loss_only |
| SAMEQ | SAMEQ | locked_single_seed |
| Hard Negative | SAMEQ-HNMB | locked_single_seed |
| CEQ | CEQ-region | locked_single_seed |
| CCSH | CCSH-CEQ | locked_single_seed |
| VSL Integrated | VSL-Full | locked_single_seed_with_external_boundary |
| Teacher model | Qwen3-VL 2B | locked_current_main_bounded |

## 8. 写论文时的图表安排

建议图表如下：

1. Figure 1：VSL-CXR framework。
2. Figure 2：SAMEQ examples。
3. Figure 3：support / contradict / uncertain / insufficient examples。
4. Figure 4：CEQ attention examples。
5. Figure 5：CCSH deployable readout diagram + CCSH table。
6. Figure 6：NIH appendix external failure examples。
7. Figure 7：calibration summary；如果后续导出 bin points，再画正式 calibration curve。

建议主表：

1. Table 1：Phase 1/2 formal VSL training results。
2. Table 2：CEQ variants。
3. Table 3：CCSH/AUCH deployable readout。
4. Table 4：Integrated candidates。
5. Table 5：External appendix/stress results。
6. Table 6：Locked final comparison。

## 9. 当前最需要补强的下一步

如果要把这篇推到更强论文形态，最值得继续补的是：

1. 主 external：
   - 完成 VinDr-CXR 解压、CRC/SHA 抽样与 DICOM 解码验收；
   - 完成 Raw、SAMEQ、VSL-Core、VSL-CEQ、VSL-Full 的官方 test-3k 推理；
   - 以七个直接可比且 test 非退化的标签报告主 macro-AUC，Edema 因 test 阳性为 0 仅保留在完整映射中。
2. 多 seed：
   - 至少给 Raw、SAMEQ、VSL-Core、VSL-CEQ、VSL-Full 做 3 seed。
3. Calibration bins：
   - 保存 per-sample probabilities；
   - 导出 calibration curve points。
4. Teacher adapter：
   - 实现 InternVL 或 Mllama 的最小 VSL trainer adapter。
5. Manual visual review：
   - 审 Phase 8 casebook；
   - 填 D6/D9 manual audit template；
   - 特别审 false-hard-negative review rows。

## 10. 最终可用摘要

当前最合理的论文故事是：

> VSL-CXR is not merely another report-QA training recipe. It is a reframing of report supervision around whether an image provides sufficient visual evidence for a clinical statement. The experiments show that SAMEQ-style visual contrast, CEQ evidence queries, and CCSH deployable readouts form a coherent pipeline. The strongest deployable readout is CCSH-CEQ, and the integrated finalist is VSL-Full under current single-seed evidence. At the same time, NIH appendix transfer and calibration results reveal that cross-dataset generalization remains limited, making main-external and multi-seed validation the most important next step.

这条故事线的优点是诚实、集中、有机制解释，也能把当前已有的大量实验结果组织成一个论文可读的主线。
