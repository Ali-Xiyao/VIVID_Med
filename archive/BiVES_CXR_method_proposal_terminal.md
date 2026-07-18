# BiVES-CXR：面向胸片临床陈述验证的双极干预式视觉证据集学习

> **英文暂定题目**  
> **From Report Truth to Image Evidence: Bipolar Interventional Visual Evidence Set Learning for Chest Radiographs**
>
> **方法简称（暂定）**：**BiVES-CXR**  
> Bipolar Interventional Visual Evidence Set Learning for Chest Radiographs
>
> **目标期刊**：Medical Image Analysis（优先）或 IEEE Transactions on Medical Imaging  
> **文档状态**：可执行研究 proposal，版本 1.0，2026-07-16  
> **重要说明**：`BiVES-CXR` 是暂定名称；正式投稿前必须再做一次截至投稿日的系统检索、代码仓库检索和名称冲突检查。

> **2026-07-17 执行修订（当前生效）**：由于项目不存在且未来也无法获得合格的临床双盲审阅者，专家 U/I 审阅与显式正负人工审计不再作为工程实验启动门。当前 P0 改为可复现的弱标签代理实验：S/C/U 仅来自冻结规则解析，I 仅来自带来源哈希的合成证据移除；所有结果必须标记为 `weak_proxy_unreviewed` 和 `formal_result=false`。该修订允许验证管线可学习性与 BiVES 机制，但不能支持专家一致性、临床 U/I 有效性或专家审定测试集相关声明。文中其余专家审阅设计保留为理想投稿证据和未满足限制，不得报告为已完成。

> **2026-07-19 终局修订（当前最高优先级）**：冻结的 C6I 已排除输入坐标、控制区域连通性和执行错误，并在独立 MS-CXR publisher-test 阳性样本上得到有效的 `FAIL_FINAL_STOP`。因此，BiVES B2 的“定位证据集同时具备局部必要性/充分性”假设不再作为已成立的方法主张，也不再创建 C6J、调整干预算子或扩大到 Qwen3.5-4B/9B。当前可辩护的论文方向改为 **localization–causality audit**：系统审计定位质量、目标/对照区域干预和模型支持分数之间何时一致、何时失配。下文方法设计保留为被检验的冻结假设和审计对象，而不是成功结论。

---

## 0. 最终决策

### 0.1 对现有 VSL-CXR 方案的判断

现有方案具有一个好的科学问题：

> 报告中的语义真实性，不等于当前影像提供了足够的视觉证据。

但当前方法形态仍然是：

- SAMEQ / hard negative：数据采样与对比监督；
- VSL-4class：任务与标签定义；
- CEQ：区域查询或注意力模块；
- CCSH：一致性读出头；
- AUCH：回答性与不确定性读出头。

这些组成部分可以形成一个连贯系统，却还没有形成一个足够集中的、可用一句数学定义概括的方法创新。若以“VSL-Full = SAMEQ-HNMB + VSL-4class + CEQ + CCSH + AUCH”直接投稿，审稿人很容易将其评价为：

> 一个合理的新任务，配合若干已有思想的工程组合与增量改进。

因此，本 proposal 不再把现有模块堆叠作为最终方法，而是重新定义一个单一技术对象：

> **陈述条件下的双极视觉证据集（bipolar visual evidence set）。**

模型不直接接一个普通四分类头，而是学习：

1. 哪些图像区域提供支持该陈述的证据；
2. 哪些区域提供反驳该陈述的证据；
3. 这些证据是否足量；
4. 正反证据是否冲突；
5. 选出的证据区域是否在干预意义上既“充分”又“必要”。

由此，四种状态不再是人为并列的四个 softmax 类别，而是由同一个证据空间自然导出：

- **support**：证据充足，且支持证据占优；
- **contradict**：证据充足，且反驳证据占优；
- **uncertain**：存在相关证据，但正反证据接近或视觉表现本身模糊；
- **insufficient**：总视觉证据不足，当前影像无法作出合法判断。

### 0.2 原方法假设（已被 C6I 终局否定，不再作为主 claim）

> **A clinical statement should be verified from a K-budgeted visual evidence set that is sufficient when retained, necessary when removed, and invariant to matched perturbations of irrelevant context.**

中文表述：

> 对一个临床陈述的判断，应当来自一个固定 K 预算的视觉证据集：保留该证据时结论应保持，删除该证据时模型应转为“证据不足”，对匹配的无关区域进行干预时结论应稳定。

这一个定义统一取代原来的 CEQ、CCSH 和 AUCH：

- 不再用 CEQ 作为独立创新模块，而是直接学习空间证据场；
- 不再用 CCSH 接一个额外一致性分类头，而是从证据极性直接读出结论；
- 不再用 AUCH 接一个额外回答性头，而是用总证据量表示 insufficiency、用正反证据冲突表示 uncertainty；
- SAMEQ 仅作为构造“同一陈述、不同影像状态”训练组的采样策略；
- hard-negative mining 仅作为训练数据策略，不列为方法贡献。

### 0.2A 当前论文主命题：定位与因果证据并不等价

> **A localized evidence set can align with expert regions yet fail to be
> causally necessary or specifically sufficient under matched interventions;
> localization and causal evidence therefore require separate validation.**

中文表述：

> 模型选出的区域可以与专家标注有正向定位重合，但仍未必在匹配干预下具有特异的因果必要性或充分性；因此，医学视觉模型的定位质量与因果证据有效性必须分开验证。

当前证据链不把单次负结果包装成新方法成功，而是冻结以下可复核事实：

1. C4 在 VinDr-train protocol-design positives 上通过内部机制门；
2. C5 在 image-disjoint confirmation 上复现机制门，但 polarity gate 因 Consolidation AUPRC 低于 B0 而终止；
3. C6I 在独立 MS-CXR publisher test 上修复实际输入几何后，Pleural Effusion 的 masked blur 通过，而 Consolidation 的同一算子呈稳定负 TCIG；
4. top-K 定位增益在两个 finding 上均为正，但不能替代 target-vs-control 因果门；
5. 终局只读审计显示算子像素强度差异只能解释部分 local-mean 现象，不能解释 Consolidation blur 的负 TCIG，因此不再以调参方式“修复”结果。

该主线允许讨论的贡献是：一个严格冻结、成对 target/control、跨阶段并包含独立数据的定位—因果失配审计框架。它不允许声称 BiVES B2 已学到临床有效的因果证据，不允许把 post-stop 关联分析解释为因果证明，也不允许从 C6I 选择新阈值或新算子。

### 0.3 论文是否达到 MIA/TMI 水平的必要条件

本 proposal 只有在下列证据同时成立时，才应以 MIA/TMI 为目标：

1. BiVES-CXR 在同一 backbone、同一数据和同一训练预算下，显著优于 flat 4-class、VECL 类方法和 CORAL/CGO 类整图交换方法；
2. 提升不仅体现在普通分类指标，还体现在目标区域删除、证据保留、无关区域扰动和同题异图响应上；
3. 若要声称临床 U/I 有效性，uncertain 与 insufficient 仍需人工审阅并达到可接受一致性；当前无审阅者，因此代理 P0 不满足这一投稿条件；
4. 至少完成一个真正的主外部验证，最好再加一个外部 stress test；
5. 关键方法完成至少 3 个 seed、患者级置信区间和配对统计检验；
6. 所有主要结论来自同一个锁定 checkpoint，而不是从多个 head 或 run 中分别挑最好结果；
7. 在一个非 VIVID-Med 初始化的 backbone 上仍然成立，证明新方法不是旧工作的附属模块。

若这些条件没有满足，应继续作为方法研究推进，而不要靠重新组织叙事强行投稿顶刊。

---

## 1. 与 VIVID-Med 的关系

### 1.1 不存在“因为同一作者就没关系”的逻辑

VIVID-Med 是你们自己的预印本，这本身不构成不能投稿新工作的障碍。真正需要处理的是：

- 是否重复使用了旧论文已经声称的核心创新；
- 是否重复使用大面积实验、图表、文本或结果而未披露；
- 新论文是否只是旧框架加几个 head；
- 是否清楚引用并解释两篇工作的边界。

### 1.2 两篇论文必须采用以下科学边界

| 工作 | 科学问题 | 核心方法 | 部署对象 |
|---|---|---|---|
| VIVID-Med | 如何把报告中的结构化医学语义蒸馏到视觉编码器 | LLM structured teacher、UMS、answerability-aware masking、SPD query decomposition | 可部署 ViT backbone |
| BiVES-CXR | 如何识别当前患者影像中对一个陈述真正必要且充分的视觉证据，并区分视觉冲突与证据缺失 | 双极证据场、证据状态解码、干预式证据集闭包 | image-statement verifier / evidence-aware encoder |

### 1.3 新论文不能再次作为新贡献声称的内容

以下内容应引用 VIVID-Med，而不能重新包装成 BiVES-CXR 的新贡献：

- LLM 作为结构化语义 teacher；
- Unified Medical Schema；
- answerability-aware masking；
- Structured Prediction Decomposition；
- 训练后丢弃 LLM、部署视觉编码器这一总体卖点。

### 1.4 可合法复用但必须披露的内容

可以复用：

- 数据清洗代码；
- 报告解析基础设施；
- 视觉 backbone 或初始化权重；
- 训练框架；
- 患者级划分代码；
- 部分公开数据 manifest。

但需要：

1. 在 Methods 或 Supplement 中明确说明复用项；
2. 主实验同时包含一个通用/公开 backbone；
3. 把 VIVID-Med 初始化作为 secondary ablation，而非 BiVES-CXR 成立的必要条件；
4. 不能重复使用旧测试集做反复模型选择；
5. 若复用相同患者集合，必须重新锁定 split，并说明旧论文是否曾访问该测试集。

### 1.5 推荐写法

> VIVID-Med studies structured semantic distillation into a deployable visual backbone. In contrast, the present work investigates a different question: whether a patient-specific clinical statement is supported by a spatially localized visual evidence set that is necessary and sufficient under controlled interventions. We use VIVID-Med only as one optional initialization and demonstrate the method with an independent backbone.

---

## 2. 最近邻工作与 novelty 防线

### 2.1 不能再把这些内容单独当作核心创新

| 已有方向 | 与旧方案的重合 | 对新 proposal 的要求 |
|---|---|---|
| VECL，MICCAI 2025 | entailment / neutral / contradiction；正负 mention 对比 | “三类变四类”不够；必须提供新的证据表示和干预定义 |
| CORAL/CGO，arXiv 2026-07 | hard-negative image swap；惩罚答案对图像交换不敏感 | SAMEQ/HNMB 不能作为主创新；需要从整图依赖推进到局部证据必要性/充分性 |
| Localize-before-Answer，IJCAI 2025 | 先定位病灶再回答；区域监督 | CEQ/attention localization 本身不够；必须验证区域在因果干预下是否必要与充分 |
| Phrase-grounded fact-checking，MICCAI 2025 | 合成 finding/location 错误，做真伪预测与定位 | 不能只把任务包装成报告事实核查；必须强调四状态证据语义与局部必要/充分闭包 |
| CXR image-use causal audit，2026 | original / image swap / target mask / irrelevant mask | 不能只报告 accuracy 或 attention；必须把这些干预变成主要评价，最好进入训练目标 |
| ShoViR，2026 | target-region 与 co-occurring-region occlusion | 需要证明方法不仅诊断 shortcut，而且主动减少 shortcut |
| SwapMix，CVPR 2022 | 交换无关上下文特征，抑制视觉上下文捷径 | context swap 本身不新；它只能作为证据集干预闭包的一部分 |
| Evidential deep learning / subjective logic | 以证据量表示不确定性 | 不能声称“evidence mass”本身新；创新应是 statement-conditioned bipolar spatial evidence 与局部干预闭包的联合形式 |
| Rationale / sufficient-comprehensive explanations | 选择最小解释片段，评价 sufficiency/comprehensiveness | 必须体现医学视觉陈述验证、双极证据、四状态语义和临床区域干预的特异性 |

### 2.2 暂定 novelty hypothesis

截至本 proposal 制定时，最值得验证的 novelty hypothesis 是：

> 现有医学视觉蕴含、VQA grounding、整图 counterfactual training 和不确定性方法，尚未把**支持/反驳双极空间证据**、**vacuity–dissonance 分解**和**局部证据必要性/充分性干预**统一为胸片临床陈述验证方法。

这只是“待投稿时再次核验”的 novelty hypothesis，不应在摘要中直接使用 “first” 或 “the first”。

### 2.3 与最近邻的预期差异

| 维度 | VECL | CORAL/CGO | LobA | Phrase-grounded fact-checking | BiVES-CXR |
|---|---|---|---|---|---|
| 判断单位 | image-text mention relation | VQA answer under whole-image swap | localized VQA | report finding/location truth | image–clinical statement |
| 证据表示 | ternary relation matrix | whole-image dependence | localization mask | cross-modal veracity and localization | positive/negative spatial evidence masses |
| uncertain vs insufficient | 未显式分离 | 未显式分离 | 未显式分离 | 未显式分离 | 用冲突度与总证据量分离 |
| 干预粒度 | 观察式 contrast | 整图 hard negative | 先定位后回答 | 合成 finding/location 文本错误 | evidence-only、evidence-deleted、irrelevant-deleted、matched context |
| 必要性 | 无 | 整图层面 | 不保证 | 不保证 | 删除证据后应转为 insufficient |
| 充分性 | 无 | 无 | 不保证 | 不保证 | 只保留证据仍保持原状态 |
| 无关上下文稳定性 | 无 | 部分 | 无 | 无 | 显式约束与评价 |
| 最终输出 | alignment / downstream | VQA answer | VQA answer | veracity score + localization | 4-state evidence semantics + map |

---

## 3. 论文核心任务

### 3.1 输入与输出

给定：

- 胸片图像：\(x\)；
- 一个原子化临床陈述：\(s\)。

输出：

1. 状态 \(y\)：
   \[
   y\in\{S,C,U,I\}
   \]
   其中 \(S\)=support，\(C\)=contradict，\(U\)=uncertain，\(I\)=insufficient；
2. 支持证据图 \(M^+\)；
3. 反驳证据图 \(M^-\)；
4. 总证据量、证据冲突度和校准置信度；
5. 可选的 evidence-only 图像区域或 patch 集合。

### 3.2 原子临床陈述

建议将自然语言报告先转成一个原子 proposition：

\[
s=(f,l,r,se,v)
\]

其中：

- \(f\)：finding；
- \(l\)：location；
- \(r\)：laterality；
- \(se\)：severity；
- \(v\)：view / evidence requirement。

推荐训练时统一使用非否定的 canonical proposition，例如：

- “A right pleural effusion is present.”
- “The heart is enlarged.”
- “The endotracheal tube tip is appropriately positioned.”

而不要同时使用大量 “有/无”两种句式。正负状态由影像决定，减少语言模板泄漏。

### 3.3 四种状态的严格定义

#### Support（S）

当前影像覆盖目标解剖区域，质量足以判断，并存在清晰视觉证据支持陈述。

#### Contradict（C）

当前影像覆盖目标解剖区域，质量足以判断，并存在足够证据表明该陈述不成立。

注意：

- “报告未提到”不能自动视为 contradict；
- 应优先使用显式否定、可靠结构化标签或人工确认；
- 对于设备位置、左右侧和严重程度，必须确认反例确实排除了原陈述。

#### Uncertain（U）

目标区域可见，问题与当前影像有关，但影像表现边界性、模糊、相互冲突或不足以形成支持/反驳中的任一明确极性。

这是**视觉层面的模糊性/冲突**，不是模型自己“不自信”。

#### Insufficient（I）

当前影像缺少作出判断所需的视觉信息。典型原因包括：

- 目标区域未覆盖或被遮挡；
- 图像质量严重不足；
- 所需 view 缺失；
- 陈述要求既往片或多时点信息；
- 陈述依赖非影像临床信息；
- 该模态本身不能合法回答。

### 3.4 uncertain 与 insufficient 的操作区别

| 问题 | Uncertain | Insufficient |
|---|---|---|
| 相关解剖区域是否可见 | 是 | 可能否 |
| 是否存在与陈述相关的视觉信号 | 有，但模糊或冲突 | 不足或不存在 |
| 再提高阅片能力能否判断 | 仍可能存在真实模糊性 | 需要更多/不同信息 |
| 数学表征 | 总证据量不低，但正反证据接近 | 总证据量低 |
| 临床例子 | 轻微基底阴影，难定肺不张或浸润 | “较昨日改善”但无 prior；肺尖未完整覆盖 |

---

## 4. 方法：BiVES-CXR

## 4.1 总体结构

BiVES-CXR 只包含一个核心建模对象和一个由定义导出的训练目标：

1. **Bipolar visual evidence field**：每个 patch 对陈述产生支持证据和反驳证据；
2. **Evidence-state decoder**：根据总证据量、证据方向和冲突度得到四种状态；
3. **Interventional evidence-set closure**：通过保留、删除和对照扰动，使证据集满足充分性、必要性和无关上下文稳定性。

不存在额外的 CCSH、AUCH 或普通四分类 head 作为最终读出。

---

## 4.2 图像与陈述编码

视觉编码器输出 patch tokens：

\[
Z=f_\theta(x)=\{z_p\}_{p=1}^{P},\qquad z_p\in\mathbb{R}^{d}.
\]

主实验的临床陈述表示是冻结的 canonical statement prototype：对 Qwen3.5
input-token embedding lookup 做 mean pooling 后再 L2 normalization；这不是
完整 contextual language encoder 的输出。扩展实验才可使用另行锁定的冻结
clinical text encoder：

\[
h=g_\phi(s)\in\mathbb{R}^{d}.
\]

推荐两种实现模式：

### 模式 A：canonical ontology（主实验）

- 每个 finding/location/laterality 组合有固定 canonical statement；
- 陈述 embedding 可离线缓存；
- 最大限度控制语言模板；
- 适合严谨验证方法机制。

### 模式 B：open paraphrase（泛化实验）

- 使用冻结 clinical text encoder；
- 对同一 proposition 生成未见 paraphrase；
- 只作为鲁棒性/扩展实验，不应干扰主实验。

对每个 patch 构造多模态融合特征：

\[
\psi_p = \operatorname{MLP}\big[z_p;h;z_p\odot h;|z_p-h|\big].
\]

---

## 4.3 双极空间证据场

每个 patch 分别产生支持和反驳证据：

\[
e_p^{+}=\operatorname{softplus}(w_+^\top\psi_p),
\]

\[
e_p^{-}=\operatorname{softplus}(w_-^\top\psi_p).
\]

其中：

- \(e_p^+\)：该 patch 对陈述成立提供的证据质量；
- \(e_p^-\)：该 patch 对陈述不成立提供的证据质量。

证据选择 gate：

\[
m_p=\operatorname{HardConcrete}(a_p),\qquad m_p\in[0,1].
\]

初期实现可先使用可微 soft top-\(K\) 或固定 top-\(K\)；最终模型建议使用 hard-concrete / \(L_0\) gate，使证据区域可自适应稀疏。

聚合证据：

\[
E^+=\frac{\sum_{p}m_pe_p^+}{\sum_pm_p+\epsilon},
\qquad
E^-=\frac{\sum_{p}m_pe_p^-}{\sum_pm_p+\epsilon}.
\]

为避免证据尺度任意增大，建议：

- 对 \(e_p^\pm\) 使用有界参数化，或；
- 增加 evidence magnitude regularization，或；
- 对 \(E^\pm\) 做温度标定；
- 在独立 calibration split 上锁定温度参数。

证据图为：

\[
M_p^+=m_pe_p^+,
\qquad
M_p^-=m_pe_p^-.
\]

总证据图：

\[
M_p=M_p^++M_p^-.
\]

---

## 4.4 从证据量直接得到四种状态

定义总证据量：

\[
T=E^++E^-.
\]

定义证据可用性：

\[
A=1-\exp(-T/\tau_A).
\]

- \(A\approx0\)：几乎没有足够视觉证据；
- \(A\approx1\)：存在足量视觉证据。

定义有符号证据差：

\[
\Delta=E^+-E^-.
\]

在 evidence available 的条件下，使用单调双极条件解码器：

\[
(\pi_S,\pi_C,\pi_U)
=\operatorname{softmax}\left(
\frac{\Delta}{2\tau_P},
-\frac{\Delta}{2\tau_P},
\log(2m_U)
\right),
\]

其中 \(m_U>0\) 是 uncertain mass。它控制零极性附近的模糊质量，
而不改变 support/contradict 的方向单调性。

四类概率：

\[
p_I=1-A,
\]

\[
p_U=A\pi_U,
\]

\[
p_S=A\pi_S,
\]

\[
p_C=A\pi_C.
\]

它们满足：

\[
p_S+p_C+p_U+p_I=1.
\]

这一解码具有明确语义：

- **insufficient = low evidence availability**；
- **uncertain = available but near-zero/competing signed evidence**；
- **support / contradict = available evidence with opposite signed polarity**。

在固定总证据量 \(T\) 时，\(p_S\) 对 \(\Delta\) 严格单调递增，
\(p_C\) 对 \(\Delta\) 严格单调递减，\(p_U\) 关于 \(\Delta=0\)
对称且在零点最大。因此 support 的 NLL 不再在错误的负半轴产生驻点。
旧的 \(|\Delta|\)-exponential decoder 仅保留为历史消融，不进入 active
配置、校准或正式发布链。

主要观察损失：

\[
\mathcal L_{state}=-\log p_y.
\]

### 为什么不再使用 flat 4-class head

普通 softmax 可以学会四类边界，却不保证：

- uncertain 来自视觉冲突；
- insufficient 来自证据缺失；
- support 与 contradict 共享同一个极性轴；
- 输出与空间证据图有任何一致关系。

BiVES 解码器把四类语义绑定到可解释且可干预的证据变量上。

---

## 4.5 证据集干预

令 \(M=\{m_p\}\) 为模型选择的 evidence set。训练中在**特征空间**构造干预，减少像素遮挡产生的分布外伪影；测试中使用未参与训练的像素级干预检验泛化。

设 \(z_{mask}\) 为可学习的 mask token。

### 4.5.1 Evidence-retained intervention：充分性

仅保留证据区域：

\[
z_p^{keep}=m_pz_p+(1-m_p)z_{mask}.
\]

对于 \(y\in\{S,C,U\}\)，只保留证据后应保持原状态：

\[
\mathcal L_{suf}= -\log p_y(x^{keep},s).
\]

这要求选出的区域足以支持原判断。

### 4.5.2 Evidence-deleted intervention：必要性

删除证据区域：

\[
z_p^{drop}=(1-m_p)z_p+m_pz_{mask}.
\]

对于原本可回答的样本，删除全部证据后应转为 insufficient：

\[
\mathcal L_{nec}= -\log p_I(x^{drop},s).
\]

这要求模型所选证据对判断是必要的，而不仅是一张好看的 attention map。

### 4.5.3 Irrelevant-region control：特异性

在证据集合之外，选择与 \(M\) 面积相同、形状或空间分布匹配的对照区域 \(R\)：

\[
R\cap M=\varnothing,
\qquad |R|\approx|M|.
\]

删除对照区域：

\[
z_p^{ctrl}=(1-r_p)z_p+r_pz_{mask}.
\]

输出应保持：

\[
\mathcal L_{ctrl}=
D_{JS}\big(p(\cdot|x,s),p(\cdot|x^{ctrl},s)\big).
\]

这可防止模型对任意遮挡都不稳定，并验证目标区域效应具有特异性。

### 4.5.4 Matched-context intervention：可选增强

从同一 canonical statement 的另一患者中取 context donor，并匹配：

- AP/PA/view；
- 性别、年龄段；
- 设备负担；
- 图像质量；
- 数据来源；
- 目标解剖区域。

用 donor 的非证据区域替换当前图像的非证据区域，并要求标签跟随保留的 evidence set。

由于 context feature swap 已有先例，此项只作为增强和 ablation，不单独声称创新。若重组图像产生明显伪影，应删除该训练项，而保留 keep/drop/control 三类干预作为主方法。

---

## 4.6 同一陈述、不同影像状态的成组训练

对同一个 canonical statement \(s\)，构造：

\[
(x^S,s),\quad(x^C,s),\quad(x^U,s),\quad(x^I,s).
\]

语言输入完全相同，仅影像证据状态变化。

定义归一化证据极性：

\[
\rho(x,s)=\frac{E^+-E^-}{E^++E^-+\epsilon}.
\]

support–contradict pairwise ranking：

\[
\mathcal L_{pair}=
\max\big(0,\gamma-\rho(x^S,s)+\rho(x^C,s)\big).
\]

对于 uncertain：

\[
\mathcal L_{U-pol}=|\rho(x^U,s)|,
\]

并通过 \(\mathcal L_{state}\) 保证其总证据不低。

对于 insufficient：

\[
\mathcal L_{I-mag}=T(x^I,s),
\]

鼓励原始 insufficient 样本具有低证据量。

注意：SAMEQ 在新论文中仅作为这种成组采样的实现历史或 baseline 名称，不再作为独立方法贡献。

---

## 4.7 固定 K 预算与空间连续性

当前主实现使用 fixed exact-K selector。它学习“在给定 K 预算下选哪些
patch”，并不学习每个样本的自适应集合大小。因此主结果只声称
**K-budgeted evidence set**，不声称已证明最小基数。

固定 K 配置令 \(\lambda_{min}=0\)，并预注册
\(K\in\{4,8,16,32\}\) 的敏感性分析。自适应 hard-concrete/\(L_0\)
最小性作为后续扩展，不进入当前主方法。

对局部病灶可加入 total variation：

\[
\mathcal L_{TV}=\sum_{(p,q)\in\mathcal N}|m_p-m_q|.
\]

对弥漫性病变，过强 TV 或稀疏约束可能有害，因此应：

- 按 finding 调整稀疏先验，或；
- 使用 anatomy-level 连续区域，而不是固定小框；
- 在消融中报告局灶与弥漫 finding 的差异。

---

## 4.8 统一训练目标

把三类干预合并成一个证据集闭包损失：

\[
\mathcal L_{IES}
=
\mathcal L_{suf}
+\lambda_n\mathcal L_{nec}
+\lambda_c\mathcal L_{ctrl}.
\]

总损失：

\[
\mathcal L
=
\mathcal L_{state}
+\lambda_{ies}\mathcal L_{IES}
+\lambda_p\mathcal L_{pair}
+\lambda_i\mathcal L_{I-mag}
+\lambda_{tv}\mathcal L_{TV}.
\]

论文表述中不要把它写成六个独立模块，而应解释为：

- \(\mathcal L_{state}\)：定义证据状态；
- \(\mathcal L_{IES}\)：使证据集满足必要性、充分性和干预特异性；
- 其余项是估计该证据集所需的成组识别与正则化。

---

## 4.9 推理流程

推理时：

1. 编码图像和 statement；
2. 生成 \(M^+\) 与 \(M^-\)；
3. 聚合 \(E^+\) 与 \(E^-\)；
4. 由闭式 evidence-state decoder 输出 \(p_S,p_C,p_U,p_I\)；
5. 可选择输出 evidence-only crop、证据图和拒答决定。

不需要：

- autoregressive LLM；
- 生成式 chain-of-thought；
- 单独 CCSH；
- 单独 AUCH；
- 多阶段 agent。

对于固定 ontology，statement embeddings 可缓存。

当前主实现缓存的是：frozen canonical statement prototypes obtained by mean
pooling Qwen3.5 input-token embeddings；不得表述为 contextual Qwen language-
encoder embeddings。

---

## 4.10 可写入论文的形式化命题

### 命题 1：同陈述配对排除纯文本解

若同一个 statement \(s\) 同时存在 support 和 contradict 样本，且模型的极性分数仅依赖 \(s\)，则：

\[
\rho(x^S,s)=\rho(x^C,s),
\]

因而任何正 margin 的 \(\mathcal L_{pair}\) 均不能达到零。也就是说，纯 statement-only predictor 无法满足成组极性约束。

### 命题 2：局部干预闭包下的上下文不变性

设图像特征可写为选定证据 \(E\) 与上下文 \(C\)。若在一个给定干预族中：

1. 对所有匹配上下文 \(C'\)，\(\mathcal L_{ctrl}=0\)；
2. 删除 \(E\) 后 \(\mathcal L_{nec}=0\)；
3. 仅保留 \(E\) 后 \(\mathcal L_{suf}=0\)；

则模型在该干预族支持集内满足：

- 预测对上下文替换不变；
- 无证据输入不能维持 support/contradict/uncertain；
- 选定证据单独足以恢复原状态。

这不是对现实世界因果可识别性的无限制证明。论文中必须限定为：

> an operational certificate over the specified intervention family.

完整假设和 proof sketch 放入 Supplement。

---

## 5. 数据引擎

## 5.1 数据角色规划

### 训练主集

优先使用：

- MIMIC-CXR-JPG 图像与报告；
- 可选的 Chest ImaGenome 解剖区域/scene graph，用于 anatomy prior、分析或弱监督；
- 当前已有 VSL 数据引擎作为初始候选生成器，但全部关键标签需要重新审计。

### 内部专家测试集

建立新的 **BiVES-CXR Audit Set**：

- 患者级独立；
- 不参与任何训练或阈值选择；
- 四状态平衡或近似平衡；
- 同一 canonical statement 覆盖多个状态；
- 包含局灶、弥漫、设备和质量不足案例；
- 部分样本有人工证据区域。

### 主外部集

优先路线：

1. PadChest / PadChest-GR：报告、外部机构与可用 grounding 资源；
2. VinDr-CXR：放射科医师标签与 bounding boxes，适合 support/contradict 和 grounding；
3. CheXpert：外部分类、线性探测或 statement verification；
4. MS-CXR：短语–区域对应与人工验证 grounding；
5. NIH ChestX-ray14：仅作为额外 stress test，不再承担主 external claim。

实际选择必须根据数据许可、标签可用性和本地 manifest 完整程度锁定。

---

## 5.2 finding 范围

### 第一阶段核心 finding

优先选择视觉定义相对明确、样本量足够、区域可评价的 6–8 类：

- pleural effusion；
- pneumothorax；
- cardiomegaly；
- pulmonary edema；
- atelectasis；
- focal air-space opacity / consolidation；
- support-device position；
- 另一个经审计后具有可靠正负样本和外部标签的 finding。

### 不建议作为首批主任务的内容

- 依赖临床信息的 pneumonia；
- 没有 prior 时的 interval change；
- 仅凭胸片难以合法排除的疾病；
- 标签极度稀疏或跨数据集定义不一致的 finding；
- “报告未提及即为阴性”的弱标签。

最终 finding 清单必须由以下标准决定：

1. explicit positive/negative 可获得；
2. uncertain 和 insufficient 能被审阅者理解并稳定标注；
3. 至少一个外部数据源有可映射标签；
4. 有足够同陈述跨状态样本；
5. 目标区域或 anatomy prior 可定义。

---

## 5.3 报告解析与候选生成

每条报告解析为原子结构：

```text
finding
polarity
uncertainty
laterality
anatomical_location
severity
comparison_requirement
view_requirement
quality_limitation
```

推荐流程：

1. RadGraph/CheXpert 类规则或现有 parser 提取基础实体；
2. 规则归一化成 canonical proposition；
3. LLM 只用于离线歧义复核或补全，不参与测试推理；
4. 冻结 prompt、模型版本和解析规则；
5. 对每种标签来源保存 provenance；
6. 自动冲突检测；
7. 高风险类别进入人工审阅。

每一条训练样本保存：

```yaml
patient_id:
study_id:
image_id:
canonical_statement_id:
statement_text:
state_label:
state_source:
label_confidence:
explicit_positive_or_negative:
uncertainty_cue:
insufficiency_reason:
view:
quality_flags:
region_source:
parser_version:
```

---

## 5.4 四类样本来源

### Support

可接受来源：

- 报告显式阳性；
- 结构化标签阳性且与报告一致；
- 人工确认；
- 对设备位置、左右侧、严重程度的精确一致。

### Contradict

可接受来源：

- 报告显式否定；
- 高可靠外部阴性标签；
- 人工确认；
- 受控 laterality/location/severity counterfactual 且已排除实际成立可能。

禁止：

- 单纯“报告未提及”；
- mined hard negative 未人工或规则核验；
- 双侧病变中的左右侧简单翻转；
- 标签工具输出不确定却强制映射为阴性。

### Uncertain

来源：

- 报告中的 hedged finding；
- 边界性、小量、模糊、可能/不能排除等表达；
- 人工审阅确认相关区域可见但证据不明确；
- 多阅片者在 support/contradict 之间分歧、经裁决后认定影像本身模糊。

禁止把模型低置信度直接当作 uncertain 标签。

### Insufficient

自然样本优先：

- 图像截断、旋转、严重欠曝光/过曝光；
- 所需 anatomy 未覆盖；
- 所需 view 缺失；
- 设备位置目标未包含；
- 需要 prior 或多时点；
- 当前模态不能回答。

训练干预样本：

- 对 S/C/U 图像删除专家区域或当前 evidence set，目标设为 I；
- 这些是**干预生成的 insufficient**，必须与自然 insufficient 分开报告；
- 测试集必须包含真实 insufficient，不能只测试自己合成的遮挡。

---

## 5.5 同陈述成组与匹配

对每个 canonical statement，尽量构造 S/C/U/I 组。匹配变量至少包括：

- AP / PA / lateral；
- 年龄段；
- 性别；
- portable 标记；
- 支持设备；
- 数据来源；
- 图像质量；
- 主要共病负担。

目标不是让所有变量完全相同，而是降低模型用 acquisition artifact 或共病统计猜状态的机会。

推荐 batch sampler：

```text
每个 batch 先采样 canonical statement
→ 再从不同状态各采样若干患者
→ 对缺失状态使用匹配候选池
→ 不允许同一患者的其他 study 跨 split
```

---

## 5.6 划分与泄漏控制

必须先按患者划分，再构造 pair/group：

- train；
- validation；
- calibration；
- internal expert test；
- external test。

严格禁止：

- 同一患者不同 study 跨 split；
- 从测试报告中生成训练 paraphrase；
- 用最终测试做阈值、温度或 top-\(K\) 选择；
- hard-negative 检索跨入测试集；
- 在同一测试集上进行 33 个模型选择后再把最好一项当作无偏结果。

额外鲁棒性划分：

- template-disjoint paraphrase test；
- institution-disjoint external；
- acquisition-shift subset；
- rare-finding subset；
- natural-insufficient-only subset。

---

## 5.7 专家审阅集设计

### 推荐目标规模

理想目标：

- 约 2,400 个 image-statement pairs；
- 8 个 finding × 4 个状态 × 每格约 75 个；
- 其中 800–1,200 个有证据区域标注。

最低可接受方案：

- 至少 1,200 个 pairs；
- 每个主要 finding 和每个状态均有足够样本；
- 至少 400–600 个 region-annotated cases。

### 标注流程

1. 两位独立放射科医师标注；
2. 第三位专家对分歧样本裁决；
3. 标注内容：
   - 四状态；
   - 置信度；
   - 证据区域；
   - 证据是局灶还是弥漫；
   - insufficiency 原因；
   - 是否需要额外 view/prior；
4. 报告：
   - Gwet’s AC1 或 Cohen’s \(\kappa\)；
   - classwise agreement；
   - uncertain–insufficient confusion；
   - region agreement。

### 标注 gate

在大规模标注前先进行小规模双盲 pilot。建议 gate：

- support/contradict 标签 precision 达到可接受水平；
- uncertain 与 insufficient 的 agreement 至少达到中等以上；
- 若两者持续无法区分，必须修改定义和界面，不能直接用自动标签训练后宣称语义成立。

---

## 6. 研究问题

### RQ1：双极证据表示是否真正改善四状态识别？

比较：

- flat 4-class softmax；
- factorized answerability + polarity；
- evidential non-spatial classifier；
- BiVES bipolar spatial evidence decoder。

重点指标：

- 4-state macro-F1；
- U-vs-I AUROC；
- S-vs-C AUROC；
- per-class F1；
- confusion matrix。

### RQ2：方法是否真的更依赖当前影像？

比较 original、blank/no-image、image swap、same-label swap、opposite-state swap。

重点指标：

- unrelated-image answer rate；
- opposite-state responsiveness；
- text-only gap；
- same-statement paired accuracy。

### RQ3：证据集是否在干预意义上必要且充分？

比较：

- evidence-only；
- evidence-deleted；
- irrelevant-region-deleted；
- random equal-area mask；
- expert-region mask；
- model-region mask。

重点指标：

- evidence-only sufficiency；
- evidence-removal-to-insufficient rate；
- irrelevant-mask stability；
- target-versus-control effect gap。

### RQ4：证据图是否对应临床区域？

在 MS-CXR、VinDr、Chest ImaGenome 或人工标注 subset 上评价：

- pointing game；
- BoxRecall@K；
- mIoU / soft-IoU；
- anatomy-level accuracy；
- 局灶与弥漫 finding 分层结果。

### RQ5：方法是否具有外部泛化与校准优势？

在主 external 与 stress external 上评价：

- state metrics；
- intervention metrics；
- calibration；
- subgroup shift；
- label mapping sensitivity。

### RQ6：方法是否独立于 VIVID-Med？

比较：

- 通用公开 radiology/vision backbone；
- 当前 Qwen3-VL vision tower；
- VIVID-Med initialization。

主 claim 必须在非 VIVID-Med 初始化下成立。

---

## 7. Baselines

## 7.1 必须包含的主 baseline

所有主 baseline 尽量采用相同：

- 视觉 backbone；
- statement encoder；
- 数据 split；
- 数据量；
- batch 组成；
- 训练步数；
- 参数更新范围；
- 模型选择规则。

### B0 Text-only

只输入 statement，验证语言泄漏。

### B1 Image-only

只输入图像和 canonical statement ID 的固定查询，验证任务是否退化为普通 finding 分类。

### B2 Flat-4

普通 image-statement fusion + 4-class softmax。对应当前 VSL-4class 的最公平版本。

### B3 Hierarchical-4

先 answerable/insufficient，再对 answerable 做 S/C/U 分类。

### B4 VECL-style

在相同 backbone 上复现 entailment/neutral/contradiction 的 visual entailment contrastive objective，并适配到当前协议。

### B5 CGO/CORAL-style

使用整图 hard-negative swap 和 answer-invariance penalty 的 matched baseline。若与原论文架构不同，必须写成 “CGO-style matched implementation”，不能声称完全复现原 CORAL。

### B6 Localize-before-answer

使用区域预测或弱 region supervision 后再分类，验证“先定位”是否已经足够。

### B6b Phrase-grounded fact-checking baseline

在当前 statement ontology 上构造 finding/location perturbations，并复现 cross-modal veracity + localization 的 matched baseline，检验 BiVES 的优势是否来自四状态证据建模与局部干预，而不是普通报告错误检测。

### B7 Selector-only rationale baseline

学习稀疏 mask + 普通分类，但没有 bipolar evidence 和 intervention closure。

### B8 Bipolar-no-intervention

使用 BiVES 证据解码，但不加 keep/drop/control，隔离表示创新。

### B9 BiVES-CXR full

完整模型。

---

## 7.2 表征学习 baseline（次要）

用于 CheXpert linear probing 或 transfer：

- raw backbone；
- report/image contrastive baseline；
- BioViL/MedCLIP/相关可复现实验；
- VECL；
- 当前 SAMEQ；
- BiVES-CXR。

注意：线性探测只是 secondary endpoint，不能作为 visual evidence claim 的主要证据。

---

## 8. 指标

## 8.1 状态识别指标

主要：

- macro-F1；
- balanced accuracy；
- per-class F1；
- one-vs-rest AUROC/AUPRC；
- confusion matrix。

结构化指标：

- answerability AUROC：\(I\) vs \(S/C/U\)；
- polarity AUROC：\(S\) vs \(C\)；
- ambiguity AUROC：\(U\) vs \(S/C\)；
- U-vs-I AUROC：检验冲突与缺证据分离。

## 8.2 干预指标

### Evidence-Only Sufficiency（EOS）

在原图判断正确的样本中，仅保留 evidence set 后仍保持原状态的比例：

\[
EOS=\Pr[\hat y(x^{keep},s)=y\mid \hat y(x,s)=y].
\]

### Evidence Removal to Insufficient（ERI）

对原图判断正确且可回答的样本，删除 evidence set 后转为 insufficient 的比例：

\[
ERI=\Pr[\hat y(x^{drop},s)=I\mid \hat y(x,s)=y,y\neq I].
\]

同时报告连续变化：

\[
\Delta_I=p_I(x^{drop},s)-p_I(x,s).
\]

### Irrelevant Intervention Stability（IIS）

删除等面积无关区域后保持原状态的比例：

\[
IIS=\Pr[\hat y(x^{ctrl},s)=\hat y(x,s)].
\]

### Target-Control Intervention Gap（TCIG）

\[
TCIG=\Delta_{target}-\Delta_{control},
\]

其中 \(\Delta\) 可定义为原状态概率下降或预测翻转率。

### Opposite-State Responsiveness（OSR）

同一 statement 换成相反证据状态的图像后，预测极性正确改变的比例。

### Unrelated-Image Answer Rate（UAR）

沿用近期 causal audit 的定义，便于与已有工作比较。

不要只给一个混合总分。EOS、ERI、IIS、OSR 应分别报告；可在 supplement 中给一个调和平均作为 secondary summary。

---

## 8.3 Grounding 指标

- pointing game；
- BoxRecall@1 / @K；
- mIoU / soft-IoU；
- anatomy-region accuracy；
- target-vs-irrelevant mask effect；
- support-map 与 contradict-map 分别评价；
- 局灶 vs 弥漫 finding 分层。

注意：attention overlap 不能替代干预证据。Grounding overlap 是辅助结果，因果区域效应才是主结果。

---

## 8.4 Calibration 与 selective prediction

报告：

- NLL；
- Brier score；
- classwise ECE；
- adaptive ECE；
- reliability diagrams；
- risk–coverage curves；
- AURC；
- 固定 coverage 下 selective risk。

阈值与温度必须只在独立 calibration split 上拟合。

---

## 8.5 外部泛化

至少报告：

- internal expert test；
- main external；
- second external / stress；
- common-label subset；
- per-dataset prevalence；
- 有无 recalibration 两种结果；
- acquisition/view subgroup；
- label mapping sensitivity analysis。

---

## 8.6 效率

报告：

- trainable / total parameters；
- FLOPs；
- 单图单 statement latency；
- 批量多 statement latency；
- GPU memory；
- 是否运行 text encoder；
- statement embedding cache 成本；
- 是否需要生成式 LLM。

---

## 9. 统计计划

### 9.1 多 seed

至少对以下方法做 3 seeds：

- Flat-4；
- VECL-style；
- CGO-style；
- Bipolar-no-intervention；
- BiVES-CXR full；
- 最强 localize-before-answer baseline。

其余大规模 ablation 可先单 seed 筛选，但最终保留的关键结论应做 3 seeds。

### 9.2 置信区间与检验

- 患者级 bootstrap，建议 10,000 次；
- 所有模型使用相同 bootstrap samples 做 paired comparison；
- AUROC 可用 paired bootstrap 或 DeLong；
- accuracy/F1 用 paired bootstrap 或 permutation test；
- 干预前后用配对检验；
- 多个主要比较进行 Holm–Bonferroni 或预注册的层级检验。

### 9.3 模型选择

- 在 validation 上选择 architecture 和 loss；
- 在 calibration split 上选择温度和拒答阈值；
- final test 只运行一次锁定配置；
- 不比较不同任务的 training loss；
- 不从不同 checkpoint 分别抽取最佳 AUC、AUPRC 和 ECE 拼成“最终模型”。

### 9.4 Primary endpoints 预注册

建议预先锁定：

1. Expert audit 4-state macro-F1；
2. U-vs-I AUROC；
3. ERI；
4. IIS；
5. main external macro-F1。

其余为 secondary endpoints。

---

## 10. 实验运行矩阵

## 10.1 阶段 P0：数据与标签可行性

| Run ID | 内容 | 输出 | Gate |
|---|---|---|---|
| P0-W | 冻结规则弱标签代理 P0（5,000-study intake、parser v3、三 finding 扩展验证） | target-local 规则作用域、唯一候选 ID、规则/报告/图像哈希、48/48 患者隔离 S/C/U/I proxy rows | 排序门通过：总体 held-out S/C AUROC 0.8056、U/I 1.0，consolidation/pleural effusion/pulmonary edema 的 S/C 分别为 0.875/0.8125/1.0；决策门失败：四分类 argmax 全落到 insufficient、accuracy 0.25；train-proxy 参数拟合只能把 held-out accuracy 提到 0.5417，故仍停止 4B/9B 扩展且不得作为专家真值或正式 P0 结果 |
| P0-1 | 报告解析人工抽样审计（当前不可用） | 各字段 precision/recall | 未完成，不阻塞代理 P0，但阻塞临床可靠性声明 |
| P0-2 | U/I 双盲标注 pilot（当前取消） | agreement、混淆 | 无审阅者；不得声称定义已被专家稳定区分 |
| P0-3 | same-statement 覆盖统计 | 每 finding 每状态样本量 | 主 finding 有足够跨状态组 |
| P0-4 | 泄漏审计 | text-only、模板频率 | text-only 不应接近多模态主模型 |
| P0-5 | 外部 manifest 审计 | 可用标签、box、许可 | 锁定主 external |

若 P0 失败，应优先修数据，不进入大规模模型堆叠。

---

## 10.2 阶段 P1：最小机制证明

只使用 4–6 个高可靠 finding，缩小数据，验证方法机制。

| Run ID | 方法 | 目的 |
|---|---|---|
| P1-B0 | Text-only | 测语言泄漏 |
| P1-B1 | Flat-4 | 最小分类 baseline |
| P1-B2 | VECL-style | 最近邻视觉蕴含 baseline |
| P1-B3 | CGO-style | 整图交换依赖 baseline |
| P1-M0 | Bipolar decoder，无空间 mask | 验证 vacuity/dissonance 分解 |
| P1-M1 | Bipolar + evidence mask | 验证空间证据 |
| P1-M2 | + sufficiency | 验证 evidence-only |
| P1-M3 | + necessity | 验证 evidence deletion |
| P1-M4 | + irrelevant control | 完整 BiVES closure |

### P1 go/no-go

BiVES full 至少需要同时表现出：

- 状态识别优于 Flat-4；
- U-vs-I 分离优于 hierarchical baseline；
- target removal 效应显著大于 irrelevant removal；
- evidence-only 不出现严重性能崩溃；
- 同题异图响应优于 VECL/CGO matched baseline。

若只提高 linear probing 而干预指标不提高，则核心假设未成立。

---

## 10.3 阶段 P2：完整内部实验

扩展到 6–8 个 finding，完成：

- 3 seeds；
- 专家 audit set；
- region subset；
- calibration split；
- 所有主 baselines；
- locked checkpoint；
- failure taxonomy。

主表只保留 6–8 个关键方法，不再呈现 30 多行 variant。完整筛选放 Supplement。

---

## 10.4 阶段 P3：外部验证

至少完成：

1. 一个主 external：具备机构差异且能构造 statement verification；
2. 一个 region/grounding external 或 stress external；
3. common ontology mapping；
4. 无 recalibration 与 external recalibration 两套结果；
5. 主要 subgroup 分析。

当前 NIH 结果可以保留为历史 stress 参考，但不能代替主 external。

---

## 10.5 阶段 P4：最终锁定与论文证据

- 冻结代码 commit；
- 冻结 data manifest；
- 冻结 parser/version；
- 运行 3 seeds；
- 保存全部 per-sample predictions；
- 生成 bootstrap CI；
- 输出 calibration bins；
- 固定论文 case selection 规则；
- 由专家复核所有展示案例；
- 执行最终 novelty search；
- 完成 VIVID-Med overlap disclosure。

---

## 11. 消融设计

## 11.1 表示消融

| 变体 | 目的 |
|---|---|
| Flat softmax | 普通四分类 |
| Hierarchical softmax | answerability 分层 |
| Non-spatial evidential | 证据量但无区域 |
| Bipolar spatial | 正反空间证据 |
| Bipolar + learned thresholds | 检验固定/可学习温度 |

## 11.2 干预消融

| 变体 | 保留 | 删除 | 无关对照 | same-statement pair |
|---|---:|---:|---:|---:|
| A0 | × | × | × | × |
| A1 | ✓ | × | × | × |
| A2 | ✓ | ✓ | × | × |
| A3 | ✓ | ✓ | ✓ | × |
| Full | ✓ | ✓ | ✓ | ✓ |

## 11.3 mask 消融

- attention soft mask；
- fixed top-\(K\)；
- soft top-\(K\)；
- hard-concrete \(L_0\)；
- anatomy-restricted mask；
- 弱 region warm-start vs 无 region supervision。

## 11.4 数据消融

- random pair；
- same-statement pair；
- metadata-matched pair；
- random negatives；
- hard negatives；
- explicit negatives only；
- 含/不含 uncertain；
- synthetic vs natural insufficient。

## 11.5 backbone 消融

- 公共基础 backbone；
- 当前 Qwen3-VL vision tower；
- VIVID-Med initialization。

## 11.6 语言消融

- canonical statement ID；
- canonical text；
- seen paraphrase；
- unseen paraphrase；
- negation template；
- text-only。

---

## 12. 初始实现建议

这些是起始配置，不应在未验证前写成最终固定值。

### 12.1 Backbone

主实验建议：

- 选择一个公开、可复现、非 VIVID-Med 专属的视觉 backbone；
- 同时复用当前 Qwen3-VL vision tower 以减少工程风险；
- VIVID-Med 权重作为 initialization ablation。

### 12.2 分辨率与 patch

- 初始统一到 448×448 或当前 backbone 的标准输入；
- 保存原始比例并使用合理 padding；
- 对 laterality 和设备位置避免会改变方向的随机水平翻转；
- 任何翻转必须同步更新 statement laterality。

### 12.3 优化

可从以下范围开始：

```yaml
optimizer: AdamW
lr_new_layers: 1e-4
lr_backbone: 1e-5
weight_decay: 0.05
scheduler: cosine
warmup_ratio: 0.05
precision: bf16
```

建议：

1. 先冻结 backbone，训练 evidence head/gate；
2. 再解冻后若干 block 或全量低学习率微调；
3. 先只训练 \(\mathcal L_{state}\)，再逐步 ramp-up \(\mathcal L_{IES}\)；
4. batch 以 statement group 为单位组织。

### 12.4 mask

初始 screening：

```yaml
mask_type: soft_topk
topk_candidates: [8, 16, 32]
```

最终：

```yaml
mask_type: hard_concrete
l0_regularization: tuned_on_validation
```

对弥漫病变可用更高 \(K\) 或 anatomy-level gate。

### 12.5 损失系数初始搜索

```yaml
lambda_ies: [0.25, 0.5, 1.0]
lambda_n: [0.5, 1.0]
lambda_c: [0.25, 0.5, 1.0]
lambda_pair: [0.1, 0.5]
lambda_min: [1e-4, 1e-3, 1e-2]
lambda_tv: [0, 1e-4, 1e-3]
```

先小规模单 seed 选择可行区间，再锁定有限候选进行 3 seeds。不要进行无边界的超参数搜索。

### 12.6 防止 mask collapse

监控：

- 平均 mask area；
- 每类 mask area；
- evidence magnitude；
- keep/drop/control 的状态分布；
- 是否总选图像边缘、文字标记、导管或固定角落。

措施：

- state-loss warm-up；
- fixed top-\(K\) warm-start；
- patch dropout；
- acquisition artifact augmentation；
- anatomy-aware negative controls；
- 对 mask area 设置上下界或 curriculum；
- 定期执行 target-vs-random intervention audit。

---

## 13. 预期主表与图

## Figure 1：问题定义

同一个 statement 配四种图像状态，展示：

- support；
- contradict；
- uncertain；
- insufficient；
- uncertain 与 insufficient 的区别。

## Figure 2：BiVES-CXR 方法

只画三个核心步骤：

1. 双极 patch evidence；
2. evidence-state decoder；
3. keep/drop/control intervention closure。

不要画成五个独立模块框。

## Figure 3：干预示例

对同一病例显示：

- original；
- evidence-only；
- evidence-deleted；
- irrelevant-deleted；
- 各状态概率变化。

## Figure 4：vacuity–dissonance 平面

横轴：总证据量或 availability；
纵轴：决断度/冲突度；
颜色：S/C/U/I；
显示四类是否形成符合定义的结构。

## Figure 5：Grounding 与失败案例

- 正确支持区域；
- 正确反驳区域；
- diffuse finding；
- artifact shortcut；
- uncertain–insufficient confusion；
- external shift。

## Figure 6：Calibration / risk–coverage

展示 selective prediction 是否能用 \(p_I\) 和 \(p_U\) 降低风险。

### Table 1：数据与人工审阅

- 数据集；
- 患者/图像/pair 数；
- 四状态分布；
- finding 分布；
- region annotations；
- inter-rater agreement。

### Table 2：主四状态结果

主 baselines + BiVES，internal expert test 和 main external。

### Table 3：干预与图像依赖

EOS、ERI、IIS、OSR、UAR、TCIG。

### Table 4：Grounding、calibration 与效率

Grounding、ECE/Brier/AURC、参数量、延迟。

### Supplement

- 所有消融；
- 数据规则；
- parser audit；
- 统计细节；
- 更多 external；
- 证明；
- casebook；
- 完整 per-finding 表。

---

## 14. 当前已有资产如何迁移

| 现有资产 | 新 proposal 中的用途 | 是否保留为贡献 |
|---|---|---|
| SAMEQ 数据与 sampler | 构造 same-statement cross-state group | 否，数据策略 |
| SAMEQ-HNMB | hard baseline / candidate mining | 否 |
| VSL-4class 数据 | 候选标签池，重新审计 | 任务基础，不单独作为方法创新 |
| CEQ 代码 | 可用于 evidence gate 初始化或 baseline | 否 |
| CCSH | baseline；最终方法用闭式 evidence decoder 取代 | 否 |
| AUCH | baseline；最终用 availability/dissonance 取代 | 否 |
| CheXpert LP scripts | secondary representation evaluation | 保留工具 |
| NIH scripts | stress test | 保留但降级 |
| casebook | 人工审阅候选池 | 需重新复核 |
| teacher comparison | 不作为主线 | 可删除或放 supplement |

### 当前已有结果的定位

已有结果说明：

- same-statement 类监督具有信号；
- evidence-aware readout 有可行性；
- integrated training 能改善部分 in-domain representation；
- external shift 与 calibration 仍是明显问题。

它们应当被视为**提出 BiVES-CXR 的 pilot evidence**，而不是最终论文的主结果。旧 33-row 表不应直接变成主论文的主表。

---

## 15. 失败模式与预案

### 15.1 uncertain 与 insufficient 无法稳定标注

表现：专家一致性低，自动标签混淆严重。

处理：

1. 收紧定义；
2. 将原因细分并优化标注界面；
3. 暂时将主任务设为 S/C/not-verifiable 做机制 pilot；
4. 只有当 U/I gate 通过后，才恢复四状态作为顶刊主 claim。

若最终仍无法区分，当前论文的核心 thesis 需要重做，而不是用模型结果掩盖标签问题。

### 15.2 evidence mask 选中伪影

表现：总关注边框、文字、portable 标记、导管或固定位置。

处理：

- metadata-matched grouping；
- artifact perturbation；
- external institution test；
- anatomy prior；
- expert-region audit；
- control-region intervention；
- 报告 artifact-specific failure rate。

### 15.3 删除操作产生分布外伪影

处理：

- 训练用 feature mask token；
- 测试同时使用 blur、inpainting、mean replacement、feature deletion 等多个算子；
- 一个算子训练，另一个算子测试；
- 结论只在多个干预算子一致时成立。

### 15.4 mask 过大

表现：整张图都被认为是 evidence，必要性指标虚高。

处理：

- \(L_0\) 最小性；
- area-matched control；
- 报告 evidence fraction；
- 在同等面积下比较 baseline；
- 对弥漫病变使用 anatomy region 而非任意全图。

### 15.5 mask 过小或单点 collapse

处理：

- top-\(K\) warm-start；
- 局部连续性；
- evidence-only 性能 gate；
- multi-scale patch；
- finding-specific sparsity。

### 15.6 只在 VIVID-Med 初始化上有效

处理：

- 在公开 backbone 上复现；
- 把 VIVID-Med 作为性能增强而非方法成立条件；
- 若无法复现，必须将论文改写成明确的 VIVID-Med extension，并重新评估重复贡献风险。

### 15.7 外部结果不提升

处理：

- 检查 label mapping；
- 分离 prevalence shift、acquisition shift 和 parser shift；
- 报告 external recalibration；
- 不声称 universal generalization；
- 若主外部机制指标也无提升，则方法尚不具备 MIA/TMI-ready 证据。

### 15.8 对 VECL/CORAL 无优势

这不是“再加一个模块”的信号。应检查：

- 双极表示是否真正有用；
- 局部干预是否学到临床证据；
- U/I 标签是否可靠；
- 训练干预是否与测试干预一致；
- 任务是否本质上可以用更简单方法解决。

若在公平协议下没有机制优势，应该停止顶刊包装并修改核心假设。

---

## 16. 投稿级 claim discipline

### 16.1 只有结果支持后才能写的 claim

- BiVES-CXR distinguishes ambiguous evidence from insufficient evidence through bipolar evidence availability and decisiveness.
- The learned evidence set is sufficient when retained and necessary when removed under controlled interventions.
- BiVES reduces image-independent predictions relative to matched visual-entailment and whole-image hard-negative baselines.
- The method generalizes to an external institution and remains calibrated after prespecified calibration.

### 16.2 不应写的 claim

- “attention proves causal grounding”；
- “first visual sufficiency method”，除非投稿当日系统检索确认；
- “human-level” 或 “clinical deployment ready”；
- “insufficient equals epistemic uncertainty” 而无严格限定；
- “uncertain equals aleatoric uncertainty” 而无人工证据；
- 仅凭 linear probing 宣称模型更看图；
- 用 synthetic deletion 的好结果替代真实 insufficient 验证；
- 用不同 checkpoint 的最好数字拼接最终模型；
- 把 Qwen3-VL 与未完成 adapter 的模型比较写成 superiority；
- 把 NIH stress 写成主 external success。

---

## 17. 建议论文结构

## 17.1 Introduction

四段即可：

1. 报告监督丰富，但语义正确不等于当前图像提供证据；
2. 现有 entailment、grounding 和 hard-negative 方法仍不能区分冲突证据与证据缺失，也不保证局部证据必要且充分；
3. 提出 statement-conditioned bipolar evidence set 与 interventional closure；
4. 概括人工审阅、因果干预、外部验证和主要结果。

## 17.2 Related Work

仅保留四组：

- medical image–text representation / visual entailment；
- grounded medical VQA / report fact checking；
- shortcut and counterfactual image-use audits；
- evidential uncertainty and sufficient rationales。

## 17.3 Method

1. Problem definition；
2. Bipolar visual evidence field；
3. Evidence-state decoder；
4. Interventional evidence-set closure；
5. Optimization and inference；
6. Operational proposition。

## 17.4 Data and Evaluation

1. Statement construction；
2. Four-state annotation；
3. Expert audit set；
4. External datasets；
5. Interventions and metrics；
6. Statistical plan。

## 17.5 Results

按研究问题组织，而不是按 module 组织：

1. Four-state verification；
2. Image reliance；
3. Evidence necessity/sufficiency；
4. Grounding；
5. External/calibration；
6. Ablations/failure analysis。

## 17.6 Discussion

必须主动讨论：

- causal claim 的干预族边界；
- report-derived label noise；
- negative evidence 是否可局部化；
- diffuse findings；
- synthetic vs natural insufficient；
- external ontology mismatch；
- 非自主诊断用途。

---

## 18. 预期贡献表述

最终摘要中最多保留三项贡献：

1. **Problem/representation**：提出双极空间证据表示，将 support、contradict、uncertain 和 insufficient 统一为证据方向、决断度和可用性；
2. **Method**：提出干预式证据集闭包，使选定区域在保留时充分、删除时必要、无关区域扰动时稳定；
3. **Evidence**：构建专家审阅的四状态与区域测试协议，并在公平 baselines、外部数据和多 seed 下验证真实图像依赖。

不要再把 SAMEQ、CEQ、CCSH、AUCH 分别列为四项贡献。

---

## 19. 摘要草案（结果占位版）

> Radiology reports provide rich supervision for chest radiographs, but a statement that is clinically plausible or present in a report is not necessarily verifiable from the current image. Existing image–text entailment and hard-negative training methods mainly model semantic agreement or whole-image dependence, without identifying which patient-specific visual evidence is necessary and sufficient for a decision, or distinguishing ambiguous evidence from missing evidence. We introduce BiVES-CXR, a bipolar interventional visual evidence-set framework for clinical statement verification. Given a radiograph and an atomic statement, BiVES-CXR estimates spatial support and contradiction evidence, and derives four states—support, contradict, uncertain, and insufficient—from evidence availability, decisiveness, and polarity. The learned evidence set is trained under an interventional closure: retaining it should preserve the original state, deleting it should render the statement visually insufficient, and perturbing matched irrelevant regions should leave the prediction stable. We evaluate BiVES-CXR on a patient-disjoint expert-audited test set and external chest-radiograph cohorts, using four-state accuracy, calibration, spatial grounding, and target-versus-control intervention metrics. Under matched backbones and training budgets, BiVES-CXR improves [PRIMARY METRIC] over visual-entailment and whole-image hard-negative baselines, while increasing evidence-removal sensitivity and irrelevant-region stability. These results suggest a route from report-level semantic supervision to patient-specific, interventionally testable visual evidence.

结果出来后再填数字，禁止在实验完成前预写 superiority。

---

## 20. MIA 与 TMI 的最终选择

### 优先选择 Medical Image Analysis 的条件

更适合 MIA 的版本应包含：

- 新的 evidence representation 与 objective；
- 专家审阅数据或 evaluation protocol；
- 充分的 causal/interventional 分析；
- 两类外部验证；
- 较完整的 failure taxonomy；
- extensive supplement。

本 proposal 更自然地走这一版本，因为问题、方法、数据审计和机制评价共同构成贡献。

### 选择 TMI 的条件

TMI 版本需要进一步压缩为：

- 一个非常清晰的数学方法；
- 一个主要 proposition；
- 3–4 张关键表；
- 直接超过 VECL-style、CGO-style 和 localize-before-answer；
- 强 external；
- 去掉大量数据引擎细节和非核心 variants。

若最终结果只是多个实验面向的综合优势，而方法本身不够尖锐，优先 MIA；若方法在公平协议下有显著、稳定、可复现的机制提升，TMI 也可行。

---

## 21. Paper-ready gate checklist

### 方法

- [ ] 双极证据场与四状态解码稳定训练；
- [ ] keep/drop/control 均有预期行为；
- [ ] 没有普通四分类 head 偷偷承担全部预测；
- [ ] mask area 与 evidence scale 无明显 collapse；
- [ ] 在公开/独立 backbone 上成立。

### 数据

- [ ] 患者级划分先于 pair 构造；
- [ ] explicit negative 不等于 report omission；
- [ ] natural insufficient 单独保留；
- [ ] U/I 人工一致性达标；
- [ ] 主 external manifest 完整；
- [ ] parser、prompt 和版本冻结。

### 实验

- [ ] VECL-style matched baseline；
- [ ] CGO/CORAL-style matched baseline；
- [ ] localize-before-answer baseline；
- [ ] text-only 与 image-swap audit；
- [ ] target vs irrelevant intervention；
- [ ] evidence-only sufficiency；
- [ ] external；
- [ ] calibration；
- [ ] 3 seeds；
- [ ] paired CI 与统计检验。

### 论文完整性

- [ ] 所有主指标来自同一锁定 checkpoint；
- [ ] VIVID-Med 被引用并明确区分；
- [ ] 重用代码、权重、数据和 split 已披露；
- [ ] 不使用未经证实的 “first”；
- [ ] 所有 qualitative cases 经专家复核；
- [ ] 全部 per-sample prediction 可追溯；
- [ ] 代码、配置和 manifest 可复现；
- [ ] final novelty search 已完成。

---

## 22. 最终执行优先级

### 第一优先级：先证明标签和机制成立

1. U/I 人工 pilot；
2. same-statement 跨状态覆盖；
3. Flat-4、Bipolar-no-intervention、BiVES full 的小规模比较；
4. target-vs-control intervention；
5. 检查 mask 是否选中临床区域。

### 第二优先级：补最近邻公平比较

1. VECL-style；
2. CGO-style；
3. localize-before-answer；
4. selector-only rationale baseline。

### 第三优先级：形成顶刊证据链

1. 专家 audit set；
2. 主 external；
3. 3 seeds；
4. calibration 与 selective risk；
5. 独立 backbone；
6. 完整统计与 failure taxonomy。

### 明确不再优先的内容

- 扩展更多 teacher family，但没有公平 adapter；
- 继续增加 CEQ 变体；
- 再接新的 consistency/uncertainty head；
- 只追求更低 training loss；
- 在 NIH stress 上反复调参；
- 通过更多模块名称制造贡献数量。

---

## 23. 关键文献锚点（投稿前需更新）

1. **VIVID-Med: LLM-Supervised Structured Pretraining for Deployable Medical ViTs.** arXiv:2603.09109, 2026.
2. **Medical Contrastive Learning of Positive and Negative Mentions / VECL.** MICCAI 2025.
3. **Do Medical Vision Language Models Actually See? A Counterfactual Grounding Framework and Hard-Negative Contrastive Training for Visually-Reliant Medical VLMs.** arXiv:2607.03647, 2026. CORAL/CGO.
4. **Vision-language models for chest radiography do not always need the image.** arXiv:2606.17710, 2026.
5. **Localizing Before Answering: A Hallucination Evaluation Benchmark for Grounded Medical Multimodal LLMs.** IJCAI 2025 / arXiv:2505.00744.
6. **Phrase-grounded Fact-checking for Automatically Generated Chest X-ray Reports.** MICCAI 2025 / arXiv:2509.21356.
7. **ShoViR: A Benchmark for Evaluating Vision Shortcut Learning in Radiology Report Generation.** arXiv:2606.30201, 2026.
8. **SwapMix: Diagnosing and Regularizing the Over-Reliance on Visual Context in Visual Question Answering.** CVPR 2022 / arXiv:2204.02285.
9. **Evidential Deep Learning to Quantify Classification Uncertainty.** NeurIPS 2018.
10. 充分性/完整性 rationale learning、subjective logic vacuity/dissonance、counterfactual explanation 等基础工作，投稿前系统补齐。
11. Chest ImaGenome、MS-CXR、PadChest-GR、VinDr-CXR 等 grounding/外部数据资源的原始论文。

---

## 24. 一句话最终版本

> **BiVES-CXR 不再把报告监督做成四分类加多个辅助 head，而是学习一个可被干预验证的双极视觉证据集：证据保留时结论成立，证据删除时转为不足，无关区域变化时结论稳定；证据的总量、冲突和方向分别决定 insufficient、uncertain 以及 support/contradict。**

这是最终论文必须始终围绕的唯一技术主线。
