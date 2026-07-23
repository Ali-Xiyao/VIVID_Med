
# RCSD-CXR 完整研究 Proposal 与实验纲要

> **工作题目（暂定）**  
> **From Noisy Reports to Reliable Visual Representations: Reliability-Calibrated Structured Semantic Distillation for Chest Radiographs**  
> 中文：**从噪声报告到可靠视觉表征：面向胸片编码器的可靠性校准结构化语义蒸馏**
>
> **方法工作名**：RCSD-CXR（Reliability-Calibrated Structured Distillation for Chest X-rays）  
> **目标期刊**：IEEE Transactions on Medical Imaging（优先）/ Medical Image Analysis  
> **论文类型**：方法学论文；不是数据论文、报告生成论文或因果解释论文  
> **核心部署产物**：单图像胸片视觉编码器；推理时不需要 LLM、报告解析器或结构化标签模块  
> **方案版本**：2026-07-22，执行版 v1.0  
> **名称注意**：正式投稿前重新检索方法名、缩写和代码仓库名称冲突。

---

## 0. 一页式最终决策

### 0.1 论文只回答一个问题

大规模胸片数据通常带有自由文本报告和自动提取标签，但这些监督存在三个问题：

1. 不同解析器对同一报告可能给出冲突标签；
2. “未提及”经常被错误当作“阴性”；
3. 报告语义与图像专家标签并不完全一致。

本论文研究：

> **如何在不把报告来源当作无噪声真值的前提下，将多源、缺失且相互冲突的结构化报告语义，可靠地蒸馏到一个最终可独立部署的视觉编码器中。**

### 0.2 方法只新增两个机制

1. **可靠性校准的结构化语义后验**  
   将 CheXpert、CheXbert、NegBio、RadGraph、Chest ImaGenome 等来源转换为带缺失掩码的软后验；显式估计不同来源和字段的可靠性，不把未提及映射为阴性。

2. **字段锚定的语义分解蒸馏**  
   保留原 VIVID-Med 的 ViT + SPD 思路，但将查询组绑定到明确语义字段，并用冻结文本教师生成的原子语义原型进行软分布蒸馏。训练后丢弃文本教师和查询投影器，只保留 ViT。

### 0.3 明确不做

- 不加入 GFTM；
- 不加入 FSA consistency；
- 不做 CXR+CT 混合预训练；
- 不使用 HFP；
- 不做 VSL 四状态；
- 不使用 CEQ/CCSH/AUCH；
- 不声称学习到因果证据；
- 不把报告 uncertain 解释为视觉不确定；
- 不把 CT 或病理数据塞进主论文；
- 不用更多模块补救核心假设失败。

### 0.4 论文成败的核心标准

论文能否进入 TMI/MIA，不取决于模型是否在一个表格中多 0.2 个点，而取决于以下证据链是否同时成立：

1. 多源软后验在独立人工文本标注集上优于最佳单一解析器；
2. 可靠性分数能预测报告标签与专家图像标签的一致程度；
3. 在相同 MIMIC 数据、相同 ViT、相同训练预算下，方法优于原始 A+UMS+SPD；
4. 提升在专家标注 CheXpert、NIH、VinDr 和至少一个新外部域中保持；
5. 在低标注量和长尾条件下优势更明显；
6. 结果跨 3 个 seed 稳定；
7. 所有最终结论来自预先锁定的代码、manifest、checkpoint 选择规则和统计方案。

---

# 1. 科学背景与问题定义

## 1.1 原 VIVID-Med 可保留的稳定核心

原项目最终相对稳定的科学问题是：

> 使用冻结语言模型的结构化语义空间，训练一个可部署的医学视觉编码器。

原 V12 的核心组件为：

- 可训练 ViT；
- 结构化 UMS 目标；
- SPD query projector；
- 冻结文本/语言模型；
- 训练后仅保留 ViT。

历史结果提示：

- 自由文本监督到 UMS 有增益；
- UMS 到 SPD 有进一步增益；
- NIH 跨机构结果方向一致。

这些历史数字只能作为 pilot 和复现预期，不能直接进入新论文主表。新论文必须重新锁定数据、配置和 checkpoint 后重跑。

## 1.2 当前方法空缺

现有胸片视觉语言和知识增强方法已经覆盖：

- 原始图文对比；
- 疾病实体/三元组抽取；
- 知识图谱；
- disease query；
- patch-level 对齐；
- 多源数据训练；
- 大规模自监督基础模型。

因此本文不能声称“首次使用结构化报告”“首次使用疾病 query”或“首次使用知识图谱”。

本文聚焦的空缺是：

> 现有方法通常将自动抽取的结构化语义作为确定标签或统一可信的文本监督，而没有把来源相关错误、字段相关错误、缺失和冲突作为训练目标的一部分。

## 1.3 形式化问题

给定胸片图像 \(x_i\)、报告 \(r_i\)、概念集合 \(\mathcal C\) 和语义字段集合 \(\mathcal F\)。

对于概念 \(c\) 和字段 \(f\)，存在不可直接观测的结构化语义状态：

\[
z_{icf}\in\mathcal Y_f.
\]

多个监督来源 \(s\in\mathcal S\) 产生：

\[
\tilde z_{icfs},
\]

其中可能包含：

- 错误；
- 冲突；
- 缺失；
- 不同定义口径。

目标不是强行生成一个硬标签，而是估计：

\[
q_{icf}(z)
=
P(z_{icf}=z\mid \tilde z_{icf1},\ldots,\tilde z_{icfS}),
\]

并将该软后验及其可靠性蒸馏到视觉编码器 \(E_\theta\)。

---

# 2. 研究假设

## H1：多源监督融合假设

在独立人工标注报告上，来源校准后的软后验应当在 NLL、ECE 和宏平均 F1 上优于任何单一自动标签来源。

## H2：可靠性可排序假设

软后验的置信度/熵应能排序标签质量：高可靠性样本与专家图像标签的一致率应显著高于低可靠性样本。

## H3：软语义优于硬 UMS 假设

在相同 ViT、相同数据和相同训练预算下，使用可靠性加权软后验，应优于把某个 parser 输出直接序列化为硬 UMS。

## H4：字段分解提供额外增益

在软后验固定后，字段锚定查询应优于未锚定 SPD 或普通 projector，证明增益不仅来自更好的标签。

## H5：优势在监督稀缺和域偏移下更明显

若方法确实改善了表征而非只拟合训练标签，则其优势应在低数据 linear probe、跨机构测试和长尾类别中更明显。

---

# 3. 预期贡献

1. **多源、字段级可靠性校准的结构化监督模型**  
   统一处理报告标签冲突、来源缺失和 uncertainty，不把未提及当作阴性。

2. **面向软语义后验的字段锚定视觉蒸馏方法**  
   使用概念条件查询和冻结语义原型，将不同字段的结构化语义蒸馏到 ViT。

3. **严格控制的 deployable encoder 实验协议**  
   同数据、同骨干、同预算比较自由文本、硬 UMS、原 SPD、软监督和完整方法。

4. **标签可靠性—视觉表征收益的机制分析**  
   定量检验标签融合质量、可靠性分层、低数据迁移和外部泛化之间的关系。

---

# 4. Novelty 边界

## 4.1 与 MedKLIP/KAD/图知识预训练的区别

已有工作使用医学实体、三元组、知识图谱或疾病 query 进行图文/patch 对齐。本方法不把“query”或“结构化文本”本身作为主要创新。

主要区别：

- 结构化标签不是单一 parser 的硬输出；
- 每个来源和字段经过独立校准；
- 未提及是缺失，不是阴性；
- 训练目标是软语义后验；
- 可靠性分数必须通过人工 gold 和图像专家标签验证；
- 语言教师只用于训练，最终部署为纯视觉编码器。

## 4.2 与 Ark+/UniChest 的区别

Ark+、UniChest 重点研究多数据集和异构标注的规模化利用。本论文的主实验先固定为 MIMIC-only，以隔离方法贡献；MIMIC+CheXpert Plus 仅作为通过主闸门后的 scale track。

## 4.3 与 BioViL/CheXzero/CLIP 类方法的区别

这些方法主要学习图像—原始报告全局或局部对齐。本方法显式建模报告解析噪声、字段缺失和多源冲突，并不要求保留文本编码器进行零样本部署。

## 4.4 与原 VIVID-Med 的区别

原方法将 UMS 视为确定性目标，SPD query 语义没有被明确绑定。新方法：

- 将 UMS 从硬 JSON 改为可靠性校准软后验；
- 将 query group 绑定至可审计字段；
- 预计算冻结语义原型；
- 不在训练中动态调用 LLM；
- 明确验证标签可靠性和专家标签泛化。

---

# 5. 数据与使用边界

## 5.1 主训练资源

### MIMIC-CXR：Primary controlled pretraining

用途：

- 所有受控方法对比的唯一预训练数据；
- 只使用官方 train split；
- 一位患者只能属于一个 split；
- 主协议仅保留 frontal AP/PA；
- 每个 study 只选择一个 canonical frontal image；
- 使用报告、CheXpert/NegBio 标签、RadGraph、Chest ImaGenome 等训练资源；
- MIMIC test 不用于模型选择。

选择一个图像/每 study 的原因：

- 避免同一 study 多图重复监督；
- 降低 frontal/lateral 与整份报告错配；
- 保持视觉编码器单图输入定义。

canonical image 规则必须在看结果前冻结，例如：

1. 优先 PA；
2. 无 PA 时选择 AP；
3. 同 view 多图时按固定 DICOM ID 排序取第一张；
4. 不使用图像内容或模型分数选择。

### CheXpert Plus：Scale track

只有在 MIMIC-only 主实验通过后才启动。

用途：

- 第二机构规模化训练；
- 研究可靠性方法在来源异构条件下的增益；
- 不是主方法成立的必要条件。

要求：

- 排除原 CheXpert validation/test 的全部患者和图像；
- 通过 DICOM ID、患者 ID 和图像 SHA-256 三重去重；
- CheXlocalize validation/test 图像不得进入预训练；
- train/val 按患者划分。

## 5.2 标签与结构扩展资源

### Chest ImaGenome

- MIMIC 派生资源；
- 用于 anatomy relation 和场景图银标；
- 不能作为独立外部数据；
- 仅使用与 MIMIC train 对齐部分；
- gold subset 用于 anatomy 质量评估。

### CheXTemporal

- 若完整性、许可和标识映射通过，可用于 temporal 字段补充；
- temporal 不作为第一版核心字段；
- temporal 质量未通过时全部放入 supplement，不阻塞主论文。

### RadGraph

- 用于人工报告 gold、来源校准和实体/关系质量评价；
- 自动推理结果作为弱来源；
- gold 与 inference 必须分离。

### LUNGUAGE（建议新下载）

- 用于独立结构化报告测试；
- 不能参与标签模型参数拟合；
- 可用于单报告结构和 temporal appendix；
- 因其来自 MIMIC test，使用后 MIMIC test 不得作为盲测图像面。

## 5.3 下游与外部评价

### CheXpert

- CheXpert train：下游 linear probe/full fine-tuning；
- CheXpert validation：专家图像标签主评价；
- CheXpert test/CheXlocalize test：保留给论文二，不打开。

### NIH ChestX-ray14

- 跨机构无适配测试；
- 报告 8 个与 CheXpert 对齐的共同标签；
- 患者级划分；
- 阈值只能在 CheXpert validation 选择。

### VinDr-CXR

- 使用 radiologist image labels 进行分类迁移/外部复现；
- 历史项目已查看过部分 test 结果，必须标注为 previously inspected external replication；
- 不允许用于本论文超参数选择。

### PadChest（建议新下载）

- 作为西班牙机构、不同语言报告和细粒度标签的外部域；
- 主要用于共同标签分类、长尾评价和跨语言域偏移；
- 下载后先做许可、完整性、患者拆分和标签映射审计。

### CXR-LT/PadChest-GR（可选增强）

- 若访问和使用条款允许，作为长尾和专家标注外部评价；
- 不依赖比赛私有测试标签；
- 任何挑战限制必须严格遵守。

## 5.4 不进入主论文的数据

- AMOS22；
- KiTS21；
- OrganAMNIST；
- CAMELYON16；
- LIDC-IDRI（最多一个 appendix）；
- IU/OpenI（最多做小规模检索或定性分析）。

---

# 6. 核心语义 ontology

## 6.1 核心 finding 集合

第一版保留 12 个 CheXpert findings：

1. Enlarged Cardiomediastinum
2. Cardiomegaly
3. Lung Opacity
4. Lung Lesion
5. Edema
6. Consolidation
7. Pneumonia
8. Atelectasis
9. Pneumothorax
10. Pleural Effusion
11. Fracture
12. Support Devices

处理原则：

- No Finding 不作为独立疾病概念；
- Pleural Other 因频率和映射稳定性不足，默认不进入核心宏平均；
- 若某 finding 不满足样本量/质量闸门，从 core endpoint 移至 secondary，不用调参强行保留。

## 6.2 Assertion 字段

\[
\mathcal Y_{\rm assertion}
=
\{\text{present},\text{absent},\text{uncertain}\}.
\]

**unmentioned 不是类别**，而是缺失掩码。

这三个状态描述报告/监督来源的 assertion，不声称对应“视觉不确定性”。

## 6.3 Anatomy 字段

将 RadGraph/Chest ImaGenome 位置统一到预先冻结的 coarse ontology。建议先从 8 个大区开始：

1. left lung
2. right lung
3. bilateral lungs
4. pleural/costophrenic regions
5. cardiac silhouette
6. mediastinum/hila
7. bones/chest wall
8. devices/other

允许多标签 anatomy。

只有满足以下条件时 anatomy 才进入主方法：

- gold relation macro-F1 ≥ 0.75；
- 至少 6 个 finding 的有效 coverage ≥ 30%；
- anatomy 加入后主 pilot 不恶化 assertion 表现。

否则 anatomy 仅作为 supplement ablation。

## 6.4 Temporal/Severity 字段

不进入第一版必需核心。只有在：

- CheXTemporal/LUNGUAGE 完整；
- 映射准确；
- gold F1 ≥ 0.75；
- 有效训练 study ≥ 10,000；

时作为扩展字段加入 supplement。

---

# 7. 方法：可靠性校准的结构化语义后验

## 7.1 弱监督来源

建议第一版来源：

| 数据域 | 来源 |
|---|---|
| MIMIC | CheXpert labeler、NegBio、RadGraph assertion、Chest ImaGenome |
| CheXpert Plus | CheXpert、CheXbert、RadGraph |
| 可选 | 冻结 LLM parser，仅作 ablation，不作唯一来源 |

禁止：

- 根据图像 test 结果修改解析规则；
- 将模型预测作为无审计 gold；
- 将 blank 直接填充为 absent。

## 7.2 来源混淆矩阵

对每个来源 \(s\)、字段 \(f\)，在人工 gold 上估计：

\[
M_{sf}(a,b)
=
P(\tilde z=a\mid z=b).
\]

对于低频 finding，采用 pooled-to-finding shrinkage：

\[
\widehat M_{sfc}
=
\rho_c\widehat M^{\rm local}_{sfc}
+
(1-\rho_c)\widehat M^{\rm pooled}_{sf},
\]

\[
\rho_c=\frac{n_c}{n_c+\kappa}.
\]

第一版固定 \(\kappa=50\)，不在图像验证集上调节。

## 7.3 校准的 log-opinion pool

每个来源通过混淆矩阵转换成状态概率 \(p_s(y)\)。

融合后验：

\[
q_{icf}(y)
=
\operatorname{softmax}
\left[
\log \pi_{cf}(y)
+
\sum_{s\in\mathcal S_i}
\alpha_{sf}\log p_s(y)
\right].
\]

约束：

\[
\alpha_{sf}\ge0,\qquad
\sum_s\alpha_{sf}=1.
\]

这样可减少高度相关 parser 被重复计数造成的过度自信。

\(\alpha\) 仅在报告 gold 的 calibration fold 上，以 NLL 最小化拟合；使用五折交叉验证报告性能。

## 7.4 可靠性分数

对融合后验做 temperature calibration：

\[
q^{(T)}(y)
=
\operatorname{softmax}\left(\frac{\log q(y)}{T_f}\right).
\]

样本—概念—字段可靠性：

\[
r_{icf}
=
\max_y q^{(T)}_{icf}(y).
\]

同时保存：

- normalized entropy；
- source coverage；
- source agreement；
- posterior margin；
- missingness pattern。

这些变量用于审计，不额外训练复杂网络。

## 7.5 监督 manifest

每一条监督必须保存：

- patient_id/study_id/image_id；
- source outputs；
- source versions；
- ontology version；
- posterior vector；
- reliability；
- missing mask；
- parser model hash；
- report hash；
- image hash；
- calibration artifact hash。

标签模型必须在视觉训练前冻结并生成 canonical SHA-256。

---

# 8. 方法：字段锚定结构化语义蒸馏

## 8.1 冻结语义原型库

冻结文本教师 \(T\) 对固定模板编码。

对 concept \(c\)、字段 \(f\)、状态 \(y\)，定义 3–5 个预声明 canonical paraphrases：

例：

- “Pleural effusion is present.”
- “There is pleural fluid.”
- “The radiograph demonstrates pleural effusion.”

negative：

- “No pleural effusion is present.”
- “There is no pleural fluid.”

uncertain：

- “Pleural effusion is reported as uncertain.”
- “The report cannot confirm pleural effusion.”

原型：

\[
p_{cfy}
=
\operatorname{normalize}
\left(
\frac{1}{K}
\sum_{k=1}^{K}T(s^{(k)}_{cfy})
\right).
\]

要求：

- 模板在图像实验前冻结；
- 不根据下游表现改 prompt；
- 保存模板列表、tokenizer、teacher 权重 hash；
- teacher 选择基于文本任务，不使用图像 test。

第一版建议：

- primary teacher：原项目 Qwen2.5-1.5B-Instruct；
- sensitivity：一个医学文本 encoder；
- 若 teacher 替换改变结论，必须报告，不可挑最好 teacher。

## 8.2 视觉编码

\[
H_i=E_\theta(x_i)\in\mathbb R^{P\times d}.
\]

primary backbone：

- ViT-B/16；
- 224×224；
- 输入为单张 canonical frontal CXR。

## 8.3 字段锚定查询

保持总 query token 数与原 SPD 4×2 相同：8 tokens。

四个组：

1. concept/observation；
2. assertion；
3. anatomy；
4. global structured context。

对 concept \(c\)、field \(f\)：

\[
Q_{cf}
=
Q_f^{\rm learn}
+
W_f p_c,
\]

其中 \(p_c\) 是冻结 concept prototype。

跨注意力：

\[
Z_{icf}
=
\operatorname{CrossAttn}(Q_{cf},H_i,H_i).
\]

聚合：

\[
u_{icf}
=
\operatorname{Pool}(Z_{icf}).
\]

## 8.4 软后验预测

对 assertion state：

\[
\ell_{ic}(y)
=
\frac{
\operatorname{cos}(u_{ic,\rm assertion},p_{c,\rm assertion,y})
}{\tau}.
\]

\[
p_\theta(y\mid x_i,c)
=
\operatorname{softmax}(\ell_{ic}).
\]

训练目标：

\[
L_{\rm state}
=
\sum_{i,c}
m_{ic}
r_{ic}
\operatorname{KL}
\left(
q_{ic}
\Vert
p_\theta(\cdot\mid x_i,c)
\right).
\]

## 8.5 字段语义对齐

对于 concept、anatomy 和 context：

\[
L_{\rm field}
=
\sum_{i,c,f}
m_{icf}r_{icf}
\left[
1-\operatorname{cos}(u_{icf},t_{icf})
\right].
\]

其中 \(t_{icf}\) 是对应字段的冻结语义目标。

## 8.6 分解约束

沿用轻量正交/去相关项：

\[
L_{\rm dec}
=
\frac{1}{|\mathcal F|(|\mathcal F|-1)}
\sum_{f\ne g}
\left|
\operatorname{cos}
(
\bar A_{icf},
\bar A_{icg}
)
\right|^2.
\]

不将 attention map 解释为临床定位，只用于防止所有字段完全坍缩。

## 8.7 总目标

\[
L
=
L_{\rm state}
+
\lambda_{\rm field}L_{\rm field}
+
\lambda_{\rm dec}L_{\rm dec}.
\]

第一版固定：

- \(\lambda_{\rm field}=1.0\)；
- \(\lambda_{\rm dec}=0.01\)。

只允许在 20k pilot 上预声明的小网格：

- \(\lambda_{\rm field}\in\{0.5,1.0\}\)；
- \(\lambda_{\rm dec}\in\{0,0.01\}\)。

一旦 pilot 冻结，不再在外部数据上调整。

## 8.8 部署

训练结束后丢弃：

- 文本 teacher；
- prototype bank；
- label model；
- field queries/projector。

保留：

\[
E_\theta(x)
\]

作为通用 CXR encoder。

主部署指标：

- 参数量；
- 单图推理时间；
- GPU 显存；
- 是否需要文本；
- 是否需要额外解剖模型。

---

# 9. 训练设置

## 9.1 图像预处理

primary：

- frontal AP/PA；
- 224×224；
- DICOM/PNG 使用统一像素标准化；
- 不水平翻转；
- rotation ≤ 5°；
- scale crop 0.9–1.0；
- 轻度亮度/对比度扰动；
- 所有增强记录 seed。

不使用：

- 大角度旋转；
- aggressive crop；
- posterize；
- cutout；
- GFTM；
- 可能改变 laterality 的增强。

## 9.2 优化器

primary：

- AdamW；
- backbone LR \(2\times10^{-5}\)；
- query/projector LR \(1\times10^{-4}\)；
- weight decay 0.05；
- warmup 5%；
- cosine decay；
- BF16；
- gradient clipping 1.0；
- effective batch size 64 或 128；
- 20 epochs。

训练预算以“有效 study 数 × epoch”锁定，所有受控 baseline 相同。

## 9.3 概念采样

每个 study 采 4–6 个概念：

- 50% 从有监督/被提及概念均匀采样；
- 50% 使用 inverse-square-root frequency；
- 不使用原 V11 类逆频率 loss；
- 同一 sampling policy 用于所有结构化 baseline。

## 9.4 checkpoint 选择

预训练 checkpoint 只按：

> MIMIC validation reliability-weighted structured NLL 最低

选择。

禁止：

- 根据 CheXpert、NIH、VinDr、PadChest AUC 选择预训练 checkpoint；
- F1 和 AUC 分别挑不同 step；
- 多个 checkpoint 拼主表。

下游 probe checkpoint 以目标 dataset validation NLL 最低选择，并一次性报告所有指标。

---

# 10. Baseline 设计

## 10.1 同数据、同骨干、同预算的受控 baseline

| ID | 方法 | 目的 |
|---|---|---|
| B0 | ImageNet ViT-B | 常规初始化 |
| B1 | MAE on same MIMIC images | 无语言自监督 |
| B2 | Raw-report image-text contrastive | 原始报告监督 |
| B3 | Hard 12-label BCE | 硬结构标签 |
| B4 | 原 A+UMS | 结构化 JSON 监督 |
| B5 | 原 A+UMS+SPD 4×2 | 原 MICCAI 稳定核心 |
| B6 | Fused soft posterior + original SPD | 只测试标签融合 |
| B7 | Hard UMS + field-anchored queries | 只测试新架构 |
| B8 | RCSD-CXR full | 完整方法 |

B0/B1 若使用公开权重，需增加“same-data MAE”或明确分离公开权重块。

## 10.2 公开强模型统一评估

在权重和许可可用时，统一 frozen probe：

- BioViL；
- CheXzero；
- MedKLIP；
- KAD；
- Ark+；
- EVA-X；
- Rad-DINO/RadJEPA；
- BiomedCLIP。

不直接复制论文数字；必须使用本项目统一 split、预处理和 probe。

分成两块报告：

1. controlled same-data；
2. off-the-shelf foundation encoders。

---

# 11. 下游评价协议

## 11.1 Linear probe

- backbone 完全冻结；
- 线性分类层；
- 3 seeds；
- identical patient subsets；
- fixed optimizer；
- checkpoint by validation NLL；
- primary metrics：macro AUROC、macro AUPRC；
- secondary：macro F1、micro F1、NLL、Brier、ECE。

## 11.2 Full fine-tuning

仅对最终 4 个模型：

- ImageNet；
- strongest public baseline；
- original SPD；
- RCSD-CXR。

报告：

- 100% labeled；
- 10% labeled；
- 相同 training budget；
- layer-wise LR decay 固定。

## 11.3 Low-data

CheXpert patient-level：

- 1%；
- 5%；
- 10%；
- 25%；
- 100%。

每个比例建立 3 个固定 patient subset seeds；所有模型共享。

主假设：

> RCSD 对 1%/5%/10% 的增益应大于 100%。

## 11.4 Cross-domain frozen transfer

从 CheXpert train 训练的 probe，不重新训练，直接测试：

- NIH；
- VinDr；
- PadChest。

只计算预声明的共同标签。

## 11.5 Target-specific probe

在 NIH、VinDr、PadChest 各自 train/validation 上训练 probe，test 上评价，衡量表示可适配性。

## 11.6 Calibration

- temperature scaling 仅在 validation；
- 15 equal-mass bins；
- ECE；
- adaptive ECE；
- Brier；
- NLL；
- calibration slope/intercept。

## 11.7 Subgroup

CheXpert Plus/外部 metadata 允许时：

- sex；
- age：<40、40–64、≥65；
- AP/PA；
- inpatient/outpatient（若合法且完整）；
- institution/domain。

只有每 subgroup 每 finding 正负例均 ≥50 时报告该 finding AUC。

报告：

- worst-group AUC；
- max-min gap；
- subgroup calibration gap。

## 11.8 Semantic representation analysis

不使用仅凭 UMAP 的定性结论。

定量分析：

1. visual class centroid 与 frozen semantic prototype 的距离相关；
2. pairwise semantic distance vs visual centroid distance 的 Spearman；
3. rare/common finding 的 centroid margin；
4. class retrieval mAP；
5. reliability decile 与 downstream correctness 的关系。

---

# 12. 统计分析

## 12.1 基本统计

- 所有 test CI 按患者 bootstrap；
- 10,000 replicates；
- 同一患者的多图作为一个 cluster；
- 模型对比使用 paired bootstrap；
- 主要 endpoint 报 95% CI。

## 12.2 多 seed

- 预训练核心模型 3 seeds；
- 报 mean ± SD；
- primary comparison 要求 3 个 seed 方向一致；
- 使用 hierarchical bootstrap 同时重采样 seed 和患者；
- 不用仅有 3 个 seed 的普通 t-test 作为唯一证据。

## 12.3 多重比较

- primary endpoint 不校正；
- per-finding exploratory tests 使用 Benjamini–Hochberg FDR；
- 所有主/次 endpoint 在实验前列入 registry。

## 12.4 非劣效界值

外部域非劣效界：

\[
\Delta{\rm AUROC}=-0.005.
\]

即 lower CI 不低于 -0.5 percentage points。

---

# 13. 分阶段实验队列

## Phase 0：项目分离与旧方法复现

| ID | 实验 | 输出 |
|---|---|---|
| P0-01 | 创建独立 paper1 目录/branch | 独立代码入口 |
| P0-02 | 数据 lineage 和历史 test exposure ledger | audit 表 |
| P0-03 | 找回原 V12 真实 config/checkpoint | 4×2 SPD 证明 |
| P0-04 | 原 A+UMS 重跑 ×3 seeds | legacy reproduction |
| P0-05 | 原 A+UMS+SPD 重跑 ×3 seeds | legacy reproduction |
| P0-06 | CheXpert/NIH 历史指标复算 | 单 checkpoint 结果 |

## Phase 1：标签模型

| ID | 实验 | 输出 |
|---|---|---|
| L1-01 | 统一 12 finding ontology | mapping 表 |
| L1-02 | 运行/导入所有弱来源 | source manifest |
| L1-03 | RadGraph gold 5-fold 单来源评价 | 单来源表 |
| L1-04 | majority/static/fused posterior 比较 | 融合表 |
| L1-05 | temperature calibration | ECE/NLL |
| L1-06 | CheXpert expert val 可靠性分层 | reliability 图 |
| L1-07 | anatomy gold 与 coverage | anatomy gate |
| L1-08 | 冻结 posterior manifest | SHA lock |

## Phase 2：最小可学习性

| ID | 实验 | 数据 | 目的 |
|---|---|---|---|
| M2-01 | 256-row high-confidence overfit | real subset | 实现正确性 |
| M2-02 | 5k study run | MIMIC train | loss/gradient |
| M2-03 | 20k hard UMS+SPD | pilot | baseline |
| M2-04 | 20k soft posterior+SPD | pilot | 标签贡献 |
| M2-05 | 20k hard+field query | pilot | 架构贡献 |
| M2-06 | 20k full RCSD | pilot | 完整方法 |
| M2-07 | 2-seed 50k confirmation | MIMIC | 冻结超参 |

## Phase 3：MIMIC-only 主实验

| ID | 方法 | seeds |
|---|---|---:|
| F3-01 | Raw report contrastive | 3 |
| F3-02 | Hard 12-label BCE | 3 |
| F3-03 | A+UMS | 3 |
| F3-04 | A+UMS+SPD | 3 |
| F3-05 | Soft posterior+SPD | 3 |
| F3-06 | Hard+field queries | 3 |
| F3-07 | Full RCSD | 3 |

MAE/ImageNet/公开模型按统一 probe 评价。

## Phase 4：下游

| ID | 实验 |
|---|---|
| D4-01 | CheXpert linear probe |
| D4-02 | CheXpert full fine-tuning |
| D4-03 | 1/5/10/25/100% low-data |
| D4-04 | CheXpert → NIH frozen transfer |
| D4-05 | CheXpert → VinDr frozen transfer |
| D4-06 | CheXpert → PadChest frozen transfer |
| D4-07 | NIH target-specific probe |
| D4-08 | VinDr target-specific probe |
| D4-09 | PadChest target-specific probe |
| D4-10 | calibration |
| D4-11 | subgroup |
| D4-12 | semantic topology analysis |

## Phase 5：Scale track

仅在主方法通过 G5 后：

| ID | 实验 |
|---|---|
| S5-01 | original SPD，MIMIC+CheXpert Plus |
| S5-02 | full RCSD，MIMIC+CheXpert Plus |
| S5-03 | source-held-out generalization |
| S5-04 | teacher sensitivity |
| S5-05 | second backbone reduced-scale |

## Phase 6：最终锁定

- freeze code；
- freeze config；
- freeze manifests；
- freeze prototype bank；
- freeze label posterior；
- freeze statistics plan；
- final test one-time release；
- auto-build tables。

---

# 14. 闸门与停止规则

> 这些是本项目的预声明决策阈值，不是期刊官方要求。

## G0：数据资格闸门

必须全部通过：

- train/val/test 患者交集 = 0；
- image SHA-256 overlap = 0；
- CheXpert val/test 与 CheXpert Plus pretrain overlap = 0；
- 100% manifest 行有来源和 split provenance；
- 失效图像在 lock 前排除并有原因；
- test 未被训练/选模代码访问；
- 每个 study 只有一个 canonical frontal 输入。

失败动作：禁止训练。

## G1：旧方法复现闸门

要求：

- 真实 SPD config 确认为 4 groups × 2 tokens；
- A+UMS 和 A+UMS+SPD 完整重跑；
- 历史 CheXpert/NIH 指标在 ±1.0 pp 内，或差异由明确协议变更解释；
- 三个 seed 中 SPD 相对 A+UMS 至少两个方向为正；
- 指标全部来自同一 validation-NLL-selected checkpoint。

失败动作：暂停新方法；先解决历史不可复现问题。

## G2：标签模型闸门

### 报告 gold

- fused posterior macro-F1 ≥ best single source +1.0 pp；
- NLL 相对 best source 至少降低 5%；
- ECE ≤0.05，或相对 best source 降低 ≥20%；
- present/absent/uncertain 三类均不能完全坍缩。

### 图像专家标签外推

在未参与拟合的 CheXpert expert validation：

- label correctness detection AUROC ≥0.65；
- reliability top quartile 与 bottom quartile 的标签一致率差 ≥10 pp；
- reliability decile correctness 基本单调；
- 至少 8/12 findings 满足质量和覆盖标准。

### Anatomy

- macro-F1 ≥0.75；
- 至少 6 findings coverage ≥30%。

失败动作：

- 若 assertion 融合失败：停止 RCSD 主方法，使用最佳单来源；不强行写可靠性创新；
- 若 anatomy 失败：删除 anatomy 主组件，转 supplement；
- 不新增 parser 无限补救。

## G3：最小可学习性闸门

256-row 高可靠 subset：

- train state accuracy ≥0.98；
- loss 相对初始降低 ≥80%；
- 每个 eligible field 都有非零梯度；
- balanced audit set 中任何一个 state 的预测占比不得 >95%。

5k studies：

- validation NLL 相对 uniform/initial 至少降低 20%；
- 无 NaN/Inf；
- query groups 不完全相同；
- 数据加载和患者分组稳定复现。

失败动作：仅允许修实现；不改科学主张。

## G4：20k pilot 闸门

Full RCSD vs original SPD：

- structured validation NLL 相对降低 ≥3%；
- CheXpert development macro-AUROC 提升 ≥0.5 pp；
- ECE 不恶化超过 0.01；
- 不得有超过 2 个 finding 各下降 >2 pp；
- soft-label-only 和 architecture-only 结果能区分贡献。

失败动作：

- 若 soft-label-only 有效而 full 无额外收益：删除新架构，论文改为更简单的可靠性监督方法；
- 若都无效：停止全量训练。

## G5：全量单 seed 晋级闸门

相对 original SPD：

### Primary CheXpert expert endpoint

满足任一：

- paired bootstrap ΔAUROC lower 95% CI > 0；或
- mean ΔAUROC ≥1.0 pp 且预声明检验显著。

### 外部域

- NIH/VinDr/PadChest 中至少两个 ΔAUROC ≥0.5 pp；
- 任何外部域 lower CI 不得低于 -0.5 pp；
- 10% low-data 至少提高 1.0 pp；
- AUPRC 与 AUROC 方向不能严重冲突。

失败动作：不进入 3-seed full scale，不做更多 teacher/backbone。

## G6：最终论文闸门

3 seeds：

- primary ΔAUROC 三个 seed 方向一致；
- mean ΔAUROC ≥0.8 pp；
- hierarchical 95% CI >0；
- 至少两个外部数据 CI >0；
- 其余外部数据达到非劣效；
- low-data 1% 或 10% 提升 ≥1.5 pp；
- ECE/Brier 至少一项改善，另一项非劣；
- full 方法显著优于 soft-only 和 architecture-only 中较强者。

若 full 不优于 simpler variant：

> 删除无贡献组件，以最简单有效版本投稿。

## G7：TMI/MIA 定位闸门

### TMI-ready

- G0–G6 全通过；
- 技术实现完整；
- 多 seed；
- ≥3 外部域；
- calibration/subgroup；
- 公开代码和锁定 protocol。

### MIA-ready 增强

额外需要：

- label model 更深入的误差与理论分析；
- 独立结构化报告 gold（例如 LUNGUAGE）；
- 长尾/rare finding 评价；
- 第二 backbone 或一个空间任务迁移；
- semantic geometry 与数据效率机制分析；
- 明确讨论何时可靠性监督失败。

若增益只存在于自动报告标签、不存在于专家图像标签：

> 不按 TMI/MIA 方法成功投稿。

---

# 15. 主文必须制作的表格

## Table 1：数据、独立性和使用角色

列：

- dataset/resource；
- institution/country；
- patients；
- studies；
- images；
- reports；
- annotation type；
- role；
- derived from；
- previous test exposure；
- final eligibility。

目的：证明 15 个资源不是 15 个独立域，并明确无泄漏。

## Table 2：结构化监督质量

行：

- CheXpert；
- NegBio；
- CheXbert；
- RadGraph；
- majority；
- static average；
- RCSD posterior。

列：

- coverage；
- assertion macro-F1；
- present F1；
- absent F1；
- uncertain F1；
- NLL；
- ECE；
- image-label agreement；
- reliability correctness AUROC。

这是本论文最关键的方法表之一。

## Table 3：Controlled MIMIC-only 主结果

行：

- ImageNet；
- MAE；
- raw report contrastive；
- hard BCE；
- A+UMS；
- A+UMS+SPD；
- soft posterior+SPD；
- hard+field queries；
- full RCSD。

列：

- CheXpert LP 5-label AUROC；
- CheXpert LP 12-label AUROC；
- macro AUPRC；
- 10% LP AUROC；
- full FT AUROC；
- NLL；
- ECE；
- params at inference。

必须 mean±SD，3 seeds。

## Table 4：跨机构泛化

行：选定核心模型。

列：

- NIH common-label AUROC/AUPRC；
- VinDr mapped AUROC/AUPRC；
- PadChest mapped AUROC/AUPRC；
- worst-domain AUROC；
- average rank；
- non-inferiority pass。

VinDr 标注 previously inspected。

## Table 5：低数据效率

行：核心模型。

列：

- 1%；
- 5%；
- 10%；
- 25%；
- 100%。

每格：

- macro AUROC；
- 可在括号中给 AUPRC。

推荐主文展示 AUROC，AUPRC 放 supplement。

## Table 6：单变量消融

严格递进：

1. hard UMS + plain projector；
2. hard UMS + SPD；
3. soft posterior + SPD；
4. reliability weighting + SPD；
5. + field anchors；
6. + anatomy；
7. + global context；
8. full。

列：

- label-model NLL；
- CheXpert AUROC；
- NIH AUROC；
- 10% AUROC；
- ECE；
- compute。

## Table 7：Scale track（若通过）

| Pretrain data | original SPD | RCSD |
|---|---:|---:|
| MIMIC |  |  |
| CheXpert Plus |  |  |
| MIMIC + CheXpert Plus |  |  |

同时报告：

- 数据量；
- steps；
- GPU hours；
- cross-source worst performance。

## Table 8：Calibration 和 subgroup（可放 supplement）

列：

- overall；
- sex groups；
- age groups；
- AP/PA；
- worst group；
- gap。

---

# 16. 主文图形

## Figure 1：方法总览

左：

- multi-source report labels；
- missing/conflict。

中：

- calibrated posterior；
- reliability。

右：

- field-anchored queries；
- frozen prototype bank；
- ViT-only deployment。

必须明确显示 LLM 只在原型构建阶段使用。

## Figure 2：标签可靠性

四个 panel：

1. source confusion matrices；
2. reliability calibration curve；
3. correctness by reliability decile；
4. 三个典型冲突报告案例。

## Figure 3：低数据曲线

x：标注比例。  
y：macro AUROC/AUPRC。  
模型：ImageNet、original SPD、RCSD。

## Figure 4：跨域 forest plot

每个外部域报告 RCSD−SPD 的 paired ΔAUROC 和 95% CI。

## Figure 5：语义结构分析

- teacher semantic distance vs visual centroid distance；
- common/tail findings；
- 不用 UMAP 作为主要证据。

## Figure 6：训练与部署开销

- pretraining cost；
- inference params；
- latency；
- LLM removed。

---

# 17. Supplementary 必须表格

- S1：12 finding ontology 与所有数据集 mapping；
- S2：anatomy ontology；
- S3：每个 source 的版本和 hash；
- S4：全部 confusion matrices；
- S5：per-finding label quality；
- S6：per-finding CheXpert AUROC/AUPRC；
- S7：per-finding NIH；
- S8：per-finding VinDr；
- S9：per-finding PadChest；
- S10：每个 seed 结果；
- S11：训练 hyperparameters；
- S12：probe hyperparameters；
- S13：teacher/template sensitivity；
- S14：query 数/field 数 sensitivity；
- S15：224 vs 384；
- S16：second backbone；
- S17：missingness policy；
- S18：source ablation；
- S19：calibration；
- S20：subgroup；
- S21：compute/carbon；
- S22：排除样本和失败原因；
- S23：data leakage/hash audit；
- S24：统计检验和 FDR；
- S25：历史结果与新复现差异；
- S26：可选 CT appendix。

---

# 18. 论文 Results 的叙事顺序

严格按证据链写：

1. **标签来源确实有冲突，且可靠性可校准；**
2. **可靠性后验优于单一 parser；**
3. **软后验改善原 SPD；**
4. **字段锚定提供额外但较小的独立增益；**
5. **优势在低数据更强；**
6. **优势跨机构保持；**
7. **推理成本与普通 ViT 相同；**
8. **失败类别和限制被明确列出。**

禁止：

- 先展示最好外部结果，再回头挑方法；
- 用单个 finding 的巨大提升掩盖其他下降；
- 用不同 checkpoint 报不同指标；
- 用 CT/病理简单任务放大“通用性”。

---

# 19. 论文结构

## Abstract

四段逻辑：

1. 报告监督丰富但噪声、缺失和冲突；
2. 提出可靠性后验 + 字段蒸馏；
3. 多机构、多 seed、低数据和 calibration 评价；
4. 训练后得到纯视觉编码器。

## 1 Introduction

- 报告是自然监督；
- 自动标签并非图像真值；
- 结构化方法仍把 parser 输出当硬标签；
- 提出两个组件；
- 列贡献。

## 2 Related Work

- CXR self-supervised / VLP；
- knowledge/graph structured report；
- heterogeneous labels / noisy supervision；
- deployable visual encoders。

## 3 Method

- ontology；
- label model；
- posterior；
- prototype bank；
- query distillation；
- objectives；
- deployment。

## 4 Data and Experimental Protocol

- lineage；
- splits；
- baselines；
- metrics；
- statistics；
- gates。

## 5 Results

按第 18 节顺序。

## 6 Discussion

必须讨论：

- report semantics 不等于 image truth；
- reliability 只能降低风险，不能制造缺失真值；
- anatomy 是报告关系，不是像素定位；
- teacher 可能带有语义偏差；
- 只覆盖胸片；
- previously inspected external test 的限制；
- 对罕见 finding 的样本限制。

## 7 Conclusion

只声称：

> 可靠性校准的结构化报告监督可在不增加推理成本的情况下，改善胸片视觉编码器的迁移和校准。

不声称：

- 临床诊断替代；
- 因果证据；
- 视觉 uncertainty；
- 通用医学多模态基础模型。

---

# 20. 代码与仓库建议

不要在 BiVES active mainline 上继续堆代码。

建议新目录：

```text
papers/rcsd_cxr/
rcsd_cxr/
  ontology.py
  source_schema.py
  label_model.py
  calibration.py
  prototype_bank.py
  field_queries.py
  model.py
  losses.py
  dataset.py
  metrics.py
  locks.py
configs/rcsd_cxr/
scripts/
  build_rcsd_source_manifest.py
  fit_rcsd_label_model.py
  audit_rcsd_posterior.py
  build_rcsd_prototypes.py
  train_rcsd_cxr.py
  evaluate_rcsd_probe.py
  evaluate_rcsd_external.py
  lock_rcsd_release.py
tests/
  test_rcsd_label_model.py
  test_rcsd_missingness.py
  test_rcsd_split_lock.py
  test_rcsd_prototype_lock.py
  test_rcsd_checkpoint_selection.py
```

legacy VIVID 代码保持只读；复制最小必要代码后再重构。

---

# 21. 必须实现的自动审计

1. patient overlap；
2. study/image hash overlap；
3. CheXpert Plus 与 CheXpert val/test overlap；
4. MIMIC derived resource split alignment；
5. report hash/source version；
6. source missingness；
7. posterior sums to one；
8. posterior finite；
9. reliability within [0,1]；
10. template/prototype hash；
11. checkpoint config binding；
12. 单 checkpoint 全指标；
13. test evaluator 独立入口；
14. final table 自动生成；
15. seed 完整性。

---

# 22. 最小可发表版本与顶刊扩展版本

## 22.1 最小方法学闭环

必须完成：

- MIMIC controlled pretraining；
- CheXpert expert validation；
- NIH；
- VinDr；
- 3 seeds；
- label quality；
- low-data；
- calibration；
- ablation。

若结果扎实，可以形成 TMI submission。

## 22.2 MIA 增强

增加：

- PadChest/CXR-LT 长尾；
- LUNGUAGE text gold；
- MIMIC+CheXpert Plus scale track；
- second backbone；
- semantic topology；
- source-held-out；
- failure analysis；
- 一个空间任务 transfer。

不建议用更多模块替代这些实证增强。

---

# 23. 立即执行清单

## 第一批：不训练

- [ ] 创建 paper1 独立目录；
- [ ] 锁定 12 finding ontology；
- [ ] 完成 MIMIC/CheXpert Plus/CheXpert overlap audit；
- [ ] 冻结 CheXpert test/CheXlocalize test；
- [ ] 找到原 4×2 SPD config；
- [ ] 导出历史 checkpoint/config/metrics 对照；
- [ ] 下载并资格审计 PadChest；
- [ ] 下载并资格审计 LUNGUAGE；
- [ ] 建立 weak-source version ledger；
- [ ] 写 `PRIMARY_ENDPOINTS.md`。

## 第二批：标签模型

- [ ] 单来源 gold 评价；
- [ ] posterior fusion；
- [ ] reliability calibration；
- [ ] CheXpert expert val 外推检查；
- [ ] G2 决策；
- [ ] 冻结 posterior manifest。

## 第三批：旧方法复现

- [ ] A+UMS ×3；
- [ ] A+UMS+SPD ×3；
- [ ] CheXpert/NIH 复算；
- [ ] G1 决策。

## 第四批：pilot

- [ ] 256-row overfit；
- [ ] 5k；
- [ ] 20k ablation；
- [ ] 50k 2-seed；
- [ ] G3/G4 决策；
- [ ] 冻结超参。

## 第五批：full

- [ ] MIMIC-only core runs；
- [ ] downstream；
- [ ] 3 seeds；
- [ ] G5/G6；
- [ ] scale track；
- [ ] final locked test。

---

# 24. 最终停止条件

出现任一情况，应停止把该方案包装成 TMI/MIA 方法成功：

1. 融合 posterior 不优于最佳单 parser；
2. reliability 无法预测标签正确性；
3. full 方法不优于 original SPD；
4. 改善仅出现在自动报告标签测试；
5. 专家图像标签不改善；
6. 外部域出现系统性下降；
7. 只有一个 seed 成功；
8. full 不优于 simpler soft-label variant；
9. 需要通过查看 test 继续调参才能成立。

最好的简化版本优先于复杂完整版本。

---

# 25. 参考工作的最低覆盖

正式 proposal 和论文至少讨论：

1. CheXzero；
2. BioViL；
3. MedKLIP；
4. KAD；
5. Image-Graph Contrastive Pretraining；
6. DeViDe；
7. GK-MVLP；
8. UniChest；
9. Ark+；
10. EVA-X；
11. Rad-DINO/RadJEPA；
12. VisualCheXbert；
13. RadGraph；
14. CheXpert Plus；
15. PadChest/CXR-LT。

---

# 26. 最终建议

本项目不应直接开始全量训练。

正确顺序是：

1. 先复现原 VIVID 稳定核心；
2. 再证明新的可靠性后验确实更好；
3. 再做 20k 小规模单变量实验；
4. 只有标签和模型两条证据同时通过，才启动 MIMIC 全量；
5. 最后才做多源扩展和顶刊增强。

方法贡献应保持：

> **更可靠的监督 + 更明确的蒸馏结构，而不是更多模块。**

