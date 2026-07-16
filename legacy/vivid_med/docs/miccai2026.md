下面是我的整体判断：**这篇不是“没戏”的论文，而是一个有潜力但论证链断了的 MICCAI 方法论文**。两个 reviewer 给了 **3 / weak reject**，而且都说问题主要是实验验证不足；真正致命的是第 3 个 reviewer 给了 **2 / reject**，把问题上升到了 **novelty 低 + evaluation weak**。所以这篇值得继续做，但前提是你不能只改写作或补一两个小实验，必须按“重建证据链”的方式改。论文和 review 我分别参考了你上传的稿件与审稿意见。

------

# 1. 一句话复盘

你的 paper 当前最大的问题不是方法本身，而是：

**proposal 讲得太大，实验支撑太窄，writing 又把 claim 写得太满，导致 reviewer 不相信你的提升来自 VIVID-Med 本身。**

更具体地说，VIVID-Med 的核心 idea——**用 frozen LLM + UMS JSON + SPD 去训练 deployable ViT**——是有吸引力的。reviewer 也认可 SPD、teacher forcing、结构化监督这些点。Reviewer #1 明确认为 SPD 设计不错，teacher forcing 合理；Reviewer #2 也认可 frozen LLM target space 和 SPD orthogonality 的思路。

但是审稿人质疑的是：**你现在的实验还不能证明“LLM 的语义空间”真的起了作用，也不能证明比 BiomedCLIP 强是因为方法好，而不是因为 CheXpert domain/label advantage。** Reviewer #1 和 #2 都点名了 data-matched comparison 缺失，Reviewer #2 还点名 LLM role 没有被直接测试。

------

# 2. 按严重程度分级

我建议用四级来理解：

| 等级            | 含义                                 | 对录用影响            |
| --------------- | ------------------------------------ | --------------------- |
| **P0 致命问题** | 不修基本无法转 strong venue          | 会被认为 claim 不成立 |
| **P1 重大问题** | 修了能明显提高可信度                 | 影响评分从 2/3 到 4   |
| **P2 中等问题** | 影响 polish、reproducibility、说服力 | 影响 reviewer 信任    |
| **P3 小问题**   | 表达、图、措辞层面                   | 不单独致命            |

------

# 3. Proposal 层面的问题

## P0-1：中心 claim 没有被直接证明

你在 introduction 和 discussion 里反复强调：**pretrained LLM 提供 structured semantic space，临床概念之间的 proximity 反映真实临床相关性。** 稿件中也明确说 frozen LLM defines a stable target manifold，ViT 学会预测这个 structured output space。

但问题是：你目前的训练目标是 **teacher forcing 下预测 ground-truth UMS JSON tokens**。这会让 reviewer 产生一个非常自然的怀疑：

**LLM 真的在提供医学语义吗？还是只是一个固定 decoder / token prediction loss？**

Reviewer #2 已经直接指出：你只比较了 free-text vs UMS-JSON，没有测试不同 LLM、随机 LLM、无 LLM，因而无法证明 LLM 在做 meaningful semantic work。

这是当前最核心的断点。

你需要把 claim 从：

> “LLM provides a clinical semantic manifold.”

变成可验证命题：

> “A pretrained frozen language decoder provides more useful gradients than random / non-semantic / no-LLM decoders under the same UMS target.”

然后通过实验支撑。

------

## P0-2：novelty positioning 不够，尤其被 R3 抓住了

Reviewer #3 认为你的方法和两个 prior works 太像，并且说 novelty low。它点名：

- **Frozen Transformers in Language Models Are Effective Visual Encoder Layers**
- **Beyond Text: Frozen Large Language Models in Visual Signal Comprehension**

这不是一个小问题。因为 MICCAI 方法论文如果被认为“只是换到 medical domain”，分数很难上 4。

我查了一下这两个工作：Pang et al. 的工作强调 frozen LLM transformer block 可以作为 visual encoder layer，直接处理 visual tokens，并在多类视觉任务上提升性能；它的核心是把 frozen LLM block 当视觉编码层。([arXiv](https://arxiv.org/abs/2310.12973)) Zhu et al. 的 CVPR 2024 工作则把图像视为一种 linguistic entity，通过 Vision-to-Language Tokenizer 把图像转成 LLM vocabulary 派生的离散 token，让 LLM 理解视觉信号。([arXiv](https://arxiv.org/abs/2403.07874))

你和它们确实不完全一样。你的差异应该是：

**不是用 LLM block 当视觉 encoder，也不是把 image tokenize 成 language 给 LLM 做理解，而是用 frozen LLM 的 next-token loss 作为结构化医学标签监督，把最终可部署的 ViT backbone 蒸馏出来。**

但是这个差异在当前稿子里没有被充分钉牢。你主要对比的是 ViTP，但没有系统解释和 R1/R2 的区别。ViTP 本身也很相关，因为它同样把 ViT 嵌进 vision-language model 做 domain-specific visual instruction pretraining。([arXiv](https://arxiv.org/abs/2509.17562))

所以 proposal 需要重写，不然 R3 这种 reviewer 还会继续打 novelty。

------

## P0-3：“500× less data”这个 claim 会伤害可信度

摘要里写 VIVID-Med 用 CheXpert linear probing 达到 0.8588 macro-AUC，比 BiomedCLIP 高 6.65 points，同时使用 500× less data。

这个 claim 很容易被攻击，Reviewer #1 和 #2 都攻击了。原因是：

**BiomedCLIP 的 15M 是 diverse biomedical image-text pairs，而你的 30k 是 CheXpert domain-specific CXR，并且 CheXpert 还直接用于 evaluation benchmark。**

这不是严格的数据效率比较，而是：

- BiomedCLIP：大规模、多域、通用 biomedical VLP；
- VIVID-Med：小规模、目标域、结构化标签监督；
- evaluation：又在 CheXpert / CXR 上。

所以 reviewer 会觉得你把 **domain specificity** 伪装成了 **data efficiency**。Reviewer #1 明确说这个说法 misleading，Reviewer #2 也说缺少 apples-to-apples comparison。

这个 claim 建议直接降级，不要放在 abstract 主打。

------

## P1-1：“pretraining”定位有点危险

你说这是 medical ViT pretraining framework，但当前主数据是 **30k CheXpert CXR with labels/findings**，并且 CheXpert 也是主要 in-domain evaluation。论文写明 pretraining/evaluation 在 CheXpert，NIH 做 zero-shot transfer，CT 做 cross-modality。

这会让 reviewer 怀疑：

**这到底是 foundation pretraining，还是 CheXpert label-supervised representation learning？**

如果你要叫 pretraining，就需要更强的外部 generalization：

- pretrain on CheXpert, evaluate on MIMIC-CXR / PadChest / VinDr-CXR / NIH；
- 或 pretrain on MIMIC-CXR, evaluate on CheXpert / NIH；
- 或做 data scaling：3k / 10k / 30k / larger。

否则建议把定位收窄成：

> structured semantic supervision for data-efficient CXR representation learning

比 “deployable medical ViTs” 更稳。

------

## P1-2：UMS 的 clinical assumption 需要更谨慎

UMS 把 finding state 设成 present / absent / uncertain / null，并且 null 表示 finding 不可从图像判断；answerability mask 只对 answerable tokens 计算 loss。

这个想法不错，但 CheXpert label 里的 blank / unmentioned 不一定等价于 “not assessable from image”。有时候只是报告没提，有时候可能是默认 negative，有时候是 NLP labeler 没抽取到。你把它写成 “not assessable” 会有临床语义风险。

建议改成更保守的说法：

> null denotes unannotated or non-supervised fields under the available label extraction protocol.

不要说它医学上不可判断，除非你有额外 annotation 支撑。

------

# 4. 实验层面的问题

这是最需要补的部分。

## P0-1：缺少 data-matched baselines

当前主表里有 ImageNet supervised、MAE、DINOv3、BiomedCLIP、Random-mask proxy、VIVID-Med。VIVID-Med 在 CheXpert-12 上 0.8588 macro-AUC，在 NIH-8 上 0.7225 macro-AUC。

问题是这些 baseline 不能回答最关键的问题：

**在同样 30k CheXpert、同样 ViT-B、同样训练步数、同样 label supervision 下，VIVID-Med 是否真的更好？**

必须补以下 data-matched baselines：

| Baseline                                                    | 目的                                        |
| ----------------------------------------------------------- | ------------------------------------------- |
| **ViT-B + CheXpert multi-label BCE supervised pretraining** | 判断 UMS+LLM 是否优于普通 label supervision |
| **ViT-B + UMS multi-head classifier, no LLM**               | 判断收益来自 UMS schema 还是 LLM            |
| **ViT-B + free-text target + same frozen LLM**              | 你已有 free-text，但要写清楚完全同设置      |
| **ViT-B + random initialized frozen LM / random decoder**   | 判断 pretrained LLM 是否真的重要            |
| **ViT-B + CLIP-style text from same 30k labels/reports**    | 判断是否只是 text supervision baseline 不足 |
| **Q-Former same-token baseline without orthogonality**      | 判断 SPD 相对标准 Q-Former 的贡献           |

现在的 ablation 只有 free-text、no SPD、full、Q-Former proxy。这个方向是对的，但还不够。

------

## P0-2：LLM 作用必须被拆开

你现在声称 frozen LLM 是 structured semantic teacher，但没有证明它比 non-semantic decoder 更好。必须补一个 **LLM role ablation table**。

建议表格如下：

| Variant                                 | 预期回答的问题                         |
| --------------------------------------- | -------------------------------------- |
| Qwen2.5-1.5B frozen                     | full method                            |
| same architecture random weights frozen | pretrained language knowledge 是否重要 |
| small pretrained LM frozen              | size 是否重要                          |
| medical/scientific LM frozen, 如可用    | domain language knowledge 是否重要     |
| no LLM: MLP predicts field-state logits | JSON schema 本身是否足够               |
| shuffled field names / arbitrary tokens | field semantic names 是否重要          |
| blank image prefix / no visual prefix   | 模型是否真的使用 image                 |

尤其是最后一个很重要。因为 teacher forcing 下，LLM 可能部分依赖 target prefix 和 label prior，而不是 visual prefix。你需要报告：

- image prefix vs zero prefix 的 token NLL；
- state-token accuracy；
- per-finding AUC；
- visual token attention / gradient norm；
- 如果去掉图像 performance 大幅下降，才能证明 visual features 真被学到了。

------

## P0-3：外部泛化证据不够强

Reviewer #3 说 evaluation 不 comprehensive，只看到一个 in-domain 和一个 cross-dataset。 你的论文其实有 CT 表，但 reviewer 可能觉得它不够支撑主 claim。

当前 CT 结果的问题是：

- OrganAMNIST AUC 基本饱和，ImageNet 已经 0.9928，BiomedCLIP 0.9913，VIVID-Med 0.9969；这个提升看起来没有那么有信息量。
- LIDC-IDRI 上 VIVID-Med AUC 是 0.8413，BiomedCLIP 是 0.8465，AUC 并没有赢；你主要赢在 F1。

所以 “strong cross-modality generalization” 写得过强。Reviewer #2 也说 cross-modality results are weaker than presented。

建议转投版本把 CT 放成 supporting evidence，而不是主卖点。更重要的是补 CXR 外部泛化：

| Setting                                     | 推荐做法                                                     |
| ------------------------------------------- | ------------------------------------------------------------ |
| CheXpert → NIH                              | 保留，但写清楚是 CheXpert-trained head zero-shot transfer 还是重新 linear probe |
| CheXpert → PadChest / VinDr-CXR / MIMIC-CXR | 至少补一个外部 CXR 数据集                                    |
| MIMIC-CXR → CheXpert / NIH                  | 更有说服力，因为避免 CheXpert pretrain/eval 过近             |
| low-label regime                            | 1%、5%、10%、100% labels，看 data efficiency                 |

如果你的目标仍是 MICCAI / MIDL 级别，至少要有 **两个外部 CXR 数据集** 或者 **一个外部 CXR + 一个强 data scaling experiment**。

------

## P1-1：SPD 超参数敏感性缺失

Reviewer #1 和 #3 都点了这个。你的方法里固定：

- G = 4 groups；
- M = 2 tokens/group；
- λ_ortho = 0.01；
- field sampling k ∈ [4,6]；
- low-frequency oversampling p = 0.6。

但这些都是 central hyperparameters。必须补 sensitivity。

最低限度：

| Hyperparameter     | Values                    |
| ------------------ | ------------------------- |
| groups G           | 1, 2, 4, 8                |
| tokens/group M     | 1, 2, 4                   |
| λ_ortho            | 0, 1e-4, 1e-3, 1e-2, 1e-1 |
| sampled fields k   | 2, 4, 6, all              |
| tail sampling prob | 0, 0.3, 0.6, 0.9          |

不需要所有组合全跑，可以分组单变量 sweep。主文放核心表，appendix 放完整结果。

------

## P1-2：单 backbone 不够支撑“pretraining method”

Reviewer #3 明确说只测 ViT-B 不够，因为你宣传的是 pretraining technique。

最少补：

- ViT-S/16；
- ViT-B/16；
- 如果资源够，再加 ViT-L/16 或 ViT-B/32。

你不一定要每个 backbone 都跑所有 ablation，但 full method + 2 个关键 baseline 必须有。

------

## P1-3：训练成本分析缺失

Reviewer #2 指出：你强调 deployment lightweight，但训练时每一步跑 1.5B frozen LLM，成本并不小。

必须补一个 efficiency table：

| Method | Train params | Frozen params | Peak memory | Throughput | GPU-hours | Inference params | Inference FLOPs |
| ------ | ------------ | ------------- | ----------- | ---------- | --------- | ---------------- | --------------- |
|        |              |               |             |            |           |                  |                 |

你的优势应该重新表述为：

**training is heavier than standard supervised ViT pretraining, but the LLM is a one-time training teacher and is fully removed at deployment.**

不要只说 resource-friendly，否则 reviewer 会说你偷换 training cost 和 inference cost。

------

## P1-4：t-SNE claim 建议删除或量化

Reviewer #1 说 Figure 4 “tighter, more separable clusters” 不convincing。

t-SNE 这种图本来就很主观。建议：

- 主文删除这句强 claim；
- 或增加 quantitative embedding separability：
  - silhouette score；
  - Davies-Bouldin index；
  - linear separability；
  - class centroid distance；
  - kNN retrieval precision；
  - bootstrap CI。

如果没有显著优势，就把 t-SNE 放 appendix，当 qualitative visualization，不作为证据。

------

# 5. 写作层面的问题

## P0/P1：overclaim 太多

当前稿子里有几类词会让 reviewer 反感：

- “exceptional”
- “proving”
- “strong cross-modality”
- “robust zero-shot”
- “500× less data”
- “clinical semantic manifold”
- “substantially reducing inference costs”但没有训练成本对照

这些不是不能写，而是你现在证据还没支撑到这个程度。比如 Table 2 里 LIDC AUC 没赢 BiomedCLIP，OrganAMNIST AUC 又接近饱和，所以 “strong cross-modality generalization” 容易被认为夸大。

建议改成更稳的表达：

| 当前表达                                                | 建议表达                                                     |
| ------------------------------------------------------- | ------------------------------------------------------------ |
| outperforms BiomedCLIP with 500× less data              | achieves strong performance under domain-specific 30k-sample pretraining; direct data-efficiency comparison requires controlled baselines |
| proving orthogonal decomposition fundamentally improves | suggesting that orthogonal decomposition improves tail-class ranking |
| strong cross-modality generalization                    | encouraging cross-modality transfer, with gains strongest on OrganAMNIST F1 and mixed AUC results on LIDC |
| LLM inherently embeds clinical relatedness              | we hypothesize that pretrained LMs provide useful structured priors; we empirically test this through LLM ablations |

------

## P1：Related Work 要重写

你现在主要拿 ViTP 对比，但 reviewer #3 认为你漏了 R1/R2。转投稿必须增加一个 subsection：

**Frozen LLMs for visual representation learning.**

里面明确分三类：

1. **LLM as visual encoder layer**：例如 Pang et al. 把 frozen LLM transformer block 用作 visual encoder layer。你的方法不是把 LLM block 部署进视觉 encoder，而是用 LLM 作为 training-only structured decoder/teacher，最终只保留 ViT。([arXiv](https://arxiv.org/abs/2310.12973))
2. **Image-to-language tokenization for LLM comprehension**：例如 Beyond Text / V2T Tokenizer 把图像转成 LLM vocabulary 中的 token，让 LLM 理解视觉信号。你的目标不是让 LLM 直接理解图像，而是训练一个 standalone medical ViT。([arXiv](https://arxiv.org/abs/2403.07874))
3. **Visual instruction pretraining**：ViTP 把 ViT backbone 嵌入 VLM，用 visual instruction data 做 domain-specific pretraining；你需要强调 UMS、answerability masking、SPD、LLM-free deployment，以及最好加官方或近似 baseline。([arXiv](https://arxiv.org/abs/2509.17562))

------

## P1：Abstract 要重写

现在 abstract 把所有卖点都塞进去，但 claim 太强。建议转投版 abstract 逻辑改成：

1. 医学视觉 pretraining 里，label supervision 太离散，free text 太 noisy；
2. 我们提出 schema-constrained LLM-supervised training；
3. 关键不是“LLM 神奇懂医学”，而是 structured target + frozen decoder + decomposed visual prefix；
4. 我们通过 controlled baselines 验证：
   - UMS 是否有效；
   - pretrained LLM 是否优于 random/no LLM；
   - SPD 是否稳定；
   - data-matched baselines 是否公平；
5. deployment only keeps ViT。

也就是说，abstract 里不要先打 BiomedCLIP 500×，而要先打 **controlled evidence**。

------

# 6. 这篇值不值得继续补实验、修改、转投？

我的判断：**值得，但要有止损条件。**

## 值得继续的原因

第一，reviewer 没有说你的结果造假、方法完全不合理、医学方向无意义。相反，两个 reviewer 都认可方法 interesting / technically sound。Reviewer #1 说 SPD 设计好，Reviewer #2 说 frozen LLM target 和 SPD 有意思。

第二，两个 3 分 reviewer 的措辞是“fixable”。Reviewer #2 明确说 gaps are fixable but substantial。 这说明不是完全没救。

第三，当前分数结构是 **3, 3, 2**，这通常表示 paper 已经有基本兴趣点，但被一个 reviewer 的 novelty/evaluation concern 拉下去了。你转投时只要把 R3 类型的问题解决，提升空间比较大。

## 不值得继续的情况

如果你补完关键实验后出现下面结果，就不建议继续主打 VIVID-Med 这个故事：

1. **普通 CheXpert BCE supervised ViT 在同数据同 backbone 下接近或超过 VIVID-Med**；
2. **random frozen LM / no LLM 和 Qwen2.5 frozen LLM 差不多**；
3. **外部 CXR 数据集上没有稳定提升**；
4. **SPD 的 G、M、λ 稍微变动 performance 就崩**；
5. **去掉 visual prefix 后 token loss / downstream AUC 几乎不变**。

如果出现这些，说明真正起作用的可能只是 CheXpert label supervision 或 label prior，不是 LLM structured semantic pretraining。那就应该 pivot 成更窄的 paper，比如：

> structured label decomposition for long-tail CXR representation learning

而不是继续讲 frozen LLM semantic teacher。

------

# 7. 推荐的转投版本方案

## 版本目标

我建议你把转投稿改成：

> **VIVID-Med is a controlled study of schema-guided frozen-LM supervision for deployable CXR representation learning.**

不要再泛泛说 “medical ViTs” 或 “cross-modality foundation model”。先把 CXR 场景做扎实，CT 作为 secondary evidence。

------

## 新 contribution 建议

当前 contribution 是：

1. frozen-LLM distillation；
2. UMS；
3. SPD；
4. comprehensive evaluations。

建议改成：

1. **Schema-guided frozen-LM supervision**：将 CXR finding labels 转成 answerability-aware UMS，并通过 frozen LM prefix objective 训练 deployable ViT。
2. **Mechanistic validation of the LM teacher**：通过 pretrained/random/no-LM、field-shuffling、no-image-prefix 等 ablation 证明 LLM 与 visual prefix 的作用。
3. **Structured Prediction Decomposition**：用 orthogonality-regularized query groups 提高 long-tail finding ranking，并给出 sensitivity analysis。
4. **Controlled evaluation**：在 data-matched baselines、external CXR transfer、backbone scaling 和 efficiency 上验证。

这个改法会直接回应三个 reviewer。

------

# 8. 必补实验清单：按优先级

## Tier A：不做就不要转强 venue

### A1. Data-matched baseline table

主表建议这样设计：

| Method                     | Pretrain data | Supervision | LLM? | SPD?    | CheXpert AUC | NIH AUC | External CXR AUC |
| -------------------------- | ------------- | ----------- | ---- | ------- | ------------ | ------- | ---------------- |
| ViT-B from scratch BCE     | 30k CheXpert  | labels      | no   | no      |              |         |                  |
| ImageNet ViT-B + BCE       | 30k CheXpert  | labels      | no   | no      |              |         |                  |
| UMS classifier no LLM      | 30k CheXpert  | UMS         | no   | no      |              |         |                  |
| Frozen LM + free text      | 30k CheXpert  | text        | yes  | no/SPD  |              |         |                  |
| Frozen LM + UMS no SPD     | 30k CheXpert  | UMS         | yes  | no      |              |         |                  |
| Frozen LM + UMS + Q-Former | 30k CheXpert  | UMS         | yes  | qformer |              |         |                  |
| **VIVID-Med**              | 30k CheXpert  | UMS         | yes  | yes     |              |         |                  |

这张表是救命表。它解决 R1/R2 的 fairness concern。

------

### A2. LLM role ablation

主文单独一张表：

| Teacher             | Semantic field names? | Visual prefix?   | CheXpert AUC | NIH AUC | Token NLL |
| ------------------- | --------------------- | ---------------- | ------------ | ------- | --------- |
| Qwen2.5 frozen      | yes                   | yes              |              |         |           |
| Qwen2.5 frozen      | shuffled field names  | yes              |              |         |           |
| Qwen2.5 frozen      | yes                   | no / blank image |              |         |           |
| random LM frozen    | yes                   | yes              |              |         |           |
| small pretrained LM | yes                   | yes              |              |         |           |
| no LM classifier    | yes                   | yes              |              |         |           |

你需要证明：

- pretrained LM > random LM；
- semantic field names > shuffled names；
- visual prefix > blank image；
- VIVID-Med > no-LM UMS classifier。

只要这四个成立，Reviewer #2 的核心质疑基本就能被解决。

------

### A3. SPD sensitivity

主文放 compact 结果：

| Variant | CheXpert AUC | NIH AUC | Tail AUC | Attention diversity |
| ------- | ------------ | ------- | -------- | ------------------- |
| G=1     |              |         |          |                     |
| G=2     |              |         |          |                     |
| G=4     |              |         |          |                     |
| G=8     |              |         |          |                     |
| λ=0     |              |         |          |                     |
| λ=1e-3  |              |         |          |                     |
| λ=1e-2  |              |         |          |                     |
| λ=1e-1  |              |         |          |                     |

重点不是每个都最好，而是要证明：

**G=4, λ=0.01 不是拍脑袋；performance 对合理范围稳定；λ=0 明显弱于合适 λ。**

------

### A4. 至少一个新的外部 CXR 数据集

当前 NIH 不够。建议补一个外部 CXR：

- MIMIC-CXR；
- PadChest；
- VinDr-CXR。

最好报告两种 setting：

1. **CheXpert-trained linear head direct transfer**；
2. **frozen backbone linear probe on target dataset**。

这样可以区分 domain transfer 和 representation quality。

------

## Tier B：强烈建议做

### B1. Backbone scaling

至少 ViT-S 和 ViT-B。表格：

| Backbone        | Params | BCE baseline | VIVID-Med | Gain |
| --------------- | ------ | ------------ | --------- | ---- |
| ViT-S           |        |              |           |      |
| ViT-B           |        |              |           |      |
| ViT-L, optional |        |              |           |      |

如果 ViT-S 也有效，可以回应 deployable；如果 ViT-L 更有效，可以回应 pretraining scalability。

------

### B2. Training + inference cost

表格必须补：

| Method | Training GPU memory | Training throughput | GPU-hours | Inference params | Inference latency |
| ------ | ------------------- | ------------------- | --------- | ---------------- | ----------------- |
|        |                     |                     |           |                  |                   |

并且文字里承认：

> VIVID-Med pays an additional one-time training cost due to the frozen LLM, but deploys only the ViT backbone.

这比单纯说 lightweight 更可信。

------

### B3. Quantitative embedding analysis

替代 t-SNE 主观图：

| Method | Silhouette ↑ | Davies-Bouldin ↓ | kNN retrieval ↑ | Linear AUC ↑ |
| ------ | ------------ | ---------------- | --------------- | ------------ |
|        |              |                  |                 |              |

如果结果不明显，删除 t-SNE claim。

------

## Tier C：有资源再做

### C1. Better CT evidence

如果你还想保留 cross-modality claim，需要更强 CT 设置：

- patient-level LIDC split；
- 3D MedMNIST tasks；
- CT organ / nodule / lesion datasets；
- frozen backbone linear probe + fine-tuning；
- 不要只靠 OrganAMNIST，因为 AUC 太饱和。

如果资源有限，建议把 CT 降级为 appendix 或 secondary experiment。

------

# 9. 写作重构方案

## 新标题建议

当前标题：

> VIVID-Med: LLM-Supervised Structured Pretraining for Deployable Medical ViTs

这个标题过大，容易让 reviewer 期待 multi-modality/multi-dataset/multi-backbone。

更稳的标题：

> **VIVID-Med: Schema-Guided Frozen-Language-Model Supervision for Deployable Chest X-ray ViTs**

或者：

> **Structured Frozen-LM Supervision for Data-Efficient Chest X-ray Representation Learning**

如果你补了多外部数据和多 backbone，再用 “Medical ViTs” 没问题。否则建议收窄到 CXR。

------

## Introduction 重写逻辑

建议 4 段：

1. **Problem**：CXR representation learning 需要可迁移、低部署成本的 backbone。
2. **Gap**：one-hot labels 忽略 finding relations；free text noisy；large VLM deployment expensive。
3. **Idea**：用 structured schema 把 findings 转成 answerable field-state supervision，用 frozen LM 作为 training-only decoder，把语义监督蒸馏进 ViT。
4. **Validation promise**：本文不是只报 SOTA，而是 controlled 验证 LLM role、UMS、SPD、data-matched baselines、external transfer、efficiency。

这样比现在直接说 “LLM inherently embeds related clinical concepts” 更稳。

------

## Related Work 必须新增的段落

建议加一个小节：

**Frozen LLMs for visual signals.**

写法核心：

- Pang et al.：frozen LLM transformer blocks can act as visual encoder layers；你的不同是 LLM 不作为 visual encoder layer 部署，而是 training-only structured teacher。([arXiv](https://arxiv.org/abs/2310.12973))
- Zhu et al.：V2T 把图像转成 LLM vocabulary tokens 让 LLM 直接理解视觉信号；你的不同是训练 standalone ViT，而不是让 LLM 作为最终视觉理解器。([arXiv](https://arxiv.org/abs/2403.07874))
- ViTP：把 ViT 嵌入 VLM 做 visual instruction pretraining；你的不同是 UMS、answerability-aware masking、SPD、LLM-free deployment，但需要通过实验对比证明。([arXiv](https://arxiv.org/abs/2509.17562))

------

## Claims 修改清单

| 位置       | 当前风险                        | 修改建议                            |
| ---------- | ------------------------------- | ----------------------------------- |
| Abstract   | 500× less data claim 太刺激     | 删除或放 limitation 后面            |
| Intro      | LLM clinical proximity 被当事实 | 改成 hypothesis，并用 ablation 验证 |
| Method     | null = not assessable 太绝对    | 改成 unannotated / non-supervised   |
| Results    | “exceptional”                   | 改成 “strong”                       |
| Results    | t-SNE “tighter clusters”        | 删除或量化                          |
| Discussion | “proving”                       | 改成 “supports / suggests”          |
| CT         | “strong cross-modality”         | 改成 “encouraging but mixed”        |
| Deployment | 只讲 inference cost             | 同时报告 training cost              |

------

# 10. 转投策略

## 方案 1：最推荐——补完整证据链后转方法型会议/期刊

适合条件：

- 能补 data-matched baselines；
- 能补 LLM role ablation；
- 能补 SPD sensitivity；
- 能补至少一个外部 CXR；
- 能补 efficiency table。

目标定位：

- **MIDL / MICCAI workshop → full conference version**
- **IEEE TMI / Medical Image Analysis**，如果你能扩到多数据集、多 backbone、多任务。

这个方向最值得，因为 paper 的 idea 仍然有价值。

------

## 方案 2：资源有限——收窄成 CXR representation learning paper

如果资源不够跑大量数据集，就不要继续讲 “medical ViT foundation / cross-modality”。

改成：

> schema-guided CXR representation learning with deployable ViT

保留：

- CheXpert；
- NIH；
- 一个外部 CXR；
- 核心 ablation；
- efficiency。

这样可以投更聚焦的 medical imaging / machine learning venue，成功率比硬讲 foundation model 更高。

------

## 方案 3：如果关键 ablation 不支持 LLM，立即 pivot

如果 random LM 或 no-LM classifier 和 Qwen2.5 差不多，那你不要再讲 frozen LLM semantic teacher。

可以改成：

> UMS-SPD: answerability-aware structured label decomposition for long-tail CXR classification

这时 contribution 从 LLM 转向：

- structured label schema；
- answerability-aware supervision；
- SPD for long-tail findings；
- deployable ViT。

这会比强行保留 LLM story 更诚实，也更容易防守。

------

# 11. 我给你的最终优先级清单

按顺序做：

1. **删除 / 降级 500× less data claim。**
2. **补同数据同 backbone baseline：BCE supervised ViT、UMS no-LLM、random LM、Q-Former。**
3. **补 LLM role ablation：pretrained vs random vs no LM，field-name shuffle，blank image prefix。**
4. **补 SPD sensitivity：G、M、λ_ortho、field sampling。**
5. **补一个外部 CXR 数据集。**
6. **补 training/inference cost table。**
7. **重写 related work，正面回应 Pang et al.、Beyond Text、ViTP。**
8. **删除主观 t-SNE claim，换成 quantitative separability。**
9. **把 CT claim 降级，除非你补更强 CT 实验。**
10. **把 title 和 abstract 收窄，避免 medical foundation model 级别的过度承诺。**

------

# 12. 最终判断

**值得继续，但不要小修小补。**

当前版本的问题是：**想讲一个大故事，但实验像一个中等规模 ablation paper。** 转投版要么把故事收窄，要么把实验补到足以支撑大故事。

我的建议是：

**继续做，目标是“中修到大修后转投”，而不是放弃。**

最关键的成败点只有三个：

1. **VIVID-Med 是否明显强于同数据、同 backbone、同 label supervision 的普通 supervised baseline？**
2. **pretrained frozen LLM 是否明显强于 random/no-LM teacher？**
3. **外部 CXR transfer 是否稳定提升？**

这三个都成立，这篇还有很好的投稿价值；如果其中两个不成立，就应该 pivot，别再主打 LLM-supervised pretraining。

------

# 13. 自动补实验进度（2026-05-08 起）

执行约束：
- 环境：`conda run -n vivid python ...`
- GPU：只用物理 GPU1，命令前置 `CUDA_VISIBLE_DEVICES=1`
- 规则：每补完一个实验立刻更新本文档；若关键停止准则触发，则停止继续烧卡并转为 pivot 判断。

## 13.1 停止 / pivot 准则

若下面三项中至少两项不成立，则不再主打 “frozen LLM semantic teacher”，应 pivot 到 “structured label decomposition / UMS-SPD for CXR representation learning”：

1. VIVID-Med 明显强于同数据、同 ViT-B、同 CheXpert supervision 的 BCE baseline。
2. Pretrained frozen LLM 明显强于 random/no-LM teacher。
3. 外部 CXR transfer 稳定提升。

## 13.2 已有结果复盘

| 返修问题 | 实验/输出目录 | Macro-AUC | Macro-F1 | 判断 |
| --- | --- | ---: | ---: | --- |
| Data-matched BCE baseline | `outputs/baseline_vit_full14` | 0.7927 | 0.8987 | VIVID/SPD 线性探针 0.8208，高于 BCE；暂不触发停止。 |
| UMS + frozen LLM, no SPD | `outputs/lp_A_ums_12label` | 0.8439 | 0.9095 | 高于 SPD 版本；SPD 主张存在风险，需补 sensitivity 或降低 SPD claim。 |
| UMS + frozen LLM + SPD | `outputs/lp_A_ums_spd_12label` | 0.8208 | 0.9114 | 高于 BCE 和部分通用预训练，但低于 no-SPD UMS。 |
| Free-text target + frozen LLM | `outputs/lp_A_freetext_12label` | 0.8126 | 0.9064 | UMS no-SPD 明显优于 free-text；结构化监督成立。 |
| Random mask proxy | `outputs/lp_random_mask_12label` | 0.7387 | 0.8876 | 低于 UMS/LLM；不是随机 mask 解释。 |
| Q-Former proxy | `outputs/lp_qformer_proxy_12label` | 0.8008 | 0.9186 | AUC 低于 SPD，但 F1 高；SPD 证据不够稳。 |
| BiomedCLIP linear probe | `outputs/lp_biomedclip_baseline_seed0` | 0.8076 | 0.9040 | 当前 VIVID/SPD 略高，但 UMS no-SPD 更高。 |

当前判断：
- 继续补实验，但要优先验证 LLM role。
- SPD 目前不能作为强 claim，除非 sensitivity 或重跑能证明稳定收益。

## 13.3 正在补的实验

### E1. UMS classifier no LLM

目的：回答 “UMS schema 本身是否已经足够，还是 frozen pretrained LLM 提供了额外训练信号？”

新增文件：
- `scripts/train_ums_classifier.py`
- `configs/ums_classifier_no_llm_12label.yaml`
- `configs/lp_ums_classifier_no_llm_12label.yaml`

Debug 验证：
- `CUDA_VISIBLE_DEVICES=1 conda run -n vivid python scripts/train_ums_classifier.py --config configs/ums_classifier_no_llm_12label.yaml --debug`
- 结果：脚本、CUDA、数据读取、验证和 checkpoint 保存均跑通；短跑输出误写入 `outputs/ums_classifier_no_llm_12label`，正式实验改用 `_full` 后缀目录，避免覆盖/删除 debug artifact。

计划命令：
```bash
CUDA_VISIBLE_DEVICES=1 conda run -n vivid python scripts/train_ums_classifier.py --config configs/ums_classifier_no_llm_12label.yaml
CUDA_VISIBLE_DEVICES=1 conda run -n vivid python scripts/train_vit_baseline.py --config configs/lp_ums_classifier_no_llm_12label.yaml
```

正式运行：
- 启动命令：`CUDA_VISIBLE_DEVICES=1 conda run -n vivid python scripts/train_ums_classifier.py --config configs/ums_classifier_no_llm_12label.yaml`
- PID：`26900`（conda 子进程使用物理 GPU1）
- step 500 中间结果：`outputs/ums_classifier_no_llm_12label_full/metrics_step_500.json`
- step 500 macro-AUC：0.7234；state accuracy all fields：0.7075；answerable fields：0.2777。
- step 2500 macro-AUC：0.7147；macro-F1：0.3165；answerable state accuracy：0.4089。
- step 5000 macro-AUC：0.7241；macro-F1：0.3532；answerable state accuracy：0.4144。
- step 7000 macro-AUC：0.7233；macro-F1：0.3981；answerable state accuracy：0.4765。
- step 9500 macro-AUC：0.7488；macro-F1：0.4222；answerable state accuracy：0.4736。
- final macro-AUC：0.7495；macro-F1：0.4339；answerable state accuracy：0.4898；输出：`outputs/ums_classifier_no_llm_12label_full/metrics_final.json`。
- 判断：no-LLM UMS state classifier 本身明显低于 frozen-LLM UMS/no-SPD（0.8439）和 frozen-LLM UMS+SPD（0.8208）的 downstream linear probe，也低于 BCE baseline（0.7927）。这支持 “UMS schema alone 不足以解释主要收益”，但仍需跑该 no-LLM backbone 的 14-label linear probe 作 apples-to-apples 下游比较。

正式结果：预训练完成；linear probe 运行中。

Linear probe：
- 启动命令：`CUDA_VISIBLE_DEVICES=1 conda run -n vivid python scripts/train_vit_baseline.py --config configs/lp_ums_classifier_no_llm_12label.yaml`
- step 600 macro-AUC：0.7830；macro-F1：0.9159；输出：`outputs/lp_ums_classifier_no_llm_12label_full/metrics_step_600.json`。
- step 1400 macro-AUC：0.8116；macro-F1：0.9169。
- step 2200 macro-AUC：0.8365；macro-F1：0.9129。
- 临时判断：no-LLM backbone 已明显超过 BCE baseline（0.7927）并超过 frozen-LLM UMS+SPD（0.8208），但仍略低于 frozen-LLM UMS/no-SPD（0.8439）。这削弱 “LLM 必要性很强” 的 claim；更稳的说法应是 UMS schema 是主要贡献之一，pretrained LM 可能提供额外增益，需要 random-LM / blank-prefix 继续拆解。
- final macro-AUC：0.8273；macro-F1：0.9143；输出：`outputs/lp_ums_classifier_no_llm_12label_full/metrics_final.json`。

E1 结论：
- UMS schema 本身有效：no-LLM UMS classifier backbone 的 downstream macro-AUC 0.8273，高于 BCE baseline 0.7927。
- pretrained frozen LLM 的额外收益存在但不大：frozen-LLM UMS/no-SPD 0.8439，比 no-LLM 高 0.0167 macro-AUC。
- SPD 不能强 claim：当前 SPD 0.8208 低于 no-LLM UMS classifier 0.8273，也低于 UMS/no-SPD 0.8439。
- 对论文定位的影响：可以继续做，但应把主线改成 “UMS schema + frozen LM provide controlled structured supervision”，不要说 LLM 是唯一/主要来源；SPD 降级为 optional design，除非 sensitivity 后证明稳定。

### E2. Visual-prefix dependency / blank-prefix token NLL

目的：回答 teacher forcing 下 LLM 是否真的使用视觉 prefix，而不是只靠 label prior / JSON prefix。

计划：对已有 frozen-LLM checkpoint 做验证集 token NLL 对比：
- image prefix：正常输入图像。
- blank image：把图像置零再输入同一模型。
- shuffled image：batch 内图像打乱，文本 target 不变。

若 image prefix 的 token NLL 明显低于 blank/shuffled，说明视觉 token 被使用；若差异很小，LLM story 需要降级。

结果：待运行。

Debug note：
- `scripts/train_cxr.py --debug` 会把 LLM 改成 `sshleifer/tiny-gpt2`；`vivid` conda 的 torch 2.5 被 transformers 安全检查拦截 `.bin` 权重加载，因此该 debug 失败不代表 G=2 配置失败。
- 新增 `configs/debug_ablation_spd_g2_12label.yaml`，继续使用 Qwen safetensors，只跑 5 step 验证配置。

Debug：
- 命令：`CUDA_VISIBLE_DEVICES=1 conda run -n vivid python scripts/eval_visual_prefix_dependency.py --config configs/ablation_A_ums_12label.yaml --checkpoint outputs/ablation_A_ums_12label/checkpoints/best.pt --output outputs/visual_prefix_dependency_A_ums_12label_debug.json --max-samples 4 --batch-size 1 --num-workers 0`
- 结果：脚本和 checkpoint 加载跑通；4 样本上 image NLL 0.0454，blank/shuffled NLL 0.0671，blank 相对 image 高 47.7%。

正式命令：
```bash
CUDA_VISIBLE_DEVICES=1 conda run -n vivid python scripts/eval_visual_prefix_dependency.py --config configs/ablation_A_ums_12label.yaml --checkpoint outputs/ablation_A_ums_12label/checkpoints/best.pt --output outputs/visual_prefix_dependency_A_ums_12label_128.json --max-samples 128 --batch-size 2 --num-workers 0
```

正式结果（128 val samples, `outputs/visual_prefix_dependency_A_ums_12label_128.json`）：

| Variant | Token NLL | Δ vs image | Relative Δ |
| --- | ---: | ---: | ---: |
| image prefix | 0.0482 | 0 | 0 |
| blank image prefix | 0.0743 | +0.0262 | +54.3% |
| shuffled image prefix | 0.0689 | +0.0208 | +43.1% |

E2 结论：
- 正常图像 prefix 的 token NLL 明显低于 blank/shuffled，说明 teacher-forcing 目标不是纯靠 JSON/text prior，视觉 prefix 被模型使用。
- 这支持继续保留 “visual-prefix-conditioned frozen-LM supervision” 这条主线。
- 但结合 E1，claim 应谨慎：pretrained LLM + visual prefix 有实证作用；UMS schema 也是强贡献；SPD 当前证据不足，需降级或补 sensitivity。

## 13.4 阶段性继续 / 停止判断

截至 E1 + E2：

| 成败点 | 当前证据 | 判断 |
| --- | --- | --- |
| 是否强于同数据 BCE | no-LLM UMS LP 0.8273、frozen-LM UMS 0.8439、BCE 0.7927 | 成立 |
| pretrained frozen LLM 是否强于 no-LM | frozen-LM UMS/no-SPD 0.8439 vs no-LM UMS 0.8273；差值 +0.0167 | 弱成立，但不是压倒性 |
| visual prefix 是否被使用 | image NLL 比 blank/shuffled 低 54.3% / 43.1% | 成立 |
| SPD 是否稳定有益 | SPD 0.8208 < UMS/no-SPD 0.8439，且 < no-LM UMS 0.8273 | 不成立 |

阶段性判断：
- 项目值得继续，不触发停止 / pivot 条件。
- 论文主线应继续，但要调整：主打 UMS + frozen-LM visual-prefix supervision；弱化 “LLM semantic manifold” 和 SPD 强 claim。
- 下一优先级：补 SPD sensitivity，判断 SPD 是不是应作为 appendix/negative result；随后补 external CXR 或 cost table。

## 13.5 SPD sensitivity

已有覆盖：
- no-SPD UMS：`outputs/lp_A_ums_12label`，macro-AUC 0.8439。
- Q-Former proxy / G=1,M=8,λ=0：`outputs/lp_qformer_proxy_12label`，macro-AUC 0.8008。
- SPD default / G=4,M=2,λ=0.02：`outputs/lp_A_ums_spd_12label`，macro-AUC 0.8208。

新增计划：
- E3：G=2,M=4,λ=0.02，保持总 SPD query token=8。

新增文件：
- `configs/ablation_spd_g2_12label.yaml`
- `configs/lp_spd_g2_12label.yaml`

计划命令：
```bash
CUDA_VISIBLE_DEVICES=1 conda run -n vivid python scripts/train_cxr.py --config configs/ablation_spd_g2_12label.yaml
CUDA_VISIBLE_DEVICES=1 conda run -n vivid python scripts/train_vit_baseline.py --config configs/lp_spd_g2_12label.yaml
```

Debug 验证：
- 直接使用 `scripts/train_cxr.py --debug` 会把 LLM 换成 `sshleifer/tiny-gpt2`，在当前 `vivid` conda 环境中被 transformers 对 `.bin` 权重的 torch>=2.6 安全检查拦截；这不是 G=2/SPD 配置本身的问题。
- 新增 `configs/debug_ablation_spd_g2_12label.yaml`，保留正式 Qwen safetensors，只跑 5 step。
- 命令：`CUDA_VISIBLE_DEVICES=1 conda run -n vivid python scripts/train_cxr.py --config configs/debug_ablation_spd_g2_12label.yaml`
- 结果：Qwen + G=2,M=4,λ=0.02 配置可正常构建、前向、反向、验证和保存 checkpoint；step 2 val loss 1.3125，step 4 val loss 0.9972；输出 `outputs/debug_ablation_spd_g2_12label/checkpoints/{best.pt,step_5.pt,final.pt}`。

正式运行：
- 启动命令：`CUDA_VISIBLE_DEVICES=1 conda run -n vivid python scripts/train_cxr.py --config configs/ablation_spd_g2_12label.yaml`
- 运行环境：`vivid` conda 环境；物理 GPU1（RTX 3090）已占用约 23GB 显存，GPU0 为其他进程占用。
- 监控方式：`scripts/train_cxr.py` 的正式结果以 `outputs/ablation_spd_g2_12label/checkpoints/` 下的 `best.pt/final.pt/step_*.pt` 为准；`conda run` 的 stdout/stderr 可能在进程结束前保持缓冲。
- 中间状态：GPU1 显存约 24.2GB、利用率持续波动，属于当前 `batch_size=4` + Qwen2.5-1.5B + ViT-B + SPD 配置的高水位；连续采样显示进程仍在正常计算。
- 已保存第一轮验证 checkpoint：`outputs/ablation_spd_g2_12label/checkpoints/best.pt`（2026-05-09 01:32），说明正式训练已通过验证与写盘阶段，无 OOM。
- step 1000 checkpoint 已保存：`outputs/ablation_spd_g2_12label/checkpoints/step_1000.pt`（2026-05-09 02:25）；同一时刻 `best.pt` 更新，训练继续正常。
- step 2000 checkpoint 已保存：`outputs/ablation_spd_g2_12label/checkpoints/step_2000.pt`（2026-05-09 03:37）；`best.pt` 同步更新。
- step 3000 checkpoint 已保存：`outputs/ablation_spd_g2_12label/checkpoints/step_3000.pt`（2026-05-09 04:48）；`best.pt` 于 05:22 再次更新，说明验证 loss 仍在改善。
- step 4000/5000/6000 checkpoints 已保存；`best.pt` 于 2026-05-09 08:28 在 step 6000 附近再次更新。训练仍稳定运行在 GPU1。
- step 7000 checkpoint 已保存：`outputs/ablation_spd_g2_12label/checkpoints/step_7000.pt`（2026-05-09 09:45）；`best.pt` 于 10:38 再次更新。
- step 8000 checkpoint 已保存：`outputs/ablation_spd_g2_12label/checkpoints/step_8000.pt`（2026-05-09 11:41）；`best.pt` 于 12:53 再次更新。
- step 9000 checkpoint 已保存：`outputs/ablation_spd_g2_12label/checkpoints/step_9000.pt`（2026-05-09 14:02）；`best.pt` 于 15:00 再次更新。
- 预训练完成：`step_10000.pt` 与 `final.pt` 已保存（2026-05-09 15:59）；总耗时约 15:37:35。
- 验证 loss 摘要：step 2000 0.0385；step 4000 0.0364；step 6000 0.0355；step 8000 0.0344；step 9500 0.0343（best 更新）；step 10000 0.0345。

正式结果：预训练完成；已用 `outputs/ablation_spd_g2_12label/checkpoints/best.pt` 跑 linear probe。

Linear probe：
- 命令：`CUDA_VISIBLE_DEVICES=1 conda run -n vivid python scripts/train_vit_baseline.py --config configs/lp_spd_g2_12label.yaml`
- 输出：`outputs/lp_spd_g2_12label/metrics_final.json`
- final macro-AUC：0.8291；macro-F1：0.9149；micro-F1：0.8973；val loss：0.2423。

E3 结论：
- G=2,M=4,λ=0.02 明显好于 default SPD G=4,M=2,λ=0.02（0.8291 vs 0.8208 macro-AUC），说明 SPD 分组方式对结果敏感。
- 但 G=2 仍低于 UMS/no-SPD（0.8291 vs 0.8439 macro-AUC），也只略高于 no-LM UMS classifier backbone（0.8291 vs 0.8273）。
- 因此 SPD 不能作为主贡献或稳定增益 claim；更稳妥的论文处理是把 SPD sensitivity 放 appendix/negative analysis，正文主线转为 UMS + frozen-LM visual-prefix supervision。
- 项目仍值得继续：E1/E2/E3 共同支持 UMS 与 visual-prefix-conditioned frozen LM 有效；需要继续补 external CXR / cost / seed 稳定性，而不是继续投入大量 GPU 在 SPD sweep。

## 13.6 External CXR / NIH 补充

已有 NIH cross-domain 结果（CheXpert-trained head 直接在 NIH 8 common labels 上推理）：
- UMS/no-SPD seeds：0.7068 / 0.6977 / 0.6996 NIH macro-AUC。
- UMS+SPD default seeds：0.7214 / 0.7247 / 0.7215 NIH macro-AUC。
- BiomedCLIP seeds：0.6730 / 0.6799 / 0.6647 NIH macro-AUC。
- ImageNet baseline seeds：约 0.6135-0.6159 NIH macro-AUC。
- Random-mask seeds：约 0.5982-0.6017 NIH macro-AUC。

临时判断：
- 外部 CXR transfer 目前是正向证据：UMS/no-SPD 与 default SPD 都显著高于 ImageNet/BiomedCLIP/random-mask。
- 但 E1/E3 新增的 no-LM UMS classifier 与 G2/SPD 还没有对应 NIH 推理结果；这两个只需要 inference，优先补齐。

计划命令：
```bash
CUDA_VISIBLE_DEVICES=1 conda run -n vivid python scripts/eval_nih_crossdomain.py --checkpoint outputs/lp_ums_classifier_no_llm_12label_full/best.pt --label no_lm_ums_classifier --output outputs/lp_ums_classifier_no_llm_12label_full/nih_crossdomain.json
CUDA_VISIBLE_DEVICES=1 conda run -n vivid python scripts/eval_nih_crossdomain.py --checkpoint outputs/lp_spd_g2_12label/best.pt --label spd_g2 --output outputs/lp_spd_g2_12label/nih_crossdomain.json
```

no-LM UMS NIH 结果：
- 输出：`outputs/lp_ums_classifier_no_llm_12label_full/nih_crossdomain.json`
- macro-AUC：0.7262；macro-F1：0.2630；micro-F1：0.4328。
- 判断：no-LM UMS 在 NIH 上高于已有 UMS/no-SPD seeds（约 0.6977-0.7068）和 default SPD seeds（约 0.7214-0.7247）中的大多数/全部；这说明外部 CXR 提升很大一部分来自 UMS/state supervision 与 representation，而不是只能归因于 pretrained frozen LLM。

G2/SPD NIH 结果：
- 输出：`outputs/lp_spd_g2_12label/nih_crossdomain.json`
- macro-AUC：0.7176；macro-F1：0.2627；micro-F1：0.4433。
- 判断：G2/SPD 在 NIH 上低于 no-LM UMS（0.7176 vs 0.7262），也没有超过 default SPD seed 均值附近；SPD 的外部泛化证据仍不稳定。

E4 结论：
- 外部 CXR 这个停止条件总体成立：UMS-family 结果明显高于 ImageNet、BiomedCLIP、random-mask 等旧 baseline。
- 但 LLM-specific claim 进一步变弱：no-LM UMS 在 NIH 上反而是当前最强单次结果之一。
- 因此不应再把论文主打成 “pretrained LLM semantic knowledge is the main source of transfer”；更稳的主线是 “answerability-aware UMS supervision gives transferable CXR representations; frozen LM visual-prefix loss is a useful but modest variant.”
- 项目仍可继续，但应停止继续烧卡做 SPD sweep；下一步优先做成本表、seed 汇总表和写作降 claim。如果继续补实验，最有价值的是 random-LM/frozen-LM 同架构对照或第二个外部 CXR 数据集，而不是 SPD 超参。

## 13.7 当前总判断（2026-05-09）

| 成败点 | 证据 | 判断 |
| --- | --- | --- |
| 是否强于同数据 BCE | BCE 0.7927；no-LM UMS 0.8273；UMS/no-SPD 0.8439；G2/SPD 0.8291 | 成立 |
| pretrained frozen LLM 是否强于 no-LM | CheXpert：UMS/no-SPD 0.8439 > no-LM 0.8273；NIH：no-LM 0.7262 > UMS/no-SPD/default SPD 单次或均值 | 弱成立/不稳定 |
| visual prefix 是否被使用 | image-prefix NLL 比 blank/shuffled 低 54.3% / 43.1% | 成立 |
| 外部 CXR transfer 是否稳定提升 | UMS-family NIH AUC 约 0.70-0.73，高于 BiomedCLIP 约 0.66-0.68 与 ImageNet 约 0.61 | 成立 |
| SPD 是否稳定有益 | CheXpert 和 NIH 均不稳定；G2 低于 no-SPD，default SPD 外部好但 in-domain 弱 | 不成立 |

停止 / 继续决策：
- 不触发“项目停止”条件：同数据 BCE 和外部 CXR transfer 都成立，visual-prefix dependency 也成立。
- 触发“降 claim / pivot 部分叙事”条件：LLM 的额外贡献不稳定，SPD 不能主打。
- 推荐继续项目，但改写定位：从 “frozen LLM semantic teacher + SPD” 改为 “UMS-based structured CXR representation learning, with frozen-LM visual-prefix supervision as one controlled variant”；SPD 放 appendix/negative sensitivity。

## 13.8 当前所有补充实验总总结（可直接用于改稿）

本节把 2026-05-08 至 2026-05-09 自动补实验的结果集中整理，避免只散落在过程日志中。完整机器可读汇总表另存为 `outputs/summary_key_results_20260509.csv`。

### 13.8.1 主结果与关键对照

| Method / Variant | CheXpert macro-AUC | CheXpert macro-F1 | NIH macro-AUC | NIH macro-F1 | 主要作用 |
| --- | ---: | ---: | ---: | ---: | --- |
| Data-matched BCE ViT-B | 0.7927 | 0.8987 | - | - | 同数据、同 backbone 的普通监督 baseline |
| Frozen-LM UMS / no-SPD | 0.8439 | 0.9095 | 0.7068 | 0.2453 | 当前 CheXpert 最强主线 |
| Frozen-LM UMS + SPD default G=4,M=2 | 0.8208 | 0.9114 | 0.7214 | 0.2620 | SPD 默认设置；NIH 好，但 CheXpert 弱 |
| Frozen-LM UMS + SPD G=2,M=4 | 0.8291 | 0.9149 | 0.7176 | 0.2627 | SPD sensitivity；比 default 好但仍低于 no-SPD |
| no-LM UMS state classifier | 0.8273 | 0.9143 | 0.7262 | 0.2630 | 拆解 UMS schema 本身的贡献 |
| Frozen-LM free-text target | 0.8126 | 0.9064 | 0.6365 | 0.2201 | 证明 structured UMS 明显优于 free text |
| Random-mask proxy | 0.7387 | 0.8876 | 0.6002 | 0.1940 | 非语义/弱结构 proxy |
| Q-Former proxy G=1,M=8,λ=0 | 0.8008 | 0.9186 | - | - | 无 SPD orthogonality 的 proxy |
| BiomedCLIP linear probe seed0 | 0.8076 | 0.9040 | 0.6730 | 0.2235 | 外部强 baseline |

直接结论：
- UMS-family 明显强于 data-matched BCE：best UMS/no-SPD 0.8439 vs BCE 0.7927，CheXpert macro-AUC +0.0512。
- UMS structure 明显优于 free text：0.8439 vs 0.8126，CheXpert macro-AUC +0.0314。
- no-LM UMS 也很强：CheXpert 0.8273，NIH 0.7262，说明 UMS/state supervision 本身是主要贡献之一。
- frozen LM 的额外收益在 CheXpert 上存在但不大：UMS/no-SPD 0.8439 vs no-LM UMS 0.8273，+0.0167；但 NIH 上 no-LM UMS 反而更高。
- SPD 不稳定：G=2 CheXpert 0.8291 低于 no-SPD 0.8439；default SPD CheXpert 0.8208 更低。SPD 不应作为主贡献。

### 13.8.2 no-LM UMS classifier 细节

新增 `scripts/train_ums_classifier.py`，用 ViT-B 直接预测每个 UMS field 的 4-state label（null / absent / uncertain / present），不使用 LLM。

预训练结果：
- 输出：`outputs/ums_classifier_no_llm_12label_full/metrics_final.json`
- macro-AUC：0.7495
- macro-F1：0.4339
- state accuracy all fields：0.7434
- state accuracy answerable fields：0.4898

下游 linear probe：
- 输出：`outputs/lp_ums_classifier_no_llm_12label_full/metrics_final.json`
- CheXpert macro-AUC：0.8273
- CheXpert macro-F1：0.9143
- NIH macro-AUC：0.7262
- NIH macro-F1：0.2630

解释：
- no-LM UMS 的下游表示明显超过 BCE baseline 和 free-text target，说明“把 CXR 标签转成 answerability-aware UMS state supervision”本身已经有价值。
- 这削弱了“pretrained LLM 是唯一/主要来源”的 claim，但不否定 frozen-LM objective：frozen-LM UMS/no-SPD 在 CheXpert 上仍比 no-LM 高 0.0167 macro-AUC。

### 13.8.3 Visual-prefix dependency 细节

新增 `scripts/eval_visual_prefix_dependency.py`，对 frozen-LM UMS/no-SPD checkpoint 做 teacher-forcing token NLL 对照。

结果文件：`outputs/visual_prefix_dependency_A_ums_12label_128.json`

| Prefix variant | Token NLL | Δ vs image | Relative Δ |
| --- | ---: | ---: | ---: |
| normal image prefix | 0.0482 | 0 | 0 |
| blank image prefix | 0.0743 | +0.0262 | +54.3% |
| shuffled image prefix | 0.0689 | +0.0208 | +43.1% |

解释：
- 正常图像 prefix 的 token NLL 明显低于 blank/shuffled，说明 frozen LM 在 teacher forcing 下确实使用视觉 prefix，不是只靠 JSON/text prior。
- 这条证据可以支持正文保留 “visual-prefix-conditioned frozen-LM supervision”。
- 但它只能证明视觉 prefix 被使用，不能单独证明“pretrained LLM semantic manifold 是主要收益来源”。

### 13.8.4 SPD sensitivity 细节

新增：
- `configs/ablation_spd_g2_12label.yaml`
- `configs/lp_spd_g2_12label.yaml`
- `configs/debug_ablation_spd_g2_12label.yaml`

G=2,M=4,λ=0.02 预训练：
- 输出：`outputs/ablation_spd_g2_12label/checkpoints/`
- 训练完成：`step_10000.pt` 与 `final.pt`，总耗时约 15:37:35。
- validation loss：step 2000 0.0385；step 4000 0.0364；step 6000 0.0355；step 8000 0.0344；step 9500 0.0343；step 10000 0.0345。

G=2 downstream：
- 输出：`outputs/lp_spd_g2_12label/metrics_final.json`
- CheXpert macro-AUC：0.8291
- CheXpert macro-F1：0.9149
- NIH macro-AUC：0.7176
- NIH macro-F1：0.2627

SPD 总判断：
- G=2 比 default G=4 好：0.8291 vs 0.8208 CheXpert macro-AUC。
- 但 G=2 仍低于 no-SPD UMS：0.8291 vs 0.8439。
- default SPD 在 NIH 上较好，但 CheXpert 不好；G=2 在 CheXpert 改善后 NIH 也不超过 no-LM UMS。
- SPD 对结果敏感且不稳定，不能主打为 stable gain。建议放 appendix，作为 negative/sensitivity analysis。

### 13.8.5 External CXR / NIH 细节

NIH 使用 CheXpert-trained 14-label head 直接映射到 8 个 common labels 做 cross-domain inference，不重新训练。

关键 NIH 结果：
- no-LM UMS：0.7262 macro-AUC
- default SPD：0.7214 / 0.7247 / 0.7215 macro-AUC（已有 3 seeds）
- G2/SPD：0.7176 macro-AUC
- UMS/no-SPD：0.7068 / 0.6977 / 0.6996 macro-AUC（已有 3 seeds）
- BiomedCLIP：0.6730 / 0.6799 / 0.6647 macro-AUC
- ImageNet：约 0.6135-0.6159 macro-AUC
- Random-mask：约 0.5982-0.6017 macro-AUC

解释：
- 外部 CXR transfer 是正向证据：UMS-family 明显高于 BiomedCLIP、ImageNet 和 random-mask。
- 但 NIH 上最强的是 no-LM UMS 单次结果，说明外部迁移收益不能主要归因于 pretrained LLM。
- 更稳的论文表达是：UMS/state supervision improves transferable CXR representations；frozen-LM visual-prefix loss provides a controlled training variant and improves in-domain CheXpert, but its external advantage is not stable in current evidence。

### 13.8.6 当前应如何改论文 claim

建议保留 / 强化：
- “Answerability-aware UMS provides stronger structured supervision than flat BCE or free-text targets.”
- “The visual prefix is used by the frozen-LM objective, as blank/shuffled image prefixes substantially increase token NLL.”
- “UMS-family representations transfer better to NIH than ImageNet, BiomedCLIP, and random-mask baselines under the current evaluation.”
- “The final deployed model remains a ViT backbone; the LLM is only a training-time component.”

建议降级 / 删除：
- 不要说 “LLM semantic manifold is the main source of improvement”。
- 不要把 SPD 写成核心贡献或稳定提升模块。
- 不要把 “500× less data” 写成强 apples-to-apples data-efficiency claim；最多作为背景性 resource note，并明确 domain-specific CheXpert supervision 与 BiomedCLIP pretraining data 不可直接等价。
- 不要把 CT/cross-modality 写成 strong generalization 主卖点，除非再补更强外部数据。

推荐新主线：

> VIVID-Med is a controlled study of answerability-aware structured supervision for deployable CXR representation learning. UMS supervision is the main stable contributor; frozen-LM visual-prefix training is a useful in-domain variant whose visual dependency can be verified, while SPD is treated as a sensitivity/negative-result component rather than a core claim.

### 13.8.7 是否继续做项目

结论：继续，但不要继续围绕 SPD 大量烧 GPU。

继续的理由：
- 同数据 BCE 对照已经被明显超过。
- structured UMS 明显优于 free text 和 random proxy。
- visual-prefix dependency 成立。
- 外部 NIH transfer 整体成立。

需要调整的地方：
- LLM 贡献只能说 “modest / not fully stable”，不能说 “dominant”。
- SPD 只能作为 appendix/sensitivity，不作为主模块。
- 论文应 pivot 到 “UMS structured CXR representation learning”，而不是继续主打 “frozen LLM semantic teacher + SPD”。

下一步优先级：
1. 整理 seed mean/std 表，尤其是 CheXpert 和 NIH 的 3-seed 结果。
2. 补 training/inference cost table，明确 LLM 只在训练时使用。
3. 若还要补实验，优先 random-LM/frozen-LM same-architecture 对照或第二个外部 CXR 数据集；不建议继续 SPD sweep。

## 14. VIVID_review1 后续补实验更新（2026-05-10）

本节按 `VIVID_review1.md` 的建议，优先补 EMNLP 线需要的 grounding 诊断实验，并同步检查 AAAI/general 线是否能在本地数据和现有代码基础上直接启动。

运行约束：
- 环境：`conda run -n vivid`
- GPU：只暴露物理 GPU1，命令中使用 `$env:CUDA_VISIBLE_DEVICES='1'`
- 不重新训练新主干；优先使用已有 frozen-LM UMS/no-SPD checkpoint 做轻量评估。

### 14.1 EMNLP P0: counterfactual schema scoring

新增脚本：`scripts/eval_counterfactual_schema_grounding.py`

正式运行命令：

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n vivid python scripts/eval_counterfactual_schema_grounding.py \
  --config configs/ablation_A_ums_12label.yaml \
  --checkpoint outputs/ablation_A_ums_12label/checkpoints/best.pt \
  --output outputs/counterfactual_schema_grounding_A_ums_12label_128.json \
  --max-samples 128 \
  --batch-size 2 \
  --num-workers 0
```

评估设计：
- 对每张图像保留真实 UMS schema 作为 `z+`。
- 构造四类反事实 `z-`：`state_flip`、`field_swap`、`image_swap`、`null_to_present`。
- 用 teacher forcing token NLL 打分，判断模型是否给真实 schema 更低 NLL。
- 指标：pairwise accuracy、positive/negative token NLL、mean margin，其中 `margin = NLL(z-) - NLL(z+)`，越大越好。

正式结果文件：`outputs/counterfactual_schema_grounding_A_ums_12label_128.json`

Positive schema 平均 token NLL：0.0482；平均 target token 数：220.89。

| Counterfactual variant | n | Pairwise acc. | Positive NLL | Negative NLL | Mean margin | Median margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| state flip | 128 | 0.8750 | 0.0482 | 0.0597 | +0.0115 | +0.0120 |
| field swap | 77 | 0.8831 | 0.0532 | 0.0665 | +0.0133 | +0.0128 |
| image swap | 128 | 0.7734 | 0.0482 | 0.0690 | +0.0208 | +0.0206 |
| null to present | 128 | 0.8750 | 0.0482 | 0.0563 | +0.0081 | +0.0080 |

结论：
- frozen-LM UMS/no-SPD 不只是复述 JSON prior：在真实图像与真实 schema 配对时，token NLL 系统性低于 state/field/image/null 反事实。
- `image_swap` 的 margin 最大，说明图像-schema 配对关系被模型捕获；但 pairwise acc. 只有 0.7734，表示仍存在较多图像间 schema 混淆。
- `state_flip` 与 `field_swap` 均约 0.88 pairwise acc.，可以作为 EMNLP 线的直接 grounding 证据。
- `null_to_present` 平均成立，但 Lung Opacity 和 Pleural Effusion 的部分 null corruption 较弱，提示 answerability/null 边界仍是需要继续加强的点。

可写入论文的谨慎表述：

> Counterfactual schema scoring shows that the frozen-LM UMS model assigns lower token NLL to image-matched schemas than to state-flipped, field-swapped, image-swapped, and null-corrupted schemas, supporting that the visual prefix is used for structured schema grounding rather than only JSON-format completion.

### 14.2 通用域 / AAAI 线可行性检查

本地公共数据目录检查位置：`H:\Xiyao_Wang\000_Public Dataset`

一级目录中可见的数据主要是：
- `CheXpert-v1.0-small`
- `NIH Chest X-rays`
- `mimic-cxr`
- `mimic-cxr_less`
- `AMOS22`
- `KITS21`
- `LIDC-IDRI-slices`
- `CAMELYON16`
- `Qwen*`
- `stg-master`

进一步检查：
- `00 Raw` 只看到 `amos22.zip`、`CheXpert_v1.0_small.zip`、`NIH Chest X-rays.zip`。
- `新建文件夹` 只看到 `.accelerate`。
- `新建文件夹 (2)` 为空。
- `stg-master` 是代码/工具目录，不是 CUB/CLEVR/COCO 这类可直接跑的通用视觉属性数据。
- 递归查找 CUB/CLEVR/COCO/attribute/bird 等候选名时因数据目录过大超时；分层检查没有发现可直接使用的通用域数据入口。

判断：
- 目前不能在“不下载新数据、不大改数据管线”的前提下直接启动 CUB/CLEVR/COCO 通用域实验。
- `VIVID_review1.md` 中的 AAAI/general 线需要通用属性数据与 schema 构造，目前本项目代码和本地数据更适合先做 CXR schema grounding base study。
- 因此当前执行策略应转为 base study：先把 CXR 内的 schema grounding、answerability/null、field-name robustness、random/shuffled control 做完整，再决定是否另开通用域数据准备。

### 14.3 当前项目是否继续

最新判断：继续，但方向需要收窄。

继续理由：
- 已有 MICCAI 补实验说明 UMS-family 明显优于 BCE、free-text、random-mask、BiomedCLIP 和 ImageNet baseline。
- visual-prefix dependency 成立：blank/shuffled image prefix 会显著提高 token NLL。
- 新的 counterfactual schema scoring 成立：真实 schema 相比四类反事实 schema 具有更低 token NLL。
- 外部 NIH transfer 整体支持 structured UMS supervision 的可迁移性。

需要收窄的原因：
- SPD 不是稳定增益，不能继续作为核心主卖点。
- pretrained LLM 的贡献目前是 in-domain 有益但 external 不稳定，不能写成 dominant source。
- 通用域/AAAI 线目前缺少可直接运行的数据入口，不应在没有数据准备的情况下继续烧 GPU。

当前最稳主线：

> VIVID-Med should be framed as answerability-aware structured CXR representation learning. Frozen-LM visual-prefix training is a controllable schema-grounding objective with verified visual dependency and counterfactual sensitivity, while SPD is treated as sensitivity/negative analysis rather than the main contribution.

下一步建议：
1. 继续补 EMNLP/base-study：answerability ablation（mask / no mask / null-as-negative）。
2. 补 shuffled-field-name 或 paraphrased-field-name 诊断，验证是否依赖固定 schema key。
3. 整理 seed mean/std 和 cost table。
4. 若一定要做 AAAI/general，需要先准备 CUB/CLEVR/COCO-attributes 之一的数据与 schema JSONL，再迁移当前 evaluation/training scaffold。

### 14.4 EMNLP/base-study: schema key/order robustness

新增脚本：`scripts/eval_schema_key_robustness.py`

目的：
- 检查 frozen-LM UMS/no-SPD 是否过度依赖训练时固定 JSON 顺序和固定 clinical field names。
- 该实验不重新训练，只在同一图像、同一 finding states 下扰动 target schema 的 key/order，再比较 teacher-forcing token NLL。

正式运行约束：
- 环境：`conda run -n vivid`
- GPU：`$env:CUDA_VISIBLE_DEVICES='1'`
- checkpoint：`outputs/ablation_A_ums_12label/checkpoints/best.pt`
- 样本数：128

正式结果文件：`outputs/schema_key_robustness_A_ums_12label_128.json`

完整性检查：
- JSON 可解析。
- `sample_count=128`。
- `rows=512`，对应 128 samples × 4 perturbation variants。

扰动定义：
- `reversed_order`：finding key 顺序完全反转，clinical key/value 不变。
- `shuffled_order`：finding key 顺序随机打乱，clinical key/value 不变。
- `clinical_key_shift`：clinical key 名称循环错位，value 保持原样。
- `generic_keys`：clinical key 替换为 `field_00`, `field_01`, ...，value 保持原样。

| Variant | n | Original NLL | Variant NLL | Mean margin | Median margin | Original better |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reversed order | 128 | 0.0543 | 0.5887 | +0.5344 | +0.5343 | 1.0000 |
| shuffled order | 128 | 0.0543 | 0.4926 | +0.4382 | +0.4320 | 1.0000 |
| clinical key shift | 128 | 0.0543 | 0.2664 | +0.2121 | +0.2081 | 1.0000 |
| generic keys | 128 | 0.0543 | 0.4131 | +0.3587 | +0.3596 | 1.0000 |

解释：
- 原始 schema 在所有 128 个样本上都比扰动 schema 更低 NLL，说明模型强烈学习了训练时 schema surface form。
- 纯顺序扰动（反转/随机）造成最大 NLL 上升，这说明当前 objective 对固定 JSON 序列顺序非常敏感。
- clinical key shift 和 generic keys 也明显升高 NLL，说明模型不仅利用视觉 prefix，也利用 clinical key token 的语言先验和训练格式先验。
- 这不是坏结果，但需要在论文中谨慎表述：counterfactual schema scoring 证明了 visual-prefix-conditioned grounding；schema-key robustness 说明 frozen-LM objective 同时高度依赖 schema serialization，不能把收益完全解释为“通用语义理解”。

论文建议写法：

> A schema-key robustness diagnostic shows that perturbing field order or replacing clinical keys substantially increases teacher-forcing NLL, indicating that the frozen-LM objective learns a strong schema serialization prior. Therefore, we interpret the method as structured schema-supervised representation learning rather than as unconstrained language understanding.

对后续实验的影响：
- `VIVID_review1.md` 里建议的 shuffled-field-name / field paraphrase robustness 已经有了一个 base-study 版本。
- 若继续往 EMNLP 线推进，下一步应补 paraphrased clinical keys（例如 `Pleural Effusion` -> `Pleural fluid`）而不是只做 generic keys。
- 若继续提升方法，应考虑训练时随机化 schema order 或加入 field-query prompt，让模型减少对固定 JSON 顺序的依赖。

### 14.5 EMNLP/base-study: answerability ablation 准备状态

`VIVID_review1.md` 把 answerability ablation 列为 P0：with mask / no mask / null-as-negative。当前状态如下。

已经存在的 baseline：
- `configs/ablation_A_ums_12label.yaml`：schema / no answerability token mask / null 保持 null。
- 输出：`outputs/ablation_A_ums_12label/checkpoints/best.pt`
- 下游：`outputs/lp_A_ums_12label/metrics_final.json`
- CheXpert macro-AUC：0.8439；CheXpert macro-F1：0.9095；NIH macro-AUC：0.7068。

新增配置：
- `configs/ablation_ums_ansmask_12label.yaml`：answerability-aware token mask。不可答字段的 state value token loss 权重为 0，可答字段权重为 1。
- `configs/ablation_ums_null_as_negative_12label.yaml`：null-as-negative。通过 `json_null_state: "absent"` 把 null 字段序列化为 absent 并完整监督。
- `configs/lp_ums_ansmask_12label.yaml`：answerability-mask checkpoint 的 CheXpert linear probe。
- `configs/lp_ums_null_as_negative_12label.yaml`：null-as-negative checkpoint 的 CheXpert linear probe。

新增 debug 配置：
- `configs/debug_ablation_ums_ansmask_12label.yaml`
- `configs/debug_ablation_ums_null_as_negative_12label.yaml`

debug 验证：
- answerability-mask debug 已跑通，输出：`outputs/debug_ablation_ums_ansmask_12label/checkpoints/final.pt`。日志确认进入 `Using answerability-aware token mask` 分支，5-step debug 完成。
- null-as-negative debug 已跑通，输出：`outputs/debug_ablation_ums_null_as_negative_12label/checkpoints/final.pt`。日志确认 `json_null_state: "absent"` 配置可进入训练并完成 5-step debug。
- 最初使用 `train_cxr.py --debug` 会失败，因为该模式强制把 LLM 换成 `sshleifer/tiny-gpt2`，当前 transformers/torch 组合因 `.bin` 权重安全限制拒绝加载；改用独立 debug YAML 后使用正式 Qwen safetensors 路径可正常运行。

正式训练命令（待 GPU1 空档启动）：

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n vivid python scripts/train_cxr.py \
  --config configs/ablation_ums_ansmask_12label.yaml

CUDA_VISIBLE_DEVICES=1 conda run -n vivid python scripts/train_cxr.py \
  --config configs/ablation_ums_null_as_negative_12label.yaml
```

对应 linear probe：

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n vivid python scripts/train_vit_baseline.py \
  --config configs/lp_ums_ansmask_12label.yaml

CUDA_VISIBLE_DEVICES=1 conda run -n vivid python scripts/train_vit_baseline.py \
  --config configs/lp_ums_null_as_negative_12label.yaml
```

当前判断：
- answerability P0 的工程入口已经补齐并通过短测，但正式 10k-step 训练和下游 probe 尚未完成。
- 因为此前 GPU1 被其他 `envs\\worse` 评估进程占用约 9.9GB，本轮没有贸然启动完整训练。
- 这项仍是下一步最优先的未完成实验；完成后才能把 “answerability is central” 写成强实验证据。

### 14.6 Answerability 正式训练运行状态

最新运行状态：
- `configs/ablation_ums_ansmask_12label.yaml` 已正式启动，当前在 GPU1 上训练。
- `configs/ablation_ums_null_as_negative_12label.yaml` 已正式启动，当前在 GPU0 上训练。
- 两个任务都是通过 `conda run -n vivid` 启动，使用正式 Qwen safetensors 路径，不再触发 `--debug` 的 tiny-gpt2 安全加载问题。

2026-05-11 10:33 复查：
- GPU1: `vivid` 训练主进程 `25804`，`nvidia-smi -i 1` 显示约 24.3GB / 24.6GB，GPU 利用率约 87%，正在训练 answerability-mask 版本。
- GPU0: `nvidia-smi` 报错 `GPU is lost. Reboot the system to recover this GPU`；因此原本绑定到 GPU0 的 `null-as-negative` 训练主进程 `26524` 虽仍在进程表中，但当前不能视为可靠完成中的任务。
- 两个训练目录目前尚未产生正式 checkpoint；`conda run` 的重定向日志仍为 0 字节，主要依赖进程、GPU 和输出目录监控判断状态。

2026-05-11 10:36 处理：
- 因 GPU0 已不可见且 `null-as-negative` 没有 checkpoint / 输出文件，已清理该条无效挂起任务的进程树。
- 当前只保留 GPU1 上的 `ansmask` 正式训练；`null-as-negative` 需要等 GPU0 恢复，或等 GPU1 空闲后重跑。

2026-05-11 10:44 复查：
- `ansmask` 训练仍在 GPU1 上运行，`nvidia-smi -i 1` 显示约 24.3GB / 24.6GB，GPU 利用率约 79%。
- `outputs/ablation_ums_ansmask_12label/checkpoints/best.pt` 已生成，文件大小约 1.07GB，时间为 2026-05-11 10:38:48；这说明正式训练已经通过第一次验证并保存了 best checkpoint。
- 还未生成 `outputs/ablation_ums_ansmask_12label/checkpoints/final.pt`，因此 `ansmask` 预训练尚未完成，linear probe 尚未启动。

2026-05-11 11:22 资源策略更新：
- GPU0 已恢复可见，但后续不再让 GPU0/GPU1 同时满载；必须始终保证至少一张卡低于 50% 负载。
- 实际执行策略改为：重负载训练只排到 GPU1；GPU0 保持空闲或低负载。这样即使 GPU1 满载，GPU0 也满足低于 50% 的要求。
- `ansmask` 原训练停在约 step 906，日志未见 traceback，但没有 `final.pt`；已保留 step 500 的 `best.pt`，后续从该 checkpoint 在 GPU1 上恢复。
- `scripts/answerability_gpu1_queue.ps1` 已加入资源门控：只有 GPU0 利用率低于 50% 且 GPU1 空闲时，才会在 GPU1 启动下一条重负载任务。

2026-05-11 11:33 恢复：
- 为避免当前沙箱网络限制触发 ModelScope/HuggingFace 请求，已将两个正式 answerability 配置的 `llm_model_name` 改为本地缓存路径 `C:/Users/Admin/.cache/modelscope/hub/models/Qwen/Qwen2___5-1___5B-Instruct`。
- 已通过 `scripts/run_ansmask_resume_gpu1.cmd` 从 `outputs/ablation_ums_ansmask_12label/checkpoints/best.pt` 在 GPU1 上恢复 `ansmask` 训练。
- 11:35 复查时 GPU0 为 0% / 11MiB，GPU1 正在加载任务，约 4913MiB；当前满足“至少一张卡低于 50%”的资源约束。

2026-05-11 11:43 复查：
- `ansmask` 恢复任务已稳定进入 GPU1：GPU1 约 92% / 22.9GB，GPU0 仍为 0% / 11MiB。
- 当前资源状态满足新约束：两张卡没有同时满载，且 GPU0 低于 50%。
- `outputs/logs/ablation_ums_ansmask_12label_resume_from_best_gpu1.log` 已开始记录启动信息；`conda run` 训练输出仍会缓冲，训练中主要依据进程、GPU 与 checkpoint 状态监控。

2026-05-11 11:47 复查：
- 已改为 `conda run --no-capture-output`，训练日志可以实时写入。
- `ansmask` 恢复确认成功：日志显示 `Checkpoint loaded ... Resuming from step 500`，随后开始训练剩余 9500 step。
- GPU 状态：GPU0 0% / 11MiB，GPU1 约 92% / 24.3GB；满足“至少一张卡低于 50%”的要求。

2026-05-11 12:11 复查：
- 普通 `Start-Process` 后台任务会随当前 shell 清理，不适合长跑；已改用 Windows Task Scheduler 启动 `scripts/run_ansmask_resume_gpu1.cmd`。
- 任务计划程序启动后的 `ansmask` 已稳定运行超过 5 分钟，日志显示从 step 500 恢复后推进到约 step 590。
- GPU 状态：GPU0 0% / 11MiB，GPU1 约 54% / 24.3GB；仍满足“至少一张卡低于 50%”的资源约束。

2026-05-11 12:28 复查：
- `ansmask` 仍在 GPU1 上稳定训练，日志进度约为恢复后的 339 / 9500 step，即全局约 step 839。
- GPU 状态：GPU0 0% / 11MiB，GPU1 约 92% / 24.3GB；资源约束满足。
- 当前尚未到下一个保存点，全局 step 1000 时预计会生成 `step_1000.pt`。

2026-05-11 12:44 复查：
- `ansmask` 已完成全局 step 1000 的验证与保存。
- 新文件：`outputs/ablation_ums_ansmask_12label/checkpoints/step_1000.pt`，同时 `best.pt` 在 2026-05-11 12:40:56 更新。
- 训练已继续向后推进，日志约为恢复后的 547 / 9500 step。
- GPU 状态：GPU0 0% / 11MiB，GPU1 约 76% / 24.3GB；资源约束满足。

2026-05-11 12:55 自动队列：
- 已创建 `scripts/answerability_gpu1_queue_once.ps1` 和 `scripts/run_answerability_queue_once.cmd`，并注册计划任务 `VIVID_answerability_queue_once`，每 5 分钟执行一次。
- 队列策略：只在 GPU0 利用率低于 50% 且 GPU1 空闲时启动下一阶段；当前 GPU1 忙，因此不会重复启动。
- 已预注册后续计划任务：`VIVID_lp_ansmask_gpu1`、`VIVID_null_as_negative_gpu1`、`VIVID_lp_null_as_negative_gpu1`，用于在上游产物完成后顺序接续。

2026-05-11 13:22 资源守卫更新：
- 根据最新约束，后续仍采用保守策略：重负载实验只让一张卡执行，另一张卡必须保持低于 50% 负载；当前继续让 GPU1 跑 `ansmask`，GPU0 保持空闲/低负载。
- 新增 `scripts/answerability_resource_guard_once.ps1` 与 `scripts/run_answerability_resource_guard_once.cmd`，并注册计划任务 `VIVID_answerability_resource_guard_once`，每 5 分钟检查一次 GPU0/GPU1。
- 守卫逻辑：若连续两次采样确认 GPU0 和 GPU1 利用率都超过 50%，立即停止当前 VIVID 训练/linear-probe 计划任务与对应进程；之后由队列在资源恢复后从最新 checkpoint 继续。
- 本次手动守卫检查结果：GPU0 为 0% / 11MiB，GPU1 为 99% / 24301MiB，满足“至少一张卡不超过 50%”的要求，因此未暂停训练。
- `ansmask` 当前仍在 GPU1 上运行，日志进度约为恢复后的 835 / 9500 step；`outputs/ablation_ums_ansmask_12label/checkpoints/final.pt` 尚未生成，后续 linear probe 暂不启动。

2026-05-11 13:33 复查：
- `VIVID_answerability_resource_guard_once` 已按计划任务实际运行，13:27 与 13:32 两次记录均显示 GPU0 为 0% / 11MiB，GPU1 正在训练但 GPU0 保持低负载。
- `VIVID_answerability_queue_once` 仍每 5 分钟检查一次；因 GPU1 已有训练进程和约 24.3GB 显存占用，队列正确阻止重复启动 `ansmask`。
- `ansmask` 日志已推进到恢复后的约 974 / 9500 step；当前仍只有 `step_1000.pt` 与 `best.pt`，尚未到下一保存点或最终 checkpoint。

2026-05-11 13:35 快速复查：
- GPU0 为 0% / 11MiB，GPU1 为 88% / 24301MiB；仍满足至少一张卡不超过 50% 负载。
- `ansmask` 日志推进到恢复后的约 995 / 9500 step，训练仍在继续。

2026-05-11 13:39 复查与恢复脚本修正：
- `ansmask` 已完成一次验证，随后继续训练到恢复后的约 1002 / 9500 step；`best.pt` 在 2026-05-11 13:37:21 更新。
- 当前 checkpoint 状态：`best.pt` 最新，`step_1000.pt` 仍是 12:41 的旧 step 文件，尚未生成 `final.pt`。
- 已新增 `scripts/select_latest_checkpoint.ps1`，并更新 `scripts/run_ansmask_resume_gpu1.cmd`：后续若资源守卫停止任务，恢复时会在 `best.pt` 与 `step_*.pt` 中按修改时间选择最新 checkpoint，避免错误回退到旧 `step_1000.pt`。
- 13:37 守卫检查：GPU0 为 0% / 11MiB，GPU1 为 73% / 24301MiB；资源约束仍满足。

2026-05-11 13:42 EMNLP P1 诊断脚本准备：
- 根据 `VIVID_review1.md` 中的 field paraphrase robustness 要求，新增 `scripts/eval_field_paraphrase_robustness.py`。
- 该脚本比较同一图像/同一 finding state 下，原始临床字段名、医学同义改写字段名、通俗改写字段名的 teacher-forcing token NLL，用于判断 schema supervision 是否只是依赖固定字段 token。
- 当前只完成脚本准备与静态语法检查：`python -m py_compile scripts/eval_field_paraphrase_robustness.py` 通过。尚未运行正式评估，因为 GPU1 正在训练且 GPU0 需要保持低负载。
- checkpoint 选择脚本 `scripts/select_latest_checkpoint.ps1` 已手动验证，会返回当前最新的 `best.pt`。

2026-05-11 13:43 EMNLP P0 诊断脚本准备：
- 根据 `VIVID_review1.md` 中“Prefix dependency 完整扩展到 positive/counterfactual schemas”的要求，新增 `scripts/eval_counterfactual_prefix_dependency.py`。
- 该脚本对同一批 positive/counterfactual schema，在 `image`、`blank`、`shuffled` 三种 prefix 下分别计算 positive NLL、pairwise accuracy、margin；用于验证 counterfactual schema scoring 是否真的依赖正确图像 prefix。
- 静态语法检查通过：`python -m py_compile scripts/eval_counterfactual_prefix_dependency.py scripts/eval_field_paraphrase_robustness.py`。
- 尚未启动该评估，原因同上：GPU1 正在跑 `ansmask`，GPU0 必须保持低负载。
- 13:42 守卫检查：GPU0 为 0% / 11MiB，GPU1 为 70% / 24301MiB；资源约束仍满足。

2026-05-11 13:46 自动队列扩展：
- 新增 `scripts/run_cf_prefix_dependency_gpu1.cmd` 与 `scripts/run_field_paraphrase_gpu1.cmd`，均绑定 `CUDA_VISIBLE_DEVICES=1`，使用 `conda run --no-capture-output -n vivid`。
- 已注册计划任务 `VIVID_cf_prefix_dependency_gpu1` 和 `VIVID_field_paraphrase_gpu1`。
- `scripts/answerability_gpu1_queue_once.ps1` 已扩展：在 answerability-mask 训练/LP、null-as-negative 训练/LP 全部完成后，自动顺序运行 counterfactual-prefix dependency eval 和 field-paraphrase robustness eval。
- `scripts/answerability_resource_guard_once.ps1` 已把这两个新评估任务纳入资源守卫；若 GPU0/GPU1 同时超过 50%，同样会停止这些评估进程。
- 手动测试队列脚本通过，当前因 GPU1 正忙而正确阻止启动新任务；13:46 复查 GPU0 为 0% / 11MiB，GPU1 为 80% / 24301MiB。

2026-05-11 13:48 复查：
- `ansmask` 训练日志推进到恢复后的约 1139 / 9500 step；`best.pt` 仍是最新 checkpoint，`final.pt` 尚未生成。
- 计划任务状态正常：`VIVID_answerability_queue_once` 下一次 13:49，`VIVID_answerability_resource_guard_once` 下一次 13:52。
- GPU0 为 0% / 11MiB，GPU1 为 35% / 24301MiB；仍满足至少一张卡不超过 50% 负载。

2026-05-11 13:50 复查：
- `ansmask` 训练日志推进到恢复后的约 1165 / 9500 step；尚未产生新的 step checkpoint 或 `final.pt`。
- `VIVID_answerability_queue_once` 在 13:49 正常执行，识别到 GPU1 忙并阻止重复启动；状态日志已包含新增的 `cf_prefix_final=False` 与 `field_paraphrase_final=False`。
- GPU0 为 0% / 11MiB，GPU1 为约 72% / 24301MiB；资源约束满足。

2026-05-11 13:52 EMNLP P1 manual-audit 入口：
- 新增 `scripts/export_schema_manual_audit.py`，用于从 UMS JSONL 导出人工审计表。
- 已生成 `outputs/schema_manual_audit_chexpert_val_200.csv`，共 201 行（1 行表头 + 200 个 val 样本），包含 image path、sample metadata、12 个 selected finding 的 state/answerability/uncertainty 以及空白 `audit_notes` 字段。
- 已生成 `outputs/schema_manual_audit_chexpert_val_200_summary.json`；200 样本平均 answerable finding 数为 3.915，平均 null 数为 8.085，平均 present 数为 2.63，平均 uncertain 数为 0.48。
- 该产物只是人工审计入口，不等价于人工审计完成；后续仍需人工或半自动 agreement analysis。
- 同时 `ansmask` 训练推进到恢复后的约 1208 / 9500 step；GPU0 仍为 0% / 11MiB。

2026-05-11 13:53 answerability / missingness 数据分析：
- 新增 `scripts/analyze_schema_answerability.py`，用于统计 UMS JSONL 中每个 finding 的 state 与 answerability 分布。
- 已生成 `outputs/schema_answerability_chexpert_val.json` / `.csv`：val split 共 1000 样本、12000 个 label slots，answerable rate 为 0.3293，null rate 为 0.6707，present rate 为 0.2245，absent rate 为 0.0627，uncertain rate 为 0.0422。
- 已生成 `outputs/schema_answerability_chexpert_train.json` / `.csv`：train split 共 29000 样本、348000 个 label slots，answerable rate 为 0.3312，null rate 为 0.6688，present rate 为 0.2256，absent rate 为 0.0636，uncertain rate 为 0.0421。
- 该结果支持论文中“CXR schema 存在大量不可答/null 字段，因此 answerability 不是装饰变量”的论证，也解释为什么 answerability ablation 是 P0。
- val split 中 null rate 最高的字段包括 Fracture 0.902、Lung Lesion 0.892、Pneumonia 0.835、Cardiomegaly 0.774、Enlarged Cardiomediastinum 0.773；这些是 answerability/missingness 讨论的主要例子。
- uncertain rate 最高的字段包括 Atelectasis 0.124、Consolidation 0.100、Pneumonia 0.077、Pleural Effusion 0.050、Edema 0.043；这些可作为 uncertain/null failure case 的候选。
- answerable rate 最高的字段包括 Pleural Effusion 0.586、Support Devices 0.568、Lung Opacity 0.538、Pneumothorax 0.395、Atelectasis 0.354；这些字段更适合观察 image-grounded schema scoring。

2026-05-11 13:55 复查：
- `ansmask` 训练日志推进到恢复后的约 1266 / 9500 step；当前仍只有 `best.pt` 与 `step_1000.pt`，尚未生成 `final.pt`。
- `VIVID_answerability_queue_once` 在 13:54 正常执行，因 GPU1 忙而阻止重复启动。
- GPU0 为 0% / 11MiB，GPU1 为 42% / 24301MiB；当前满足至少一张卡不超过 50% 负载。
- 新增的 4 个 Python 工具脚本已统一通过 `py_compile`：counterfactual-prefix eval、field-paraphrase eval、manual-audit export、answerability distribution analysis。

2026-05-11 13:58 复查：
- `ansmask` 训练日志推进到恢复后的约 1300 / 9500 step；尚未到下一次验证/保存完成点。
- `VIVID_answerability_resource_guard_once` 在 13:57 正常执行，记录 GPU0 0% / 11MiB、GPU1 64% / 24301MiB，策略通过。
- 当前即时 GPU 状态：GPU0 0% / 11MiB，GPU1 68% / 24301MiB；资源约束满足。

2026-05-11 14:00 复查：
- `ansmask` 训练日志推进到恢复后的约 1329 / 9500 step；仍未生成新的 step checkpoint 或 `final.pt`。
- `VIVID_answerability_queue_once` 在 13:59 正常执行，记录 GPU0 0% / 11MiB、GPU1 49% / 24301MiB；虽然瞬时利用率低于 50%，但 GPU1 仍有训练 compute app 和 24.3GB 显存占用，因此队列正确阻止重复启动。
- 即时 GPU 状态：GPU0 0% / 11MiB，GPU1 47% / 24301MiB；资源约束满足。
- 训练日志错误扫描未见 traceback / CUDA OOM / RuntimeError；仅有启动时 FlashAttention2 不可用并回退 eager attention 的提示，当前不影响训练继续。

2026-05-11 14:01 复查：
- `ansmask` 训练日志推进到恢复后的约 1353 / 9500 step；当前仍未生成新的 checkpoint。
- 计划任务状态正常：`VIVID_ansmask_resume_gpu1` 仍为 Running；队列下一次约 14:04，资源守卫下一次约 14:02。
- GPU0 为 0% / 11MiB，GPU1 为 68% / 24301MiB；资源约束满足。

2026-05-11 14:03 复查：
- `VIVID_answerability_resource_guard_once` 在 14:02 正常执行，记录 GPU0 0% / 11MiB、GPU1 63% / 24301MiB，策略通过。
- `ansmask` 训练日志推进到恢复后的约 1378 / 9500 step；尚未产生新的 checkpoint。
- 即时 GPU 状态：GPU0 0% / 11MiB，GPU1 49% / 24301MiB；资源约束满足。

2026-05-11 14:05 复查：
- `VIVID_answerability_queue_once` 在 14:04 正常执行，记录 GPU0 0% / 11MiB、GPU1 58% / 24301MiB；因 GPU1 忙而阻止重复启动。
- `ansmask` 训练日志推进到恢复后的约 1402 / 9500 step；仍未生成新的 checkpoint。
- 即时 GPU 状态：GPU0 0% / 11MiB，GPU1 20% / 24301MiB；虽然瞬时利用率较低，但训练进程仍占用 GPU1 显存，队列继续视为忙。

2026-05-11 14:06 复查：
- `ansmask` 训练日志推进到恢复后的约 1422 / 9500 step；尚未开始下一轮验证，未生成新的 checkpoint。
- `VIVID_answerability_queue_once` 最近一次 14:04 检查正常，因 GPU1 忙而阻止重复启动。
- 即时 GPU 状态：GPU0 0% / 11MiB，GPU1 55% / 24301MiB；资源约束满足。

2026-05-11 14:08 复查：
- `VIVID_answerability_resource_guard_once` 在 14:07 正常执行，记录 GPU0 0% / 11MiB、GPU1 60% / 24301MiB，策略通过。
- `ansmask` 训练日志推进到恢复后的约 1443 / 9500 step；距离下一次验证点约 57 step，尚未生成新的 checkpoint。
- 即时 GPU 状态：GPU0 0% / 11MiB，GPU1 100% / 24301MiB；资源约束满足，因为 GPU0 仍低于 50%。

2026-05-11 14:16 验证/保存点：
- `ansmask` 已完成恢复后约 1500 step 附近的一次验证；日志记录 `Step 1500: val_loss = 0.0426`，随后继续训练。
- 由于训练从全局 step 500 恢复，恢复后进度条约 1500 / 9500 对应全局 step 2000；已生成 `outputs/ablation_ums_ansmask_12label/checkpoints/step_2000.pt`。
- `best.pt` 在 2026-05-11 14:14:02 更新，`step_2000.pt` 在 2026-05-11 14:14:04 生成；说明该保存点已真实落盘。
- 即时 GPU 状态：GPU0 0% / 11MiB，GPU1 66% / 24301MiB；资源约束满足。
- `VIVID_answerability_queue_once` 在 14:14 正常执行，因 GPU1 忙而阻止重复启动；`VIVID_answerability_resource_guard_once` 在 14:12 正常执行并通过。
- 验证后训练继续推进到恢复后的约 1534 / 9500 step。

2026-05-11 14:18 复查：
- `VIVID_answerability_resource_guard_once` 在 14:17 正常执行，记录 GPU0 0% / 11MiB、GPU1 49% / 24301MiB，策略通过。
- `ansmask` 训练日志推进到恢复后的约 1553 / 9500 step；`step_2000.pt` 与最新 `best.pt` 均存在，尚未生成 `final.pt`。
- 即时 GPU 状态：GPU0 0% / 11MiB，GPU1 47% / 24301MiB；资源约束满足。

2026-05-11 14:19 复查：
- `VIVID_answerability_queue_once` 在 14:19 正常执行，记录 GPU0 0% / 11MiB、GPU1 28% / 24301MiB；因 GPU1 仍有训练显存/compute app，占用判定为 busy，未重复启动。
- `ansmask` 训练日志推进到恢复后的约 1579 / 9500 step；`step_2000.pt` 与最新 `best.pt` 仍是当前可恢复点，尚未生成 `final.pt`。
- 即时 GPU 状态：GPU0 0% / 11MiB，GPU1 12% / 24301MiB；资源约束满足。

2026-05-11 14:20 复查：
- `ansmask` 训练日志推进到恢复后的约 1593 / 9500 step；当前仍未生成 `final.pt`。
- 当前可恢复点仍为 `best.pt` 与 `step_2000.pt`。
- 即时 GPU 状态：GPU0 0% / 11MiB，GPU1 2% / 24301MiB；虽然 GPU1 瞬时利用率低，但显存仍由训练进程占用，因此继续视为 busy。

2026-05-11 14:23 复查：
- `VIVID_answerability_resource_guard_once` 在 14:22 正常执行，记录 GPU0 0% / 11MiB、GPU1 91% / 24301MiB，策略通过。
- `ansmask` 训练日志推进到恢复后的约 1628 / 9500 step；当前仍未生成 `final.pt`。
- 即时 GPU 状态：GPU0 0% / 11MiB，GPU1 0% / 24301MiB；GPU1 仍被训练进程占用显存，队列应继续视为 busy。

2026-05-11 14:24 复查：
- `VIVID_answerability_queue_once` 在 14:24 正常执行，记录 GPU0 0% / 11MiB、GPU1 65% / 24301MiB；因 GPU1 忙而阻止重复启动。
- `ansmask` 训练日志推进到恢复后的约 1655 / 9500 step；当前仍未生成 `final.pt`。
- 即时 GPU 状态：GPU0 0% / 11MiB，GPU1 15% / 24301MiB；资源约束满足。

2026-05-11 14:28 复查：
- `VIVID_answerability_resource_guard_once` 在 14:27 正常执行，记录 GPU0 0% / 11MiB、GPU1 84% / 24301MiB，策略通过。
- `ansmask` 训练日志推进到恢复后的约 1699 / 9500 step；当前仍未生成 `final.pt`。
- 即时 GPU 状态：GPU0 0% / 11MiB，GPU1 46% / 24301MiB；资源约束满足。

2026-05-11 14:30 复查：
- `VIVID_answerability_queue_once` 在 14:29 正常执行，记录 GPU0 0% / 11MiB、GPU1 82% / 24301MiB；因 GPU1 忙而阻止重复启动。
- `ansmask` 训练日志推进到恢复后的约 1735 / 9500 step；当前仍未生成 `final.pt`。
- 即时 GPU 状态：GPU0 0% / 11MiB，GPU1 26% / 24301MiB；资源约束满足。

2026-05-11 14:38 复查：
- `ansmask` 训练日志推进到恢复后的约 1829 / 9500 step；当前仍未生成 `final.pt`。
- `VIVID_answerability_queue_once` 在 14:34 正常执行，因 GPU1 忙而阻止重复启动。
- `VIVID_answerability_resource_guard_once` 在 14:37 正常执行，记录 GPU0 0% / 11MiB、GPU1 90% / 24301MiB，策略通过。
- 即时 GPU 状态：GPU0 0% / 11MiB，GPU1 9% / 24301MiB；资源约束满足。

2026-05-11 14:39 复查：
- `VIVID_answerability_queue_once` 在 14:39 正常执行，记录 GPU0 0% / 11MiB、GPU1 30% / 24301MiB；因 GPU1 仍被训练进程占用显存而阻止重复启动。
- `ansmask` 训练日志推进到恢复后的约 1851 / 9500 step；当前仍未生成 `final.pt`。
- 即时 GPU 状态：GPU0 0% / 11MiB，GPU1 0% / 24301MiB；资源约束满足。

2026-05-11 14:52 复查：
- `ansmask` 到达恢复后 2000 / 9500 step 验证点，日志记录 `Step 2000: val_loss = 0.0412`，随后训练继续推进到约 2004 / 9500 step。
- checkpoint 内部核验：`best.pt` 与 `step_2000.pt` 的 `global_step` 均为 2000，`best_val_loss` 均为 0.04117919921875；因此保存点有效，即使文件时间显示为 14:14。
- `VIVID_answerability_queue_once` 在 14:49 正常执行，因 GPU1 忙而阻止重复启动；`VIVID_answerability_resource_guard_once` 在 14:47 正常执行并通过。
- 即时 GPU 状态：GPU0 0% / 11MiB，GPU1 71% / 24301MiB；资源约束满足。

2026-05-11 14:53 复查：
- `ansmask` 训练日志推进到恢复后的约 2021 / 9500 step；当前仍未生成 `final.pt`，尚未进入下游 LP。
- `VIVID_answerability_resource_guard_once` 在 14:52 正常执行，记录 GPU0 0% / 11MiB、GPU1 69% / 24301MiB，策略通过。
- 即时 GPU 状态：GPU0 0% / 11MiB，GPU1 53% / 24301MiB；资源约束满足。

### 14.7 VIVID_review1 对照审计（当前未完成）

| 要求 | 当前证据 | 状态 |
| --- | --- | --- |
| 通用域 / AAAI 可行性 | 已检查本地公共数据目录，当前没有可直接运行的 CUB/CLEVR/COCO 属性入口；见 14.2 | 通用域暂失败，转 base study |
| EMNLP P0: counterfactual schema scoring | `outputs/counterfactual_schema_grounding_A_ums_12label_128.json` 已生成并汇总在 14.1 | 已完成 |
| EMNLP P0: answerability ablation | `configs/ablation_ums_ansmask_12label.yaml` 正在 GPU1 训练；`null-as-negative` 队列待跑 | 进行中 |
| EMNLP P0: random-LM 或 shuffled-field-name | `outputs/schema_key_robustness_A_ums_12label_128.json` 已完成 shuffled/order/key diagnostic；random-LM same-architecture 已实现入口并接入 GPU1 队列，尚未运行完成 | 部分完成 |
| EMNLP P0: prefix dependency 扩展到 counterfactual schemas | `scripts/eval_counterfactual_prefix_dependency.py` 已实现并通过语法检查；已接入 GPU1 队列 | 待运行 |
| EMNLP P1: field paraphrase robustness | `scripts/eval_field_paraphrase_robustness.py` 已实现并通过语法检查；已接入 GPU1 队列 | 待运行 |
| EMNLP P1: report-to-schema / manual audit | `outputs/schema_manual_audit_chexpert_val_200.csv` 与 summary 已生成，但尚无人工标注/agreement | 部分完成 |
| 资源约束 | 最新策略已覆盖为只使用 GPU1；`VIVID_answerability_queue_once` 每 5 分钟只检查 GPU1 是否空闲，`VIVID_answerability_resource_guard_once` 只记录 GPU1 状态且不再按 GPU0/50% 规则停止任务 | 已更新 |

当前结论：目标尚未完成。最关键的阻塞项是 answerability ablation 的正式结果；在 `ansmask` 训练完成前，下游实验继续等待 GPU1 空闲后按队列顺序执行。

监控文件：
- `outputs/logs/ablation_ums_ansmask_12label_train.log`
- `outputs/logs/ablation_ums_null_as_negative_12label_train.log`
- `outputs/logs/answerability_watch.log`
- `outputs/logs/answerability_queue.log`
- `outputs/logs/answerability_queue_once.log`
- `outputs/logs/answerability_resource_guard_once.log`

当前状态判断：
- answerability P0 不再只是配置入口，已经进入正式训练阶段；`ansmask` 正在 GPU1 上继续训练，`null-as-negative` 已排队等待 GPU1 空闲后重跑。
- 还没有最终 checkpoint / metrics，因此不能提前把它写成完成。
- 后续按 5 分钟间隔检查进程、GPU1、checkpoint 和日志；若 GPU1 的 `ansmask` 完成，将顺序启动 `lp_ums_ansmask_12label`、`ablation_ums_null_as_negative_12label` 重跑、`lp_ums_null_as_negative_12label`、counterfactual-prefix eval、random-LM train/LP、field-paraphrase eval。
- 当前资源策略以 2026-05-11 16:41 记录为准：后续只使用 GPU1，不再管 GPU0，也不再执行 GPU0/50% 负载约束。

2026-05-11 14:55 资源策略复核（GPU0 已恢复后）：
- 当前用户约束更新为：GPU0 和 GPU1 可以都可用，但不能同时满载；任意时刻两张卡中必须至少有一张卡利用率不超过 50%。
- 当前执行策略保持保守：`ansmask` 仍只在 GPU1 上运行，GPU0 作为低负载保留卡；后续队列也只在 GPU1 空闲且 GPU0 低于 50% 时启动下一项重负载实验。
- 当前 GPU 快照：GPU0 0% / 11MiB，GPU1 9% / 24301MiB；GPU1 显存仍由 `C:\Users\Admin\anaconda3\envs\vivid\python.exe` 训练进程占用，因此队列继续判定 GPU1 busy。
- `VIVID_answerability_resource_guard_once` 已按规则每 5 分钟执行；守卫逻辑是若连续两次采样确认 GPU0 与 GPU1 都超过 50%，则停止当前 VIVID 训练/评估任务，避免双卡同时高负载。
- `VIVID_answerability_queue_once` 最近一次 14:54:54 正常完成，下一次约 14:59:59；`VIVID_answerability_resource_guard_once` 最近一次 14:52:52 正常完成，下一次约 14:57:57。
- `ansmask` 训练日志推进到恢复后的约 2056 / 9500 step；当前可恢复 checkpoint 为 `best.pt` 与 `step_2000.pt`，尚未生成 `final.pt`，因此下游 LP 和 `null-as-negative` 尚未启动。

2026-05-11 15:03 EMNLP P0 random-LM same-architecture 对照入口：
- 新增模型开关：`models/vivid_model.py` 支持 `llm_random_init: true`，会加载同一 Qwen2.5 tokenizer/config，但不加载 pretrained 权重，而是从 config 随机初始化同架构 causal LM；随后仍冻结 LLM，只训练 ViT 与 projector。
- 新增训练配置：`configs/ablation_ums_random_lm_12label.yaml`，输出到 `outputs/ablation_ums_random_lm_12label`；新增调试配置 `configs/debug_ablation_ums_random_lm_12label.yaml`。
- 新增对应 linear probe 配置：`configs/lp_ums_random_lm_12label.yaml`，读取 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`。
- 新增 GPU1 runner：`scripts/run_random_lm_gpu1.cmd` 与 `scripts/run_lp_random_lm_gpu1.cmd`。
- 已注册计划任务：`VIVID_random_lm_gpu1`、`VIVID_lp_random_lm_gpu1`，当前均为 `Ready`，没有立即启动。
- 已把 random-LM 两阶段纳入 `scripts/answerability_gpu1_queue_once.ps1` 队列末尾：在 answerability ablation、counterfactual-prefix eval、field-paraphrase eval 全部完成后，才依次启动 random-LM 预训练与 LP。
- 已把 random-LM 两阶段纳入 `scripts/answerability_resource_guard_once.ps1` 守卫：若两张 GPU 同时超过 50%，会一并停止这些任务。
- 静态验证：`python -m py_compile models/vivid_model.py scripts/train_cxr.py` 通过；手动运行 `answerability_gpu1_queue_once.ps1` 与 `answerability_resource_guard_once.ps1` 通过，且因 GPU1 busy 没有启动新任务。
- 当前训练状态：`ansmask` 继续在 GPU1 运行，日志推进到恢复后的约 2162 / 9500 step；GPU0 0% / 11MiB，GPU1 36% / 24301MiB，满足至少一张卡不超过 50% 的资源约束。

2026-05-11 15:10 EMNLP P1 report-to-schema 数据源审计：
- 新增脚本：`scripts/audit_schema_source_fields.py`，用于检查本地 CheXpert/NIH CSV 与 UMS JSONL 是否包含可用于 report-to-schema extraction 的原始自由文本字段。
- 生成产物：`outputs/schema_source_field_audit.json`。
- 审计范围包括 `data/dataset/CheXpert-v1.0-small/train.csv`、`valid.csv`、`processed/chexpert_sampled_30k.csv`、`chexpert_ums_train.jsonl`、`chexpert_ums_val.jsonl`、`nih_external_test.csv`、`nih_external_test_ums.jsonl`。
- 结论：`has_local_report_text_field=false`；当前本地源暴露的是 labels、paths、metadata 和 UMS schema，没有 raw report text 字段。
- 因此 `VIVID_review1.md` 里的 report-to-schema extraction 在当前 workspace 不能完整完成，需要额外加入 MIMIC-CXR reports 或其他 report-text source；当前可完成的是 manual-audit export 与 schema source audit，而不是 report-text extraction agreement。
- 该脚本已通过 `python -m py_compile scripts/audit_schema_source_fields.py`，并用 `conda run --no-capture-output -n vivid python scripts/audit_schema_source_fields.py --output outputs/schema_source_field_audit.json` 生成结果。

2026-05-11 15:12 队列优先级修正：
- 根据 `VIVID_review1.md`，`random-LM same-architecture` 属于 EMNLP/AAAI P0，`field paraphrase robustness` 属于 P1。
- 已更新 `scripts/answerability_gpu1_queue_once.ps1`：在 `null-as-negative` 及 counterfactual-prefix eval 完成后，优先运行 `VIVID_random_lm_gpu1` 和 `VIVID_lp_random_lm_gpu1`；二者完成后再运行 `VIVID_field_paraphrase_gpu1`。
- 手动运行队列脚本验证通过；由于 GPU1 仍被 `ansmask` 占用，脚本只记录状态并阻止重复启动，没有启动新任务。
- 当前 GPU 约束仍满足：GPU0 0% / 11MiB，GPU1 显存由训练进程占用但瞬时利用率波动；至少一张卡低于 50%。

2026-05-11 15:13 复查：
- `ansmask` 训练日志推进到恢复后的约 2302 / 9500 step；当前仍未到下一次验证/保存点，尚未生成 `final.pt`。
- 当前 checkpoint 仍为 `best.pt` 与 `step_2000.pt`；`outputs/ablation_ums_ansmask_12label/checkpoints/final.pt` 不存在，因此下游 LP、`null-as-negative`、counterfactual-prefix、random-LM、field-paraphrase 均未启动。
- `VIVID_answerability_queue_once` 在 15:12 手动/计划检查中正确识别 GPU1 busy，并阻止重复启动 `ansmask`。
- `VIVID_answerability_resource_guard_once` 在 15:12 正常执行，记录 GPU0 0% / 11MiB、GPU1 67% / 24301MiB，策略通过。
- 即时 GPU 状态：GPU0 0% / 11MiB，GPU1 22% / 24301MiB；资源约束满足。

2026-05-11 15:19 复查：
- `ansmask` 训练日志推进到恢复后的约 2391 / 9500 step；当前仍未到下一次验证/保存点，尚未生成 `final.pt`。
- 当前 checkpoint 仍为 `best.pt` 与 `step_2000.pt`，未出现新的 `step_*.pt`。
- `VIVID_answerability_queue_once` 在 15:19 正常执行，记录 GPU0 0% / 11MiB、GPU1 54% / 24301MiB；因 GPU1 busy 阻止重复启动。
- `VIVID_answerability_resource_guard_once` 在 15:17 正常执行，记录 GPU0 0% / 11MiB、GPU1 40% / 24301MiB，策略通过。
- 即时 GPU 状态：GPU0 0% / 11MiB，GPU1 55% / 24301MiB；资源约束满足，因为 GPU0 低于 50%。

2026-05-11 15:27 复查：
- `ansmask` 已到恢复后的 2500 / 9500 step 验证点；日志记录 `Step 2500: val_loss = 0.0423`，当前正在验证循环中。
- checkpoint 目录暂未出现新文件；仍只有 `best.pt`、`step_1000.pt`、`step_2000.pt`。需要等待验证/保存结束后再判断是否更新 `best.pt` 或生成下一步 checkpoint。
- `VIVID_answerability_queue_once` 在 15:24 正常执行，记录 GPU0 0% / 11MiB、GPU1 58% / 24301MiB；因 GPU1 busy 阻止重复启动。
- `VIVID_answerability_resource_guard_once` 在 15:27 正常执行，记录 GPU0 0% / 11MiB、GPU1 64% / 24301MiB，策略通过。
- 即时 GPU 状态：GPU0 0% / 11MiB，GPU1 4% / 24301MiB；验证期间瞬时利用率波动，但 GPU1 仍由训练进程占用，队列继续视为 busy。

2026-05-11 15:35 验证/保存点：
- `ansmask` 完成恢复后 2500 / 9500 step 附近的验证，并继续训练到约 2593 / 9500 step。
- 验证日志：`Step 2500: val_loss = 0.0423`；训练器随后更新了 `best.pt` 并保存了新的 `step_3000.pt`。
- checkpoint 内部核验：`best.pt` 与 `step_3000.pt` 的 `global_step` 均为 3000，`best_val_loss` 均为 0.039765625。
- 当前可恢复点更新为 `outputs/ablation_ums_ansmask_12label/checkpoints/best.pt` 或 `step_3000.pt`；`final.pt` 尚未生成，因此下游 LP 仍未启动。
- `VIVID_answerability_queue_once` 在 15:34 正常执行，记录 GPU0 0% / 11MiB、GPU1 55% / 24301MiB；因 GPU1 busy 阻止重复启动。
- 即时 GPU 状态：GPU0 0% / 11MiB，GPU1 38% / 24301MiB；资源约束满足。

2026-05-11 15:42 复查：
- `ansmask` 训练日志推进到恢复后的约 2688 / 9500 step；训练继续运行。
- 当前最新保存点仍为 `best.pt` / `step_3000.pt`，二者内部 `global_step=3000`，尚未生成 `final.pt`。
- `VIVID_answerability_queue_once` 在 15:39 正常执行，记录 GPU0 0% / 11MiB、GPU1 83% / 24301MiB；因 GPU1 busy 阻止重复启动。
- `VIVID_answerability_resource_guard_once` 在 15:42 正常执行，记录 GPU0 0% / 11MiB、GPU1 34% / 24301MiB，策略通过。
- 即时 GPU 状态：GPU0 0% / 11MiB，GPU1 34% / 24301MiB；资源约束满足。

2026-05-11 15:48 复查：
- `ansmask` 训练日志推进到恢复后的约 2762 / 9500 step；训练继续运行，尚未到下一个验证点。
- 当前最新保存点仍为 `best.pt` / `step_3000.pt`；尚未生成 `final.pt`。
- `VIVID_answerability_queue_once` 在 15:44 正常执行，记录 GPU0 0% / 11MiB、GPU1 57% / 24301MiB；因 GPU1 busy 阻止重复启动。
- `VIVID_answerability_resource_guard_once` 在 15:47 正常执行，记录 GPU0 0% / 11MiB、GPU1 91% / 24301MiB，策略通过。
- 即时 GPU 状态：GPU0 0% / 11MiB，GPU1 0% / 24301MiB；GPU1 显存仍由训练进程占用，队列继续视为 busy。

2026-05-11 16:00 复查：
- `ansmask` 训练日志推进到恢复后的约 2936 / 9500 step，接近下一次验证点但尚未开始验证。
- 当前最新保存点仍为 `best.pt` / `step_3000.pt`；尚未生成 `final.pt`。
- `VIVID_answerability_queue_once` 在 15:59 正常执行，记录 GPU0 0% / 11MiB、GPU1 42% / 24301MiB；因 GPU1 busy 阻止重复启动。
- `VIVID_answerability_resource_guard_once` 在 15:57 正常执行，记录 GPU0 0% / 11MiB、GPU1 92% / 24301MiB，策略通过。
- 即时 GPU 状态：GPU0 0% / 11MiB，GPU1 66% / 24301MiB；资源约束满足，因为 GPU0 低于 50%。

2026-05-11 16:06 复查：
- `ansmask` 已进入恢复后约 3000 / 9500 step 的验证点；日志显示当前正在 `Validating`，尚未完成该轮验证。
- 当前日志中最近的已落盘信息仍是此前的全局 `step_3000.pt`；需要等待本轮验证结束后再确认是否更新 `best.pt` 或生成新的 checkpoint。
- 当前最新保存点仍为 `best.pt` / `step_3000.pt`，尚未生成 `final.pt`。
- `VIVID_answerability_queue_once` 在 16:04 正常执行，记录 GPU0 0% / 11MiB、GPU1 39% / 24301MiB；因 GPU1 busy 阻止重复启动。
- `VIVID_answerability_resource_guard_once` 在 16:02 正常执行，记录 GPU0 0% / 11MiB、GPU1 50% / 24301MiB，策略通过。
- 即时 GPU 状态：GPU0 0% / 11MiB，GPU1 18% / 24301MiB；验证期间 GPU 利用率波动，但 GPU1 仍由训练进程占用。

2026-05-11 16:10 复查：
- 恢复后约 3000 / 9500 step 的验证已经结束，训练继续推进到约 3049 / 9500 step。
- 该轮验证未产生新的 checkpoint；当前 checkpoint 目录仍为 `best.pt`、`step_1000.pt`、`step_2000.pt`、`step_3000.pt`。
- 解释：本轮 validation 后没有优于当前 `best_val_loss=0.039765625`，并且尚未到下一个保存间隔，因此 `best.pt` 与 `step_3000.pt` 均未更新。
- `VIVID_answerability_queue_once` 在 16:09 正常执行，记录 GPU0 0% / 11MiB、GPU1 0% / 24301MiB；虽然瞬时利用率为 0%，GPU1 显存仍被训练进程占用，因此队列正确判定 busy。
- `VIVID_answerability_resource_guard_once` 在 16:07 正常执行，记录 GPU0 0% / 11MiB、GPU1 46% / 24301MiB，策略通过。
- 即时 GPU 状态：GPU0 0% / 11MiB，GPU1 0% / 24301MiB；资源约束满足。

2026-05-11 16:16 复查：
- `ansmask` 训练日志推进到恢复后的约 3137 / 9500 step；训练继续运行。
- 当前 checkpoint 仍为 `best.pt` / `step_3000.pt`，尚未生成 `final.pt`。
- `VIVID_answerability_queue_once` 在 16:14 正常执行，记录 GPU0 0% / 11MiB、GPU1 0% / 24301MiB；GPU1 显存仍被训练进程占用，因此队列正确判定 busy。
- `VIVID_answerability_resource_guard_once` 在 16:12 正常执行，记录 GPU0 0% / 11MiB、GPU1 31% / 24301MiB，策略通过。
- 即时 GPU 状态：GPU0 0% / 11MiB，GPU1 80% / 24301MiB；资源约束满足，因为 GPU0 低于 50%。

2026-05-11 16:23 复查：
- `ansmask` 训练日志推进到恢复后的约 3198 / 9500 step；训练继续运行。
- 当前 checkpoint 仍为 `best.pt` / `step_3000.pt`，尚未生成 `final.pt`。
- `VIVID_answerability_queue_once` 在 16:19 正常执行，记录 GPU0 0% / 11MiB、GPU1 10% / 24301MiB；GPU1 显存仍被训练进程占用，因此队列正确判定 busy。
- `VIVID_answerability_resource_guard_once` 在 16:22 正常执行，记录 GPU0 0% / 11MiB、GPU1 16% / 24301MiB，策略通过。
- 即时 GPU 状态：GPU0 0% / 11MiB，GPU1 1% / 24301MiB；资源约束满足。

2026-05-11 16:41 资源策略更新（覆盖此前 GPU0/50% 约束）：
- 根据最新要求，后续只使用 GPU1；不再管 GPU0，也不再要求“两张卡中至少一张不超过 50% 负载”。
- 已更新 `scripts/answerability_gpu1_queue_once.ps1`：队列启动条件只检查 GPU1；若 GPU1 显存已占用或存在 compute app，则阻止重复启动；不再查询或依赖 GPU0。
- 已更新 `scripts/answerability_resource_guard_once.ps1`：守护脚本只记录 GPU1 状态；GPU0/50% 守护和停止训练逻辑已经禁用/移除，不会再因为 GPU0 或 50% 规则停止 VIVID 任务。
- 手动验证：16:39 队列脚本记录 `GPU1-only policy active; GPU1 util=97%, mem=24301MiB`，因 GPU1 正在跑 `ansmask` 而正确阻止重复启动；16:38 守护脚本记录 `GPU1-only policy active; GPU1 util=99%, mem=24301MiB; GPU0/50% guard disabled; no stop action`。
- 5 分钟检查仍在：`VIVID_answerability_queue_once` 最近一次 16:39:39 成功，下一次 16:44:44；`VIVID_answerability_resource_guard_once` 最近一次 16:37:37 成功，下一次 16:42:42。
- 当前训练状态：`VIVID_ansmask_resume_gpu1` 仍在 GPU1 运行，日志推进到恢复后的约 3308 / 9500 step；`outputs/ablation_ums_ansmask_12label/checkpoints/final.pt` 尚未生成，所以下游 LP、`null-as-negative`、counterfactual-prefix、random-LM、field-paraphrase 仍按队列等待 GPU1 空闲后继续。

2026-05-11 16:47 GPU1-only 队列一致性复查：
- 旧的长循环队列脚本 `scripts/answerability_gpu1_queue.ps1` 也已同步为 GPU1-only：启动判定只查 GPU1，不再查 GPU0，也不再使用 50% 门控。
- 当前计划任务实际使用的是 `scripts/answerability_gpu1_queue_once.ps1`；旧循环脚本没有注册为计划任务，但已同步以避免误用。
- 所有 `scripts/run_*gpu1.cmd` 检查结果均绑定 `CUDA_VISIBLE_DEVICES=1`，未发现 VIVID runner 绑定 GPU0。
- 语法验证：`scripts/answerability_gpu1_queue.ps1` 通过 PowerShell parser；计划队列最近一次 16:44:44 成功，下一次 16:49:49；守护任务下一次 16:47:47。
- 训练继续推进：`ansmask` 日志约为恢复后的 3359 / 9500 step，`final.pt` 仍未生成。

2026-05-11 16:54 复查：
- GPU1 上的 VIVID 训练仍在运行：`C:\Users\Admin\anaconda3\envs\vivid\python.exe`，日志推进到恢复后的约 3409 / 9500 step。
- GPU 快照：GPU1 0% / 24301MiB（瞬时利用率波动但显存仍占用）；GPU0 上存在非 VIVID 进程（`GameViewer.exe` 与 `envs\worse\python.exe`），按最新要求不处理 GPU0。
- 计划任务正常：`VIVID_answerability_queue_once` 最近一次 16:49:49 成功，下一次 16:54:54；`VIVID_answerability_resource_guard_once` 最近一次 16:52:52 成功，下一次 16:57:57。
- 队列日志确认 GPU1-only：16:49 记录 `GPU1-only policy active; GPU1 util=64%, mem=24301MiB`，因 GPU1 busy 阻止重复启动。
- 守护日志确认不会停止任务：16:52 记录 `GPU1-only policy active; GPU1 util=10%, mem=24301MiB; GPU0/50% guard disabled; no stop action`。
- checkpoint 仍为 `best.pt` / `step_3000.pt`，尚未生成 `final.pt`；错误扫描未见 traceback / CUDA OOM / RuntimeError，仅有启动时 FlashAttention2 不可用并回退 eager attention 的提示。

2026-05-11 17:00 复查：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上仍是 `vivid` 环境的训练进程，当前日志推进到恢复后的约 3449 / 9500 step。
- GPU 快照：GPU1 18% / 24301MiB；GPU0 0% / 11MiB。后续仍只关心 GPU1 是否可继续跑队列。
- 计划任务正常：`VIVID_answerability_queue_once` 最近一次 16:59:59 成功，下一次 17:04:04；`VIVID_answerability_resource_guard_once` 最近一次 16:57:57 成功，下一次 17:02:02。
- 队列日志：16:59 记录 `GPU1-only policy active; GPU1 util=91%, mem=24301MiB`，因 GPU1 busy 阻止重复启动。
- 守护日志：16:57 记录 `GPU1-only policy active; GPU1 util=4%, mem=24301MiB; GPU0/50% guard disabled; no stop action`，没有停止任何任务。
- checkpoint 未变化：`best.pt` / `step_3000.pt` 仍为最新可恢复点，`final.pt` 尚未生成；错误扫描仍未见 traceback / CUDA OOM / RuntimeError。

2026-05-11 17:05 复查：
- `ansmask` 训练继续推进到恢复后的约 3498 / 9500 step，接近下一轮验证点但尚未看到新的 validation 结果。
- 最近已记录的验证仍是 `Step 3000: val_loss = 0.0398`；checkpoint 目录仍只有 `best.pt`、`step_1000.pt`、`step_2000.pt`、`step_3000.pt`，尚未生成 `final.pt`。
- GPU 快照：GPU1 13% / 24301MiB；显存仍由 VIVID 训练进程占用，队列继续判定 GPU1 busy。
- 计划任务正常：`VIVID_answerability_queue_once` 最近一次 17:04:04 成功，下一次 17:09:09；`VIVID_answerability_resource_guard_once` 最近一次 17:02:02 成功，下一次 17:07:07。
- 队列日志：17:04 记录 `GPU1-only policy active; GPU1 util=29%, mem=24301MiB`，因 GPU1 busy 阻止重复启动。
- 守护日志：17:02 记录 `GPU1-only policy active; GPU1 util=0%, mem=24301MiB; GPU0/50% guard disabled; no stop action`，无停止动作。

2026-05-11 17:13 验证/保存点：
- `ansmask` 已完成恢复后约 3500 / 9500 step 附近的验证，日志记录 `Step 3500: val_loss = 0.0474`，随后训练继续推进到约 3526 / 9500 step。
- 该轮之后产生了新的 checkpoint：`outputs/ablation_ums_ansmask_12label/checkpoints/step_4000.pt`，同时 `best.pt` 在 17:09 更新。
- checkpoint 内部核验：`best.pt` 与 `step_4000.pt` 的 `global_step` 均为 4000，`best_val_loss` 均为 0.0389794921875，说明当前最新可恢复点有效。
- `final.pt` 尚未生成，因此下游 `lp_ums_ansmask_12label`、`null-as-negative`、counterfactual-prefix、random-LM、field-paraphrase 仍未启动。
- 队列日志：17:09 记录 `GPU1-only policy active; GPU1 util=34%, mem=24301MiB`，因 GPU1 busy 阻止重复启动。
- 守护日志：17:12 记录 `GPU1-only policy active; GPU1 util=56%, mem=24301MiB; GPU0/50% guard disabled; no stop action`，未停止任务。

2026-05-11 17:18 复查：
- `ansmask` 继续运行，日志推进到恢复后的约 3580 / 9500 step；当前 `VIVID_ansmask_resume_gpu1` 仍为 Running。
- GPU 快照：GPU1 31% / 24301MiB；GPU1 显存仍由训练占用，队列继续判定 busy。
- 计划任务正常：`VIVID_answerability_queue_once` 最近一次 17:14:14 成功，下一次 17:19:19；`VIVID_answerability_resource_guard_once` 最近一次 17:17:17 成功，下一次 17:22:22。
- 队列日志：17:14 记录 `GPU1-only policy active; GPU1 util=13%, mem=24301MiB`，阻止重复启动。
- 守护日志：17:17 记录 `GPU1-only policy active; GPU1 util=96%, mem=24301MiB; GPU0/50% guard disabled; no stop action`。
- checkpoint 仍以 `best.pt` / `step_4000.pt` 为最新可恢复点，`outputs/ablation_ums_ansmask_12label/checkpoints/final.pt` 仍不存在。

2026-05-11 17:24 复查：
- `ansmask` 继续运行，日志推进到恢复后的约 3607 / 9500 step；当前仍未到下一次验证/保存点。
- GPU 快照：GPU1 0% / 24301MiB（瞬时利用率波动，显存仍占用）；队列继续将 GPU1 判定为 busy。
- 计划任务正常：`VIVID_answerability_queue_once` 最近一次 17:24:24 成功，下一次 17:29:29；`VIVID_answerability_resource_guard_once` 最近一次 17:22:22 成功，下一次 17:27:27。
- 队列日志：17:24 记录 `GPU1-only policy active; GPU1 util=0%, mem=24301MiB`，因显存/compute 占用阻止重复启动。
- 守护日志：17:22 记录 `GPU1-only policy active; GPU1 util=72%, mem=24301MiB; GPU0/50% guard disabled; no stop action`。
- checkpoint 未变化：最新可恢复点仍为 `best.pt` / `step_4000.pt`，`final.pt` 仍不存在；最近 validation 仍为 `Step 3500: val_loss = 0.0474`。

2026-05-11 17:33 用户要求暂停 GPU 进程：
- 已停止当前项目的 GPU 训练进程：`ablation_ums_ansmask_12label.yaml` 对应的 `vivid` 环境 python 进程已终止。
- 已禁用所有 `VIVID_*` 计划任务，包括 `VIVID_ansmask_resume_gpu1`、`VIVID_answerability_queue_once`、`VIVID_answerability_resource_guard_once`、下游 LP、`null-as-negative`、counterfactual-prefix、random-LM、field-paraphrase 任务，避免 5 分钟队列自动重启。
- 停止后 GPU1 快照为 0MiB / 0%：本项目已不再占用 GPU1。GPU0 上仍有非本项目进程（例如 `envs\worse\python.exe`、`GameViewer.exe`），未处理。
- 停止前训练日志推进到恢复后的约 3625 / 9500 step；当前最新可恢复点仍为 `outputs/ablation_ums_ansmask_12label/checkpoints/best.pt` 或 `step_4000.pt`，二者内部此前已核验 `global_step=4000`。
- `final.pt` 尚未生成，因此 answerability ablation 仍未完成；下游 LP、`null-as-negative`、counterfactual-prefix、random-LM、field-paraphrase 均处于暂停/未启动状态。

2026-05-30 16:08 复跑恢复与 GPU0 优先策略：
- 当前目标恢复执行；两张 3090 初始均空闲，GPU0/GPU1 温度约 30-34C、总功耗约 16-18W。
- 已按最新要求改为 GPU0 优先：新增 `scripts/run_*_gpu0.cmd` runner，队列目标卡改为 `target_gpu=0`，计划任务改为调用 `VIVID_*_gpu0`。
- 旧 GPU1 上刚恢复的 `VIVID_ansmask_resume_gpu1` 已终止，避免继续占用 GPU1；GPU1 当前空闲。
- `ansmask` 训练已从 `outputs/ablation_ums_ansmask_12label/checkpoints/step_4000.pt` 在 GPU0 上恢复，日志进入正式训练循环，当前为 global step 4000 后继续跑剩余 6000 step。
- 当前 GPU 快照：GPU0 约 24.3GB / 24GB 显存占用、util 约 80%+、功耗约 218W、温度约 62C；GPU1 空闲约 8W；总功耗约 226W，低于 350-400W 长期功耗边界。
- 速度估计：恢复后约 3.2-3.5 秒/step；`ansmask` 预训练预计还需约 5.5-6 小时。之后队列顺序仍为 `lp_ums_ansmask` -> `null-as-negative` 预训练/LP -> counterfactual-prefix eval -> random-LM 预训练/LP -> field-paraphrase eval。
- 当前结果判断仍沿用上一阶段：已有 P0 结果总体支持 UMS-family 表示，但 LLM 贡献只能说 modest；answerability ablation 尚未出最终 downstream 结果，所以还不能把 “answerability is central” 写成强结论。

2026-05-30 16:10 复查：
- `VIVID_ansmask_resume_gpu0` 为 Running；GPU0 上有唯一 VIVID compute app，GPU1 空闲。
- `ansmask` 从 global step 4000 恢复后已推进到约 66 / 6000 step；尚未生成新的 `step_5000.pt` 或 `final.pt`。
- 当前日志未见 traceback / RuntimeError / CUDA OOM；训练 loss 在 0.034-0.039 附近波动，属于恢复初期可接受范围。
- GPU 快照：GPU0 约 24.3GB 显存、util 约 43-66%、功耗约 253-289W、温度约 73-78C；GPU1 空闲约 9W；总功耗约 262-298W，未超过 350-400W 长期边界。
- 队列与守卫均在 16:10 正常运行：队列因 GPU0 busy 阻止重复启动，守卫记录资源 OK 未停止任务。
- 基于当前 4.0s/step 左右速度，`ansmask` 预训练剩余约 6.7 小时；全部剩余队列仍粗估约 29-35 小时。

2026-05-30 16:12 复查：
- `VIVID_ansmask_resume_gpu0` 仍为 Running；GPU0 上的 VIVID compute app PID 为 21840（进程名因权限不足无法展开），GPU1 空闲。
- GPU 快照：GPU0 约 24.3GB 显存、util 约 37%、功耗约 212W、温度约 65C；GPU1 约 8W、33C；当前总功耗约 220W，资源状态健康。
- 训练日志推进到恢复后的约 89 / 6000 step，loss 在 0.0325-0.0378 附近；未见 traceback / RuntimeError / CUDA OOM / exitcode。
- checkpoint 仍停在 `step_4000.pt` / `best.pt`；`step_5000.pt` 和 `final.pt` 尚未生成，下游 `lp_ums_ansmask` 尚未启动。
- 队列与资源守卫 16:10 最近一次执行成功，下一次 16:15；队列正确阻止重复启动，守卫未触发停止。
- 当前剩余实验产物 8 项均未完成：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 基于当前 4.0-4.6s/step 波动，`ansmask` 预训练剩余约 6.5-7.5 小时；全部剩余队列仍粗估约 29-36 小时。阶段性结果仍符合运行预期，但 answerability 的论文结论要等 LP 指标完成后再定。

2026-05-30 16:14 复查：
- `ansmask` 仍在 GPU0 正常训练，恢复后进度约 109 / 6000 step；当前 checkpoint 仍是 `step_4000.pt` / `best.pt`，尚未到 `step_5000.pt` 保存点。
- GPU 快照：GPU0 约 24.3GB 显存、util 约 93%、功耗约 217W、温度约 68C；GPU1 空闲约 8W、32C；总功耗约 225W。
- 训练 loss 仍在 0.0325-0.0377 区间内波动；日志未见 traceback / RuntimeError / CUDA OOM / exitcode。
- 计划任务状态：`VIVID_ansmask_resume_gpu0` 为 Running；队列/守卫最近一次 16:10 成功，下一次 16:15。当前无需人工干预。
- 剩余时间估计基本不变：`ansmask` 预训练剩余约 6.4-7.4 小时；全部剩余队列约 29-36 小时。

2026-05-30 16:15 复查：
- `VIVID_ansmask_resume_gpu0` 继续 Running；队列与资源守卫 16:15 成功执行，下一次 16:20。
- 训练日志推进到恢复后的约 127 / 6000 step；尚未生成 `step_5000.pt` 或 `final.pt`。
- GPU 快照：GPU0 约 24.3GB 显存、util 在 0-93% 间波动、功耗约 148-217W、温度约 64-68C；GPU1 空闲约 8W、32C；16:15 守卫记录总功耗 212.85W、最高温 69C。
- 队列在 16:15 正确阻止重复启动，守卫判定 resource OK；无 traceback / RuntimeError / CUDA OOM / exitcode。
- 当前没有失败 case；运行符合预期，但仍未产生 answerability ablation 的最终指标。

2026-05-30 16:17 复查：
- `ansmask` 训练继续在 GPU0 推进，恢复后约 144 / 6000 step；仍未到 `step_5000.pt` 保存点。
- GPU 快照：GPU0 约 24.3GB 显存、util 约 87%、功耗约 197W、温度约 69C；GPU1 空闲约 8W、32C；总功耗约 205W。
- 训练 loss 最近在 0.0314-0.0359 区间内波动；未见 traceback / RuntimeError / CUDA OOM / exitcode。
- 基于当前约 4.2s/step 的平均速度，`ansmask` 剩余约 6.8 小时；全部剩余队列仍约 29-36 小时。

2026-05-30 16:18 复查：
- `ansmask` 恢复后进度约 159 / 6000 step；仍在 GPU0 训练，未到新 checkpoint。
- GPU 快照：GPU0 约 24.3GB 显存、util 约 59%、功耗约 207W、温度约 67C；GPU1 空闲约 8W、32C；总功耗约 215W。
- 最近 loss 在 0.0358-0.0399 区间内波动；日志未见 traceback / RuntimeError / CUDA OOM / exitcode。
- 队列/守卫最新有效记录仍为 16:15；状态正常，下一轮 16:20。当前无需人工干预。

2026-05-30 16:20 复查：
- `ansmask` 训练继续运行，恢复后约 170 / 6000 step；仍未到 `step_5000.pt` 保存点。
- GPU 快照：GPU0 约 24.3GB 显存，瞬时 util 约 42-56%、功耗约 174-252W、温度约 66-68C；GPU1 空闲约 8W、32C。
- 16:20 资源守卫记录总功耗 181.87W、最高温 66C，判定 resource OK；队列正确阻止重复启动。
- 最近 loss 在 0.0371-0.0415 区间内波动；未见 traceback / RuntimeError / CUDA OOM / exitcode。
- 当前没有失败 case，也没有新产物完成；`ansmask` 预计仍需约 6.5-7 小时，整体剩余约 29-36 小时。

2026-05-30 16:40 复查（按 20 分钟同步节奏）：
- `VIVID_ansmask_resume_gpu0` 仍为 Running；GPU0 上的 VIVID compute app PID 仍为 21840，GPU1 空闲。
- `ansmask` 从 global step 4000 恢复后推进到约 417 / 6000 step，即全局约 4417 / 10000 step；`step_5000.pt` 和 `final.pt` 仍未生成。
- GPU 快照：GPU0 约 24.3GB 显存、瞬时 util 约 83%、功耗约 131W、温度约 59C；GPU1 约 8W、32C；16:40 守卫记录总功耗约 264.5W、最高温 66C，未触发停止。
- 队列在 16:40 正确识别 GPU0 busy 并阻止重复启动；资源守卫判定 resource OK，无人工介入需求。
- 最近 loss 在 0.0344-0.0405 区间内波动；日志未见 traceback / RuntimeError / CUDA OOM / exitcode，因此当前没有失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。其中 `ansmask final` 正在跑，另外 7 项等待队列自动接续。
- 按最近约 5.2s/step 的速度估计，距离 `step_5000.pt` 约 0.8-0.9 小时，距离 `ansmask final.pt` 约 8.0-8.5 小时；全部目标队列粗估仍需约 31-39 小时。
- 阶段性判断：运行状态符合预期；但 answerability ablation 的论文结论仍需等 `ansmask final` 与下游 LP 指标完成后再确认。

2026-05-30 17:00 复查（按 20 分钟同步节奏）：
- `VIVID_ansmask_resume_gpu0` 仍为 Running；GPU0 上的 VIVID compute app PID 仍为 21840，GPU1 空闲。
- `ansmask` 从 global step 4000 恢复后推进到约 590 / 6000 step，即全局约 4590 / 10000 step；训练已穿过 global 4500 附近的验证段并继续推进。
- checkpoint 目录仍只有 `best.pt`、`step_4000.pt`、`step_3000.pt`、`step_2000.pt`、`step_1000.pt`；`step_5000.pt` 和 `final.pt` 尚未生成。下一次保存点仍是 global step 5000。
- GPU 快照：GPU0 约 24.3GB 显存、瞬时 util 约 0-81%、功耗约 110-252W、温度约 55-68C；GPU1 约 8W、32C。17:00 守卫记录总功耗约 250.69W、最高温 68C，资源安全。
- 队列在 17:00 正确识别 GPU0 busy 并阻止重复启动；资源守卫判定 resource OK，无人工介入需求。
- 最近 loss 在 0.0347-0.0425 区间内波动；日志未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。其中 `ansmask final` 正在跑，另外 7 项等待队列自动接续。
- 按当前含验证开销的速度估计，距离 `step_5000.pt` 约 0.6-0.8 小时，距离 `ansmask final.pt` 约 8.5-10.5 小时；全部目标队列粗估约 32-40 小时。
- 阶段性判断：训练运行与资源状态符合预期；answerability ablation 的最终结论仍需等 `ansmask final` 和后续 LP 指标完成后确认。

2026-05-30 17:20 复查（按 20 分钟同步节奏）：
- `VIVID_ansmask_resume_gpu0` 仍为 Running；GPU0 上的 VIVID compute app PID 仍为 21840，GPU1 空闲。
- `ansmask` 从 global step 4000 恢复后推进到约 764 / 6000 step，即全局约 4764 / 10000 step；尚未到 global step 5000 保存点。
- checkpoint 目录仍只有 `best.pt`、`step_4000.pt`、`step_3000.pt`、`step_2000.pt`、`step_1000.pt`；`step_5000.pt` 和 `final.pt` 尚未生成。
- GPU 快照：GPU0 约 24.3GB 显存、瞬时 util 约 1-62%、功耗约 72-272W、温度约 53-67C；GPU1 约 7-8W、32C。17:20 守卫记录总功耗约 279.89W、最高温 67C，低于 350-400W 边界。
- 队列在 17:20 正确识别 GPU0 busy 并阻止重复启动；资源守卫判定 resource OK，无人工介入需求。
- 最近 loss 在 0.0307-0.0418 区间内波动；日志未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。其中 `ansmask final` 正在跑，另外 7 项等待队列自动接续。
- 按当前速度估计，距离 `step_5000.pt` 约 0.4-0.6 小时，距离 `ansmask final.pt` 约 8.0-10.0 小时；全部目标队列粗估约 32-40 小时。
- 阶段性判断：训练运行与资源状态符合预期；answerability ablation 的最终结论仍需等 `ansmask final` 和后续 LP 指标完成后确认。

2026-05-30 17:40 复查（按 20 分钟同步节奏）：
- `VIVID_ansmask_resume_gpu0` 仍为 Running；GPU0 上的 VIVID compute app PID 仍为 21840，GPU1 空闲。
- `ansmask` 从 global step 4000 恢复后推进到约 939 / 6000 step，即全局约 4939 / 10000 step；已经接近 global step 5000 保存点。
- checkpoint 目录仍只有 `best.pt`、`step_4000.pt`、`step_3000.pt`、`step_2000.pt`、`step_1000.pt`；`step_5000.pt` 和 `final.pt` 尚未生成。下一轮重点检查 `step_5000.pt` 是否落盘。
- GPU 快照：GPU0 约 24.3GB 显存、瞬时 util 约 96%、功耗约 133W、温度约 59C；GPU1 约 7-8W、32C。17:40 守卫记录总功耗约 241.1W、最高温 66C，低于 350-400W 边界。
- 队列在 17:40 正确识别 GPU0 busy 并阻止重复启动；资源守卫判定 resource OK，无人工介入需求。
- 最近 loss 在 0.0344-0.0441 区间内波动；日志未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。其中 `ansmask final` 正在跑，另外 7 项等待队列自动接续。
- 按当前速度估计，距离 `step_5000.pt` 约 5-10 分钟，距离 `ansmask final.pt` 约 7.5-9.5 小时；全部目标队列粗估约 31-39 小时。
- 阶段性判断：训练运行与资源状态符合预期；answerability ablation 的最终结论仍需等 `ansmask final` 和后续 LP 指标完成后确认。

2026-05-30 18:00 复查（按 20 分钟同步节奏）：
- `VIVID_ansmask_resume_gpu0` 仍为 Running；GPU0 上的 VIVID compute app PID 仍为 21840，GPU1 空闲。
- `ansmask` 已成功越过 global step 5000，当前从 global step 4000 恢复后推进到约 1111 / 6000 step，即全局约 5111 / 10000 step。
- 新 checkpoint 已落盘：`outputs/ablation_ums_ansmask_12label/checkpoints/step_5000.pt`，时间为 2026-05-30 17:50:55，大小约 1.07GB。已用 `torch.load(..., map_location='cpu')` 轻量核验元数据，内部 `global_step=5000`、`best_val_loss=0.0389794921875`，可作为新的有效恢复点。
- 最近验证结果：日志记录 `Step 4500: val_loss = 0.0435`，未优于当前 `best_val_loss`，因此 `best.pt` 未更新属于预期。
- `final.pt` 尚未生成，因此下游 `lp_ums_ansmask` 仍未启动；队列在 18:00 仍正确识别 GPU0 busy 并阻止重复启动。
- GPU 快照：GPU0 约 24.3GB 显存、瞬时 util 约 33-55%、功耗约 167-244W、温度约 60-69C；GPU1 约 7-8W、31C。18:00 守卫记录总功耗约 252.66W、最高温 69C，低于 350-400W 边界。
- 最近 loss 在 0.0339-0.0402 区间内波动；日志未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。`step_5000.pt` 是中间恢复点，不计入目标产物完成数。
- 按当前含验证/保存开销的速度估计，距离 `ansmask final.pt` 约 7.5-9.0 小时；全部目标队列粗估约 30-38 小时。
- 阶段性判断：训练运行、恢复点保存和资源状态均符合预期；answerability ablation 的最终结论仍需等 `ansmask final` 和后续 LP 指标完成后确认。

2026-05-30 18:20 复查（按 20 分钟同步节奏）：
- `VIVID_ansmask_resume_gpu0` 仍为 Running；GPU0 上的 VIVID compute app PID 仍为 21840，GPU1 空闲。
- `ansmask` 从 global step 4000 恢复后推进到约 1291 / 6000 step，即全局约 5291 / 10000 step；继续稳定向 `final.pt` 推进。
- checkpoint 目录最新有效恢复点仍为 `outputs/ablation_ums_ansmask_12label/checkpoints/step_5000.pt`；`final.pt` 尚未生成，因此下游 `lp_ums_ansmask` 仍未启动。
- GPU 快照：GPU0 约 24.3GB 显存、瞬时 util 约 28-47%、功耗约 149-166W、温度约 58-64C；GPU1 约 7-8W、32C。18:20 守卫记录总功耗约 162.56W、最高温 64C，低于 350-400W 边界。
- 队列在 18:20 正确识别 GPU0 busy 并阻止重复启动；资源守卫判定 resource OK，无人工介入需求。
- 最近 loss 在 0.0322-0.0402 区间内波动；日志未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。`step_5000.pt` 是中间恢复点，不计入目标产物完成数。
- 按最近 20 分钟约 180 step 的速度估计，距离 `ansmask final.pt` 约 8.5-9.5 小时；全部目标队列粗估约 30-38 小时。
- 阶段性判断：训练运行、恢复点状态和资源状态均符合预期；answerability ablation 的最终结论仍需等 `ansmask final` 和后续 LP 指标完成后确认。

2026-05-30 18:40 复查（按 20 分钟同步节奏）：
- `VIVID_ansmask_resume_gpu0` 仍为 Running；GPU0 上的 VIVID compute app PID 仍为 21840，GPU1 空闲。
- `ansmask` 从 global step 4000 恢复后推进到约 1453 / 6000 step，即全局约 5453 / 10000 step；继续向 `final.pt` 推进。
- checkpoint 目录最新有效恢复点仍为 `outputs/ablation_ums_ansmask_12label/checkpoints/step_5000.pt`；`final.pt` 尚未生成，因此下游 `lp_ums_ansmask` 仍未启动。
- GPU 快照：GPU0 约 24.3GB 显存、瞬时 util 约 45-49%、功耗约 76-181W、温度约 53-60C；GPU1 约 7-8W、32C。18:40 守卫记录总功耗约 176.58W、最高温 60C，低于 350-400W 边界。
- 队列在 18:40 正确识别 GPU0 busy 并阻止重复启动；资源守卫判定 resource OK，无人工介入需求。
- 最近 loss 在 0.0344-0.0390 区间内波动；日志未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。`step_5000.pt` 是中间恢复点，不计入目标产物完成数。
- 按最近 20 分钟约 162 step 的速度估计，距离 `ansmask final.pt` 约 9.0-9.5 小时；全部目标队列粗估约 30-38 小时。
- 阶段性判断：训练运行、恢复点状态和资源状态均符合预期；answerability ablation 的最终结论仍需等 `ansmask final` 和后续 LP 指标完成后确认。

2026-05-30 19:05 复查（按 20 分钟同步节奏，含 19:00/19:05 队列与守卫记录）：
- `VIVID_ansmask_resume_gpu0` 仍为 Running；GPU0 上的 VIVID compute app PID 仍为 21840，GPU1 保持空闲。计划任务 `VIVID_ansmask_resume_gpu0` 最近启动时间为 2026-05-30 16:05:16，当前状态仍是 Running。
- `ansmask` 从 global step 4000 恢复后推进到约 1595 / 6000 step，即全局约 5595 / 10000 step；当前仍在向 `final.pt` 推进。
- checkpoint 状态：`outputs/ablation_ums_ansmask_12label/checkpoints/step_5000.pt` 已是有效中间恢复点；`best.pt` 在 2026-05-30 18:52:42 更新。已用 `torch.load(..., map_location='cpu')` 核验 `best.pt` 元数据，内部 `global_step=5500`、`best_val_loss=0.03747265625`，说明 5500 附近验证结果优于此前 best。
- 日志中可直接检索到的最近验证行包括 `Step 5000: val_loss = 0.0452`；快速文本检索未抓到 `Step 5500` 的逐行输出，但 `best.pt` 元数据已确认 5500 处产生了新的 best checkpoint。
- `final.pt` 尚未生成，因此下游 `lp_ums_ansmask` 仍未启动；队列日志路径为 `outputs/logs/answerability_queue_once.log`，19:00 与 19:05 均正确识别目标 GPU/总功耗处于 busy 或 launch-threshold 状态，并阻止重复启动。
- GPU 快照：19:03 左右 GPU0 约 24.3GB 显存、瞬时 util 约 24%、功耗约 127W、温度约 59C；GPU1 为 0MiB、约 7W、32C。19:00 守卫记录总功耗约 181.39W、最高温 63C；19:05 队列侧记录总功耗约 127.42W、最高温 53C，均低于 350-400W 边界。
- 最近 loss 在约 0.0309-0.0399 区间内波动；日志未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。`step_5000.pt` 与 `best.pt` 都是中间训练/恢复证据，不计入目标产物完成数。
- 按最近 20-40 分钟含验证开销的速度估计，距离 `ansmask final.pt` 约 10-12 小时；如果训练速度恢复到 18:20 前后的水平，可能回落到约 8.5-10.5 小时。全部目标队列粗估约 32-41 小时。
- 阶段性判断：训练、checkpoint 更新、资源守卫和自动队列均符合预期；`best_val_loss` 更新是积极信号。但 answerability ablation 的论文结论仍需等 `ansmask final` 和后续 LP 指标完成后再确认。

2026-05-30 19:25 复查（按 20 分钟同步节奏）：
- `VIVID_ansmask_resume_gpu0` 仍为 Running；GPU0 上的 VIVID compute app PID 仍为 21840，GPU1 空闲。
- `ansmask` 从 global step 4000 恢复后推进到约 1754 / 6000 step，即全局约 5754 / 10000 step；当前仍在第一项 `ansmask` 预训练阶段。
- checkpoint 状态未出现异常变化：最新中间恢复点仍为 `step_5000.pt`，当前 best 仍为 2026-05-30 18:52:42 更新的 `best.pt`；`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动。
- GPU 快照：19:27 左右 GPU0 约 24.3GB 显存、瞬时 util 约 99%、功耗约 108W、温度约 56C；GPU1 为 0MiB、约 7W、32C。19:25 资源守卫记录总功耗约 276.21W、最高温 64C，低于 350-400W 边界，也低于 83C 温度硬阈值。
- 队列在 19:25 正确识别所有目标产物仍未完成，并因 GPU0 busy / 当前功耗阈值阻止重复启动；资源守卫判定 resource OK，无停止动作。
- 最近 loss 在约 0.0322-0.0375 区间内波动；日志未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 按 19:05 到 19:25 这一段约 159 step / 20 分钟的速度估计，距离 `ansmask final.pt` 约 9-10.5 小时；考虑验证/保存阶段开销，保守按 9-11 小时看。全部目标队列粗估约 31-40 小时。
- 阶段性判断：训练进度、资源状态和自动队列都符合预期；目前只能确认训练过程健康，最终论文判断仍需等 `ansmask final` 与后续 LP 指标完成。

2026-05-30 19:45 复查（按 20 分钟同步节奏）：
- `VIVID_ansmask_resume_gpu0` 仍为 Running；GPU0 上的 VIVID compute app PID 仍为 21840，GPU1 空闲。
- `ansmask` 从 global step 4000 恢复后推进到约 1903 / 6000 step，即全局约 5903 / 10000 step；仍在第一项 `ansmask` 预训练阶段，尚未到下一个 `step_6000.pt` 保存点。
- checkpoint 状态未出现异常变化：最新中间恢复点仍为 `step_5000.pt`，当前 best 仍为 2026-05-30 18:52:42 更新的 `best.pt`；`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动。
- GPU 快照：19:48 左右 GPU0 约 24.3GB 显存、瞬时 util 约 20%、功耗约 160W、温度约 58C；GPU1 为 0MiB、约 8W、32C。19:45 资源守卫记录总功耗约 209.04W、最高温 62C，低于 350-400W 边界，也低于 83C 温度硬阈值。
- 队列在 19:45 正确识别所有目标产物仍未完成，并因 GPU0 busy / 当前功耗阈值阻止重复启动；资源守卫判定 resource OK，无停止动作。
- 最近 loss 在约 0.0304-0.0389 区间内波动；日志未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 按 19:25 到 19:45 这一段约 149 step / 20 分钟的速度估计，距离 `ansmask final.pt` 约 9.5-11 小时；全部目标队列粗估约 31-40 小时。
- 阶段性判断：训练进度、资源状态和自动队列都符合预期；`best.pt` 仍保留 18:52 的改进结果，但最终论文判断仍需等 `ansmask final` 与后续 LP 指标完成。

2026-05-30 20:05 复查（按 20 分钟同步节奏，实际检查到 20:09）：
- `VIVID_ansmask_resume_gpu0` 仍为 Running；GPU0 上的 VIVID compute app PID 仍为 21840，GPU1 空闲。
- `ansmask` 从 global step 4000 恢复后推进到约 2000 / 6000 step，即全局约 6000 / 10000 step；当前处在 global 6000 附近的验证阶段。原始日志尾部显示 `Validating: 0/250`，因此 GPU0 显存仍占用但瞬时 util 降到 0% 属于验证/保存阶段的可解释现象。
- 验证与 best 状态：日志现在可直接检索到 `Step 5500: val_loss = 0.0375` 和 `Checkpoint saved: .../best.pt`；这与此前 `best.pt` 元数据 `global_step=5500`、`best_val_loss=0.03747265625` 一致。
- checkpoint 状态：`step_6000.pt` 尚未落盘，最新中间恢复点仍为 `step_5000.pt`；当前 best 仍为 2026-05-30 18:52:42 更新的 `best.pt`。下一轮重点检查 global 6000 验证完成后是否生成 `step_6000.pt`。
- GPU 快照：20:09 左右 GPU0 约 24.3GB 显存、瞬时 util 0%、功耗约 9W、温度约 35C；GPU1 为 0MiB、约 8W、32C。20:05 资源守卫记录总功耗约 145.65W、最高温 51C，低于 350-400W 边界，也低于 83C 温度硬阈值。
- 队列在 20:05 正确识别所有目标产物仍未完成，并因 GPU0 busy / 当前功耗阈值阻止重复启动；资源守卫判定 resource OK，无停止动作。
- 最近 loss 在约 0.0288-0.0396 区间内波动；日志未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 以当前从 16:05 到 20:09 约 2000 step 的整体速度估算，距离 `ansmask final.pt` 约 8-10 小时；考虑当前验证/保存阶段开销，保守按约 8.5-10.5 小时看。全部目标队列粗估约 30-39 小时。
- 阶段性判断：训练进度、验证进入状态、资源状态和自动队列都符合预期；`best.pt` 在 step 5500 的改进是积极信号，但最终论文判断仍需等 `ansmask final` 与后续 LP 指标完成。

2026-05-30 20:30 复查（按 20 分钟同步节奏，实际检查到 20:31）：
- `VIVID_ansmask_resume_gpu0` 仍为 Running；GPU0 上的 VIVID compute app PID 仍为 21840，GPU1 空闲。
- global 6000 附近的验证已完成，训练已继续推进。`ansmask` 从 global step 4000 恢复后推进到约 2120 / 6000 step，即全局约 6120 / 10000 step。
- 新 checkpoint 已落盘：`outputs/ablation_ums_ansmask_12label/checkpoints/step_6000.pt`，时间为 2026-05-30 20:16:17，大小约 1.07GB。已用 `torch.load(..., map_location='cpu')` 轻量核验元数据，内部 `global_step=6000`、`best_val_loss=0.03747265625`，可作为新的有效恢复点。
- `best.pt` 未在 global 6000 再次更新，当前 best 仍来自 global step 5500；这说明 step 6000 验证没有刷新最佳值，但训练继续推进，属于可接受状态。
- GPU 快照：20:31 左右 GPU0 约 24.3GB 显存、瞬时 util 约 12%、功耗约 125W、温度约 55C；GPU1 为 0MiB、约 8W、32C。20:30 资源守卫记录总功耗约 264.25W、最高温 62C，低于 350-400W 边界，也低于 83C 温度硬阈值。
- 队列在 20:30 正确识别所有目标产物仍未完成，并因 GPU0 busy / 当前功耗阈值阻止重复启动；资源守卫判定 resource OK，无停止动作。
- 最近 loss 在约 0.0310-0.0379 区间内波动；日志未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。`step_6000.pt` 是中间恢复点，不计入目标产物完成数。
- 以当前整体速度估算，距离 `ansmask final.pt` 约 8-9.5 小时；全部目标队列粗估约 29-38 小时。
- 阶段性判断：训练、验证完成、checkpoint 保存、资源状态和自动队列都符合预期；最终论文判断仍需等 `ansmask final` 与后续 LP 指标完成。

2026-05-30 20:50 复查（按 20 分钟同步节奏，实际检查到 20:54）：
- `VIVID_ansmask_resume_gpu0` 仍为 Running；GPU0 上的 VIVID compute app PID 仍为 21840，GPU1 空闲。
- `ansmask` 从 global step 4000 恢复后推进到约 2226 / 6000 step，即全局约 6226 / 10000 step；已稳定越过 global 6000 保存点并继续训练。
- checkpoint 状态未出现异常变化：最新中间恢复点仍为 `step_6000.pt`，当前 best 仍为 global step 5500 的 `best.pt`；`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动。
- GPU 快照：20:54 左右 GPU0 约 24.3GB 显存、瞬时 util 约 52%、功耗约 134W、温度约 54C；GPU1 为 0MiB、约 8W、31C。20:50 资源守卫记录总功耗约 248.8W、最高温 59C，低于 350-400W 边界，也低于 83C 温度硬阈值。
- 队列在 20:50 正确识别所有目标产物仍未完成，并因 GPU0 busy / 当前功耗阈值阻止重复启动；资源守卫判定 resource OK，无停止动作。
- 最近 loss 在约 0.0325-0.0390 区间内波动；日志未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 最近 20 分钟速度比 20:30 前后慢一些；按整体速度估算，距离 `ansmask final.pt` 约 8.5-10 小时，按最近窗口保守估算约 10-13 小时。当前采用保守 ETA：`ansmask final.pt` 约 9-12 小时；全部目标队列粗估约 30-40 小时。
- 阶段性判断：训练、资源状态和自动队列都符合预期；当前只是速度波动，没有失败迹象。最终论文判断仍需等 `ansmask final` 与后续 LP 指标完成。

2026-05-30 21:10 复查（按 20 分钟同步节奏，实际检查到 21:15）：
- `VIVID_ansmask_resume_gpu0` 仍为 Running；GPU0 上的 VIVID compute app PID 仍为 21840，GPU1 空闲。
- `ansmask` 从 global step 4000 恢复后推进到约 2336 / 6000 step，即全局约 6336 / 10000 step；继续向 global 7000 保存点推进。
- checkpoint 状态未出现异常变化：最新中间恢复点仍为 `step_6000.pt`，当前 best 仍为 global step 5500 的 `best.pt`；`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动。
- GPU 快照：21:15 左右 GPU0 约 24.3GB 显存、瞬时 util 约 21%、功耗约 114W、温度约 54C；GPU1 为 0MiB、约 7W、31C。21:15 资源守卫记录总功耗约 120.86W、最高温 50C，低于 350-400W 边界，也低于 83C 温度硬阈值。
- 队列在 21:15 正确识别所有目标产物仍未完成，并因 GPU0 busy / 当前功耗阈值阻止重复启动；资源守卫判定 resource OK，无停止动作。
- 最近 loss 在约 0.0297-0.0387 区间内波动；日志未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 最近窗口速度仍偏慢，但状态稳定；当前保守估计距离 `ansmask final.pt` 约 9-12 小时，全部目标队列粗估约 30-40 小时。
- 阶段性判断：训练、资源状态和自动队列都符合预期；没有失败迹象。最终论文判断仍需等 `ansmask final` 与后续 LP 指标完成。

2026-05-30 21:35 复查（按 20 分钟同步节奏，实际检查到 21:36）：
- `VIVID_ansmask_resume_gpu0` 仍为 Running；GPU0 上的 VIVID compute app PID 仍为 21840，GPU1 空闲。
- `ansmask` 从 global step 4000 恢复后推进到约 2444 / 6000 step，即全局约 6444 / 10000 step；继续向 global 7000 保存点推进。
- checkpoint 状态未出现异常变化：最新中间恢复点仍为 `step_6000.pt`，当前 best 仍为 global step 5500 的 `best.pt`；`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动。
- GPU 快照：21:36 左右 GPU0 约 24.3GB 显存、瞬时 util 0%、功耗约 148W、温度约 52C；GPU1 为 0MiB、约 7W、31C。21:35 资源守卫记录总功耗约 246.44W、最高温 59C，低于 350-400W 边界，也低于 83C 温度硬阈值。
- 队列在 21:35 正确识别所有目标产物仍未完成，并因 GPU0 busy / 当前功耗阈值阻止重复启动；资源守卫判定 resource OK，无停止动作。
- 最近 loss 在约 0.0329-0.0385 区间内波动；日志未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 最近窗口速度继续偏慢但持续推进；当前保守估计距离 `ansmask final.pt` 约 9-12 小时，全部目标队列粗估约 30-40 小时。
- 阶段性判断：训练、资源状态和自动队列都符合预期；没有失败迹象。最终论文判断仍需等 `ansmask final` 与后续 LP 指标完成。

2026-05-30 21:55 复查（按 20 分钟同步节奏，实际检查到 21:58）：
- `VIVID_ansmask_resume_gpu0` 仍为 Running；GPU0 上的 VIVID compute app PID 仍为 21840，GPU1 空闲。
- 日志开头确认本轮从 `step_4000.pt` 恢复：`Resuming from step 4000`。当前训练进度条推进到约 2532 / 6000 step，按恢复点折算全局约 6532 / 10000 step；已越过 global 6000 保存点，继续向 global 7000 保存点推进。
- 日志现在可直接检索到 `Step 6000: val_loss = 0.0409` 和 `Checkpoint saved: .../step_6000.pt`。这次验证没有刷新 `best.pt`，当前 best 仍为 global step 5500 的 `best_val_loss=0.03747265625`。
- checkpoint 状态未出现异常变化：最新中间恢复点仍为 `step_6000.pt`，`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动。
- GPU 快照：21:58 左右 GPU0 约 24.3GB 显存、瞬时 util 0%、功耗约 133W、温度约 54C；GPU1 为 0MiB、约 8W、31C。21:55 资源守卫记录总功耗约 167.95W、最高温 61C，低于 350-400W 边界，也低于 83C 温度硬阈值。
- 队列在 21:55 正确识别所有目标产物仍未完成，并因 GPU0 busy / 当前功耗阈值阻止重复启动；资源守卫判定 resource OK，无停止动作。
- 最近 loss 在约 0.0304-0.0383 区间内波动；日志未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 最近窗口速度仍偏慢；按全程平均速度估算，距离 `ansmask final.pt` 约 8-9 小时，按最近窗口保守估算约 12-14 小时。当前采用保守 ETA：`ansmask final.pt` 约 10-13 小时；全部目标队列粗估约 31-42 小时。
- 阶段性判断：训练、资源状态和自动队列都符合预期；没有失败迹象。最终论文判断仍需等 `ansmask final` 与后续 LP 指标完成。

2026-05-30 22:15 复查（按 20 分钟同步节奏，实际检查到 22:20）：
- `VIVID_ansmask_resume_gpu0` 仍为 Running；GPU0 上的 VIVID compute app PID 仍为 21840，GPU1 空闲。
- 当前训练进度条推进到约 2654 / 6000 step，按 `step_4000.pt` 恢复点折算全局约 6654 / 10000 step；距离 global 7000 保存点约 346 step。
- checkpoint 状态未出现异常变化：最新中间恢复点仍为 `step_6000.pt`，当前 best 仍为 global step 5500 的 `best.pt`；`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动。
- GPU 快照：22:20 左右 GPU0 约 24.3GB 显存、瞬时 util 约 36%、功耗约 80W、温度约 53C；GPU1 为 0MiB、约 8W、32C。22:15 资源守卫记录总功耗约 149.14W、最高温 57C，低于 350-400W 边界，也低于 83C 温度硬阈值。
- 队列在 22:15 正确识别所有目标产物仍未完成，并因 GPU0 busy / 当前功耗阈值阻止重复启动；资源守卫判定 resource OK，无停止动作。
- 最近 loss 在约 0.0309-0.0382 区间内波动；日志未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 按最近窗口速度估算，距离 `step_7000.pt` 约 1 小时左右；距离 `ansmask final.pt` 保守估计约 8.5-11 小时。全部目标队列粗估约 30-40 小时。
- 阶段性判断：训练、资源状态和自动队列都符合预期；没有失败迹象。最终论文判断仍需等 `ansmask final` 与后续 LP 指标完成。

2026-05-30 22:40 复查（按 20 分钟同步节奏，实际检查到 22:42）：
- `VIVID_ansmask_resume_gpu0` 仍为 Running；GPU0 上的 VIVID compute app PID 仍为 21840，GPU1 空闲。
- 当前训练进度条推进到约 2761 / 6000 step，按 `step_4000.pt` 恢复点折算全局约 6761 / 10000 step；距离 global 7000 保存点约 239 step。
- checkpoint 状态未出现异常变化：最新中间恢复点仍为 `step_6000.pt`，当前 best 仍为 global step 5500 的 `best.pt`；`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动。
- GPU 快照：22:42 左右 GPU0 约 24.3GB 显存、瞬时 util 0%、功耗约 75W、温度约 48C；GPU1 为 0MiB、约 7W、32C。22:40 资源守卫记录总功耗约 262.74W、最高温 62C，低于 350-400W 边界，也低于 83C 温度硬阈值。
- 队列在 22:40 正确识别所有目标产物仍未完成，并因 GPU0 busy / 当前功耗阈值阻止重复启动；资源守卫判定 resource OK，无停止动作。
- 最近 loss 在约 0.0304-0.0391 区间内波动；日志未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 按最近窗口速度估算，距离 `step_7000.pt` 约 45-60 分钟；距离 `ansmask final.pt` 保守估计约 8.5-11.5 小时。全部目标队列粗估约 30-40 小时。
- 阶段性判断：训练、资源状态和自动队列都符合预期；没有失败迹象。最终论文判断仍需等 `ansmask final` 与后续 LP 指标完成。

2026-05-30 23:00 复查（按 20 分钟同步节奏，实际检查到 23:03）：
- `VIVID_ansmask_resume_gpu0` 仍为 Running；GPU0 上的 VIVID compute app PID 仍为 21840，GPU1 空闲。
- 当前训练进度条推进到约 2863 / 6000 step，按 `step_4000.pt` 恢复点折算全局约 6863 / 10000 step；距离 global 7000 保存点约 137 step。
- checkpoint 状态未出现异常变化：最新中间恢复点仍为 `step_6000.pt`，当前 best 仍为 global step 5500 的 `best.pt`；`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动。下一轮有机会进入 global 7000 验证/保存阶段。
- GPU 快照：23:03 左右 GPU0 约 24.3GB 显存、瞬时 util 0%、功耗约 138W、温度约 51C；GPU1 为 0MiB、约 8W、32C。23:00 资源守卫记录总功耗约 121.79W、最高温 54C，低于 350-400W 边界，也低于 83C 温度硬阈值。
- 队列在 23:00 正确识别所有目标产物仍未完成，并因 GPU0 busy / 当前功耗阈值阻止重复启动；资源守卫判定 resource OK，无停止动作。
- 最近 loss 在约 0.0304-0.0392 区间内波动；日志未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 按最近窗口速度估算，距离 `step_7000.pt` 约 25-40 分钟；距离 `ansmask final.pt` 保守估计约 8.5-11.5 小时。全部目标队列粗估约 30-40 小时。
- 阶段性判断：训练、资源状态和自动队列都符合预期；没有失败迹象。最终论文判断仍需等 `ansmask final` 与后续 LP 指标完成。

2026-05-30 23:20 复查（按 20 分钟同步节奏，实际检查到 23:24）：
- `VIVID_ansmask_resume_gpu0` 仍为 Running；GPU0 上的 VIVID compute app PID 仍为 21840，GPU1 空闲。
- 当前训练进度条推进到约 2961 / 6000 step，按 `step_4000.pt` 恢复点折算全局约 6961 / 10000 step；距离 global 7000 保存点约 39 step。
- checkpoint 状态未出现异常变化：最新中间恢复点仍为 `step_6000.pt`，当前 best 仍为 global step 5500 的 `best.pt`；`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动。预计下一轮会重点确认 global 7000 验证/保存是否完成。
- GPU 快照：23:24 左右 GPU0 约 24.3GB 显存、瞬时 util 99%、功耗约 75W、温度约 50C；GPU1 为 0MiB、约 7W、33C。23:20 资源守卫记录总功耗约 175.74W、最高温 56C，低于 350-400W 边界，也低于 83C 温度硬阈值。
- 队列在 23:20 正确识别所有目标产物仍未完成，并因 GPU0 busy / 当前功耗阈值阻止重复启动；资源守卫判定 resource OK，无停止动作。
- 最近 loss 在约 0.0326-0.0390 区间内波动；日志未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 按最近窗口速度估算，距离 `step_7000.pt` 约 5-15 分钟；距离 `ansmask final.pt` 保守估计约 8.5-11 小时。全部目标队列粗估约 30-40 小时。
- 阶段性判断：训练、资源状态和自动队列都符合预期；没有失败迹象。最终论文判断仍需等 `ansmask final` 与后续 LP 指标完成。

2026-05-30 23:45 复查（按 20 分钟同步节奏，实际检查到 23:47）：
- `VIVID_ansmask_resume_gpu0` 仍为 Running；GPU0 上的 VIVID compute app PID 仍为 21840，GPU1 继续空闲。
- global 7000 验证/保存已经完成，训练已经恢复并推进到进度条约 3012 / 6000 step；按 `step_4000.pt` 恢复点折算，全局约 7012 / 10000 step。
- 新中间恢复点已落盘：`outputs/ablation_ums_ansmask_12label/checkpoints/step_7000.pt`，写入时间 2026-05-30 23:43:45，大小约 1.07GB。已用 `torch.load(..., map_location='cpu')` 轻量核验内部元数据，`global_step=7000`、`best_val_loss=0.03747265625`。
- `best.pt` 未在 global 7000 刷新，当前 best 仍来自 global step 5500；这说明本轮验证没有刷新最佳值，但中间 checkpoint 正常、训练继续推进，属于可接受状态。`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动。
- GPU 快照：23:47 左右 GPU0 约 24.3GB 显存、瞬时 util 47%、功耗约 99W、温度约 49C；GPU1 为 0MiB、约 7W、33C。23:45 资源守卫记录总功耗约 150.61W、最高温 54C，低于 350-400W 边界，也低于 83C 温度硬阈值。
- 队列在 23:45 正确识别所有目标产物仍未完成，并因 GPU0 busy / 当前功耗阈值阻止重复启动；资源守卫判定 resource OK，无停止动作。
- 最近 loss 在约 0.0341-0.0363 区间内波动；日志未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。`step_7000.pt` 是中间恢复点，不计入目标产物完成数。
- 按当前推进速度与后续验证开销保守估计，距离 `ansmask final.pt` 约 10-13 小时；全部目标队列粗估约 31-43 小时。
- 阶段性判断：global 7000 checkpoint、训练续跑、资源状态和自动队列都符合预期；没有失败迹象。最终论文判断仍需等 `ansmask final` 与后续 LP 指标完成。

2026-05-31 00:10 复查（按 20 分钟同步节奏）：
- `VIVID_ansmask_resume_gpu0` 仍为 Running；GPU0 上的 VIVID compute app PID 仍为 21840，GPU1 继续空闲。
- `ansmask` 在 global 7000 之后继续稳定推进，当前训练进度条约 3084 / 6000 step；按 `step_4000.pt` 恢复点折算，全局约 7084 / 10000 step。
- checkpoint 状态正常：最新中间恢复点仍为 `step_7000.pt`，`best.pt` 仍为 global step 5500 的最佳点；`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动。
- GPU 快照：00:10 左右 GPU0 约 24.3GB 显存、瞬时 util 100%、功耗约 84W、温度约 50C；GPU1 为 0MiB、约 8W、33C。00:10 资源守卫记录总功耗约 78.46W、最高温 49C，明显低于 350-400W 边界，也低于 83C 温度硬阈值。
- 队列在 00:10 正确识别 8 项目标产物仍未完成，并因 GPU0 busy / 当前功耗阈值阻止重复启动；资源守卫判定 resource OK，无停止动作。
- 最近 loss 在约 0.0311-0.0384 区间内波动；日志未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 当前窗口速度偏慢，按最近窗口与后续验证开销保守估计，距离 `ansmask final.pt` 约 12-16 小时；全部目标队列粗估约 33-47 小时。
- 阶段性判断：训练、资源状态和自动队列符合预期；目前没有失败迹象，只是速度存在波动。最终论文判断仍需等 `ansmask final` 与后续 LP 指标完成。

2026-05-31 01:15 GPU1 迁移复查与 case study：
- 用户更新资源策略为“以后都用 GPU1”。已将 `scripts/answerability_gpu1_queue_once.ps1` 的目标卡切回 `target_gpu=1`，并把队列中的 ansmask、LP、null-as-negative、counterfactual-prefix、random-LM、field-paraphrase 启动项全部切到 `*_gpu1` 计划任务。
- 已同步更新 `scripts/register_answerability_tasks.ps1`，以后重新注册 answerability 任务时只注册 GPU1 runner；旧 GPU0 runner 计划任务已禁用，避免队列或误触发再次使用 GPU0。
- 当前 `VIVID_ansmask_resume_gpu1` 已通过 Task Scheduler 启动，日志确认从 `outputs/ablation_ums_ansmask_12label/checkpoints/step_7000.pt` 恢复：`Resuming from step 7000`。截至 01:15，训练进度约 12 / 3000 step，折算全局约 7012 / 10000 step。
- GPU 状态：01:15 GPU0 空闲 0MiB；GPU1 约 16.8GB 显存、util 约 59%、功耗约 82-105W、温度约 58-60C。资源守卫 01:15 记录总功耗约 114.71W、最高温 60C，低于 350-400W 边界和 83C 温度硬阈值。
- 队列状态：01:10 与 01:15 已正确识别 `target_gpu=1`，且因 GPU1 busy 阻止重复启动 `VIVID_ansmask_resume_gpu1`。这说明后续自动队列已经按 GPU1 策略工作。
- 项目结构整理：旧的 GPU1 ansmask 日志已归档到 `history/gpu1_migration_20260531/ablation_ums_ansmask_12label_resume_from_best_gpu1_before_20260531_010603.log`，避免新 GPU1 续跑日志覆盖历史证据。
- case study：00:45 左右 GPU0 查询失败，队列与资源守卫均记录 `GPU is lost. Reboot the system to recover this GPU`；00:50 GPU0 状态恢复后，旧队列策略又启动了一次 `VIVID_ansmask_resume_gpu0`。该 GPU0 续跑只推进了约 57 个局部 step，日志末尾出现 `^C`，没有形成新的 checkpoint。因此影响范围是：global 7000 之后的一小段未保存训练进度丢失，但 `step_7000.pt` 已在此前完成并经元数据核验，可作为可靠恢复点；模型结果主线不受破坏。
- case study 处置：已禁用 GPU0 runner，把队列和未来注册入口改为 GPU1，当前从 `step_7000.pt` 在 GPU1 重启，避免继续依赖出现过 `GPU is lost` 的 GPU0。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 按 GPU1 当前早期速度与后续验证开销保守估计，距离 `ansmask final.pt` 约 12-14 小时；全部目标队列粗估约 33-48 小时。
- 阶段性判断：当前迁移成功，训练与资源状态符合预期；GPU0 异常已记录并通过 GPU1 策略规避。最终论文结果判断仍需等 `ansmask final` 与后续 LP 指标完成。

2026-05-31 01:35 复查（GPU1-only 策略）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上的 VIVID compute app PID 为 8468，GPU0 空闲且 `VIVID_ansmask_resume_gpu0` 计划任务保持 Disabled。
- 当前训练进度约 77 / 3000 step，按 `step_7000.pt` 恢复点折算全局约 7077 / 10000 step；距离下一个 global 8000 验证/保存点约 923 step。
- checkpoint 状态未出现异常变化：最新中间恢复点仍为 `step_7000.pt`，`best.pt` 仍为 global step 5500 的最佳点；`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动。
- GPU 快照：01:36 左右 GPU1 约 16.8GB 显存、util 约 62%、功耗约 110W、温度约 59C；GPU0 为 0MiB、约 9W、38C。01:35 资源守卫记录总功耗约 115.33W、最高温 61C，低于 350-400W 边界和 83C 温度硬阈值。
- 队列在 01:20、01:25、01:30、01:35 均正确识别 `target_gpu=1`，并因 GPU1 busy 阻止重复启动 `VIVID_ansmask_resume_gpu1`。
- 日志可见 `Flash attention not available, falling back to eager`，这是依赖缺失导致的性能回退，不是运行失败；训练仍在推进。最近 loss 在约 0.0334-0.0374 区间内波动，未见 traceback / RuntimeError / CUDA OOM / exitcode。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 按 GPU1 最近窗口速度与后续验证开销保守估计，距离 `ansmask final.pt` 约 15-17 小时；全部目标队列粗估约 36-52 小时。
- 阶段性判断：GPU1-only 迁移后的训练、资源状态和自动队列均符合预期；目前没有新的失败 case。最终论文结果判断仍需等 `ansmask final` 与后续 LP 指标完成。

2026-05-31 01:55 复查（GPU1-only 策略）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上的 VIVID compute app PID 仍为 8468，GPU0 空闲。
- 当前训练进度约 129 / 3000 step，按 `step_7000.pt` 恢复点折算全局约 7129 / 10000 step；距离下一个 global 8000 验证/保存点约 871 step。
- checkpoint 状态未出现异常变化：最新中间恢复点仍为 `step_7000.pt`，`best.pt` 仍为 global step 5500 的最佳点；8 项目标产物均未完成。
- GPU 快照：01:57 左右 GPU1 约 16.8GB 显存、util 约 21%、功耗约 23W、温度约 54C；GPU0 为 0MiB、约 9W、41C。01:55 资源守卫记录总功耗约 128.17W、最高温 58C，低于 350-400W 边界和 83C 温度硬阈值。
- 队列在 01:40、01:45、01:50、01:55 均正确识别 `target_gpu=1`，并因 GPU1 busy 或当前功耗阈值阻止重复启动 `VIVID_ansmask_resume_gpu1`。
- 最近 loss 在约 0.0301-0.0348 区间内波动，未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- GPU1 最近窗口速度偏慢，按当前窗口与后续验证开销保守估计，距离 `ansmask final.pt` 约 18-22 小时；全部目标队列粗估约 39-57 小时。
- 阶段性判断：训练、资源状态和自动队列均符合预期；速度偏慢但没有失败迹象。最终论文结果判断仍需等 `ansmask final` 与后续 LP 指标完成。

2026-05-31 02:15 复查（GPU1-only 策略，实际检查到 02:18）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上的 VIVID compute app PID 仍为 8468，GPU0 空闲。
- 当前训练进度约 182 / 3000 step，按 `step_7000.pt` 恢复点折算全局约 7182 / 10000 step；距离下一个 global 8000 验证/保存点约 818 step。
- checkpoint 状态未出现异常变化：最新中间恢复点仍为 `step_7000.pt`，`best.pt` 仍为 global step 5500 的最佳点；`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动。
- GPU 快照：02:18 左右 GPU1 约 16.8GB 显存、util 约 23%、功耗约 97W、温度约 58C；GPU0 为 0MiB、约 9W、39C。02:15 资源守卫记录总功耗约 127.52W、最高温 59C，低于 350-400W 边界和 83C 温度硬阈值。
- 队列在 02:00、02:05、02:10、02:15 均正确识别 `target_gpu=1`，并因 GPU1 busy 或当前功耗阈值阻止重复启动 `VIVID_ansmask_resume_gpu1`。
- 最近 loss 在约 0.0310-0.0394 区间内波动，未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 按 GPU1 最近窗口速度与后续验证开销保守估计，距离 `ansmask final.pt` 约 18-22 小时；全部目标队列粗估约 39-57 小时。
- 阶段性判断：训练、资源状态和自动队列均符合预期；速度偏慢但稳定，没有失败迹象。最终论文结果判断仍需等 `ansmask final` 与后续 LP 指标完成。

2026-05-31 02:35 复查（GPU1-only 策略，实际检查到 02:39）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上的 VIVID compute app PID 仍为 8468，GPU0 空闲。
- 当前训练进度约 236 / 3000 step，按 `step_7000.pt` 恢复点折算全局约 7236 / 10000 step；距离下一个 global 8000 验证/保存点约 764 step。
- checkpoint 状态未出现异常变化：最新中间恢复点仍为 `step_7000.pt`，`best.pt` 仍为 global step 5500 的最佳点；`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动。
- GPU 快照：02:39 左右 GPU1 约 16.8GB 显存、util 约 71%、功耗约 81W、温度约 57C；GPU0 为 0MiB、约 9W、39C。02:35 资源守卫记录总功耗约 119.52W、最高温 58C；02:25 窗口曾到总功耗约 176.69W、最高温 63C，仍明显低于 350-400W 边界和 83C 温度硬阈值。
- 队列在 02:20、02:25、02:30、02:35 均正确识别 `target_gpu=1`，并因 GPU1 busy 或当前功耗阈值阻止重复启动 `VIVID_ansmask_resume_gpu1`。
- 最近 loss 在约 0.0298-0.0376 区间内波动，未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 按 GPU1 最近窗口速度与后续验证开销保守估计，距离 `ansmask final.pt` 约 18-21 小时；全部目标队列粗估约 39-56 小时。
- 阶段性判断：训练、资源状态和自动队列均符合预期；速度偏慢但稳定，没有失败迹象。最终论文结果判断仍需等 `ansmask final` 与后续 LP 指标完成。

2026-05-31 02:55 复查（GPU1-only 策略，实际检查到 02:59）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上的 VIVID compute app PID 仍为 8468，GPU0 空闲。
- 当前训练进度约 306 / 3000 step，按 `step_7000.pt` 恢复点折算全局约 7306 / 10000 step；距离下一个 global 8000 验证/保存点约 694 step。
- checkpoint 状态未出现异常变化：最新中间恢复点仍为 `step_7000.pt`，`best.pt` 仍为 global step 5500 的最佳点；`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动。
- GPU 快照：02:59 左右 GPU1 约 16.8GB 显存、util 约 98%、功耗约 82W、温度约 59C；GPU0 为 0MiB、约 8W、40C。02:55 资源守卫记录总功耗约 145.86W、最高温 62C；02:50 窗口最高温约 64C，仍低于 350-400W 边界和 83C 温度硬阈值。
- 队列在 02:40、02:45、02:50、02:55 均正确识别 `target_gpu=1`，并因 GPU1 busy 或当前功耗阈值阻止重复启动 `VIVID_ansmask_resume_gpu1`。
- 最近 loss 在约 0.0288-0.0395 区间内波动，未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 最近窗口速度有所回升；按 GPU1 近几个窗口与后续验证开销保守估计，距离 `ansmask final.pt` 约 15-19 小时；全部目标队列粗估约 36-53 小时。
- 阶段性判断：训练、资源状态和自动队列均符合预期；没有失败迹象。最终论文结果判断仍需等 `ansmask final` 与后续 LP 指标完成。

2026-05-31 03:20 复查（GPU1-only 策略）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上的 VIVID compute app PID 仍为 8468，GPU0 空闲。
- 当前训练进度约 370 / 3000 step，按 `step_7000.pt` 恢复点折算全局约 7370 / 10000 step；距离下一个 global 8000 验证/保存点约 630 step。
- checkpoint 状态未出现异常变化：最新中间恢复点仍为 `step_7000.pt`，`best.pt` 仍为 global step 5500 的最佳点；`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动。
- GPU 快照：03:20 左右 GPU1 约 18.3GB 显存、util 约 41-71%、功耗约 19-88W、温度约 52-58C；GPU0 为 0MiB、约 9W、40C。03:20 资源守卫记录总功耗约 96.66W、最高温 58C，明显低于 350-400W 边界和 83C 温度硬阈值。
- 队列在 03:00、03:05、03:10、03:15、03:20 均正确识别 `target_gpu=1`，并因 GPU1 busy 或当前功耗阈值阻止重复启动 `VIVID_ansmask_resume_gpu1`。
- 最近 loss 在约 0.0330-0.0385 区间内波动，未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 最近窗口速度继续回升；按 GPU1 近几个窗口与后续验证开销保守估计，距离 `ansmask final.pt` 约 15-18 小时；全部目标队列粗估约 36-52 小时。
- 阶段性判断：训练、资源状态和自动队列均符合预期；没有失败迹象。最终论文结果判断仍需等 `ansmask final` 与后续 LP 指标完成。

2026-05-31 03:40 复查（GPU1-only 策略，实际检查到 03:41）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上的 VIVID compute app PID 仍为 8468，GPU0 空闲。
- 当前训练进度约 435 / 3000 step，按 `step_7000.pt` 恢复点折算全局约 7435 / 10000 step；距离下一个 global 8000 验证/保存点约 565 step。
- checkpoint 状态未出现异常变化：最新中间恢复点仍为 `step_7000.pt`，`best.pt` 仍为 global step 5500 的最佳点；`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动。
- GPU 快照：03:41 左右 GPU1 约 18.3GB 显存、util 约 33%、功耗约 108W、温度约 58C；GPU0 为 0MiB、约 8W、40C。03:40 资源守卫记录总功耗约 136.23W、最高温 60C；03:35 最高温约 62C，仍低于 350-400W 边界和 83C 温度硬阈值。
- 队列在 03:25、03:30、03:35、03:40 均正确识别 `target_gpu=1`，并因 GPU1 busy 或当前功耗阈值阻止重复启动 `VIVID_ansmask_resume_gpu1`。
- 最近 loss 在约 0.0328-0.0382 区间内波动，未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 按 GPU1 近几个窗口与后续验证开销保守估计，距离 `ansmask final.pt` 约 15-18 小时；全部目标队列粗估约 36-52 小时。
- 阶段性判断：训练、资源状态和自动队列均符合预期；没有失败迹象。最终论文结果判断仍需等 `ansmask final` 与后续 LP 指标完成。

2026-05-31 04:00 复查（GPU1-only 策略，实际检查到 04:01）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上的 VIVID compute app PID 仍为 8468，GPU0 空闲。
- 当前训练进度约 497 / 3000 step，按 `step_7000.pt` 恢复点折算全局约 7497 / 10000 step；距离下一个 global 8000 验证/保存点约 503 step。
- checkpoint 状态未出现异常变化：最新中间恢复点仍为 `step_7000.pt`，`best.pt` 仍为 global step 5500 的最佳点；`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动。
- GPU 快照：04:01 左右 GPU1 约 18.3GB 显存、util 约 53%、功耗约 75W、温度约 56C；GPU0 为 0MiB、约 8W、39C。03:55 资源守卫记录总功耗约 191.78W、最高温 64C，04:00 记录总功耗约 169.85W、最高温 63C，均低于 350-400W 边界和 83C 温度硬阈值。
- 队列在 03:45、03:50、03:55、04:00 均正确识别 `target_gpu=1`，并因 GPU1 busy 或当前功耗阈值阻止重复启动 `VIVID_ansmask_resume_gpu1`。
- 最近 loss 在约 0.0347-0.0397 区间内波动，未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 按 GPU1 近几个窗口与后续验证开销保守估计，距离 `ansmask final.pt` 约 15-17 小时；全部目标队列粗估约 36-51 小时。
- 阶段性判断：训练、资源状态和自动队列均符合预期；没有失败迹象。最终论文结果判断仍需等 `ansmask final` 与后续 LP 指标完成。

2026-05-31 04:20 复查（GPU1-only 策略，实际检查到 04:24）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上的 VIVID compute app PID 仍为 8468，GPU0 空闲。
- 当前训练已完成本地 step 500 的验证阶段并继续训练，最新进度约 542 / 3000 step；按 `step_7000.pt` 恢复点折算全局约 7542 / 10000 step；距离下一个 global 8000 验证/保存点约 458 step。
- checkpoint 状态仍正常但未出现新目标产物：最新中间恢复点仍为 `step_7000.pt`，`best.pt` 仍为 global step 5500 的最佳点；本地 step 500 验证后未观察到新的命名 checkpoint 或 best 刷新，`final.pt` 仍未生成，因此 `lp_ums_ansmask` 尚未启动。
- GPU 快照：04:23 左右 GPU1 约 18.3GB 显存、util 约 99%、功耗约 83W、温度约 56C；GPU0 为 0MiB、约 8W、39C。04:20 资源守卫记录总功耗约 128.9W、最高温 59C，低于 350-400W 边界和 83C 温度硬阈值。
- 队列在 04:05、04:10、04:15、04:20 均正确识别 `target_gpu=1`，并因 GPU1 busy 或当前功耗阈值阻止重复启动 `VIVID_ansmask_resume_gpu1`。
- 最近 loss 在约 0.0358-0.0396 区间内波动，未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 由于本窗口包含一次完整验证，短窗速度会被拉低；按 GPU1 近几个窗口与后续验证开销保守估计，距离 `ansmask final.pt` 约 15-18 小时；全部目标队列粗估约 36-52 小时。
- 阶段性判断：训练、验证完成后的续跑、资源状态和自动队列均符合预期；没有失败迹象。最终论文结果判断仍需等 `ansmask final` 与后续 LP 指标完成。

2026-05-31 04:40 复查（GPU1-only 策略，实际检查到 04:44）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上的 VIVID compute app PID 仍为 8468，GPU0 空闲。
- 当前训练进度约 615 / 3000 step，按 `step_7000.pt` 恢复点折算全局约 7615 / 10000 step；距离下一个 global 8000 验证/保存点约 385 step。
- checkpoint 状态未出现异常变化：最新中间恢复点仍为 `step_7000.pt`，`best.pt` 仍为 global step 5500 的最佳点；`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动，8 项目标产物仍未完成。
- GPU 快照：04:44 左右 GPU1 约 18.3GB 显存、util 约 46%、功耗约 34W、温度约 56C；GPU0 为 0MiB、约 9W、39C。04:25-04:40 资源守卫记录总功耗约 109.81W、112.65W、117.69W、157.18W，最高温约 57-63C，均低于 350-400W 边界和 83C 温度硬阈值。
- 队列在 04:25、04:30、04:35、04:40 均正确识别 `target_gpu=1`，并因 GPU1 busy 或当前功耗阈值阻止重复启动 `VIVID_ansmask_resume_gpu1`。
- 最近 loss 在约 0.0323-0.0399 区间内波动，未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 本窗口训练速度回升；按 GPU1 近几个窗口与后续验证开销保守估计，距离 `ansmask final.pt` 约 13-16 小时；全部目标队列粗估约 34-49 小时。
- 阶段性判断：训练、资源状态和自动队列均符合预期；没有失败迹象。最终论文结果判断仍需等 `ansmask final` 与后续 LP 指标完成。

2026-05-31 05:00 复查（GPU1-only 策略，实际检查到 05:05）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上的 VIVID compute app PID 仍为 8468，GPU0 空闲。
- 当前训练进度约 702 / 3000 step，按 `step_7000.pt` 恢复点折算全局约 7702 / 10000 step；距离下一个 global 8000 验证/保存点约 298 step。
- checkpoint 状态未出现异常变化：最新中间恢复点仍为 `step_7000.pt`，`best.pt` 仍为 global step 5500 的最佳点；`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动，8 项目标产物仍未完成。
- GPU 快照：05:05 左右 GPU1 约 18.3GB 显存、util 约 89%、功耗约 22W、温度约 54C；GPU0 为 0MiB、约 8W、40C。04:45-05:05 资源守卫记录总功耗约 127.95W、110.88W、103.76W、154.55W、105.69W，最高温约 57-68C，均低于 350-400W 边界和 83C 温度硬阈值。
- 队列在 04:45、04:50、04:55、05:00、05:05 均正确识别 `target_gpu=1`，并因 GPU1 busy 或当前功耗阈值阻止重复启动 `VIVID_ansmask_resume_gpu1`。
- 最近 loss 在约 0.0294-0.0378 区间内波动，未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 最近窗口训练速度继续较好；按 GPU1 近几个窗口与后续验证开销保守估计，距离 `ansmask final.pt` 约 11-14 小时；全部目标队列粗估约 32-47 小时。
- 阶段性判断：训练、资源状态和自动队列均符合预期；没有失败迹象。最终论文结果判断仍需等 `ansmask final` 与后续 LP 指标完成。

2026-05-31 05:20 复查（GPU1-only 策略，实际检查到 05:26）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上的 VIVID compute app PID 仍为 8468，GPU0 空闲。
- 当前训练进度约 790 / 3000 step，按 `step_7000.pt` 恢复点折算全局约 7790 / 10000 step；距离下一个 global 8000 验证/保存点约 210 step。
- checkpoint 状态未出现异常变化：最新中间恢复点仍为 `step_7000.pt`，`best.pt` 仍为 global step 5500 的最佳点；`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动，8 项目标产物仍未完成。
- GPU 快照：05:26 左右 GPU1 约 18.3GB 显存、util 约 20%、功耗约 81W、温度约 59C；GPU0 为 0MiB、约 9W、38C。05:10-05:25 资源守卫记录总功耗约 83.84W、185.58W、125.12W、165.29W，最高温约 56-66C，均低于 350-400W 边界和 83C 温度硬阈值。
- 队列在 05:10、05:15、05:20、05:25 均正确识别 `target_gpu=1`，并因 GPU1 busy 或当前功耗阈值阻止重复启动 `VIVID_ansmask_resume_gpu1`。
- 最近 loss 在约 0.0343-0.0412 区间内波动，未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 按 GPU1 近几个窗口与后续验证开销保守估计，距离 `ansmask final.pt` 约 10-13 小时；全部目标队列粗估约 31-45 小时。下一关键观察点是 global 8000 的验证与 `step_8000.pt` 保存。
- 阶段性判断：训练、资源状态和自动队列均符合预期；没有失败迹象。最终论文结果判断仍需等 `ansmask final` 与后续 LP 指标完成。

2026-05-31 05:40 复查（GPU1-only 策略，实际检查到 05:46）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上的 VIVID compute app PID 仍为 8468，GPU0 空闲。
- 当前训练进度约 896 / 3000 step，按 `step_7000.pt` 恢复点折算全局约 7896 / 10000 step；距离下一个 global 8000 验证/保存点约 104 step。
- checkpoint 状态未出现异常变化：最新中间恢复点仍为 `step_7000.pt`，`best.pt` 仍为 global step 5500 的最佳点；`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动，8 项目标产物仍未完成。
- GPU 快照：05:46 左右 GPU1 约 18.3GB 显存、util 约 66%、功耗约 108W、温度约 59C；GPU0 为 0MiB、约 9W、40C。05:30-05:45 资源守卫记录总功耗约 116.48W、215.5W、205.59W、202.41W，最高温约 59-67C，均低于 350-400W 边界和 83C 温度硬阈值。
- 队列在 05:30、05:35、05:40、05:45 均正确识别 `target_gpu=1`，并因 GPU1 busy 或当前功耗阈值阻止重复启动 `VIVID_ansmask_resume_gpu1`。
- 最近 loss 在约 0.0325-0.0421 区间内波动，未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 按 GPU1 近几个窗口与后续验证开销保守估计，距离 global 8000 验证/保存约 20-40 分钟；距离 `ansmask final.pt` 约 9-12 小时；全部目标队列粗估约 30-44 小时。
- 阶段性判断：训练、资源状态和自动队列均符合预期；没有失败迹象。下一轮重点确认是否进入 global 8000 验证与 `step_8000.pt` 保存。

2026-05-31 06:00 复查（GPU1-only 策略，实际检查到 06:07）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上的 VIVID compute app PID 仍为 8468，GPU0 空闲。
- 当前训练进度约 993 / 3000 step，按 `step_7000.pt` 恢复点折算全局约 7993 / 10000 step；距离 global 8000 验证/保存点约 7 step。
- checkpoint 状态暂未变化：最新中间恢复点仍为 `step_7000.pt`，`best.pt` 仍为 global step 5500 的最佳点；截至本次检查，尚未看到 `step_8000.pt`、新的 best 或 `final.pt`，因此 `lp_ums_ansmask` 尚未启动，8 项目标产物仍未完成。
- GPU 快照：06:07 左右 GPU1 约 18.3GB 显存、瞬时 util 约 0%、功耗约 20W、温度约 55C；GPU0 为 0MiB、约 9W、40C。05:50-06:05 资源守卫记录总功耗约 143.77W、105.81W、120.58W、147.66W，最高温约 56-65C，均低于 350-400W 边界和 83C 温度硬阈值。
- 队列在 05:50、05:55、06:00、06:05 均正确识别 `target_gpu=1`，并因 GPU1 busy 或当前功耗阈值阻止重复启动 `VIVID_ansmask_resume_gpu1`。
- 最近 loss 在约 0.0336-0.0383 区间内波动，未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- global 8000 只差数个训练 step，预计会在下一小段运行后进入验证/保存；考虑验证开销，`step_8000.pt` 预计在数分钟到下一轮同步内出现。距离 `ansmask final.pt` 仍约 9-12 小时；全部目标队列粗估约 30-44 小时。
- 阶段性判断：训练、资源状态和自动队列均符合预期；没有失败迹象。下一步重点核验 `step_8000.pt` 元数据。

2026-05-31 06:18 短查（global 8000 保存点核验）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上的 VIVID compute app PID 仍为 8468，GPU0 空闲。
- global 8000 验证/保存已经完成，训练已继续到约 1023 / 3000 step；按 `step_7000.pt` 恢复点折算全局约 8023 / 10000 step。
- 新中间恢复点已落盘：`outputs/ablation_ums_ansmask_12label/checkpoints/step_8000.pt`，写入时间 2026-05-31 06:14:27，大小约 1.07GB。已用 `torch.load(..., map_location='cpu')` 轻量核验内部元数据，`global_step=8000`、`best_val_loss=0.03747265625`。
- `best.pt` 未在本轮刷新，当前 best 仍来自 global step 5500；日志中最近可见 `Step 7500: val_loss = 0.0403`，高于当前 best，符合 best 未刷新的现象。`step_8000.pt` 是中间恢复点，不计入目标产物完成数；`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动。
- GPU 快照：06:18 左右 GPU1 约 18.3GB 显存、util 约 43%、功耗约 84W、温度约 59C；GPU0 为 0MiB、约 9W、39C。06:10 与 06:15 资源守卫记录总功耗约 19.08W、94.52W，最高温约 48-59C，均低于 350-400W 边界和 83C 温度硬阈值。
- 队列在 06:10、06:15 均正确识别 `target_gpu=1`，并因 GPU1 busy 阻止重复启动 `VIVID_ansmask_resume_gpu1`。
- 验证结束后训练已恢复；最近 loss 在约 0.0301-0.0393 区间内波动，未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 距离 `ansmask final.pt` 仍约 9-12 小时；全部目标队列粗估约 30-44 小时。下一关键观察点是 global 9000 / local 2000 附近的验证/保存，以及最终 global 10000 的 `final.pt`。
- 阶段性判断：global 8000 checkpoint、验证后续跑、资源状态和自动队列均符合预期；目前没有失败迹象。最终论文结果判断仍需等 `ansmask final` 与后续 LP 指标完成。

2026-05-31 06:40 复查（GPU1-only 策略）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上的 VIVID compute app PID 仍为 8468，GPU0 空闲。
- 当前训练进度约 1121 / 3000 step，按 `step_7000.pt` 恢复点折算全局约 8121 / 10000 step；距离下一个 global 9000 验证/保存点约 879 step。
- checkpoint 状态正常：最新中间恢复点为已核验的 `step_8000.pt`，`best.pt` 仍为 global step 5500 的最佳点；`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动，8 项目标产物仍未完成。
- GPU 快照：06:40 左右 GPU1 约 18.3GB 显存、瞬时 util 约 0%、功耗约 116W、温度约 59C；GPU0 为 0MiB、约 9W、42C。06:20-06:40 资源守卫记录总功耗约 105.36W、115.63W、141.38W、118.73W，最高温约 57-66C，均低于 350-400W 边界和 83C 温度硬阈值。
- 队列在 06:20、06:25、06:30、06:35、06:40 均正确识别 `target_gpu=1`，并因 GPU1 busy 或当前功耗阈值阻止重复启动 `VIVID_ansmask_resume_gpu1`。
- 最近 loss 在约 0.0333-0.0369 区间内波动，未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 按 GPU1 近几个窗口与后续验证开销保守估计，距离 `ansmask final.pt` 约 8-11 小时；全部目标队列粗估约 29-43 小时。
- 阶段性判断：训练、资源状态和自动队列均符合预期；没有失败迹象。下一关键观察点仍是 global 9000 / local 2000 附近的验证/保存。

2026-05-31 07:00 复查（GPU1-only 策略，实际检查到 07:01）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上的 VIVID compute app PID 仍为 8468，GPU0 空闲。
- 当前训练进度约 1196 / 3000 step，按 `step_7000.pt` 恢复点折算全局约 8196 / 10000 step；距离下一个 global 9000 验证/保存点约 804 step。
- checkpoint 状态正常：最新中间恢复点仍为已核验的 `step_8000.pt`，`best.pt` 仍为 global step 5500 的最佳点；`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动，8 项目标产物仍未完成。
- GPU 快照：07:01 左右 GPU1 约 18.3GB 显存、util 约 37%、功耗约 18W、温度约 53C；GPU0 为 0MiB、约 8W、42C。06:45-07:00 资源守卫记录总功耗约 118.04W、127.5W、89.47W、174.83W，最高温约 58-63C，均低于 350-400W 边界和 83C 温度硬阈值。
- 队列在 06:45、06:50、06:55、07:00 均正确识别 `target_gpu=1`，并因 GPU1 busy 或当前功耗阈值阻止重复启动 `VIVID_ansmask_resume_gpu1`。
- 最近 loss 在约 0.0295-0.0404 区间内波动，未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 按 GPU1 近几个窗口与后续验证开销保守估计，距离 `ansmask final.pt` 约 8-10.5 小时；全部目标队列粗估约 29-42 小时。
- 阶段性判断：训练、资源状态和自动队列均符合预期；没有失败迹象。下一关键观察点仍是 global 9000 / local 2000 附近的验证/保存。

2026-05-31 07:20 复查（GPU1-only 策略，实际检查到 07:22）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上的 VIVID compute app PID 仍为 8468，GPU0 空闲。
- 当前训练进度约 1238 / 3000 step，按 `step_7000.pt` 恢复点折算全局约 8238 / 10000 step；距离下一个 global 9000 验证/保存点约 762 step。
- checkpoint 状态正常：最新中间恢复点仍为已核验的 `step_8000.pt`，`best.pt` 仍为 global step 5500 的最佳点；`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动，8 项目标产物仍未完成。
- GPU 快照：07:22 左右 GPU1 约 18.3GB 显存、util 约 53%、功耗约 100W、温度约 58C；GPU0 为 0MiB、约 9W、42C。07:05-07:20 资源守卫记录总功耗约 117.38W、55.57W、90.62W、121.17W，最高温约 57-59C，均低于 350-400W 边界和 83C 温度硬阈值。
- 队列在 07:05、07:10、07:15、07:20 均正确识别 `target_gpu=1`，并因 GPU1 busy 或当前功耗阈值阻止重复启动 `VIVID_ansmask_resume_gpu1`。
- 最近 loss 在约 0.0317-0.0404 区间内波动，未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 本窗口速度偏慢；按 GPU1 近几个窗口与后续验证开销保守估计，距离 `ansmask final.pt` 约 10-14 小时；全部目标队列粗估约 31-46 小时。
- 阶段性判断：训练、资源状态和自动队列均符合预期；速度有波动但没有失败迹象。下一关键观察点仍是 global 9000 / local 2000 附近的验证/保存。

2026-05-31 07:40 复查（GPU1-only 策略，实际检查到 07:43）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上的 VIVID compute app PID 仍为 8468，GPU0 空闲。
- 当前训练进度约 1289 / 3000 step，按 `step_7000.pt` 恢复点折算全局约 8289 / 10000 step；距离下一个 global 9000 验证/保存点约 711 step。
- checkpoint 状态正常：最新中间恢复点仍为已核验的 `step_8000.pt`，`best.pt` 仍为 global step 5500 的最佳点；`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动，8 项目标产物仍未完成。
- GPU 快照：07:43 左右 GPU1 约 18.3GB 显存、util 约 19%、功耗约 12W、温度约 52C；GPU0 为 0MiB、约 9W、42C。07:25-07:40 资源守卫记录总功耗约 119.98W、103.74W、91.11W、184.84W，最高温约 58-64C，均低于 350-400W 边界和 83C 温度硬阈值。
- 队列在 07:25、07:30、07:35、07:40 均正确识别 `target_gpu=1`，并因 GPU1 busy 或当前功耗阈值阻止重复启动 `VIVID_ansmask_resume_gpu1`。
- 最近 loss 在约 0.0320-0.0366 区间内波动，未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 最近两个窗口速度偏慢；按 GPU1 近几个窗口与后续验证开销保守估计，距离 `ansmask final.pt` 约 11-15 小时；全部目标队列粗估约 32-47 小时。
- 阶段性判断：训练、资源状态和自动队列均符合预期；速度偏慢但没有失败迹象。下一关键观察点仍是 global 9000 / local 2000 附近的验证/保存。

2026-05-31 08:00 复查（GPU1-only 策略，实际检查到 08:04）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上的 VIVID compute app PID 仍为 8468，GPU0 空闲。
- 当前训练进度约 1355 / 3000 step，按 `step_7000.pt` 恢复点折算全局约 8355 / 10000 step；距离下一个 global 9000 验证/保存点约 645 step。
- checkpoint 状态正常：最新中间恢复点仍为已核验的 `step_8000.pt`，`best.pt` 仍为 global step 5500 的最佳点；`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动，8 项目标产物仍未完成。
- GPU 快照：08:04 左右 GPU1 约 18.3GB 显存、瞬时 util 约 0%、功耗约 22W、温度约 54C；GPU0 为 0MiB、约 9W、43C。07:45-08:00 资源守卫记录总功耗约 59.42W、123.58W、167.28W、98.39W，最高温约 59-66C，均低于 350-400W 边界和 83C 温度硬阈值。
- 队列在 07:45、07:50、07:55、08:00 均正确识别 `target_gpu=1`，并因 GPU1 busy 或当前功耗阈值阻止重复启动 `VIVID_ansmask_resume_gpu1`。
- 最近 loss 在约 0.0322-0.0385 区间内波动，未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 本窗口速度略有回升；按 GPU1 近几个窗口与后续验证开销保守估计，距离 `ansmask final.pt` 约 9-13 小时；全部目标队列粗估约 30-45 小时。
- 阶段性判断：训练、资源状态和自动队列均符合预期；速度仍有波动但没有失败迹象。下一关键观察点仍是 global 9000 / local 2000 附近的验证/保存。

2026-05-31 08:20 复查（GPU1-only 策略，实际检查到 08:25）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上的 VIVID compute app PID 仍为 8468，GPU0 空闲。
- 当前训练进度约 1423 / 3000 step，按 `step_7000.pt` 恢复点折算全局约 8423 / 10000 step；距离下一个 global 9000 验证/保存点约 577 step。
- checkpoint 状态正常：最新中间恢复点仍为已核验的 `step_8000.pt`，`best.pt` 仍为 global step 5500 的最佳点；`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动，8 项目标产物仍未完成。
- GPU 快照：08:25 左右 GPU1 约 18.3GB 显存、util 约 98%、功耗约 40W、温度约 55C；GPU0 为 0MiB、约 8W、43C。08:05-08:20 资源守卫记录总功耗约 92.46W、85.1W、189.39W、100.2W，最高温约 57-67C，均低于 350-400W 边界和 83C 温度硬阈值。
- 队列在 08:05、08:10、08:15、08:20 均正确识别 `target_gpu=1`，并因 GPU1 busy 或当前功耗阈值阻止重复启动 `VIVID_ansmask_resume_gpu1`。
- 最近 loss 在约 0.0333-0.0385 区间内波动，未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 速度较上一轮略有回升；按 GPU1 近几个窗口与后续验证开销保守估计，距离 `ansmask final.pt` 约 8.5-12.5 小时；全部目标队列粗估约 30-44 小时。
- 阶段性判断：训练、资源状态和自动队列均符合预期；速度仍有波动但没有失败迹象。下一关键观察点仍是 global 9000 / local 2000 附近的验证/保存。

2026-05-31 08:40 复查（GPU1-only 策略，实际检查到 08:46）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上的 VIVID compute app PID 仍为 8468，GPU0 空闲。
- 当前训练进度约 1492 / 3000 step，按 `step_7000.pt` 恢复点折算全局约 8492 / 10000 step；距离下一个 global 9000 验证/保存点约 508 step。
- checkpoint 状态正常：最新中间恢复点仍为已核验的 `step_8000.pt`，`best.pt` 仍为 global step 5500 的最佳点；`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动，8 项目标产物仍未完成。
- GPU 快照：08:46 左右 GPU1 约 18.3GB 显存、util 约 83%、功耗约 31W、温度约 56C；GPU0 为 0MiB、约 9W、44C。08:25-08:45 资源守卫记录总功耗约 76.9W、141.54W、97.76W、162.02W、168.13W，最高温约 54-63C，均低于 350-400W 边界和 83C 温度硬阈值。
- 队列在 08:25、08:30、08:35、08:40、08:45 均正确识别 `target_gpu=1`，并因 GPU1 busy 或当前功耗阈值阻止重复启动 `VIVID_ansmask_resume_gpu1`。
- 最近 loss 在约 0.0335-0.0385 区间内波动，未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 按 GPU1 近几个窗口与后续验证开销保守估计，距离 `ansmask final.pt` 约 8-12 小时；全部目标队列粗估约 29-43 小时。
- 阶段性判断：训练、资源状态和自动队列均符合预期；没有失败迹象。下一关键观察点仍是 global 9000 / local 2000 附近的验证/保存。

2026-05-31 09:05 复查（GPU1-only 策略，实际检查到 09:07）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上的 VIVID compute app PID 仍为 8468，GPU0 空闲。
- 当前训练进度约 1540 / 3000 step，按 `step_7000.pt` 恢复点折算全局约 8540 / 10000 step；距离下一个 global 9000 验证/保存点约 460 step。
- 本轮有正向结果：`outputs/ablation_ums_ansmask_12label/checkpoints/best.pt` 在 2026-05-31 08:56:25 刷新。已用 `torch.load(..., map_location='cpu')` 轻量核验内部元数据，`global_step=8500`、`best_val_loss=0.037125732421875`，优于此前 best 的 `0.03747265625`。
- checkpoint 状态：最新命名中间恢复点仍为已核验的 `step_8000.pt`；`best.pt` 已刷新为 global step 8500 的最佳点。`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动，8 项目标产物仍未完成。
- GPU 快照：09:07 左右 GPU1 约 18.3GB 显存、瞬时 util 约 0%、功耗约 18W、温度约 53C；GPU0 为 0MiB、约 9W、43C。08:50-09:05 资源守卫记录总功耗约 21.64W、164.49W、135.55W、186.82W，最高温约 52-65C，均低于 350-400W 边界和 83C 温度硬阈值。
- 队列在 08:50、08:55、09:00、09:05 均正确识别 `target_gpu=1`，并因 GPU1 busy 或当前功耗阈值阻止重复启动 `VIVID_ansmask_resume_gpu1`。
- 验证后训练已恢复；最近 loss 在约 0.0305-0.0347 区间内波动，未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 按 GPU1 近几个窗口与后续验证开销保守估计，距离 `ansmask final.pt` 约 9-13 小时；全部目标队列粗估约 30-45 小时。
- 阶段性判断：训练、资源状态和自动队列均符合预期；global 8500 刷新 best 是积极信号，当前结果符合预期且略好于前一最佳。下一关键观察点是 global 9000 / local 2000 的验证/保存。

2026-05-31 09:25 复查（GPU1-only 策略，实际检查到 09:29）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上的 VIVID compute app PID 仍为 8468，GPU0 空闲。
- 当前训练进度约 1599 / 3000 step，按 `step_7000.pt` 恢复点折算全局约 8599 / 10000 step；距离下一个 global 9000 验证/保存点约 401 step。
- checkpoint 状态正常：最新命名中间恢复点仍为已核验的 `step_8000.pt`；`best.pt` 仍为 08:56 刷新的 global step 8500 最佳点，`best_val_loss=0.037125732421875`。`final.pt` 尚未生成，因此 `lp_ums_ansmask` 尚未启动，8 项目标产物仍未完成。
- GPU 快照：09:29 左右 GPU1 约 18.3GB 显存、瞬时 util 约 5%、功耗约 98W、温度约 61C；GPU0 为 0MiB、约 9W、43C。09:10-09:25 资源守卫记录总功耗约 86.06W、99.75W、97.02W、92.36W，最高温约 57-60C，均低于 350-400W 边界和 83C 温度硬阈值。
- 队列在 09:10、09:15、09:20、09:25 均正确识别 `target_gpu=1`，并因 GPU1 busy 阻止重复启动 `VIVID_ansmask_resume_gpu1`。
- 最近 loss 在约 0.0321-0.0380 区间内波动，未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 按 GPU1 近几个窗口与后续验证开销保守估计，距离 `ansmask final.pt` 约 8-12 小时；全部目标队列粗估约 29-43 小时。
- 阶段性判断：训练、资源状态和自动队列均符合预期；global 8500 best 刷新后训练稳定，没有失败迹象。下一关键观察点是 global 9000 / local 2000 的验证/保存。

2026-05-31 09:50 复查（GPU1-only 策略，实际检查到 09:51）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上的 VIVID compute app PID 仍为 8468，GPU0 空闲。后续继续按用户最新指令固定使用 GPU1。
- 当前训练进度约 1666 / 3000 step，按 `step_7000.pt` 恢复点折算全局约 8666 / 10000 step；距离下一个 global 9000 验证/保存点约 334 step。
- checkpoint 状态正常：最新命名中间恢复点仍为已核验的 `step_8000.pt`；`best.pt` 仍为 08:56 刷新的 global step 8500 最佳点，`best_val_loss=0.037125732421875`。截至本次检查，尚未看到 `step_9000.pt` 或 `final.pt`，因此 `lp_ums_ansmask` 尚未启动，8 项目标产物仍未完成。
- GPU 快照：09:51 左右 GPU1 约 18.3GB 显存、util 约 31%、功耗约 62W、温度约 53C；GPU0 为 0MiB、约 9W、43C。09:30-09:50 资源守卫记录总功耗约 91.96W、95.67W、155.73W、116.92W、100.83W，最高温约 57-66C，均低于 350-400W 边界和 83C 温度硬阈值。
- 队列在 09:30、09:35、09:40、09:45、09:50 均正确识别 `target_gpu=1`，并因 GPU1 busy 或当前功耗阈值阻止重复启动 `VIVID_ansmask_resume_gpu1`。
- 最近 loss 在约 0.0328-0.0380 区间内波动，未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 按 GPU1 近几个窗口与后续验证开销保守估计，距离 `ansmask final.pt` 约 7.5-11.5 小时；全部目标队列粗估约 28-42 小时。
- 阶段性判断：训练、资源状态和自动队列均符合预期；global 8500 best 刷新仍是积极信号，当前没有失败迹象。下一关键观察点是 global 9000 / local 2000 的验证/保存。

2026-05-31 10:12 复查（GPU1-only 策略）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上的 VIVID compute app PID 仍为 8468，GPU0 空闲。
- 当前训练进度约 1724 / 3000 step，按 `step_7000.pt` 恢复点折算全局约 8724 / 10000 step；距离下一个 global 9000 验证/保存点约 276 step。
- checkpoint 状态正常：最新命名中间恢复点仍为已核验的 `step_8000.pt`；`best.pt` 仍为 08:56 刷新的 global step 8500 最佳点，`best_val_loss=0.037125732421875`。截至本次检查，尚未看到 `step_9000.pt` 或 `final.pt`，因此 `lp_ums_ansmask` 尚未启动，8 项目标产物仍未完成。
- GPU 快照：10:12 左右 GPU1 约 18.3GB 显存、util 约 68%、功耗约 77W、温度约 58C；GPU0 为 0MiB、约 8W、41C。09:55-10:10 资源守卫记录总功耗约 185.55W、112.38W、108.59W、85.36W，最高温约 59-62C，均低于 350-400W 边界和 83C 温度硬阈值。
- 队列在 09:55、10:00、10:05、10:10 均正确识别 `target_gpu=1`，并因 GPU1 busy 或当前功耗阈值阻止重复启动 `VIVID_ansmask_resume_gpu1`。
- 最近 loss 在约 0.0317-0.0358 区间内波动，学习率约 `1.04e-05`，未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 按 GPU1 最近一个窗口的速度和后续验证开销保守估计，距离 `step_9000.pt` 约 1.5-2.5 小时；距离 `ansmask final.pt` 约 8-11.5 小时；全部目标队列粗估约 28-42 小时。
- 阶段性判断：训练、资源状态和自动队列均符合预期；没有失败迹象。下一关键观察点仍是 global 9000 / local 2000 的验证/保存。

2026-05-31 10:33 复查（GPU1-only 策略）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上的 VIVID compute app PID 仍为 8468，GPU0 空闲。
- 当前训练进度约 1806 / 3000 step，按 `step_7000.pt` 恢复点折算全局约 8806 / 10000 step；距离下一个 global 9000 验证/保存点约 194 step。
- checkpoint 状态正常：最新命名中间恢复点仍为已核验的 `step_8000.pt`；`best.pt` 仍为 08:56 刷新的 global step 8500 最佳点，`best_val_loss=0.037125732421875`。截至本次检查，尚未看到 `step_9000.pt` 或 `final.pt`，因此 `lp_ums_ansmask` 尚未启动，8 项目标产物仍未完成。
- GPU 快照：10:33 左右 GPU1 约 18.3GB 显存、瞬时 util 约 0%、功耗约 22W、温度约 55C；GPU0 为 0MiB、约 9W、42C。10:15-10:30 资源守卫记录总功耗约 105.78W、132.4W、194.43W、101.9W，最高温约 58-67C，均低于 350-400W 边界和 83C 温度硬阈值。
- 队列在 10:15、10:20、10:25、10:30 均正确识别 `target_gpu=1`，并因 GPU1 busy 或当前功耗阈值阻止重复启动 `VIVID_ansmask_resume_gpu1`。
- 最近 loss 在约 0.0305-0.0405 区间内波动，学习率约 `1.04e-05`，未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 按 GPU1 最近两个窗口的速度和后续验证开销保守估计，距离 `step_9000.pt` 约 1-1.5 小时；距离 `ansmask final.pt` 约 7-10.5 小时；全部目标队列粗估约 27-41 小时。
- 阶段性判断：训练、资源状态和自动队列均符合预期；没有失败迹象。下一关键观察点仍是 global 9000 / local 2000 的验证/保存。

2026-05-31 10:54 复查（GPU1-only 策略）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上的 VIVID compute app PID 仍为 8468，GPU0 空闲。
- 当前训练进度约 1888 / 3000 step，按 `step_7000.pt` 恢复点折算全局约 8888 / 10000 step；距离下一个 global 9000 验证/保存点约 112 step。
- checkpoint 状态正常：最新命名中间恢复点仍为已核验的 `step_8000.pt`；`best.pt` 仍为 08:56 刷新的 global step 8500 最佳点，`best_val_loss=0.037125732421875`。截至本次检查，尚未看到 `step_9000.pt` 或 `final.pt`，因此 `lp_ums_ansmask` 尚未启动，8 项目标产物仍未完成。
- GPU 快照：10:54 左右 GPU1 约 18.3GB 显存、util 约 57%、功耗约 17W、温度约 55C；GPU0 为 0MiB、约 9W、41C。10:35-10:50 资源守卫记录总功耗约 102.73W、108.77W、115.68W、168.25W，最高温约 53-66C，均低于 350-400W 边界和 83C 温度硬阈值。
- 队列在 10:35、10:40、10:45、10:50 均正确识别 `target_gpu=1`，并因 GPU1 busy 或当前功耗阈值阻止重复启动 `VIVID_ansmask_resume_gpu1`。
- 最近 loss 在约 0.0300-0.0405 区间内波动，学习率约 `1.03e-05` 到 `1.04e-05`，未见 traceback / RuntimeError / CUDA OOM / exitcode。10:23 左右有单步耗时拉长，但随后训练恢复正常，不构成失败 case。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 按 GPU1 最近窗口速度和后续验证开销保守估计，距离 `step_9000.pt` 约 0.5-1 小时；距离 `ansmask final.pt` 约 6.5-9.5 小时；全部目标队列粗估约 26-40 小时。
- 阶段性判断：训练、资源状态和自动队列均符合预期；没有失败迹象。下一关键观察点是 imminent 的 global 9000 / local 2000 验证与 `step_9000.pt` 元数据核验。

2026-05-31 11:15 复查（GPU1-only 策略，接近 global 9000）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上的 VIVID compute app PID 仍为 8468，GPU0 空闲。
- 当前训练进度约 1969 / 3000 step，按 `step_7000.pt` 恢复点折算全局约 8969 / 10000 step；距离下一个 global 9000 验证/保存点约 31 step。
- checkpoint 状态正常：最新命名中间恢复点仍为已核验的 `step_8000.pt`；`best.pt` 仍为 08:56 刷新的 global step 8500 最佳点，`best_val_loss=0.037125732421875`。截至本次检查，尚未看到 `step_9000.pt` 或 `final.pt`，因此 `lp_ums_ansmask` 尚未启动，8 项目标产物仍未完成。
- GPU 快照：11:15 左右 GPU1 约 18.3GB 显存、util 约 95%、功耗约 32W、温度约 54C；GPU0 为 0MiB、约 9W、42C。10:55-11:10 资源守卫记录总功耗约 169.19W、89.49W、153.4W、109.17W，最高温约 57-64C，均低于 350-400W 边界和 83C 温度硬阈值。
- 队列在 10:55、11:00、11:05、11:10 均正确识别 `target_gpu=1`，并因 GPU1 busy 或当前功耗阈值阻止重复启动 `VIVID_ansmask_resume_gpu1`。
- 最近 loss 在约 0.0297-0.0390 区间内波动，学习率约 `1.03e-05`，未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败 case 需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- 按当前速度和后续验证开销估计，`step_9000.pt` 预计会在下一小段运行和验证后出现；距离 `ansmask final.pt` 约 6-9 小时；全部目标队列粗估约 26-39 小时。
- 阶段性判断：训练、资源状态和自动队列均符合预期；没有失败迹象。由于 global 9000 已很近，下一轮临时缩短检查间隔，重点核验 `step_9000.pt` 的内部元数据。

2026-05-31 12:18 复查与失败 case study（GPU1-only 恢复）：
- 异常现象：`VIVID_ansmask_resume_gpu1` 在接近 global 9000 前掉线。11:15 时已到约 global 8969 / 10000；11:31 资源守卫显示 GPU1 已变为 0MiB，checkpoint 目录未出现 `step_9000.pt`，`final.pt` 也未生成。由于当时 runner 仍使用覆盖日志，掉线前的最后 traceback/退出原因没有被保留下来。
- 直接证据：11:31 队列检测到 GPU1 空闲并尝试重启 `VIVID_ansmask_resume_gpu1`；第一次重启日志只留下 `start ansmask resume...` 和 `^C`，计划任务结果出现 `-1073741510`。因此本次按失败 case 处理，而不是正常 checkpoint 边界。
- 影响范围：没有损坏已有 checkpoint；最新可靠训练恢复点仍是 08:56 刷新的 `best.pt`，内部元数据此前已核验为 `global_step=8500`、`best_val_loss=0.037125732421875`。因为 `step_9000.pt` 尚未保存，本次丢失的是 global 8500 之后、约到 8969 的未落盘训练进度，约 469 step。
- 排查结论：GPU0 上可见的 PID 9888 是 `GameViewerServer`，不是 VIVID 训练；VIVID 训练仍按用户要求恢复到 GPU1。旧队列的启动阈值 `120W` 过低，在 GPU0 有非训练轻负载时会误阻塞 GPU1 重启；此外 ansmask resume 重新构造模型时无需再次加载预训练 ViT 权重，因为 checkpoint 会覆盖 ViT/projector/optimizer 状态。
- 修复措施：
  - 将 GPU1 队列启动阈值从 `120W` 调整为 `220W`，仍低于用户给定的 350-400W 长期功耗边界，并继续保留 400W hard cap 与 83C hard temp。
  - 新增 `configs/ablation_ums_ansmask_12label_resume_gpu1.yaml`，仅用于 ansmask resume，设置 `vit_pretrained: false`，避免恢复时卡在无必要的预训练 ViT 权重加载。
  - 修改 `scripts/run_ansmask_resume_gpu1.cmd` 使用恢复专用 config，并将日志改为 append；其余 GPU1 后续 runner 也改为 append 日志并设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID`、`PYTHONUNBUFFERED=1`，后续失败证据不再被覆盖。
- 恢复结果：12:18 检查时 `VIVID_ansmask_resume_gpu1` 已重新进入训练循环；日志显示 `Resuming from step 8500`，并开始 `Training: 1/1500`。GPU1 上 VIVID compute PID 为 5300，显存约 15.2GB；GPU0 上只有非训练 `GameViewerServer` compute app。12:18 快照总功耗约 184W、最高温约 47C，均低于安全边界。
- 当前进度：可靠全局进度回到 8500 / 10000，当前本轮续跑约 1 / 1500 step；距离 `step_9000.pt` 约 500 step，距离 `ansmask final.pt` 约 1500 step。`lp_ums_ansmask` 尚未启动，8 项目标产物仍未完成。
- ETA 更新：考虑本次回退和重启开销，`step_9000.pt` 粗估约 2-4 小时后出现；`ansmask final.pt` 粗估约 8-12 小时；全部目标队列粗估约 29-43 小时。
- 结果判断：此前 global 8500 的 best 刷新仍是正向结果；本次掉线不符合预期，但已有可用 checkpoint 且已恢复到 GPU1，当前恢复后的资源与训练状态符合预期。下一步继续盯 `step_9000.pt` 与 `final.pt`。

2026-05-31 13:14 复查（GPU1-only，半小时低频巡检）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；计划任务最后启动时间为 2026-05-31 12:10:10。GPU1 上的 VIVID compute PID 为 5300。
- 当前训练进度约 54 / 1500 step，按 `best.pt` 的可靠恢复点 `global_step=8500` 折算，全局约 8554 / 10000 step；距离下一个 `step_9000.pt` 约 446 step，距离 `ansmask final.pt` 约 1446 step。
- checkpoint 状态：`outputs/ablation_ums_ansmask_12label/checkpoints/` 中仍只有 `best.pt`、`step_8000.pt` 及更早 checkpoint；截至本轮未出现 `step_9000.pt` 或 `final.pt`。因此 `lp_ums_ansmask` 尚未启动，8 项目标产物仍全部未完成。
- GPU 快照：13:14 左右 GPU1 约 16.8GB 显存、util 约 87%、功耗约 78W、温度约 48C；GPU0 约 32MiB 显存、util 0%、功耗约 113W、温度约 51C。13:10 资源守卫记录总功耗约 207W、最高温约 54C，低于 350-400W 长期功耗边界和 83C 温度硬阈值。
- 队列在 12:45、12:50、12:55、13:00、13:05、13:10 均正确识别 `target_gpu=1` busy，并阻止重复启动。资源守卫均为 OK，无 stop action。
- 最近可见 loss 约在 0.0307-0.0344 区间，学习率约 `1.06e-05`；未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败 case 需要 case study。
- 重要更正：当前 12:10 启动的日志仍显示 `Loading config from configs\ablation_ums_ansmask_12label.yaml`，说明这一个正在运行的进程并未使用后续已写入 runner 的 resume 专用 config；runner 文件当前已经改为 `configs\ablation_ums_ansmask_12label_resume_gpu1.yaml`，会在下一次重启时生效。由于当前进程已经稳定训练且资源安全，暂不为此中断训练。
- ETA 更新：按 12:10 之后实际步速重新估计，`step_9000.pt` 约还需 7-10 小时；`ansmask final.pt` 约还需 23-31 小时。后续 7 个目标产物仍未开始，当前全目标剩余粗估约 45-65 小时；若步速恢复到 09:00-11:00 窗口水平，ETA 会相应缩短。
- 阶段性判断：训练已从失败点恢复并稳定推进，结果指标仍符合预期；主要变化是当前续跑窗口明显慢于上午的乐观估计，但没有温度、功耗或 OOM 证据指向硬件异常。

2026-05-31 13:47 复查（GPU1-only，半小时低频巡检）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；计划任务最后启动时间仍为 2026-05-31 12:10:10。GPU1 上的 VIVID compute PID 仍为 5300。
- 当前训练进度约 74 / 1500 step，按 `best.pt` 的可靠恢复点 `global_step=8500` 折算，全局约 8574 / 10000 step；距离下一个 `step_9000.pt` 约 426 step，距离 `ansmask final.pt` 约 1426 step。
- checkpoint 状态：仍未出现 `step_9000.pt` 或 `final.pt`，`best.pt` 仍为 2026-05-31 08:56:25 刷新的 global 8500 最佳点。因此 `lp_ums_ansmask` 尚未启动，8 项目标产物仍全部未完成。
- GPU 快照：13:47 左右 GPU1 约 16.8GB 显存、util 约 68%、功耗约 16W、温度约 46C；GPU0 0MiB、util 0%、功耗约 9W、温度约 36C。13:45 资源守卫记录总功耗约 64.8W、最高温约 47C，低于 350-400W 长期功耗边界和 83C 温度硬阈值。
- 队列在 13:15、13:20、13:25、13:30、13:35、13:40、13:45 均正确识别 `target_gpu=1` busy，并阻止重复启动。资源守卫均为 OK，无 stop action。
- 最近可见 loss 约在 0.0321-0.0344 区间，学习率约 `1.05e-05` 到 `1.06e-05`；未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败 case 需要 case study。
- 步速观察：13:14 到 13:47 之间只从 local 54 推进到 74，约 20 step / 33 分钟；明显慢于上午窗口，也慢于 13:14 时的均速。GPU 温度与总功耗都很安全，因此目前更像数据加载、CPU/系统调度或 Windows 后台状态带来的吞吐波动，不像过热或功耗保护。
- ETA 更新：按最近半小时慢速窗口重新估计，`step_9000.pt` 约还需 10-13 小时；`ansmask final.pt` 约还需 35-45 小时。全目标剩余粗估约 55-80 小时；若后续步速回升，ETA 会下调。
- 阶段性判断：训练仍稳定推进，loss 水平符合预期，没有新失败；但速度不符合上午预期，后续继续按 30 分钟低频观察是否只是短时吞吐波动。

2026-05-31 14:19 复查（GPU1-only，半小时低频巡检）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上的 VIVID compute PID 仍为 5300。
- 当前训练进度约 101 / 1500 step，按 `best.pt` 的可靠恢复点 `global_step=8500` 折算，全局约 8601 / 10000 step；距离下一个 `step_9000.pt` 约 399 step，距离 `ansmask final.pt` 约 1399 step。
- checkpoint 状态：仍未出现 `step_9000.pt` 或 `final.pt`，`best.pt` 仍为 2026-05-31 08:56:25 刷新的 global 8500 最佳点。因此 `lp_ums_ansmask` 尚未启动，8 项目标产物仍全部未完成。
- GPU 快照：14:19 左右 GPU1 约 16.8GB 显存、util 约 28%、功耗约 33W、温度约 54C；GPU0 0MiB、util 0%、功耗约 9W、温度约 39C。14:15 资源守卫记录总功耗约 84.87W、最高温约 58C，低于 350-400W 长期功耗边界和 83C 温度硬阈值。
- 队列在 13:50、13:55、14:00、14:05、14:10、14:15 均正确识别 `target_gpu=1` busy，并阻止重复启动。资源守卫均为 OK，无 stop action。
- 最近可见 loss 约在 0.0280-0.0335 区间，学习率约 `1.05e-05`；未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败 case 需要 case study。
- 步速观察：13:47 到 14:19 从 local 74 推进到 101，约 27 step / 32 分钟，较上一轮慢速窗口明显回升，但仍慢于上午 09:00-11:00 附近的较好窗口。
- ETA 更新：按最近半小时回升后的步速估计，`step_9000.pt` 约还需 7-9 小时；`ansmask final.pt` 约还需 26-32 小时。全目标剩余粗估约 45-65 小时。
- 阶段性判断：训练稳定推进，loss 仍在合理范围且出现更低的短窗 loss；速度波动有所缓解，当前整体符合恢复后的预期。

2026-05-31 14:54 复查与失败 case study（GPU1-only 自动恢复）：
- 异常现象：12:10 启动的 `VIVID_ansmask_resume_gpu1` 在 `step_9000.pt` 之前再次掉线。日志最后可见训练进度为 local `105 / 1500`，按 `best.pt` 的可靠恢复点 `global_step=8500` 折算约 global `8605 / 10000`；14:40 资源守卫记录 GPU1 仅约 82MiB 显存，说明训练进程已不再保持完整训练状态。checkpoint 目录仍未出现 `step_9000.pt` 或 `final.pt`。
- 直接证据：14:20 队列仍看到 GPU1 busy，14:26 资源守卫仍记录 GPU1 约 16.8GB；到 14:40 GPU1 降到约 82MiB；14:50 队列看到 GPU1 0MiB 且自动重新启动 `VIVID_ansmask_resume_gpu1`。runner 日志没有写出 `exitcode`、traceback、RuntimeError 或 CUDA OOM，因此当前只能判为无 traceback 的掉线/中断。
- 影响范围：没有发现 checkpoint 损坏；最新可靠可恢复点仍是 2026-05-31 08:56:25 的 `best.pt`，内部元数据此前已核验为 `global_step=8500`、`best_val_loss=0.037125732421875`。本次丢失的是从 global 8500 到约 8605 的未落盘进度，约 105 step。
- 当前恢复状态：14:50:05 队列重新拉起 `VIVID_ansmask_resume_gpu1`；14:53 进程 PID 为 14032，使用 `CUDA_VISIBLE_DEVICES=1`、`configs\ablation_ums_ansmask_12label_resume_gpu1.yaml` 和 checkpoint `best.pt`。日志显示已重新加载 LLM、创建 trainer，并 `Resuming from step 8500`；截至 14:54 尚未看到新一轮第 1 个训练 step 输出，GPU1 显存约 4.9GB、GPU0 空闲。
- GPU 与资源判断：14:50 GPU0/GPU1 总功耗约 17.06W，最高温约 37C；14:53 快照 GPU0 为 0MiB、约 8.36W、34-35C，GPU1 约 4.9GB、util 0%、约 7.66W、35-36C。没有过热或超过功耗阈值的证据。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- ETA 更新：由于第二次回退到 global 8500，距离 `step_9000.pt` 仍约 500 step，距离 `ansmask final.pt` 约 1500 step。按 12:10-14:17 的慢速窗口估计，`step_9000.pt` 约还需 8-12 小时，`ansmask final.pt` 约还需 25-35 小时；全目标剩余粗估约 50-75 小时。若新一轮恢复到上午较快窗口，ETA 会下调。
- 阶段性判断：训练结果本身仍符合预期，主要依据是 `best.pt` 在 global 8500 刷新到 `best_val_loss=0.037125732421875` 且最近可见 loss 在约 0.0280-0.0335 区间；但运行稳定性不符合预期，连续两次在 `step_9000.pt` 之前掉线。下一步先确认 14:50 新进程是否进入训练 step；如果再次掉线，需要优先改为更频繁 checkpoint 或缩短验证/保存间隔，避免反复丢失 8500 之后的进度。

2026-05-31 14:59 受控短重启（降低后续掉线回退风险）：
- 措施：将 `configs/ablation_ums_ansmask_12label_resume_gpu1.yaml` 的 `training.save_interval` 从 `1000` 改为 `100`，保持 `eval_interval=500`、`max_steps=10000`、`vit_pretrained=false` 不变。该改动只作用于 GPU1 resume 专用配置，不改变主实验定义。
- 原因：14:50 自动恢复进程已进入训练并跑到 local 9/1500，但它是在配置改动前启动的，不会读取新的 `save_interval=100`。为避免第三次在 `step_9000.pt` 前掉线又回退到 global 8500，主动停止该进程并立即重启；本次有意损失约 9 step，换取后续每 100 step 保存一次恢复点。
- 执行结果：14:56:56 重新启动 `VIVID_ansmask_resume_gpu1`；日志显示加载 `configs\ablation_ums_ansmask_12label_resume_gpu1.yaml`，并从 `best.pt` 恢复到 `global_step=8500`。14:58 已进入新一轮训练，当前约 local `4 / 1500`，折算 global 约 `8504 / 10000`。
- GPU 状态：14:58 GPU0 为 0MiB、空闲；GPU1 约 15.2GB 显存、util 约 62%、功耗约 245W、温度约 65C，低于 350-400W 长期功耗边界和 83C 温度硬阈值。
- 预期变化：如果这轮能跑到 local 100，将生成 `step_8600.pt`，之后即使掉线也不再回退到 8500；`step_9000.pt` 仍是下一次重要验证/保存节点。

2026-05-31 15:00 复查（GPU1-only，短重启后稳定运行）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running，最近启动时间为 14:56:56；队列在 15:00 正确识别 `target_gpu=1` busy，并阻止重复启动。
- 当前训练进度约 local `12 / 1500`，折算 global 约 `8512 / 10000`；距离新的短间隔 checkpoint `step_8600.pt` 约 88 step，距离 `step_9000.pt` 约 488 step，距离 `ansmask final.pt` 约 1488 step。
- GPU 快照：15:00 GPU0 为 0MiB、空闲；GPU1 约 16.8GB 显存、util 约 61-69%、功耗约 144-171W、温度约 63-65C，低于功耗与温度边界。
- 最近 loss：local 10 step 处 loss 约 `0.0362`，学习率约 `1.06e-05`；未见 traceback / RuntimeError / CUDA OOM。
- ETA 更新：按 14:56 后前 12 step 的较快窗口估计，`step_8600.pt` 约 20-40 分钟，`step_9000.pt` 约 2-4 小时，`ansmask final.pt` 约 7-12 小时；考虑到此前速度波动，全目标剩余保守仍按约 45-70 小时记录。

2026-05-31 15:01 复查（GPU1-only，小时同步）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running，计划任务最近启动时间为 2026-05-31 14:56:56；GPU1 上 VIVID compute PID 为 7108。GPU0 仍空闲。
- 当前训练进度约 local `21 / 1500`，按 `best.pt` 的可靠恢复点 `global_step=8500` 折算 global 约 `8521 / 10000`；距离新的短间隔 checkpoint `step_8600.pt` 约 79 step，距离 `step_9000.pt` 约 479 step，距离 `ansmask final.pt` 约 1479 step。
- checkpoint 状态：`outputs/ablation_ums_ansmask_12label/checkpoints/` 中仍未出现 `step_8600.pt`、`step_9000.pt` 或 `final.pt`；最新可靠落盘点仍是 `best.pt` 和 `step_8000.pt`。因此 `lp_ums_ansmask` 尚未启动，8 项目标产物仍全部未完成。
- GPU 快照：15:01 GPU0 为 0MiB、util 0%、约 11W、36C；GPU1 约 16.8GB 显存、util 约 12-69% 波动、功耗约 160-185W、温度约 63-66C，低于 350-400W 功耗边界和 83C 温度硬阈值。
- 队列/守卫状态：15:00 队列正确识别 `target_gpu=1` busy 并阻止重复启动；资源守卫记录 OK，无 stop action。
- 最近 loss：local 10 step loss 约 `0.0362`，local 20 step loss 约 `0.0308`，学习率约 `1.06e-05`；未见 traceback / RuntimeError / CUDA OOM。当前没有新的失败 case 需要 case study。
- ETA 更新：按 14:56 后约 21 step / 5 分钟的短窗估计，`step_8600.pt` 约 20-35 分钟；`step_9000.pt` 约 2-4 小时；`ansmask final.pt` 约 7-12 小时。考虑到此前 Windows/数据加载速度波动，全目标剩余保守仍按约 45-70 小时记录。
- 阶段性判断：当前训练、GPU1-only 策略、资源守卫和 loss 水平均符合预期；主要剩余风险仍是此前无 traceback 掉线，因此下一关键观察点是确认 `step_8600.pt` 成功保存。

2026-05-31 15:03 快速复查（GPU1-only）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上 VIVID compute PID 仍为 7108，GPU0 空闲。
- 当前训练进度约 local `29 / 1500`，折算 global 约 `8529 / 10000`；距离 `step_8600.pt` 约 71 step，距离 `step_9000.pt` 约 471 step，距离 `ansmask final.pt` 约 1471 step。
- checkpoint 状态：尚未出现 `step_8600.pt`、`step_9000.pt` 或 `final.pt`；这与当前 local step 尚未到 100 一致。
- GPU 快照：15:02 GPU1 约 16.8GB 显存、util 约 43%、功耗约 181W、温度约 68C；GPU0 0MiB、约 10W、37C。15:00 资源守卫 OK，无 stop action。
- 最近 loss：local 20 step loss 约 `0.0308`，学习率约 `1.06e-05`；未见 traceback / RuntimeError / CUDA OOM。当前没有新的失败 case。
- ETA：按 14:56 后短窗速度，`step_8600.pt` 约 15-30 分钟；`step_9000.pt` 约 2-4 小时；全目标保守仍约 45-70 小时。

2026-05-31 15:04 续查（GPU1-only）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上 VIVID compute PID 仍为 7108，GPU0 空闲，GPU 绑定符合用户“只用 gpu1”的要求。
- 当前训练进度约 local `38 / 1500`，折算 global 约 `8538 / 10000`；距离 `step_8600.pt` 约 62 step，距离 `step_9000.pt` 约 462 step，距离 `ansmask final.pt` 约 1462 step。
- checkpoint 状态：仍未出现 `step_8600.pt`、`step_9000.pt` 或 `final.pt`；最新可靠落盘点仍是 `best.pt` 和 `step_8000.pt`。
- GPU 快照：15:04 GPU1 约 16.8GB 显存、util 约 79%、功耗约 97W、温度约 61C；GPU0 为 0MiB、约 9W、38C。资源仍低于 350-400W 功耗边界和 83C 温度硬阈值。
- 最近 loss：local 30 step loss 约 `0.0322`，学习率约 `1.06e-05`；未见 traceback / RuntimeError / CUDA OOM。当前没有新的失败 case。
- ETA：按 14:56 后当前短窗速度，`step_8600.pt` 约 10-20 分钟；`step_9000.pt` 约 1.5-3 小时；`ansmask final.pt` 约 5-9 小时。考虑此前掉线和速度波动，全目标剩余保守仍约 40-65 小时。

2026-05-31 15:09 续查（GPU1-only，等待短间隔 checkpoint）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上 VIVID compute PID 仍为 7108，GPU0 空闲。15:05 队列正确识别 `target_gpu=1` busy，并阻止重复启动。
- 当前训练进度约 local `57 / 1500`，折算 global 约 `8557 / 10000`；距离 `step_8600.pt` 约 43 step，距离 `step_9000.pt` 约 443 step，距离 `ansmask final.pt` 约 1443 step。
- checkpoint 状态：仍未出现 `step_8600.pt`、`step_9000.pt` 或 `final.pt`；当前 local step 尚未到 100，因此仍符合预期。
- GPU 快照：15:09 GPU1 约 16.8GB 显存、util 约 47%、功耗约 238W、温度约 69C；GPU0 为 0MiB、约 12W、39C。15:05 资源守卫 OK，无 stop action，仍低于 350-400W 功耗边界和 83C 温度硬阈值。
- 最近 loss：local 50 step loss 约 `0.0344`，学习率约 `1.06e-05`；未见 traceback / RuntimeError / CUDA OOM。当前没有新的失败 case。
- ETA：按 14:56 后当前短窗速度，`step_8600.pt` 约 8-15 分钟；`step_9000.pt` 约 1.5-3 小时；`ansmask final.pt` 约 5-9 小时。考虑此前掉线和速度波动，全目标剩余保守仍约 40-65 小时。

2026-05-31 15:19 复查（短间隔 checkpoint 已验证）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上 VIVID compute PID 仍为 7108，GPU0 空闲。15:15 队列正确识别 `target_gpu=1` busy，并阻止重复启动。
- `step_8600.pt` 已成功生成：文件时间为 2026-05-31 15:17:08，大小约 1.072GB。只读 `torch.load(..., map_location='cpu')` 核验通过，内部 `global_step=8600`、`best_val_loss=0.037125732421875`，keys 为 `['best_val_loss', 'global_step', 'optimizer', 'projector', 'scheduler', 'vit']`。
- 当前训练已继续到约 local `115 / 1500`，折算 global 约 `8615 / 10000`；距离下一个短间隔 checkpoint `step_8700.pt` 约 85 step，距离 `step_9000.pt` 约 385 step，距离 `ansmask final.pt` 约 1385 step。
- GPU 快照：15:19 GPU1 约 16.8GB 显存、util 约 74%、功耗约 82W、温度约 61C；GPU0 为 0MiB、约 9W、42C。15:15 资源守卫 OK，无 stop action，仍低于功耗和温度边界。
- 最近 loss：local 100 step 附近保存 `step_8600.pt` 时 loss 约 `0.0331`，local 110 step loss 约 `0.0295`，学习率约 `1.05e-05`；未见 traceback / RuntimeError / CUDA OOM。当前没有新的失败 case。
- 阶段性判断：短间隔保存策略已生效，之前“掉线后总退回 global 8500”的主要风险已明显降低；训练指标和资源状态符合预期。
- ETA：按 14:56 后到 15:19 的速度估计，`step_8700.pt` 约 15-30 分钟，`step_9000.pt` 约 1.2-2.5 小时，`ansmask final.pt` 约 5-9 小时。考虑此前掉线和速度波动，全目标剩余保守约 38-62 小时。

2026-05-31 15:20 续查（GPU1-only，短间隔防护继续生效）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上 VIVID compute PID 仍为 7108，GPU0 空闲。15:20 队列正确识别 `target_gpu=1` busy，并阻止重复启动。
- 当前训练约 local `122 / 1500`，折算 global 约 `8622 / 10000`；距离下一个短间隔 checkpoint `step_8700.pt` 约 78 step，距离 `step_9000.pt` 约 378 step，距离 `ansmask final.pt` 约 1378 step。
- checkpoint 状态：最新可靠恢复点为已核验的 `step_8600.pt`；尚未出现 `step_8700.pt`、`step_9000.pt` 或 `final.pt`。
- GPU 快照：15:20 GPU1 约 16.8GB 显存、util 约 14-26%、功耗约 116-143W、温度约 62-63C；GPU0 为 0MiB、约 9-10W、42C。资源守卫 OK，无 stop action。
- 最近 loss：local 120 step loss 约 `0.0300`，学习率约 `1.05e-05`；未见 traceback / RuntimeError / CUDA OOM。当前没有新的失败 case。
- ETA：按最近短窗速度，`step_8700.pt` 约 15-30 分钟，`step_9000.pt` 约 1.2-2.5 小时，`ansmask final.pt` 约 5-9 小时；全目标保守约 38-62 小时。

2026-05-31 15:23 续查（GPU1-only，短间隔 checkpoint 已落盘）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；计划任务最近启动时间为 2026-05-31 14:56:56，GPU1 上 VIVID compute PID 为 7108。GPU0 仍为空闲，GPU 绑定符合“只用 gpu1”的要求。
- 当前训练约 local `136 / 1500`，按已核验的 `best.pt` 恢复点 `global_step=8500` 折算 global 约 `8636 / 10000`；距离下一个短间隔 checkpoint `step_8700.pt` 约 64 step，距离 `step_9000.pt` 约 364 step，距离 `ansmask final.pt` 约 1364 step。
- checkpoint 状态：最新可靠恢复点为 `step_8600.pt`，文件时间 2026-05-31 15:17:08，大小约 1.072GB；本轮只读 `torch.load(..., map_location='cpu')` 再次核验通过，内部 `global_step=8600`、`best_val_loss=0.037125732421875`，keys 为 `['best_val_loss', 'global_step', 'optimizer', 'projector', 'scheduler', 'vit']`。尚未出现 `step_8700.pt`、`step_9000.pt` 或 `final.pt`。
- GPU 快照：15:22 左右 GPU1（bus `00000000:05:00.0`）约 16.8GB 显存、util 约 38%、功耗约 150.59W、温度 64C；GPU0（bus `00000000:01:00.0`）为 0MiB、util 0%、约 10.80W、42C。15:20 资源守卫记录 total_power 约 124.52W、max_temp 63C，OK，无 stop action。
- 队列状态：15:20 队列正确识别 `target_gpu=1` busy，并阻止重复启动 `VIVID_ansmask_resume_gpu1`；GPU0 相关任务仍为 Disabled，其余 GPU1 后续任务为 Ready。
- 最近 loss：local 130 step 附近 loss 约 `0.0324`，学习率约 `1.05e-05`；未见 traceback / RuntimeError / CUDA OOM / exitcode。日志中的 Flash attention fallback 是已知非致命降级，不是失败 case。当前没有新的失败案例需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- ETA：按 14:56 后本轮短窗速度估计，`step_8700.pt` 约 15-30 分钟，`step_9000.pt` 约 1-2.5 小时，`ansmask final.pt` 约 5-8.5 小时；考虑此前无 traceback 掉线和 Windows/数据加载波动，全目标剩余保守约 38-62 小时。
- 阶段性判断：当前训练、资源、GPU1-only 队列和 loss 水平均符合预期；短间隔保存已经把恢复点推进到 global 8600，之前反复退回 global 8500 的主要风险已降低。下一关键观察点是 `step_8700.pt` 是否按预期保存。

2026-05-31 15:34 续查（`step_8700.pt` 已验证）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上 VIVID compute PID 为 7108。GPU0 为空闲，GPU0 相关计划任务仍 Disabled，GPU1 后续任务为 Ready。
- `step_8700.pt` 已成功生成：文件时间 2026-05-31 15:33:26，大小约 1.072GB。只读 `torch.load(..., map_location='cpu')` 核验通过，内部 `global_step=8700`、`best_val_loss=0.037125732421875`，keys 为 `['best_val_loss', 'global_step', 'optimizer', 'projector', 'scheduler', 'vit']`。
- 当前训练已继续到约 local `204 / 1500`，按 `best.pt` 的可靠恢复点 `global_step=8500` 折算 global 约 `8704 / 10000`；距离下一个短间隔 checkpoint `step_8800.pt` 约 96 step，距离带验证的 `step_9000.pt` 约 296 step，距离 `ansmask final.pt` 约 1296 step。
- GPU 快照：15:34 左右 GPU1（bus `00000000:05:00.0`）约 16.8GB 显存、util 约 41%、功耗约 200.51W、温度 69C；GPU0（bus `00000000:01:00.0`）为 0MiB、util 0%、约 10.93W、43C。15:30 资源守卫记录 total_power 约 168.7W、max_temp 65C，OK，无 stop action，低于 350-400W 功耗边界和 83C 温度硬阈值。
- 队列状态：15:25、15:30 队列均正确识别 `target_gpu=1` busy，并阻止重复启动；目标完成状态仍为 `ans_train_final=False; ans_lp_final=False; null_train_final=False; null_lp_final=False; cf_prefix_final=False; field_paraphrase_final=False; random_lm_train_final=False; random_lm_lp_final=False`。
- 最近 loss：local 200 step 保存 `step_8700.pt` 时 loss 约 `0.0281`，学习率约 `1.05e-05`；未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败案例需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- ETA：按 14:56 到 15:34 的实际速度估计，`step_8800.pt` 约 15-25 分钟，`step_9000.pt` 约 0.9-1.8 小时（含 global 9000 验证/保存开销），`ansmask final.pt` 约 4.5-7.5 小时；考虑此前无 traceback 掉线和后续 7 项目标，全目标剩余保守约 36-60 小时。
- 阶段性判断：当前训练、资源状态、短间隔 checkpoint 策略和 loss 水平均符合预期；恢复点已推进到 global 8700，掉线回退风险进一步降低。下一关键观察点是 `step_8800.pt` 与 global 9000 验证。

2026-05-31 15:52 续查（`step_8800.pt` 已验证）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；GPU1 上 VIVID compute PID 仍为 7108。物理 GPU1 为 bus `00000000:05:00.0`，GPU0 为 bus `00000000:01:00.0` 且空闲，继续符合“只用 gpu1”的要求。
- `step_8800.pt` 已成功生成：文件时间 2026-05-31 15:51:00，大小约 1.072GB。只读 `torch.load(..., map_location='cpu')` 核验通过，内部 `global_step=8800`、`best_val_loss=0.037125732421875`，keys 为 `['best_val_loss', 'global_step', 'optimizer', 'projector', 'scheduler', 'vit']`。
- 当前训练已继续到约 local `306 / 1500`，按 `best.pt` 的可靠恢复点 `global_step=8500` 折算 global 约 `8806 / 10000`；距离下一个短间隔 checkpoint `step_8900.pt` 约 94 step，距离带验证的 `step_9000.pt` 约 194 step，距离 `ansmask final.pt` 约 1194 step。
- GPU 快照：15:51 左右 GPU1 约 16.8GB 显存、瞬时 util 0%、功耗约 109.50W、温度 61C；GPU0 为 0MiB、util 0%、约 8.97W、40C。15:50 资源守卫记录 total_power 约 185.49W、max_temp 67C，OK，无 stop action，低于 350-400W 功耗边界和 83C 温度硬阈值。
- 队列状态：15:35、15:40、15:45、15:50 队列均正确识别 `target_gpu=1` busy，并阻止重复启动；目标完成状态仍为 `ans_train_final=False; ans_lp_final=False; null_train_final=False; null_lp_final=False; cf_prefix_final=False; field_paraphrase_final=False; random_lm_train_final=False; random_lm_lp_final=False`。
- 最近 loss：local 300 step 保存 `step_8800.pt` 时 loss 约 `0.0352`，local 270 step 曾到约 `0.0275`，学习率约 `1.04e-05`；未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败案例需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- ETA：按 14:56 到 15:52 的实际速度估计，`step_8900.pt` 约 15-25 分钟，`step_9000.pt` 约 35-75 分钟（含 global 9000 验证/保存开销），`ansmask final.pt` 约 4-7 小时；考虑此前无 traceback 掉线和后续 7 项目标，全目标剩余保守约 35-58 小时。
- 阶段性判断：当前训练、资源状态、GPU1-only 队列、短间隔 checkpoint 策略和 loss 水平均符合预期；恢复点已推进到 global 8800，掉线回退风险继续降低。下一关键观察点是 `step_8900.pt`，再下一关键观察点是 global 9000 验证与 `step_9000.pt` 元数据核验。

2026-05-31 16:42 续查（`step_8900.pt`、`step_9000.pt` 已验证）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；VIVID 训练 PID 仍为 7108，并仍在物理 GPU1（bus `00000000:05:00.0`）。所有 VIVID 的 GPU0 计划任务仍为 Disabled。
- `step_8900.pt` 已成功生成：文件时间 2026-05-31 16:11:20，大小约 1.072GB。只读 `torch.load(..., map_location='cpu')` 核验通过，内部 `global_step=8900`、`best_val_loss=0.037125732421875`，keys 为 `['best_val_loss', 'global_step', 'optimizer', 'projector', 'scheduler', 'vit']`。
- `step_9000.pt` 已成功生成：文件时间 2026-05-31 16:38:21，大小约 1.072GB。只读 `torch.load(..., map_location='cpu')` 核验通过，内部 `global_step=9000`、`best_val_loss=0.037125732421875`，keys 同样完整。`best.pt` 未刷新，仍为 2026-05-31 08:56 的 global 8500 / `best_val_loss=0.037125732421875`。
- global 9000 验证已完成：日志显示 `Step 9000: val_loss = 0.0382`。该值略高于当前 best `0.0371257`，所以不刷新 best 是符合预期的；从数值看仍在同一低损失区间，不是失败信号。
- 当前训练已继续到约 local `504 / 1500`，折算 global 约 `9004 / 10000`；距离下一个短间隔 checkpoint `step_9100.pt` 约 96 step，距离 `ansmask final.pt` 约 996 step。
- GPU 快照：16:39 左右 VIVID PID 7108 在 GPU1 上占约 18.3GB 显存，温度约 55C；16:35 资源守卫记录 VIVID 目标 GPU1 为 OK，无 stop action。16:39 起 GPU0 上出现另一个 Python PID 320，占约 14.8GB，但命令行为 `scripts/run_local_hf_smoke.py --model-path H:\Xiyao_Wang\001_models\Qwen2.5-Coder-7B-Instruct ...`；当前 VIVID repo 内不存在该脚本、输入或输出文件，且所有 VIVID GPU0 计划任务均 disabled，因此判定为非当前 VIVID 目标进程，未做终止操作。
- 队列状态：16:00、16:05、16:10、16:15、16:20、16:25、16:30、16:35 队列均正确识别 `target_gpu=1` busy，并阻止重复启动；目标完成状态仍为 `ans_train_final=False; ans_lp_final=False; null_train_final=False; null_lp_final=False; cf_prefix_final=False; field_paraphrase_final=False; random_lm_train_final=False; random_lm_lp_final=False`。
- 最近 loss：global 9000 前后训练 loss 约 `0.0319-0.0352`，学习率约 `1.03e-05`；未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败案例需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- ETA：按 14:56 到 16:42 的实际速度和 global 9000 验证开销估计，`step_9100.pt` 约 20-35 分钟，`ansmask final.pt` 约 3.5-6 小时；考虑后续 7 项目标和可能的验证/队列切换开销，全目标剩余保守约 34-56 小时。
- 阶段性判断：当前训练、验证、checkpoint、资源守卫和 GPU1-only 的 VIVID 队列均符合预期；`step_9000.pt` 已把可靠恢复点推进到 global 9000，之前掉线回退风险显著降低。下一关键观察点是 `step_9100.pt`，中期观察点是 `final.pt` 生成后自动切到 `lp_ums_ansmask`。

2026-05-31 17:00 续查（`step_9100.pt` 已验证）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；VIVID 训练 PID 仍为 7108，并仍在物理 GPU1（bus `00000000:05:00.0`）。所有 VIVID 的 GPU0 计划任务仍为 Disabled。
- `step_9100.pt` 已成功生成：文件时间 2026-05-31 16:58:41，大小约 1.072GB。只读 `torch.load(..., map_location='cpu')` 核验通过，内部 `global_step=9100`、`best_val_loss=0.037125732421875`，keys 为 `['best_val_loss', 'global_step', 'optimizer', 'projector', 'scheduler', 'vit']`。
- 当前训练已继续到约 local `604 / 1500`，折算 global 约 `9104 / 10000`；距离下一个短间隔 checkpoint `step_9200.pt` 约 96 step，距离 `ansmask final.pt` 约 896 step。
- GPU 快照：17:00 左右 VIVID PID 7108 在 GPU1 上占约 18.3GB 显存，util 约 99%，功耗约 83.65W，温度约 58C；资源低于 350-400W 功耗边界和 83C 温度硬阈值。GPU0 上另有非本项目 Python PID 2160，占约 8.8GB，命令行为 `scripts/run_local_hf_smoke.py --model-path H:\Xiyao_Wang\001_models\Qwen3.5-4B ... --out H:\Xiyao_Wang\034_HIOAPD\outputs\qwen35_4b_gpu0_eval25_seed6026.jsonl`，判定为外部 `034_HIOAPD` 任务，不是当前 VIVID 目标进程，未做终止操作。
- 队列状态：16:40、16:45、16:50、16:55 队列均正确识别 `target_gpu=1` busy，并阻止重复启动；目标完成状态仍为 `ans_train_final=False; ans_lp_final=False; null_train_final=False; null_lp_final=False; cf_prefix_final=False; field_paraphrase_final=False; random_lm_train_final=False; random_lm_lp_final=False`。
- 最近 loss：`step_9100.pt` 保存时 local 600 step loss 约 `0.0338`，保存前短窗 loss 曾到约 `0.0310`；学习率约 `1.02e-05`。未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败案例需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- ETA：按 14:56 到 17:00 的实际速度和 global 9000 验证开销估计，`step_9200.pt` 约 20-35 分钟，`ansmask final.pt` 约 3-5.5 小时；考虑后续 7 项目标和可能的验证/队列切换开销，全目标剩余保守约 33-55 小时。
- 阶段性判断：当前训练、checkpoint、资源守卫和 VIVID 的 GPU1-only 队列均符合预期；可靠恢复点已推进到 global 9100。下一关键观察点是 `step_9200.pt`，中期观察点仍是 `final.pt` 生成后自动切到 `lp_ums_ansmask`。

2026-05-31 17:34 续查（`step_9200.pt` 已验证）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；VIVID 训练 PID 仍为 7108，并仍在物理 GPU1（bus `00000000:05:00.0`）。所有 VIVID 的 GPU0 计划任务仍为 Disabled，继续符合“只用 gpu1”跑当前 VIVID 目标的要求。
- `step_9200.pt` 已成功生成：文件时间 2026-05-31 17:31:36，大小约 1.072GB。只读 `torch.load(..., map_location='cpu')` 核验通过，内部 `global_step=9200`、`best_val_loss=0.037125732421875`，keys 为 `['best_val_loss', 'global_step', 'optimizer', 'projector', 'scheduler', 'vit']`。
- 当前训练已继续到约 local `701 / 1500`，折算 global 约 `9201 / 10000`；距离下一个短间隔 checkpoint `step_9300.pt` 约 99 step，距离 `ansmask final.pt` 约 799 step。
- GPU 快照：17:32 左右 VIVID PID 7108 在 GPU1 上占约 18.3GB 显存，util 约 39%，功耗约 92.29W，温度约 57C。17:30 资源守卫记录 total_power 约 183.37W、max_temp 60C，低于 350-400W 功耗边界和 83C 温度硬阈值，无 stop action。GPU0 上另有非本项目 Python PID 11712，占约 8.7GB，命令行为 `scripts/run_local_hf_smoke.py --model-path H:\Xiyao_Wang\001_models\Qwen3.5-4B ... --out H:\Xiyao_Wang\034_HIOAPD\outputs\qwen35_4b_gpu0_eval_offset10_limit15_stream_seed6026.jsonl`，判定为外部 `034_HIOAPD` 任务，不是当前 VIVID 目标进程，未做终止操作。
- 队列状态：17:10、17:15、17:20、17:25、17:30 队列均正确识别 `target_gpu=1` busy，并阻止重复启动；目标完成状态仍为 `ans_train_final=False; ans_lp_final=False; null_train_final=False; null_lp_final=False; cf_prefix_final=False; field_paraphrase_final=False; random_lm_train_final=False; random_lm_lp_final=False`。
- 最近 loss：`step_9200.pt` 保存时 local 700 step loss 约 `0.0336`，保存前短窗 loss 约 `0.0284-0.0313`；学习率约 `1.02e-05`。未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败案例需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- ETA：按 `step_9100.pt` 到 `step_9200.pt` 的最近 100-step 慢速窗口估计，`step_9300.pt` 约 25-45 分钟，`ansmask final.pt` 约 4-6.5 小时；若回到 14:56 后的平均速度会更快。考虑后续 7 项目标和可能的验证/队列切换开销，全目标剩余保守约 32-54 小时。
- 阶段性判断：当前训练、checkpoint、资源守卫和 VIVID 的 GPU1-only 队列均符合预期；可靠恢复点已推进到 global 9200。下一关键观察点是 `step_9300.pt`，中期观察点仍是 `final.pt` 生成后自动切到 `lp_ums_ansmask`。

2026-05-31 18:05 小时同步（`step_9300.pt` 已验证）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；VIVID 训练 PID 仍为 7108，并仍在物理 GPU1（bus `00000000:05:00.0`）。所有 VIVID 的 GPU0 计划任务仍为 Disabled。
- `step_9300.pt` 已成功生成：文件时间 2026-05-31 18:03:26，大小约 1.072GB。只读 `torch.load(..., map_location='cpu')` 核验通过，内部 `global_step=9300`、`best_val_loss=0.037125732421875`，keys 为 `['best_val_loss', 'global_step', 'optimizer', 'projector', 'scheduler', 'vit']`。
- 当前训练已继续到约 local `801 / 1500`，折算 global 约 `9301 / 10000`；距离下一个短间隔 checkpoint `step_9400.pt` 约 99 step，距离 `ansmask final.pt` 约 699 step。
- GPU 快照：18:04 左右 GPU0 空闲（0MiB、util 0%）；VIVID PID 7108 在 GPU1 上占约 18.3GB 显存，util 约 64%，功耗约 75.93W，温度约 56C。18:00 资源守卫记录 total_power 约 174.69W、max_temp 64C，低于 350-400W 功耗边界和 83C 温度硬阈值，无 stop action。
- 队列状态：17:55、18:00 队列均正确识别 `target_gpu=1` busy，并阻止重复启动；目标完成状态仍为 `ans_train_final=False; ans_lp_final=False; null_train_final=False; null_lp_final=False; cf_prefix_final=False; field_paraphrase_final=False; random_lm_train_final=False; random_lm_lp_final=False`。
- 最近 loss：`step_9300.pt` 保存时 local 800 step loss 约 `0.0322`，保存前短窗 loss 约 `0.0335-0.0344`；学习率约 `1.01e-05`。未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败案例需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- ETA：按 `step_9200.pt` 到 `step_9300.pt` 的最近 100-step 窗口估计，`step_9400.pt` 约 25-40 分钟，`ansmask final.pt` 约 3.5-5.5 小时；考虑后续 7 项目标和可能的验证/队列切换开销，全目标剩余保守约 31-53 小时。
- 阶段性判断：当前训练、checkpoint、资源守卫和 VIVID 的 GPU1-only 队列均符合预期；可靠恢复点已推进到 global 9300。下一关键观察点是 `step_9400.pt`，中期观察点仍是 `final.pt` 生成后自动切到 `lp_ums_ansmask`。

2026-05-31 18:38 续查（`step_9400.pt` 已验证）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；VIVID 训练 PID 仍为 7108，并仍在物理 GPU1（bus `00000000:05:00.0`）。所有 VIVID 的 GPU0 计划任务仍为 Disabled；GPU0 在本轮快照中为空闲（0MiB、util 0%）。
- `step_9400.pt` 已成功生成：文件时间 2026-05-31 18:36:02，大小约 1.072GB。只读 `torch.load(..., map_location='cpu')` 核验通过，内部 `global_step=9400`、`best_val_loss=0.037125732421875`，keys 为 `['best_val_loss', 'global_step', 'optimizer', 'projector', 'scheduler', 'vit']`。
- 当前训练已继续到约 local `901 / 1500`，折算 global 约 `9401 / 10000`；距离下一个短间隔 checkpoint `step_9500.pt` 约 99 step，距离 `ansmask final.pt` 约 599 step。
- GPU 快照：18:36 左右 VIVID PID 7108 在 GPU1 上占约 18.3GB 显存，功耗约 48.40W，温度约 52C；保存 checkpoint 后瞬时 util 为 0%，但计划任务仍 Running 且训练日志已继续到 local 901。18:35 资源守卫记录 total_power 约 194.11W、max_temp 64C，低于 350-400W 功耗边界和 83C 温度硬阈值，无 stop action。
- 队列状态：18:20、18:25、18:30、18:35 队列均正确识别 `target_gpu=1` busy，并阻止重复启动；目标完成状态仍为 `ans_train_final=False; ans_lp_final=False; null_train_final=False; null_lp_final=False; cf_prefix_final=False; field_paraphrase_final=False; random_lm_train_final=False; random_lm_lp_final=False`。
- 最近 loss：`step_9400.pt` 保存时 local 900 step loss 约 `0.0372`，保存前短窗 loss 约 `0.0312-0.0349`；学习率约 `1.01e-05`。未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败案例需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- ETA：按 `step_9300.pt` 到 `step_9400.pt` 的最近 100-step 窗口估计，`step_9500.pt` 约 25-45 分钟，`ansmask final.pt` 约 3-5 小时；考虑后续 7 项目标和可能的验证/队列切换开销，全目标剩余保守约 30-52 小时。
- 阶段性判断：当前训练、checkpoint、资源守卫和 VIVID 的 GPU1-only 队列均符合预期；可靠恢复点已推进到 global 9400。下一关键观察点是 `step_9500.pt`，中期观察点仍是 `final.pt` 生成后自动切到 `lp_ums_ansmask`。

2026-05-31 19:30 小时同步（`step_9500.pt` 已验证）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；VIVID 训练 PID 仍为 7108，并仍在物理 GPU1（bus `00000000:05:00.0`）。所有 VIVID 的 GPU0 计划任务仍为 Disabled。
- `step_9500.pt` 已成功生成并确认写入完成：文件时间 2026-05-31 19:27:36，大小约 1.072GB。只读 `torch.load(..., map_location='cpu')` 核验通过，内部 `global_step=9500`、`best_val_loss=0.037125732421875`，keys 为 `['best_val_loss', 'global_step', 'optimizer', 'projector', 'scheduler', 'vit']`。
- global 9500 验证已完成：日志显示 500/500 validation 耗时约 11:46，`Step 9500: val_loss = 0.0376`。该值略高于当前 best `0.0371257`，所以不刷新 `best.pt` 是符合预期的；数值仍在低损失区间，不是失败信号。
- 当前训练已继续到约 local `1001 / 1500`，折算 global 约 `9501 / 10000`；距离下一个短间隔 checkpoint `step_9600.pt` 约 99 step，距离 `ansmask final.pt` 约 499 step。
- GPU 快照：19:28 左右 VIVID PID 7108 在 GPU1 上占约 18.3GB 显存；保存/验证结束后瞬时 util 为 0%，但计划任务仍 Running 且日志已继续到 local 1001。GPU0 上另有非本项目 Python PID 6908，占约 8.8GB，命令行为 `scripts/run_local_hf_smoke.py --model-path H:\Xiyao_Wang\001_models\Qwen3.5-4B ... --out H:\Xiyao_Wang\034_HIOAPD\outputs\qwen35_4b_gpu0_eval_offset25_limit500_stream_seed6026.jsonl`，判定为外部 `034_HIOAPD` 任务，不是当前 VIVID 目标进程，未做终止操作。
- 队列/资源守卫：19:15、19:20、19:25 队列均正确识别 `target_gpu=1` busy 或总功耗超过启动阈值，因此阻止重复启动；资源守卫记录 max_temp 最高约 58C、total_power 最高约 256.25W，低于 350-400W 功耗边界和 83C 温度硬阈值，无 stop action。
- 最近 loss：`step_9500.pt` 保存前训练 loss 约 `0.0319-0.0376`，学习率约 `1.01e-05`。未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败案例需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- ETA：按 `step_9400.pt` 到 `step_9500.pt` 的最近 100-step 窗口和 9500 验证开销估计，`step_9600.pt` 约 25-45 分钟，`ansmask final.pt` 约 2.5-4.5 小时；考虑后续 7 项目标和可能的验证/队列切换开销，全目标剩余保守约 30-51 小时。
- 阶段性判断：当前训练、验证、checkpoint、资源守卫和 VIVID 的 GPU1-only 队列均符合预期；可靠恢复点已推进到 global 9500。下一关键观察点是 `step_9600.pt`，中期观察点是 `final.pt` 生成后自动切到 `lp_ums_ansmask`。

2026-05-31 20:05 小时同步（`step_9600.pt` 已验证）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；VIVID 训练 PID 仍为 7108，并仍在物理 GPU1（bus `00000000:05:00.0`）。所有 VIVID 的 GPU0 计划任务仍为 Disabled。
- `step_9600.pt` 已成功生成：文件时间 2026-05-31 20:01:42，大小约 1.072GB。只读 `torch.load(..., map_location='cpu')` 核验通过，内部 `global_step=9600`、`best_val_loss=0.037125732421875`，keys 为 `['best_val_loss', 'global_step', 'optimizer', 'projector', 'scheduler', 'vit']`。
- 当前训练已继续到约 local `1101 / 1500`，折算 global 约 `9601 / 10000`；距离下一个短间隔 checkpoint `step_9700.pt` 约 99 step，距离 `ansmask final.pt` 约 399 step。
- GPU 快照：20:02 左右 VIVID PID 7108 在 GPU1 上占约 18.3GB 显存，util 约 99%，温度约 57C。GPU0 上仍有非本项目 Python PID 6908，占约 8.8GB，判定为外部 `034_HIOAPD` 任务，不是当前 VIVID 目标进程，未做终止操作。
- 队列/资源守卫：19:50、19:55、20:00 队列均正确识别 `target_gpu=1` busy 或总功耗超过启动阈值，因此阻止重复启动；资源守卫记录 max_temp 最高约 62C、total_power 最高约 247.07W，低于 350-400W 功耗边界和 83C 温度硬阈值，无 stop action。
- 最近 loss：`step_9600.pt` 保存时 local 1100 step loss 约 `0.0321`，保存前短窗 loss 约 `0.0326-0.0357`；学习率约 `1.00e-05`。未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败案例需要 case study。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- ETA：按 `step_9500.pt` 到 `step_9600.pt` 的最近 100-step 窗口估计，`step_9700.pt` 约 25-45 分钟，`ansmask final.pt` 约 2-4 小时；考虑后续 7 项目标和可能的验证/队列切换开销，全目标剩余保守约 29-50 小时。
- 阶段性判断：当前训练、checkpoint、资源守卫和 VIVID 的 GPU1-only 队列均符合预期；可靠恢复点已推进到 global 9600。下一关键观察点是 `step_9700.pt`，中期观察点是 `final.pt` 生成后自动切到 `lp_ums_ansmask`。

2026-05-31 21:05 小时同步（`step_9700.pt` 未到，慢速窗口复查）：
- `VIVID_ansmask_resume_gpu1` 仍为 Running；计划任务 `schtasks /Query` 显示状态为 Running，最近启动时间仍为 2026-05-31 14:56:10。VIVID 训练 PID 仍为 7108，并仍在物理 GPU1（bus `00000000:05:00.0`）。所有 VIVID 的 GPU0 计划任务仍为 Disabled。
- 当前最新可靠 checkpoint 仍为 `step_9600.pt`：文件时间 2026-05-31 20:01:42，已在上一轮只读 `torch.load(..., map_location='cpu')` 核验通过，内部 `global_step=9600` 且 keys 完整。`step_9700.pt` 截至 21:02 尚未生成。
- 当前训练已继续到约 local `1155 / 1500`，折算 global 约 `9655 / 10000`；距离下一个短间隔 checkpoint `step_9700.pt` 约 45 step，距离 `ansmask final.pt` 约 345 step。
- 慢速窗口：20:03 之后训练从约 13-25s/step 降到约 60-110s/step；21:02 日志尾部仍在持续推进（local 1143 -> 1155），未见长时间完全停住。该现象影响 ETA，但还不能判为失败 case。
- GPU/资源证据：21:02 左右 GPU1 占约 18.3GB 显存、温度约 47C、瞬时 util 约 0%，但 PID 7108 仍为 compute app；GPU0 上仍有非本项目 Python PID 6908 和 `GameViewerServer` PID 11812，占用约 8.8GB。20:30-20:45 资源守卫均为 OK，无 stop action，max_temp 最高约 56C、total_power 最高约 238.93W，低于 350-400W 功耗边界和 83C 温度硬阈值。
- 队列状态：20:30、20:35、20:40、20:45 队列均正确识别 `target_gpu=1` busy，并阻止重复启动；目标完成状态仍为 `ans_train_final=False; ans_lp_final=False; null_train_final=False; null_lp_final=False; cf_prefix_final=False; field_paraphrase_final=False; random_lm_train_final=False; random_lm_lp_final=False`。
- 最近 loss：20:45-21:02 慢速窗口中 loss 约 `0.0289-0.0308`，学习率约 `1.00e-05`。未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有新的失败案例需要 case study；这是吞吐异常/系统调度观察点。
- 当前剩余目标产物仍为 8 项：`ansmask final`、`ansmask LP`、`null-as-negative final/LP`、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`。
- ETA：若当前 60-110s/step 慢速窗口持续，`step_9700.pt` 约还需 45-85 分钟，`ansmask final.pt` 约还需 6-11 小时；若恢复到 19:30-20:00 的速度，则 `step_9700.pt` 约 15-30 分钟、`ansmask final.pt` 约 2-4 小时。考虑后续 7 项目标和队列切换开销，全目标剩余保守上调为约 34-60 小时。
- 阶段性判断：结果质量和 checkpoint 可靠性仍符合预期，训练没有崩溃；但当前吞吐不符合 20:05 时的 ETA，需要继续观察 `step_9700.pt` 是否在慢速窗口内顺利保存。下一关键观察点是 `step_9700.pt`，中期观察点仍是 `final.pt` 生成后自动切到 `lp_ums_ansmask`。

2026-05-31 21:18 目标收口（只跑完当前 `ansmask` 后停止）：
- 用户最新要求：当前只把 `ansmask` 训练跑到完成，然后停止，不再继续自动跑后续 LP / null-as-negative / random-LM / field-paraphrase / counterfactual eval。
- 已更新 `scripts/answerability_gpu1_queue_once.ps1`：新增 `$StopAfterAnsmaskFinal = $true`，并在检测到 `outputs/ablation_ums_ansmask_12label/checkpoints/final.pt` 后只写日志 `stop-after-ansmask enabled`，不再启动 `VIVID_lp_ansmask_gpu1` 或后续任务。
- 当前正在运行的 `VIVID_ansmask_resume_gpu1` 不受影响，继续只用 GPU1 跑到 `final.pt`；后续完成后的验收标准改为：`final.pt` 生成并可被只读 `torch.load(..., map_location='cpu')` 验证，队列不再启动下一项任务。
- 截至 21:18，`step_9700.pt` 尚未生成，训练日志已推进到约 local `1169 / 1500`，折算 global 约 `9669 / 10000`；距离 `step_9700.pt` 约 31 step，距离 `ansmask final.pt` 约 331 step。当前目标剩余从“8 项产物”收口为“只剩 `ansmask final.pt` 这一项需要完成并停止”。

2026-05-31 21:36 续查（`step_9700.pt` 已验证，进入最后 300 step）：
- `step_9700.pt` 已成功生成：文件时间 2026-05-31 21:33:19，大小约 1.072GB。只读 `torch.load(..., map_location='cpu')` 核验通过，内部 `global_step=9700`、`best_val_loss=0.037125732421875`，keys 为 `['best_val_loss', 'global_step', 'optimizer', 'projector', 'scheduler', 'vit']`。
- 当前训练已继续到约 local `1205 / 1500`，折算 global 约 `9705 / 10000`；距离 `ansmask final.pt` 约 295 step。按最新要求，`ansmask final.pt` 完成并验证后即停止，不再自动跑后续实验。
- 慢速窗口已有缓解：`step_9700.pt` 保存前后速度从 60-110s/step 回落到约 16-19s/step。GPU1 仍为 VIVID PID 7108，占约 18.3GB 显存，温度约 58C；GPU0 上外部 PID 6908 仍非 VIVID 目标进程。
- 最近 loss：`step_9700.pt` 保存时 local 1200 step loss 约 `0.0332`，保存前短窗 loss 约 `0.0298-0.0384`；学习率约 `1.00e-05`。未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有失败 case。
- ETA：若维持恢复后的 16-25s/step，`ansmask final.pt` 约 1.5-2.5 小时；若慢速窗口再次出现，则约 3-6 小时。检查频次已按接近 final 调整：先每 5-8 分钟看一次，到 `step_9800.pt` 之后进一步缩短到 20-30 秒。

2026-05-31 22:02 续查（`step_9800.pt` 已验证，进入最后约 200 step）：
- `step_9800.pt` 已成功生成：文件时间 2026-05-31 21:59:12，大小约 1.072GB。只读 `torch.load(..., map_location='cpu')` 核验通过，内部 `global_step=9800`、`best_val_loss=0.037125732421875`，keys 为 `['best_val_loss', 'global_step', 'optimizer', 'projector', 'scheduler', 'vit']`。
- 当前训练仍由 `VIVID_ansmask_resume_gpu1` 承载，计划任务状态为 Running，VIVID 训练 PID 仍为 7108，并仍在物理 GPU1（bus `00000000:05:00.0`）上占约 18.3GB 显存。GPU0 上的非 VIVID 外部任务未做干预。
- 日志已继续到约 local `1306 / 1500`，折算 global 约 `9806 / 10000`；距离 `ansmask final.pt` 约 194 step。按最新收口目标，当前只剩 `ansmask final.pt` 这一项，完成并验证后即停止，不启动 LP 或其他后续实验。
- 最近速度已明显恢复：`step_9800.pt` 保存前后约 13-18s/step；保存时 local 1300 step loss 约 `0.0352`，保存后短窗 loss 继续在约 `0.0316-0.0352` 区间，学习率约 `1.00e-05`。未见 traceback / RuntimeError / CUDA OOM / exitcode，当前没有失败 case 需要 case study。
- ETA：若维持当前 13-18s/step，距离 `final.pt` 约 45-70 分钟；考虑 global 10000 处可能有最终验证/保存开销，保守按约 1-1.5 小时看。若慢速窗口再次出现，则可能延长到约 2-3 小时。检查频次已切到更密集模式：先盯 `step_9900.pt`，之后对 `final.pt` 做 20-30 秒级轮询并等待文件大小稳定后再加载验证。

2026-05-31 22:30 续查（`step_9900.pt` 已验证，进入最后约 100 step）：
- `step_9900.pt` 已成功生成：文件时间 2026-05-31 22:27:45，大小约 1.072GB。只读 `torch.load(..., map_location='cpu')` 核验通过，内部 `global_step=9900`、`best_val_loss=0.037125732421875`，keys 为 `['best_val_loss', 'global_step', 'optimizer', 'projector', 'scheduler', 'vit']`。
- 当前训练仍由 `VIVID_ansmask_resume_gpu1` 承载，计划任务状态为 Running，VIVID PID 7108 仍在物理 GPU1（bus `00000000:05:00.0`）上运行；GPU1 快照约 18.3GB 显存、util 约 95%、温度约 59C。
- 日志已继续到约 local `1402 / 1500`，折算 global 约 `9902 / 10000`；距离 `ansmask final.pt` 约 98 step。按最新收口目标，仍只等待 `final.pt`，完成并验证后停止，不启动 LP 或其他后续实验。
- 最近 loss：`step_9900.pt` 保存时 local 1400 step loss 约 `0.0355`，保存前短窗 loss 约 `0.0316-0.0355`，学习率约 `1.00e-05`。未见 traceback / RuntimeError / CUDA OOM / exitcode，当前没有失败 case。
- ETA：按最近 15-22s/step，最后 98 step 约 25-40 分钟；若结尾有验证/保存开销，保守约 40-70 分钟。检查频次已调整为 `final.pt` 20-30 秒级轮询，并会先等待文件大小稳定再做只读加载验证。

2026-06-01 00:12 最终同步（当前收口目标 `ansmask final.pt` 已完成并停止）：
- `ansmask` 预训练已完成。最终产物为 `outputs/ablation_ums_ansmask_12label/checkpoints/final.pt`，文件时间 2026-06-01 00:08:01，大小约 1.072GB；同时生成 `step_10000.pt`，文件时间 2026-06-01 00:07:53，大小约 1.072GB。
- 只读 `torch.load(..., map_location='cpu')` 核验 `final.pt` 通过：内部 `global_step=10000`、`best_val_loss=0.037125732421875`，keys 为 `['best_val_loss', 'global_step', 'optimizer', 'projector', 'scheduler', 'vit']`。
- 训练日志显示最终验证完成：`Step 10000: val_loss = 0.0415`，随后保存 `step_10000.pt` 和 `final.pt`，并输出 `Training completed!`；计划任务 `VIVID_ansmask_resume_gpu1` 已回到 Ready，Last Result 为 0。
- 当前 GPU 状态：物理 GPU1（bus `00000000:05:00.0`）已空闲，0MiB 显存、util 0%；GPU0 上仍有外部非 VIVID 任务，占约 8.7GB 显存，未做干预。
- 队列停止验证：00:10 的 `VIVID_answerability_queue_once` 已识别 `ans_train_final=True`，并写入 `stop-after-ansmask enabled: ansmask final exists; not launching LP or downstream experiments`。`VIVID_lp_ansmask_gpu1` 仍为 Ready，Last Run Time 仍是 1999-11-30，说明没有启动 LP 或后续实验。
- 当前用户收口目标已经完成：只跑完当前 `ansmask` 并停止。后续 LP / null-as-negative / random-LM / field-paraphrase / counterfactual eval 按最新要求未继续运行。

Case study：23:00-23:45 末段中断与自动恢复
- 现象：第一轮在 `step_9900.pt` 后继续推进到约 local 1467/1500，但截至 23:45 仍未生成 `final.pt`；队列在 23:45 看到 GPU1 空闲且 `ans_train_final=False`，自动重新启动 `VIVID_ansmask_resume_gpu1`。
- 证据：`answerability_queue_once.log` 在 23:45 记录 `target_gpu=1 util=0%, mem=0MiB` 并执行 `launching VIVID_ansmask_resume_gpu1`；新进程命令行为 `--resume ...\step_9900.pt`。训练主日志未见 traceback / RuntimeError / CUDA OOM / KeyboardInterrupt。
- 影响：没有损坏 checkpoint，可靠恢复点仍为已验证的 `step_9900.pt`；实际多跑了最后 100 step 的一部分，增加约 20 分钟左右墙钟时间，但最终 `final.pt` 与 `step_10000.pt` 均成功生成并通过只读加载验证。
- 处理：未手动终止任何训练/外部进程；让队列从 `step_9900.pt` 自动恢复并完成。后续队列被 `StopAfterAnsmaskFinal` 拦截，没有启动 LP 或下游任务。

2026-06-01 00:18 LP 启动同步（按最新要求只跑 `lp_ums_ansmask` 后停止）：
- 已手动启动计划任务 `VIVID_lp_ansmask_gpu1`，启动时间 2026-06-01 00:15:37；命令为 `scripts/run_lp_ansmask_gpu1.cmd`，其中 `CUDA_VISIBLE_DEVICES=1`，只使用物理 GPU1（bus `00000000:05:00.0`）。
- 配置为 `configs/lp_ums_ansmask_12label.yaml`：冻结 ViT backbone，只训练线性分类头；`max_steps=3000`、`batch_size=16`、`gradient_accumulation_steps=2`、`eval_interval=200`、`save_interval=600`，输出目录为 `outputs/lp_ums_ansmask_12label`。
- 训练日志确认已加载 `outputs/ablation_ums_ansmask_12label/checkpoints/best.pt`，进入 linear probe mode，冻结 150 组 backbone 参数，仅训练 10,766 个 head 参数。
- 启动后约 00:18 已推进到约 `91 / 3000` step，GPU1 显存约 1.8GB，loss 从约 `0.4195` 降到约 `0.1370`，未见 traceback / RuntimeError / CUDA OOM。
- ETA：按当前 1-2 step/s 的早期吞吐，纯训练约 30-40 分钟；考虑每 200 step 一次验证与保存开销，保守估计本次 LP 总耗时约 0.5-1 小时，偏向 35-55 分钟。

2026-06-01 00:25 LP 首个验证点：
- `metrics_step_200.json` 已生成，日志记录 `Step 200: val_loss = 0.2973`；验证后训练继续推进到约 `231 / 3000` step。
- 第一段实际耗时包含模型加载、200 step 训练和一次 63-batch 验证；验证阶段约 2-3 分钟，明显是当前 LP 的主要额外开销。
- 当前判断：训练本身正常，loss 在 warmup 后已降至约 `0.12-0.15` 区间，未见 traceback / RuntimeError / CUDA OOM。按当前节奏，完整 LP 更现实的剩余 ETA 约 65-80 分钟，可能略超过 1 小时。

2026-06-01 00:53 LP 进度同步：
- `VIVID_lp_ansmask_gpu1` 仍为 Running；GPU1 上的 VIVID LP 进程为 `python.exe` PID 16764，显存约 1.9GB，仍在物理 GPU1（bus `00000000:05:00.0`）。
- 最新已落盘指标为 `outputs/lp_ums_ansmask_12label/metrics_step_1000.json`，日志记录 `Step 1000: val_loss = 0.2716`；同时 `best.pt` 与 `step_600.pt` 已生成。
- 日志尾部已推进到 `1200 / 3000` step 并进入下一轮验证。训练 loss 主要在约 `0.12-0.17` 区间波动，未见 traceback / RuntimeError / CUDA OOM。
- 实际节奏：从 00:15 启动到 00:53 约 38 分钟，完成约 40% 训练与多个验证点。按当前速度，剩余约 45-60 分钟；总耗时可能略超过最初 0.5-1 小时估计。

2026-06-01 01:25 LP 失败 case study 与重跑处理：
- 失败现象：第一次 `lp_ums_ansmask` 在约 `1647 / 3000` step 处退出，没有生成 `metrics_final.json`。最后可靠指标为 `metrics_step_1600.json`，日志记录 `Step 1600: val_loss = 0.2587`。
- 错误栈：`torch.utils.data.dataloader._shutdown_workers()` 在 Windows 下调用 worker `terminate()` 时触发 `PermissionError: [WinError 5] 拒绝访问`。未见 CUDA OOM、模型 forward/backward 错误或指标异常。
- 影响判断：这是 Windows Task Scheduler + 多进程 DataLoader worker 清理权限问题，不是 LP 模型训练本身失败。GPU1 已空闲，计划任务回到 Ready，但旧 batch 脚本因 `%ERRORLEVEL%` 在括号块内提前展开，显示 `exitcode 0`，不能作为成功证据。
- 处理：将失败产物和日志归档到 `history/20260601_0015_failed_dataloader_workers/`，避免与最终 rerun 混淆；将 `configs/lp_ums_ansmask_12label.yaml` 的 `num_workers` 改为 0，避免 Windows worker shutdown；修复 `scripts/run_lp_ansmask_gpu1.cmd`，使用 delayed expansion 捕获真实 exit code 并 `exit /b`。
- 下一步：从同一 `outputs/ablation_ums_ansmask_12label/checkpoints/best.pt` 重跑完整 LP，输出仍写回干净的 `outputs/lp_ums_ansmask_12label`。由于 `num_workers=0` 可能降低吞吐，重跑 ETA 按约 1-1.5 小时保守估计。

2026-06-01 01:30 LP 重跑启动确认：
- 已重新启动 `VIVID_lp_ansmask_gpu1`，启动时间约 2026-06-01 01:27:14；GPU1 上进程为 `python.exe` PID 2492，显存约 1.8GB。
- 新日志确认仍加载 `outputs/ablation_ums_ansmask_12label/checkpoints/best.pt`，进入 linear probe mode，并且使用干净输出目录 `outputs/lp_ums_ansmask_12label`。
- `num_workers=0` 后训练更稳定但吞吐下降，约 `40 / 3000` step 用 1 分钟多。当前保守 ETA 调整为约 1.5-2 小时。

2026-06-01 02:26 LP 最终完成同步（当前目标已完成并停止）：
- `lp_ums_ansmask` linear probe 已完整跑完。计划任务 `VIVID_lp_ansmask_gpu1` 已从 Running 回到 Ready，`Last Result=0`；GPU1（bus `00000000:05:00.0`）已空闲，显存 0MiB，未发现仍在 GPU1 上运行的 VIVID 进程。
- 最终产物已落盘：`outputs/lp_ums_ansmask_12label/final.pt`、`outputs/lp_ums_ansmask_12label/step_3000.pt`、`outputs/lp_ums_ansmask_12label/metrics_final.json`、`outputs/lp_ums_ansmask_12label/metrics_step_3000.json`。只读加载 `final.pt` 核验通过，checkpoint 内部 keys 为 `['model', 'step']`，`step=3000`。
- 最终指标：`val_loss=0.249564`、`macro_f1=0.912003`、`micro_f1=0.896418`、`macro_auc=0.817777`。训练日志记录 `Step 3000: val_loss = 0.2496`，随后输出 `Training completed!` 并写入真实 `exitcode 0`。
- 指标趋势符合预期：重跑过程中 `val_loss` 从 `step 1000 = 0.2732` 降到 `step 2000 = 0.2615`，最终到 `step 3000 = 0.2496`；没有出现 CUDA OOM、RuntimeError 或指标异常。
- 首次 LP 失败 case 已归档在 `history/20260601_0015_failed_dataloader_workers/`。结论保持不变：失败点是 Windows Task Scheduler 下多进程 DataLoader worker shutdown 触发 `PermissionError: [WinError 5]`，不是模型训练失败；处理方式为将 `num_workers` 改为 0，并修复 batch 脚本的真实 exit code 捕获。重跑已验证该处理有效。
- 当前用户收口目标完成：只把 `lp_ums_ansmask` 跑完并停止。没有启动 null-as-negative、random-LM、field-paraphrase、counterfactual eval 或其他后续实验。

2026-06-01 10:36 `null-as-negative` 10k 训练状态与中断 case study：
- 按用户要求，在 `lp_ums_ansmask` 完成后启动 `VIVID_null_as_negative_gpu1`，目标为 `configs/ablation_ums_null_as_negative_12label.yaml` 的 10k-step 训练，只绑定 GPU1（bus `00000000:05:00.0`）。
- 启动前处理：保留 `max_text_length=256` 以降低显存风险；先短暂试过 `batch_size=2 / gradient_accumulation_steps=16`，实测约 20-22s/step，会把总 ETA 拉长到 55h+，因此立即归档短跑日志到 `history/20260601_0231_null_as_negative_safe_start_abort/`，并恢复到原实验等效 batch 设置 `batch_size=4 / gradient_accumulation_steps=8`。
- batch4 版本从 2026-06-01 02:43:47 开始运行，实际推进到约 `3637 / 10000` step；最新可靠 checkpoint 为 `outputs/ablation_ums_null_as_negative_12label/checkpoints/step_3000.pt`，只读 `torch.load(..., map_location='cpu')` 核验通过：`global_step=3000`、`best_val_loss=0.028968754708766937`，keys 为 `['best_val_loss', 'global_step', 'optimizer', 'projector', 'scheduler', 'vit']`。
- 指标/结果判断：训练本身符合预期。日志记录 `Step 3500: val_loss = 0.0297`，训练 loss 主要在约 `0.025-0.032` 区间；没有看到模型 forward/backward 错误或 CUDA OOM。
- 中断现象：计划任务 `VIVID_null_as_negative_gpu1` 当前为 Ready，`Last Result=-1073741510`；日志尾部显示 `ERROR conda.cli.main_run...` 后出现 `^C Terminate batch job`。全局 `nvidia-smi` 查询报 `Unable to determine the device handle for GPU0000:01:00.0: GPU is lost. Reboot the system to recover this GPU`。
- GPU/CUDA 证据：`nvidia-smi -i 1` 仍可单独看到 GPU1，0MiB 显存且空闲；但在 `CUDA_VISIBLE_DEVICES=1` 或 GPU1 UUID 下运行 `torch.cuda.is_available()` 均返回 False，`torch.cuda.device_count()` 为 0，并报 `cudaGetDeviceCount ... Error 1: invalid argument`。因此当前主机 CUDA runtime 已不可用，不能继续 GPU 训练。
- 处理：中断日志已归档到 `history/20260601_0243_null_as_negative_interrupted_step3637/`；`scripts/run_null_as_negative_gpu1.cmd` 已改为启动时自动选择最新 checkpoint 并传入 `--resume`，后续恢复 CUDA 后可直接从最新 checkpoint 继续。
- 当前剩余：从可靠 checkpoint `step_3000.pt` 算还剩约 `7000` step。按 batch4 已观察到的 `step_1000/2000/3000` 保存节奏估计，CUDA 恢复后还需约 13-16 小时（包含后续验证/保存开销）。当前不能继续跑，必须先重启机器或恢复 NVIDIA 驱动/CUDA 状态。

2026-06-08 11:46 `null-as-negative` CUDA 恢复后续跑启动：
- CUDA/GPU 已恢复：`CUDA_VISIBLE_DEVICES=1` 下 `torch.cuda.is_available()` 为 True，设备为 RTX 3090；当前训练进程 PID 22512 只在物理 GPU1（bus `00000000:05:00.0`）上运行，GPU0 空闲，符合“只用 gpu1”约束。
- 恢复脚本修复：旧 `scripts/select_latest_checkpoint.ps1 -Default ""` 在 Windows PowerShell 下会把错误输出误传给 `--resume`；相关失败日志已归档到 `history/20260608_1136_null_as_negative_cuda_recovered_resume/` 和 `history/20260608_1137_null_as_negative_resume_script_quote_fail/`。当前 `scripts/run_null_as_negative_gpu1.cmd` 改为 batch 内部按修改时间选择最新 `step_*.pt`，必要时才回退 `best.pt`。
- 多智能体只读审计确认：当前 null-as-negative 脚本只暴露 GPU1，优先 `step_*.pt` 再兜底 `best.pt`，并且 `trainer.py` 会以 checkpoint 内的 `global_step=3000` 继续跑到 `max_steps=10000`。同时已将通用 `scripts/select_latest_checkpoint.ps1` 修正为优先选择最大数字的 `step_N.pt`，没有 step 时才使用 `best.pt`；在真实 checkpoint 目录上验证返回 `step_3000.pt`。该 helper 修复不影响已经启动的当前训练进程。
- 已通过计划任务 `VIVID_null_as_negative_gpu1` 重启训练，日志确认从 `outputs\ablation_ums_null_as_negative_12label\checkpoints\step_3000.pt` 加载，并显示 `Resuming from step 3000`、`Max steps: 10000`、`Output dir: outputs\ablation_ums_null_as_negative_12label`。
- 当前进度：恢复后约 `34 / 7000` step，折算 global 约 `3034 / 10000`；最新可靠 checkpoint 仍是 `step_3000.pt`，下一关键观察点是重新到达 global `3500` 验证点，下一可靠落盘 checkpoint 是 `step_4000.pt`。
- 结果判断：已有可靠 `best_val_loss=0.028968754708766937`，旧中断日志中 `Step 3500: val_loss = 0.0297`，当前恢复后训练 loss 约 `0.0256-0.0322`，未见 traceback / RuntimeError / CUDA OOM；结果质量仍符合预期。注意旧 `3500` 之后未形成可靠 checkpoint，本轮会从 `3000` 开始重跑这段。
- ETA：按当前纯训练 tqdm 约 3-4s/step，剩余纯训练约 6-8 小时；按 2026-06-01 已观察到的 checkpoint/验证/保存总墙钟节奏，完整剩余更保守约 13-16 小时。等 `step_4000.pt` 生成后再用真实 1000-step 间隔刷新 ETA。

2026-06-08 11:59 `null-as-negative` 温度与早期推进复查：
- 计划任务 `VIVID_null_as_negative_gpu1` 仍为 Running；VIVID 训练 PID 22512 仍只在物理 GPU1（bus `00000000:05:00.0`）上，GPU0 空闲。
- 资源守卫已启用，每 5 分钟执行一次；11:55 记录 GPU1 `temp=82C`、`power=327.6W`、总功耗 `338.36W`，低于硬停阈值 `hard_temp=83C` / `hard_power=400W`，因此未触发 stop action。11:59 手动快照 GPU1 已回落到 `78C`、`266.62W`。
- 训练日志已推进到恢复后约 `274 / 7000` step，折算 global 约 `3274 / 10000`；距离重新到达 global `3500` 验证点约 226 step，距离下一可靠 checkpoint `step_4000.pt` 约 726 step。
- 最近训练 loss 仍在约 `0.0245-0.0317` 区间；未见 traceback / RuntimeError / CUDA OOM / exitcode。当前没有失败 case，需要继续观察温度是否反复触及 83C。

Case study：2026-06-08 12:05 `null-as-negative` 温度守卫硬停
- 现象：资源守卫在 12:05 记录 GPU1 `temp=83C`、`power=314.46W`、`max_temp=83C`，达到 `hard_temp=83C`，因此执行 `schtasks /End` 终止 VIVID 计划任务；`VIVID_null_as_negative_gpu1` 回到 Ready，`Last Result=267014`。
- 影响：训练日志停在恢复后约 `380 / 7000` step，折算 global 约 `3380 / 10000`，尚未到 global `3500` 验证点，也没有生成 `step_4000.pt` 或新的指标文件。可靠恢复点仍是 `outputs/ablation_ums_null_as_negative_12label/checkpoints/step_3000.pt`，本轮 380 个未落盘 step 需要重跑。
- 判断：这是温度/资源守卫保护停机，不是模型 forward/backward 错误、CUDA OOM、数据错误或指标异常；停止前 loss 仍在约 `0.0256-0.0310` 区间。
- 处理：已将训练日志和资源守卫日志归档到 `history/20260608_1205_null_as_negative_temp_guard_stop/`，并新增 `case_study.md`。尝试用 `nvidia-smi -i 1 -pl 280` 降低 GPU1 power limit 失败，原因为权限不足。
- 恢复策略：新增 `configs/ablation_ums_null_as_negative_12label_thermal_resume_gpu1.yaml`，保持相同 output_dir 与 `max_steps=10000`，将 `batch_size=4 / gradient_accumulation_steps=8` 调整为 `batch_size=3 / gradient_accumulation_steps=11`，有效 batch 约 33，尽量贴近原 32，同时降低 micro-batch 热负载；`scripts/run_null_as_negative_gpu1.cmd` 已切换到该 thermal resume 配置。

2026-06-08 12:20 `null-as-negative` thermal resume 重启确认：
- 已重新启动计划任务 `VIVID_null_as_negative_gpu1`，启动时间 2026-06-08 12:18:39；当前 VIVID PID 为 11616，只在物理 GPU1（bus `00000000:05:00.0`）运行，GPU0 空闲。
- 日志确认使用 thermal 配置并再次从 `step_3000.pt` 恢复，显示 `Resuming from step 3000`、`Max steps: 10000`。
- thermal 配置生效后的早期快照：GPU1 显存约 `12499MiB`，低于原 batch4 运行时约 `15193MiB`；12:20 手动快照 GPU1 `temp=68C`、`power=300.52W`、util 49%。训练推进到恢复后约 `10 / 7000` step，早期 loss `0.0315`，暂无 traceback / RuntimeError / CUDA OOM。
- 当前剩余仍按可靠 checkpoint `step_3000.pt` 计算，需重跑 `3000-3500` 观察窗；下一关键检查点仍是 global `3500` 验证和 `step_4000.pt`。

2026-06-08 12:27 thermal resume 早期温度复查：
- `VIVID_null_as_negative_gpu1` 仍为 Running；VIVID PID 11616 仍在物理 GPU1，GPU0 空闲。
- 12:25 资源守卫记录 GPU1 `temp=81C`、`power=331.6W`、显存 `12499MiB`，低于 `hard_temp=83C`，未触发 stop action；12:27 手动快照 GPU1 `temp=79C`、`power=291.69W`、util 29%。
- 训练已推进到恢复后约 `143 / 7000` step，折算 global 约 `3143 / 10000`；最近 loss 约 `0.0258-0.0313`。暂无 traceback / RuntimeError / CUDA OOM。
- 判断：thermal 配置已降低显存，但温度仍接近硬阈值，需要继续观察 12:30/12:35 守卫；如果再次触及 83C，则改用更冷的 `batch_size=2 / gradient_accumulation_steps=16` 或暂停等待人工冷却/权限级 power limit。

2026-06-08 12:32 thermal resume 温度趋稳：
- 12:30 资源守卫记录 GPU1 `temp=73C`、`power=247.85W`、显存 `12499MiB`，未触发 stop action；12:32 手动快照 GPU1 `temp=71C`、`power=226.97W`。
- 训练继续推进到恢复后约 `203 / 7000` step，折算 global 约 `3203 / 10000`；最近 loss 约 `0.0248-0.0313`，未见 traceback / RuntimeError / CUDA OOM。
- 代价：thermal 配置在稳定后吞吐约 `4-5.5s/step`，比最初 batch4 快窗更慢，但温度从 83C 硬停区间回落到 70 多度。下一关键点为 global `3500` 验证，预计约 20-30 分钟内到达。

2026-06-08 12:53 `null-as-negative` global 3500 验证通过：
- global `3500` 验证已完成，日志记录 `Step 3500: val_loss = 0.0282`，并刷新 `outputs/ablation_ums_null_as_negative_12label/checkpoints/best.pt`。
- 只读加载确认：`best.pt` 内部 `global_step=3500`、`best_val_loss=0.02821943630927248`；旧 `step_3000.pt` 仍为 `global_step=3000`、`best_val_loss=0.028968754708766937`。因此当前可靠恢复点已推进到 global 3500。
- 结果判断：新 `3500` val_loss 比 2026-06-01 中断日志里的旧 `Step 3500: val_loss = 0.0297` 更低，符合预期，没有模型质量异常。
- 温度/守卫：12:35/12:40/12:45/12:50 资源守卫均 OK，GPU1 温度约 `75-78C`；12:52 手动快照处于验证/低负载期，GPU1 `temp=65C`。未见新的 traceback / RuntimeError / CUDA OOM。
- 恢复脚本修正：由于 `best.pt` 已成为 global 3500 的最新可靠 checkpoint，`scripts/select_latest_checkpoint.ps1` 和 `scripts/run_null_as_negative_gpu1.cmd` 已调整为选择最近落盘的 `best.pt` 或 `step_*.pt`。真实目录验证返回 `best.pt`，避免若 `step_4000.pt` 前再次中断时回退到 `step_3000.pt`。
- 下一关键点：global `4000` 验证和 `step_4000.pt`；按 thermal 后实际速度，预计约 35-50 分钟后到达。完成 `step_4000.pt` 后再刷新全程 ETA。

2026-06-08 13:36 `null-as-negative` step 4000 checkpoint 已验证：
- global `4000` 验证已完成，日志记录 `Step 4000: val_loss = 0.0310`，随后保存 `outputs/ablation_ums_null_as_negative_12label/checkpoints/step_4000.pt`。
- 只读加载确认：`step_4000.pt` 内部 `global_step=4000`、`best_val_loss=0.02821943630927248`；`best.pt` 仍为 global 3500。当前可靠恢复点已推进到 `step_4000.pt`。
- 训练已继续到恢复后约 `1007 / 7000` step，折算 global 约 `4007 / 10000`；剩余约 `5993` step。下一关键验证点为 global `4500`，下一 step checkpoint 为 `step_5000.pt`。
- 温度/守卫：13:00-13:35 资源守卫均 OK，GPU1 温度约 `51-77C`；13:36 手动快照 GPU1 `temp=66C`、显存约 `12501MiB`，仍只用物理 GPU1，GPU0 空闲。
- 结果判断：`4000` 的 val_loss 高于当前 best 但仍在低损失区间，训练 loss 约 `0.0254-0.0282`，未见 traceback / RuntimeError / CUDA OOM；结果仍符合预期。
- ETA：thermal 配置下从 12:18:39 的 global 3000 恢复到 13:35:58 的 `step_4000.pt`，约 77 分钟 / 1000 step（含 3500、4000 两轮验证和保存）。按此估计，当前 `null-as-negative` 训练还需约 7.5-9.5 小时，若温度或系统调度变慢则保守看 10-12 小时。

2026-06-08 14:22 小时同步（global 4500 验证完成）：
- `VIVID_null_as_negative_gpu1` 仍为 Running；VIVID PID 11616 仍只在物理 GPU1 上运行，GPU0 空闲。
- global `4500` 验证已完成，日志记录 `Step 4500: val_loss = 0.0288`。该值高于当前 best `0.0282194`，因此没有刷新 `best.pt`，这是符合预期的。
- 当前训练已继续到恢复后约 `1553 / 7000` step，折算 global 约 `4553 / 10000`；剩余约 `5447` step。最新可靠 step checkpoint 仍为 `step_4000.pt`，下一可靠 checkpoint 为 `step_5000.pt`。
- 温度/守卫：13:40-14:20 资源守卫均 OK，GPU1 温度约 `68-76C`；14:22 手动快照 GPU1 `temp=64C`、显存约 `12501MiB`。未见 traceback / RuntimeError / CUDA OOM。
- 结果判断：`4500` val_loss 接近 best，训练 loss 近期约 `0.0213-0.0303`，结果继续符合预期。
- ETA：按 `step_4000.pt` 后到 global 4500 的实际节奏，`step_5000.pt` 预计约 45-70 分钟内生成；完整 `null-as-negative` 训练剩余约 7-9 小时，保守看 10 小时左右。

2026-06-08 15:24 小时同步（`step_5000.pt` 已验证）：
- `VIVID_null_as_negative_gpu1` 仍为 Running；VIVID PID 11616 仍只在物理 GPU1 上运行，GPU0 空闲。
- global `5000` 验证已完成，日志记录 `Step 5000: val_loss = 0.0280`，随后保存 `best.pt` 和 `step_5000.pt`。
- 只读加载确认：`step_5000.pt` 内部 `global_step=5000`、`best_val_loss=0.02795240507477861`；`best.pt` 也已刷新到 global 5000。当前可靠恢复点已推进到 `step_5000.pt`。
- 当前训练已继续到恢复后约 `2133 / 7000` step，折算 global 约 `5133 / 10000`；剩余约 `4867` step。下一验证点为 global `5500`，下一 step checkpoint 为 `step_6000.pt`。
- 温度/守卫：14:25-15:25 资源守卫全部 OK，GPU1 温度约 `48-75C`，明显低于 83C 硬停阈值；15:24 手动快照 GPU1 `temp=64C`、显存约 `12501MiB`。
- 结果判断：`5000` val_loss 刷新 best，说明 thermal resume 后质量走势优于预期；训练 loss 近期约 `0.0234-0.0303`，未见 traceback / RuntimeError / CUDA OOM。
- ETA：`step_4000.pt` 到 `step_5000.pt` 约 95 分钟。按此较慢窗口估计，当前 `null-as-negative` 训练剩余约 7.5-8.5 小时；若后续维持 14:25-15:25 的慢速波动，保守看 9-10 小时。

2026-06-08 16:27 小时同步（global 5500 验证完成）：
- `VIVID_null_as_negative_gpu1` 仍为 Running；VIVID PID 11616 仍只在物理 GPU1 上运行，GPU0 空闲。
- global `5500` 验证已完成，日志记录 `Step 5500: val_loss = 0.0307`。该值高于当前 best `0.0279524`，因此未刷新 `best.pt`，属于正常波动。
- 当前训练已继续到恢复后约 `2551 / 7000` step，折算 global 约 `5551 / 10000`；剩余约 `4449` step。最新可靠 checkpoint 仍为 `step_5000.pt`，下一可靠 checkpoint 为 `step_6000.pt`。
- 温度/守卫：15:30-16:25 资源守卫全部 OK，GPU1 温度约 `61-71C`；16:27 手动快照 GPU1 `temp=64C`、显存约 `12501MiB`。未见 traceback / RuntimeError / CUDA OOM。
- 结果判断：`5500` val_loss 高于 best 但仍在可接受低损失区间；训练 loss 近期约 `0.0233-0.0303`，整体仍符合预期。
- ETA：当前 5000-5500 区间有明显慢速波动，`step_6000.pt` 预计约 70-100 分钟后；完整 `null-as-negative` 训练剩余约 7-9 小时，保守看 10 小时。

2026-06-08 17:30 小时同步（慢速窗口，`step_6000.pt` 尚未到）：
- `VIVID_null_as_negative_gpu1` 仍为 Running；VIVID PID 11616 仍只在物理 GPU1 上运行，GPU0 空闲。
- 当前训练推进到恢复后约 `2836 / 7000` step，折算 global 约 `5836 / 10000`；剩余约 `4164` step。最新可靠 checkpoint 仍为 `step_5000.pt`，`step_6000.pt` 尚未生成。
- 温度/守卫：16:30-17:30 资源守卫全部 OK，GPU1 温度约 `49-70C`；17:30 手动快照 GPU1 `temp=56C`、显存约 `12501MiB`。未见 traceback / RuntimeError / CUDA OOM。
- 现象：global 5500 后出现吞吐慢速窗口，部分 step 约 `20-30s/step`，GPU 功耗/利用率偏低；但日志持续推进，没有停住，也没有错误。因此当前不是失败 case，而是系统调度/吞吐观察点。
- 结果判断：最近训练 loss 约 `0.0236-0.0314`，仍符合预期。
- ETA：由于慢速窗口，`step_6000.pt` 预计还需约 25-45 分钟；完整 `null-as-negative` 训练剩余保守上调为约 8-11 小时。

2026-06-08 18:32 小时同步（慢速窗口加重，仍未失败）：
- `VIVID_null_as_negative_gpu1` 仍为 Running；VIVID PID 11616 仍只在物理 GPU1 上运行，GPU0 空闲。
- 当前训练推进到恢复后约 `2913 / 7000` step，折算 global 约 `5913 / 10000`；剩余约 `4087` step。最新可靠 checkpoint 仍为 `step_5000.pt`，`step_6000.pt` 尚未生成。
- 温度/守卫：17:55-18:27 资源守卫全部 OK，GPU1 温度约 `49-58C`，显存约 `12501MiB`；未触发 stop action。未见 traceback / RuntimeError / CUDA OOM / GPU lost。
- 现象：18:00 后吞吐慢速窗口明显加重，部分 step 约 `90-120s/step`，同时只读诊断命令也出现超时，倾向于系统调度或 I/O 抖动；但训练日志仍持续推进，因此当前仍不是失败 case。
- 结果判断：最新验证仍是 global `5500` 的 `val_loss=0.0307`；最近训练 loss 约 `0.0243-0.0285`，质量指标没有异常，结果仍符合预期。下一次质量判断点是 global `6000` 验证。
- ETA：离 `step_6000.pt` 还约 `87` step；若维持当前慢速窗口，可能还需约 2-3 小时，若吞吐恢复则约 1-1.5 小时。完整 `null-as-negative` 剩余时间当前不稳定：若恢复到 20-35s/step 约 23-40 小时，若维持 90s/step 以上会显著拉长到 4 天量级；下一轮以 `step_6000.pt` 的实测落盘时间重新校准。

2026-06-08 19:26 重启后恢复处理（增加 500-step 中间断点）：
- 用户手动重启后，旧任务已停止；重启前日志最后推进到恢复后约 `2941 / 7000` step，折算 global 约 `5941 / 10000`，但没有生成 `step_6000.pt`。
- 当前最新可靠 checkpoint 仍为 `step_5000.pt`；因此 global `5000-5941` 区间需要重跑。此前 global `5500` 的 `val_loss=0.0307` 只作为诊断记录，不作为恢复点。
- 已按用户要求将 `configs/ablation_ums_null_as_negative_12label_thermal_resume_gpu1.yaml` 的 `save_interval` 从 `1000` 改为 `500`，后续会保存 `step_5500.pt`、`step_6000.pt` 等中间断点，减少再次中断时的重跑损失。
- 重启/中断 case study 已归档到 `history/20260608_1924_null_as_negative_reboot_step5941/`；旧训练日志已移动到该目录，资源守卫日志已复制保存。
- 恢复脚本 `scripts/run_null_as_negative_gpu1.cmd` 当前会选择最新 checkpoint，实测返回 `outputs/ablation_ums_null_as_negative_12label/checkpoints/step_5000.pt`。下一步从 `step_5000.pt` 只用 GPU1 续跑。

2026-06-08 20:27 启动排查后成功续跑（已跑上 GPU1）：
- 19:29 的首次重启后启动没有进入训练，只停在 LLM 初始化阶段并以 `exitcode 1073807364` 退出；训练日志已归档到 `history/20260608_2017_null_as_negative_startup_aborted_llm_load/`，并写入 case study。
- 排查结果：GPU1 torch smoke test 通过；`step_5000.pt` 可读，内部 `global_step=5000`、`best_val_loss=0.02795240507477861`；本地 Qwen 模型可加载到 GPU1；VIVIDModel 构建测试也通过。
- 配置微调：因当前环境未安装 FlashAttention2，已在 thermal resume config 中显式设置 `model.use_flash_attention: false`，避免每次启动先尝试不可用的 flash attention。实际训练仍使用 eager attention，与之前 fallback 后的有效路径一致。
- 20:21 重新启动计划任务，日志确认从 `step_5000.pt` 加载并显示 `Resuming from step 5000`。20:26 训练进度已到本次恢复 `25 / 5000` step，折算 global `5025 / 10000`。
- GPU 约束：VIVID PID 20176 在物理 GPU1（bus `00000000:05:00.0`）上运行，GPU1 显存约 `12499MiB`；GPU0 仍为 `0MiB`，符合“只用 gpu1”。
- 下一关键点：由于 `save_interval=500`，下一可靠中间断点改为 `step_5500.pt`；当前距离 `step_5500.pt` 约 `475` step。按重启后初始 4-7s/step 估计，约 35-60 分钟可到；若再次进入慢速窗口则顺延。

2026-06-08 21:27 小时同步（接近 `step_5500.pt`）：
- `VIVID_null_as_negative_gpu1` 仍为 Running；VIVID PID 20176 仍只在物理 GPU1 上运行，GPU0 空闲。
- 当前训练推进到本次恢复约 `428 / 5000` step，折算 global 约 `5428 / 10000`；剩余约 `4572` step。最新可靠 checkpoint 仍为 `step_5000.pt`，`step_5500.pt` 尚未生成。
- 温度/守卫：20:35-21:20 资源守卫全部 OK，GPU1 温度约 `58-75C`；21:26 手动快照 GPU1 `temp=69C`、显存约 `12499MiB`。未见 traceback / RuntimeError / CUDA OOM / GPU lost。
- 结果判断：重启后训练 loss 约 `0.0227-0.0303`，与此前区间一致；当前没有质量异常，结果仍符合预期。
- ETA：距离新增中间断点 `step_5500.pt` 约 `72` step，按当前 8-15s/step 波动估计还需约 10-15 分钟。`step_5500.pt` 生成后会立即加载核验 checkpoint 元数据，并以 500-step 断点节奏继续监控。

2026-06-08 21:50 中间断点同步（`step_5500.pt` 已验证）：
- global `5500` 验证已完成，日志记录 `Step 5500: val_loss = 0.0266`，随后保存新的 `best.pt` 与 `step_5500.pt`。
- 只读加载确认：`step_5500.pt` 内部 `global_step=5500`、`best_val_loss=0.026638102756735095`；`best.pt` 也已刷新到 global 5500。当前可靠恢复点已推进到 `step_5500.pt`。
- 当前训练已继续到本次恢复约 `503 / 5000` step，折算 global 约 `5503 / 10000`；剩余约 `4497` step。下一验证点为 global `6000`，下一中间 checkpoint 为 `step_6000.pt`。
- GPU 约束：计划任务 `VIVID_null_as_negative_gpu1` 仍为 Running；VIVID PID 20176 仍只在物理 GPU1（bus `00000000:05:00.0`）运行，GPU0 显存 `0MiB`，符合“只用 gpu1”约束。
- 温度/守卫：21:30-21:50 资源守卫全部 OK，GPU1 温度约 `43-68C`，显存约 `12499-12501MiB`；未见 traceback / RuntimeError / CUDA OOM / GPU lost。
- 结果判断：`5500` val_loss 从 `step_5000.pt` 的 best `0.02795240507477861` 进一步降到 `0.026638102756735095`，刷新 best，结果优于预期且无失败 case。
- ETA：从 20:21 重启到 21:49 生成 `step_5500.pt`，约 88 分钟完成 500 step 加一次验证/保存。按此窗口估算，当前剩余约 13-15 小时；若后续再次进入慢速窗口，保守看约 16-18 小时。下一次 500-step 断点 `step_6000.pt` 预计约 1.5-2 小时后生成。

2026-06-08 22:10 结构整理与短同步：
- 子智能体只读审计确认：`miccai2026.md` 已覆盖 2026-06-08 的 CUDA 恢复、温度硬停 case、重启 case、启动排查 case、GPU1-only 约束以及 `step_5500.pt` 中间断点；当前没有新的异常信号。
- 按项目结构要求，将两个 2026-05-11 的旧非当前主线失败日志从 `outputs/logs/` 归档到 `History/20260608_2210_old_active_logs_cleanup/`：`ablation_ums_ansmask_12label_train.log` 和 `answerability_watch.log`。当前活跃的 `ablation_ums_null_as_negative_12label_train.log` 与资源守卫日志未移动。
- 当前训练继续推进到本次恢复约 `568 / 5000` step，折算 global 约 `5568 / 10000`；剩余约 `4432` step。最新可靠 checkpoint 仍为已验证的 `step_5500.pt`。
- GPU 约束：VIVID PID 20176 仍只在物理 GPU1（bus `00000000:05:00.0`）运行，GPU0 显存 `0MiB`，GPU1 显存约 `12501MiB`，符合“只用 gpu1”。
- 温度/守卫：22:05 资源守卫 OK；22:10 手动快照 GPU1 `temp=55C`、功耗约 `55.85W`、util `91%`。未见 traceback / RuntimeError / CUDA OOM / GPU lost。

2026-06-08 22:43 小时同步：
- `VIVID_null_as_negative_gpu1` 仍在运行；VIVID PID 20176 仍只在物理 GPU1（bus `00000000:05:00.0`）运行，GPU0 显存 `0MiB`，符合“只用 gpu1”。
- 当前训练推进到本次恢复约 `706 / 5000` step，折算 global 约 `5706 / 10000`；剩余约 `4294` step。最新可靠 checkpoint 仍为已验证的 `step_5500.pt`，`step_6000.pt` 尚未生成。
- 温度/守卫：22:15-22:40 资源守卫全部 OK，GPU1 温度约 `58-66C`；22:43 手动快照 GPU1 `temp=70C`、显存约 `12501MiB`。未见 traceback / RuntimeError / CUDA OOM / GPU lost。
- 结果判断：`step_5500.pt` 已刷新 best，之后训练 loss 近期约 `0.0227-0.0283`，没有质量异常；当前结果仍符合预期。
- ETA：距离下一中间断点 `step_6000.pt` 还约 `294` step。按当前 8-18s/step 的恢复后窗口估计，`step_6000.pt` 预计约 1-1.5 小时后生成；完整 `null-as-negative` 训练剩余约 13-16 小时，若再次进入慢速窗口则保守看 18 小时左右。

2026-06-08 23:46 中间断点同步（`step_6000.pt` 已验证）：
- global `6000` 验证已完成，日志记录 `Step 6000: val_loss = 0.0268`，随后保存 `outputs/ablation_ums_null_as_negative_12label/checkpoints/step_6000.pt`。
- 只读加载确认：`step_6000.pt` 内部 `global_step=6000`、`best_val_loss=0.026638102756735095`；`best.pt` 仍为 global 5500，说明 `6000` 未刷新 best，但与 best 很接近。
- 当前训练已继续到本次恢复约 `1005 / 5000` step，折算 global 约 `6005 / 10000`；剩余约 `3995` step。最新可靠 step checkpoint 已推进到 `step_6000.pt`，下一中间 checkpoint 为 `step_6500.pt`。
- GPU 约束：VIVID PID 20176 仍只在物理 GPU1（bus `00000000:05:00.0`）运行，GPU0 显存 `0MiB`，GPU1 显存约 `12501MiB`，符合“只用 gpu1”。计划任务状态查询在本轮曾超时，但 GPU 进程、日志和 checkpoint 均显示训练正常继续。
- 温度/守卫：23:25-23:45 资源守卫全部 OK，GPU1 温度约 `62-72C`；23:45 手动快照 GPU1 `temp=60C`。未见 traceback / RuntimeError / CUDA OOM / GPU lost。
- 结果判断：`6000` val_loss 为 `0.0268`，略高于 `5500` best `0.0266381`，但明显优于早期 `5000` best `0.0279524`，属于正常低损失区间，结果继续符合预期，无新的失败 case。
- ETA：从 `step_5500.pt` 到 `step_6000.pt` 约 115 分钟（含验证/保存），当前剩余约 3995 step。按该窗口估算完整 `null-as-negative` 训练还需约 15-17 小时；若后续维持 8-12s/step 的较快窗口，则可能缩短到约 12-14 小时。

2026-06-09 10:58 系统错误后恢复准备：
- 用户报告系统错误/死机后，现场确认 GPU0/GPU1 均为空闲（显存 `0MiB`），`VIVID_null_as_negative_gpu1` 为 `Ready`，训练没有继续运行。
- 最新可靠 checkpoint 为 `step_6000.pt`；只读加载确认内部 `global_step=6000`、`best_val_loss=0.026638102756735095`。`scripts/select_latest_checkpoint.ps1 -Dir outputs/ablation_ums_null_as_negative_12label/checkpoints` 实测返回 `step_6000.pt`。
- 崩溃前训练日志最后推进到本次恢复约 `1011 / 5000` step，折算 global 约 `6011 / 10000`；因此未落盘损失约 11 step，可靠恢复点仍是 global 6000。
- 资源守卫日志显示 2026-06-08 23:45 前均 OK，之后到 2026-06-09 10:55 才恢复记录，且两张 GPU 都为空闲；没有 guard stop action。判断这是系统级中断，不是模型错误、数据错误、CUDA OOM 或质量异常。
- 已新增 case study 并归档崩溃前 active 训练日志：`History/20260609_1058_null_as_negative_system_crash_step6011/`。资源守卫日志复制到该目录，原始守卫日志保留继续使用。
- 下一步：从 `step_6000.pt` 只用物理 GPU1 恢复训练，目标推进到下一中间断点 `step_6500.pt`。当前剩余约 `4000` step；按上一窗口估计完整 `null-as-negative` 训练仍需约 12-17 小时，恢复后会用真实速度重新校准。

2026-06-09 11:08 系统错误后恢复已确认：
- 已通过计划任务 `VIVID_null_as_negative_gpu1` 重新启动训练；新 active 日志确认 `CUDA_VISIBLE_DEVICES=1`、resume checkpoint 为 `step_6000.pt`，并显示 `Resuming from step 6000`。
- GPU 约束：当前 VIVID PID 912 只在物理 GPU1（bus `00000000:05:00.0`）运行，GPU1 显存约 `12499MiB`；GPU0 显存 `0MiB`，符合“只用 gpu1”。
- 当前训练推进到本次恢复约 `52 / 4000` step，折算 global 约 `6052 / 10000`；剩余约 `3948` step。最新可靠 checkpoint 仍为 `step_6000.pt`，下一中间 checkpoint 为 `step_6500.pt`。
- 温度/守卫：11:05 资源守卫 OK，GPU1 `temp=67C`；11:08 手动快照 GPU1 `temp=71C`、功耗约 `236.79W`。未见 traceback / RuntimeError / CUDA OOM / GPU lost。
- 结果判断：恢复后训练 loss 约 `0.0220-0.0291`，与此前低损失区间一致，质量没有异常；当前结果仍符合预期。
- ETA：恢复初始窗口约 `4-5s/step`，如果保持该速度，`step_6500.pt` 预计约 35-45 分钟到达；完整剩余约 5-6 小时加验证/保存开销。考虑此前慢速波动，保守仍按约 12-16 小时观察，等 `step_6500.pt` 生成后重新校准。

2026-06-09 11:46 中间断点同步（`step_6500.pt` 已验证）：
- global `6500` 验证已完成，日志记录 `Step 6500: val_loss = 0.0269`，随后保存 `outputs/ablation_ums_null_as_negative_12label/checkpoints/step_6500.pt`。
- 只读加载确认：`step_6500.pt` 内部 `global_step=6500`、`best_val_loss=0.026638102756735095`；`best.pt` 仍为 global 5500。`6500` 未刷新 best，但与 best 很接近。
- 当前训练已继续到本次恢复约 `511 / 4000` step，折算 global 约 `6511 / 10000`；剩余约 `3489` step。最新可靠 step checkpoint 已推进到 `step_6500.pt`，下一中间 checkpoint 为 `step_7000.pt`。
- GPU 约束：`VIVID_null_as_negative_gpu1` 仍为 Running；VIVID PID 912 仍只在物理 GPU1（bus `00000000:05:00.0`）运行，GPU0 显存 `0MiB`，GPU1 显存约 `12501MiB`，符合“只用 gpu1”。
- 温度/守卫：11:25-11:45 资源守卫全部 OK，GPU1 温度约 `69-76C`；11:45 手动快照 GPU1 `temp=74C`。未见 traceback / RuntimeError / CUDA OOM / GPU lost。
- 结果判断：`6500` val_loss 为 `0.0269`，略高于 `5500` best `0.0266381`，但仍处于正常低损失区间；恢复后训练 loss 约 `0.0215-0.0278`。结果继续符合预期，无新的失败 case。
- ETA：从 11:00 恢复启动到 11:44 生成 `step_6500.pt`，约 44 分钟完成 500 step 加验证/保存，明显快于昨晚慢速窗口。按这个窗口估计剩余 3489 step 约 5-6 小时；保守考虑温度/调度波动，完整 `null-as-negative` 训练还需约 7-10 小时。下一次 `step_7000.pt` 预计约 45-70 分钟后生成。

2026-06-09 12:27 小时同步（接近 `step_7000.pt`）：
- `VIVID_null_as_negative_gpu1` 仍为 Running；VIVID PID 912 仍只在物理 GPU1（bus `00000000:05:00.0`）运行，GPU0 显存 `0MiB`，GPU1 显存约 `12501MiB`。
- 当前训练推进到本次恢复约 `924 / 4000` step，折算 global 约 `6924 / 10000`；剩余约 `3076` step。最新可靠 checkpoint 仍为 `step_6500.pt`，`step_7000.pt` 尚未生成。
- 温度/守卫：11:55-12:25 资源守卫全部 OK，GPU1 温度约 `66-75C`；12:27 手动快照 GPU1 `temp=69C`。未见 traceback / RuntimeError / CUDA OOM / GPU lost。
- 结果判断：最近训练 loss 约 `0.0211-0.0282`，与此前低损失区间一致；当前结果仍符合预期，无新的失败 case。
- ETA：距离 `step_7000.pt` 约 `76` step；按当前 5-7s/step 加验证/保存，预计约 15-25 分钟生成。完整 `null-as-negative` 剩余约 5-8 小时，取决于后续是否再次出现慢速窗口。

2026-06-09 12:54 中间断点同步（`step_7000.pt` 已验证）：
- global `7000` 验证已完成，日志记录 `Step 7000: val_loss = 0.0278`，随后保存 `outputs/ablation_ums_null_as_negative_12label/checkpoints/step_7000.pt`。
- 只读加载确认：`step_7000.pt` 内部 `global_step=7000`、`best_val_loss=0.026638102756735095`；`best.pt` 仍为 global 5500。`7000` 未刷新 best。
- 当前训练已继续到本次恢复约 `1150 / 4000` step，折算 global 约 `7150 / 10000`；剩余约 `2850` step。最新可靠 step checkpoint 已推进到 `step_7000.pt`，下一中间 checkpoint 为 `step_7500.pt`。
- GPU 约束：`VIVID_null_as_negative_gpu1` 仍为 Running；VIVID PID 912 仍只在物理 GPU1（bus `00000000:05:00.0`）运行，GPU0 显存 `0MiB`，GPU1 显存约 `12501MiB`。
- 温度/守卫：12:30-12:50 资源守卫全部 OK，GPU1 温度约 `69-76C`；12:53 手动快照 GPU1 `temp=65C`。未见 traceback / RuntimeError / CUDA OOM / GPU lost。
- 结果判断：`7000` val_loss 为 `0.0278`，高于 `5500/6000/6500`，但仍处于低损失区间，且训练 loss 近期约 `0.0190-0.0304`；目前没有失败 case，结果仍符合预期，但后续会继续看 `7500` 是否回落或稳定。
- ETA：从 `step_6500.pt` 到 `step_7000.pt` 约 53 分钟（含验证/保存）。按当前窗口估算，剩余约 5-6 小时；保守考虑慢速波动，完整 `null-as-negative` 训练还需约 6-8 小时。

2026-06-09 13:46 中间断点同步（`step_7500.pt` 刷新 best）：
- global `7500` 验证已完成，日志记录 `Step 7500: val_loss = 0.0266`，随后保存新的 `best.pt` 与 `step_7500.pt`。
- 只读加载确认：`step_7500.pt` 与 `best.pt` 内部均为 `global_step=7500`、`best_val_loss=0.026576380329039282`。当前可靠 best 已从 global 5500 推进到 global 7500。
- 当前训练已继续到本次恢复约 `1596 / 4000` step，折算 global 约 `7596 / 10000`；剩余约 `2404` step。最新可靠 step checkpoint 已推进到 `step_7500.pt`，下一中间 checkpoint 为 `step_8000.pt`。
- GPU 约束：`VIVID_null_as_negative_gpu1` 仍为 Running；VIVID PID 912 仍只在物理 GPU1（bus `00000000:05:00.0`）运行，GPU0 显存 `0MiB`，GPU1 显存约 `12501MiB`。
- 温度/守卫：13:15-13:45 资源守卫全部 OK，GPU1 温度约 `58-73C`；13:45 手动快照 GPU1 `temp=68C`。未见 traceback / RuntimeError / CUDA OOM / GPU lost。
- 结果判断：`7500` val_loss 回落并刷新 best，说明 `7000` 的小幅回升只是正常波动；训练 loss 近期约 `0.0192-0.0279`。结果符合预期，且优于上一 best。
- ETA：从 `step_7000.pt` 到 `step_7500.pt` 约 57 分钟（含验证/保存）。按当前窗口估算，剩余 2404 step 约 4.5-5.5 小时；保守考虑慢速波动，完整 `null-as-negative` 训练还需约 5-7 小时。

2026-06-09 14:38 中间断点同步（`step_8000.pt` 已验证）：
- global `8000` 验证已完成，日志记录 `Step 8000: val_loss = 0.0274`，随后保存 `outputs/ablation_ums_null_as_negative_12label/checkpoints/step_8000.pt`。
- 只读加载确认：`step_8000.pt` 内部 `global_step=8000`、`best_val_loss=0.026576380329039282`；`best.pt` 仍为 global 7500。`8000` 未刷新 best。
- 当前训练已继续到本次恢复约 `2008 / 4000` step，折算 global 约 `8008 / 10000`；剩余约 `1992` step。最新可靠 step checkpoint 已推进到 `step_8000.pt`，下一中间 checkpoint 为 `step_8500.pt`。
- GPU 约束：`VIVID_null_as_negative_gpu1` 仍为 Running；VIVID PID 912 仍只在物理 GPU1（bus `00000000:05:00.0`）运行，GPU0 显存 `0MiB`，GPU1 显存约 `12501MiB`。
- 温度/守卫：14:05-14:35 资源守卫全部 OK，GPU1 温度约 `49-74C`；14:38 手动快照 GPU1 `temp=68C`。未见 traceback / RuntimeError / CUDA OOM / GPU lost。
- 结果判断：`8000` val_loss 为 `0.0274`，高于当前 best 但低于 `7000` 的 `0.0278`，仍属正常波动；训练 loss 近期约 `0.0237-0.0278`。结果仍符合预期，无新的失败 case。
- ETA：从 `step_7500.pt` 到 `step_8000.pt` 约 62 分钟（含验证/保存）。按当前窗口估算，剩余 1992 step 约 4-5 小时；保守考虑慢速波动，完整 `null-as-negative` 训练还需约 5-6 小时。

2026-06-09 15:30 小时同步（接近 `step_8500.pt`）：
- `VIVID_null_as_negative_gpu1` 仍为 Running；VIVID PID 912 仍只在物理 GPU1（bus `00000000:05:00.0`）运行，GPU0 显存 `0MiB`，GPU1 显存约 `12501MiB`。
- 当前训练推进到本次恢复约 `2425 / 4000` step，折算 global 约 `8425 / 10000`；剩余约 `1575` step。最新可靠 checkpoint 仍为 `step_8000.pt`，`step_8500.pt` 尚未生成。
- 温度/守卫：15:00-15:30 资源守卫全部 OK，GPU1 温度约 `66-72C`；15:30 手动快照 GPU1 `temp=62C`。未见 traceback / RuntimeError / CUDA OOM / GPU lost。
- 结果判断：最近训练 loss 约 `0.0225-0.0283`，与此前低损失区间一致；当前结果仍符合预期，无新的失败 case。
- ETA：距离 `step_8500.pt` 约 `75` step；按当前速度加验证/保存，预计约 15-25 分钟生成。完整 `null-as-negative` 剩余约 3.5-5 小时。

2026-06-09 16:01 中间断点同步（`step_8500.pt` 刷新 best）：
- global `8500` 验证已完成，日志记录 `Step 8500: val_loss = 0.0262`，随后保存新的 `best.pt` 与 `step_8500.pt`。
- 只读加载确认：`step_8500.pt` 与 `best.pt` 内部均为 `global_step=8500`、`best_val_loss=0.026174051193576194`。当前可靠 best 已从 global 7500 推进到 global 8500。
- 当前训练已继续到本次恢复约 `2627 / 4000` step，折算 global 约 `8627 / 10000`；剩余约 `1373` step。最新可靠 step checkpoint 已推进到 `step_8500.pt`，下一中间 checkpoint 为 `step_9000.pt`。
- GPU 约束：`VIVID_null_as_negative_gpu1` 仍为 Running；VIVID PID 912 仍只在物理 GPU1（bus `00000000:05:00.0`）运行，GPU0 显存 `0MiB`，GPU1 显存约 `12501MiB`。
- 温度/守卫：15:35-15:55 资源守卫全部 OK，GPU1 温度约 `68-72C`；15:58 手动快照 GPU1 `temp=62C`、功耗约 `135.57W`。未见 traceback / RuntimeError / CUDA OOM / GPU lost。
- 结果判断：`8500` val_loss 进一步降到 `0.026174051193576194` 并刷新 best，优于 `7500` 的 `0.026576380329039282`；训练 loss 近期约 `0.0208-0.0271`。结果符合预期且偏正向，无新的失败 case。
- ETA：从 `step_8000.pt` 到 `step_8500.pt` 约 67 分钟（含验证/保存）。按当前窗口估算，剩余 1373 step 约 3-4 小时；保守考虑慢速波动，完整 `null-as-negative` 训练还需约 3.5-5 小时。下一次 `step_9000.pt` 预计约 45-70 分钟后生成。

2026-06-09 16:44 近 `step_9000.pt` 同步：
- `VIVID_null_as_negative_gpu1` 仍为 Running；VIVID PID 912 仍只在物理 GPU1（bus `00000000:05:00.0`）运行，GPU0 显存 `0MiB`，GPU1 显存约 `12501MiB`。
- 当前训练推进到本次恢复约 `2887 / 4000` step，折算 global 约 `8887 / 10000`；剩余约 `1113` step。最新可靠 checkpoint 仍为已验证的 `step_8500.pt`，`step_9000.pt` 尚未生成。
- 温度/守卫：16:10-16:40 资源守卫全部 OK，GPU1 温度约 `60-74C`；16:44 手动快照 GPU1 `temp=61C`、显存约 `12501MiB`。未见 traceback / RuntimeError / CUDA OOM / GPU lost。
- 结果判断：`8500` 已刷新 best，之后训练 loss 近期约 `0.0197-0.0281`，仍在正常低损失区间；当前结果符合预期，无新的失败 case。
- ETA：距离 `step_9000.pt` 约 `113` step。按当前速度加验证/保存，预计约 20-35 分钟生成。完整 `null-as-negative` 训练剩余约 2.5-4 小时，若后续慢速窗口增多则保守约 4-5 小时。

2026-06-09 17:12 中间断点同步（`step_9000.pt` 已验证）：
- global `9000` 验证已完成，日志记录 `Step 9000: val_loss = 0.0265`，随后保存 `outputs/ablation_ums_null_as_negative_12label/checkpoints/step_9000.pt`。
- 只读加载确认：`step_9000.pt` 内部 `global_step=9000`、`best_val_loss=0.026174051193576194`；`best.pt` 仍为 global 8500。`9000` 未刷新 best，但与 best 很接近。
- 当前训练已继续到本次恢复约 `3020 / 4000` step，折算 global 约 `9020 / 10000`；剩余约 `980` step。最新可靠 step checkpoint 已推进到 `step_9000.pt`，下一中间 checkpoint 为 `step_9500.pt`。
- GPU 约束：VIVID PID 912 仍只在物理 GPU1（bus `00000000:05:00.0`）运行，GPU0 显存 `0MiB`，GPU1 显存约 `12501MiB`，符合“只用 gpu1”。
- 温度/守卫：16:45-17:10 资源守卫全部 OK，GPU1 温度约 `53-70C`；17:12 手动快照 GPU1 `temp=59C`、显存约 `12501MiB`。未见 traceback / RuntimeError / CUDA OOM / GPU lost。
- 结果判断：`9000` val_loss 为 `0.0265`，略高于 `8500` best `0.026174051193576194`，但仍明显处于低损失区间；训练 loss 近期约 `0.0242-0.0295`。结果符合预期，无新的失败 case。
- ETA：从 `step_8500.pt` 到 `step_9000.pt` 约 84 分钟（含验证/保存，验证阶段本轮偏慢）。按该窗口估算，剩余 980 step 约 2.8-3.5 小时；保守考虑慢速波动，完整 `null-as-negative` 训练还需约 3-4.5 小时。下一次 `step_9500.pt` 预计约 1.2-2 小时后生成。

2026-06-09 18:15 小时同步（接近 `step_9500.pt`）：
- 当前训练推进到本次恢复约 `3439 / 4000` step，折算 global 约 `9439 / 10000`；剩余约 `561` step。最新可靠 checkpoint 仍为已验证的 `step_9000.pt`，`step_9500.pt` 尚未生成。
- GPU 约束：VIVID PID 912 仍只在物理 GPU1（bus `00000000:05:00.0`）运行，GPU0 显存 `0MiB`，GPU1 显存约 `12501MiB`。本轮计划任务状态查询超时，但 GPU 进程、日志和 checkpoint 状态均显示训练仍正常继续。
- 温度/守卫：17:35-18:15 资源守卫全部 OK，GPU1 温度约 `66-72C`；18:15 手动快照 GPU1 `temp=60C`、显存约 `12501MiB`。未见 traceback / RuntimeError / CUDA OOM / GPU lost。
- 结果判断：最近训练 loss 约 `0.0198-0.0272`，仍在正常低损失区间；当前结果符合预期，无新的失败 case。
- ETA：距离 `step_9500.pt` 约 `61` step。按当前速度加验证/保存，预计约 20-35 分钟生成。完整 `null-as-negative` 剩余约 1.5-2.5 小时；如验证或保存阶段变慢，保守约 2.5-3.5 小时。

2026-06-09 18:48 中间断点同步（`step_9500.pt` 已验证）：
- global `9500` 验证已完成，日志记录 `Step 9500: val_loss = 0.0264`，随后保存 `outputs/ablation_ums_null_as_negative_12label/checkpoints/step_9500.pt`。
- 只读加载确认：`step_9500.pt` 内部 `global_step=9500`、`best_val_loss=0.026174051193576194`；`best.pt` 仍为 global 8500。`9500` 未刷新 best，但与 best 很接近。
- 当前训练已继续到本次恢复约 `3617 / 4000` step，折算 global 约 `9617 / 10000`；剩余约 `383` step。最新可靠 step checkpoint 已推进到 `step_9500.pt`，下一关键点为最终 global `10000`。
- GPU 约束：VIVID PID 912 仍只在物理 GPU1（bus `00000000:05:00.0`）运行，GPU0 显存 `0MiB`，GPU1 显存约 `12501MiB`，符合“只用 gpu1”。
- 温度/守卫：18:20-18:45 资源守卫全部 OK，GPU1 温度约 `60-68C`；18:48 手动快照 GPU1 `temp=63C`、显存约 `12501MiB`。未见 traceback / RuntimeError / CUDA OOM / GPU lost。
- 结果判断：`9500` val_loss 为 `0.0264`，略高于 `8500` best `0.026174051193576194`，仍属正常低损失区间；训练 loss 近期约 `0.0199-0.0293`。结果符合预期，无新的失败 case。
- ETA：进入最后约 `383` step。按当前波动速度估计还需约 1-1.5 小时；若后续慢速窗口或最终验证/保存更慢，保守约 1.5-2.5 小时。

2026-06-09 19:20 最终阶段同步：
- 当前训练推进到本次恢复约 `3824 / 4000` step，折算 global 约 `9824 / 10000`；剩余约 `176` step。最新可靠 checkpoint 仍为已验证的 `step_9500.pt`，下一关键点为最终 global `10000`。
- GPU 约束：VIVID PID 912 仍只在物理 GPU1（bus `00000000:05:00.0`）运行，GPU0 显存 `0MiB`，GPU1 显存约 `12501MiB`，符合“只用 gpu1”。
- 温度/守卫：18:50-19:20 资源守卫全部 OK，GPU1 温度约 `62-66C`；19:20 手动快照 GPU1 `temp=62C`、util `94%`、显存约 `12501MiB`。未见 traceback / RuntimeError / CUDA OOM / GPU lost。
- 结果判断：最近训练 loss 约 `0.0222-0.0269`，仍在正常低损失区间；当前结果符合预期，无新的失败 case。
- ETA：距离 global `10000` 约 `176` step。按当前速度预计约 25-45 分钟到最终训练步，之后还需最终验证/保存/退出收尾；保守估计完整结束约 45-75 分钟。

2026-06-09 20:00 最终完成同步（`null-as-negative` 10k 已完成并停止）：
- 训练已完成 global `10000 / 10000`，最终日志记录 `Step 10000: val_loss = 0.0258`，随后保存 `best.pt`、`step_10000.pt`、`final.pt`，并以 `exitcode 0` 正常退出。
- 只读加载确认：`step_10000.pt`、`best.pt`、`final.pt` 内部均为 `global_step=10000`、`best_val_loss=0.025843591105839805`。最终 best 已从 global 8500 的 `0.026174051193576194` 进一步推进到 global 10000。
- 输出文件确认：`outputs/ablation_ums_null_as_negative_12label/checkpoints/final.pt`、`step_10000.pt`、`best.pt` 均已落盘，时间戳约为 2026-06-09 19:58。
- GPU/任务状态：20:00 手动快照显示 GPU0/GPU1 显存均为 `0MiB`，无 compute process；计划任务 `VIVID_null_as_negative_gpu1` 为 `Ready`。当前没有训练进程在跑，符合“跑完就停止”。
- 温度/守卫：19:30-19:55 资源守卫全部 OK；最终阶段未见 traceback / RuntimeError / CUDA OOM / GPU lost。
- 结果判断：最终 val_loss `0.025843591105839805` 刷新全程 best，优于 `8500` best `0.026174051193576194`，结果符合预期且偏正向。没有新的失败 case；本轮系统崩溃相关 case study 已在 2026-06-09 10:58 归档。
- 当前剩余：`null-as-negative` 10k 训练剩余 `0` step。按用户后续收敛后的目标，本轮当前训练已完成并停止；未启动新的实验。

2026-06-09 21:05 调度更新（GPU0 承接短 LP，GPU1 优先 random-LM 长任务）：
- 用户最新指令允许当前短任务使用 GPU0，并要求 GPU1 尽快启动 `random-LM same-architecture` 预训练；因此当前执行策略从严格 GPU1-only 调整为：GPU0 跑 `lp_ums_null_as_negative`，GPU1 跑 `ablation_ums_random_lm_12label`。
- 已将旧队列收口开关 `StopAfterAnsmaskFinal` 恢复为 `false`，并给 `scripts/answerability_gpu1_queue_once.ps1` 增加运行中检测：当 `null-as-negative` LP 或 random-LM 已有进程时，队列不再重复启动同一任务。
- `VIVID_lp_null_as_negative_gpu1` 在约 1301/3000 step 处按用户调度要求停止，部分输出与日志已归档到 `History/20260609_2053_lp_null_gpu1_to_gpu0_migration/`。这不是模型失败；只是为 GPU1 让出长任务。
- 已用 `scripts/run_lp_null_as_negative_gpu0.cmd` 在 GPU0 启动干净重跑，active 日志为 `outputs/logs/lp_ums_null_as_negative_12label_train_gpu0.log`。21:05 左右已推进到约 1500/3000，`metrics_step_800.json` 已落盘，GPU0 约 1.9GB 显存。
- 已启动 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。截至 21:05，进程仍在 random-init Qwen 架构加载阶段，日志停在 `ModelScope available, will try ModelScope first`，尚未占用 GPU1 显存；进程仍有 CPU/内存活动，暂未判定失败。
- 当前剩余目标产物：`lp_ums_null_as_negative` final/metrics、`counterfactual-prefix eval`、`random-LM final/LP`、`field-paraphrase eval`，以及后续成本/汇总表整理。

2026-06-09 21:28 random-LM GPU1 启动排查与修复：
- 用户询问为什么 GPU1 一直启动不了。系统化排查结论：GPU1/CUDA 本身可用；问题是 `llm_random_init: true` 路径在 `models/vivid_model.py` 中先用 CPU 执行 `AutoModelForCausalLM.from_config(...)` 构建完整 Qwen 1.5B 随机权重，外层 `model.to(cuda)` 必须等 CPU 构建完成后才执行。因此 GPU1 在长时间内显示 0MiB，看起来像没有启动。
- 轻量探针显示同一本地 Qwen 路径的 tokenizer/config 加载很快：tokenizer 约 3.35s，config 约 0.003s；卡点不是本地模型路径或 tokenizer，而是随机 1.5B 模型构建。
- 旧 random-LM 进程最终完成 CPU 构建并短暂跑到约 step 7，但按修复计划中止；相关日志和部分输出归档到 `History/20260609_2119_random_lm_cpu_init_slow_restart/`，case study 已写入该目录。
- 已修复 `models/vivid_model.py`：random-init 分支在构建 LLM 时临时设置 PyTorch default device 为目标 CUDA 设备，并传入目标 dtype，使随机权重直接在 GPU1 可见设备上创建。patched 日志确认出现 `Random-init LLM will be constructed on cuda with dtype torch.bfloat16`，随后快速完成 `LLM loaded` 并进入训练。
- 当前 `VIVID_random_lm_gpu1` 已在物理 GPU1（bus `00000000:05:00.0`）运行，21:25-21:28 快照显示 GPU1 显存约 24.1GB、util 约 93%、温度约 67C；日志已推进到约 step 19/10000，loss 约 11.2853，属于 random-LM 起步阶段的预期高 loss。
- `counterfactual-prefix` 在 21:20 队列竞态中被误启动，已为 random-LM 让路停止；日志归档到 `History/20260609_2120_cf_prefix_preempted_for_random_lm/`。队列顺序已改为 `null LP final` 后优先 random-LM train，并手动验证 21:28 队列写入 `not launching VIVID_random_lm_gpu1: random-LM already running`。
- `lp_ums_null_as_negative` 已在 GPU0 完成：`outputs/lp_ums_null_as_negative_12label/metrics_final.json`，macro-AUC `0.8334`、macro-F1 `0.9133`，结果符合预期。当前 GPU0 空闲。
- 当前剩余目标产物：`random-LM final`、`random-LM LP`、`counterfactual-prefix eval`、`field-paraphrase eval`，以及后续成本/汇总表整理。按当前 random-LM 早期速度粗估，该 10k 长任务约需 18-24 小时；等 step 500/1000 验证后重新校准。

2026-06-09 21:36 状态同步（random-LM 已稳定占用 GPU1）：
- 当前阶段：`random-LM same-architecture` 预训练正在运行，任务名 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。
- GPU 约束：GPU0 空闲，GPU1（bus `00000000:05:00.0`）上仅剩 random-LM 训练进程 PID 25132；21:35 快照 GPU1 显存约 `24107MiB / 24576MiB`、util 约 `98%`、temp `72C`，符合“GPU1 跑长任务”的最新调度要求。
- 训练进度：日志已推进到约 `102 / 10000` step。loss 从 step 10 的约 `11.2853` 逐步降到 step 100 的约 `6.4221`，这是 random-init LM 对照的预期高起点下降趋势；当前结果符合预期。
- 队列验证：21:30 与 21:35 队列均识别 `random_lm_running=True` 并写入 `not launching VIVID_random_lm_gpu1: random-LM already running`，未再启动 `counterfactual-prefix` 或其他 GPU1 任务。
- 资源守卫：21:25、21:30、21:35 均 OK；最高温度约 `72C`，低于 `83C` hard stop，无 CUDA OOM / traceback / GPU lost。
- ETA：前 100 step 约 11.75 分钟，粗略约 7.0s/step；剩余约 `9898` step，当前估算 random-LM 预训练还需约 `19-22` 小时（含后续验证/保存开销）。下一关键点是 step 500 验证/best，step 1000 checkpoint 后会重新校准。
- 当前剩余目标产物：`random-LM final`、`random-LM LP`、`counterfactual-prefix eval`、`field-paraphrase eval`，以及成本/汇总表整理。

2026-06-09 22:00 小时同步（random-LM 继续运行；后续队列风险已修复）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务名 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）仅有 random-LM 训练进程 PID 25132。22:00 手动快照显示 GPU1 显存约 `24107MiB / 24576MiB`，温度约 `58-60C`；显存占用稳定，符合 GPU1 承接长任务的调度要求。
- 训练进度：日志已推进到约 `220 / 10000` step，剩余约 `9780` step。loss 从 step 10 的约 `11.2853`、step 100 的约 `6.4221`，继续下降到 step 220 的约 `5.1003`；这是 random-init LM 对照的预期下降趋势，当前结果符合预期。
- 队列与结构维护：旁路多智能体审计指出两个后续风险：random-LM 完成后队列会先跑 `counterfactual-prefix` 而不是 `lp_ums_random_lm`，以及 random-LM runner 中断后没有自动 resume。已修复 `scripts/answerability_gpu1_queue_once.ps1`：顺序改为 `random-LM final -> random-LM LP -> counterfactual-prefix -> field-paraphrase`，并增加 random-LM LP / cf-prefix / field-paraphrase 的运行中检测，避免 5 分钟队列重复触发。已修复 `scripts/run_random_lm_gpu1.cmd`：若存在 `outputs/ablation_ums_random_lm_12label/checkpoints/step_*.pt`，重启时自动从最新 step checkpoint resume。
- 队列验证：21:59 手动运行队列、22:00 定时队列均识别 `random_lm_running=True`，写入 `not launching VIVID_random_lm_gpu1: random-LM already running`，没有误启动其他任务。
- 资源守卫：21:40-22:00 均 OK；最高温度约 `71C`，低于 `83C` hard stop；未见 CUDA OOM / traceback / GPU lost。当前还未到 step 500 首次验证，因此无 random-LM checkpoint 是正常现象。
- ETA：从启动到 step 220 的平均速度约 10-12s/step，较 21:36 的早期粗估更慢。保守估计 random-LM 预训练剩余约 `30-38` 小时；下一关键点是 step 500 首次验证/best，预计约 1-1.5 小时后到达，届时重新校准全程 ETA。
- 当前剩余目标产物：`random-LM final`、`random-LM LP`、`counterfactual-prefix eval`、`field-paraphrase eval`，以及成本/汇总表整理。

2026-06-09 22:23 batch-size 调整同步（按用户要求降低 random-LM 显存）：
- 用户指出此前实验显存没有这么高，要求调小 batch size。配置审计结果：`ablation_ums_ansmask_12label` 使用 `batch_size=2, gradient_accumulation_steps=16`；`ablation_ums_null_as_negative_12label`、`ablation_A_ums_12label`、`ablation_A_freetext_12label`、`ablation_spd_g2_12label` 等多数 1.5B LLM 预训练使用 `batch_size=4, gradient_accumulation_steps=8`；LP 任务使用 `batch_size=16, gradient_accumulation_steps=2`，但 LP 不加载完整 LLM，显存不可直接比较。
- 已将 `configs/ablation_ums_random_lm_12label.yaml` 从 `batch_size=4, gradient_accumulation_steps=8` 改为 `batch_size=2, gradient_accumulation_steps=16`，保持有效 batch size 为 32，同时降低单个 microbatch 的激活显存。
- 为了让新 batch 设置生效，已停止 22:14 前的 batch-4 random-LM partial run；该 run 尚未到 step 500，因此无 checkpoint 可 resume。partial 日志已归档到 `History/20260609_2211_random_lm_batch_size_tuning/`，这是资源调参重启，不是模型失败。
- 已重新启动 `VIVID_random_lm_gpu1`，新日志确认 `Train batches: 14500`，对应 batch size 2；模型仍在 GPU1 上构建，任务状态 Running。
- 新配置运行状态：22:23 日志已推进到约 `13 / 10000` step，step 10 loss 约 `11.2202`，与 random-init LM 起步阶段预期一致。GPU1 显存约 `15985MiB / 24576MiB`，相比 batch-4 时约 `24107MiB` 明显下降；GPU0 仍为 `0MiB`。
- 资源守卫：22:20 守卫显示 GPU1 显存约 `14467MiB`、温度约 `51C`、总功耗约 `107W`，OK，无 stop action。
- ETA：batch 变小后单 step 速度预计变慢。当前刚重启，ETA 暂不以 13 step 的早期窗口精算；下一关键点仍是 step 500 首次验证/best，之后重新估计 full random-LM 和全目标剩余时间。

2026-06-09 23:00 小时同步（batch-2 random-LM 稳定运行）：
- 当前阶段：`random-LM same-architecture` 预训练以 `batch_size=2, gradient_accumulation_steps=16` 继续运行，任务名 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）仅有 random-LM 训练进程 PID 22104。23:00 快照显示 GPU1 显存约 `15985MiB / 24576MiB`、温度约 `58C`，比 batch-4 时约 `24107MiB` 明显降低。
- 训练进度：日志已推进到约 `114 / 10000` step，剩余约 `9886` step。loss 从 step 10 的 `11.2202` 下降到 step 50 的 `8.1624`、step 80 的 `6.8766`、step 110 的 `6.1547`，符合 random-init LM 对照的预期高起点下降趋势；当前结果符合预期。
- 验证/checkpoint：尚未到 step 500 首次验证，因此当前没有 random-LM checkpoint 属正常现象；下一关键点仍是 step 500 的首次 validation/best。
- 队列验证：22:35、22:40、22:45、22:50、22:55、23:00 队列均识别 `random_lm_running=True` 并写入 `not launching VIVID_random_lm_gpu1: random-LM already running`，未误启动 LP、counterfactual-prefix 或 field-paraphrase。
- 资源守卫：22:40-23:00 均 OK；GPU1 温度约 `58-60C`，总功耗约 `104-141W`，远低于 hard stop；未见 traceback / RuntimeError / CUDA OOM / GPU lost。
- ETA：batch-2 当前窗口约 20-25s/optimizer step，step 500 预计还需约 2.5-3.5 小时。按该速度粗估，random-LM 预训练剩余约 `55-70` 小时；如果后续吞吐回升，ETA 会相应下调。全目标剩余仍包括 `random-LM final`、`random-LM LP`、`counterfactual-prefix eval`、`field-paraphrase eval` 和成本/汇总表整理，保守约 `58-74` 小时。

2026-06-10 00:45 小时同步（random-LM 首次验证与 best checkpoint 已通过）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务名 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）仅有 random-LM 训练进程 PID 22104。00:45 快照显示 GPU1 显存约 `17511MiB / 24576MiB`、util 约 `74%`、温度约 `61C`；资源守卫 00:15-00:45 均 OK，无 stop action。
- 训练进度：日志已推进到约 `516 / 10000` step，剩余约 `9484` step。step 500 验证结束后训练继续到 501+，说明验证/保存后没有卡死。
- 验证/checkpoint：首次 validation 已完成，日志记录 `Step 500: val_loss = 3.7193`，并保存 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`。只读加载确认 `best.pt` 内部 `global_step=500`、`best_val_loss=3.7193317666053773`、文件大小约 `1.07GB`。
- 结果判断：random-init LM 对照的绝对 loss 明显高于加载预训练 LLM 的主线实验是预期内；本轮关注的是同架构随机初始化对照是否稳定收敛。loss 从 step 10 的约 `11.2202`、step 110 的约 `6.1547`，下降到 step 500 validation `3.7193`，趋势健康，当前结果符合预期。
- 队列验证：00:00-00:45 队列均识别 `random_lm_running=True` 并写入 `not launching VIVID_random_lm_gpu1: random-LM already running`，未误启动 LP、counterfactual-prefix 或 field-paraphrase。
- 异常检查：active 训练日志未检出 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 等错误模式。当前没有新的失败 case 需要 case study。
- ETA：从 batch-2 重启到 step 500 加首次验证约 2.45 小时，实际速度优于 23:00 的保守估计。按当前窗口，random-LM 预训练剩余约 `40-50` 小时；考虑后续每 500 step validation、每 1000 step checkpoint 与吞吐波动，全目标剩余保守估计约 `42-55` 小时。当前仍剩 `random-LM final`、`random-LM LP`、`counterfactual-prefix eval`、`field-paraphrase eval` 和成本/汇总表整理。

2026-06-10 00:55 风险收敛记录（random-LM 500-1000 step 续跑窗口）：
- 旁路审计指出当前已生成 `best.pt` 但尚未生成 `step_1000.pt`，原 `scripts/run_random_lm_gpu1.cmd` 只会自动寻找 `step_*.pt` 续跑；若恰好在 500-1000 之间中断，自动续跑可能不会使用 step 500 的 `best.pt`。
- 已将 runner 改为优先选择最新 `step_*.pt`，若不存在 step checkpoint 但存在 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`，则 fallback 到 `best.pt` 续跑。该改动只影响未来异常重启，不影响当前 PID 22104 正在运行的训练过程。
- dry-run 确认 fallback 选择为 `outputs\ablation_ums_random_lm_12label\checkpoints\best.pt`；同时 GPU1 仍在运行 random-LM，日志已推进到约 `535 / 10000` step。

2026-06-10 01:00 小时同步（random-LM 500 后稳定推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务名 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）仅有 random-LM 训练进程 PID 22104。01:00 快照显示 GPU1 显存约 `17511MiB / 24576MiB`、util 约 `85%`、温度约 `61C`，符合 GPU1 长任务调度。
- 训练进度：日志已推进到约 `550 / 10000` step，剩余约 `9450` step。step 500 验证与 `best.pt` 保存后，训练继续到 550，说明验证/保存后持续稳定推进。
- 验证/checkpoint：当前最新可靠 checkpoint 仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`，内部已在 00:45 同步中确认 `global_step=500`、`best_val_loss=3.7193317666053773`；`step_1000.pt` 尚未生成，属正常进度。
- 结果判断：最新训练 loss 约 `3.67-3.72`，低于 step 500 前后的约 `3.79-3.82` 区间；作为 random-init LLM 对照，当前下降趋势正常，结果符合预期。
- 队列/守卫：01:00 队列识别 `random_lm_running=True` 并写入 `not launching VIVID_random_lm_gpu1: random-LM already running`；01:00 资源守卫 OK，无 stop action。active 日志尾部未检出 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`。
- ETA：step 500 后当前窗口约 `22-26s/optimizer step`，距离 `step_1000.pt` 约 `450` step，预计约 `3-3.5` 小时到 step 1000 验证/保存窗口。按该移动窗口保守估算，random-LM 预训练剩余约 `58-68` 小时；完整目标仍需 `random-LM final`、`random-LM LP`、`counterfactual-prefix eval`、`field-paraphrase eval` 与成本/汇总表整理，保守约 `60-72` 小时。

2026-06-10 02:00 小时同步（random-LM batch-2 长跑正常）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务名 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）仅有 random-LM 训练进程 PID 22104。02:01 手动快照显示 GPU1 显存约 `17511MiB / 24576MiB`、util 约 `80%`、温度约 `62C`，仍符合 GPU1 长任务调度。
- 训练进度：日志已推进到约 `679 / 10000` step，剩余约 `9321` step。log mtime 为 02:01:56，说明训练仍在写日志并继续推进。
- 验证/checkpoint：当前最新可靠 checkpoint 仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`，来自 step 500 validation，`step_1000.pt` 尚未生成，属正常进度。
- 结果判断：step 500 validation 为 `val_loss=3.7193`；step 670 后训练 loss 约 `3.3426`，继续低于 step 500 附近水平。作为 random-init LLM 对照，loss 高于 pretrained frozen-LM 主线实验是预期内；当前趋势健康，结果符合预期。
- 关于 batch 设置：当前 random-LM 使用 `batch_size=2, gradient_accumulation_steps=16`，有效 batch size 仍为 32；后续 random-LM LP 使用 `batch_size=16, gradient_accumulation_steps=2`，但 LP 不加载完整 LLM，显存和预训练不可直接比较。这不改变 `miccai2026.md` 要求的 random-LM same-architecture 对照定义，也不影响最终 LP/验证口径。
- 队列/守卫：02:00 队列识别 `random_lm_running=True` 并写入 `not launching VIVID_random_lm_gpu1: random-LM already running`，未误启动 LP、counterfactual-prefix 或 field-paraphrase。02:00 资源守卫 OK，无 stop action；active 日志尾部未检出 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost` 或 `Exception`。
- ETA：step 500 后移动窗口约 `24-28s/optimizer step`，距离 `step_1000.pt` 约 `321` step，预计约 `2.3-2.8` 小时到 step 1000，再加 validation/save 约为 `2.5-3.2` 小时。按当前窗口保守估算，random-LM 预训练剩余约 `58-75` 小时；完整目标仍需 `random-LM final`、`random-LM LP`、`counterfactual-prefix eval`、`field-paraphrase eval` 与成本/汇总表整理，保守约 `60-80` 小时。
- 多智能体只读侧查：剩余产物与队列顺序一致，未发现硬错配。`lp_ums_random_lm_12label.yaml` 与其他 LP 配置一致使用对应预训练的 `best.pt` 初始化；队列等待 random-LM `final.pt` 只是确保预训练完整结束后再启动下游 LP，不代表 LP 必须从 `final.pt` 初始化。

2026-06-10 03:00 小时同步（random-LM 稳定推进，尚未到 step 1000）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务名 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）仅有 random-LM 训练进程 PID 22104。03:01 手动快照显示 GPU1 显存约 `17511MiB / 24576MiB`、温度约 `61C`；瞬时 util 可波动，但进程、显存和日志均显示训练仍在推进。
- 训练进度：日志已推进到约 `818 / 10000` step，剩余约 `9182` step；距离下一关键 `step_1000.pt` 约 `182` step。log mtime 为 03:01:38，说明训练持续写入。
- 验证/checkpoint：当前最新可靠 checkpoint 仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`，来自 step 500 validation；`step_1000.pt` 和 `final.pt` 尚未生成，属当前进度下的正常状态。
- 结果判断：step 500 validation 为 `val_loss=3.7193`；最近训练 loss 从 02:00 附近的约 `3.34` 继续下降到约 `2.9843`。作为 random-init LLM 对照，当前下降趋势健康，结果符合预期。
- 队列/守卫：03:00 队列识别 `random_lm_running=True` 并写入 `not launching VIVID_random_lm_gpu1: random-LM already running`，未误启动 random-LM LP、counterfactual-prefix 或 field-paraphrase。03:00 资源守卫 OK，无 stop action；active 日志尾部未检出 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`Exception`、`failed` 或 `error`。
- 多智能体只读侧查：已完成的关键产物包括 no-mask UMS anchor、answerability-mask branch、null-as-negative branch、counterfactual schema grounding 和 schema-key robustness；当前硬缺口仍是 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`。
- 后处理待办：random-LM 与两个 eval 产物齐全后，还需做成本/summary table consolidation、result consolidation 与 claim cleanup；manual audit 相关 CSV/summary 已存在，但人工标注/agreement 仍是 partial，不作为当前自动队列的已完成主证据。
- ETA：按 02:00-03:00 移动窗口约 `20-26s/optimizer step`，距离 `step_1000.pt` 约 `182` step，预计约 `1.1-1.4` 小时进入 step 1000 验证窗口；加 validation/save 后保守约 `1.5-2.2` 小时看到 `step_1000.pt`。random-LM 预训练剩余约 `55-75` 小时；完整目标仍需 `random-LM final`、`random-LM LP`、`counterfactual-prefix eval`、`field-paraphrase eval` 与最终汇总，保守约 `58-82` 小时。

2026-06-10 04:00 小时同步（random-LM 接近 step 1000，训练健康）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务名 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）仅有 random-LM 训练进程 PID 22104。04:00 队列/守卫快照显示 GPU1 显存约 `17511MiB / 24576MiB`，资源守卫 OK，无 stop action。
- 训练进度：04:01-04:02 日志已推进到约 `958 / 10000` step，剩余约 `9042` step；距离下一关键 `step_1000.pt` 约 `42` step。log mtime 为 04:01 后，说明训练仍在写入并继续推进。
- 验证/checkpoint：当前最新可靠 checkpoint 仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`，来自 step 500 validation；`step_1000.pt` 和 `final.pt` 尚未生成，属当前进度下的正常状态。
- 结果判断：step 500 validation 为 `val_loss=3.7193`；最近训练 loss 约 `2.79`，继续低于 03:00 附近的约 `2.98`。作为 random-init LLM 对照，loss 仍高于加载预训练 LLM 的主线是预期内；下降趋势健康，当前结果符合预期。
- 队列/守卫：04:00 队列识别 `random_lm_running=True` 并写入 `not launching VIVID_random_lm_gpu1: random-LM already running`，未误启动 random-LM LP、counterfactual-prefix 或 field-paraphrase。04:00 资源守卫 OK；active 日志尾部未检出 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`Exception`、`failed` 或 `error`。
- 短暂低功耗窗口说明：03:42-03:47 之间曾出现 GPU1 util/power 偏低且日志短暂停顿，但进程 CPU 时间继续增长，并随后从 step 906 推进到 step 958，GPU1 恢复训练负载；当前不构成失败 case，也不需要 case study。
- 多智能体只读侧查：队列顺序复核为 `random-LM final -> random-LM LP -> counterfactual-prefix dependency -> field-paraphrase robustness`；`lp_ums_random_lm_12label.yaml` 读取 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`，而队列用 `final.pt` 作为启动 LP 的完整预训练 gate，未发现硬错配。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：近 50 step 约 `16.8s/step`，近 200 step 约 `24.2s/step`；保守估计约 `15-35` 分钟进入 step 1000 validation/save 窗口，看到 `step_1000.pt` 可能约 `0.5-1.0` 小时。按近 200-400 step 窗口估算，random-LM 预训练剩余约 `60-70` 小时；完整目标仍需 random-LM LP、两个 eval 与最终汇总，保守约 `63-78` 小时。

2026-06-10 04:30 里程碑同步（random-LM step 1000 checkpoint 已落盘）：
- 当前阶段：`random-LM same-architecture` 预训练完成 step 1000 validation/save 后继续运行，任务名仍为 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）仅有 random-LM 训练进程 PID 22104。04:25 资源守卫显示 GPU1 显存约 `17511MiB / 24576MiB`、util `99%`、温度 `60C`，无 stop action。
- 训练进度：step 1000 validation 于 04:26 左右完成，随后训练已继续到约 `1008 / 10000` step；当前还剩约 `8992` step 到 random-LM pretrain final。
- 验证/checkpoint：日志记录 `Step 1000: val_loss = 2.5412`，并保存 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 与 `outputs/ablation_ums_random_lm_12label/checkpoints/step_1000.pt`。只读加载确认二者内部均为 `global_step=1000`、`best_val_loss=2.541152367115021`。
- 结果判断：step 500 validation 为 `val_loss=3.7193`，step 1000 降至 `2.5412`，改善明显；训练 loss 也在 step 1000 附近保持约 `2.69`。作为 random-init LLM 同架构对照，当前趋势健康，符合预期。
- 队列/守卫：04:15、04:20、04:25 队列均识别 `random_lm_running=True`，未误启动 random-LM LP、counterfactual-prefix 或 field-paraphrase；resource guard 均 OK。active 日志尾部未检出 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`Exception`、`failed` 或 `error`。
- 多智能体只读侧查：当前文件层面仍只有 random-LM 中间 checkpoint；队列 gate 正确，会继续等待 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt` 后再启动 `VIVID_lp_random_lm_gpu1`，随后顺序进入 counterfactual-prefix dependency 与 field-paraphrase robustness。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：排除刚过 validation 后的短期波动，近 100-500 step 窗口约 `25-27s/optimizer step`。random-LM pretrain 剩余约 `63-70` 小时；完整目标还需 random-LM LP、两个 eval 与最终汇总，保守约 `66-80` 小时。

2026-06-10 05:00 小时同步（random-LM step 1000 后稳定续跑）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务名 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）仅有 random-LM 训练进程 PID 22104。05:00 快照显示 GPU1 显存约 `17511MiB / 24576MiB`、util `57%`、温度 `63C`；resource guard OK，无 stop action。
- 训练进度：05:01 日志推进到约 `1074 / 10000` step，剩余约 `8926` step；距离下一次 validation（step 1500）约 `426` step。log mtime 为 05:01 后，说明训练仍在写入。
- 验证/checkpoint：当前可靠 checkpoint 为 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 与 `outputs/ablation_ums_random_lm_12label/checkpoints/step_1000.pt`，二者均已确认 `global_step=1000`、`best_val_loss=2.541152367115021`；`final.pt` 尚未生成。
- 结果判断：step 500 validation `val_loss=3.7193`，step 1000 validation `val_loss=2.5412`；05:00 附近训练 loss 约 `2.53`。random-init LLM 对照仍在按预期下降，目前结果符合预期。
- 队列/守卫：04:50、04:55、05:00 队列均识别 `random_lm_running=True`，未误启动 random-LM LP、counterfactual-prefix 或 field-paraphrase。04:55 与 05:00 resource guard 均 OK。active 日志尾部未检出 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`Exception`、`failed` 或 `error`。
- 多智能体只读侧查：文件与队列复核一致；当前 random-LM pretrain 尚未 final，队列 gate 正确等待 `final.pt` 后再启动 `VIVID_lp_random_lm_gpu1`，随后进入 counterfactual-prefix dependency 与 field-paraphrase robustness。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：近 50-500 step 窗口约 `27-32s/optimizer step`；random-LM pretrain 剩余约 `67-80` 小时，若后续吞吐回升会下调。完整目标还需 random-LM LP、两个 eval 与最终汇总，保守约 `70-90` 小时。

2026-06-10 06:00 小时同步（random-LM 持续健康推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务名 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）仅有 random-LM 训练进程 PID 22104。06:00 快照显示 GPU1 显存约 `17511MiB / 24576MiB`、util `29-40%`、温度 `62-65C`；resource guard OK，无 stop action。
- 训练进度：06:01 日志推进到约 `1209 / 10000` step，剩余约 `8791` step；距离下一次 validation（step 1500）约 `291` step。log mtime 为 06:00 后，说明训练仍在写入。
- 验证/checkpoint：当前可靠 checkpoint 仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 与 `outputs/ablation_ums_random_lm_12label/checkpoints/step_1000.pt`，二者均已确认 `global_step=1000`、`best_val_loss=2.541152367115021`；`final.pt` 尚未生成。
- 结果判断：step 500 validation `val_loss=3.7193`，step 1000 validation `val_loss=2.5412`；06:00 附近训练 loss 约 `2.38`。random-init LLM 对照继续按预期下降，目前结果符合预期。
- 队列/守卫：05:50、05:55、06:00 队列均识别 `random_lm_running=True`，未误启动 random-LM LP、counterfactual-prefix 或 field-paraphrase。05:55 与 06:00 resource guard 均 OK。active 日志尾部未检出 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`Exception`、`failed` 或 `error`。
- 多智能体只读侧查：文件与队列复核一致；当前 random-LM pretrain 尚未 final，队列 gate 正确等待 `final.pt` 后再启动 `VIVID_lp_random_lm_gpu1`，随后进入 counterfactual-prefix dependency 与 field-paraphrase robustness。历史 GPU-lost 记录不属于当前状态。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：近 50-500 step 窗口约 `23.5-27.4s/optimizer step`；random-LM pretrain 剩余约 `58-67` 小时。完整目标还需 random-LM LP、两个 eval 与最终汇总，保守约 `61-77` 小时。

2026-06-10 07:00 小时同步（random-LM 接近 step 1500 validation）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务名 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）仅有 random-LM 训练进程 PID 22104。07:00 快照显示 GPU1 显存约 `17511MiB / 24576MiB`、util `25-96%`、温度 `60-66C`；resource guard OK，无 stop action。
- 训练进度：07:02 日志推进到约 `1370 / 10000` step，剩余约 `8630` step；距离下一次 validation（step 1500）约 `130` step。log mtime 为 07:02，说明训练仍在写入。
- 验证/checkpoint：当前可靠 checkpoint 仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 与 `outputs/ablation_ums_random_lm_12label/checkpoints/step_1000.pt`；`final.pt` 尚未生成，step 1500 validation 尚未开始。
- 结果判断：step 500 validation `val_loss=3.7193`，step 1000 validation `val_loss=2.5412`；07:00 附近训练 loss 约 `2.11-2.24`。random-init LLM 对照继续按预期下降，目前结果符合预期。
- 队列/守卫：06:50、06:55、07:00 队列均识别 `random_lm_running=True`，未误启动 random-LM LP、counterfactual-prefix 或 field-paraphrase。06:55 与 07:00 resource guard 均 OK。active 日志尾部未检出 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`Exception`、`failed` 或 `error`。
- 多智能体只读侧查：文件与队列复核一致；当前 random-LM pretrain 尚未 final，队列 gate 正确等待 `final.pt` 后再启动 `VIVID_lp_random_lm_gpu1`，随后进入 counterfactual-prefix dependency 与 field-paraphrase robustness。历史 GPU-lost 记录不属于当前状态。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：近 200-500 step 稳定窗口约 `22.8-26.4s/optimizer step`；预计约 `45-60` 分钟进入 step 1500 validation/save 窗口。random-LM pretrain 剩余约 `55-64` 小时；完整目标还需 random-LM LP、两个 eval 与最终汇总，保守约 `58-74` 小时。

2026-06-10 08:00 里程碑同步（random-LM step 1500 validation 已完成）：
- 当前阶段：`random-LM same-architecture` 预训练完成 step 1500 validation 后继续运行，任务名仍为 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）仅有 random-LM 训练进程 PID 22104。08:00-08:05 资源守卫显示 GPU1 显存约 `17511MiB / 24576MiB`、温度 `58-62C`，无 stop action。
- 训练进度：step 1500 validation 于 07:58 左右完成，随后训练已继续到约 `1524 / 10000` step；当前还剩约 `8476` step 到 random-LM pretrain final，距离下一次 step checkpoint（step 2000）约 `476` step。
- 验证/checkpoint：日志记录 `Step 1500: val_loss = 1.8702`，并保存/更新 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`。只读加载确认 `best.pt` 内部为 `global_step=1500`、`best_val_loss=1.8702103219032287`；`step_1000.pt` 仍是最新 step checkpoint，step 1500 按当前保存策略只更新 best，不生成 `step_1500.pt`。
- 结果判断：validation 从 step 500 的 `3.7193`、step 1000 的 `2.5412` 继续降到 step 1500 的 `1.8702`；训练 loss 在 step 1500 后约 `1.91-1.95`。random-init LLM 同架构对照继续按预期收敛，目前结果符合预期。
- 队列/守卫：07:55、08:00、08:05 队列均识别 `random_lm_running=True`，未误启动 random-LM LP、counterfactual-prefix 或 field-paraphrase。08:00 与 08:05 resource guard 均 OK。active 日志尾部未检出 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`Exception`、`failed` 或 `error`。
- 多智能体只读侧查：文件与队列复核一致；当前 random-LM pretrain 尚未 final，队列 gate 正确等待 `final.pt` 后再启动 `VIVID_lp_random_lm_gpu1`，随后进入 counterfactual-prefix dependency 与 field-paraphrase robustness。历史 GPU-lost 记录不属于当前状态。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：排除刚过 validation 的短期波动，近 100-500 step 稳定窗口约 `24.4-26.8s/optimizer step`；random-LM pretrain 剩余约 `58-64` 小时。完整目标还需 random-LM LP、两个 eval 与最终汇总，保守约 `61-74` 小时。

2026-06-10 09:00 小时同步（random-LM step 1500 后稳定续跑）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务名 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）仅有 random-LM 训练进程 PID 22104。09:00 快照显示 GPU1 显存约 `17511MiB / 24576MiB`、util 约 `58-60%`、温度 `59C`；resource guard OK，无 stop action。
- 训练进度：09:01 日志推进到约 `1647 / 10000` step，剩余约 `8353` step；距离下一次 step checkpoint 与 validation（step 2000）约 `353` step。log mtime 为 09:01 后，说明训练仍在写入。
- 验证/checkpoint：当前可靠 checkpoint 为 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`，内部已确认 `global_step=1500`、`best_val_loss=1.8702103219032287`；`step_1000.pt` 仍是最新 step checkpoint，`final.pt` 尚未生成。
- 结果判断：step 500 validation `val_loss=3.7193`，step 1000 validation `val_loss=2.5412`，step 1500 validation `val_loss=1.8702`；09:00 附近训练 loss 约 `1.92`。random-init LLM 同架构对照继续按预期收敛，目前结果符合预期。
- 队列/守卫：08:55 与 09:00 队列均识别 `random_lm_running=True`，未误启动 random-LM LP、counterfactual-prefix 或 field-paraphrase。08:55 与 09:00 resource guard 均 OK。active 日志尾部未检出 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`Exception`、`failed` 或 `error`。
- 多智能体只读侧查：文件与队列复核一致；当前 random-LM pretrain 尚未 final，队列 gate 正确等待 `final.pt` 后再启动 `VIVID_lp_random_lm_gpu1`，随后进入 counterfactual-prefix dependency 与 field-paraphrase robustness。历史 GPU-lost 记录不属于当前状态。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：近 50-500 step 稳定窗口约 `24.1-26.7s/optimizer step`；random-LM pretrain 剩余约 `56-62` 小时。完整目标还需 random-LM LP、两个 eval 与最终汇总，保守约 `59-72` 小时。

2026-06-10 10:20 小时同步与重启恢复记录（random-LM 从最新 checkpoint 恢复）：
- 当前阶段：`random-LM same-architecture` 预训练仍是当前主任务；10:00 前最后可见训练日志曾推进到约 `1727 / 10000`，但当时尚未生成 `step_2000.pt`，因此可恢复的最新可靠 checkpoint 仍是 step 1500 的 `best.pt`。
- 重启/失败现象：10:00 左右 GPU1 变为空闲，队列检测到 `random_lm_running=False` 并尝试重启。10:00 首次重启在 model creation 早期退出，日志记录 `conda run ... failed` 与 exit code `1073807364`。这不是新的 CUDA OOM 或 `GPU is lost`，但属于重启恢复失败案例。
- Case study：已保存 `History/20260610_1000_random_lm_reboot_resume_checkpoint_fix/case_study.md`。根因是 `scripts/run_random_lm_gpu1.cmd` 原先优先选择 `step_*.pt`，会选到较旧的 `step_1000.pt`，而不是更新的 `best.pt`。
- 修复动作：已把 runner 改为在 `step_*.pt` 与 `best.pt` 中按 `LastWriteTime` 选择最新 checkpoint；随后终止误从 `step_1000.pt` 拉起的任务，并重新启动 `VIVID_random_lm_gpu1`。
- 恢复验证：10:15 后新日志显示 `resume random-LM from ...\best.pt`、`Checkpoint loaded from ...\best.pt`、`Resuming from step 1500`。10:20 快照显示 GPU1（bus `00000000:05:00.0`）由 PID 20324 占用约 `15221MiB / 24576MiB`，当前恢复段已推进到 `8 / 8500` local progress step；10:20 队列识别 `random_lm_running=True`，resource guard OK。
- 结果判断：step 500/1000/1500 validation 仍为 `3.7193 -> 2.5412 -> 1.8702`，符合 random-init LLM 对照逐步收敛预期。未checkpoint的 step 1500 到约 step 1727 区间无法恢复，但不会改变实验定义和验证口径。
- 当前硬缺口：仍需 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：从已恢复的 step 1500 checkpoint 继续计算，还剩约 `8492` optimizer steps 到 random-LM pretrain final。按重启前稳定窗口约 `26-32s/step` 粗估，random-LM pretrain 约还需 `61-76` 小时；完整目标保守约 `64-86` 小时。

2026-06-10 11:00 小时同步（random-LM 重启恢复后健康续跑）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务名 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本轮由主线程巡检，未新增子智能体。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）仅有 random-LM 训练进程 PID `20324`。11:01 快照显示 GPU1 显存约 `16739MiB / 24576MiB`、util `73%`、温度 `64C`；11:00 resource guard OK，无 stop action。
- 训练进度：从 step 1500 的 `best.pt` 恢复后，当前恢复段推进到 `121 / 8500` local progress step，折算全局约 `1621 / 10000`；还剩约 `8379` step 到 random-LM pretrain final，距离下一次 step checkpoint/validation（step 2000）约 `379` step。log mtime 为 11:01 后，说明训练仍在写入。
- 验证/checkpoint：当前可靠 checkpoint 仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`（step 1500 best）与 `step_1000.pt`；`final.pt` 尚未生成，`step_2000.pt` 尚未到达。
- 结果判断：validation 仍为 step 500/1000/1500 的 `3.7193 -> 2.5412 -> 1.8702`；恢复段最近训练 loss 约 `1.91 -> 1.905`，符合 random-init LLM 同架构对照继续收敛的预期。
- 队列/守卫：10:50、10:55、11:00 队列均识别 `random_lm_running=True`，未误启动 `VIVID_lp_random_lm_gpu1`、counterfactual-prefix 或 field-paraphrase；resource guard 均 OK。当前恢复段未检出 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`Exception`、`failed` 或 `error`。
- Case study：本小时无新增失败案例；10:00 重启恢复问题已记录在 `History/20260610_1000_random_lm_reboot_resume_checkpoint_fix/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：当前窗口约 `21.2s/optimizer step`，预计约 `2.2-2.8` 小时到 step 2000 validation/save 窗口。random-LM pretrain 剩余约 `49-58` 小时；完整目标还需 random-LM LP、两个 eval 与最终汇总，保守约 `53-68` 小时。

2026-06-10 12:00 小时同步（random-LM 重启恢复后继续健康推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务名 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本轮由主线程巡检，未新增子智能体。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）仅有 random-LM 训练进程 PID `20324`。12:01 快照显示 GPU1 显存约 `16739MiB / 24576MiB`、util `66%`、温度 `59C`；12:00 resource guard OK，无 stop action。
- 训练进度：从 step 1500 的 `best.pt` 恢复后，当前恢复段推进到 `242 / 8500` local progress step，折算全局约 `1742 / 10000`；还剩约 `8258` step 到 random-LM pretrain final，距离下一次 step checkpoint/validation（step 2000）约 `258` step。log mtime 为 12:01 后，说明训练仍在写入。
- 验证/checkpoint：当前可靠 checkpoint 仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`（step 1500 best）与 `step_1000.pt`；`final.pt` 尚未生成，`step_2000.pt` 尚未到达。
- 结果判断：validation 仍为 step 500/1000/1500 的 `3.7193 -> 2.5412 -> 1.8702`；恢复段最近训练 loss 约 `1.79-1.82`，符合 random-init LLM 同架构对照继续收敛的预期。
- 队列/守卫：11:50、11:55、12:00 队列均识别 `random_lm_running=True`，未误启动 `VIVID_lp_random_lm_gpu1`、counterfactual-prefix 或 field-paraphrase；12:00 resource guard OK。当前恢复段未检出 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`Exception`、`failed` 或 `error`。
- Case study：本小时无新增失败案例；10:00 重启恢复问题仍以 `History/20260610_1000_random_lm_reboot_resume_checkpoint_fix/case_study.md` 为已记录案例。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：当前窗口约 `25.4s/optimizer step`，预计约 `1.8-2.0` 小时到 step 2000 validation/save 窗口。random-LM pretrain 剩余约 `58-62` 小时；完整目标还需 random-LM LP、两个 eval 与最终汇总，保守约 `61-74` 小时。

2026-06-10 13:00 小时同步（重启后继续运行，已收束到主线程巡检）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务名 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。按用户要求，本轮不再新增子智能体，后续巡检由主线程执行。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）仅有 random-LM 训练进程 PID `20324`。13:00 resource guard 显示 GPU1 显存 `18263MiB / 24576MiB`、util `61%`、温度 `58C`，13:03 现场快照仍为同一 PID；resource guard OK，无 stop action。所有 GPU0 版本计划任务保持 Disabled，当前 Running 的 VIVID 任务只有 `VIVID_random_lm_gpu1`。
- 训练进度：从 step 1500 的 `best.pt` 恢复后，当前恢复段推进到 `399 / 8500` local progress step，折算全局约 `1899 / 10000`；还剩约 `8101` step 到 random-LM pretrain final，距离下一次 step checkpoint/validation（step 2000）约 `101` step。log mtime 为 13:03 后，说明训练仍在写入。
- 验证/checkpoint：当前可靠 checkpoint 仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`（step 1500 best）与 `step_1000.pt`；`final.pt` 尚未生成，`step_2000.pt` 尚未到达。runner 已确认按 `LastWriteTime` 在 `step_*.pt` 与 `best.pt` 中选择最新 checkpoint，避免再次退回旧的 `step_1000.pt`。
- 结果判断：validation 仍为 step 500/1000/1500 的 `3.7193 -> 2.5412 -> 1.8702`；13:00 附近训练 loss 约 `1.62-1.72`。random-init LLM 同架构对照继续按预期收敛，目前结果符合预期。
- 队列/守卫：12:50、12:55、13:00 队列均识别 `random_lm_running=True`，未误启动 `VIVID_lp_random_lm_gpu1`、counterfactual-prefix 或 field-paraphrase；13:00 resource guard OK。当前恢复段日志尾部未检出 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`Exception`、`failed` 或 `error`。
- Case study：本小时无新增失败案例；10:00 重启恢复问题仍以 `History/20260610_1000_random_lm_reboot_resume_checkpoint_fix/case_study.md` 为已记录案例。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：近 50 step 窗口约 `21.65s/optimizer step`；预计约 `0.6-0.8` 小时到 step 2000 validation/save 窗口。按当前短窗口计算 random-LM pretrain 约 `49` 小时，考虑 validation、checkpoint 与吞吐波动，保守约 `50-60` 小时；完整目标还需 random-LM LP、两个 eval 与最终汇总，保守约 `54-70` 小时。

2026-06-10 13:45 里程碑同步（random-LM step 2000 validation/checkpoint 已完成）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务名 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本轮继续由主线程巡检，未新增子智能体。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）仅有 random-LM 训练进程 PID `20324`。13:46 快照显示 GPU1 显存 `18265MiB / 24576MiB`、util `93%`、温度 `59C`；所有 GPU0 版本计划任务保持 Disabled，当前 Running 的训练任务只有 `VIVID_random_lm_gpu1`。
- 训练进度：从 step 1500 的 `best.pt` 恢复后，13:38 左右到达恢复段 `500 / 8500`，折算全局 `2000 / 10000`；13:45 validation 完成后训练已继续到恢复段 `501 / 8500`。random-LM pretrain 还剩约 `7999-8000` optimizer steps 到 `final.pt`。
- 验证/checkpoint：step 2000 validation 完成，日志记录 `Step 2000: val_loss = 1.4182`。已写出 `outputs/ablation_ums_random_lm_12label/checkpoints/step_2000.pt`，并更新 `best.pt`；当前可靠 checkpoint 为 `step_1000.pt`、`best.pt`（step 2000 best）和 `step_2000.pt`，`final.pt` 尚未生成。
- 结果判断：validation loss 继续从 step 500/1000/1500/2000 的 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182` 单调下降；random-init LLM 同架构对照仍按预期收敛，目前结果符合预期。
- 队列/守卫：13:35、13:40、13:45 队列均识别 `random_lm_running=True`，未误启动 `VIVID_lp_random_lm_gpu1`、counterfactual-prefix 或 field-paraphrase；13:45 resource guard OK。当前运行日志未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory` 或 `GPU is lost`。
- Case study：本轮无新增失败案例；10:00 重启恢复问题仍以 `History/20260610_1000_random_lm_reboot_resume_checkpoint_fix/case_study.md` 为已记录案例。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：step 2000 前最近训练窗口约 `20-24s/optimizer step`，另需计入后续 validation/checkpoint 开销；random-LM pretrain 剩余保守约 `46-56` 小时。完整目标还需 random-LM pretrain final、random-LM LP、两个 eval 与最终汇总，保守约 `50-66` 小时。

2026-06-10 14:00 小时同步（已收敛为主线程巡检，random-LM step 2000 后续跑）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务名 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。按用户要求，本轮不再新增子智能体；当前巡检、ETA 与文档同步均由主线程执行。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）仅有 random-LM 训练进程 PID `20324`。14:00 resource guard 显示 GPU1 显存 `18265MiB / 24576MiB`、util `36%`、温度 `63C`，resource guard OK，无 stop action。所有 GPU0 版本计划任务保持 Disabled，当前 Running 的训练任务只有 `VIVID_random_lm_gpu1`。
- 训练进度：从 step 1500 的 `best.pt` 恢复后，当前恢复段推进到约 `544 / 8500` local progress step，折算全局约 `2044 / 10000`；还剩约 `7956` optimizer steps 到 random-LM pretrain `final.pt`，距离下一次 step 2500 validation/checkpoint 约 `456` step。log mtime 为 13:59:53，说明训练仍在写入。
- 验证/checkpoint：step 2000 validation 已完成，日志记录 `Step 2000: val_loss = 1.4182`。当前可靠 checkpoint 为 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`（13:45 更新）与 `step_2000.pt`（13:45 写出），`final.pt` 尚未生成。
- 结果判断：validation loss 继续从 step 500/1000/1500/2000 的 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182` 单调下降；14:00 附近训练 loss 约 `1.5853`。random-init LLM 同架构对照仍按预期收敛，目前结果符合预期。
- 队列/守卫：13:55 与 14:00 队列均识别 `random_lm_running=True`，未误启动 `VIVID_lp_random_lm_gpu1`、counterfactual-prefix 或 field-paraphrase；14:00 resource guard OK。日志中仍可见 10:00 旧的 `conda run ... step_1000.pt failed` 记录，但这是已记录的重启恢复失败案例，不是当前新增失败。
- Case study：本小时无新增失败案例；10:00 重启恢复问题仍以 `History/20260610_1000_random_lm_reboot_resume_checkpoint_fix/case_study.md` 为已记录案例。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：最近 20 个进度样本均值约 `19.6s/optimizer step`，短窗口估计约 `2.5` 小时到 step 2500 validation/checkpoint；考虑 validation/checkpoint 与吞吐波动，random-LM pretrain 剩余保守约 `45-58` 小时。完整目标还需 random-LM pretrain final、random-LM LP、两个 eval 与最终汇总，保守约 `49-68` 小时。

2026-06-10 15:00 小时同步（random-LM step 2000 后继续健康推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务名 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续保持主线程巡检，不新增子智能体。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）仅有 random-LM 训练进程 PID `20324`。15:00 resource guard 显示 GPU1 显存 `18265MiB / 24576MiB`、util `89%`、温度 `61C`，resource guard OK，无 stop action。15:01 现场快照仍显示 GPU1 由同一 PID 占用，GPU0 仍为 `0MiB`。
- 训练进度：从 step 1500 的 `best.pt` 恢复后，当前恢复段推进到 `689 / 8500` local progress step，折算全局约 `2189 / 10000`；还剩约 `7811` optimizer steps 到 random-LM pretrain `final.pt`，距离下一次 step 2500 validation/checkpoint 约 `311` step。log mtime 为 15:01:48，说明训练仍在持续写入。
- 验证/checkpoint：step 2000 validation 已完成，日志记录 `Step 2000: val_loss = 1.4182`。当前可靠 checkpoint 为 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`（13:45 更新）与 `step_2000.pt`（13:45 写出），`final.pt` 尚未生成。
- 结果判断：validation loss 继续从 step 500/1000/1500/2000 的 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182` 单调下降；15:00 附近训练 loss 约 `1.4676`，learning rate 约 `1.92e-05`。random-init LLM 同架构对照仍按预期收敛，目前结果符合预期。
- 队列/守卫：14:50、14:55、15:00 队列均识别 `random_lm_running=True`，未误启动 `VIVID_lp_random_lm_gpu1`、counterfactual-prefix 或 field-paraphrase；15:00 resource guard OK。日志中仍可见 10:00 旧的 `conda run ... step_1000.pt failed` 记录，但这是已记录的重启恢复失败案例，不是当前新增失败。
- Case study：本小时无新增失败案例；10:00 重启恢复问题仍以 `History/20260610_1000_random_lm_reboot_resume_checkpoint_fix/case_study.md` 为已记录案例。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：最近 20/50/100 个进度样本均值约 `20.7/23.0/24.6s/optimizer step`；预计约 `1.8-2.1` 小时到 step 2500 validation/checkpoint。按当前窗口估计 random-LM pretrain 剩余约 `45-53` 小时，考虑 validation/checkpoint 与吞吐波动保守约 `48-58` 小时；完整目标还需 random-LM LP、两个 eval 与最终汇总，保守约 `50-62` 小时。

2026-06-10 16:00 小时同步（主线程巡检，random-LM step 2500 前继续推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务名 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。按用户要求，本小时继续收缩为主线程巡检，不再新增子智能体。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）仅有 random-LM 训练进程 PID `20324`。16:00 resource guard 显示 GPU0 `0MiB`，GPU1 显存 `18265MiB / 24576MiB`、温度 `58C`，resource guard OK，无 stop action。16:01 现场快照仍为同一 GPU1 PID，GPU0 仍空闲。
- 训练进度：从 step 1500 的 `best.pt` 恢复后，当前恢复段推进到 `827 / 8500` local progress step，折算全局约 `2327 / 10000`；还剩约 `7673` optimizer steps 到 random-LM pretrain `final.pt`，距离下一次 step 2500 validation/checkpoint 约 `173` step。log mtime 为 16:01:44，说明训练仍在持续写入。
- 验证/checkpoint：step 2000 validation 已完成，日志记录 `Step 2000: val_loss = 1.4182`。当前可靠 checkpoint 为 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`（13:45 更新）与 `step_2000.pt`（13:45 写出），`final.pt` 尚未生成。
- 结果判断：validation loss 继续从 step 500/1000/1500/2000 的 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182` 单调下降；16:00 附近训练 loss 约 `1.4348`，learning rate 约 `1.91e-05`。random-init LLM 同架构对照仍按预期收敛，目前结果符合预期。
- 队列/守卫：15:50、15:55、16:00 队列均识别 `random_lm_running=True`，未误启动 `VIVID_lp_random_lm_gpu1`、counterfactual-prefix 或 field-paraphrase；16:00 resource guard OK。日志中仍可见 10:00 旧的 `conda run ... step_1000.pt failed` 记录，但这是已记录的重启恢复失败案例，不是当前新增失败。
- Case study：本小时无新增失败案例；10:00 重启恢复问题仍以 `History/20260610_1000_random_lm_reboot_resume_checkpoint_fix/case_study.md` 为已记录案例。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：最近 20/50/100 个进度样本均值约 `23.2/23.6/25.6s/optimizer step`；预计约 `1.1-1.3` 小时到 step 2500 validation/checkpoint。按当前窗口估计 random-LM pretrain 剩余约 `49-55` 小时，考虑后续 validation/checkpoint 与吞吐波动保守约 `50-58` 小时；完整目标还需 random-LM LP、两个 eval 与最终汇总，保守约 `52-62` 小时。

2026-06-10 17:00 小时同步（random-LM 即将到达 step 2500 validation/checkpoint）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务名 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续保持主线程巡检，不新增子智能体。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）仅有 random-LM 训练进程 PID `20324`。17:00 resource guard 显示 GPU0 `0MiB`，GPU1 显存 `18265MiB / 24576MiB`、温度 `61C`，resource guard OK，无 stop action。17:01 现场快照仍为同一 GPU1 PID，GPU0 仍空闲。
- 训练进度：从 step 1500 的 `best.pt` 恢复后，当前恢复段推进到 `974 / 8500` local progress step，折算全局约 `2474 / 10000`；还剩约 `7526` optimizer steps 到 random-LM pretrain `final.pt`，距离下一次 step 2500 validation/checkpoint 约 `26` step。log mtime 为 17:01:43，说明训练仍在持续写入。
- 验证/checkpoint：step 2000 validation 已完成，日志记录 `Step 2000: val_loss = 1.4182`。当前可靠 checkpoint 仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`（13:45 更新）与 `step_2000.pt`（13:45 写出），`final.pt` 尚未生成，`step_2500.pt` 尚未到达。
- 结果判断：validation loss 继续从 step 500/1000/1500/2000 的 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182` 单调下降；17:00 附近训练 loss 约 `1.4030`，learning rate 约 `1.90e-05`。random-init LLM 同架构对照仍按预期收敛，目前结果符合预期。
- 队列/守卫：16:50、16:55、17:00 队列均识别 `random_lm_running=True`，未误启动 `VIVID_lp_random_lm_gpu1`、counterfactual-prefix 或 field-paraphrase；17:00 resource guard OK。日志中仍可见 10:00 旧的 `conda run ... step_1000.pt failed` 记录，但这是已记录的重启恢复失败案例，不是当前新增失败。
- Case study：本小时无新增失败案例；10:00 重启恢复问题仍以 `History/20260610_1000_random_lm_reboot_resume_checkpoint_fix/case_study.md` 为已记录案例。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：最近 20/50/100 个进度样本均值约 `17.3/16.3/21.5s/optimizer step`；预计约 `7-10` 分钟到 step 2500 validation/checkpoint。按当前窗口估计 random-LM pretrain 剩余约 `34-45` 小时，考虑 validation/checkpoint 与吞吐波动保守约 `42-58` 小时；完整目标还需 random-LM LP、两个 eval 与最终汇总，保守约 `45-65` 小时。

2026-06-10 17:15 里程碑同步（random-LM step 2500 validation 已完成）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务名 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本轮继续保持主线程巡检，不新增子智能体。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）仅有 random-LM 训练进程 PID `20324`。17:15 后现场快照显示 GPU1 显存 `18265MiB / 24576MiB`、温度 `58C`，GPU0 仍为 `0MiB`。
- 训练进度：从 step 1500 的 `best.pt` 恢复后，17:15 后已推进到恢复段 `1003 / 8500` local progress step，折算全局约 `2503 / 10000`；还剩约 `7497` optimizer steps 到 random-LM pretrain `final.pt`，距离下一次 step 3000 validation/checkpoint 约 `497` step。
- 验证/checkpoint：step 2500 validation 已完成，日志记录 `Step 2500: val_loss = 1.1340`；该轮保存行为为更新 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`（17:15 更新），日志未显示单独写出 `step_2500.pt`。当前可靠 checkpoint 为 `step_1000.pt`、`step_2000.pt` 与最新 `best.pt`，`final.pt` 尚未生成。
- 结果判断：validation loss 继续从 step 500/1000/1500/2000/2500 的 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340` 单调下降；random-init LLM 同架构对照仍按预期收敛，目前结果符合预期。
- 队列/守卫：17:05、17:10、17:15 队列均识别 `random_lm_running=True`，未误启动 `VIVID_lp_random_lm_gpu1`、counterfactual-prefix 或 field-paraphrase；17:15 resource guard OK。日志中仍可见 10:00 旧的 `conda run ... step_1000.pt failed` 记录，但这是已记录的重启恢复失败案例，不是当前新增失败。
- Case study：本轮无新增失败案例；10:00 重启恢复问题仍以 `History/20260610_1000_random_lm_reboot_resume_checkpoint_fix/case_study.md` 为已记录案例。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：最近 20/50/100 个进度样本均值约 `28.8/22.6/19.7s/optimizer step`，预计约 `2.7-4.0` 小时到 step 3000 validation/checkpoint。按当前窗口估计 random-LM pretrain 剩余约 `41-60` 小时，考虑 validation/checkpoint 与吞吐波动保守约 `45-62` 小时；完整目标还需 random-LM LP、两个 eval 与最终汇总，保守约 `48-68` 小时。

2026-06-10 18:05 暂停记录（按用户要求先暂停）：
- 暂停动作：已暂停 Codex hourly heartbeat 自动化；已停止 `VIVID_random_lm_gpu1` 训练进程，GPU1 上不再有 VIVID random-LM 训练进程。Windows 计划任务权限不允许直接 Disable（`Access is denied`），因此增加暂停哨兵 `outputs/run_state/VIVID_MICCAI_PAUSED.flag`，并让 `scripts/answerability_gpu1_queue_once.ps1` 与 `scripts/answerability_resource_guard_once.ps1` 在哨兵存在时提前退出。
- 暂停验证：18:05 队列计划任务实际触发后写入 `pause flag present at outputs/run_state/VIVID_MICCAI_PAUSED.flag; queue launch skipped`，未重新启动 random-LM；resource guard 同样写入 pause-skip。GPU1 当前 VIVID 占用为 0；GPU1 上若出现非 VIVID 的 Python 进程，应视为其他实验，不属于本 MICCAI 队列。
- 暂停点：random-LM same-architecture 预训练最后观测为恢复段 `1116 / 8500` local progress step，折算全局约 `2616 / 10000`；因为暂停不是在 checkpoint 边界，step 2500 之后约 `116` 个未持久化 optimizer steps 需要下次从 checkpoint 重放。
- 最新可靠 checkpoint：`outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`（step 2500 validation，`val_loss = 1.1340`，17:15 更新）；`final.pt` 尚未生成。
- 结果判断：step 500/1000/1500/2000/2500 validation loss 为 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340`，暂停前结果仍符合预期；本次暂停不是 failure case。旧的 10:00 resume/checkpoint 失败案例仍记录在 `History/20260610_1000_random_lm_reboot_resume_checkpoint_fix/case_study.md`。
- 恢复方式：移除 `outputs/run_state/VIVID_MICCAI_PAUSED.flag` 后，继续只用 GPU1，从最新 `best.pt`/最新 checkpoint 重新启动 `VIVID_random_lm_gpu1` 或队列；恢复后预期会从 step 2500 附近继续，而不是从 2616 精确续跑。
- 暂停后剩余硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。

2026-06-10 20:30 恢复记录（用户要求继续）：
- 恢复动作：已恢复 Codex hourly heartbeat 自动化为 ACTIVE；已删除暂停哨兵 `outputs/run_state/VIVID_MICCAI_PAUSED.flag`。20:16 手动触发 `scripts/answerability_gpu1_queue_once.ps1` 后，队列启动 `VIVID_random_lm_gpu1`；20:20 与 20:25 队列均识别 `random_lm_running=True`，未重复启动 LP/counterfactual/field-paraphrase。
- GPU 约束：当前 VIVID 训练进程 PID `22396` 绑定 GPU1（bus `00000000:05:00.0`），GPU0 为 `0MiB`；20:29 现场快照显示 GPU1 显存约 `15221MiB / 24576MiB`、util `79%`、温度 `59C`。
- 恢复 checkpoint：日志明确显示 `Checkpoint loaded from ...\outputs\ablation_ums_random_lm_12label\checkpoints\best.pt` 与 `Resuming from step 2500`。由于 18:05 暂停不是 checkpoint 边界，暂停前 step 2500 之后的约 `116` 个未持久化 optimizer steps 已按预期从 step 2500 重新计算。
- 当前阶段：`random-LM same-architecture` 预训练继续运行，新恢复段总进度为 `0/7500` 起步；20:29 已推进到约 `9/7500`，折算全局约 `2509/10000`。下一次 validation/checkpoint 目标仍为全局 step 3000，距离约 `491` optimizer steps。
- 结果判断：恢复前最近 validation 仍为 step 500/1000/1500/2000/2500 的 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340`，趋势符合预期；本次继续/重放不是 failure case。日志中可见的 `step_1000.pt failed` 仍是 10:00 已归档案例，不是新增失败。
- 剩余硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：刚恢复后的短窗口约 `20-23s/optimizer step`；预计约 `3.0-3.6` 小时到 step 3000 validation/checkpoint。random-LM pretrain 剩余约 `7491` step，按当前短窗口约 `42-48` 小时，考虑 validation/checkpoint 与吞吐波动保守约 `46-62` 小时；完整目标还需 random-LM LP、两个 eval 与最终汇总，保守约 `49-68` 小时。

2026-06-10 21:00 小时同步（random-LM 从 step 2500 checkpoint 恢复后继续运行）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务名 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）由 random-LM 训练进程 PID `22396` 占用。21:02 现场快照显示 GPU0 `0MiB / 24576MiB`、util `0%`；GPU1 `16739MiB / 24576MiB`、util `99%`、温度 `56C`。21:00 resource guard 记录 GPU1 `16739MiB`、util `100%`、温度 `55C`，resource guard OK，无 stop action。
- 训练进度：20:16 从 `best.pt` 恢复，日志明确处于 step 2500 续跑段；21:02 已推进到恢复段约 `53 / 7500` local progress step，折算全局约 `2553 / 10000`。距离下一次 step 3000 validation/checkpoint 约 `447` step，random-LM pretrain 到 `final.pt` 还剩约 `7447` step。
- 验证/checkpoint：最新可靠 checkpoint 仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`（17:15，step 2500 validation best），并保留 `step_1000.pt`、`step_2000.pt`；`final.pt` 尚未生成。
- 结果判断：validation loss 仍为 step 500/1000/1500/2000/2500 的 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340`，趋势单调下降；21:02 当前训练 loss 约 `1.2727`，learning rate 约 `1.89e-05`。random-init LLM 同架构对照仍按预期收敛，目前结果符合预期。
- 队列/守卫：20:55 与 21:00 队列均识别 `random_lm_running=True`，未误启动 `VIVID_lp_random_lm_gpu1`、counterfactual-prefix 或 field-paraphrase；21:00 resource guard OK。当前 20:16 恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`。
- Case study：本小时无新增失败案例；10:00 重启后错误 checkpoint 恢复问题仍以 `History/20260610_1000_random_lm_reboot_resume_checkpoint_fix/case_study.md` 为已记录案例。18:05 暂停与 20:16 继续导致的 step 2500 后重放属于预期恢复行为，不是 failure case。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：最近 20/50/100 个进度样本均值约 `48.8/42.9/40.3s/optimizer step`；按当前慢窗口预计约 `5.0-6.1` 小时到 step 3000 validation/checkpoint。random-LM pretrain 剩余约 `83-101` 小时（取决于后续吞吐是否恢复）；完整目标还需 random-LM LP、两个 eval 与最终汇总，当前保守估计约 `86-108` 小时。该 ETA 受 Windows/WDDM 吞吐波动影响较大，后续以每小时滚动窗口更新。

2026-06-10 22:00 小时同步（random-LM 继续健康推进，尚未到 step 3000）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务名 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）由 random-LM 训练进程 PID `22396` 占用。22:00 后现场快照显示 GPU0 `0MiB / 24576MiB`、util `0%`；GPU1 `16739MiB / 24576MiB`、util `99%`、温度 `58C`。22:00 resource guard 记录 GPU1 `16739MiB`、util `72%`、温度 `60C`，resource guard OK，无 stop action。
- 训练进度：20:16 从 `best.pt` 恢复，日志明确处于 step 2500 续跑段；22:02 已推进到恢复段约 `120 / 7500` local progress step，折算全局约 `2620 / 10000`。距离下一次 step 3000 validation/checkpoint 约 `380` step，random-LM pretrain 到 `final.pt` 还剩约 `7380` step。
- 验证/checkpoint：最新可靠 checkpoint 仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`（17:15，step 2500 validation best），并保留 `step_1000.pt`、`step_2000.pt`；`final.pt` 尚未生成。
- 结果判断：validation loss 仍为 step 500/1000/1500/2000/2500 的 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340`，趋势单调下降；22:02 当前训练 loss 约 `1.2489`，learning rate 约 `1.88e-05`。random-init LLM 同架构对照仍按预期收敛，目前结果符合预期。
- 队列/守卫：21:50、21:55、22:00 队列均识别 `random_lm_running=True`，未误启动 `VIVID_lp_random_lm_gpu1`、counterfactual-prefix 或 field-paraphrase；22:00 resource guard OK。当前 20:16 恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`。
- Case study：本小时无新增失败案例；10:00 重启后错误 checkpoint 恢复问题仍以 `History/20260610_1000_random_lm_reboot_resume_checkpoint_fix/case_study.md` 为已记录案例。18:05 暂停与 20:16 继续导致的 step 2500 后重放属于预期恢复行为，不是 failure case。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：最近 20/50/100 个进度样本均值约 `49.7/51.7/52.2s/optimizer step`；预计约 `5.3-5.5` 小时到 step 3000 validation/checkpoint。random-LM pretrain 剩余约 `102-107` 小时；完整目标还需 random-LM LP、两个 eval 与最终汇总，当前保守估计约 `105-115` 小时。该 ETA 受 Windows/WDDM 吞吐波动影响较大，后续以每小时滚动窗口更新。

2026-06-10 23:00 小时同步（random-LM 继续推进，等待 step 3000 validation/checkpoint）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务名 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）由 random-LM 训练进程 PID `22396` 占用。23:01 现场快照显示 GPU0 `0MiB / 24576MiB`、util `0%`；GPU1 `16739MiB / 24576MiB`、util `65%`、温度 `57C`。23:00 resource guard 记录 GPU1 `16739MiB`、util `25%`、温度 `61C`，resource guard OK，无 stop action。
- 训练进度：20:16 从 `best.pt` 恢复，日志明确处于 step 2500 续跑段；23:02 已推进到恢复段约 `192 / 7500` local progress step，折算全局约 `2692 / 10000`。距离下一次 step 3000 validation/checkpoint 约 `308` step，random-LM pretrain 到 `final.pt` 还剩约 `7308` step。
- 验证/checkpoint：最新可靠 checkpoint 仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`（17:15，step 2500 validation best），并保留 `step_1000.pt`、`step_2000.pt`；`final.pt` 尚未生成。
- 结果判断：validation loss 仍为 step 500/1000/1500/2000/2500 的 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340`，趋势单调下降；23:02 当前训练 loss 约 `1.2360`，learning rate 约 `1.87e-05`。random-init LLM 同架构对照仍按预期收敛，目前结果符合预期。
- 队列/守卫：22:50、22:55、23:00 队列均识别 `random_lm_running=True`，未误启动 `VIVID_lp_random_lm_gpu1`、counterfactual-prefix 或 field-paraphrase；23:00 resource guard OK。当前 20:16 恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`。
- Case study：本小时无新增失败案例；10:00 重启后错误 checkpoint 恢复问题仍以 `History/20260610_1000_random_lm_reboot_resume_checkpoint_fix/case_study.md` 为已记录案例。18:05 暂停与 20:16 继续导致的 step 2500 后重放属于预期恢复行为，不是 failure case。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：最近 20/50/100 个进度样本均值约 `46.3/50.8/50.5s/optimizer step`；预计约 `4.0-4.35` 小时到 step 3000 validation/checkpoint。random-LM pretrain 剩余约 `94-103` 小时；完整目标还需 random-LM LP、两个 eval 与最终汇总，当前保守估计约 `97-111` 小时。该 ETA 受 Windows/WDDM 吞吐波动影响较大，后续以每小时滚动窗口更新。

2026-06-11 00:00 小时同步（random-LM 继续推进，仍未到 step 3000 validation/checkpoint）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务名 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）由 random-LM 训练进程 PID `22396` 占用。00:00 现场快照显示 GPU0 `0MiB / 24576MiB`、util `0%`；GPU1 `16739MiB / 24576MiB`、util `29%`、温度 `53C`。00:00 resource guard 记录 GPU1 `16739MiB`、util `46%`、温度 `54C`，resource guard OK，无 stop action。
- 训练进度：20:16 从 `best.pt` 恢复，日志明确处于 step 2500 续跑段；00:01 已推进到恢复段约 `247 / 7500` local progress step，折算全局约 `2747 / 10000`。距离下一次 step 3000 validation/checkpoint 约 `253` step，random-LM pretrain 到 `final.pt` 还剩约 `7253` step。
- 验证/checkpoint：最新可靠 checkpoint 仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`（17:15，step 2500 validation best），并保留 `step_1000.pt`、`step_2000.pt`；`final.pt` 尚未生成。
- 结果判断：validation loss 仍为 step 500/1000/1500/2000/2500 的 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340`，趋势单调下降；00:01 当前训练 loss 约 `1.2456`，learning rate 约 `1.87e-05`。random-init LLM 同架构对照仍按预期收敛，目前结果符合预期。
- 队列/守卫：23:50、23:55、00:00 队列均识别 `random_lm_running=True`，未误启动 `VIVID_lp_random_lm_gpu1`、counterfactual-prefix 或 field-paraphrase；00:00 resource guard OK。当前 20:16 恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`。
- Case study：本小时无新增失败案例；10:00 重启后错误 checkpoint 恢复问题仍以 `History/20260610_1000_random_lm_reboot_resume_checkpoint_fix/case_study.md` 为已记录案例。18:05 暂停与 20:16 继续导致的 step 2500 后重放属于预期恢复行为，不是 failure case。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：最近 20/50/100 个进度样本均值约 `62.9/64.5/57.4s/optimizer step`；预计约 `4.0-4.55` 小时到 step 3000 validation/checkpoint。random-LM pretrain 剩余约 `116-130` 小时；完整目标还需 random-LM LP、两个 eval 与最终汇总，当前保守估计约 `119-139` 小时。该 ETA 受 Windows/WDDM 吞吐波动影响较大，后续以每小时滚动窗口更新。

2026-06-11 01:00 小时同步（random-LM 遭遇 GPU lost 后已恢复重启，需从 step 2500 重放）：
- 当前阶段：`random-LM same-architecture` 预训练仍是当前阻塞项，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- 失败事件：00:55 后 `nvidia-smi` 报 `GPU0000:01:00.0: GPU is lost`，训练进程退出，日志记录 `conda run ... failed` 与 `exitcode 1073807364`。失败前 random-LM 已推进到恢复段约 `321 / 7500` local progress step，折算全局约 `2821 / 10000`，但尚未到 step 3000 validation/checkpoint，因此 step 2500 之后这约 `321` 个未持久化 optimizer steps 需要重放。
- Case study：已保存 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`，记录 GPU lost 症状、队列阻塞、修复动作与恢复边界。
- 恢复动作：01:20 后 GPU 查询恢复；队列一度因 GPU1 上约 `60MiB` 的 `GameViewerServer.exe` 小占用误判 busy。已将 `scripts/answerability_gpu1_queue_once.ps1` 的 launch busy 阈值调整为 `500MiB`，避免小型显示/系统占用阻塞续跑。01:22 手动触发队列，成功启动 `VIVID_random_lm_gpu1`。
- 恢复验证：日志确认 `resume random-LM from ...\best.pt`、`Checkpoint loaded from ...\best.pt`、`Resuming from step 2500`、`Starting training...`。01:25 队列识别 `random_lm_running=True`，未重复启动；01:25 resource guard 显示 GPU0 空闲、GPU1 已由重启中的训练进程占用约 `4901MiB`，resource guard OK。
- GPU 约束：恢复后仍只使用 GPU1（`CUDA_VISIBLE_DEVICES=1`，进程内显示 `cuda:0` 为可见设备 0，对应物理 GPU1）。GPU0 当前空闲。
- 验证/checkpoint：最新可靠 checkpoint 仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`（step 2500 validation best），并保留 `step_1000.pt`、`step_2000.pt`；`final.pt` 尚未生成，`step_3000.pt` 尚未到达。
- 结果判断：step 500/1000/1500/2000/2500 validation loss 仍为 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340`，趋势符合预期；本次 GPU lost 是运行稳定性失败，不改变已保存 checkpoint 的实验有效性，但会增加重放时间。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：由于恢复后从 step 2500 重新开始，距离 step 3000 validation/checkpoint 约 `500` optimizer steps。当前刚重启，稳定吞吐尚未重新形成；按本机近期窗口粗估约 `3-8` 小时到 step 3000，random-LM pretrain 仍需约 `80-130` 小时量级，完整目标保守约 `85-140` 小时。下一小时以恢复后真实训练窗口重新估算。

2026-06-11 02:00 小时同步（random-LM GPU lost 后恢复段健康推进，尚未到 step 3000）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务名 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）由 random-LM 训练进程 PID `6464` 占用。02:00 现场快照显示 GPU0 `0MiB / 24576MiB`、util `0%`；GPU1 `16739MiB / 24576MiB`、util `39%`、温度 `60C`。02:00 resource guard 记录 GPU1 `16739MiB`、util `32%`、温度 `63C`，resource guard OK，无 stop action。
- 训练进度：01:22 从 `best.pt` 恢复，日志明确处于 step 2500 续跑段；02:01 已推进到恢复段约 `100 / 7500` local progress step，折算全局约 `2600 / 10000`。距离下一次 step 3000 validation/checkpoint 约 `400` step，random-LM pretrain 到 `final.pt` 还剩约 `7400` step。
- 验证/checkpoint：最新可靠 checkpoint 仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`（step 2500 validation best，17:15 更新），并保留 `step_1000.pt`、`step_2000.pt`；`final.pt` 尚未生成，`step_3000.pt` 尚未到达。
- 结果判断：validation loss 仍为 step 500/1000/1500/2000/2500 的 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340`，趋势单调下降；02:01 当前训练 loss 约 `1.2472`，learning rate 约 `1.88e-05`。random-init LLM 同架构对照仍按预期收敛，目前结果符合预期。
- 队列/守卫：01:50、01:55、02:00 队列均识别 `random_lm_running=True`，未误启动 `VIVID_lp_random_lm_gpu1`、counterfactual-prefix 或 field-paraphrase；02:00 resource guard OK。当前 01:22 恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`。
- Case study：本小时无新增失败案例；00:55 GPU lost 已记录在 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`。01:22 后从 step 2500 重放属于预期恢复行为，不是新的 failure case。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：最近 20/50/100 个进度样本均值约 `20.9/22.4/21.3s/optimizer step`；预计约 `2.3-2.5` 小时到 step 3000 validation/checkpoint。random-LM pretrain 剩余约 `43-46` 小时；完整目标还需 random-LM LP、两个 eval 与最终汇总，当前保守估计约 `46-52` 小时。后续继续以每小时滚动窗口更新。

2026-06-11 03:00 小时同步（random-LM 恢复段继续健康推进，仍未到 step 3000）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务名 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）由 random-LM 训练进程 PID `6464` 占用。03:01 现场快照显示 GPU0 `0MiB / 24576MiB`、util `0%`；GPU1 `16739MiB / 24576MiB`、util `86%`、温度 `61C`。03:00 resource guard 记录 GPU1 `16739MiB`、util `43%`、温度 `65C`，resource guard OK，无 stop action。
- 训练进度：01:22 从 `best.pt` 恢复，日志明确处于 step 2500 续跑段；03:01 已推进到恢复段约 `231 / 7500` local progress step，折算全局约 `2731 / 10000`。距离下一次 step 3000 validation/checkpoint 约 `269` step，random-LM pretrain 到 `final.pt` 还剩约 `7269` step。
- 验证/checkpoint：最新可靠 checkpoint 仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`（step 2500 validation best，17:15 更新），并保留 `step_1000.pt`、`step_2000.pt`；`final.pt` 尚未生成，`step_3000.pt` 尚未到达。
- 结果判断：validation loss 仍为 step 500/1000/1500/2000/2500 的 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340`，趋势单调下降；03:01 当前训练 loss 约 `1.2270`，learning rate 约 `1.87e-05`。random-init LLM 同架构对照仍按预期收敛，目前结果符合预期。
- 队列/守卫：02:50、02:55、03:00 队列均识别 `random_lm_running=True`，未误启动 `VIVID_lp_random_lm_gpu1`、counterfactual-prefix 或 field-paraphrase；03:00 resource guard OK。当前 01:22 恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`。
- Case study：本小时无新增失败案例；00:55 GPU lost 已记录在 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`。01:22 后从 step 2500 重放属于预期恢复行为，不是新的 failure case。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：最近 20/50/100 个进度样本均值约 `24.9/27.5/27.5s/optimizer step`；预计约 `1.9-2.1` 小时到 step 3000 validation/checkpoint。random-LM pretrain 剩余约 `50-55.5` 小时；完整目标还需 random-LM LP、两个 eval 与最终汇总，当前保守估计约 `53-62` 小时。后续继续以每小时滚动窗口更新。

2026-06-11 04:00 小时同步（random-LM 已过 step 2800，仍未到 step 3000）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务名 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）由 random-LM 训练进程 PID `6464` 占用。04:01 现场快照显示 GPU0 `0MiB / 24576MiB`、util `0%`；GPU1 `16739MiB / 24576MiB`、util `87%`、温度 `61C`。04:00 resource guard 记录 GPU1 `16739MiB`、util `28%`、温度 `61C`，resource guard OK，无 stop action。
- 训练进度：01:22 从 `best.pt` 恢复，日志明确处于 step 2500 续跑段；04:01 已推进到恢复段约 `339 / 7500` local progress step，折算全局约 `2839 / 10000`。距离下一次 step 3000 validation/checkpoint 约 `161` step，random-LM pretrain 到 `final.pt` 还剩约 `7161` step。
- 验证/checkpoint：最新可靠 checkpoint 仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`（step 2500 validation best，17:15 更新），并保留 `step_1000.pt`、`step_2000.pt`；`final.pt` 尚未生成，`step_3000.pt` 尚未到达。
- 结果判断：validation loss 仍为 step 500/1000/1500/2000/2500 的 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340`，趋势单调下降；04:01 当前训练 loss 约 `1.1904`，learning rate 约 `1.86e-05`。random-init LLM 同架构对照仍按预期收敛，目前结果符合预期。
- 队列/守卫：03:50、03:55、04:00 队列均识别 `random_lm_running=True`，未误启动 `VIVID_lp_random_lm_gpu1`、counterfactual-prefix 或 field-paraphrase；04:00 resource guard OK。当前 01:22 恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`。
- Case study：本小时无新增失败案例；00:55 GPU lost 已记录在 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`。01:22 后从 step 2500 重放属于预期恢复行为，不是新的 failure case。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/step_3000.pt`、`outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：最近 20/50/100 个进度样本均值约 `25.4/31.0/33.3s/optimizer step`；预计约 `1.1-1.5` 小时到 step 3000 validation/checkpoint。random-LM pretrain 剩余约 `50.5-66.2` 小时；完整目标还需 random-LM LP、两个 eval 与最终汇总，当前保守估计约 `54-73` 小时。后续继续以每小时滚动窗口更新。

2026-06-11 05:00 小时同步（random-LM 已过 step 2940，仍未到 step 3000）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务名 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）由 random-LM 训练进程 PID `6464` 占用。05:00 现场快照显示 GPU0 `0MiB / 24576MiB`、util `0%`；GPU1 `18263MiB / 24576MiB`、util `27%`、温度 `63C`。05:00 resource guard 记录 GPU1 `18263MiB`、util `39%`、温度 `62C`，resource guard OK，无 stop action。
- 训练进度：01:22 从 `best.pt` 恢复，日志明确处于 step 2500 续跑段；05:00 已推进到恢复段约 `443 / 7500` local progress step，折算全局约 `2943 / 10000`。距离下一次 step 3000 validation/checkpoint 约 `57` step，random-LM pretrain 到 `final.pt` 还剩约 `7057` step。
- 验证/checkpoint：最新可靠 checkpoint 仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`（step 2500 validation best，17:15 更新），并保留 `step_1000.pt`、`step_2000.pt`；`final.pt` 尚未生成，`step_3000.pt` 尚未到达。
- 结果判断：validation loss 仍为 step 500/1000/1500/2000/2500 的 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340`，趋势单调下降；05:00 当前训练 loss 约 `1.1897`，learning rate 约 `1.85e-05`。random-init LLM 同架构对照仍按预期收敛，目前结果符合预期。
- 队列/守卫：04:50、04:55、05:00 队列均识别 `random_lm_running=True`，未误启动 `VIVID_lp_random_lm_gpu1`、counterfactual-prefix 或 field-paraphrase；05:00 resource guard OK。当前 01:22 恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`。
- Case study：本小时无新增失败案例；00:55 GPU lost 已记录在 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`。01:22 后从 step 2500 重放属于预期恢复行为，不是新的 failure case。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/step_3000.pt`、`outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：最近 20/50/100 个进度样本均值约 `39.3/37.3/33.8s/optimizer step`；预计约 `0.6` 小时到 step 3000 validation/checkpoint。random-LM pretrain 剩余约 `66-73` 小时；完整目标还需 random-LM LP、两个 eval 与最终汇总，当前保守估计约 `70-80` 小时。后续继续以每小时滚动窗口更新，并在 step 3000 落盘后单独记录 validation/checkpoint 里程碑。

2026-06-11 05:45 里程碑同步（random-LM step 3000 已落盘）：
- 当前阶段：`random-LM same-architecture` 预训练已完成 step 3000 validation/checkpoint，并继续向 step 10000 运行；任务仍为 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。
- GPU 约束：GPU0 仍空闲；GPU1（bus `00000000:05:00.0`）由 PID `6464` 的 random-LM 训练进程占用。05:45 快照显示 GPU0 `0MiB / 24576MiB`、util `0%`；GPU1 `18265MiB / 24576MiB`、util `26%`、温度 `62C`。
- 训练进度：01:22 从 `best.pt` 恢复后，05:41 完成全局 step 3000 validation；05:45 已继续到恢复段约 `509 / 7500` local progress step，折算全局约 `3009 / 10000`。random-LM pretrain 到 `final.pt` 还剩约 `6991` step。
- 验证/checkpoint：step 3000 validation loss 为 `0.9361`；`outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 于 05:41:52 更新，`outputs/ablation_ums_random_lm_12label/checkpoints/step_3000.pt` 于 05:41:59 生成。`final.pt` 尚未生成。
- 结果判断：validation loss 轨迹更新为 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340 -> 0.9361`，继续单调下降；random-init LLM 同架构对照仍按预期收敛，目前结果符合预期。
- 队列/守卫：05:35、05:40、05:45 队列均识别 `random_lm_running=True`，未误启动 `VIVID_lp_random_lm_gpu1` 或后续 eval；当前 01:22 恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`。
- Case study：本里程碑无新增失败案例；00:55 GPU lost 仍是唯一已记录 failure case，路径为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`。
- 当前硬缺口更新为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：step 3000 已完成；random-LM pretrain 剩余约 `69-75` 小时（以 validation 前后滚动速度保守估计）。完整目标还需 random-LM LP、两个 eval 与最终汇总，当前保守估计约 `73-82` 小时。后续继续每小时同步，并在 `final.pt` 落盘后自动进入 random-LM LP。

2026-06-11 06:00 小时同步（random-LM step 3000 后继续推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）由 PID `6464` 的 random-LM 训练进程占用。06:01 现场快照显示 GPU0 `0MiB / 24576MiB`、util `0%`；GPU1 `18265MiB / 24576MiB`、util `71%`、温度 `59C`。06:00 resource guard 记录 GPU1 `18265MiB`、util `17%`、温度 `64C`，resource guard OK，无 stop action。
- 训练进度：01:22 从 `best.pt` 恢复后，05:41 完成全局 step 3000 validation/checkpoint；06:01 已推进到恢复段约 `544 / 7500` local progress step，折算全局约 `3044 / 10000`。random-LM pretrain 到 `final.pt` 还剩约 `6956` step。
- 验证/checkpoint：最新验证点仍为 step 3000，validation loss `0.9361`；`outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 于 05:41:52 更新，`outputs/ablation_ums_random_lm_12label/checkpoints/step_3000.pt` 于 05:41:59 生成。`step_4000.pt` 与 `final.pt` 尚未生成。
- 结果判断：validation loss 轨迹为 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340 -> 0.9361`，继续单调下降；06:01 当前训练 loss 约 `1.1253`，learning rate 约 `1.83e-05`。random-init LLM 同架构对照仍按预期收敛，目前结果符合预期。
- 队列/守卫：05:50、05:55、06:00 队列均识别 `random_lm_running=True`，未误启动 `VIVID_lp_random_lm_gpu1` 或后续 eval；06:00 resource guard OK。当前 01:22 恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`。
- Case study：本小时无新增失败案例；00:55 GPU lost 仍是唯一已记录 failure case，路径为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：最近 20/50/100 个进度样本均值约 `23.5/36.0/36.3s/optimizer step`；短窗受 step 3000 validation 后恢复影响偏乐观，按 50/100 步窗口估计 random-LM pretrain 剩余约 `69.5-70.2` 小时。完整目标还需 random-LM LP、两个 eval 与最终汇总，当前保守估计约 `73-79` 小时。后续继续每小时同步，并在 `final.pt` 落盘后自动进入 random-LM LP。

2026-06-11 07:00 小时同步（random-LM step 3100+，继续向 step 4000 推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）由 PID `6464` 的 random-LM 训练进程占用。07:02 现场快照显示 GPU0 `0MiB / 24576MiB`、util `0%`；GPU1 `18265MiB / 24576MiB`、util 瞬时 `0%`、温度 `58C`。07:00 resource guard 记录 GPU1 `18265MiB`、util `52%`、温度 `59C`，resource guard OK，无 stop action。
- 训练进度：01:22 从 `best.pt` 恢复后，05:41 完成全局 step 3000 validation/checkpoint；07:02 已推进到恢复段约 `644 / 7500` local progress step，折算全局约 `3144 / 10000`。距离下一次 step 4000 validation/checkpoint 约 `856` step；random-LM pretrain 到 `final.pt` 还剩约 `6856` step。
- 验证/checkpoint：最新验证点仍为 step 3000，validation loss `0.9361`；`outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 于 05:41:52 更新，`outputs/ablation_ums_random_lm_12label/checkpoints/step_3000.pt` 于 05:41:59 生成。`step_4000.pt` 与 `final.pt` 尚未生成。
- 结果判断：validation loss 轨迹为 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340 -> 0.9361`，继续单调下降；07:02 当前训练 loss 约 `1.0528`，learning rate 约 `1.82e-05`。random-init LLM 同架构对照仍按预期收敛，目前结果符合预期。
- 队列/守卫：06:50、06:55、07:00 队列均识别 `random_lm_running=True`，未误启动 `VIVID_lp_random_lm_gpu1` 或后续 eval；07:00 resource guard OK。当前 01:22 恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`。
- Case study：本小时无新增失败案例；00:55 GPU lost 仍是唯一已记录 failure case，路径为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：最近 20/50/100 个进度样本均值约 `25.5/32.7/35.9s/optimizer step`；预计约 `7.8` 小时到 step 4000 validation/checkpoint。按 50/100 步窗口估计 random-LM pretrain 剩余约 `62.3-68.4` 小时。完整目标还需 random-LM LP、两个 eval 与最终汇总，当前保守估计约 `66-77` 小时。后续继续每小时同步，并在 `step_4000.pt` 或 `final.pt` 落盘时记录里程碑。

2026-06-11 08:00 小时同步（random-LM step 3200+，继续向 step 4000 推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）由 PID `6464` 的 random-LM 训练进程占用。07:59 现场快照显示 GPU0 `0MiB / 24576MiB`、util `0%`；GPU1 `18265MiB / 24576MiB`、util `47%`、温度 `56C`。08:00 resource guard 记录 GPU1 `18265MiB`、util `20%`、温度 `59C`，resource guard OK，无 stop action。
- 训练进度：01:22 从 `best.pt` 恢复后，05:41 完成全局 step 3000 validation/checkpoint；08:00 已推进到恢复段约 `725 / 7500` local progress step，折算全局约 `3225 / 10000`。距离下一次 step 4000 validation/checkpoint 约 `775` step；random-LM pretrain 到 `final.pt` 还剩约 `6775` step。
- 验证/checkpoint：最新验证点仍为 step 3000，validation loss `0.9361`；`outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 于 05:41:52 更新，`outputs/ablation_ums_random_lm_12label/checkpoints/step_3000.pt` 于 05:41:59 生成。`step_4000.pt` 与 `final.pt` 尚未生成。
- 结果判断：validation loss 轨迹为 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340 -> 0.9361`，继续单调下降；08:00 当前训练 loss 约 `1.1254`，learning rate 约 `1.81e-05`。random-init LLM 同架构对照仍按预期收敛，目前结果符合预期。
- 队列/守卫：07:45、07:50、07:55、08:00 队列均识别 `random_lm_running=True`，未误启动 `VIVID_lp_random_lm_gpu1` 或后续 eval；08:00 resource guard OK。当前 01:22 恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`。
- Case study：本小时无新增失败案例；00:55 GPU lost 仍是唯一已记录 failure case，路径为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：最近 20/50/100 个进度样本均值约 `43.2/43.4/40.6s/optimizer step`；预计约 `9.35` 小时到 step 4000 validation/checkpoint。按 50/100 步窗口估计 random-LM pretrain 剩余约 `76.4-81.8` 小时。完整目标还需 random-LM LP、两个 eval 与最终汇总，当前保守估计约 `80-91` 小时。后续继续每小时同步，并在 `step_4000.pt` 或 `final.pt` 落盘时记录里程碑。

2026-06-11 09:00 小时同步（random-LM step 3300+，继续向 step 4000 推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）由 PID `6464` 的 random-LM 训练进程占用。09:00 现场快照显示 GPU0 `0MiB / 24576MiB`、util `0%`；GPU1 `18265MiB / 24576MiB`、util `43%`、温度 `56C`。09:00 resource guard 记录 GPU1 `18265MiB`、util `99%`、温度 `59C`，resource guard OK，无 stop action。
- 训练进度：01:22 从 `best.pt` 恢复后，05:41 完成全局 step 3000 validation/checkpoint；09:01 已推进到恢复段约 `810 / 7500` local progress step，折算全局约 `3310 / 10000`。距离下一次 step 4000 validation/checkpoint 约 `690` step；random-LM pretrain 到 `final.pt` 还剩约 `6690` step。
- 验证/checkpoint：最新验证点仍为 step 3000，validation loss `0.9361`；`outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 于 05:41:52 更新，`outputs/ablation_ums_random_lm_12label/checkpoints/step_3000.pt` 于 05:41:59 生成。`step_4000.pt` 与 `final.pt` 尚未生成。
- 结果判断：validation loss 轨迹为 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340 -> 0.9361`，继续单调下降；09:01 当前训练 loss 约 `1.0122`，learning rate 约 `1.80e-05`。random-init LLM 同架构对照仍按预期收敛，目前结果符合预期。
- 队列/守卫：08:45、08:50、08:55、09:00 队列均识别 `random_lm_running=True`，未误启动 `VIVID_lp_random_lm_gpu1` 或后续 eval；09:00 resource guard OK。当前 01:22 恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`。
- Case study：本小时无新增失败案例；00:55 GPU lost 仍是唯一已记录 failure case，路径为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：最近 20/50/100 个进度样本均值约 `43.0/43.2/43.2s/optimizer step`；预计约 `8.3` 小时到 step 4000 validation/checkpoint。按 50/100 步窗口估计 random-LM pretrain 剩余约 `80.3` 小时。完整目标还需 random-LM LP、两个 eval 与最终汇总，当前保守估计约 `84-91` 小时。后续继续每小时同步，并在 `step_4000.pt` 或 `final.pt` 落盘时记录里程碑。

2026-06-11 10:00 小时同步（random-LM step 3400+，继续向 step 4000 推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）由 PID `6464` 的 random-LM 训练进程占用。10:01 现场快照显示 GPU0 `0MiB / 24576MiB`、util `0%`、温度 `41C`；GPU1 `18265MiB / 24576MiB`、util `82%`、温度 `59C`。10:00 resource guard 记录 GPU1 `18265MiB`、util `80%`、温度 `58C`，resource guard OK，无 stop action。
- 训练进度：01:22 从 `best.pt` 恢复后，05:41 完成全局 step 3000 validation/checkpoint；10:02 已推进到恢复段约 `903 / 7500` local progress step，折算全局约 `3403 / 10000`。距离下一次 step 4000 validation/checkpoint 约 `597` step；random-LM pretrain 到 `final.pt` 还剩约 `6597` step。
- 验证/checkpoint：最新验证点仍为 step 3000，validation loss `0.9361`；`outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 于 05:41:52 更新，`outputs/ablation_ums_random_lm_12label/checkpoints/step_3000.pt` 于 05:41:59 生成。`step_4000.pt` 与 `final.pt` 尚未生成。
- 结果判断：validation loss 轨迹为 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340 -> 0.9361`，继续单调下降；10:02 当前训练 loss 约 `0.9961`，learning rate 约 `1.79e-05`。random-init LLM 同架构对照仍按预期收敛，目前结果符合预期。
- 队列/守卫：09:45、09:50、09:55、10:00 队列均识别 `random_lm_running=True`，未误启动 `VIVID_lp_random_lm_gpu1` 或后续 eval；10:00 resource guard OK。当前 01:22 恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`。
- Case study：本小时无新增失败案例；00:55 GPU lost 仍是唯一已记录 failure case，路径为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：最近 20/50/100 个进度样本均值约 `34.9/36.5/40.0s/optimizer step`；预计约 `6.1` 小时到 step 4000 validation/checkpoint。按 50/100 步窗口估计 random-LM pretrain 剩余约 `66.9-73.2` 小时。完整目标还需 random-LM LP、两个 eval 与最终汇总，当前保守估计约 `70-82` 小时。后续继续每小时同步，并在 `step_4000.pt` 或 `final.pt` 落盘时记录里程碑。

2026-06-11 11:00 小时同步（10:20 GPU lost 后已恢复，random-LM 从 step 3000 重放）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- 中断/恢复：10:20 起 resource guard 和 queue 均记录 GPU0 `GPU is lost`，队列在 10:20、10:25、10:30、10:35、10:40 因 `GPU status unavailable` 阻塞启动；重启后 10:42 手动运行 `scripts/answerability_gpu1_queue_once.ps1`，成功触发 scheduled task `VIVID_random_lm_gpu1`。恢复日志显示 `CUDA_VISIBLE_DEVICES=1`、`--resume ...\step_3000.pt`、`Checkpoint loaded ... step_3000.pt`、`Resuming from step 3000`。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）由 PID `6152` 的 random-LM 训练进程占用。11:01 现场快照显示 GPU0 `0MiB / 24576MiB`、util `0%`、温度 `39C`；GPU1 `16739MiB / 24576MiB`、util `17%`、温度 `65C`。11:00 resource guard 记录 GPU1 `16739MiB`、util `65%`、温度 `64C`，resource guard OK，无 stop action。
- 训练进度：10:14 中断前最后解析到 post-step-3000 恢复段约 `921 / 7500` local progress step，折算全局约 `3421 / 10000`，但未生成 `step_4000.pt`，因此进度从 `step_3000.pt` 重放。11:02 已推进到本次重启后的 `70 / 7000` local progress step，折算全局约 `3070 / 10000`。距离下一次 step 4000 validation/checkpoint 约 `930` step；random-LM pretrain 到 `final.pt` 还剩约 `6930` step。
- 验证/checkpoint：最新验证点仍为 step 3000，validation loss `0.9361`；`outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 于 05:41:52 更新，`outputs/ablation_ums_random_lm_12label/checkpoints/step_3000.pt` 于 05:41:59 生成。`step_4000.pt` 与 `final.pt` 尚未生成。
- 结果判断：validation loss 轨迹仍为 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340 -> 0.9361`，最新 checkpoint 前趋势符合预期；11:02 当前训练 loss 约 `1.0338`，learning rate 约 `1.83e-05`。本次 GPU lost 只造成墙钟回退与 step 3000 后重放，不改变实验定义、batch size 或验证协议，目前结果仍符合预期。
- 队列/守卫：10:45、10:50、10:55、11:00 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`；11:00 resource guard OK。恢复后日志未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`。
- Case study：本小时新增 failure case，路径为 `History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`。此前 00:55 GPU lost case study 保留在 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：恢复后 20/50/100 个进度样本均值约 `12.9/13.3/12.3s/optimizer step`；刚重启后的短窗偏乐观，预计约 `3.4` 小时到 step 4000 validation/checkpoint。按 50/100 步窗口估计 random-LM pretrain 剩余约 `23.8-25.6` 小时；考虑此前长窗和 GPU lost 重放风险，完整目标当前保守估计约 `28-40` 小时。后续继续每小时同步，并在 `step_4000.pt` 或 `final.pt` 落盘时记录里程碑。

2026-06-11 12:00 小时同步（random-LM step 3200+，重启后稳定推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）由 PID `6152` 的 random-LM 训练进程占用。12:01 现场快照显示 GPU0 `0MiB / 24576MiB`、util `0%`、温度 `40C`；GPU1 `16739MiB / 24576MiB`、util `98%`、温度 `60C`。12:00 resource guard 记录 GPU1 `16739MiB`、util `19%`、温度 `63C`，resource guard OK，无 stop action。
- 训练进度：10:42 从 `outputs/ablation_ums_random_lm_12label/checkpoints/step_3000.pt` 恢复后，12:01 已推进到本次重启后的 `202 / 7000` local progress step，折算全局约 `3202 / 10000`。距离下一次 step 4000 validation/checkpoint 约 `798` step；random-LM pretrain 到 `final.pt` 还剩约 `6798` step。
- 验证/checkpoint：最新验证点仍为 step 3000，validation loss `0.9361`；`outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 于 05:41:52 更新，`outputs/ablation_ums_random_lm_12label/checkpoints/step_3000.pt` 于 05:41:59 生成。`step_4000.pt` 与 `final.pt` 尚未生成。
- 结果判断：validation loss 轨迹仍为 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340 -> 0.9361`，最新 checkpoint 前趋势符合预期；12:01 当前训练 loss 约 `1.0309`，learning rate 约 `1.81e-05`。恢复后日志持续写入，未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果仍符合预期。
- 队列/守卫：11:45、11:50、11:55、12:00 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；12:00 resource guard OK。
- Case study：本小时无新增失败案例；已记录的两次 GPU lost case study 保留在 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md` 与 `History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：恢复后最近 20/50/100 个进度样本均值约 `25.6/27.9/27.7s/optimizer step`；预计约 `6.2` 小时到 step 4000 validation/checkpoint。按 50/100 步窗口估计 random-LM pretrain 剩余约 `52.4-52.6` 小时；完整目标还需 random-LM LP、两个 eval 与最终汇总，当前保守估计约 `56-65` 小时。后续继续每小时同步，并在 `step_4000.pt` 或 `final.pt` 落盘时记录里程碑。

2026-06-11 15:00 小时同步（补记 12:15 静默退出恢复；random-LM step 4000 已落盘）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- 中断/恢复补记：12:05 队列仍识别 `random_lm_running=True`；12:15 队列识别 `random_lm_running=False`，resource guard 同时显示 GPU1 空闲（`0MiB`、util `0%`），随后自动重新启动 `VIVID_random_lm_gpu1` 并从 `outputs/ablation_ums_random_lm_12label/checkpoints/step_3000.pt` 恢复。日志未发现新的 traceback/OOM/GPU lost；该事件已作为静默退出/队列恢复 case study 保存至 `History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）由 PID `15648` 的 random-LM 训练进程占用。15:03 现场快照显示 GPU0 `0MiB / 24576MiB`、util `0%`、温度 `39C`；GPU1 `18265MiB / 24576MiB`、util `67%`、温度 `59C`。15:00 resource guard 记录 GPU1 `18265MiB`、util `43%`、温度 `57C`，resource guard OK，无 stop action。
- 训练进度：12:15 从 `step_3000.pt` 自动重启后，15:03 已推进到本次恢复段 `1298 / 7000` local progress step，折算全局约 `4298 / 10000`。距离下一次 step 5000 validation/checkpoint 约 `702` step；random-LM pretrain 到 `final.pt` 还剩约 `5702` step。
- 验证/checkpoint：`outputs/ablation_ums_random_lm_12label/checkpoints/step_4000.pt` 已于 14:25:14 生成，`best.pt` 于 14:25:09 更新。validation loss 轨迹更新为 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340 -> 0.9361 -> 0.8275 -> 0.7444`。`final.pt` 尚未生成。
- 结果判断：step 3500 与 step 4000 validation loss 继续下降，random-init LLM 同架构对照仍按预期收敛；15:03 当前训练 loss 约 `0.8769`，learning rate 约 `1.66e-05`。当前 post-12:15 segment 未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：14:35、14:40、14:45、14:50、14:55、15:00 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；15:00 resource guard OK。
- Case study：本小时新增静默退出/队列恢复 case study：`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`。此前两次 GPU lost case study 仍保留在 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md` 与 `History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`。
- 当前硬缺口更新为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。`step_4000.pt` 已不再是缺口，可作为下一次恢复点。
- ETA：最近 20/50/100 个进度样本均值约 `24.3/14.3/10.4s/optimizer step`；预计约 `2.8` 小时到 step 5000 validation/checkpoint。按 50/100 步窗口估计 random-LM pretrain 剩余约 `16.5-22.7` 小时；考虑 12:15 静默退出和此前 GPU lost 风险，完整目标还需 random-LM LP、两个 eval 与最终汇总，当前保守估计约 `22-32` 小时。后续继续每小时同步，并在 `step_5000.pt` 或 `final.pt` 落盘时记录里程碑。

2026-06-11 16:00 小时同步（random-LM step 4400+，继续向 step 5000 推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）由 PID `15648` 的 random-LM 训练进程占用。16:01 现场快照显示 GPU0 `0MiB / 24576MiB`、util `0%`、温度 `40C`；GPU1 `18265MiB / 24576MiB`、util `99%`、温度 `54C`。16:00 resource guard 记录 GPU1 `18265MiB`、util `39%`、温度 `57C`，resource guard OK，无 stop action。
- 训练进度：12:15 从 `step_3000.pt` 自动重启后，16:02 已推进到本次恢复段 `1418 / 7000` local progress step，折算全局约 `4418 / 10000`。距离下一次 step 5000 validation/checkpoint 约 `582` step；random-LM pretrain 到 `final.pt` 还剩约 `5582` step。
- 验证/checkpoint：`outputs/ablation_ums_random_lm_12label/checkpoints/step_4000.pt` 已于 14:25:14 生成，`best.pt` 于 14:25:09 更新。validation loss 轨迹仍为 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340 -> 0.9361 -> 0.8275 -> 0.7444`。`step_5000.pt` 与 `final.pt` 尚未生成。
- 结果判断：step 3500 与 step 4000 validation loss 继续下降，random-init LLM 同架构对照仍按预期收敛；16:02 当前训练 loss 约 `0.8617`，learning rate 约 `1.64e-05`。当前 post-12:15 segment 未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：15:50、15:55、16:00 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；16:00 resource guard OK。
- Case study：本小时无新增失败案例；已记录 case study 仍为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。`step_4000.pt` 仍可作为下一次恢复点。
- ETA：最近 20/50/100 个进度样本均值约 `30.3/31.2/29.0s/optimizer step`；预计约 `5.0` 小时到 step 5000 validation/checkpoint。按 50/100 步窗口估计 random-LM pretrain 剩余约 `45.0-48.3` 小时；完整目标还需 random-LM LP、两个 eval 与最终汇总，当前保守估计约 `48-55` 小时。后续继续每小时同步，并在 `step_5000.pt` 或 `final.pt` 落盘时记录里程碑。

2026-06-11 17:00 小时同步（random-LM step 4500 验证完成，继续向 step 5000 推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：GPU0 空闲；GPU1（bus `00000000:05:00.0`）由 PID `15648` 的 random-LM 训练进程占用。16:53 现场快照显示 GPU0 `0MiB / 24576MiB`、util `0%`；GPU1 `18265MiB / 24576MiB`、util `24%`、温度 `60C`。16:50 resource guard 记录 GPU1 `18265MiB`、温度 `50C`，resource guard OK，无 stop action。
- 训练进度：12:15 从 `step_3000.pt` 自动重启后，16:59 已推进到本次恢复段 `1508 / 7000` local progress step，折算全局约 `4508 / 10000`。距离下一次 step 5000 validation/checkpoint 约 `492` step；random-LM pretrain 到 `final.pt` 还剩约 `5492` step。
- 验证/checkpoint：step 4500 validation 已完成，`val_loss = 0.6806`，并于 16:56 刷新 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`。`step_5000.pt` 与 `final.pt` 尚未生成；最新固定恢复点仍包括 `step_4000.pt`。
- 结果判断：validation loss 轨迹更新为 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340 -> 0.9361 -> 0.8275 -> 0.7444 -> 0.6806`，继续单调下降；16:59 当前训练 loss 约 `0.8672`，learning rate 约 `1.62e-05`。random-init LLM 同架构对照仍按预期收敛，目前结果符合预期。
- 队列/守卫：16:25、16:30、16:35、16:40、16:45、16:50 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；16:50 resource guard OK。当前 post-12:15 segment 未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`。
- Case study：本小时无新增失败案例；已记录 case study 仍为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：过滤验证后最近 50 个训练进度样本均值约 `35.1s/optimizer step`；预计约 `4.8` 小时到 step 5000 validation/checkpoint。按该速度估计 random-LM pretrain 剩余约 `53.5` 小时；完整目标还需 random-LM LP、两个 eval 与最终汇总，当前保守估计约 `56-64` 小时。后续继续每小时同步，并在 `step_5000.pt` 或 `final.pt` 落盘时记录里程碑。

2026-06-11 18:00 小时同步（17:50 静默退出后已从 best.pt 恢复，继续向 step 5000 推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- 中断/恢复：17:35 队列仍识别 `random_lm_running=True`；17:50 resource guard 显示 GPU1 空闲（`0MiB`），队列识别 `random_lm_running=False`，随后自动重新启动 `VIVID_random_lm_gpu1`。训练日志显示 `CUDA_VISIBLE_DEVICES=1`，并从 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 恢复，`Resuming from step 4500`。该事件已保存 case study：`History/20260611_1750_random_lm_silent_exit_best_relaunch/case_study.md`。
- GPU 约束：18:02 现场快照显示 VIVID 主训练 PID `11500` 在 GPU1（bus `00000000:05:00.0`）上运行，GPU1 `16739MiB / 24576MiB`。GPU0 上存在一个小型外部 `GameViewerServer.exe` compute-app 进程，约 `66MiB`，不是 VIVID 训练；VIVID 训练仍只绑定 GPU1。18:00 resource guard OK，无 stop action。
- 训练进度：17:50 从 `best.pt` 恢复后，18:02 已推进到本次恢复段 `48 / 5500` local progress step，折算全局约 `4548 / 10000`。17:50 之前曾推进到约全局 `4566 / 10000`，但未生成 `step_5000.pt`，因此本次恢复重放约 `66` step；当前距离 step 5000 validation/checkpoint 约 `452` step，random-LM pretrain 到 `final.pt` 还剩约 `5452` step。
- 一步多久：18:02 最新一步约 `13.57s/optimizer step`；本次恢复段最近 20/50/100 个有效进度样本均值约 `13.67/12.97/12.77s/optimizer step`。这是重启后短窗速度，后续仍需观察是否回到 30-60s/step 的慢段。
- 验证/checkpoint：最新验证点仍为 step 4500，`val_loss = 0.6806`，`best.pt` 于 16:56 更新。`step_5000.pt` 与 `final.pt` 尚未生成；固定恢复点仍包括 `step_4000.pt` 和当前 `best.pt`。
- 结果判断：validation loss 轨迹仍为 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340 -> 0.9361 -> 0.8275 -> 0.7444 -> 0.6806`，继续单调下降。17:50 静默退出只造成墙钟时间损失和约 66 step 重放，不改变实验定义、batch size、数据或验证协议；目前结果仍符合预期。当前恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`。
- 队列/守卫：17:50 队列完成自动 relaunch；17:55 与 18:00 队列均识别 `random_lm_running=True`，未误启动 random-LM LP 或后续 eval；18:00 resource guard OK。
- Case study：本小时新增 `History/20260611_1750_random_lm_silent_exit_best_relaunch/case_study.md`。此前 case study 仍保留在 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按恢复后 50 步短窗估计，约 `1.6` 小时到 step 5000 validation/checkpoint，random-LM pretrain 约 `19-20` 小时；考虑此前慢段、静默退出和 GPU lost 风险，完整目标保守估计仍按 `24-36` 小时观察。后续继续每小时同步，并在 `step_5000.pt` 或 `final.pt` 落盘时记录里程碑。

2026-06-11 19:00 小时同步（random-LM step 4660+，恢复后稳定推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：19:01 现场快照显示 VIVID 主训练 PID `11500` 在 GPU1（bus `00000000:05:00.0`）上运行，GPU1 `16739MiB / 24576MiB`。GPU0 上仍只有小型外部 `GameViewerServer.exe` compute-app 进程，约 `66MiB`，不是 VIVID 训练；VIVID 训练仍只绑定 GPU1。19:00 resource guard 记录 GPU1 `16739MiB`、util `40%`、温度 `63C`，resource guard OK，无 stop action。
- 训练进度：17:50 从 `best.pt` 恢复后，19:01 已推进到本次恢复段 `161 / 5500` local progress step，折算全局约 `4661 / 10000`。当前距离 step 5000 validation/checkpoint 约 `339` step，random-LM pretrain 到 `final.pt` 还剩约 `5339` step。
- 一步多久：19:01 最新一步约 `23.11s/optimizer step`；本次恢复段最近 20/50/100 个有效进度样本均值约 `20.92/24.05/29.39s/optimizer step`。最近一小时速度比 18:00 刚恢复时更稳，但仍存在 Windows/WDDM 与验证段造成的短窗波动。
- 验证/checkpoint：最新验证点仍为 step 4500，`val_loss = 0.6806`，`best.pt` 于 16:56 更新。`step_5000.pt` 与 `final.pt` 尚未生成；固定恢复点仍包括 `step_4000.pt` 和当前 `best.pt`。
- 结果判断：validation loss 轨迹仍为 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340 -> 0.9361 -> 0.8275 -> 0.7444 -> 0.6806`，继续单调下降。19:01 当前训练 loss 约 `0.8204`，learning rate 约 `1.60e-05`；当前恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：18:35、18:40、18:45、18:50、18:55、19:00 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；19:00 resource guard OK。
- Case study：本小时无新增失败案例；已记录 case study 仍为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`、`History/20260611_1750_random_lm_silent_exit_best_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `24.05s/optimizer step` 估计，约 `2.3` 小时到 step 5000 validation/checkpoint，random-LM pretrain 剩余约 `35.7` 小时；按最近 100 步平均 `29.39s/step` 估计，pretrain 剩余约 `43.6` 小时。完整目标还需 random-LM LP、两个 eval 与最终汇总，考虑历史静默退出/GPU lost 风险，当前保守估计约 `40-52` 小时。后续继续每小时同步，并在 `step_5000.pt` 或 `final.pt` 落盘时记录里程碑。

2026-06-11 20:00 小时同步（random-LM step 4850+，接近 step 5000 验证点）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：20:01 现场快照显示 VIVID 主训练 PID `11500` 在 GPU1（bus `00000000:05:00.0`）上运行，GPU1 `18263MiB / 24576MiB`。GPU0 上仍只有小型外部 `GameViewerServer.exe` compute-app 进程，约 `66MiB`，不是 VIVID 训练；VIVID 训练仍只绑定 GPU1。20:00 resource guard 记录 GPU1 `18263MiB`、util `5%`、温度 `59C`，resource guard OK，无 stop action。
- 训练进度：17:50 从 `best.pt` 恢复后，20:01 已推进到本次恢复段 `351 / 5500` local progress step，折算全局约 `4851 / 10000`。当前距离 step 5000 validation/checkpoint 约 `149` step，random-LM pretrain 到 `final.pt` 还剩约 `5149` step。
- 一步多久：20:01 最新一步约 `24.75s/optimizer step`；本次恢复段最近 20/50/100 个有效进度样本均值约 `16.00/13.90/16.80s/optimizer step`。本小时平均速度明显好于 19:00 窗口，预计较快进入 step 5000 验证段。
- 验证/checkpoint：最新验证点仍为 step 4500，`val_loss = 0.6806`，`best.pt` 于 16:56 更新。`step_5000.pt` 与 `final.pt` 尚未生成；固定恢复点仍包括 `step_4000.pt` 和当前 `best.pt`。
- 结果判断：validation loss 轨迹仍为 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340 -> 0.9361 -> 0.8275 -> 0.7444 -> 0.6806`，继续单调下降。20:01 当前训练 loss 约 `0.8318`，learning rate 约 `1.57e-05`；当前恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：19:30、19:35、19:40、19:45、19:50、19:55、20:00 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；20:00 resource guard OK。
- Case study：本小时无新增失败案例；已记录 case study 仍为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`、`History/20260611_1750_random_lm_silent_exit_best_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `13.90s/optimizer step` 估计，约 `0.6` 小时到 step 5000 validation/checkpoint，random-LM pretrain 剩余约 `19.9` 小时；按最近 100 步平均 `16.80s/step` 估计，pretrain 剩余约 `24.0` 小时。完整目标还需 random-LM LP、两个 eval 与最终汇总，考虑历史静默退出/GPU lost 风险，当前保守估计约 `24-36` 小时。后续继续每小时同步，并在 `step_5000.pt` 或 `final.pt` 落盘时记录里程碑。

2026-06-11 20:52 里程碑补记（random-LM step 5000 验证完成）：
- step 5000 validation 已完成，`val_loss = 0.6277`，继续优于 step 4500 的 `0.6806`；validation loss 轨迹更新为 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340 -> 0.9361 -> 0.8275 -> 0.7444 -> 0.6806 -> 0.6277`。
- `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 于 20:46:18 更新，`outputs/ablation_ums_random_lm_12label/checkpoints/step_5000.pt` 于 20:46:22 生成；`final.pt` 尚未生成。
- 训练已自动继续进入 step 5000 之后的段落，20:52 日志尾部显示本次恢复段已推进到约 `527 / 5500` local progress step，折算全局约 `5027 / 10000`。当前无新增错误；该里程碑不需要 case study。
- `step_5000.pt` 已成为新的固定恢复点；当前硬缺口更新后仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。

2026-06-11 21:00 小时同步（random-LM step 5000 已落盘，继续向 final 推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：21:02 现场快照显示 VIVID 主训练 PID `11500` 在 GPU1（bus `00000000:05:00.0`）上运行，GPU1 `18265MiB / 24576MiB`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；VIVID 训练仍只绑定 GPU1。21:00 resource guard 记录 GPU1 `18265MiB`、util `31%`、温度 `60C`，resource guard OK，无 stop action。
- 训练进度：17:50 从 `best.pt` 恢复后，21:02 已推进到本次恢复段 `557 / 5500` local progress step，折算全局约 `5057 / 10000`。random-LM pretrain 到 `final.pt` 还剩约 `4943` optimizer step。
- 一步多久：21:02 最新一步约 `15.47s/optimizer step`；本次恢复段最近 20/50/100 个有效进度样本均值约 `15.54/17.01/19.74s/optimizer step`。step 5000 验证后已回到正常训练节奏。
- 验证/checkpoint：step 5000 validation 已完成，`val_loss = 0.6277`；`outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 于 20:46:18 更新，`outputs/ablation_ums_random_lm_12label/checkpoints/step_5000.pt` 于 20:46:22 生成。`final.pt` 尚未生成；`step_5000.pt` 是当前最新固定恢复点。
- 结果判断：validation loss 轨迹为 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340 -> 0.9361 -> 0.8275 -> 0.7444 -> 0.6806 -> 0.6277`，继续单调下降。21:02 当前训练 loss 约 `0.7741`，learning rate 约 `1.53e-05`；当前恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：20:30、20:35、20:40、20:45、20:50、20:55、21:00 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；21:00 resource guard OK。
- Case study：本小时无新增失败案例；已记录 case study 仍为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`、`History/20260611_1750_random_lm_silent_exit_best_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `17.01s/optimizer step` 估计，random-LM pretrain 剩余约 `23.4` 小时；按最近 100 步平均 `19.74s/step` 估计，pretrain 剩余约 `27.1` 小时。完整目标还需 random-LM LP、两个 eval 与最终汇总，考虑历史静默退出/GPU lost 风险，当前保守估计约 `28-40` 小时。后续继续每小时同步，并在 `final.pt` 落盘或队列切到 LP 时记录里程碑。

2026-06-11 22:00 小时同步（random-LM step 5290+，step 5000 后稳定推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：22:02 现场快照显示 VIVID 主训练 PID `11500` 在 GPU1（bus `00000000:05:00.0`）上运行，GPU1 `18265MiB / 24576MiB`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；VIVID 训练仍只绑定 GPU1。22:00 resource guard 记录 GPU1 `18265MiB`、util `25%`、温度 `64C`，resource guard OK，无 stop action。
- 训练进度：17:50 从 `best.pt` 恢复后，22:02 已推进到本次恢复段 `797 / 5500` local progress step，折算全局约 `5297 / 10000`。random-LM pretrain 到 `final.pt` 还剩约 `4703` optimizer step。
- 一步多久：22:02 最新一步约 `19.74s/optimizer step`；本次恢复段最近 20/50/100 个有效进度样本均值约 `14.63/13.88/13.41s/optimizer step`。step 5000 后训练节奏稳定，未见明显卡顿。
- 验证/checkpoint：最新验证点仍为 step 5000，`val_loss = 0.6277`；`outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 于 20:46:18 更新，`outputs/ablation_ums_random_lm_12label/checkpoints/step_5000.pt` 于 20:46:22 生成。`final.pt` 尚未生成；`step_5000.pt` 是当前最新固定恢复点。
- 结果判断：validation loss 轨迹为 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340 -> 0.9361 -> 0.8275 -> 0.7444 -> 0.6806 -> 0.6277`，继续单调下降。22:02 当前训练 loss 约 `0.8433`，learning rate 约 `1.49e-05`；当前恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：21:30、21:35、21:40、21:45、21:50、21:55、22:00 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；22:00 resource guard OK。
- Case study：本小时无新增失败案例；已记录 case study 仍为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`、`History/20260611_1750_random_lm_silent_exit_best_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `13.88s/optimizer step` 估计，random-LM pretrain 剩余约 `18.1` 小时；按最近 100 步平均 `13.41s/step` 估计，pretrain 剩余约 `17.5` 小时。完整目标还需 random-LM LP、两个 eval 与最终汇总，考虑历史静默退出/GPU lost 风险，当前保守估计约 `22-34` 小时。后续继续每小时同步，并在 `final.pt` 落盘或队列切到 LP 时记录里程碑。

2026-06-11 23:00 小时同步（random-LM step 5500 验证完成，继续向 final 推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：23:01 现场快照显示 VIVID 主训练 PID `11500` 在 GPU1（bus `00000000:05:00.0`）上运行，GPU1 `18265MiB / 24576MiB`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；VIVID 训练仍只绑定 GPU1。23:00 resource guard 记录 GPU1 `18265MiB`、util `24%`、温度 `67C`，resource guard OK，无 stop action。
- 训练进度：17:50 从 `best.pt` 恢复后，23:01 已推进到本次恢复段 `1027 / 5500` local progress step，折算全局约 `5527 / 10000`。random-LM pretrain 到 `final.pt` 还剩约 `4473` optimizer step。
- 一步多久：23:01 最新一步约 `14.98s/optimizer step`；本次恢复段最近 20/50/100 个有效进度样本均值约 `12.25/16.72/14.05s/optimizer step`。本小时有短暂低 util/log flush 空档，但复查后日志恢复新鲜，不构成失败案例。
- 验证/checkpoint：step 5500 validation 已完成，`val_loss = 0.5917`，`outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 于 22:55:43 更新。`step_5000.pt` 仍是最新固定 step checkpoint，`final.pt` 尚未生成。
- 结果判断：validation loss 轨迹为 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340 -> 0.9361 -> 0.8275 -> 0.7444 -> 0.6806 -> 0.6277 -> 0.5917`，继续单调下降。23:01 当前训练 loss 约 `0.6779`，learning rate 约 `1.46e-05`；当前恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：22:30、22:35、22:40、22:45、22:50、22:55、23:00 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；23:00 resource guard OK。
- Case study：本小时无新增失败案例；已记录 case study 仍为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`、`History/20260611_1750_random_lm_silent_exit_best_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `16.72s/optimizer step` 估计，random-LM pretrain 剩余约 `20.8` 小时；按最近 100 步平均 `14.05s/step` 估计，pretrain 剩余约 `17.5` 小时。完整目标还需 random-LM LP、两个 eval 与最终汇总，考虑历史静默退出/GPU lost 风险，当前保守估计约 `22-34` 小时。后续继续每小时同步，并在 `final.pt` 落盘或队列切到 LP 时记录里程碑。

2026-06-12 00:00 小时同步（跨日后 random-LM step 5770+，继续向 final 推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：00:02 现场快照显示 VIVID 主训练 PID `11500` 在 GPU1（bus `00000000:05:00.0`）上运行，GPU1 `18265MiB / 24576MiB`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；VIVID 训练仍只绑定 GPU1。00:00 resource guard 记录 GPU1 `18265MiB`、util `3%`、温度 `61C`，resource guard OK，无 stop action。
- 训练进度：17:50 从 `best.pt` 恢复后，00:02 已推进到本次恢复段 `1272 / 5500` local progress step，折算全局约 `5772 / 10000`。random-LM pretrain 到 `final.pt` 还剩约 `4228` optimizer step。
- 一步多久：00:02 最新一步约 `24.07s/optimizer step`；本次恢复段最近 20/50/100 个有效进度样本均值约 `16.28/16.44/15.27s/optimizer step`。跨日后日志仍持续刷新，速度保持在正常波动范围。
- 验证/checkpoint：最新验证点仍为 step 5500，`val_loss = 0.5917`，`outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 于 2026-06-11 22:55:43 更新。`step_5000.pt` 仍是最新固定 step checkpoint，`final.pt` 尚未生成。
- 结果判断：validation loss 轨迹为 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340 -> 0.9361 -> 0.8275 -> 0.7444 -> 0.6806 -> 0.6277 -> 0.5917`，继续单调下降。00:02 当前训练 loss 约 `0.7213`，learning rate 约 `1.41e-05`；当前恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：23:30、23:35、23:40、23:45、23:50、23:55、00:00 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；00:00 resource guard OK。
- Case study：本小时无新增失败案例；已记录 case study 仍为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`、`History/20260611_1750_random_lm_silent_exit_best_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `16.44s/optimizer step` 估计，random-LM pretrain 剩余约 `19.3` 小时；按最近 100 步平均 `15.27s/step` 估计，pretrain 剩余约 `17.9` 小时。完整目标还需 random-LM LP、两个 eval 与最终汇总，考虑历史静默退出/GPU lost 风险，当前保守估计约 `22-34` 小时。后续继续每小时同步，并在 `final.pt` 落盘或队列切到 LP 时记录里程碑。

2026-06-12 01:00 小时同步（random-LM step 6000 验证完成，继续向 final 推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：01:18 现场快照显示 VIVID 主训练 PID `11500` 在 GPU1（bus `00000000:05:00.0`）上运行，GPU1 `18265MiB / 24576MiB`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；VIVID 训练仍只绑定 GPU1。01:05 resource guard 记录 GPU1 `18265MiB`、util `99%`、温度 `62C`，resource guard OK，无 stop action。
- 训练进度：17:50 从 `best.pt` 恢复后，01:18 已推进到本次恢复段 `1514 / 5500` local progress step，折算全局约 `6014 / 10000`。random-LM pretrain 到 `final.pt` 还剩约 `3986` optimizer step。
- 一步多久：01:18 最新一步约 `18.94s/optimizer step`；包含 step 6000 验证后恢复段的最近 20/50/100 个有效进度样本均值约 `43.83/27.12/21.72s/optimizer step`。该短窗受刚完成验证后的几个高耗时 step 影响，后续会继续用 50/100 步窗口观察回落情况。
- 验证/checkpoint：step 6000 validation 已完成，`val_loss = 0.5609`，`outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 于 01:14:35 更新，`outputs/ablation_ums_random_lm_12label/checkpoints/step_6000.pt` 于 01:14:43 生成。`final.pt` 尚未生成；`step_6000.pt` 是当前最新固定恢复点。
- 结果判断：validation loss 轨迹为 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340 -> 0.9361 -> 0.8275 -> 0.7444 -> 0.6806 -> 0.6277 -> 0.5917 -> 0.5609`，继续单调下降。01:18 当前训练 loss 约 `0.6545`，learning rate 约 `1.38e-05`；当前恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：00:30、00:35、00:40、00:45、00:50、00:55、01:00、01:05 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK。
- Case study：本小时无新增失败案例；已记录 case study 仍为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`、`History/20260611_1750_random_lm_silent_exit_best_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `27.12s/optimizer step` 估计，random-LM pretrain 剩余约 `30.0` 小时；按最近 100 步平均 `21.72s/step` 估计，pretrain 剩余约 `24.1` 小时。完整目标还需 random-LM LP、两个 eval 与最终汇总，考虑历史静默退出/GPU lost 风险，当前保守估计约 `28-42` 小时。后续继续每小时同步，并在 `final.pt` 落盘或队列切到 LP 时记录里程碑。

2026-06-12 01:25 失败案例补记（random-LM 静默退出后从 step_6000.pt 自动恢复）：
- 事件：01:15 队列仍识别 `random_lm_running=True`；01:25 resource guard 显示 GPU1 空闲（`0MiB`、util `0%`），队列识别 `random_lm_running=False`，随后自动重新启动 `VIVID_random_lm_gpu1`。
- 恢复点：训练日志显示 `CUDA_VISIBLE_DEVICES=1`，并从 `outputs/ablation_ums_random_lm_12label/checkpoints/step_6000.pt` 恢复，`Checkpoint loaded ... step_6000.pt`，`Resuming from step 6000`。新主训练 PID 为 `9556`，运行在 GPU1（bus `00000000:05:00.0`）。
- 影响：step 6000 验证和 checkpoint 已完成，`val_loss = 0.5609`，`step_6000.pt` 于 01:14:43 生成；此次静默退出只造成 step 6000 之后少量训练步重放和墙钟时间损失，不改变实验定义、batch size、数据、验证协议或 GPU 约束。
- Case study：已保存至 `History/20260612_0125_random_lm_silent_exit_step6000_relaunch/case_study.md`。
- 当前状态：重启后任务继续运行在 GPU1；后续进度计算应以 `step_6000.pt` 为恢复基准，即 `global_step ~= 6000 + 当前 local step`。当前硬缺口不变：`outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。

2026-06-12 02:00 小时同步（random-LM 从 step_6000.pt 恢复后继续稳定推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：02:01 现场快照显示 VIVID 主训练 PID `9556` 在 GPU1（bus `00000000:05:00.0`）上运行，GPU1 `16739MiB / 24576MiB`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；VIVID 训练仍只绑定 GPU1。02:00 resource guard 记录 GPU1 `16739MiB`、util `15%`、温度 `62C`，resource guard OK，无 stop action。
- 训练进度：01:25 从 `step_6000.pt` 自动重启后，02:01 已推进到本次恢复段 `113 / 4000` local progress step，折算全局约 `6113 / 10000`。random-LM pretrain 到 `final.pt` 还剩约 `3887` optimizer step。
- 一步多久：02:01 最新一步约 `16.73s/optimizer step`；本次恢复段最近 20/50/100 个有效进度样本均值约 `12.43/12.83/16.32s/optimizer step`。恢复后速度已回到正常区间。
- 验证/checkpoint：最新验证点仍为 step 6000，`val_loss = 0.5609`，`outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 于 01:14:35 更新，`outputs/ablation_ums_random_lm_12label/checkpoints/step_6000.pt` 于 01:14:43 生成。`final.pt` 尚未生成；`step_6000.pt` 是当前最新固定恢复点。
- 结果判断：validation loss 轨迹为 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340 -> 0.9361 -> 0.8275 -> 0.7444 -> 0.6806 -> 0.6277 -> 0.5917 -> 0.5609`，继续单调下降。02:01 当前训练 loss 约 `0.6720`，learning rate 约 `1.36e-05`；当前恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：01:30、01:35、01:40、01:45、01:50、01:55、02:00 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK。
- Case study：本小时新增静默退出 case study：`History/20260612_0125_random_lm_silent_exit_step6000_relaunch/case_study.md`。此前 case study 仍为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`、`History/20260611_1750_random_lm_silent_exit_best_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `12.83s/optimizer step` 估计，random-LM pretrain 剩余约 `13.9` 小时；按最近 100 步平均 `16.32s/step` 估计，pretrain 剩余约 `17.6` 小时。完整目标还需 random-LM LP、两个 eval 与最终汇总，考虑历史静默退出/GPU lost 风险，当前保守估计约 `18-30` 小时。后续继续每小时同步，并在 `final.pt` 落盘或队列切到 LP 时记录里程碑。

2026-06-12 03:00 小时同步（random-LM step 6310+，继续向 step 6500 推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：03:01 现场快照显示 VIVID 主训练 PID `9556` 在 GPU1（bus `00000000:05:00.0`）上运行，GPU1 `16739MiB / 24576MiB`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；VIVID 训练仍只绑定 GPU1。03:00 resource guard 记录 GPU1 `16739MiB`、util `32%`、温度 `66C`，resource guard OK，无 stop action。
- 训练进度：01:25 从 `step_6000.pt` 自动重启后，03:01 已推进到本次恢复段 `315 / 4000` local progress step，折算全局约 `6315 / 10000`。距离 step 6500 validation/checkpoint 约 `185` step；random-LM pretrain 到 `final.pt` 还剩约 `3685` optimizer step。
- 一步多久：03:01 最新一步约 `21.29s/optimizer step`；本次恢复段最近 20/50/100 个有效进度样本均值约 `17.56/20.16/19.30s/optimizer step`。按最近 50 步估计，约 `1.0` 小时到 step 6500。
- 验证/checkpoint：最新验证点仍为 step 6000，`val_loss = 0.5609`；`outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 于 01:14:35 更新，`outputs/ablation_ums_random_lm_12label/checkpoints/step_6000.pt` 于 01:14:43 生成。`final.pt` 尚未生成；`step_6000.pt` 是当前最新固定恢复点。
- 结果判断：validation loss 轨迹为 `3.7193 -> 2.5412 -> 1.8702 -> 1.4182 -> 1.1340 -> 0.9361 -> 0.8275 -> 0.7444 -> 0.6806 -> 0.6277 -> 0.5917 -> 0.5609`，继续单调下降。03:01 当前训练 loss 约 `0.7188`，learning rate 约 `1.33e-05`；当前恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：02:30、02:35、02:40、02:45、02:50、02:55、03:00 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK。
- Case study：本小时无新增失败案例；已记录 case study 仍为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`、`History/20260611_1750_random_lm_silent_exit_best_relaunch/case_study.md`、`History/20260612_0125_random_lm_silent_exit_step6000_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `20.16s/optimizer step` 估计，random-LM pretrain 剩余约 `20.6` 小时；按最近 100 步平均 `19.30s/step` 估计，pretrain 剩余约 `19.8` 小时。完整目标还需 random-LM LP、两个 eval 与最终汇总，考虑历史静默退出/GPU lost 风险，当前保守估计约 `22-34` 小时。后续继续每小时同步，并在 step 6500、`final.pt` 落盘或队列切到 LP 时记录里程碑。

2026-06-12 04:00 小时同步（random-LM step 6500 验证完成，继续向 step 7000 推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：04:04 现场快照显示 VIVID 主训练 PID `9556` 在 GPU1（bus `00000000:05:00.0`）上运行，GPU1 `18265MiB / 24576MiB`，util `76%`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；VIVID 训练仍只绑定 GPU1。04:00 resource guard 记录 GPU1 `18265MiB`、util `36%`、温度 `66C`，resource guard OK，无 stop action。
- 训练进度：01:25 从 `step_6000.pt` 自动重启后，04:04 已推进到本次恢复段 `517 / 4000` local progress step，折算全局约 `6517 / 10000`。距离 step 7000 validation/checkpoint 约 `483` step；random-LM pretrain 到 `final.pt` 还剩约 `3483` optimizer step。
- 一步多久：04:04 最新一步约 `27.03s/optimizer step`；本次恢复段最近 20/50/100 个有效进度样本均值约 `33.40/22.17/18.21s/optimizer step`。最近 20 步偏慢主要受刚完成 step 6500 验证后的短窗影响，50/100 步窗口仍在可接受范围。
- 验证/checkpoint：step 6500 validation 已完成，`val_loss = 0.5364`，`outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 于 03:58:34 更新。当前固定 step checkpoint 仍是 `step_6000.pt`；未发现 `step_6500.pt`，符合当前固定 checkpoint 每 1000 step 落盘、best checkpoint 每个改进验证点更新的模式。`final.pt` 尚未生成。
- 结果判断：validation loss 轨迹为 `2.5412 -> 1.8702 -> 1.4182 -> 1.1340 -> 0.9361 -> 0.8275 -> 0.7444 -> 0.6806 -> 0.6277 -> 0.5917 -> 0.5609 -> 0.5364`，继续下降。04:04 当前训练 loss 约 `0.6899`，learning rate 约 `1.30e-05`；当前恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：03:30、03:35、03:40、03:45、03:50、03:55、04:00 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；04:00 resource guard OK。
- Case study：本小时无新增失败案例；已记录 case study 仍为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`、`History/20260611_1750_random_lm_silent_exit_best_relaunch/case_study.md`、`History/20260612_0125_random_lm_silent_exit_step6000_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `22.17s/optimizer step` 估计，约 `3.0` 小时到 step 7000 validation/checkpoint，random-LM pretrain 剩余约 `21.4` 小时；按最近 100 步平均 `18.21s/step` 估计，pretrain 剩余约 `17.6` 小时。完整目标还需 random-LM LP、两个 eval 与最终汇总，考虑历史静默退出/GPU lost 风险，当前保守估计约 `22-34` 小时。后续继续每小时同步，并在 step 7000、`final.pt` 落盘或队列切到 LP 时记录里程碑。

2026-06-12 05:00 小时同步（random-LM step 6575+，慢速但持续推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：05:00 现场快照显示 VIVID 主训练 PID `9556` 在 GPU1（bus `00000000:05:00.0`）上运行，GPU1 `18265MiB / 24576MiB`，util `72%`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；VIVID 训练仍只绑定 GPU1。04:55 resource guard 记录 GPU1 `18265MiB`、util `37%`、温度 `62C`，resource guard OK，无 stop action。
- 训练进度：01:25 从 `step_6000.pt` 自动重启后，05:00 已推进到本次恢复段 `575 / 4000` local progress step，折算全局约 `6575 / 10000`。距离 step 7000 validation/checkpoint 约 `425` step；random-LM pretrain 到 `final.pt` 还剩约 `3425` optimizer step。
- 一步多久：05:00 最新一步约 `58.53s/optimizer step`；本次恢复段最近 20/50/100 个有效进度样本均值约 `55.32/57.86/45.35s/optimizer step`。本小时 04:40 后明显进入慢速区间，但日志仍在更新、GPU1 进程仍在运行，当前判断为慢速推进而非失败。
- 验证/checkpoint：最新验证点仍为 step 6500，`val_loss = 0.5364`，`outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 于 03:58:34 更新。当前固定 step checkpoint 仍是 `step_6000.pt`；`final.pt` 尚未生成。
- 结果判断：最近 validation loss 轨迹为 `0.5917 -> 0.5609 -> 0.5364`，继续下降。05:00 当前训练 loss 约 `0.6241`，learning rate 约 `1.29e-05`；当前恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：04:30、04:35、04:40、04:45、04:50、04:55 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK。
- Case study：本小时无新增失败案例；已记录 case study 仍为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`、`History/20260611_1750_random_lm_silent_exit_best_relaunch/case_study.md`、`History/20260612_0125_random_lm_silent_exit_step6000_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `57.86s/optimizer step` 估计，约 `6.8` 小时到 step 7000 validation/checkpoint，random-LM pretrain 剩余约 `55.0` 小时；按最近 100 步平均 `45.35s/step` 估计，pretrain 剩余约 `43.1` 小时。完整目标还需 random-LM LP、两个 eval 与最终汇总，考虑当前慢速区间和历史静默退出/GPU lost 风险，当前保守估计约 `46-66` 小时。后续继续每小时同步，并在 step 7000、`final.pt` 落盘或队列切到 LP 时记录里程碑。

2026-06-12 06:00 小时同步（random-LM step 6632，慢速但稳定推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：06:02 现场快照显示 VIVID 主训练 PID `9556` 在 GPU1（bus `00000000:05:00.0`）上运行，GPU1 `18265MiB / 24576MiB`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；VIVID 训练仍只绑定 GPU1。06:00 resource guard 记录 GPU0 `0MiB`、GPU1 `18265MiB`、util `95%`、温度 `60C`，resource guard OK，无 stop action。
- 训练进度：01:25 从 `step_6000.pt` 自动重启后，06:02 已推进到本次恢复段 `632 / 4000` local progress step，折算全局约 `6632 / 10000`。距离 step 7000 validation/checkpoint 约 `368` step；random-LM pretrain 到 `final.pt` 还剩约 `3368` optimizer step。
- 一步多久：06:02 最新一步约 `64.03s/optimizer step`；本次恢复段最近 20/50/100 个有效进度样本均值约 `71.55/66.87/61.84s/optimizer step`。本小时仍处于慢速区间，但日志持续刷新、GPU1 进程和队列状态正常，当前判断为慢速推进而非失败。
- 验证/checkpoint：最新验证点仍为 step 6500，`val_loss = 0.5364`，`outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 于 03:58:34 更新。当前固定 step checkpoint 仍是 `step_6000.pt`；`final.pt` 尚未生成。
- 结果判断：最近 validation loss 轨迹为 `0.5917 -> 0.5609 -> 0.5364`，继续下降。06:02 当前训练 loss 约 `0.7015`，learning rate 约 `1.28e-05`；当前恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：05:35、05:40、05:45、05:50、05:55、06:00 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK。
- Case study：本小时无新增失败案例；已记录 case study 仍为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`、`History/20260611_1750_random_lm_silent_exit_best_relaunch/case_study.md`、`History/20260612_0125_random_lm_silent_exit_step6000_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `66.87s/optimizer step` 估计，约 `6.8` 小时到 step 7000 validation/checkpoint，random-LM pretrain 剩余约 `62.6` 小时；按最近 100 步平均 `61.84s/step` 估计，pretrain 剩余约 `57.9` 小时。完整目标还需 random-LM LP、两个 eval 与最终汇总，考虑当前慢速区间和历史静默退出/GPU lost 风险，当前保守估计约 `64-84` 小时。后续继续每小时同步，并在 step 7000、`final.pt` 落盘或队列切到 LP 时记录里程碑。

2026-06-12 07:00 小时同步（random-LM step 6682，仍在慢速区间但持续推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：07:00 现场快照显示 VIVID 主训练 PID `9556` 在 GPU1（bus `00000000:05:00.0`）上运行，GPU1 `18265MiB / 24576MiB`，util `51%`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；VIVID 训练仍只绑定 GPU1。06:55 resource guard 记录 GPU0 `0MiB`、GPU1 `18265MiB`、util `99%`、温度 `58C`，resource guard OK，无 stop action。
- 训练进度：01:25 从 `step_6000.pt` 自动重启后，07:00 已推进到本次恢复段 `682 / 4000` local progress step，折算全局约 `6682 / 10000`。距离 step 7000 validation/checkpoint 约 `318` step；random-LM pretrain 到 `final.pt` 还剩约 `3318` optimizer step。
- 一步多久：07:00 最新一步约 `86.78s/optimizer step`；本次恢复段最近 20/50/100 个有效进度样本均值约 `71.74/69.30/68.49s/optimizer step`。本小时仍处于慢速区间，但日志持续刷新、GPU1 进程和队列状态正常，当前判断为慢速推进而非失败。
- 验证/checkpoint：最新验证点仍为 step 6500，`val_loss = 0.5364`，`outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 于 03:58:34 更新。当前固定 step checkpoint 仍是 `step_6000.pt`；`final.pt` 尚未生成。
- 结果判断：最近 validation loss 轨迹为 `0.5917 -> 0.5609 -> 0.5364`，继续下降。07:00 当前训练 loss 约 `0.6237`，learning rate 约 `1.27e-05`；当前恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：06:30、06:35、06:40、06:45、06:50、06:55 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK。
- Case study：本小时无新增失败案例；已记录 case study 仍为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`、`History/20260611_1750_random_lm_silent_exit_best_relaunch/case_study.md`、`History/20260612_0125_random_lm_silent_exit_step6000_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `69.30s/optimizer step` 估计，约 `6.1` 小时到 step 7000 validation/checkpoint，random-LM pretrain 剩余约 `63.9` 小时；按最近 100 步平均 `68.49s/step` 估计，pretrain 剩余约 `63.1` 小时。完整目标还需 random-LM LP、两个 eval 与最终汇总，考虑当前慢速区间和历史静默退出/GPU lost 风险，当前保守估计约 `66-86` 小时。后续继续每小时同步，并在 step 7000、`final.pt` 落盘或队列切到 LP 时记录里程碑。

2026-06-12 08:00 小时同步（random-LM step 6734，慢速区间内稳定推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：07:59 现场快照显示 VIVID 主训练 PID `9556` 在 GPU1（bus `00000000:05:00.0`）上运行，GPU1 `18265MiB / 24576MiB`，util `67%`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；VIVID 训练仍只绑定 GPU1。07:55 resource guard 记录 GPU0 `0MiB`、GPU1 `18265MiB`、util `96%`、温度 `60C`，resource guard OK，无 stop action。
- 训练进度：01:25 从 `step_6000.pt` 自动重启后，08:00 已推进到本次恢复段 `734 / 4000` local progress step，折算全局约 `6734 / 10000`。距离 step 7000 validation/checkpoint 约 `266` step；random-LM pretrain 到 `final.pt` 还剩约 `3266` optimizer step。
- 一步多久：08:00 最新一步约 `69.96s/optimizer step`；本次恢复段最近 20/50/100 个有效进度样本均值约 `70.23/68.52/69.63s/optimizer step`。当前仍处于慢速区间，但日志持续刷新、GPU1 进程和队列状态正常，当前判断为慢速推进而非失败。
- 验证/checkpoint：最新验证点仍为 step 6500，`val_loss = 0.5364`，`outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 于 03:58:34 更新。当前固定 step checkpoint 仍是 `step_6000.pt`；`final.pt` 尚未生成。
- 结果判断：最近 validation loss 轨迹为 `0.5917 -> 0.5609 -> 0.5364`，继续下降。08:00 当前训练 loss 约 `0.6768`，learning rate 约 `1.26e-05`；当前恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：07:30、07:35、07:40、07:45、07:50、07:55 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK。
- Case study：本小时无新增失败案例；已记录 case study 仍为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`、`History/20260611_1750_random_lm_silent_exit_best_relaunch/case_study.md`、`History/20260612_0125_random_lm_silent_exit_step6000_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `68.52s/optimizer step` 估计，约 `5.1` 小时到 step 7000 validation/checkpoint，random-LM pretrain 剩余约 `62.2` 小时；按最近 100 步平均 `69.63s/step` 估计，pretrain 剩余约 `63.2` 小时。完整目标还需 random-LM LP、两个 eval 与最终汇总，考虑当前慢速区间和历史静默退出/GPU lost 风险，当前保守估计约 `66-86` 小时。后续继续每小时同步，并在 step 7000、`final.pt` 落盘或队列切到 LP 时记录里程碑。

2026-06-12 09:00 小时同步（random-LM step 6790，仍在 GPU1 稳定推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：09:01 现场快照显示 VIVID 主训练 PID `9556` 在 GPU1（bus `00000000:05:00.0`）上运行，GPU1 `18265MiB / 24576MiB`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；VIVID 训练仍只绑定 GPU1。09:00 resource guard 记录 GPU0 `0MiB`、GPU1 `18265MiB`、util `21%`、温度 `63C`，resource guard OK，无 stop action。
- 训练进度：01:25 从 `step_6000.pt` 自动重启后，09:01 已推进到本次恢复段 `790 / 4000` local progress step，折算全局约 `6790 / 10000`。距离 step 7000 validation/checkpoint 约 `210` step；random-LM pretrain 到 `final.pt` 还剩约 `3210` optimizer step。
- 一步多久：09:01 最新一步约 `48.89s/optimizer step`；本次恢复段最近 20/50/100 个有效进度样本均值约 `56.25/65.61/68.09s/optimizer step`。相较 08:00，短窗速度略有恢复，但 50/100 步窗口仍按慢速区间估计；日志持续刷新、GPU1 进程和队列状态正常，当前判断为慢速推进而非失败。
- 验证/checkpoint：最新验证点仍为 step 6500，`val_loss = 0.5364`，`outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 于 03:58:34 更新。当前固定 step checkpoint 仍是 `step_6000.pt`；`step_7000.pt` 和 `final.pt` 尚未生成。
- 结果判断：最近 validation loss 轨迹为 `0.5917 -> 0.5609 -> 0.5364`，继续下降。09:01 当前训练 loss 约 `0.7482`，learning rate 约 `1.26e-05`；当前恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：08:30、08:35、08:40、08:45、08:50、08:55、09:00 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK。
- Case study：本小时无新增失败案例；已记录 case study 仍为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`、`History/20260611_1750_random_lm_silent_exit_best_relaunch/case_study.md`、`History/20260612_0125_random_lm_silent_exit_step6000_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `65.61s/optimizer step` 估计，约 `3.8` 小时到 step 7000 validation/checkpoint，random-LM pretrain 剩余约 `58.5` 小时；按最近 100 步平均 `68.09s/step` 估计，pretrain 剩余约 `60.7` 小时。完整目标还需 random-LM LP、两个 eval 与最终汇总，考虑当前慢速区间和历史静默退出/GPU lost 风险，当前保守估计约 `62-82` 小时。后续继续每小时同步，并在 step 7000、`final.pt` 落盘或队列切到 LP 时记录里程碑。

2026-06-12 10:00 小时同步（random-LM 10:00 静默退出，10:10 从 best.pt 恢复）：
- 当前阶段：`random-LM same-architecture` 预训练仍是当前瓶颈任务。10:00 前上一段训练已推进到本次 `step_6000.pt` 恢复段 local `889 / 4000`，折算全局约 `6889 / 10000`，但尚未到 step 7000 固定 checkpoint。
- 失败/恢复：10:00 队列检测到 `random_lm_running=False`，GPU1 显存降到 `0MiB`，说明原训练进程已静默退出。10:00 自动拉起一次，模型加载到 checkpoint load 附近后以 exit code `1073807364` 退出，日志中未见明确 Python `Traceback`、`CUDA out of memory` 或 `GPU is lost`。10:10 队列第二次自动拉起成功，从 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 恢复，日志显示 `Resuming from step 6500`，并已重新进入 training loop。
- GPU 约束：10:23 现场快照显示当前 VIVID 主训练 PID `4840` 在 GPU1（bus `00000000:05:00.0`）运行，GPU1 `16739MiB / 24576MiB`，util `42%`。GPU0 `0MiB / 24576MiB`；10:20 resource guard 记录 GPU1 `16739MiB`、util `99%`、温度 `56C`，resource guard OK，无 stop action。恢复后仍只使用 GPU1。
- 训练进度：恢复后的当前段从 step 6500 开始，10:23 已推进到 local `27 / 3500`，折算全局约 `6527 / 10000`。由于 `step_7000.pt` 尚未生成，本次静默退出造成约 `389` 个未落盘 optimizer step 损失（约 `6889 -> 6500`）。
- 一步多久：10:23 最新一步约 `27.62s/optimizer step`；恢复后当前段最近 20/50/100 个有效进度样本均值约 `17.86/17.90/17.90s/optimizer step`。这个速度只代表刚恢复后的短窗，后续仍需观察是否回到 09:00 前后的慢速区间。
- 验证/checkpoint：最新验证点仍为 step 6500，`val_loss = 0.5364`，`outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 于 03:58:34 更新。`step_7000.pt` 和 `final.pt` 尚未生成。当前恢复源为 `best.pt`，不是 `step_7000.pt`。
- 结果判断：验证损失仍沿 `0.5917 -> 0.5609 -> 0.5364` 下降，实验结果本身仍符合预期；但运行稳定性出现新的失败案例，已记录为 `History/20260612_1000_random_lm_silent_exit_best_relaunch/case_study.md`。当前恢复段未检出新的 `Traceback`、`CUDA out of memory` 或 `GPU is lost`。
- 队列/守卫：10:00、10:10 队列均检测到 random-LM 非运行并尝试拉起；10:15、10:20 队列已识别 `random_lm_running=True`，未重复启动，也未误启动 random-LM LP 或后续 eval；resource guard OK。
- Case study：本小时新增 `History/20260612_1000_random_lm_silent_exit_best_relaunch/case_study.md`。此前 case study 仍为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`、`History/20260611_1750_random_lm_silent_exit_best_relaunch/case_study.md`、`History/20260612_0125_random_lm_silent_exit_step6000_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按恢复后短窗最近 50 步平均 `17.90s/optimizer step` 估计，约 `2.4` 小时到 step 7000 validation/checkpoint，random-LM pretrain 剩余约 `17.3` 小时；但考虑刚发生静默退出且短窗速度可能偏乐观，完整目标当前保守估计约 `24-48` 小时。后续重点观察是否稳定通过 step 7000 并落盘 checkpoint。

2026-06-12 11:00 小时同步（random-LM 从 10:10 恢复后稳定推进到 step 6632）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：11:01 现场快照显示当前 VIVID 主训练 PID `4840` 在 GPU1（bus `00000000:05:00.0`）运行，GPU1 `16739MiB / 24576MiB`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；恢复后仍只使用 GPU1。11:00 resource guard 记录 GPU0 `0MiB`、GPU1 `16739MiB`、util `28%`、温度 `63C`，resource guard OK，无 stop action。
- 训练进度：10:10 从 `best.pt` 恢复到 step 6500 后，11:01 已推进到恢复段 local `132 / 3500`，折算全局约 `6632 / 10000`。距离 step 7000 validation/checkpoint 约 `368` step；random-LM pretrain 到 `final.pt` 还剩约 `3368` optimizer step。
- 一步多久：11:01 最新一步约 `30.37s/optimizer step`；恢复后当前段最近 20/50/100 个有效进度样本均值约 `21.85/21.49/20.72s/optimizer step`。这一小时比 08:00-09:00 慢速区间明显快，但仍需持续观察能否稳定通过 step 7000。
- 验证/checkpoint：最新验证点仍为 step 6500，`val_loss = 0.5364`，`outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 于 03:58:34 更新。`step_7000.pt` 和 `final.pt` 尚未生成；当前恢复源仍是 `best.pt`。
- 结果判断：validation loss 轨迹仍为 `0.5917 -> 0.5609 -> 0.5364`，继续下降。11:01 当前训练 loss 约 `0.7143`，learning rate 约 `1.28e-05`；当前恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：10:20、10:25、10:30、10:35、10:40、10:45、10:50、10:55、11:00 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK。
- Case study：本小时无新增失败案例；10:00 静默退出已记录为 `History/20260612_1000_random_lm_silent_exit_best_relaunch/case_study.md`。此前 case study 仍为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`、`History/20260611_1750_random_lm_silent_exit_best_relaunch/case_study.md`、`History/20260612_0125_random_lm_silent_exit_step6000_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `21.49s/optimizer step` 估计，约 `2.2` 小时到 step 7000 validation/checkpoint，random-LM pretrain 剩余约 `20.1` 小时；按最近 100 步平均 `20.72s/step` 估计，pretrain 剩余约 `19.4` 小时。考虑 10:00 刚发生静默退出，完整目标当前保守估计约 `26-50` 小时。后续重点仍是稳定通过 step 7000 并落盘 checkpoint。

2026-06-12 12:00 小时同步（random-LM step 6798，接近 step 7000 里程碑）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：12:00 现场快照显示当前 VIVID 主训练 PID `4840` 在 GPU1（bus `00000000:05:00.0`）运行，GPU1 `16739MiB / 24576MiB`，util `38%`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；仍只使用 GPU1。12:00 resource guard 记录 GPU0 `0MiB`、GPU1 `16739MiB`、util `99%`、温度 `62C`，resource guard OK，无 stop action。
- 训练进度：10:10 从 `best.pt` 恢复到 step 6500 后，12:00 已推进到恢复段 local `298 / 3500`，折算全局约 `6798 / 10000`。距离 step 7000 validation/checkpoint 约 `202` step；random-LM pretrain 到 `final.pt` 还剩约 `3202` optimizer step。
- 一步多久：12:00 最新一步约 `21.88s/optimizer step`；恢复后当前段最近 20/50/100 个有效进度样本均值约 `19.08/20.25/20.79s/optimizer step`。恢复后速度基本稳定，当前主要等待 step 7000 验证/checkpoint。
- 验证/checkpoint：最新验证点仍为 step 6500，`val_loss = 0.5364`，`outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 于 03:58:34 更新。`step_7000.pt` 和 `final.pt` 尚未生成；当前恢复源仍是 `best.pt`。
- 结果判断：validation loss 轨迹仍为 `0.6277 -> 0.5917 -> 0.5609 -> 0.5364`，继续下降。12:00 当前训练 loss 约 `0.6579`，learning rate 约 `1.26e-05`；当前恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：11:20、11:25、11:30、11:35、11:40、11:45、11:50、11:55、12:00 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK。
- Case study：本小时无新增失败案例；10:00 静默退出已记录为 `History/20260612_1000_random_lm_silent_exit_best_relaunch/case_study.md`。此前 case study 仍为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`、`History/20260611_1750_random_lm_silent_exit_best_relaunch/case_study.md`、`History/20260612_0125_random_lm_silent_exit_step6000_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `20.25s/optimizer step` 估计，约 `1.1` 小时到 step 7000 validation/checkpoint，random-LM pretrain 剩余约 `18.0` 小时；按最近 100 步平均 `20.79s/step` 估计，pretrain 剩余约 `18.5` 小时。考虑 10:00 曾发生静默退出，完整目标当前保守估计约 `24-48` 小时。后续重点是确认 step 7000 checkpoint/validation 是否正常落盘。

2026-06-12 13:00 小时同步（random-LM step 7000 验证/checkpoint 已完成）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：13:24 现场快照显示当前 VIVID 主训练 PID `4840` 在 GPU1（bus `00000000:05:00.0`）运行，GPU1 `18265MiB / 24576MiB`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；仍只使用 GPU1。13:20 resource guard 记录 GPU0 `0MiB`、GPU1 `18265MiB`、util `61%`、温度 `58C`，resource guard OK，无 stop action。
- 训练进度：10:10 从 `best.pt` 恢复到 step 6500 后，13:24 已推进到恢复段 local `511 / 3500`，折算全局约 `7011 / 10000`。random-LM pretrain 到 `final.pt` 还剩约 `2989` optimizer step。
- 一步多久：13:24 最新一步约 `26.64s/optimizer step`；恢复后当前段最近 20/50/100 个有效进度样本均值约 `41.62/28.42/24.56s/optimizer step`。最近 20/50 步被 step 7000 validation/checkpoint 明显拉长；剔除验证影响后，训练本身仍在约 20-25s/step 区间。
- 验证/checkpoint：step 7000 validation 已完成，`val_loss = 0.5167`。`outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 于 13:20:57 更新，`outputs/ablation_ums_random_lm_12label/checkpoints/step_7000.pt` 于 13:21:05 落盘。`final.pt` 尚未生成。
- 结果判断：validation loss 轨迹为 `0.6806 -> 0.6277 -> 0.5917 -> 0.5609 -> 0.5364 -> 0.5167`，继续下降。13:24 当前训练 loss 约 `0.6669`，learning rate 约 `1.23e-05`；当前恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：12:55、13:00、13:05、13:10、13:15、13:20 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK。
- Case study：本小时无新增失败案例；10:00 静默退出已记录为 `History/20260612_1000_random_lm_silent_exit_best_relaunch/case_study.md`。此前 case study 仍为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`、`History/20260611_1750_random_lm_silent_exit_best_relaunch/case_study.md`、`History/20260612_0125_random_lm_silent_exit_step6000_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `28.42s/optimizer step` 估计，random-LM pretrain 剩余约 `23.6` 小时；按最近 100 步平均 `24.56s/step` 估计，pretrain 剩余约 `20.4` 小时。考虑 step 7000 后已有稳定 checkpoint、但历史有静默退出风险，完整目标当前保守估计约 `26-48` 小时。后续重点是继续向 step 8000/9000/10000 推进，并等待 `final.pt` 落盘后自动进入 random-LM LP。

2026-06-12 14:00 小时同步（random-LM step 7095，step 7000 后稳定推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：14:01 现场快照显示当前 VIVID 主训练 PID `4840` 在 GPU1（bus `00000000:05:00.0`）运行，GPU1 `18265MiB / 24576MiB`，util `26%`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；仍只使用 GPU1。14:00 resource guard 记录 GPU0 `0MiB`、GPU1 `18265MiB`、util `25%`、温度 `62C`，resource guard OK，无 stop action。
- 训练进度：10:10 从 `best.pt` 恢复到 step 6500 后，14:01 已推进到恢复段 local `595 / 3500`，折算全局约 `7095 / 10000`。距离 step 8000 validation/checkpoint 约 `905` step；random-LM pretrain 到 `final.pt` 还剩约 `2905` optimizer step。
- 一步多久：14:01 最新一步约 `31.71s/optimizer step`；恢复后当前段最近 20/50/100 个有效进度样本均值约 `24.52/25.34/26.60s/optimizer step`。step 7000 后速度已基本回稳，当前估计用 50/100 步窗口更可靠。
- 验证/checkpoint：最新验证点为 step 7000，`val_loss = 0.5167`。`outputs/ablation_ums_random_lm_12label/checkpoints/step_7000.pt` 已于 13:21:05 落盘；`step_8000.pt` 和 `final.pt` 尚未生成。
- 结果判断：validation loss 轨迹为 `0.6806 -> 0.6277 -> 0.5917 -> 0.5609 -> 0.5364 -> 0.5167`，继续下降。14:01 当前训练 loss 约 `0.5924`，learning rate 约 `1.21e-05`；当前恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：13:20、13:25、13:30、13:35、13:40、13:45、13:50、13:55、14:00 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK。
- Case study：本小时无新增失败案例；10:00 静默退出已记录为 `History/20260612_1000_random_lm_silent_exit_best_relaunch/case_study.md`。此前 case study 仍为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`、`History/20260611_1750_random_lm_silent_exit_best_relaunch/case_study.md`、`History/20260612_0125_random_lm_silent_exit_step6000_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `25.34s/optimizer step` 估计，约 `6.4` 小时到 step 8000 validation/checkpoint，random-LM pretrain 剩余约 `20.4` 小时；按最近 100 步平均 `26.60s/step` 估计，pretrain 剩余约 `21.5` 小时。考虑 step 7000 后已有稳定 checkpoint、但历史有静默退出风险，完整目标当前保守估计约 `26-48` 小时。后续重点是继续向 step 8000/9000/10000 推进，并等待 `final.pt` 落盘后自动进入 random-LM LP。

2026-06-12 15:00 小时同步（random-LM step 7233，稳定向 step 8000 推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：15:00 现场快照显示当前 VIVID 主训练 PID `4840` 在 GPU1（bus `00000000:05:00.0`）运行，GPU1 `18265MiB / 24576MiB`，util `21%`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；仍只使用 GPU1。15:00 resource guard 记录 GPU0 `0MiB`、GPU1 `18265MiB`、util `33%`、温度 `56C`，resource guard OK，无 stop action。
- 训练进度：10:10 从 `best.pt` 恢复到 step 6500 后，15:00 已推进到恢复段 local `733 / 3500`，折算全局约 `7233 / 10000`。距离 step 8000 validation/checkpoint 约 `767` step；random-LM pretrain 到 `final.pt` 还剩约 `2767` optimizer step。
- 一步多久：15:00 最新一步约 `24.63s/optimizer step`；恢复后当前段最近 20/50/100 个有效进度样本均值约 `24.83/24.21/24.00s/optimizer step`。step 7000 后速度已经稳定在约 24s/step。
- 验证/checkpoint：最新验证点为 step 7000，`val_loss = 0.5167`。`outputs/ablation_ums_random_lm_12label/checkpoints/step_7000.pt` 已于 13:21:05 落盘；`step_8000.pt` 和 `final.pt` 尚未生成。
- 结果判断：validation loss 轨迹为 `0.6806 -> 0.6277 -> 0.5917 -> 0.5609 -> 0.5364 -> 0.5167`，继续下降。15:00 当前训练 loss 约 `0.6554`，learning rate 约 `1.20e-05`；当前恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：14:15、14:20、14:25、14:30、14:35、14:40、14:45、14:50、14:55、15:00 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK。
- Case study：本小时无新增失败案例；10:00 静默退出已记录为 `History/20260612_1000_random_lm_silent_exit_best_relaunch/case_study.md`。此前 case study 仍为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`、`History/20260611_1750_random_lm_silent_exit_best_relaunch/case_study.md`、`History/20260612_0125_random_lm_silent_exit_step6000_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `24.21s/optimizer step` 估计，约 `5.2` 小时到 step 8000 validation/checkpoint，random-LM pretrain 剩余约 `18.6` 小时；按最近 100 步平均 `24.00s/step` 估计，pretrain 剩余约 `18.4` 小时。考虑历史静默退出风险，完整目标当前保守估计约 `24-46` 小时。后续重点是继续向 step 8000/9000/10000 推进，并等待 `final.pt` 落盘后自动进入 random-LM LP。

2026-06-12 16:00 小时同步（random-LM step 7386，稳定推进且速度略有改善）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：16:00 现场快照显示当前 VIVID 主训练 PID `4840` 在 GPU1（bus `00000000:05:00.0`）运行，GPU1 `18265MiB / 24576MiB`，util `40%`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；仍只使用 GPU1。16:00 resource guard 记录 GPU0 `0MiB`、GPU1 `18265MiB`、util `28%`、温度 `61C`，resource guard OK，无 stop action。
- 训练进度：10:10 从 `best.pt` 恢复到 step 6500 后，16:00 已推进到恢复段 local `886 / 3500`，折算全局约 `7386 / 10000`。距离 step 8000 validation/checkpoint 约 `614` step；random-LM pretrain 到 `final.pt` 还剩约 `2614` optimizer step。
- 一步多久：16:00 最新一步约 `20.11s/optimizer step`；恢复后当前段最近 20/50/100 个有效进度样本均值约 `18.98/20.54/21.90s/optimizer step`。本小时速度较 15:00 略有改善。
- 验证/checkpoint：最新验证点为 step 7000，`val_loss = 0.5167`。`outputs/ablation_ums_random_lm_12label/checkpoints/step_7000.pt` 已于 13:21:05 落盘；`step_8000.pt` 和 `final.pt` 尚未生成。
- 结果判断：validation loss 轨迹为 `0.6806 -> 0.6277 -> 0.5917 -> 0.5609 -> 0.5364 -> 0.5167`，继续下降。16:00 当前训练 loss 约 `0.6613`，learning rate 约 `1.18e-05`；当前恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：15:20、15:25、15:30、15:35、15:40、15:45、15:50、15:55、16:00 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK。
- Case study：本小时无新增失败案例；10:00 静默退出已记录为 `History/20260612_1000_random_lm_silent_exit_best_relaunch/case_study.md`。此前 case study 仍为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`、`History/20260611_1750_random_lm_silent_exit_best_relaunch/case_study.md`、`History/20260612_0125_random_lm_silent_exit_step6000_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `20.54s/optimizer step` 估计，约 `3.5` 小时到 step 8000 validation/checkpoint，random-LM pretrain 剩余约 `14.9` 小时；按最近 100 步平均 `21.90s/step` 估计，pretrain 剩余约 `15.9` 小时。考虑历史静默退出风险，完整目标当前保守估计约 `22-42` 小时。后续重点是继续向 step 8000/9000/10000 推进，并等待 `final.pt` 落盘后自动进入 random-LM LP。

2026-06-12 17:00 小时同步（random-LM step 7509，step 7500 验证继续下降）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：17:00 现场快照显示当前 VIVID 主训练 PID `4840` 在 GPU1（bus `00000000:05:00.0`）运行，GPU1 `18265MiB / 24576MiB`，util `95%`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；仍只使用 GPU1。17:00 resource guard 记录 GPU0 `0MiB`、GPU1 `18265MiB`、util `15%`、温度 `61C`，resource guard OK，无 stop action。
- 训练进度：10:10 从 `best.pt` 恢复到 step 6500 后，17:00 已推进到恢复段 local `1009 / 3500`，折算全局约 `7509 / 10000`。距离 step 8000 validation/checkpoint 约 `491` step；random-LM pretrain 到 `final.pt` 还剩约 `2491` optimizer step。
- 一步多久：17:00 最新一步约 `32.65s/optimizer step`；恢复后当前段最近 20/50/100 个有效进度样本均值约 `44.76/31.89/26.79s/optimizer step`。最近 20/50 步被 step 7500 validation 窗口拉长，100 步窗口更接近当前训练速度。
- 验证/checkpoint：最新验证点为 step 7500，`val_loss = 0.5077`。`outputs/ablation_ums_random_lm_12label/checkpoints/step_7000.pt` 已于 13:21:05 落盘；`step_8000.pt` 和 `final.pt` 尚未生成。
- 结果判断：validation loss 轨迹为 `0.6277 -> 0.5917 -> 0.5609 -> 0.5364 -> 0.5167 -> 0.5077`，继续下降。17:00 当前训练 loss 约 `0.6457`，learning rate 约 `1.16e-05`；当前恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：16:20、16:25、16:30、16:35、16:40、16:45、16:50、16:55、17:00 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK。
- Case study：本小时无新增失败案例；10:00 静默退出已记录为 `History/20260612_1000_random_lm_silent_exit_best_relaunch/case_study.md`。此前 case study 仍为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`、`History/20260611_1750_random_lm_silent_exit_best_relaunch/case_study.md`、`History/20260612_0125_random_lm_silent_exit_step6000_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `31.89s/optimizer step` 估计，约 `4.3` 小时到 step 8000 validation/checkpoint，random-LM pretrain 剩余约 `22.1` 小时；按最近 100 步平均 `26.79s/step` 估计，pretrain 剩余约 `18.5` 小时。考虑 step 7500 验证刚拉长短窗、且历史有静默退出风险，完整目标当前保守估计约 `24-44` 小时。后续重点是继续向 step 8000/9000/10000 推进，并等待 `final.pt` 落盘后自动进入 random-LM LP。

2026-06-12 18:00 小时同步（random-LM step 7655，稳定接近 step 8000）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：18:00 现场快照显示当前 VIVID 主训练 PID `4840` 在 GPU1（bus `00000000:05:00.0`）运行，GPU1 `18265MiB / 24576MiB`，util `85%`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；仍只使用 GPU1。18:00 resource guard 记录 GPU0 `0MiB`、GPU1 `18265MiB`、util `54%`、温度 `63C`，resource guard OK，无 stop action。
- 训练进度：10:10 从 `best.pt` 恢复到 step 6500 后，18:00 已推进到恢复段 local `1155 / 3500`，折算全局约 `7655 / 10000`。距离 step 8000 validation/checkpoint 约 `345` step；random-LM pretrain 到 `final.pt` 还剩约 `2345` optimizer step。
- 一步多久：18:00 最新一步约 `29.42s/optimizer step`；恢复后当前段最近 20/50/100 个有效进度样本均值约 `24.97/23.99/23.92s/optimizer step`。验证窗口影响已消退，速度回到约 24s/step。
- 验证/checkpoint：最新验证点为 step 7500，`val_loss = 0.5077`。`outputs/ablation_ums_random_lm_12label/checkpoints/step_7000.pt` 已于 13:21:05 落盘；`step_8000.pt` 和 `final.pt` 尚未生成。
- 结果判断：validation loss 轨迹为 `0.7444 -> 0.6806 -> 0.6277 -> 0.5917 -> 0.5609 -> 0.5364 -> 0.5167 -> 0.5077`，继续下降。18:00 当前训练 loss 约 `0.5730`，learning rate 约 `1.14e-05`；当前恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：17:20、17:25、17:30、17:35、17:40、17:45、17:50、17:55、18:00 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK。
- Case study：本小时无新增失败案例；10:00 静默退出已记录为 `History/20260612_1000_random_lm_silent_exit_best_relaunch/case_study.md`。此前 case study 仍为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`、`History/20260611_1750_random_lm_silent_exit_best_relaunch/case_study.md`、`History/20260612_0125_random_lm_silent_exit_step6000_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `23.99s/optimizer step` 估计，约 `2.3` 小时到 step 8000 validation/checkpoint，random-LM pretrain 剩余约 `15.6` 小时；按最近 100 步平均 `23.92s/step` 估计，pretrain 剩余约 `15.6` 小时。考虑历史静默退出风险，完整目标当前保守估计约 `22-40` 小时。后续重点是通过 step 8000 并等待 `final.pt` 落盘后自动进入 random-LM LP。

2026-06-12 19:00 小时同步（random-LM step 7730，仍在 GPU1 正常推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：18:59 现场快照显示当前 VIVID 主训练 PID `4840` 在 GPU1（bus `00000000:05:00.0`）运行，GPU1 `18265MiB / 24576MiB`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；仍只使用 GPU1。18:55 resource guard 记录 GPU0 `0MiB`、GPU1 `18265MiB`、util `99%`、温度 `58C`，resource guard OK，无 stop action。
- 训练进度：10:10 从 `best.pt` 恢复到 step 6500 后，19:00 已推进到恢复段 local `1230 / 3500`，折算全局约 `7730 / 10000`。距离 step 8000 validation/checkpoint 约 `270` step；距离 step 9000 约 `1270` step；random-LM pretrain 到 `final.pt` 还剩约 `2270` optimizer step。
- 一步多久：19:00 最新一步约 `41.91s/optimizer step`；恢复后当前段最近 20/50/100 个有效进度样本均值约 `49.14/46.63/42.56s/optimizer step`。这一小时短窗速度变慢，但日志仍在 1 分钟内刷新，GPU1 进程仍在，当前判断为慢速推进而非失败。
- 验证/checkpoint：最新验证点为 step 7500，`val_loss = 0.5077`。`outputs/ablation_ums_random_lm_12label/checkpoints/step_7000.pt` 已于 13:21:05 落盘；`step_8000.pt`、`step_9000.pt` 和 `final.pt` 尚未生成。
- 结果判断：validation loss 轨迹为 `0.7444 -> 0.6806 -> 0.6277 -> 0.5917 -> 0.5609 -> 0.5364 -> 0.5167 -> 0.5077`，继续下降。19:00 当前训练 loss 约 `0.6042`，learning rate 约 `1.13e-05`；当前恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：18:20、18:25、18:30、18:35、18:40、18:45、18:50、18:55 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK。
- Case study：本小时无新增失败案例；10:00 静默退出已记录为 `History/20260612_1000_random_lm_silent_exit_best_relaunch/case_study.md`。此前 case study 仍为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`、`History/20260611_1750_random_lm_silent_exit_best_relaunch/case_study.md`、`History/20260612_0125_random_lm_silent_exit_step6000_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `46.63s/optimizer step` 估计，约 `3.5` 小时到 step 8000 validation/checkpoint，约 `16.4` 小时到 step 9000，random-LM pretrain 剩余约 `29.4` 小时；按最近 100 步平均 `42.56s/step` 估计，pretrain 剩余约 `26.8` 小时。考虑历史静默退出风险，完整目标当前保守估计约 `32-48` 小时。后续重点是通过 step 8000 并等待 `final.pt` 落盘后自动进入 random-LM LP。

2026-06-12 20:00 小时同步（random-LM step 7805，继续向 step 8000 推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：20:01 现场快照显示当前 VIVID 主训练 PID `4840` 在 GPU1（bus `00000000:05:00.0`）运行，GPU1 `18265MiB / 24576MiB`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；仍只使用 GPU1。20:00 resource guard 记录 GPU0 `0MiB`、GPU1 `18265MiB`、util `68%`、温度 `62C`，resource guard OK，无 stop action。
- 训练进度：10:10 从 `best.pt` 恢复到 step 6500 后，20:01 已推进到恢复段 local `1305 / 3500`，折算全局约 `7805 / 10000`。距离 step 8000 validation/checkpoint 约 `195` step；距离 step 9000 约 `1195` step；random-LM pretrain 到 `final.pt` 还剩约 `2195` optimizer step。
- 一步多久：20:01 最新一步约 `43.88s/optimizer step`；恢复后当前段最近 20/50/100 个有效进度样本均值约 `47.49/49.16/49.15s/optimizer step`。这一段速度仍慢于 18:00 前后，但日志在 1 分钟内刷新，GPU1 进程仍在，当前判断为慢速推进而非失败。
- 验证/checkpoint：最新验证点为 step 7500，`val_loss = 0.5077`。`outputs/ablation_ums_random_lm_12label/checkpoints/step_7000.pt` 已于 13:21:05 落盘；`step_8000.pt`、`step_9000.pt` 和 `final.pt` 尚未生成。
- 结果判断：validation loss 轨迹为 `0.7444 -> 0.6806 -> 0.6277 -> 0.5917 -> 0.5609 -> 0.5364 -> 0.5167 -> 0.5077`，继续下降。20:01 当前训练 loss 约 `0.6709`，learning rate 约 `1.13e-05`；当前恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：19:20、19:25、19:30、19:35、19:40、19:45、19:50、19:55、20:00 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK。
- Case study：本小时无新增失败案例；10:00 静默退出已记录为 `History/20260612_1000_random_lm_silent_exit_best_relaunch/case_study.md`。此前 case study 仍为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`、`History/20260611_1750_random_lm_silent_exit_best_relaunch/case_study.md`、`History/20260612_0125_random_lm_silent_exit_step6000_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `49.16s/optimizer step` 估计，约 `2.7` 小时到 step 8000 validation/checkpoint，约 `16.3` 小时到 step 9000，random-LM pretrain 剩余约 `30.0` 小时；按最近 100 步平均 `49.15s/step` 估计，pretrain 剩余约 `30.0` 小时。考虑历史静默退出风险，完整目标当前保守估计约 `34-50` 小时。后续重点是通过 step 8000 并等待 `final.pt` 落盘后自动进入 random-LM LP。

2026-06-12 21:00 小时同步（random-LM step 7872，继续接近 step 8000）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：21:00 现场快照显示当前 VIVID 主训练 PID `4840` 在 GPU1（bus `00000000:05:00.0`）运行，GPU1 `18265MiB / 24576MiB`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；仍只使用 GPU1。21:00 resource guard 记录 GPU0 `0MiB`、GPU1 `18265MiB`、util `43%`、温度 `58C`，resource guard OK，无 stop action。
- 训练进度：10:10 从 `best.pt` 恢复到 step 6500 后，21:00 已推进到恢复段 local `1372 / 3500`，折算全局约 `7872 / 10000`。距离 step 8000 validation/checkpoint 约 `128` step；距离 step 9000 约 `1128` step；random-LM pretrain 到 `final.pt` 还剩约 `2128` optimizer step。
- 一步多久：21:00 最新一步约 `49.61s/optimizer step`；恢复后当前段最近 20/50/100 个有效进度样本均值约 `52.83/52.66/51.56s/optimizer step`。本小时速度稳定在约 52-53s/step，日志仍持续刷新，当前判断为慢速但健康推进。
- 验证/checkpoint：最新验证点仍为 step 7500，`val_loss = 0.5077`。`outputs/ablation_ums_random_lm_12label/checkpoints/step_7000.pt` 已于 13:21:05 落盘；`step_8000.pt`、`step_9000.pt` 和 `final.pt` 尚未生成。
- 结果判断：validation loss 轨迹为 `0.7444 -> 0.6806 -> 0.6277 -> 0.5917 -> 0.5609 -> 0.5364 -> 0.5167 -> 0.5077`，继续下降。21:00 当前训练 loss 约 `0.6269`，learning rate 约 `1.12e-05`；当前恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：20:30、20:35、20:40、20:45、20:50、20:55、21:00 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK。
- Case study：本小时无新增失败案例；10:00 静默退出已记录为 `History/20260612_1000_random_lm_silent_exit_best_relaunch/case_study.md`。此前 case study 仍为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`、`History/20260611_1750_random_lm_silent_exit_best_relaunch/case_study.md`、`History/20260612_0125_random_lm_silent_exit_step6000_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `52.66s/optimizer step` 估计，约 `1.9` 小时到 step 8000 validation/checkpoint，约 `16.5` 小时到 step 9000，random-LM pretrain 剩余约 `31.1` 小时；按最近 100 步平均 `51.56s/step` 估计，pretrain 剩余约 `30.5` 小时。考虑历史静默退出风险，完整目标当前保守估计约 `34-50` 小时。后续重点是通过 step 8000 并等待 `final.pt` 落盘后自动进入 random-LM LP。

2026-06-12 22:00 小时同步（random-LM step 7942，稳定逼近 step 8000）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：21:59 现场快照显示当前 VIVID 主训练 PID `4840` 在 GPU1（bus `00000000:05:00.0`）运行，GPU1 `18265MiB / 24576MiB`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；仍只使用 GPU1。21:55 resource guard 记录 GPU0 `0MiB`、GPU1 `18265MiB`、util `69%`、温度 `62C`，resource guard OK，无 stop action。
- 训练进度：10:10 从 `best.pt` 恢复到 step 6500 后，22:00 已推进到恢复段 local `1442 / 3500`，折算全局约 `7942 / 10000`。距离 step 8000 validation/checkpoint 约 `58` step；距离 step 9000 约 `1058` step；random-LM pretrain 到 `final.pt` 还剩约 `2058` optimizer step。
- 一步多久：22:00 最新一步约 `49.46s/optimizer step`；恢复后当前段最近 20/50/100 个有效进度样本均值约 `49.91/51.13/51.52s/optimizer step`。本小时速度仍在约 50-52s/step，日志持续刷新，当前判断为慢速但健康推进。
- 验证/checkpoint：最新验证点仍为 step 7500，`val_loss = 0.5077`。`outputs/ablation_ums_random_lm_12label/checkpoints/step_7000.pt` 已于 13:21:05 落盘；`step_8000.pt`、`step_9000.pt` 和 `final.pt` 尚未生成。
- 结果判断：validation loss 轨迹为 `0.7444 -> 0.6806 -> 0.6277 -> 0.5917 -> 0.5609 -> 0.5364 -> 0.5167 -> 0.5077`，继续下降。22:00 当前训练 loss 约 `0.5785`，learning rate 约 `1.11e-05`；当前恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：21:30、21:35、21:40、21:45、21:50、21:55 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK。
- Case study：本小时无新增失败案例；10:00 静默退出已记录为 `History/20260612_1000_random_lm_silent_exit_best_relaunch/case_study.md`。此前 case study 仍为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`、`History/20260611_1750_random_lm_silent_exit_best_relaunch/case_study.md`、`History/20260612_0125_random_lm_silent_exit_step6000_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `51.13s/optimizer step` 估计，约 `0.82` 小时到 step 8000 validation/checkpoint，约 `15.0` 小时到 step 9000，random-LM pretrain 剩余约 `29.2` 小时；按最近 100 步平均 `51.52s/step` 估计，pretrain 剩余约 `29.4` 小时。考虑历史静默退出风险，完整目标当前保守估计约 `33-49` 小时。后续重点是通过 step 8000 并等待 `final.pt` 落盘后自动进入 random-LM LP。

2026-06-12 23:00 小时同步（random-LM step 8000 validation/checkpoint 已完成）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，任务仍为 `VIVID_random_lm_gpu1`，active 日志仍为 `outputs/logs/ablation_ums_random_lm_12label_train.log`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：22:59 现场快照显示当前 VIVID 主训练 PID `4840` 在 GPU1（bus `00000000:05:00.0`）运行，GPU1 `18265MiB / 24576MiB`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；仍只使用 GPU1。22:55 resource guard 记录 GPU0 `0MiB`、GPU1 `18265MiB`、util `24%`、温度 `56C`，resource guard OK，无 stop action。
- 训练进度：10:10 从 `best.pt` 恢复到 step 6500 后，23:01 已推进到恢复段 local `1506 / 3500`，折算全局约 `8006 / 10000`。step 8000 validation 已完成；距离 step 9000 约 `994` step；random-LM pretrain 到 `final.pt` 还剩约 `1994` optimizer step。
- 一步多久：23:01 最新一步约 `61.69s/optimizer step`；恢复后当前段最近 20/50/100 个有效进度样本均值约 `67.51/58.37/54.15s/optimizer step`。该窗口包含 step 8000 validation 和 checkpoint 保存开销，所以短窗速度被拉慢；训练已继续到 step 8006。
- 验证/checkpoint：`Step 8000: val_loss = 0.4969`，`outputs/ablation_ums_random_lm_12label/checkpoints/step_8000.pt` 已于 22:58:26 落盘，大小约 `1022.72MB`；`best.pt` 也于 22:58:12 刷新。`step_9000.pt` 和 `final.pt` 尚未生成。
- 结果判断：validation loss 轨迹为 `0.6806 -> 0.6277 -> 0.5917 -> 0.5609 -> 0.5364 -> 0.5167 -> 0.5077 -> 0.4969`，继续下降。23:01 当前训练 loss 约 `0.6376`，learning rate 约 `1.11e-05`；当前恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：22:20、22:25、22:30、22:35、22:40、22:45、22:50、22:55 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK。
- Case study：本小时无新增失败案例；10:00 静默退出已记录为 `History/20260612_1000_random_lm_silent_exit_best_relaunch/case_study.md`。此前 case study 仍为 `History/20260611_0055_random_lm_gpu_lost_recovery/case_study.md`、`History/20260611_1020_random_lm_gpu_lost_reboot_recovery/case_study.md`、`History/20260611_1215_random_lm_silent_exit_queue_relaunch/case_study.md`、`History/20260611_1750_random_lm_silent_exit_best_relaunch/case_study.md`、`History/20260612_0125_random_lm_silent_exit_step6000_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `58.37s/optimizer step` 估计，约 `16.1` 小时到 step 9000，random-LM pretrain 剩余约 `32.3` 小时；按最近 100 步平均 `54.15s/step` 估计，pretrain 剩余约 `30.0` 小时。考虑 step 8000 validation 刚拉长短窗且历史有静默退出风险，完整目标当前保守估计约 `34-52` 小时。后续重点是继续向 step 9000/10000 推进，并等待 `final.pt` 落盘后自动进入 random-LM LP。

2026-06-13 00:35 小时同步（random-LM 在 step 8050 附近被中断后已从 step 8000 恢复）：
- 当前阶段：`random-LM same-architecture` 预训练仍是当前主任务。23:38 前一段训练日志在恢复段 local `1550 / 3500` 停止，折算全局约 `8050 / 10000`，日志尾部出现 `^CTerminate batch job (Y/N)?`，未见 Python traceback、CUDA OOM 或训练代码异常。
- GPU/队列事件：23:45 到 00:30 队列持续检测到 `random_lm_running=False`，但因 GPU0 查询返回 `GPU is lost`，resource policy 阻止自动重启。00:33 现场 `nvidia-smi` 恢复正常，手动触发 `scripts/answerability_gpu1_queue_once.ps1` 后，队列成功启动计划任务 `VIVID_random_lm_gpu1`。
- GPU 约束：恢复任务日志显示 `CUDA_VISIBLE_DEVICES=1`，训练内 `Using device: cuda:0` 对应物理 GPU1。00:37 现场 GPU0 `0MiB / 24576MiB`，GPU1 由 PID `6964` 使用；00:36 后恢复段显存升至约 `15221MiB / 24576MiB`，仍只使用 GPU1。
- 恢复/checkpoint：恢复任务从 `outputs/ablation_ums_random_lm_12label/checkpoints/step_8000.pt` 加载，日志显示 `Checkpoint loaded` 和 `Resuming from step 8000`。`step_9000.pt` 与 `final.pt` 尚未生成。
- 当前训练进度：恢复后新段为 local `10 / 2000`，折算全局约 `8010 / 10000`。本次中断损失约 `50` 个未落盘 optimizer step（约 step 8000 到 8050），没有证据显示 checkpoint 损坏。
- 一步多久：中断前最近 20/50/100 步约 `54.58/51.29/54.65s/optimizer step`；恢复后早期 10 步约 `6-10s/step`，但这是刚启动短窗，后续 ETA 仍需用更稳定窗口重估。
- 结果判断：最新有效验证仍为 step 8000，`val_loss = 0.4969`，继续优于 step 7500 的 `0.5077`。本次属于运行/设备状态中断，不是模型指标退化；当前结果趋势仍符合预期。
- Case study：已新增 `History/20260613_0035_random_lm_ctrlc_gpu0_lost_relaunch/case_study.md`，记录 `Ctrl-C/批处理终止 + GPU0 lost 阻塞重启 + 从 step_8000 恢复`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/step_9000.pt`、`outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按中断前稳定窗口估算，random-LM pretrain 从 step 8000 到 final 约 `28-30` 小时；恢复后早期速度明显更快，但短窗过短，暂不把 `6-10s/step` 当作长期速度。下一小时重点是确认恢复段持续推进，并重估到 step 9000 与 final 的 ETA。

2026-06-13 01:00 小时同步（random-LM 恢复后健康推进到 step 8140）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，仍为 `VIVID_random_lm_gpu1`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：01:00 现场快照显示当前训练 PID `6964` 在 GPU1（bus `00000000:05:00.0`）运行，GPU1 `16739MiB / 24576MiB`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；仍只使用 GPU1。01:00 resource guard 记录 GPU0 `0MiB`、GPU1 `16739MiB`、util `65%`、温度 `62C`、总功耗 `123.63W`，resource guard OK，无 stop action。
- 训练进度：00:33 从 `step_8000.pt` 恢复后，01:01 已推进到恢复段 local `140 / 2000`，折算全局约 `8140 / 10000`。距离 `step_9000.pt` 约 `860` step；random-LM pretrain 到 `final.pt` 还剩约 `1860` optimizer step。
- 一步多久：01:01 最新一步约 `20.59s/optimizer step`；恢复段最近 20/50/100 个有效进度样本均值约 `21.27/19.47/13.31s/optimizer step`。恢复早期的 6-10s/step 窗口已回落，当前更可信速度约 20s/step。
- 验证/checkpoint：最新验证点仍为 `Step 8000: val_loss = 0.4969`，`outputs/ablation_ums_random_lm_12label/checkpoints/step_8000.pt` 已落盘；`step_9000.pt` 和 `final.pt` 尚未生成。
- 结果判断：validation loss 轨迹为 `0.5917 -> 0.5609 -> 0.5364 -> 0.5167 -> 0.5077 -> 0.4969`，继续下降。01:01 当前训练 loss 约 `0.5910`，learning rate 约 `1.09e-05`；恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：00:35、00:40、00:45、00:50、00:55、01:00 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK。
- Case study：本小时无新增失败案例；00:35 的 `Ctrl-C/批处理终止 + GPU0 lost 阻塞重启` 已记录为 `History/20260613_0035_random_lm_ctrlc_gpu0_lost_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/step_9000.pt`、`outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `19.47s/optimizer step` 估计，约 `4.7` 小时到 `step_9000.pt`，random-LM pretrain 剩余约 `10.1` 小时；按最近 100 步平均 `13.31s/step` 估计，pretrain 剩余约 `6.9` 小时。考虑恢复段速度仍在收敛，完整目标当前保守估计约 `12-24` 小时，待 `final.pt` 落盘后会重新估算 random-LM LP 与两个 eval。

2026-06-13 02:00 小时同步（random-LM 恢复后推进到 step 8275）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，仍为当前主任务 `VIVID_random_lm_gpu1`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：02:00 现场快照显示当前训练 PID `6964` 在 GPU1（bus `00000000:05:00.0`）运行，GPU1 `16739MiB / 24576MiB`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；仍只使用 GPU1。02:00 单次 util 采样为 `0%`，但训练日志 6 秒内刷新且 GPU1 显存/进程持续存在，判断为采样窗口落在短暂空档而非停止。
- 训练进度：00:33 从 `step_8000.pt` 恢复后，02:00 已推进到恢复段 local `275 / 2000`，折算全局约 `8275 / 10000`。距离 `step_9000.pt` 约 `725` step；random-LM pretrain 到 `final.pt` 还剩约 `1725` optimizer step。
- 一步多久：02:00 最新一步约 `26.91s/optimizer step`；恢复段最近 20/50/100 个有效进度样本均值约 `30.40/28.57/27.35s/optimizer step`。相比 01:00 的短窗，速度已稳定到约 `27-30s/step`。
- 验证/checkpoint：最新验证点仍为 `Step 8000: val_loss = 0.4969`，`outputs/ablation_ums_random_lm_12label/checkpoints/step_8000.pt` 已落盘；`step_9000.pt` 和 `final.pt` 尚未生成。
- 结果判断：validation loss 轨迹为 `0.5917 -> 0.5609 -> 0.5364 -> 0.5167 -> 0.5077 -> 0.4969`，继续下降。02:00 当前训练 loss 约 `0.6276`，learning rate 约 `1.08e-05`；恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：01:15 至 01:55 队列持续识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK，无 stop action。
- Case study：本小时无新增失败案例；最新失败案例仍为 `History/20260613_0035_random_lm_ctrlc_gpu0_lost_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/step_9000.pt`、`outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `28.57s/optimizer step` 估计，约 `5.8` 小时到 `step_9000.pt`，random-LM pretrain 剩余约 `13.7` 小时；按最近 100 步平均 `27.35s/step` 估计，pretrain 剩余约 `13.1` 小时。完整目标在 pretrain 之后还剩 random-LM LP、两个 eval 和最终表格/claim cleanup，当前保守估计约 `16-28` 小时，待 `final.pt` 落盘后重新按实际 LP/eval 速度收窄。

2026-06-13 03:00 小时同步（random-LM 恢复后推进到 step 8398）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，仍为当前主任务 `VIVID_random_lm_gpu1`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：03:01 现场快照显示当前训练 PID `6964` 在 GPU1（bus `00000000:05:00.0`）运行，GPU1 `18263MiB / 24576MiB`，util `56%`，温度 `60C`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；仍只使用 GPU1。03:00 resource guard 记录 GPU0 `0MiB`、GPU1 `18263MiB`、util `68%`、温度 `61C`，resource guard OK，无 stop action。
- 训练进度：00:33 从 `step_8000.pt` 恢复后，03:01 已推进到恢复段 local `398 / 2000`，折算全局约 `8398 / 10000`。距离 `step_9000.pt` 约 `602` step；random-LM pretrain 到 `final.pt` 还剩约 `1602` optimizer step。
- 一步多久：03:01 最新一步约 `27.64s/optimizer step`；恢复段最近 20/50/100 个有效进度样本均值约 `30.56/30.80/29.95s/optimizer step`。这一小时速度稳定在约 `30s/step`，日志 27 秒内刷新，判断为健康推进。
- 验证/checkpoint：最新验证点仍为 `Step 8000: val_loss = 0.4969`，`outputs/ablation_ums_random_lm_12label/checkpoints/step_8000.pt` 已落盘；`step_9000.pt` 和 `final.pt` 尚未生成。
- 结果判断：validation loss 轨迹为 `0.5917 -> 0.5609 -> 0.5364 -> 0.5167 -> 0.5077 -> 0.4969`，继续下降。03:01 当前训练 loss 约 `0.6364`，learning rate 约 `1.07e-05`；恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：02:20 至 03:00 队列持续识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK，无 stop action。
- Case study：本小时无新增失败案例；最新失败案例仍为 `History/20260613_0035_random_lm_ctrlc_gpu0_lost_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/step_9000.pt`、`outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `30.80s/optimizer step` 估计，约 `5.2` 小时到 `step_9000.pt`，random-LM pretrain 剩余约 `13.7` 小时；按最近 100 步平均 `29.95s/step` 估计，pretrain 剩余约 `13.3` 小时。完整目标在 pretrain 之后还剩 random-LM LP、两个 eval 和最终表格/claim cleanup，当前保守估计约 `16-28` 小时，待 `final.pt` 落盘后按实际 LP/eval 速度重新收窄。

2026-06-13 04:00 小时同步（random-LM 恢复后推进到 step 8497）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，仍为当前主任务 `VIVID_random_lm_gpu1`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：04:00 现场快照显示当前训练 PID `6964` 在 GPU1（bus `00000000:05:00.0`）运行，GPU1 `18263MiB / 24576MiB`，04:00 resource guard 记录 GPU1 util `47%`、温度 `63C`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；仍只使用 GPU1。resource guard OK，无 stop action。
- 训练进度：00:33 从 `step_8000.pt` 恢复后，04:00 已推进到恢复段 local `497 / 2000`，折算全局约 `8497 / 10000`。距离 `step_9000.pt` 约 `503` step；random-LM pretrain 到 `final.pt` 还剩约 `1503` optimizer step。
- 一步多久：04:00 最新一步约 `40.49s/optimizer step`；恢复段最近 20/50/100 个有效进度样本均值约 `40.03/38.12/35.83s/optimizer step`。相较 03:00 的约 `30s/step` 有所变慢，但日志 14 秒内刷新、GPU1 进程/显存稳定，判断为慢速但健康推进。
- 验证/checkpoint：最新验证点仍为 `Step 8000: val_loss = 0.4969`，`outputs/ablation_ums_random_lm_12label/checkpoints/step_8000.pt` 已落盘；`step_9000.pt` 和 `final.pt` 尚未生成。
- 结果判断：validation loss 轨迹为 `0.5917 -> 0.5609 -> 0.5364 -> 0.5167 -> 0.5077 -> 0.4969`，继续下降。04:00 当前训练 loss 约 `0.6368`，learning rate 约 `1.06e-05`；恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：03:40 至 04:00 队列持续识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK，无 stop action。
- Case study：本小时无新增失败案例；最新失败案例仍为 `History/20260613_0035_random_lm_ctrlc_gpu0_lost_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/step_9000.pt`、`outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `38.12s/optimizer step` 估计，约 `5.3` 小时到 `step_9000.pt`，random-LM pretrain 剩余约 `15.9` 小时；按最近 100 步平均 `35.83s/step` 估计，pretrain 剩余约 `15.0` 小时。完整目标在 pretrain 之后还剩 random-LM LP、两个 eval 和最终表格/claim cleanup，当前保守估计约 `18-30` 小时，待 `final.pt` 落盘后按实际 LP/eval 速度重新收窄。

2026-06-13 04:16 里程碑补记（random-LM step 8500 validation 完成）：
- 当前阶段：`random-LM same-architecture` 预训练仍在 GPU1 上由 PID `6964` 运行；训练从 00:33 的 `step_8000.pt` 恢复后已完成全局 step 8500 validation，并继续进入后续训练。
- 验证结果：`Step 8500: val_loss = 0.4804`，优于 step 8000 的 `0.4969`；validation loss 仍保持下降，结果符合预期。该点刷新了 `outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`，但当前未生成独立 `step_8500.pt` 文件，后续硬目标仍以 `step_9000.pt` 与 `final.pt` 为准。
- 当前进度：04:16 解析日志为恢复段 local `505 / 2000`，折算全局约 `8505 / 10000`。距离 `step_9000.pt` 约 `495` step；random-LM pretrain 到 `final.pt` 还剩约 `1495` optimizer step。
- 一步多久：04:16 最新短窗受刚结束的 validation/checkpoint 影响，最近 20/50/100 个有效进度样本均值约 `67.03/49.85/42.47s/optimizer step`。按最近 100 步估计，pretrain 剩余约 `17.6` 小时；按最近 50 步估计约 `20.7` 小时，下一小时会在 validation 开销被窗口淡出后重估。
- GPU/队列：04:16 现场 GPU0 `0MiB / 24576MiB`；GPU1 `18265MiB / 24576MiB`，bus `00000000:05:00.0`，仍只用 GPU1。04:10 队列识别 `random_lm_running=True`，未重复启动；resource guard OK。
- Case study：本次是正常验证与 best checkpoint 刷新，不是失败案例；未新增 case study。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/step_9000.pt`、`outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。

2026-06-13 05:00 小时同步（random-LM 恢复后推进到 step 8578）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，仍为当前主任务 `VIVID_random_lm_gpu1`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：05:00 现场快照显示当前训练 PID `6964` 在 GPU1（bus `00000000:05:00.0`）运行，GPU1 `18265MiB / 24576MiB`，现场 util `77%`；05:00 resource guard 记录 GPU1 util `61%`、温度 `63C`、power `123.2W`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；仍只使用 GPU1。resource guard OK，无 stop action。
- 训练进度：00:33 从 `step_8000.pt` 恢复后，05:00 已推进到恢复段 local `578 / 2000`，折算全局约 `8578 / 10000`。距离 `step_9000.pt` 约 `422` step；random-LM pretrain 到 `final.pt` 还剩约 `1422` optimizer step。
- 一步多久：05:00 最新一步约 `40.43s/optimizer step`；恢复段最近 20/50/100 个有效进度样本均值约 `37.55/36.83/43.35s/optimizer step`。当前速度比 04:16 validation 后短窗稳定，日志 29 秒内刷新、GPU1 进程/显存稳定，判断为健康推进。
- 验证/checkpoint：最新验证点为 `Step 8500: val_loss = 0.4804`，优于 step 8000 的 `0.4969`；`outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 已于 04:14 刷新。`step_9000.pt` 和 `final.pt` 尚未生成。
- 结果判断：validation loss 轨迹为 `0.5917 -> 0.5609 -> 0.5364 -> 0.5167 -> 0.5077 -> 0.4969 -> 0.4804`，继续下降。05:00 当前训练 loss 约 `0.5552`，learning rate 约 `1.05e-05`；恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：04:50、04:55、05:00 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK，无 stop action。
- Case study：本小时无新增失败案例；最新失败案例仍为 `History/20260613_0035_random_lm_ctrlc_gpu0_lost_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/step_9000.pt`、`outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `36.83s/optimizer step` 估计，约 `4.3` 小时到 `step_9000.pt`，random-LM pretrain 剩余约 `14.5` 小时；按最近 100 步平均 `43.35s/step` 估计，pretrain 剩余约 `17.1` 小时。完整目标在 pretrain 之后还剩 random-LM LP、两个 eval 和最终表格/claim cleanup，当前保守估计约 `18-28` 小时，待 `final.pt` 落盘后按实际 LP/eval 速度重新收窄。

2026-06-13 06:00 小时同步（random-LM 恢复后推进到 step 8673）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，仍为当前主任务 `VIVID_random_lm_gpu1`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：06:00 现场快照显示当前训练 PID `6964` 在 GPU1（bus `00000000:05:00.0`）运行，GPU1 `18265MiB / 24576MiB`，现场 util `11%`；06:00 resource guard 记录 GPU1 util `75%`、温度 `59C`、power `113.72W`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；仍只使用 GPU1。resource guard OK，无 stop action。
- 训练进度：00:33 从 `step_8000.pt` 恢复后，06:00 已推进到恢复段 local `673 / 2000`，折算全局约 `8673 / 10000`。距离 `step_9000.pt` 约 `327` step；random-LM pretrain 到 `final.pt` 还剩约 `1327` optimizer step。
- 一步多久：06:00 最新一步约 `45.30s/optimizer step`；恢复段最近 20/50/100 个有效进度样本均值约 `40.76/38.85/37.98s/optimizer step`。日志 5 秒内刷新、GPU1 进程/显存稳定，判断为健康推进。
- 验证/checkpoint：最新验证点仍为 `Step 8500: val_loss = 0.4804`，优于 step 8000 的 `0.4969`；`outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 已刷新。`step_9000.pt` 和 `final.pt` 尚未生成。
- 结果判断：validation loss 轨迹为 `0.5917 -> 0.5609 -> 0.5364 -> 0.5167 -> 0.5077 -> 0.4969 -> 0.4804`，继续下降。06:00 当前训练 loss 约 `0.5897`，learning rate 约 `1.05e-05`；恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：05:20 至 06:00 队列持续识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK，无 stop action。
- Case study：本小时无新增失败案例；最新失败案例仍为 `History/20260613_0035_random_lm_ctrlc_gpu0_lost_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/step_9000.pt`、`outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `38.85s/optimizer step` 估计，约 `3.5` 小时到 `step_9000.pt`，random-LM pretrain 剩余约 `14.3` 小时；按最近 100 步平均 `37.98s/step` 估计，pretrain 剩余约 `14.0` 小时。完整目标在 pretrain 之后还剩 random-LM LP、两个 eval 和最终表格/claim cleanup，当前保守估计约 `17-25` 小时，待 `final.pt` 落盘后按实际 LP/eval 速度重新收窄。

2026-06-13 07:00 小时同步（random-LM 恢复后推进到 step 8767）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，仍为当前主任务 `VIVID_random_lm_gpu1`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：07:00 现场快照显示当前训练 PID `6964` 在 GPU1（bus `00000000:05:00.0`）运行，GPU1 `18265MiB / 24576MiB`，现场 util `55%`、温度 `56C`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；仍只使用 GPU1。
- 训练进度：00:33 从 `step_8000.pt` 恢复后，07:00 已推进到恢复段 local `767 / 2000`，折算全局约 `8767 / 10000`。距离 `step_9000.pt` 约 `233` step；random-LM pretrain 到 `final.pt` 还剩约 `1233` optimizer step。
- 一步多久：07:00 最新一步约 `34.76s/optimizer step`；恢复段最近 20/50/100 个有效进度样本均值约 `36.41/36.25/38.39s/optimizer step`。日志 11 秒内刷新、GPU1 进程/显存稳定，判断为健康推进。
- 验证/checkpoint：最新验证点仍为 `Step 8500: val_loss = 0.4804`，优于 step 8000 的 `0.4969`；`outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 已刷新。`step_9000.pt` 和 `final.pt` 尚未生成。
- 结果判断：validation loss 轨迹为 `0.5917 -> 0.5609 -> 0.5364 -> 0.5167 -> 0.5077 -> 0.4969 -> 0.4804`，继续下降。07:00 当前训练 loss 约 `0.5899`，learning rate 约 `1.04e-05`；恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：06:50、06:55、07:00 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard 06:55 OK，无 stop action。
- Case study：本小时无新增失败案例；最新失败案例仍为 `History/20260613_0035_random_lm_ctrlc_gpu0_lost_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/step_9000.pt`、`outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `36.25s/optimizer step` 估计，约 `2.35` 小时到 `step_9000.pt`，random-LM pretrain 剩余约 `12.4` 小时；按最近 100 步平均 `38.39s/step` 估计，pretrain 剩余约 `13.1` 小时。完整目标在 pretrain 之后还剩 random-LM LP、两个 eval 和最终表格/claim cleanup，当前保守估计约 `16-24` 小时，待 `final.pt` 落盘后按实际 LP/eval 速度重新收窄。

2026-06-13 08:00 小时同步（random-LM 恢复后推进到 step 8857）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，仍为当前主任务 `VIVID_random_lm_gpu1`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：08:00 现场快照显示当前训练 PID `6964` 在 GPU1（bus `00000000:05:00.0`）运行，GPU1 `18265MiB / 24576MiB`，现场 util `56%`、温度 `61C`。GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；仍只使用 GPU1。08:00 resource guard 记录 GPU1 util `37%`、温度 `59C`、power `92.44W`，resource guard OK，无 stop action。
- 训练进度：00:33 从 `step_8000.pt` 恢复后，08:01 已推进到恢复段 local `857 / 2000`，折算全局约 `8857 / 10000`。距离 `step_9000.pt` 约 `143` step；random-LM pretrain 到 `final.pt` 还剩约 `1143` optimizer step。
- 一步多久：08:01 最新一步约 `39.11s/optimizer step`；恢复段最近 20/50/100 个有效进度样本均值约 `41.24/39.51/40.03s/optimizer step`。日志约 30 秒内刷新、GPU1 进程/显存稳定，判断为健康推进。
- 验证/checkpoint：最新验证点仍为 `Step 8500: val_loss = 0.4804`，优于 step 8000 的 `0.4969`；`outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 已刷新。`step_9000.pt` 和 `final.pt` 尚未生成。
- 结果判断：validation loss 轨迹为 `0.5917 -> 0.5609 -> 0.5364 -> 0.5167 -> 0.5077 -> 0.4969 -> 0.4804`，继续下降。08:01 当前训练 loss 约 `0.5592`，learning rate 约 `1.04e-05`；恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：07:50、07:55、08:00 队列均识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK，无 stop action。
- Case study：本小时无新增失败案例；最新失败案例仍为 `History/20260613_0035_random_lm_ctrlc_gpu0_lost_relaunch/case_study.md`。
- 当前硬缺口仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/step_9000.pt`、`outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `39.51s/optimizer step` 估计，约 `1.57` 小时到 `step_9000.pt`，random-LM pretrain 剩余约 `12.5` 小时；按最近 100 步平均 `40.03s/step` 估计，pretrain 剩余约 `12.7` 小时。完整目标在 pretrain 之后还剩 random-LM LP、两个 eval 和最终表格/claim cleanup，当前保守估计约 `16-24` 小时，待 `final.pt` 落盘后按实际 LP/eval 速度重新收窄。

2026-06-13 09:56 里程碑补记（random-LM step 9000 checkpoint 已落盘）：
- 当前阶段：`random-LM same-architecture` 预训练仍在 GPU1 上由 PID `6964` 运行；训练从 00:33 的 `step_8000.pt` 恢复后已完成全局 step 9000 validation，并继续向 `final.pt` 推进。本次巡检仍为主线程巡检，未新增子智能体。
- GPU 约束：09:50 现场快照显示 GPU0 `0MiB / 24576MiB`，无 VIVID 训练；GPU1（bus `00000000:05:00.0`）由 PID `6964` 占用约 `18265MiB / 24576MiB`，util `58%`、温度 `62C`，仍只使用 GPU1。
- 验证/checkpoint：`Step 9000: val_loss = 0.4782`，优于 step 8500 的 `0.4804`；`outputs/ablation_ums_random_lm_12label/checkpoints/step_9000.pt` 已于 09:48:47 落盘，大小约 `1022.72MB`，同时刷新 `best.pt`。
- 当前进度：09:56 解析日志为恢复段 local `1014 / 2000`，折算全局约 `9014 / 10000`。random-LM pretrain 到 `final.pt` 还剩约 `986` optimizer step。
- 一步多久：09:56 最新一步约 `36.85s/optimizer step`；最近 20/50/100 个有效进度样本均值约 `65.59/48.71/45.04s/optimizer step`。短窗被 step 9000 validation 与 checkpoint 写盘拉高，实际训练步已恢复到约 `37s/step`。
- 结果判断：validation loss 轨迹为 `0.6806 -> 0.6277 -> 0.5917 -> 0.5609 -> 0.5364 -> 0.5167 -> 0.5077 -> 0.4969 -> 0.4804 -> 0.4782`，仍在下降；当前训练 loss 约 `0.5487`，learning rate 约 `1.03e-05`。本段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- Case study：本次是正常 validation/checkpoint 里程碑，不是失败案例；未新增 case study。最新失败案例仍为 `History/20260613_0035_random_lm_ctrlc_gpu0_lost_relaunch/case_study.md`。
- 当前硬缺口更新为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup；`step_9000.pt` 已从硬缺口中移除。
- ETA：按最近 100 步平均 `45.04s/optimizer step` 估计，random-LM pretrain 剩余约 `12.3` 小时；按最近 50 步平均 `48.71s/step` 估计约 `13.3` 小时。考虑 validation 开销会逐步从窗口淡出，当前保守估计 `final.pt` 约还需 `12-14` 小时；完整目标在 pretrain 后还剩 random-LM LP、两个 eval 和最终表格/claim cleanup，当前保守估计约 `16-24` 小时，待 `final.pt` 落盘后按实际 LP/eval 速度重新收窄。

2026-06-13 10:00 小时同步（random-LM 已过 step 9000，继续向 final.pt 推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，仍为当前主任务 `VIVID_random_lm_gpu1`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：10:00 现场快照显示当前训练 PID `6964` 在 GPU1（bus `00000000:05:00.0`）运行，GPU1 `18265MiB / 24576MiB`；GPU0 `0MiB / 24576MiB`，没有 VIVID 训练。10:00 resource guard 记录 GPU0 `0MiB`、GPU1 util `28%`、温度 `58C`，resource guard OK，无 stop action；仍只使用 GPU1。
- 训练进度：00:33 从 `step_8000.pt` 恢复后，10:00 已推进到恢复段 local `1021 / 2000`，折算全局约 `9021 / 10000`。random-LM pretrain 到 `final.pt` 还剩约 `979` optimizer step。
- 一步多久：10:00 最新一步约 `43.97s/optimizer step`；恢复段最近 20/50/100 个有效进度样本均值约 `45.65/48.73/45.02s/optimizer step`。短窗仍包含 step 9000 validation/checkpoint 写盘开销，日志 6 秒内刷新、GPU1 进程/显存稳定，判断为健康推进。
- 验证/checkpoint：`outputs/ablation_ums_random_lm_12label/checkpoints/step_9000.pt` 已于 09:48:47 落盘，大小约 `1022.72MB`；最新验证点为 `Step 9000: val_loss = 0.4782`，优于 step 8500 的 `0.4804`。`final.pt` 尚未生成。
- 结果判断：validation loss 轨迹为 `0.6806 -> 0.6277 -> 0.5917 -> 0.5609 -> 0.5364 -> 0.5167 -> 0.5077 -> 0.4969 -> 0.4804 -> 0.4782`，仍在下降。10:00 当前训练 loss 约 `0.5427`，learning rate 约 `1.03e-05`；恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：09:40 至 10:00 队列持续识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK，无 stop action。
- Case study：本小时无新增失败案例；最新失败案例仍为 `History/20260613_0035_random_lm_ctrlc_gpu0_lost_relaunch/case_study.md`。
- 当前硬缺口为 `outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup；`step_9000.pt` 已完成。
- ETA：按最近 100 步平均 `45.02s/optimizer step` 估计，random-LM pretrain 剩余约 `12.2` 小时；按最近 50 步平均 `48.73s/step` 估计约 `13.3` 小时。完整目标在 pretrain 后还剩 random-LM LP、两个 eval 和最终表格/claim cleanup，当前保守估计约 `16-24` 小时，待 `final.pt` 落盘后按实际 LP/eval 速度重新收窄。

2026-06-13 11:00 小时同步（random-LM 继续向 final.pt 推进）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，仍为当前主任务 `VIVID_random_lm_gpu1`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：11:02 现场快照显示当前训练 PID `6964` 在 GPU1（bus `00000000:05:00.0`）运行，GPU1 `18265MiB / 24576MiB`、现场 util `34%`、温度 `60C`；GPU0 `0MiB / 24576MiB`，没有 VIVID 训练。11:00 resource guard 记录 GPU0 `0MiB`、GPU1 util `32%`、温度 `61C`、power `153.7W`，resource guard OK，无 stop action；仍只使用 GPU1。
- 训练进度：00:33 从 `step_8000.pt` 恢复后，11:02 已推进到恢复段 local `1145 / 2000`，折算全局约 `9145 / 10000`。距离 `step_9500.pt` 约 `355` step；random-LM pretrain 到 `final.pt` 还剩约 `855` optimizer step。
- 一步多久：11:02 最新一步约 `19.89s/optimizer step`；恢复段最近 20/50/100 个有效进度样本均值约 `27.49/25.53/27.99s/optimizer step`。日志约 13 秒内刷新、GPU1 进程/显存稳定，判断为健康推进。
- 验证/checkpoint：`outputs/ablation_ums_random_lm_12label/checkpoints/step_9000.pt` 已完成；最新验证点仍为 `Step 9000: val_loss = 0.4782`，优于 step 8500 的 `0.4804`。`step_9500.pt` 与 `final.pt` 尚未生成。
- 结果判断：validation loss 轨迹为 `0.6806 -> 0.6277 -> 0.5917 -> 0.5609 -> 0.5364 -> 0.5167 -> 0.5077 -> 0.4969 -> 0.4804 -> 0.4782`，仍在下降。11:02 当前训练 loss 约 `0.5357`，learning rate 约 `1.02e-05`；恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`，目前结果符合预期。
- 队列/守卫：10:30 至 11:00 队列持续识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK，无 stop action。
- Case study：本小时无新增失败案例；最新失败案例仍为 `History/20260613_0035_random_lm_ctrlc_gpu0_lost_relaunch/case_study.md`。
- 当前硬缺口为 `outputs/ablation_ums_random_lm_12label/checkpoints/step_9500.pt`、`outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `25.53s/optimizer step` 估计，约 `2.5` 小时到 `step_9500.pt`，random-LM pretrain 剩余约 `6.1` 小时；按最近 100 步平均 `27.99s/step` 估计，pretrain 剩余约 `6.6` 小时。完整目标在 pretrain 后还剩 random-LM LP、两个 eval 和最终表格/claim cleanup，当前保守估计约 `13-20` 小时，待 `final.pt` 落盘后按实际 LP/eval 速度重新收窄。

2026-06-13 11:46 重启恢复记录（random-LM 从 step_9000 重新接续）：
- 事件：用户提示刚重启后，本轮巡检发现重启前约 global `9201 / 10000` 的非持久化进度没有落盘；最新可靠 checkpoint 仍为 `outputs/ablation_ums_random_lm_12label/checkpoints/step_9000.pt`。11:35 队列识别 `random_lm_running=False` 且 `random_lm_train_final=False`，自动重新启动 `VIVID_random_lm_gpu1`。
- 恢复证据：11:46 现场进程表显示 `conda run --no-capture-output -n vivid python scripts\train_cxr.py --config configs\ablation_ums_random_lm_12label.yaml --resume ...\step_9000.pt`，训练 Python PID `20716` 在 GPU1（bus `00000000:05:00.0`）运行；日志显示 `Checkpoint loaded from ...\step_9000.pt`、`Resuming from step 9000`，并已推进到恢复段 local `24 / 1000`。
- 影响：未观察到 checkpoint 损坏；影响主要是丢失 `step_9000` 到重启前约 `step_9201` 之间约 `201` 个未落盘 optimizer step。`step_9500.pt` 与 `final.pt` 仍未生成，后续 ETA 必须以重启后的 `step_9000` 恢复段重新估算。
- GPU/队列：GPU0 仅有非 VIVID 低显存占用；VIVID 训练仍绑定 GPU1。11:40、11:45 队列均识别 `random_lm_running=True`，未重复启动，也未提前启动 LP/eval；resource guard OK。
- 当前进度：11:46 日志尾部显示恢复段 local `24 / 1000`，折算全局约 `9024 / 10000`。距离 `step_9500.pt` 约 `476` step；random-LM pretrain 到 `final.pt` 还剩约 `976` step。
- 一步多久与 ETA：重启后早期最近步速约 `10-20s/optimizer step`，但窗口还短；保守按早期窗口估计，`step_9500.pt` 约 `2-4` 小时，pretrain final 约 `4-8` 小时。完整目标在 pretrain 后仍需 random-LM LP、counterfactual-prefix eval、field-paraphrase eval、最终汇总/claim cleanup，当前保守估计约 `12-20` 小时。
- 结果判断：`Step 9000: val_loss = 0.4782` 仍是最新可靠验证点，优于 step 8500 的 `0.4804`；恢复后日志未见新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`。当前恢复状态符合预期，但重启导致未落盘进度回退。
- Case study：已保存 `History/20260613_1146_random_lm_reboot_step9000_relaunch/case_study.md`。

2026-06-13 12:00 小时同步（random-LM 重启后恢复段推进到 step 9043）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，仍是当前主任务 `VIVID_random_lm_gpu1`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：12:01 现场快照显示训练 Python PID `20716` 在 GPU1（bus `00000000:05:00.0`）运行，GPU1 `16739MiB / 24576MiB`、现场 util `42%`、温度 `56C`；GPU0 `0MiB / 24576MiB`，没有 VIVID 训练。12:00 resource guard 记录 GPU1 util `32%`、温度 `60C`、power `122.27W`，resource guard OK，无 stop action；仍只使用 GPU1。
- 训练进度：11:35 从 `step_9000.pt` 重启恢复后，12:01 已推进到恢复段 local `43 / 1000`，折算全局约 `9043 / 10000`。距离 `step_9500.pt` 约 `457` step；random-LM pretrain 到 `final.pt` 还剩约 `957` optimizer step。
- 一步多久：12:01 最新一步约 `40.61s/optimizer step`；重启恢复段最近 20/50/100 个有效进度样本均值约 `46.86/27.78/27.78s/optimizer step`。恢复窗口仍较短且含启动抖动，ETA 需要继续按后续窗口收敛；日志约 33 秒内刷新、GPU1 进程/显存稳定，判断为健康推进。
- 验证/checkpoint：最新可靠验证点仍为 `Step 9000: val_loss = 0.4782`，优于 step 8500 的 `0.4804`；`outputs/ablation_ums_random_lm_12label/checkpoints/step_9000.pt` 已完成。`step_9500.pt` 与 `final.pt` 尚未生成。
- 结果判断：validation loss 轨迹仍为 `0.6806 -> 0.6277 -> 0.5917 -> 0.5609 -> 0.5364 -> 0.5167 -> 0.5077 -> 0.4969 -> 0.4804 -> 0.4782`，整体符合预期。12:01 当前训练 loss 约 `0.6400`，learning rate 约 `1.02e-05`；恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`。
- 队列/守卫：11:40 至 12:00 队列持续识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK，无 stop action。
- Case study：本小时无新增失败案例；本次重启恢复的 case study 已保存为 `History/20260613_1146_random_lm_reboot_step9000_relaunch/case_study.md`。
- 当前硬缺口为 `outputs/ablation_ums_random_lm_12label/checkpoints/step_9500.pt`、`outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 50 步平均 `27.78s/optimizer step` 估计，约 `3.5` 小时到 `step_9500.pt`，random-LM pretrain 剩余约 `7.4` 小时；恢复窗口仍短，保守估计 pretrain final 约 `6-8` 小时。完整目标在 pretrain 后还剩 random-LM LP、两个 eval 和最终表格/claim cleanup，当前保守估计约 `12-20` 小时，待 `final.pt` 落盘后按实际 LP/eval 速度重新收窄。

2026-06-13 13:00 小时同步（random-LM 重启后恢复段推进到 step 9156）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，仍是当前主任务 `VIVID_random_lm_gpu1`。本小时继续主线程巡检，未新增子智能体。
- GPU 约束：13:00 现场快照显示训练 Python PID `20716` 在 GPU1（bus `00000000:05:00.0`）运行，GPU1 `16739MiB / 24576MiB`、现场 util `1%`、温度 `59C`；GPU0 `0MiB / 24576MiB`，没有 VIVID 训练。13:00 resource guard 记录 GPU0 `0MiB`、GPU1 util `33%`、温度 `63C`、power `148.81W`，resource guard OK，无 stop action；仍只使用 GPU1。
- 训练进度：11:35 从 `step_9000.pt` 重启恢复后，13:00 已推进到恢复段 local `156 / 1000`，折算全局约 `9156 / 10000`。距离 `step_9500.pt` 约 `344` step；random-LM pretrain 到 `final.pt` 还剩约 `844` optimizer step。
- 一步多久：13:00 最新一步约 `27.58s/optimizer step`；重启恢复段最近 20/50/100 个有效进度样本均值约 `21.11/24.09/28.74s/optimizer step`。日志约 17 秒内刷新、GPU1 进程/显存稳定，判断为健康推进。
- 验证/checkpoint：最新可靠验证点仍为 `Step 9000: val_loss = 0.4782`，优于 step 8500 的 `0.4804`；`outputs/ablation_ums_random_lm_12label/checkpoints/step_9000.pt` 已完成。`step_9500.pt` 与 `final.pt` 尚未生成。
- 结果判断：validation loss 轨迹仍为 `0.6806 -> 0.6277 -> 0.5917 -> 0.5609 -> 0.5364 -> 0.5167 -> 0.5077 -> 0.4969 -> 0.4804 -> 0.4782`，整体符合预期。13:00 当前训练 loss 约 `0.6481`，learning rate 约 `1.02e-05`；恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`。
- 队列/守卫：12:55 至 13:00 队列持续识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK，无 stop action。
- Case study：本小时无新增失败案例；最新失败/恢复案例仍为 `History/20260613_1146_random_lm_reboot_step9000_relaunch/case_study.md`。
- 当前硬缺口为 `outputs/ablation_ums_random_lm_12label/checkpoints/step_9500.pt`、`outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 20/50 步平均估计，约 `2.0-2.3` 小时到 `step_9500.pt`，random-LM pretrain 剩余约 `4.9-5.6` 小时；按最近 100 步平均估计，pretrain 剩余约 `6.7` 小时。完整目标在 pretrain 后还剩 random-LM LP、两个 eval 和最终表格/claim cleanup，当前保守估计约 `10-18` 小时，待 `final.pt` 落盘后按实际 LP/eval 速度重新收窄。

2026-06-13 14:00 小时同步（random-LM 重启后恢复段推进到 step 9272）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，仍是当前主任务 `VIVID_random_lm_gpu1`。本小时按用户要求改为 20 分钟巡检，正式文档同步仍按整点写入；继续主线程巡检，未新增子智能体。
- GPU 约束：14:00 现场快照显示训练 Python PID `20716` 在 GPU1（bus `00000000:05:00.0`）运行，GPU1 `16739MiB / 24576MiB`、现场 util `30%`、温度 `59C`；GPU0 `0MiB / 24576MiB`，没有 VIVID 训练。14:00 resource guard 记录 GPU0 `0MiB`、GPU1 util `30%`、温度 `58C`、power `97.63W`，resource guard OK，无 stop action；仍只使用 GPU1。
- 训练进度：11:35 从 `step_9000.pt` 重启恢复后，14:00 已推进到恢复段 local `272 / 1000`，折算全局约 `9272 / 10000`。距离 `step_9500.pt` 约 `228` step；random-LM pretrain 到 `final.pt` 还剩约 `728` optimizer step。
- 一步多久：14:00 最新一步约 `27.53s/optimizer step`；重启恢复段最近 20/50/100 个有效进度样本均值约 `34.64/34.13/31.93s/optimizer step`。日志约 8 秒内刷新、GPU1 进程/显存稳定，判断为健康推进。
- 验证/checkpoint：最新可靠验证点仍为 `Step 9000: val_loss = 0.4782`，优于 step 8500 的 `0.4804`；`outputs/ablation_ums_random_lm_12label/checkpoints/step_9000.pt` 已完成。`step_9500.pt` 与 `final.pt` 尚未生成。
- 结果判断：validation loss 轨迹仍为 `0.6806 -> 0.6277 -> 0.5917 -> 0.5609 -> 0.5364 -> 0.5167 -> 0.5077 -> 0.4969 -> 0.4804 -> 0.4782`，整体符合预期。14:00 当前训练 loss 约 `0.6068`，learning rate 约 `1.01e-05`；恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`。
- 队列/守卫：13:55 至 14:00 队列持续识别 `random_lm_running=True`，未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval；resource guard OK，无 stop action。
- Case study：本小时无新增失败案例；最新失败/恢复案例仍为 `History/20260613_1146_random_lm_reboot_step9000_relaunch/case_study.md`。
- 当前硬缺口为 `outputs/ablation_ums_random_lm_12label/checkpoints/step_9500.pt`、`outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 20/50 步平均估计，约 `2.16-2.19` 小时到 `step_9500.pt`，random-LM pretrain 剩余约 `6.90-7.01` 小时；按最近 100 步平均估计，pretrain 剩余约 `6.46` 小时。完整目标在 pretrain 后还剩 random-LM LP、两个 eval 和最终表格/claim cleanup，当前保守估计约 `10-18` 小时，待 `final.pt` 落盘后按实际 LP/eval 速度重新收窄。

2026-06-13 15:00 小时同步（random-LM 重启后恢复段推进到 step 9402）：
- 当前阶段：`random-LM same-architecture` 预训练继续运行，仍是当前主任务 `VIVID_random_lm_gpu1`。本小时继续按 20 分钟巡检节奏监控，正式文档同步按整点写入；继续主线程巡检，未新增子智能体。
- GPU 约束：15:01 现场快照显示训练 Python PID `20716` 在 GPU1（bus `00000000:05:00.0`）运行，GPU1 `18263MiB / 24576MiB`、现场 util `71%`、温度 `59C`；GPU0 `0MiB / 24576MiB`，没有 VIVID 训练；仍只使用 GPU1。
- 训练进度：11:35 从 `step_9000.pt` 重启恢复后，15:01 已推进到恢复段 local `402 / 1000`，折算全局约 `9402 / 10000`。距离 `step_9500.pt` 约 `98` step；random-LM pretrain 到 `final.pt` 还剩约 `598` optimizer step。
- 一步多久：15:01 最新一步约 `27.99s/optimizer step`；重启恢复段最近 20/50/100 个有效进度样本均值约 `29.62/27.99/28.76s/optimizer step`。日志约 5 秒内刷新、GPU1 进程/显存稳定，判断为健康推进。
- 验证/checkpoint：最新可靠验证点仍为 `Step 9000: val_loss = 0.4782`，优于 step 8500 的 `0.4804`；`outputs/ablation_ums_random_lm_12label/checkpoints/step_9000.pt` 已完成。`step_9500.pt` 与 `final.pt` 尚未生成。
- 结果判断：validation loss 轨迹仍为 `0.6806 -> 0.6277 -> 0.5917 -> 0.5609 -> 0.5364 -> 0.5167 -> 0.5077 -> 0.4969 -> 0.4804 -> 0.4782`，整体符合预期。15:01 当前训练 loss 约 `0.5759`，learning rate 约 `1.01e-05`；恢复段未检出新的 `Traceback`、`RuntimeError`、`CUDA out of memory`、`GPU is lost`、`failed` 或 `error`。
- 队列/守卫：本轮现场进程列表显示 `conda run ... scripts\train_cxr.py --config configs\ablation_ums_random_lm_12label.yaml --resume ...step_9000.pt` 仍由 PID `20716` 执行；未重复启动 `VIVID_random_lm_gpu1`，也未误启动 random-LM LP 或后续 eval。
- Case study：本小时无新增失败案例；最新失败/恢复案例仍为 `History/20260613_1146_random_lm_reboot_step9000_relaunch/case_study.md`。
- 当前硬缺口为 `outputs/ablation_ums_random_lm_12label/checkpoints/step_9500.pt`、`outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终成本/汇总表整理和 claim cleanup。
- ETA：按最近 20/50 步平均估计，约 `0.76-0.81` 小时到 `step_9500.pt`，random-LM pretrain 剩余约 `4.65-4.92` 小时；按最近 100 步平均估计，pretrain 剩余约 `4.78` 小时。完整目标在 pretrain 后还剩 random-LM LP、两个 eval 和最终表格/claim cleanup，当前保守估计约 `10-18` 小时，待 `final.pt` 落盘后按实际 LP/eval 速度重新收窄。

2026-06-13 16:00 小时同步（random-LM 通过 step 9500 验证并继续训练）：
- 当前阶段：`random-LM same-architecture` UMS 预训练仍在运行，继续使用 `configs/ablation_ums_random_lm_12label.yaml`，从 `step_9000.pt` 恢复后的当前段已越过 9500 验证点并继续向 `max_steps=10000` 推进。
- GPU/进程：仅 GPU1 在用，GPU1 `18265/24576 MiB`，训练 PID `20716`；GPU0 `0 MiB`，符合“只用 GPU1”的约束。可见命令仍是 `python scripts\train_cxr.py --config configs\ablation_ums_random_lm_12label.yaml --resume H:\Xiyao_Wang\021_260129VIVID\outputs\ablation_ums_random_lm_12label\checkpoints\step_9000.pt`。
- 进度：日志最新活跃时间约 `2026-06-13 16:06:25`，当前段约 `523/1000`，全局约 `9523/10000`，还剩约 `477` 个 optimizer steps 才到 random-LM 预训练 `final.pt`。
- 每一步耗时/ETA：最新单步约 `30.84s/it`；近 20 步均值 `27.51s/it`，近 50 步均值 `36.79s/it`，近 100 步均值 `32.87s/it`。按这三个窗口估算，random-LM 预训练 final 还需约 `3.65-4.87h`，中位视角约 `4.36h`。
- 9500 验证/checkpoint：`Step 9500: val_loss = 0.4665` 已完成；`outputs/ablation_ums_random_lm_12label/checkpoints/best.pt` 于 `2026-06-13 15:57:38` 更新，torch metadata 验证为 `global_step=9500`、`best_val_loss=0.4664952790737152`。当前配置 `save_interval=1000`，所以 9500 不是常规 `step_9500.pt` 保存点；9500 关键 checkpoint 由 `best.pt` 承载，不再把缺少单独 `step_9500.pt` 视为失败。
- 结果是否符合预期：符合预期且走势更好。最近验证损失为 `7500=0.5077 -> 8000=0.4969 -> 8500=0.4804 -> 9000=0.4782 -> 9500=0.4665`，random-LM control 的 pretrain loss/val_loss 持续下降，没有看到异常反弹。
- 失败/病例记录：当前恢复段未发现 `Traceback`、`RuntimeError`、OOM、GPU lost 或进程退出。9500 验证开头单 batch 曾显得偏慢，但随后完整跑完约 500/500，并成功刷新 `best.pt`；暂不构成失败 case study。
- 仍未完成的硬缺口：`outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终 summary/claim cleanup/cost tables。
- 下一步：继续每 20 分钟主线程巡检 GPU1、日志活跃、checkpoint 和失败信息；到 random-LM `final.pt` 后立即启动对应 LP 与剩余 answerability diagnostics；整点继续写正式同步。

2026-06-13 17:00 小时同步（random-LM 已过 step 9500，继续冲刺 final.pt）：
- 当前阶段：`random-LM same-architecture` UMS 预训练仍在 GPU1 上运行，从 `step_9000.pt` 继续训练；当前进程为 PID `20716`，`CUDA_VISIBLE_DEVICES=1` 后进程内显示为 cuda:0。
- GPU/进程：物理 GPU1 bus `00000000:05:00.0`，显存约 `18265/24576 MiB`，现场 util 约 `12%`、温度 `58C`、功耗约 `116.26W`；GPU0 显存 `0MiB`，未被本实验占用，仍符合“只用 GPU1”的约束。
- 进度：恢复段已完成 local `643/1000` 个 optimizer steps，折算全局约 `9643/10000`，距离 random-LM 预训练 `final.pt` 还剩约 `357` 步。
- 每一步耗时/ETA：最新单步约 `24.75s/optimizer step`；近 20 步均值 `22.91s/step`，近 50 步均值 `24.49s/step`，近 100 步均值 `26.18s/step`。按当前窗口估计，预训练完成还需约 `2.27-2.60h`。
- 验证/checkpoint：`Step 9500: val_loss = 0.4665` 已完成；`best.pt` 已在 global step 9500 更新，`best_val_loss=0.4665`。`final.pt` 尚未生成。
- 结果是否符合预期：符合预期，而且验证趋势继续变好。最近验证损失为 `7500=0.5077 -> 8000=0.4969 -> 8500=0.4804 -> 9000=0.4782 -> 9500=0.4665`；当前最新训练 loss 约 `0.5423`，learning rate `1.00e-05`。
- 失败/病例记录：当前运行段没有发现 OOM、Traceback、RuntimeError、GPU lost 或进程退出；本小时没有新增失败案例，因此无需新增 case study。
- 仍未完成的硬缺口：`outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终结果汇总、claim cleanup 与 cost table。
- 说明：当前配置 `save_interval=1000`、`eval_interval=500`，所以 global step 9500 只触发验证和 `best.pt` 更新，不会生成独立的 `step_9500.pt`；9500 关键 checkpoint 已由 `best.pt` 覆盖。
- 下一步：继续 20 分钟级别巡检；一旦 `final.pt` 生成，立即在 GPU1 上启动 random-LM LP，然后继续跑剩余两个诊断实验。

2026-06-13 18:00 小时同步（random-LM 距离 final.pt 约 227 steps）：
- 当前阶段：`random-LM same-architecture` UMS 预训练仍在 GPU1 上运行，从 `step_9000.pt` 继续训练；当前训练进程仍为 PID `20716`，命令为 `python scripts\train_cxr.py --config configs\ablation_ums_random_lm_12label.yaml --resume H:\Xiyao_Wang\021_260129VIVID\outputs\ablation_ums_random_lm_12label\checkpoints\step_9000.pt`。
- GPU/进程：物理 GPU1 bus `00000000:05:00.0`，显存约 `18265/24576 MiB`，现场 util 约 `5%`、温度 `62C`、功耗约 `158.48W`；GPU0 显存 `0MiB`，未被本实验占用，仍符合“只用 GPU1”的约束。
- 进度：18:00 采样时日志最新活跃时间约 `2026-06-13 18:00:36`，恢复段已完成 local `773/1000` 个 optimizer steps，折算全局约 `9773/10000`，距离 random-LM 预训练 `final.pt` 还剩约 `227` 步。
- 每一步耗时/ETA：最新单步约 `24.28s/optimizer step`；近 20 步均值 `29.47s/step`，近 50 步均值 `29.41s/step`，近 100 步均值 `28.95s/step`。按当前窗口估计，random-LM 预训练完成还需约 `1.83-1.86h`。
- 验证/checkpoint：最新可靠验证仍为 `Step 9500: val_loss = 0.4665`；`best.pt` 已在 global step 9500 更新并承载该关键 checkpoint。`outputs/ablation_ums_random_lm_12label/checkpoints/final.pt` 尚未生成。
- 结果是否符合预期：符合预期且验证趋势继续改善。最近验证损失为 `7500=0.5077 -> 8000=0.4969 -> 8500=0.4804 -> 9000=0.4782 -> 9500=0.4665`；当前最新训练 loss 约 `0.5907`，learning rate `1.00e-05`。
- 失败/病例记录：当前运行段没有发现 `Traceback`、`RuntimeError`、CUDA OOM、GPU lost 或进程退出；本小时无新增失败案例，因此无需新增 case study。
- 仍未完成的硬缺口：`outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终结果汇总、claim cleanup 与 cost table。
- 下一步：继续 20 分钟主线程巡检；一旦 `final.pt` 生成，立即在 GPU1 上启动 random-LM LP，然后继续跑剩余两个诊断实验。

2026-06-13 19:00 小时同步（random-LM 距离 final.pt 约 96 steps）：
- 当前阶段：`random-LM same-architecture` UMS 预训练仍在 GPU1 上运行，从 `step_9000.pt` 继续训练；当前训练进程仍为 PID `20716`，命令为 `python scripts\train_cxr.py --config configs\ablation_ums_random_lm_12label.yaml --resume H:\Xiyao_Wang\021_260129VIVID\outputs\ablation_ums_random_lm_12label\checkpoints\step_9000.pt`。
- GPU/进程：物理 GPU1 bus `00000000:05:00.0`，显存约 `18265/24576 MiB`，现场 util 约 `42%`、温度 `56C`、功耗约 `60.72W`；GPU0 显存 `0MiB`，未被本实验占用，仍符合“只用 GPU1”的约束。队列守卫继续识别 `random_lm_running=True`，未误启动 random-LM LP 或后续诊断。
- 进度：19:00 采样时日志最新活跃时间约 `2026-06-13 18:59:39`，恢复段已完成 local `904/1000` 个 optimizer steps，折算全局约 `9904/10000`，距离 random-LM 预训练 `final.pt` 还剩约 `96` 步。
- 每一步耗时/ETA：最新单步约 `20.46s/optimizer step`；近 20 步均值 `21.79s/step`，近 50 步均值 `25.60s/step`，近 100 步均值 `26.85s/step`。按当前窗口估计，random-LM 预训练完成还需约 `0.58-0.72h`。
- 验证/checkpoint：最新可靠验证仍为 `Step 9500: val_loss = 0.4665`；`best.pt` 已在 global step 9500 更新并承载该关键 checkpoint。`outputs/ablation_ums_random_lm_12label/checkpoints/final.pt` 尚未生成，因此尚未启动 random-LM LP。
- 结果是否符合预期：符合预期且验证趋势继续改善。最近验证损失为 `7500=0.5077 -> 8000=0.4969 -> 8500=0.4804 -> 9000=0.4782 -> 9500=0.4665`；当前最新训练 loss 约 `0.5416`，learning rate `1.00e-05`。
- 失败/病例记录：当前运行段没有发现 `Traceback`、`RuntimeError`、CUDA OOM、GPU lost 或进程退出；本小时无新增失败案例，因此无需新增 case study。
- 仍未完成的硬缺口：`outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终结果汇总、claim cleanup 与 cost table。
- 下一步：继续 20 分钟主线程巡检；一旦 `final.pt` 生成，立即在 GPU1 上启动 random-LM LP，然后继续跑剩余两个诊断实验。

2026-06-13 19:15 失败案例与恢复记录（random-LM dataloader PermissionError）：
- 失败点：19:10 巡检发现 11:35 从 `step_9000.pt` 恢复的 random-LM 进程已在 local `906/1000`、折算 global `9906/10000` 处失败退出，`final.pt` 未生成。
- 错误类型：日志显示 `training/trainer.py:632 batch = next(train_iter)` 触发 dataloader shutdown，随后在 Windows multiprocessing `popen_spawn_win32.py` 的 `TerminateProcess` 出现 `PermissionError: [WinError 5] 拒绝访问`，conda run exitcode `1`。这不是 CUDA OOM，也不是 GPU0/GPU1 绑定错误。
- 影响：由于 `save_interval=1000`，global `9500 -> 9906` 的未保存进度无法恢复；当前可用 checkpoint 仍是 `best.pt`，metadata 为 `global_step=9500`、`best_val_loss=0.4664952790737152`。队列 19:05 自动从 `best.pt` 重启，因此最终 500-step 段需要重跑。
- 缓解：已将 `configs/ablation_ums_random_lm_12label.yaml` 的 `data.num_workers` 从 `4` 改为 `0`，仅改变 data loading 并行度，用于降低 Windows dataloader worker 终止权限错误风险；batch size `2`、gradient accumulation `16`、seed、模型与训练目标不变。
- 恢复动作：19:05 队列自动重启后，我在约 local `12/500` 时停止该多 worker 恢复段，并于 19:14:50 通过 `VIVID_random_lm_gpu1` 重新从 `best.pt` 启动 GPU1 恢复段，使新配置生效。case study 已写入 `History/20260613_1905_random_lm_dataloader_permission_relaunch/case_study.md`。
- 当前状态：等待 19:14:50 新进程完成模型加载并进入训练；下一轮继续检查 GPU1 显存、训练日志和 `final.pt`。

2026-06-13 20:00 小时同步（random-LM dataloader 恢复段推进到 global step 9655）：
- 当前阶段：`random-LM same-architecture` UMS 预训练仍在 GPU1 上运行；这是 19:05 dataloader `PermissionError` 后的稳定恢复段，19:14:50 从 `best.pt`（`global_step=9500`）重新启动。当前训练进程为 PID `6136`，命令为 `python scripts\train_cxr.py --config configs\ablation_ums_random_lm_12label.yaml --resume H:\Xiyao_Wang\021_260129VIVID\outputs\ablation_ums_random_lm_12label\checkpoints\best.pt`。
- GPU/进程：物理 GPU1 bus `00000000:05:00.0`，显存约 `16739/24576 MiB`，现场 util 约 `54%`、温度 `60C`、功耗约 `74.98W`；GPU0 显存 `0MiB`，未被本实验占用，仍符合“只用 GPU1”的约束。
- 进度：20:01 采样时日志最新活跃时间约 `2026-06-13 20:00:46`，19:14:50 恢复段已完成 local `155/500` 个 optimizer steps，折算全局约 `9655/10000`，距离 random-LM 预训练 `final.pt` 还剩约 `345` 步。
- 每一步耗时/ETA：最新单步约 `18.51s/optimizer step`；近 20 步均值 `19.45s/step`，近 50 步均值 `16.54s/step`，近 100 步均值 `17.38s/step`，近 200 步均值 `17.17s/step`。按当前窗口估计，random-LM 预训练完成还需约 `1.58-1.86h`。
- 验证/checkpoint：最新可靠验证仍为 `Step 9500: val_loss = 0.4665`；`best.pt` 已在 global step 9500 更新并承载该关键 checkpoint。`outputs/ablation_ums_random_lm_12label/checkpoints/final.pt` 尚未生成，因此尚未启动 random-LM LP。
- 结果是否符合预期：当前恢复段运行稳定，没有新错误；最近验证损失趋势仍为 `7500=0.5077 -> 8000=0.4969 -> 8500=0.4804 -> 9000=0.4782 -> 9500=0.4665`，符合预期。当前最新训练 loss 约 `0.6401`，learning rate `1.00e-05`，属于单步训练波动。
- 失败/病例记录：19:05 dataloader `PermissionError` 已在 19:15 记录并保存 case study；20:00 当前恢复段没有发现 `Traceback`、`RuntimeError`、CUDA OOM、GPU lost、`PermissionError` 或进程退出，无新增失败案例。
- 仍未完成的硬缺口：`outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终结果汇总、claim cleanup 与 cost table。
- 下一步：继续 20 分钟主线程巡检；一旦 `final.pt` 生成，立即验证 checkpoint metadata，随后在 GPU1 上启动 random-LM LP，再继续跑剩余两个诊断实验。

2026-06-13 21:00 小时同步（random-LM 距离 final.pt 约 70 steps；21:17 现场采样）：
- 当前阶段：`random-LM same-architecture` UMS 预训练仍在 GPU1 上运行；这是 19:05 dataloader `PermissionError` 后的稳定恢复段，19:14:50 从 `best.pt`（`global_step=9500`）重新启动。当前训练进程为 PID `6136`，命令为 `python scripts\train_cxr.py --config configs\ablation_ums_random_lm_12label.yaml --resume H:\Xiyao_Wang\021_260129VIVID\outputs\ablation_ums_random_lm_12label\checkpoints\best.pt`。
- GPU/进程：物理 GPU1 bus `00000000:05:00.0`，显存约 `18263/24576 MiB`，现场 util 约 `40%`、温度 `57C`、功耗约 `76.28W`；GPU0 显存 `0MiB`，未被本实验占用，仍符合“只用 GPU1”的约束。
- 进度：21:17 采样时日志最新活跃时间约 `2026-06-13 21:17:26`，19:14:50 恢复段已完成 local `430/500` 个 optimizer steps，折算全局约 `9930/10000`，距离 random-LM 预训练 `final.pt` 还剩约 `70` 步。
- 每一步耗时/ETA：最新单步约 `20.08s/optimizer step`；近 20 步均值 `14.27s/step`，近 50 步均值 `14.97s/step`，近 100 步均值 `16.82s/step`，近 200 步均值 `16.90s/step`。按当前窗口估计，random-LM 预训练完成还需约 `0.28-0.33h`。
- 验证/checkpoint：最新可靠验证仍为 `Step 9500: val_loss = 0.4665`；`best.pt` 已在 global step 9500 更新并承载该关键 checkpoint。`outputs/ablation_ums_random_lm_12label/checkpoints/final.pt` 尚未生成，因此尚未启动 random-LM LP。
- 结果是否符合预期：当前恢复段运行稳定，没有新错误；最近验证损失趋势仍为 `7500=0.5077 -> 8000=0.4969 -> 8500=0.4804 -> 9000=0.4782 -> 9500=0.4665`，符合预期。当前最新训练 loss 约 `0.6149`，learning rate `1.00e-05`，属于单步训练波动。
- 失败/病例记录：19:05 dataloader `PermissionError` 已在 19:15 记录并保存 case study；21:17 当前恢复段没有发现 `Traceback`、`RuntimeError`、CUDA OOM、GPU lost、`PermissionError` 或进程退出，无新增失败案例。
- 仍未完成的硬缺口：`outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终结果汇总、claim cleanup 与 cost table。
- 下一步：继续 20 分钟主线程巡检；一旦 `final.pt` 生成，立即验证 checkpoint metadata，随后在 GPU1 上启动 random-LM LP，再继续跑剩余两个诊断实验。

2026-06-13 21:46 关键 checkpoint 记录（random-LM pretrain 完成）：
- 完成项：`random-LM same-architecture` UMS 预训练已完成，训练日志显示 `Training completed!`，exitcode `0`。最终恢复段从 19:14:50 开始，local `500/500` 完成，折算全局 `10000/10000`。
- 验证结果：`Step 10000: val_loss = 0.4608`，优于此前 `Step 9500: val_loss = 0.4665`，结果符合预期且验证趋势继续改善。
- checkpoint 验证：`outputs/ablation_ums_random_lm_12label/checkpoints/final.pt` 已生成；metadata 为 `global_step=10000`、`best_val_loss=0.46084351754188535`。同轮生成/更新 `best.pt` 与 `step_10000.pt`，metadata 同为 `global_step=10000`、`best_val_loss=0.46084351754188535`。
- GPU/进程：21:46 复查时 GPU0 与 GPU1 显存均已释放到 `0MiB`，random-LM 预训练进程已结束。下一步将使用 `scripts/run_lp_random_lm_gpu1.cmd` 在 GPU1 上启动 random-LM LP，配置为 `configs/lp_ums_random_lm_12label.yaml`，其 `transfer.init_vit_checkpoint` 指向 `./outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`（当前 best 已是 global step 10000）。
- 仍未完成的硬缺口：`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终结果汇总、claim cleanup 与 cost table。

2026-06-13 22:00 小时同步（random-LM pretrain 完成，LP 已在 GPU1 运行；22:17 现场采样）：
- 当前阶段：`random-LM same-architecture` 预训练已完成并验证通过；当前主任务已切到 random-LM LP。LP 于 `2026-06-13 21:50:04` 通过 `scripts/run_lp_random_lm_gpu1.cmd` 启动，命令为 `python scripts\train_vit_baseline.py --config configs\lp_ums_random_lm_12label.yaml`，配置 `configs/lp_ums_random_lm_12label.yaml`，初始化 checkpoint 为 `./outputs/ablation_ums_random_lm_12label/checkpoints/best.pt`（metadata `global_step=10000`、`best_val_loss=0.46084351754188535`）。
- GPU/进程：物理 GPU1 bus `00000000:05:00.0`，LP 训练进程 PID `5840`，显存约 `1933/24576 MiB`；GPU0 显存 `0MiB`，未被本实验占用，仍符合“只用 GPU1”的约束。
- LP 进度：22:17 采样时 LP 日志最新活跃时间约 `2026-06-13 22:17:31`，已推进到 step `1800/3000` 附近并进入 step 1800 validation；采样解析到最新训练步约 `1775/3000`，随后日志尾部显示已到 `1800/3000`。当前已生成 `metrics_step_200/400/600/800/1000/1200/1400/1600.json`，并在 step 600/1200 保存 checkpoint；`best.pt` 已于 22:14 更新。
- 每一步耗时/ETA：22:17 解析窗口显示最新单步约 `1.11s/step`；近 20/50/100/200 步均值约 `4.21/5.73/6.95/6.76s/step`（含 dataloader 抖动与 validation 附近开销）。按训练步窗口粗估，LP 剩余约 `1225` steps，约 `1.43-2.36h`；实际总 ETA 会受后续 1800/2000/2200/2400/2600/2800/3000 validation 影响。
- 结果是否符合预期：LP 已正常加载 random-LM pretrain 的 ViT backbone，日志显示 `Loaded params: 150`，backbone 冻结、trainable head 参数 `10,766`，训练 loss 已降到约 `0.17-0.19` 区间，运行符合预期。当前尚未生成 `metrics_final.json`。
- 失败/病例记录：当前 LP 运行没有发现 `Traceback`、`RuntimeError`、CUDA OOM、GPU lost、`PermissionError` 或进程退出；无新增失败案例。
- 仍未完成的硬缺口：`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终结果汇总、claim cleanup 与 cost table。
- 下一步：继续 20 分钟主线程巡检 LP；LP `metrics_final.json` 生成后，立即验证关键指标并在 GPU1 上衔接剩余两个诊断实验。

2026-06-13 22:37 关键结果记录（random-LM LP 完成）：
- 完成项：random-LM LP 已完成，日志显示 `Training completed!`，exitcode `0`；`outputs/lp_ums_random_lm_12label/final.pt` 与 `outputs/lp_ums_random_lm_12label/metrics_final.json` 已生成。
- 最终指标：`val_loss=0.3198051466828301`、`macro_auc=0.7411244610094135`、`macro_f1=0.8552836154498756`、`micro_f1=0.8575395723410164`。最终 checkpoint/metrics 来自 step `3000/3000`。
- 结果是否符合预期：LP 顺利加载 random-LM pretrain 的 ViT backbone（pretrain best 已是 `global_step=10000`），运行到 final，无 OOM、Traceback、PermissionError 或 GPU lost。指标低于 null-as-negative LP，但 random-LM 是 same-architecture 随机语言模型对照，作为能力削弱/结构对照符合预期。
- GPU/进程：22:37 复查时 GPU0 与 GPU1 显存均为 `0MiB`，LP 进程已结束。下一步使用 GPU1 启动 `counterfactual_prefix_dependency_A_ums_12label_128` 诊断。
- 仍未完成的硬缺口：`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`，以及最终结果汇总、claim cleanup 与 cost table。

2026-06-13 22:57 关键诊断记录（prefix 与 field paraphrase 完成）：
- 完成项：`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json` 与 `outputs/field_paraphrase_robustness_A_ums_12label_128.json` 均已生成；两个诊断日志均显示 exitcode `0`。22:57 复查时 GPU0/GPU1 显存均为 `0MiB`，无诊断进程残留。
- `counterfactual_prefix_dependency` 摘要：`sample_count=128`、`batch_size=2`；image 条件下 `state_flip pairwise_accuracy=0.875`、`field_swap pairwise_accuracy=0.8831168831168831`、`null_to_present pairwise_accuracy=0.875`。该结果支持模型对 schema 字段/状态扰动有较强敏感性，符合预期。
- `field_paraphrase_robustness` 摘要：`sample_count=128`、`batch_size=2`；original NLL `0.04818403269720607`，clinical paraphrase `variant_nll_mean=1.1919437535107136`、`original_better_rate=1.0`、`relative_delta_vs_original=23.743913253700374`；lay paraphrase `variant_nll_mean=1.333443476818502`、`original_better_rate=1.0`、`relative_delta_vs_original=26.68134790079511`。该结果显示模型明显偏好原始 schema 字段名，对 paraphrase 敏感，属于需要在论文 claim 中谨慎表述的 failure/limitation 信号。
- 失败/病例记录：本轮无运行失败、无 OOM、无 GPU lost、无 PermissionError；但 field paraphrase 结果本身构成鲁棒性限制案例，后续最终 summary/claim cleanup 需将其作为 case study 纳入。
- 当前硬缺口：核心实验产物已齐；仍需完成最终结果汇总、claim cleanup、cost/time table，并严格复核 `miccai2026.md` 中所有硬缺口是否关闭。

2026-06-13 23:00 最终汇总与 claim cleanup（MICCAI 实验目标完成前复核）：
- 产物闭环：本轮 `miccai2026.md` 要求的剩余核心产物已全部生成并验证存在：`outputs/ablation_ums_random_lm_12label/checkpoints/final.pt`、`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`。此前已完成的 `outputs/counterfactual_schema_grounding_A_ums_12label_128.json` 与 `outputs/schema_key_robustness_A_ums_12label_128.json` 也已复核存在。
- random-LM pretrain：最终 checkpoint `final.pt` metadata 为 `global_step=10000`、`best_val_loss=0.46084351754188535`；最终验证 `Step 10000: val_loss=0.4608`，优于 step 9500 的 `0.4665`，pretrain 结果符合预期。
- LP 对照结果表：
  - `ansmask` LP：`macro_auc=0.8177771890161666`、`macro_f1=0.9120033620176754`、`micro_f1=0.8964176617606221`、`val_loss=0.2495641630320322`。
  - `null-as-negative` LP：`macro_auc=0.8334071676048904`、`macro_f1=0.9132557148420648`、`micro_f1=0.897528464315468`、`val_loss=0.2472998457295554`。
  - `random-LM same-architecture` LP：`macro_auc=0.7411244610094135`、`macro_f1=0.8552836154498756`、`micro_f1=0.8575395723410164`、`val_loss=0.3198051466828301`。
- 诊断结果表：
  - `counterfactual_prefix_dependency`：`sample_count=128`；image 条件下 `state_flip pairwise_accuracy=0.875`、`field_swap pairwise_accuracy=0.8831168831168831`、`null_to_present pairwise_accuracy=0.875`。
  - `field_paraphrase_robustness`：`sample_count=128`；original NLL `0.04818403269720607`；clinical paraphrase `variant_nll_mean=1.1919437535107136`、`original_better_rate=1.0`、`relative_delta_vs_original=23.743913253700374`；lay paraphrase `variant_nll_mean=1.333443476818502`、`original_better_rate=1.0`、`relative_delta_vs_original=26.68134790079511`。
- Claim cleanup：
  - 支持的表述：UMS/schema-aware 训练产物完成；random-LM same-architecture control 显著弱于 null-as-negative/ansmask LP，说明性能并非仅来自相同 ViT/训练管线或随机语言模型结构。
  - 支持的表述：prefix/counterfactual 诊断显示模型对 schema 字段和状态扰动具有较强 pairwise sensitivity。
  - 必须谨慎的表述：不能声称模型对任意 schema paraphrase 鲁棒。field paraphrase 结果显示 original schema NLL 明显低于 clinical/lay paraphrase，`original_better_rate=1.0`，应作为 limitation/case study 写入论文。
  - 失败/限制案例：运行失败方面，19:05 Windows dataloader `PermissionError` 已记录在 `History/20260613_1905_random_lm_dataloader_permission_relaunch/case_study.md` 并通过 `num_workers=0` 恢复；行为层面，field paraphrase sensitivity 是最终需要呈现的鲁棒性限制案例。
- Time/cost table：
  - 运行资源：全程按约束只使用物理 GPU1（bus `00000000:05:00.0`）执行 VIVID 任务；最终复查 GPU0/GPU1 均为空闲 `0MiB`。未使用云实例，云成本 `$0`。
  - random-LM 最终恢复段：`2026-06-13 19:14:50 -> 21:45:21`，约 `2.5h`，完成 `best.pt global_step=9500 -> final.pt global_step=10000`。
  - random-LM LP：`2026-06-13 21:50:04 -> 22:35:34`，约 `45.5min`，完成 `3000/3000` steps 与 final validation。
  - counterfactual prefix 诊断：`2026-06-13 22:38:39 -> 22:42:05`，约 `3.5min`。
  - field paraphrase 诊断：`2026-06-13 22:45:06 -> 22:46:35`，约 `1.5min`。
- 结构/归档：失败和中断案例均已归入 `History/`；本轮未新增无用散落文件。当前核心输出留在 `outputs/`，日志留在 `outputs/logs/`，配置留在 `configs/`，项目结构保持规整。
- 最终状态：核心实验、诊断、case study、结果汇总、claim cleanup 与 time/cost table 均已完成。除后续论文写作排版外，`miccai2026.md` 本轮实验目标的硬缺口已关闭。

2026-06-27 17:58 当前状态复核（重启后巡检）：
- 核心 MICCAI 收尾产物复核：`outputs/ablation_ums_random_lm_12label/checkpoints/final.pt` 与 `best.pt` 均存在，metadata 为 `global_step=10000`、`best_val_loss=0.46084351754188535`；`outputs/lp_ums_random_lm_12label/metrics_final.json`、`outputs/counterfactual_prefix_dependency_A_ums_12label_128.json`、`outputs/field_paraphrase_robustness_A_ums_12label_128.json`、`outputs/counterfactual_schema_grounding_A_ums_12label_128.json`、`outputs/schema_key_robustness_A_ums_12label_128.json` 均存在。`History/20260613_1905_random_lm_dataloader_permission_relaunch/case_study.md` 仍在位。
- 核心结果复核：random-LM LP `macro_auc=0.7411244610094135`、`macro_f1=0.8552836154498756`、`micro_f1=0.8575395723410164`、`val_loss=0.3198051466828301`；prefix 诊断 `sample_count=128`，image 条件下 `state_flip=0.875`、`field_swap=0.8831168831168831`、`null_to_present=0.875`；field paraphrase 诊断仍显示 original schema 明显优于 clinical/lay paraphrase，作为 limitation/case study 结论不变。
- 当前正在运行的额外 data scaling 训练：GPU0 上 `configs/data_scaling/frozen_lm_ums_10k.yaml`，PID `10424`，日志 `outputs/logs/data_scaling_frozen_lm_ums_10k_source_gpu0.log`；GPU1 上 `configs/data_scaling/frozen_lm_ums_30k.yaml`，PID `19372`，日志 `outputs/logs/data_scaling_frozen_lm_ums_30k_source_gpu1.log`。两个启动脚本分别设置 `CUDA_VISIBLE_DEVICES=0/1`，配置内 `device: cuda:0` 是可见 GPU 映射后的局部编号。
- 训练配置：两个 data scaling run 均为 `max_steps=10000`、`batch_size=4`、`gradient_accumulation_steps=8`、`num_workers=0`、`bf16=true`，符合此前降低 batch size 与 Windows dataloader 稳定性的设置。
- 当前进度与 ETA：10k/GPU0 run 约 `4746/10000`，单步约 `2.90s/it`，剩余约 `4.2h`；最近验证 `step 4500 val_loss=0.0401`。30k/GPU1 run 约 `2573/10000`，单步约 `3.11s/it`，剩余约 `6.4h`；最近验证 `step 2500 val_loss=0.0412`。整体完成时间由 GPU1 的 30k run 决定，粗估还需约 `6.5h`，再加最终 validation/checkpoint 写盘波动。
- 失败检查：两个 data scaling 日志当前均未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`PermissionError` 或 `EXITCODE`；训练 loss/val_loss 下降趋势正常，结果符合预期。
- 注意事项：当前 GPU0 的 10k data scaling run 是正在占用 GPU0 的 VIVID 进程；若后续严格恢复“只用 GPU1”约束，需要单独决定是否中止 GPU0 run。目前未中断任何进程。

2026-06-27 17:59 当前状态复核（data scaling 继续运行）：
- GPU/进程：GPU0 `24299/24576 MiB`、GPU1 `24305/24576 MiB`。GPU0 运行 `frozen_lm_ums_10k`，PID `10424`；GPU1 运行 `frozen_lm_ums_30k`，PID `19372`。两个进程均由对应 `scripts/run_data_scaling_frozen_lm_ums_*` 启动脚本拉起。
- 10k/GPU0 进度：`4778/10000`，约 `48%`，最近窗口约 `2.91s/step`，日志 ETA 约 `4:13:07`；最近验证点仍为 `step 4500 val_loss=0.0401`。已生成 `step_1000/2000/3000/4000.pt` 与 `best.pt`。
- 30k/GPU1 进度：`2603/10000`，约 `26%`，最近窗口约 `3.10s/step`，日志 ETA 约 `6:21:40`；最近验证点仍为 `step 2500 val_loss=0.0412`。已生成 `step_1000/2000.pt` 与 `best.pt`。
- 剩余工作量：10k run 剩余 `5222` steps；30k run 剩余 `7397` steps。当前整体 ETA 由 GPU1 的 30k run 决定，约 `6.4h`，后续仍需等待 final checkpoint/metrics 生成并复核。
- 失败检查与结果判断：两个 data scaling 日志仍未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`PermissionError` 或 `EXITCODE`；loss 与 val_loss 走势正常，目前符合预期。无新增 failure case，因此不新增 case study。

2026-06-27 18:00 整点同步（data scaling 运行中）：
- GPU/进程：GPU0 `24299/24576 MiB`，运行 `frozen_lm_ums_10k` PID `10424`；GPU1 `24305/24576 MiB`，运行 `frozen_lm_ums_30k` PID `19372`。两条训练均仍活跃。
- 10k/GPU0 进度：`4801/10000`，约 `48%`；剩余 `5199` steps；当前约 `2.90s/step`，日志 ETA 约 `4:11:15`。最近验证仍为 `step 4500 val_loss=0.0401`，尚未到 `step 5000` 验证/保存点。
- 30k/GPU1 进度：`2625/10000`，约 `26%`；剩余 `7375` steps；当前约 `3.10s/step`，日志 ETA 约 `6:20:49`。最近验证仍为 `step 2500 val_loss=0.0412`，尚未到 `step 3000` 保存点。
- 当前目标剩余时间：若以当前正在运行的 data scaling 作为剩余目标，整体由 GPU1 `30k` 决定，粗估仍需约 `6.35h`，另加 final validation/checkpoint 写盘与结果复核时间。
- 失败检查与结果判断：两个日志仍未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`PermissionError` 或退出失败；训练 loss/val_loss 正常，阶段性结果符合预期。无新增 failure case。

2026-06-27 18:01 运行巡检（data scaling 继续推进）：
- GPU/进程：GPU0 `24299/24576 MiB`，`frozen_lm_ums_10k` PID `10424`；GPU1 `24305/24576 MiB`，`frozen_lm_ums_30k` PID `19372`。两条训练仍活跃。
- 10k/GPU0 进度：`4825/10000`，约 `48%`；剩余 `5175` steps；最近约 `2.91s/step`，日志 ETA 约 `4:11:03`。最近验证点仍为 `step 4500 val_loss=0.0401`；当前未产生新的 `step_5000` checkpoint。
- 30k/GPU1 进度：`2647/10000`，约 `26%`；剩余 `7353` steps；最近约 `3.11s/step`，日志 ETA 约 `6:20:43`。最近验证点仍为 `step 2500 val_loss=0.0412`；当前未产生新的 `step_3000` checkpoint。
- 当前目标剩余时间：整体仍由 GPU1 `30k` 决定，粗估约 `6.35h`，之后需要 final validation/checkpoint 写盘和结果复核。
- 失败检查与结果判断：两个日志未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`PermissionError` 或退出失败；loss 处于正常训练区间，阶段性结果符合预期。无新增 case study。

2026-06-27 18:02 运行巡检（data scaling 继续推进）：
- GPU/进程：GPU0 `24299/24576 MiB`，`frozen_lm_ums_10k` PID `10424`；GPU1 `24305/24576 MiB`，`frozen_lm_ums_30k` PID `19372`。两条训练仍活跃。
- 10k/GPU0 进度：`4847/10000`，约 `48%`；剩余 `5153` steps；最近约 `2.91s/step`，日志 ETA 约 `4:09:33`。最近验证点仍为 `step 4500 val_loss=0.0401`；当前未产生新的 `step_5000` checkpoint。
- 30k/GPU1 进度：`2668/10000`，约 `27%`；剩余 `7332` steps；最近约 `3.13s/step`，日志 ETA 约 `6:22:42`。最近验证点仍为 `step 2500 val_loss=0.0412`；当前未产生新的 `step_3000` checkpoint。
- 当前目标剩余时间：整体仍由 GPU1 `30k` 决定，粗估约 `6.4h`，之后需要 final validation/checkpoint 写盘和结果复核。
- 失败检查与结果判断：两个日志未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`PermissionError` 或退出失败；loss 处于正常训练区间，阶段性结果符合预期。无新增 case study。

2026-06-27 18:03 运行巡检（data scaling 继续推进）：
- GPU/进程：GPU0 `24299/24576 MiB`，`frozen_lm_ums_10k` PID `10424`；GPU1 `24305/24576 MiB`，`frozen_lm_ums_30k` PID `19372`。两条训练仍活跃。
- 10k/GPU0 进度：`4873/10000`，约 `49%`；剩余 `5127` steps；最近约 `2.92s/step`，日志 ETA 约 `4:09:09`。最近验证点仍为 `step 4500 val_loss=0.0401`；当前未产生新的 `step_5000` checkpoint。
- 30k/GPU1 进度：`2691/10000`，约 `27%`；剩余 `7309` steps；最近约 `3.09s/step`，日志 ETA 约 `6:16:48`。最近验证点仍为 `step 2500 val_loss=0.0412`；当前未产生新的 `step_3000` checkpoint。
- 当前目标剩余时间：整体仍由 GPU1 `30k` 决定，粗估约 `6.3h`，之后需要 final validation/checkpoint 写盘和结果复核。
- 失败检查与结果判断：两个日志未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`PermissionError` 或退出失败；训练 loss 正常波动，阶段性结果符合预期。无新增 case study。

2026-06-27 18:05 运行巡检（data scaling 继续推进）：
- GPU/进程：GPU0 `24299/24576 MiB`，`frozen_lm_ums_10k` PID `10424`；GPU1 `24305/24576 MiB`，`frozen_lm_ums_30k` PID `19372`。两条训练仍活跃。
- 10k/GPU0 进度：`4899/10000`，约 `49%`；剩余 `5101` steps；最近日志窗口约 `3.31s/step`，ETA 波动到约 `4:41:20`。最近验证点仍为 `step 4500 val_loss=0.0401`；当前未产生新的 `step_5000` checkpoint。
- 30k/GPU1 进度：`2714/10000`，约 `27%`；剩余 `7286` steps；最近日志窗口约 `3.54s/step`，ETA 波动到约 `7:09:26`。最近验证点仍为 `step 2500 val_loss=0.0412`；当前未产生新的 `step_3000` checkpoint。
- 当前目标剩余时间：整体仍由 GPU1 `30k` 决定。按本次日志瞬时 ETA 约 `7.2h`，较上一轮变慢；当前没有错误信号，暂按局部速度/IO 抖动处理，继续观察到下一次稳定窗口和 `step 3000` 验证点。
- 失败检查与结果判断：两个日志未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`PermissionError` 或退出失败；训练 loss 正常波动，阶段性结果仍符合预期。无新增 case study。

2026-06-27 18:06 运行巡检（data scaling 继续推进）：
- GPU/进程：GPU0 `24299/24576 MiB`，`frozen_lm_ums_10k` PID `10424`；GPU1 `24305/24576 MiB`，`frozen_lm_ums_30k` PID `19372`。两条训练仍活跃；另有一个只读监控 PowerShell 进程在等待下一次采样。
- 10k/GPU0 进度：`4927/10000`，约 `49%`；剩余 `5073` steps；最近约 `2.93s/step`，日志 ETA 约 `4:07:28`。最近验证点仍为 `step 4500 val_loss=0.0401`；当前未产生新的 `step_5000` checkpoint。
- 30k/GPU1 进度：`2741/10000`，约 `27%`；剩余 `7259` steps；最近约 `3.07s/step`，日志 ETA 约 `6:11:01`。最近验证点仍为 `step 2500 val_loss=0.0412`；当前未产生新的 `step_3000` checkpoint。
- 当前目标剩余时间：整体仍由 GPU1 `30k` 决定，ETA 已从上轮瞬时 `7.2h` 回落到约 `6.2h`；继续观察到 `step 3000` 验证/保存节点。
- 失败检查与结果判断：两个日志未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`PermissionError` 或退出失败；训练 loss 正常波动，阶段性结果仍符合预期。无新增 case study。

2026-06-27 18:07 运行巡检（data scaling 继续推进）：
- GPU/进程：GPU0 `24299/24576 MiB`，`frozen_lm_ums_10k` PID `10424`；GPU1 `24305/24576 MiB`，`frozen_lm_ums_30k` PID `19372`。两条训练仍活跃；另有一个只读监控 PowerShell 进程在等待下一次采样。
- 10k/GPU0 进度：`4954/10000`，约 `50%`；剩余 `5046` steps；最近约 `2.92s/step`，日志 ETA 约 `4:05:33`。最近验证点仍为 `step 4500 val_loss=0.0401`；当前未产生新的 `step_5000` checkpoint，但已接近 `step 5000` 验证/保存节点。
- 30k/GPU1 进度：`2766/10000`，约 `28%`；剩余 `7234` steps；最近约 `3.05s/step`，日志 ETA 约 `6:08:12`。最近验证点仍为 `step 2500 val_loss=0.0412`；当前未产生新的 `step_3000` checkpoint。
- 当前目标剩余时间：整体仍由 GPU1 `30k` 决定，粗估约 `6.1h`；下一关键节点是 GPU1 `step 3000` 验证/保存，GPU0 则很快到 `step 5000`。
- 失败检查与结果判断：两个日志未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`PermissionError` 或退出失败；训练 loss 正常波动，阶段性结果仍符合预期。无新增 case study。

2026-06-27 18:09 运行巡检（data scaling 继续推进）：
- GPU/进程：GPU0 `24299/24576 MiB`，`frozen_lm_ums_10k` PID `10424`；GPU1 `24305/24576 MiB`，`frozen_lm_ums_30k` PID `19372`。两条训练仍活跃。
- 10k/GPU0 进度：`4981/10000`，约 `50%`；剩余 `5019` steps；最近约 `2.91s/step`，日志 ETA 约 `4:03:33`。最近验证点仍为 `step 4500 val_loss=0.0401`；距离 `step 5000` 验证/保存节点约 `19` steps，当前尚未产生新的 `step_5000` checkpoint。
- 30k/GPU1 进度：`2792/10000`，约 `28%`；剩余 `7208` steps；最近约 `3.02s/step`，日志 ETA 约 `6:02:53`。最近验证点仍为 `step 2500 val_loss=0.0412`；当前未产生新的 `step_3000` checkpoint。
- 当前目标剩余时间：整体仍由 GPU1 `30k` 决定，粗估约 `6.05h`；下一关键节点是 GPU1 `step 3000` 验证/保存，GPU0 即将进入 `step 5000` 验证点。
- 失败检查与结果判断：两个日志未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`PermissionError` 或退出失败；训练 loss 正常波动，阶段性结果仍符合预期。无新增 case study。

2026-06-27 18:10 运行巡检（10k 进入 step 5000 validation）：
- GPU/进程：GPU0 `24299/24576 MiB`，`frozen_lm_ums_10k` PID `10424`；GPU1 `24305/24576 MiB`，`frozen_lm_ums_30k` PID `19372`。两条训练仍活跃。
- 10k/GPU0 进度：训练已进入 `step 5000` validation，当前 validation 约 `186/250` batches；因此训练主进度暂时不再按 `10000` step 递增。最近训练验证点仍为 `step 4500 val_loss=0.0401`；`step_5000` checkpoint 尚未生成，需等待本轮 validation 完成后再复核。
- 30k/GPU1 进度：`2821/10000`，约 `28%`；剩余 `7179` steps；最近约 `3.04s/step`，日志 ETA 约 `6:04:01`。最近验证点仍为 `step 2500 val_loss=0.0412`；当前未产生新的 `step_3000` checkpoint。
- 当前目标剩余时间：整体仍由 GPU1 `30k` 决定，粗估约 `6.1h`；下一关键节点是 GPU1 `step 3000` 验证/保存，GPU0 当前正在 `step 5000` validation/checkpoint 前置阶段。
- 失败检查与结果判断：两个日志未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`PermissionError` 或退出失败；GPU0 当前 validation 进度正常，GPU1 训练 loss 正常波动，阶段性结果仍符合预期。无新增 case study。

2026-06-27 18:12 关键 checkpoint 记录（10k step 5000 完成）：
- GPU/进程：GPU0 `24299/24576 MiB`，`frozen_lm_ums_10k` PID `10424`；GPU1 `24305/24576 MiB`，`frozen_lm_ums_30k` PID `19372`。两条训练仍活跃。
- 10k/GPU0 进度：`step 5000` validation 已完成，日志记录 `Step 5000: val_loss = 0.0392`；`outputs/data_scaling/frozen_lm_ums_10k/checkpoints/step_5000.pt` 已生成。随后训练继续推进到约 `5025/10000`，剩余 `4975` steps，最近约 `2.94s/step`，日志 ETA 约 `4:03:46`。该验证点较 `step 4500 val_loss=0.0401` 略有提升，符合预期。
- 30k/GPU1 进度：`2849/10000`，约 `28%`；剩余 `7151` steps；最近约 `3.16s/step`，日志 ETA 约 `6:17:06`。最近验证点仍为 `step 2500 val_loss=0.0412`；当前未产生新的 `step_3000` checkpoint。
- 当前目标剩余时间：整体仍由 GPU1 `30k` 决定，粗估约 `6.3h`；下一关键节点是 GPU1 `step 3000` validation/checkpoint。
- 失败检查与结果判断：两个日志未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`PermissionError` 或退出失败；10k 的 `step 5000` checkpoint 已正常落盘，GPU1 继续训练，阶段性结果符合预期。无新增 failure case。

2026-06-27 18:13 运行巡检（data scaling 继续推进）：
- GPU/进程：GPU0 `24299/24576 MiB`，`frozen_lm_ums_10k` PID `10424`；GPU1 `24305/24576 MiB`，`frozen_lm_ums_30k` PID `19372`。两条训练仍活跃。
- 10k/GPU0 进度：`5051/10000`，约 `51%`；剩余 `4949` steps；最近约 `3.28s/step`，日志 ETA 约 `4:30:39`。`step 5000 val_loss=0.0392` 与 `step_5000.pt` 已确认，训练已从 checkpoint 后继续推进。
- 30k/GPU1 进度：`2875/10000`，约 `29%`；剩余 `7125` steps；最近约 `3.42s/step`，日志 ETA 约 `6:45:58`。最近验证点仍为 `step 2500 val_loss=0.0412`；当前未产生新的 `step_3000` checkpoint。
- 当前目标剩余时间：整体仍由 GPU1 `30k` 决定；本轮 ETA 因短时速度波动升至约 `6.8h`，需继续观察是否回落。下一关键节点仍是 GPU1 `step 3000` validation/checkpoint。
- 失败检查与结果判断：两个日志未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`PermissionError` 或退出失败；10k checkpoint 后继续训练，GPU1 loss 正常波动，阶段性结果仍符合预期。无新增 failure case。

2026-06-27 18:15 运行巡检（GPU1 ETA 回落）：
- GPU/进程：GPU0 `24299/24576 MiB`，`frozen_lm_ums_10k` PID `10424`；GPU1 `24305/24576 MiB`，`frozen_lm_ums_30k` PID `19372`。两条训练仍活跃。
- 10k/GPU0 进度：`5080/10000`，约 `51%`；剩余 `4920` steps；最近约 `2.91s/step`，日志 ETA 约 `3:58:26`。`step 5000 val_loss=0.0392` 与 `step_5000.pt` 已确认，训练稳定继续推进。
- 30k/GPU1 进度：`2902/10000`，约 `29%`；剩余 `7098` steps；最近约 `3.03s/step`，日志 ETA 约 `5:58:53`。最近验证点仍为 `step 2500 val_loss=0.0412`；距离 `step 3000` validation/checkpoint 约 `98` steps。
- 当前目标剩余时间：整体仍由 GPU1 `30k` 决定；ETA 已从上轮短时 `6.8h` 回落到约 `6.0h`。下一关键节点是 GPU1 `step 3000` validation/checkpoint。
- 失败检查与结果判断：两个日志未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`PermissionError` 或退出失败；GPU1 速度恢复，10k checkpoint 后继续训练，阶段性结果仍符合预期。无新增 failure case。

2026-06-27 18:16 运行巡检（GPU1 接近 step 3000）：
- GPU/进程：GPU0 `24299/24576 MiB`，`frozen_lm_ums_10k` PID `10424`；GPU1 `24305/24576 MiB`，`frozen_lm_ums_30k` PID `19372`。两条训练仍活跃。
- 10k/GPU0 进度：`5109/10000`，约 `51%`；剩余 `4891` steps；最近约 `2.91s/step`，日志 ETA 约 `3:57:10`。`step 5000 val_loss=0.0392` 与 `step_5000.pt` 已确认，训练稳定继续推进。
- 30k/GPU1 进度：`2930/10000`，约 `29%`；剩余 `7070` steps；最近约 `3.03s/step`，日志 ETA 约 `5:56:28`。最近验证点仍为 `step 2500 val_loss=0.0412`；距离 `step 3000` validation/checkpoint 约 `70` steps。
- 当前目标剩余时间：整体仍由 GPU1 `30k` 决定，粗估约 `5.95h`。下一关键节点是 GPU1 `step 3000` validation/checkpoint。
- 失败检查与结果判断：两个日志未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`PermissionError` 或退出失败；GPU1 速度稳定在约 `3.0s/step`，10k 继续训练，阶段性结果仍符合预期。无新增 failure case。

2026-06-27 18:17 运行巡检（GPU1 临近 step 3000）：
- GPU/进程：GPU0 `24299/24576 MiB`，`frozen_lm_ums_10k` PID `10424`；GPU1 `24305/24576 MiB`，`frozen_lm_ums_30k` PID `19372`。两条训练仍活跃。
- 10k/GPU0 进度：`5137/10000`，约 `51%`；剩余 `4863` steps；最近约 `2.91s/step`，日志 ETA 约 `3:56:01`。`step 5000 val_loss=0.0392` 与 `step_5000.pt` 已确认，训练稳定继续推进。
- 30k/GPU1 进度：`2957/10000`，约 `30%`；剩余 `7043` steps；最近约 `2.99s/step`，日志 ETA 约 `5:50:58`。最近验证点仍为 `step 2500 val_loss=0.0412`；距离 `step 3000` validation/checkpoint 约 `43` steps，当前尚未进入 validation，也未产生 `step_3000` checkpoint。
- 当前目标剩余时间：整体仍由 GPU1 `30k` 决定，粗估约 `5.85h`。下一关键节点是 GPU1 `step 3000` validation/checkpoint，预计几分钟内触发。
- 失败检查与结果判断：两个日志未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`PermissionError` 或退出失败；GPU1 速度稳定在约 `3.0s/step`，10k 继续训练，阶段性结果仍符合预期。无新增 failure case。

2026-06-27 18:19 运行巡检（GPU1 距 step 3000 约 17 steps）：
- GPU/进程：GPU0 `24299/24576 MiB`，`frozen_lm_ums_10k` PID `10424`；GPU1 `24305/24576 MiB`，`frozen_lm_ums_30k` PID `19372`。两条训练仍活跃。
- 10k/GPU0 进度：`5165/10000`，约 `52%`；剩余 `4835` steps；最近约 `2.91s/step`，日志 ETA 约 `3:54:10`。`step 5000 val_loss=0.0392` 与 `step_5000.pt` 已确认，训练稳定继续推进。
- 30k/GPU1 进度：`2983/10000`，约 `30%`；剩余 `7017` steps；最近约 `3.13s/step`，日志 ETA 约 `6:05:49`。最近验证点仍为 `step 2500 val_loss=0.0412`；距离 `step 3000` validation/checkpoint 约 `17` steps，当前尚未进入 validation，也未产生 `step_3000` checkpoint。
- 当前目标剩余时间：整体仍由 GPU1 `30k` 决定，粗估约 `6.1h`。下一关键节点是 GPU1 `step 3000` validation/checkpoint，预计很快触发。
- 失败检查与结果判断：两个日志未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`PermissionError` 或退出失败；GPU1 训练 loss 正常波动，10k 继续训练，阶段性结果仍符合预期。无新增 failure case。

2026-06-27 18:20 运行巡检（GPU1 进入 step 3000 validation）：
- GPU/进程：GPU0 `24299/24576 MiB`，`frozen_lm_ums_10k` PID `10424`；GPU1 `24305/24576 MiB`，`frozen_lm_ums_30k` PID `19372`。两条训练仍活跃。
- 10k/GPU0 进度：`5195/10000`，约 `52%`；剩余 `4805` steps；最近约 `2.97s/step`，日志 ETA 约 `3:57:29`。`step 5000 val_loss=0.0392` 与 `step_5000.pt` 已确认，训练稳定继续推进。
- 30k/GPU1 进度：已触发 `step 3000` validation，当前 validation 约 `236/250` batches；最近完整验证点仍为 `step 2500 val_loss=0.0412`。`step_3000` checkpoint 尚未生成，需等待 validation 完成后复核。
- 当前目标剩余时间：整体仍由 GPU1 `30k` 决定；validation 完成后将恢复训练并更新 ETA。下一关键检查项是 `step 3000 val_loss`、`best.pt` 是否更新，以及 `step_3000.pt` 是否落盘。
- 失败检查与结果判断：两个日志未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`PermissionError` 或退出失败；GPU1 当前 validation 正常推进，10k 继续训练，阶段性结果仍符合预期。无新增 failure case。
2026-06-27 18:23 关键 checkpoint 记录（GPU1 30k step 3000 完成）：
- GPU/进程：GPU0 `24299/24576 MiB`、util `71%`，运行 `frozen_lm_ums_10k` PID `10424`；GPU1 `24305/24576 MiB`、util `59%`，运行 `frozen_lm_ums_30k` PID `19372`。两条训练均仍活跃；另有一个只读监控 PowerShell 进程在等待下一次采样。
- 10k/GPU0 进度：`5245/10000`，约 `52%`；剩余 `4755` steps；当前日志 ETA 约 `3:50:48`，最近速度约 `2.91s/step`。最新验证点为 `step 5000 val_loss=0.0392`，`outputs/data_scaling/frozen_lm_ums_10k/checkpoints/step_5000.pt` 已确认落盘。
- 30k/GPU1 进度：`step 3000` validation 已完成，日志记录 `Step 3000: val_loss = 0.0391`；`outputs/data_scaling/frozen_lm_ums_30k/checkpoints/step_3000.pt` 已确认落盘，随后训练继续到约 `3045/10000`，约 `30%`；剩余 `6955` steps；当前日志 ETA 约 `5:50:10`，最近速度约 `3.02s/step`。
- 当前目标剩余时间：整体仍由 GPU1 `30k` 决定，按当前日志 ETA 粗估约 `5.8-6.0h` 到训练末尾，之后还需要 final validation/checkpoint 写盘、结果文件复核和文档汇总；若速度保持稳定，当前 data-scaling 阶段保守估计约 `6-7h` 完成。
- 结果判断：10k 的验证损失从 `0.0401` 到 `0.0392`，30k 从 `0.0412` 到 `0.0391`，都在小幅改善；两条日志未发现 `Traceback`、`RuntimeError`、`CUDA out of memory`、`PermissionError` 或退出失败，阶段性结果符合预期。
- Case study：本次为正常 checkpoint 里程碑，无新增失败案例；继续按 20 分钟巡检，若出现失败会立即归档到 `History/` 并总结。
