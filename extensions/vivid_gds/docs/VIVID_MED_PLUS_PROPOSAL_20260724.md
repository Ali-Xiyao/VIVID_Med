# VIVID-Med+: 基于冻结生成式 LLM 的生成—判别双路径结构化蒸馏

## 1. 最终建议

新论文不再围绕 SPD、query decomposition 或 attention orthogonality 展开。
保留冻结 Qwen3.5-2B、UMS 结构化监督、answerability-aware masking、稳定的
prefix4 projector 和 ViT-only deployment，新增唯一主模块：

> **UMS Schema Readout Bridge (UMS-SRB)**。

方法名固定为：

> **VIVID-GDS: Generative–Discriminative Structured Distillation**。

## 2. 科学问题

原 VIVID 通过可微的冻结 Qwen 生成 UMS，但更低的生成 token loss 不一定
产生更可迁移的 ViT。VIVID-GDS 增加一条绕过 projector 和 Qwen 的判别路径，
直接从最终 ViT CLS 表征预测同一批 UMS 字段状态。

## 3. 方法

生成路径保持不变：

```text
image -> ViT -> prefix4 projector -> frozen Qwen3.5-2B -> hard UMS
```

判别路径为：

```text
ViT CLS -> LayerNorm -> Linear(768,512) -> GELU -> Dropout(0.1)
        -> Linear(512,12*3)
```

三个状态为 `present`、`absent`、`uncertain`。`missing` 不是类别，而是
监督 mask。两条路径必须使用同一行 hard UMS 中完全相同的字段集合，
因此 schema head 不会获得生成路径看不到的标签。

联合损失：

\[
L=L_{\mathrm{gen}}+\lambda_s(t)L_{\mathrm{schema}},
\qquad
\lambda_s(t)=0.5\min(1,t/500).
\]

`L_schema` 先对每个 finding 的可观察样本求平均，再在当前 batch 有监督的
findings 上求平均，防止高覆盖字段支配损失。首轮只允许
`lambda_schema=0.5`；`0.25` 仅是生存门通过后的预注册敏感性。

预训练结束后删除 Qwen、prefix projector 和 schema head，只部署 ViT。

## 4. Stage A 核心生存实验

四个实验使用相同 20k MIMIC study 锁、患者划分、ViT-B/16 初始化、3000
optimizer steps、seed 0、增强、checkpoint 规则和 CheXpert
expert-development probe：

| ID | 监督 | Qwen | UMS | Schema head |
|---|---|---:|---:|---:|
| A0 | direct schema classification | 否 | 标签形式 | 是 |
| A1 | deterministic free-text generation | 是 | 否 | 否 |
| A2 | hard-UMS generation, prefix4 | 是 | 是 | 否 |
| A3 | VIVID-GDS | 是 | 是 | 是 |

A2 复用已经冻结的 strict prefix4 checkpoint；A0、A1、A3 从同一 ViT
初始化重新训练。A1 只把 A2 每行已选字段转换为冻结的自由文本模板，
不读取原始报告、不改变字段身份，也不把 missing 转成 absent。

### Gate A2 > A1：UMS 是否有效

- macro AUROC delta >= +0.005；
- macro AUPRC delta >= -0.005；
- 五个 findings 至少四个 AUROC delta 非负；
- 最多一个 finding 下降超过 0.02。

### Gate A2 > A0：冻结 Qwen 是否有增益

- macro AUROC delta >= +0.003；
- macro AUPRC delta >= -0.005；
- 若内部未通过，只能记录为未证实；不得用已经暴露的 test 事后救援。

原 proposal 中“内部持平后用外部/low-data 救援”的条款被收紧：在 Stage A
生存门中，外部数据不参与身份选择。

### Gate A3 > A2：新模块是否有效

- macro AUROC delta >= +0.005；
- macro AUPRC delta >= -0.005；
- 五个 findings 至少四个 AUROC delta 非负；
- 最多一个 finding 下降超过 0.02；
- 全部指标来自各自预注册 checkpoint。

三道门全部通过后，才允许进入三 seed、全量 MIMIC、low-data、NIH 和
PadChest。VinDr 只允许标注为已暴露的 retrospective replication；
CheXlocalize test 保持关闭。

## 5. 机制控制

只有 Stage A 生存后才运行：

1. pretrained Qwen 对同架构 random-init Qwen；
2. real UMS 对字段名语义置换 UMS；
3. correct answerability mask 对 null-supervised 和 null-as-absent。

这些控制用于区分语言预训练知识、UMS 语义、网络规模和 missing 处理。

## 6. Stage B 与最终实验

Stage B 首先在 20k 上运行 A0–A3 的 seeds 0/1/2。A3 相对 A2 需要至少
2/3 seeds 为正、平均 delta AUROC >= +0.005，并且患者级配对 bootstrap
95% CI 下界大于 0。之后才运行 full MIMIC；多机构
MIMIC+CheXpert-Plus 必须先排除 CheXpert expert val/test 和
CheXlocalize val/test，并完成 patient/image hash 去重。

最终评价矩阵包括 linear probe、full fine-tuning、1/5/10/25/100% low-data、
head/medium/tail、NLL/ECE/Brier、AP/PA、sex、age 和 institution 亚组。

## 7. 论文主表

1. 数据集与独立性；
2. A0/A1/A2/A3 核心机制；
3. pretrained/random Qwen 与 real/permuted UMS；
4. 外部泛化；
5. low-data；
6. synchronized fields 与 answerability 消融；
7. 数据规模；
8. 训练与部署效率。

## 8. 停止规则

以下任一科学条件成立即停止扩张：

1. A2 不优于 A1；
2. A2 不优于 A0；
3. A3 不优于 A2；
4. 增益只来自一个 finding；
5. 需要修改门槛或 checkpoint 规则；
6. 需要 Qwen3.5-4B/9B 才能产生增益；
7. 需要查看外部 test 才能选择 `lambda_schema`；
8. 三 seed 方向不一致。

失败后不得加入 contrastive loss、graph、anatomy query、新 projector、
SPD、consistency network 或 causal selector。唯一方法身份保持
VIVID-GDS。
