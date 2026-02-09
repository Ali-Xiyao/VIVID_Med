# 14 V4 方案：Answerability Masked Training

## 1. 版本定义
- 版本名：`V4`
- 核心改动：在 `V3(state-aware token weighting)` 基础上，新增 **answerability-aware token masking**。
- 一句话目标：对 `answerability=false` 的 findings token 不再给予和可答字段相同的训练权重，缓解 `null` 主导训练。

## 2. 背景与问题
- 当前 CheXpert 30k 采样中，14 findings 的 `blank/NaN` 总体约 `70.47%`。
- 在 UMS 编译口径中，`blank/NaN -> state=null + answerability=false`。
- 训练时若仍强制“全量 14 finding 全监督”，会导致：
- 有效状态词监督过稀（`present/absent/uncertain` 梯度占比小）。
- 模型更容易学到“安全输出 null”而不是判别特征。

## 3. V4 训练原则（必须固定）
- 不改变 UMS 语义：`blank` 仍是 unknown，不强行改成 negative。
- 训练上做掩码：`answerability=false` 的 finding token 采用 `mask` 或极低权重。
- 保留双口径评估：
- 严格 UMS：`blank=unknown`，评估时 mask NaN。
- 对比口径：`blank->absent`，仅用于 baseline 公平对比，不作为 UMS 真值结论。

## 4. 具体落地规范
## 4.1 Loss 设计
- 令 `w_tok` 为 token 权重，默认 `1.0`。
- 当 token 属于 `answerability=false` 的 finding block 时，设置为：
- `w_tok = false_weight`（推荐起点 `0.0` 或 `0.1`）。
- 可答字段保持 `w_tok = true_weight`（推荐 `1.0`）。
- 与 V3 叠加：状态词加权仍保留（`present/uncertain/absent/null`）。

## 4.2 配置建议（V4）
- 建议新增配置段：
- `training.answerability_mask.enabled: true`
- `training.answerability_mask.false_weight: 0.0`
- `training.answerability_mask.true_weight: 1.0`
- `training.answerability_mask.scope: findings_only`
- `training.answerability_mask.keep_structural_tokens: true`

## 4.3 训练参数起点
- 先延续 V3 学习率与 batch 配置，不同时改太多变量。
- 首轮只改一个主变量：`false_weight`。
- 推荐扫描：`0.0 -> 0.1 -> 0.2`（每轮固定其他超参）。

## 5. 实验矩阵（最小可用）
- `V3`：当前 best 配置（作为主对照）。
- `V4-a`：`false_weight=0.0`（已完成，结果失败）。
- `V4-b`：`false_weight=0.1`（下一步）。
- `V4-c`：`false_weight=0.2`（下一步，可选）。

每组都输出：
- `json_success_rate`
- `pred_nan_rate`
- `macro_f1`
- `micro_f1`
- `per-label f1`

## 6. 成功判据
- 不牺牲结构化能力：`json_success_rate` 不显著下降。
- 分类有效性提升：`macro_f1` 高于 V3，且可复现。
- 高 missing 标签有改善：重点看 `Pleural Other/Fracture/Lung Lesion/Pneumonia`。

## 7. 失败判据与回滚策略
- 若 `json_success_rate` 明显下降：
- 优先把 `false_weight` 从 `0.0` 调回 `0.1` 或 `0.2`。
- 若仍不稳定，关闭 answerability mask 回到 V3。
- 若 `macro_f1` 无提升且 `pred_nan_rate` 不降：
- 保留 V3，转入“伪标签 + calibration 标注”路线。

## 7.1 V4-a 实测结果（2026-02-09，`false_weight=0.0`）
- 配置文件：`configs/cxr_chexpert_v4.yaml`
- 训练输出：`outputs/cxr_chexpert_v4/checkpoints/final.pt`
- 评估输出：`outputs/cxr_chexpert_v4/checkpoints/best.metrics.full.json`
- 指标：`json_success_rate=1.0`，`pred_nan_rate=1.0`，`macro_f1=0.0`，`micro_f1=0.2091`
- 现象：14 个标签 `per-label f1` 全为 0，模型几乎全部输出 `state=null`
- 结论：`false_weight=0.0` 过于激进，当前策略在本数据分布上不可用

## 8. 与伪标签/人工标注的衔接
- V4 是低成本先手方案，不依赖新标注即可验证有效性。
- 若 V4 提升有限，再接入：
- teacher 高置信伪标签（仅填充确定 blank）。
- 人工 calibration set（200-500 张）校准阈值。

## 9. 报告话术（可直接用）
- “V4 不是改语义，而是改训练梯度分配。我们保持 blank=unknown 的医学语义，用 answerability mask 让模型更关注可答字段，减少 70% 不可答字段对主任务的干扰。”

## 10. 文档联动
- 训练现状：`profle/10_训练结果记录.md`
- 汇报提纲：`profle/13_汇报清单.md`
- 数据覆盖：`profle/schema/coverage_table.md`

## 11. V4.1 执行指令（先短程再全量）
- 目标：先确认 `pred_nan_rate` 能否从 1.0 拉下来，再投入全量计算。
- 配置建议：
- `false_weight: 0.1`（第一轮）
- `false_weight: 0.2`（第二轮）
- 试验流程：
- 每轮 1000-2000 step + full eval
- 若 `macro_f1` 仍为 0 或 `pred_nan_rate` 仍接近 1.0，则停止该路线
