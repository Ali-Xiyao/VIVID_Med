# 训练目标（Loss）与 Verifier-guided Sample Policy

## 1. 训练目标（围绕结构化 token 生成）

记 target JSON token 序列为 `y = (y1…yT)`，模型输出 `p(y_t | image, prompt, y_<t)`。

### 1.1 `L_tok`：Next-token prediction（主监督）

- 标准 cross-entropy，让模型生成 target JSON tokens。
- 对 `answerable=false` 的字段，target 固定为 `null` 或 `"uncertain"`（由 schema 统一约定）。

字段级 answerability（v1.0 口径，避免“不可判定”污染枚举空间）：

- UMS 的 `answerability` 是**字段级**（`answerability.<field> ∈ {true,false}`）。
- 训练与 verifier 共同遵循：若 `answerability.<field>=false`，则该字段在 JSON 中必须落到 schema 合法的 abstain 值（v1.0 默认 `null`）。
- 对 `findings` 这类“字典字段”，建议把 `<field>` 具体到单个 finding（例如 `findings.<name>.state` 或 `findings.<name>`），从而把“不可答”落实到可训练的 token 目标上。

### 1.2 `L_rank`：Hard-negative JSON ranking（拒绝“错得很像真的”）

对每条样本生成若干 hard negatives（格式合法但内容错），训练偏好约束（二选一）：

1. **margin ranking**：`score(y_true) > score(y_neg) + m`
2. **DPO/IPO 风格偏好损失**：只更新 ViT+projector（LLM 冻结）

### 1.3 `L_vdep`：Visual-dependence（anti-shortcut）

- prompt corruption：语义不变，措辞变化 → 输出应一致
- image swap：prompt 不变，图像变化 → 输出应显著变化（尤其 findings/measurements）

总损失（示意）：

`L = L_tok + λ_rank * L_rank + λ_vdep * L_vdep + λ_ans * L_ans`

> `λ_rank/λ_vdep/λ_ans`：待定（见 `11_待确认与留空项.md`）。

### 1.4 `L_ans`：Answerability head（强烈建议，v1.0 默认开启）

为每个字段 `f` 预测可答性 `â_f`（二分类），并最小化：

`L_ans = Σ_f BCE(â_f, a_f)`

推理时用 `â_f` 做 gating：当 `â_f < τ` 时，强制该字段输出 `null/uncertain`（或进入 abstain），并在 risk–coverage 中报告“更安全”的覆盖-风险曲线。

## 2. Hard negatives 的生成规则（v1.0：三层体系 + 在线挖掘）

### 2.1 规则级（cheap）

- laterality flip：left↔right
- unit swap：mm↔cm（以及 mm2↔cm2 如涉及面积）
- scale perturbation：×10 / ÷10
- range boundary：推到合理范围边界附近（用于 stress test）

### 2.2 语义级（关键）

- finding confusion：用“混淆对（confusable pairs）”替换 findings（例如 A↔B）

混淆对来源（两种方式二选一或结合）：
- 医学知识/手工列表（先少量即可）
- 先训练一个 **task-head baseline（结构化 heads）**，通过 confusion matrix 自动挖 top-confusable pairs

v1.0 口径（避免和 verifier 逻辑打架）：

- confusable pairs **仅用于 `L_rank` 的 hard negatives**（提高区分度），不作为 verifier 的互斥/一致性约束（临床上很多所见可共存，强行互斥会错杀样本）。
- 构造 negatives 时要保证“对当前样本来说确实是错的”（例如只在 `A=present 且 B≠present` 时做 A↔B 替换，避免制造“同样也可能为真”的噪声负例）。

### 2.3 在线挖掘（最像算法）

每轮从候选 negatives 中选择当前模型打分最高（最容易误判真）的进入 `L_rank`（hardness-aware sampling）。

建议报告 hardness（审稿人很吃）：
- negatives 平均 score/log-prob（越高越 hard）
- 加入语义级 + 在线挖掘后，hard error 与迁移性能变化（写进消融表一行）

要求：
- **必须通过 schema**（否则不算 hard negative）
- `failure_type` 要可追踪（用于统计“模型最容易被哪类假负样本骗过”）

## 3. Verifier-guided Sample Policy（决定你是不是“预处理论文”）

每条样本根据 verifier 输出分配训练角色（role）：

| verifier 结果 | 训练角色 | 进入的 loss |
|---|---|---|
| pass & answerable | positive | `L_tok + L_rank + (可选)L_vdep` |
| pass & unanswerable/ambiguity | abstain | `L_tok`（目标为 null/uncertain） |
| fail: unit/laterality/... | negative-only | `L_rank`（只作为 hard negatives） |
| fail: schema_violation/leakage | drop | 不参与训练 |

## 4. 训练循环（工程执行顺序）

建议按“先能跑通，再加约束”的顺序：

1. 只跑 `L_tok`（确保结构化生成可学）
2. 加 `L_rank`（学会拒绝“错但像”）
3. 加 `L_vdep`（逼迫看图）
4. 接入 sample policy（role 分流与 failure taxonomy 统计）
