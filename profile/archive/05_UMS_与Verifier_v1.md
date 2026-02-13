# UMS + Programmatic Verifier + Failure Taxonomy（v0.x）

> 本文档合并了原 `06_UMS_Schema.md` 与 `04_Verifier_与FailureTaxonomy.md`：把 **UMS-JSON（结构化答案）** 和 **programmatic verifier（可验证监督）** 放在同一处，便于实现与写作对齐。

## 0. 三条红线（审稿与工程都靠它们）

1. **schema 统一，但 coverage 不统一**：不同数据集可提供的字段不同，缺失必须显式写 `null`（不要省略 key）。
2. **`null` 是“不可答”的正式答案**：不能硬填；用 `answerability/uncertainty` 把不可答变成可训练、可统计的信号。
3. **verifier 不是清洗器**：它输出 `role`，决定样本进入哪些 loss（训练分流口径见 `06_训练目标与SamplePolicy.md`）。

---

## 1. UMS（Unified Medical Schema）：统一结构化答案

### 1.1 目标

把“医学视觉事实”落在一个 **可程序验证** 的 JSON 上：
- 训练时：teacher forcing / constrained decoding 可以严格对齐字段与格式
- 评测时：schema violation / clinical hard errors 可以直接统计
- 写作时：Answerability、训练/推理一致性、可审计性 都有落点

### 1.2 设计原则（必须遵守）

1. **schema 统一，但 coverage 不统一**：靠 coverage 表承认差异，而不是靠“省略字段”掩盖差异。
2. **`null/missing` 合法**：不可判定字段必须显式 `null`（或 enum 的 `"uncertain"`），并用 answerability/uncertainty 表达原因。
3. **provenance 可追溯**：每个字段要能回溯来源（dataset/derived/...）与 verifier 版本。
4. **能算就算，不让 LLM 编**：geometry、spacing、measurement 走 deterministic compiler（规则/脚本），不是让 LLM“生成真值”。

### 1.3 顶层字段（v0.2 的最小可用集）

UMS 顶层建议保持稳定（数据集特有字段放 `extensions`）：
- `modality`：`CT | CXR | MRI | ...`
- `anatomy`：器官/部位列表
- `findings`：字典结构（每个 finding 至少有 `state`，可选 `score`）
- `laterality`：`left | right | bilateral | null`
- `study_view`：`AP | PA | LAT | null`
- `geometry`：`bbox/mask/keypoints`（允许为 `null`）
- `measurements`：字典结构（每个 measurement 至少 `value/unit/state/spacing_source`）
- `answerability`：字段级 `true/false`
- `uncertainty`：字段级（`0~1` / boolean / `null` 均可，v1.0 可先简单）
- `provenance`：字段级溯源（label_source/verifier_version/failure_type 等）
- `verifier`：样本级结果（pass/failure_type/confidence/role）
- `extensions`：数据集特有扩展（不破坏 UMS core）

### 1.4 `null` 字段怎么处理（你现在最关心的部分）

#### (A) 什么时候用 `null`

1. **数据集中没有该字段**（coverage 不支持）：例如 NIH 没有稳定的 `study_view`。
2. **该字段在该样本不可答**：例如 CXR 的 laterality 在多数情况下无法确定，或存在冲突证据。
3. **该字段需要确定性输入但缺失**：例如 measurement 需要 spacing，但 DICOM/metadata 缺失。

#### (B) `null` 时必须同步写什么

- `answerability.<field> = false`
- `uncertainty.<field>`：建议用 `1.0`（或 `true`）代表“完全不确定/不可答”
- `provenance.<field>`：写清 `label_source` 与 `failure_type/ambiguity`（若适用）

> 核心思想：把“缺失/不可答”变成 **可监督目标**（模型学会输出 `null/uncertain`），并且可进入 risk–coverage 报告。

#### (C) 训练/推理如何消费这些 `null`

- 训练：对 `answerable=false` 的字段，目标 token 固定为 `null` 或 `"uncertain"`；不进入 ranking/对比等“真假”损失（详见 `06_训练目标与SamplePolicy.md`）。
- 推理：若 `answerability head` 预测 `â_f < τ`，强制输出 `null/uncertain`，并在可靠性指标中统计覆盖-风险曲线（risk–coverage）。

### 1.5 机器可读 schema 与示例在哪里

- JSON Schema：`schema/ums_v0_2.schema.json`
- coverage 表：`schema/coverage_table.md`
- examples：`schema/examples/`

---

## 2. Dataset → UMS：确定性编译（deterministic compiler）

> 这一步的原则是：**真值来自数据集/规则/几何计算**，而不是来自 LLM。

常见编译来源（示意）：
- AMOS / KiTS（CT 分割）：`mask + spacing` → measurement（面积/体积/最长径等），并填 `spacing_source`
- CheXpert（CXR 弱标签）：`present/absent/uncertain` → `findings.<name>.state`
- NIH（CXR 标签）：需要与 CheXpert 做标签对齐子集（映射表待定），无法对齐的字段 → `null + answerability=false`

编译产物必须：
- 满足 JSON Schema（字段齐全；允许为 `null`）
- 在 `provenance` 里标明字段来源（dataset/derived/...）

---

## 3. Programmatic Verifier：让 UMS 变成“可验证监督”

### 3.1 verifier 输出（样本级）

verifier 对每条 UMS-JSON 输出：
- `pass/fail`
- `failure_type`（失败类型）
- `confidence`
- `role`：`positive / abstain / negative-only / drop`

> 论文里必须写清楚：`role` 会改变样本进入哪些 loss（否则容易被看成纯工程清洗）。

### 3.2 最小规则集（v0：先做 8–11 条）

1. **schema 校验**：合法 JSON 且满足 JSON Schema（失败→`schema_violation`）。
2. **missing-field policy**：不允许省略顶层 key；缺失→`missing_field`（即使业务上“可省略”，在 UMS 里也不省略）。
3. **enum/type 校验**：`laterality/study_view/state/unit` 等必须在允许集合中（失败→`schema_violation` 或更细分）。
4. **answerability 一致性（v1.0 必做）**：若 `answerability.<field>=false`，则该字段必须输出为 schema 允许的 abstain 值（v1.0 默认 `null`），避免把“不可判定”混入枚举空间。
5. **geometry consistency**：bbox/mask 坐标合法；若同时有 bbox+mask，要求 bbox ⊇ mask（失败→`geometry_mismatch`）。
6. **measurement unit**：单位合法（mm/cm/mm2...）（失败→`unit_error`）。
7. **measurement range**：数值在合理区间（失败→`range_error`）。
8. **spacing source presence（CT）**：需要 measurement 时必须有 `spacing_source`（失败→`metadata_missing`）。
9. **uncertainty consistency**：数据集标签含 uncertain 时，`findings` 与 `uncertainty` 映射需一致（失败→`ambiguity` 或自定义 inconsistency）。
10. **prompt-schema consistency**：prompt 询问的字段必须在 UMS 中出现；不可答时应输出 `null/uncertain`（失败→`missing_field`/`schema_violation`/`ambiguity`）。
11. **leakage check**：检测 prompt 或字段是否直接泄漏答案（失败→`leakage_suspected`，并 `role=drop`）。

> 重要澄清：`confusable pairs`（容易混淆的 finding 对）只用于构造 hard negatives / `L_rank`，不作为 verifier 的“互斥规则”（医学上常见共存，强行互斥会错杀样本）。

可选增强（后续版本）：
- laterality/方位 sanity（例如 ROI 中心落在左右半区的一致性检查）：只能作为弱规则；冲突应转为 `ambiguity`，不要当作 hard error。

### 3.3 Failure taxonomy（建议 v0 覆盖）

- `schema_violation`：JSON/字段/enum 不合法
- `missing_field`：顶层 key 缺失或关键子字段缺失（违反“显式 null”政策）
- `geometry_mismatch`：bbox/mask 不一致或坐标非法
- `unit_error`：measurement unit 不合法
- `range_error`：measurement 超出合理范围
- `metadata_missing`：measurement 所需 spacing/metadata 缺失
- `laterality_error` / `view_error`：仅在“数据集/规则能确定”前提下才算 error；否则应归为 `ambiguity`
- `ambiguity`：信息不足/冲突，按不可答处理（`null + answerability=false`）
- `leakage_suspected`：怀疑答案泄漏
- `verifier_timeout`：如实现中有超时

### 3.4 role 分流（训练口径）

- **positive**：pass 且字段可答 → 进入 `L_tok + L_rank + (可选)L_vdep`
- **abstain**：pass 但字段不可答/歧义 → 只做 `L_tok`（目标输出 `null/uncertain`）
- **negative-only**：内容错但格式对（unit/laterality/×10 等）→ 不喂正确答案，只作为 hard negative 做 `L_rank`
- **drop**：schema_violation/leakage 等 → 丢弃，避免污染

---

## 4. 必须产出的报告（写进 repo/论文）

- `reports/verifier_pass_rate.csv`：总体/分数据集/分 failure type 的通过率
- `reports/failure_taxonomy.csv`：failure type 计数与占比
- `figs/failure_taxonomy_bar.png`：柱状图（正文/附录）
- （可选）医生抽检：`reports/verifier_precision_recall.md`（200–500 例即可）

---

## 5. 本文档与其它文档的关系（导航）

- 方法主线与前向路径：`02_方法.md`
- 数据集与字段覆盖表：`03_数据集.md`
- loss / sample policy / answerability head：`06_训练目标与SamplePolicy.md`
- 实验矩阵与可靠性套件：`07_实验矩阵与评估协议.md`
