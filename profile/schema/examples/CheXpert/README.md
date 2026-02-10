# CheXpert 示例（UMS-JSON v0.2）

目标：提供 **10–20 条**“真实样本 → UMS 编译后 JSON”的示例，用于展示：
- findings 的 `present/absent/uncertain/null` 映射
- `study_view` 有/无元数据时的口径
- laterality 不可判定时的 `null + answerability=false`

建议覆盖：
- positive 标签映射为 `state=present` 的样本
- negative 标签映射为 `state=absent` 的样本
- uncertain 标签映射为 `state=uncertain` 的样本
- study_view 缺失的样本（显式 `null` + answerability）
- laterality 不可判定 → `null` + `role=abstain`

