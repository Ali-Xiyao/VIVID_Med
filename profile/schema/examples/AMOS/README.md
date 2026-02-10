# AMOS 示例（UMS-JSON v0.2）

目标：提供 **10–20 条**“真实样本 → UMS 编译后 JSON”的示例，用于：
- 验证 `mask/bbox/measurement` 的确定性编译口径
- 覆盖 `answerability=false` 的典型场景（laterality/view 不可答）

建议覆盖：
- 有 organ mask 的样本（`geometry.mask` 非空）
- 由 `mask + spacing` 计算 measurement 的样本
- spacing 缺失 → `metadata_missing` 的样本（verifier 应能抓到）

