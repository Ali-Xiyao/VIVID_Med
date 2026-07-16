# UMS-JSON 示例目录（UMS v0.2）

本目录存放 **UMS-JSON（v0.2）** 的“数据集 → JSON”示例模板与样例占位，用于：
- 让 schema 不停留在概念层：审稿人能看到具体长什么样
- 让实现不走样：dataloader / verifier / 训练/评测都围绕同一对象操作

## 目录结构

- `AMOS/`：CT（分割/器官）示例模板与覆盖建议
- `CheXpert/`：CXR（多标签 findings）示例模板与覆盖建议

## 约束

- 所有示例都应满足 `../ums_v0_2.schema.json`
- 顶层字段齐全（允许为 `null`，但不要省略 key）

