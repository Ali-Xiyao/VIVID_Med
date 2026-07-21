# PIPM / FiVPU Official Fisher 方向—强度解耦诊断实验卡

> 项目：IndexMemory / Provenance-Indexed Parametric Memory（PIPM）<br>
> 分支：`codex/fivpu-implementation`<br>
> 基线结果提交：`679c24a`<br>
> 当前状态：`official Tiny1 = FAIL_OFFICIAL_COORDINATE`<br>
> 本卡性质：**单样本、诊断-only、document-first**。不解锁 checkpoint、Tiny64、训练、E5 或正式实验。<br>
> 建议实现文件：`scripts/fivpu/run_tiny1_fisher_direction_calibration.py`<br>
> 建议测试文件：`tests/fivpu/test_tiny1_fisher_direction_calibration.py`

---

## 0. 一页结论

当前证据已经排除工程链路、梯度符号、Qwen hook、注入可见性和线性求解器数值失败。正式 S32/K8 FiVPU 更新失败的剩余问题是：

1. **强度过大**：正式 materialized update norm 为 `26.0951`；已通过的 direct projected control 只在 `alpha=10` 通过。
2. **Fisher 旋转明显**：正式更新与负 projected 方向余弦为 `0.82303`，夹角约 `34.61°`；相对该方向的正交残差范数为 `14.822`，约占正式更新平方能量的 `32.26%`。
3. **目标函数错位仍存在**：之前通过的 projected 配置往往不是靠提高 gold，而是更强地压低主要负候选。因此，后续正式强度校准不能只用 gold CE，必须使用与 T 一致的候选身份/候选 margin，且只能在与 scored Tiny1 target 隔离的 calibration facts 上拟合。

本轮只回答一个问题：

> 固定完全相同的 S32/K8 support basis 和 Fisher，先把方向单位化，再用同一 alpha 网格比较 normalized Fisher 方向与 direct projected 方向。正式失败究竟主要是 Fisher 方向旋转，还是未校准的 raw step magnitude？

---

## 1. 已冻结的基线事实

### 1.1 Official Tiny1 结果

| 项目 | 值 |
|---|---:|
| T candidate identity EM | 0.0 |
| F gold answer EM | 0.0 |
| D candidate identity EM | 0.0 |
| T predicted candidate | candidate-2 |
| gold rank | 4 |
| gold margin | -2.9074530819 |
| no-update gold CE | 9.1523199671 |
| official gold CE | 14.7164468136 |
| rho | 0.2652358780 |
| rho² | 0.0703500710 |
| official update norm | 26.0950948824 |
| signed alpha equivalent along projected direction | 21.4770509633 |
| cosine with negative projected direction | 0.8230301925 |
| orthogonal residual norm | 14.8219519240 |
| Fisher system condition number | 10.9368721984 |
| solve relative residual | 2.2790476923e-16 |

### 1.2 必须保持不变的配置

| 轴 | 冻结值 |
|---|---|
| support profile | `stratified_relation_template_raw_s32_k8_v1` |
| support selection | `stratified_relation_template` |
| SVD preprocessing | `raw` |
| support count S | 32 |
| basis dimension K | 8 |
| layers | `[0, 1]` |
| tangent rank | 2 |
| damping | `1e-3` |
| scale | `1.0` |
| seed | `7` |
| target row | official Tiny1 row 0 |
| prompt/candidates/evaluator | 与 official Tiny1 完全一致 |
| primary protocol | T only |

本轮禁止改变：supports、basis、Fisher、damping、scale、layers、rank、target、candidate set、scorer 或 gate。

---

## 2. 方向定义

令：

- `U ∈ R^(P×K)`：固定 S32/K8 正交 basis；
- `g ∈ R^P`：official Tiny1 target gradient；
- `z = Uᵀg`；
- `F ∈ R^(K×K)`：固定 projected Fisher；
- `λ = 1e-3`；
- `c_off`：production closed-form coordinate；
- `u_off = U c_off`：production materialized update。

### 2.1 Direct projected unit direction

```text
q_projected = -z / ||Uz||
d_projected = U q_projected
```

由于 U 列正交，`||d_projected|| = 1`。

### 2.2 Normalized production Fisher direction

不要重新实现一个近似 solver；直接从 production artifact 中读取 `c_off`：

```text
q_fisher = c_off / ||U c_off||
d_fisher = U q_fisher
```

`c_off` 已经包含负号，所以 sweep 时使用：

```text
coordinate(alpha) = alpha * q_fisher
```

不要再额外乘负号。

### 2.3 重要区分

- `21.4771` 是 official update 在 `d_projected` 上的平行分量，不是 normalized Fisher 方向的 official alpha。
- normalized Fisher 方向重建 official update 时，alpha 应为 `||u_off|| = 26.0951`。
- 本轮 scored alpha grid 仍冻结到 `10`；`26.0951` 只做代数重建断言，不作为新的 target-scored sweep 点。

---

## 3. 冻结 alpha 网格

使用现有 `R3A_ALPHA_GRID`：

```python
ALPHA_GRID = (
    0.0,
    1.0e-4,
    3.0e-4,
    1.0e-3,
    3.0e-3,
    1.0e-2,
    3.0e-2,
    1.0e-1,
    3.0e-1,
    1.0,
    3.0,
    10.0,
)
```

总计：

```text
2 directions × 12 alpha × 1 T record = 24 T records
```

本轮不跑 F/D，因为它们不能回答方向—强度解耦问题，而且 official T hard gate 已经失败。

---

## 4. 实现要求

### 4.1 建议新增通用 artifact helper

避免复用带隐式负号的 `_projected_artifact`，新增一个符号透明的 helper：

```python
def _unit_coordinate_artifact(reference, *, unit_coordinate, alpha):
    # assert exact shape [K]
    # assert ||U @ unit_coordinate|| == 1
    # coordinates are all zero except row 0
    # coordinates[0] = alpha * unit_coordinate
    # active_mask only activates row 0
    ...
```

### 4.2 构建顺序

```text
1. metadata-only freeze exact S32/K8 supports
2. build probe/frame with frozen layers/rank/seed
3. collect support gradients
4. build exact production Fisher basis
5. collect target gradient / production coordinates
6. derive q_projected and q_fisher
7. perform algebraic invariants
8. run paired T grid
9. persist evidence and classify
```

### 4.3 必须保存的文件

```text
direction_definitions.json
fisher_eigen_diagnostics.json
direction_grid.json
records.json
metrics.json
run_manifest.json
summary.md
support_profile.json
```

### 4.4 每个 direction × alpha 必须记录

| 字段 | 说明 |
|---|---|
| direction | `projected` / `fisher_normalized` |
| alpha | 单位化后的 Euclidean update norm |
| materialized_update_norm | 应等于 alpha |
| target_ce | gold token CE |
| gold_mean_logp | T scorer gold score |
| best_negative_candidate_id | 当前最佳负候选身份 |
| best_negative_mean_logp | 当前最佳负候选分数 |
| margin | gold - max negative |
| gold_rank | gold 排名 |
| predicted_candidate_id | T 输出 |
| tie | 是否并列 |
| t_candidate_identity | official T identity |
| candidate_deltas | 每个候选相对 alpha=0 的分数变化 |

### 4.5 必须增加的 eigenspace 诊断

对 `F = V diag(μ) Vᵀ`，保存 K=8 的逐轴表：

| 字段 | 说明 |
|---|---|
| eigen_index | 0..7 |
| fisher_eigenvalue | μ_i |
| inverse_gain | `1/(μ_i+λ)` |
| target_component | `(Vᵀz)_i` |
| projected_energy_fraction | `z_i² / ||z||²` |
| fisher_direction_energy_fraction | `(z_i/(μ_i+λ))² / sum_j(...)` |

这张表用于判断 Fisher 是否因放大低曲率轴而产生有害旋转。它不增加任何模型 forward。

---

## 5. RED/GREEN 测试要求

### 5.1 纯 CPU / 代数测试

| 测试 | 通过条件 |
|---|---|
| exact profile freeze | S=32、K=8、raw、layers/rank/damping/scale/seed 全匹配 |
| basis orthonormal | `UᵀU ≈ I` |
| projected direction unit norm | `||U q_projected|| ≈ 1` |
| Fisher direction unit norm | `||U q_fisher|| ≈ 1` |
| sign correctness | `gᵀ(U q_projected) < 0` 且 `gᵀ(U q_fisher) < 0` |
| official reconstruction | `26.0950948824 * q_fisher ≈ c_off` |
| direction cosine | `cos(Uq_fisher, Uq_projected) ≈ 0.8230301925` |
| active row | 只有 row 0 active；其他 coordinates 严格为 0 |
| no mutation | reference basis、Fisher、production artifact 不被修改 |

### 5.2 GPU 行为测试

| 测试 | 通过条件 |
|---|---|
| alpha=0 identity | 两个方向的 candidate scores 逐候选一致 |
| local CE sign | 至少一个最小非零 alpha 上，两方向 target CE 均低于 baseline；否则实现回归 |
| projected reproduction | projected alpha=10 重新得到唯一 gold top-1；否则停止并查 evidence mismatch |
| record cardinality | 恰好 24 条 T records |
| no F/D | manifest 只含 T |
| no persistence/training | optimizer/checkpoint/Tiny64/formal 全为 false |

---

## 6. 主判定规则

定义：

```text
P_pass = projected direction 是否存在 alpha>0 使 T identity=true 且 tie=false
F_pass = normalized Fisher direction 是否存在 alpha>0 使 T identity=true 且 tie=false
```

| P_pass | F_pass | 判定 | 下一步 |
|---|---|---|---|
| false | 任意 | `FAIL_CONTROL_REPRODUCTION` | 停止；核对 basis hash、supports、model fingerprint、seed、scorer、target |
| true | false | `FAIL_FISHER_DIRECTION_ROTATION` | 标量缩放无法救该 Fisher 方向；进入 identity/shrinkage curvature 方案，不做 blind scale sweep |
| true | true | `PASS_FISHER_DIRECTION_MAGNITUDE_BLOCKER` | Fisher 方向具备 T 容量；下一张卡做 target-isolated magnitude calibration |
| false | true | `UNEXPECTED_FISHER_ONLY_PASS` | 与既有 projected 证据冲突；先做重现实验，不解锁后续 |

### 6.1 次级解释

| 现象 | 解释 |
|---|---|
| Fisher 仅在 alpha=10 通过，official norm 26.1 失败 | 主要是 raw magnitude 未校准 |
| Fisher 所有 alpha 都失败，projected alpha=10 通过 | Fisher rotation 是主 blocker |
| 两者 T 通过点都伴随 gold CE 恶化 | CE 与 candidate ranking 一阶错位；正式校准必须使用 T-like margin/IMW |
| Fisher 小 alpha CE 降，但大 alpha CE 升 | 方向局部正确，official raw step 已离开局部二次近似有效区 |
| Fisher margin 在 alpha→0 的局部导数为负 | 该方向局部损害排序；若大 alpha 才通过，则通过依赖强非线性，不宜直接正式化 |

---

## 7. 结果空表

### 7.1 Direction summary

| Direction | Cosine vs projected | First CE-lowering alpha | Best-CE alpha | Best-margin alpha | T-passing alphas | First T-pass alpha | Verdict |
|---|---:|---:|---:|---:|---|---:|---|
| projected | 1.0 |  |  |  |  |  |  |
| fisher_normalized | 0.8230301925 expected |  |  |  |  |  |  |

### 7.2 Paired T grid

| alpha | Projected CE | Projected margin | Projected pred | Projected T | Fisher CE | Fisher margin | Fisher pred | Fisher T |
|---:|---:|---:|---|---|---:|---:|---|---|
| 0 |  |  |  |  |  |  |  |  |
| 0.0001 |  |  |  |  |  |  |  |  |
| 0.0003 |  |  |  |  |  |  |  |  |
| 0.001 |  |  |  |  |  |  |  |  |
| 0.003 |  |  |  |  |  |  |  |  |
| 0.01 |  |  |  |  |  |  |  |  |
| 0.03 |  |  |  |  |  |  |  |  |
| 0.1 |  |  |  |  |  |  |  |  |
| 0.3 |  |  |  |  |  |  |  |  |
| 1 |  |  |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |  |  |

### 7.3 Candidate delta at decisive alpha

| Direction | alpha | candidate_id | answer | score_baseline | score_after | delta | rank_after |
|---|---:|---|---|---:|---:|---:|---:|
| projected |  | candidate-0 / gold |  |  |  |  |  |
| projected |  | candidate-1 |  |  |  |  |  |
| projected |  | candidate-2 |  |  |  |  |  |
| projected |  | candidate-replace |  |  |  |  |  |
| fisher_normalized |  | candidate-0 / gold |  |  |  |  |  |
| fisher_normalized |  | candidate-1 |  |  |  |  |  |
| fisher_normalized |  | candidate-2 |  |  |  |  |  |
| fisher_normalized |  | candidate-replace |  |  |  |  |  |

### 7.4 Fisher eigenspace

| Axis | eigenvalue μ | inverse gain | target component | projected energy fraction | Fisher-direction energy fraction |
|---:|---:|---:|---:|---:|---:|
| 0 |  |  |  |  |  |
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |

---

## 8. 若 Fisher 方向通过：下一张卡的正式校准原则

本轮 alpha sweep 使用 scored Tiny1 target，因此只能证明方向容量，**不能把最佳 target alpha 直接写进 official 方法**。

推荐下一阶段使用 target-isolated、support-derived trust-region calibration：

1. 保留 exact S32/K8 basis construction；
2. 从不属于 Tiny1 target sources、且未进入 basis 的事实中冻结 calibration pool；
3. 把 calibration facts 逐个当 pseudo-target，生成与 official T 一致的 candidate set；
4. 对每个 pseudo-target 计算 raw Fisher coordinate；
5. 在 calibration pool 上预注册一个固定的 norm cap `τ` 或固定 scalar `η`；
6. 正式写入使用例如：

```text
c_raw = -(F + λI)^(-1) z
c_cal = min(1, τ / ||U c_raw||) * c_raw
```

7. `τ` 必须只由 calibration facts 决定，不能查看 official Tiny1 target 的 T、margin 或最佳 alpha；
8. 因 CE-versus-ranking 已明显错位，calibration metric 应优先是 unique candidate identity / candidate margin，而不是仅最小 gold CE。

只有 target-isolated calibration rule 被文档冻结后，才允许重新运行 unchanged official Tiny1。

---

## 9. 若 Fisher 方向失败：下一张卡的候选方案

若 normalized Fisher 在全部 alpha 上失败，而 projected alpha=10 可复现通过，则说明缩放无法修复方向。按最小改动顺序：

1. **Identity curvature control**：同一 U，令 projected Fisher 为 I；其方向与 direct projected 一致；
2. **Shrinkage Fisher**：

```text
F_beta = (1-beta) I + beta F
```

`beta` 必须在 target-isolated calibration facts 上冻结，不能在 Tiny1 target 上扫完后挑选；
3. 如果通过仍主要依靠负候选抑制而非 gold CE 改善，进入 pairwise/listwise candidate-margin 或 IMW write objective；
4. 仍不允许 Tiny64，直到 support-only、target-isolated 的正式 coordinate rule 通过 unchanged Tiny1。

---

## 10. Slurm preflight 与异常 step

当前曾观察到非本阶段创建的 `3066.13004 bash`。保持“不擅自取消”是正确的。下一次启动前先记录：

```bash
squeue -s -j 3066
scontrol show step 3066.13004
sstat -j 3066.13004 --format=JobID,AveCPU,AveRSS,MaxRSS,MaxVMSize
```

执行规则：

```text
- 若 step 仍存在且来源/用途不明确：不启动新 GPU step；先确认归属。
- 若只是无 GPU 占用的已知 shell：记录证据后可继续。
- 若占用 GPU 或运行未知命令：由 allocation owner 确认后决定，不自动 scancel。
```

---

## 11. 本轮硬停止规则

无论结果如何，本轮结束后都必须停止并写回文档。禁止自动继续：

```text
checkpoint replay
Tiny64
formal T/F/D
E5
训练或 optimizer
blind damping sweep
blind scale sweep
换 support/basis/layer/rank
把 target 最佳 alpha 写入 official 方法
```

---

## 12. 最终报告模板

| 项目 | 结果 |
|---|---|
| 执行 commit |  |
| allocation / node / step |  |
| exact support profile verified |  |
| basis/support hashes match |  |
| projected alpha=10 reproduced |  |
| normalized Fisher T-passing alphas |  |
| projected T-passing alphas |  |
| Fisher best CE alpha |  |
| Fisher best margin alpha |  |
| Fisher local CE direction valid |  |
| Fisher local margin derivative sign |  |
| Primary decision |  |
| Direction rotation blocker | yes / no / mixed |
| Magnitude blocker | yes / no / mixed |
| CE-ranking mismatch remains | yes / no |
| Next permitted card |  |
| checkpoint/Tiny64/formal executed | no |
