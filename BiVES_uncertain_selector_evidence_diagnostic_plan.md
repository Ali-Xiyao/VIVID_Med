# BiVES-CXR Uncertain 稳定性：下一步定向诊断与修复计划

**基于提交：** `b8bafd3`
**当前状态：**

- monotone bipolar conditional decoder 已修复 support 极性；
- transform-order 修复后 train/val uncertain 都保留 8 个灰度级；
- 同一 100-step gate 仍为 train accuracy `1.0`、val accuracy `0.75`；
- validation uncertain `|rho| = 0.8424913883`；
- mini-P0 与 formal run 继续暂停。

---

## 1. 先不要把根因直接定为 exact-K

当前失败可能来自三层：

1. **Evidence-field drift**：同一空间位置的 `e+ - e-` 在 train→val 后改变；
2. **Selector drift**：evidence field 尚可，但 hard top-K 选中了不同极性的 patch；
3. **Synthetic uncertain 定义不稳**：posterization 只是第三种纹理，不天然对应 bipolar evidence balance。

下一步应做零训练的 cross-replay，把三者分开。

---

## 2. 先修 replay 工具的四个诊断问题

### 2.1 直接读取真实 uncertain pair

不要默认从 train support 图重新生成 uncertain。

增加：

```bash
--train-uncertain-image path/to/train_uncertain.png
--val-uncertain-image path/to/val_uncertain.png
```

这两个文件应直接来自本轮 100-step gate 的 manifests。

### 2.2 对齐后再计算指标

当前旋转样本的以下指标必须在 affine 对齐后计算：

- gate-logit Spearman；
- signed-evidence correlation；
- total-evidence correlation；
- patch-token cosine；
- top-K Jaccard；
- Recall@1 patch。

优先使用 `torch.nn.functional.affine_grid` + `grid_sample`，不要用手写整数索引旋转作为主结果。

### 2.3 验证 PIL 旋转方向

用一个只有单个亮 patch 的 28×28 synthetic grid：

1. 按 PIL `rotate(+1°)` 生成像素结果；
2. 用 patch mapping 预测目标位置；
3. 检查二者方向一致。

图像坐标中 y 轴向下，手写标准笛卡尔旋转公式容易把正角度方向写反。

### 2.4 保存原始 patch arrays

每个 pair 保存：

```text
qwen_tokens_train / val
gate_logits_train / val
relaxed_gate_train / val
evidence_pos_train / val
evidence_neg_train / val
signed_evidence_train / val
total_evidence_train / val
hard_mask_train / val
valid_mask
grid_h / grid_w
affine_transform
```

建议保存 `.pt`，JSON 只保存摘要。

---

## 3. 决定性 2×2 selector/evidence cross-replay

定义：

\[
d_p=e_p^+-e_p^-,
\qquad
t_p=e_p^++e_p^-.
\]

令：

- \(M_t\)：train top-K mask；
- \(M_v\)：val top-K mask；
- \(W(M_t)\)：映射到 val 坐标的 train mask；
- \(W^{-1}(M_v)\)：映射到 train 坐标的 val mask。

计算四组：

\[
R_{tt}=\operatorname{Agg}(e_t,M_t),
\]

\[
R_{vv}=\operatorname{Agg}(e_v,M_v),
\]

\[
R_{vt}=\operatorname{Agg}(e_v,W(M_t)),
\]

\[
R_{tv}=\operatorname{Agg}(e_t,W^{-1}(M_v)).
\]

每组输出：

```text
E+
E-
Delta = E+ - E-
T = E+ + E-
rho = Delta / (T + eps)
decoded state probabilities
```

### 解释

| 现象 | 主要结论 |
|---|---|
| `rho_vv` 失败，`rho_vt` 恢复 | selector-dominant |
| `rho_vv` 与 `rho_vt` 都失败 | evidence-field / transform-definition dominant |
| `rho_tv` 也失败 | val mask 本身偏向正极性 patch |
| `rho_vt` 正常、soft pooling 正常、hard K 失败 | hard top-K boundary problem |

---

## 4. 用 Delta 而不是 rho 分解漂移

由于 rho 是比值，建议对 signed evidence mean 做加性分解：

\[
\Delta_{\rm selector}
=
\operatorname{mean}_{M_v}d_v
-
\operatorname{mean}_{W(M_t)}d_v,
\]

\[
\Delta_{\rm field}
=
\operatorname{mean}_{W(M_t)}d_v
-
\operatorname{mean}_{M_t}d_t.
\]

定义：

\[
f_{\rm selector}
=
\frac{
|\Delta_{\rm selector}|
}{
|\Delta_{\rm selector}|+
|\Delta_{\rm field}|+\epsilon
}.
\]

工程解释：

- \(f_{\rm selector}>0.7\)：selector-dominant；
- \(f_{\rm selector}<0.3\)：field-dominant；
- 中间：mixed。

同时对 total evidence \(T\) 做相同分解。

---

## 5. 必须补的 pooling counterfactual

对 train/val evidence field 离线计算：

1. own exact-K；
2. mapped opposite-view exact-K；
3. relaxed soft-topK；
4. uniform all-valid-patch mean；
5. `K ∈ {8,16,32,64}`。

### 判读

| 结果 | 判断 |
|---|---|
| soft/all-patch `rho≈0`，hard-K 很大 | hard selector |
| K 增大后 `rho→0` | small-budget sensitivity |
| 所有 K、soft、all 都很大 | evidence-field drift |
| K=16 的第 16/17 logit margin 极小 | near-tie boundary |
| gate rank稳定但 signed evidence 不稳定 | evidence head |
| Qwen tokens 已不稳定 | frozen backbone / transform |

---

## 6. Patch contribution decomposition

对 val evidence field，将 patch 分成：

- common：\(M_v\cap W(M_t)\)；
- added：\(M_v\setminus W(M_t)\)；
- removed：\(W(M_t)\setminus M_v\)。

分别输出：

```text
patch count
mean signed evidence
mean total evidence
sum signed contribution
max absolute contribution
top contributing patch indices
```

这能回答：

> `rho=0.842` 是由少数 swapped patches 造成，还是整张 evidence field 都偏正？

---

## 7. 先修 gate 设计，不要立即修模型

当前 synthetic uncertain 是 posterization。它不天然表达：

\[
E^+\approx E^-.
\]

如果 fixed-mask replay 仍失败，优先重做 engineering uncertain：

### 推荐：spatial bipolar mixture

构造大尺度、等面积的 support-like 与 contradict-like区域，例如 2×2 tile：

```text
support-like      contradict-like
contradict-like   support-like
```

要求：

- tile 宽高明显大于一个 patch；
- 两种 cue 面积相等；
- train/val 只加 cue-preserving 弱变换；
- 验证变换前后仍严格保持两类区域面积；
- 可生成已知 positive/negative region masks。

这个 gate 更直接检验：

- gate 是否同时选择两种相关证据；
- evidence head 是否正确赋予双极性；
- uncertain 是否由正负证据平衡产生。

---

## 8. 若确定是 selector-dominant，只做一个有界修复

不改 decoder、K、backbone 或 state loss。

增加跨视图 soft-selector consistency：

\[
\mathcal L_{\rm gate-eq}
=
1-
\operatorname{Dice}
\left(
\tilde g_t,
W^{-1}(\tilde g_v)
\right),
\]

其中 \(\tilde g\) 是 hard top-K 前的 relaxed gate。

建议：

- stop-gradient 一侧；
- 对所有状态使用，不只 uncertain；
- exact-K 仍用于 forward evidence set 和 interventions；
- 只跑一个预声明系数的 bounded candidate。

若 top-K margin 很小，再加：

\[
\mathcal L_{\rm boundary}
=
\max(0,m_g-g_{(K)}+g_{(K+1)}).
\]

boundary loss 只作为第二选择。

---

## 9. 若确定是 evidence-field dominant

增加跨视图 bipolar evidence consistency：

\[
\mathcal L_{\Delta\text{-eq}}
=
\operatorname{SmoothL1}
\left(
d_t,
W^{-1}(d_v)
\right),
\]

\[
\mathcal L_{T\text{-eq}}
=
\operatorname{SmoothL1}
\left(
t_t,
W^{-1}(t_v)
\right).
\]

原则：

- 对所有状态使用；
- 使用 soft/per-patch field；
- 一侧 stop-gradient；
- 不强迫 hard mask完全相同；
- 不增加新分类 head。

若 balanced synthetic uncertain 通过而 posterization 仍失败，则删除 posterization gate，不为人工纹理修改模型。

---

## 10. 重新定义三个 gate

### Gate A：Mechanism Fit

检查：

```text
train four states correct
rho_S > 0
rho_C < 0
abs(rho_U) small
T_I lowest
drop -> insufficient
control stable
```

当前基本已通过。

### Gate B：Evidence Equivariance

检查弱变换下：

```text
aligned gate/evidence stability
mapped-mask rho stability
soft-vs-hard pooling consistency
top-K boundary margin
```

当前正在诊断。

### Gate C：Mini-P0

只有 A/B 通过后进入真实 64–256 quartet。

不要让“单一 posterized image 的 zero-shot 分类”无限期取代真正机制诊断。

---

## 11. 唯一建议执行顺序

1. 用真实 `train_uncertain.png` / `val_uncertain.png` 重跑 direct pair replay；
2. 修正 affine alignment 和 rotation-direction test；
3. 输出 2×2 cross-replay；
4. 输出 soft/all/K sweep；
5. 输出 common/added/removed patch contributions；
6. 根据结果三选一：
   - selector consistency；
   - evidence-field consistency；
   - redesign synthetic uncertain；
7. 只做一次 bounded 100-step rerun；
8. 通过后再进入 mini-P0。

---

## 12. Go / No-Go

### 允许进入 mini-P0

至少满足：

```text
rho_S > 0
rho_C < 0
train abs(rho_U) < 0.1
val abs(rho_U) < 0.1
T_I lowest
pair violation = 0
removal-to-insufficient = 1
eligible target-control gap > 0
control effect small
weak-view evidence stability passes
```

建议换 3 张不同源 CXR 重复 engineering gate，不改任何超参数。

### 继续阻断

以下任一成立：

```text
balanced uncertain cue仍无法保持 rho
mapped fixed mask仍产生高 rho
soft/all pooling也高度偏极性
weak transform使 signed evidence field系统性翻转
```

此时不能用更大模型或更多训练步数掩盖问题。
