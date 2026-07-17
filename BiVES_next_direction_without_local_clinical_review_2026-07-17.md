# BiVES-CXR：取消本地临床评审后的结果审计与下一步方案

**日期：** 2026-07-17
**对应提交：** `c113b2a`
**当前实验：** 5,000-study parser-v3 candidate expansion；48 train / 48 validation proxy rows；Qwen3.5-2B frozen vision；50 optimization steps。

---

# 1. 当前结果应如何定性

## 可以成立的结论

1. 在三个保留 finding 上，冻结的 Qwen3.5-2B 表征和当前 BiVES 读出包含一定的 **support-vs-contradict 相对排序信号**。
2. 合成 evidence-removal 图像的总 evidence 明显低于原图，因此 **intervention-induced availability** 信号已经出现。
3. 当前 pipeline、parser provenance、patient/image split 和本地执行链已经可以稳定运行。
4. 4B/9B 未启动是正确决策。

## 不能成立的结论

1. 不能声称四状态临床分类已经成立。
2. 不能把 parser uncertain 当作 visual uncertainty 的临床真值。
3. 不能把合成 evidence-removal 当作自然临床 insufficient。
4. 不能把 U/I AUROC=1.0 写成临床 answerability 成功。
5. 不能把 5,000 studies 写成 5,000 个训练样本；实际训练与验证各只有 12 个 quartet。
6. 不能因为 contradict 的 signed evidence 为正，就直接断言方法在理论上不可学习；当前 50-step 训练连训练集四状态也没有拟合。

---

# 2. 数值诊断

当前训练与验证：

```text
train accuracy    = 0.25
validation acc    = 0.25
train macro-F1    = 0.10
validation F1     = 0.10
train NLL         = 1.3675
validation NLL    = 1.3692
uniform 4-class NLL = log(4) = 1.3863
```

训练 NLL 只比均匀预测略低，说明当前 head 尚未完成基本训练拟合。

当前每个 split：

```text
12 quartets
12 support
12 contradict
12 uncertain
12 insufficient
```

每个 finding 只有：

```text
4 support + 4 contradict
```

因此 per-finding AUROC 只有 16 个正负 pair comparisons：

```text
0.875  = 14 / 16
0.8125 = 13 / 16
1.0    = 16 / 16
```

这些数字适合作为 pilot ranking signal，不是稳定外部验证。

---

# 3. All-insufficient collapse 的两部分原因

## 3.1 Availability 尺度未校准

当前：

\[
A=1-\exp(-T/\tau_A),
\qquad \tau_A=1.
\]

在 S/C/U 的中位总证据约 0.85–0.94 时：

\[
A\approx0.57-0.61,
\qquad p_I=1-A\approx0.39-0.43.
\]

而 answerable mass \(A\) 还要在 S/C/U 三类之间分配，因此 argmax 很容易全部落到 I。

训练代理数据拟合 \(\tau_A,\tau_P,m_U\) 后，S/I 明显恢复，证明尺度问题是真实存在的。

## 3.2 Signed evidence 没有绝对原点

中位 signed evidence：

```text
S = +0.1309
C = +0.0464
U = +0.0863
I = +0.0215
```

S/C 排序存在，但 C 没跨到负侧，U 也没有居中。

当前 pair loss 只要求：

\[
\rho_S \ge \rho_C + m.
\]

它不要求：

\[
\rho_S>0,
\qquad \rho_C<0.
\]

不过 state NLL 本身应提供绝对方向梯度。因此在修改方法前，必须先区分：

```text
训练预算不足 / 梯度冲突
vs
参数化原点不可辨识
```

---

# 4. 下一轮只允许做一个有界的 Optimization–Identifiability Gate

不启动 4B/9B，不扩大 parser pool，不做广泛超参数搜索。

使用同一 48-row train proxy、冻结 Qwen3.5-2B、固定 seed 和 exact-K。

## 4.1 零训练诊断

保存 step 0 与 step 50 的：

```text
E+
E-
T
Delta
rho
state probabilities
gate logits
per-loss gradient norm
```

检查：

1. S/C AUROC 在 step 0 已有多少；
2. step 50 对 signed evidence 分布实际移动多少；
3. state NLL 对 S/C 的 polarity gradient 是否方向正确；
4. state / IES / pair / U-pol 梯度余弦是否冲突；
5. evidence head、gate、context block 是否有接近零的梯度；
6. 每个 finding 是否有不同的 signed-evidence offset。

增加 frozen-feature logistic regression：

```text
pooled S/C classifier, global
pooled S/C classifier, per finding
```

报告 AUROC、intercept 和 calibration。

## 4.2 两个预声明训练诊断

### Run A：state-only overfit diagnostic

```text
same model
same 48 train rows
frozen backbone
state NLL only
300–500 steps
no validation model selection
```

目标不是论文结果，而是检查：

```text
train accuracy = 1.0
C median Delta < 0
U median |rho| small
I median T lowest
```

### Run B：full objective overfit diagnostic

```text
same initialization protocol
same 300–500 steps
current state + IES + pair + U-pol + I-magnitude
```

## 4.3 判读

| 结果 | 结论 |
|---|---|
| A 不能拟合 train | implementation / optimizer / selector / capacity 问题 |
| A 能拟合，B 不能 | auxiliary objectives 与 state objective 冲突 |
| A/B 都能拟合，但 val C 始终为正 | proxy shift、finding offset 或表示原点问题 |
| A/B 都能拟合并泛化 | 原先主要是 50-step undertraining |
| 简单 logistic probe 明显优于 BiVES head | BiVES optimization/readout 问题，不是 frozen representation 问题 |

只运行预声明的两个诊断，不根据 validation 继续扫权重。

---

# 5. 若确认绝对原点确实不稳定，优先修改证据参数化

不要先叠加多个额外 head。

建议将两个独立的非负 evidence heads 改为：

\[
m_p=e_{\max}\sigma(a_p),
\]

\[
r_p=\tanh(b_p),
\]

\[
e_p^+=\frac{m_p(1+r_p)}{2},
\qquad
e_p^-=\frac{m_p(1-r_p)}{2}.
\]

于是：

\[
e_p^++e_p^-=m_p,
\]

\[
e_p^+-e_p^-=m_pr_p.
\]

优点：

1. availability magnitude 与 polarity 被结构性解耦；
2. 只有一个 signed polarity logit，不再由两个独立 head 的 bias 差定义原点；
3. \(r=0\) 是明确的 bipolar neutral origin；
4. 保留双极 evidence map；
5. 不引入 flat four-class head。

可加一个小的 proxy sign-margin：

\[
L_{\rm origin}
=
\operatorname{softplus}(m-\rho_S)
+
\operatorname{softplus}(m+\rho_C).
\]

但它只能用于 weak-proxy pretraining，不得作为临床有效性的证据。

Availability 端建议使用可学习的正参数：

\[
\tau_A=\operatorname{softplus}(\alpha_A)+\epsilon,
\]

或：

\[
A=\sigma((T-b_A)/\tau_A).
\]

这些参数必须在 train 上学习或在独立 calibration split 上拟合，不能在 test 上选择。

---

# 6. 取消本地临床评审后的论文路线

## 原四状态临床 claim 必须降级

当前 S/C/U 是规则解析候选，I 是合成 evidence removal。它们可以用于：

```text
weak proxy pretraining
mechanism debugging
ablation
```

不能用于：

```text
clinical four-state ground truth
clinical U/I validation
expert agreement
clinical safety claim
```

## 推荐把主评价改成三个可验证轴

### Axis 1：Expert polarity

在现成的公开放射科医师标注数据上评价：

```text
support vs contradict
```

primary metrics：

```text
AUROC
AUPRC
balanced accuracy
patient bootstrap CI
```

### Axis 2：Intervention-induced availability

利用专家病灶区域：

```text
original image
target-region deletion
equal-area irrelevant deletion
evidence-only image
```

评价：

```text
target deletion -> lower availability
irrelevant deletion -> stable
retained evidence -> preserve polarity
```

这里应称：

> intervention-induced insufficiency

而不是自然临床 insufficient。

### Axis 3：Reader ambiguity（可选）

只有在公开数据提供：

```text
radiologist uncertain labels
individual reader disagreement
```

时才评价 uncertainty。

没有合格标签时，U 不进入 primary endpoint。

---

# 7. 推荐的公开专家标注数据路线

## 首选组合

### CheXpert expert validation/test

用途：

```text
S/C polarity
expert uncertainty / disagreement（在许可和标签可用范围内）
calibration
```

### CheXlocalize

用途：

```text
radiologist segmentation / representative points
target deletion
control deletion
grounding overlap
```

### VinDr-CXR

用途：

```text
external S/C classification
radiologist bounding boxes
external target/control intervention
```

## 可选补充

### RSNA Pneumonia Detection Challenge

用途：

```text
单病种 localization/intervention stress test
```

---

# 8. 建议的新论文主张

不再以：

> expert-validated four-state clinical statement verifier

作为当前主张。

建议改为：

> Weakly supervised bipolar evidence learning separates expert-validated statement polarity from intervention-induced evidence availability.

核心贡献可写为：

1. statement-conditioned magnitude–polarity evidence field；
2. monotone bipolar conditional decoder；
3. target/retain/control intervention closure；
4. weak report supervision + public expert evaluation。

四状态输出可以保留，但：

- S/C 是专家评价主轴；
- I 是 intervention-defined 轴；
- U 只在专家不确定/分歧标签存在时评价。

---

# 9. Go / No-Go

## 允许进入真实 2B P0

必须先满足：

```text
state-only diagnostic can overfit train
full objective does not destroy absolute polarity
C median rho < 0 on train
U median |rho| is controlled on train
availability does not trivially collapse all answerable states
```

## 允许进入 4B

2B 必须在公开专家数据上同时满足：

```text
S/C better than matched baseline
target deletion effect > irrelevant deletion effect
evidence localization better than random/attention baseline
3 seeds
patient-level CI
single locked checkpoint
```

## 不允许继续扩大

以下情况下停止 BiVES four-state 路线：

```text
state-only cannot fit 48 proxy rows
expert S/C evaluation fails
target deletion is not stronger than control deletion
results depend on parser-U or synthetic-I shortcuts
```

---

# 10. 最短执行顺序

1. 冻结 `c113b2a` 为 Proxy-P0-A 诊断结果。
2. 做 step0/step50 evidence 与 gradient audit。
3. 跑 state-only 与 full-objective 两个 300–500-step overfit diagnostics。
4. 根据结果决定是否使用 magnitude–polarity factorization。
5. 不再扩展 parser U/I。
6. 下载/整理 CheXpert expert labels、CheXlocalize 和 VinDr-CXR。
7. 先完成公开专家 S/C + target/control intervention 的 2B experiment。
8. 2B 通过后做 3 seeds。
9. 再决定是否运行 4B；9B 只做 scale validation。
10. 没有公开专家评价时，不以 MIA/TMI-ready clinical paper 定稿。
