# BiVES-CXR Support 极性失败：解析诊断与定向修复方案

**审查对象：** `Ali-Xiyao/VIVID_Med@41a6efd`
**日期：** 2026-07-17
**结论：** 当前失败不是普通超参数问题，而是现有 closed-form decoder 的优化几何缺陷。

---

## 1. 观测结果

最佳 100-step engineering gate：

- train accuracy = 0.75
- validation accuracy = 0.75
- support 是唯一错误类别
- support:
  - \(E^+=2.8493\)
  - \(E^-=3.7504\)
  - \(\Delta=E^+-E^-=-0.9011\)
  - \(T=E^++E^-=6.5997\)
  - \(\rho=\Delta/T=-0.13654\)
- S-vs-C ranking AUROC = 1.0
- pair-margin violation = 0
- uncertain absolute polarity = 0.0394
- removal-to-insufficient = 1.0
- irrelevant-control stability = 1.0
- eligible target-control gap = 1.3570

因此模型学会了相对排序、U/I 和 intervention direction，但没有学会 support 的绝对正极性。

---

## 2. 根因：现有 decoder 在错误半轴存在解析局部最优

当前 decoder：

\[
T=E^++E^-,
\qquad
\Delta=E^+-E^-,
\]

\[
A=1-\exp(-T/\tau_A),
\]

\[
D=1-\exp(-|\Delta|/\tau_D),
\]

\[
Q=\sigma(\Delta/\tau_P),
\]

\[
p_S=ADQ,
\qquad
p_C=AD(1-Q),
\qquad
p_U=A(1-D),
\qquad
p_I=1-A.
\]

当前 overfit config 使用：

\[
\tau_A=\tau_D=\tau_P=1.
\]

固定 \(T\)，对 support 样本考虑错误半轴 \(\Delta<0\)。

令：

\[
x=e^\Delta,\qquad 0<x<1.
\]

则：

\[
D=1-x,
\qquad
Q=\frac{x}{1+x},
\]

所以：

\[
\frac{p_S}{A}
=
\frac{x(1-x)}{1+x}.
\]

其驻点满足：

\[
1-2x-x^2=0,
\]

因此：

\[
x^\star=\sqrt 2-1,
\]

\[
\Delta^\star
=
\log(\sqrt 2-1)
=
-\operatorname{asinh}(1)
\approx -0.8813736.
\]

实际 support：

\[
\Delta=-0.9011,
\]

与错误半轴的解析驻点只差约：

\[
0.0197.
\]

这不是巧合。模型几乎精确停在 decoder 的 wrong-polarity attractor。

---

## 3. 实际概率复算

使用：

\[
E^+=2.8493,
\qquad
E^-=3.7504,
\]

得到：

\[
A=0.99864,
\]

\[
D=0.59388,
\]

\[
Q=0.28882.
\]

因此：

\[
p_S=0.17129,
\]

\[
p_C=0.42178,
\]

\[
p_U=0.40557,
\]

\[
p_I=0.00136.
\]

所以它确实被预测为 contradict，而且与 uncertain 很接近。

---

## 4. 为什么 pair loss 没有解决

当前 pair loss：

\[
\mathcal L_{\rm pair}
=
\max
\left(
0,\,
m-\rho_S+\rho_C
\right),
\]

其中：

\[
\rho=\frac{E^+-E^-}{E^++E^-+\epsilon}.
\]

它只要求：

\[
\rho_S\geq \rho_C+m.
\]

它不要求：

\[
\rho_S>0,
\qquad
\rho_C<0.
\]

因此下面的结果完全满足 pair loss：

\[
\rho_S=-0.1365,
\qquad
\rho_C\leq-0.3365.
\]

这解释了：

- S-vs-C AUROC = 1；
- pair violation = 0；
- support 仍位于负半轴。

---

## 5. 为什么普通 state-only、IES ramp 和 lambda 调整没有解决

对 support：

\[
\mathcal L_S
=
-\log A-\log D-\log Q.
\]

在错误半轴：

- \(-\log Q\) 希望把 \(\Delta\) 向正方向推；
- \(-\log D\) 希望增大 \(|\Delta|\)，因此把负 \(\Delta\) 继续向负方向推。

在：

\[
\Delta\approx-0.88137
\]

两项梯度正好抵消。

更严重的是：

\[
D(0)=0,
\]

因此：

\[
p_S(0)=p_C(0)=0.
\]

所以正负半轴之间存在一个零概率屏障。单纯：

- 增加训练步数；
- 改 `lambda_IES`；
- state-only warm-up；
- support class weighting；
- 更换 seed；
- 换 4B/9B；

都没有消除这个结构。

---

## 6. 不建议的修法

不建议继续：

1. 扫描 `lambda_IES`；
2. 仅增大 `lambda_pair`；
3. 仅增加 support 权重；
4. 增加训练步数；
5. 解冻 backbone；
6. 换更大模型；
7. 只加有限权重的 polarity BCE，同时保留原 decoder NLL。

最后一项不够稳健，因为原 decoder 在 \(\Delta=0\) 的 S/C 概率为零，仍存在强屏障。

---

## 7. 推荐修复：Monotone Bipolar Conditional Decoder

保留：

- 双极证据 \(E^+,E^-\)；
- availability；
- decisiveness；
- polarity；
- 四状态闭式输出；
- 无 trainable flat four-class head。

只替换 conditional S/C/U 几何。

定义：

\[
T=E^++E^-,
\qquad
\Delta=E^+-E^-,
\]

\[
A=1-\exp(-T/\tau_A),
\]

\[
z=\frac{\Delta}{2\tau_P}.
\]

设置一个正的 uncertainty mass：

\[
m_U>0.
\]

构造 conditional logits：

\[
\ell_S=z,
\qquad
\ell_C=-z,
\qquad
\ell_U=\log(2m_U).
\]

然后：

\[
(\pi_S,\pi_C,\pi_U)
=
\operatorname{softmax}
(\ell_S,\ell_C,\ell_U).
\]

最终：

\[
p_S=A\pi_S,
\]

\[
p_C=A\pi_C,
\]

\[
p_U=A\pi_U,
\]

\[
p_I=1-A.
\]

仍可定义：

\[
D=\pi_S+\pi_C,
\]

\[
Q=\frac{\pi_S}{\pi_S+\pi_C}
=
\sigma(\Delta/\tau_P).
\]

因此仍满足：

\[
p_S=ADQ,
\]

\[
p_C=AD(1-Q),
\]

\[
p_U=A(1-D).
\]

只是 \(D\) 改为归一化一致的形式：

\[
D
=
\frac{
2\cosh(\Delta/(2\tau_P))
}{
2\cosh(\Delta/(2\tau_P))+2m_U
}.
\]

---

## 8. 新 decoder 的关键性质

### Support 单调

\[
\pi_S
=
\frac{e^z}
{e^z+e^{-z}+2m_U}.
\]

所以：

\[
\frac{\partial \pi_S}{\partial z}
=
\frac{
e^z(2e^{-z}+2m_U)
}{
(e^z+e^{-z}+2m_U)^2
}
>0.
\]

因此 support probability 对 \(\Delta\) 严格递增。

### Contradict 单调

\[
\frac{\partial \pi_C}{\partial z}<0.
\]

### Uncertain 对称

\[
\pi_U(\Delta)=\pi_U(-\Delta),
\]

并在：

\[
\Delta=0
\]

达到最大值。

### 无错误半轴局部最优

有限 \(\Delta\) 下三类 conditional probability 都严格大于 0，没有 \(\Delta=0\) 的零概率屏障。

---

## 9. 建议代码

```python
class EvidenceStateDecoder(nn.Module):
    decoder_kind = "monotone_bipolar_conditional"
    has_flat_state_head = False

    def __init__(
        self,
        tau_a: float = 1.0,
        tau_p: float = 1.0,
        uncertainty_mass: float = 1.0,
    ) -> None:
        super().__init__()

        for name, value in (
            ("tau_a", tau_a),
            ("tau_p", tau_p),
            ("uncertainty_mass", uncertainty_mass),
        ):
            if float(value) <= 0:
                raise ValueError(f"{name} must be positive")

            self.register_buffer(
                name,
                torch.tensor(float(value)),
            )

    def forward(
        self,
        evidence_pos: torch.Tensor,
        evidence_neg: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if evidence_pos.shape != evidence_neg.shape:
            raise ValueError(
                "positive and negative evidence must have "
                "identical shapes"
            )

        total = evidence_pos + evidence_neg
        delta = evidence_pos - evidence_neg

        availability = 1.0 - torch.exp(
            -total / self.tau_a
        )

        signed_logit = delta / (2.0 * self.tau_p)

        uncertain_logit = torch.log(
            2.0 * self.uncertainty_mass
        ).expand_as(signed_logit)

        conditional = torch.softmax(
            torch.stack(
                (
                    signed_logit,
                    -signed_logit,
                    uncertain_logit,
                ),
                dim=-1,
            ),
            dim=-1,
        )

        conditional_support = conditional[..., 0]
        conditional_contradict = conditional[..., 1]
        conditional_uncertain = conditional[..., 2]

        decisiveness = (
            conditional_support
            + conditional_contradict
        )

        polarity = conditional_support / (
            decisiveness + 1e-8
        )

        support = availability * conditional_support
        contradict = (
            availability * conditional_contradict
        )
        uncertain = (
            availability * conditional_uncertain
        )
        insufficient = 1.0 - availability

        probabilities = torch.stack(
            (
                support,
                contradict,
                uncertain,
                insufficient,
            ),
            dim=-1,
        )

        return {
            "state_probs": probabilities,
            "availability": availability,
            "decisiveness": decisiveness,
            "polarity": polarity,
            "total_evidence": total,
            "signed_evidence": delta,
        }
```

建议初始：

```yaml
decoder:
  type: monotone_bipolar_conditional
  tau_a: 1.0
  tau_p: 1.0
  uncertainty_mass: 1.0
```

此时 \(\Delta=0\)：

\[
\pi_U=0.5,
\qquad
\pi_S=\pi_C=0.25.
\]

---

## 10. 必须增加的单元测试

### 10.1 旧 decoder 陷阱回归

保留一个明确的 legacy failure test/documentation：

```python
delta_star = -torch.asinh(torch.tensor(1.0))
```

确认旧公式的 support loss 在错误半轴存在驻点。

### 10.2 单调性

```python
delta = torch.linspace(-6.0, 6.0, 1001)

positive = 4.0 + 0.5 * delta
negative = 4.0 - 0.5 * delta

probs = decoder(
    positive,
    negative,
)["state_probs"]

assert torch.all(
    probs[1:, 0] > probs[:-1, 0]
)

assert torch.all(
    probs[1:, 1] < probs[:-1, 1]
)
```

### 10.3 Uncertain 对称且在零点最大

```python
assert torch.allclose(
    probs[:, 2],
    probs.flip(0)[:, 2],
    atol=1e-6,
)

mid = len(delta) // 2

assert probs[mid, 2] == probs[:, 2].max()
```

### 10.4 Target gradient direction

对所有负 \(\Delta\)：

```text
support NLL gradient must increase delta
```

对所有正 \(\Delta\)：

```text
contradict NLL gradient must decrease delta
```

### 10.5 概率归一化和交换对称

继续保留：

```text
sum = 1
swap E+ / E-
→ swap support / contradict
→ preserve uncertain / insufficient
```

---

## 11. 唯一下一次 GPU gate

不要同时改其他超参数。

固定：

- 相同 synthetic quartet；
- 相同 seed；
- 相同 Qwen3.5-2B frozen vision；
- 相同 K=16；
- 相同 100 steps；
- 相同 learning rate；
- 相同 IES/pair/U/I loss；
- 只替换 decoder。

必须通过：

```text
train accuracy = 1.0
validation accuracy = 1.0
rho_support > 0
rho_contradict < 0
abs(rho_uncertain) small
insufficient total evidence lowest
pair violation = 0
removal-to-insufficient = 1
eligible target-control gap > 0
control effect remains small
```

如果单调 decoder 仍不能通过，再诊断：

- synthetic transform separability；
- exact-K selector instability；
- statement-table/head capacity；
- optimizer schedule。

在替换 decoder 之前，不应再做超参数搜索。

---

## 12. 对论文故事的影响

这不是负面结果，反而可以强化方法学：

> A naive product decomposition of decisiveness and polarity can introduce
> polarity-inverted optimization basins. BiVES therefore uses a monotone
> bipolar conditional decoder whose support and contradiction probabilities
> are provably monotonic in signed evidence.

这比“加一个 support loss”更适合作为 MIA/TMI 方法贡献。

新 decoder 仍然：

- 无生成式 LLM；
- 无 flat four-class head；
- 由同一 bipolar evidence field 产生四状态；
- 保留 availability、decisiveness 和 polarity；
- 可进行温度/uncertainty-mass calibration；
- 与 keep/drop/control evidence closure 完全兼容。
