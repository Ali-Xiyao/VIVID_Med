# BiVES-CXR：`995fb81` 最新代码审计与下一阶段执行方案

**日期：** 2026-07-17
**代码版本：** `995fb81` — `Close BiVES optimization gate`
**当前决定：** 当前弱标签四状态路线冻结；Run B、4B、9B 均不启动。

---

# 1. 执行结论

`995fb81` 对预注册 optimization gate 的执行是正确的：

- State-only Run A 固定运行 400 steps；
- 最终 train accuracy 为 `0.7917`，低于预注册的 `1.0`；
- Run B 按规则没有启动；
- 没有继续进行 loss sweep、LR sweep、decoder 改动或模型扩容；
- 结果明确标为 weak-label diagnostic，而非临床结果。

但是，这个结果只关闭：

> 当前 parser-S/C/U + synthetic-I + exact-K + 当前优化协议的四状态弱标签路线。

它不证明：

- 双极证据概念本身无效；
- frozen Qwen3.5 表征没有 S/C 信息；
- 公开专家 S/C + 区域干预路线不可行。

现有结果反而支持新路线：

- frozen-feature global S/C AUROC `0.7889`；
- State-only train S/C AUROC `0.9861`；
- State-only validation S/C AUROC `0.8333`；
- support median signed evidence `+1.9700`；
- contradict median signed evidence `-2.8745`；
- insufficient median total evidence最低。

因此绝对 polarity 已经能学到；当前失败集中于 selector/readout/optimization 与弱 U 标签，不再是 decoder polarity 原点。

---

# 2. 对 `0.7917` 的正确解释

## 可以解释为

当前 exact-K BiVES 四状态模型在固定优化协议下，无法完全拟合 48 条弱代理训练记录。

错误主要位于：

- support：8/12；
- uncertain：8/12；
- contradict：10/12；
- insufficient：12/12。

## 不能解释为

- 方法已经被理论性证伪；
- 训练 400 步已经排除所有 optimization 问题；
- U/I 标签具有临床可靠性；
- 只需换 4B/9B 即可解决。

## 为什么该 gate 不是架构容量的纯测试

当前协议同时包含：

- noisy parser S/C/U；
- synthetic I；
- hard exact-K straight-through selector；
- learned statement-ID embedding；
- global gradient clipping；
- AdamW weight decay；
- 400-step cosine decay到零；
- decoder/evidence initialization与四状态先验不匹配。

所以 gate 结果是一个合法的**整体路线 survival test**，不是单独的 universal-capacity test。

---

# 3. 最新代码中的关键技术发现

## 3.1 State-only 仍执行全部 intervention branches

训练和 primary evaluation 当前固定调用：

```python
run_interventions=True
```

即使：

```yaml
lambda_ies: 0
lambda_pair: 0
lambda_u_pol: 0
lambda_i_mag: 0
lambda_tv: 0
```

模型仍然计算：

- original；
- keep；
- drop；
- 4 个 control。

这些分支不会进入 state-only loss，但会增加大量 GPU 计算和显存。

### 建议

自动判断：

```python
needs_interventions = (
    loss_config.lambda_ies != 0
    or loss_config.lambda_i_mag != 0
    or loss_config.lambda_tv != 0
)
```

state-only diagnostic 和 S/C polarity route 使用：

```python
run_interventions=False
```

这不改变优化目标，只去除无效计算。

---

## 3.2 “Gate 梯度占 99%”只来自一个 quartet

`fixed_batch_optimization_audit()` 使用：

```python
feasibility_batch = next(iter(train_loader))
```

当前 batch 只有一个 S/C/U/I quartet。

因此 step-400 的：

```text
gate head 264.79
context   20.99
fusion     5.79
evidence   2.08
```

不能直接解释为全部 48 条训练样本的总体梯度结构。

### 建议

建立固定 stratified audit loader：

- 每个 finding 至少一个 quartet；
- 或遍历全部 12 quartets；
- 分批累计每个 parameter group 的：
  - mean gradient norm；
  - median；
  - max；
  - clipping coefficient；
  - clipped-step fraction。

当前单-quartet结果可保留，但应改名：

```text
fixed_quartet_gradient_audit
```

而不是全局 optimization diagnosis。

---

## 3.3 Evidence initialization 与 decoder availability 严重失配

当前：

```python
evidence_pm =
    sigmoid(evidence_head(contextual)) * 8.0
```

默认 Linear 初始化下，初始 logit 约为 0，因此：

\[
E^+\approx4,\qquad E^-\approx4,
\]

\[
T\approx8.
\]

decoder 使用：

\[
A=1-\exp(-T/\tau_A),
\qquad \tau_A=1.
\]

所以初始：

\[
A\approx1-e^{-8}=0.999665,
\]

\[
p_I\approx0.000335.
\]

这与 step-0 NLL `2.7761` 的现象一致：初始 insufficient 极难预测，早期梯度会强烈推动 evidence/gate 去寻找低总证据 patch。

### 意义

这可能是 gate-gradient dominance 与前期不稳定的重要来源，但当前四状态路线已经冻结，不建议为它再开一轮调参。

### 新路线建议

S/C-only polarity 训练不应一开始同时承担 availability 学习。

若未来重新启用 availability，使用 decoder-aware initialization，例如选择初始：

\[
A_0=0.5\sim0.75
\]

后按：

\[
T_0=-\tau_A\log(1-A_0)
\]

计算 evidence-head bias，而不是默认 bias 0。

---

## 3.4 Global clipping 需要记录实际 clipping，而不是只记录 raw norm

当前每步：

```python
clip_grad_norm_(
    experiment.parameters(),
    max_grad_norm=1.0,
)
```

但事件中没有记录：

- pre-clip total norm；
- post-clip norm；
- clipping coefficient；
- 每组参数被 clip 的比例。

在 gate raw norm达到数百时，必须知道训练是否几乎每一步都在 clipping。

### 建议

记录：

```python
pre_clip_norm = clip_grad_norm_(...)
clip_coef = min(1.0, max_grad_norm / (pre_clip_norm + 1e-6))
```

并按 parameter group 输出 update/parameter ratio。

---

## 3.5 README 仍包含与代码冲突的命令

README 仍展示：

```bash
python scripts/train_bives_cxr.py \
  --config configs/bives_cxr/qwen35_2b_p0.yaml \
  --debug
```

但训练代码只允许：

```text
--debug + experiment.mode=local_debug
```

`qwen35_2b_p0.yaml` 是 `local_formal` 时该命令会被拒绝。

应删除该示例，防止后续复现实验误用。

---

# 4. VinDr intake 代码的优点与未闭环处

## 已做对

`scripts/prepare_bives_vindr_expert_sc.py` 已经：

- 只使用官方 test consensus labels；
- 将 1 映射为 same-finding support；
- 将 0 映射为 same-finding contradict；
- 不制造 U/I；
- 检查 positive label 必须有 box；
- 检查 negative label不能有同 finding box；
- 排除 test 中无阳性的 Edema；
- 明确不声称 patient-level CI；
- 明确标记 `formal_result=false`。

当前 intake：

```text
Pleural effusion:
111 positive / 2,889 negative

Consolidation:
96 positive / 2,904 negative

Edema:
0 positive / 3,000 negative
```

## 仍未闭环：当前 BiVES Dataset 不能读取 VinDr intake

### 1. DICOM 不兼容

VinDr manifest 输出：

```text
test/<image_id>.dicom
```

但 `BiVESManifestDataset` 使用：

```python
PIL.Image.open(image_path)
```

PIL 不能作为当前 VinDr DICOM 主加载器。

### 2. Manifest schema 不兼容

当前 `read_manifest()` 强制要求：

```text
patient_id
group_id
canonical_statement_id
statement_text
state
```

并且训练 sampler 强制 exact S/C/U/I quartet。

VinDr intake：

- 只有 S/C；
- `patient_id=None`；
- 使用 `unit_id=image_id`；
- 没有 `group_id` quartet。

所以不能直接送入现有四状态 evaluator。

### 正确方向

不要强行把 VinDr 伪装成 quartet manifest。

新增独立接口：

```text
VinDrExpertSCDataset
evaluate_bives_vindr_sc.py
evaluate_bives_vindr_interventions.py
```

---

# 5. 建议的新主线：Expert Polarity + Interventional Evidence

取消本地临床评审后，不再把 S/C/U/I 当作同一种来源的临床四状态真值。

将论文主评价拆为两个可验证轴。

## Axis A：Expert statement polarity

给定图像与固定 statement，预测：

\[
y\in\{\text{support},\text{contradict}\}.
\]

直接使用 bipolar signed evidence：

\[
\Delta=E^+-E^-.
\]

因为 monotone decoder 中：

\[
\log\frac{p_S}{p_C}
=
\frac{\Delta}{\tau_P},
\]

所以不需要添加 flat binary classifier。

训练 loss：

\[
L_{\rm pol}
=
\operatorname{softplus}
\left(
-y\frac{\Delta}{\tau_P}
\right),
\]

其中：

\[
y=+1\ \text{for support},
\qquad
y=-1\ \text{for contradict}.
\]

## Axis B：Interventional evidence sufficiency

不再把 synthetic I 解释成自然临床 insufficient。

评价：

- original；
- expert target-box deletion；
- equal-area random-disjoint pixel deletion；
- evidence-only retention。

主要指标：

\[
\Delta_{\rm target}
=
s(x)-s(x_{\rm target-drop}),
\]

\[
\Delta_{\rm control}
=
s(x)-s(x_{\rm control-drop}),
\]

\[
TCIG
=
\Delta_{\rm target}
-
\Delta_{\rm control}.
\]

所有 pixel intervention 必须重新运行完整 Qwen vision tower。

## U 的处理

U 不作为 primary endpoint。

只有获得：

- expert uncertain label；
- 多读者 disagreement；
- CheXpert expert uncertainty；

时再作为 secondary analysis。

---

# 6. 新 S/C 路线的代码结构

## 6.1 DICOM loader

新增：

```text
bives_cxr/dicom.py
```

处理：

1. `pydicom.dcmread`；
2. modality LUT；
3. VOI LUT / Window Center / Window Width；
4. `MONOCHROME1` 反转；
5. robust percentile clipping；
6. deterministic uint8/RGB conversion；
7. DICOM preprocessing version/hash。

需要 synthetic DICOM unit tests：

- MONOCHROME1；
- MONOCHROME2；
- windowed；
- no-window fallback；
- constant image fail；
- deterministic output hash。

## 6.2 Expert S/C dataset

新增：

```python
class ExpertSCDataset(Dataset):
    ...
```

允许：

```text
sample_id
unit_id
image_path
canonical_statement_id
statement_text
binary_label
bounding_boxes
```

不要求：

```text
group_id
U/I
patient_id
```

## 6.3 Expert evaluator

新增：

```text
scripts/evaluate_bives_vindr_sc.py
```

输出每个 finding：

- AUROC；
- AUPRC；
- prevalence；
- Brier；
- NLL；
- sensitivity at fixed specificity；
- image-level clustered bootstrap CI。

不要把 6,000 rows 独立 bootstrap。

由于每个 image 对应两个 statements，应按：

```text
unit_id=image_id
```

做 cluster bootstrap。

准确名称：

> image-level clustered confidence interval

不能叫 patient-level CI。

## 6.4 Pixel intervention evaluator

新增：

```text
scripts/evaluate_bives_vindr_interventions.py
```

只在 positive images 上使用 box：

- union of boxes；
- small dilation sensitivity；
- target deletion；
- equal-area disjoint control；
- evidence-only retention；
- full vision-tower rerun。

保存：

```text
original score
target-drop score
control-drop score
keep score
box mask
evidence mask
per-image paired differences
```

---

# 7. 下一轮训练不要再使用四状态 quartet

## 训练数据

从 MIMIC parser 中只保留高置信：

- explicit positive；
- explicit negative；
- pleural effusion；
- consolidation。

不再使用：

- parser uncertain；
- synthetic insufficient；
- report omission negative。

构建：

```text
weak_sc_train.jsonl
weak_sc_val.jsonl
```

按 patient 分离，并按 finding / state 平衡。

VinDr test 只用于最后 expert evaluation，不用于：

- 模型选择；
- threshold selection；
- loss选择；
- K选择。

## 训练前先缓存 frozen patch tokens

当前 backbone 冻结且数据固定。

新增：

```text
scripts/cache_qwen35_patch_tokens.py
```

缓存：

```text
patch_tokens
valid_mask
grid_hw
sample_id
image SHA
processor snapshot SHA
Qwen snapshot SHA
```

优点：

- 不再每个 epoch重复跑 Qwen；
- 可以快速做 dense/sparse readout ablation；
- 便于3 seeds；
- 可完整审计 selector，而不混入 vision runtime。

---

# 8. 只允许三个模型进入下一轮 2B S/C P0

使用同一 weak S/C train/val 与同一 cached tokens。

## B0：Frozen pooled logistic baseline

\[
z=\operatorname{mean}_p Z_p.
\]

标准 logistic regression 或单层 linear。

用途：

- 表征下限；
- 不是主方法。

## B1：Dense bipolar evidence

不学习 selector，所有 valid patches参与聚合。

输出：

\[
E^+,E^-,\Delta.
\]

用途：

- 检查 bipolar evidence head 能否稳定学习 polarity；
- 隔离 selector。

## B2：Sparse exact-K BiVES polarity

当前 contextual evidence + exact-K selector，但只优化 S/C polarity。

用途：

- 主候选；
- 与 dense 相比判断 selector是否损害分类或改善证据质量。

## 判读

| 结果 | 结论 |
|---|---|
| B0 强、B1/B2 弱 | evidence readout/optimization问题 |
| B1 强、B2 弱 | exact-K selector是瓶颈 |
| B1/B2 均强 | 可进入 intervention evaluation |
| B2 分类略低但 intervention显著更强 | 可能仍是有效主方法 |
| 所有模型在 VinDr 接近随机 | weak-label/domain-shift路线停止 |

---

# 9. 若 B1 强、B2 弱，推荐 dense-to-sparse evidence preservation

不增加分类 head。

训练一个 dense bipolar teacher，然后 sparse selector满足：

\[
L_{\rm preserve}
=
\left|
\Delta_K
-
\operatorname{sg}(\Delta_{\rm dense})
\right|,
\]

\[
L_{\rm sign}
=
\operatorname{softplus}
\left(
-y\Delta_K/\tau_P
\right).
\]

再加入：

\[
L_{\rm drop}
=
\max
\left(
0,\,
m_D-|\Delta_{\rm dense}|+|\Delta_{\rm drop}|
\right),
\]

\[
L_{\rm control}
=
|\Delta_{\rm control}-\Delta_{\rm original}|.
\]

这仍围绕同一个 bipolar evidence field，不是 CEQ/CCSH/AUCH 式模块堆叠。

---

# 10. VinDr 评估协议

## Primary metrics

每个 finding 分开：

1. AUROC；
2. AUPRC；
3. image-level bootstrap 95% CI。

Macro 平均只在两个 finding 指标计算后进行。

不能将 6,000 rows 混合为一个普通二分类数据集，因为：

- statement 不同；
- prevalence 不同；
- 同一 image 出现两次；
- statement ID 可能产生不同 intercept。

## 不使用 accuracy 作为 primary

VinDr test prevalence：

- pleural effusion 约 3.7%；
- consolidation 约 3.2%。

全负模型 accuracy 已超过 96%。

阈值指标必须在独立 development set 上锁定。

## Formal integrity

最终专家评估前运行：

```bash
python scripts/audit_vindr_cxr_integrity.py \
  --full-sha256
```

并建议对全部 3,000 test DICOM 做 decode preflight，而不只抽样 8 个。

---

# 11. CheXlocalize 应成为下一优先公共数据

CheXlocalize 提供：

- pixel-level radiologist segmentations；
- representative points；
- relevant pathologies包含：
  - consolidation；
  - edema；
  - pleural effusion；
- validation/test分别来自 200/500 patients。

它可补足 VinDr 的两个缺口：

1. patient-level evaluation；
2. 精细 segmentation，而非仅 bounding box。

推荐分工：

| Dataset | 用途 |
|---|---|
| VinDr test consensus | external expert S/C + box intervention |
| CheXlocalize val | development / threshold / intervention protocol |
| CheXlocalize test | patient-level localization/intervention final |
| MIMIC weak S/C | weak training |

---

# 12. 新路线的 Go/No-Go

## 允许继续 2B

必须满足：

```text
VinDr per-finding AUROC/AUPRC
至少不低于 frozen pooled baseline；

target deletion effect
显著大于 equal-area control；

top-K evidence localization
优于 random mask；

结果在 3 seeds 中方向一致。
```

## 允许启动 4B

2B 必须同时满足：

1. 两个 VinDr findings 均有正向结果；
2. TCIG 的 image-level CI 不跨 0；
3. CheXlocalize patient-level结果正向；
4. sparse model相对 dense/Grad-CAM提供额外证据价值；
5. 所有主结果来自同一 locked model family/config。

## 停止条件

```text
Dense bipolar和exact-K都不优于简单 pooled probe；

target deletion不比control更强；

专家数据结果只靠一个 finding；

模型仅在弱 parser validation 上有效；

公开 expert test AUPRC接近 prevalence baseline。
```

---

# 13. 最短下一步清单

1. 冻结 `995fb81` 为四状态弱标签路线 closeout。
2. 不运行 Run B、4B 或 9B。
3. 修 README 中无效的 formal `--debug` 命令。
4. 给 state-only 路径增加 `run_interventions=False`。
5. 将 optimization audit扩展为全部 quartets，并记录 clip coefficient。
6. 新增 deterministic VinDr DICOM loader。
7. 新增独立 Expert S/C schema/evaluator。
8. 对 VinDr test执行 full SHA与全量 decode preflight。
9. 构建 pleural-effusion / consolidation weak S/C train/val。
10. 缓存 frozen Qwen patch tokens。
11. 跑 B0 pooled、B1 dense bipolar、B2 sparse bipolar 三个 2B/P0模型。
12. 只有 B2 在 expert polarity和pixel intervention上同时通过，才考虑 4B。
13. 下载/验证 CheXlocalize，补 patient-level与segmentation证据。
14. 论文主线改为：
    `expert polarity + interventional evidence sufficiency`。
15. U/I 四状态结果只保留为 exploratory/appendix，不再作为 primary clinical claim。
