# BiVES-CXR C6G：MS-CXR 几何控制协议与后续执行方案

**基于提交：** `248eacde65e00c403d0c1744df7d1014448d602f`  
**日期：** 2026-07-18  
**当前状态：** C6F `FAIL_PREOPEN_GEOMETRY_NO_MODEL_ACCESS`；28/29 几何可行；未创建 opening marker，未解码 JPG，未加载 Qwen/checkpoint，未使用 GPU，未产生模型分数。

---

## 1. 决策

1. C6F 永久冻结，不修改、不覆盖、不删除失败样本。
2. 不直接运行模型，不重用 C6F authority。
3. 新建 **C6G geometry-only authority**，只允许 score-free control construction。
4. C6G 若 29/29 通过，再新建独立 **C6H one-time model-evaluation authority**。
5. C6G 或 C6H 任一失败后停止该 MS-CXR rescue route；不继续 C6I/C6J 式逐次放宽。

---

## 2. 当前失败的准确含义

当前 v1 control generator：

- 使用 17×17 lattice seeds；
- 加入 valid connected-component centroids；
- 对每个 seed 只执行一条 deterministic greedy connected growth；
- 要求 exact area、target disjoint、一个 4-connected component；
- 最后要求 control centroid 与 target 落入完全相同的 3×3 coordinate zone。

因此当前错误：

```text
no exact-area connected control has the target coordinate zone
```

应解释为：

```text
v1 deterministic candidate family found no qualifying control
```

而不是：

```text
在所有可能的 connected masks 中不存在满足条件的 control
```

C6G 中建议修改 failure wording，并为失败行保存最接近候选的完整 geometry certificate。

---

## 3. C6G 首先应输出的 score-free failure certificate

对全部 29 行，尤其失败行，保存：

```text
target area
target component count
target perimeter
target compactness/aspect ratio
target normalized centroid x/y
target categorical zone

valid-content area
valid connected-component sizes

seed count
grown-candidate count
same-zone candidate count

nearest rejected candidate:
  exact area
  connected component count
  normalized centroid x/y
  categorical zone
  continuous centroid distance
  perimeter ratio
  compactness ratio
  reason rejected
```

这一步禁止：

- JPG pixel decoding；
- image-intensity matching；
- model/checkpoint load；
- CUDA；
- score computation。

---

## 4. 推荐的新统一 control 定义

名称建议：

```text
bives_continuous_location_connected_control_v2
```

对 **全部 29 行统一使用**，不得只为失败行设置例外。

### 硬约束

\[
|C|=|T|
\]

\[
C\cap T=\varnothing
\]

\[
C\subseteq \text{content}
\]

\[
\operatorname{components}_{4}(C)=1
\]

并且：

- geometry-only；
- deterministic；
- target/control mask hashes locked；
- no model/image-score input；
- no row exclusion。

### 删除的脆弱硬约束

不再要求：

```text
control horizontal zone == target horizontal zone
control vertical zone   == target vertical zone
```

因为 zone 由以下不连续边界定义：

```text
x < 0.40         left
0.40 <= x <=0.60 central
x > 0.60         right

y < 1/3          upper
1/3 <= y < 2/3   middle
y >= 2/3         lower
```

一个仅跨过边界少量像素、但空间上更接近的 control，不应被比更远的 same-zone control 更差地处理。

### 连续匹配目标

令 target/control 的 content-normalized centroid 为：

\[
c_T=(x_T,y_T),\qquad c_C=(x_C,y_C).
\]

定义：

\[
d_{\rm loc}
=
|x_C-x_T|+|y_C-y_T|.
\]

沿用当前 perimeter matching 思路：

\[
J(C)
=
d_{\rm loc}
+
0.10\left|
\log\frac{P_C}{P_T}
\right|.
\]

在所有满足硬约束的候选中，选择 \(J(C)\) 最小者。

---

## 5. Location mismatch gate 如何预注册

不要根据 MS-CXR 模型分数选择 location threshold。

建议从 **冻结的 C4/C5 已接受 geometry rows** 中读取：

```text
target normalized x/y
control normalized x/y
target perimeter
control perimeter
```

预先计算：

\[
d_{\rm loc}
\]

和：

\[
r_P=\left|\log(P_C/P_T)\right|.
\]

在 C6G authority 中冻结数值阈值，例如：

```text
max_location_distance =
    frozen C4/C5 accepted-control maximum

max_log_perimeter_ratio =
    frozen C4/C5 accepted-control maximum
```

或使用事先声明的 99th percentile。阈值一旦写入 C6G authority，不能在查看 MS-CXR model scores 后修改。

C6G 仍要求：

```text
29/29 controls feasible
0 denominator exclusions
0 invariant failures
```

---

## 6. Candidate generator v2

建议保留单一 connected control，但扩展 deterministic candidate family：

1. 当前 17×17 geodesic-growth seeds；
2. 33×33 dense lattice；
3. 每个 valid connected component 的 centroid；
4. distance-transform local maxima；
5. feasible translated-target-shape candidates；
6. 固定的多种 frontier tie orders。

所有候选必须先满足：

```text
exact area
target disjoint
inside content
one 4-connected component
```

再按统一 \(J(C)\) 排序。

必须保存：

```text
candidate family
seed identity
frontier-order identity
candidate counts
selected objective
selected mask SHA
```

---

## 7. 不建议的第一选择

### 不删除失败样本

29 例本来就小，删除一例会改变预先固定分母，并与 C6F authority 冲突。

### 不优先允许多连通分量

多 component 会改变：

- perimeter；
- local-ring area；
- blur boundary；
- operator artifact。

这可能直接改变 TCIG 的可比性。

### 不使用 contralateral mirror 作为唯一 control

非侧别 canonical statement 下，另一肺可能也有同一病灶或未被完整框出的异常。

### 不使用 image-intensity/model-score matching

这会让 control selection 接触图像内容或模型反应，增加选择偏差，并破坏纯 geometry pre-open 边界。

### 不只对失败行使用特殊规则

control definition 必须对全部 29 行一致。

---

## 8. C6G authority 建议字段

```text
status=AUTHORIZED_SCORE_FREE_GEOMETRY_ONLY
model_evaluation_authorized=false
gpu_authorized=false
image_decode_authorized=false

source_rows=29
source_patients=29
source_findings:
  consolidation=15
  pleural_effusion=14

target_boxes=native_no_dilation
control_version=bives_continuous_location_connected_control_v2

hard_constraints:
  exact_area=true
  target_disjoint=true
  within_content=true
  connected_components_4=1
  denominator_exclusion=false

objective:
  centroid_l1_weight=1.0
  log_perimeter_ratio_weight=0.10

threshold_source=frozen_C4_C5_geometry_only
geometry_gate=29_of_29
```

C6G 不应包含任何模型生存阈值；它只决定 geometry lock 是否可建立。

---

## 9. C6H 条件性模型评估

只有 C6G：

```text
status=pass
eligible=29
infeasible=0
evaluation_gate_open_geometry=true
```

后，才创建 C6H。

C6H 可继续冻结：

- 同一 Qwen3.5-2B snapshot；
- 同一 B2 exact-K=16 step-450 checkpoint；
- 同一两个 canonical statements；
- native boxes/no dilation；
- local-ring mean width 8；
- masked Gaussian sigma 8/truncate 3；
- patient bootstrap 2,000/seed 17；
- 同一 TCIG survival thresholds；
- positive-only；
- no AUROC/AUPRC/accuracy；
- no training/tuning/model selection；
- no 4B/9B。

C6H 仍是：

```text
small independent external mechanism evaluation
```

不能写成：

```text
clinical validation
classification validation
C5 failure reversal
4B/9B authorization
```

---

## 10. 必须新增的测试

### Geometry candidate family

```text
all selected controls exact-area
all selected controls target-disjoint
all selected controls inside content
all selected controls one 4-connected component
deterministic across runs/workers
```

### Boundary robustness

构造 target/control centroid 位于 zone 边界两侧、但连续距离很小的 synthetic case，确认 v2 选择连续更近的候选，而不是被 categorical zone 阻断。

### Failure certificate

任何失败必须保存 nearest rejected candidate 与完整 reason，不得只返回字符串。

### Fail closed

```text
28/29 -> no geometry opening marker
29/29 -> geometry lock only
no model/GPU allowed under C6G
```

### Artifact immutability

```text
C6E byte-identical
C6F authority/log/locks unchanged
new C6G artifacts use new filenames/hashes
```

---

## 11. 最短执行顺序

1. 冻结并标记 C6F 为 final pre-open failure。
2. 新建 C6G geometry-only authority。
3. 添加 failure-certificate tooling。
4. 从 frozen C4/C5 geometry rows计算并冻结 continuous mismatch thresholds。
5. 实现统一 v2 connected control generator。
6. 只运行 score-free 29-row geometry build。
7. 若不是 29/29，停止 MS-CXR route。
8. 若 29/29，冻结 C6G geometry/data lock。
9. 新建独立 C6H one-time 2B model-evaluation authority。
10. C6H 完成后，无论 pass/fail 均停止该 rescue route；不继续改 control。

---

## 12. 最终建议

优先修复的是：

```text
hard categorical zone + incomplete candidate search
```

而不是：

```text
删除一例
允许碎片化 control
修改 model/operator
```

最小、科学上最合理的变化是：

> 保留 exact-area、disjoint、single-connected 和 geometry-only；将脆弱的离散 same-zone 硬筛选改为预注册的连续 centroid/perimeter matching，并对全部 29 行统一重建 control。
