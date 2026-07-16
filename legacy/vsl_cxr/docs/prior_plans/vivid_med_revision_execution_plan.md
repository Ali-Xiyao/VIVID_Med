# VIVID-Med 大修执行计划（实验 + 模块 + 论文重构）

> 目标：把论文从 **“frozen LLM semantic manifold + SPD”** 重构为更稳的 **“schema-aware / answerability-aware structured CXR representation learning”**。  
> 执行原则：**先定位 frozen-LM 在什么场景下真正有用，再根据 failure case 设计模块**；不要沿 SPD 思路继续烧卡，也不要预设一定要做视觉正则。

---

## 0. 当前结论快照

### 0.1 当前最稳的主线

推荐新主线：

> **VIVID-Med is a controlled study of schema-aware structured supervision for deployable CXR ViTs. UMS/schema supervision is the stable contributor; frozen-LM visual-prefix training is useful in controlled settings but not the dominant source; SPD is unstable and should be treated as sensitivity/negative analysis.**

中文版本：

> **VIVID-Med 是一个面向可部署 CXR ViT 的结构化 UMS/schema 监督框架。UMS/state supervision 是主要稳定贡献；frozen-LM visual-prefix objective 有作用，但不是主要来源；SPD 不稳定，不能主打。**

### 0.2 当前主要证据

| 方法 / Variant | CheXpert macro-AUC | CheXpert macro-F1 | NIH macro-AUC | 当前解释 |
|---|---:|---:|---:|---|
| Data-matched BCE ViT-B | 0.7927 | 0.8987 | - | 同数据普通监督 baseline |
| Frozen-LM UMS / no-SPD | 0.8439 | 0.9095 | 0.7068 | 当前 CheXpert 最强主线 |
| no-LM UMS state classifier | 0.8273 | 0.9143 | 0.7262 | UMS/state supervision 本身很强 |
| Frozen-LM UMS + SPD default | 0.8208 | 0.9114 | 0.7214 | SPD 默认，in-domain 弱 |
| Frozen-LM UMS + SPD G=2 | 0.8291 | 0.9149 | 0.7176 | SPD sensitivity，仍低于 no-SPD |
| Frozen-LM free-text target | 0.8126 | 0.9064 | 0.6365 | UMS 明显强于 free text |
| Random-mask proxy | 0.7387 | 0.8876 | 0.6002 | 不是随机 mask 解释 |
| Random-LM same-architecture | 0.7411 | 0.8553 | - | pretrained LM 不是随机 decoder 可替代 |
| ansmask | 0.8178 | 0.9120 | - | missingness-faithful objective |
| null-as-negative | 0.8334 | 0.9133 | - | dense classification baseline，不等于临床正确 |

### 0.3 当前必须改的 claim

| 原 claim | 新写法 |
|---|---|
| LLM semantic manifold 是主要收益来源 | Frozen-LM provides a useful training-time structured objective, but UMS/schema supervision is the stable contributor. |
| SPD 是核心模块 | SPD is unstable; report as sensitivity / negative analysis. |
| 500× less data | 删除或降级。改成 domain-specific 30k structured supervision，不做 apples-to-apples data-efficiency claim。 |
| schema paraphrase robust | 不要写。当前方法依赖 fixed schema serialization。 |
| answerability mask 被 null-as-negative 推翻 | 不成立。null-as-negative 是 aggressive dense classification baseline，不保留 UMS missingness semantics。 |

---

## 1. 总体执行阶段

| 阶段 | 目标 | 主要产物 | 是否重训 | 用途 |
|---|---|---|---|---|
| Phase 0 | 锁定现有证据和主表 | consolidated results / claim map / cost table | 否 | 必须 |
| Phase 1 | 找 frozen-LM 什么时候必要 | LLM necessity map | 部分重训 | 必须 |
| Phase 2 | 根据 failure case 试新模块 | 2-4 个候选模块结果 | 是 | 选成功者进主文 |
| Phase 3 | answerability 与 schema 依赖分析 | missingness semantics + schema robustness | 少量重训/诊断 | 必须 |
| Phase 4 | 论文重构与最终表格 | paper tables + writing checklist | 否/少量 | 必须 |

### 总体停止 / pivot 准则

如果以下情况出现，应停止主打 frozen-LM：

1. no-LM UMS 在大多数设置中稳定追平或超过 frozen-LM UMS；
2. random-LM same-architecture 追平 pretrained frozen-LM；
3. frozen-LM 的收益不能在低数据、复杂 schema、rare/high-null/uncertain subgroup 中体现；
4. external CXR 上 frozen-LM 不优于 no-LM，且 in-domain gain 很小。

若触发 pivot，则论文改为：

> **Answerability-aware structured label decomposition for CXR representation learning**

而不是继续讲 frozen-LM semantic teacher。

---

# Phase 0：结果整理与证据链锁定

## P0-1. 结果汇总表

### 目标

把所有已完成实验汇总成一个统一表，避免后续写作时混乱。

### Codex 任务

```text
Task ID: P0_RESULT_CONSOLIDATION

Goal:
  Consolidate all finished experiment metrics into CSV/Markdown tables.

Inputs:
  outputs/baseline_vit_full14/metrics_final.json
  outputs/lp_A_ums_12label/metrics_final.json
  outputs/lp_ums_classifier_no_llm_12label_full/metrics_final.json
  outputs/lp_A_ums_spd_12label/metrics_final.json
  outputs/lp_spd_g2_12label/metrics_final.json
  outputs/lp_A_freetext_12label/metrics_final.json
  outputs/lp_random_mask_12label/metrics_final.json
  outputs/lp_biomedclip_baseline_seed0/metrics_final.json
  outputs/lp_ums_ansmask_12label/metrics_final.json
  outputs/lp_ums_null_as_negative_12label/metrics_final.json
  outputs/lp_ums_random_lm_12label/metrics_final.json
  NIH cross-domain json files if available

Outputs:
  outputs/final_tables/main_controlled_results.csv
  outputs/final_tables/main_controlled_results.md
  outputs/final_tables/claim_support_matrix.md
  outputs/final_tables/missing_artifacts.md

Success criteria:
  - Every row has method, supervision, LLM type, module, checkpoint path, CheXpert AUC/F1, NIH AUC/F1.
  - Every row marks which claim it supports or weakens.
  - Missing files are reported explicitly; do not silently ignore missing metrics.
```

### 输出表模板

| Method | Data | Supervision | LLM | Module | CheXpert AUC | NIH AUC | Claim supported | Claim weakened |
|---|---|---|---|---|---:|---:|---|---|
| BCE ViT-B | CheXpert 30k | labels | no | none | 0.7927 | - | data-matched baseline | - |
| no-LM UMS | CheXpert 30k | UMS state | no | classifier | 0.8273 | 0.7262 | UMS/schema | LLM dominant |
| frozen-LM UMS | CheXpert 30k | UMS JSON | pretrained | no-SPD | 0.8439 | 0.7068 | frozen-LM in-domain | LLM external dominance |
| random-LM UMS | CheXpert 30k | UMS JSON | random | no-SPD | 0.7411 | - | pretrained > random | - |
| SPD default | CheXpert 30k | UMS JSON | pretrained | SPD | 0.8208 | 0.7214 | weak external signal | SPD stable gain |
| SPD G2 | CheXpert 30k | UMS JSON | pretrained | SPD | 0.8291 | 0.7176 | sensitivity | SPD stable gain |

---

## P0-2. 成本表

### 目标

回应 reviewer 对训练成本的质疑。部署不用 LLM 不等于训练成本可以忽略。

### Codex 任务

```text
Task ID: P0_COST_TABLE

Goal:
  Report training and deployment cost for main methods.

Collect:
  - frozen params during training
  - trainable params
  - peak GPU memory
  - wall-clock time
  - GPU-hours
  - throughput images/sec or steps/sec
  - inference params
  - whether LLM is used at deployment

Outputs:
  outputs/final_tables/cost_table.csv
  outputs/final_tables/cost_table.md

Success criteria:
  - Distinguish training cost from deployment cost.
  - Report LLM-free deployment clearly.
  - Do not claim resource-friendly without reporting training-time overhead.
```

### 表格模板

| Method | Frozen params during train | Trainable params | Peak memory | GPU-hours | Throughput | Deployment params | Deployment LLM? |
|---|---:|---:|---:|---:|---:|---:|---|
| BCE ViT-B | 0 |  |  |  |  | ViT | no |
| no-LM UMS | 0 |  |  |  |  | ViT | no |
| frozen-LM UMS | 1.5B frozen |  |  |  |  | ViT | no |
| random-LM UMS | 1.5B random frozen |  |  |  |  | ViT | no |
| best final method |  |  |  |  |  | ViT | no |

---

# Phase 1：定位 frozen-LM 什么时候必要

Phase 1 是最重要阶段。不要先设计模块，要先回答：

> **Frozen-LM 到底在什么条件下比 no-LM UMS 值得用？**

---

## P1-A. Schema complexity sweep

### 假设

当前 12-label state schema 太像普通 multi-label classification，所以 no-LM UMS 接近 frozen-LM。Frozen-LM 的优势可能出现在更组合、更语言化的 schema 上。

### 实验矩阵

| Schema level | Target 内容 | no-LM | frozen-LM | random-LM | 目的 |
|---|---|---|---|---|---|
| S1 | finding → state | 必跑 | 必跑 | 选跑 | 当前 12-label |
| S2 | finding → state + answerability | 必跑 | 必跑 | 选跑 | missingness-aware |
| S3 | finding → state + uncertainty | 必跑 | 必跑 | 选跑 | ambiguity-aware |
| S4 | finding → state + uncertainty + location/severity | 必跑 | 必跑 | 可选 | compositional schema |
| S5 | finding-anatomy-state-severity-temporality | 可选 | 可选 | 可选 | 最高复杂度 |

### Codex 任务

```text
Task ID: P1_SCHEMA_COMPLEXITY_SWEEP

Goal:
  Compare no-LM UMS vs frozen-LM UMS under increasing schema complexity.

Implement:
  Add `schema_mode` to UMS serializer:
    - state_only
    - state_answerability
    - state_uncertainty
    - state_uncertainty_location_severity
    - compositional_full

Configs:
  configs/schema_sweep/no_lm_state_only.yaml
  configs/schema_sweep/frozen_lm_state_only.yaml
  configs/schema_sweep/no_lm_state_answerability.yaml
  configs/schema_sweep/frozen_lm_state_answerability.yaml
  configs/schema_sweep/no_lm_state_uncertainty.yaml
  configs/schema_sweep/frozen_lm_state_uncertainty.yaml
  configs/schema_sweep/no_lm_compositional.yaml
  configs/schema_sweep/frozen_lm_compositional.yaml

Outputs:
  outputs/schema_sweep/<method>_<schema_mode>/
  outputs/final_tables/schema_complexity_sweep.csv

Metrics:
  - CheXpert macro-AUC / macro-F1
  - NIH macro-AUC if available
  - per-field AUC
  - rare/high-null/uncertain subgroup AUC
  - token NLL for frozen-LM variants
  - counterfactual pairwise accuracy for frozen-LM variants

Success boundary:
  Frozen-LM becomes meaningful if:
    - gain over no-LM increases as schema complexity increases; OR
    - frozen-LM improves complex / rare / uncertain subset by >= +0.01 AUC; OR
    - frozen-LM improves schema counterfactual diagnostics.

Negative boundary:
  If frozen-LM gain stays <= +0.005 across complexity levels, do not claim LLM is necessary.
```

### 主文表模板

| Schema | no-LM AUC | frozen-LM AUC | random-LM AUC | LLM gain | Counterfactual acc. | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| State-only |  |  |  |  |  | label-like |
| State + answerability |  |  |  |  |  | missingness-aware |
| State + uncertainty |  |  |  |  |  | ambiguity-aware |
| Compositional |  |  |  |  |  | language-like |

---

## P1-B. Data-size scaling

### 假设

Frozen-LM 可能不是在 full 30k 最有用，而是在低数据下有用。no-LM classifier 在 30k 下可以学到 schema，小数据时 pretrained LM 可能提供更强先验。

### 实验矩阵

| Data size | BCE | no-LM UMS | frozen-LM UMS | random-LM |
|---:|---|---|---|---|
| 1k | 必跑 | 必跑 | 必跑 | 选跑 |
| 3k | 必跑 | 必跑 | 必跑 | 选跑 |
| 10k | 必跑 | 必跑 | 必跑 | 选跑 |
| 30k | 已有/补齐 | 已有 | 已有 | 已有 |

### Codex 任务

```text
Task ID: P1_DATA_SCALING

Goal:
  Test whether frozen-LM benefit increases in low-data regimes.

Implement:
  Create deterministic CheXpert subsets:
    - 1k
    - 3k
    - 10k
    - 30k
  Use patient-level sampling if patient IDs are available.
  Preserve label distribution when possible.

Runs per subset:
  - BCE ViT-B
  - no-LM UMS
  - frozen-LM UMS no-SPD
  - random-LM for 3k and 30k if compute constrained

Outputs:
  data/splits/chexpert_train_{1k,3k,10k,30k}.csv
  outputs/data_scaling/<method>_<size>/
  outputs/final_tables/data_scaling.csv

Metrics:
  - CheXpert macro-AUC
  - NIH macro-AUC for 10k/30k if possible
  - rare-field macro-AUC
  - AUC vs log(N) slope

Success boundary:
  Frozen-LM is useful if:
    - gain over no-LM is >= +0.015 at 1k/3k; OR
    - frozen-LM reaches no-LM 30k performance with <= 1/3 data.

Negative boundary:
  If no-LM matches or beats frozen-LM at all sizes, LLM necessity is weak.
```

### 图模板

**Figure: Frozen-LM gain vs data size**

- x-axis: pretraining samples, log scale
- y-axis: CheXpert macro-AUC
- curves: BCE, no-LM UMS, frozen-LM UMS, random-LM

---

## P1-C. Field difficulty / rare-high-null-uncertain subgroup

### 假设

Frozen-LM 的价值可能不体现在总 macro-AUC，而体现在 rare / uncertain / high-null fields。

### 字段分组

| Group | Finding examples | Criterion |
|---|---|---|
| Common | Pleural Effusion, Lung Opacity, Support Devices | high answerable/present rate |
| Rare | Fracture, Lung Lesion, Pneumonia | low positive rate |
| Uncertain-heavy | Atelectasis, Consolidation, Pneumonia | high uncertain rate |
| High-null | Fracture, Lung Lesion, Pneumonia, Cardiomegaly | high null rate |

### Codex 任务

```text
Task ID: P1_FIELD_DIFFICULTY_ANALYSIS

Goal:
  Determine whether frozen-LM helps on rare, uncertain, or high-null fields.

Implement:
  Use existing schema answerability statistics and label distribution.
  Group labels into:
    - common
    - rare
    - uncertain-heavy
    - high-null
  Compute per-field AUC/F1 and group macro averages for all main methods.

Inputs:
  prediction logits from main LP runs
  metrics_final.json files
  schema answerability stats

Outputs:
  outputs/final_tables/per_field_results.csv
  outputs/final_tables/grouped_field_results.csv

Metrics:
  - per-field AUC/F1
  - group macro-AUC
  - frozen-LM vs no-LM delta
  - frozen-LM vs BCE delta
  - null/uncertain rate per field

Success boundary:
  Frozen-LM has a real use case if it improves rare/high-null/uncertain groups by >= +0.01 to +0.02 AUC.

Negative boundary:
  If improvement is uniform tiny noise or no-LM wins on all difficult groups, LLM necessity is weak.
```

### 表模板

| Field group | BCE | no-LM UMS | frozen-LM UMS | random-LM | frozen vs no-LM gain | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| Common |  |  |  |  |  |  |
| Rare |  |  |  |  |  |  |
| Uncertain-heavy |  |  |  |  |  |  |
| High-null |  |  |  |  |  |  |

---

## P1-D. Failure case mining

### 目标

找出 frozen-LM 什么时候赢、什么时候输，然后再决定模块。

### Codex 任务

```text
Task ID: P1_LLM_FAILURE_CASE_MINING

Goal:
  Mine cases where frozen-LM and no-LM differ, then categorize failure modes.

Inputs:
  prediction logits from:
    - frozen-LM UMS no-SPD LP
    - no-LM UMS LP
    - BCE LP
    - random-LM LP
  validation/test labels
  original UMS schema jsonl

Implement:
  For each sample and label:
    - mark frozen_correct, no_lm_correct, bce_correct, random_lm_correct
    - compute confidence gap
    - attach field state, answerability, null/uncertain/present status
    - attach image path and schema fields

Outputs:
  outputs/failure_cases/frozen_better_than_no_lm.csv
  outputs/failure_cases/no_lm_better_than_frozen.csv
  outputs/failure_cases/all_methods_fail.csv
  outputs/failure_cases/random_lm_failure_examples.csv
  outputs/final_tables/failure_case_summary.csv

Manual review columns:
  - image_path
  - label
  - ground_truth
  - frozen_prob
  - no_lm_prob
  - bce_prob
  - random_lm_prob
  - UMS_state
  - answerability
  - null_or_uncertain_flag
  - notes

Success boundary:
  If frozen-LM wins disproportionately on uncertain/high-null/rare/compositional cases, use that to motivate adaptive/gated module.

Negative boundary:
  If no-LM wins on most difficult cases and frozen-LM only wins common/easy cases, LLM claim should be downgraded.
```

### 输出表模板

| Failure class | Count | frozen-LM better | no-LM better | common fields | Example path |
|---|---:|---:|---:|---|---|
| rare present missed |  |  |  |  |  |
| uncertain confused |  |  |  |  |  |
| null over-negative |  |  |  |  |  |
| schema paraphrase failure |  |  |  |  |  |
| image-schema mismatch |  |  |  |  |  |

---

## P1-E. External CXR transfer

### 目标

验证 UMS/frozen-LM 是否在 CheXpert 之外稳定迁移。NIH 已有正向，但最好补一个外部 CXR。

### 推荐数据

优先级：

1. MIMIC-CXR
2. PadChest
3. VinDr-CXR
4. NIH target linear probe 补充

### Codex 任务

```text
Task ID: P1_EXTERNAL_CXR_TRANSFER

Goal:
  Test whether UMS/frozen-LM representations transfer beyond CheXpert/NIH.

Preferred datasets:
  - MIMIC-CXR if labels available
  - PadChest if accessible
  - VinDr-CXR if accessible
  - Otherwise keep NIH and add target linear probe, not only direct transfer.

Runs:
  For each backbone:
    - BCE
    - no-LM UMS
    - frozen-LM UMS no-SPD
    - best new module if available

Evaluation modes:
  - frozen backbone linear probe on target dataset
  - CheXpert-trained head direct transfer when label mapping exists

Outputs:
  outputs/external_cxr/<dataset>/<method>/
  outputs/final_tables/external_cxr_transfer.csv

Success boundary:
  UMS story is strong if UMS-family beats BCE/BiomedCLIP/ImageNet on at least one external CXR dataset.

LLM-specific success:
  frozen-LM beats no-LM on external CXR by >= +0.01 AUC.

Negative boundary:
  If only CheXpert improves and all external CXR datasets fail, narrow title to CheXpert representation learning.
```

---

# Phase 2：候选模块池（不沿 SPD 思路）

模块不要提前押注。先做 Phase 1，再根据 failure case 选择 2-4 个模块。

推荐优先组合：

1. **Adaptive LLM Gating**：最贴合“什么时候需要 LLM”。
2. **Hierarchical UMS Head**：最稳，贴合 UMS 主线。
3. **Field/State-Balanced Loss**：成本低，可能提升 rare/high-null。
4. **Field-Query Bottleneck**：可解释、非 SPD 的结构模块，作为备选。

---

## Module A. Adaptive LLM Gating

### 想法

不是所有样本/字段都需要 frozen-LM。让 difficult samples/fields 使用 LLM loss，普通样本使用 no-LM UMS / classifier loss。

### 适用条件

Phase 1 发现 frozen-LM 主要在以下场景有优势：

- low-data
- rare finding
- uncertain-heavy
- high-null
- complex schema
- counterfactual low-margin samples

### 实验清单

| Variant | 内容 | 是否必跑 |
|---|---|---|
| all LLM | 当前 frozen-LM UMS | 已有 |
| no LLM | 当前 no-LM UMS | 已有 |
| rare-field gate | rare/high-null fields 使用 LLM loss | 必跑 |
| uncertain gate | uncertain samples/fields 使用 LLM loss | 必跑 |
| loss-margin gate | counterfactual margin 低时使用 LLM | 选跑 |
| learned gate | 小 MLP 学习 LLM weight | 选跑 |

### Codex 任务

```text
Task ID: P2_ADAPTIVE_LLM_GATING

Goal:
  Use frozen-LM loss selectively for difficult samples/fields instead of all samples.

Implement:
  Add `llm_loss_weight_mode`:
    - all
    - none
    - rare_field
    - high_null_field
    - uncertain_field
    - loss_margin
    - learned_gate

Training loss:
  total_loss = w_llm * lm_token_loss + (1 - w_llm) * ums_classifier_loss

Configs:
  configs/adaptive_gate/all_llm.yaml
  configs/adaptive_gate/no_llm.yaml
  configs/adaptive_gate/rare_field_gate.yaml
  configs/adaptive_gate/high_null_gate.yaml
  configs/adaptive_gate/uncertain_gate.yaml
  configs/adaptive_gate/loss_margin_gate.yaml
  configs/adaptive_gate/learned_gate.yaml

Outputs:
  outputs/adaptive_gate/<variant>/
  outputs/final_tables/adaptive_gate_results.csv

Metrics:
  - CheXpert AUC/F1
  - NIH AUC
  - rare/high-null/uncertain group AUC
  - training GPU-hours
  - fraction of samples/fields using LLM loss
  - token NLL diagnostics on gated subset

Success boundary:
  Keep if:
    - AUC >= frozen-LM no-SPD - 0.002 while using <= 50% LLM loss; OR
    - AUC improves over frozen-LM no-SPD by >= +0.005; OR
    - rare/uncertain group improves by >= +0.01.

Drop boundary:
  If gating gives same/worse AUC and does not reduce cost, drop.
```

---

## Module B. Hierarchical UMS Head

### 想法

UMS 不只作为 JSON target。把 supervision 显式分解：

```text
answerability -> state -> uncertainty -> optional severity/location
```

这测试的是 **supervision decomposition**，不是新视觉正则。

### 实验清单

| Variant | 内容 | 是否必跑 |
|---|---|---|
| Flat BCE | 普通标签监督 | 已有 |
| 4-state UMS head | null / absent / uncertain / present | 已有部分 |
| Hierarchical head | answerability + state + uncertainty | 必跑 |
| Hierarchical + field-balanced | 加字段权重 | 必跑 |
| Hierarchical + LM auxiliary | head loss + frozen-LM loss | 选跑 |

### Codex 任务

```text
Task ID: P2_HIERARCHICAL_UMS_HEAD

Goal:
  Train ViT with decomposed UMS objectives instead of JSON-only or BCE-only supervision.

Implement heads:
  - answerability_head: binary per field
  - state_head: present/absent/uncertain per answerable field
  - null_head or missingness head if needed
  - optional severity/location heads if schema exists

Loss:
  L = λ_ans * BCE(answerability)
    + λ_state * CE(state | answerable)
    + λ_uncertainty * BCE(uncertainty)
    + optional λ_lm * LM token loss

Configs:
  configs/hier_ums/no_lm_hier.yaml
  configs/hier_ums/hier_field_balanced.yaml
  configs/hier_ums/hier_plus_lm.yaml

Outputs:
  outputs/hier_ums/<variant>/
  outputs/final_tables/hier_ums_results.csv

Metrics:
  - CheXpert AUC/F1
  - NIH AUC
  - answerability prediction AUC
  - state accuracy on answerable fields
  - uncertainty F1
  - calibration on high-null fields

Success boundary:
  Keep if:
    - no-LM hierarchical head >= no-LM UMS +0.005 AUC; OR
    - hierarchical + LM >= frozen-LM no-SPD +0.005 AUC; OR
    - missingness/uncertainty calibration clearly improves.

Drop boundary:
  If downstream AUC and answerability calibration both fail, keep current UMS objective.
```

---

## Module C. Field/State-Balanced Loss

### 想法

UMS 的字段和状态分布非常不均衡。先修 loss imbalance，可能比加复杂模块更有效。

### 实验清单

| Variant | 权重 | 是否必跑 |
|---|---|---|
| uniform | 当前 | 已有 |
| inverse field frequency | 1/sqrt(freq) | 必跑 |
| inverse state frequency | present/absent/uncertain/null | 必跑 |
| effective number | class-balanced loss | 选跑 |
| focal UMS loss | hard examples | 选跑 |
| hybrid | field-balanced + focal | 选跑 |

### Codex 任务

```text
Task ID: P2_FIELD_STATE_BALANCED_LOSS

Goal:
  Improve rare/uncertain/high-null field learning via loss weighting.

Implement config:
  loss_weighting:
    mode: uniform | inverse_field | inverse_state | effective_num | focal | hybrid
    gamma: for focal
    beta: for effective number
    max_weight: clip upper bound

Apply to:
  - no-LM UMS classifier
  - frozen-LM UMS token loss
  - hierarchical UMS head if implemented

Configs:
  configs/loss_balance/no_lm_inverse_field.yaml
  configs/loss_balance/no_lm_focal.yaml
  configs/loss_balance/frozen_lm_inverse_field.yaml
  configs/loss_balance/frozen_lm_focal.yaml

Outputs:
  outputs/loss_balance/<variant>/
  outputs/final_tables/loss_balance_results.csv

Metrics:
  - CheXpert macro-AUC
  - rare-field macro-AUC
  - common-field macro-AUC
  - per-field AUC variance
  - NIH AUC

Success boundary:
  Keep if:
    - rare-field AUC improves by >= +0.015 without overall AUC drop >0.003; OR
    - overall AUC improves by >= +0.005.

Drop boundary:
  If it only improves rare fields but macro-AUC drops >0.005, move to appendix.
```

---

## Module D. Field-Query Bottleneck

### 想法

不是 SPD 的 group orthogonality，而是每个 UMS finding 一个 field query：

```text
image tokens -> field queries -> field embeddings -> UMS state / LM prefix
```

### 为什么不是 SPD

- 不做 query group decomposition。
- 不做 orthogonality regularization。
- query 与 clinical field 一一对应，更可解释。

### 实验清单

| Variant | 内容 | 是否必跑 |
|---|---|---|
| CLS/no-SPD | 当前 baseline | 已有 |
| mean pooling | 简单 pooling | 选跑 |
| generic query pooling | N 个 learnable queries，不绑定字段 | 必跑 |
| field-query no-LM | field queries -> state head | 必跑 |
| field-query frozen-LM | field embeddings -> LM visual prefix | 必跑 |

### Codex 任务

```text
Task ID: P2_FIELD_QUERY_BOTTLENECK

Goal:
  Replace SPD with schema-aligned field queries, without orthogonality or group decomposition.

Implement:
  models/field_query_pooler.py

Architecture:
  - Input: ViT patch tokens [B, N, D]
  - Learnable field queries [F, D], F = number of UMS findings
  - Cross-attention: field_queries attend to image tokens
  - Output: field embeddings [B, F, D]
  - Optional:
      a) state classifier head per field
      b) projection to LM prefix tokens
      c) aggregate field embeddings for LP backbone

Configs:
  configs/field_query/fq_no_lm.yaml
  configs/field_query/fq_frozen_lm.yaml
  configs/field_query/generic_query_frozen_lm.yaml
  configs/field_query/cls_frozen_lm.yaml

Outputs:
  outputs/field_query/<variant>/
  outputs/final_tables/field_query_results.csv

Metrics:
  - CheXpert AUC/F1
  - NIH AUC
  - per-field AUC
  - counterfactual field-swap accuracy
  - optional attention visualization per field query

Success boundary:
  Keep if:
    - improves over frozen-LM no-SPD by >= +0.005 CheXpert AUC; OR
    - improves rare/high-null groups by >= +0.01; OR
    - matches no-SPD AUC but improves field-level interpretability / counterfactual field sensitivity.

Drop boundary:
  If underperforms no-SPD by >0.01 and diagnostics do not improve, drop to appendix.
```

---

## Module E. Counterfactual Margin Training

### 想法

把已有 counterfactual schema scoring 变成训练信号：

```text
L = NLL(true_schema) + max(0, margin + NLL(true_schema) - NLL(counterfactual_schema))
```

这不是视觉正则，而是 schema-grounding margin loss。

### 实验清单

| Variant | Negative type |
|---|---|
| state_flip margin | present ↔ absent/uncertain |
| field_swap margin | swap field names |
| image_swap margin | wrong image prefix |
| null_to_present margin | missingness stress |
| mixed margin | all negatives |

### Codex 任务

```text
Task ID: P2_COUNTERFACTUAL_MARGIN_TRAINING

Goal:
  Improve image-schema grounding by training with schema counterfactual negatives.

Implement:
  For each batch:
    - positive schema z+
    - construct one negative schema z-
    - compute token NLL for z+ and z-
    - margin_loss = max(0, margin + NLL(z+) - NLL(z-))
    - total_loss = lm_loss + lambda_margin * margin_loss

Negative modes:
  - state_flip
  - field_swap
  - image_swap
  - null_to_present
  - mixed

Configs:
  configs/cf_margin/state_flip.yaml
  configs/cf_margin/image_swap.yaml
  configs/cf_margin/mixed.yaml

Outputs:
  outputs/cf_margin/<variant>/
  outputs/final_tables/cf_margin_results.csv

Metrics:
  - CheXpert AUC
  - NIH AUC
  - counterfactual pairwise accuracy
  - visual-prefix dependency NLL gap
  - field paraphrase robustness

Success boundary:
  Keep if:
    - counterfactual pairwise accuracy improves by >= +0.05 AND CheXpert AUC drop <= 0.005; OR
    - AUC improves >= +0.005 and diagnostics improve.

Drop boundary:
  If diagnostics improve but AUC drops >0.01, report only as diagnostic appendix.
```

---

## Module F. Schema Augmentation / Canonicalization

### 想法

当前模型强依赖 fixed schema serialization。目标不是完全解决，而是做一个轻量 mitigation，降低 reviewer 风险。

### 实验清单

| Variant | 内容 | 是否必跑 |
|---|---|---|
| original fixed schema | 当前 | 已有 |
| order dropout | field order 随机 | 选跑 |
| alias augmentation | clinical synonym 替换 | 选跑 |
| canonical field ID | field_id + clinical name | 选跑 |
| mixed serialization | JSON / compact / kv list | 选跑 |
| schema dropout | subset fields | 选跑 |

### Codex 任务

```text
Task ID: P2_SCHEMA_AUGMENTATION

Goal:
  Reduce over-dependence on fixed schema serialization while preserving AUC.

Implement serializer augmentations:
  - random_field_order_prob
  - alias_replace_prob
  - include_field_id
  - serialization_format: json | kv_list | compact
  - schema_dropout_prob

Alias examples:
  Pleural Effusion -> Pleural fluid
  Cardiomegaly -> Enlarged heart
  Pneumothorax -> Collapsed lung
  Edema -> Fluid overload pattern

Configs:
  configs/schema_aug/order_dropout.yaml
  configs/schema_aug/alias_aug.yaml
  configs/schema_aug/order_alias_aug.yaml
  configs/schema_aug/canonical_field_id.yaml

Outputs:
  outputs/schema_aug/<variant>/
  outputs/final_tables/schema_aug_results.csv

Diagnostics:
  - eval_schema_key_robustness.py
  - eval_field_paraphrase_robustness.py
  - CheXpert LP
  - token NLL original vs paraphrase

Success boundary:
  Keep if:
    - clinical paraphrase NLL gap reduces by >= 30%; AND
    - CheXpert AUC drops by <= 0.005.

Drop boundary:
  If AUC drops >0.01 for modest robustness gain, keep fixed schema and report limitation.
```

---

## Module G. UMS Prototype State Space

### 想法

用 `(field, state)` prototypes 代替完整 frozen-LM token prediction，作为 no-LM 与 frozen-LM 之间的桥梁。

### 实验清单

| Variant | 内容 |
|---|---|
| global state prototypes | present/absent/uncertain/null 共用 |
| field-state prototypes | 每个 finding 独立 prototypes |
| LM-initialized prototypes | 用 frozen-LM embeddings 初始化 |
| random prototypes | 随机初始化 |
| prototype + LM auxiliary | 同时训练 |

### Codex 任务

```text
Task ID: P2_UMS_PROTOTYPE_SPACE

Goal:
  Learn a compact field-state semantic space without relying fully on frozen-LM token prediction.

Implement:
  Prototype table:
    P[field_id, state_id, D]
  Image encoder outputs field embedding:
    E[field_id, D]
  Loss:
    - CE over cosine similarity E_i vs P_i,state
    - optional supervised contrastive loss across same field/state
    - optional initialize P from frozen-LM embeddings of field/state text

Configs:
  configs/prototype/random_proto.yaml
  configs/prototype/lm_initialized_proto.yaml
  configs/prototype/proto_plus_lm.yaml

Outputs:
  outputs/prototype/<variant>/
  outputs/final_tables/prototype_results.csv

Metrics:
  - CheXpert AUC/F1
  - NIH AUC
  - per-field AUC
  - prototype retrieval / nearest state accuracy
  - comparison to no-LM UMS and frozen-LM UMS

Success boundary:
  Keep if:
    - LM-initialized prototypes beat random prototypes by >= +0.005 AUC; OR
    - prototype method matches frozen-LM within 0.003 AUC at much lower training cost.

Drop boundary:
  If prototypes collapse or match no-LM with no benefit, drop.
```

---

## Module H. Cost-Aware LLM Schedule

### 想法

减少 LLM 使用成本：不是每一步都跑 full LLM。

### 实验清单

| Variant | 内容 |
|---|---|
| full LLM | 每步 LLM |
| warmup 20% then no-LM | 前 20% 用 LLM |
| periodic k | 每 k step 用一次 LLM |
| prototype distill | LLM 初始化 prototype 后不用 LLM |
| adaptive gate | 只在 difficult cases 用 LLM |

### Codex 任务

```text
Task ID: P2_COST_AWARE_LLM_SCHEDULE

Goal:
  Reduce training cost while preserving frozen-LM benefits.

Implement:
  Add `llm_schedule`:
    - always
    - warmup_then_off
    - periodic_k
    - prototype_distill
    - adaptive_gate

Configs:
  configs/cost_schedule/always.yaml
  configs/cost_schedule/warmup_20pct.yaml
  configs/cost_schedule/periodic_4.yaml
  configs/cost_schedule/prototype_distill.yaml

Outputs:
  outputs/cost_schedule/<variant>/
  outputs/final_tables/cost_schedule_results.csv

Metrics:
  - CheXpert AUC
  - NIH AUC
  - GPU-hours
  - peak memory
  - throughput
  - fraction of steps with LLM

Success boundary:
  Keep if:
    - AUC within 0.003 of full frozen-LM while reducing GPU-hours by >= 30%; OR
    - AUC improves or matches full while cost drops.

Drop boundary:
  If cost drops but AUC drops >0.01, keep only as efficiency appendix.
```

---

# Phase 3：answerability 与 schema serialization

---

## P3-A. Answerability semantics

### 核心判断

`null-as-negative` AUC 更高，不能说明 answerability mask 不成立。它只能说明：

> dense negative supervision 对 CheXpert classification AUC 有利。

但这不等于：

> null 在临床语义上等于 absent。

因此论文应区分：

| 问题 | 当前答案 |
|---|---|
| 哪个 objective 分类 AUC 更高？ | null-as-negative 更高 |
| 哪个 objective 更保留 UMS missingness semantics？ | answerability mask 更合理 |

### Codex 任务

```text
Task ID: P3_ANSWERABILITY_SEMANTICS

Goal:
  Separate classification performance from missingness-faithful supervision.

Evaluate:
  - no mask / null kept
  - answerability mask
  - null-as-negative

Metrics:
  - CheXpert AUC/F1
  - high-null field AUC
  - predicted absent rate on null fields
  - calibration on null-heavy fields
  - uncertain-field behavior
  - optional manual audit agreement

Inputs:
  outputs/lp_A_ums_12label/metrics_final.json
  outputs/lp_ums_ansmask_12label/metrics_final.json
  outputs/lp_ums_null_as_negative_12label/metrics_final.json
  schema answerability stats
  optional manual audit csv

Outputs:
  outputs/final_tables/answerability_semantics.csv
  outputs/final_tables/null_field_calibration.csv
  outputs/failure_cases/null_as_negative_over_absent.csv

Success boundary for answerability:
  Answerability is supported if:
    - it reduces over-confident absent predictions on null-heavy fields; OR
    - manual audit shows many nulls are not true negatives; OR
    - calibration improves even if AUC is lower.

Negative boundary:
  If null fields are mostly true negatives by audit and null-as-negative improves AUC + calibration, answerability should be downgraded.
```

### 表模板

| Objective | AUC | High-null AUC | Null calibration | Semantics preserved? | Interpretation |
|---|---:|---:|---:|---|---|
| no mask / null kept | 0.8439 |  |  | partial | strong baseline |
| answerability mask | 0.8178 |  |  | yes | missingness-faithful |
| null-as-negative | 0.8334 |  |  | questionable | dense classification baseline |

### 推荐论文写法

```text
Although null-as-negative improves downstream CheXpert AUC, it changes the intended semantics of null fields. We therefore treat it as a dense classification-oriented baseline rather than the default clinically faithful UMS objective.
```

---

## P3-B. Schema dependency writeup

### 当前事实

当前方法强依赖 fixed schema serialization。不要写 schema-agnostic language understanding。

### Codex 任务

```text
Task ID: P3_SCHEMA_DEPENDENCY_WRITEUP

Goal:
  Convert schema robustness diagnostics into a paper-ready limitation and optional mitigation.

Inputs:
  outputs/schema_key_robustness_A_ums_12label_128.json
  outputs/field_paraphrase_robustness_A_ums_12label_128.json

Outputs:
  outputs/final_tables/schema_dependency_diagnostics.csv
  outputs/final_tables/schema_dependency_case_study.md

Required table columns:
  - variant
  - original_nll
  - variant_nll
  - margin
  - original_better_rate
  - interpretation

Success boundary:
  Paper-ready if:
    - limitation is quantified
    - claim cleanup explicitly says fixed-schema deployment
    - no statement implies arbitrary paraphrase robustness
```

### 表模板

| Perturbation | Original NLL | Variant NLL | Original better rate | Interpretation |
|---|---:|---:|---:|---|
| reversed order | 0.0543 | 0.5887 | 1.0 | order-sensitive |
| shuffled order | 0.0543 | 0.4926 | 1.0 | order-sensitive |
| clinical key shift | 0.0543 | 0.2664 | 1.0 | key-sensitive |
| generic keys | 0.0543 | 0.4131 | 1.0 | field-name-sensitive |
| clinical paraphrase | 0.0482 | 1.1919 | 1.0 | not paraphrase robust |
| lay paraphrase | 0.0482 | 1.3334 | 1.0 | not paraphrase robust |

### 推荐论文写法

```text
Our objective is schema-supervised rather than schema-agnostic. The model is intentionally trained against a fixed clinical serialization; paraphrased or reordered schemas substantially increase teacher-forcing NLL. Therefore, the current system should be interpreted as a fixed-schema interface for CXR representation learning, not as unconstrained natural-language schema understanding.
```

---

# Phase 4：最终论文表格包

主文建议只放 5-6 张核心表/图，其他放 appendix。

## Table 1. Main controlled results

| Method | Data | Supervision | LLM | Module | CheXpert AUC | NIH AUC | Claim |
|---|---|---|---|---|---:|---:|---|
| BCE | CheXpert 30k | labels | no | none | 0.7927 | - | baseline |
| free-text | CheXpert 30k | free text | frozen | no-SPD | 0.8126 | 0.6365 | text baseline |
| no-LM UMS | CheXpert 30k | UMS state | no | classifier | 0.8273 | 0.7262 | schema contribution |
| frozen-LM UMS | CheXpert 30k | UMS JSON | pretrained | no-SPD | 0.8439 | 0.7068 | LLM variant |
| random-LM UMS | CheXpert 30k | UMS JSON | random | no-SPD | 0.7411 | - | pretrained LM control |
| best new module | CheXpert 30k | UMS | depends | new |  |  | final method |

## Table 2. When is frozen-LM needed?

| Condition | no-LM | frozen-LM | random-LM | Gain | Interpretation |
|---|---:|---:|---:|---:|---|
| full 30k | 0.8273 | 0.8439 | 0.7411 | +0.0167 | modest in-domain gain |
| 10k |  |  |  |  |  |
| 3k |  |  |  |  |  |
| 1k |  |  |  |  |  |
| rare fields |  |  |  |  |  |
| uncertain fields |  |  |  |  |  |
| complex schema |  |  |  |  |  |

## Table 3. Module candidate results

| Module | Motivation | CheXpert AUC | NIH AUC | Rare AUC | Cost | Decision |
|---|---|---:|---:|---:|---:|---|
| Adaptive LLM gating | use LLM where needed |  |  |  |  | keep/drop |
| Hierarchical UMS head | decomposed supervision |  |  |  |  | keep/drop |
| Field/state-balanced loss | long-tail correction |  |  |  |  | keep/drop |
| Field-query bottleneck | schema-aligned visual evidence |  |  |  |  | keep/drop |
| Counterfactual margin | grounding improvement |  |  |  |  | appendix/main |
| Schema augmentation | robustness mitigation |  |  |  |  | appendix/main |

## Table 4. Answerability semantics

| Objective | AUC | High-null AUC | Null calibration | Semantics preserved | Interpretation |
|---|---:|---:|---:|---|---|
| no mask / null kept | 0.8439 |  |  | partial | strong baseline |
| answerability mask | 0.8178 |  |  | yes | faithful missingness |
| null-as-negative | 0.8334 |  |  | questionable | dense classification baseline |

## Table 5. Schema dependency diagnostics

| Perturbation | Original NLL | Variant NLL | Original better rate | Interpretation |
|---|---:|---:|---:|---|
| reversed order | 0.0543 | 0.5887 | 1.0 | order-sensitive |
| shuffled order | 0.0543 | 0.4926 | 1.0 | order-sensitive |
| clinical key shift | 0.0543 | 0.2664 | 1.0 | key-sensitive |
| generic keys | 0.0543 | 0.4131 | 1.0 | field-name-sensitive |
| clinical paraphrase | 0.0482 | 1.1919 | 1.0 | not paraphrase robust |
| lay paraphrase | 0.0482 | 1.3334 | 1.0 | not paraphrase robust |

## Table 6. Cost table

| Method | Training LLM? | Frozen params | Trainable params | GPU-hours | Peak memory | Deployment model | Deployment LLM? |
|---|---|---:|---:|---:|---:|---|---|
| BCE | no | 0 |  |  |  | ViT | no |
| no-LM UMS | no | 0 |  |  |  | ViT | no |
| frozen-LM UMS | yes | 1.5B |  |  |  | ViT | no |
| adaptive LLM | partial | 1.5B partial |  |  |  | ViT | no |
| best final | depends |  |  |  |  | ViT | no |

---

# 5. 写作重构

## 5.1 新标题候选

最稳：

```text
VIVID-Med: Schema-Aware Structured Supervision for Deployable Chest X-ray ViTs
```

如果一定保留 LLM：

```text
VIVID-Med: Schema-Guided Frozen-LM Supervision for Deployable Chest X-ray ViTs
```

## 5.2 新 contribution

### Contribution 1：UMS structured supervision

```text
We convert CXR findings into an answerability-aware field-state schema and show that structured UMS supervision outperforms flat BCE and free-text supervision.
```

### Contribution 2：Controlled frozen-LM analysis

```text
We show that pretrained frozen-LM supervision provides modest in-domain gains over no-LM UMS and is substantially stronger than a random same-architecture LM.
```

### Contribution 3：Failure-driven module

根据 Phase 1 结果选择一个：

- Adaptive LLM Gating
- Hierarchical UMS Head
- Field/state-balanced loss
- Field-query bottleneck

写法：

```text
Motivated by failure-case analysis, we introduce a lightweight mechanism that targets the settings where structured/frozen-LM supervision is most useful.
```

### Contribution 4：Transparent limitations

```text
We quantify fixed-schema dependency, answerability/null semantics, and training/deployment cost rather than claiming schema-agnostic language understanding.
```

## 5.3 Abstract 新逻辑

不要先打 BiomedCLIP 500×。改成：

1. CXR representation learning 里 flat labels 太粗，free text noisy。
2. UMS/schema 把 findings 转成 answerability-aware field-state supervision。
3. 我们系统比较 no-LM、pretrained frozen-LM、random-LM、free-text、BCE。
4. 结果显示 UMS 是稳定主贡献；pretrained frozen-LM 在部分 setting 有增益且强于 random-LM；SPD 不稳定。
5. 最终部署只保留 ViT。

## 5.4 Related work 必须新增

新增小节：

```text
Frozen LLMs for visual representation learning
```

必须明确区分：

1. **LLM as visual encoder layer**：你的 LLM 不作为 deployed visual encoder。
2. **Image-to-language tokenizer**：你的目标不是让 LLM 最终理解图像，而是蒸馏出 standalone ViT。
3. **Visual instruction pretraining**：你的核心是 UMS / answerability / fixed-schema structured supervision，不是通用 VLM instruction tuning。

---

# 6. 推荐执行优先级

## Priority 0：必须先完成

1. `P0_RESULT_CONSOLIDATION`
2. `P0_COST_TABLE`
3. `P1_FIELD_DIFFICULTY_ANALYSIS`（先不重训，直接用预测结果）
4. `P1_DATA_SCALING`（1k/3k/10k/30k）
5. `P1_SCHEMA_COMPLEXITY_SWEEP`（至少 S1/S2/S3）
6. `P3_ANSWERABILITY_SEMANTICS`
7. `P3_SCHEMA_DEPENDENCY_WRITEUP`

## Priority 1：选 2-3 个模块

推荐组合：

1. `P2_ADAPTIVE_LLM_GATING`
2. `P2_HIERARCHICAL_UMS_HEAD`
3. `P2_FIELD_STATE_BALANCED_LOSS`

备选：

4. `P2_FIELD_QUERY_BOTTLENECK`
5. `P2_COUNTERFACTUAL_MARGIN_TRAINING`

## Priority 2：有余力再做

1. `P1_EXTERNAL_CXR_TRANSFER`
2. `P2_SCHEMA_AUGMENTATION`
3. `P2_COST_AWARE_LLM_SCHEDULE`
4. `P2_UMS_PROTOTYPE_SPACE`
5. backbone scaling：ViT-S / ViT-B / optional ViT-L

---

# 7. Codex 顶层任务清单

```text
Project: VIVID-Med revision experiments

Global rules:
  - Do not implement new SPD variants.
  - Treat SPD only as historical baseline / appendix / negative sensitivity.
  - All new modules must be motivated by LLM necessity or failure-case analysis.
  - Every experiment must write:
      metrics_final.json
      config copy
      checkpoint path
      runtime/cost summary
      failure log if interrupted
  - All final tables go to outputs/final_tables/.
  - Do not silently ignore missing artifacts.

Phase 0:
  P0_RESULT_CONSOLIDATION
  P0_COST_TABLE
  P0_CLAIM_SUPPORT_MATRIX

Phase 1:
  P1_SCHEMA_COMPLEXITY_SWEEP
  P1_DATA_SCALING
  P1_FIELD_DIFFICULTY_ANALYSIS
  P1_LLM_FAILURE_CASE_MINING
  P1_EXTERNAL_CXR_TRANSFER optional

Phase 2 candidates:
  P2_ADAPTIVE_LLM_GATING
  P2_HIERARCHICAL_UMS_HEAD
  P2_FIELD_STATE_BALANCED_LOSS
  P2_FIELD_QUERY_BOTTLENECK optional
  P2_COUNTERFACTUAL_MARGIN_TRAINING optional
  P2_SCHEMA_AUGMENTATION optional
  P2_UMS_PROTOTYPE_SPACE optional
  P2_COST_AWARE_LLM_SCHEDULE optional

Phase 3:
  P3_ANSWERABILITY_SEMANTICS
  P3_SCHEMA_DEPENDENCY_WRITEUP

Final outputs:
  outputs/final_tables/main_controlled_results.csv
  outputs/final_tables/llm_necessity.csv
  outputs/final_tables/module_candidates.csv
  outputs/final_tables/answerability_semantics.csv
  outputs/final_tables/schema_dependency_diagnostics.csv
  outputs/final_tables/cost_table.csv
  outputs/final_tables/claim_support_matrix.md
```

---

# 8. 最后执行建议

最推荐的执行顺序：

1. **先整理结果表和 cost table。** 这是写作和后续实验的底座。
2. **先做 field difficulty analysis。** 可能不需要重训，能快速告诉你 frozen-LM 到底赢在哪里。
3. **再做 low-data scaling 和 schema complexity sweep。** 这是决定 LLM 主线能不能保留的关键。
4. **根据 Phase 1 结果选模块。** 如果 frozen-LM 只在 rare/high-null/uncertain 赢，优先做 Adaptive LLM Gating；如果 UMS 分解本身更重要，优先做 Hierarchical UMS Head + balanced loss。
5. **不要继续围绕 SPD 烧卡。** SPD 只放 appendix / negative sensitivity。
6. **answerability 不按 AUC 一票否决。** 它的价值是 missingness semantics，不是一定要赢 null-as-negative。
7. **schema serialization 依赖要承认。** 不必完全解决，但必须量化，并写成 fixed-schema interface。

最终目标不是证明原始版本每个 claim 都成立，而是重构成一个更稳的论文：

> **UMS/schema-aware structured supervision is the main contribution; frozen-LM is useful in controlled, identifiable scenarios; the final method is deployable because the LLM is training-only; limitations are explicitly quantified rather than hidden.**

---

# 9. 执行日志

## 2026-06-17 Phase 0 / P0_RESULT_CONSOLIDATION 执行前记录

### 任务计划

1. 只做结果整理，不启动重训，不继续 SPD 新变体。
2. 核对 P0_RESULT_CONSOLIDATION 指定的 `metrics_final.json`、checkpoint 和 NIH/cross-domain artifact。
3. 新增可重复运行的汇总脚本，输出：
   - `outputs/final_tables/main_controlled_results.csv`
   - `outputs/final_tables/main_controlled_results.md`
   - `outputs/final_tables/claim_support_matrix.md`
   - `outputs/final_tables/missing_artifacts.md`
4. 对缺失 artifact 显式记录，不静默跳过。
5. 完成后把结果、关键指标、失败/缺失原因和下一步写回本执行日志。

### 计划命令

```powershell
python scripts/consolidate_revision_results.py
```

### 输入

- `outputs/baseline_vit_full14/metrics_final.json`
- `outputs/lp_A_ums_12label/metrics_final.json`
- `outputs/lp_ums_classifier_no_llm_12label_full/metrics_final.json`
- `outputs/lp_A_ums_spd_12label/metrics_final.json`
- `outputs/lp_spd_g2_12label/metrics_final.json`
- `outputs/lp_A_freetext_12label/metrics_final.json`
- `outputs/lp_random_mask_12label/metrics_final.json`
- `outputs/lp_biomedclip_baseline_seed0/metrics_final.json`
- `outputs/lp_ums_ansmask_12label/metrics_final.json`
- `outputs/lp_ums_null_as_negative_12label/metrics_final.json`
- `outputs/lp_ums_random_lm_12label/metrics_final.json`
- 每个方法目录下可用的 `nih_crossdomain.json`
- 每个方法目录下可用的 `best.pt`、`final.pt`、`step_*.pt`

### 输出

- `outputs/final_tables/main_controlled_results.csv`
- `outputs/final_tables/main_controlled_results.md`
- `outputs/final_tables/claim_support_matrix.md`
- `outputs/final_tables/missing_artifacts.md`

### 停止条件

- 汇总脚本无法解析现有 metric schema。
- 输出表缺少 P0 success criteria 要求的核心列。
- 发现输入 artifact 缺失但没有写入 `missing_artifacts.md`。
- 任何步骤需要下载模型、访问外部 API 或启动重训。

## 2026-06-17 Phase 0 / P0_RESULT_CONSOLIDATION 执行后记录

### 实际命令

```powershell
python scripts/consolidate_revision_results.py
```

### 生成文件

- `scripts/consolidate_revision_results.py`
- `outputs/final_tables/main_controlled_results.csv`
- `outputs/final_tables/main_controlled_results.md`
- `outputs/final_tables/claim_support_matrix.md`
- `outputs/final_tables/missing_artifacts.md`

### 结果摘要

- 成功汇总 `11` 个方法行。
- 记录 `5` 个缺失/溯源问题。
- 主要指标：
  - Data-matched BCE ViT-B：CheXpert AUC `0.7927`，F1 `0.8987`。
  - Frozen-LM UMS / no-SPD：CheXpert AUC `0.8439`，F1 `0.9095`；NIH AUC `0.7068`，F1 `0.2453`。
  - no-LM UMS state classifier：CheXpert AUC `0.8273`，F1 `0.9143`；NIH AUC `0.7262`，F1 `0.2630`。
  - Frozen-LM UMS + SPD default：CheXpert AUC `0.8208`，F1 `0.9114`；NIH AUC `0.7214`，F1 `0.2620`。
  - Frozen-LM UMS + SPD G=2：CheXpert AUC `0.8291`，F1 `0.9149`；NIH AUC `0.7176`，F1 `0.2627`。
  - Frozen-LM free-text target：CheXpert AUC `0.8126`，F1 `0.9064`；NIH AUC `0.6365`，F1 `0.2201`。
  - Random-mask proxy：CheXpert AUC `0.7387`，F1 `0.8876`；NIH AUC `0.6002`，F1 `0.1940`。
  - BiomedCLIP baseline：CheXpert AUC `0.8076`，F1 `0.9040`；NIH AUC `0.6730`，F1 `0.2235`。
  - Answerability mask：CheXpert AUC `0.8178`，F1 `0.9120`。
  - Null-as-negative：CheXpert AUC `0.8334`，F1 `0.9133`。
  - Random-LM same-architecture：CheXpert AUC `0.7411`，F1 `0.8553`。

### 缺失 artifact / 失败原因

- 未发生脚本失败；没有启动训练、下载模型或访问外部 API。
- `outputs/baseline_vit_full14/nih_crossdomain.json` 缺失，因此 BCE baseline 的 NIH 指标保留为空并写入 `missing_artifacts.md`。
- BCE baseline 的原始 config 未能定位，写入 provenance 缺失。
- `outputs/lp_ums_ansmask_12label/nih_crossdomain.json` 缺失。
- `outputs/lp_ums_null_as_negative_12label/nih_crossdomain.json` 缺失。
- `outputs/lp_ums_random_lm_12label/nih_crossdomain.json` 缺失。

### 初步解释边界

- P0 只整理现有证据，不新增 claim。
- Frozen-LM UMS / no-SPD 是当前 CheXpert AUC 最强行，但 NIH 上 no-LM UMS 更高；这削弱“LLM 外部迁移主导”叙事。
- no-LM UMS 的 CheXpert F1 和 NIH 指标强，支持 UMS/schema 是稳定贡献。
- Random-LM 明显低于 pretrained frozen-LM，支持 pretrained LM 不是 architecture-only control。
- Answerability mask 与 null-as-negative 这里只能作为分类指标对比；missingness semantics 需要进入 `P3_ANSWERABILITY_SEMANTICS` 再判断。
- SPD 仅作为历史 baseline / negative sensitivity，不继续开发新变体。

### 下一步

进入 `P0_COST_TABLE`：先写执行前记录，再从现有 checkpoint/config/log 中尽量抽取训练/部署成本；不能从现有 artifact 可靠恢复的字段必须标为 missing/unknown，不能补写猜测值。

## 2026-06-17 Phase 0 / P0_COST_TABLE 执行前记录

### 任务计划

1. 不启动训练，不运行 GPU job，不下载模型。
2. 从现有 `configs/`、`outputs/<run>/`、`outputs/logs/` 和 checkpoint 元数据中抽取成本相关信息。
3. 区分 training cost 与 deployment cost；明确标注 deployment 是否使用 LLM。
4. 能可靠解析的字段写入表格，不能可靠恢复的字段写 `unknown` 或 `missing_artifact`。
5. 输出成本表，并把缺失字段和下一步写回本日志。

### 计划命令

```powershell
python scripts/consolidate_cost_table.py
```

### 输入

- `outputs/final_tables/main_controlled_results.csv`
- `configs/*.yaml`
- `outputs/<method>/best.pt`
- `outputs/<method>/final.pt`
- `outputs/<method>/step_*.pt`
- `outputs/logs/*`（如存在可匹配运行日志）

### 输出

- `outputs/final_tables/cost_table.csv`
- `outputs/final_tables/cost_table.md`
- `outputs/final_tables/cost_missing_artifacts.md`

### 停止条件

- 需要启动重训/GPU job 才能获得字段。
- 需要下载模型或访问外部 API。
- checkpoint/config/log 解析失败且无法安全标记为 unknown。
- 表格不能清楚区分 training-time LLM 与 deployment-time LLM。

## 2026-06-17 Phase 0 / P0_COST_TABLE 执行后记录

### 实际命令

```powershell
python scripts/consolidate_cost_table.py
```

### 生成文件

- `scripts/consolidate_cost_table.py`
- `outputs/final_tables/cost_table.csv`
- `outputs/final_tables/cost_table.md`
- `outputs/final_tables/cost_missing_artifacts.csv`
- `outputs/final_tables/cost_missing_artifacts.md`

### 结果摘要

- 成功生成 `11` 行成本表。
- 记录 `51` 个缺失成本/溯源字段。
- 所有 frozen-LM / random-LM 方法均明确区分：
  - training-time LLM：`Qwen2.5-1.5B` frozen 或 random frozen。
  - deployment-time LLM：`no`，部署只需要 ViT/backbone + linear probe head。
- 从 config 可恢复的共同训练设置：
  - 主要 representation run：`max_steps=10000`，effective batch `32`，`bf16`。
  - LP/evaluation run：`max_steps=3000`，effective batch `32`。
- 从日志可恢复的 observed source runtime：
  - no-LM UMS state classifier：`1.256` 小时，`10000/10000`，`2.21it/s`。
  - Frozen-LM UMS + SPD G=2：`15.626` 小时，`10000/10000`，`5.63s/it`。
  - Answerability mask：`0.336` 小时，`100/100`，`12.09s/it`；这是可见 resume/log segment，不代表完整训练成本。
  - Null-as-negative：`8.901` 小时，`4000/4000`，`8.01s/it`；这是可见 log segment。
  - Random-LM same-architecture：`2.483` 小时，`500/500`，`17.88s/it`；这是可见 final recovery segment，不代表完整排队/恢复总耗时。

### 缺失 artifact / 失败原因

- 没有发生脚本失败；没有启动训练、下载模型或访问外部 API。
- `metrics_final.json` 不包含 runtime/cost/checkpoint/config 字段，因此成本信息只能从 config/log/checkpoint 文件恢复。
- 多数早期 February run 缺少可匹配训练日志，source wall-clock / throughput / GPU-hours 标为 unknown。
- peak GPU memory 没有统一日志记录，全部标为 `not_logged`。
- trainable params / deployment params 未在当前 artifact 中以结构化数值保存，表中标为 `not_logged`，没有用手工估计值替代。
- BCE baseline 的训练 config 仍未定位，保留 provenance 缺失。

### 下一步

Phase 0 的 P0 产物已经覆盖 result consolidation、claim support matrix 和 cost table。下一步进入 Priority 0 中不需要重训的 `P1_FIELD_DIFFICULTY_ANALYSIS`，先写执行前记录，再用现有 per-label metrics 和 schema answerability stats 计算 rare/high-null/uncertain/common group 表。

## 2026-06-17 Phase 1 / P1_FIELD_DIFFICULTY_ANALYSIS 执行前记录

### 任务计划

1. 不重训，不启动 GPU job。
2. 从现有 `metrics_final.json` 的 `metrics.per_label` 抽取每个 field 的 AUC/F1/support。
3. 从 `outputs/schema_answerability_chexpert_train.csv` 和 `outputs/schema_answerability_chexpert_val.csv` 抽取 present/null/uncertain/answerable rate。
4. 按计划中的字段组计算 group macro-AUC / macro-F1：
   - common：`Pleural Effusion`, `Lung Opacity`, `Support Devices`
   - rare：`Fracture`, `Lung Lesion`, `Pneumonia`
   - uncertain-heavy：`Atelectasis`, `Consolidation`, `Pneumonia`
   - high-null：`Fracture`, `Lung Lesion`, `Pneumonia`, `Cardiomegaly`
5. 输出 frozen-LM vs no-LM、frozen-LM vs BCE、pretrained frozen-LM vs random-LM 的 group-level delta。
6. 明确记录：当前任务做 field/group 级分析；如果缺少 sample-level logits，不做 confidence-gap failure mining。

### 计划命令

```powershell
python scripts/analyze_field_difficulty.py
```

### 输入

- `outputs/final_tables/main_controlled_results.csv`
- `outputs/*/metrics_final.json`
- `outputs/schema_answerability_chexpert_train.csv`
- `outputs/schema_answerability_chexpert_val.csv`

### 输出

- `outputs/final_tables/per_field_results.csv`
- `outputs/final_tables/per_field_results.md`
- `outputs/final_tables/grouped_field_results.csv`
- `outputs/final_tables/grouped_field_results.md`
- `outputs/final_tables/field_difficulty_summary.md`
- `outputs/final_tables/field_difficulty_missing_artifacts.md`

### 停止条件

- 关键方法的 `metrics.per_label` 无法解析。
- schema answerability stats 缺失且无法标注 null/uncertain/rare group provenance。
- 输出缺少 frozen-LM vs no-LM delta。
- 分析需要 sample-level logits 或重训才能继续。

## 2026-06-17 Phase 1 / P1_FIELD_DIFFICULTY_ANALYSIS 执行后记录

### 实际命令

```powershell
python scripts/analyze_field_difficulty.py
```

### 生成文件

- `scripts/analyze_field_difficulty.py`
- `outputs/final_tables/per_field_results.csv`
- `outputs/final_tables/per_field_results.md`
- `outputs/final_tables/grouped_field_results.csv`
- `outputs/final_tables/grouped_field_results.md`
- `outputs/final_tables/field_difficulty_summary.md`
- `outputs/final_tables/field_difficulty_missing_artifacts.csv`
- `outputs/final_tables/field_difficulty_missing_artifacts.md`

### 结果摘要

- 成功生成 `154` 行 per-field 结果和 `4` 行 field-group 结果。
- 没有缺失 required artifact。
- Group-level AUC：
  - common：BCE `0.8087`，no-LM `0.8422`，frozen-LM `0.8384`，random-LM `0.7317`；frozen - no-LM = `-0.0038`。
  - rare：BCE `0.7839`，no-LM `0.7277`，frozen-LM `0.7714`，random-LM `0.7489`；frozen - no-LM = `+0.0437`。
  - uncertain-heavy：BCE `0.8606`，no-LM `0.8386`，frozen-LM `0.8541`，random-LM `0.8215`；frozen - no-LM = `+0.0155`。
  - high-null：BCE `0.8152`，no-LM `0.7753`，frozen-LM `0.8076`，random-LM `0.7642`；frozen - no-LM = `+0.0323`。

### 初步解释边界

- frozen-LM 的相对价值不体现在 common fields；common 上 no-LM 略高。
- frozen-LM 相对 no-LM 在 rare/high-null/uncertain-heavy groups 上有可识别优势，支持后续用这些场景定义 frozen-LM 使用条件或 adaptive gating 动机。
- 这不是“LLM 全面主导”的证据：BCE 在 rare、uncertain-heavy、high-null group 上仍可超过 frozen-LM，因此论文应写成 controlled / field-specific benefit。
- random-LM 在 rare group 上不算崩溃，但 frozen-LM 仍高 `+0.0225`；需要结合整体主表的 random-LM 低性能一起解释。
- 本任务没有 sample-level logits，因此不能替代 `P1_LLM_FAILURE_CASE_MINING` 的 confidence-gap / case mining。

### 下一步

按用户优先级，先继续做无需重训且直接影响 claim cleanup 的 `P3_ANSWERABILITY_SEMANTICS` 和 `P3_SCHEMA_DEPENDENCY_WRITEUP`，再决定是否启动 `P1_DATA_SCALING` / `P1_SCHEMA_COMPLEXITY_SWEEP` 的重训队列。

## 2026-06-17 Phase 3 / P3_ANSWERABILITY_SEMANTICS 执行前记录

### 任务计划

1. 不重训，不启动 GPU job。
2. 对比三个 objective：
   - no mask / null kept：`outputs/lp_A_ums_12label/metrics_final.json`
   - answerability mask：`outputs/lp_ums_ansmask_12label/metrics_final.json`
   - null-as-negative：`outputs/lp_ums_null_as_negative_12label/metrics_final.json`
3. 复用 `outputs/final_tables/per_field_results.csv` 计算 high-null fields AUC/F1。
4. 复用 `outputs/schema_answerability_chexpert_train.csv` / `val.csv` 和 `schema_manual_audit_chexpert_val_200_summary.json` 描述 null/missingness prevalence。
5. 搜索 sample-level predictions/logits；如果没有，则将 predicted absent rate on null fields、null calibration 和 over-absent failure cases 标记为未能从当前 artifacts 恢复。
6. 输出 answerability 语义表、null-heavy field 表和可供后续人工/概率审计的 null candidate CSV。

### 计划命令

```powershell
python scripts/analyze_answerability_semantics.py
```

### 输入

- `outputs/lp_A_ums_12label/metrics_final.json`
- `outputs/lp_ums_ansmask_12label/metrics_final.json`
- `outputs/lp_ums_null_as_negative_12label/metrics_final.json`
- `outputs/final_tables/per_field_results.csv`
- `outputs/schema_answerability_chexpert_train.csv`
- `outputs/schema_answerability_chexpert_val.csv`
- `outputs/schema_manual_audit_chexpert_val_200.csv`
- `outputs/schema_manual_audit_chexpert_val_200_summary.json`

### 输出

- `outputs/final_tables/answerability_semantics.csv`
- `outputs/final_tables/answerability_semantics.md`
- `outputs/final_tables/null_field_calibration.csv`
- `outputs/final_tables/null_field_calibration.md`
- `outputs/final_tables/answerability_missing_artifacts.md`
- `outputs/failure_cases/null_as_negative_over_absent.csv`

### 停止条件

- 三个 objective 的核心 metrics 无法解析。
- high-null field 指标无法从 per-field table 或 metrics_final 计算。
- 当前 artifacts 缺少 sample-level logits/probabilities 时，不能编造 null calibration 或 predicted absent rate。

## 2026-06-17 Phase 3 / P3_ANSWERABILITY_SEMANTICS 执行后记录

### 实际命令

```powershell
python scripts/analyze_answerability_semantics.py
```

### 生成文件

- `scripts/analyze_answerability_semantics.py`
- `outputs/final_tables/answerability_semantics.csv`
- `outputs/final_tables/answerability_semantics.md`
- `outputs/final_tables/null_field_calibration.csv`
- `outputs/final_tables/null_field_calibration.md`
- `outputs/final_tables/answerability_missing_artifacts.md`
- `outputs/failure_cases/null_as_negative_over_absent.csv`

### 结果摘要

- 成功生成 `3` 行 objective-level answerability 表。
- 成功生成 `4` 行 high-null field 表。
- 生成 `674` 个 high-null field 的 null candidate slots，用于后续 sample-level probability/manual audit。
- 核心指标：
  - no mask / null kept：AUC `0.8439`，F1 `0.9095`，high-null AUC `0.8076`，high-null F1 `0.9512`。
  - answerability mask：AUC `0.8178`，F1 `0.9120`，high-null AUC `0.8070`，high-null F1 `0.9473`。
  - null-as-negative：AUC `0.8334`，F1 `0.9133`，high-null AUC `0.7950`，high-null F1 `0.9415`。
- 200-sample schema audit summary：mean null fields `8.0850` / image，mean answerable fields `3.9150` / image。

### 缺失 artifact / 失败原因

- 未找到 sample-level probabilities/logits/OOF predictions。
- 因此当前不能计算：
  - predicted absent rate on null fields；
  - null-heavy field calibration；
  - 真正的 null-as-negative over-absent failure cases。
- `outputs/failure_cases/null_as_negative_over_absent.csv` 当前只是 candidate null slots，不是已验证 failure cases。

### 初步解释边界

- null-as-negative 提升/保持部分分类 F1，不等于证明 null 在临床语义上等于 absent。
- 在 high-null group 上，null-as-negative AUC `0.7950` 低于 no-mask `0.8076` 和 answerability mask `0.8070`，不能用 high-null 证据推翻 answerability。
- answerability mask 的总体 AUC 较低，但它保留 missingness-faithful semantics；应写成语义/校准问题，而不是单纯 AUC 排名问题。
- 如果后续要强支持 answerability，需要补 sample-level prediction export 或 manual audit：验证 null slots 是否被 null-as-negative 推向 over-confident absent。

### 下一步

继续执行 `P3_SCHEMA_DEPENDENCY_WRITEUP`，将已有 schema-key robustness / field paraphrase diagnostics 转成 paper-ready limitation 表，明确 fixed-schema interface 边界。

## 2026-06-17 Phase 3 / P3_SCHEMA_DEPENDENCY_WRITEUP 执行前记录

### 任务计划

1. 不重训，不做 schema augmentation。
2. 读取已有 schema robustness diagnostics：
   - `outputs/schema_key_robustness_A_ums_12label_128.json`
   - `outputs/field_paraphrase_robustness_A_ums_12label_128.json`
3. 统一抽取 original NLL、variant NLL、margin、original better rate、relative delta。
4. 将诊断转成 paper-ready limitation：
   - 当前方法是 fixed-schema supervised interface；
   - 不声称 schema-agnostic language understanding；
   - reordered/paraphrased schema 会显著提高 teacher-forcing NLL。
5. 生成 diagnostics CSV/Markdown 和 case study Markdown。

### 计划命令

```powershell
python scripts/write_schema_dependency_diagnostics.py
```

### 输入

- `outputs/schema_key_robustness_A_ums_12label_128.json`
- `outputs/field_paraphrase_robustness_A_ums_12label_128.json`

### 输出

- `outputs/final_tables/schema_dependency_diagnostics.csv`
- `outputs/final_tables/schema_dependency_diagnostics.md`
- `outputs/final_tables/schema_dependency_case_study.md`
- `outputs/final_tables/schema_dependency_missing_artifacts.md`

### 停止条件

- schema-key 或 paraphrase diagnostic JSON 无法解析。
- 输出表不能量化 original vs variant NLL gap。
- 文案出现 arbitrary paraphrase robustness / schema-agnostic understanding 暗示。

## 2026-06-17 Phase 3 / P3_SCHEMA_DEPENDENCY_WRITEUP 执行后记录

### 实际命令

```powershell
python scripts/write_schema_dependency_diagnostics.py
```

### 生成文件

- `scripts/write_schema_dependency_diagnostics.py`
- `outputs/final_tables/schema_dependency_diagnostics.csv`
- `outputs/final_tables/schema_dependency_diagnostics.md`
- `outputs/final_tables/schema_dependency_case_study.md`
- `outputs/final_tables/schema_dependency_missing_artifacts.md`

### 结果摘要

- 成功生成 `6` 行 schema dependency diagnostics。
- 成功生成 `48` 行 case-study 候选样本，并在 case study 中展示 margin 最大的 examples。
- 没有缺失 artifact。
- 核心诊断：
  - reversed order：original NLL `0.0543`，variant NLL `0.5887`，margin `0.5344`，original better rate `1.0000`。
  - shuffled order：original NLL `0.0543`，variant NLL `0.4926`，margin `0.4382`，original better rate `1.0000`。
  - clinical key shift：original NLL `0.0543`，variant NLL `0.2664`，margin `0.2121`，original better rate `1.0000`。
  - generic keys：original NLL `0.0543`，variant NLL `0.4131`，margin `0.3587`，original better rate `1.0000`。
  - clinical paraphrase：original NLL `0.0482`，variant NLL `1.1919`，margin `1.1438`，original better rate `1.0000`。
  - lay paraphrase：original NLL `0.0482`，variant NLL `1.3334`，margin `1.2853`，original better rate `1.0000`。

### 初步解释边界

- 当前 objective 是 schema-supervised fixed clinical serialization，不是 arbitrary schema paraphrase understanding。
- field order、field key、clinical paraphrase、lay paraphrase 都显著增加 teacher-forcing NLL。
- 论文应明确 fixed-schema interface；不要写 schema-agnostic / paraphrase robust。
- 如果要缓解该限制，应作为后续 `P2_SCHEMA_AUGMENTATION` 或 appendix mitigation，而不是当前主 claim。

### 下一步

无重训诊断阶段已经完成：P0 结果/成本、P1 field difficulty、P3 answerability、P3 schema dependency。下一步应根据当前证据更新 execution queue：优先设计 `P1_DATA_SCALING` 与 `P1_SCHEMA_COMPLEXITY_SWEEP` 的最小重训队列，同时避免 SPD 新变体。

## 2026-06-17 无重训诊断后的队列决策

### 已锁定证据

- `P0_RESULT_CONSOLIDATION`：现有主表支持 UMS/schema 是稳定贡献；frozen-LM 在 CheXpert 上最强，但 NIH 上 no-LM 更高，不能写 LLM external dominance。
- `P0_COST_TABLE`：frozen-LM 是 training-time cost，deployment-time LLM 均为 `no`；历史 artifact 缺少 peak memory / structured parameter count。
- `P1_FIELD_DIFFICULTY_ANALYSIS`：frozen-LM 相对 no-LM 的优势集中在 rare/high-null/uncertain-heavy groups，而不是 common fields。
- `P3_ANSWERABILITY_SEMANTICS`：null-as-negative 不能按 AUC 一票否定 answerability；当前缺 sample-level probabilities 来验证 null calibration / over-absent failure。
- `P3_SCHEMA_DEPENDENCY_WRITEUP`：fixed schema serialization 依赖很强，不能写 schema-agnostic/paraphrase robust。

### 下一批任务选择

1. `P1_DATA_SCALING_PREP`：先创建 deterministic 1k/3k/10k/30k CheXpert subsets 和对应最小 configs，不启动完整训练。
2. `P1_DATA_SCALING`：优先跑最小矩阵：
   - BCE ViT-B：1k / 3k / 10k / 30k。
   - no-LM UMS：1k / 3k / 10k / 30k。
   - frozen-LM UMS no-SPD：1k / 3k / 10k / 30k。
   - random-LM：先只做 3k（如果 compute 允许再补 30k）。
3. `P1_SCHEMA_COMPLEXITY_SWEEP_PREP`：先确认 serializer/code 是否已有 `schema_mode`；若没有，先做 S1/S2/S3 的最小实现方案，不直接开完整 sweep。

### 明确不做

- 不启动新的 SPD variant。
- 不把 schema augmentation 当主线；当前先作为 limitation / optional mitigation。
- 不在缺 sample-level probabilities 的情况下声称 answerability calibration 已验证。

## 2026-06-17 Phase 1 / P1_DATA_SCALING_PREP 执行前记录

### 任务计划

1. 不启动训练，不启动 GPU job。
2. 从 `data/dataset/processed/chexpert_ums_train.jsonl` 创建 deterministic patient-level subsets：
   - 1k
   - 3k
   - 10k
   - 30k/all available train records
3. 同时输出 JSONL（训练入口使用）和 CSV（provenance / label distribution 审计使用）。
4. 生成 split summary，记录实际样本数、patient 数、present/null/uncertain/answerable rates。
5. 生成 data scaling 最小 config 草案：
   - BCE ViT-B：1k / 3k / 10k / 30k。
   - no-LM UMS：1k / 3k / 10k / 30k。
   - frozen-LM UMS no-SPD：1k / 3k / 10k / 30k。
   - random-LM：先生成 3k / 30k config，但不启动。
6. 所有 config 指向 `outputs/data_scaling/<method>_<size>/`，避免覆盖既有结果。

### 计划命令

```powershell
python scripts/prepare_data_scaling.py
```

### 输入

- `data/dataset/processed/chexpert_ums_train.jsonl`
- `data/dataset/processed/chexpert_ums_val.jsonl`
- `configs/ablation_A_ums_12label.yaml`
- `configs/ums_classifier_no_llm_12label.yaml`
- `configs/ablation_ums_random_lm_12label.yaml`
- BCE baseline config template：如无现成文件，则由脚本生成最小 `train_vit_baseline.py` config 并在 summary 中标记 provenance。

### 输出

- `data/splits/chexpert_train_1k.jsonl`
- `data/splits/chexpert_train_3k.jsonl`
- `data/splits/chexpert_train_10k.jsonl`
- `data/splits/chexpert_train_30k.jsonl`
- `data/splits/chexpert_train_{1k,3k,10k,30k}.csv`
- `data/splits/data_scaling_split_summary.csv`
- `data/splits/data_scaling_split_summary.md`
- `configs/data_scaling/*.yaml`
- `outputs/final_tables/data_scaling_prep_summary.md`

### 停止条件

- 无法从 UMS records 中恢复 patient id。
- subset 生成会打破 patient-level boundary。
- 训练入口不能读取生成的 JSONL。
- 需要启动训练才能验证准备结果。

## 2026-06-17 Phase 1 / P1_DATA_SCALING_PREP 执行后记录

### 实际命令

```powershell
python scripts/prepare_data_scaling.py
python -m py_compile scripts/prepare_data_scaling.py
```

### 生成文件

- `scripts/prepare_data_scaling.py`
- `data/splits/chexpert_train_1k.jsonl`
- `data/splits/chexpert_train_3k.jsonl`
- `data/splits/chexpert_train_10k.jsonl`
- `data/splits/chexpert_train_30k.jsonl`
- `data/splits/chexpert_train_1k.csv`
- `data/splits/chexpert_train_3k.csv`
- `data/splits/chexpert_train_10k.csv`
- `data/splits/chexpert_train_30k.csv`
- `data/splits/data_scaling_split_summary.csv`
- `data/splits/data_scaling_split_summary.md`
- `configs/data_scaling/*.yaml`（`24` 个 config drafts）
- `outputs/final_tables/data_scaling_prep_summary.md`

### 结果摘要

- 生成 `4` 个 deterministic patient-level nested subsets。
- Split 规模：
  - 1k：`1000` records，`670` patients。
  - 3k：`3000` records，`1984` patients。
  - 10k：`10000` records，`6603` patients。
  - 30k：`29000` records，`19257` patients；这里的 `30k` 指当前 train JSONL 全量可用 records，不是正好 30000。
- Split distribution：
  - 1k：present rate `0.218667`，null rate `0.672167`，uncertain rate `0.043000`。
  - 3k：present rate `0.223083`，null rate `0.670472`，uncertain rate `0.041444`。
  - 10k：present rate `0.226567`，null rate `0.668808`，uncertain rate `0.042075`。
  - 30k：present rate `0.225575`，null rate `0.668761`，uncertain rate `0.042083`。
- Patient-level validation：
  - `1k <= 3k <= 10k <= 30k` nested patient sets all true。
  - partial patients：1k `0`，3k `0`，10k `0`，30k `0`。
- Config drafts：
  - BCE：`bce_{1k,3k,10k,30k}.yaml`。
  - no-LM source：`no_lm_ums_{1k,3k,10k,30k}.yaml`。
  - frozen-LM source：`frozen_lm_ums_{1k,3k,10k,30k}.yaml`。
  - LP for no-LM/frozen-LM：`lp_no_lm_ums_*` and `lp_frozen_lm_ums_*`。
  - random-LM source + LP：`random_lm_ums_{3k,30k}.yaml` and `lp_random_lm_ums_{3k,30k}.yaml`。
- 所有生成 config 使用 `num_workers: 0`，避免 Windows dataloader failure 复现。
- 没有生成 SPD config，没有启动训练。

### 失败/边界

- 第一次重跑脚本时 `data/splits/chexpert_train_30k.jsonl` 触发短暂 Windows `PermissionError`；文件不是只读，也无 Python 进程占用，随后重试成功。
- Active `configs/` 中没有 BCE full baseline config；BCE data scaling configs 使用脚本生成的最小 `train_vit_baseline.py` 模板，并在 summary 中标记 provenance。
- LP configs 也使用同一个 subset JSONL，而不是全量 train JSONL，以避免低数据 setting 泄漏全量 supervised labels。

### 下一步

执行 `P1_DATA_SCALING_DRYRUN`：只做 tiny/debug 级别的入口验证，确认 BCE / no-LM / frozen-LM / LP configs 能被脚本解析并启动到最小步数；仍不启动完整训练队列。通过后再按 GPU/时间预算注册或启动 1k 队列。

## 2026-06-17 Phase 1 / P1_DATA_SCALING_DRYRUN 执行前记录

### 任务计划

1. 不启动训练，不加载 frozen LLM，不启动 GPU job。
2. 静态验证 `configs/data_scaling/*.yaml`：
   - YAML 可解析。
   - `data.train_ums_path` / `data.val_ums_path` 存在。
   - train JSONL 可读取且记录数匹配 split summary。
   - `training.output_dir` 不覆盖既有 completed run。
   - LP config 的 `transfer.init_vit_checkpoint` 是合理的前置依赖；当前未生成 checkpoint 时标为 `blocked_until_source_run`，不视为失败。
3. 输出 config validation 表和后续运行顺序。

### 计划命令

```powershell
python scripts/validate_data_scaling_configs.py
```

### 输入

- `configs/data_scaling/*.yaml`
- `data/splits/chexpert_train_{1k,3k,10k,30k}.jsonl`
- `data/dataset/processed/chexpert_ums_val.jsonl`

### 输出

- `outputs/final_tables/data_scaling_config_validation.csv`
- `outputs/final_tables/data_scaling_config_validation.md`
- `outputs/final_tables/data_scaling_run_order.md`

### 停止条件

- 任一 source-training config 的 train/val JSONL 缺失或无法解析。
- 任一 config 会覆盖已有 `metrics_final.json`。
- LP checkpoint 前置依赖无法从 output_dir 规则推导。
- 验证过程需要加载模型或启动训练。

## 2026-06-17 Phase 1 / P1_DATA_SCALING_DRYRUN 执行后记录

### 实际命令

```powershell
python scripts/prepare_data_scaling.py
python scripts/validate_data_scaling_configs.py
```

### 生成/更新文件

- `data/splits/chexpert_val_fixed.jsonl`
- `data/splits/chexpert_val_fixed.csv`
- `data/splits/chexpert_train_{1k,3k,10k,30k}.jsonl`
- `data/splits/chexpert_train_{1k,3k,10k,30k}.csv`
- `data/splits/data_scaling_split_summary.csv`
- `data/splits/data_scaling_split_summary.md`
- `configs/data_scaling/*.yaml`
- `outputs/final_tables/data_scaling_prep_summary.md`
- `outputs/final_tables/data_scaling_config_validation.csv`
- `outputs/final_tables/data_scaling_config_validation.md`
- `outputs/final_tables/data_scaling_run_order.md`

### 关键修正

- 只读审计发现原 `chexpert_ums_train.jsonl` 与 `chexpert_ums_val.jsonl` 有 patient overlap，因此不能直接作为严格 patient-level evaluation。
- 已从 `data/dataset/processed/chexpert_ums.jsonl` 全量 30k records 重新生成 fixed validation split 和 train pool。
- 所有 data scaling configs 的 `data.val_ums_path` 现在指向 `data/splits/chexpert_val_fixed.jsonl`。

### 结果摘要

- Fixed val：`1000` records，`661` patients。
- Train splits：
  - 1k：`1000` records，`673` patients。
  - 3k：`3000` records，`1974` patients。
  - 10k：`10000` records，`6548` patients。
  - 30k：`29000` records，`19047` patients；此处 30k = hold out fixed val 后的 full train pool。
- Split distribution：
  - val_fixed：present `0.218750`，null `0.673917`，uncertain `0.042000`，answerable `0.326083`。
  - 1k：present `0.219833`，null `0.675083`，uncertain `0.039083`，answerable `0.324917`。
  - 3k：present `0.224306`，null `0.670417`，uncertain `0.041111`，answerable `0.329583`。
  - 10k：present `0.226808`，null `0.668558`，uncertain `0.041850`，answerable `0.331442`。
  - 30k：present `0.225773`，null `0.668649`，uncertain `0.042089`，answerable `0.331351`。
- Config validation：
  - `24` configs validated.
  - Failures: `0`.
  - LP blocked until source checkpoints: `10`.
  - All config train/val patient overlap: `0`.

### 失败/边界

- `validate_data_scaling_configs.py` 初版重复扫描 large JSONL 导致 timeout；已加 JSONL metadata cache 后通过。
- LP configs 当前正确标记为 `blocked_until_source_run`，因为 source checkpoints 尚未训练生成。
- 这一步没有加载模型、没有启动训练、没有下载权重。

### 下一步

可以进入 `P1_DATA_SCALING_1K_SOURCE_RUNS`：先只启动 1k source runs（BCE、no-LM UMS、frozen-LM UMS），每个 run 前继续记录命令、输入输出、停止条件；等 source checkpoints 完成后再跑对应 LP，不一次启动全矩阵。

## 2026-06-17 Phase 1 / P1_DATA_SCALING_RUNTIME_PROBE 执行后记录

### 实际命令

```powershell
nvidia-smi
python -c "import torch, timm, yaml; print(...)"
conda run -n vivid python -c "import torch, timm, yaml; print(...)"
& .\vivid_env\Scripts\python.exe -c "import torch, timm, yaml; print(...)"
```

### 结果摘要

- GPU 状态：2 张 RTX 3090 空闲，仅 Codex GUI 占少量显存；没有训练进程。
- `outputs/data_scaling` 在启动训练前不存在，因此不会覆盖已完成 data-scaling run。
- Python runtime probe 失败：
  - system `python` import `torch/timm/yaml` 超过 `30s`；
  - `conda run -n vivid python` import 超过 `60s`；
  - repo `vivid_env\Scripts\python.exe` import 超过 `120s`。
- 上述探针残留的 Python 进程已清理。

### 停止原因

- 训练入口依赖 `torch` / `timm`，但当前 Python runtime 在 import 阶段卡住。
- 因此未启动 `P1_DATA_SCALING_1K_BCE_SOURCE` 训练，避免留下不可控后台训练进程或半初始化 GPU job。

### 下一步

- 暂缓所有需要模型 import / GPU 的训练 run。
- 继续推进不依赖 runtime 的 `P1_SCHEMA_COMPLEXITY_SWEEP_PREP`：先明确现有 code 缺少 `schema_mode`，并准备最小实现/验证边界。

## 2026-06-17 Phase 1 / P1_SCHEMA_COMPLEXITY_SWEEP_PREP 执行前记录

### 任务计划

1. 不启动训练，不加载模型，不启动 GPU job。
2. 静态审计现有 serializer / config / no-LM classifier 是否支持：
   - `state_only`
   - `state_answerability`
   - `state_uncertainty`
   - `state_uncertainty_location_severity`
   - `compositional_full`
3. 明确当前能直接跑的 S1/current 等价项，以及 S2/S3/S4 需要的代码改动。
4. 输出 schema complexity prep report，供后续实现前审阅。
5. 不生成 SPD 相关配置。

### 计划命令

```powershell
python scripts/audit_schema_complexity_support.py
```

### 输入

- `data/chexpert_dataset.py`
- `scripts/train_cxr.py`
- `scripts/train_ums_classifier.py`
- `configs/ablation_A_ums_12label.yaml`
- `configs/ums_classifier_no_llm_12label.yaml`

### 输出

- `outputs/final_tables/schema_complexity_prep.md`
- `outputs/final_tables/schema_complexity_support_matrix.csv`

### 停止条件

- 现有 code 无法静态确认 serializer/target capabilities。
- 报告无法区分“可直接跑”和“必须先实现”的 schema levels。
- 需要训练或模型 import 才能继续。

## 2026-06-17 Phase 1 / P1_SCHEMA_COMPLEXITY_SWEEP_PREP 执行后记录

### 实际命令

```powershell
python scripts/audit_schema_complexity_support.py
python -m py_compile scripts/audit_schema_complexity_support.py scripts/validate_data_scaling_configs.py scripts/prepare_data_scaling.py
```

### 生成文件

- `scripts/audit_schema_complexity_support.py`
- `outputs/final_tables/schema_complexity_prep.md`
- `outputs/final_tables/schema_complexity_support_matrix.csv`

### 结果摘要

- S1 / `state_only`：当前 JSON serializer 与 fixed 4-state no-LM classifier 可以作为 S1/current equivalent。
- S2 / `state_answerability`：
  - UMS records 已有 `answerability` data fields。
  - frozen-LM serializer 需要显式 `schema_mode`。
  - no-LM 当前没有 separate answerability supervision/head。
- S3 / `state_uncertainty`：
  - UMS records 已有 uncertainty fields。
  - no-LM 当前只把 uncertainty 折叠为 4-state 中的 `uncertain`。
  - 若要 fair comparison，需要定义 separate uncertainty target/head 或清楚标成 state-space variant。
- S4/S5：
  - 当前 CheXpert label-derived UMS 缺少 location/severity/temporality 等 enriched fields。
  - 不应从当前标签硬造 compositional schema。

### 缺失/边界

- `data/chexpert_dataset.py` 当前没有显式 `schema_mode`。
- `scripts/train_cxr.py` 当前只传 `target_format/json_null_state/answerability_mask` 等开关。
- `scripts/train_ums_classifier.py` 当前是固定 `STATE_TO_INDEX={null, absent, uncertain, present}`。
- 因 runtime import 卡住，本步骤没有启动任何训练。

### 下一步

- 若继续推进 schema sweep，应先做小范围代码实现：给 frozen-LM path 添加 backward-compatible `schema_mode=state_only/state_answerability/state_uncertainty`，并为 no-LM 决定 S2/S3 的目标设计。
- 在 runtime 恢复前，优先保留当前 prep 产物，不启动 S2/S3 训练。

## 2026-06-17 Phase 1 / P1_SCHEMA_MODE_MINIMAL_IMPLEMENTATION 执行前记录

### 任务计划

1. 不启动训练，不加载模型，不启动 GPU job。
2. 只实现 frozen-LM JSON target path 的最小 `schema_mode`：
   - `state_only`：默认，保持当前 JSON target 兼容。
   - `state_answerability`：在每个 finding target 中加入 `answerable`。
   - `state_uncertainty`：在每个 finding target 中加入 `uncertain`。
3. 在 `scripts/train_cxr.py` 中透传 `data.schema_mode` 到 `CheXpertUMSDataset`。
4. 不改 no-LM classifier head；S2/S3 no-LM 仍标记为待设计。
5. 更新静态 audit 输出，验证 code 中存在 `schema_mode`。

### 计划命令

```powershell
python -m py_compile data/chexpert_dataset.py scripts/train_cxr.py
python scripts/audit_schema_complexity_support.py
```

### 输入

- `data/chexpert_dataset.py`
- `scripts/train_cxr.py`

### 输出

- 更新 `data/chexpert_dataset.py`
- 更新 `scripts/train_cxr.py`
- 更新 `outputs/final_tables/schema_complexity_prep.md`
- 更新 `outputs/final_tables/schema_complexity_support_matrix.csv`

### 停止条件

- 默认 `state_only` 不能保持旧 target schema。
- 实现需要 torch/model import 才能验证。
- 改动会影响 no-LM classifier 或 SPD path。

## 2026-06-17 Phase 1 / P1_SCHEMA_MODE_MINIMAL_IMPLEMENTATION 执行后记录

### 执行命令

```powershell
python -m py_compile data/chexpert_dataset.py scripts/train_cxr.py
python scripts/audit_schema_complexity_support.py
```

### 结果

- 已在 `data/chexpert_dataset.py` 为 `CheXpertUMSDataset` 增加 backward-compatible `schema_mode`：
  - `state_only`：默认值；不向 JSON target 增加字段，保持旧 target schema。
  - `state_answerability`：在每个 finding target 中加入 `answerable`。
  - `state_uncertainty`：在每个 finding target 中加入 `uncertain`。
- 已在 `scripts/train_cxr.py` 中从 `data.schema_mode` 读取并传入 train/val dataset。
- 已更新 `scripts/audit_schema_complexity_support.py` 与静态输出：
  - `outputs/final_tables/schema_complexity_prep.md`
  - `outputs/final_tables/schema_complexity_support_matrix.csv`
- `py_compile` 通过；未加载 torch/model，未启动训练或 GPU job。

### 指标 / 静态证据

- `schema_complexity_prep.md` 显示：
  - `data/chexpert_dataset.py` contains explicit `schema_mode`: True
  - `data/chexpert_dataset.py` serializes `answerability`: True
  - `data/chexpert_dataset.py` serializes `uncertainty`: True
  - `scripts/train_ums_classifier.py` has fixed `STATE_TO_INDEX`: True
- 支持矩阵边界：
  - S1 frozen-LM：`supported_explicit_schema_mode`
  - S2/S3 frozen-LM：`minimal_serializer_supported`
  - S2/S3 no-LM：仍未支持 separate supervision/head；不能把 no-LM 纳入同等 schema complexity comparison。
  - S4/S5：当前 CheXpert label-derived UMS 不支持，不应伪造 location/severity/temporality。

### 失败 / 阻塞原因

- 无代码级失败。
- 训练级验证仍受 `P1_DATA_SCALING_RUNTIME_PROBE` 中记录的 Python/runtime import hang 阻塞；本任务故意只做静态验证。

### 下一步

- 可继续做 `P1_SCHEMA_COMPLEXITY_CONFIG_PREP`：只生成 frozen-LM S1/S2/S3 的最小 config 草案，并用静态检查确认 `schema_mode`、输入 split、输出目录；不启动训练。
- no-LM S2/S3 需要先设计 target/head，暂不排队。

## 2026-06-17 Phase 1 / P1_SCHEMA_COMPLEXITY_CONFIG_PREP 执行前记录

### 任务计划

1. 不启动训练，不导入 torch/model。
2. 只生成 frozen-LM source + downstream LP 的 S1/S2/S3 config 草案：
   - S1：`state_only`
   - S2：`state_answerability`
   - S3：`state_uncertainty`
3. 使用 `data/splits/chexpert_train_30k.jsonl` 与 `data/splits/chexpert_val_fixed.jsonl`，继承 `P1_DATA_SCALING_PREP` 的 patient-disjoint fixed split，避免旧 train/val overlap。
4. 不生成 no-LM S2/S3 config；no-LM 目标/head 设计未完成。
5. 输出 manifest，记录每个 config 的输入、输出目录、patient overlap 和后续运行状态。

### 计划命令

```powershell
python -m py_compile scripts/prepare_schema_sweep_configs.py
python scripts/prepare_schema_sweep_configs.py
```

### 输入

- `configs/ablation_A_ums_12label.yaml`
- `configs/lp_A_ums_12label.yaml`
- `data/splits/chexpert_train_30k.jsonl`
- `data/splits/chexpert_val_fixed.jsonl`

### 输出

- `configs/schema_sweep/frozen_lm_s1_state_only.yaml`
- `configs/schema_sweep/frozen_lm_s2_state_answerability.yaml`
- `configs/schema_sweep/frozen_lm_s3_state_uncertainty.yaml`
- `configs/schema_sweep/lp_frozen_lm_s1_state_only.yaml`
- `configs/schema_sweep/lp_frozen_lm_s2_state_answerability.yaml`
- `configs/schema_sweep/lp_frozen_lm_s3_state_uncertainty.yaml`
- `outputs/final_tables/schema_sweep_config_manifest.csv`
- `outputs/final_tables/schema_sweep_config_prep.md`

### 停止条件

- fixed train/val split 缺失或 patient overlap 不为 0。
- 任一 source config 缺少 `data.schema_mode`。
- 脚本需要导入 torch/model 才能生成或验证 config。
- 需要生成 no-LM S2/S3 config 才能继续。

## 2026-06-17 Phase 1 / P1_SCHEMA_COMPLEXITY_CONFIG_PREP 执行后记录

### 执行命令

```powershell
python -m py_compile scripts/prepare_schema_sweep_configs.py
python scripts/prepare_schema_sweep_configs.py
```

### 结果

- 已生成 frozen-LM schema complexity source/LP config pairs：
  - S1 `state_only`
  - S2 `state_answerability`
  - S3 `state_uncertainty`
- 未生成 no-LM S2/S3 config；原因是 no-LM classifier 目前仍是固定 4-state target，缺少 separate answerability/uncertainty head 或显式 target 设计。
- 未启动训练，未导入 torch/model，未启动 GPU job。

### 指标 / provenance

- 固定 split 检查：
  - train: `data/splits/chexpert_train_30k.jsonl`，29000 records，19047 patients
  - val: `data/splits/chexpert_val_fixed.jsonl`，1000 records，661 patients
  - train/val patient overlap: 0
- 输出：
  - `configs/schema_sweep/frozen_lm_s1_state_only.yaml`
  - `configs/schema_sweep/frozen_lm_s2_state_answerability.yaml`
  - `configs/schema_sweep/frozen_lm_s3_state_uncertainty.yaml`
  - `configs/schema_sweep/lp_frozen_lm_s1_state_only.yaml`
  - `configs/schema_sweep/lp_frozen_lm_s2_state_answerability.yaml`
  - `configs/schema_sweep/lp_frozen_lm_s3_state_uncertainty.yaml`
  - `outputs/final_tables/schema_sweep_config_manifest.csv`
  - `outputs/final_tables/schema_sweep_config_prep.md`
- `schema_sweep_config_prep.md` 的 LP run order 已校正为 `python scripts/train_vit_baseline.py --config ...`，与现有 data scaling run order 一致。

### 失败 / 阻塞原因

- config 级别无失败。
- 运行状态统一标记为 `config_ready_but_runtime_import_blocked`；训练仍受 `P1_DATA_SCALING_RUNTIME_PROBE` 中记录的 Python/runtime import hang 阻塞。

### 下一步

- runtime 恢复后，按 manifest 顺序先运行 source，再运行对应 LP。
- schema sweep 数字应与历史 P0 主表分开报告，因为这里使用新的 patient-disjoint fixed split。
- 如果要纳入 no-LM S2/S3，必须先新增 no-LM answerability/uncertainty target/head 设计并单独记录执行计划。

## 2026-06-17 Phase 1 / P1_RUNTIME_IMPORT_DIAGNOSIS 执行前记录

### 任务计划

1. 不启动训练，不加载模型权重，不启动 GPU job。
2. 先确认当前 Python 入口、版本、`yaml` import 是否正常。
3. 用短超时和 `faulthandler.dump_traceback_later` 检查 `torch` import 是否仍然 hang。
4. 若 `torch` import hang，记录具体命令、超时阈值和可见 traceback；不继续排队 data scaling/schema sweep 训练。

### 计划命令

```powershell
where.exe python
python -c "import sys; print(sys.executable); print(sys.version)"
python -c "import yaml; print(yaml.__version__)"
python -X faulthandler -c "import faulthandler; faulthandler.dump_traceback_later(20, repeat=False); import torch; print(torch.__version__)"
conda run --no-capture-output -n vivid python -X faulthandler -c "import faulthandler; faulthandler.dump_traceback_later(20, repeat=False); import torch; print(torch.__version__)"
D:\python\python.exe -c "import sys; print(sys.executable); print(sys.version)"
C:\Users\Admin\anaconda3\python.exe -c "import sys; print(sys.executable); print(sys.version)"
C:\Users\Admin\anaconda3\envs\vivid\python.exe -c "import sys; print(sys.executable); print(sys.version)"
```

### 输入

- 当前 shell Python
- conda env `vivid`
- direct Python candidates from `where.exe python`

### 输出

- Python/yaml/torch import diagnosis
- 若仍失败，写入 runtime 阻塞原因与下一步建议

### 停止条件

- 任一 `torch` import 超过短超时阈值。
- 诊断命令留下未结束 Python 进程。
- 需要执行训练脚本或加载 checkpoint 才能继续。

## 2026-06-17 Phase 1 / P1_RUNTIME_IMPORT_DIAGNOSIS 执行后记录

### 执行命令与结果

```powershell
where.exe python
```

- 当前 PATH 顺序：
  - `C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe`
  - `D:\python\python.exe`
  - `C:\Users\Admin\anaconda3\python.exe`
  - `C:\Users\Admin\AppData\Local\Microsoft\WindowsApps\python.exe`

```powershell
python -c "import sys; print(sys.executable); print(sys.version)"
python -c "import yaml; print(yaml.__version__)"
```

- 当前 shell Python：`C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe`
- 版本：Python 3.12.8
- `yaml` import 正常：6.0.3

```powershell
python -X faulthandler -c "import faulthandler; faulthandler.dump_traceback_later(20, repeat=False); import torch; print(torch.__version__)"
```

- `torch` 单独 import 成功：`2.5.1+cu121`，约 14 秒。

```powershell
python -X faulthandler -c "import faulthandler; faulthandler.dump_traceback_later(20, repeat=False); import timm; print(timm.__version__)"
python -X faulthandler -c "import faulthandler; faulthandler.dump_traceback_later(20, repeat=False); import torch, timm, yaml; print('ok', torch.__version__, timm.__version__, yaml.__version__)"
python -X faulthandler -c "import faulthandler; faulthandler.dump_traceback_later(20, repeat=False); import torchvision; print(torchvision.__version__)"
```

- `timm` import 超时：约 49 秒后被终止。
- `torch+timm+yaml` 组合 import 超时：约 49 秒后被终止。
- `torchvision` import 超时：约 39 秒后被终止。
- package metadata 可读：
  - `torch`: 2.5.1+cu121
  - `torchvision`: 0.20.1+cu121
  - `timm`: 1.0.24

```powershell
conda run --no-capture-output -n vivid python -c "import sys; print(sys.executable); print(sys.version)"
conda info --envs
```

- `conda run -n vivid` 版本查询超时：约 24 秒。
- `conda info --envs` 超时：约 24 秒。

```powershell
C:\Users\Admin\anaconda3\envs\vivid\python.exe -c "import sys; print(sys.executable); print(sys.version)"
C:\Users\Admin\anaconda3\envs\vivid\python.exe -X faulthandler -c "import faulthandler; faulthandler.dump_traceback_later(20, repeat=False); import torch, timm, yaml; print('ok', torch.__version__, timm.__version__, yaml.__version__)"
D:\python\python.exe -X faulthandler -c "import faulthandler; faulthandler.dump_traceback_later(20, repeat=False); import torch, torchvision, timm, yaml; print('ok', torch.__version__, torchvision.__version__, timm.__version__, yaml.__version__)"
```

- direct `C:\Users\Admin\anaconda3\envs\vivid\python.exe` 可输出版本：Python 3.10.19。
- direct `envs\vivid` 的 `torch+timm+yaml` import 超时：约 49 秒。
- `D:\python\python.exe` 有旧版包 metadata：torch 1.13.1, torchvision 0.14.1, timm 0.6.12, PyYAML 6.0；但 `torch+torchvision+timm+yaml` import 仍超时：约 49 秒。

### 指标 / failure boundary

- 失败层级已从 “torch import hang” 收窄为 “torchvision/timm import stack hang”。
- 训练入口 `scripts/train_cxr.py`、`scripts/train_vit_baseline.py`、`scripts/train_ums_classifier.py` 都会 import `timm` 或 vision stack，因此当前不适合启动任何训练。
- `torch` 单独可用不等于训练 runtime 可用；需要 `torchvision/timm` import 恢复后才可排队。
- 超时产生的 Python 诊断进程已清理；最终 `Get-Process python` / `Get-Process conda` 无残留。

### 失败原因

- 当前可见根因：`torchvision` import 卡住，进而导致 `timm` import 和训练脚本入口卡住。
- `conda` 命令自身也异常慢/超时，不能作为可靠 runner wrapper。

### 下一步

- 暂停所有需要 `timm`/`torchvision` 的训练任务：`P1_DATA_SCALING`、`P1_SCHEMA_COMPLEXITY_SWEEP` source/LP runs。
- 可继续做不依赖 vision stack import 的文档/manifest/结果整合。
- 若要恢复训练，应先修复 Python vision stack，例如在一个干净环境中验证：
  - `python -c "import torch; import torchvision; import timm; print('ok')"`
  - 再用同一解释器启动 `scripts/train_cxr.py` / `scripts/train_vit_baseline.py`。

## 2026-06-17 Phase 1 / P1_RUNTIME_DEPENDENCY_REPAIR 执行前记录

### 任务计划

1. 先确认是否真的缺依赖；若缺失则下载/安装。
2. 若依赖已存在但 import 卡死，抓取 `torchvision` import verbose/faulthandler 日志定位卡点。
3. 只修复 runtime dependency；不启动训练、不加载 checkpoint、不注册 GPU 队列。
4. 修复验证标准：
   - `python -c "import torch; import torchvision; import timm; print('ok')"` 能在 60 秒内完成。
   - `python -c "import yaml; import torch; import torchvision; import timm"` 能在 60 秒内完成。
5. 若需要下载，优先使用当前 Python 对应的 `pip`，并记录 wheel 版本；不要同时改多个 Python 环境。

### 计划命令

```powershell
python -m pip show torch torchvision timm PyYAML
python -m pip check
python -c "import importlib.util as u; print(u.find_spec('torchvision')); print(u.find_spec('timm'))"
python -X faulthandler -v -c "import torchvision" > outputs/logs/diagnose_torchvision_import_verbose.log 2>&1
```

若确认需要重新下载/安装：

```powershell
python -m pip install --upgrade --force-reinstall --no-cache-dir torch torchvision timm PyYAML --index-url https://download.pytorch.org/whl/cu121
```

或若 `timm/PyYAML` 不在 PyTorch index：

```powershell
python -m pip install --upgrade --force-reinstall --no-cache-dir timm PyYAML
```

### 输入

- 当前 shell Python：`C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe`
- 当前已安装 package metadata
- import verbose/faulthandler 日志

### 输出

- `outputs/logs/diagnose_torchvision_import_verbose.log`
- dependency repair result
- 可运行/不可运行判定

### 停止条件

- install 命令需要卸载/替换大范围 CUDA/PyTorch 包但无法确认目标 wheel。
- import verbose 仍无法定位且连续 3 次 repair hypothesis 均失败。
- 修复过程留下未结束 Python/pip 进程。

## 2026-06-17 Phase 1 / P1_RUNTIME_SLOW_IMPORT_DIAGNOSIS 执行前记录

### 任务计划

1. 不启动训练，不加载 checkpoint，不注册 GPU 队列。
2. 检查是否有前序超时留下的 Python/pip 进程、CPU/内存/磁盘/GPU 异常负载。
3. 用 `-X importtime` 定位 `torchvision`、`timm`、`scripts/train_vit_baseline.py --help` 的慢 import 层级。
4. 检查 Python site 初始化、editable package finder、用户 site-packages、Defender/安全扫描等可能导致 Windows import 极慢的因素。
5. 只有明确缺包或损坏时才下载/重装；如果是环境/扫描/路径问题，先给最小修复路径。

### 计划命令

```powershell
Get-Process python,pip -ErrorAction SilentlyContinue
Get-CimInstance Win32_Processor
Get-CimInstance Win32_ComputerSystem
Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3"
nvidia-smi
python -X importtime -c "import torchvision" 2> outputs/logs/importtime_torchvision.log
python -X importtime -c "import timm" 2> outputs/logs/importtime_timm.log
python -X importtime scripts/train_vit_baseline.py --help 2> outputs/logs/importtime_train_vit_baseline_help.log
python -S -c "import sys; print(sys.executable)"
python -c "import site; print(site.getsitepackages()); print(site.getusersitepackages())"
Get-MpPreference
```

### 输入

- 当前 Python environment
- `scripts/train_vit_baseline.py`
- `outputs/logs/*importtime*.log`
- 系统 CPU/内存/磁盘/GPU/Defender 状态

### 输出

- import slow-path diagnosis
- 是否需要下载/重装依赖的判定
- 训练 runtime 是否可以恢复的判定

### 停止条件

- 任一诊断命令超过 180 秒。
- 诊断命令留下未结束 Python/pip 进程。
- 需要进行卸载/重装前仍无法确认根因。

## 2026-06-17 Phase 1 / P1_SYSTEM_OCCUPANCY_SECURITY_AUDIT 执行前记录

### 任务计划

1. 只读排查，不杀进程、不删除文件、不改 Defender/计划任务配置。
2. 检查是否存在：
   - 残留 Python/pip/conda 进程；
   - 异常 CPU/内存/磁盘占用进程；
   - Windows Defender 实时扫描或近期威胁记录；
   - 正在运行的计划任务/自动化任务；
   - 常见同步/索引/杀毒/备份进程抢占；
   - GPU 上异常进程。
3. 如果发现可疑项，只记录证据和建议；修复/禁用前另写执行前记录。

### 计划命令

```powershell
Get-Process python,pip,conda -ErrorAction SilentlyContinue
Get-Process | Sort-Object CPU -Descending | Select-Object -First 20
Get-Process | Sort-Object WorkingSet -Descending | Select-Object -First 20
Get-Counter '\Processor(_Total)\% Processor Time','\Memory\Available MBytes','\PhysicalDisk(_Total)\% Disk Time'
Get-MpComputerStatus
Get-MpPreference
Get-MpThreatDetection
Get-ScheduledTask | Where-Object {$_.State -eq 'Running'}
Get-Service | Where-Object {$_.Status -eq 'Running'}
nvidia-smi
```

### 输入

- Windows process table
- Windows Defender status/preference/threat history
- Running scheduled tasks/services
- CPU/memory/disk/GPU counters

### 输出

- system occupancy/security audit notes
- 是否存在高风险可疑进程/自动化占用的判定
- 若无明显病毒迹象，继续定位 Python import 慢路径

### 停止条件

- 命令需要管理员权限才能继续。
- 发现明确恶意或高风险进程，需要先暂停并征求用户确认。
- 查询命令本身持续超时，说明系统管理接口/WMI 也异常，需要单独记录。

## 2026-06-17 Phase 1 / P1_GIT_AUTOMATION_THROTTLE 执行前记录

### 任务计划

1. 不删除文件，不改代码，不 reset，不 checkout。
2. 只停止由 Codex 应用拉起且正在处理大量未跟踪/历史文件的 Git 子进程：
   - `git add -- History/... configs/... data/... scripts/...`
   - `git diff --no-ext-diff --numstat ...`
3. 不停止用户前台应用、远程控制软件、VPN、安全软件。
4. 停止后立即验证：
   - `.git/index.lock` 不存在；
   - `git diff --cached --stat` 为空；
   - `git status --short --untracked-files=no` 不出现 staged 新文件；
   - Git 进程数量下降。

### 计划命令

```powershell
Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'git.exe'}
Stop-Process -Id <Codex-launched git add/diff pids> -Force
Test-Path .git\index.lock
git diff --cached --stat
git status --short --untracked-files=no
Get-Process git -ErrorAction SilentlyContinue
```

### 输入

- 当前 Git 子进程命令行
- 当前 `.git` index 状态

### 输出

- 停止的 Git PIDs
- staged/index 状态验证
- 对系统占用的影响

### 停止条件

- 发现 `.git/index.lock` 正在被活跃 Git 写入且无法确认安全停止。
- Git 进程不是 Codex/当前 repo 相关。
- 停止后出现 staged/unintended changes。

## 2026-06-17 Phase 1 / P1_GITIGNORE_LOCAL_ARTIFACTS_FIX 执行前记录

### 任务计划

1. 只修改 `.gitignore`，不删除任何文件。
2. 将明显本地/历史训练产物加入 ignore：
   - `History/`
   - `data/splits/`
3. 保留 `configs/` 和 `scripts/` 可见，避免隐藏真实代码/config 变更。
4. 修改后验证 `git check-ignore` 和 `git status --short`。

### 计划命令

```powershell
git check-ignore -v History/... data/splits/...
git status --short
```

### 输入

- `.gitignore`
- 当前未跟踪目录 `History/`、`data/splits/`

### 输出

- 更新 `.gitignore`
- Git 自动化不再扫描历史 checkpoint/log 和 split JSONL/CSV

### 停止条件

- `History/` 或 `data/splits/` 已经被 Git 跟踪。
- ignore 规则会隐藏 `configs/` 或 `scripts/` 的代码/config 变更。

## 2026-06-17 Phase 1 / P1_RUNTIME_DEPENDENCY_REPAIR 执行后记录

### 执行命令

```powershell
python -m pip show torch torchvision timm PyYAML
python -m pip check
python -c "import importlib.util as u; print(u.find_spec('torchvision')); print(u.find_spec('timm'))"
python -X faulthandler -v -c "import torchvision" > outputs/logs/diagnose_torchvision_import_verbose.log 2>&1
```

### 结果

- 依赖不是缺失状态：
  - `torch`: 2.5.1+cu121
  - `torchvision`: 0.20.1+cu121
  - `timm`: 1.0.24
  - `PyYAML`: 6.0.3
- `pip check`: `No broken requirements found.`
- `torchvision` / `timm` module spec 均存在。
- `import torchvision` verbose 日志最终可完整退出；前面的 timeout 判断包含 PowerShell 进程等待误判和并行冷启动干扰。

### 判定

- 当前不建议直接 force reinstall / download PyTorch 依赖。
- 真正问题不是“没下载”，而是 CUDA PyTorch + torchvision/timm 冷启动慢，以及 repo/Git 自动化在后台扫描大文件。

## 2026-06-17 Phase 1 / P1_SYSTEM_OCCUPANCY_SECURITY_AUDIT 执行后记录

### 执行命令

```powershell
Get-Process / Get-Counter / nvidia-smi
Get-MpComputerStatus
Get-MpThreatDetection
Get-ScheduledTask | Where-Object {$_.State -eq 'Running'}
Get-CimInstance Win32_StartupCommand
Get-Service
Get-AuthenticodeSignature
```

### 结果

- GPU：2x RTX 3090 均空闲，约 13 MiB memory，0% util；没有异常 GPU compute 进程。
- 内存：约 64 GB 总内存，检查时 available memory 约 49 GB；无内存压力。
- 磁盘：总体 disk time 低；无明显全盘 I/O 打满证据。
- CPU：总体约 22-38% 波动；前台 Codex/PowerShell/EdgeWebView/Taskmgr/ToDesk 有占用，但不是满载。
- Defender：
  - `AntivirusEnabled=False`
  - `RealTimeProtectionEnabled=False`
  - `Get-MpThreatDetection` 无记录
  - SecurityCenter2 只登记 Windows Defender；火绒进程存在但未登记为 SecurityCenter2 AV product。
- 正在运行的计划任务：
  - `Clash Verge (Admin)`
  - Microsoft `SystemSoundsService`
  - Microsoft `NetworkStateChangeTask`
  - Microsoft `CacheTask`
- 启动项包含多个远程/网络/同步入口：
  - `ToDesk`
  - `SunloginClient`
  - `GameViewer`
  - `Tailscale`
  - `Warp`
  - `WebVPN` / `MotionPro VPN`
  - `OneDrive`
  - `Docker Desktop`
  - `BaiduYunDetect`
  - `QuarkUpdater`
  - `Ollama`
- 签名检查：
  - ToDesk: valid signer `Hainan YouQu Technology Co., Ltd`
  - GameViewer: valid signer `NetEase (Hangzhou) Network Co., Ltd`
  - HipsTray / 火绒: valid signer `北京火绒网络科技有限公司`
  - Sangfor EAIO: valid signer `Sangfor Technologies Inc.`

### 判定

- 当前没有发现挖矿/GPU 病毒型证据。
- 没有发现明显伪装成系统文件的高占用进程。
- 但机器上远控/网络/同步/安全常驻项很多；如果不是用户主动安装，应人工核对：
  - ToDesk、Sunlogin、GameViewer、Tailscale/Warp/WebVPN/MotionPro、BaiduNetdisk/QuarkUpdater。
- Defender 关闭，因此“无 Defender 威胁记录”不能作为完整杀毒结论。

## 2026-06-17 Phase 1 / P1_GIT_AUTOMATION_THROTTLE 执行后记录

### 执行命令

```powershell
Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'git.exe'}
Stop-Process -Id 14716,18948,9364,21352 -Force
Test-Path .git\index.lock
git diff --cached --stat
git status --short --untracked-files=no
Get-Process git -ErrorAction SilentlyContinue
```

### 结果

- 停止了 4 个由 Codex 应用父进程拉起的 Git 子进程：
  - huge `git diff --no-ext-diff --numstat`
  - huge `git add -- History/... configs/... data/... scripts/...`
- 停止后验证：
  - `.git/index.lock`: False
  - `git diff --cached --stat`: empty
  - `git status --short --untracked-files=no` 只显示 tracked modified files:
    - `.gitignore`
    - `data/chexpert_dataset.py`
    - `models/vivid_model.py`
    - `scripts/train_cxr.py`
  - 没有 staged changes 或意外新增 staged 文件。
- CPU 采样下降；高内存 Git 进程消失。

### 判定

- 这是明确的自动化占用源，不是病毒证据。
- 触发原因是 repo 中未跟踪本地产物太多，尤其 `History/` checkpoint/log 和 `data/splits/` JSONL/CSV。

## 2026-06-17 Phase 1 / P1_GITIGNORE_LOCAL_ARTIFACTS_FIX 执行后记录

### 执行命令

```powershell
git check-ignore -v History/20260601_0015_failed_dataloader_workers/lp_ums_ansmask_12label/best.pt data/splits/chexpert_train_30k.jsonl outputs/final_tables/main_controlled_results.md
git status --short
```

### 结果

- `.gitignore` 已新增：
  - `data/splits/`
  - `History/`
- 验证结果：
  - `History/.../best.pt` matched `.gitignore:5:History/`
  - `data/splits/chexpert_train_30k.jsonl` matched `.gitignore:3:data/splits/`
  - `outputs/final_tables/main_controlled_results.md` 仍 matched existing `.gitignore:4:outputs/`
- `configs/` 和 `scripts/` 仍显示为 untracked，代码/config 变更未被隐藏。

### 判定

- 已降低 Codex/Git UI 反复扫描历史 checkpoint/log 和 split 数据的概率。
- 仍需保持 `configs/` / `scripts/` 可见，因为它们是当前实验执行产物和代码产物。

## 2026-06-17 Phase 1 / P1_RUNTIME_SLOW_IMPORT_DIAGNOSIS 执行后记录

### 执行命令

```powershell
python -X importtime -c "import torch" 2> outputs/logs/importtime_torch.log
python -X importtime -c "import torchvision" 2> outputs/logs/importtime_torchvision.log
python -X importtime -c "import timm" 2> outputs/logs/importtime_timm.log
python -X importtime scripts/train_vit_baseline.py --help 2> outputs/logs/importtime_train_vit_baseline_help.log
python -c "import time; import torch; import timm; from evaluation.metrics import compute_classification_metrics"
python -S -c "pass"
python -I -c "pass"
```

### 结果 / 指标

- `torch` importtime:
  - total cumulative: about 7.95s in one run
  - major modules:
    - `torch._meta_registrations`: 2.17s
    - `torch._decomp`: 1.93s
    - `torch._prims`: 1.55s
    - `torch._C`: 0.78s
- `torchvision` importtime:
  - total cumulative: about 13.0s
  - major modules:
    - `torch`: 7.31s
    - `torchvision.models`: 5.03s
    - `torchvision.models.convnext` / `torchvision.ops` / `torch._dynamo`: 4-5s range
    - `sympy`: about 2.18s via `torch._dynamo`
- `timm` importtime:
  - total cumulative: about 16.8s
  - major modules:
    - `timm.layers`: 14.13s
    - `timm.layers._fx`: 13.92s
    - `torch`: 7.55s
    - `torchvision.models.feature_extraction`: 6.36s
- `train_vit_baseline.py --help`:
  - completed after Git throttle; importtime run about 33s
  - major modules:
    - `timm`: 9.35s
    - `evaluation.metrics`: 8.28s cumulative because `evaluation/__init__.py` imports verifier and metrics
    - `sklearn.metrics`: 4.26s
    - `jsonschema`: 4.01s via verifier import
- Python empty startup:
  - normal `python -c "pass"`: about 1.68s
  - `python -S -c "pass"`: about 0.52s
  - site hooks contribute about 1s, but are not the main 20-50s import cost.
- Alternate runtimes:
  - `C:\Users\Admin\anaconda3\envs\vivid\python.exe`: `torch` about 20.4s, `timm` about 77.6s; worse.
  - `D:\python\python.exe`: torch 1.13.1+cpu, no GPU; not suitable for RTX 3090 training.

### 判定

- 慢的主因不是缺依赖，也不是发现的病毒/挖矿。
- 主要原因：
  1. CUDA PyTorch 2.5 + torchvision/timm 在 Windows 下冷启动很重；
  2. `timm` import 会拉 `torchvision.models.feature_extraction`、`torch._dynamo` 和 `sympy`；
  3. repo 的 `evaluation/__init__.py` 顶层导入 verifier，导致只 import metrics 也会加载 `jsonschema`；
  4. 之前 Codex/Git 自动化扫描大量未跟踪本地产物进一步放大了卡顿。
- 当前推荐训练 runner：PATH 上的 Python 3.12 CUDA env，而不是 conda `vivid` 或 `D:\python` CPU env。

### 下一步

- 不重装 PyTorch，除非后续出现真实 import exception 或 wheel mismatch。
- 可做小型代码优化：
  - 将 `evaluation/__init__.py` 改为 lazy exports，避免 `from evaluation.metrics` 拉起 verifier/jsonschema；
  - 或把训练脚本 heavy imports 移到 argparse 之后，让 `--help` 快速返回。
- 训练任务可以恢复前仍应先做单条 source run，不一次开完整矩阵。

## 2026-06-17 Phase 1 / P1_EVALUATION_IMPORT_LAZY_FIX 执行前记录

### 任务计划

1. 不改训练逻辑和指标公式。
2. 只降低 import side effect：
   - `evaluation/__init__.py` 改为 lazy export，避免 `from evaluation.metrics` 时加载 verifier/jsonschema；
   - `evaluation/metrics.py` 将 `sklearn.metrics` 延迟到 `compute_classification_metrics()` 内首次调用。
3. 验证：
   - `python -m py_compile evaluation/__init__.py evaluation/metrics.py`
   - `python -c "from evaluation.metrics import compute_classification_metrics; print('ok')"`
   - `python -c "from evaluation import UMSVerifier, compute_classification_metrics; print('ok')"`
   - `python scripts/train_vit_baseline.py --help`

### 输入

- `evaluation/__init__.py`
- `evaluation/metrics.py`

### 输出

- 更快、更轻的 evaluation import path
- 不改变 metric function output

### 停止条件

- lazy import 改变 public API。
- 指标函数无法调用 sklearn metrics。
- 训练脚本 import 失败。

## 2026-06-17 Phase 1 / P1_EVALUATION_IMPORT_LAZY_FIX 执行后记录

### 执行命令

```powershell
python -m py_compile evaluation/__init__.py evaluation/metrics.py
python -c "from evaluation.metrics import compute_classification_metrics; print('ok')"
python -c "from evaluation import UMSVerifier, compute_classification_metrics; print('ok')"
python -c "import numpy as np; from evaluation.metrics import compute_classification_metrics; ..."
Measure-Command { python scripts/train_vit_baseline.py --help > $null }
```

### 结果

- 已更新 `evaluation/__init__.py`：
  - 使用 `__getattr__` lazy export `UMSVerifier`、`compute_classification_metrics`、`compute_reliability_metrics`。
  - `from evaluation.metrics import ...` 不再强制加载 verifier/jsonschema。
- 已更新 `evaluation/metrics.py`：
  - `sklearn.metrics` 延迟到 `compute_classification_metrics()` 首次调用。
  - 去掉未使用的 `confusion_matrix` import。
- 验证：
  - `py_compile` 最终通过；首次失败是 Windows pycache 写入瞬时冲突 `[WinError 5]`，源码 `compile()` 通过，复跑 `py_compile` 通过。
  - `from evaluation.metrics import compute_classification_metrics`: about 2.4-3.1s。
  - `from evaluation import UMSVerifier, compute_classification_metrics`: about 10.66s，因为显式请求 `UMSVerifier` 时仍需加载 verifier/jsonschema，符合 lazy 预期。
  - 小指标调用输出 `macro_f1=1.0`, `macro_auc=1.0`。
  - 单独 `train_vit_baseline.py --help`: about 33.9s。

### 判定

- evaluation import path 已减轻，但训练脚本 `--help` 仍慢，剩余主因是顶层 `torch/timm/data` heavy imports。
- 若要让 `--help` 秒级返回，需要进一步重构训练脚本，把 heavy imports 移到 argparse 之后；该优化不影响训练速度，只改善 CLI 启动/诊断体验。

## 2026-06-17 Phase 0-3 / REVISION_EXECUTION_STATUS_SYNC 执行前记录

### 任务计划

1. 不启动训练，不导入 torch/model。
2. 汇总当前执行状态，区分：
   - completed analysis/artifact
   - config-ready but blocked
   - deferred by claim priority
   - not validated
3. 明确下一步 stop condition：只有 `torchvision/timm` import 修复后才能恢复训练队列。

### 计划命令

```powershell
python -m py_compile scripts/summarize_revision_execution_status.py
python scripts/summarize_revision_execution_status.py
```

### 输入

- `outputs/final_tables/*.md`
- `configs/data_scaling/*.yaml`
- `configs/schema_sweep/*.yaml`
- 当前执行日志

### 输出

- `outputs/final_tables/revision_execution_status.csv`
- `outputs/final_tables/revision_execution_status.md`

### 停止条件

- 脚本需要导入 torch/model。
- 状态表把 blocked training 写成 completed experiment。
- 状态表把 SPD 新变体列为下一步优先项。

## 2026-06-17 Phase 0-3 / REVISION_EXECUTION_STATUS_SYNC 执行后记录

### 执行命令

```powershell
python -m py_compile scripts/summarize_revision_execution_status.py
python scripts/summarize_revision_execution_status.py
```

### 结果

- 已新增 `scripts/summarize_revision_execution_status.py`。
- 已生成：
  - `outputs/final_tables/revision_execution_status.csv`
  - `outputs/final_tables/revision_execution_status.md`
- 状态表共 9 行，明确区分：
  - `completed_analysis`
  - `completed_analysis_with_missing_fields`
  - `completed_analysis_limited_by_oof`
  - `config_ready`
  - `frozen_lm_config_ready_no_lm_deferred`
  - `runtime_available_but_slow`
  - `deferred`

### 指标 / 核对

- `P0_RESULT_CONSOLIDATION`: `completed_analysis`, evidence `yes`
- `P0_COST_TABLE`: `completed_analysis_with_missing_fields`, evidence `yes`
- `P1_FIELD_DIFFICULTY_ANALYSIS`: `completed_analysis`, evidence `yes`
- `P3_ANSWERABILITY_SEMANTICS`: `completed_analysis_limited_by_oof`, evidence `yes`
- `P3_SCHEMA_DEPENDENCY_WRITEUP`: `completed_limitation_writeup`, evidence `yes`
- `P1_DATA_SCALING_PREP`: `config_ready`, evidence `yes`
- `P1_SCHEMA_COMPLEXITY_SWEEP_PREP`: `frozen_lm_config_ready_no_lm_deferred`, evidence `yes`
- `P1_RUNTIME_REPAIR_AND_AUDIT`: `runtime_available_but_slow`, evidence `yes`
- `P2_NEW_MODULES`: `deferred`, evidence `n/a`

### 失败 / 阻塞原因

- 无脚本失败。
- 表中没有把 training config-ready 误写为 completed experiment。
- 表中明确写明：
  - 不一次运行完整 Phase 1 矩阵；
  - LP runs 等 source checkpoint；
  - no-LM S2/S3 仍 design-blocked；
  - SPD new variants out of scope。

### 下一步

- 做 fresh process/GPU check 后，只进入一个 1k source run 的入口验证。
- 若入口验证通过，再选择一个最小 source run；仍不启动完整 data scaling 或 schema sweep 矩阵。

## 2026-06-17 Phase 1 / P1_DATA_SCALING_1K_BCE_DEBUG_ENTRY 执行前记录

### 任务计划

1. 不启动完整 data scaling 矩阵。
2. 只验证 `configs/data_scaling/bce_1k.yaml` 的 BCE 1k source 入口能否创建 dataloader/model 并完成 tiny debug loop。
3. 先修复 debug-mode scheduler 边界：`max_steps=20` 时同步降低 `warmup_steps`，避免 warmup 大于训练步数。
4. 使用独立 seed 后缀输出，避免污染正式 `outputs/data_scaling/bce_1k`：
   - `--seed 900117`
   - output_dir 自动变为 `outputs/data_scaling/bce_1k_seed900117`
5. 若该 debug run 成功，仍不继续启动 full 1k source；下一步需另写执行前记录。

### 计划命令

```powershell
python -m py_compile scripts/train_vit_baseline.py
nvidia-smi
python scripts/train_vit_baseline.py --config configs/data_scaling/bce_1k.yaml --debug --seed 900117
```

### 输入

- `scripts/train_vit_baseline.py`
- `configs/data_scaling/bce_1k.yaml`
- `data/splits/chexpert_train_1k.jsonl`
- `data/splits/chexpert_val_fixed.jsonl`
- local CheXpert image files under `data/dataset`

### 输出

- Debug-only output directory:
  - `outputs/data_scaling/bce_1k_seed900117/`
- Debug metrics/checkpoints if run succeeds.

### 停止条件

- `py_compile` fails。
- GPU/process check shows active training already running。
- Debug run attempts to write to official `outputs/data_scaling/bce_1k`。
- Missing images/checkpoints/download failure prevents dataloader/model creation。
- Run exceeds 10 minutes or leaves a Python process after timeout。

## 2026-06-17 Phase 1 / P1_VIT_PRETRAINED_WEIGHT_DOWNLOAD 执行前记录

### 任务计划

1. 不启动训练。
2. 修复 BCE debug entry 的权重缺失问题：
   - `timm` 需要 `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`
   - 本地 HF cache 未发现该 repo 的权重。
3. 当前环境 `HF_ENDPOINT=https://hf-mirror.com`，但 mirror 触发了 metadata/HEAD error；直连 Hugging Face API 返回 200。
4. 使用单条命令临时设置 `HF_ENDPOINT=https://huggingface.co`，下载 `model.safetensors` 到 HF cache。
5. 下载后只验证 `timm.create_model(..., pretrained=True)` 能从 cache 创建模型；不训练。

### 计划命令

```powershell
$env:HF_ENDPOINT='https://huggingface.co'
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('timm/vit_base_patch16_224.augreg2_in21k_ft_in1k', filename='model.safetensors'))"
python -c "import timm; m=timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=14); print(type(m).__name__)"
```

### 输入

- Hugging Face repo: `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`
- HF cache: `H:\.cache\huggingface\hub`

### 输出

- Cached pretrained ViT-B weights under HF cache.
- Model creation check result.

### 停止条件

- Download fails due proxy/SSL/403/timeout.
- Downloaded file is missing or zero bytes.
- `timm.create_model(... pretrained=True ...)` still attempts network and fails after cache download.

## 2026-06-17 Phase 1 / P1_VIT_PRETRAINED_WEIGHT_DOWNLOAD 执行后记录

### 执行命令

```powershell
$env:HF_ENDPOINT='https://huggingface.co'
python -c "from huggingface_hub import hf_hub_download; path=hf_hub_download('timm/vit_base_patch16_224.augreg2_in21k_ft_in1k', filename='model.safetensors'); print(path)"
python -c "import timm; m=timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=14); print(type(m).__name__)"
```

### 结果

- `timm` 所需 repo:
  - `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`
- 原因定位：
  - 当前环境默认 `HF_ENDPOINT=https://hf-mirror.com`
  - mirror 触发 `FileMetadataError` / `LocalEntryNotFoundError`
  - 直连 `https://huggingface.co/api/models/...` 返回 200
- 已下载：
  - `H:\.cache\huggingface\hub\models--timm--vit_base_patch16_224.augreg2_in21k_ft_in1k\snapshots\063c6c38a5d8510b2e57df480445e94b231dad2c\model.safetensors`
  - 文件大小：346,284,714 bytes
- `timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=14)` 成功，模型类型 `VisionTransformer`，约 30.92 秒。

### 失败 / 注意

- 下载过程中出现 HF unauthenticated warning；不影响本次下载，但后续大规模下载可能受限。
- Windows symlink warning：HF cache 在该目录不能使用 symlink，缓存会占用更多磁盘；不影响可用性。
- 建议后续训练命令继续临时设置 `HF_ENDPOINT=https://huggingface.co`，避免 hf-mirror metadata issue。

### 下一步

- 重跑 `P1_DATA_SCALING_1K_BCE_DEBUG_ENTRY`，确认 pretrained ViT-B source entry 能完整通过 tiny debug loop。

## 2026-06-17 Phase 1 / P1_DATA_SCALING_1K_BCE_DEBUG_ENTRY 执行后记录

### 执行命令

```powershell
python -m py_compile scripts/train_vit_baseline.py
nvidia-smi
$env:HF_ENDPOINT='https://huggingface.co'
python scripts/train_vit_baseline.py --config configs/data_scaling/bce_1k.yaml --debug --seed 900117
```

### 结果

- 已修复 `scripts/train_vit_baseline.py` debug scheduler 边界：
  - debug `max_steps=20`
  - debug `warmup_steps=min(existing, max(1, max_steps//4))`
- `py_compile` 通过。
- GPU/process pre-check：
  - 两张 RTX 3090 空闲，约 13 MiB memory，0% util。
  - 无残留 Python process。
- 第一次 debug run 失败在 model creation：
  - 本地缺 `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`
  - `HF_ENDPOINT=https://hf-mirror.com` metadata/HEAD error
- 下载 pretrained weight 后重跑成功。

### Debug run 指标

- 输入：
  - train: `data/splits/chexpert_train_1k.jsonl`，debug 读取 200 samples
  - val: `data/splits/chexpert_val_fixed.jsonl`，debug 读取 50 samples
  - labels: 14 CheXpert labels
- 训练：
  - device: cuda
  - max_steps: 20
  - output_dir: `outputs/data_scaling/bce_1k_seed900117`
  - final val_loss: 0.4649109627519335
  - macro_f1: 0.7993894993894994
  - macro_auc: 0.4984288953676708
  - micro_f1: 0.7707006369426752
- 输出：
  - `outputs/data_scaling/bce_1k_seed900117/best.pt`
  - `outputs/data_scaling/bce_1k_seed900117/final.pt`
  - `outputs/data_scaling/bce_1k_seed900117/step_20.pt`
  - `outputs/data_scaling/bce_1k_seed900117/metrics_final.json`
  - `outputs/data_scaling/bce_1k_seed900117/metrics_step_5.json`
  - `outputs/data_scaling/bce_1k_seed900117/metrics_step_10.json`
  - `outputs/data_scaling/bce_1k_seed900117/metrics_step_15.json`
  - `outputs/data_scaling/bce_1k_seed900117/metrics_step_20.json`

### 失败 / 边界

- Debug metrics 不作为 paper/result table 证据，只证明入口可运行。
- 官方输出 `outputs/data_scaling/bce_1k/metrics_final.json` 不存在；正式 1k BCE source 尚未完成。
- 运行结束后：
  - 无残留 Python process。
  - GPU 回到空闲状态。

### 下一步

- 可以进入 `P1_DATA_SCALING_1K_BCE_SOURCE_RUN`：只启动正式 BCE 1k source，不启动 no-LM/frozen-LM/LP/full matrix。
- 正式 run 前必须另写执行前记录，并设置 `HF_ENDPOINT=https://huggingface.co`。

## 2026-06-17 Phase 1 / P1_DATA_SCALING_1K_BCE_SOURCE_RUN 执行前记录

### 任务计划

1. 只启动正式 BCE 1k source run。
2. 不启动 no-LM UMS、frozen-LM UMS、random-LM、LP runs、schema sweep 或完整 data scaling 矩阵。
3. 使用正式 config：
   - `configs/data_scaling/bce_1k.yaml`
4. 使用单 GPU：
   - `CUDA_VISIBLE_DEVICES=0`
5. 临时设置：
   - `HF_ENDPOINT=https://huggingface.co`
   避免 hf-mirror metadata issue。
6. 后台运行并写日志；持续监控到完成或失败。

### 计划命令

```powershell
$env:HF_ENDPOINT='https://huggingface.co'
$env:CUDA_VISIBLE_DEVICES='0'
python scripts/train_vit_baseline.py --config configs/data_scaling/bce_1k.yaml
```

后台日志：

```powershell
outputs/logs/data_scaling_bce_1k_source.log
```

### 输入

- `configs/data_scaling/bce_1k.yaml`
- `data/splits/chexpert_train_1k.jsonl`
- `data/splits/chexpert_val_fixed.jsonl`
- `H:\.cache\huggingface\hub\models--timm--vit_base_patch16_224.augreg2_in21k_ft_in1k\...\model.safetensors`

### 输出

- `outputs/data_scaling/bce_1k/`
- Expected:
  - `metrics_final.json`
  - `best.pt`
  - `final.pt`
  - periodic `metrics_step_*.json`
  - periodic checkpoints

### 停止条件

- `outputs/data_scaling/bce_1k/metrics_final.json` already exists before launch。
- GPU/process check shows another training job running。
- Training exits nonzero。
- Run leaves orphan Python process after failure。
- Any sign that no-LM/frozen-LM/LP/full matrix started unintentionally。

## 2026-06-17 Phase 1 / P1_RUNTIME_SLOWDOWN_RECHECK 执行前记录

### 触发原因

正式 `P1_DATA_SCALING_1K_BCE_SOURCE_RUN` 已启动后，用户观察到“在这台机器配置下这么慢不正常”，要求排查是否有病毒、自动化进程或其他占用。

### 任务计划

1. 不中断正在运行的 BCE 1k source training。
2. 分层检查：
   - training process / GPU utilization / log progress；
   - current process list；
   - Git/Codex automation process；
   - CPU / memory / disk instantaneous counters；
   - startup entries；
   - Windows Defender status / threat history。
3. 仅在确认是当前 repo/Codex 相关 Git 子进程、且正在执行大范围 `git add`/`git diff` 时停止 Git 子进程。
4. 不删除文件、不关闭远程控制软件、不修改 Defender/启动项配置。

### 计划命令

```powershell
Get-Date
nvidia-smi
Get-ChildItem outputs\data_scaling\bce_1k -Filter metrics_step_*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 5
Get-Content outputs\logs\data_scaling_bce_1k_source.log -Tail 80
cmd /c tasklist /fo csv
cmd /c wmic process where "name='git.exe' or name='python.exe' or name='powershell.exe' or name='codex.exe' or name='Codex.exe'" get ProcessId,Name,CommandLine /format:list
Get-Process git -ErrorAction SilentlyContinue | Stop-Process -Force
cmd /c wmic startup get Caption,Command,Location,User /format:list
Get-MpComputerStatus | Select-Object AMServiceEnabled,AntivirusEnabled,RealTimeProtectionEnabled,BehaviorMonitorEnabled,IoavProtectionEnabled,QuickScanStartTime,QuickScanEndTime,FullScanStartTime,FullScanEndTime,AntivirusSignatureLastUpdated
Get-MpThreat | Select-Object ThreatName,SeverityID,DidThreatExecute,ActionSuccess,Resources
cmd /c wmic cpu get LoadPercentage
cmd /c wmic OS get FreePhysicalMemory,TotalVisibleMemorySize /format:list
cmd /c wmic path Win32_PerfFormattedData_PerfDisk_PhysicalDisk get Name,DiskReadBytesPerSec,DiskWriteBytesPerSec,PercentDiskTime,AvgDiskQueueLength /format:csv
cmd /c wmic path Win32_PerfFormattedData_PerfProc_Process get IDProcess,Name,PercentProcessorTime,WorkingSet /format:csv
```

### 输入

- Running training PID:
  - parent PowerShell: `21556`
  - Python training: `11840`
- Log:
  - `outputs/logs/data_scaling_bce_1k_source.log`
- Output directory:
  - `outputs/data_scaling/bce_1k`
- Windows process table / startup entries / Defender status.

### 输出

- Runtime slowdown diagnosis in this execution plan.
- If safe, stopped Git automation PIDs.
- Updated recommendation for continuing/stopping the active training.

### 停止条件

- Diagnosis command itself repeatedly hangs over 30-90 seconds.
- Evidence points to unknown high-GPU or high-CPU process unrelated to current repo, in which case do not kill it automatically.
- Training exits or fails during diagnosis.
- `.git/index.lock` remains after stopping Git, requiring manual recovery note before any further Git operation.

## 2026-06-17 Phase 1 / P1_RUNTIME_SLOWDOWN_RECHECK 执行后记录

### 执行命令

```powershell
Get-Date
nvidia-smi
Get-ChildItem outputs\data_scaling\bce_1k -Filter metrics_step_*.json -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 5 Name,LastWriteTime,Length
Get-Content outputs\logs\data_scaling_bce_1k_source.log -Tail 80
cmd /c tasklist /fo csv
cmd /c wmic process where "name='git.exe' or name='python.exe' or name='powershell.exe' or name='codex.exe' or name='Codex.exe'" get ProcessId,Name,CommandLine /format:list
Get-Process git -ErrorAction SilentlyContinue | Stop-Process -Force
cmd /c wmic startup get Caption,Command,Location,User /format:list
Get-MpComputerStatus | Select-Object AMServiceEnabled,AntivirusEnabled,RealTimeProtectionEnabled,BehaviorMonitorEnabled,IoavProtectionEnabled,QuickScanStartTime,QuickScanEndTime,FullScanStartTime,FullScanEndTime,AntivirusSignatureLastUpdated
Get-MpThreat | Select-Object ThreatName,SeverityID,DidThreatExecute,ActionSuccess,Resources
cmd /c wmic cpu get LoadPercentage
cmd /c wmic OS get FreePhysicalMemory,TotalVisibleMemorySize /format:list
cmd /c wmic path Win32_PerfFormattedData_PerfDisk_PhysicalDisk get Name,DiskReadBytesPerSec,DiskWriteBytesPerSec,PercentDiskTime,AvgDiskQueueLength /format:csv
cmd /c wmic path Win32_PerfFormattedData_PerfProc_Process get IDProcess,Name,PercentProcessorTime,WorkingSet /format:csv
```

### 结果

- `Get-Date` 单独执行约 8.2 秒，说明 PowerShell/process startup 本身也偏慢。
- Training process 仍在运行：
  - parent PowerShell PID `21556`
  - Python PID `11840`
  - GPU0 memory about `4027 MiB`
  - GPU1 idle
- 慢速证据：
  - `metrics_step_2000.json` 写入时间为 `2026-06-17 15:49:21`。
  - 15:53 左右 log tail 显示从 step 2228 以后出现大量 1-5 秒/step 抖动。
  - 同一时刻 `nvidia-smi` 抓拍 GPU0 util 为 `0%`，但显存仍被 Python 占用，说明训练没有退出，而是 GPU 在等待 CPU/IO/调度。
- 直接占用源：
  - `tasklist` 与 `wmic process` 发现多条 `git.exe`。
  - 命令行包括：
    - `git ... status --porcelain`
    - `git ... add -- configs/... scripts/... vivid_med_revision_execution_plan.md`
    - `git ... diff ... --numstat -z`
  - 这些进程来自当前 repo/Codex/Git 自动化路径，不像未知恶意程序。
  - 已停止当前 Git 子进程；停止后 `.git/index.lock` 不存在。
- 停止 Git 后的恢复迹象：
  - GPU0 util 抓拍恢复到 `70%`，随后一次抓拍为 `100%`。
  - log 从 step 2290 继续推进到 step 2462。
  - 速度仍有 1-2 秒/step 抖动，但已经不是完全 GPU 空转状态。
- 系统资源采样：
  - CPU load: `31%`
  - memory: about `50,138,848 KiB` free of `66,876,976 KiB`
  - disk instantaneous counters: sampled `0` queue/read/write on all disks;该瞬时采样未显示磁盘持续打满。
  - Python training process `PercentProcessorTime=190`，约 2 个 CPU core。
  - Codex renderer `PercentProcessorTime=97`，约 1 个 CPU core；Codex/Git UI 仍会贡献后台负载。
- Defender / 安全状态：
  - `AMServiceEnabled=False`
  - `AntivirusEnabled=False`
  - `RealTimeProtectionEnabled=False`
  - `BehaviorMonitorEnabled=False`
  - `IoavProtectionEnabled=False`
  - `Get-MpThreat` 未返回 threat rows。
  - 因 Defender 关闭，不能把“无 Defender 威胁记录”解释为完整杀毒通过。
- 启动项中存在多种远程控制 / 网络 / 同步 / 更新类自启动：
  - `GameViewer`
  - `SunloginClient`
  - `Tailscale`
  - `WebVPN`
  - `MotionPro`
  - `BaiduYunDetect`
  - `Docker Desktop`
  - `Warp`
  - `Ollama`
  - `QuarkUpdaterTaskUser1.0.0.21`
  这些未必是病毒；但如果不是用户主动安装/需要，应人工确认并考虑禁用自启。

### 诊断结论

当前证据不支持“GPU 被未知挖矿/病毒占用”：

- GPU 进程只有当前 training Python 和 Codex。
- 没有未知进程占用大量 GPU memory。
- CPU/内存/磁盘采样没有出现全机持续满载。

更可信的原因是：

1. Codex/Git 自动化反复对当前 repo 做大范围 `git add` / `git diff` / `status`，抢占 IO/CPU，并造成 GPU 等待。
2. Windows + CUDA PyTorch/timm/torchvision import/启动本身偏慢。
3. 常驻远控/网络/安全/同步进程较多，会增加调度抖动。
4. 当前训练 dataloader/source run 仍能推进，但速度在 Git 干扰后没有完全恢复到最初约 2-3 it/s。

### 失败 / 限制

- 最初尝试的多路 PowerShell JSON 诊断命令超时；这本身也支持“系统/PowerShell 当前响应慢”的判断。
- 一个 `cmd /c ... & ...` 性能采样命令被 PowerShell `&` 解析规则拒绝；后续改为独立命令采样。
- Defender 关闭，无法给出“电脑无病毒”的强结论。
- `wmic` performance counter 是瞬时采样，不能证明磁盘永远不拥塞。

### 下一步

- 保持 `P1_DATA_SCALING_1K_BCE_SOURCE_RUN` 继续运行，不启动 no-LM/frozen-LM/LP/full matrix。
- 每 10-15 分钟检查：
  - latest `metrics_step_*.json`
  - log step 进度
  - `nvidia-smi`
  - 是否再次出现大范围 `git add` / `git diff`
- 若 Git 自动化再次复发：
  - 停止 Git 子进程；
  - 继续保持 `outputs/`、`History/`、`data/splits/` 在 ignore；
  - 避免从 Codex/Git UI 对所有 untracked 文件执行 stage。
- 若用户要严格排除病毒，需要在训练结束或暂停窗口做一次完整 AV 扫描；当前阶段不建议边训练边全盘扫描。

## 2026-06-17 Phase 1 / P1_DATA_SCALING_1K_BCE_SOURCE_RUN 执行后记录

### 执行命令

```powershell
$env:HF_ENDPOINT='https://huggingface.co'
$env:CUDA_VISIBLE_DEVICES='0'
python scripts/train_vit_baseline.py --config configs/data_scaling/bce_1k.yaml
```

实际以后台 PowerShell 运行并 tee 到：

```powershell
outputs/logs/data_scaling_bce_1k_source.log
```

### 结果

- Formal BCE 1k source run completed。
- 输入：
  - train: `data/splits/chexpert_train_1k.jsonl`
  - val: `data/splits/chexpert_val_fixed.jsonl`
  - config: `configs/data_scaling/bce_1k.yaml`
  - labels: 14 CheXpert labels
  - pretrained ViT cache: `H:\.cache\huggingface\hub\models--timm--vit_base_patch16_224.augreg2_in21k_ft_in1k\...\model.safetensors`
- 训练：
  - device: GPU0 / CUDA
  - max_steps: 10000
  - training progress elapsed shown by tqdm: `1:45:45`
  - final validation completed after step 10000
- 输出：
  - `outputs/data_scaling/bce_1k/metrics_final.json`
  - `outputs/data_scaling/bce_1k/metrics_step_10000.json`
  - `outputs/data_scaling/bce_1k/best.pt`
  - `outputs/data_scaling/bce_1k/final.pt`
  - `outputs/data_scaling/bce_1k/step_10000.pt`
  - periodic metrics/checkpoints through the run
- Process/GPU cleanup：
  - `tasklist /fi "PID eq 11840"`: no matching task
  - `tasklist /fi "PID eq 21556"`: no matching task
  - `tasklist /fi "imagename eq python.exe"`: no matching task
  - `nvidia-smi`: both GPUs returned to idle except Codex app context; no training Python remains。

### Formal final metrics

Source: `outputs/data_scaling/bce_1k/metrics_final.json`

| metric | value |
|---|---:|
| val_loss | 1.293984705582261 |
| macro_f1 | 0.8860601194865824 |
| macro_auc | 0.6843658366076173 |
| micro_f1 | 0.850125593078426 |
| n_labels | 14 |

### Step-level notes

- `best.pt` timestamp is `2026-06-17 15:37:19`, corresponding to early best validation loss.
- `final.pt` timestamp is `2026-06-17 17:19:00`.
- `metrics_step_10000.json` and `metrics_final.json` contain the same final metrics.
- Selected step metrics:
  - step 500: val_loss 0.611610, macro_f1 0.885786, macro_auc 0.698255, micro_f1 0.849009
  - step 1000: val_loss 0.790982, macro_f1 0.874371, macro_auc 0.712484, micro_f1 0.842869
  - step 2500: val_loss 1.100452, macro_f1 0.890697, macro_auc 0.664715, micro_f1 0.850126
  - step 10000/final: val_loss 1.293985, macro_f1 0.886060, macro_auc 0.684366, micro_f1 0.850126

### 失败 / 异常 / 限制

- Runtime slowdown occurred mid-run:
  - repeated Codex/Git automation scans appeared as `git add`, `git diff`, `status --porcelain`, `rev-parse`, and `remote -v`;
  - when Git scans were active, `nvidia-smi` snapshots sometimes showed GPU0 util `0%` while Python still held 4027 MiB;
  - after stopping Git child processes, throughput repeatedly recovered toward 2-3 it/s.
- `.git/index.lock` was checked after stopping Git and was absent.
- The log contains garbled tqdm block characters due Windows console encoding; not a training failure.
- `outputs/logs/data_scaling_bce_1k_source.log` contains `Training completed!`, but the wrapper did not append the expected `EXITCODE 0` line. Success is therefore verified by:
  - final metrics/checkpoints present;
  - training Python and parent PowerShell exited;
  - GPU returned to idle.
- This is a fixed-split data-scaling result and must not be mixed into the historical P0 main controlled table.
- The run is BCE source only; no no-LM/frozen-LM/random-LM/LP/schema-sweep run was started.

### 下一步

- Refresh `outputs/final_tables/revision_execution_status.csv/.md` to mark `P1_DATA_SCALING_1K_BCE_SOURCE_RUN` completed.
- Next executable experiment should still be one source run only, with a fresh execution-before record and preflight process/GPU/Git check.
- Priority choices:
  - `configs/data_scaling/frozen_lm_ums_1k.yaml` if the next question is frozen-LM use-case/scaling.
- `configs/data_scaling/no_lm_ums_1k.yaml` if the next question is UMS/schema contribution independent of frozen-LM.
- Do not start LP runs until their source checkpoints exist.
- Do not launch the full Phase 1 matrix at once.

## 2026-06-17 Phase 1 / P1_DATA_SCALING_1K_NO_LM_UMS_DEBUG_ENTRY 执行前记录

### 任务计划

1. 不启动完整 data-scaling 矩阵。
2. 不启动 frozen-LM / LP / schema sweep / random-LM。
3. 只验证 `no_lm_ums_1k` source 入口：
   - dataloader 能读取 fixed patient-disjoint split；
   - ViT pretrained backbone 能从 cache 加载；
   - 4-state UMS classifier 能完成 tiny train/eval/save；
   - 运行结束后无 Python/GPU 残留。
4. 使用独立 seed 输出，避免污染正式目录：
   - `--seed 900118`
   - output_dir 自动变为 `outputs/data_scaling/no_lm_ums_1k_seed900118`
5. Debug metrics 只作为入口验证，不作为 paper evidence。

### 计划命令

```powershell
python -m py_compile scripts/train_ums_classifier.py
nvidia-smi
$env:HF_ENDPOINT='https://huggingface.co'
python scripts/train_ums_classifier.py --config configs/data_scaling/no_lm_ums_1k.yaml --debug --seed 900118
```

### 输入

- `scripts/train_ums_classifier.py`
- `configs/data_scaling/no_lm_ums_1k.yaml`
- `data/splits/chexpert_train_1k.jsonl`
- `data/splits/chexpert_val_fixed.jsonl`
- local CheXpert image files under `data/dataset`
- cached ViT-B pretrained weights

### 输出

- Debug-only output directory:
  - `outputs/data_scaling/no_lm_ums_1k_seed900118/`
- Expected if successful:
  - `metrics_final.json`
  - `best.pt`
  - `final.pt`
  - `step_20.pt`
  - periodic debug metrics

### 停止条件

- `outputs/data_scaling/no_lm_ums_1k_seed900118/metrics_final.json` already exists before launch。
- GPU/process check shows active training already running。
- Git/Codex automation resumes large `add`/`diff` scans before launch。
- `py_compile` fails。
- Model/dataloader creation fails。
- Debug run exceeds 10 minutes or leaves a Python process after failure。

## 2026-06-17 Phase 1 / P1_DATA_SCALING_1K_NO_LM_UMS_DEBUG_ENTRY 执行后记录

### 执行命令

```powershell
python -m py_compile scripts\train_ums_classifier.py
nvidia-smi
$env:HF_ENDPOINT='https://huggingface.co'
python scripts\train_ums_classifier.py --config configs\data_scaling\no_lm_ums_1k.yaml --debug --seed 900118
```

### 结果

- Debug run completed。
- 输入：
  - train: `data/splits/chexpert_train_1k.jsonl`，debug 读取 200 samples
  - val: `data/splits/chexpert_val_fixed.jsonl`，debug 读取 50 samples
  - labels: 12 selected CheXpert labels
- 训练：
  - device: cuda
  - max_steps: 20
  - output_dir: `outputs/data_scaling/no_lm_ums_1k_seed900118`
- 输出：
  - `outputs/data_scaling/no_lm_ums_1k_seed900118/config_snapshot.json`
  - `outputs/data_scaling/no_lm_ums_1k_seed900118/metrics_step_5.json`
  - `outputs/data_scaling/no_lm_ums_1k_seed900118/metrics_step_10.json`
  - `outputs/data_scaling/no_lm_ums_1k_seed900118/metrics_step_15.json`
  - `outputs/data_scaling/no_lm_ums_1k_seed900118/metrics_step_20.json`
  - `outputs/data_scaling/no_lm_ums_1k_seed900118/metrics_final.json`
  - `outputs/data_scaling/no_lm_ums_1k_seed900118/best.pt`
  - `outputs/data_scaling/no_lm_ums_1k_seed900118/final.pt`
  - `outputs/data_scaling/no_lm_ums_1k_seed900118/step_20.pt`

### Debug run 指标

Source: `outputs/data_scaling/no_lm_ums_1k_seed900118/metrics_final.json`

| metric | value |
|---|---:|
| val_loss | 0.8085320932524545 |
| macro_f1 | 0.16296296296296295 |
| macro_auc | 0.3946954481648359 |
| micro_f1 | 0.5364238410596026 |
| n_labels | 12 |

### 失败 / 注意

- Debug metrics 不作为 paper evidence，只证明 no-LM UMS source entry 可运行。
- PyTorch emitted lr_scheduler epoch deprecation warning；不影响本次 tiny run。
- 运行结束后：
  - `tasklist /fi "imagename eq python.exe"`: no matching task
  - `nvidia-smi`: GPUs idle except Codex app context
  - `wmic process where name='git.exe'`: no Git process

### 下一步

- 可以进入 `P1_DATA_SCALING_1K_NO_LM_UMS_SOURCE_RUN`。
- 正式 run 前必须另写执行前记录。
- 正式 no-LM metrics 才能用于 1k UMS/schema contribution 对照；debug metrics 不进入主表。

## 2026-06-17 Phase 1 / P1_DATA_SCALING_1K_NO_LM_UMS_SOURCE_RUN 执行前记录

### 任务计划

1. 只启动正式 no-LM UMS 1k source run。
2. 不启动 frozen-LM、BCE rerun、random-LM、LP runs、schema sweep 或完整 data-scaling 矩阵。
3. 使用正式 config：
   - `configs/data_scaling/no_lm_ums_1k.yaml`
4. 使用单 GPU：
   - `CUDA_VISIBLE_DEVICES=0`
5. 临时设置：
   - `HF_ENDPOINT=https://huggingface.co`
   避免 timm/HF metadata issue。
6. 后台运行并写日志；持续监控到完成或失败。
7. 该 run 用于与已完成的 `P1_DATA_SCALING_1K_BCE_SOURCE_RUN` 比较 1k low-data 下 UMS/schema contribution；不能与历史 P0 主表混用。

### 计划命令

```powershell
$env:HF_ENDPOINT='https://huggingface.co'
$env:CUDA_VISIBLE_DEVICES='0'
python scripts/train_ums_classifier.py --config configs/data_scaling/no_lm_ums_1k.yaml
```

后台日志：

```powershell
outputs/logs/data_scaling_no_lm_ums_1k_source.log
```

### 输入

- `configs/data_scaling/no_lm_ums_1k.yaml`
- `data/splits/chexpert_train_1k.jsonl`
- `data/splits/chexpert_val_fixed.jsonl`
- local CheXpert image files under `data/dataset`
- cached ViT-B pretrained weights

### 输出

- `outputs/data_scaling/no_lm_ums_1k/`
- Expected:
  - `metrics_final.json`
  - `best.pt`
  - `final.pt`
  - periodic `metrics_step_*.json`
  - periodic checkpoints
- Log:
  - `outputs/logs/data_scaling_no_lm_ums_1k_source.log`
  - expected final line: `EXITCODE 0`

### 停止条件

- `outputs/data_scaling/no_lm_ums_1k/metrics_final.json` already exists before launch。
- GPU/process check shows another training job running。
- Git/Codex resumes large `add`/`diff` scans before launch。
- Training exits nonzero。
- Run leaves orphan Python process after failure。
- Any sign that frozen-LM/LP/full matrix started unintentionally。

## 2026-06-17 Phase 1 / P1_DATA_SCALING_1K_NO_LM_UMS_SOURCE_RUN 执行后记录

### 执行命令

```powershell
$env:HF_ENDPOINT='https://huggingface.co'
$env:CUDA_VISIBLE_DEVICES='0'
python scripts/train_ums_classifier.py --config configs/data_scaling/no_lm_ums_1k.yaml
```

实际以后台 PowerShell `EncodedCommand` 运行并写入：

```powershell
outputs/logs/data_scaling_no_lm_ums_1k_source.log
```

### 结果

- Formal no-LM UMS 1k source run completed。
- 输入：
  - train: `data/splits/chexpert_train_1k.jsonl`
  - val: `data/splits/chexpert_val_fixed.jsonl`
  - config: `configs/data_scaling/no_lm_ums_1k.yaml`
  - labels: 12 selected CheXpert labels
- 训练：
  - device: GPU0 / CUDA
  - max_steps: 10000
  - training progress elapsed shown by tqdm: about `1:43:34`
- 输出：
  - `outputs/data_scaling/no_lm_ums_1k/metrics_final.json`
  - `outputs/data_scaling/no_lm_ums_1k/metrics_step_10000.json`
  - `outputs/data_scaling/no_lm_ums_1k/best.pt`
  - `outputs/data_scaling/no_lm_ums_1k/final.pt`
  - `outputs/data_scaling/no_lm_ums_1k/step_10000.pt`
  - periodic metrics/checkpoints through the run
- Exit/process/GPU cleanup：
  - log contains `EXITCODE 0`
  - `tasklist /fi "PID eq 23036"`: no matching task
  - `tasklist /fi "PID eq 9980"`: no matching task
  - `tasklist /fi "imagename eq python.exe"`: no matching task
  - `nvidia-smi`: both GPUs idle except Codex app context

### Formal final metrics

Source: `outputs/data_scaling/no_lm_ums_1k/metrics_final.json`

| metric | value |
|---|---:|
| val_loss | 2.9547683000564575 |
| macro_f1 | 0.3609847388866225 |
| macro_auc | 0.6365509825963275 |
| micro_f1 | 0.5467879143443826 |
| state_accuracy_all_fields | 0.6795833333333333 |
| state_accuracy_answerable_fields | 0.38985039601056026 |
| n_labels | 12 |

### Step-level notes

- `best.pt` timestamp is `2026-06-17 17:38:15`, corresponding to the best validation loss at/near step 500.
- `final.pt` timestamp is `2026-06-17 19:17:06`.
- `metrics_step_10000.json` and `metrics_final.json` contain the same final metrics.
- Selected step metrics:
  - step 500: val_loss 1.101613, macro_f1 0.334863, macro_auc 0.663464, micro_f1 0.514520
  - step 2000: val_loss 2.201571, macro_f1 0.391824, macro_auc 0.662816, micro_f1 0.545615
  - step 3500: val_loss 2.625572, macro_f1 0.401490, macro_auc 0.634938, micro_f1 0.560575
  - step 8000: val_loss 3.011741, macro_f1 0.397748, macro_auc 0.634498, micro_f1 0.575829
  - step 10000/final: val_loss 2.954768, macro_f1 0.360985, macro_auc 0.636551, micro_f1 0.546788

### Comparison boundary vs BCE 1k source

The completed fixed-split 1k source runs are not directly comparable to historical P0 table rows, but they are comparable to each other within this data-scaling protocol:

| run | labels | macro_f1 | macro_auc | micro_f1 |
|---|---:|---:|---:|---:|
| BCE 1k source final | 14 | 0.886060 | 0.684366 | 0.850126 |
| no-LM UMS 1k source final | 12 | 0.360985 | 0.636551 | 0.546788 |

Interpretation:

- This fixed 1k protocol does not support a simple claim that no-LM UMS/source schema alone beats BCE.
- no-LM UMS learns nontrivial 4-state structure (`state_accuracy_all_fields` 0.6796), but present/absent binary metrics are weak under the current readout/evaluation.
- This makes the frozen-LM 1k source run more important: it can test whether frozen LM/schema serialization improves low-data UMS beyond the no-LM classifier.

### 失败 / 异常 / 限制

- No process-level failure; command exited with `EXITCODE 0`.
- Log contains garbled tqdm block characters due Windows console encoding; not a training failure.
- The no-LM source uses 12 selected labels, whereas the BCE source config used 14 labels. Comparison must flag this label-set mismatch.
- The final checkpoint is not necessarily the best binary-metric checkpoint; step-level metrics show some earlier steps have higher macro_f1 or micro_f1.
- This result should not be used to claim no-LM UMS is invalid globally; it is a fixed-split 1k source result under the current 4-state-to-binary evaluation path.

### 下一步

- Refresh `outputs/final_tables/revision_execution_status.csv/.md` to mark `P1_DATA_SCALING_1K_NO_LM_UMS_DEBUG_ENTRY` and `P1_DATA_SCALING_1K_NO_LM_UMS_SOURCE_RUN` completed.
- Next executable source run should be `configs/data_scaling/frozen_lm_ums_1k.yaml`, because frozen-LM is now the missing member of the 1k BCE/no-LM/frozen-LM source triad.
- Before frozen-LM full run, run a debug entry if needed to confirm Qwen cache/model creation and avoid starting a multi-hour run with missing LLM weights.
- Do not start LP runs until corresponding source checkpoints exist and the label-set/checkpoint-path boundary is reviewed.

## 2026-06-17 Phase 1 / P1_RUNTIME_SLOWDOWN_RECHECK 执行记录

### 任务计划

1. 回答“这台电脑是不是因为病毒、自动化进程或其他占用导致异常慢”。
2. 只做诊断，不启动新训练。
3. 分层检查：
   - GPU/process: 是否有残留 `python.exe`、`git.exe`、训练进程或未知 GPU consumer。
   - System load: 实时 CPU、内存、磁盘占用。
   - Automation/startup: 远控、VPN、同步、更新器、Codex/Git 自动扫描。
   - Security state: Windows Defender/威胁记录是否可用。
4. 将结论限定为“当前证据支持/不支持”，不在 Defender 关闭时宣称系统已确认无病毒。

### 执行命令

```powershell
git status --short
Get-Process | Sort-Object CPU -Descending | Select-Object -First 20 Id,ProcessName,CPU,PM,WS,StartTime
nvidia-smi
Get-Process python,powershell,git,codex -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,PM,WS,StartTime,Path
Get-Counter '\Processor(_Total)\% Processor Time','\Memory\Available MBytes','\PhysicalDisk(_Total)\% Disk Time' -SampleInterval 1 -MaxSamples 5
Get-MpComputerStatus | Select-Object AMServiceEnabled,AntivirusEnabled,RealTimeProtectionEnabled,BehaviorMonitorEnabled,IoavProtectionEnabled,AntispywareEnabled,QuickScanAge,FullScanAge,AntivirusSignatureLastUpdated
Get-MpThreat
Get-CimInstance Win32_StartupCommand | Select-Object Name,Command,Location,User | Sort-Object Name
Get-ScheduledTask | Where-Object { $_.State -eq 'Running' } | Select-Object TaskName,TaskPath,State
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|git|powershell|codex|GameViewer|Sunlogin|ToDesk|Baidu|Docker|Ollama|Warp|Quark|Tailscale|Motion|WebVPN' } | Select-Object ProcessId,Name,ExecutablePath,CommandLine
Test-Path .git\index.lock
Get-PSDrive -PSProvider FileSystem | Select-Object Name,Used,Free,Root
```

### 输入

- 当前 Windows process table。
- NVIDIA driver/GPU process table。
- Windows Defender status/threat table。
- Startup command registry。
- Running scheduled tasks。
- Current repo `.git` state。
- Current training logs under `outputs/logs/`。

### 输出

- Process/GPU diagnosis in this section。
- Updated `outputs/final_tables/revision_execution_status.csv/.md` after `python scripts\summarize_revision_execution_status.py`。
- No new model checkpoint and no new training run。

### 停止条件

- 若发现未知 GPU process、残留 training `python.exe`、active `git.exe` scan、`.git/index.lock`、CPU/disk saturation、或 Defender threat entry，先停止后续实验并处理该层。
- 若诊断命令本身超时但已有足够层级证据，记录超时并继续只读检查。

### 结果

System/GPU layer:

- `nvidia-smi`: GPU0/GPU1 idle，均约 13MiB，仅 Codex app C+G context；没有 unknown compute process。
- `tasklist/Get-Process`: no residual training `python.exe` after `P1_DATA_SCALING_1K_NO_LM_UMS_SOURCE_RUN`。
- `Get-Process git`: no active `git.exe`。
- `.git/index.lock`: `False`。
- Disk: H: free about 666GB；不是输出空间瓶颈。
- `Get-Counter` realtime samples: CPU about 18-24%，available memory about 50GB，disk time near 0%。该命令最后因 timeout 提前结束，但已返回多条有效 sample。

Automation/startup layer:

- 常驻/启动项包括：`ToDesk`、`GameViewer`、`SunloginClient`、`Tailscale`、`MotionPro/WebVPN`、`Docker Desktop`、`Ollama`、`Warp`、`QuarkUpdater`、`BaiduYunDetect`、`OneDrive`、`WPS cloud service`。
- Running scheduled tasks include `Clash Verge (Admin)` and Windows background tasks。
- Realtime CPU delta sample 中主要 activity 来自当前 PowerShell 诊断、Codex、WPS cloud service、Task Manager、Edge WebView、ToDesk；没有 `python.exe` 或 `git.exe`。
- Earlier slowdown source remains consistent with Codex/Git UI scanning large untracked/ignored experiment artifacts; `.gitignore` now includes `data/splits/` and `History/` while `outputs/` and `data/dataset/` were already ignored。

Security layer:

- `Get-MpComputerStatus`: `AMServiceEnabled=False`、`AntivirusEnabled=False`、`RealTimeProtectionEnabled=False`、`BehaviorMonitorEnabled=False`、`IoavProtectionEnabled=False`、`AntispywareEnabled=False`。
- `Get-MpThreat`: no rows。
- Because Defender is disabled and scan ages are unset (`4294967295`), this diagnostic cannot certify the machine is virus-free。

Training/runtime layer:

- `outputs/logs/data_scaling_no_lm_ums_1k_source.log` ends with `Training completed!` and `EXITCODE 0`。
- The no-LM UMS formal run took about `1:43:34` for 10000 steps, mostly around 1.6-2.2 it/s near the end. Given ViT-B, image loading, validation, checkpointing, and Windows overhead, this is not by itself evidence of malware。
- Current bottleneck for next frozen-LM run is likely model/cache startup and LLM forward cost, not system saturation。

### 失败 / 注意

- One PowerShell realtime CPU delta command had a parser error due an invalid pipeline after `foreach`; rerun with an explicit `$rows` array succeeded。
- `Get-Counter` timed out after returning partial valid samples; treated as partial evidence, not a failed system check。
- A Python heredoc command using Unix `python - <<'PY'` failed in PowerShell; rerun should use `@' ... '@ | python -`。
- `Get-MpThreat` being empty is weak evidence only because Defender protection is disabled。

### 当前判断

- 当前证据不支持“未知 GPU miner / hidden Python training / disk saturation / active Git scan 正在拖慢机器”。
- 当前证据支持“Windows 常驻远控/同步/VPN/更新器较多 + Codex/Git 曾经扫描大型 repo artifacts + CUDA/timm/transformers import/model load 本身慢”这一解释。
- 若要做真正病毒排查，需要先启用可信 AV/Defender 或离线扫描；本实验流程只能报告运行时证据，不能替代杀毒结论。

### 下一步

- frozen-LM 运行前先做 Qwen cache/model preflight；如果 snapshot 不完整则下载/修复。
- 启动正式 frozen-LM 前继续确认：no `python.exe` training process、no `git.exe` scan、GPU idle、`metrics_final.json` 不存在。
- 尽量不要在训练时同时打开大量 repo diff/status UI；保持 `outputs/`、`data/splits/`、`History/` ignored。

## 2026-06-17 Phase 1 / P1_DATA_SCALING_1K_FROZEN_LM_CACHE_PREFLIGHT 执行前记录

### 任务计划

1. 只验证 `configs/data_scaling/frozen_lm_ums_1k.yaml` 的 frozen-LM 前置条件。
2. 不启动正式 10000-step frozen-LM training。
3. 检查 `Qwen/Qwen2.5-1.5B-Instruct` HuggingFace cache 是否完整。
4. 如果 cache 缺失或 snapshot incomplete，则按用户指示下载/修复，但仍停在 preflight 层。
5. 运行最小模型加载/搬运测试，确认当前 Python/CUDA/transformers 可以加载 Qwen 1.5B。
6. 运行 `train_cxr.py --debug` 只作为 pipeline entry check；注意 debug 默认替换为 `sshleifer/tiny-gpt2`，不能证明 Qwen full path。

### 计划命令

```powershell
python -m py_compile scripts\train_cxr.py models\vivid_model.py training\trainer.py
nvidia-smi
tasklist /fi "imagename eq python.exe"
Get-Process git -ErrorAction SilentlyContinue
Test-Path outputs\data_scaling\frozen_lm_ums_1k\metrics_final.json
Get-Content H:\.cache\huggingface\hub\models--Qwen--Qwen2.5-1.5B-Instruct\refs\main
Get-ChildItem -Force H:\.cache\huggingface\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots -Directory
Get-ChildItem -Force H:\.cache\huggingface\hub\models--Qwen--Qwen2.5-1.5B-Instruct\blobs | Sort-Object Length -Descending
```

Cache integrity / download repair:

```powershell
$env:HF_ENDPOINT='https://huggingface.co'
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_files_only=True))"
```

If local-only fails:

```powershell
$env:HF_ENDPOINT='https://huggingface.co'
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-1.5B-Instruct'))"
```

Minimal load check:

```powershell
$env:HF_ENDPOINT='https://huggingface.co'
python -c "import torch; from transformers import AutoTokenizer, AutoModelForCausalLM; m='Qwen/Qwen2.5-1.5B-Instruct'; tok=AutoTokenizer.from_pretrained(m, trust_remote_code=True); model=AutoModelForCausalLM.from_pretrained(m, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation='eager'); model=model.to('cuda:0'); print(tok.__class__.__name__, model.config.hidden_size, next(model.parameters()).device, next(model.parameters()).dtype)"
```

Debug entry:

```powershell
$env:HF_ENDPOINT='https://huggingface.co'
python scripts\train_cxr.py --config configs\data_scaling\frozen_lm_ums_1k.yaml --debug --seed 900119
```

### 输入

- `configs/data_scaling/frozen_lm_ums_1k.yaml`
- HF cache under `H:\.cache\huggingface`
- cached/fetched `Qwen/Qwen2.5-1.5B-Instruct`
- cached/fetched `sshleifer/tiny-gpt2` for debug mode
- `data/splits/chexpert_train_1k.jsonl`
- `data/splits/chexpert_val_fixed.jsonl`

### 输出

- Preflight console/log evidence。
- If debug entry succeeds:
  - `outputs/data_scaling/frozen_lm_ums_1k_seed900119/metrics_final.json`
  - `outputs/data_scaling/frozen_lm_ums_1k_seed900119/final.pt`
- If cache repair downloads files:
  - updated HF cache under `H:\.cache\huggingface\hub\models--Qwen--Qwen2.5-1.5B-Instruct`

### 停止条件

- Formal output `outputs/data_scaling/frozen_lm_ums_1k/metrics_final.json` already exists。
- Any residual training `python.exe` or active `git.exe` scan is present before launch。
- GPU is not idle before load check。
- Local cache is incomplete and network download fails。
- Qwen minimal load cannot complete or OOMs。
- Debug run exceeds 10 minutes, exits nonzero, or leaves orphan Python process。
- Any command attempts to start formal frozen-LM training or another Phase 1 matrix job unintentionally。

## 2026-06-17 Phase 1 / P1_DATA_SCALING_1K_FROZEN_LM_CACHE_PREFLIGHT 执行后记录

### 执行命令

```powershell
python -m py_compile scripts\train_cxr.py models\vivid_model.py training\trainer.py
nvidia-smi
tasklist /fi "imagename eq python.exe"
Get-Process git -ErrorAction SilentlyContinue
Test-Path outputs\data_scaling\frozen_lm_ums_1k\metrics_final.json
$env:HF_ENDPOINT='https://huggingface.co'
$env:HF_HUB_DISABLE_XET='1'
python -u -c "from huggingface_hub import snapshot_download; import os; p=snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_files_only=True); print('SNAPSHOT', p); print('FILES', os.listdir(p))"
python -u -c "from huggingface_hub import snapshot_download; infos=snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', dry_run=True, max_workers=4); print('N', len(infos)); [print(getattr(x,'filename',None), getattr(x,'is_cached',None), getattr(x,'will_download',None)) for x in infos[:20]]"
python -u -c "from huggingface_hub import snapshot_download; import os; p=snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', max_workers=4); print('SNAPSHOT', p); print('FILES', sorted(os.listdir(p)))"
python -u -  # AutoTokenizer/AutoModelForCausalLM Qwen minimal load and cuda move check
python scripts\train_cxr.py --config configs\data_scaling\frozen_lm_ums_1k.yaml --debug --seed 900119
```

### 结果

- Preflight compile passed。
- Initial process/GPU guard:
  - GPUs idle before model load。
  - no residual `python.exe` before Qwen load。
  - formal output `outputs/data_scaling/frozen_lm_ums_1k/metrics_final.json`: `False`。
  - one transient `python.exe` from the parallel guard disappeared on recheck。
  - one transient set of `git.exe` processes appeared after config edit but disappeared within 5 seconds; `.git/index.lock=False`。
- HuggingFace cache:
  - `HF_HOME=H:\.cache\huggingface`
  - `refs/main=989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
  - local-only `snapshot_download` initially returned an empty snapshot (`FILES []`) despite blobs totaling about 3.10GB。
  - `dry_run=True` reported 10 expected files; large `model.safetensors` and tokenizer/config files were cached, while small metadata files were missing。
  - non-local `snapshot_download` repaired the snapshot; final files:
    - `.gitattributes`
    - `LICENSE`
    - `README.md`
    - `config.json`
    - `generation_config.json`
    - `merges.txt`
    - `model.safetensors`
    - `tokenizer.json`
    - `tokenizer_config.json`
    - `vocab.json`
  - Windows symlink support is disabled for the HF cache, so HuggingFace uses degraded non-symlink caching; this can cost extra disk and copy time。
- Qwen minimal load:
  - tokenizer: `Qwen2Tokenizer`, about 21.84s
  - model CPU load: `Qwen2ForCausalLM`, hidden size 1536, about 22.69s
  - move to GPU0: about 9.61s
  - CUDA memory allocated: about 2945.3MB
  - completed and freed GPU memory。
- First debug entry failed before training because `train_cxr.py --debug` defaulted to `sshleifer/tiny-gpt2`:
  - `transformers` refused to load `.bin` weights with current `torch 2.5.1+cu121` because it now requires `torch>=2.6` for `torch.load`-based checkpoint loading。
  - This does not apply to Qwen because Qwen uses `safetensors`。
- Config-specific repair:
  - Added `debug_llm_model_name: Qwen/Qwen2.5-1.5B-Instruct` to `configs/data_scaling/frozen_lm_ums_1k.yaml`。
  - Scope is limited to this config's debug behavior; formal `llm_model_name` already used Qwen。
- Qwen debug entry succeeded:
  - train: 20 samples from `data/splits/chexpert_train_1k.jsonl`
  - val: 4 samples from `data/splits/chexpert_val_fixed.jsonl`
  - model: ViT-B + frozen `Qwen/Qwen2.5-1.5B-Instruct`
  - trainable params: 89,349,888
  - frozen params: 1,543,714,304
  - max_steps: 5
  - validation losses:
    - step 2: 1.0007
    - step 4: 0.9703
  - outputs:
    - `outputs/data_scaling/frozen_lm_ums_1k_seed900119/checkpoints/best.pt` (1,072,361,762 bytes)
    - `outputs/data_scaling/frozen_lm_ums_1k_seed900119/checkpoints/step_5.pt` (1,072,387,794 bytes)
    - `outputs/data_scaling/frozen_lm_ums_1k_seed900119/checkpoints/final.pt` (1,072,362,394 bytes)
  - after run: no `python.exe`, no `git.exe`, GPUs idle except Codex app context。

### 失败 / 注意

- The original debug expectation that `train_cxr.py` would emit `metrics_final.json` was wrong for this source trainer. `VIVIDTrainer` saves checkpoints and prints validation loss; classification metrics require downstream LP/eval。
- `ModelScope` is installed. During `train_cxr.py --debug`, `VIVIDModel` tried ModelScope first and used `C:\Users\Admin\.cache\modelscope\hub\models\Qwen\Qwen2___5-1___5B-Instruct`。
- ModelScope cache already contained the same Qwen safetensors model; `.mdl/.mv` metadata timestamps were refreshed during the debug run。
- Qwen debug validates the full frozen-LM path, but it is not formal data-scaling evidence。
- The full source config has `batch_size=4` and `gradient_accumulation_steps=8`; one formal optimizer step requires 8 micro-batches. Based on the debug timing, the formal 10000-step source run can be long-running and should not be mistaken for an abnormal virus/hidden-process slowdown。

### 下一步

- Update `outputs/final_tables/revision_execution_status.csv/.md` with this preflight。
- Before launching formal `P1_DATA_SCALING_1K_FROZEN_LM_UMS_SOURCE_RUN`, write a separate execution-before record。
- Formal source success criterion is checkpoint/provenance, not `metrics_final.json`。
- LP metrics for frozen-LM 1k remain blocked until formal source checkpoint exists。

## 2026-06-17 Phase 1 / P1_DATA_SCALING_1K_FROZEN_LM_UMS_SOURCE_RUN 执行前记录

### 任务计划

1. 只启动正式 frozen-LM UMS 1k source run。
2. 不启动 BCE/no-LM rerun、LP runs、schema sweep、random-LM、SPD variants 或完整 data-scaling matrix。
3. 使用正式 config：
   - `configs/data_scaling/frozen_lm_ums_1k.yaml`
4. 使用 GPU0：
   - `CUDA_VISIBLE_DEVICES=0`
5. 保留：
   - `HF_ENDPOINT=https://huggingface.co`
   - `HF_HUB_DISABLE_XET=1`
6. 记录到日志并监控到稳定运行或失败。
7. 该 source run 只产生 frozen-LM source checkpoint；分类指标需要后续 LP run。

### 计划命令

```powershell
$env:HF_ENDPOINT='https://huggingface.co'
$env:HF_HUB_DISABLE_XET='1'
$env:CUDA_VISIBLE_DEVICES='0'
python scripts\train_cxr.py --config configs\data_scaling\frozen_lm_ums_1k.yaml
```

后台日志：

```powershell
outputs/logs/data_scaling_frozen_lm_ums_1k_source.log
```

### 输入

- `configs/data_scaling/frozen_lm_ums_1k.yaml`
- `data/splits/chexpert_train_1k.jsonl`
- `data/splits/chexpert_val_fixed.jsonl`
- local CheXpert image files under `data/dataset`
- cached ViT-B pretrained weights
- cached `Qwen/Qwen2.5-1.5B-Instruct` under HF and/or ModelScope cache

### 输出

- `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/best.pt`
- `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/final.pt`
- periodic `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/step_*.pt`
- `outputs/logs/data_scaling_frozen_lm_ums_1k_source.log`
- Expected final line in wrapper log: `EXITCODE 0`

### 停止条件

- `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/final.pt` already exists before launch。
- Any residual training `python.exe` or active `git.exe` scan is present before launch。
- GPU0 is not idle before launch。
- Qwen cache/model load fails or OOMs。
- Training exits nonzero。
- Run leaves orphan Python process after failure。
- Training speed indicates a multi-day runtime and no stable checkpoint/progress can be produced。
- Any sign that LP/schema/full matrix jobs started unintentionally。

## 2026-06-17 Phase 1 / P1_DATA_SCALING_1K_FROZEN_LM_UMS_SOURCE_RUN 运行中记录

### 启动命令

Initial `EncodedCommand`/PowerShell wrapper was not retained because it failed to produce reliable live logs. A reproducible wrapper was added instead:

```powershell
scripts\run_data_scaling_frozen_lm_ums_1k_source.ps1
scripts\run_data_scaling_frozen_lm_ums_1k_source.cmd
```

The `.ps1` wrapper exposed a PowerShell `Tee-Object` encoding/logging problem and was not used for the stable run. The active stable run uses:

```powershell
Start-Process -FilePath cmd.exe -ArgumentList @('/c','H:\Xiyao_Wang\021_260129VIVID\scripts\run_data_scaling_frozen_lm_ums_1k_source.cmd') -WindowStyle Hidden -PassThru
```

Actual command inside the wrapper:

```cmd
set HF_ENDPOINT=https://huggingface.co
set HF_HUB_DISABLE_XET=1
set CUDA_VISIBLE_DEVICES=0
set PYTHONIOENCODING=utf-8
python -u scripts\train_cxr.py --config configs\data_scaling\frozen_lm_ums_1k.yaml >> outputs\logs\data_scaling_frozen_lm_ums_1k_source.log 2>&1
```

### Current process/log evidence

- Stable wrapper PID: `17936` (`cmd.exe`)
- Stable Python PID: `17472`
- Log: `outputs/logs/data_scaling_frozen_lm_ums_1k_source.log`
- Training output dir: `outputs/data_scaling/frozen_lm_ums_1k`
- Formal source trainer does not emit `metrics_final.json`; it emits checkpoints and validation loss logs。

### Initial runtime observations

- Model creation succeeded:
  - train samples: 1000
  - val samples: 1000
  - labels: 12 selected CheXpert labels
  - train batches: 250
  - val batches: 250
  - LLM: `Qwen/Qwen2.5-1.5B-Instruct`
  - trainable params: 89,349,888
  - frozen params: 1,543,714,304
- FlashAttention2 is not installed; model falls back to eager attention。
- GPU0 during training:
  - memory about 24,289MiB / 24,576MiB
  - utilization observed around 28-44%
  - power about 175-196W
  - temperature about 51-60C
- Early progress:
  - step 10: about 1:26 elapsed, loss 0.8724
  - step 20: about 2:40 elapsed, loss 0.6793
  - step 23: about 3:05 elapsed
- Rough ETA from early progress: about 18-22 hours for 10000 optimizer steps, before accounting for validation/checkpoint overhead。

### 失败 / 注意

- First background launch with a PowerShell encoded command timed out before starting a durable process/log。
- Second launch with `.ps1`/`Tee-Object` entered dataloader/model creation but exited without an `EXITCODE`; foreground diagnostics showed the model path itself was healthy, so this was treated as wrapper/logging failure rather than training failure。
- Foreground formal create-model diagnostic succeeded:
  - dataloaders OK: 250 train batches / 250 val batches
  - model OK on `cuda:0`
  - Qwen and ViT load path OK
- `.cmd` raw-byte redirection fixed the live logging issue。
- The run is expected to be slow and memory-bound/near-capacity on a 24GB RTX 3090; this supports the runtime diagnosis that slowness is workload/model-size driven rather than unknown malware or hidden automation。

### 下一步

- Continue monitoring until at least the first validation/checkpoint boundary.
- If OOM/nonzero exit occurs, record as GPU memory/resource failure and consider smaller batch/gradient accumulation only after documenting the failure.
- If first checkpoint appears and speed remains stable, keep the run active until completion before launching LP。

## 2026-06-17 Phase 1 / P1_DATA_SCALING_1K_FROZEN_LM_UMS_SOURCE_RUN first-checkpoint 记录

### 监控命令

```powershell
Get-Content -Tail 140 outputs\logs\data_scaling_frozen_lm_ums_1k_source.log
nvidia-smi
Get-Process -Id 17472,17936 -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,PM,WS,StartTime
Get-ChildItem -Force outputs\data_scaling\frozen_lm_ums_1k -Recurse
```

### 结果

- Active wrapper PID: `17936`
- Active Python PID: `17472`
- First validation completed at step 500。
- Validation:
  - `Step 500: val_loss = 0.0514`
  - validation batches: 250/250
  - validation duration from tqdm: about 1:51
- Checkpoint:
  - `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/best.pt`
  - size: 1,072,361,762 bytes
  - timestamp: `2026-06-17 21:07:09`
- Training continued after validation:
  - step 523 observed at about 1:18:58 elapsed
  - no `EXITCODE` yet because run remains active
- GPU/process:
  - GPU0 memory about 24,291MiB / 24,576MiB
  - GPU0 temperature about 65C
  - GPU0 power about 221W in one sample
  - Python working set about 14.1GB; private memory about 41.2GB

### 失败 / 注意

- No OOM or nonzero exit through first checkpoint。
- GPU memory remains extremely close to capacity. Avoid launching any other GPU workload while this run is active。
- The first checkpoint makes LP technically unblocked for this source, but LP must not start until the source run is either completed or intentionally stopped with documented rationale。

### 下一步

- Continue the source run toward completion。
- Next checkpoint boundary: step 1000 (`save_interval=1000`) plus validation at step 1000。
- If runtime must be shortened later, use this first-checkpoint record as the minimum evidence that the frozen-LM 1k source path is executable; do not treat it as complete source-run evidence。

## 2026-06-17 Phase 1 / P1_DATA_SCALING_1K_FROZEN_LM_UMS_SOURCE_RUN step-1000 记录

### 监控命令

```powershell
Get-Content -Tail 160 outputs\logs\data_scaling_frozen_lm_ums_1k_source.log
nvidia-smi
Get-Process -Id 17472,17936 -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,PM,WS,StartTime
Get-ChildItem -Force outputs\data_scaling\frozen_lm_ums_1k\checkpoints
```

### 结果

- Active wrapper PID: `17936`
- Active Python PID: `17472`
- Step 1000 validation completed。
- Validation:
  - `Step 1000: val_loss = 0.0487`
  - validation batches: 250/250
  - validation duration from tqdm: about 1:55
- Checkpoints:
  - `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/best.pt`
    - size: 1,072,361,762 bytes
    - timestamp refreshed at `2026-06-17 22:12:57`
  - `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/step_1000.pt`
    - size: 1,072,395,066 bytes
    - timestamp: `2026-06-17 22:13:00`
- Training continued after checkpoint:
  - latest observed around step 1093 at about `2:35:59` elapsed
  - no `EXITCODE` yet because run remains active
- Runtime:
  - speed generally fluctuates around 6-10 seconds/step after warmup, with occasional 20-30s spikes around validation/checkpoint/IO or Windows scheduling。
  - no sustained slowdown was observed after step 1000。
  - current full-run ETA remains roughly 23-26 hours total, depending on validation/checkpoint overhead。
- GPU/process:
  - GPU0 memory about 24,291MiB / 24,576MiB
  - GPU0 temperature around 59-60C in latest samples
  - GPU0 utilization is bursty; instantaneous `nvidia-smi` samples range widely and should not be interpreted alone。
  - GPU1 remains idle。

### 当前判断

- Training speed is slow but normal for this formal frozen-LM/Qwen path on a 24GB RTX 3090。
- There is no evidence that the current slowdown is caused by virus/miner/hidden Python training/Git scan。
- The main constraint is that frozen-LM source training keeps GPU0 near memory capacity and performs expensive LLM forward/backward-through-input steps。

### 下一步

- Continue the run toward completion。
- Next checkpoint boundary: step 2000 validation/checkpoint。
- Do not start LP/schema sweep while this source run is active。

## 2026-06-17 Phase 1 / P1_DATA_SCALING_1K_FROZEN_LM_UMS_SOURCE_RUN step-1500 speed-sanity 记录

### 监控计划

1. Verify the active frozen-LM source process is still alive and responsive。
2. Read the latest log boundary lines for validation, errors, and progress。
3. Check checkpoint directory to distinguish expected no-save validation from a failed save。
4. Sample GPU state, but do not interpret a single utilization sample as training speed。

### 监控命令

```powershell
rg -n "Step [0-9]+: val_loss|EXITCODE|Error|Traceback|out of memory" outputs\logs\data_scaling_frozen_lm_ums_1k_source.log
Get-Content -Tail 40 outputs\logs\data_scaling_frozen_lm_ums_1k_source.log
Get-Item outputs\logs\data_scaling_frozen_lm_ums_1k_source.log | Select-Object LastWriteTime,Length
Get-ChildItem outputs\data_scaling\frozen_lm_ums_1k\checkpoints | Sort-Object LastWriteTime
Get-Process -Id 17472,17936 -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime,Responding,WorkingSet64,PrivateMemorySize64
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

### 输入

- Active log: `outputs/logs/data_scaling_frozen_lm_ums_1k_source.log`
- Active output dir: `outputs/data_scaling/frozen_lm_ums_1k`
- Active wrapper PID: `17936`
- Active Python PID: `17472`

### 输出 / 结果

- Process:
  - wrapper `cmd.exe` PID `17936` remains alive and responding。
  - Python PID `17472` remains alive and responding。
  - Python CPU time sampled at about `11476.98s`; working set about `14.25GB`; private memory about `41.19GB`。
- Validation boundary:
  - `Step 500: val_loss = 0.0514`
  - `Step 1000: val_loss = 0.0487`
  - `Step 1500: val_loss = 0.0513`
  - Step 1500 validation completed 250/250 batches; tqdm duration about `2:39`。
- Checkpoints:
  - `best.pt` remains the step-1000 best checkpoint, timestamp `2026-06-17 22:12:57`。
  - `step_1000.pt` remains the latest periodic checkpoint, timestamp `2026-06-17 22:13:00`。
  - No `step_1500.pt` is expected because `save_interval=1000`; no new best checkpoint is expected because `0.0513` is worse than the current best `0.0487`。
- Latest observed training progress:
  - training continued after validation through at least step `1526` at about `3:41:19` elapsed。
  - post-validation steps 1501-1526 include normal short-term fluctuations, roughly 7-22s/step in the sampled tail。
- GPU:
  - GPU0 memory remains about `24291MiB / 24576MiB`。
  - GPU0 sampled temperature about `51-56C`。
  - GPU0 sampled power about `104-193W`。
  - GPU0 sampled utilization ranged from `18%` during/near validation to `100%` during active compute。
  - GPU1 remains idle except for minimal desktop/Codex allocation。
- 10-second CPU delta sample:
  - training Python PID `17472`: about `14.53 CPU-seconds / 10s`, the dominant compute process。
  - Codex/PowerShell monitoring processes were visible but expected from this diagnostic session。
  - `ToDesk` and `GameViewer` were present, but their sampled CPU use was low and they did not appear as GPU compute workloads。

### 指标 / 当前速度判断

- The run is slow but still normal for this frozen-LM/Qwen 1.5B source path on a 24GB RTX 3090。
- The apparent slowdown is mostly validation/checkpoint/progress-bar artifact plus near-capacity GPU memory pressure。
- No sustained hang is visible: log timestamp continues to move, process is responsive, GPU0 remains occupied by the expected Python training process, and no OOM/Traceback/nonzero exit has appeared。
- Step 1500 validation is slower than step 1000 validation (`2:39` vs about `1:55`), but still completed; treat this as runtime variability unless the next 100+ training steps stay above about 20s/step or the log stops advancing。
- No evidence in the sampled process/GPU view suggests a hidden miner, second Python training job, or high-CPU remote-control process causing the current slowdown。

### 失败 / 停止条件

- No current failure。
- Stop and diagnose if any of the following appears:
  - `Traceback`, CUDA OOM, or nonzero `EXITCODE` in the log。
  - Python PID `17472` exits while wrapper remains active。
  - log `LastWriteTime` stops advancing for more than 10 minutes while GPU0 utilization/power are low。
  - step cadence stays above about 20s/step for more than 100 consecutive training steps outside validation/checkpoint windows。
  - any LP/schema/matrix job starts before this source run is complete or intentionally stopped。

### 下一步

- Continue monitoring the active source run。
- Next hard checkpoint boundary: step 2000 validation plus `step_2000.pt` periodic checkpoint。
- Do not launch LP/schema sweep while this source run is active。

## 2026-06-18 Phase 1 / P1_DATA_SCALING_1K_FROZEN_LM_UMS_SOURCE_RUN step-2000 记录

### 监控计划

1. Wait for the step 2000 validation/checkpoint boundary。
2. Confirm whether validation finishes, whether `best.pt` refreshes, and whether `step_2000.pt` is saved。
3. Confirm the training process continues after checkpoint rather than exiting or hanging。
4. Record runtime/speed interpretation without treating source checkpoint loss as final LP/classification metrics。

### 监控命令

```powershell
rg -n "Step [0-9]+: val_loss|EXITCODE|Traceback|out of memory|Error" outputs\logs\data_scaling_frozen_lm_ums_1k_source.log
Get-Content -Tail 80 outputs\logs\data_scaling_frozen_lm_ums_1k_source.log
Get-ChildItem outputs\data_scaling\frozen_lm_ums_1k\checkpoints | Sort-Object LastWriteTime
Get-Process -Id 17472,17936 -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime,Responding,WorkingSet64,PrivateMemorySize64
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

### 输入

- Active log: `outputs/logs/data_scaling_frozen_lm_ums_1k_source.log`
- Active output dir: `outputs/data_scaling/frozen_lm_ums_1k`
- Active wrapper PID: `17936`
- Active Python PID: `17472`

### 输出 / 结果

- Validation boundary:
  - `Step 500: val_loss = 0.0514`
  - `Step 1000: val_loss = 0.0487`
  - `Step 1500: val_loss = 0.0513`
  - `Step 2000: val_loss = 0.0481`
  - Step 2000 validation completed 250/250 batches; tqdm duration about `1:35`。
- Checkpoints:
  - `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/best.pt`
    - refreshed because `0.0481` improved over prior best `0.0487`
    - timestamp: `2026-06-18 00:42:35`
    - size: `1,072,361,762` bytes
  - `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/step_2000.pt`
    - timestamp: `2026-06-18 00:42:41`
    - size: `1,072,395,066` bytes
  - prior `step_1000.pt` remains available。
- Training continued after checkpoint:
  - latest observed after this boundary: step `2004` at about `4:52:08` elapsed。
  - no `EXITCODE` yet because the source run remains active。
- Process/GPU:
  - wrapper `cmd.exe` PID `17936` remains alive and responding。
  - Python PID `17472` remains alive and responding。
  - Python working set about `14.26GB`; private memory about `41.19GB`。
  - GPU0 memory remains about `24291MiB / 24576MiB`。
  - GPU0 sampled temperature about `61C`; power about `239W`; sampled utilization about `54%` after checkpoint。
  - GPU1 remains idle except for minimal desktop/Codex allocation。

### 指标 / 当前解释

- Step 2000 confirms the formal frozen-LM/Qwen source path is not merely alive: it reaches repeated validation/checkpoint boundaries and improves source validation loss。
- Runtime is slow but normal for this path. The apparent earlier slowdown was transient; later training segments reached roughly 5-10s/step before validation。
- This is still source-run evidence only. It does not produce `metrics_final.json`, AUROC/AUPRC, or LP classification results。

### 失败 / 停止条件

- No current failure。
- No OOM, Traceback, nonzero exit, orphan process, or checkpoint write failure through step 2000。
- Stop/diagnose if a later checkpoint fails to save, if process exits without `EXITCODE 0`, or if log/GPU activity stalls for more than 10 minutes outside validation/checkpoint windows。

### 下一步

- Continue the active frozen-LM source run toward completion。
- Next validation boundary: step 2500。
- Next periodic checkpoint boundary: step 3000。
- Do not launch LP/schema sweep while this source run is still active。

## 2026-06-18 Phase 1 / P1_DATA_SCALING_1K_FROZEN_LM_UMS_SOURCE_RUN step-2500 记录

### 监控计划

1. Wait for the step 2500 validation boundary。
2. Confirm whether validation finishes and whether any checkpoint behavior matches config expectations。
3. Record current source validation trajectory and failure status。
4. Keep LP/schema sweep blocked while the source run remains active。

### 监控命令

```powershell
rg -n "Step [0-9]+: val_loss|EXITCODE|Traceback|out of memory|Error" outputs\logs\data_scaling_frozen_lm_ums_1k_source.log
Get-Content -Tail 80 outputs\logs\data_scaling_frozen_lm_ums_1k_source.log
Get-ChildItem outputs\data_scaling\frozen_lm_ums_1k\checkpoints | Sort-Object LastWriteTime
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

### 输入

- Active log: `outputs/logs/data_scaling_frozen_lm_ums_1k_source.log`
- Active output dir: `outputs/data_scaling/frozen_lm_ums_1k`
- Current best source checkpoint: `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/best.pt`

### 输出 / 结果

- Validation boundary:
  - `Step 500: val_loss = 0.0514`
  - `Step 1000: val_loss = 0.0487`
  - `Step 1500: val_loss = 0.0513`
  - `Step 2000: val_loss = 0.0481`
  - `Step 2500: val_loss = 0.0533`
  - Step 2500 validation completed 250/250 batches; tqdm duration about `1:41`。
- Checkpoints:
  - No `step_2500.pt` is expected because `save_interval=1000`。
  - `best.pt` was not refreshed because `0.0533` is worse than the current best `0.0481`。
  - Existing checkpoints remain:
    - `step_1000.pt`
    - `best.pt` from step 2000
    - `step_2000.pt`
- Training continued after validation:
  - latest observed after this boundary: step `2502` at about `5:57:08` elapsed。
  - no `EXITCODE` yet because the source run remains active。
- GPU:
  - GPU0 memory remains about `24291MiB / 24576MiB`。
  - GPU0 sampled temperature about `61C`; power about `158W`; sampled utilization about `34%` near/after validation。
  - GPU1 remains idle except for minimal desktop/Codex allocation。

### 指标 / 当前解释

- Source validation loss trajectory through step 2500: `0.0514 -> 0.0487 -> 0.0513 -> 0.0481 -> 0.0533`。
- Step 2500 is worse than the step 2000 best, but it is a normal validation fluctuation rather than a runtime failure。
- Runtime remains healthy: validation completed, training resumed, no OOM/Traceback/nonzero exit observed。
- This remains source-run evidence only and still must not be treated as LP/classification metrics。

### 失败 / 停止条件

- No current failure。
- Stop/diagnose if a later checkpoint boundary fails to save, if process exits without `EXITCODE 0`, or if log/GPU activity stalls for more than 10 minutes outside validation/checkpoint windows。

### 下一步

- Continue the active frozen-LM source run toward completion。
- Next hard checkpoint boundary: step 3000 validation plus `step_3000.pt` periodic checkpoint。
- Do not launch LP/schema sweep while this source run is still active。

## 2026-06-18 Phase 1 / P1_DATA_SCALING_1K_FROZEN_LM_UMS_SOURCE_RUN step-3000 记录

### 监控计划

1. Wait for the step 3000 validation/checkpoint boundary。
2. Confirm validation, periodic checkpoint writing, and whether `best.pt` refreshes。
3. Confirm training continues after checkpoint。
4. Preserve the boundary between source loss evidence and downstream LP/classification metrics。

### 监控命令

```powershell
rg -n "Step [0-9]+: val_loss|Checkpoint saved|EXITCODE|Traceback|out of memory|Error" outputs\logs\data_scaling_frozen_lm_ums_1k_source.log
Get-Content -Tail 80 outputs\logs\data_scaling_frozen_lm_ums_1k_source.log
Get-ChildItem outputs\data_scaling\frozen_lm_ums_1k\checkpoints | Sort-Object LastWriteTime
Get-Process -Id 17472,17936 -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime,Responding,WorkingSet64,PrivateMemorySize64
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

### 输入

- Active log: `outputs/logs/data_scaling_frozen_lm_ums_1k_source.log`
- Active output dir: `outputs/data_scaling/frozen_lm_ums_1k`
- Current best source checkpoint: `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/best.pt`

### 输出 / 结果

- Validation boundary:
  - `Step 500: val_loss = 0.0514`
  - `Step 1000: val_loss = 0.0487`
  - `Step 1500: val_loss = 0.0513`
  - `Step 2000: val_loss = 0.0481`
  - `Step 2500: val_loss = 0.0533`
  - `Step 3000: val_loss = 0.0710`
  - Step 3000 validation completed 250/250 batches; tqdm duration about `2:01`。
- Checkpoints:
  - `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/step_3000.pt`
    - timestamp: `2026-06-18 02:54:17`
    - size: `1,072,395,066` bytes
  - `best.pt` was not refreshed because `0.0710` is worse than the current best `0.0481` from step 2000。
  - Existing periodic checkpoints now include `step_1000.pt`, `step_2000.pt`, and `step_3000.pt`。
- Training continued after checkpoint:
  - latest observed after this boundary: step `3006` at about `7:04:05` elapsed。
  - no `EXITCODE` yet because the source run remains active。
- Process/GPU:
  - wrapper `cmd.exe` PID `17936` remains alive and responding。
  - Python PID `17472` remains alive and responding。
  - Python working set about `14.25GB`; private memory about `41.19GB`。
  - GPU0 memory remains about `24291MiB / 24576MiB`。
  - GPU0 sampled temperature about `58C`; power about `205W`; sampled utilization reached `100%` near the checkpoint boundary。
  - GPU1 remains idle except for minimal desktop/Codex allocation。

### 指标 / 当前解释

- Source validation loss trajectory through step 3000: `0.0514 -> 0.0487 -> 0.0513 -> 0.0481 -> 0.0533 -> 0.0710`。
- Step 3000 loss is worse, but the run remains operational and checkpointing works。
- Current best source checkpoint remains step 2000 (`val_loss=0.0481`)。
- This remains source-run evidence only. Do not report it as AUROC/AUPRC/Macro-F1 or downstream LP evidence。

### 失败 / 停止条件

- No runtime failure through step 3000。
- No OOM, Traceback, nonzero exit, orphan process, or checkpoint write failure observed。
- Stop/diagnose if a later checkpoint boundary fails to save, if process exits without `EXITCODE 0`, or if log/GPU activity stalls for more than 10 minutes outside validation/checkpoint windows。

### 下一步

- Continue the active frozen-LM source run toward completion。
- Next validation boundary: step 3500。
- Next periodic checkpoint boundary: step 4000。
- Do not launch LP/schema sweep while this source run is still active。

## 2026-06-18 Phase 1 / P1_DATA_SCALING_1K_FROZEN_LM_UMS_SOURCE_RUN step-3500 记录

### 监控计划

1. Wait for the step 3500 validation boundary。
2. Confirm validation finishes and checkpoint behavior matches `save_interval=1000`。
3. Record validation trajectory and runtime health。
4. Keep downstream LP/schema sweep blocked while the source run remains active。

### 监控命令

```powershell
rg -n "Step [0-9]+: val_loss|EXITCODE|Traceback|out of memory|Error" outputs\logs\data_scaling_frozen_lm_ums_1k_source.log
Get-Content -Tail 80 outputs\logs\data_scaling_frozen_lm_ums_1k_source.log
Get-ChildItem outputs\data_scaling\frozen_lm_ums_1k\checkpoints | Sort-Object LastWriteTime
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

### 输入

- Active log: `outputs/logs/data_scaling_frozen_lm_ums_1k_source.log`
- Active output dir: `outputs/data_scaling/frozen_lm_ums_1k`
- Current best source checkpoint: `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/best.pt`

### 输出 / 结果

- Validation boundary:
  - `Step 500: val_loss = 0.0514`
  - `Step 1000: val_loss = 0.0487`
  - `Step 1500: val_loss = 0.0513`
  - `Step 2000: val_loss = 0.0481`
  - `Step 2500: val_loss = 0.0533`
  - `Step 3000: val_loss = 0.0710`
  - `Step 3500: val_loss = 0.0882`
  - Step 3500 validation completed 250/250 batches; tqdm duration about `1:48`。
- Checkpoints:
  - No `step_3500.pt` is expected because `save_interval=1000`。
  - `best.pt` was not refreshed because `0.0882` is worse than the current best `0.0481` from step 2000。
  - Existing periodic checkpoints remain `step_1000.pt`, `step_2000.pt`, and `step_3000.pt`。
- Training continued after validation:
  - latest observed after this boundary: step `3503` at about `8:09:32` elapsed。
  - no `EXITCODE` yet because the source run remains active。
- GPU:
  - GPU0 memory remains about `24291MiB / 24576MiB`。
  - GPU0 sampled temperature about `62C`; power about `249W`; sampled utilization about `77%`。
  - GPU1 remains idle except for minimal desktop/Codex allocation。

### 指标 / 当前解释

- Source validation loss trajectory through step 3500: `0.0514 -> 0.0487 -> 0.0513 -> 0.0481 -> 0.0533 -> 0.0710 -> 0.0882`。
- The source objective appears to overfit or drift after the step 2000 best, but this is not a runtime failure。
- The run remains operational and training resumes after validation。
- This remains source-run evidence only; downstream LP/classification metrics are still unavailable。

### 失败 / 停止条件

- No runtime failure through step 3500。
- No OOM, Traceback, nonzero exit, orphan process, or checkpoint write failure observed。
- Stop/diagnose if a later checkpoint boundary fails to save, if process exits without `EXITCODE 0`, or if log/GPU activity stalls for more than 10 minutes outside validation/checkpoint windows。

### 下一步

- Continue the active frozen-LM source run toward completion。
- Next hard checkpoint boundary: step 4000 validation plus `step_4000.pt` periodic checkpoint。
- Do not launch LP/schema sweep while this source run is still active。

## 2026-06-18 Phase 1 / P1_DATA_SCALING_1K_FROZEN_LM_UMS_SOURCE_RUN step-4000 记录

### 监控计划

1. Wait for the step 4000 validation/checkpoint boundary。
2. Confirm validation finishes and `step_4000.pt` is saved。
3. Confirm whether `best.pt` refreshes。
4. Record runtime health and the source validation-loss trend。

### 监控命令

```powershell
rg -n "Step [0-9]+: val_loss|Checkpoint saved|EXITCODE|Traceback|out of memory|Error" outputs\logs\data_scaling_frozen_lm_ums_1k_source.log
Get-Content -Tail 80 outputs\logs\data_scaling_frozen_lm_ums_1k_source.log
Get-ChildItem outputs\data_scaling\frozen_lm_ums_1k\checkpoints | Sort-Object LastWriteTime
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

### 输入

- Active log: `outputs/logs/data_scaling_frozen_lm_ums_1k_source.log`
- Active output dir: `outputs/data_scaling/frozen_lm_ums_1k`
- Current best source checkpoint: `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/best.pt`

### 输出 / 结果

- Validation boundary:
  - `Step 500: val_loss = 0.0514`
  - `Step 1000: val_loss = 0.0487`
  - `Step 1500: val_loss = 0.0513`
  - `Step 2000: val_loss = 0.0481`
  - `Step 2500: val_loss = 0.0533`
  - `Step 3000: val_loss = 0.0710`
  - `Step 3500: val_loss = 0.0882`
  - `Step 4000: val_loss = 0.1067`
  - Step 4000 validation completed 250/250 batches; tqdm duration about `1:32`。
- Checkpoints:
  - `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/step_4000.pt`
    - timestamp: `2026-06-18 05:17:37`
    - size: `1,072,395,066` bytes
  - `best.pt` was not refreshed because `0.1067` is worse than the current best `0.0481` from step 2000。
  - Existing periodic checkpoints now include `step_1000.pt`, `step_2000.pt`, `step_3000.pt`, and `step_4000.pt`。
- Training continued after checkpoint:
  - latest observed after this boundary: step `4001` at about `9:26:42` elapsed。
  - no `EXITCODE` yet because the source run remains active。
- GPU:
  - GPU0 memory sampled at about `24253MiB / 24576MiB`。
  - GPU0 sampled temperature about `58C`; power about `170W`; sampled utilization about `12%` immediately near/after validation/checkpoint。
  - GPU1 remains idle except for minimal desktop/Codex allocation。

### 指标 / 当前解释

- Source validation loss trajectory through step 4000: `0.0514 -> 0.0487 -> 0.0513 -> 0.0481 -> 0.0533 -> 0.0710 -> 0.0882 -> 0.1067`。
- The source objective is clearly worsening after step 2000, likely overfitting/drift on this fixed 1k source split。
- This is not a runtime failure: validation completed, `step_4000.pt` saved, and training continued。
- Current best source checkpoint remains step 2000 (`val_loss=0.0481`)。
- This remains source-run evidence only; LP/classification metrics are still unavailable。

### 失败 / 停止条件

- No runtime failure through step 4000。
- No OOM, Traceback, nonzero exit, orphan process, or checkpoint write failure observed。
- Stop/diagnose if a later checkpoint boundary fails to save, if process exits without `EXITCODE 0`, or if log/GPU activity stalls for more than 10 minutes outside validation/checkpoint windows。

### 下一步

- Continue the active frozen-LM source run toward completion unless an explicit early-stop decision is recorded later。
- Next validation boundary: step 4500。
- Next periodic checkpoint boundary: step 5000。
- Do not launch LP/schema sweep while this source run is still active。

## 2026-06-18 Phase 1 / P1_DATA_SCALING_1K_FROZEN_LM_UMS_SOURCE_RUN step-4500 记录

### 监控计划

1. Wait for the step 4500 validation boundary。
2. Confirm validation completes and checkpoint behavior matches `save_interval=1000`。
3. Record source validation trajectory and runtime health。
4. Do not launch LP/schema sweep while source run remains active。

### 监控命令

```powershell
rg -n "Step [0-9]+: val_loss|Checkpoint saved|EXITCODE|Traceback|out of memory|Error" outputs\logs\data_scaling_frozen_lm_ums_1k_source.log
Get-Content -Tail 80 outputs\logs\data_scaling_frozen_lm_ums_1k_source.log
Get-ChildItem outputs\data_scaling\frozen_lm_ums_1k\checkpoints | Sort-Object LastWriteTime
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

### 输入

- Active log: `outputs/logs/data_scaling_frozen_lm_ums_1k_source.log`
- Active output dir: `outputs/data_scaling/frozen_lm_ums_1k`
- Current best source checkpoint: `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/best.pt`

### 输出 / 结果

- Validation boundary:
  - `Step 500: val_loss = 0.0514`
  - `Step 1000: val_loss = 0.0487`
  - `Step 1500: val_loss = 0.0513`
  - `Step 2000: val_loss = 0.0481`
  - `Step 2500: val_loss = 0.0533`
  - `Step 3000: val_loss = 0.0710`
  - `Step 3500: val_loss = 0.0882`
  - `Step 4000: val_loss = 0.1067`
  - `Step 4500: val_loss = 0.1131`
  - Step 4500 validation completed 250/250 batches; tqdm duration about `1:39`。
- Checkpoints:
  - No `step_4500.pt` is expected because `save_interval=1000`。
  - `best.pt` was not refreshed because `0.1131` is worse than the current best `0.0481` from step 2000。
  - Existing periodic checkpoints remain `step_1000.pt`, `step_2000.pt`, `step_3000.pt`, and `step_4000.pt`。
- Training continued after validation:
  - latest observed after this boundary: step `4502` at about `10:35:49` elapsed。
  - no `EXITCODE` yet because the source run remains active。
- GPU:
  - GPU0 memory sampled at about `24253MiB / 24576MiB`。
  - GPU0 sampled temperature about `60C`; power about `227W`; sampled utilization about `40%` near/after validation。
  - GPU1 remains idle except for minimal desktop/Codex allocation。

### 指标 / 当前解释

- Source validation loss trajectory through step 4500: `0.0514 -> 0.0487 -> 0.0513 -> 0.0481 -> 0.0533 -> 0.0710 -> 0.0882 -> 0.1067 -> 0.1131`。
- Worsening validation loss after step 2000 is now consistent; the best checkpoint remains step 2000。
- This remains a source-run behavior/overfitting signal, not a runtime failure。
- LP/classification metrics are still unavailable until the source run is complete or an explicit early-stop decision is documented and downstream evaluation is launched。

### 失败 / 停止条件

- No runtime failure through step 4500。
- No OOM, Traceback, nonzero exit, orphan process, or checkpoint write failure observed。
- Stop/diagnose if a later checkpoint boundary fails to save, if process exits without `EXITCODE 0`, or if log/GPU activity stalls for more than 10 minutes outside validation/checkpoint windows。

### 下一步

- Continue the active frozen-LM source run toward completion unless an explicit early-stop decision is recorded later。
- Next hard checkpoint boundary: step 5000 validation plus `step_5000.pt` periodic checkpoint。
- Do not launch LP/schema sweep while this source run is still active。

## 2026-06-18 Phase 1 / P1_DATA_SCALING_1K_FROZEN_LM_UMS_SOURCE_RUN step-5000 记录

### 监控计划

1. Wait for the step 5000 validation/checkpoint boundary。
2. Confirm validation finishes and `step_5000.pt` is saved。
3. Recheck checkpoint file size after a short delay to avoid recording a partially written file。
4. Confirm training continues after checkpoint。

### 监控命令

```powershell
rg -n "Step [0-9]+: val_loss|Checkpoint saved|EXITCODE|Traceback|out of memory|Error" outputs\logs\data_scaling_frozen_lm_ums_1k_source.log
Get-Content -Tail 80 outputs\logs\data_scaling_frozen_lm_ums_1k_source.log
Get-ChildItem outputs\data_scaling\frozen_lm_ums_1k\checkpoints | Sort-Object LastWriteTime
Start-Sleep -Seconds 60; Get-ChildItem outputs\data_scaling\frozen_lm_ums_1k\checkpoints | Sort-Object LastWriteTime
Get-Process -Id 17472,17936 -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime,Responding,WorkingSet64,PrivateMemorySize64
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

### 输入

- Active log: `outputs/logs/data_scaling_frozen_lm_ums_1k_source.log`
- Active output dir: `outputs/data_scaling/frozen_lm_ums_1k`
- Current best source checkpoint: `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/best.pt`

### 输出 / 结果

- Validation boundary:
  - `Step 500: val_loss = 0.0514`
  - `Step 1000: val_loss = 0.0487`
  - `Step 1500: val_loss = 0.0513`
  - `Step 2000: val_loss = 0.0481`
  - `Step 2500: val_loss = 0.0533`
  - `Step 3000: val_loss = 0.0710`
  - `Step 3500: val_loss = 0.0882`
  - `Step 4000: val_loss = 0.1067`
  - `Step 4500: val_loss = 0.1131`
  - `Step 5000: val_loss = 0.1198`
  - Step 5000 validation completed 250/250 batches; tqdm duration about `1:53`。
- Checkpoints:
  - `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/step_5000.pt`
    - timestamp after write settled: `2026-06-18 07:34:25`
    - final observed size after a 60s recheck: `1,072,395,066` bytes
  - `best.pt` was not refreshed because `0.1198` is worse than the current best `0.0481` from step 2000。
  - Existing periodic checkpoints now include `step_1000.pt`, `step_2000.pt`, `step_3000.pt`, `step_4000.pt`, and `step_5000.pt`。
  - Initial checkpoint listing caught `step_5000.pt` mid-write at `921,083,904` bytes; the 60s recheck confirmed the final size matched prior periodic checkpoints。
- Training continued after checkpoint:
  - latest observed after this boundary: step `5008` at about `11:44:47` elapsed。
  - no `EXITCODE` yet because the source run remains active。
- Process/GPU:
  - wrapper `cmd.exe` PID `17936` remains alive and responding。
  - Python PID `17472` remains alive and responding。
  - Python working set about `14.23GB`; private memory about `41.20GB`。
  - GPU0 memory sampled at about `24253MiB / 24576MiB`。
  - GPU0 sampled temperature about `55C`; power about `132W`; sampled utilization about `18%` near/after validation/checkpoint。
  - GPU1 remains idle except for minimal desktop/Codex allocation。

### 指标 / 当前解释

- Source validation loss trajectory through step 5000: `0.0514 -> 0.0487 -> 0.0513 -> 0.0481 -> 0.0533 -> 0.0710 -> 0.0882 -> 0.1067 -> 0.1131 -> 0.1198`。
- The formal source run is now past the halfway checkpoint and remains operational。
- Validation loss continues to worsen after the step 2000 best, making step 2000 the likely source checkpoint to use for any downstream LP evaluation unless later validation improves。
- This remains source-run evidence only; LP/classification metrics are still unavailable。

### 失败 / 停止条件

- No runtime failure through step 5000。
- No OOM, Traceback, nonzero exit, orphan process, or checkpoint write failure observed。
- The temporary small `step_5000.pt` size was a read-during-write artifact, not a persistent file corruption。
- Stop/diagnose if a later checkpoint boundary fails to save, if process exits without `EXITCODE 0`, or if log/GPU activity stalls for more than 10 minutes outside validation/checkpoint windows。

### 下一步

- Continue the active frozen-LM source run toward completion unless an explicit early-stop decision is recorded later。
- Next validation boundary: step 5500。
- Next periodic checkpoint boundary: step 6000。
- Do not launch LP/schema sweep while this source run is still active。

## 2026-06-18 Phase 1 / P1_DATA_SCALING_1K_FROZEN_LM_UMS_SOURCE_RUN step-5500 记录

### 监控计划

1. Wait for the step 5500 validation boundary。
2. Confirm validation completes and checkpoint behavior matches `save_interval=1000`。
3. Record source validation trajectory and runtime health。
4. Keep downstream LP/schema sweep blocked while source run remains active。

### 监控命令

```powershell
rg -n "Step [0-9]+: val_loss|Checkpoint saved|EXITCODE|Traceback|out of memory|Error" outputs\logs\data_scaling_frozen_lm_ums_1k_source.log
Get-Content -Tail 80 outputs\logs\data_scaling_frozen_lm_ums_1k_source.log
Get-ChildItem outputs\data_scaling\frozen_lm_ums_1k\checkpoints | Sort-Object LastWriteTime
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

### 输入

- Active log: `outputs/logs/data_scaling_frozen_lm_ums_1k_source.log`
- Active output dir: `outputs/data_scaling/frozen_lm_ums_1k`
- Current best source checkpoint: `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/best.pt`

### 输出 / 结果

- Validation boundary:
  - `Step 500: val_loss = 0.0514`
  - `Step 1000: val_loss = 0.0487`
  - `Step 1500: val_loss = 0.0513`
  - `Step 2000: val_loss = 0.0481`
  - `Step 2500: val_loss = 0.0533`
  - `Step 3000: val_loss = 0.0710`
  - `Step 3500: val_loss = 0.0882`
  - `Step 4000: val_loss = 0.1067`
  - `Step 4500: val_loss = 0.1131`
  - `Step 5000: val_loss = 0.1198`
  - `Step 5500: val_loss = 0.1265`
  - Step 5500 validation completed 250/250 batches; tqdm duration about `1:45`。
- Checkpoints:
  - No `step_5500.pt` is expected because `save_interval=1000`。
  - `best.pt` was not refreshed because `0.1265` is worse than the current best `0.0481` from step 2000。
  - Existing periodic checkpoints remain `step_1000.pt`, `step_2000.pt`, `step_3000.pt`, `step_4000.pt`, and `step_5000.pt`。
- Training continued after validation:
  - latest observed after this boundary: step `5502` at about `12:49:32` elapsed。
  - no `EXITCODE` yet because the source run remains active。
- GPU:
  - GPU0 memory sampled at about `24253MiB / 24576MiB`。
  - GPU0 sampled temperature about `62C`; power about `211W`; sampled utilization about `98%`。
  - GPU1 remains idle except for minimal desktop/Codex allocation。

### 指标 / 当前解释

- Source validation loss trajectory through step 5500: `0.0514 -> 0.0487 -> 0.0513 -> 0.0481 -> 0.0533 -> 0.0710 -> 0.0882 -> 0.1067 -> 0.1131 -> 0.1198 -> 0.1265`。
- The post-step-2000 worsening trend persists。
- The run remains operational and training resumes after validation。
- This remains source-run evidence only; downstream LP/classification metrics are still unavailable。

### 失败 / 停止条件

- No runtime failure through step 5500。
- No OOM, Traceback, nonzero exit, orphan process, or checkpoint write failure observed。
- Stop/diagnose if a later checkpoint boundary fails to save, if process exits without `EXITCODE 0`, or if log/GPU activity stalls for more than 10 minutes outside validation/checkpoint windows。

### 下一步

- Continue the active frozen-LM source run toward completion unless an explicit early-stop decision is recorded later。
- Next hard checkpoint boundary: step 6000 validation plus `step_6000.pt` periodic checkpoint。
- Do not launch LP/schema sweep while this source run is still active。

## 2026-06-18 Phase 1 / P1_DATA_SCALING_1K_FROZEN_LM_UMS_SOURCE_RUN step-6000 记录

### 监控计划

1. Wait for the step 6000 validation/checkpoint boundary。
2. Confirm validation finishes and `step_6000.pt` is saved。
3. Recheck checkpoint file size after a short delay if the first listing catches a partial write。
4. Confirm training continues after checkpoint。

### 监控命令

```powershell
rg -n "Step [0-9]+: val_loss|Checkpoint saved|EXITCODE|Traceback|out of memory|Error" outputs\logs\data_scaling_frozen_lm_ums_1k_source.log
Get-Content -Tail 80 outputs\logs\data_scaling_frozen_lm_ums_1k_source.log
Get-ChildItem outputs\data_scaling\frozen_lm_ums_1k\checkpoints | Sort-Object LastWriteTime
Start-Sleep -Seconds 90; Get-ChildItem outputs\data_scaling\frozen_lm_ums_1k\checkpoints | Sort-Object LastWriteTime
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

### 输入

- Active log: `outputs/logs/data_scaling_frozen_lm_ums_1k_source.log`
- Active output dir: `outputs/data_scaling/frozen_lm_ums_1k`
- Current best source checkpoint: `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/best.pt`

### 输出 / 结果

- Validation boundary:
  - `Step 500: val_loss = 0.0514`
  - `Step 1000: val_loss = 0.0487`
  - `Step 1500: val_loss = 0.0513`
  - `Step 2000: val_loss = 0.0481`
  - `Step 2500: val_loss = 0.0533`
  - `Step 3000: val_loss = 0.0710`
  - `Step 3500: val_loss = 0.0882`
  - `Step 4000: val_loss = 0.1067`
  - `Step 4500: val_loss = 0.1131`
  - `Step 5000: val_loss = 0.1198`
  - `Step 5500: val_loss = 0.1265`
  - `Step 6000: val_loss = 0.1355`
  - Step 6000 validation completed 250/250 batches; tqdm duration about `1:51`。
- Checkpoints:
  - `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/step_6000.pt`
    - timestamp after write settled: `2026-06-18 09:48:21`
    - final observed size after a 90s recheck: `1,072,395,066` bytes
  - `best.pt` was not refreshed because `0.1355` is worse than the current best `0.0481` from step 2000。
  - Existing periodic checkpoints now include `step_1000.pt`, `step_2000.pt`, `step_3000.pt`, `step_4000.pt`, `step_5000.pt`, and `step_6000.pt`。
  - Initial checkpoint listing caught `step_6000.pt` mid-write at `389,431,296` bytes; the 90s recheck confirmed the final size matched prior periodic checkpoints。
- Training continued after checkpoint:
  - latest observed after this boundary: step `6007` at about `13:58:50` elapsed。
  - no `EXITCODE` yet because the source run remains active。
- GPU:
  - GPU0 memory sampled at about `24253MiB / 24576MiB`。
  - GPU0 sampled temperature about `56C`; power about `142W`; sampled utilization about `22%` near/after validation/checkpoint。
  - GPU1 remains idle except for minimal desktop/Codex allocation。

### 指标 / 当前解释

- Source validation loss trajectory through step 6000: `0.0514 -> 0.0487 -> 0.0513 -> 0.0481 -> 0.0533 -> 0.0710 -> 0.0882 -> 0.1067 -> 0.1131 -> 0.1198 -> 0.1265 -> 0.1355`。
- The source run remains stable operationally and checkpointing works。
- Validation loss continues to worsen after the step 2000 best, strengthening the interpretation that the best frozen-LM source checkpoint is step 2000 for downstream LP evaluation unless later validation unexpectedly improves。
- This remains source-run evidence only; LP/classification metrics are still unavailable。

### 失败 / 停止条件

- No runtime failure through step 6000。
- No OOM, Traceback, nonzero exit, orphan process, or checkpoint write failure observed。
- The temporary small `step_6000.pt` size was a read-during-write artifact, not persistent file corruption。
- Stop/diagnose if a later checkpoint boundary fails to save, if process exits without `EXITCODE 0`, or if log/GPU activity stalls for more than 10 minutes outside validation/checkpoint windows。

### 下一步

- Continue the active frozen-LM source run toward completion unless an explicit early-stop decision is recorded later。
- Next validation boundary: step 6500。
- Next periodic checkpoint boundary: step 7000。
- Do not launch LP/schema sweep while this source run is still active。

## 2026-06-18 Phase 1 / P1_DATA_SCALING_1K_FROZEN_LM_UMS_SOURCE_RUN interrupted-after-step6329 记录

### 诊断计划

1. Check whether the formal frozen-LM source process is still alive。
2. Check GPU occupancy and whether training is still consuming GPU memory。
3. Inspect the active log for last progress, `EXITCODE`, Python/CUDA/OOM errors, and validation/checkpoint boundaries。
4. Inspect checkpoint artifacts to determine the last durable recovery point。
5. Inspect Windows system events around the log stop time to distinguish training-internal failure from external shutdown/reboot。
6. Stop all diagnosis once process/GPU/log/checkpoint/system-event evidence can explain the interruption without launching new training。

### 诊断命令

```powershell
Get-Process python,cmd,powershell -ErrorAction SilentlyContinue |
  Select-Object Id,ProcessName,CPU,StartTime,Path
nvidia-smi
Get-Item outputs\logs\data_scaling_frozen_lm_ums_1k_source.log |
  Select-Object FullName,Length,LastWriteTime
Get-Content outputs\logs\data_scaling_frozen_lm_ums_1k_source.log -Tail 80
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_1k_source.log `
  -Pattern 'Traceback|CUDA|RuntimeError|OutOfMemory|KeyboardInterrupt|EXITCODE|Validation|val_loss|Saved checkpoint|best' `
  -CaseSensitive:$false | Select-Object -Last 80
Get-ChildItem outputs\data_scaling\frozen_lm_ums_1k\checkpoints |
  Select-Object Name,Length,LastWriteTime | Sort-Object LastWriteTime
Get-ChildItem outputs\data_scaling\frozen_lm_ums_1k |
  Select-Object Name,Length,LastWriteTime
Get-CimInstance Win32_OperatingSystem |
  Select-Object LastBootUpTime,TotalVisibleMemorySize,FreePhysicalMemory,TotalVirtualMemorySize,FreeVirtualMemory
Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=(Get-Date).AddHours(-2)} |
  Where-Object { $_.Id -in 41,1074,109,6005,6006,6008,7036,7040,7045 } |
  Select-Object TimeCreated,ProviderName,Id,LevelDisplayName,Message
```

### 输入

- Active log: `outputs/logs/data_scaling_frozen_lm_ums_1k_source.log`
- Training output dir: `outputs/data_scaling/frozen_lm_ums_1k`
- Checkpoint dir: `outputs/data_scaling/frozen_lm_ums_1k/checkpoints`
- Wrapper script: `scripts/run_data_scaling_frozen_lm_ums_1k_source.cmd`
- Relevant process ids from the running phase: wrapper `cmd.exe` PID `17936`, Python PID `17472`

### 预期输出 / 停止条件

- If Python and wrapper remain alive and GPU0 remains occupied, classify as slow-but-running and continue monitoring。
- If process/GPU are gone but the log contains `EXITCODE 0`, classify as completed and wait for expected artifacts。
- If process/GPU are gone with Python/CUDA/OOM traceback, classify as training/runtime failure。
- If process/GPU are gone with no `EXITCODE` and system events show shutdown/reboot near the log stop time, classify as external interruption。
- Stop without resuming or launching LP/schema sweep once the interruption class is established。

### 输出 / 结果

- Current process/GPU state:
  - No active `python.exe` training process was found。
  - GPU0 and GPU1 both reported `0MiB / 24576MiB` and no running GPU processes。
  - A stray `cmd.exe` PID `15192` existed after reboot, but not the original training wrapper PID `17936`。
- Log state:
  - `outputs/logs/data_scaling_frozen_lm_ums_1k_source.log`
    - size: `939,965` bytes
    - last write time: `2026-06-18 10:42:19`
  - Last training progress reached step `6329 / 10000`。
  - No `EXITCODE` line is present, even though the wrapper script should append one after Python returns。
  - No `Traceback`, `RuntimeError`, `CUDA`, `OutOfMemory`, or `KeyboardInterrupt` line was found in the log。
- Speed behavior immediately before interruption:
  - Before step `6315`, the local cadence was mostly around `6-12s/it`。
  - From step `6315` onward, cadence became abnormal and sustained:
    - `6315`: about `20.86s/it`
    - `6316`: about `29.30s/it`
    - `6317`: about `32.93s/it`
    - `6318`: about `34.19s/it`
    - `6319-6329`: roughly `26-36s/it`
  - This is a real local slowdown relative to the preceding segment, but it occurred immediately before the system reboot window。
- System event evidence:
  - `User32` event ID `1074` at `2026-06-18 10:42:25`: `RuntimeBroker.exe` initiated a computer restart on behalf of user `DESKTOP-FMKOM61\Admin`; shutdown type `重启`; reason `其他(计划外)`。
  - `EventLog` ID `6006` at `2026-06-18 10:43:40`: event log service stopped。
  - `Microsoft-Windows-Kernel-Power` ID `109` at `2026-06-18 10:43:45`: kernel power manager started shutdown transition; shutdown reason `Kernel API`。
  - `Win32_OperatingSystem.LastBootUpTime`: `2026-06-18 10:45:03`。
  - `EventLog` ID `6005` at `2026-06-18 10:46:17`: event log service started。
  - `Service Control Manager` ID `7045` at `2026-06-18 10:47:41`: `aTrustNetflt_1` kernel-mode driver installed from `C:\Program Files (x86)\Sangfor\aTrust\NetfltDriver\SdpNetFilter1.sys` after reboot。
- Checkpoint/artifact state:
  - Durable checkpoints:
    - `step_1000.pt`: `1,072,395,066` bytes, `2026-06-17 22:13:00`
    - `best.pt`: `1,072,361,762` bytes, `2026-06-18 00:42:35`
    - `step_2000.pt`: `1,072,395,066` bytes, `2026-06-18 00:42:41`
    - `step_3000.pt`: `1,072,395,066` bytes, `2026-06-18 02:54:17`
    - `step_4000.pt`: `1,072,395,066` bytes, `2026-06-18 05:17:37`
    - `step_5000.pt`: `1,072,395,066` bytes, `2026-06-18 07:34:25`
    - `step_6000.pt`: `1,072,395,066` bytes, `2026-06-18 09:48:21`
  - `final.pt` and `metrics_final.json` do not exist for `outputs/data_scaling/frozen_lm_ums_1k`。

### 指标 / 当前解释

- Source validation trajectory through the last completed validation boundary:
  `0.0514 -> 0.0487 -> 0.0513 -> 0.0481 -> 0.0533 -> 0.0710 -> 0.0882 -> 0.1067 -> 0.1131 -> 0.1198 -> 0.1265 -> 0.1355`。
- Best source validation loss remains step `2000` at `0.0481`。
- The post-step-2000 source validation loss was monotonically worse at observed validation boundaries, so the interruption does not hide a late improvement up to step 6000。
- The step `6315-6329` slowdown was abnormal relative to immediately preceding steps, but the strongest root-cause evidence for run termination is an external Windows restart/shutdown sequence initiated by `RuntimeBroker.exe` / Kernel API。
- This run is not a completed source run: no `EXITCODE 0`, no `final.pt`, and no final source-run completion marker。

### 失败原因 / 停止条件结果

- Primary failure class: external system restart killed the training wrapper and Python process。
- Not supported by evidence:
  - Python exception as root cause。
  - CUDA OOM as root cause。
  - checkpoint corruption as root cause。
  - hidden GPU miner or another GPU process at diagnosis time。
- The stop condition for external interruption is met: process/GPU gone, no wrapper `EXITCODE`, no Python/CUDA traceback, and system reboot events overlap the log stop time。

### 下一步

- Do not silently mark this source run as complete。
- Do not launch downstream LP/schema sweep before recording a source-run decision。
- Two valid next branches:
  1. **Strict completion branch**: resume from `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/step_6000.pt` with a new log path and a new execution-before record, then require `EXITCODE 0` / completion artifact evidence。
  2. **Documented early-stop branch**: record an explicit early-stop rationale because validation worsened after step 2000, then use `best.pt` only as a source checkpoint for LP after a separate execution-before record。
- Current recommendation: pause the run queue at `interrupted_after_step6329_resume_decision_needed` until the next branch is chosen; this preserves the failure narrative and avoids mixing incomplete source evidence into LP/classification claims。

## 2026-06-18 Phase 1 / P1_DATA_SCALING_1K_FROZEN_LM_SOURCE_DECISION documented-early-stop 记录

### 执行前计划

1. Resolve the branch created by the interrupted frozen-LM 1k source run。
2. Inspect whether downstream LP depends on `best.pt` or `final.pt`。
3. If LP depends on `best.pt` and the validation trajectory supports early stop, explicitly mark the source checkpoint policy as documented early-stop。
4. Keep this decision scoped to the 1k frozen-LM data-scaling source run; do not generalize it to 3k/10k/30k or schema sweep。
5. Do not launch LP in this decision step; LP requires its own execution-before record。

### 命令

```powershell
rg -n "lp_frozen|source_checkpoint|checkpoint|frozen_lm_ums_1k|best.pt|final.pt|resume" configs scripts training models -g "*.py" -g "*.yaml"
Get-Content configs\data_scaling\lp_frozen_lm_ums_1k.yaml
Get-ChildItem outputs\data_scaling\lp_frozen_lm_ums_1k -ErrorAction SilentlyContinue -Force
Get-ChildItem outputs\logs -Filter '*lp*frozen*1k*' -ErrorAction SilentlyContinue
```

### 输入

- Source checkpoint candidates:
  - `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/best.pt`
  - `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/step_6000.pt`
- LP config: `configs/data_scaling/lp_frozen_lm_ums_1k.yaml`
- Source validation trajectory from the interrupted source run。
- Runtime interruption diagnosis above。

### 预期输出

- A source checkpoint policy for the interrupted frozen-LM 1k source run。
- A clear decision on whether downstream LP may use `best.pt`。
- No new training outputs in this decision step。

### 停止条件

- Stop if LP depends on `final.pt` or an absent source artifact。
- Stop if the source validation trajectory does not support early stop。
- Stop if `outputs/data_scaling/lp_frozen_lm_ums_1k/metrics_final.json` already exists。
- Stop if any command would start training during this decision step。

### 执行后结果

- `configs/data_scaling/lp_frozen_lm_ums_1k.yaml` uses:
  - `transfer.init_vit_checkpoint: ./outputs/data_scaling/frozen_lm_ums_1k/checkpoints/best.pt`
  - `transfer.freeze_backbone: true`
  - `training.output_dir: ./outputs/data_scaling/lp_frozen_lm_ums_1k`
- No existing `outputs/data_scaling/lp_frozen_lm_ums_1k` output was found。
- No existing `outputs/logs/*lp*frozen*1k*` log was found。
- The interrupted source run has a durable `best.pt`, and best source validation occurred at step `2000` with `val_loss=0.0481`。
- Subsequent observed source validation losses were worse through step `6000`: `0.0533`, `0.0710`, `0.0882`, `0.1067`, `0.1131`, `0.1198`, `0.1265`, `0.1355`。

### 决策 / 指标解释

- Decision: use the **documented early-stop branch** for the frozen-LM 1k source run。
- Rationale:
  - The downstream LP protocol depends on `checkpoints/best.pt`, not on `final.pt`。
  - The best source validation point is already saved durably。
  - The external system restart prevented a clean `EXITCODE 0`, but it did not remove the best checkpoint or create evidence of late source validation improvement。
  - Resuming from `step_6000.pt` would mainly spend additional compute after a sustained worsening trend and is not required to produce the configured LP input。
- Boundary:
  - This decision does **not** claim the source run completed normally。
  - This decision does **not** create final source-run metrics。
  - Any LP result must cite `best.pt` plus this early-stop decision as provenance。

### 失败原因

- The source run remains externally interrupted, not completed。
- The accepted recovery is methodological early-stop from `best.pt`, not silent completion。

### 下一步

- Proceed to `P1_DATA_SCALING_1K_FROZEN_LM_LP_DEBUG_ENTRY`。
- The next task must first record plan/command/input/output/stop conditions, then run only the LP debug entry。
- Do not start formal LP until the debug entry confirms checkpoint loading and training entry behavior。

## 2026-06-18 Phase 1 / P1_DATA_SCALING_1K_FROZEN_LM_LP_DEBUG_ENTRY 执行前记录

### 计划

1. Run a tiny/debug LP entry for the documented early-stop frozen-LM 1k checkpoint。
2. Verify `scripts/train_vit_baseline.py` can:
   - parse `configs/data_scaling/lp_frozen_lm_ums_1k.yaml`;
   - load `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/best.pt`;
   - freeze the ViT backbone;
   - train/evaluate a small linear head on CUDA;
   - write debug metrics/checkpoints under a seed-suffixed output dir。
3. Keep this as entry validation only; do not treat debug metrics as paper evidence。
4. Do not start formal LP or any schema/LP matrix job in this step。

### 命令

```powershell
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime,Path
nvidia-smi
Test-Path outputs\data_scaling\frozen_lm_ums_1k\checkpoints\best.pt
Test-Path outputs\data_scaling\lp_frozen_lm_ums_1k_seed900120\metrics_final.json
python scripts\train_vit_baseline.py --config configs\data_scaling\lp_frozen_lm_ums_1k.yaml --debug --seed 900120
```

### 输入

- Config: `configs/data_scaling/lp_frozen_lm_ums_1k.yaml`
- Source checkpoint: `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/best.pt`
- Fixed split:
  - `data/splits/chexpert_train_1k.jsonl`
  - `data/splits/chexpert_val_fixed.jsonl`
- Seed override: `900120`

### 预期输出

- Debug output dir: `outputs/data_scaling/lp_frozen_lm_ums_1k_seed900120`
- Expected artifacts:
  - `outputs/data_scaling/lp_frozen_lm_ums_1k_seed900120/best.pt`
  - `outputs/data_scaling/lp_frozen_lm_ums_1k_seed900120/step_20.pt`
  - `outputs/data_scaling/lp_frozen_lm_ums_1k_seed900120/final.pt`
  - `outputs/data_scaling/lp_frozen_lm_ums_1k_seed900120/metrics_final.json`

### 停止条件

- Stop if `metrics_final.json` already exists before launch。
- Stop if another training process is active。
- Stop if GPU is occupied by an unrelated process。
- Stop if checkpoint loading fails or loads zero useful ViT parameters。
- Stop if debug run raises an exception, exits nonzero, or writes no final metrics。

## 2026-06-18 Phase 1 / P1_RUNTIME_SPEED_STATUS_RECHECK_AFTER_LP 执行记录

### 计划

1. 回答用户关于“目前训练速度是否变慢”的即时问题。
2. 只做只读诊断：检查 Python 训练进程、GPU 占用、最近日志和系统资源。
3. 将当前机器状态与最近完成的 LP 训练、此前中断的 frozen-LM source 尾段分开解释。
4. 不启动任何新训练、不修改实验指标、不扩展 SPD 新变体。

### 命令

```powershell
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime,Path
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits
Get-ChildItem outputs\logs -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 12 Name,Length,LastWriteTime
Get-Content outputs\logs\data_scaling_lp_no_lm_ums_1k.log -Tail 80
Get-Content outputs\logs\data_scaling_lp_frozen_lm_ums_1k.log -Tail 80
Get-Content outputs\logs\data_scaling_frozen_lm_ums_1k_source.log -Tail 80
Get-Counter '\Processor(_Total)\% Processor Time','\Memory\Available MBytes','\PhysicalDisk(_Total)\% Disk Time' -SampleInterval 1 -MaxSamples 3
Get-Process | Sort-Object CPU -Descending | Select-Object -First 20 Id,ProcessName,CPU,WorkingSet64,Path
nvidia-smi pmon -c 1
Get-Process -Name MsMpEng,SearchIndexer,OneDrive,python,python3 -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,WorkingSet64,StartTime,Path
Get-CimInstance Win32_StartupCommand | Select-Object Name,Command,Location,User | Sort-Object Name | Select-Object -First 30
Get-ScheduledTask | Where-Object {$_.State -eq 'Running'} | Select-Object TaskName,TaskPath,State | Sort-Object TaskPath,TaskName | Select-Object -First 50
```

### 输入

- Live GPU/process/resource state。
- Recent LP logs:
  - `outputs/logs/data_scaling_lp_no_lm_ums_1k.log`
  - `outputs/logs/data_scaling_lp_frozen_lm_ums_1k.log`
- Interrupted source log:
  - `outputs/logs/data_scaling_frozen_lm_ums_1k_source.log`
- Windows startup/scheduled-task snapshots for obvious automation/background-load signals。

### 预期输出

- 当前是否有训练进程。
- 当前 GPU/CPU/内存/磁盘是否饱和。
- 最近完成训练的速度是否正常。
- 若存在失败命令，记录失败层级和解释边界。

### 停止条件

- 若发现正在运行的训练进程或 GPU 被未知进程占用，停止后续实验启动并进入占用诊断。
- 若发现 CPU/磁盘长时间饱和，停止后续训练启动并先定位系统层占用。
- 若只读诊断命令超时但已有足够证据回答当前问题，记录超时命令，不做破坏性修复。

### 执行后结果

- Current training state:
  - `Get-Process python` returned no active Python training process。
  - GPU0 RTX 3090: `0%` utilization, `0 / 24576 MB`。
  - GPU1 RTX 3090: `0%` utilization, `0 / 24576 MB`。
- System state:
  - CPU total sampled around `20-27%`。
  - Available memory around `55 GB`。
  - Disk total busy time near idle (`0-1%` in sampled windows)。
- Recent completed LP runs:
  - `data_scaling_lp_no_lm_ums_1k.log` ended with `Training completed!` and `EXITCODE 0`。
  - `data_scaling_lp_frozen_lm_ums_1k.log` ended with `Training completed!` and `EXITCODE 0`。
- Background/process observations:
  - Visible high-cumulative CPU processes include Codex, ToDesk, Edge WebView, SunloginClient, proxy/security/vendor utilities。
  - Startup entries include BaiduYunDetect, Docker Desktop, GameViewer, MotionPro/WebVPN, Edge autolaunch, Ollama, OneDrive, QuarkUpdater, SunloginClient, Huorong Sysdiag, Tailscale, Warp。
  - Running scheduled tasks include Clash Verge, Windows multimedia/system/network/cache tasks。
  - None of the visible checks showed a current GPU miner, second Python training job, or disk-saturating automation during this snapshot。

### 指标

- Current training speed:
  - Not applicable: no training is currently running。
- Most recent LP speed:
  - no-LM LP 1k: `3000/3000` in `16:23`, reported `3.05 it/s`。
  - frozen-LM LP 1k: `3000/3000` in `22:02`, reported `2.27 it/s`。
  - Both are within expected range for these LP jobs on this Windows + RTX 3090 setup。
- Earlier interrupted frozen-LM source tail:
  - Tail around steps `6315-6329` slowed to roughly `20-35 s/it`。
  - This was abnormal relative to its preceding segment, but it occurred before the documented reboot/interruption and is not evidence of persistent current slowdown。

### 失败 / 限制

- `nvidia-smi pmon -c 1` failed with “feature is not supported in this configuration”; fallback `nvidia-smi --query-gpu` was sufficient for GPU memory/utilization。
- One live per-process CPU delta sampling command timed out at 10 seconds; fallback cumulative process list, system counters, startup entries, scheduled tasks, and GPU/process checks were sufficient for the current question。
- This is not a full antivirus or forensic scan. It only shows no current GPU/training-like load and no obvious CPU/disk saturation at diagnosis time。

### 当前判断

- 目前没有正在跑的训练，因此当前训练速度不存在“正在变慢”的问题。
- 机器当前资源状态正常：GPU 空闲、CPU/内存/磁盘没有明显瓶颈。
- 最近两个正式 LP 训练速度正常并成功结束。
- 之前 frozen-LM source 尾段确实异常变慢，但更像外部 runtime/reboot 前的局部事件，不应扩大解释为现在电脑持续变慢或训练系统整体异常。

### 下一步

- 不因为本次速度复查重跑 frozen-LM source。
- 若继续启动下一项长训练，启动前再次做 GPU/Python/process preflight，并保留 wrapper `EXITCODE` 日志。
- 若用户仍怀疑病毒或远控占用，应另开一个系统安全检查任务：Windows Defender/Huorong 扫描状态、启动项白名单、网络连接和未知签名进程；该任务与实验训练指标分开记录。

## 2026-06-18 Phase 1 / P1_RUNTIME_SPEED_STATUS_CHECK 执行记录

### 计划

1. Respond to the user interruption asking whether training speed is currently slower or normal。
2. Check live GPU/CPU/process state without launching any training。
3. Compare the most recent completed LP logs with the interrupted frozen-LM source log tail。
4. Keep the conclusion scoped to runtime/speed only; do not infer new scientific claims。

### 命令

```powershell
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits
Get-Process python,python3,cmd,powershell,pwsh -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,WorkingSet64,StartTime
Get-CimInstance Win32_OperatingSystem | Select-Object LastBootUpTime,LocalDateTime
Get-Counter '\Processor(_Total)\% Processor Time','\Memory\Available MBytes','\PhysicalDisk(_Total)\% Disk Time' -SampleInterval 1 -MaxSamples 3
python - <<'PY'
# parse tqdm speed tokens from:
# outputs/logs/data_scaling_frozen_lm_ums_1k_source.log
# outputs/logs/data_scaling_lp_frozen_lm_ums_1k.log
# outputs/logs/data_scaling_lp_no_lm_ums_1k.log
PY
```

### 输入

- Live process/GPU state。
- Logs:
  - `outputs/logs/data_scaling_frozen_lm_ums_1k_source.log`
  - `outputs/logs/data_scaling_lp_frozen_lm_ums_1k.log`
  - `outputs/logs/data_scaling_lp_no_lm_ums_1k.log`

### 预期输出

- A user-facing answer on whether current training speed is abnormal。
- A document record separating live machine state from historical interrupted-run speed。

### 停止条件

- Stop if a live training process is found; do not launch any new experiment during diagnosis。
- Stop if GPU is occupied by an unknown process。
- Stop if logs are missing; report the boundary instead of guessing。

### 执行后结果

- Live GPU state at diagnosis time:
  - GPU0 RTX 3090: `0 / 24576 MB`, `0%` utilization。
  - GPU1 RTX 3090: `0 / 24576 MB`, `0%` utilization。
- No active Python training process was present。
- System load snapshot:
  - CPU total was roughly `20-26%` during the sample window。
  - Available memory was roughly `56 GB`。
  - Disk total busy time was near `0-1%`。
- System boot time:
  - `2026-06-18 10:45:03` local time, consistent with the earlier documented reboot boundary。

### 指标

- Recent completed LP runs:
  - `data_scaling_lp_frozen_lm_ums_1k.log`: `3000/3000` in `22:02`, reported `2.27 it/s`, `EXITCODE 0`。
  - `data_scaling_lp_no_lm_ums_1k.log`: `3000/3000` in `16:23`, reported `3.05 it/s`, `EXITCODE 0`。
- Interrupted frozen-LM source log:
  - Overall parsed median training speed: about `7.805 s/it`。
  - Last 20 parsed `s/it` median: about `29.245 s/it`。
  - Last 10 parsed `s/it`: `35.63, 33.83, 29.79, 27.45, 27.76, 29.19, 28.66, 29.78, 26.70, 31.03`。

### 解释

- Current state is normal in the sense that no training is running and GPU/CPU/disk are not saturated。
- The most recent completed LP jobs ran at plausible speed for this Windows + RTX 3090 setup。
- The frozen-LM source run tail was abnormally slow, but that slowdown occurred immediately before the documented system reboot/interruption and should be treated as an external runtime-event tail, not as evidence that all training is now persistently slower。

### 失败 / 限制

- This was a runtime snapshot, not a full antivirus scan。
- Visible process snapshots do not prove absence of malware; they only show no current GPU miner/training-like Python load。
- Remote-control / vendor background apps may exist on this machine, but the diagnosis did not show them consuming GPU or blocking the current VIVID run path。

### 下一步

- Do not rerun the interrupted frozen-LM source solely for speed unless its source-final checkpoint becomes necessary; current LP evidence already used the documented `best.pt` boundary。
- Continue with the planned no-LM S2/S3 comparator design audit before launching any formal schema-complexity runs。

## 2026-06-18 Phase 1 / P1_SCHEMA_COMPLEXITY_NO_LM_COMPARATOR_DESIGN_AUDIT 执行前记录

### 计划

1. Inspect the no-LM UMS classifier/evaluation path to decide whether S2 answerability and S3 uncertainty comparators can be derived from the existing 4-state logits。
2. Produce a compact design table that separates:
   - current support;
   - derivable diagnostic metrics;
   - missing export/head/loss requirements;
   - what can be compared against frozen-LM S2/S3 without overclaiming。
3. Do not launch training and do not create no-LM S2/S3 configs unless the target/head design is explicit。
4. Write the result back into this plan and refresh `revision_execution_status`。

### 命令

```powershell
rg -n "STATE_TO_INDEX|labels_to_state_targets|evaluate|state_accuracy|macro_auc|answerable|uncertain" scripts\train_ums_classifier.py evaluation\metrics.py
Get-Content scripts\train_ums_classifier.py
Get-Content evaluation\metrics.py
python scripts\design_no_lm_schema_comparator.py
python -m py_compile scripts\design_no_lm_schema_comparator.py scripts\summarize_revision_execution_status.py
python scripts\summarize_revision_execution_status.py
```

### 输入

- `scripts/train_ums_classifier.py`
- `evaluation/metrics.py`
- Existing task records for frozen-LM S2/S3 serializer debug。
- Existing no-LM UMS 4-state source/LP evidence。

### 预期输出

- `outputs/final_tables/no_lm_schema_comparator_design.csv`
- `outputs/final_tables/no_lm_schema_comparator_design.md`
- Updated `outputs/final_tables/revision_execution_status.csv`
- Updated `outputs/final_tables/revision_execution_status.md`
- Updated execution-plan after-record。

### 停止条件

- Stop before implementation if the comparator would require retraining or a new classifier head/loss。
- Stop if no-LM S2/S3 would be compared as equal supervision to frozen-LM S2/S3 without explicit target semantics。
- Stop if the task drifts into full schema sweep, SPD variants, or broader data-scaling runs。

### 执行后结果

- Static audit confirmed the no-LM classifier trains a 4-state target:
  - `null`
  - `absent`
  - `uncertain`
  - `present`
- Code evidence:
  - `scripts/train_ums_classifier.py` defines `STATE_TO_INDEX` with the four states。
  - `labels_to_state_targets()` maps NaN labels to `null`, `0` to `absent`, `-1` to `uncertain`, and `1` to `present`。
  - `evaluate()` computes softmax probabilities internally but exports only present-vs-nonpresent classification metrics plus state accuracy。
- Generated design artifacts:
  - `outputs/final_tables/no_lm_schema_comparator_design.csv`
  - `outputs/final_tables/no_lm_schema_comparator_design.md`
- Refreshed status artifacts:
  - `outputs/final_tables/revision_execution_status.csv`
  - `outputs/final_tables/revision_execution_status.md`
- Status table now has `25` rows and includes `P1_SCHEMA_COMPLEXITY_NO_LM_COMPARATOR_DESIGN_AUDIT`。

### 指标 / 设计结论

| Schema level | no-LM support | Derived signal | Fair comparison boundary |
| --- | --- | --- | --- |
| S1 `state_only` | already supported | 4-state prediction / `p_present` | comparable after matched frozen-LM S1 run |
| S2 `state_answerability` | diagnostic only | `answerable = state != null`, `p_answerable = 1 - p_null` | not equal to explicit S2 supervision |
| S3 `state_uncertainty` | diagnostic only | `uncertain = state == uncertain`, `p_uncertain` | not equal to explicit S3 supervision |
| S2/S3 multi-head no-LM | not implemented | would need new heads/losses | defer as new model variant |

### 失败 / 限制

- No runtime failure。
- No training was launched。
- Current no-LM metrics do not export answerability AUROC/F1/ECE or uncertainty AUROC/F1/ECE。
- Derived S2/S3 diagnostics are useful for reviewer response, but they are not equivalent to frozen-LM explicit JSON serialization targets。

### 下一步

- Implement a no-training eval/export path that loads an existing no-LM checkpoint and computes:
  - answerability metrics from `p_answerable = 1 - p_null`;
  - uncertainty metrics from `p_uncertain`;
  - prevalence and per-label support。
- Do not run formal schema sweep until this derived no-LM diagnostic exists, or explicitly label any frozen-LM-only work as serialization-only diagnostics。

## 2026-06-18 Phase 1 / P1_SCHEMA_COMPLEXITY_NO_LM_DERIVED_METRIC_EXPORT 执行前记录

### 计划

1. Implement a validation-only export script for existing no-LM 4-state checkpoints。
2. Load the fixed-split no-LM UMS 1k source checkpoint, run validation forward passes, and derive:
   - answerability target/probability from `state != null` and `1 - p_null`;
   - uncertainty target/probability from `state == uncertain` and `p_uncertain`。
3. Export summary/per-label metrics for answerability and uncertainty。
4. Do not train, resume, fine-tune, or launch schema sweep jobs。

### 命令

```powershell
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime,Path
nvidia-smi
Test-Path outputs\data_scaling\no_lm_ums_1k\best.pt
Get-Content configs\data_scaling\no_lm_ums_1k.yaml
python -m py_compile scripts\evaluate_no_lm_schema_derivatives.py
python scripts\evaluate_no_lm_schema_derivatives.py --config configs\data_scaling\no_lm_ums_1k.yaml --checkpoint outputs\data_scaling\no_lm_ums_1k\best.pt --split val
python scripts\summarize_revision_execution_status.py
```

### 输入

- Config: `configs/data_scaling/no_lm_ums_1k.yaml`
- Checkpoint: `outputs/data_scaling/no_lm_ums_1k/best.pt`
- Validation split from config:
  - `data/splits/chexpert_val_fixed.jsonl`
- Code:
  - `scripts/train_ums_classifier.py`
  - `data/chexpert_dataset.py`
  - `evaluation/metrics.py`

### 预期输出

- `outputs/final_tables/no_lm_schema_derivative_metrics.csv`
- `outputs/final_tables/no_lm_schema_derivative_metrics.md`
- `outputs/final_tables/no_lm_schema_derivative_per_label.csv`
- Updated `outputs/final_tables/revision_execution_status.csv`
- Updated `outputs/final_tables/revision_execution_status.md`

### 停止条件

- Stop before running if another training process is active。
- Stop before running if GPU is occupied by an unknown process。
- Stop before running if checkpoint or validation split is missing。
- Stop if the script attempts optimizer/training/resume behavior。
- Stop if checkpoint state cannot be loaded into the no-LM classifier。
- Stop if metric export has no positive/negative support for the target; report missing metrics rather than fabricating AUROC。

### 执行后结果

- Added validation-only export script:
  - `scripts/evaluate_no_lm_schema_derivatives.py`
- Syntax checks passed:
  - `python -m py_compile scripts\evaluate_no_lm_schema_derivatives.py scripts\summarize_revision_execution_status.py`
- Ran:
  - `python scripts\evaluate_no_lm_schema_derivatives.py --config configs\data_scaling\no_lm_ums_1k.yaml --checkpoint outputs\data_scaling\no_lm_ums_1k\best.pt --split val`
- Loaded:
  - `1000` validation samples from `data/splits/chexpert_val_fixed.jsonl`
  - `12` labels from `configs/data_scaling/no_lm_ums_1k.yaml`
  - checkpoint `outputs/data_scaling/no_lm_ums_1k/best.pt`
- Runtime:
  - validation forward loop: `16` batches, roughly `13` seconds from tqdm；
  - total command wall time: about `39` seconds including imports and checkpoint/model setup。
- Post-run status:
  - no residual Python training process；
  - GPU0/GPU1 returned to `0 / 24576 MB`, `0%` utilization。
- Generated artifacts:
  - `outputs/final_tables/no_lm_schema_derivative_metrics.csv`
  - `outputs/final_tables/no_lm_schema_derivative_metrics.md`
  - `outputs/final_tables/no_lm_schema_derivative_per_label.csv`
  - `outputs/final_tables/no_lm_schema_derivative_manifest.json`
- Refreshed status artifacts:
  - `outputs/final_tables/revision_execution_status.csv`
  - `outputs/final_tables/revision_execution_status.md`
  - status table now has `26` rows。

### 指标

| Target | support_positive / support_total | prevalence | prob_mean | pred_rate | accuracy | macro_f1 | micro_f1 | macro_auc | brier | ECE | MCE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| answerability | `3913 / 12000` | `0.326083` | `0.300014` | `0.270250` | `0.690167` | `0.351833` | `0.690167` | `0.607103` | `0.233198` | `0.173478` | `0.325286` |
| uncertainty | `504 / 12000` | `0.042000` | `0.026857` | `0.006417` | `0.952917` | `0.014694` | `0.952917` | `0.608101` | `0.043748` | `0.035577` | `0.949211` |

### 解释

- no-LM 4-state classifier can produce derived answerability and uncertainty diagnostics without retraining。
- Answerability signal is weak-to-moderate (`macro_auc=0.6071`) and thresholded F1 is low, so this does not show strong no-LM answerability modeling。
- Uncertainty has similar AUC (`0.6081`) but is rare (`4.2%` positive) and the default `0.5` threshold predicts uncertain only `0.64%` of field instances; F1 is therefore very low。
- These diagnostics are useful for reviewer-facing semantics/schema analysis, but they must be described as derived from the existing 4-state no-LM head, not as explicit S2/S3 serialization training。

### 失败 / 限制

- No runtime failure and no nonzero exit。
- PyTorch emitted the known `torch.load(..., weights_only=False)` FutureWarning for a local trusted checkpoint。
- This export used the no-LM `best.pt` checkpoint only; it does not compare best-vs-final no-LM checkpoints。
- Metrics use the default threshold `0.5`; uncertainty F1 is threshold-sensitive because positives are rare。
- This is not a formal frozen-LM-vs-no-LM S2/S3 training comparison。

### 下一步

- Use these exported diagnostics as the no-LM S2/S3 comparator boundary:
  - derived diagnostic evidence is available；
  - explicit no-LM S2/S3 head/loss remains a deferred new variant, not part of this priority pass。
- Next scientifically useful step is to consolidate schema evidence:
  - frozen-LM S2/S3 serializer debug passed；
  - no-LM derived answerability/uncertainty diagnostics are now exported；
  - schema-complexity claims should stay diagnostic unless a matched explicit no-LM S2/S3 training objective is implemented later。
- Do not launch SPD variants or full schema/data-scaling matrices in the same step。

## 2026-06-18 Phase 1 / P1_SCHEMA_COMPLEXITY_DIAGNOSTIC_CONSOLIDATION 执行前记录

### 计划

1. Consolidate schema-complexity diagnostics into one reader-facing table。
2. Combine:
   - frozen-LM S2/S3 serializer debug evidence；
   - no-LM answerability/uncertainty derived metric export；
   - explicit claim boundaries for diagnostic vs formal comparative evidence。
3. Produce final-table artifacts and refresh global execution status。
4. Do not run training, schema sweep, data-scaling matrix, or SPD variants。

### 命令

```powershell
Get-Content outputs\final_tables\no_lm_schema_derivative_metrics.csv
Get-ChildItem outputs\schema_sweep\frozen_lm_s2_state_answerability_seed900122\checkpoints
Get-ChildItem outputs\schema_sweep\frozen_lm_s3_state_uncertainty_seed900123\checkpoints
python scripts\summarize_schema_complexity_diagnostics.py
python -m py_compile scripts\summarize_schema_complexity_diagnostics.py scripts\summarize_revision_execution_status.py
python scripts\summarize_revision_execution_status.py
```

### 输入

- `outputs/final_tables/no_lm_schema_derivative_metrics.csv`
- Frozen-LM S2 debug artifacts:
  - `outputs/schema_sweep/frozen_lm_s2_state_answerability_seed900122/checkpoints/best.pt`
  - `outputs/schema_sweep/frozen_lm_s2_state_answerability_seed900122/checkpoints/final.pt`
- Frozen-LM S3 debug artifacts:
  - `outputs/schema_sweep/frozen_lm_s3_state_uncertainty_seed900123/checkpoints/best.pt`
  - `outputs/schema_sweep/frozen_lm_s3_state_uncertainty_seed900123/checkpoints/final.pt`
- Prior execution records in this plan。

### 预期输出

- `outputs/final_tables/schema_complexity_diagnostic_summary.csv`
- `outputs/final_tables/schema_complexity_diagnostic_summary.md`
- Updated `outputs/final_tables/revision_execution_status.csv`
- Updated `outputs/final_tables/revision_execution_status.md`

### 停止条件

- Stop if no-LM derived metrics are missing。
- Stop if frozen-LM S2/S3 debug artifacts are missing。
- Stop if the table would imply formal no-LM-vs-frozen-LM S2/S3 training comparison。
- Stop if the task requires launching formal schema sweep or new no-LM heads。

### 执行后结果

- Added consolidation script:
  - `scripts/summarize_schema_complexity_diagnostics.py`
- Generated:
  - `outputs/final_tables/schema_complexity_diagnostic_summary.csv`
  - `outputs/final_tables/schema_complexity_diagnostic_summary.md`
- Refreshed:
  - `outputs/final_tables/revision_execution_status.csv`
  - `outputs/final_tables/revision_execution_status.md`
- Status table now has `27` rows and includes:
  - `P1_SCHEMA_COMPLEXITY_DIAGNOSTIC_CONSOLIDATION`
- No training, schema sweep, data-scaling matrix, or SPD variant was launched。

### 指标 / 汇总结论

| Evidence | Pathway | Primary result | Boundary |
| --- | --- | --- | --- |
| no-LM answerability derivative | 4-state no-LM logits | macro-AUC `0.607103`, macro-F1 `0.351833` | derived diagnostic, not explicit S2 supervision |
| no-LM uncertainty derivative | 4-state no-LM logits | macro-AUC `0.608101`, macro-F1 `0.014694` | derived diagnostic, rare positives, not explicit S3 supervision |
| frozen-LM S2 debug | JSON serializer | `target_json` contains `answerable`; runtime passed | debug only, not formal performance evidence |
| frozen-LM S3 debug | JSON serializer | `target_json` contains `uncertain`; runtime passed | debug only, not formal performance evidence |

### 解释

- Current schema-complexity evidence is now consolidated as a diagnostic package。
- The evidence supports:
  - frozen-LM can serialize richer S2/S3 fields in debug mode；
  - no-LM can expose answerability/uncertainty diagnostics from existing 4-state logits；
  - no formal S2/S3 performance comparison is complete。
- This is the right claim boundary for revision writing unless a later phase explicitly designs and trains a matched no-LM S2/S3 objective。

### 失败 / 限制

- No runtime failure。
- frozen-LM S2/S3 rows remain debug-entry evidence only。
- no-LM S2/S3 rows are derived diagnostics only。
- No explicit no-LM multi-head comparator was implemented; it remains deferred as a new variant。

### 下一步

- Use `outputs/final_tables/schema_complexity_diagnostic_summary.md` as the current schema-complexity boundary table。
- If continuing Phase 1, the next decision should be whether to:
  - stop schema work at diagnostic evidence for the current revision；or
  - open a separate execution plan for a true no-LM S2/S3 multi-head variant。
- Given current user priority and “do not continue SPD new variants”, prefer stopping schema work at diagnostic evidence unless the paper absolutely needs a formal S2/S3 sweep。

## 2026-06-18 Phase 0-4 / REVISION_EXECUTION_COMPLETION_GAP_AUDIT 执行前记录

### 计划

1. Re-read the original required tasks in `vivid_med_revision_execution_plan.md` and compare them against current file-provenance status。
2. Identify which tasks are:
   - completed with evidence；
   - completed only as diagnostic/limited evidence；
   - intentionally deferred because evidence already weakens the claim or would require a new variant；
   - still genuinely missing for the revision objective。
3. Use read-only subagents for independent checks:
   - original plan requirement gap audit；
   - availability of sample-level artifacts for `P1_LLM_FAILURE_CASE_MINING`。
4. Produce a gap/decision table under `outputs/final_tables/` and refresh execution status。
5. Do not launch training, full matrices, schema sweeps, SPD variants, or destructive file operations。

### 命令

```powershell
rg -n "Task ID:|Priority|Success boundary|Success criteria|Phase 4|P1_LLM_FAILURE_CASE_MINING|P1_EXTERNAL_CXR_TRANSFER|P2_" vivid_med_revision_execution_plan.md
Get-Content outputs\final_tables\revision_execution_status.md
Get-ChildItem outputs -Recurse -File -Include *.csv,*.json,*.jsonl,*.md,*.pt -ErrorAction SilentlyContinue
python scripts\summarize_revision_completion_gap_audit.py
python -m py_compile scripts\summarize_revision_completion_gap_audit.py scripts\summarize_revision_execution_status.py
python scripts\summarize_revision_execution_status.py
```

### 输入

- Plan: `vivid_med_revision_execution_plan.md`
- Status: `outputs/final_tables/revision_execution_status.md`
- Existing final tables under `outputs/final_tables/`
- Existing metrics/checkpoints/log artifacts under `outputs/`
- Read-only subagent findings。

### 预期输出

- `outputs/final_tables/revision_completion_gap_audit.csv`
- `outputs/final_tables/revision_completion_gap_audit.md`
- Updated `outputs/final_tables/revision_execution_status.csv`
- Updated `outputs/final_tables/revision_execution_status.md`
- Execution-after record in this document。

### 停止条件

- Stop if the audit would require guessing about missing artifacts。
- Stop if evidence is only indirect and cannot support a completion/defer decision。
- Stop if the next action implies running the full plan at once。
- Stop if the path drifts into SPD new variants or new model-module implementation。

### 执行后结果

- Generated completion gap audit:
  - `outputs/final_tables/revision_completion_gap_audit.csv`
  - `outputs/final_tables/revision_completion_gap_audit.md`
- Refreshed execution status:
  - `outputs/final_tables/revision_execution_status.csv`
  - `outputs/final_tables/revision_execution_status.md`
- Status table now has `28` rows and includes:
  - `REVISION_EXECUTION_COMPLETION_GAP_AUDIT`
- Read-only subagent audit confirmed:
  - Priority 0 diagnostics P0/P3/field difficulty are complete or complete-with-limits。
  - `P1_DATA_SCALING` is partial: only 1k matched LP is complete, and it is negative/mixed for frozen-LM macro-AUC。
  - `P1_SCHEMA_COMPLEXITY_SWEEP` is diagnostic-only, not a formal sweep。
  - `P1_LLM_FAILURE_CASE_MINING` is missing and should be the next safest no-training target。
  - Phase 4 `llm_necessity.csv` and `module_candidates.csv` are still absent。
- Second read-only subagent confirmed:
  - Existing `outputs/failure_cases/null_as_negative_over_absent.csv` is useful but insufficient。
  - Required per-sample/per-label probabilities for frozen-LM/no-LM/BCE/random-LM are not exported。
  - Minimal path is an eval-only probability export from existing LP checkpoints, not retraining。

### 指标 / gap 表结论

| Requirement | Current status | Decision |
| --- | --- | --- |
| P0 result/cost | complete or complete-with-missing-fields | use with caveats |
| P1 field difficulty | complete aggregate group evidence | use as frozen-LM use-case evidence |
| P1 data scaling | partial 1k only, negative/mixed for frozen-LM macro-AUC | do not claim low-data frozen-LM necessity |
| P1 schema complexity | diagnostic-only | do not claim formal S2/S3 performance comparison |
| P1 failure mining | missing | next no-training target |
| P2 modules | deferred | do not start before failure mining |
| Phase 4 final packet | missing dedicated `llm_necessity.csv` / `module_candidates.csv` | generate after failure-mining boundary |

### 失败 / 限制

- No runtime failure。
- This audit does not complete missing experiments; it makes their evidence boundary explicit。
- Completion is not proven for the full original plan because several original-scope items remain partial or deferred。

### 下一步

- Proceed to `P1_LLM_FAILURE_CASE_MINING_PRELIGHT_AND_EXPORT`:
  - first inspect configs/checkpoints and data split；
  - implement an eval-only LP probability export if checkpoint compatibility is verified；
  - only then mine frozen-LM vs no-LM sample/label differences。
- Do not start P2 modules, full data-scaling matrix, formal schema sweep, external CXR transfer, or SPD variants before this failure-mining boundary is resolved。

## 2026-06-18 Phase 1 / P1_LLM_FAILURE_CASE_MINING_PREFLIGHT_AND_EXPORT 执行前记录

### 计划

1. Preflight existing LP checkpoints and configs for four methods:
   - frozen-LM UMS no-SPD；
   - no-LM UMS state-classifier backbone LP；
   - BCE ViT-B；
   - random-LM same-architecture UMS LP。
2. Implement an eval-only script that:
   - loads existing checkpoints；
   - runs validation forward passes only；
   - exports per-sample/per-label probabilities；
   - attaches `sample_id`, `image_path`, label, UMS state, answerability and null/uncertain flags；
   - computes correctness only for binary present/absent labels under `uncertain_policy=ignore`。
3. Mine failure-case tables:
   - frozen-LM better than no-LM；
   - no-LM better than frozen-LM；
   - all methods fail；
   - random-LM-specific failures；
   - aggregate failure summary。
4. Do not train, resume, or modify checkpoints。

### 命令

```powershell
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime,Path
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
Test-Path outputs\lp_A_ums_12label\best.pt
Test-Path outputs\lp_ums_classifier_no_llm_12label_full\best.pt
Test-Path outputs\baseline_vit_full14\best.pt
Test-Path outputs\lp_ums_random_lm_12label\best.pt
python -m py_compile scripts\export_failure_mining_predictions.py
python scripts\export_failure_mining_predictions.py --split val --max-samples 1000
python scripts\summarize_revision_execution_status.py
```

### 输入

- Validation split:
  - `data/dataset/processed/chexpert_ums_val.jsonl`
- Checkpoints:
  - `outputs/lp_A_ums_12label/best.pt`
  - `outputs/lp_ums_classifier_no_llm_12label_full/best.pt`
  - `outputs/baseline_vit_full14/best.pt`
  - `outputs/lp_ums_random_lm_12label/best.pt`
- Config references:
  - `configs/lp_A_ums_12label.yaml`
  - `configs/lp_ums_classifier_no_llm_12label.yaml`
  - `configs/lp_ums_random_lm_12label.yaml`
  - BCE config is absent in the current repo; use the checkpoint plus default full-14 CheXpert val loader as provenance。
- Code:
  - `scripts/train_vit_baseline.py`
  - `data/chexpert_dataset.py`
  - `evaluation/metrics.py`

### 预期输出

- `outputs/failure_cases/lp_failure_mining_predictions.csv`
- `outputs/failure_cases/frozen_better_than_no_lm.csv`
- `outputs/failure_cases/no_lm_better_than_frozen.csv`
- `outputs/failure_cases/all_methods_fail.csv`
- `outputs/failure_cases/random_lm_failure_examples.csv`
- `outputs/final_tables/failure_case_summary.csv`
- `outputs/final_tables/failure_case_summary.md`
- Updated `outputs/final_tables/revision_execution_status.csv`
- Updated `outputs/final_tables/revision_execution_status.md`

### 停止条件

- Stop before running if another Python training process is active。
- Stop before running if GPU is occupied by an unknown process。
- Stop if any required checkpoint is missing。
- Stop if checkpoint model state cannot be loaded into a 14-label ViT LP model。
- Stop if the script attempts optimizer/training/resume behavior。
- Stop if sample-level probabilities cannot be exported; document blocker rather than fabricating failure cases。
- Stop if null/uncertain labels are treated as binary present/absent correctness without explicit policy。

### 执行后结果

- Added eval-only export/mining script:
  - `scripts/export_failure_mining_predictions.py`
- Syntax check passed:
  - `python -m py_compile scripts\export_failure_mining_predictions.py`
- Preflight:
  - No active training process before launch。
  - GPU0/GPU1 were `0 / 24576 MB`, `0%` utilization。
  - Required checkpoints existed:
    - `outputs/lp_A_ums_12label/best.pt`
    - `outputs/lp_ums_classifier_no_llm_12label_full/best.pt`
    - `outputs/baseline_vit_full14/best.pt`
    - `outputs/lp_ums_random_lm_12label/best.pt`
- Ran:
  - `python scripts\export_failure_mining_predictions.py --split val --max-samples 1000`
- Loaded:
  - `1000` samples from `data/dataset/processed/chexpert_ums_val.jsonl`
  - full CheXpert `14` labels。
- Exported:
  - `14000` per-sample/per-label prediction rows。
- Generated artifacts:
  - `outputs/failure_cases/lp_failure_mining_predictions.csv`
  - `outputs/failure_cases/frozen_better_than_no_lm.csv`
  - `outputs/failure_cases/no_lm_better_than_frozen.csv`
  - `outputs/failure_cases/all_methods_fail.csv`
  - `outputs/failure_cases/random_lm_failure_examples.csv`
  - `outputs/final_tables/failure_case_summary.csv`
  - `outputs/final_tables/failure_case_summary.md`
  - `outputs/final_tables/failure_case_manifest.json`
- Refreshed:
  - `outputs/final_tables/revision_completion_gap_audit.csv`
  - `outputs/final_tables/revision_completion_gap_audit.md`
  - `outputs/final_tables/revision_execution_status.csv`
  - `outputs/final_tables/revision_execution_status.md`
- Status table now has `29` rows and includes:
  - `P1_LLM_FAILURE_CASE_MINING_PREFLIGHT_AND_EXPORT`

### 指标

Correctness was computed only on binary present/absent fields; null and uncertain rows are retained in the long table but excluded from correct/incorrect case definitions。

| Failure class | Count | Share of valid binary fields | Top fields | Rare count | Uncertain-heavy count | High-null count |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| frozen better than no-LM | `95` | `0.026382` | Pneumothorax `30`; Enlarged Cardiomediastinum `17`; Pleural Effusion `14`; Cardiomegaly `12`; Edema `11` | `0` | `8` | `12` |
| no-LM better than frozen | `102` | `0.028325` | Pneumothorax `39`; Pleural Effusion `23`; Edema `11`; Enlarged Cardiomediastinum `11`; Lung Opacity `7` | `1` | `6` | `5` |
| all methods fail | `214` | `0.059428` | Pneumothorax `49`; Edema `30`; Enlarged Cardiomediastinum `25`; Lung Opacity `21`; Pleural Effusion `19` | `22` | `23` | `38` |
| random-LM failure examples | `297` | `0.082477` | Pleural Effusion `89`; Pneumothorax `82`; Consolidation `46`; Enlarged Cardiomediastinum `35`; Edema `28` | `0` | `46` | `16` |

### 解释

- Failure mining does not show a strong frozen-LM dominance pattern:
  - frozen-LM fixes no-LM errors in `95` binary field instances；
  - no-LM fixes frozen-LM errors in `102` binary field instances。
- The main disagreement field is Pneumothorax for both directions, so this is not a clean rare/high-null-only frozen-LM story。
- The all-methods-fail set has more rare/high-null examples and is a better source for appendix qualitative cases than for justifying a new Phase 2 module。
- Random-LM failures are common, especially Pleural Effusion/Pneumothorax/Consolidation, supporting that random same-architecture LM remains a weak control rather than a replacement for pretrained frozen-LM。

### 失败 / 限制

- No runtime failure and no nonzero exit。
- PyTorch emitted the known `torch.load(..., weights_only=False)` FutureWarning for local trusted checkpoints。
- BCE config path is absent in the current repo; provenance uses `outputs/baseline_vit_full14/best.pt` plus a default full-14 CheXpert val loader。
- Null/uncertain labels are not treated as binary correctness; they are available in `lp_failure_mining_predictions.csv` for qualitative/semantic filtering only。
- This is validation-split failure mining, not train/test OOF prediction export。

### 下一步

- Do not start P2 modules from this evidence alone; failure mining does not strongly motivate a new module。
- Proceed to Phase 4 synthesis:
  - `outputs/final_tables/llm_necessity.csv/.md`
  - `outputs/final_tables/module_candidates.csv/.md`
  - writing/claim checklist that states which claims are supported, downgraded, or forbidden。
- Keep data-scaling and schema-complexity limitations explicit:
  - `P1_DATA_SCALING` is only 1k matched LP, not full 1k/3k/10k/30k；
  - `P1_SCHEMA_COMPLEXITY_SWEEP` is diagnostic-only, not a formal sweep。

## 2026-06-18 Phase 4 / PHASE4_REVISION_SYNTHESIS 执行前记录

### 计划

1. Generate final revision synthesis tables required by the top-level plan:
   - `llm_necessity.csv/.md`
   - `module_candidates.csv/.md`
   - writing/claim checklist。
2. Use only existing consolidated evidence:
   - main controlled results；
   - 1k matched LP data scaling；
   - field difficulty；
   - schema complexity diagnostics；
   - failure-case mining；
   - answerability semantics；
   - schema dependency diagnostics；
   - cost table。
3. Clearly mark:
   - supported claims；
   - downgraded claims；
   - forbidden claims；
   - missing/incomplete original plan scope。
4. Do not start P2 modules, SPD variants, full data-scaling matrix, schema sweep, or external dataset work。

### 命令

```powershell
Get-Content outputs\final_tables\main_controlled_results.csv
Get-Content outputs\final_tables\data_scaling_1k_matched_deltas.csv
Get-Content outputs\final_tables\grouped_field_results.csv
Get-Content outputs\final_tables\schema_complexity_diagnostic_summary.csv
Get-Content outputs\final_tables\failure_case_summary.csv
python scripts\summarize_phase4_revision_synthesis.py
python -m py_compile scripts\summarize_phase4_revision_synthesis.py scripts\summarize_revision_execution_status.py
python scripts\summarize_revision_execution_status.py
```

### 输入

- `outputs/final_tables/main_controlled_results.csv`
- `outputs/final_tables/claim_support_matrix.md`
- `outputs/final_tables/cost_table.csv`
- `outputs/final_tables/grouped_field_results.csv`
- `outputs/final_tables/data_scaling_1k_matched_summary.csv`
- `outputs/final_tables/data_scaling_1k_matched_deltas.csv`
- `outputs/final_tables/schema_complexity_diagnostic_summary.csv`
- `outputs/final_tables/failure_case_summary.csv`
- `outputs/final_tables/answerability_semantics.csv`
- `outputs/final_tables/schema_dependency_diagnostics.csv`
- `outputs/final_tables/revision_completion_gap_audit.csv`

### 预期输出

- `outputs/final_tables/llm_necessity.csv`
- `outputs/final_tables/llm_necessity.md`
- `outputs/final_tables/module_candidates.csv`
- `outputs/final_tables/module_candidates.md`
- `outputs/final_tables/phase4_writing_claim_checklist.md`
- Updated `outputs/final_tables/revision_execution_status.csv`
- Updated `outputs/final_tables/revision_execution_status.md`

### 停止条件

- Stop if an input table is missing。
- Stop if the synthesis would imply full data-scaling or formal schema sweep completion。
- Stop if module candidates are presented as implemented when they are deferred。
- Stop if forbidden claims are softened into supported claims。

### 执行后结果

- Added Phase 4 synthesis script:
  - `scripts/summarize_phase4_revision_synthesis.py`
- Syntax check passed:
  - `python -m py_compile scripts\summarize_phase4_revision_synthesis.py scripts\summarize_revision_execution_status.py`
- Ran:
  - `python scripts\summarize_phase4_revision_synthesis.py`
- Generated:
  - `outputs/final_tables/llm_necessity.csv`
  - `outputs/final_tables/llm_necessity.md`
  - `outputs/final_tables/module_candidates.csv`
  - `outputs/final_tables/module_candidates.md`
  - `outputs/final_tables/phase4_writing_claim_checklist.md`
- Regenerated:
  - `outputs/final_tables/revision_completion_gap_audit.csv`
  - `outputs/final_tables/revision_completion_gap_audit.md`
  - `outputs/final_tables/revision_execution_status.csv`
  - `outputs/final_tables/revision_execution_status.md`
- Status table now has `30` rows and includes:
  - `PHASE4_REVISION_SYNTHESIS`

### 指标 / synthesis 结论

| Claim area | Status | Key result |
| --- | --- | --- |
| UMS/schema contribution | supported | no-LM UMS CheXpert AUC `0.8273` vs BCE `0.7927`; no-LM NIH AUC `0.7262` |
| pretrained frozen-LM in-domain gain | limited supported | frozen CheXpert AUC `0.8439` vs no-LM `0.8273` and random-LM `0.7411` |
| external/NIH LLM dominance | weakened | no-LM NIH AUC `0.7262` > frozen NIH AUC `0.7068` |
| low-data frozen-LM necessity | not supported by current 1k | matched LP final frozen-minus-no-LM macro-AUC `-0.030367` |
| rare/high-null/uncertain use case | supported as subgroup signal | rare `+0.0437`; uncertain-heavy `+0.0155`; high-null `+0.0323` |
| schema complexity | diagnostic only | no formal S1/S2/S3 performance sweep |
| failure mining | mixed | frozen better `95` vs no-LM better `102` |
| answerability/null semantics | semantic boundary supported | null-as-negative changes missingness semantics |
| schema serialization robustness | limitation supported | fixed schema/paraphrase dependence quantified |

### Module candidate decisions

- `Adaptive LLM Gating`: defer; field groups are promising but failure mining is balanced。
- `Hierarchical UMS Head`: defer as future work; requires new heads/losses。
- `Field/state-balanced loss`: possible future appendix, not current priority。
- `Field-query bottleneck`: defer。
- `Counterfactual margin training`: defer。
- `Schema augmentation/canonicalization`: optional future mitigation。
- New SPD variants: forbidden / out of scope。

### 失败 / 限制

- No runtime failure。
- Phase 4 synthesis does not complete the original full experimental scope:
  - `P1_DATA_SCALING` remains partial: only 1k matched LP is complete；
  - `P1_SCHEMA_COMPLEXITY_SWEEP` remains diagnostic-only, not a formal S1/S2/S3 sweep。
- The synthesis explicitly records these boundaries rather than hiding them。

### 下一步

- Current non-training revision packet is complete and ready for paper rewrite:
  - result tables；
  - claim-control table；
  - LLM necessity map；
  - module candidate decision table；
  - writing/claim checklist。
- Remaining original-scope gaps are long-running full-matrix items:
  - 3k/10k/30k data scaling；
  - formal schema-complexity sweep。
- Do not start P2 modules or broad training matrices unless the paper explicitly requires additional evidence。

### 执行后结果

- Preflight:
  - No active `python.exe` training process。
  - GPU0/GPU1 had `0MiB / 24576MiB` memory usage and no running GPU process。
  - Source checkpoint exists: `outputs/data_scaling/no_lm_ums_1k/best.pt`
    - size: `1,030,215,238` bytes
    - timestamp: `2026-06-17 17:38:15`
  - Debug completion marker did not exist before launch:
    - `outputs/data_scaling/lp_no_lm_ums_1k_seed900121/metrics_final.json`: `False`
- Command completed with exit code `0`:
  - `python scripts/train_vit_baseline.py --config configs\data_scaling\lp_no_lm_ums_1k.yaml --debug --seed 900121`
- Checkpoint loading:
  - Loaded ViT backbone from `outputs/data_scaling/no_lm_ums_1k/best.pt`。
  - Loaded params: `150`。
  - Missing keys: `head.weight`, `head.bias` only, as expected for a new LP head。
- LP mode:
  - Backbone frozen params/modules count printed as `150`。
  - Trainable head parameters: `10,766`。
  - Optimizer backbone params: `0`。
  - Optimizer head params: `10,766`。
- Debug data:
  - Train samples: `200` from `data/splits/chexpert_train_1k.jsonl`。
  - Validation samples: `50` from `data/splits/chexpert_val_fixed.jsonl`。
  - Labels: 14 CheXpert labels。
  - Train batches: `25`; validation batches: `7`。
- Validation trajectory:
  - Step 5: `val_loss=0.5998`
  - Step 10: `val_loss=0.4222`
  - Step 15: `val_loss=0.4120`
  - Step 20: `val_loss=0.4047`
- Artifacts created:
  - `outputs/data_scaling/lp_no_lm_ums_1k_seed900121/best.pt`
  - `outputs/data_scaling/lp_no_lm_ums_1k_seed900121/step_20.pt`
  - `outputs/data_scaling/lp_no_lm_ums_1k_seed900121/final.pt`
  - `outputs/data_scaling/lp_no_lm_ums_1k_seed900121/metrics_step_5.json`
  - `outputs/data_scaling/lp_no_lm_ums_1k_seed900121/metrics_step_10.json`
  - `outputs/data_scaling/lp_no_lm_ums_1k_seed900121/metrics_step_15.json`
  - `outputs/data_scaling/lp_no_lm_ums_1k_seed900121/metrics_step_20.json`
  - `outputs/data_scaling/lp_no_lm_ums_1k_seed900121/metrics_final.json`
- Post-run GPU/process state:
  - No active training process remains。
  - GPU0/GPU1 returned to `0MiB / 24576MiB` and `0%` utilization。

### 指标

- Debug `metrics_final.json`:
  - `val_loss`: `0.40469602388995035`
  - `macro_f1`: `0.843812987930635`
  - `macro_auc`: `0.7928839414553701`
  - `micro_f1`: `0.7898089171974523`
- These metrics are debug-entry sanity metrics only, from 200 train / 50 validation samples; they are not paper/data-scaling evidence。

### 失败 / 限制

- No execution failure observed。
- Warnings:
  - `torch.load(..., weights_only=False)` FutureWarning from PyTorch; acceptable for local trusted checkpoint, but worth noting for future hardening。
  - `torch.optim.lr_scheduler` deprecation warning about epoch parameter; not blocking。
- Debug metrics are intentionally small-sample and should not be compared against formal runs。

### 下一步

- The no-LM UMS 1k LP entry is runnable and may proceed to formal LP after a new execution-before record。
- Formal LP should use:
  - `python scripts\train_vit_baseline.py --config configs\data_scaling\lp_no_lm_ums_1k.yaml`
  - output dir: `outputs/data_scaling/lp_no_lm_ums_1k`
- Do not launch 3k/10k/30k LPs or schema sweep in the same step。

## 2026-06-18 Phase 1 / P1_DATA_SCALING_1K_NO_LM_UMS_LP_FORMAL_RUN 执行前记录

### 计划

1. Launch only the formal no-LM UMS 1k LP run。
2. Use the matched no-LM source checkpoint:
   - `outputs/data_scaling/no_lm_ums_1k/best.pt`
3. Use a dedicated wrapper/log so the run records `START`, command, and `EXITCODE`。
4. Monitor process/GPU/log and first validation boundary before treating the run as stable。
5. Do not launch frozen-LM rerun, schema sweep, random-LM, 3k/10k/30k jobs, SPD variants, or a broader LP matrix in this step。

### 命令

```powershell
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime,Path
nvidia-smi
Test-Path outputs\data_scaling\lp_no_lm_ums_1k\metrics_final.json
Test-Path outputs\data_scaling\no_lm_ums_1k\best.pt

scripts\run_data_scaling_lp_no_lm_ums_1k.cmd

Get-Content outputs\logs\data_scaling_lp_no_lm_ums_1k.log -Tail 80
Get-ChildItem outputs\data_scaling\lp_no_lm_ums_1k -ErrorAction SilentlyContinue
```

### 输入

- Config: `configs/data_scaling/lp_no_lm_ums_1k.yaml`
- Source checkpoint: `outputs/data_scaling/no_lm_ums_1k/best.pt`
- Fixed splits:
  - train: `data/splits/chexpert_train_1k.jsonl`
  - validation: `data/splits/chexpert_val_fixed.jsonl`
- Script: `scripts/train_vit_baseline.py`
- Wrapper: `scripts/run_data_scaling_lp_no_lm_ums_1k.cmd`
- Log: `outputs/logs/data_scaling_lp_no_lm_ums_1k.log`

### 预期输出

- Formal output dir: `outputs/data_scaling/lp_no_lm_ums_1k`
- Expected artifacts if complete:
  - `outputs/data_scaling/lp_no_lm_ums_1k/best.pt`
  - `outputs/data_scaling/lp_no_lm_ums_1k/final.pt`
  - `outputs/data_scaling/lp_no_lm_ums_1k/metrics_final.json`
  - periodic `metrics_step_*.json`
  - periodic `step_*.pt`
- Log should end with `EXITCODE 0` if the run completes normally。

### 停止条件

- Stop before launch if `outputs/data_scaling/lp_no_lm_ums_1k/metrics_final.json` already exists。
- Stop before launch if another training process is active or GPU is occupied by an unrelated process。
- Stop before launch if source checkpoint `best.pt` is absent。
- Stop/diagnose if checkpoint loading fails, loaded params are zero, process exits without `EXITCODE 0`, or a Python/CUDA/OOM traceback appears。
- Stop/diagnose if no log/GPU progress occurs for more than 10 minutes outside import/startup/checkpoint windows。

### 执行后结果

- Wrapper launched:
  - `scripts/run_data_scaling_lp_no_lm_ums_1k.cmd`
  - wrapper PID at launch: `6940`
  - log: `outputs/logs/data_scaling_lp_no_lm_ums_1k.log`
- Preflight:
  - No active `python.exe` training process before launch。
  - GPU0/GPU1 had no running GPU process。
  - `outputs/data_scaling/lp_no_lm_ums_1k/metrics_final.json`: `False` before launch。
  - `outputs/data_scaling/no_lm_ums_1k/best.pt`: `True`。
- Checkpoint loading / LP mode:
  - Loaded ViT backbone from `outputs/data_scaling/no_lm_ums_1k/best.pt`。
  - Loaded params: `150`。
  - Missing keys were the new classifier head only。
  - Linear probe mode froze the ViT backbone。
  - Trainable head parameters: `10,766`。
- Formal data:
  - Train samples: `1000` from `data/splits/chexpert_train_1k.jsonl`。
  - Validation samples: `1000` from `data/splits/chexpert_val_fixed.jsonl`。
  - Labels: 14 CheXpert labels。
  - Train batches: `62`; validation batches: `63`。
- Completion:
  - Training reached `3000 / 3000` steps。
  - Log contains `Training completed!`。
  - Wrapper wrote `EXITCODE 0`。
  - GPU memory returned to `0MiB / 24576MiB` on both GPUs after completion。
- Artifacts:
  - `outputs/data_scaling/lp_no_lm_ums_1k/best.pt`
  - `outputs/data_scaling/lp_no_lm_ums_1k/final.pt`
  - `outputs/data_scaling/lp_no_lm_ums_1k/metrics_final.json`
  - `outputs/data_scaling/lp_no_lm_ums_1k/metrics_step_{200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000}.json`
  - `outputs/data_scaling/lp_no_lm_ums_1k/step_{600,1200,1800,2400,3000}.pt`

### 指标

- Validation loss trajectory:
  - step 200: `0.4113`
  - step 400: `0.4672`
  - step 600: `0.5029`
  - step 800: `0.5356`
  - step 1000: `0.5593`
  - step 1200: `0.5941`
  - step 1400: `0.6140`
  - step 1600: `0.6211`
  - step 1800: `0.6355`
  - step 2000: `0.6415`
  - step 2200: `0.6483`
  - step 2400: `0.6533`
  - step 2600: `0.6592`
  - step 2800: `0.6652`
  - step 3000: `0.6633`
- Best validation checkpoint:
  - `best.pt` corresponds to step `200` by lowest observed `val_loss=0.4113499453616521`。
  - Step 200 metrics:
    - `macro_f1`: `0.8819374479649502`
    - `macro_auc`: `0.7169596759786138`
    - `micro_f1`: `0.8492883058889199`
- Final checkpoint metrics from `metrics_final.json`:
  - `val_loss`: `0.6633415856058635`
  - `macro_f1`: `0.8844983233539759`
  - `macro_auc`: `0.719718373475444`
  - `micro_f1`: `0.8464973485905666`

### 解释

- The formal no-LM UMS 1k LP run is complete and provides the matched LP comparator for frozen-LM 1k LP。
- no-LM validation loss worsens steadily after step 200, but final macro-AUC remains higher than the frozen-LM final macro-AUC。
- This is a negative or mixed signal for the low-data frozen-LM necessity hypothesis at 1k:
  - frozen-LM has slightly higher macro-F1 / micro-F1 under matched LP。
  - no-LM has higher macro-AUC under both final-policy and best-validation-step-policy。

### 失败 / 限制

- No runtime failure, no CUDA/OOM/Python traceback, and no nonzero exit。
- PyTorch emitted the known `torch.load(..., weights_only=False)` FutureWarning for local checkpoint loading。
- This run resolves only the no-LM 1k LP row; 3k/10k/30k scaling remains unrun。

### 下一步

- Update status tables to mark `P1_DATA_SCALING_1K_NO_LM_UMS_LP_FORMAL_RUN` complete。
- Consolidate the 1k matched data-scaling evidence into a table with explicit metric policy:
  - BCE 1k final classifier metrics。
  - no-LM source metrics (not matched LP; keep separate)。
  - no-LM 1k LP final + best-step metrics。
  - frozen-LM 1k LP final + best-step metrics。
- Do not claim low-data frozen-LM benefit unless the table explicitly shows the metric-policy boundary。

## 2026-06-18 Phase 1 / P1_DATA_SCALING_1K_MATCHED_RESULT_CONSOLIDATION 执行前记录

### 计划

1. Read completed 1k formal artifacts only; do not start training。
2. Build a compact table separating:
   - classifier baseline final metrics;
   - source-run metrics that are not LP-comparable;
   - matched LP final metrics;
   - matched LP best-validation-step metrics。
3. Compute frozen-LM minus no-LM deltas under matched LP policies。
4. Compute LP vs BCE reference deltas where the metric policy is explicit。
5. Write CSV/Markdown outputs under `outputs/final_tables/`。
6. Update status table after outputs are created。

### 命令

```powershell
python scripts\summarize_data_scaling_1k_results.py
Get-Content outputs\final_tables\data_scaling_1k_matched_summary.md
Get-Content outputs\final_tables\data_scaling_1k_matched_deltas.csv
```

### 输入

- `outputs/data_scaling/bce_1k/metrics_final.json`
- `outputs/data_scaling/no_lm_ums_1k/metrics_final.json`
- `outputs/data_scaling/lp_no_lm_ums_1k/metrics_final.json`
- `outputs/data_scaling/lp_no_lm_ums_1k/metrics_step_200.json`
- `outputs/data_scaling/lp_frozen_lm_ums_1k/metrics_final.json`
- `outputs/data_scaling/lp_frozen_lm_ums_1k/metrics_step_400.json`
- Source/checkpoint provenance already recorded above。

### 预期输出

- `outputs/final_tables/data_scaling_1k_matched_summary.csv`
- `outputs/final_tables/data_scaling_1k_matched_summary.md`
- `outputs/final_tables/data_scaling_1k_matched_deltas.csv`
- `outputs/final_tables/data_scaling_1k_matched_deltas.md`

### 停止条件

- Stop if any required formal metrics JSON is missing。
- Stop if the script would mix source metrics with LP metrics in one comparison row。
- Stop if no-LM/frozen-LM LP deltas cannot be computed under the same metric policy。
- Stop if any output implies frozen-LM source completed normally instead of documented early-stop provenance。

### 执行后结果

- Command completed with exit code `0`:
  - `python scripts\summarize_data_scaling_1k_results.py`
- Created outputs:
  - `outputs/final_tables/data_scaling_1k_matched_summary.csv`
  - `outputs/final_tables/data_scaling_1k_matched_summary.md`
  - `outputs/final_tables/data_scaling_1k_matched_deltas.csv`
  - `outputs/final_tables/data_scaling_1k_matched_deltas.md`
- The summary explicitly separates:
  - `classifier_reference`: BCE 1k classifier rows。
  - `source_not_lp`: no-LM source/state-classifier rows。
  - `matched_lp`: no-LM and frozen-LM LP rows。
- The frozen-LM source provenance is explicitly written as documented early-stop `best.pt`, not normal completion。

### 指标

Matched LP final-policy:

| Method | macro-AUC | macro-F1 | micro-F1 | val_loss |
|---|---:|---:|---:|---:|
| no-LM UMS LP final | `0.719718` | `0.884498` | `0.846497` | `0.663342` |
| frozen-LM UMS LP final | `0.689351` | `0.887014` | `0.850126` | `0.396421` |
| frozen - no-LM | `-0.030367` | `+0.002516` | `+0.003629` |  |

Matched LP best-validation-step policy:

| Method | best step | macro-AUC | macro-F1 | micro-F1 | val_loss |
|---|---:|---:|---:|---:|---:|
| no-LM UMS LP best | `200` | `0.716960` | `0.881937` | `0.849288` | `0.411350` |
| frozen-LM UMS LP best | `400` | `0.705861` | `0.891938` | `0.855708` | `0.354914` |
| frozen - no-LM |  | `-0.011099` | `+0.010001` | `+0.006420` |  |

BCE final-policy reference:

| Method | macro-AUC | macro-F1 | micro-F1 |
|---|---:|---:|---:|
| BCE ViT final | `0.684366` | `0.886060` | `0.850126` |
| frozen-LM LP final - BCE final | `+0.004985` | `+0.000954` | `+0.000000` |
| no-LM LP final - BCE final | `+0.035352` | `-0.001562` | `-0.003629` |

### 解释

- 1k matched LP evidence does **not** support a broad low-data frozen-LM necessity claim on macro-AUC。
- Under both final-policy and best-validation-step-policy, no-LM UMS LP has higher macro-AUC than frozen-LM UMS LP。
- frozen-LM UMS LP has small positive macro-F1/micro-F1 deltas, so the evidence is mixed, not a total no-LM win。
- Relative to BCE final, frozen-LM LP is only marginally higher in macro-AUC, while no-LM LP has the larger macro-AUC gain。
- no-LM source/state-classifier metrics remain provenance-only in this table and should not be directly compared to LP rows。

### 失败 / 限制

- No consolidation failure。
- The table covers only 1k; it does not answer whether frozen-LM becomes more useful at 3k/10k or under more complex schema。
- The matched LP best-step policy uses different best steps (`200` vs `400`), so it should be labelled as best-validation-step policy rather than final checkpoint policy。

### 下一步

- Update status tables to mark `P1_DATA_SCALING_1K_MATCHED_RESULT_CONSOLIDATION` complete。
- Scientific implication for next queue:
  - Do not claim frozen-LM low-data necessity from 1k data。
  - If continuing Phase 1, prefer schema complexity / answerability-aware schema tests over blindly extending 3k/10k scaling, because the 1k low-data result is negative on macro-AUC。
- If extending scaling, run only one next size/task after a fresh execution-before record。

## 2026-06-18 Phase 1 / P1_SCHEMA_SWEEP_DEBUG_MODEL_CONFIG_FIX 执行记录

### 计划

1. Inspect frozen-LM schema sweep configs before running any schema debug entry。
2. Ensure debug mode uses the already validated Qwen path instead of falling back to `sshleifer/tiny-gpt2`。
3. Patch both existing configs and the config generator so regeneration preserves the fix。
4. Do not launch training in this config-fix step。

### 命令

```powershell
rg -n "schema_mode|answerability|uncertainty|debug_llm_model_name|llm_model_name" data\chexpert_dataset.py configs\schema_sweep\*.yaml configs\data_scaling\frozen_lm_ums_1k.yaml
```

### 输入

- `configs/schema_sweep/frozen_lm_s1_state_only.yaml`
- `configs/schema_sweep/frozen_lm_s2_state_answerability.yaml`
- `configs/schema_sweep/frozen_lm_s3_state_uncertainty.yaml`
- `scripts/prepare_schema_sweep_configs.py`
- Prior runtime evidence that Qwen debug path works and tiny-gpt2 debug path is unsafe under current torch/transformers behavior。

### 输出 / 结果

- Added `model.debug_llm_model_name: Qwen/Qwen2.5-1.5B-Instruct` to:
  - `configs/schema_sweep/frozen_lm_s1_state_only.yaml`
  - `configs/schema_sweep/frozen_lm_s2_state_answerability.yaml`
  - `configs/schema_sweep/frozen_lm_s3_state_uncertainty.yaml`
- Updated `scripts/prepare_schema_sweep_configs.py` so generated frozen-LM schema configs set:
  - `config["model"]["debug_llm_model_name"] = config["model"]["llm_model_name"]`

### 停止条件结果

- No training was launched。
- No output artifacts were generated。
- This change only affects debug model selection; formal `llm_model_name` remains Qwen。

### 下一步

- Proceed to a single S2 `state_answerability` frozen-LM debug entry after a fresh execution-before record。
- Do not launch S1/S2/S3 full source runs or schema LPs as a batch。

## 2026-06-18 Phase 1 / P1_SCHEMA_COMPLEXITY_S2_FROZEN_LM_DEBUG_ENTRY 执行前记录

### 计划

1. Run only a tiny/debug entry for frozen-LM schema S2 (`state_answerability`)。
2. Verify that `schema_mode=state_answerability` is accepted by `CheXpertUMSDataset` and `scripts/train_cxr.py`。
3. Verify that the target serializer includes answerability fields without breaking tokenization/model forward。
4. Verify that Qwen debug model path loads from cache and the debug run writes expected checkpoint artifacts。
5. Treat this as serializer/runtime evidence only, not a formal schema-complexity result。

### 命令

```powershell
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime,Path
nvidia-smi
Test-Path outputs\schema_sweep\frozen_lm_s2_state_answerability_seed900122\metrics_final.json
python scripts\train_cxr.py --config configs\schema_sweep\frozen_lm_s2_state_answerability.yaml --debug --seed 900122
```

### 输入

- Config: `configs/schema_sweep/frozen_lm_s2_state_answerability.yaml`
- Schema mode: `state_answerability`
- Train split in debug mode: `data/splits/chexpert_train_30k.jsonl` truncated to 20 samples by `--debug`
- Val split in debug mode: `data/splits/chexpert_val_fixed.jsonl` truncated to 4 samples by `--debug`
- Debug LLM: `Qwen/Qwen2.5-1.5B-Instruct`
- Seed override: `900122`

### 预期输出

- Debug output dir: `outputs/schema_sweep/frozen_lm_s2_state_answerability_seed900122`
- Expected artifacts:
  - `outputs/schema_sweep/frozen_lm_s2_state_answerability_seed900122/checkpoints/best.pt`
  - `outputs/schema_sweep/frozen_lm_s2_state_answerability_seed900122/checkpoints/step_5.pt`
  - `outputs/schema_sweep/frozen_lm_s2_state_answerability_seed900122/checkpoints/final.pt`

### 停止条件

- Stop if another training process is active。
- Stop if GPU is occupied by an unrelated process。
- Stop if debug completion marker already exists。
- Stop if config does not contain `schema_mode: state_answerability` and `debug_llm_model_name: Qwen/Qwen2.5-1.5B-Instruct`。
- Stop if the run raises schema serialization, tokenizer/model, CUDA/OOM, or checkpoint-write errors。

### 执行后结果

- Preflight:
  - No active training process。
  - GPU0/GPU1 had `0MiB / 24576MiB` and no running GPU process。
  - Debug completion marker did not exist:
    - `outputs/schema_sweep/frozen_lm_s2_state_answerability_seed900122/metrics_final.json`: `False`
  - Config contains:
    - `schema_mode: state_answerability`
    - `debug_llm_model_name: Qwen/Qwen2.5-1.5B-Instruct`
- Command completed with exit code `0`:
  - `python scripts\train_cxr.py --config configs\schema_sweep\frozen_lm_s2_state_answerability.yaml --debug --seed 900122`
- Runtime:
  - Loaded 20 train samples and 4 validation samples under debug mode。
  - Used 12 selected CheXpert labels。
  - LLM: `Qwen/Qwen2.5-1.5B-Instruct`。
  - LLM loaded from ModelScope cache path; hidden size `1536`。
  - Frozen LLM parameters: `1,543,714,304`。
  - Trainable parameters: `89,349,888`。
  - Output dir: `outputs/schema_sweep/frozen_lm_s2_state_answerability_seed900122`。
- Validation trajectory:
  - Step 2: `val_loss=0.7578`
  - Step 4: `val_loss=0.7426`
- Artifacts:
  - `outputs/schema_sweep/frozen_lm_s2_state_answerability_seed900122/checkpoints/best.pt`
    - size: `1,072,361,762` bytes
  - `outputs/schema_sweep/frozen_lm_s2_state_answerability_seed900122/checkpoints/step_5.pt`
    - size: `1,072,387,794` bytes
  - `outputs/schema_sweep/frozen_lm_s2_state_answerability_seed900122/checkpoints/final.pt`
    - size: `1,072,362,394` bytes
- Serializer inspection:
  - First attempted to read `item["target_text"]`; this failed with `KeyError: 'target_text'` because the dataset uses `target_json` as the serialized target field。
  - Re-inspection of one `state_answerability` sample showed item keys:
    - `answerable`, `image`, `labels`, `original_path`, `prompt_text`, `query_labels`, `sample_id`, `study_view`, `target_json`
  - The printed `target_json` includes answerability entries, e.g. `"answerable": false` and `"answerable": true` inside each finding object。
  - This confirms S2 serializer emits answerability fields and the trainer consumes the resulting batch without failure。
- Post-run:
  - No residual Python process。
  - GPU0/GPU1 returned to `0MiB / 24576MiB`。

### 指标 / 当前解释

- S2 `state_answerability` frozen-LM debug entry is runtime-valid。
- This is serializer/runtime evidence only; it is not formal schema-complexity evidence。
- The debug run validates:
  - schema mode parsing；
  - answerability-field serialization；
  - Qwen frozen-LM path；
  - tokenization/model forward/training loop；
  - checkpoint writing。

### 失败 / 限制

- No training/runtime failure。
- The only failure was a post-run inspection typo/assumption (`target_text` vs `target_json`), corrected immediately。
- no-LM S2 comparator is still not implemented, so no schema-complexity claim can compare no-LM vs frozen-LM S2 yet。
- Full S2 source training was not launched。

### 下一步

- Update status tables to mark `P1_SCHEMA_COMPLEXITY_S2_FROZEN_LM_DEBUG_ENTRY` complete。
- Next safe schema runtime step is S3 `state_uncertainty` frozen-LM debug entry, still not a full run。
- Before any formal S2/S3 schema claim, design/implement no-LM answerability/uncertainty comparator or clearly label frozen-LM-only serializer diagnostics。

## 2026-06-18 Phase 1 / P1_SCHEMA_COMPLEXITY_S3_FROZEN_LM_DEBUG_ENTRY 执行前记录

### 计划

1. Run only a tiny/debug entry for frozen-LM schema S3 (`state_uncertainty`)。
2. Verify that `schema_mode=state_uncertainty` is accepted by `CheXpertUMSDataset` and `scripts/train_cxr.py`。
3. Verify that the target serializer includes uncertainty fields without breaking tokenization/model forward。
4. Verify that Qwen debug model path loads and the debug run writes expected checkpoints。
5. Treat this as serializer/runtime evidence only, not a formal schema-complexity result。

### 命令

```powershell
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime,Path
nvidia-smi
Test-Path outputs\schema_sweep\frozen_lm_s3_state_uncertainty_seed900123\metrics_final.json
python scripts\train_cxr.py --config configs\schema_sweep\frozen_lm_s3_state_uncertainty.yaml --debug --seed 900123
```

### 输入

- Config: `configs/schema_sweep/frozen_lm_s3_state_uncertainty.yaml`
- Schema mode: `state_uncertainty`
- Train split in debug mode: `data/splits/chexpert_train_30k.jsonl` truncated to 20 samples by `--debug`
- Val split in debug mode: `data/splits/chexpert_val_fixed.jsonl` truncated to 4 samples by `--debug`
- Debug LLM: `Qwen/Qwen2.5-1.5B-Instruct`
- Seed override: `900123`

### 预期输出

- Debug output dir: `outputs/schema_sweep/frozen_lm_s3_state_uncertainty_seed900123`
- Expected artifacts:
  - `outputs/schema_sweep/frozen_lm_s3_state_uncertainty_seed900123/checkpoints/best.pt`
  - `outputs/schema_sweep/frozen_lm_s3_state_uncertainty_seed900123/checkpoints/step_5.pt`
  - `outputs/schema_sweep/frozen_lm_s3_state_uncertainty_seed900123/checkpoints/final.pt`

### 停止条件

- Stop if another training process is active。
- Stop if GPU is occupied by an unrelated process。
- Stop if debug completion marker already exists。
- Stop if config does not contain `schema_mode: state_uncertainty` and `debug_llm_model_name: Qwen/Qwen2.5-1.5B-Instruct`。
- Stop if the run raises schema serialization, tokenizer/model, CUDA/OOM, or checkpoint-write errors。

### 执行后结果

- Preflight:
  - No active training process。
  - GPU0/GPU1 had `0MiB / 24576MiB` and no running GPU process。
  - Debug completion marker did not exist:
    - `outputs/schema_sweep/frozen_lm_s3_state_uncertainty_seed900123/metrics_final.json`: `False`
  - Config contains:
    - `schema_mode: state_uncertainty`
    - `debug_llm_model_name: Qwen/Qwen2.5-1.5B-Instruct`
- Command completed with exit code `0`:
  - `python scripts\train_cxr.py --config configs\schema_sweep\frozen_lm_s3_state_uncertainty.yaml --debug --seed 900123`
- Runtime:
  - Loaded 20 train samples and 4 validation samples under debug mode。
  - Used 12 selected CheXpert labels。
  - LLM: `Qwen/Qwen2.5-1.5B-Instruct`。
  - LLM loaded from ModelScope cache path; hidden size `1536`。
  - Frozen LLM parameters: `1,543,714,304`。
  - Trainable parameters: `89,349,888`。
  - Output dir: `outputs/schema_sweep/frozen_lm_s3_state_uncertainty_seed900123`。
- Validation trajectory:
  - Step 2: `val_loss=0.7468`
  - Step 4: `val_loss=0.7321`
- Artifacts:
  - `outputs/schema_sweep/frozen_lm_s3_state_uncertainty_seed900123/checkpoints/best.pt`
    - size: `1,072,361,762` bytes
  - `outputs/schema_sweep/frozen_lm_s3_state_uncertainty_seed900123/checkpoints/step_5.pt`
    - size: `1,072,387,794` bytes
  - `outputs/schema_sweep/frozen_lm_s3_state_uncertainty_seed900123/checkpoints/final.pt`
    - size: `1,072,362,394` bytes
- Serializer inspection:
  - A one-sample `state_uncertainty` dataset inspection showed item keys:
    - `answerable`, `image`, `labels`, `original_path`, `prompt_text`, `query_labels`, `sample_id`, `study_view`, `target_json`
  - The printed `target_json` includes uncertainty entries, e.g. `"uncertain": null` and `"uncertain": false` inside each finding object。
  - `contains_uncertain=True`。
  - This confirms S3 serializer emits uncertainty fields and the trainer consumes the resulting batch without failure。
- Post-run:
  - No residual Python process。
  - GPU0/GPU1 returned to `0MiB / 24576MiB`。

### 指标 / 当前解释

- S3 `state_uncertainty` frozen-LM debug entry is runtime-valid。
- This is serializer/runtime evidence only; it is not formal schema-complexity evidence。
- Together with S2, the frozen-LM path now has runtime-validated serializer support for:
  - `state_answerability`
  - `state_uncertainty`

### 失败 / 限制

- No training/runtime failure。
- no-LM S3 comparator is still not implemented, so no schema-complexity claim can compare no-LM vs frozen-LM S3 yet。
- Full S3 source training was not launched。

### 下一步

- Update status tables to mark `P1_SCHEMA_COMPLEXITY_S3_FROZEN_LM_DEBUG_ENTRY` complete。
- The next scientifically useful step is not a full frozen-LM-only schema sweep; it is a no-LM schema comparator design for answerability/uncertainty, or a clearly labelled frozen-LM-only serializer diagnostic。

### 执行后结果

- Preflight:
  - No active `python.exe` training process。
  - GPU0/GPU1 had `0MiB / 24576MiB` memory usage and no running GPU process。
  - Source checkpoint exists: `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/best.pt`
    - size: `1,072,361,762` bytes
    - timestamp: `2026-06-18 00:42:35`
  - Debug completion marker did not exist before launch:
    - `outputs/data_scaling/lp_frozen_lm_ums_1k_seed900120/metrics_final.json`: `False`
- Command completed with exit code `0`:
  - `python scripts/train_vit_baseline.py --config configs\data_scaling\lp_frozen_lm_ums_1k.yaml --debug --seed 900120`
- Checkpoint loading:
  - Loaded ViT backbone from `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/best.pt`。
  - Loaded params: `150`。
  - Missing keys: `head.weight`, `head.bias` only, as expected for a new LP head。
- LP mode:
  - Backbone frozen params/modules count printed as `150`。
  - Trainable head parameters: `10,766`。
  - Optimizer backbone params: `0`。
  - Optimizer head params: `10,766`。
- Debug data:
  - Train samples: `200` from `data/splits/chexpert_train_1k.jsonl`。
  - Validation samples: `50` from `data/splits/chexpert_val_fixed.jsonl`。
  - Labels: 14 CheXpert labels。
  - Train batches: `25`; validation batches: `7`。
- Validation trajectory:
  - Step 5: `val_loss=0.5308`
  - Step 10: `val_loss=0.3933`
  - Step 15: `val_loss=0.3643`
  - Step 20: `val_loss=0.3549`
- Artifacts created:
  - `outputs/data_scaling/lp_frozen_lm_ums_1k_seed900120/best.pt`
  - `outputs/data_scaling/lp_frozen_lm_ums_1k_seed900120/step_20.pt`
  - `outputs/data_scaling/lp_frozen_lm_ums_1k_seed900120/final.pt`
  - `outputs/data_scaling/lp_frozen_lm_ums_1k_seed900120/metrics_step_5.json`
  - `outputs/data_scaling/lp_frozen_lm_ums_1k_seed900120/metrics_step_10.json`
  - `outputs/data_scaling/lp_frozen_lm_ums_1k_seed900120/metrics_step_15.json`
  - `outputs/data_scaling/lp_frozen_lm_ums_1k_seed900120/metrics_step_20.json`
  - `outputs/data_scaling/lp_frozen_lm_ums_1k_seed900120/metrics_final.json`
- Post-run GPU/process state:
  - No active training process remains。
  - GPU0/GPU1 returned to `0MiB / 24576MiB` and `0%` utilization。

### 指标

- Debug `metrics_final.json`:
  - `val_loss`: `0.3548993170261383`
  - `macro_f1`: `0.8748402391259534`
  - `macro_auc`: `0.7382027269782371`
  - `micro_f1`: `0.8535031847133758`
- These metrics are debug-entry sanity metrics only, from 200 train / 50 validation samples; they are not paper/data-scaling evidence。

### 失败 / 限制

- No execution failure observed。
- Warnings:
  - `torch.load(..., weights_only=False)` FutureWarning from PyTorch; acceptable for local trusted checkpoint, but worth noting for future hardening。
  - `torch.optim.lr_scheduler` deprecation warning about epoch parameter; not blocking。
- Debug metrics are intentionally small-sample and should not be compared against formal BCE/no-LM/frozen-LM data-scaling rows。

### 下一步

- The frozen-LM 1k LP entry is runnable and may proceed to formal LP after a new execution-before record。
- Formal LP should use:
  - `python scripts\train_vit_baseline.py --config configs\data_scaling\lp_frozen_lm_ums_1k.yaml`
  - output dir: `outputs/data_scaling/lp_frozen_lm_ums_1k`
- Do not launch no-LM LP, schema sweep, random-LM, or broader data-scaling matrix in the same step。

## 2026-06-18 Phase 1 / P1_DATA_SCALING_1K_FROZEN_LM_LP_FORMAL_RUN 执行前记录

### 计划

1. Launch only the formal frozen-LM UMS 1k LP run。
2. Use the documented early-stop source checkpoint:
   - `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/best.pt`
3. Use a dedicated wrapper/log so the run records `START`, command, and `EXITCODE` without overwriting prior source logs。
4. Monitor process/GPU/log and first validation boundary before treating the run as stable。
5. Do not launch no-LM LP, BCE rerun, random-LM, schema sweep, SPD variant, or a broader data-scaling matrix in this step。

### 命令

```powershell
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime,Path
nvidia-smi
Test-Path outputs\data_scaling\lp_frozen_lm_ums_1k\metrics_final.json
Test-Path outputs\data_scaling\frozen_lm_ums_1k\checkpoints\best.pt

scripts\run_data_scaling_lp_frozen_lm_ums_1k.cmd

Get-Content outputs\logs\data_scaling_lp_frozen_lm_ums_1k.log -Tail 80
Get-ChildItem outputs\data_scaling\lp_frozen_lm_ums_1k -ErrorAction SilentlyContinue
```

### 输入

- Config: `configs/data_scaling/lp_frozen_lm_ums_1k.yaml`
- Source checkpoint: `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/best.pt`
- Fixed splits:
  - train: `data/splits/chexpert_train_1k.jsonl`
  - validation: `data/splits/chexpert_val_fixed.jsonl`
- Script: `scripts/train_vit_baseline.py`
- Wrapper: `scripts/run_data_scaling_lp_frozen_lm_ums_1k.cmd`
- Log: `outputs/logs/data_scaling_lp_frozen_lm_ums_1k.log`

### 预期输出

- Formal output dir: `outputs/data_scaling/lp_frozen_lm_ums_1k`
- Expected artifacts if complete:
  - `outputs/data_scaling/lp_frozen_lm_ums_1k/best.pt`
  - `outputs/data_scaling/lp_frozen_lm_ums_1k/final.pt`
  - `outputs/data_scaling/lp_frozen_lm_ums_1k/metrics_final.json`
  - periodic `metrics_step_*.json`
  - periodic `step_*.pt`
- Log should end with `EXITCODE 0` if the run completes normally。

### 停止条件

- Stop before launch if `outputs/data_scaling/lp_frozen_lm_ums_1k/metrics_final.json` already exists。
- Stop before launch if another training process is active or GPU is occupied by an unrelated process。
- Stop before launch if source checkpoint `best.pt` is absent。
- Stop/diagnose if checkpoint loading fails, loaded params are zero, process exits without `EXITCODE 0`, or a Python/CUDA/OOM traceback appears。
- Stop/diagnose if no log/GPU progress occurs for more than 10 minutes outside import/startup/checkpoint windows。

### 执行后结果

- Wrapper launched:
  - `scripts/run_data_scaling_lp_frozen_lm_ums_1k.cmd`
  - wrapper PID at launch: `2348`
  - log: `outputs/logs/data_scaling_lp_frozen_lm_ums_1k.log`
- Preflight:
  - No active `python.exe` training process before launch。
  - GPU0/GPU1 had no running GPU process。
  - `outputs/data_scaling/lp_frozen_lm_ums_1k/metrics_final.json`: `False` before launch。
  - `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/best.pt`: `True`。
- Checkpoint loading / LP mode:
  - Loaded ViT backbone from `outputs/data_scaling/frozen_lm_ums_1k/checkpoints/best.pt`。
  - Loaded params: `150`。
  - Missing keys were the new classifier head only。
  - Linear probe mode froze the ViT backbone。
  - Trainable head parameters: `10,766`。
- Formal data:
  - Train samples: `1000` from `data/splits/chexpert_train_1k.jsonl`。
  - Validation samples: `1000` from `data/splits/chexpert_val_fixed.jsonl`。
  - Labels: 14 CheXpert labels。
  - Train batches: `62`; validation batches: `63`。
- Completion:
  - Training reached `3000 / 3000` steps。
  - Log contains `Training completed!`。
  - Wrapper wrote `EXITCODE 0`。
  - GPU memory returned to `0MiB / 24576MiB` on both GPUs after completion。
- Artifacts:
  - `outputs/data_scaling/lp_frozen_lm_ums_1k/best.pt`
  - `outputs/data_scaling/lp_frozen_lm_ums_1k/final.pt`
  - `outputs/data_scaling/lp_frozen_lm_ums_1k/metrics_final.json`
  - `outputs/data_scaling/lp_frozen_lm_ums_1k/metrics_step_{200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000}.json`
  - `outputs/data_scaling/lp_frozen_lm_ums_1k/step_{600,1200,1800,2400,3000}.pt`

### 指标

- Validation loss trajectory:
  - step 200: `0.3650`
  - step 400: `0.3549`
  - step 600: `0.3682`
  - step 800: `0.3787`
  - step 1000: `0.3794`
  - step 1200: `0.3889`
  - step 1400: `0.3868`
  - step 1600: `0.3868`
  - step 1800: `0.3966`
  - step 2000: `0.3860`
  - step 2200: `0.3917`
  - step 2400: `0.3955`
  - step 2600: `0.3947`
  - step 2800: `0.3943`
  - step 3000: `0.3964`
- Best validation checkpoint:
  - `best.pt` corresponds to step `400` by lowest observed `val_loss=0.3549138678917809`。
  - Step 400 metrics:
    - `macro_f1`: `0.8919383985748169`
    - `macro_auc`: `0.7058609569920495`
    - `micro_f1`: `0.8557075076751326`
- Final checkpoint metrics from `metrics_final.json`:
  - `val_loss`: `0.39642118769032614`
  - `macro_f1`: `0.8870136422226503`
  - `macro_auc`: `0.689350781954082`
  - `micro_f1`: `0.850125593078426`

### 解释

- The formal frozen-LM 1k LP run is complete and usable as formal LP evidence。
- The best checkpoint is earlier than final, so future tables must specify whether they use `metrics_final.json` or the best-validation step metrics。
- Using final metrics, frozen-LM 1k LP is close to the formal BCE 1k source result and not clearly dominant。
- Using the best-validation step, frozen-LM 1k LP has a stronger macro-AUC signal, but this must be compared against matched best-step or final-policy metrics for BCE/no-LM LP before making a frozen-LM-usefulness claim。

### 失败 / 限制

- No runtime failure, no CUDA/OOM/Python traceback, and no nonzero exit。
- PyTorch emitted the known `torch.load(..., weights_only=False)` FutureWarning for local checkpoint loading。
- This run resolves only the frozen-LM 1k LP row; it does not yet compare against matched no-LM LP under the same LP protocol。

### 下一步

- Update status tables to mark `P1_DATA_SCALING_1K_FROZEN_LM_LP_FORMAL_RUN` complete。
- Next comparison-critical queue item: no-LM UMS 1k LP under the same fixed split and LP protocol。
- Before running no-LM LP, write a separate execution-before record and run at most the no-LM LP debug/formal pair; do not launch the whole LP matrix。

## 2026-06-18 Phase 1 / P1_DATA_SCALING_1K_NO_LM_UMS_LP_DEBUG_ENTRY 执行前记录

### 计划

1. Run a tiny/debug LP entry for the matched no-LM UMS 1k source checkpoint。
2. Verify `scripts/train_vit_baseline.py` can:
   - parse `configs/data_scaling/lp_no_lm_ums_1k.yaml`;
   - load `outputs/data_scaling/no_lm_ums_1k/best.pt`;
   - freeze the ViT backbone;
   - train/evaluate a small linear head on CUDA;
   - write debug metrics/checkpoints under a seed-suffixed output dir。
3. Keep this as entry validation only; do not treat debug metrics as paper evidence。
4. Do not start formal LP or any matrix job in this step。

### 命令

```powershell
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime,Path
nvidia-smi
Test-Path outputs\data_scaling\no_lm_ums_1k\best.pt
Test-Path outputs\data_scaling\lp_no_lm_ums_1k_seed900121\metrics_final.json
python scripts\train_vit_baseline.py --config configs\data_scaling\lp_no_lm_ums_1k.yaml --debug --seed 900121
```

### 输入

- Config: `configs/data_scaling/lp_no_lm_ums_1k.yaml`
- Source checkpoint: `outputs/data_scaling/no_lm_ums_1k/best.pt`
- Fixed split:
  - `data/splits/chexpert_train_1k.jsonl`
  - `data/splits/chexpert_val_fixed.jsonl`
- Seed override: `900121`

### 预期输出

- Debug output dir: `outputs/data_scaling/lp_no_lm_ums_1k_seed900121`
- Expected artifacts:
  - `outputs/data_scaling/lp_no_lm_ums_1k_seed900121/best.pt`
  - `outputs/data_scaling/lp_no_lm_ums_1k_seed900121/step_20.pt`
  - `outputs/data_scaling/lp_no_lm_ums_1k_seed900121/final.pt`
  - `outputs/data_scaling/lp_no_lm_ums_1k_seed900121/metrics_final.json`

### 停止条件

- Stop if `metrics_final.json` already exists before launch。
- Stop if another training process is active。
- Stop if GPU is occupied by an unrelated process。
- Stop if checkpoint loading fails or loads zero useful ViT parameters。
- Stop if debug run raises an exception, exits nonzero, or writes no final metrics。

## 2026-06-18 Phase 1 / P1_RUNTIME_SPEED_STATUS_RECHECK_AFTER_LP 执行记录

### 计划

1. 回答用户关于“目前训练速度是否变慢”的即时问题。
2. 只做只读诊断：检查 Python 训练进程、GPU 占用、最近日志和系统资源。
3. 将当前机器状态与最近完成的 LP 训练、此前中断的 frozen-LM source 尾段分开解释。
4. 不启动任何新训练、不修改实验指标、不扩展 SPD 新变体。

### 命令

```powershell
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime,Path
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits
Get-ChildItem outputs\logs -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 12 Name,Length,LastWriteTime
Get-Content outputs\logs\data_scaling_lp_no_lm_ums_1k.log -Tail 80
Get-Content outputs\logs\data_scaling_lp_frozen_lm_ums_1k.log -Tail 80
Get-Content outputs\logs\data_scaling_frozen_lm_ums_1k_source.log -Tail 80
Get-Counter '\Processor(_Total)\% Processor Time','\Memory\Available MBytes','\PhysicalDisk(_Total)\% Disk Time' -SampleInterval 1 -MaxSamples 3
Get-Process | Sort-Object CPU -Descending | Select-Object -First 20 Id,ProcessName,CPU,WorkingSet64,Path
nvidia-smi pmon -c 1
Get-Process -Name MsMpEng,SearchIndexer,OneDrive,python,python3 -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,WorkingSet64,StartTime,Path
Get-CimInstance Win32_StartupCommand | Select-Object Name,Command,Location,User | Sort-Object Name | Select-Object -First 30
Get-ScheduledTask | Where-Object {$_.State -eq 'Running'} | Select-Object TaskName,TaskPath,State | Sort-Object TaskPath,TaskName | Select-Object -First 50
```

### 输入

- Live GPU/process/resource state。
- Recent LP logs:
  - `outputs/logs/data_scaling_lp_no_lm_ums_1k.log`
  - `outputs/logs/data_scaling_lp_frozen_lm_ums_1k.log`
- Interrupted source log:
  - `outputs/logs/data_scaling_frozen_lm_ums_1k_source.log`
- Windows startup/scheduled-task snapshots for obvious automation/background-load signals。

### 预期输出

- 当前是否有训练进程。
- 当前 GPU/CPU/内存/磁盘是否饱和。
- 最近完成训练的速度是否正常。
- 若存在失败命令，记录失败层级和解释边界。

### 停止条件

- 若发现正在运行的训练进程或 GPU 被未知进程占用，停止后续实验启动并进入占用诊断。
- 若发现 CPU/磁盘长时间饱和，停止后续训练启动并先定位系统层占用。
- 若只读诊断命令超时但已有足够证据回答当前问题，记录超时命令，不做破坏性修复。

### 执行后结果

- Current training state:
  - `Get-Process python` returned no active Python training process。
  - GPU0 RTX 3090: `0%` utilization, `0 / 24576 MB`。
  - GPU1 RTX 3090: `0%` utilization, `0 / 24576 MB`。
- System state:
  - CPU total sampled around `20-27%`。
  - Available memory around `55 GB`。
  - Disk total busy time near idle (`0-1%` in sampled windows)。
- Recent completed LP runs:
  - `data_scaling_lp_no_lm_ums_1k.log` ended with `Training completed!` and `EXITCODE 0`。
  - `data_scaling_lp_frozen_lm_ums_1k.log` ended with `Training completed!` and `EXITCODE 0`。
- Background/process observations:
  - Visible high-cumulative CPU processes include Codex, ToDesk, Edge WebView, SunloginClient, proxy/security/vendor utilities。
  - Startup entries include BaiduYunDetect, Docker Desktop, GameViewer, MotionPro/WebVPN, Edge autolaunch, Ollama, OneDrive, QuarkUpdater, SunloginClient, Huorong Sysdiag, Tailscale, Warp。
  - Running scheduled tasks include Clash Verge, Windows multimedia/system/network/cache tasks。
  - None of the visible checks showed a current GPU miner, second Python training job, or disk-saturating automation during this snapshot。

### 指标

- Current training speed:
  - Not applicable: no training is currently running。
- Most recent LP speed:
  - no-LM LP 1k: `3000/3000` in `16:23`, reported `3.05 it/s`。
  - frozen-LM LP 1k: `3000/3000` in `22:02`, reported `2.27 it/s`。
  - Both are within expected range for these LP jobs on this Windows + RTX 3090 setup。
- Earlier interrupted frozen-LM source tail:
  - Tail around steps `6315-6329` slowed to roughly `20-35 s/it`。
  - This was abnormal relative to its preceding segment, but it occurred before the documented reboot/interruption and is not evidence of persistent current slowdown。

### 失败 / 限制

- `nvidia-smi pmon -c 1` failed with “feature is not supported in this configuration”; fallback `nvidia-smi --query-gpu` was sufficient for GPU memory/utilization。
- One live per-process CPU delta sampling command timed out at 10 seconds; fallback cumulative process list, system counters, startup entries, scheduled tasks, and GPU/process checks were sufficient for the current question。
- This is not a full antivirus or forensic scan. It only shows no current GPU/training-like load and no obvious CPU/disk saturation at diagnosis time。

### 当前判断

- 目前没有正在跑的训练，因此当前训练速度不存在“正在变慢”的问题。
- 机器当前资源状态正常：GPU 空闲、CPU/内存/磁盘没有明显瓶颈。
- 最近两个正式 LP 训练速度正常并成功结束。
- 之前 frozen-LM source 尾段确实异常变慢，但更像外部 runtime/reboot 前的局部事件，不应扩大解释为现在电脑持续变慢或训练系统整体异常。

### 下一步

- 不因为本次速度复查重跑 frozen-LM source。
- 若继续启动下一项长训练，启动前再次做 GPU/Python/process preflight，并保留 wrapper `EXITCODE` 日志。
- 若用户仍怀疑病毒或远控占用，应另开一个系统安全检查任务：Windows Defender/Huorong 扫描状态、启动项白名单、网络连接和未知签名进程；该任务与实验训练指标分开记录。

## 2026-06-18 Phase 1 / P1_DATA_SCALING_3K_BCE_SOURCE_RUN 执行前记录

### 计划

1. 只运行 `P1_DATA_SCALING` 的 3k BCE source 单项，作为 1k 之外的数据量扩展起点。
2. 使用现有固定 patient-disjoint split 与 `configs/data_scaling/bce_3k.yaml`。
3. 使用 dedicated wrapper/log 记录 `START`、命令、`EXITCODE`、`END`。
4. 启动前复查 Python/GPU/process、config/split/output 状态。
5. 本步骤不启动 no-LM、frozen-LM、LP、schema sweep、Phase 2 module 或任何 SPD 新变体。

### 命令

```powershell
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime,Path
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
Test-Path outputs\data_scaling\bce_3k\metrics_final.json
Test-Path configs\data_scaling\bce_3k.yaml
Test-Path data\splits\chexpert_train_3k.jsonl
Test-Path data\splits\chexpert_val_fixed.jsonl
(Get-Content data\splits\chexpert_train_3k.jsonl | Measure-Object -Line).Lines
(Get-Content data\splits\chexpert_val_fixed.jsonl | Measure-Object -Line).Lines
scripts\run_data_scaling_bce_3k_source.cmd
Get-Content outputs\logs\data_scaling_bce_3k_source.log -Tail 80
Get-ChildItem outputs\data_scaling\bce_3k -ErrorAction SilentlyContinue
```

### 输入

- Config: `configs/data_scaling/bce_3k.yaml`
- Train split: `data/splits/chexpert_train_3k.jsonl`
- Validation split: `data/splits/chexpert_val_fixed.jsonl`
- Script: `scripts/train_vit_baseline.py`
- Wrapper: `scripts/run_data_scaling_bce_3k_source.cmd`
- Log: `outputs/logs/data_scaling_bce_3k_source.log`

### 预期输出

- Formal output dir: `outputs/data_scaling/bce_3k`
- Expected artifacts if complete:
  - `outputs/data_scaling/bce_3k/best.pt`
  - `outputs/data_scaling/bce_3k/final.pt`
  - `outputs/data_scaling/bce_3k/metrics_final.json`
  - periodic `metrics_step_*.json`
  - periodic `step_*.pt`
- Log should end with `EXITCODE 0` if the run completes normally。

### 停止条件

- Stop before launch if `outputs/data_scaling/bce_3k/metrics_final.json` already exists。
- Stop before launch if another training process is active or GPU is occupied by an unrelated process。
- Stop before launch if config or either split file is absent。
- Stop/diagnose if Python/CUDA/OOM traceback appears or wrapper exits nonzero。
- Stop/diagnose if no log/GPU progress occurs for more than 10 minutes outside startup/eval/checkpoint windows。

### 预检结果

- `outputs/data_scaling/bce_3k/metrics_final.json`: `False` before launch。
- `configs/data_scaling/bce_3k.yaml`: `True`。
- `data/splits/chexpert_train_3k.jsonl`: `True`, line count `3000`。
- `data/splits/chexpert_val_fixed.jsonl`: `True`, line count `1000`。
- No active Python training process was present。
- GPU0/GPU1 both reported `0%` utilization and `0 / 24576 MB` memory used。

### 执行后结果

- Wrapper launched:
  - `scripts/run_data_scaling_bce_3k_source.cmd`
  - wrapper PID at launch: `2988`
  - training Python PID: `3680`
  - log: `outputs/logs/data_scaling_bce_3k_source.log`
- Formal data:
  - Train samples: `3000` from `data/splits/chexpert_train_3k.jsonl`
  - Validation samples: `1000` from `data/splits/chexpert_val_fixed.jsonl`
  - Labels: 14 CheXpert labels
  - Train batches: `93`
  - Validation batches: `32`
- Completion:
  - Training reached `10000 / 10000` steps.
  - Log contains `Training completed!`
  - Wrapper wrote `EXITCODE 0`
  - End timestamp: `2026-06-18 15:25:18` local time
  - Post-run GPU state returned to `0 / 24576 MB` on both GPUs.
  - No active Python training process remained after completion.
- Artifacts:
  - `outputs/data_scaling/bce_3k/best.pt`
  - `outputs/data_scaling/bce_3k/final.pt`
  - `outputs/data_scaling/bce_3k/metrics_final.json`
  - `outputs/data_scaling/bce_3k/metrics_step_{500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000}.json`
  - `outputs/data_scaling/bce_3k/step_{1000,2000,3000,4000,5000,6000,7000,8000,9000,10000}.pt`

### 指标

- Runtime:
  - Wrapper wall time: about `2h20m38s` from `13:04:40` to `15:25:18`.
  - TQDM training line reached `10000/10000` in about `2:19:23`.
- Best by validation loss:
  - step `1000`
  - `val_loss`: `0.34981973469257355`
  - `macro_auc`: `0.770397114448335`
  - `macro_f1`: `0.8799593894298514`
  - `micro_f1`: `0.8534747418364499`
- Final checkpoint / `metrics_final.json`:
  - step `10000`
  - `val_loss`: `1.148102859966457`
  - `macro_auc`: `0.7300493480420224`
  - `macro_f1`: `0.8929581417631126`
  - `micro_f1`: `0.8615685180016746`
- Step trajectory summary:

| step | val_loss | macro_auc | macro_f1 | micro_f1 |
|---:|---:|---:|---:|---:|
| 500 | 0.355769 | 0.761072 | 0.877555 | 0.845102 |
| 1000 | 0.349820 | 0.770397 | 0.879959 | 0.853475 |
| 1500 | 0.486705 | 0.761798 | 0.881079 | 0.853196 |
| 2000 | 0.609000 | 0.746742 | 0.872849 | 0.849567 |
| 2500 | 0.697949 | 0.740025 | 0.886521 | 0.852079 |
| 3000 | 0.830781 | 0.722305 | 0.871831 | 0.841195 |
| 3500 | 0.887063 | 0.736186 | 0.893890 | 0.849846 |
| 4000 | 0.987152 | 0.708376 | 0.882296 | 0.839241 |
| 4500 | 0.967793 | 0.722322 | 0.893419 | 0.858778 |
| 5000 | 0.995740 | 0.740531 | 0.886234 | 0.858778 |
| 5500 | 1.008860 | 0.736346 | 0.887068 | 0.859894 |
| 6000 | 1.054098 | 0.714512 | 0.896446 | 0.862685 |
| 6500 | 1.124487 | 0.697933 | 0.893330 | 0.856824 |
| 7000 | 1.087535 | 0.746037 | 0.883987 | 0.854591 |
| 7500 | 1.142958 | 0.724202 | 0.893026 | 0.856545 |
| 8000 | 1.133767 | 0.728592 | 0.889795 | 0.857103 |
| 8500 | 1.138541 | 0.725065 | 0.894218 | 0.859615 |
| 9000 | 1.126877 | 0.733050 | 0.886802 | 0.861010 |
| 9500 | 1.127579 | 0.732877 | 0.893221 | 0.862127 |
| 10000 | 1.148103 | 0.730049 | 0.892958 | 0.861569 |

### 解释

- `P1_DATA_SCALING_3K_BCE_SOURCE_RUN` is complete and usable as a formal BCE 3k source row.
- The best validation-loss and best macro-AUC checkpoint is step `1000`; the final checkpoint has higher macro-F1/micro-F1 but lower macro-AUC and much higher validation loss.
- This strengthens the need to report data-scaling rows with a clear metric-selection policy:
  - final checkpoint metrics for protocol consistency;
  - best-validation checkpoint metrics for model-selection sensitivity.
- The 3k BCE source row alone does not answer frozen-LM necessity. It only fills the BCE baseline side of the larger data-scaling matrix.

### Runtime / speed diagnosis

- Early training ran in the expected seconds-or-faster regime for this ViT/BCE source run.
- Around steps roughly `5621-5738`, the log showed a temporary slowdown to multi-second steps while GPU memory remained allocated.
- System snapshots during that segment showed:
  - CPU around `20-28%`;
  - available memory around `54 GB`;
  - disk activity low;
  - no second Python training process;
  - GPU0 owned by the training process.
- The run later recovered: GPU utilization snapshots returned to high values, and the job finished normally with `EXITCODE 0`.
- Therefore this was a transient runtime slowdown, not a training failure and not evidence of a persistent virus/miner/GPU contention event in this run.

### 失败 / 限制

- No Python/CUDA/OOM traceback.
- No nonzero wrapper exit.
- One PowerShell process-delta diagnostic command failed due an “empty pipe element” script typo; a corrected command later timed out, so the runtime diagnosis relies on simpler process/GPU/system-counter snapshots plus log progress.
- The 3k BCE row is only one cell of the larger data-scaling matrix; no no-LM/frozen-LM 3k comparison was launched in this step.

### 下一步

- Update `revision_execution_status.*` to mark `P1_DATA_SCALING_3K_BCE_SOURCE_RUN` complete.
- Regenerate status tables.
- Do not automatically launch 3k no-LM/frozen-LM in the same step.
- Next safe decision point: compare current evidence needs before choosing between:
  - 3k no-LM source row, to start matched 3k UMS-vs-BCE comparison;
  - formal schema sweep design, if serialization/schema dependency is still the more urgent paper gap.

## 2026-06-18 Phase 1 / P1_DATA_SCALING_3K_BCE_STATUS_SYNC 执行前记录

### 计划

1. Summarize the completed 3k BCE source run into final-table CSV/Markdown artifacts.
2. Update revision execution status and completion-gap audit so they reflect the new 3k BCE control row.
3. Regenerate status/gap tables without importing torch or launching training.
4. Recheck Python/GPU state after regeneration before reporting runtime-speed status.

### 命令

```powershell
python scripts\summarize_data_scaling_3k_bce_progress.py
python scripts\summarize_revision_execution_status.py
python scripts\summarize_revision_completion_gap_audit.py
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,WorkingSet64,StartTime,Path
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
```

### 输入

- `outputs/data_scaling/bce_3k/metrics_final.json`
- `outputs/data_scaling/bce_3k/metrics_step_*.json`
- `scripts/summarize_data_scaling_3k_bce_progress.py`
- `scripts/summarize_revision_execution_status.py`
- `scripts/summarize_revision_completion_gap_audit.py`

### 预期输出

- `outputs/final_tables/data_scaling_3k_bce_progress.csv`
- `outputs/final_tables/data_scaling_3k_bce_progress.md`
- `outputs/final_tables/data_scaling_3k_bce_trajectory.csv`
- refreshed `outputs/final_tables/revision_execution_status.csv`
- refreshed `outputs/final_tables/revision_execution_status.md`
- refreshed `outputs/final_tables/revision_completion_gap_audit.csv`
- refreshed `outputs/final_tables/revision_completion_gap_audit.md`

### 停止条件

- Stop if any 3k BCE metrics artifact is missing.
- Stop if any summary script exits nonzero.
- Stop if the regenerated status table marks 3k BCE as matched frozen-LM/no-LM evidence.
- Stop if a training process appears during this non-training status-sync step.

### 执行后结果

- `python scripts\summarize_data_scaling_3k_bce_progress.py`
  - exit code `0`
  - wrote summaries to `outputs/final_tables`
- `python scripts\summarize_revision_execution_status.py`
  - exit code `0`
  - wrote `31` rows to `outputs/final_tables`
- `python scripts\summarize_revision_completion_gap_audit.py`
  - exit code `0`
  - wrote `12` rows to `outputs/final_tables`
- Generated / refreshed artifacts:
  - `outputs/final_tables/data_scaling_3k_bce_progress.csv`
  - `outputs/final_tables/data_scaling_3k_bce_progress.md`
  - `outputs/final_tables/data_scaling_3k_bce_trajectory.csv`
  - `outputs/final_tables/revision_execution_status.csv`
  - `outputs/final_tables/revision_execution_status.md`
  - `outputs/final_tables/revision_completion_gap_audit.csv`
  - `outputs/final_tables/revision_completion_gap_audit.md`

### 验证指标 / 抽查

- `revision_execution_status.csv` now contains:
  - task_id: `P1_DATA_SCALING_3K_BCE_SOURCE_RUN`
  - status: `completed_formal_source_run_control_only`
  - evidence_present: `yes`
  - boundary: this is BCE 3k source-control only, not matched no-LM/frozen-LM evidence.
- `revision_completion_gap_audit.csv` now contains:
  - requirement: `P1_DATA_SCALING`
  - current_status: `partial_1k_matched_plus_3k_bce_control`
  - evidence_present_count: `6 / 6`
  - gap: 3k BCE source-control row complete, but matched 3k no-LM/frozen-LM and 10k/30k matrix remain unrun.
- `data_scaling_3k_bce_progress.csv` key rows:

| policy | step | val_loss | macro_auc | macro_f1 | micro_f1 |
|---|---:|---:|---:|---:|---:|
| final_checkpoint | final | 1.148103 | 0.730049 | 0.892958 | 0.861569 |
| best_val_loss | 1000 | 0.349820 | 0.770397 | 0.879959 | 0.853475 |
| best_macro_auc | 1000 | 0.349820 | 0.770397 | 0.879959 | 0.853475 |
| best_macro_f1 | 6000 | 1.054098 | 0.714512 | 0.896446 | 0.862685 |

### Runtime / speed status after sync

- No active `python.exe` training process was visible after the non-training status sync.
- GPU state after sync:
  - GPU0: `0%` utilization, `0 / 24576 MB`
  - GPU1: `0%` utilization, `0 / 24576 MB`
- Therefore the current machine state is idle, and there is no evidence of an ongoing training slowdown at this checkpoint.

### 失败 / 限制

- No summary command failed.
- No training command was launched in this status-sync step.
- This update does not complete matched 3k no-LM/frozen-LM, 10k/30k scaling, or a formal S1/S2/S3 schema sweep.

### 下一步

- For the user’s speed question: answer from verified state that no training is currently running, recent completed LP/BCE jobs finished successfully, and the observed 3k slowdown segment was transient rather than persistent.
- Do not launch another long training job automatically from this status-sync step.
- Next experimental decision should choose explicitly between:
  - matched 3k no-LM/frozen-LM only if a data-scaling curve is required;
  - schema serialization / answerability-aware formalization if paper claims need stronger schema evidence.

## 2026-06-18 Phase 1 / P1_SCHEMA_COMPLEXITY_NO_LM_EXPLICIT_HEADS_DEBUG 执行前记录

### 计划

1. Address the current `P1_SCHEMA_COMPLEXITY_SWEEP` blocker: no-LM S2/S3 currently has only derived diagnostics, not explicit answerability/uncertainty heads.
2. Add optional auxiliary heads to `scripts/train_ums_classifier.py`:
   - default behavior remains the original 4-state classifier;
   - `answerability` head uses `batch["answerable"]` as binary target;
   - `uncertainty` head uses `labels == -1` as binary target.
3. Generate no-LM schema-sweep configs for S1/S2/S3 on the fixed patient-disjoint split.
4. Run only debug entries for S2 and S3 to validate dataloading, losses, checkpoint format, and metric export.
5. Refresh schema-complexity/status artifacts so the plan distinguishes explicit-head debug evidence from formal full training.

### 命令

```powershell
rg -n "schema_auxiliary|auxiliary_targets|answerability_|uncertainty_" scripts\train_ums_classifier.py
python scripts\prepare_no_lm_schema_sweep_configs.py
python scripts\train_ums_classifier.py --config configs\schema_sweep\no_lm_s2_state_answerability.yaml --debug --seed 900124
python scripts\train_ums_classifier.py --config configs\schema_sweep\no_lm_s3_state_uncertainty.yaml --debug --seed 900125
python scripts\summarize_schema_complexity_diagnostics.py
python scripts\summarize_revision_execution_status.py
python scripts\summarize_revision_completion_gap_audit.py
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,WorkingSet64,StartTime,Path
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
```

### 输入

- Training script: `scripts/train_ums_classifier.py`
- Dataset: `data/chexpert_dataset.py`
- Base no-LM config: `configs/ums_classifier_no_llm_12label.yaml`
- Fixed train split: `data/splits/chexpert_train_30k.jsonl`
- Fixed val split: `data/splits/chexpert_val_fixed.jsonl`
- Existing schema diagnostic scripts:
  - `scripts/summarize_schema_complexity_diagnostics.py`
  - `scripts/summarize_revision_execution_status.py`
  - `scripts/summarize_revision_completion_gap_audit.py`

### 预期输出

- Updated `scripts/train_ums_classifier.py` with default-compatible optional auxiliary heads.
- New config prep script: `scripts/prepare_no_lm_schema_sweep_configs.py`
- New configs:
  - `configs/schema_sweep/no_lm_s1_state_only.yaml`
  - `configs/schema_sweep/no_lm_s2_state_answerability.yaml`
  - `configs/schema_sweep/no_lm_s3_state_uncertainty.yaml`
- Debug output dirs:
  - `outputs/schema_sweep/no_lm_s2_state_answerability_seed900124`
  - `outputs/schema_sweep/no_lm_s3_state_uncertainty_seed900125`
- Expected debug artifacts per S2/S3:
  - `metrics_final.json`
  - `best.pt`
  - `final.pt`
  - `step_20.pt`
- Refreshed status/diagnostic outputs under `outputs/final_tables`.

### 停止条件

- Stop if default no-LM classifier behavior would become incompatible with existing checkpoints or `evaluate_no_lm_schema_derivatives.py`.
- Stop if auxiliary heads require a new SPD or frozen-LM variant.
- Stop if debug run tries to execute formal `max_steps=10000` instead of the `--debug` 20-step path.
- Stop if output directories already contain `metrics_final.json`.
- Stop if Python/CUDA/OOM traceback appears or a wrapper/training command exits nonzero.
- Stop if regenerated status tables describe debug outputs as formal schema-sweep performance evidence.

### 执行后结果

- Code / config support:
  - Updated `scripts/train_ums_classifier.py` with optional no-LM schema auxiliary heads.
    - `model(images)` remains backward-compatible and returns 4-state logits.
    - `forward_with_aux()` is used only when `model.schema_auxiliary_targets` is set.
    - Supported auxiliary targets: `answerability`, `uncertainty`.
    - Checkpoints now record `schema_auxiliary_targets`.
  - Updated `scripts/evaluate_no_lm_schema_derivatives.py` so old/new no-LM checkpoints use the same `schema_auxiliary_targets` config when loading.
  - Added `scripts/prepare_no_lm_schema_sweep_configs.py`.
    - Generates no-LM S1/S2/S3 source configs.
    - Generates matching no-LM LP configs.
    - Performs train/val patient-overlap check.
  - Updated schema/status synthesis scripts:
    - `scripts/audit_schema_complexity_support.py`
    - `scripts/design_no_lm_schema_comparator.py`
    - `scripts/summarize_schema_complexity_diagnostics.py`
    - `scripts/summarize_revision_execution_status.py`
    - `scripts/summarize_revision_completion_gap_audit.py`
    - `scripts/summarize_phase4_revision_synthesis.py`

- Config generation:
  - `python scripts\prepare_no_lm_schema_sweep_configs.py`
  - exit code `0`
  - generated/refreshed:
    - `configs/schema_sweep/no_lm_s1_state_only.yaml`
    - `configs/schema_sweep/no_lm_s2_state_answerability.yaml`
    - `configs/schema_sweep/no_lm_s3_state_uncertainty.yaml`
    - `configs/schema_sweep/lp_no_lm_s1_state_only.yaml`
    - `configs/schema_sweep/lp_no_lm_s2_state_answerability.yaml`
    - `configs/schema_sweep/lp_no_lm_s3_state_uncertainty.yaml`
    - `outputs/final_tables/no_lm_schema_sweep_config_manifest.csv`
    - `outputs/final_tables/no_lm_schema_sweep_config_prep.md`

- Debug runs:
  - `python scripts\train_ums_classifier.py --config configs\schema_sweep\no_lm_s2_state_answerability.yaml --debug --seed 900124`
    - exit code `0`
    - `Schema auxiliary targets: ['answerability']`
    - output dir: `outputs/schema_sweep/no_lm_s2_state_answerability_seed900124`
    - wrote `best.pt`, `final.pt`, `step_20.pt`, `metrics_final.json`
  - `python scripts\train_ums_classifier.py --config configs\schema_sweep\no_lm_s3_state_uncertainty.yaml --debug --seed 900125`
    - exit code `0`
    - `Schema auxiliary targets: ['uncertainty']`
    - output dir: `outputs/schema_sweep/no_lm_s3_state_uncertainty_seed900125`
    - wrote `best.pt`, `final.pt`, `step_20.pt`, `metrics_final.json`

- Backward-compatibility check:
  - `python scripts\evaluate_no_lm_schema_derivatives.py --config configs\data_scaling\no_lm_ums_1k.yaml --checkpoint outputs\data_scaling\no_lm_ums_1k\best.pt --split val --device cuda`
  - exit code `0`
  - loaded 1000 validation samples and wrote derived metrics to `outputs/final_tables`
  - warning observed: PyTorch `torch.load(weights_only=False)` future security warning; not a run failure.

- Refreshed tables:
  - `outputs/final_tables/schema_complexity_support_matrix.csv`
  - `outputs/final_tables/schema_complexity_prep.md`
  - `outputs/final_tables/no_lm_schema_comparator_design.csv`
  - `outputs/final_tables/no_lm_schema_comparator_design.md`
  - `outputs/final_tables/schema_complexity_diagnostic_summary.csv`
  - `outputs/final_tables/schema_complexity_diagnostic_summary.md`
  - `outputs/final_tables/revision_execution_status.csv`
  - `outputs/final_tables/revision_execution_status.md`
  - `outputs/final_tables/revision_completion_gap_audit.csv`
  - `outputs/final_tables/revision_completion_gap_audit.md`
  - `outputs/final_tables/llm_necessity.csv`
  - `outputs/final_tables/llm_necessity.md`
  - `outputs/final_tables/module_candidates.csv`
  - `outputs/final_tables/module_candidates.md`
  - `outputs/final_tables/phase4_writing_claim_checklist.md`

### 指标 / 验证

- Syntax:
  - `python -m py_compile` passed for all edited scripts.
- Config safety:
  - `configs/schema_sweep/no_lm_s2_state_answerability.yaml`
    - `data.schema_mode`: `state_answerability`
    - `model.schema_auxiliary_targets`: `['answerability']`
    - no SPD key and no `llm_model_name`.
  - `configs/schema_sweep/no_lm_s3_state_uncertainty.yaml`
    - `data.schema_mode`: `state_uncertainty`
    - `model.schema_auxiliary_targets`: `['uncertainty']`
    - no SPD key and no `llm_model_name`.
- S2 no-LM explicit answerability debug:
  - `val_loss`: `1.3907907860619682`
  - state `macro_auc`: `0.5397981761247068`
  - state `macro_f1`: `0.16296296296296295`
  - state `micro_f1`: `0.5364238410596026`
  - `state_accuracy_all_fields`: `0.6833333333333333`
  - `answerability_support_positive`: `184`
  - `answerability_prevalence`: `0.30666667222976685`
  - `answerability_pred_rate`: `0.25`
  - `answerability_accuracy`: `0.6966666666666667`
  - `answerability_macro_auc`: `0.5114478943108262`
  - `answerability_macro_f1`: `0.1679705178263217`
  - `answerability_micro_f1`: `0.6966666666666667`
- S3 no-LM explicit uncertainty debug:
  - `val_loss`: `0.9862581746918815`
  - state `macro_auc`: `0.6237036055403403`
  - state `macro_f1`: `0.08148148148148147`
  - state `micro_f1`: `0.3973509933774834`
  - `state_accuracy_all_fields`: `0.6783333333333333`
  - `uncertainty_support_positive`: `33`
  - `uncertainty_prevalence`: `0.054999999701976776`
  - `uncertainty_pred_rate`: `0.0`
  - `uncertainty_accuracy`: `0.945`
  - `uncertainty_macro_auc`: `0.5500862011664055`
  - `uncertainty_macro_f1`: `0.0`
  - `uncertainty_micro_f1`: `0.945`
- `schema_complexity_diagnostic_summary.csv` now has 6 rows:
  - `no_lm_answerability_derivative`
  - `no_lm_uncertainty_derivative`
  - `no_lm_s2_explicit_head_debug`
  - `no_lm_s3_explicit_head_debug`
  - `frozen_lm_s2_serializer_debug`
  - `frozen_lm_s3_serializer_debug`
- `revision_execution_status.csv` now contains:
  - task_id: `P1_SCHEMA_COMPLEXITY_NO_LM_EXPLICIT_HEADS_DEBUG`
  - status: `debug_entry_passed_explicit_heads_not_formal`
  - evidence_present: `yes`
- `revision_completion_gap_audit.csv` now contains:
  - requirement: `P1_SCHEMA_COMPLEXITY_SWEEP`
  - current_status: `diagnostic_debug_only_not_formal_sweep`
  - evidence_present_count: `7 / 7`
  - gap: formal S1/S2/S3 source and LP performance rows are not run; no-LM S2/S3 explicit heads are implemented and debugged but not formal evidence.
- Runtime state after all checks:
  - no visible active `python.exe` training process.
  - GPU0: `0%` utilization, `0 / 24576 MB`
  - GPU1: `0%` utilization, `0 / 24576 MB`

### 失败 / 限制

- No Python syntax failures.
- No CUDA/OOM traceback.
- No debug training command exited nonzero.
- A PyTorch `torch.load(weights_only=False)` future warning appeared during derivative-export compatibility validation; it did not prevent checkpoint loading or metric export.
- The S2/S3 no-LM runs are debug-only:
  - 200 training samples
  - 50 validation samples
  - 20 training steps
  - not paper-performance evidence.
- This step does not complete the formal S1/S2/S3 schema sweep.
- This step does not launch frozen-LM formal S1/S2/S3, no-LM formal S1/S2/S3, LP runs, data scaling, or SPD variants.

### 下一步

- If schema complexity remains the priority, the next safe formal step is one source row at a time, for example:
  - formal `no_lm_s2_state_answerability` source, then its matched LP; or
  - formal `no_lm_s3_state_uncertainty` source, then its matched LP.
- Do not launch full S1/S2/S3 or no-LM/frozen-LM matrices at once.
- Keep the paper boundary as:
  - current schema evidence is diagnostic/debug;
  - no-LM explicit-head comparator is now implemented and debug-validated;
  - formal matched schema performance remains unproven.

## 2026-06-18 Phase 1 / P1_SCHEMA_COMPLEXITY_NO_LM_S2_FORMAL_SOURCE_RUN 执行前记录

### 计划

1. Run exactly one formal schema-complexity source row: no-LM S2 `state_answerability`.
2. Use the fixed patient-disjoint schema split and the explicit answerability auxiliary head.
3. Record execution via a dedicated wrapper/log with `START`, command, `EXITCODE`, and `END`.
4. Do not launch S1, S3, frozen-LM, LP, data scaling, random-LM, or SPD variants in this step.
5. After completion, summarize metrics and refresh status/gap tables before any next run decision.

### 命令

```powershell
Test-Path outputs\schema_sweep\no_lm_s2_state_answerability\metrics_final.json
Test-Path configs\schema_sweep\no_lm_s2_state_answerability.yaml
@'
from pathlib import Path
for p in ['data/splits/chexpert_train_30k.jsonl', 'data/splits/chexpert_val_fixed.jsonl']:
    with Path(p).open('rb') as f:
        print(p, sum(1 for _ in f))
'@ | python -
@'
import yaml
from pathlib import Path
cfg=yaml.safe_load(Path('configs/schema_sweep/no_lm_s2_state_answerability.yaml').read_text(encoding='utf-8'))
print(cfg['data'].get('schema_mode'))
print(cfg['model'].get('schema_auxiliary_targets'))
print(cfg['training'].get('output_dir'))
print(cfg['training'].get('max_steps'))
print(cfg['data'].get('num_workers'))
print('llm_model_name' in cfg.get('model', {}))
print('spd' in str(cfg).lower())
'@ | python -
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,WorkingSet64,StartTime,Path
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
scripts\run_schema_no_lm_s2_source.cmd
Get-Content outputs\logs\schema_no_lm_s2_source.log -Tail 80
Get-ChildItem outputs\schema_sweep\no_lm_s2_state_answerability
```

### 输入

- Config: `configs/schema_sweep/no_lm_s2_state_answerability.yaml`
- Training script: `scripts/train_ums_classifier.py`
- Train split: `data/splits/chexpert_train_30k.jsonl`
- Validation split: `data/splits/chexpert_val_fixed.jsonl`
- Wrapper: `scripts/run_schema_no_lm_s2_source.cmd`
- Log: `outputs/logs/schema_no_lm_s2_source.log`

### 预期输出

- Formal output dir: `outputs/schema_sweep/no_lm_s2_state_answerability`
- Expected artifacts if complete:
  - `outputs/schema_sweep/no_lm_s2_state_answerability/best.pt`
  - `outputs/schema_sweep/no_lm_s2_state_answerability/final.pt`
  - `outputs/schema_sweep/no_lm_s2_state_answerability/metrics_final.json`
  - periodic `metrics_step_*.json`
  - periodic `step_*.pt`
- Log should end with `EXITCODE 0` if the run completes normally.

### 预检结果

- `outputs/schema_sweep/no_lm_s2_state_answerability/metrics_final.json`: `False` before launch.
- `configs/schema_sweep/no_lm_s2_state_answerability.yaml`: `True`.
- Train split line count: `29000` records in `data/splits/chexpert_train_30k.jsonl`.
- Validation split line count: `1000` records in `data/splits/chexpert_val_fixed.jsonl`.
- Config safety:
  - `data.schema_mode`: `state_answerability`
  - `model.schema_auxiliary_targets`: `['answerability']`
  - `training.output_dir`: `./outputs/schema_sweep/no_lm_s2_state_answerability`
  - `training.max_steps`: `10000`
  - `data.num_workers`: `0`
  - `model.llm_model_name`: absent
  - SPD string in config: absent
- No active Python training process was visible.
- GPU0/GPU1 both reported `0%` utilization and `0 / 24576 MB` memory used.

### 停止条件

- Stop before launch if `metrics_final.json` already exists.
- Stop before launch if another Python training process is active or either GPU is occupied by unrelated work.
- Stop before launch if config contains SPD or LLM keys.
- Stop before launch if config is not exactly S2 `state_answerability` with `['answerability']` auxiliary target.
- Stop/diagnose if Python/CUDA/OOM traceback appears or wrapper exits nonzero.
- Stop/diagnose if no log/GPU/progress movement occurs for more than 10 minutes outside startup/eval/checkpoint windows.
- Stop after this source run; do not automatically launch LP or S3/frozen-LM matrix in the same step.

### 执行后结果 / 中断记录

- Wrapper launched:
  - `scripts/run_schema_no_lm_s2_source.cmd`
  - wrapper PID at launch: `6104`
  - training Python PID observed: `20784`
  - log: `outputs/logs/schema_no_lm_s2_source.log`
- Formal data:
  - Train split: `data/splits/chexpert_train_30k.jsonl`
  - Actual train records: `29000`
  - Validation split: `data/splits/chexpert_val_fixed.jsonl`
  - Validation records: `1000`
  - Labels: 12 selected CheXpert labels
  - Train batches: `906`
  - Validation batches: `16`
  - Schema mode: `state_answerability`
  - Auxiliary targets: `['answerability']`
- Runtime progress:
  - Training started normally on CUDA.
  - First formal validation/checkpoint at step `500` completed.
  - Subsequent validations/checkpoints completed at steps `1000`, `1500`, and `2000`.
  - `outputs/schema_sweep/no_lm_s2_state_answerability/best.pt` was updated at step `2000`.
  - `outputs/schema_sweep/no_lm_s2_state_answerability/step_2000.pt` exists.
- Nonzero exit:
  - Log tail reaches around step `2015 / 10000`.
  - Wrapper wrote `EXITCODE 1073807364`.
  - Hex code: `0x40010004`.
  - Wrapper end timestamp: `2026-06-18 17:50:40` local time.
  - No `Training completed!` line.
  - No Python traceback, `RuntimeError`, CUDA OOM, or CUDNN error was found in the log.
  - `outputs/schema_sweep/no_lm_s2_state_answerability/metrics_final.json`: absent.
  - `outputs/schema_sweep/no_lm_s2_state_answerability/final.pt`: absent.
  - Post-exit check showed no active Python training process and both GPUs idle.

### 中间指标

| step | val_loss | macro_auc | macro_f1 | micro_f1 | answerability_macro_auc | answerability_macro_f1 | answerability_accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 1.274807 | 0.678166 | 0.284083 | 0.570842 | 0.672188 | 0.263167 | 0.732333 |
| 1000 | 1.268587 | 0.709466 | 0.229976 | 0.474333 | 0.680125 | 0.339372 | 0.741167 |
| 1500 | 1.232073 | 0.718180 | 0.341833 | 0.570842 | 0.696508 | 0.390234 | 0.747250 |
| 2000 | 1.223948 | 0.715198 | 0.320853 | 0.548255 | 0.703864 | 0.364613 | 0.744667 |

- Best validation loss among observed checkpoints: step `2000`, `val_loss=1.223948`.
- Best state macro-AUC among observed checkpoints: step `1500`, `macro_auc=0.718180`.
- Best answerability macro-AUC among observed checkpoints: step `2000`, `answerability_macro_auc=0.703864`.

### 失败层级 / 初步解释

- Failure layer: external/runtime process termination after successful Python training/eval/checkpoint activity.
- Evidence:
  - The script completed multiple validation and checkpoint cycles before exit.
  - No Python traceback or CUDA/OOM error appears in the log.
  - Wrapper captured a Windows-style nonzero exit code after step `2015`.
- Current interpretation:
  - This is not a model/config/metric-export failure.
  - It is most consistent with external Windows process termination or runtime interruption.
  - Because `metrics_final.json` and `final.pt` are absent, the formal source run is not complete.

### 下一步

- Do not restart from scratch and overwrite partial artifacts.
- Add minimal resume support to `scripts/train_ums_classifier.py` using the existing checkpoint fields:
  - `model`
  - `optimizer`
  - `scheduler`
  - `step`
  - `best_val_loss`
- Relaunch this same formal S2 source run from `outputs/schema_sweep/no_lm_s2_state_answerability/step_2000.pt` under a new execution-before record.
- Preserve this nonzero-exit case in the plan as a runtime failure case, not as a completed result.

## 2026-06-18 Phase 1 / P1_SCHEMA_COMPLEXITY_NO_LM_S2_RESUME_SUPPORT_AND_RUN 执行前记录

### 计划

1. Add minimal resume support to `scripts/train_ums_classifier.py`.
2. Preserve default behavior when `--resume` is not provided.
3. Load model, optimizer, scheduler, step, and best validation loss from an existing checkpoint.
4. Resume the interrupted formal no-LM S2 source run from `step_2000.pt`.
5. Use a dedicated resume wrapper/log and keep this as the only active training job.

### 命令

```powershell
python -m py_compile scripts\train_ums_classifier.py
Test-Path outputs\schema_sweep\no_lm_s2_state_answerability\step_2000.pt
Test-Path outputs\schema_sweep\no_lm_s2_state_answerability\metrics_final.json
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,WorkingSet64,StartTime,Path
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
scripts\run_schema_no_lm_s2_source_resume.cmd
Get-Content outputs\logs\schema_no_lm_s2_source_resume.log -Tail 80
Get-ChildItem outputs\schema_sweep\no_lm_s2_state_answerability
```

### 输入

- Script to patch: `scripts/train_ums_classifier.py`
- Config: `configs/schema_sweep/no_lm_s2_state_answerability.yaml`
- Resume checkpoint: `outputs/schema_sweep/no_lm_s2_state_answerability/step_2000.pt`
- Existing partial output dir: `outputs/schema_sweep/no_lm_s2_state_answerability`
- Resume wrapper: `scripts/run_schema_no_lm_s2_source_resume.cmd`
- Resume log: `outputs/logs/schema_no_lm_s2_source_resume.log`

### 预期输出

- `scripts/train_ums_classifier.py` supports `--resume <checkpoint>`.
- Resume launch should report that it resumed from step `2000`.
- Existing output dir should continue toward:
  - `outputs/schema_sweep/no_lm_s2_state_answerability/final.pt`
  - `outputs/schema_sweep/no_lm_s2_state_answerability/metrics_final.json`
  - later periodic `metrics_step_*.json` and `step_*.pt`
- Resume log should end with `EXITCODE 0` only if the full 10k-step run completes.

### 停止条件

- Stop if resume checkpoint is absent.
- Stop if `metrics_final.json` already exists before relaunch.
- Stop if checkpoint/model load mismatch occurs.
- Stop if optimizer/scheduler state cannot be loaded.
- Stop if resume starts at step `0` instead of step `2000`.
- Stop if another training process or unrelated GPU job is active.
- Stop if Python/CUDA/OOM traceback appears or wrapper exits nonzero.
- Stop after relaunch/monitoring this source run; do not launch LP/S3/frozen-LM in this step.

## 2026-06-19 Phase 1 / P1_SCHEMA_COMPLEXITY_NO_LM_S2_RESUME_SUPPORT_AND_RUN 执行中记录

### 当前状态

- Resume support patch verification:
  - `python -m py_compile scripts\train_ums_classifier.py`: passed.
  - `scripts/train_ums_classifier.py` now accepts `--resume`.
  - Resume loading covers `model`, `optimizer`, `scheduler`, `step`, and `best_val_loss`.
  - Progress bar starts with `initial=global_step`, so a valid resume should visibly continue from the checkpoint step.
- Resume wrapper:
  - `scripts/run_schema_no_lm_s2_source_resume.cmd`
  - log: `outputs/logs/schema_no_lm_s2_source_resume.log`
- Relaunch gate:
  - `step_2000.pt` exists: `True`.
  - `metrics_final.json` exists before relaunch: `False`.
  - No active Python process before relaunch.
  - GPUs before relaunch: GPU0 `0% / 0 MB`, GPU1 `0% / 0 MB`.
- Relaunch:
  - wrapper PID at launch: `16856`
  - observed Python PID: `9164`
  - start timestamp in log: `2026-06-19 15:53:37` local time.

### Resume/速度诊断

- Resume log confirms:
  - `Resumed from outputs\schema_sweep\no_lm_s2_state_answerability\step_2000.pt at step 2000`
  - `best_val_loss=1.223948`
  - training did not restart from step 0.
- Runtime after resume:
  - GPU0 active around `71%`, memory around `4057 / 24576 MB`.
  - GPU1 idle.
  - `progress.json` advanced to step `2020` at `2026-06-19 15:55:38`.
- Speed:
  - immediate post-resume training steps are around `2.3-2.6 s/it` after the first warmup steps.
  - This is faster than the interrupted 2026-06-18 tail, which was around `4-6 s/it` after the step-2000 validation pause.
  - Current interpretation: training speed is normal; the apparent 2026-06-18 slowdown was not persistent.
- Failure/stop status:
  - No Python traceback, CUDA/OOM message, or nonzero resume exit observed at this point.
  - Run is still in progress; this is not yet a completed formal source result.

### 下一步

- Continue monitoring the same resumed S2 source run until one of these occurs:
  - formal completion with `final.pt`, `metrics_final.json`, and wrapper `EXITCODE 0`;
  - a new checkpoint/validation milestone that requires a runtime note;
  - nonzero exit or traceback, which must be written back as a failure record.
- Do not launch LP/S3/frozen-LM follow-up runs until this source run is complete or explicitly classified as failed.

## 2026-06-19 Phase 1 / P1_SCHEMA_COMPLEXITY_NO_LM_S3_FORMAL_SOURCE_RUN_GPU1 执行前记录

### 计划

1. Keep the active S2 resumed source run on GPU0 unchanged.
2. Use the idle GPU1 for one independent no-LM S3 source run, because S3 source does not depend on the still-changing S2 `best.pt`.
3. Do not launch S2 LP yet: S2 LP depends on the final stable S2 source `best.pt`, which may still change while S2 source is training.
4. Keep batch size unchanged because current source VRAM use is low enough for one no-LM source run per 24GB GPU.
5. Record GPU assignment, config, inputs, outputs, and stop conditions before launch.

### 命令

```powershell
Get-Content configs\schema_sweep\no_lm_s3_state_uncertainty.yaml -Raw
Test-Path outputs\schema_sweep\no_lm_s3_state_uncertainty
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,CPU,StartTime,WorkingSet64,Path
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
scripts\run_schema_no_lm_s3_source_gpu1.cmd
Get-Content outputs\logs\schema_no_lm_s3_source_gpu1.log -Tail 80
Get-ChildItem outputs\schema_sweep\no_lm_s3_state_uncertainty
```

### 输入

- Config: `configs/schema_sweep/no_lm_s3_state_uncertainty.yaml`
- Train split: `data/splits/chexpert_train_30k.jsonl`
- Validation split: `data/splits/chexpert_val_fixed.jsonl`
- Schema mode: `state_uncertainty`
- Auxiliary targets: `['uncertainty']`
- Device binding: `CUDA_VISIBLE_DEVICES=1`, so config `device: cuda` maps to physical GPU1.
- Wrapper: `scripts/run_schema_no_lm_s3_source_gpu1.cmd`
- Log: `outputs/logs/schema_no_lm_s3_source_gpu1.log`

### 预期输出

- `outputs/schema_sweep/no_lm_s3_state_uncertainty/config_snapshot.json`
- periodic `metrics_step_*.json`
- periodic `step_*.pt`
- `outputs/schema_sweep/no_lm_s3_state_uncertainty/best.pt`
- on completion:
  - `outputs/schema_sweep/no_lm_s3_state_uncertainty/final.pt`
  - `outputs/schema_sweep/no_lm_s3_state_uncertainty/metrics_final.json`
  - wrapper `EXITCODE 0`

### 停止条件

- Stop if GPU1 is not idle or if another unrelated GPU job appears.
- Stop if `outputs/schema_sweep/no_lm_s3_state_uncertainty/metrics_final.json` already exists.
- Stop if the output directory contains partial artifacts that would make this a resume/retry rather than a fresh formal run.
- Stop if config is not `state_uncertainty` or lacks `schema_auxiliary_targets: ['uncertainty']`.
- Stop if CUDA binding fails and the run lands on GPU0.
- Stop if Python/CUDA/OOM traceback appears or wrapper exits nonzero.
- Do not launch S2 LP until S2 source has a final stable `best.pt`.

## 2026-06-19 Phase 1 / dual-GPU no-LM schema source monitoring 执行中记录

### 资源策略

- User approved running one job per GPU when VRAM/power use is not large.
- Current no-LM source footprint is low enough for this:
  - S2 source on physical GPU0: about `4.3 GB / 24 GB`.
  - S3 source on physical GPU1: about `4.0 GB / 24 GB` after startup.
- No batch-size reduction was applied.
  - Reason: both runs fit comfortably on separate 24GB GPUs.
  - Reducing batch size would change throughput and may introduce a comparability concern without being needed for memory safety.

### S2 source milestone after resume

- Active run:
  - Config: `configs/schema_sweep/no_lm_s2_state_answerability.yaml`
  - Output: `outputs/schema_sweep/no_lm_s2_state_answerability`
  - Log: `outputs/logs/schema_no_lm_s2_source_resume.log`
  - Physical GPU: GPU0
- Runtime:
  - `progress.json` reached step `2600` at `2026-06-19 16:16:34`.
  - Training speed in the post-resume stable region is mostly around `1.4-2.0 s/it`, with ordinary jitter.
  - This remains normal and is faster than the interrupted 2026-06-18 tail.
- Step-2500 validation:
  - `metrics_step_2500.json` exists.
  - `best.pt` was updated at step `2500`.

| step | val_loss | macro_auc | macro_f1 | micro_f1 | answerability_macro_auc | answerability_macro_f1 | answerability_accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2500 | 1.216073 | 0.726128 | 0.324118 | 0.599883 | 0.708462 | 0.349338 | 0.745917 |

### S3 source launch on GPU1

- Active run:
  - Config: `configs/schema_sweep/no_lm_s3_state_uncertainty.yaml`
  - Output: `outputs/schema_sweep/no_lm_s3_state_uncertainty`
  - Log: `outputs/logs/schema_no_lm_s3_source_gpu1.log`
  - Wrapper: `scripts/run_schema_no_lm_s3_source_gpu1.cmd`
  - wrapper PID at launch: `8676`
  - observed Python PID: `17180`
  - physical GPU: GPU1 via `CUDA_VISIBLE_DEVICES=1`
- Startup check:
  - S3 output dir was absent before launch.
  - `metrics_final.json` was absent before launch.
  - GPU1 was idle before launch.
  - Startup log confirms `CUDA_VISIBLE_DEVICES=1`.
- Runtime after startup:
  - S3 reached step `70` at `2026-06-19 16:19:50`.
  - GPU1 memory around `4017 / 24576 MB`.
  - Initial step speed is approximately `0.5-1.2 s/it` after warmup jitter.
  - No traceback or nonzero exit observed.

### 下一步

- Continue monitoring both source runs.
- S2 next milestone: step `3000`.
- S3 first validation/checkpoint milestone: step `500`.
- Keep S2 LP deferred until S2 source has a final stable `best.pt`.

## 2026-06-19 Phase 1 / dual-GPU no-LM schema source milestone update

### 运行状态

- S2 source:
  - physical GPU0
  - Python PID: `9164`
  - progress at `2026-06-19 16:30:22`: step `3230 / 10000`
  - GPU0 memory around `4353 / 24576 MB`
  - latest checkpoint: `step_3000.pt`
  - no traceback or nonzero exit observed.
- S3 source:
  - physical GPU1
  - Python PID: `17180`
  - progress at `2026-06-19 16:30:23`: step `1230 / 10000`
  - GPU1 memory around `4313 / 24576 MB`
  - latest checkpoint: `step_1000.pt`
  - no traceback or nonzero exit observed.

### S2 source intermediate metrics

| step | val_loss | macro_auc | macro_f1 | micro_f1 | answerability_macro_auc | answerability_macro_f1 | answerability_accuracy | note |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2500 | 1.216073 | 0.726128 | 0.324118 | 0.599883 | 0.708462 | 0.349338 | 0.745917 | `best.pt` updated |
| 3000 | 1.229224 | 0.723630 | 0.305704 | 0.534174 | 0.706085 | 0.402550 | 0.746500 | val_loss worse than step 2500 |

- Current best S2 checkpoint remains step `2500` by validation loss.
- This is an intermediate trend only; the formal source run is still in progress.

### S3 source intermediate metrics

| step | val_loss | macro_auc | macro_f1 | micro_f1 | uncertainty_macro_auc | uncertainty_macro_f1 | uncertainty_accuracy | uncertainty_pred_rate | note |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 500 | 0.907132 | 0.657774 | 0.286915 | 0.599003 | 0.560164 | 0.000000 | 0.958000 | 0.000000 | first validation |
| 1000 | 0.893668 | 0.709054 | 0.236998 | 0.479906 | 0.583023 | 0.000000 | 0.958000 | 0.000000 | `best.pt` updated |

- S3 validation loss improved from step `500` to step `1000`.
- S3 uncertainty auxiliary head has nontrivial ranking signal (`uncertainty_macro_auc` increased to `0.583023`) but still predicts no positive uncertainty labels at threshold `0.5`.
- This is not yet final evidence for schema contribution; it is an intermediate diagnostic.

### 下一步

- Continue dual-GPU monitoring.
- S2 next validation: step `3500`.
- S3 next validation: step `1500`.
- Keep S2 LP deferred until S2 source completes or a final stable source checkpoint is chosen and documented.

## 2026-06-19 Phase 1 / dual-GPU no-LM schema source milestone update 2

### 运行状态

- S2 source:
  - progress at `2026-06-19 16:46:00`: step `4280 / 10000`
  - latest saved checkpoint: `step_4000.pt`
  - `best.pt` updated at step `3500`
  - GPU0 memory around `4353 / 24576 MB`
  - no traceback or nonzero exit observed.
- S3 source:
  - progress at `2026-06-19 16:46:01`: step `2290 / 10000`
  - latest saved checkpoint: `step_2000.pt`
  - `best.pt` updated at step `2000`
  - GPU1 memory around `4313 / 24576 MB`
  - no traceback or nonzero exit observed.

### S2 source intermediate metrics

| step | val_loss | macro_auc | macro_f1 | micro_f1 | answerability_macro_auc | answerability_macro_f1 | answerability_accuracy | note |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3500 | 1.208898 | 0.727103 | 0.361452 | 0.605163 | 0.712196 | 0.421418 | 0.753333 | `best.pt` updated |
| 4000 | 1.213858 | 0.726495 | 0.360729 | 0.581989 | 0.718996 | 0.419352 | 0.749917 | val_loss worse than step 3500 |

- Current best S2 checkpoint by validation loss: step `3500`.
- S2 answerability macro-AUC continues to improve through step `4000`, even though validation loss is slightly worse than step `3500`.

### S3 source intermediate metrics

| step | val_loss | macro_auc | macro_f1 | micro_f1 | uncertainty_macro_auc | uncertainty_macro_f1 | uncertainty_accuracy | uncertainty_pred_rate | note |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1500 | 0.867696 | 0.728747 | 0.323623 | 0.562042 | 0.648173 | 0.000000 | 0.958000 | 0.000000 | `best.pt` updated |
| 2000 | 0.861437 | 0.740496 | 0.305430 | 0.524201 | 0.642690 | 0.000000 | 0.958000 | 0.000000 | `best.pt` updated |

- Current best S3 checkpoint by validation loss: step `2000`.
- S3 state macro-AUC improved to `0.740496` at step `2000`.
- The uncertainty auxiliary head remains a ranking-only signal so far: AUC is above chance, but threshold `0.5` yields `uncertainty_pred_rate=0`.

### 下一步

- Continue dual-GPU monitoring.
- S2 next validation: step `4500`.
- S3 next validation: step `2500`.
- If either run exits, classify it immediately by wrapper exit code plus log traceback search.

## 2026-06-19 Phase 1 / dual-GPU no-LM schema source milestone update 3

### 运行状态

- S2 source:
  - progress at `2026-06-19 17:01:42`: step `5180 / 10000`
  - latest saved checkpoint: `step_5000.pt`
  - `best.pt` updated at step `4500`
  - no traceback or nonzero exit observed.
- S3 source:
  - progress at `2026-06-19 17:01:39`: step `3200 / 10000`
  - latest saved checkpoint: `step_3000.pt`
  - `best.pt` updated at step `3000`
  - no traceback or nonzero exit observed.
- Resource use remains acceptable:
  - GPU0 memory around `4353 / 24576 MB`
  - GPU1 memory around `4313 / 24576 MB`

### S2 source intermediate metrics

| step | val_loss | macro_auc | macro_f1 | micro_f1 | answerability_macro_auc | answerability_macro_f1 | answerability_accuracy | note |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4500 | 1.189174 | 0.708014 | 0.358887 | 0.606630 | 0.721326 | 0.378031 | 0.752167 | `best.pt` updated |
| 5000 | 1.193032 | 0.728053 | 0.320231 | 0.578175 | 0.722791 | 0.384954 | 0.753167 | val_loss worse than step 4500 |

- Current best S2 checkpoint by validation loss: step `4500`.
- S2 answerability macro-AUC is still improving at step `5000`.

### S3 source intermediate metrics

| step | val_loss | macro_auc | macro_f1 | micro_f1 | uncertainty_macro_auc | uncertainty_macro_f1 | uncertainty_accuracy | uncertainty_pred_rate | note |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2500 | 0.862488 | 0.713282 | 0.294532 | 0.540921 | 0.647439 | 0.000000 | 0.958000 | 0.000000 | val_loss worse than step 2000 |
| 3000 | 0.860726 | 0.708938 | 0.321182 | 0.582576 | 0.666334 | 0.000000 | 0.958000 | 0.000000 | `best.pt` updated |

- Current best S3 checkpoint by validation loss: step `3000`.
- S3 uncertainty AUC improved to `0.666334`, but threshold-positive predictions remain absent at `0.5`.

### 下一步

- Continue dual-GPU monitoring.
- S2 next validation: step `5500`.
- S3 next validation: step `3500`.
- S2 LP remains deferred until S2 source completes or a final checkpoint selection is documented.

## 2026-06-19 Phase 1 / dual-GPU no-LM schema source milestone update 4

### 运行状态

- S2 source:
  - progress at `2026-06-19 17:17:38`: step `6080 / 10000`
  - latest saved checkpoint: `step_6000.pt`
  - `best.pt` updated at step `6000`
  - no traceback or nonzero exit observed.
- S3 source:
  - progress at `2026-06-19 17:17:27`: step `4100 / 10000`
  - latest saved checkpoint: `step_4000.pt`
  - `best.pt` updated at step `4000`
  - no traceback or nonzero exit observed.

### S2 source intermediate metrics

| step | val_loss | macro_auc | macro_f1 | micro_f1 | answerability_macro_auc | answerability_macro_f1 | answerability_accuracy | note |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 5500 | 1.179941 | 0.728769 | 0.357615 | 0.600763 | 0.729005 | 0.403855 | 0.759167 | `best.pt` updated |
| 6000 | 1.172211 | 0.727150 | 0.370581 | 0.580229 | 0.732943 | 0.409644 | 0.760167 | `best.pt` updated |

- Current best S2 checkpoint by validation loss: step `6000`.
- S2 answerability macro-AUC improved to `0.732943`.

### S3 source intermediate metrics

| step | val_loss | macro_auc | macro_f1 | micro_f1 | uncertainty_macro_auc | uncertainty_macro_f1 | uncertainty_accuracy | uncertainty_pred_rate | note |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3500 | 0.846177 | 0.734280 | 0.331926 | 0.569669 | 0.677946 | 0.000000 | 0.958000 | 0.000000 | `best.pt` updated |
| 4000 | 0.838138 | 0.724270 | 0.323532 | 0.557348 | 0.681135 | 0.000000 | 0.958000 | 0.000000 | `best.pt` updated |

- Current best S3 checkpoint by validation loss: step `4000`.
- S3 uncertainty macro-AUC improved to `0.681135`.
- Threshold-positive uncertainty predictions remain absent at `0.5`; this points to calibration/threshold behavior, not absence of ranking signal.

### 下一步

- Continue dual-GPU monitoring.
- S2 next validation: step `6500`.
- S3 next validation: step `4500`.
- Watch for resource drift; no batch-size change unless either GPU approaches memory pressure or thermal throttling symptoms.

## 2026-06-19 Phase 1 / dual-GPU no-LM schema source milestone update 5

### 运行状态

- S2 source:
  - progress at `2026-06-19 17:33:02`: step `7000 / 10000`
  - latest validation file observed: `metrics_step_7000.json`
  - current best checkpoint by validation loss remains step `6000`
  - no traceback or nonzero exit observed.
- S3 source:
  - progress at `2026-06-19 17:33:09`: step `5020 / 10000`
  - latest saved checkpoint: `step_5000.pt`
  - `best.pt` updated at step `5000`
  - no traceback or nonzero exit observed.

### S2 source intermediate metrics

| step | val_loss | macro_auc | macro_f1 | micro_f1 | answerability_macro_auc | answerability_macro_f1 | answerability_accuracy | note |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 6500 | 1.176802 | 0.739406 | 0.349820 | 0.579349 | 0.734478 | 0.426491 | 0.759167 | val_loss worse than step 6000 |
| 7000 | 1.175787 | 0.732631 | 0.383316 | 0.603989 | 0.733797 | 0.433174 | 0.759917 | val_loss worse than step 6000 |

- Current best S2 checkpoint by validation loss: step `6000`.
- S2 answerability macro-F1 improved to `0.433174` at step `7000`, while answerability macro-AUC is essentially stable around `0.734`.

### S3 source intermediate metrics

| step | val_loss | macro_auc | macro_f1 | micro_f1 | uncertainty_macro_auc | uncertainty_macro_f1 | uncertainty_accuracy | uncertainty_pred_rate | note |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4500 | 0.841243 | 0.729619 | 0.320721 | 0.570842 | 0.671526 | 0.000000 | 0.958000 | 0.000000 | val_loss worse than step 4000 |
| 5000 | 0.836532 | 0.726236 | 0.357529 | 0.584629 | 0.683207 | 0.000000 | 0.958000 | 0.000000 | `best.pt` updated |

- Current best S3 checkpoint by validation loss: step `5000`.
- S3 uncertainty macro-AUC improved to `0.683207`.
- S3 threshold-positive uncertainty predictions remain absent at `0.5`.

### 下一步

- Continue dual-GPU monitoring.
- S2 next validation/checkpoint: step `7500` / `8000`.
- S3 next validation/checkpoint: step `5500` / `6000`.
- Keep formal LP jobs deferred until each source run has a final stable source checkpoint.

## 2026-06-19 Phase 1 / GPU scheduling policy and speed check

### 调度原则

- Do not force dual-GPU execution for its own sake.
- It is acceptable to run two jobs concurrently only when:
  - the tasks are independent;
  - each job has its own output directory and log;
  - GPU memory/power/temperature remain within normal bounds;
  - the run has an execution-before record in this plan.
- Dependent jobs must wait for their upstream evidence:
  - LP jobs that depend on `best.pt` should wait until the corresponding source run completes or a final checkpoint is explicitly selected and documented.
  - Current example: S2 LP remains deferred because S2 source `best.pt` may still change before completion.
- Current concurrent jobs are independent source runs:
  - S2 `state_answerability` source on GPU0.
  - S3 `state_uncertainty` source on GPU1.

### 当前速度结论

- Current speed is normal.
- Between roughly `2026-06-19 17:33` and `17:55`:
  - S2 advanced from step `7000` to `8180`, about `1.1-1.2 s/step` including validation/checkpoint overhead.
  - S3 advanced from step `5020` to `6230`, about `1.1-1.2 s/step` including validation/checkpoint overhead.
- GPU memory remains low:
  - GPU0 around `4353 / 24576 MB`.
  - GPU1 around `4313 / 24576 MB`.
- Instantaneous `nvidia-smi` utilization can read `0%` during dataloader gaps or between kernels; progress files, log tails, process CPU time, GPU memory, and power draw confirm both runs are still active.
- No batch-size reduction is needed at this point.

### S2 source intermediate metrics

| step | val_loss | macro_auc | macro_f1 | micro_f1 | answerability_macro_auc | answerability_macro_f1 | answerability_accuracy | note |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 7500 | 1.174702 | 0.734908 | 0.398224 | 0.600763 | 0.736488 | 0.453234 | 0.758000 | val_loss worse than step 6000 |
| 8000 | 1.172650 | 0.721436 | 0.383008 | 0.588736 | 0.739872 | 0.446421 | 0.761083 | val_loss slightly worse than step 6000 |

- Current best S2 checkpoint by validation loss remains step `6000`.
- S2 answerability macro-AUC improved to `0.739872` at step `8000`.

### S3 source intermediate metrics

| step | val_loss | macro_auc | macro_f1 | micro_f1 | uncertainty_macro_auc | uncertainty_macro_f1 | uncertainty_accuracy | uncertainty_pred_rate | note |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 5500 | 0.835497 | 0.732145 | 0.360620 | 0.580815 | 0.680220 | 0.000000 | 0.958000 | 0.000000 | `best.pt` updated |
| 6000 | 0.832226 | 0.728298 | 0.371498 | 0.574362 | 0.694729 | 0.000000 | 0.958000 | 0.000000 | `best.pt` updated |

- Current best S3 checkpoint by validation loss: step `6000`.
- S3 uncertainty macro-AUC improved to `0.694729`.
- Threshold-positive uncertainty predictions remain absent at `0.5`; this is still a calibration/threshold issue rather than lack of ranking signal.

### 下一步

- Continue monitoring current independent source runs.
- Do not launch S2 LP until S2 source completes or an upstream checkpoint is explicitly frozen.
- S2 next validation/checkpoint: step `8500` / `9000`.
- S3 next validation/checkpoint: step `6500` / `7000`.

## 2026-06-22 Phase 1 / dual-GPU source interruption audit

### 结果

- Current host check at `2026-06-22 14:29:39`:
  - No active Python training process.
  - GPU0: `0% / 0 MB`.
  - GPU1: `0% / 0 MB`.
- S2 source:
  - `metrics_final.json`: absent.
  - `final.pt`: absent.
  - latest regular checkpoint: `step_8000.pt`.
  - latest validation metric: `metrics_step_8500.json`.
  - `progress.json` is corrupted/NUL-filled after the interruption.
  - log contains no `EXITCODE`, no `Training completed!`, and no Python/CUDA/OOM traceback.
- S3 source:
  - `metrics_final.json`: absent.
  - `final.pt`: absent.
  - latest regular checkpoint: `step_6000.pt`.
  - latest best checkpoint: `best.pt`, checkpoint metadata `step=6500`.
  - latest validation metric: `metrics_step_6500.json`.
  - `progress.json` is corrupted/NUL-filled after the interruption.
  - log contains no `EXITCODE`, no `Training completed!`, and no Python/CUDA/OOM traceback.

### 中断前最后指标

| run | step | val_loss | macro_auc | macro_f1 | semantic_auc | semantic_f1 | semantic_accuracy | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| S2 answerability source | 8500 | 1.174230 | 0.731732 | 0.403595 | 0.740391 | 0.449859 | 0.759917 | no final output |
| S3 uncertainty source | 6500 | 0.828821 | 0.729690 | 0.374800 | 0.703670 | 0.000000 | 0.958000 | `best.pt` metadata step 6500 |

- For S2, `semantic_*` means answerability metrics.
- For S3, `semantic_*` means uncertainty metrics.
- S3 still has `uncertainty_pred_rate=0.0` at threshold `0.5`; its semantic signal is ranking/AUC evidence so far.

### 失败层级 / 原因

- Failure layer: external/runtime interruption after long-running training and checkpoint activity.
- Evidence:
  - Both processes are gone and GPUs are idle.
  - Neither wrapper wrote an exit code.
  - Neither log contains Python traceback, CUDA OOM, or training completion.
  - Both `progress.json` files were left as NUL-filled content, consistent with interruption during file write.
- Interpretation:
  - This is not a model/config/metric-export failure.
  - It is not evidence that training speed is abnormal.
  - It should be preserved as an external runtime interruption case.

### 下一步

- Resume only the interrupted independent source runs.
- S2 resume checkpoint: `outputs/schema_sweep/no_lm_s2_state_answerability/step_8000.pt`
  - checkpoint metadata: `step=8000`, `best_val_loss=1.1722113192081451`.
- S3 resume checkpoint: `outputs/schema_sweep/no_lm_s3_state_uncertainty/best.pt`
  - checkpoint metadata: `step=6500`, `best_val_loss=0.8288214355707169`.
- Do not launch dependent LP jobs until the corresponding source run has completed.

## 2026-06-22 Phase 1 / P1_SCHEMA_COMPLEXITY_NO_LM_S2_S3_RESUME_TO_FINAL 执行前记录

### 计划

1. Relaunch S2 source from `step_8000.pt` on physical GPU0.
2. Relaunch S3 source from `best.pt` step `6500` on physical GPU1.
3. Keep the two source runs independent; do not launch any LP jobs.
4. Confirm both logs report a nonzero resume step.
5. Monitor until each run produces `final.pt`, `metrics_final.json`, and wrapper `EXITCODE 0`, or until a new failure is observed.

### 命令

```powershell
python -m py_compile scripts\train_ums_classifier.py
Test-Path outputs\schema_sweep\no_lm_s2_state_answerability\step_8000.pt
Test-Path outputs\schema_sweep\no_lm_s3_state_uncertainty\best.pt
Test-Path outputs\schema_sweep\no_lm_s2_state_answerability\metrics_final.json
Test-Path outputs\schema_sweep\no_lm_s3_state_uncertainty\metrics_final.json
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,CPU,StartTime,WorkingSet64,Path
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
scripts\run_schema_no_lm_s2_source_resume3_gpu0.cmd
scripts\run_schema_no_lm_s3_source_resume2_gpu1.cmd
Get-Content outputs\logs\schema_no_lm_s2_source_resume3_gpu0.log -Tail 80
Get-Content outputs\logs\schema_no_lm_s3_source_resume2_gpu1.log -Tail 80
```

### 输入

- S2 config: `configs/schema_sweep/no_lm_s2_state_answerability.yaml`
- S2 resume checkpoint: `outputs/schema_sweep/no_lm_s2_state_answerability/step_8000.pt`
- S2 physical GPU binding: `CUDA_VISIBLE_DEVICES=0`
- S3 config: `configs/schema_sweep/no_lm_s3_state_uncertainty.yaml`
- S3 resume checkpoint: `outputs/schema_sweep/no_lm_s3_state_uncertainty/best.pt`
- S3 physical GPU binding: `CUDA_VISIBLE_DEVICES=1`

### 预期输出

- S2:
  - `outputs/schema_sweep/no_lm_s2_state_answerability/final.pt`
  - `outputs/schema_sweep/no_lm_s2_state_answerability/metrics_final.json`
  - `outputs/logs/schema_no_lm_s2_source_resume3_gpu0.log` ending with `EXITCODE 0`
- S3:
  - `outputs/schema_sweep/no_lm_s3_state_uncertainty/final.pt`
  - `outputs/schema_sweep/no_lm_s3_state_uncertainty/metrics_final.json`
  - `outputs/logs/schema_no_lm_s3_source_resume2_gpu1.log` ending with `EXITCODE 0`

### 停止条件

- Stop if either resume checkpoint is absent.
- Stop if either `metrics_final.json` already exists before relaunch.
- Stop if either resume starts from step `0` or an unexpected step.
- Stop if checkpoint/model/optimizer/scheduler load fails.
- Stop if either run lands on the wrong physical GPU.
- Stop if Python/CUDA/OOM traceback appears or wrapper exits nonzero.
- Stop after these source runs; do not launch S2/S3 LP in this task.

## 2026-06-22 Phase 1 / P1_SCHEMA_COMPLEXITY_NO_LM_S2_S3_RESUME_TO_FINAL 执行中记录

### Relaunch gate

- `python -m py_compile scripts\train_ums_classifier.py`: passed.
- S2 checkpoint exists: `outputs/schema_sweep/no_lm_s2_state_answerability/step_8000.pt`.
- S3 checkpoint exists: `outputs/schema_sweep/no_lm_s3_state_uncertainty/best.pt`.
- S2 `metrics_final.json` before relaunch: absent.
- S3 `metrics_final.json` before relaunch: absent.
- No active Python process before relaunch.
- GPUs before relaunch:
  - GPU0 `0% / 0 MB`
  - GPU1 `0% / 0 MB`

### Relaunch result

- S2 wrapper:
  - `scripts/run_schema_no_lm_s2_source_resume3_gpu0.cmd`
  - wrapper PID: `10436`
  - observed Python PID: `1040`
  - log: `outputs/logs/schema_no_lm_s2_source_resume3_gpu0.log`
  - physical GPU binding: `CUDA_VISIBLE_DEVICES=0`
  - resume confirmed from step `8000`
  - `best_val_loss=1.172211`
  - progress reached step `8010` at `2026-06-22 14:33:38`
- S3 wrapper:
  - `scripts/run_schema_no_lm_s3_source_resume2_gpu1.cmd`
  - wrapper PID: `1468`
  - observed Python PID: `9396`
  - log: `outputs/logs/schema_no_lm_s3_source_resume2_gpu1.log`
  - physical GPU binding: `CUDA_VISIBLE_DEVICES=1`
  - resume confirmed from step `6500`
  - `best_val_loss=0.828821`
  - progress reached step `6510` at `2026-06-22 14:33:38`

### 初始速度/资源

- S2 and S3 both loaded on separate GPUs with about `4057 MB` each.
- Initial post-resume steps were around `3.5-4.0 s/it`.
- This is slower than the 2026-06-19 mid-run stable region, but it is an initial cold-start observation; do not classify as persistent slowdown unless the 10-minute stable window remains slow or progress stops.
- No traceback or nonzero exit observed at this point.

### 下一步

- Continue monitoring the same two source runs.
- Confirm whether speed stabilizes after warmup/cache effects.
- Do not launch dependent LP jobs.

## 2026-06-25 Phase 1 / P1_SCHEMA_COMPLEXITY_NO_LM_S2_S3_RESUME_TO_FINAL 失败记录

### 结果

- Current host check at `2026-06-25 17:42:19`:
  - No active Python training process.
  - GPU0: `0% / 0 MB`.
  - GPU1: `0% / 0 MB`.
- S2 resume3:
  - Log: `outputs/logs/schema_no_lm_s2_source_resume3_gpu0.log`
  - Resumed from step `8000`.
  - Last progress file: step `8050`, timestamp `2026-06-22 14:42:07`.
  - `metrics_final.json`: absent.
  - `final.pt`: absent.
  - Wrapper exit: `EXITCODE 1073807364`.
  - Wrapper end timestamp: `2026-06-22 14:42:14`.
  - No Python traceback, CUDA OOM, or training-completed marker.
- S3 resume2:
  - Log: `outputs/logs/schema_no_lm_s3_source_resume2_gpu1.log`
  - Resumed from step `6500`.
  - Last progress file: step `6550`, timestamp `2026-06-22 14:42:07`.
  - `metrics_final.json`: absent.
  - `final.pt`: absent.
  - Log does not show a completed final output.
  - No Python traceback, CUDA OOM, or training-completed marker.

### 速度/失败解释

- Both source jobs slowed sharply before stopping:
  - S2 step time rose from about `3-6 s/it` to `30+ s/it` near step `8050`.
  - S3 showed the same pattern near step `6550`.
- Both jobs stopped around the same time, while neither reported CUDA/Python errors.
- Interpretation:
  - This is most consistent with external Windows/Codex session termination or process-tree/console closure rather than model failure.
  - It is not a reliable scientific signal and should not be interpreted as schema behavior.
  - It does suggest the launch method was not robust enough for long-running detached training.

### 调度决策

- Do not force dual-GPU execution.
- Stop dual-GPU source resume attempts for now.
- Relaunch only S2 source first, on one GPU, because S2 is closer to completion and is needed before the dependent S2 LP.
- Use a WMI-created detached process rather than a Codex-child `Start-Process` wrapper, to reduce the chance that the process receives a console/session close event when the Codex turn is interrupted.
- Keep S3 source and all LP jobs deferred until S2 source completes or the single-job resume fails again.

## 2026-06-25 Phase 1 / P1_SCHEMA_COMPLEXITY_NO_LM_S2_SINGLE_RESUME_TO_FINAL 执行前记录

### 计划

1. Run only S2 source on physical GPU0.
2. Resume from `outputs/schema_sweep/no_lm_s2_state_answerability/step_8000.pt`.
3. Launch through WMI `Win32_Process.Create` so the process is not tied to the Codex shell process tree.
4. Verify that the log reports resume from step `8000`.
5. Monitor until S2 produces `final.pt`, `metrics_final.json`, and `EXITCODE 0`, or until the same nonzero exit recurs.

### 命令

```powershell
python -m py_compile scripts\train_ums_classifier.py
Test-Path outputs\schema_sweep\no_lm_s2_state_answerability\step_8000.pt
Test-Path outputs\schema_sweep\no_lm_s2_state_answerability\metrics_final.json
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,CPU,StartTime,WorkingSet64,Path
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = 'cmd.exe /c "H:\Xiyao_Wang\021_260129VIVID\scripts\run_schema_no_lm_s2_source_resume4_gpu0.cmd"'
  CurrentDirectory = 'H:\Xiyao_Wang\021_260129VIVID'
}
Get-Content outputs\logs\schema_no_lm_s2_source_resume4_gpu0.log -Tail 80
```

### 输入

- Config: `configs/schema_sweep/no_lm_s2_state_answerability.yaml`
- Resume checkpoint: `outputs/schema_sweep/no_lm_s2_state_answerability/step_8000.pt`
- Physical GPU binding: `CUDA_VISIBLE_DEVICES=0`
- Wrapper: `scripts/run_schema_no_lm_s2_source_resume4_gpu0.cmd`
- Log: `outputs/logs/schema_no_lm_s2_source_resume4_gpu0.log`

### 预期输出

- `outputs/schema_sweep/no_lm_s2_state_answerability/final.pt`
- `outputs/schema_sweep/no_lm_s2_state_answerability/metrics_final.json`
- `outputs/logs/schema_no_lm_s2_source_resume4_gpu0.log` ending with `EXITCODE 0`

### 停止条件

- Stop if checkpoint is absent.
- Stop if S2 `metrics_final.json` already exists before relaunch.
- Stop if another Python/GPU training job is active.
- Stop if resume starts from an unexpected step.
- Stop if checkpoint/model/optimizer/scheduler load fails.
- Stop if the process lands on the wrong physical GPU.
- Stop if Python/CUDA/OOM traceback appears or wrapper exits nonzero.
- Do not launch S3 or any LP job in this step.

## 2026-06-25 Phase 1 / P1_SCHEMA_COMPLEXITY_NO_LM_S2_SINGLE_RESUME_TO_FINAL 执行中记录

### Relaunch result

- Launch method: WMI `Win32_Process.Create`.
- WMI process id: `14652`.
- Observed Python PID: `3980`.
- Wrapper/log:
  - `scripts/run_schema_no_lm_s2_source_resume4_gpu0.cmd`
  - `outputs/logs/schema_no_lm_s2_source_resume4_gpu0.log`
- Resume confirmed:
  - `Resumed from outputs\schema_sweep\no_lm_s2_state_answerability\step_8000.pt at step 8000`
  - `best_val_loss=1.172211`
- GPU binding:
  - GPU0 active, around `4057 / 24576 MB`.
  - GPU1 idle.
- Progress:
  - step `8080` at `2026-06-25 17:45:38`.
  - This passes the previous dual-run failure point at step `8050`.

### 速度诊断

- Single-GPU/WMI resume speed is normal:
  - mostly around `0.7-1.0 s/it` after startup.
- This is much faster and more stable than the failed dual-run resume3, where step time rose to `30+ s/it` before `EXITCODE 1073807364`.
- Interpretation:
  - The model/checkpoint path is runnable.
  - The previous failure is more consistent with launch/session/process interaction under dual concurrent jobs than with model, checkpoint, CUDA memory, or data failure.

### 下一步

- Continue monitoring S2 single run to completion.
- Keep GPU1 idle for now.
- Do not launch S3 or any LP job until S2 source has completed or failed under this single-run setup.

## 2026-06-25 Phase 1 / P1_SCHEMA_COMPLEXITY_NO_LM_S2_SINGLE_RESUME_TO_FINAL 完成记录

### 结果

- Status: completed.
- Launch method: WMI `Win32_Process.Create`.
- Wrapper/log:
  - `scripts/run_schema_no_lm_s2_source_resume4_gpu0.cmd`
  - `outputs/logs/schema_no_lm_s2_source_resume4_gpu0.log`
- Wrapper exit: `EXITCODE 0`.
- End timestamp: `2026-06-25 18:00:08` local time.
- Artifacts:
  - `outputs/schema_sweep/no_lm_s2_state_answerability/final.pt`
  - `outputs/schema_sweep/no_lm_s2_state_answerability/metrics_final.json`
  - `outputs/schema_sweep/no_lm_s2_state_answerability/step_10000.pt`
  - `outputs/schema_sweep/no_lm_s2_state_answerability/metrics_step_10000.json`
- `progress.json`:
  - `status=completed`
  - `global_step=10000`
  - `val_loss=1.239534`
  - `macro_auc=0.715499`
- Runtime:
  - single-GPU resume from step `8000` to `10000` completed in about `15.5` minutes after startup.
  - This confirms the S2 checkpoint/model path is runnable when launched as a single WMI-detached process.

### 尾段指标

| step/file | val_loss | macro_auc | macro_f1 | micro_f1 | answerability_macro_auc | answerability_macro_f1 | answerability_accuracy | answerability_pred_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| step 8500 | 1.182190 | 0.727682 | 0.414095 | 0.621883 | 0.739949 | 0.477299 | 0.759167 | 0.269917 |
| step 9000 | 1.198252 | 0.729971 | 0.428966 | 0.614256 | 0.739797 | 0.476601 | 0.756750 | 0.265000 |
| step 9500 | 1.210836 | 0.728175 | 0.428311 | 0.608976 | 0.738631 | 0.472324 | 0.756250 | 0.258667 |
| step 10000 / final | 1.239534 | 0.715499 | 0.435641 | 0.610443 | 0.735958 | 0.478639 | 0.755083 | 0.261833 |

### 解释

- Final checkpoint is complete, but by validation loss the run appears past its best point.
- Current best checkpoint by validation loss remains earlier:
  - `best.pt` metadata from prior audit: step `6000`, `best_val_loss=1.172211`.
- Tail behavior:
  - answerability macro-F1 improves through the tail and is highest at final among these tail checkpoints.
  - answerability macro-AUC and state macro-AUC are lower at final than the best intermediate checkpoints.
- Use `metrics_final.json` as the completed formal endpoint, and `best.pt`/intermediate metrics for best-checkpoint analysis.

### 下一步

- Keep LP jobs deferred until S3 source is also finalized or explicitly paused.
- Relaunch S3 source as a single WMI-detached job, not in parallel with another training job.

## 2026-06-25 Phase 1 / P1_SCHEMA_COMPLEXITY_NO_LM_S3_SINGLE_RESUME_TO_FINAL 执行前记录

### 计划

1. Run only S3 source as a single training job.
2. Resume from `outputs/schema_sweep/no_lm_s3_state_uncertainty/best.pt`, which has checkpoint metadata `step=6500`.
3. Use WMI `Win32_Process.Create` and physical GPU0, since the S2 single WMI launch completed successfully on GPU0.
4. Do not launch S2 LP, S3 LP, or any other job while S3 source is completing.
5. Monitor until `final.pt`, `metrics_final.json`, and wrapper `EXITCODE 0` exist, or until a nonzero exit recurs.

### 命令

```powershell
python -m py_compile scripts\train_ums_classifier.py
Test-Path outputs\schema_sweep\no_lm_s3_state_uncertainty\best.pt
Test-Path outputs\schema_sweep\no_lm_s3_state_uncertainty\metrics_final.json
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,CPU,StartTime,WorkingSet64,Path
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = 'cmd.exe /c "H:\Xiyao_Wang\021_260129VIVID\scripts\run_schema_no_lm_s3_source_resume3_gpu0.cmd"'
  CurrentDirectory = 'H:\Xiyao_Wang\021_260129VIVID'
}
Get-Content outputs\logs\schema_no_lm_s3_source_resume3_gpu0.log -Tail 80
```

### 输入

- Config: `configs/schema_sweep/no_lm_s3_state_uncertainty.yaml`
- Resume checkpoint: `outputs/schema_sweep/no_lm_s3_state_uncertainty/best.pt`
- Resume checkpoint metadata: step `6500`, `best_val_loss=0.8288214355707169`
- Physical GPU binding: `CUDA_VISIBLE_DEVICES=0`
- Wrapper: `scripts/run_schema_no_lm_s3_source_resume3_gpu0.cmd`
- Log: `outputs/logs/schema_no_lm_s3_source_resume3_gpu0.log`

### 预期输出

- `outputs/schema_sweep/no_lm_s3_state_uncertainty/final.pt`
- `outputs/schema_sweep/no_lm_s3_state_uncertainty/metrics_final.json`
- `outputs/logs/schema_no_lm_s3_source_resume3_gpu0.log` ending with `EXITCODE 0`

### 停止条件

- Stop if checkpoint is absent.
- Stop if S3 `metrics_final.json` already exists before relaunch.
- Stop if another Python/GPU training job is active.
- Stop if resume starts from an unexpected step.
- Stop if checkpoint/model/optimizer/scheduler load fails.
- Stop if the process lands on the wrong physical GPU.
- Stop if Python/CUDA/OOM traceback appears or wrapper exits nonzero.
- Do not launch any LP job in this step.

## 2026-06-25 Phase 1 / P1_SCHEMA_COMPLEXITY_NO_LM_S3_SINGLE_RESUME_TO_FINAL 执行中记录

### Relaunch result

- Launch method: WMI `Win32_Process.Create`.
- WMI process id: `216`.
- Observed Python PID: `20248`.
- Wrapper/log:
  - `scripts/run_schema_no_lm_s3_source_resume3_gpu0.cmd`
  - `outputs/logs/schema_no_lm_s3_source_resume3_gpu0.log`
- Resume confirmed:
  - `Resumed from outputs\schema_sweep\no_lm_s3_state_uncertainty\best.pt at step 6500`
  - `best_val_loss=0.828821`
- GPU binding:
  - GPU0 active, around `4057 / 24576 MB`.
  - GPU1 idle.
- Progress:
  - step `6960` at `2026-06-25 18:04:08`.
  - This passes the previous failed dual-run resume2 point around step `6550`.

### 速度诊断

- Single-GPU/WMI S3 resume speed is normal:
  - about `5.6-5.8 it/s` in the observed stable window.
- This is much faster and more stable than the failed dual-run resume2, where the process slowed sharply before stopping.
- Interpretation:
  - The S3 checkpoint/model path is runnable.
  - Single WMI-detached launch should be the preferred runtime mode for the remaining source completion.

### 下一步

- Continue monitoring S3 single run to completion.
- Keep LP jobs deferred until S3 source has completed or failed under this single-run setup.

## 2026-06-25 Phase 1 / P1_SCHEMA_COMPLEXITY_NO_LM_S3_SINGLE_RESUME_TO_FINAL 完成记录

### 结果

- Status: completed.
- Launch method: WMI `Win32_Process.Create`.
- Wrapper/log:
  - `scripts/run_schema_no_lm_s3_source_resume3_gpu0.cmd`
  - `outputs/logs/schema_no_lm_s3_source_resume3_gpu0.log`
- Wrapper exit: `EXITCODE 0`.
- End timestamp: `2026-06-25 18:13:48` local time.
- Artifacts:
  - `outputs/schema_sweep/no_lm_s3_state_uncertainty/final.pt`
  - `outputs/schema_sweep/no_lm_s3_state_uncertainty/metrics_final.json`
  - `outputs/schema_sweep/no_lm_s3_state_uncertainty/step_10000.pt`
  - `outputs/schema_sweep/no_lm_s3_state_uncertainty/metrics_step_10000.json`
- `progress.json`:
  - `status=completed`
  - `global_step=10000`
  - `val_loss=0.850186`
  - `macro_auc=0.735212`
- Runtime:
  - single-GPU resume from step `6500` to `10000` completed in about `11` minutes after startup.
  - This confirms the S3 checkpoint/model path is runnable when launched as a single WMI-detached process.

### 尾段指标

| step/file | val_loss | macro_auc | macro_f1 | micro_f1 | uncertainty_macro_auc | uncertainty_macro_f1 | uncertainty_accuracy | uncertainty_pred_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| step 7000 | 0.828504 | 0.737744 | 0.393149 | 0.616603 | 0.696613 | 0.000000 | 0.958000 | 0.000000 |
| step 7500 | 0.832087 | 0.734777 | 0.397439 | 0.595483 | 0.698396 | 0.002315 | 0.958083 | 0.000083 |
| step 8000 | 0.833004 | 0.748749 | 0.408948 | 0.608683 | 0.695020 | 0.003672 | 0.957917 | 0.000417 |
| step 8500 | 0.844488 | 0.739847 | 0.410375 | 0.607803 | 0.698899 | 0.003577 | 0.957500 | 0.000833 |
| step 9000 | 0.841864 | 0.744417 | 0.409317 | 0.597243 | 0.692666 | 0.003481 | 0.957167 | 0.001167 |
| step 9500 | 0.850309 | 0.734507 | 0.429443 | 0.613083 | 0.682917 | 0.015424 | 0.955917 | 0.003917 |
| step 10000 / final | 0.850186 | 0.735212 | 0.425318 | 0.612790 | 0.690307 | 0.006719 | 0.957333 | 0.001167 |

### 解释

- Final checkpoint is complete.
- Best checkpoint by validation loss:
  - `best.pt` metadata: step `7000`
  - `best_val_loss=0.8285037167370319`
- S3 uncertainty semantics:
  - uncertainty macro-AUC is clearly above chance and peaks in the tail around step `8500` (`0.698899`).
  - threshold-positive predictions remain extremely rare at threshold `0.5`.
  - This supports a ranking/calibration distinction: the uncertainty head learns ranking signal but is poorly calibrated for default-threshold positive prediction.

### 下一步

- Update schema diagnostics and revision status/gap summaries to include completed S2/S3 formal source evidence.
- Then plan dependent LP jobs from finalized source checkpoints under a new execution-before record.

## 2026-06-25 Runtime / CURRENT_SPEED_AND_SYSTEM_LOAD_DIAGNOSTIC 执行前记录

### 计划

1. 不启动新的训练，不下载模型，不改 batch size。
2. 检查当前是否还有 Python/GPU 训练进程。
3. 检查两张 RTX 3090 的显存、利用率、功率和温度。
4. 检查最近 S2/S3 单卡 WMI 运行日志，判断训练速度是否恢复正常。
5. 检查 CPU/后台任务/Defender 状态，判断是否有明显异常占用或病毒式行为迹象。
6. 明确后续 GPU 调度策略：不刻意双卡；只有不依赖前置结果、显存占用低、且现场资源健康时才并行。

### 命令

```powershell
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu --format=csv,noheader,nounits
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|cmd|powershell|conhost|nvidia' } | Select-Object ProcessId,ParentProcessId,Name,CommandLine
Get-Process | Sort-Object CPU -Descending | Select-Object -First 20 Id,ProcessName,CPU,WorkingSet64,Path
Get-CimInstance Win32_PerfFormattedData_PerfProc_Process | Sort-Object PercentProcessorTime -Descending | Select-Object -First 20 IDProcess,Name,PercentProcessorTime,WorkingSet,IODataBytesPersec
Get-ScheduledTask | Where-Object { $_.State -eq 'Running' } | Select-Object TaskName,TaskPath,State
Get-MpComputerStatus | Select-Object AMServiceEnabled,AntivirusEnabled,RealTimeProtectionEnabled,AntispywareEnabled,IoavProtectionEnabled,QuickScanAge,FullScanAge
Get-Content outputs\schema_sweep\no_lm_s2_state_answerability\progress.json -Raw
Get-Content outputs\schema_sweep\no_lm_s3_state_uncertainty\progress.json -Raw
Get-Content outputs\logs\schema_no_lm_s2_source_resume4_gpu0.log -Head 80
Get-Content outputs\logs\schema_no_lm_s2_source_resume4_gpu0.log -Tail 100
Get-Content outputs\logs\schema_no_lm_s3_source_resume3_gpu0.log -Head 80
Get-Content outputs\logs\schema_no_lm_s3_source_resume3_gpu0.log -Tail 100
Get-Content outputs\schema_sweep\no_lm_s2_state_answerability\metrics_final.json -Raw
Get-Content outputs\schema_sweep\no_lm_s3_state_uncertainty\metrics_final.json -Raw
```

### 输入

- Current runtime state from `nvidia-smi` and Windows process table.
- Completed S2/S3 source logs:
  - `outputs/logs/schema_no_lm_s2_source_resume4_gpu0.log`
  - `outputs/logs/schema_no_lm_s3_source_resume3_gpu0.log`
- Completed S2/S3 progress and metrics:
  - `outputs/schema_sweep/no_lm_s2_state_answerability/progress.json`
  - `outputs/schema_sweep/no_lm_s2_state_answerability/metrics_final.json`
  - `outputs/schema_sweep/no_lm_s3_state_uncertainty/progress.json`
  - `outputs/schema_sweep/no_lm_s3_state_uncertainty/metrics_final.json`

### 预期输出

- Current GPU/process status.
- Recent training throughput estimate.
- Background process and scheduled-task snapshot.
- Go/no-go policy for later parallel GPU usage.

### 停止条件

- Stop if an unknown Python/GPU process is active.
- Stop if GPU memory/power remains high without a known training process.
- Stop if CPU or scheduled-task inspection shows a high-load unknown process.
- Stop if logs show current training still active or failing.
- Do not launch LP/frozen-LM jobs from this diagnostic step.

## 2026-06-25 Runtime / CURRENT_SPEED_AND_SYSTEM_LOAD_DIAGNOSTIC 执行后记录

### 结果

- Current training status: no active training.
- GPU status at check time:
  - GPU0: `0 MiB / 24576 MiB`, `0%` utilization, about `8W`, `35C`.
  - GPU1: `0 MiB / 24576 MiB`, `0%` utilization, about `8W`, `31C`.
  - `nvidia-smi` reported `No running processes found`.
- Process table:
  - No active Python training process.
  - Visible higher cumulative CPU processes were expected desktop/session processes such as Codex, Edge WebView, ToDesk, `verge-mihomo`, Explorer, OneDrive, and related services.
- Scheduled tasks currently running:
  - `Clash Verge (Admin)`
  - `ScheduledDefrag`
  - `SystemSoundsService`
  - `NetworkStateChangeTask`
  - `CacheTask`
- Defender snapshot:
  - `AntivirusEnabled=True`
  - `RealTimeProtectionEnabled=True`
  - `QuickScanAge=0`
  - `FullScanAge=4294967295`, meaning no recent full-scan age is available/reported.
- Virus/malware conclusion boundary:
  - This is not a full antivirus forensic scan.
  - However, the live resource snapshot shows no abnormal GPU consumer, no unknown Python trainer, and no obvious high-load suspicious process.

### 速度判断

- Current speed: not applicable, because no training is currently running.
- Most recent completed single-GPU WMI runs were normal:
  - S2 source resumed from step `8000`, started `2026-06-25 17:43:59`, ended `2026-06-25 18:00:08`, and completed with `EXITCODE 0`.
  - S2 stable tail training speed was about `5.6-5.8 it/s`; validation was about `3.5 batch/s`.
  - S3 source resumed from step `6500`, started `2026-06-25 18:02:31`, ended `2026-06-25 18:13:48`, and completed with `EXITCODE 0`.
  - S3 stable training speed was about `5.6-5.8 it/s`; validation was about `3.5 batch/s`.
- Interpretation:
  - The host/GPU training speed is normal under the single-job WMI launch path.
  - The previous extremely slow dual-run behavior is not reproduced.
  - The earlier failure remains most consistent with Windows/Codex launch/session/process-tree interaction under concurrent jobs, not CUDA OOM, model failure, or sustained GPU underpower.

### 指标确认

- S2 final endpoint:
  - `val_loss=1.239534`
  - state `macro_auc=0.715499`
  - `answerability_macro_auc=0.735958`
  - `answerability_macro_f1=0.478639`
  - `answerability_accuracy=0.755083`
  - `answerability_pred_rate=0.261833`
- S3 final endpoint:
  - `val_loss=0.850186`
  - state `macro_auc=0.735212`
  - `uncertainty_macro_auc=0.690307`
  - `uncertainty_macro_f1=0.006719`
  - `uncertainty_accuracy=0.957333`
  - `uncertainty_pred_rate=0.001167`

### 后续 GPU 调度策略

- Do not force dual-GPU execution.
- Dependent jobs remain sequential:
  - S2 LP depends on completed S2 source/best checkpoint.
  - S3 LP depends on completed S3 source/best checkpoint.
- Independent jobs can use both GPUs only when all of the following are true:
  - inputs do not depend on the other active job's output;
  - each job's expected memory footprint is safely below one GPU's 24GB;
  - both jobs are launched through the more stable WMI-detached path;
  - both have separate logs/output dirs;
  - a live resource check shows no hidden GPU/Python process.
- If a job uses high memory or unstable runtime behavior appears again:
  - run only one GPU job at a time;
  - reduce batch size only after confirming the change will not alter the scientific comparison unfairly;
  - document batch-size changes as runtime mitigation, not as an equivalent reproduction unless effective batch is preserved.

### 下一步

- Update schema diagnostics/status/gap summaries to include the now-completed S2/S3 formal source evidence.
- Then write a new execution-before record for the dependent S2/S3 LP jobs before launching any LP training.

## 2026-06-25 Phase 1 / P1_SCHEMA_COMPLEXITY_SUMMARY_REFRESH_AFTER_SOURCE 执行前记录

### 计划

1. 不启动训练，不下载模型，不继续 SPD 新变体。
2. 更新 schema complexity diagnostic summary，使它包含已完成的 no-LM S2/S3 formal source rows。
3. 更新 revision execution status 和 completion gap audit，明确：
   - S2/S3 no-LM formal source 已完成；
   - S2/S3 LP 仍未跑；
   - frozen-LM formal schema rows 仍未跑；
   - 因此不能宣称完整 formal schema sweep 已完成。
4. 更新 Phase 4 synthesis 读取逻辑：优先使用 formal source evidence，但继续保留 LP/frozen-LM formal missing 的边界。
5. 重新生成相关 CSV/Markdown 表，并抽样检查关键行。
6. 将执行结果写回本计划文档。

### 命令

```powershell
Get-Content scripts\summarize_schema_complexity_diagnostics.py -Encoding UTF8
Get-Content scripts\summarize_revision_execution_status.py -Encoding UTF8
Get-Content scripts\summarize_revision_completion_gap_audit.py -Encoding UTF8
Get-Content scripts\summarize_phase4_revision_synthesis.py -Encoding UTF8
python -m py_compile scripts\summarize_schema_complexity_diagnostics.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py
python scripts\summarize_schema_complexity_diagnostics.py
python scripts\summarize_revision_execution_status.py
python scripts\summarize_revision_completion_gap_audit.py
python scripts\summarize_phase4_revision_synthesis.py
Get-Content outputs\final_tables\schema_complexity_diagnostic_summary.csv -Encoding UTF8
Get-Content outputs\final_tables\revision_execution_status.csv -Encoding UTF8
Get-Content outputs\final_tables\revision_completion_gap_audit.csv -Encoding UTF8
Get-Content outputs\final_tables\llm_necessity.csv -Encoding UTF8
Get-Content outputs\final_tables\phase4_writing_claim_checklist.md -Encoding UTF8
```

### 输入

- Completed no-LM S2 source artifacts:
  - `outputs/schema_sweep/no_lm_s2_state_answerability/metrics_final.json`
  - `outputs/schema_sweep/no_lm_s2_state_answerability/final.pt`
  - `outputs/schema_sweep/no_lm_s2_state_answerability/best.pt`
  - `outputs/logs/schema_no_lm_s2_source_resume4_gpu0.log`
- Completed no-LM S3 source artifacts:
  - `outputs/schema_sweep/no_lm_s3_state_uncertainty/metrics_final.json`
  - `outputs/schema_sweep/no_lm_s3_state_uncertainty/final.pt`
  - `outputs/schema_sweep/no_lm_s3_state_uncertainty/best.pt`
  - `outputs/logs/schema_no_lm_s3_source_resume3_gpu0.log`
- Existing summary inputs under `outputs/final_tables/`.

### 预期输出

- `outputs/final_tables/schema_complexity_diagnostic_summary.csv`
- `outputs/final_tables/schema_complexity_diagnostic_summary.md`
- `outputs/final_tables/revision_execution_status.csv`
- `outputs/final_tables/revision_execution_status.md`
- `outputs/final_tables/revision_completion_gap_audit.csv`
- `outputs/final_tables/revision_completion_gap_audit.md`
- `outputs/final_tables/llm_necessity.csv`
- `outputs/final_tables/llm_necessity.md`
- `outputs/final_tables/module_candidates.csv`
- `outputs/final_tables/module_candidates.md`
- `outputs/final_tables/phase4_writing_claim_checklist.md`

### 停止条件

- Stop if any required S2/S3 source artifact is missing.
- Stop if summary scripts cannot compile.
- Stop if regenerated tables imply formal schema sweep completion.
- Stop if LP rows are marked complete without `lp_no_lm_s2_*` / `lp_no_lm_s3_*` artifacts.
- Stop if any command tries to import model training dependencies or start a GPU job.

## 2026-06-25 Phase 1 / P1_SCHEMA_COMPLEXITY_SUMMARY_REFRESH_AFTER_SOURCE 执行后记录

### 结果

- Status: completed.
- No training was launched.
- GPU post-check:
  - GPU0: `0%`, `0 MiB`, about `9W`, `34C`.
  - GPU1: `0%`, `0 MiB`, about `8W`, `31C`.
- Compile check:
  - `python -m py_compile scripts\summarize_schema_complexity_diagnostics.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py`
  - Exit code: `0`.
- Regenerated outputs:
  - `python scripts\summarize_schema_complexity_diagnostics.py` wrote `8` rows.
  - `python scripts\summarize_revision_execution_status.py` wrote `34` rows.
  - `python scripts\summarize_revision_completion_gap_audit.py` wrote `12` rows.
  - `python scripts\summarize_phase4_revision_synthesis.py` wrote Phase 4 synthesis tables.
- Whitespace check:
  - `git diff --check -- scripts\... vivid_med_revision_execution_plan.md`
  - Exit code: `0`.

### 更新的文件

- Scripts:
  - `scripts/summarize_schema_complexity_diagnostics.py`
  - `scripts/summarize_revision_execution_status.py`
  - `scripts/summarize_revision_completion_gap_audit.py`
  - `scripts/summarize_phase4_revision_synthesis.py`
- Regenerated tables:
  - `outputs/final_tables/schema_complexity_diagnostic_summary.csv`
  - `outputs/final_tables/schema_complexity_diagnostic_summary.md`
  - `outputs/final_tables/revision_execution_status.csv`
  - `outputs/final_tables/revision_execution_status.md`
  - `outputs/final_tables/revision_completion_gap_audit.csv`
  - `outputs/final_tables/revision_completion_gap_audit.md`
  - `outputs/final_tables/llm_necessity.csv`
  - `outputs/final_tables/llm_necessity.md`
  - `outputs/final_tables/module_candidates.csv`
  - `outputs/final_tables/module_candidates.md`
  - `outputs/final_tables/phase4_writing_claim_checklist.md`

### 指标 / 抽查

- `schema_complexity_diagnostic_summary.csv` now includes:
  - `no_lm_s2_explicit_head_formal_source`
    - `answerability_macro_auc=0.735958`
    - `answerability_macro_f1=0.478639`
    - `answerability_accuracy=0.755083`
    - `answerability_pred_rate=0.261833`
    - `state_macro_auc=0.715499`
    - claim boundary: S2 source evidence only; S2 LP and frozen-LM matched rows are missing.
  - `no_lm_s3_explicit_head_formal_source`
    - `uncertainty_macro_auc=0.690307`
    - `uncertainty_macro_f1=0.006719`
    - `uncertainty_accuracy=0.957333`
    - `uncertainty_pred_rate=0.001167`
    - `state_macro_auc=0.735212`
    - claim boundary: S3 source evidence only; S3 LP and frozen-LM matched rows are missing.
- `revision_execution_status.csv` now includes:
  - `P1_SCHEMA_COMPLEXITY_NO_LM_S2_FORMAL_SOURCE_RUN`
  - `P1_SCHEMA_COMPLEXITY_NO_LM_S3_FORMAL_SOURCE_RUN`
  - both marked `completed_formal_source_run`, with evidence present.
- `revision_completion_gap_audit.csv` now marks `P1_SCHEMA_COMPLEXITY_SWEEP` as:
  - `partial_formal_no_lm_s2_s3_source_complete_lp_and_frozen_missing`
  - evidence count `13 / 13`
  - gap: no-LM S2/S3 LP rows, S1 formal rows, and frozen-LM formal source/LP matched rows are not run.
- `llm_necessity.csv` now marks schema complexity as:
  - `partial_formal_source_only`
  - key metrics: derived no-LM answerability AUC `0.607103`; no-LM S2 formal source `0.735958`; no-LM S3 formal source `0.690307`; frozen S2 serializer `passed`.
- `phase4_writing_claim_checklist.md` now requires:
  - "no-LM S2/S3 formal source rows are available; schema sweep is not complete because LP rows and frozen-LM formal matched rows are missing."

### 失败原因 / 边界

- No command failed.
- No training process was active or launched.
- This step does not complete the formal schema sweep.
- Remaining schema-complexity gaps:
  - no-LM S2 LP missing.
  - no-LM S3 LP missing.
  - S1 formal rows missing.
  - frozen-LM formal source/LP rows missing.

### 下一步

- Prepare a new execution-before record for dependent no-LM S2/S3 LP jobs.
- Because both LP jobs depend on completed source checkpoints, run them under explicit LP records rather than treating them as independent source jobs.
- Do not force dual-GPU execution:
  - run one LP first if checkpoint/output dependency or runtime stability is uncertain;
  - only consider parallel LP execution if preflight shows low memory use and both jobs have isolated logs/output dirs.

## 2026-06-25 Phase 1 / P1_SCHEMA_COMPLEXITY_NO_LM_S2_S3_LP_FORMAL_RUNS 执行前记录

### 计划

1. 不启动 SPD 新变体，不下载模型。
2. 只运行 no-LM schema complexity 的 dependent LP jobs：
   - S2 `lp_no_lm_s2_state_answerability`
   - S3 `lp_no_lm_s3_state_uncertainty`
3. 两个 LP 都依赖已完成的 source `best.pt`：
   - S2 LP depends on `outputs/schema_sweep/no_lm_s2_state_answerability/best.pt`
   - S3 LP depends on `outputs/schema_sweep/no_lm_s3_state_uncertainty/best.pt`
4. 因为两个 source checkpoints 均已完成，S2 LP 和 S3 LP 彼此不依赖；若现场 GPU 空闲，则可一张 GPU 跑一个。
5. 使用 WMI-detached launch，避免 Codex shell process-tree 关闭影响训练。
6. 不改 batch size；当前 LP 配置为 frozen-backbone, `batch_size=16`, `gradient_accumulation_steps=2`, `num_workers=0`。
7. 启动后立刻检查 GPU/日志；若任一 LP 显存/功率异常、日志异常慢化、或 nonzero exit，则停止并切回单任务诊断。

### 命令

```powershell
Test-Path outputs\schema_sweep\no_lm_s2_state_answerability\best.pt
Test-Path outputs\schema_sweep\no_lm_s3_state_uncertainty\best.pt
Test-Path outputs\schema_sweep\lp_no_lm_s2_state_answerability\metrics_final.json
Test-Path outputs\schema_sweep\lp_no_lm_s3_state_uncertainty\metrics_final.json
python -m py_compile scripts\train_vit_baseline.py
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader,nounits
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = 'cmd.exe /c "H:\Xiyao_Wang\021_260129VIVID\scripts\run_schema_lp_no_lm_s2_gpu0.cmd"'
  CurrentDirectory = 'H:\Xiyao_Wang\021_260129VIVID'
}
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = 'cmd.exe /c "H:\Xiyao_Wang\021_260129VIVID\scripts\run_schema_lp_no_lm_s3_gpu1.cmd"'
  CurrentDirectory = 'H:\Xiyao_Wang\021_260129VIVID'
}
Get-Content outputs\logs\schema_lp_no_lm_s2_gpu0.log -Tail 80
Get-Content outputs\logs\schema_lp_no_lm_s3_gpu1.log -Tail 80
```

### 输入

- S2 LP config: `configs/schema_sweep/lp_no_lm_s2_state_answerability.yaml`
- S3 LP config: `configs/schema_sweep/lp_no_lm_s3_state_uncertainty.yaml`
- S2 source checkpoint: `outputs/schema_sweep/no_lm_s2_state_answerability/best.pt`
- S3 source checkpoint: `outputs/schema_sweep/no_lm_s3_state_uncertainty/best.pt`
- Physical GPU binding:
  - S2 LP: `CUDA_VISIBLE_DEVICES=0`
  - S3 LP: `CUDA_VISIBLE_DEVICES=1`

### 预期输出

- S2 LP:
  - `outputs/schema_sweep/lp_no_lm_s2_state_answerability/best.pt`
  - `outputs/schema_sweep/lp_no_lm_s2_state_answerability/final.pt`
  - `outputs/schema_sweep/lp_no_lm_s2_state_answerability/metrics_final.json`
  - `outputs/logs/schema_lp_no_lm_s2_gpu0.log` ending with `EXITCODE 0`
- S3 LP:
  - `outputs/schema_sweep/lp_no_lm_s3_state_uncertainty/best.pt`
  - `outputs/schema_sweep/lp_no_lm_s3_state_uncertainty/final.pt`
  - `outputs/schema_sweep/lp_no_lm_s3_state_uncertainty/metrics_final.json`
  - `outputs/logs/schema_lp_no_lm_s3_gpu1.log` ending with `EXITCODE 0`

### 停止条件

- Stop if either source `best.pt` is absent.
- Stop if either LP `metrics_final.json` already exists before launch.
- Stop if any unknown Python/GPU training process is active.
- Stop if `train_vit_baseline.py` fails to compile.
- Stop if either process lands on the wrong physical GPU.
- Stop if either log shows checkpoint load failure, CUDA/OOM traceback, dataloader failure, or nonzero `EXITCODE`.
- Stop if concurrent launch reproduces severe slow-down or external termination; do not retry dual-GPU blindly.

## 2026-06-25 Phase 1 / P1_SCHEMA_COMPLEXITY_NO_LM_S2_S3_LP_FORMAL_RUNS 执行后记录

### 结果

- Status: completed.
- Launch method: WMI `Win32_Process.Create`.
- Parallelism policy:
  - S2 LP and S3 LP were run concurrently because both depended only on already-completed source `best.pt` files and did not depend on each other.
  - No batch-size change was made.
- Runtime health:
  - Peak observed memory during early run: about `1787-1893 MiB` per GPU.
  - Training speed: generally `10-13 it/s`, with expected short slowdowns during periodic validation.
  - Validation speed: about `13-14 batch/s`.
  - No severe slow-down, CUDA/OOM traceback, dataloader error, or external termination.
- Post-run GPU state:
  - GPU0: `0%`, `0 MiB`, about `12W`, `45C`.
  - GPU1: `0%`, `0 MiB`, about `9W`, `34C`.

### S2 LP

- Config: `configs/schema_sweep/lp_no_lm_s2_state_answerability.yaml`
- Source checkpoint: `outputs/schema_sweep/no_lm_s2_state_answerability/best.pt`
- Wrapper/log:
  - `scripts/run_schema_lp_no_lm_s2_gpu0.cmd`
  - `outputs/logs/schema_lp_no_lm_s2_gpu0.log`
- Physical GPU binding: `CUDA_VISIBLE_DEVICES=0`
- Start/end:
  - start `2026-06-25 18:30:29`
  - end `2026-06-25 18:36:24`
- Exit: `EXITCODE 0`
- Loaded source:
  - `Loaded ViT backbone from outputs\schema_sweep\no_lm_s2_state_answerability\best.pt`
  - `Loaded params: 150`
  - `Linear probe mode: froze 150 params, trainable: 10,766`
- Artifacts:
  - `outputs/schema_sweep/lp_no_lm_s2_state_answerability/best.pt`
  - `outputs/schema_sweep/lp_no_lm_s2_state_answerability/final.pt`
  - `outputs/schema_sweep/lp_no_lm_s2_state_answerability/metrics_final.json`
  - `outputs/schema_sweep/lp_no_lm_s2_state_answerability/step_3000.pt`
- Final metrics:
  - `val_loss=0.264131`
  - `macro_auc=0.828866`
  - `macro_f1=0.913637`
  - `micro_f1=0.889199`
- Best checkpoints / metric policy:
  - `best.pt` checkpoint step: `2800`
  - best validation loss row: step `2800`, `val_loss=0.263111`, `macro_auc=0.831034`, `macro_f1=0.915384`, `micro_f1=0.889757`
  - best macro-AUC row: step `400`, `macro_auc=0.847098`, `val_loss=0.274789`

### S3 LP

- Config: `configs/schema_sweep/lp_no_lm_s3_state_uncertainty.yaml`
- Source checkpoint: `outputs/schema_sweep/no_lm_s3_state_uncertainty/best.pt`
- Wrapper/log:
  - `scripts/run_schema_lp_no_lm_s3_gpu1.cmd`
  - `outputs/logs/schema_lp_no_lm_s3_gpu1.log`
- Physical GPU binding: `CUDA_VISIBLE_DEVICES=1`
- Start/end:
  - start `2026-06-25 18:30:32`
  - end `2026-06-25 18:36:13`
- Exit: `EXITCODE 0`
- Loaded source:
  - `Loaded ViT backbone from outputs\schema_sweep\no_lm_s3_state_uncertainty\best.pt`
  - `Loaded params: 150`
  - `Linear probe mode: froze 150 params, trainable: 10,766`
- Artifacts:
  - `outputs/schema_sweep/lp_no_lm_s3_state_uncertainty/best.pt`
  - `outputs/schema_sweep/lp_no_lm_s3_state_uncertainty/final.pt`
  - `outputs/schema_sweep/lp_no_lm_s3_state_uncertainty/metrics_final.json`
  - `outputs/schema_sweep/lp_no_lm_s3_state_uncertainty/step_3000.pt`
- Final metrics:
  - `val_loss=0.267431`
  - `macro_auc=0.815002`
  - `macro_f1=0.912928`
  - `micro_f1=0.888083`
- Best checkpoints / metric policy:
  - `best.pt` checkpoint step: `2800`
  - best validation loss row: step `2800`, `val_loss=0.266493`, `macro_auc=0.819476`, `macro_f1=0.914269`, `micro_f1=0.888920`
  - best macro-AUC row: step `400`, `macro_auc=0.844872`, `val_loss=0.279343`

### 失败原因 / 边界

- No failure occurred.
- This completes no-LM S2/S3 source+LP rows, but not the full matched schema sweep.
- Remaining schema-complexity gaps:
  - frozen-LM S2/S3 formal source+LP rows are still missing.
  - S1 formal row is still missing unless explicitly scoped out.
- The LP runs use 14 binary labels in `train_vit_baseline.py`, while no-LM source metrics use the 12-label UMS state set; comparisons must respect this source-vs-LP metric boundary.

### 下一步

- Refresh schema diagnostic/status/gap/Phase 4 summary tables again so they include no-LM S2/S3 LP completion.
- Then decide whether frozen-LM formal S2/S3 rows are necessary for the paper claim boundary before launching any additional long-running job.

## 2026-06-25 Phase 1 / P1_SCHEMA_COMPLEXITY_SUMMARY_REFRESH_AFTER_LP 执行前记录

### 计划

1. 不启动训练，不下载模型，不继续 SPD 新变体。
2. 更新 schema complexity diagnostic summary，使它包含 no-LM S2/S3 formal LP rows。
3. 更新 revision execution status 和 completion gap audit：
   - no-LM S2/S3 source+LP 已完成；
   - frozen-LM formal source+LP rows 仍未跑；
   - S1 formal row 仍未跑，除非后续明确 scope out。
4. 更新 Phase 4 synthesis 和 writing checklist，避免继续写成 source-only。
5. 重新生成相关 CSV/Markdown 表，并抽样检查关键行。
6. 将执行结果写回本计划文档。

### 命令

```powershell
python -m py_compile scripts\summarize_schema_complexity_diagnostics.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py
python scripts\summarize_schema_complexity_diagnostics.py
python scripts\summarize_revision_execution_status.py
python scripts\summarize_revision_completion_gap_audit.py
python scripts\summarize_phase4_revision_synthesis.py
Select-String -Path outputs\final_tables\schema_complexity_diagnostic_summary.csv -Pattern 'formal_lp|lp_no_lm_s2|lp_no_lm_s3'
Select-String -Path outputs\final_tables\revision_execution_status.csv -Pattern 'P1_SCHEMA_COMPLEXITY_NO_LM_S2_LP_FORMAL_RUN|P1_SCHEMA_COMPLEXITY_NO_LM_S3_LP_FORMAL_RUN'
Select-String -Path outputs\final_tables\revision_completion_gap_audit.csv -Pattern 'P1_SCHEMA_COMPLEXITY_SWEEP|frozen-LM formal'
Select-String -Path outputs\final_tables\llm_necessity.csv -Pattern 'Schema complexity'
Select-String -Path outputs\final_tables\phase4_writing_claim_checklist.md -Pattern 'Schema complexity'
```

### 输入

- Completed S2 LP artifacts:
  - `outputs/schema_sweep/lp_no_lm_s2_state_answerability/metrics_final.json`
  - `outputs/schema_sweep/lp_no_lm_s2_state_answerability/best.pt`
  - `outputs/schema_sweep/lp_no_lm_s2_state_answerability/final.pt`
  - `outputs/logs/schema_lp_no_lm_s2_gpu0.log`
- Completed S3 LP artifacts:
  - `outputs/schema_sweep/lp_no_lm_s3_state_uncertainty/metrics_final.json`
  - `outputs/schema_sweep/lp_no_lm_s3_state_uncertainty/best.pt`
  - `outputs/schema_sweep/lp_no_lm_s3_state_uncertainty/final.pt`
  - `outputs/logs/schema_lp_no_lm_s3_gpu1.log`

### 预期输出

- Updated schema diagnostic table with no-LM S2/S3 formal LP evidence.
- Updated execution status with no-LM S2/S3 LP completion.
- Updated gap audit showing no-LM S2/S3 source+LP complete but frozen-LM formal matched rows still missing.
- Updated Phase 4 synthesis/checklist using source+LP language.

### 停止条件

- Stop if either LP `metrics_final.json` is missing.
- Stop if summary scripts cannot compile.
- Stop if regenerated tables imply frozen-LM formal schema rows are complete.
- Stop if source metrics and LP metrics are merged without labeling the metric boundary.

## 2026-06-25 Phase 1 / P1_SCHEMA_COMPLEXITY_SUMMARY_REFRESH_AFTER_LP 执行后记录

### 结果

- Status: completed.
- No training was launched in this summary-refresh step.
- Compile check:
  - `python -m py_compile scripts\summarize_schema_complexity_diagnostics.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py`
  - Exit code: `0`.
- Regenerated outputs:
  - `python scripts\summarize_schema_complexity_diagnostics.py` wrote `10` rows.
  - `python scripts\summarize_revision_execution_status.py` wrote `36` rows.
  - `python scripts\summarize_revision_completion_gap_audit.py` wrote `12` rows.
  - `python scripts\summarize_phase4_revision_synthesis.py` wrote Phase 4 synthesis tables.
- Whitespace check:
  - `git diff --check -- scripts\... vivid_med_revision_execution_plan.md`
  - Exit code: `0`.

### 更新的证据

- `outputs/final_tables/schema_complexity_diagnostic_summary.csv` now includes:
  - `no_lm_s2_explicit_head_formal_lp`
    - `macro_auc=0.828866`
    - `macro_f1=0.913637`
    - `micro_f1=0.889199`
    - `val_loss=0.264131`
    - `best_loss_step=2800:0.831034`
    - `best_auc_step=400:0.847098`
  - `no_lm_s3_explicit_head_formal_lp`
    - `macro_auc=0.815002`
    - `macro_f1=0.912928`
    - `micro_f1=0.888083`
    - `val_loss=0.267431`
    - `best_loss_step=2800:0.819476`
    - `best_auc_step=400:0.844872`
- `outputs/final_tables/revision_execution_status.csv` now includes:
  - `P1_SCHEMA_COMPLEXITY_NO_LM_S2_LP_FORMAL_RUN`
  - `P1_SCHEMA_COMPLEXITY_NO_LM_S3_LP_FORMAL_RUN`
  - both marked `completed_formal_lp_run`, with evidence present.
- `outputs/final_tables/revision_completion_gap_audit.csv` now marks `P1_SCHEMA_COMPLEXITY_SWEEP` as:
  - `partial_formal_no_lm_s2_s3_source_lp_complete_frozen_missing`
  - evidence count `19 / 19`
  - gap: no-LM S2/S3 formal source+LP rows are complete, but S1 formal rows and frozen-LM formal source/LP matched rows are not run.
- `outputs/final_tables/llm_necessity.csv` now marks schema complexity as:
  - `partial_formal_no_lm_source_lp`
  - key metrics: no-LM S2 source `0.735958` / LP `0.828866`; no-LM S3 source `0.690307` / LP `0.815002`; frozen S2 serializer `passed`.
- `outputs/final_tables/phase4_writing_claim_checklist.md` now requires:
  - "no-LM S2/S3 formal source+LP rows are available; schema sweep is not complete because frozen-LM formal matched rows are missing."

### 失败原因 / 边界

- No command failed.
- No training process was launched in this refresh step.
- no-LM S2/S3 source+LP evidence is now complete for the current schema-complexity branch.
- This still does not complete a matched frozen-LM/no-LM formal schema sweep.
- Remaining schema-complexity gaps:
  - frozen-LM S2 formal source+LP missing.
  - frozen-LM S3 formal source+LP missing.
  - S1 formal row missing unless later scoped out.

### 下一步

- Decision gate before more long-running work:
  - If the paper only needs "UMS/schema contribution and no-LM schema heads are viable", current no-LM source+LP evidence may be enough with frozen-LM rows labeled debug serializer evidence.
  - If the paper needs a matched "frozen-LM vs no-LM under S2/S3 schema complexity" claim, run frozen-LM formal S2/S3 source+LP rows under separate execution-before records.
- Do not launch frozen-LM formal rows as a batch without a preflight and runtime/cost decision.

## 2026-06-25 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_FORMAL_PREFLIGHT 执行前记录

### 计划

1. 不启动训练，不下载模型，不继续 SPD 新变体。
2. 静态检查 frozen-LM S2/S3 formal configs 是否存在、是否指向 Qwen/frozen-LM path、输出目录是否已有 formal artifacts。
3. 检查 frozen-LM debug artifacts 和现有 logs，区分 debug serializer evidence 与 formal performance evidence。
4. 检查当前 GPU/进程状态。
5. 给出是否应启动 frozen-LM formal S2/S3 source+LP 的决策边界。

### 命令

```powershell
Get-Content configs\schema_sweep\frozen_lm_s2_state_answerability.yaml -Encoding UTF8
Get-Content configs\schema_sweep\frozen_lm_s3_state_uncertainty.yaml -Encoding UTF8
Get-ChildItem outputs\schema_sweep -Directory | Where-Object { $_.Name -like 'frozen_lm_s*_state_*' }
Get-ChildItem outputs\schema_sweep\frozen_lm_s2_state_answerability,outputs\schema_sweep\frozen_lm_s3_state_uncertainty -Recurse -File -ErrorAction SilentlyContinue
Get-ChildItem outputs\logs -Filter '*frozen*schema*' -ErrorAction SilentlyContinue
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader,nounits
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,CPU,StartTime,WorkingSet64,Path
```

### 输入

- `configs/schema_sweep/frozen_lm_s2_state_answerability.yaml`
- `configs/schema_sweep/frozen_lm_s3_state_uncertainty.yaml`
- Existing frozen-LM debug artifacts under `outputs/schema_sweep/*seed900122*` and `*seed900123*`.
- Current no-LM source+LP summaries under `outputs/final_tables/`.

### 预期输出

- Frozen-LM formal artifact inventory.
- Frozen-LM formal run risk/cost boundary.
- Recommendation: launch now, defer, or require user decision.

### 停止条件

- Stop if configs are missing or point to an unexpected model/path.
- Stop if formal output dirs already contain `metrics_final.json`.
- Stop if any Python/GPU training process is active.
- Stop if preflight would require importing torch/model or starting training.

## 2026-06-25 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_FORMAL_PREFLIGHT 执行后记录

### 结果

- Status: completed preflight.
- No training was launched.
- No model import was performed.
- Current runtime:
  - GPU0: `0%`, `0 MiB`, about `10W`, `36C`.
  - GPU1: `0%`, `0 MiB`, about `8W`, `32C`.
  - No active Python training process.

### Config inventory

- S2 config exists:
  - `configs/schema_sweep/frozen_lm_s2_state_answerability.yaml`
  - `schema_mode=state_answerability`
  - `llm_model_name=Qwen/Qwen2.5-1.5B-Instruct`
  - `max_steps=10000`
  - `batch_size=4`
  - `gradient_accumulation_steps=8`
  - `effective batch=32`
  - `num_workers=0`
  - `output_dir=./outputs/schema_sweep/frozen_lm_s2_state_answerability`
- S3 config exists:
  - `configs/schema_sweep/frozen_lm_s3_state_uncertainty.yaml`
  - `schema_mode=state_uncertainty`
  - `llm_model_name=Qwen/Qwen2.5-1.5B-Instruct`
  - `max_steps=10000`
  - `batch_size=4`
  - `gradient_accumulation_steps=8`
  - `effective batch=32`
  - `num_workers=0`
  - `output_dir=./outputs/schema_sweep/frozen_lm_s3_state_uncertainty`

### Artifact inventory

- Formal output dirs:
  - `outputs/schema_sweep/frozen_lm_s2_state_answerability`: no files found.
  - `outputs/schema_sweep/frozen_lm_s3_state_uncertainty`: no files found.
- Existing frozen-LM schema evidence is debug-only:
  - `outputs/schema_sweep/frozen_lm_s2_state_answerability_seed900122/checkpoints/{best.pt,step_5.pt,final.pt}`
  - `outputs/schema_sweep/frozen_lm_s3_state_uncertainty_seed900123/checkpoints/{best.pt,step_5.pt,final.pt}`
- No existing formal frozen-LM schema logs were found under `outputs/logs` matching `*frozen*schema*`.
- `scripts/train_cxr.py` supports:
  - `--config`
  - `--resume`
  - `--debug`
  - safety refusal if `metrics_final.json` already exists.

### Cost / risk interpretation

- These formal frozen-LM source jobs are long-running Qwen 1.5B training-time LM jobs.
- Historical cost table shows related frozen-LM 10k-step rows can range from multi-hour to `15.626` observed source wall-clock hours depending on variant/log segment.
- Because frozen-LM formal S2/S3 rows are still missing, a matched frozen-LM/no-LM schema-complexity claim cannot yet be made.
- However, no-LM S2/S3 source+LP evidence is now complete; if the paper only needs UMS/schema/no-LM viability and uses frozen-LM S2/S3 as serializer-debug evidence, these long-running formal frozen-LM rows may be deferred.

### Decision boundary

- Do not launch frozen-LM S2/S3 formal rows as a batch now.
- If matched frozen-LM/no-LM schema complexity is required:
  1. run one frozen-LM source row first, preferably S2 answerability;
  2. use WMI-detached launch and one GPU;
  3. monitor first 500 steps for memory, speed, and checkpoint writing;
  4. only after source completion run its LP;
  5. decide whether S3 is still necessary.
- If paper claim can stay as "no-LM source+LP complete; frozen-LM serializer debug only", stop here and do not spend the long frozen-LM runtime.

### 下一步

- No frozen-LM formal job launched in this step.
- Await claim-scope decision before spending long GPU time on frozen-LM S2/S3 formal source+LP.

## 2026-06-25 Phase 1 / P1_SCHEMA_COMPLEXITY_S1_COVERAGE_AUDIT 执行前记录

### 计划

1. 不启动训练，不下载模型，不继续 SPD 新变体。
2. 审计 S1 `state_only` formal rows 是否已经可以由历史主表的 no-LM/frozen-LM UMS rows 覆盖。
3. 对比 schema-sweep S1 configs 与历史主表 configs：
   - train/val JSONL path
   - selected labels
   - schema mode / target format
   - max steps / effective batch
   - output dirs and artifacts
4. 对比数据文件记录数和内容 fingerprint，判断 `data/splits/*` 与 `data/dataset/processed/*` 是否等价。
5. 输出 S1 coverage audit table，并更新 gap boundary：
   - 如果等价，则把 S1 视为 covered by existing formal rows；
   - 如果不等价，则保留 S1 formal fixed-split gap，不用历史主表冒充。

### 命令

```powershell
Get-Content configs\schema_sweep\no_lm_s1_state_only.yaml -Encoding UTF8
Get-Content configs\schema_sweep\frozen_lm_s1_state_only.yaml -Encoding UTF8
Get-Content configs\ums_classifier_no_llm_12label.yaml -Encoding UTF8
Get-Content configs\ablation_A_ums_12label.yaml -Encoding UTF8
Get-Content configs\lp_ums_classifier_no_llm_12label.yaml -Encoding UTF8
Get-Content configs\lp_A_ums_12label.yaml -Encoding UTF8
python scripts\audit_schema_s1_coverage.py
Get-Content outputs\final_tables\schema_s1_coverage_audit.csv -Encoding UTF8
Get-Content outputs\final_tables\schema_s1_coverage_audit.md -Encoding UTF8
```

### 输入

- Schema-sweep S1 configs:
  - `configs/schema_sweep/no_lm_s1_state_only.yaml`
  - `configs/schema_sweep/frozen_lm_s1_state_only.yaml`
- Historical main-table S1-like configs:
  - `configs/ums_classifier_no_llm_12label.yaml`
  - `configs/ablation_A_ums_12label.yaml`
  - `configs/lp_ums_classifier_no_llm_12label.yaml`
  - `configs/lp_A_ums_12label.yaml`
- Dataset files:
  - `data/splits/chexpert_train_30k.jsonl`
  - `data/splits/chexpert_val_fixed.jsonl`
  - `data/dataset/processed/chexpert_ums_train.jsonl`
  - `data/dataset/processed/chexpert_ums_val.jsonl`
- Existing historical source/LP outputs.

### 预期输出

- `outputs/final_tables/schema_s1_coverage_audit.csv`
- `outputs/final_tables/schema_s1_coverage_audit.md`
- Clear decision on whether S1 is covered or still missing under the formal fixed-split schema-sweep protocol.

### 停止条件

- Stop if any config/data file needed for comparison is missing.
- Stop if datasets are not byte/content equivalent and cannot be safely mapped.
- Stop if the audit would require loading model weights or starting training.
- Stop if historical rows lack source or LP artifacts needed for coverage.

## 2026-06-25 Phase 1 / P1_SCHEMA_COMPLEXITY_S1_COVERAGE_AUDIT 执行后记录

### 结果

- Status: completed audit.
- No training was launched.
- No model weights were loaded.
- Compile/check:
  - `python -m py_compile scripts\audit_schema_s1_coverage.py`
  - Exit code: `0`.
- Generated outputs:
  - `outputs/final_tables/schema_s1_coverage_audit.csv`
  - `outputs/final_tables/schema_s1_coverage_audit.md`
- GPU check:
  - GPU0 and GPU1 remained idle with `0 MiB` memory.

### 指标 / 审计结果

| audit_id | train counts | val counts | train hash match | val hash match | artifact issue | coverage decision |
|---|---:|---:|---|---|---|---|
| `no_lm_s1_historical_coverage` | `29000 vs 29000` | `1000 vs 1000` | `false` | `false` | source+LP artifacts present | `not_covered_fixed_split_mismatch_or_missing` |
| `frozen_lm_s1_historical_coverage` | `29000 vs 29000` | `1000 vs 1000` | `false` | `false` | historical source `final.pt` missing | `not_covered_fixed_split_mismatch_or_missing` |

### 解释

- Historical S1-like main-table configs and schema-sweep S1 configs match on core config shape:
  - labels
  - max steps
  - effective batch
  - schema mode / target format
- But they do not match on exact dataset content:
  - schema-sweep uses `data/splits/chexpert_train_30k.jsonl` and `data/splits/chexpert_val_fixed.jsonl`.
  - historical main-table rows use `data/dataset/processed/chexpert_ums_train.jsonl` and `data/dataset/processed/chexpert_ums_val.jsonl`.
  - counts match, but content hashes do not.
- Therefore historical main-table S1-like rows cannot be used as exact fixed-split formal schema-sweep S1 evidence.

### 失败原因 / 边界

- No command failed.
- This audit does not make S1 complete.
- S1 remains a formal fixed-split schema-sweep gap unless explicitly scoped out or rerun on `data/splits/*`.
- Historical S1 rows remain useful for main P0 controlled-results evidence, but not as exact P1_SCHEMA_COMPLEXITY_SWEEP S1 proof.

### 下一步

- Update execution status / gap audit / Phase 4 checklist to include `schema_s1_coverage_audit`.
- The schema-sweep gap should now say:
  - no-LM S2/S3 source+LP complete;
  - S1 fixed-split formal row still not covered by historical runs;
  - frozen-LM S2/S3 formal matched rows still missing.

## 2026-06-25 Phase 1 / P1_SCHEMA_COMPLEXITY_S1_AUDIT_INTEGRATION 执行前记录

### 计划

1. 不启动训练，不下载模型，不继续 SPD 新变体。
2. 更新 `summarize_revision_execution_status.py`，加入 S1 coverage audit row。
3. 更新 `summarize_revision_completion_gap_audit.py`，把 S1 audit 纳入 `P1_SCHEMA_COMPLEXITY_SWEEP` evidence 和 gap。
4. 更新 `summarize_phase4_revision_synthesis.py`，让 schema complexity checklist 明确 S1 fixed-split row 仍未覆盖。
5. 重新生成 status/gap/Phase 4 表并抽查关键行。
6. 写回执行结果。

### 命令

```powershell
python -m py_compile scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py
python scripts\summarize_revision_execution_status.py
python scripts\summarize_revision_completion_gap_audit.py
python scripts\summarize_phase4_revision_synthesis.py
Select-String -Path outputs\final_tables\revision_execution_status.csv -Pattern 'P1_SCHEMA_COMPLEXITY_S1_COVERAGE_AUDIT'
Select-String -Path outputs\final_tables\revision_completion_gap_audit.csv -Pattern 'P1_SCHEMA_COMPLEXITY_SWEEP|S1 fixed-split'
Select-String -Path outputs\final_tables\phase4_writing_claim_checklist.md -Pattern 'Schema complexity|S1 fixed-split'
```

### 输入

- `outputs/final_tables/schema_s1_coverage_audit.csv`
- `outputs/final_tables/schema_s1_coverage_audit.md`
- Existing status/gap/Phase 4 summary scripts.

### 预期输出

- Updated `outputs/final_tables/revision_execution_status.{csv,md}`
- Updated `outputs/final_tables/revision_completion_gap_audit.{csv,md}`
- Updated `outputs/final_tables/llm_necessity.{csv,md}`
- Updated `outputs/final_tables/module_candidates.{csv,md}`
- Updated `outputs/final_tables/phase4_writing_claim_checklist.md`

### 停止条件

- Stop if S1 audit outputs are missing.
- Stop if summaries imply S1 formal fixed-split row is covered by historical runs.
- Stop if summaries hide frozen-LM formal matched rows as missing.

## 2026-06-25 Phase 1 / P1_SCHEMA_COMPLEXITY_S1_AUDIT_INTEGRATION 执行后记录

### 结果

- 未启动训练。
- 未下载模型。
- 未继续 SPD 新变体。
- S1 coverage audit 已整合进执行状态、completion gap audit 和 Phase 4 写作检查表。
- `revision_execution_status.csv` 从 36 行更新为 37 行，新增 `P1_SCHEMA_COMPLEXITY_S1_COVERAGE_AUDIT`。
- `revision_completion_gap_audit.csv` 保持 12 行，但 `P1_SCHEMA_COMPLEXITY_SWEEP` 的 gap 现在明确包含：
  - no-LM S2/S3 formal source+LP rows complete；
  - S1 fixed-split formal rows are not covered by historical main-table rows；
  - frozen-LM formal source/LP matched rows are not run。
- `phase4_writing_claim_checklist.md` 现在明确写作边界：
  - 可以说 no-LM S2/S3 formal source+LP rows are available；
  - 不能说 matched frozen-LM/no-LM S1/S2/S3 formal source-plus-LP comparison is complete。

### 命令与输出

```powershell
python -m py_compile scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py
```

- Exit code: `0`

```powershell
python scripts\summarize_revision_execution_status.py
```

- Output: `Wrote 37 rows to outputs/final_tables`

```powershell
python scripts\summarize_revision_completion_gap_audit.py
```

- Output: `Wrote 12 rows to outputs/final_tables`

```powershell
python scripts\summarize_phase4_revision_synthesis.py
```

- Output: `Wrote Phase 4 synthesis tables to outputs/final_tables`

```powershell
git diff --check -- scripts/audit_schema_s1_coverage.py scripts/summarize_revision_execution_status.py scripts/summarize_revision_completion_gap_audit.py scripts/summarize_phase4_revision_synthesis.py vivid_med_revision_execution_plan.md
```

- Exit code: `0`

### 指标 / 验证点

| check | result |
|---|---|
| `revision_execution_status.csv` contains `P1_SCHEMA_COMPLEXITY_S1_COVERAGE_AUDIT` | pass |
| `revision_completion_gap_audit.csv` contains `S1 fixed-split` boundary | pass |
| `phase4_writing_claim_checklist.md` contains S1 fixed-split missing language | pass |
| `llm_necessity.csv` contains `S1 exact historical coverage missing` | pass |
| `git diff --check` | pass |

### 失败原因 / 边界

- No command failed.
- This integration does not close the S1 formal fixed-split row.
- It prevents accidental overclaiming that historical S1-like main-table runs are exact schema-sweep S1 evidence.
- It also preserves the boundary that frozen-LM formal matched rows remain missing.

### 下一步

- Short, low-risk next step: decide whether to run no-LM S1 fixed-split source+LP to close the no-LM side of `P1_SCHEMA_COMPLEXITY_SWEEP`.
- Long-running/gated next step: only after explicit go/no-go, start at most one frozen-LM formal row first, preferably S2, one GPU, WMI-detached, monitor first 500 steps before queueing more.
- Do not batch frozen-LM S2/S3/S1 formal runs yet, because prior runtime evidence suggests Qwen 10k source rows may take hours and should be gated by early speed/VRAM checks.

## 2026-06-25 Phase 1 / P1_SCHEMA_COMPLEXITY_NO_LM_S1_FIXED_SOURCE 执行前记录

### 计划

1. 只补跑 no-LM S1 fixed-split source row，不启动 frozen-LM，不下载 LLM，不继续 SPD 新变体。
2. 使用 `configs/schema_sweep/no_lm_s1_state_only.yaml`，保持 fixed split:
   - train: `data/splits/chexpert_train_30k.jsonl`
   - val: `data/splits/chexpert_val_fixed.jsonl`
3. 单卡 GPU0 运行，GPU1 保持空闲，避免和后续 frozen-LM gated run 抢资源。
4. 用独立日志 `outputs/logs/schema_no_lm_s1_source_gpu0.log` 记录启动命令、退出码和时间。
5. 启动后检查 GPU/进程/日志，确认速度和显存正常。

### 命令

```powershell
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,power.limit --format=csv,noheader,nounits
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|python.exe' } | Select-Object ProcessId,Name,CommandLine
Test-Path outputs\schema_sweep\no_lm_s1_state_only\metrics_final.json
wmic process call create "cmd.exe /c H:\Xiyao_Wang\021_260129VIVID\scripts\run_schema_no_lm_s1_source_gpu0.cmd"
Get-Content outputs\logs\schema_no_lm_s1_source_gpu0.log -Tail 40
```

### 输入

- `configs/schema_sweep/no_lm_s1_state_only.yaml`
- `data/splits/chexpert_train_30k.jsonl`
- `data/splits/chexpert_val_fixed.jsonl`
- `scripts/train_ums_classifier.py`

### 预期输出

- `outputs/schema_sweep/no_lm_s1_state_only/config_snapshot.json`
- `outputs/schema_sweep/no_lm_s1_state_only/progress.json`
- `outputs/schema_sweep/no_lm_s1_state_only/metrics_step_*.json`
- `outputs/schema_sweep/no_lm_s1_state_only/best.pt`
- `outputs/schema_sweep/no_lm_s1_state_only/final.pt`
- `outputs/schema_sweep/no_lm_s1_state_only/metrics_final.json`
- `outputs/logs/schema_no_lm_s1_source_gpu0.log`

### 停止条件

- Stop if `metrics_final.json` already exists before launch.
- Stop if the launch command does not create a Python process or log file.
- Stop if the log reports a Python exception, non-zero `EXITCODE`, or missing dataset path.
- Stop if GPU memory/power/utilization indicates an abnormal stall during early monitoring.
- Stop after launching and early monitoring; do not launch LP or frozen-LM until source completion is confirmed.

## 2026-06-25 Phase 1 / P1_SCHEMA_COMPLEXITY_NO_LM_S1_FIXED_SOURCE 执行后记录

### 结果

- no-LM S1 fixed-split source row completed successfully.
- No frozen-LM run was launched.
- No LLM/model download was triggered.
- No SPD new variant was launched.
- Wrapper:
  - `scripts/run_schema_no_lm_s1_source_gpu0.cmd`
- Log:
  - `outputs/logs/schema_no_lm_s1_source_gpu0.log`
- Output directory:
  - `outputs/schema_sweep/no_lm_s1_state_only`
- Exit code:
  - `0`
- Runtime:
  - start: `2026-06-25 18:51:28`
  - end: `2026-06-25 19:22:52`
  - training log speed at completion: `10000/10000 [31:11, 5.34it/s]`
- GPU:
  - early stable usage: GPU0 about `4.0-4.3 GB`, `~250-267 W`, `~46-59%` utilization.
  - GPU1 remained idle.
  - after completion both GPUs returned to `0 MiB` memory usage.

### 指标

| metric point | step | val_loss | macro_auc | macro_f1 | micro_f1 |
|---|---:|---:|---:|---:|---:|
| final | 10000 | 0.688880 | 0.713327 | 0.428571 | 0.617776 |
| best val_loss row | 7500 | 0.674552 | 0.735068 | 0.395167 | 0.603109 |
| best macro_auc row | 5500 | 0.680255 | 0.744081 | 0.364237 | 0.586682 |

Additional final state metrics:

| metric | value |
|---|---:|
| `state_accuracy_all_fields` | 0.744083 |
| `state_accuracy_answerable_fields` | 0.481373 |

### 失败原因 / 边界

- No command failed.
- No abnormal GPU or process contention was observed.
- This completes only the no-LM S1 source endpoint.
- It does not complete no-LM S1 LP.
- It does not complete frozen-LM matched S1/S2/S3 formal rows.
- Final macro-AUC is lower than the best-AUC mid-run row, so downstream summary should preserve final, best-loss, and best-AUC distinctions.

### 下一步

- Update schema complexity diagnostic summary and completion gap tables to include the formal no-LM S1 fixed-split source row.
- Then run dependent no-LM S1 LP from `outputs/schema_sweep/no_lm_s1_state_only/best.pt`, one GPU, because LP depends on source completion.
- Keep frozen-LM formal rows gated; do not launch frozen-LM in parallel with the LP unless explicitly deciding to start the gated frozen-LM test.

## 2026-06-25 Phase 1 / P1_SCHEMA_COMPLEXITY_NO_LM_S1_SOURCE_INTEGRATION 执行前记录

### 计划

1. 不启动训练，不下载模型，不继续 SPD 新变体。
2. 把 `outputs/schema_sweep/no_lm_s1_state_only/metrics_final.json` 纳入 schema complexity summary。
3. 更新 execution status / completion gap audit / Phase 4 synthesis，使边界从 “S1 fixed-split source missing” 改成 “S1 source complete, S1 LP missing”。
4. 重新生成所有相关 summary 输出并抽查关键行。
5. 写回执行结果、指标、失败状态和下一步。

### 命令

```powershell
python -m py_compile scripts\summarize_schema_complexity_diagnostics.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py
python scripts\summarize_schema_complexity_diagnostics.py
python scripts\summarize_revision_execution_status.py
python scripts\summarize_revision_completion_gap_audit.py
python scripts\summarize_phase4_revision_synthesis.py
Select-String -Path outputs\final_tables\schema_complexity_diagnostic_summary.csv -Pattern 'no_lm_s1_state_only_formal_source'
Select-String -Path outputs\final_tables\revision_completion_gap_audit.csv -Pattern 'S1 LP'
Select-String -Path outputs\final_tables\phase4_writing_claim_checklist.md -Pattern 'S1 LP'
```

### 输入

- `outputs/schema_sweep/no_lm_s1_state_only/metrics_final.json`
- `outputs/schema_sweep/no_lm_s1_state_only/metrics_step_*.json`
- `outputs/logs/schema_no_lm_s1_source_gpu0.log`
- Existing summary scripts and final-table inputs.

### 预期输出

- Updated `outputs/final_tables/schema_complexity_diagnostic_summary.{csv,md}`
- Updated `outputs/final_tables/revision_execution_status.{csv,md}`
- Updated `outputs/final_tables/revision_completion_gap_audit.{csv,md}`
- Updated `outputs/final_tables/llm_necessity.{csv,md}`
- Updated `outputs/final_tables/module_candidates.{csv,md}`
- Updated `outputs/final_tables/phase4_writing_claim_checklist.md`

### 停止条件

- Stop if S1 source metrics are missing.
- Stop if summary tables imply S1 LP is complete.
- Stop if summary tables imply frozen-LM matched schema rows are complete.
- Stop if final/best-loss/best-AUC distinctions are collapsed into one unsupported number.

## 2026-06-25 Phase 1 / P1_SCHEMA_COMPLEXITY_NO_LM_S1_SOURCE_INTEGRATION 执行后记录

### 结果

- 未启动训练。
- 未下载模型。
- 未继续 SPD 新变体。
- Formal no-LM S1 source row 已纳入 schema complexity summary。
- Execution status 新增 `P1_SCHEMA_COMPLEXITY_NO_LM_S1_FORMAL_SOURCE_RUN`。
- Completion gap / Phase 4 synthesis 已从 “S1 fixed-split source missing” 更新为：
  - no-LM S1/S2/S3 source rows complete；
  - no-LM S2/S3 LP rows complete；
  - no-LM S1 LP missing；
  - frozen-LM formal matched rows missing。

### 命令与输出

```powershell
python -m py_compile scripts\summarize_schema_complexity_diagnostics.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py
```

- Exit code: `0`

```powershell
python scripts\summarize_schema_complexity_diagnostics.py
```

- Output: `Wrote 11 rows to outputs/final_tables`

```powershell
python scripts\summarize_revision_execution_status.py
```

- Output: `Wrote 38 rows to outputs/final_tables`

```powershell
python scripts\summarize_revision_completion_gap_audit.py
```

- Output: `Wrote 12 rows to outputs/final_tables`

```powershell
python scripts\summarize_phase4_revision_synthesis.py
```

- Output: `Wrote Phase 4 synthesis tables to outputs/final_tables`

```powershell
git diff --check -- scripts\summarize_schema_complexity_diagnostics.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py scripts\run_schema_no_lm_s1_source_gpu0.cmd vivid_med_revision_execution_plan.md
```

- Exit code: `0`

### 指标 / 验证点

| check | result |
|---|---|
| `schema_complexity_diagnostic_summary.csv` has `no_lm_s1_state_only_formal_source` | pass |
| `revision_execution_status.csv` has `P1_SCHEMA_COMPLEXITY_NO_LM_S1_FORMAL_SOURCE_RUN` | pass |
| `revision_completion_gap_audit.csv` still says `S1 LP` missing | pass |
| `phase4_writing_claim_checklist.md` still says `S1 LP` and frozen-LM formal matched rows missing | pass |
| `llm_necessity.csv` includes `no-LM S1 source 0.713327` | pass |

### 失败原因 / 边界

- No command failed.
- This integration does not run or complete S1 LP.
- This integration does not run frozen-LM formal source/LP rows.
- Summary tables now allow a no-LM S1/S2/S3 source statement, but not a completed matched schema-sweep statement.

### 下一步

- Run dependent no-LM S1 LP from `outputs/schema_sweep/no_lm_s1_state_only/best.pt`.
- Use a fresh execution-before record and single GPU.
- After LP completes, update schema summary/gap/Phase 4 tables again.

## 2026-06-25 Phase 1 / P1_SCHEMA_COMPLEXITY_NO_LM_S1_LP_FORMAL_RUN 执行前记录

### 计划

1. 只运行 no-LM S1 fixed-split LP，不启动 frozen-LM，不下载 LLM，不继续 SPD 新变体。
2. 使用 `configs/schema_sweep/lp_no_lm_s1_state_only.yaml`。
3. 从 `outputs/schema_sweep/no_lm_s1_state_only/best.pt` 初始化 ViT backbone，并冻结 backbone。
4. 单卡 GPU0 运行，GPU1 保持空闲。
5. 启动后检查日志、进程、GPU、首次评估和最终退出码。

### 命令

```powershell
Test-Path outputs\schema_sweep\lp_no_lm_s1_state_only\metrics_final.json
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,power.limit --format=csv,noheader,nounits
wmic process call create "cmd.exe /c H:\Xiyao_Wang\021_260129VIVID\scripts\run_schema_lp_no_lm_s1_gpu0.cmd"
Get-Content outputs\logs\schema_lp_no_lm_s1_gpu0.log -Tail 40
Get-Content outputs\schema_sweep\lp_no_lm_s1_state_only\metrics_final.json
```

### 输入

- `configs/schema_sweep/lp_no_lm_s1_state_only.yaml`
- `outputs/schema_sweep/no_lm_s1_state_only/best.pt`
- `data/splits/chexpert_train_30k.jsonl`
- `data/splits/chexpert_val_fixed.jsonl`
- `scripts/train_vit_baseline.py`

### 预期输出

- `outputs/schema_sweep/lp_no_lm_s1_state_only/metrics_final.json`
- `outputs/schema_sweep/lp_no_lm_s1_state_only/best.pt`
- `outputs/schema_sweep/lp_no_lm_s1_state_only/final.pt`
- `outputs/schema_sweep/lp_no_lm_s1_state_only/step_3000.pt`
- `outputs/logs/schema_lp_no_lm_s1_gpu0.log`

### 停止条件

- Stop if `metrics_final.json` already exists before launch.
- Stop if source checkpoint `outputs/schema_sweep/no_lm_s1_state_only/best.pt` is missing.
- Stop if the log shows checkpoint-load, dataset, CUDA, or non-zero-exit failure.
- Stop if GPU memory/power/utilization indicates an abnormal stall during early monitoring.
- Stop after S1 LP completion; do not launch frozen-LM rows until the LP result is integrated and a fresh gated decision is recorded.

## 2026-06-25 Phase 1 / P1_SCHEMA_COMPLEXITY_NO_LM_S1_LP_FORMAL_RUN 执行后记录

### 结果

- no-LM S1 fixed-split LP completed successfully.
- No frozen-LM run was launched.
- No LLM/model download was triggered.
- No SPD new variant was launched.
- Wrapper:
  - `scripts/run_schema_lp_no_lm_s1_gpu0.cmd`
- Log:
  - `outputs/logs/schema_lp_no_lm_s1_gpu0.log`
- Output directory:
  - `outputs/schema_sweep/lp_no_lm_s1_state_only`
- Source checkpoint:
  - `outputs/schema_sweep/no_lm_s1_state_only/best.pt`
- Exit code:
  - `0`
- Runtime:
  - start: `2026-06-25 19:28:22`
  - end: `2026-06-25 19:33:29`
  - training log speed at completion: `3000/3000 [04:55, 10.17it/s]`
- GPU:
  - early stable usage: GPU0 about `1.8-1.9 GB`, `~100-130 W`.
  - GPU1 remained idle.
  - after completion both GPUs returned to `0 MiB` memory usage.

### 指标

| metric point | step | val_loss | macro_auc | macro_f1 | micro_f1 |
|---|---:|---:|---:|---:|---:|
| final | 3000 | 0.263213 | 0.821601 | 0.913358 | 0.890315 |
| best val_loss row | 2000 | 0.261554 | 0.820109 | 0.912632 | 0.890036 |
| best macro_auc row | 400 | 0.271630 | 0.850335 | 0.906785 | 0.887524 |

### 失败原因 / 边界

- No command failed.
- Source checkpoint loaded successfully.
- Backbone freeze was confirmed in the log: `froze 150 params, trainable: 10,766`.
- This completes the no-LM S1 LP endpoint only.
- It does not complete frozen-LM matched schema rows.
- Final macro-AUC is lower than the early best-AUC row, so summaries should preserve final/best-loss/best-AUC distinctions.

### 下一步

- Update schema complexity summary, execution status, completion gap audit, and Phase 4 synthesis to include formal no-LM S1 LP.
- After integration, the no-LM side of S1/S2/S3 source+LP will be complete.
- The remaining formal schema gap will be frozen-LM matched S1/S2/S3 source+LP, unless explicitly scoped out.

## 2026-06-25 Phase 1 / P1_SCHEMA_COMPLEXITY_NO_LM_S1_LP_INTEGRATION 执行前记录

### 计划

1. 不启动训练，不下载模型，不继续 SPD 新变体。
2. 把 `outputs/schema_sweep/lp_no_lm_s1_state_only/metrics_final.json` 纳入 schema complexity summary。
3. 更新 execution status / completion gap audit / Phase 4 synthesis，使 schema gap 明确变为：
   - no-LM S1/S2/S3 source+LP complete；
   - frozen-LM formal matched rows missing。
4. 重新生成所有相关 summary 输出并抽查关键行。
5. 写回执行结果、指标、失败状态和下一步。

### 命令

```powershell
python -m py_compile scripts\summarize_schema_complexity_diagnostics.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py
python scripts\summarize_schema_complexity_diagnostics.py
python scripts\summarize_revision_execution_status.py
python scripts\summarize_revision_completion_gap_audit.py
python scripts\summarize_phase4_revision_synthesis.py
Select-String -Path outputs\final_tables\schema_complexity_diagnostic_summary.csv -Pattern 'no_lm_s1_state_only_formal_lp'
Select-String -Path outputs\final_tables\revision_completion_gap_audit.csv -Pattern 'no-LM S1/S2/S3 formal source\+LP rows are complete'
Select-String -Path outputs\final_tables\phase4_writing_claim_checklist.md -Pattern 'frozen-LM formal matched rows are missing'
```

### 输入

- `outputs/schema_sweep/lp_no_lm_s1_state_only/metrics_final.json`
- `outputs/schema_sweep/lp_no_lm_s1_state_only/metrics_step_*.json`
- `outputs/logs/schema_lp_no_lm_s1_gpu0.log`
- Existing summary scripts and final-table inputs.

### 预期输出

- Updated `outputs/final_tables/schema_complexity_diagnostic_summary.{csv,md}`
- Updated `outputs/final_tables/revision_execution_status.{csv,md}`
- Updated `outputs/final_tables/revision_completion_gap_audit.{csv,md}`
- Updated `outputs/final_tables/llm_necessity.{csv,md}`
- Updated `outputs/final_tables/module_candidates.{csv,md}`
- Updated `outputs/final_tables/phase4_writing_claim_checklist.md`

### 停止条件

- Stop if S1 LP metrics are missing.
- Stop if summary tables imply frozen-LM matched schema rows are complete.
- Stop if summary tables imply no-LM-only rows are a completed matched frozen-LM/no-LM schema sweep.
- Stop if final/best-loss/best-AUC distinctions are collapsed into one unsupported number.

## 2026-06-25 Phase 1 / P1_SCHEMA_COMPLEXITY_NO_LM_S1_LP_INTEGRATION 执行后记录

### 结果

- 未启动训练。
- 未下载模型。
- 未继续 SPD 新变体。
- Formal no-LM S1 LP row 已纳入 schema complexity summary。
- Execution status 新增 `P1_SCHEMA_COMPLEXITY_NO_LM_S1_LP_FORMAL_RUN`。
- Completion gap / Phase 4 synthesis 已更新为：
  - no-LM S1/S2/S3 formal source+LP rows complete；
  - frozen-LM formal matched rows missing。
- `llm_necessity.csv` 中的 S1 表述已从容易误解的 “S1 exact historical coverage missing” 改为 `S1 historical coverage mismatch documented`，避免把历史覆盖审计误写成当前 formal S1 仍缺失。

### 命令与输出

```powershell
python -m py_compile scripts\summarize_schema_complexity_diagnostics.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py
```

- Exit code: `0`

```powershell
python scripts\summarize_schema_complexity_diagnostics.py
```

- Output: `Wrote 12 rows to outputs/final_tables`

```powershell
python scripts\summarize_revision_execution_status.py
```

- Output: `Wrote 39 rows to outputs/final_tables`

```powershell
python scripts\summarize_revision_completion_gap_audit.py
```

- Output: `Wrote 12 rows to outputs/final_tables`

```powershell
python scripts\summarize_phase4_revision_synthesis.py
```

- Output: `Wrote Phase 4 synthesis tables to outputs/final_tables`

```powershell
git diff --check -- scripts\summarize_schema_complexity_diagnostics.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py scripts\run_schema_no_lm_s1_source_gpu0.cmd scripts\run_schema_lp_no_lm_s1_gpu0.cmd vivid_med_revision_execution_plan.md
```

- Exit code: `0`

### 指标 / 验证点

| check | result |
|---|---|
| `schema_complexity_diagnostic_summary.csv` has `no_lm_s1_state_only_formal_lp` | pass |
| `revision_execution_status.csv` has `P1_SCHEMA_COMPLEXITY_NO_LM_S1_LP_FORMAL_RUN` | pass |
| `revision_completion_gap_audit.csv` says no-LM S1/S2/S3 source+LP complete | pass |
| `phase4_writing_claim_checklist.md` says frozen-LM formal matched rows are missing | pass |
| `llm_necessity.csv` includes S1 source/LP metrics and historical mismatch wording | pass |

### 失败原因 / 边界

- No command failed.
- This integration completes the no-LM side of the formal S1/S2/S3 source+LP schema evidence.
- It does not complete matched frozen-LM formal schema evidence.
- Frozen-LM serializer/debug evidence remains debug/runtime evidence, not matched formal performance evidence.

### 下一步

- Remaining formal schema-complexity gap: frozen-LM matched S1/S2/S3 source+LP rows.
- Because frozen-LM source rows may be multi-hour and use large Qwen weights, do not launch the whole matrix.
- If continuing, start with one gated frozen-LM source row only, preferably S2 state_answerability, one GPU, monitor first 500 steps, and stop before any second frozen-LM row unless speed/VRAM and first metrics are acceptable.

## 2026-06-25 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S2_SOURCE_GATED_RUN 执行前记录

### 计划

1. 只启动 frozen-LM S2 state_answerability formal source row，不启动 S1/S3 frozen rows，不启动 LP，不继续 SPD 新变体。
2. 使用 `configs/schema_sweep/frozen_lm_s2_state_answerability.yaml` 原始 formal config：
   - model: `Qwen/Qwen2.5-1.5B-Instruct`
   - schema_mode: `state_answerability`
   - max_steps: `10000`
   - batch_size: `4`
   - gradient_accumulation_steps: `8`
3. 单卡 GPU0 运行，GPU1 保持空闲，先监控启动和前 500 steps。
4. 不改 batch size，除非启动后显存/功率显示明显低效且需要另开独立 config decision；本次 formal run 保持协议一致。
5. 若 500-step 以内速度、显存、日志正常，则继续单条 run；不自动启动第二条 frozen-LM row。

### 命令

```powershell
Test-Path outputs\schema_sweep\frozen_lm_s2_state_answerability\metrics_final.json
Get-ChildItem "$env:USERPROFILE\.cache\huggingface\hub" -Directory -Filter 'models--Qwen--Qwen2.5-1.5B-Instruct'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,power.limit --format=csv,noheader,nounits
wmic process call create "cmd.exe /c H:\Xiyao_Wang\021_260129VIVID\scripts\run_schema_frozen_lm_s2_source_gpu0.cmd"
Get-Content outputs\logs\schema_frozen_lm_s2_source_gpu0.log -Tail 80
```

### 输入

- `configs/schema_sweep/frozen_lm_s2_state_answerability.yaml`
- `data/splits/chexpert_train_30k.jsonl`
- `data/splits/chexpert_val_fixed.jsonl`
- Cached model: `C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-1.5B-Instruct`
- `scripts/train_cxr.py`

### 预期输出

- `outputs/schema_sweep/frozen_lm_s2_state_answerability/config_snapshot.yaml` or trainer config artifacts if emitted.
- `outputs/schema_sweep/frozen_lm_s2_state_answerability/checkpoints/best.pt`
- `outputs/schema_sweep/frozen_lm_s2_state_answerability/checkpoints/step_*.pt`
- `outputs/schema_sweep/frozen_lm_s2_state_answerability/checkpoints/final.pt`
- `outputs/schema_sweep/frozen_lm_s2_state_answerability/metrics_final.json`
- `outputs/logs/schema_frozen_lm_s2_source_gpu0.log`

### 停止条件

- Stop if `metrics_final.json` already exists before launch.
- Stop if Qwen cache is missing and download fails.
- Stop if the log shows model-load, dataset, CUDA, OOM, or non-zero-exit failure.
- Stop if GPU memory/power/utilization indicates an abnormal stall during early monitoring.
- Stop before launching S1/S3 frozen-LM rows or any frozen-LM LP until S2 source completion and integration are recorded.

## 2026-06-25 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S2_SOURCE_GATED_RUN 500-step gate 记录

### 结果

- frozen-LM S2 source run launched successfully and passed the first 500-step gate.
- No S1/S3 frozen-LM row was launched.
- No frozen-LM LP was launched.
- No SPD new variant was launched.
- Qwen weights were already available locally in Hugging Face cache; the loader also populated/used the local ModelScope cache path.
- Model loaded successfully:
  - LLM: `Qwen/Qwen2.5-1.5B-Instruct`
  - LLM frozen parameters: `1,543,714,304`
  - trainable parameters: `89,349,888`
- First validation ran at step 500:
  - validation batches: `250`
  - validation wall time in log: about `43s`
  - checkpoint side effect: `outputs/schema_sweep/frozen_lm_s2_state_answerability/checkpoints/best.pt` exists.

### runtime / speed

| item | observed |
|---|---|
| training speed before validation | about `2.8-3.0 s/step` |
| estimated full 10k source runtime | about `7.5-8.5 h` plus evaluation/checkpoint overhead |
| GPU0 memory | about `24.3 GB / 24.6 GB` |
| GPU0 power | about `340-344 W` |
| GPU0 utilization | about `90-99%` |
| GPU1 | idle |

### 失败原因 / 边界

- No command failed.
- No OOM or CUDA failure observed.
- This is a high-VRAM run; do not run another training job on GPU0.
- Because GPU0 memory is almost full, do not increase batch size inside this formal run.
- GPU1 is idle, but do not launch another frozen-LM row until this gated S2 row finishes or is explicitly stopped; the purpose is to avoid compounding long-run risk.
- This trainer saves checkpoints but does not emit `metrics_step_500.json`; early gate evidence is therefore log + checkpoint + resource behavior, not step-level metric JSON.

### 下一步

- Let the single frozen-LM S2 formal source run continue.
- Monitor checkpoint/log/GPU periodically.
- After completion, write final metrics and failure/success boundary back to this document, then decide whether to run frozen-LM S2 LP or another frozen-LM source row.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S2_SOURCE_GATED_RUN 执行后记录

### 结果

- frozen-LM S2 state_answerability formal source row completed successfully.
- No S1/S3 frozen-LM row was launched.
- No frozen-LM LP was launched.
- No SPD new variant was launched.
- Wrapper:
  - `scripts/run_schema_frozen_lm_s2_source_gpu0.cmd`
- Log:
  - `outputs/logs/schema_frozen_lm_s2_source_gpu0.log`
- Output directory:
  - `outputs/schema_sweep/frozen_lm_s2_state_answerability`
- Checkpoints:
  - `outputs/schema_sweep/frozen_lm_s2_state_answerability/checkpoints/best.pt`
  - `outputs/schema_sweep/frozen_lm_s2_state_answerability/checkpoints/final.pt`
  - `outputs/schema_sweep/frozen_lm_s2_state_answerability/checkpoints/step_10000.pt`
- Exit code:
  - `0`
- Runtime:
  - start: `2026-06-25 19:39:09`
  - end: `2026-06-26 03:56:24`
  - training log speed at completion: `10000/10000 [8:16:39, 2.98s/it]`
- GPU:
  - GPU0 used about `24.3 GB / 24.6 GB`.
  - GPU0 power was generally `~286-344 W`.
  - GPU1 remained idle.
  - after completion both GPUs returned to `0 MiB` memory usage.

### 指标

| metric | value |
|---|---:|
| final/global step | 10000 |
| final log val_loss | 0.0265 |
| checkpoint `best_val_loss` | 0.026451 |
| best checkpoint step | 10000 |
| validation interval | every 500 steps |
| final validation batches | 250 |
| final validation wall time | about 44s |

Step-level validation loss from log:

| step | val_loss |
|---:|---:|
| 500 | 0.0376 |
| 1000 | 0.0350 |
| 1500 | 0.0314 |
| 2000 | 0.0304 |
| 2500 | 0.0304 |
| 3000 | 0.0292 |
| 3500 | 0.0288 |
| 4000 | 0.0279 |
| 4500 | 0.0280 |
| 5000 | 0.0275 |
| 5500 | 0.0276 |
| 6000 | 0.0278 |
| 6500 | 0.0271 |
| 7000 | 0.0267 |
| 7500 | 0.0265 |
| 8000 | 0.0267 |
| 8500 | 0.0267 |
| 9000 | 0.0266 |
| 9500 | 0.0266 |
| 10000 | 0.0265 |

### 失败原因 / 边界

- No command failed.
- No OOM or CUDA failure observed.
- This run completed the frozen-LM S2 source checkpoint, not the frozen-LM S2 LP endpoint.
- `train_cxr.py` / `training/trainer.py` does not emit `metrics_final.json` for this source path; source evidence is therefore log val_loss plus checkpoint metadata, not JSON classification metrics.
- Because this run used almost all GPU0 VRAM, it should remain single-card/single-run evidence; do not infer that frozen-LM rows can be safely batched on one GPU.
- Matched schema-complexity performance comparison remains incomplete until frozen-LM S2 LP is run and S1/S3 frozen-LM rows are either run or explicitly scoped out.

### 下一步

- Integrate frozen-LM S2 source evidence into schema complexity summary/status/gap tables with a source-val-loss boundary.
- Then decide between:
  - dependent frozen-LM S2 LP from `checkpoints/best.pt`; or
  - another gated frozen-LM source row.
- Prefer frozen-LM S2 LP next because it depends directly on the completed S2 source and is likely much shorter than another source row.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S2_SOURCE_INTEGRATION 执行前记录

### 计划

1. 不启动训练，不下载模型，不继续 SPD 新变体。
2. 新增 frozen-LM source log/checkpoint 汇总脚本，输出 source summary 和 val-loss trace。
3. 把 completed frozen-LM S2 source row 纳入 schema complexity summary / execution status / gap audit / Phase 4 synthesis。
4. 明确边界：
   - frozen-LM S2 source complete；
   - frozen-LM S2 LP missing；
   - frozen-LM S1/S3 source+LP missing；
   - no-LM-only rows cannot be reported as matched frozen-LM/no-LM comparison。
5. 重新生成并抽查 final tables。

### 命令

```powershell
python -m py_compile scripts\summarize_frozen_lm_source_training.py scripts\summarize_schema_complexity_diagnostics.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py
python scripts\summarize_frozen_lm_source_training.py
python scripts\summarize_schema_complexity_diagnostics.py
python scripts\summarize_revision_execution_status.py
python scripts\summarize_revision_completion_gap_audit.py
python scripts\summarize_phase4_revision_synthesis.py
Select-String -Path outputs\final_tables\frozen_lm_source_training_summary.csv -Pattern 'frozen_lm_s2_state_answerability_formal_source'
Select-String -Path outputs\final_tables\schema_complexity_diagnostic_summary.csv -Pattern 'frozen_lm_s2_state_answerability_formal_source'
Select-String -Path outputs\final_tables\revision_completion_gap_audit.csv -Pattern 'frozen-LM S2 source'
```

### 输入

- `outputs/logs/schema_frozen_lm_s2_source_gpu0.log`
- `outputs/schema_sweep/frozen_lm_s2_state_answerability/checkpoints/best.pt`
- `outputs/schema_sweep/frozen_lm_s2_state_answerability/checkpoints/final.pt`
- Existing summary scripts and final-table inputs.

### 预期输出

- `outputs/final_tables/frozen_lm_source_training_summary.{csv,md}`
- `outputs/final_tables/frozen_lm_source_val_loss_trace.{csv,md}`
- Updated `outputs/final_tables/schema_complexity_diagnostic_summary.{csv,md}`
- Updated `outputs/final_tables/revision_execution_status.{csv,md}`
- Updated `outputs/final_tables/revision_completion_gap_audit.{csv,md}`
- Updated `outputs/final_tables/llm_necessity.{csv,md}`
- Updated `outputs/final_tables/module_candidates.{csv,md}`
- Updated `outputs/final_tables/phase4_writing_claim_checklist.md`

### 停止条件

- Stop if frozen-LM source log cannot be parsed.
- Stop if checkpoint metadata cannot confirm step `10000`.
- Stop if summaries describe frozen-LM S2 source as LP/classification evidence.
- Stop if summaries describe the schema sweep as complete while frozen-LM S2 LP and S1/S3 rows are missing.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S2_SOURCE_INTEGRATION 执行后记录

### 结果

- 未启动训练。
- 未下载模型。
- 未继续 SPD 新变体。
- 新增 frozen-LM source 汇总脚本：
  - `scripts/summarize_frozen_lm_source_training.py`
- 新增 final-table 输出：
  - `outputs/final_tables/frozen_lm_source_training_summary.csv`
  - `outputs/final_tables/frozen_lm_source_training_summary.md`
  - `outputs/final_tables/frozen_lm_source_val_loss_trace.csv`
  - `outputs/final_tables/frozen_lm_source_val_loss_trace.md`
- Formal frozen-LM S2 source row 已纳入 schema complexity summary。
- Execution status 新增 `P1_SCHEMA_COMPLEXITY_FROZEN_LM_S2_FORMAL_SOURCE_RUN`。
- Completion gap / Phase 4 synthesis 已更新为：
  - no-LM S1/S2/S3 source+LP complete；
  - frozen-LM S2 source complete；
  - frozen-LM S2 LP missing；
  - frozen-LM S1/S3 source+LP missing。

### 命令与输出

```powershell
python -m py_compile scripts\summarize_frozen_lm_source_training.py scripts\summarize_schema_complexity_diagnostics.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py
```

- Exit code: `0`

```powershell
python scripts\summarize_frozen_lm_source_training.py
```

- Output: `Wrote 1 summary rows and 20 trace rows to outputs/final_tables`

```powershell
python scripts\summarize_schema_complexity_diagnostics.py
```

- Output: `Wrote 13 rows to outputs/final_tables`

```powershell
python scripts\summarize_revision_execution_status.py
```

- Output: `Wrote 40 rows to outputs/final_tables`

```powershell
python scripts\summarize_revision_completion_gap_audit.py
```

- Output: `Wrote 12 rows to outputs/final_tables`

```powershell
python scripts\summarize_phase4_revision_synthesis.py
```

- Output: `Wrote Phase 4 synthesis tables to outputs/final_tables`

```powershell
git diff --check -- scripts\summarize_frozen_lm_source_training.py scripts\summarize_schema_complexity_diagnostics.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py scripts\run_schema_frozen_lm_s2_source_gpu0.cmd vivid_med_revision_execution_plan.md
```

- Exit code: `0`

### 指标 / 验证点

| check | result |
|---|---|
| `frozen_lm_source_training_summary.csv` has S2 source row | pass |
| `frozen_lm_source_val_loss_trace.csv` has 20 validation-loss rows | pass |
| `schema_complexity_diagnostic_summary.csv` has `frozen_lm_s2_state_answerability_formal_source` | pass |
| `revision_execution_status.csv` has `P1_SCHEMA_COMPLEXITY_FROZEN_LM_S2_FORMAL_SOURCE_RUN` | pass |
| `revision_completion_gap_audit.csv` still says frozen-LM S2 LP missing | pass |
| `llm_necessity.csv` includes `frozen S2 source val_loss 0.026451` | pass |
| `phase4_writing_claim_checklist.md` forbids completed matched schema sweep wording | pass |

### 失败原因 / 边界

- No command failed.
- This integration converts source log/checkpoint evidence into final-table rows.
- It does not create downstream LP/classification metrics for frozen-LM S2.
- Matched S2 comparison is still incomplete until frozen-LM S2 LP runs.
- Matched S1/S3 comparison is still incomplete until frozen-LM S1/S3 source+LP rows are either run or explicitly scoped out.

### 下一步

- Preferred next task: frozen-LM S2 LP from `outputs/schema_sweep/frozen_lm_s2_state_answerability/checkpoints/best.pt`.
- This is directly dependent on the completed S2 source and should be shorter than launching another frozen-LM source row.
- Do not launch frozen-LM S1/S3 source rows until S2 LP is completed and integrated.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S2_LP_FORMAL_RUN 执行前记录

### 计划

1. 只运行 frozen-LM S2 dependent LP，不启动 frozen-LM S1/S3 source rows，不继续 SPD 新变体。
2. 使用 `configs/schema_sweep/lp_frozen_lm_s2_state_answerability.yaml`。
3. 从 `outputs/schema_sweep/frozen_lm_s2_state_answerability/checkpoints/best.pt` 初始化 ViT backbone，并冻结 backbone。
4. 单卡 GPU0 运行，GPU1 保持空闲。
5. 完成后提取 final、best-loss、best-AUC 指标，再写回文档和 summary tables。

### 命令

```powershell
Test-Path outputs\schema_sweep\lp_frozen_lm_s2_state_answerability\metrics_final.json
Test-Path outputs\schema_sweep\frozen_lm_s2_state_answerability\checkpoints\best.pt
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,power.limit --format=csv,noheader,nounits
wmic process call create "cmd.exe /c H:\Xiyao_Wang\021_260129VIVID\scripts\run_schema_lp_frozen_lm_s2_gpu0.cmd"
Get-Content outputs\logs\schema_lp_frozen_lm_s2_gpu0.log -Tail 60
Get-Content outputs\schema_sweep\lp_frozen_lm_s2_state_answerability\metrics_final.json
```

### 输入

- `configs/schema_sweep/lp_frozen_lm_s2_state_answerability.yaml`
- `outputs/schema_sweep/frozen_lm_s2_state_answerability/checkpoints/best.pt`
- `data/splits/chexpert_train_30k.jsonl`
- `data/splits/chexpert_val_fixed.jsonl`
- `scripts/train_vit_baseline.py`

### 预期输出

- `outputs/schema_sweep/lp_frozen_lm_s2_state_answerability/metrics_final.json`
- `outputs/schema_sweep/lp_frozen_lm_s2_state_answerability/best.pt`
- `outputs/schema_sweep/lp_frozen_lm_s2_state_answerability/final.pt`
- `outputs/schema_sweep/lp_frozen_lm_s2_state_answerability/step_3000.pt`
- `outputs/logs/schema_lp_frozen_lm_s2_gpu0.log`

### 停止条件

- Stop if LP `metrics_final.json` already exists before launch.
- Stop if source checkpoint is missing.
- Stop if the log shows checkpoint-load, dataset, CUDA, or non-zero-exit failure.
- Stop if GPU memory/power/utilization indicates an abnormal stall during early monitoring.
- Stop after S2 LP completion; do not launch frozen-LM S1/S3 rows until S2 LP is integrated.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S2_LP_FORMAL_RUN 执行后记录

### 结果

- frozen-LM S2 dependent LP completed successfully.
- No frozen-LM S1/S3 source rows were launched.
- No SPD new variant was started.
- GPU state after completion:
  - GPU0: `0 MiB`, `0%`, about `9.53 W`
  - GPU1: `0 MiB`, `0%`, about `8.07 W`
- Process state after completion:
  - no active Python training process was found.

### 命令与输出

```powershell
Get-Content outputs\schema_sweep\lp_frozen_lm_s2_state_answerability\metrics_final.json -Raw
```

- Output:
  - `val_loss=0.2642261591695604`
  - `macro_auc=0.7912591395709081`
  - `macro_f1=0.9089131366993349`
  - `micro_f1=0.8827797934691599`

```powershell
Get-ChildItem outputs\schema_sweep\lp_frozen_lm_s2_state_answerability -Filter 'metrics_step_*.json'
```

- Best validation loss:
  - step `2800`
  - `val_loss=0.262902311389408`
  - `macro_auc=0.790000169517732`
  - `macro_f1=0.911499180715165`
  - `micro_f1=0.885291655037678`
- Best macro-AUC:
  - step `800`
  - `val_loss=0.285972873133326`
  - `macro_auc=0.816738508455639`
  - `macro_f1=0.900018001056212`
  - `micro_f1=0.879988836170807`

```powershell
Get-Content outputs\logs\schema_lp_frozen_lm_s2_gpu0.log -Tail 80
```

- Log tail confirms:
  - `Training: 100%|...| 3000/3000 [04:59<00:00, 10.03it/s`
  - `Step 3000: val_loss = 0.2642`
  - `Training completed!`
  - `EXITCODE 0`
  - `END 2026/06/26 ... 4:12:28.70`

### 指标

| metric | value |
|---|---:|
| final val_loss | 0.264226 |
| final macro_auc | 0.791259 |
| final macro_f1 | 0.908913 |
| final micro_f1 | 0.882780 |
| best-loss step | 2800 |
| best-loss macro_auc | 0.790000 |
| best-AUC step | 800 |
| best-AUC macro_auc | 0.816739 |
| runtime | about 5 min |
| train speed | 10.03 it/s |

### 失败原因 / 边界

- No command failed.
- No CUDA/OOM/checkpoint-load/dataset failure appeared in the log.
- This is a dependent LP result from the completed frozen-LM S2 source checkpoint.
- This closes the focused frozen-LM S2 source+LP pair, but it does not close frozen-LM S1/S3 source+LP rows.
- The full matched S1/S2/S3 schema sweep is still incomplete unless frozen-LM S1/S3 are run or explicitly scoped out.

### 下一步

- Integrate `frozen_lm_s2_state_answerability_formal_lp` into:
  - `schema_complexity_diagnostic_summary`
  - `revision_execution_status`
  - `revision_completion_gap_audit`
  - `llm_necessity`
  - `module_candidates`
  - `phase4_writing_claim_checklist`
- Do not launch frozen-LM S1/S3 rows during this integration step.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S2_LP_INTEGRATION 执行前记录

### 计划

1. 不启动训练，不下载模型，不继续 SPD 新变体。
2. 将 frozen-LM S2 LP 指标作为单独 evidence row 写入 schema-complexity summary。
3. 将 execution status / gap audit / Phase 4 synthesis 从 `frozen-LM S2 LP missing` 更新为 `frozen-LM S2 source+LP complete; frozen-LM S1/S3 missing`。
4. 重新生成 final tables。
5. 抽查关键文件，确保没有把 focused S2 结果写成完整 S1/S2/S3 schema sweep。

### 命令

```powershell
python -m py_compile scripts\summarize_schema_complexity_diagnostics.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py
python scripts\summarize_schema_complexity_diagnostics.py
python scripts\summarize_revision_execution_status.py
python scripts\summarize_revision_completion_gap_audit.py
python scripts\summarize_phase4_revision_synthesis.py
Select-String -Path outputs\final_tables\schema_complexity_diagnostic_summary.csv -Pattern 'frozen_lm_s2_state_answerability_formal_lp'
Select-String -Path outputs\final_tables\revision_execution_status.csv -Pattern 'P1_SCHEMA_COMPLEXITY_FROZEN_LM_S2_LP_FORMAL_RUN'
Select-String -Path outputs\final_tables\revision_completion_gap_audit.csv -Pattern 'frozen-LM S1/S3'
Select-String -Path outputs\final_tables\llm_necessity.csv -Pattern 'frozen S2 source val_loss'
Select-String -Path outputs\final_tables\phase4_writing_claim_checklist.md -Pattern 'frozen-LM S1/S3'
git diff --check -- scripts\summarize_schema_complexity_diagnostics.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py scripts\run_schema_lp_frozen_lm_s2_gpu0.cmd vivid_med_revision_execution_plan.md
```

### 输入

- `outputs/schema_sweep/lp_frozen_lm_s2_state_answerability/metrics_final.json`
- `outputs/schema_sweep/lp_frozen_lm_s2_state_answerability/best.pt`
- `outputs/schema_sweep/lp_frozen_lm_s2_state_answerability/final.pt`
- `outputs/schema_sweep/lp_frozen_lm_s2_state_answerability/step_3000.pt`
- `outputs/logs/schema_lp_frozen_lm_s2_gpu0.log`
- Existing final-table summary scripts.

### 预期输出

- Updated `outputs/final_tables/schema_complexity_diagnostic_summary.{csv,md}`
- Updated `outputs/final_tables/revision_execution_status.{csv,md}`
- Updated `outputs/final_tables/revision_completion_gap_audit.{csv,md}`
- Updated `outputs/final_tables/llm_necessity.{csv,md}`
- Updated `outputs/final_tables/module_candidates.{csv,md}`
- Updated `outputs/final_tables/phase4_writing_claim_checklist.md`

### 停止条件

- Stop if any required S2 LP artifact is missing.
- Stop if summary scripts fail compile or execution.
- Stop if final tables still say frozen-LM S2 LP is missing.
- Stop if final tables describe the focused S2 pair as a completed S1/S2/S3 schema sweep.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S2_LP_INTEGRATION 执行后记录

### 结果

- 未启动训练。
- 未下载模型。
- 未继续 SPD 新变体。
- `schema_complexity_diagnostic_summary` 新增:
  - `frozen_lm_s2_state_answerability_formal_lp`
- `revision_execution_status` 新增:
  - `P1_SCHEMA_COMPLEXITY_FROZEN_LM_S2_LP_FORMAL_RUN`
- `revision_completion_gap_audit` / `llm_necessity` / `phase4_writing_claim_checklist` 已更新为:
  - no-LM S1/S2/S3 source+LP complete
  - frozen-LM S2 source+LP complete
  - frozen-LM S1/S3 source+LP missing
  - full matched S1/S2/S3 schema sweep still incomplete unless frozen-LM S1/S3 are run or explicitly scoped out

### 命令与输出

```powershell
python -m py_compile scripts\summarize_schema_complexity_diagnostics.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py
```

- Exit code: `0`

```powershell
python scripts\summarize_schema_complexity_diagnostics.py
```

- Output: `Wrote 14 rows to outputs/final_tables`

```powershell
python scripts\summarize_revision_execution_status.py
```

- Output: `Wrote 41 rows to outputs/final_tables`

```powershell
python scripts\summarize_revision_completion_gap_audit.py
```

- Output: `Wrote 12 rows to outputs/final_tables`

```powershell
python scripts\summarize_phase4_revision_synthesis.py
```

- Output: `Wrote Phase 4 synthesis tables to outputs/final_tables`

```powershell
Select-String -Path outputs\final_tables\schema_complexity_diagnostic_summary.csv -Pattern 'frozen_lm_s2_state_answerability_formal_lp'
Select-String -Path outputs\final_tables\revision_execution_status.csv -Pattern 'P1_SCHEMA_COMPLEXITY_FROZEN_LM_S2_LP_FORMAL_RUN'
Select-String -Path outputs\final_tables\revision_completion_gap_audit.csv -Pattern 'frozen-LM S1/S3'
Select-String -Path outputs\final_tables\llm_necessity.csv -Pattern 'frozen S2 source val_loss'
Select-String -Path outputs\final_tables\phase4_writing_claim_checklist.md -Pattern 'frozen-LM S1/S3'
```

- All expected patterns were found.

```powershell
Select-String -Path outputs\final_tables\schema_complexity_diagnostic_summary.md,outputs\final_tables\revision_execution_status.md,outputs\final_tables\revision_completion_gap_audit.md,outputs\final_tables\llm_necessity.md,outputs\final_tables\phase4_writing_claim_checklist.md -Pattern 'frozen-LM S2 LP missing|frozen-LM S2 LP and S1/S3|frozen-LM S2 source complete; frozen-LM S2 LP'
```

- Output: no matches.

```powershell
git diff --check -- scripts\summarize_schema_complexity_diagnostics.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py scripts\run_schema_lp_frozen_lm_s2_gpu0.cmd vivid_med_revision_execution_plan.md
```

- Exit code: `0`

```powershell
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit --format=csv
```

- Output:
  - GPU0 RTX 3090: `0%`, `0 MiB / 24576 MiB`, about `9.07 W / 350 W`
  - GPU1 RTX 3090: `0%`, `0 MiB / 24576 MiB`, about `7.98 W / 350 W`

### 指标 / 验证点

| check | result |
|---|---|
| `schema_complexity_diagnostic_summary.csv` has frozen-LM S2 LP row | pass |
| `revision_execution_status.csv` has frozen-LM S2 LP task row | pass |
| `revision_completion_gap_audit.csv` says frozen-LM S1/S3 rows are missing | pass |
| `llm_necessity.csv` includes frozen S2 source val_loss and LP macro-AUC | pass |
| `phase4_writing_claim_checklist.md` says frozen-LM S1/S3 formal matched rows are missing | pass |
| old `frozen-LM S2 LP missing` wording in final-table markdown | none found |
| current GPU training/process occupancy | idle / no active Python training |

### 失败原因 / 边界

- No integration command failed.
- This integration closes focused frozen-LM S2 source+LP evidence only.
- It does not close frozen-LM S1/S3 source+LP.
- It does not justify a completed matched S1/S2/S3 schema-complexity sweep claim.

### 当前速度结论

- 当前没有训练在跑，因此没有正在变慢的训练进程。
- 刚完成的 frozen-LM S2 LP speed is normal for LP:
  - `3000/3000 [04:59, 10.03it/s]`
  - low GPU memory/power is expected because only the LP head is trained with a frozen backbone.
- 刚完成的 frozen-LM S2 source speed is normal for this workload:
  - about `8:16:39` for 10k steps
  - GPU0 near full VRAM and high power during source training, which is expected for Qwen frozen-LM source path on a 24GB RTX 3090.
- 当前证据不支持“病毒/挖矿/GPU 被未知进程占用导致变慢”:
  - no active Python training remains
  - both GPUs are idle after completion
  - no unknown GPU process was observed
- This runtime check is not a full antivirus scan; it is a process/GPU/runtime evidence check.

### 下一步

- Preferred next decision: decide whether the paper needs frozen-LM S1/S3 formal source+LP rows.
- If yes, run one frozen-LM source row at a time with the same gated protocol; do not batch S1/S3 automatically.
- If no, explicitly scope out frozen-LM S1/S3 and use the focused S2 matched evidence plus no-LM S1/S2/S3 formal evidence as the schema-complexity boundary.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S3_SOURCE_GATED_RUN 执行前记录

### 计划

1. 只启动 frozen-LM S3 `state_uncertainty` formal source，不启动 frozen-LM S1，不启动任何 LP，不继续 SPD 新变体。
2. 因 frozen-LM source 显存接近满 24GB RTX 3090，单卡 GPU0 运行，GPU1 保持空闲。
3. 先确认 S3 formal source final checkpoint 不存在，避免覆盖完成结果。
4. 创建/使用 `scripts/run_schema_frozen_lm_s3_source_gpu0.cmd` wrapper，记录完整 log 和 exit code。
5. 启动后做 early gate：
   - process exists
   - GPU0 memory/power/utilization consistent with frozen-LM source workload
   - log reaches model/data loading and starts training
   - first 500-step validation/checkpoint正常后继续跑
6. 不根据 S3 source 直接更新 LP/classification结论；source 完成后再单独写 integration / LP 前记录。

### 命令

```powershell
Test-Path outputs\schema_sweep\frozen_lm_s3_state_uncertainty\checkpoints\final.pt
Test-Path configs\schema_sweep\frozen_lm_s3_state_uncertainty.yaml
Test-Path outputs\schema_sweep\frozen_lm_s3_state_uncertainty_seed900123\checkpoints\final.pt
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,power.limit --format=csv,noheader,nounits
wmic process call create "cmd.exe /c H:\Xiyao_Wang\021_260129VIVID\scripts\run_schema_frozen_lm_s3_source_gpu0.cmd"
Get-Content outputs\logs\schema_frozen_lm_s3_source_gpu0.log -Tail 80
```

### 输入

- `configs/schema_sweep/frozen_lm_s3_state_uncertainty.yaml`
- `data/splits/chexpert_train_30k.jsonl`
- `data/splits/chexpert_val_fixed.jsonl`
- `scripts/train_cxr.py`
- Qwen/Qwen2.5-1.5B-Instruct local cache / ModelScope cache as resolved by the training runtime.

### 预期输出

- `outputs/logs/schema_frozen_lm_s3_source_gpu0.log`
- `outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/best.pt`
- `outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/final.pt`
- `outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/step_10000.pt`

### 停止条件

- Stop if S3 formal final checkpoint already exists before launch.
- Stop if config or required cache/runtime path is missing.
- Stop if log shows model-load, dataset, CUDA/OOM, or checkpoint-save failure.
- Stop if early speed is much worse than S2 source without matching GPU utilization/memory evidence.
- Stop if GPU0 memory leaves no headroom and causes CUDA OOM; do not compensate with dual-GPU unless code explicitly supports distributed execution.
- Stop after S3 source completion; do not auto-launch S3 LP or S1 source without a new execution-before record.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S3_SOURCE_GATED_RUN 500-step gate 记录

### 结果

- S3 formal source has passed the 500-step runtime gate.
- Training remains active on GPU0 only.
- GPU1 remains idle.
- `best.pt` was created under:
  - `outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/best.pt`

### 命令与输出

```powershell
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,power.limit --format=csv
```

- Around/after the gate:
  - GPU0: `91-99%`, `24305-24307 MiB / 24576 MiB`, about `342.88-343.55 W / 350 W`
  - GPU1: `0%`, `0 MiB / 24576 MiB`, about `8.7 W / 350 W`

```powershell
Get-Content outputs\logs\schema_frozen_lm_s3_source_gpu0.log -Tail 160
```

- Log evidence:
  - validation at step 500 completed across `250/250` validation batches
  - validation speed was about `5.9 it/s`
  - after validation, training resumed from step `501`
  - apparent `501` slow step is tqdm averaging validation overhead into the next training-step estimate; by steps `520+`, speed returned to about `2.83s/it`

```powershell
python -c "import torch; p='outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/best.pt'; ck=torch.load(p,map_location='cpu'); print({k: ck.get(k) for k in ['global_step','best_val_loss','epoch']})"
```

- Output:
  - `global_step=500`
  - `best_val_loss=0.09415725156664849`
  - `epoch=None`

### 指标 / 当前速度判断

| metric | value |
|---|---:|
| checkpoint step | 500 |
| checkpoint best_val_loss | 0.094157 |
| training step speed before validation | about 2.82-3.0 s/it |
| validation speed | about 5.9 it/s |
| post-validation recovered speed | about 2.83 s/it |
| GPU0 memory | about 24.3 GB / 24.6 GB |
| GPU0 power | about 343 W / 350 W |
| GPU1 memory | 0 MiB |

### 失败原因 / 边界

- No CUDA/OOM/model-load/dataset/checkpoint failure observed through step 500.
- No evidence of a hidden GPU process or virus/miner-style GPU contention during this gate.
- The workload is memory-bound/near-capacity; this supports single-GPU execution only.
- Do not launch another frozen-LM source concurrently.
- Do not treat step-500 source loss as final source result or downstream LP/classification evidence.

### 下一步

- Continue the same S3 source run on GPU0 toward step 10000.
- Monitor next validation checkpoints and stop only on explicit runtime failure, nonzero exit, OOM, or sustained abnormal slow speed with low GPU utilization.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S3_SOURCE_GATED_RUN 1000-step 记录

### 结果

- S3 formal source passed the 1000-step checkpoint.
- Training is still active on GPU0 only.
- GPU1 remains idle.
- `step_1000.pt` was created under:
  - `outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/step_1000.pt`
- `best.pt` was updated at step 1000.

### 命令与输出

```powershell
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,power.limit --format=csv
```

- GPU0: `90-99%`, `24307 MiB / 24576 MiB`, about `341-343 W / 350 W`
- GPU1: `0%`, `0 MiB / 24576 MiB`, about `8.7-9.0 W / 350 W`

```powershell
Get-Content outputs\logs\schema_frozen_lm_s3_source_gpu0.log -Tail 180
```

- Log evidence:
  - validation at step 1000 completed across `250/250` validation batches
  - validation speed was about `5.9 it/s`
  - after validation, training resumed past step `1001`
  - speed recovered to about `2.83s/it` by steps `1030+`

```powershell
python -c "import torch; paths=['outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/best.pt','outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/step_1000.pt']; for p in paths: ck=torch.load(p,map_location='cpu'); print(p, {k: ck.get(k) for k in ['global_step','best_val_loss','epoch']})"
```

- Output:
  - `best.pt`: `global_step=1000`, `best_val_loss=0.06919634526968002`
  - `step_1000.pt`: `global_step=1000`, `best_val_loss=0.06919634526968002`

### 指标 / 当前速度判断

| metric | value |
|---|---:|
| checkpoint step | 1000 |
| checkpoint best_val_loss | 0.069196 |
| post-validation recovered speed | about 2.83 s/it |
| GPU0 memory | about 24.3 GB / 24.6 GB |
| GPU0 power | about 341-343 W / 350 W |
| GPU1 memory | 0 MiB |

### 失败原因 / 边界

- No CUDA/OOM/model-load/dataset/checkpoint failure observed through step 1000.
- No hidden GPU process or unknown heavy GPU contention observed.
- The run remains high-memory/high-power and should stay single-GPU.
- No batch-size adjustment is needed because speed is normal and no OOM has occurred.
- Do not launch frozen-LM S1 or S3 LP until this source run completes and a fresh execution-before record is written.

### 下一步

- Continue S3 source run on GPU0.
- Next checkpoint/health target: step `1500` or `2000`, depending on monitoring interval.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S3_SOURCE_GATED_RUN 2000-step 记录

### 结果

- S3 formal source passed the 2000-step checkpoint.
- Training is still active on GPU0 only.
- GPU1 remains idle.
- `step_2000.pt` was created under:
  - `outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/step_2000.pt`
- `best.pt` remains step 1500, so step 2000 did not improve best validation loss.

### 命令与输出

```powershell
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,power.limit --format=csv
```

- GPU0: `91%`, `24307 MiB / 24576 MiB`, about `344 W / 350 W`
- GPU1: `0%`, `0 MiB / 24576 MiB`, about `8.9 W / 350 W`

```powershell
Get-Content outputs\logs\schema_frozen_lm_s3_source_gpu0.log -Tail 160
```

- Log evidence:
  - validation at step 2000 completed across `250/250` validation batches
  - validation speed remained about `5.9 it/s`
  - after validation, training resumed past step `2001`
  - speed recovered to about `2.82-2.95s/it`

```powershell
python -c "import torch; paths=['outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/best.pt','outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/step_2000.pt']; for p in paths: ck=torch.load(p,map_location='cpu'); print(p, {k: ck.get(k) for k in ['global_step','best_val_loss','epoch']})"
```

- Output:
  - `best.pt`: `global_step=1500`, `best_val_loss=0.04641297536343336`
  - `step_2000.pt`: `global_step=2000`, `best_val_loss=0.04641297536343336`

### 指标 / 当前速度判断

| metric | value |
|---|---:|
| latest named checkpoint | step 2000 |
| current best checkpoint step | 1500 |
| current best_val_loss | 0.046413 |
| post-validation recovered speed | about 2.82-2.95 s/it |
| GPU0 memory | about 24.3 GB / 24.6 GB |
| GPU0 power | about 344 W / 350 W |
| GPU1 memory | 0 MiB |

### 失败原因 / 边界

- No runtime failure through step 2000.
- Step 2000 not improving best loss is validation behavior, not a speed/hardware failure.
- The process remains high-memory/high-power and single-GPU.
- Do not start frozen-LM S1 or any LP concurrently.

### 下一步

- Continue S3 source run on GPU0.
- Next checkpoint/health target: step `3000` or later if the run remains stable.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S3_SOURCE_GATED_RUN 3000-step 记录

### 结果

- S3 formal source passed the 3000-step checkpoint.
- Training is still active on GPU0 only.
- GPU1 remains idle.
- `step_3000.pt` was created under:
  - `outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/step_3000.pt`
- `best.pt` was updated at step 3000.

### 命令与输出

```powershell
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,power.limit --format=csv
```

- GPU0: `99%`, `24259 MiB / 24576 MiB`, about `343.77 W / 350 W`
- GPU1: `0%`, `0 MiB / 24576 MiB`, about `9.26 W / 350 W`

```powershell
Get-Content outputs\logs\schema_frozen_lm_s3_source_gpu0.log -Tail 180
```

- Log evidence:
  - validation at step 3000 completed across `250/250` validation batches
  - validation speed was about `5.8 it/s`
  - after validation, training resumed past step `3001`
  - speed recovered to about `2.9-3.1s/it`

```powershell
python -c "import torch; paths=['outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/best.pt','outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/step_3000.pt']; for p in paths: ck=torch.load(p,map_location='cpu'); print(p, {k: ck.get(k) for k in ['global_step','best_val_loss','epoch']})"
```

- Output:
  - `best.pt`: `global_step=3000`, `best_val_loss=0.040528286695480344`
  - `step_3000.pt`: `global_step=3000`, `best_val_loss=0.040528286695480344`

### 指标 / 当前速度判断

| metric | value |
|---|---:|
| latest named checkpoint | step 3000 |
| current best checkpoint step | 3000 |
| current best_val_loss | 0.040528 |
| post-validation recovered speed | about 2.9-3.1 s/it |
| GPU0 memory | about 24.26 GB / 24.6 GB |
| GPU0 power | about 344 W / 350 W |
| GPU1 memory | 0 MiB |

### 失败原因 / 边界

- No runtime failure through step 3000.
- The source objective continues to improve best validation loss.
- Speed and GPU utilization remain consistent with the completed frozen-LM S2 source run.
- The run remains high-memory/high-power and should stay single-GPU.
- Do not start frozen-LM S1 or any LP concurrently.

### 下一步

- Continue S3 source run on GPU0.
- Next checkpoint/health target: step `4000` or later if stable.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S3_SOURCE_GATED_RUN 4000-step 记录

### 结果

- S3 formal source passed the 4000-step checkpoint.
- Training is still active on GPU0 only.
- GPU1 remains idle.
- `step_4000.pt` was created under:
  - `outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/step_4000.pt`
- `best.pt` was updated at step 4000.

### 命令与输出

```powershell
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,power.limit --format=csv
```

- GPU0: `99%`, `24259 MiB / 24576 MiB`, about `341.97 W / 350 W`
- GPU1: `0%`, `0 MiB / 24576 MiB`, about `9.09 W / 350 W`

```powershell
Get-Content outputs\logs\schema_frozen_lm_s3_source_gpu0.log -Tail 180
```

- Log evidence:
  - validation at step 4000 completed across `250/250` validation batches
  - validation speed was about `5.8 it/s`
  - after validation, training resumed past step `4001`
  - speed recovered to about `2.9-3.0s/it`

```powershell
python -c "import torch; paths=['outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/best.pt','outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/step_4000.pt']; for p in paths: ck=torch.load(p,map_location='cpu'); print(p, {k: ck.get(k) for k in ['global_step','best_val_loss','epoch']})"
```

- Output:
  - `best.pt`: `global_step=4000`, `best_val_loss=0.03523650147020817`
  - `step_4000.pt`: `global_step=4000`, `best_val_loss=0.03523650147020817`

### 指标 / 当前速度判断

| metric | value |
|---|---:|
| latest named checkpoint | step 4000 |
| current best checkpoint step | 4000 |
| current best_val_loss | 0.035237 |
| post-validation recovered speed | about 2.9-3.0 s/it |
| GPU0 memory | about 24.26 GB / 24.6 GB |
| GPU0 power | about 342 W / 350 W |
| GPU1 memory | 0 MiB |

### 失败原因 / 边界

- No runtime failure through step 4000.
- The source objective continues to improve best validation loss.
- Speed and GPU utilization remain normal for frozen-LM source.
- Do not start frozen-LM S1 or any LP concurrently.

### 下一步

- Continue S3 source run on GPU0.
- Next checkpoint/health target: step `5000` or later if stable.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S3_SOURCE_GATED_RUN 5000-step 记录

### 结果

- S3 formal source passed the 5000-step checkpoint.
- Training is still active on GPU0 only.
- GPU1 remains idle.
- `step_5000.pt` was created under:
  - `outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/step_5000.pt`
- `best.pt` was updated at step 5000.

### 命令与输出

```powershell
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,power.limit --format=csv
```

- GPU0: `91%`, `24259 MiB / 24576 MiB`, about `342.23 W / 350 W`
- GPU1: `0%`, `0 MiB / 24576 MiB`, about `9.09 W / 350 W`

```powershell
Get-Content outputs\logs\schema_frozen_lm_s3_source_gpu0.log -Tail 180
```

- Log evidence:
  - validation at step 5000 completed across `250/250` validation batches
  - validation speed was about `5.8 it/s`
  - after validation, training resumed past step `5001`
  - speed recovered to about `2.9-3.0s/it`

```powershell
python -c "import torch; paths=['outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/best.pt','outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/step_5000.pt']; for p in paths: ck=torch.load(p,map_location='cpu'); print(p, {k: ck.get(k) for k in ['global_step','best_val_loss','epoch']})"
```

- Output:
  - `best.pt`: `global_step=5000`, `best_val_loss=0.031996499739587306`
  - `step_5000.pt`: `global_step=5000`, `best_val_loss=0.031996499739587306`

### 指标 / 当前速度判断

| metric | value |
|---|---:|
| latest named checkpoint | step 5000 |
| current best checkpoint step | 5000 |
| current best_val_loss | 0.031996 |
| post-validation recovered speed | about 2.9-3.0 s/it |
| GPU0 memory | about 24.26 GB / 24.6 GB |
| GPU0 power | about 342 W / 350 W |
| GPU1 memory | 0 MiB |

### 失败原因 / 边界

- No runtime failure through step 5000.
- The source objective continues to improve best validation loss through the halfway point.
- Speed and GPU utilization remain normal for frozen-LM source.
- Do not start frozen-LM S1 or any LP concurrently.

### 下一步

- Continue S3 source run on GPU0.
- Next checkpoint/health target: step `6000` or later if stable.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S3_SOURCE_GATED_RUN 8000-step 外部中断记录

### 结果

- S3 formal source reached step 8000 and saved `step_8000.pt`.
- `best.pt` was updated at step 8000.
- The active training process stopped before step 10000.
- The log ends around step `8128/10000` with `^C`.
- GPU0/GPU1 are now idle and no S3 training process remains.

### 命令与输出

```powershell
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit --format=csv
```

- Output after interruption:
  - GPU0 RTX 3090: `0%`, `0 MiB / 24576 MiB`, about `12.7 W / 350 W`
  - GPU1 RTX 3090: `0%`, `0 MiB / 24576 MiB`, about `8.5 W / 350 W`

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|cmd' -and $_.CommandLine -match 'frozen_lm_s3|train_cxr|run_schema_frozen_lm_s3|schema_sweep' }
```

- Output: no active S3 training `python.exe` / wrapper `cmd.exe`.

```powershell
Get-Content outputs\logs\schema_frozen_lm_s3_source_gpu0.log -Tail 80
```

- Output tail:
  - training had resumed after step 8000 validation
  - last visible progress: `8128/10000`
  - final visible marker: `^C`
  - no CUDA/OOM traceback, no Python exception, no normal `EXITCODE 0`

```powershell
python -c "import torch; paths=['outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/best.pt','outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/step_8000.pt']; for p in paths: ck=torch.load(p,map_location='cpu'); print(p, {k: ck.get(k) for k in ['global_step','best_val_loss','epoch']})"
```

- Output:
  - `best.pt`: `global_step=8000`, `best_val_loss=0.030963554482907055`
  - `step_8000.pt`: `global_step=8000`, `best_val_loss=0.030963554482907055`

### 指标 / 当前速度判断

| metric | value |
|---|---:|
| last complete named checkpoint | step 8000 |
| current best checkpoint step | 8000 |
| current best_val_loss | 0.030964 |
| last visible progress | step 8128 |
| failure marker | `^C` |
| GPU after interruption | idle |

### 失败原因 / 边界

- Failure layer: external/runtime interruption after successful validation/checkpoint activity.
- This is not an OOM, model-load, dataset, or checkpoint-save failure based on current log evidence.
- This is not evidence that training speed was abnormal; prior checkpoints showed normal high-GPU-utilization source speed.
- Current evidence is not a completed S3 source row because `final.pt` and `EXITCODE 0` are missing.

### 下一步

- Resume from `outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/step_8000.pt` on GPU0.
- Keep the same config, batch size, and single-GPU policy.
- Do not start S3 LP or frozen-LM S1 until source resumes to step 10000 and writes a normal final checkpoint.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S3_SOURCE_RESUME8000_TO_FINAL 执行前记录

### 计划

1. 只恢复 frozen-LM S3 source，不启动 frozen-LM S1，不启动 S3 LP，不继续 SPD 新变体。
2. 使用 `configs/schema_sweep/frozen_lm_s3_state_uncertainty.yaml`。
3. 从 `outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/step_8000.pt` 恢复 optimizer/scheduler/global_step/best_val_loss。
4. 单卡 GPU0 运行，GPU1 保持空闲。
5. 输出独立 resume log，保留原始中断 log 作为 failure case。
6. 完成后检查:
   - `final.pt`
   - `step_10000.pt`
   - `best.pt`
   - log `EXITCODE 0`

### 命令

```powershell
Test-Path outputs\schema_sweep\frozen_lm_s3_state_uncertainty\checkpoints\step_8000.pt
Test-Path outputs\schema_sweep\frozen_lm_s3_state_uncertainty\checkpoints\final.pt
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,power.limit --format=csv,noheader,nounits
wmic process call create "cmd.exe /c H:\Xiyao_Wang\021_260129VIVID\scripts\run_schema_frozen_lm_s3_source_resume8000_gpu0.cmd"
Get-Content outputs\logs\schema_frozen_lm_s3_source_resume8000_gpu0.log -Tail 80
```

### 输入

- `configs/schema_sweep/frozen_lm_s3_state_uncertainty.yaml`
- `outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/step_8000.pt`
- `scripts/train_cxr.py`
- Existing train/val split files already used by the interrupted run.

### 预期输出

- `outputs/logs/schema_frozen_lm_s3_source_resume8000_gpu0.log`
- `outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/step_10000.pt`
- `outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/final.pt`
- Updated or preserved `outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/best.pt`

### 停止条件

- Stop if `step_8000.pt` is missing or checkpoint metadata cannot load.
- Stop if `final.pt` already exists before resume.
- Stop if resume log shows checkpoint-load, CUDA/OOM, dataset, or nonzero-exit failure.
- Stop if GPU0 has low utilization plus abnormal slow speed for a sustained window.
- Stop after S3 source completion; do not launch S3 LP without a new execution-before record.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S3_SOURCE_RESUME8000_TO_FINAL 执行后记录

### 结果

- Resume run completed normally with `EXITCODE 0`.
- `outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/final.pt` now exists.
- `outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/step_10000.pt` now exists.
- `best.pt`, `final.pt`, and `step_10000.pt` all report `global_step=10000`.
- GPU0/GPU1 are idle after completion, and no Python training process remains.
- The original interrupted log is preserved as failure provenance.

### 命令与输入输出

```powershell
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit --format=csv
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime,Path
Get-Content outputs\logs\schema_frozen_lm_s3_source_gpu0.log -Tail 30
Get-Content outputs\logs\schema_frozen_lm_s3_source_resume8000_gpu0.log -Tail 40
python -c "import torch; paths=['outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/best.pt','outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/final.pt','outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/step_10000.pt','outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/step_8000.pt']; [print(p, {k: torch.load(p,map_location='cpu').get(k) for k in ['global_step','best_val_loss','epoch']}) for p in paths]"
```

Inputs:

- `configs/schema_sweep/frozen_lm_s3_state_uncertainty.yaml`
- `outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/step_8000.pt`
- `scripts/run_schema_frozen_lm_s3_source_resume8000_gpu0.cmd`

Outputs:

- `outputs/logs/schema_frozen_lm_s3_source_gpu0.log`
- `outputs/logs/schema_frozen_lm_s3_source_resume8000_gpu0.log`
- `outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/best.pt`
- `outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/final.pt`
- `outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/step_10000.pt`

### 指标

| metric | value |
|---|---:|
| original last visible progress | `8128/10000` |
| original complete checkpoint before interruption | `step_8000.pt` |
| step 8000 best_val_loss | `0.030963554482907055` |
| resume training wall-clock shown by progress bar | `1:41:09` |
| resume final log val_loss | `0.0306` |
| checkpoint best step after resume | `10000` |
| checkpoint best_val_loss after resume | `0.03057638929784298` |
| resume exit code | `0` |

### 失败原因 / 边界

- Failure layer for the first run: external/runtime interruption, visible as final `^C` around step `8128/10000`.
- The interrupted first run shows no CUDA OOM, dataset error, checkpoint-save error, or Python traceback.
- Resume from `step_8000.pt` recovered optimizer/scheduler/global_step/best_val_loss and completed step 10000.
- This is now valid frozen-LM S3 source checkpoint evidence.
- It is not downstream LP/classification evidence and does not complete matched S3 source+LP.

### 下一步

- Integrate S3 source into frozen-LM source summary, schema diagnostic summary, execution status, gap audit, and Phase 4 synthesis.
- After integration verification, open a separate execution-before record for frozen-LM S3 LP if continuing the matched S3 pair.
- Do not launch frozen-LM S1 or any broad S1/S2/S3 batch.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S3_SOURCE_INTEGRATION 执行前记录

### 计划

1. Extend `scripts/summarize_frozen_lm_source_training.py` from S2-only to S2+S3.
2. Treat S3 as a completed source row with two logs:
   - original interrupted log
   - resume-to-final log
3. Update downstream summaries so they state:
   - no-LM S1/S2/S3 source+LP complete
   - frozen-LM S2 source+LP complete
   - frozen-LM S3 source complete
   - frozen-LM S3 LP and frozen-LM S1 source+LP still missing
4. Rebuild final tables and run targeted checks.
5. Stop if any summary calls the full matched S1/S2/S3 schema sweep complete.

### 命令

```powershell
python -m py_compile scripts\summarize_frozen_lm_source_training.py scripts\summarize_schema_complexity_diagnostics.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py
python scripts\summarize_frozen_lm_source_training.py
python scripts\summarize_schema_complexity_diagnostics.py
python scripts\summarize_revision_execution_status.py
python scripts\summarize_revision_completion_gap_audit.py
python scripts\summarize_phase4_revision_synthesis.py
Select-String -Path outputs\final_tables\schema_complexity_diagnostic_summary.csv -Pattern 'frozen_lm_s3_state_uncertainty_formal_source'
Select-String -Path outputs\final_tables\revision_execution_status.csv -Pattern 'P1_SCHEMA_COMPLEXITY_FROZEN_LM_S3_FORMAL_SOURCE_RUN'
Select-String -Path outputs\final_tables\revision_completion_gap_audit.csv -Pattern 'frozen-LM S3 LP|frozen-LM S1'
Select-String -Path outputs\final_tables\llm_necessity.csv -Pattern 'frozen S3 source val_loss'
Select-String -Path outputs\final_tables\phase4_writing_claim_checklist.md -Pattern 'frozen-LM S3 LP|matched frozen-LM/no-LM S1/S2/S3'
```

### 输入

- `outputs/final_tables/frozen_lm_source_training_summary.csv`
- `outputs/logs/schema_frozen_lm_s3_source_gpu0.log`
- `outputs/logs/schema_frozen_lm_s3_source_resume8000_gpu0.log`
- `outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/best.pt`
- `outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/final.pt`
- `outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/step_10000.pt`

### 预期输出

- Updated `outputs/final_tables/frozen_lm_source_training_summary.csv`
- Updated `outputs/final_tables/frozen_lm_source_val_loss_trace.csv`
- Updated `outputs/final_tables/schema_complexity_diagnostic_summary.csv`
- Updated `outputs/final_tables/revision_execution_status.csv`
- Updated `outputs/final_tables/revision_completion_gap_audit.csv`
- Updated `outputs/final_tables/llm_necessity.csv`
- Updated `outputs/final_tables/phase4_writing_claim_checklist.md`

### 停止条件

- Stop if S3 source final checkpoint metadata does not report `global_step=10000`.
- Stop if S3 source is written as S3 LP/classification evidence.
- Stop if the original `^C` interruption is hidden.
- Stop if final outputs say matched frozen-LM/no-LM S1/S2/S3 source-plus-LP comparison is complete.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S3_SOURCE_INTEGRATION 执行后记录

### 结果

- `scripts/summarize_frozen_lm_source_training.py` now writes two source rows:
  - `frozen_lm_s2_state_answerability_formal_source`
  - `frozen_lm_s3_state_uncertainty_formal_source`
- `outputs/final_tables/frozen_lm_source_training_summary.csv` has 2 rows.
- `outputs/final_tables/schema_complexity_diagnostic_summary.csv` has 15 rows and includes `frozen_lm_s3_state_uncertainty_formal_source`.
- `outputs/final_tables/revision_execution_status.csv` has 42 rows and includes `P1_SCHEMA_COMPLEXITY_FROZEN_LM_S3_FORMAL_SOURCE_RUN`.
- `outputs/final_tables/revision_completion_gap_audit.csv` reports frozen-LM S3 LP and frozen-LM S1 rows as the remaining schema-complexity gap.
- `outputs/final_tables/llm_necessity.csv` includes `frozen S3 source val_loss 0.030576`.
- `outputs/final_tables/phase4_writing_claim_checklist.md` still forbids claiming completed matched frozen-LM/no-LM S1/S2/S3 source-plus-LP comparison.

### 指标

| output | value |
|---|---:|
| frozen source summary rows | `2` |
| frozen source val-loss trace rows | `39` |
| schema diagnostic rows | `15` |
| revision execution status rows | `42` |
| gap audit rows | `12` |
| S3 source checkpoint_best_val_loss | `0.030576` |
| S3 source exitcode | `0` |

### 失败原因 / 边界

- No integration-script failure.
- The first S3 source run interruption remains visible in the source summary runtime field:
  - `interrupted_at_8128/10000`
- Current schema-complexity status is partial:
  - frozen-LM S2 source+LP complete
  - frozen-LM S3 source complete
  - frozen-LM S3 LP missing
  - frozen-LM S1 source+LP missing
- Therefore the full matched S1/S2/S3 schema sweep remains incomplete.

### 下一步

- Run frozen-LM S3 LP only after a fresh execution-before record.
- Use one GPU for S3 LP; do not batch frozen-LM S1 or any new SPD variant.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S3_LP_FORMAL_RUN 执行前记录

### 计划

1. Run only the dependent frozen-LM S3 `state_uncertainty` LP.
2. Use `configs/schema_sweep/lp_frozen_lm_s3_state_uncertainty.yaml`.
3. Initialize the frozen ViT backbone from:
   - `outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/best.pt`
4. Keep `batch_size=16`, `gradient_accumulation_steps=2`, `num_workers=0`, and `bf16=true` as configured.
5. Run on GPU0 with `scripts/run_schema_lp_frozen_lm_s3_gpu0.cmd`.
6. Do not launch frozen-LM S1, do not launch another frozen-LM source job, and do not continue SPD variants.

### 预检命令与结果

```powershell
Test-Path scripts\run_schema_lp_frozen_lm_s3_gpu0.cmd
Test-Path outputs\schema_sweep\lp_frozen_lm_s3_state_uncertainty\final.pt
Test-Path outputs\schema_sweep\frozen_lm_s3_state_uncertainty\checkpoints\best.pt
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit --format=csv
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime,Path
```

Preflight outputs:

- wrapper before creation: `False`; created `scripts/run_schema_lp_frozen_lm_s3_gpu0.cmd`.
- S3 LP `final.pt`: `False`, so no completed LP result will be overwritten.
- S3 source `best.pt`: `True`.
- GPU0: `0%`, `13 MiB / 24576 MiB`, about `12.77 W / 350 W`.
- GPU1: `0%`, `13 MiB / 24576 MiB`, about `8.08 W / 350 W`.
- Active Python process check: no Python training process found.

### 执行命令

```powershell
wmic process call create "cmd.exe /c H:\Xiyao_Wang\021_260129VIVID\scripts\run_schema_lp_frozen_lm_s3_gpu0.cmd"
Get-Content outputs\logs\schema_lp_frozen_lm_s3_gpu0.log -Tail 80
```

### 输入

- `configs/schema_sweep/lp_frozen_lm_s3_state_uncertainty.yaml`
- `outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/best.pt`
- `scripts/train_vit_baseline.py`
- `scripts/run_schema_lp_frozen_lm_s3_gpu0.cmd`

### 预期输出

- `outputs/logs/schema_lp_frozen_lm_s3_gpu0.log`
- `outputs/schema_sweep/lp_frozen_lm_s3_state_uncertainty/metrics_final.json`
- `outputs/schema_sweep/lp_frozen_lm_s3_state_uncertainty/best.pt`
- `outputs/schema_sweep/lp_frozen_lm_s3_state_uncertainty/final.pt`
- `outputs/schema_sweep/lp_frozen_lm_s3_state_uncertainty/step_3000.pt`

### 停止条件

- Stop if S3 source `best.pt` is missing or cannot load.
- Stop if S3 LP `final.pt` already exists before launch.
- Stop if the wrapper exits nonzero or the log shows checkpoint-load, CUDA/OOM, dataset, or metric-export failure.
- Stop after S3 LP completion and write results back before considering frozen-LM S1.
- Stop if any summary describes S3 source alone as S3 LP evidence.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S3_LP_FORMAL_RUN 执行后记录

### 结果

- S3 dependent LP completed normally with `EXITCODE 0`.
- `outputs/schema_sweep/lp_frozen_lm_s3_state_uncertainty/final.pt` exists.
- `outputs/schema_sweep/lp_frozen_lm_s3_state_uncertainty/metrics_final.json` exists.
- `outputs/schema_sweep/lp_frozen_lm_s3_state_uncertainty/step_3000.pt` exists.
- GPU0/GPU1 returned to idle after completion.
- No Python training process remains.

### 命令与输入输出

```powershell
wmic process call create "cmd.exe /c H:\Xiyao_Wang\021_260129VIVID\scripts\run_schema_lp_frozen_lm_s3_gpu0.cmd"
Get-Content outputs\logs\schema_lp_frozen_lm_s3_gpu0.log -Tail 80
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit --format=csv
python -c "import json; from pathlib import Path; base=Path('outputs/schema_sweep/lp_frozen_lm_s3_state_uncertainty'); final=json.loads((base/'metrics_final.json').read_text()); print(final['val_loss'], final['metrics']['macro_auc'], final['metrics']['macro_f1'], final['metrics']['micro_f1'])"
```

Inputs:

- `configs/schema_sweep/lp_frozen_lm_s3_state_uncertainty.yaml`
- `outputs/schema_sweep/frozen_lm_s3_state_uncertainty/checkpoints/best.pt`
- `scripts/train_vit_baseline.py`
- `scripts/run_schema_lp_frozen_lm_s3_gpu0.cmd`

Outputs:

- `outputs/logs/schema_lp_frozen_lm_s3_gpu0.log`
- `outputs/schema_sweep/lp_frozen_lm_s3_state_uncertainty/metrics_final.json`
- `outputs/schema_sweep/lp_frozen_lm_s3_state_uncertainty/best.pt`
- `outputs/schema_sweep/lp_frozen_lm_s3_state_uncertainty/final.pt`
- `outputs/schema_sweep/lp_frozen_lm_s3_state_uncertainty/step_3000.pt`

### 指标

| metric | value |
|---|---:|
| runtime | `05:23` |
| final val_loss | `0.2650233433833198` |
| final macro_auc | `0.8187573849298394` |
| final macro_f1 | `0.9108463687663829` |
| final micro_f1 | `0.8866871336868546` |
| best-loss step | `2800` |
| best-loss val_loss | `0.26396165551647305` |
| best-loss macro_auc | `0.8207804403879092` |
| best-AUC step | `800` |
| best-AUC macro_auc | `0.8248514807767141` |
| exit code | `0` |

### 失败原因 / 边界

- No runtime failure.
- LP memory/power usage was low relative to frozen-LM source:
  - about `1.9 GB` GPU memory during training
  - about `10-13 it/s` training speed
- This is downstream binary LP evidence from frozen-LM S3 source `best.pt`.
- It must not be mixed with 12-label source-token validation loss.
- This closes the focused frozen-LM S3 source+LP pair, but does not close frozen-LM S1 source+LP.

### 下一步

- Integrate S3 LP into schema diagnostic summary, execution status, completion gap audit, and Phase 4 synthesis.
- After integration, the remaining formal schema-complexity gap is frozen-LM S1 source+LP unless explicitly scoped out.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S3_LP_INTEGRATION 执行前记录

### 计划

1. Add `frozen_lm_s3_state_uncertainty_formal_lp` to `schema_complexity_diagnostic_summary`.
2. Add `P1_SCHEMA_COMPLEXITY_FROZEN_LM_S3_LP_FORMAL_RUN` to `revision_execution_status`.
3. Update gap audit and Phase 4 synthesis from `frozen-LM S3 LP missing` to `frozen-LM S2/S3 source+LP complete; frozen-LM S1 source+LP missing`.
4. Regenerate final tables.
5. Run targeted searches to ensure no output still treats S3 LP as missing.

### 命令

```powershell
python -m py_compile scripts\summarize_frozen_lm_source_training.py scripts\summarize_schema_complexity_diagnostics.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py
python scripts\summarize_frozen_lm_source_training.py
python scripts\summarize_schema_complexity_diagnostics.py
python scripts\summarize_revision_execution_status.py
python scripts\summarize_revision_completion_gap_audit.py
python scripts\summarize_phase4_revision_synthesis.py
Select-String -Path outputs\final_tables\schema_complexity_diagnostic_summary.csv -Pattern 'frozen_lm_s3_state_uncertainty_formal_lp'
Select-String -Path outputs\final_tables\revision_execution_status.csv -Pattern 'P1_SCHEMA_COMPLEXITY_FROZEN_LM_S3_LP_FORMAL_RUN'
Select-String -Path outputs\final_tables\revision_completion_gap_audit.csv -Pattern 'frozen-LM S1 source\+LP|frozen-LM S2/S3 source\+LP|frozen-LM S3 LP'
Select-String -Path outputs\final_tables\llm_necessity.csv -Pattern 'frozen S3 source val_loss|S3 LP|S1 rows are missing'
Select-String -Path outputs\final_tables\phase4_writing_claim_checklist.md -Pattern 'frozen-LM S1 formal|matched frozen-LM/no-LM S1/S2/S3|frozen-LM S3 LP'
```

### 输入

- `outputs/schema_sweep/lp_frozen_lm_s3_state_uncertainty/metrics_final.json`
- `outputs/schema_sweep/lp_frozen_lm_s3_state_uncertainty/best.pt`
- `outputs/schema_sweep/lp_frozen_lm_s3_state_uncertainty/final.pt`
- `outputs/schema_sweep/lp_frozen_lm_s3_state_uncertainty/step_3000.pt`
- `outputs/logs/schema_lp_frozen_lm_s3_gpu0.log`

### 预期输出

- Updated `outputs/final_tables/schema_complexity_diagnostic_summary.csv`
- Updated `outputs/final_tables/revision_execution_status.csv`
- Updated `outputs/final_tables/revision_completion_gap_audit.csv`
- Updated `outputs/final_tables/llm_necessity.csv`
- Updated `outputs/final_tables/module_candidates.csv`
- Updated `outputs/final_tables/phase4_writing_claim_checklist.md`

### 停止条件

- Stop if S3 LP metrics cannot be parsed.
- Stop if S3 LP metrics are merged into source-token loss.
- Stop if final tables say full matched frozen-LM/no-LM S1/S2/S3 schema sweep is complete.
- Stop if final tables still say frozen-LM S3 LP is missing.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S3_LP_INTEGRATION 执行后记录

### 结果

- `outputs/final_tables/schema_complexity_diagnostic_summary.csv` now has 16 rows.
- It includes `frozen_lm_s3_state_uncertainty_formal_lp`.
- `outputs/final_tables/revision_execution_status.csv` now has 43 rows.
- It includes `P1_SCHEMA_COMPLEXITY_FROZEN_LM_S3_LP_FORMAL_RUN`.
- `outputs/final_tables/revision_completion_gap_audit.csv` now states:
  - no-LM S1/S2/S3 source+LP complete
  - frozen-LM S2/S3 source+LP complete
  - frozen-LM S1 source+LP missing
- `outputs/final_tables/llm_necessity.csv` now reports:
  - frozen S3 source val_loss `0.030576`
  - frozen S3 LP macro_auc `0.818757`
- `outputs/final_tables/phase4_writing_claim_checklist.md` still forbids claiming completed matched frozen-LM/no-LM S1/S2/S3 source-plus-LP comparison.

### 指标

| output | value |
|---|---:|
| schema diagnostic rows | `16` |
| execution status rows | `43` |
| gap audit rows | `12` |
| S3 LP final macro_auc | `0.818757` |
| S3 LP best-loss macro_auc | `0.820780` |
| S3 LP best-AUC macro_auc | `0.824851` |

### 失败原因 / 边界

- No integration-script failure.
- Final tables no longer treat frozen-LM S3 LP as missing.
- Full matched schema sweep remains incomplete because frozen-LM S1 source+LP is missing.
- The safe next decision is whether to run frozen-LM S1 source+LP or explicitly scope it out.

### 下一步

- Do not launch frozen-LM S1 automatically.
- If the paper needs a full matched S1/S2/S3 formal schema sweep, write a fresh execution-before record for frozen-LM S1 source.
- If not, explicitly scope out frozen-LM S1 and use the focused S2/S3 matched evidence plus no-LM S1/S2/S3 evidence.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S1_SOURCE_GATED_RUN 执行前记录

### 计划

1. Run only frozen-LM S1 `state_only` formal source.
2. Use `configs/schema_sweep/frozen_lm_s1_state_only.yaml`.
3. Run on GPU0 via `scripts/run_schema_frozen_lm_s1_source_gpu0.cmd`.
4. Keep the configured `batch_size=4`, `gradient_accumulation_steps=8`, `num_workers=0`, and `bf16=true`.
5. Apply the same gated protocol as S2/S3:
   - early log/checkpoint/GPU check after launch
   - stop if load, OOM, dataset, or abnormal sustained speed failure appears
   - do not launch S1 LP until source finishes and is integrated
6. Do not run frozen-LM S1 LP, another frozen-LM source job, or any SPD variant concurrently.

### 预检命令与结果

```powershell
Test-Path configs\schema_sweep\frozen_lm_s1_state_only.yaml
Test-Path outputs\schema_sweep\frozen_lm_s1_state_only\checkpoints\final.pt
Get-ChildItem outputs\schema_sweep\frozen_lm_s1_state_only -Recurse -File -ErrorAction SilentlyContinue
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit --format=csv
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime,Path
```

Preflight outputs:

- S1 source config exists: `True`.
- S1 formal source `final.pt`: `False`, so no completed formal result will be overwritten.
- Existing S1 formal source output files: none found.
- GPU0: `0%`, `13 MiB / 24576 MiB`, about `12.89 W / 350 W`.
- GPU1: `0%`, `13 MiB / 24576 MiB`, about `8.19 W / 350 W`.
- Active Python process check: no Python training process found.
- Created wrapper:
  - `scripts/run_schema_frozen_lm_s1_source_gpu0.cmd`

### 执行命令

```powershell
wmic process call create "cmd.exe /c H:\Xiyao_Wang\021_260129VIVID\scripts\run_schema_frozen_lm_s1_source_gpu0.cmd"
Get-Content outputs\logs\schema_frozen_lm_s1_source_gpu0.log -Tail 80
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit --format=csv
```

### 输入

- `configs/schema_sweep/frozen_lm_s1_state_only.yaml`
- `data/splits/chexpert_train_30k.jsonl`
- `data/splits/chexpert_val_fixed.jsonl`
- `scripts/train_cxr.py`
- `scripts/run_schema_frozen_lm_s1_source_gpu0.cmd`

### 预期输出

- `outputs/logs/schema_frozen_lm_s1_source_gpu0.log`
- `outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/best.pt`
- `outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/step_1000.pt`
- `outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/step_10000.pt`
- `outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/final.pt`

### 停止条件

- Stop if S1 formal final checkpoint already exists before launch.
- Stop if model/tokenizer load fails.
- Stop if CUDA OOM, dataset read failure, checkpoint-save failure, or nonzero wrapper exit appears.
- Stop if GPU utilization is persistently low with abnormal slow step time after warmup.
- Stop after source completion; do not launch S1 LP without a new execution-before record.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S1_SOURCE_GATED_RUN early runtime 记录

### 结果

- S1 formal frozen-LM source launched on GPU0.
- Model/tokenizer/data loading completed.
- Training started successfully.
- No checkpoint has been written yet at this early warmup point.
- GPU1 remains idle.

### 命令与输入输出

```powershell
wmic process call create "cmd.exe /c H:\Xiyao_Wang\021_260129VIVID\scripts\run_schema_frozen_lm_s1_source_gpu0.cmd"
Get-Content outputs\logs\schema_frozen_lm_s1_source_gpu0.log -Tail 100
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit --format=csv
Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -eq 20368 -or ($_.CommandLine -match 'frozen_lm_s1|run_schema_frozen_lm_s1|train_cxr') }
```

Outputs:

- WMI ProcessId: `20368`.
- Python training ProcessId: `17488`.
- Log confirms:
  - 29000 train samples loaded
  - 1000 fixed-val samples loaded
  - Qwen/Qwen2.5-1.5B-Instruct loaded
  - frozen LLM parameters: `1,543,714,304`
  - trainable parameters: `89,349,888`
  - max steps: `10000`
- Early training speed: about `2.8-3.0s/it`.
- GPU0: `89%`, `24265 MiB / 24576 MiB`, about `330.92 W / 350 W`.
- GPU1: `0%`, `13 MiB / 24576 MiB`.

### 指标 / 当前速度判断

| metric | value |
|---|---:|
| current visible step | `26/10000` |
| step time | about `2.8-3.0s/it` |
| GPU0 memory | `24265 MiB / 24576 MiB` |
| GPU0 power | `330.92 W / 350 W` |
| GPU1 memory | `13 MiB / 24576 MiB` |

### 失败原因 / 边界

- No load/runtime failure in warmup.
- Speed is consistent with prior frozen-LM S2/S3 source runs.
- Memory is near the GPU0 limit, so do not run another high-memory frozen-LM source concurrently.
- This is only an early runtime gate, not a completed source result.

### 下一步

- Continue monitoring to the first validation/checkpoint gate around step `500`.
- Stop immediately if OOM, nonzero exit, or sustained abnormal slowdown appears.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S1_SOURCE_GATED_RUN 500-step gate 记录

### 结果

- S1 formal frozen-LM source passed the first validation/checkpoint gate.
- Validation at step 500 completed across `250/250` fixed-val batches.
- `outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/best.pt` was created.
- Training continued after validation and recovered normal step speed.
- GPU0 remains the only active GPU; GPU1 is idle.

### 命令与输入输出

```powershell
Get-Content outputs\logs\schema_frozen_lm_s1_source_gpu0.log -Tail 160
python -c "import torch; p='outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/best.pt'; ck=torch.load(p,map_location='cpu'); print({k: ck.get(k) for k in ['global_step','best_val_loss','epoch']})"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit --format=csv
```

Outputs:

- `best.pt`: `global_step=500`, `best_val_loss=0.05026683175563812`.
- Validation speed: about `6.4-6.6 it/s`.
- Training resumed past step `534`.
- Post-validation recovered speed: about `2.78-2.80s/it`.
- GPU0: about `99%`, `24267 MiB / 24576 MiB`, `336.61 W / 350 W`.
- GPU1: `0%`, `13 MiB / 24576 MiB`.

### 指标 / 当前速度判断

| metric | value |
|---|---:|
| latest checkpoint | `best.pt` |
| checkpoint step | `500` |
| best_val_loss | `0.05026683175563812` |
| latest visible step | `534/10000` |
| post-validation step time | about `2.78-2.80s/it` |
| GPU0 memory | `24267 MiB / 24576 MiB` |
| GPU0 power | `336.61 W / 350 W` |

### 失败原因 / 边界

- No runtime failure through step 500.
- The temporary slow-looking progress entries immediately after validation are validation overhead being amortized into the progress bar; speed recovered normally.
- This remains high-memory source training and should stay single-GPU.
- This is not a completed source run yet; no `final.pt` or `step_10000.pt` exists.

### 下一步

- Continue S1 source on GPU0.
- Next checkpoint/health target: step `1000`.
- Do not launch S1 LP until source completion and integration are recorded.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S1_SOURCE_GATED_RUN 1000-step 记录

### 结果

- S1 formal frozen-LM source passed the 1000-step checkpoint.
- Validation at step 1000 completed across `250/250` fixed-val batches.
- `outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/step_1000.pt` was created.
- `best.pt` was updated at step 1000.
- Training continued after validation and recovered normal step speed.

### 命令与输入输出

```powershell
Get-Content outputs\logs\schema_frozen_lm_s1_source_gpu0.log -Tail 160
python -c "import torch; paths=['outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/best.pt','outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/step_1000.pt']; [print(p, {k: torch.load(p,map_location='cpu').get(k) for k in ['global_step','best_val_loss','epoch']}) for p in paths]"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit --format=csv
```

Outputs:

- `best.pt`: `global_step=1000`, `best_val_loss=0.04403838207572699`.
- `step_1000.pt`: `global_step=1000`, `best_val_loss=0.04403838207572699`.
- Validation speed: about `6.4-6.6 it/s`.
- Training resumed past step `1057`.
- Post-validation recovered speed: about `2.78-2.81s/it`.
- GPU0: about `97%`, `24267 MiB / 24576 MiB`, `329.85 W / 350 W`.
- GPU1: `0%`, `13 MiB / 24576 MiB`.

### 指标 / 当前速度判断

| metric | value |
|---|---:|
| latest named checkpoint | `step_1000.pt` |
| current best checkpoint step | `1000` |
| current best_val_loss | `0.04403838207572699` |
| latest visible step | `1057/10000` |
| post-validation step time | about `2.78-2.81s/it` |
| GPU0 memory | `24267 MiB / 24576 MiB` |
| GPU0 power | `329.85 W / 350 W` |

### 失败原因 / 边界

- No runtime failure through step 1000.
- Validation overhead again caused temporary slow progress-bar estimates, then speed recovered.
- This remains high-memory/high-power single-GPU source training.
- This is not a completed source row yet; no `final.pt` or `step_10000.pt` exists.

### 下一步

- Continue S1 source on GPU0.
- Next checkpoint/health target: step `2000`.
- Do not launch S1 LP until source completion and integration are recorded.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S1_SOURCE_GATED_RUN 2000-5000-step 记录

### 结果

- S1 formal frozen-LM source passed named checkpoints through step 5000.
- `step_2000.pt`, `step_3000.pt`, `step_4000.pt`, and `step_5000.pt` exist.
- `best.pt` was updated through step 5000.
- Training remains active on GPU0 only.
- GPU1 remains idle.

### 命令与输入输出

```powershell
Get-ChildItem outputs\schema_sweep\frozen_lm_s1_state_only\checkpoints -File
python -c "import torch; paths=['outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/best.pt','outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/step_2000.pt','outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/step_3000.pt','outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/step_4000.pt','outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/step_5000.pt']; [print(p, {k: torch.load(p,map_location='cpu').get(k) for k in ['global_step','best_val_loss','epoch']}) for p in paths]"
Select-String -Path outputs\logs\schema_frozen_lm_s1_source_gpu0.log -Pattern 'Step 2000: val_loss|Step 3000: val_loss|Step 4000: val_loss'
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit --format=csv
```

Outputs:

- `step_2000.pt`: `global_step=2000`, `best_val_loss=0.04071743554621935`.
- `step_3000.pt`: `global_step=3000`, `best_val_loss=0.039485231019556526`.
- `step_4000.pt`: `global_step=4000`, `best_val_loss=0.03763267385214567`.
- `step_5000.pt`: `global_step=5000`, `best_val_loss=0.037151597172021864`.
- `best.pt`: `global_step=5000`, `best_val_loss=0.037151597172021864`.
- Log text confirms:
  - `Step 2000: val_loss = 0.0407`
  - `Step 3000: val_loss = 0.0395`
  - `Step 4000: val_loss = 0.0376`
- GPU0 during post-5000 training: about `23961 MiB / 24576 MiB`, about `326.51 W / 350 W`.

### 指标 / 当前速度判断

| metric | value |
|---|---:|
| latest named checkpoint | `step_5000.pt` |
| current best checkpoint step | `5000` |
| current best_val_loss | `0.037151597172021864` |
| latest visible step during check | about `5456/10000` |
| typical post-validation step time | about `2.83-2.90s/it` |
| GPU0 memory | about `23961 MiB / 24576 MiB` |
| GPU0 power | about `326.51 W / 350 W` |

### 失败原因 / 边界

- No runtime failure through step 5000.
- Checkpoint metadata shows validation loss improved from step 2000 through step 5000.
- This remains a running source job; no `final.pt` or `step_10000.pt` exists yet.
- Keep as single-GPU high-memory source training.

### 下一步

- Continue S1 source on GPU0.
- Next checkpoint/health target: step `6000` or `7000` depending on monitoring interval.
- Do not launch S1 LP until source completion and integration are recorded.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S1_SOURCE_GATED_RUN 6000-step 记录

### 结果

- S1 formal frozen-LM source passed the 6000-step checkpoint.
- `outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/step_6000.pt` was created.
- `best.pt` was updated at step 6000.
- Training continued after validation and recovered normal speed.
- GPU0 remains active; GPU1 remains idle.

### 命令与输入输出

```powershell
Get-Content outputs\logs\schema_frozen_lm_s1_source_gpu0.log -Tail 160
python -c "import torch; paths=['outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/best.pt','outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/step_6000.pt']; [print(p, {k: torch.load(p,map_location='cpu').get(k) for k in ['global_step','best_val_loss','epoch']}) for p in paths]"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit --format=csv
```

Outputs:

- `best.pt`: `global_step=6000`, `best_val_loss=0.03637683048099279`.
- `step_6000.pt`: `global_step=6000`, `best_val_loss=0.03637683048099279`.
- `step_6000.pt` file time: `2026/6/26 18:59:55`.
- Post-validation recovered speed: about `2.85-2.90s/it`.
- GPU0: about `99%`, `23961 MiB / 24576 MiB`, `325.38 W / 350 W`.
- GPU1: `0%`, `0 MiB / 24576 MiB`.

### 指标 / 当前速度判断

| metric | value |
|---|---:|
| latest named checkpoint | `step_6000.pt` |
| current best checkpoint step | `6000` |
| current best_val_loss | `0.03637683048099279` |
| latest visible step during check | about `6100/10000` |
| post-validation step time | about `2.85-2.90s/it` |
| GPU0 memory | about `23961 MiB / 24576 MiB` |
| GPU0 power | about `325.38 W / 350 W` |

### 失败原因 / 边界

- No runtime failure through step 6000.
- Checkpoint metadata confirms continued validation-loss improvement.
- This remains a running source job; no `final.pt` or `step_10000.pt` exists yet.
- Continue single-GPU policy because GPU0 memory remains near full.

### 下一步

- Continue S1 source on GPU0.
- Next checkpoint/health target: step `7000`.
- Do not launch S1 LP until source completion and integration are recorded.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S1_SOURCE_GATED_RUN 7000-step record

### Plan

- Keep the current S1 frozen-LM source run on GPU0; do not start S1 LP or any new SPD variant.
- Verify the 7000-step checkpoint, current best checkpoint, log speed, GPU state, and possible external process contention.
- Treat this as a health gate only; the source row is not complete until `step_10000.pt` and `final.pt` exist.

### Commands / Inputs / Outputs / Stop Conditions

```powershell
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|cmd|powershell' } | Select-Object ProcessId,Name,CommandLine
Get-Content outputs\logs\schema_frozen_lm_s1_source_gpu0.log -Tail 120
python -c "from pathlib import Path; import torch; ckpt_dir=Path('outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints'); names=['best.pt','step_6000.pt','step_7000.pt','step_8000.pt','final.pt']; [print(name, 'MISSING' if not (ckpt_dir/name).exists() else {'global_step': torch.load(ckpt_dir/name, map_location='cpu').get('global_step'), 'best_val_loss': torch.load(ckpt_dir/name, map_location='cpu').get('best_val_loss')}) for name in names]"
```

Inputs:

- Config: `configs/schema_sweep/frozen_lm_s1_state_only.yaml`.
- Log: `outputs/logs/schema_frozen_lm_s1_source_gpu0.log`.
- Checkpoints: `outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/`.

Expected outputs:

- `step_7000.pt` checkpoint metadata.
- Log tail showing post-validation training speed.
- GPU/process table showing whether any non-VIVID process is using the GPUs.

Stop conditions:

- Stop or intervene only on traceback, OOM, stalled log/checkpoint writes, missing checkpoint after validation, or unknown high-utilization process contention.
- Otherwise continue to the 8000-step gate.

### Results

- S1 formal frozen-LM source passed the 7000-step checkpoint.
- `outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/step_7000.pt` exists.
- `best.pt` remained at step 6000, so step 7000 did not improve the best validation loss.
- Training continued after validation and recovered normal speed.
- GPU0 remains the active VIVID training GPU.
- GPU1 is not idle: it has an external Python process from another repo, not a VIVID job.

Outputs:

- `best.pt`: `global_step=6000`, `best_val_loss=0.03637683048099279`.
- `step_6000.pt`: `global_step=6000`, `best_val_loss=0.03637683048099279`.
- `step_7000.pt`: `global_step=7000`, `best_val_loss=0.03637683048099279`.
- `step_8000.pt`: missing, as expected at this gate.
- `final.pt`: missing, as expected before completion.
- Latest visible log step during check: about `7157/10000`.
- Post-validation recovered speed: about `2.85-2.90s/it`.
- GPU0: `98%`, `23961 MiB / 24576 MiB`, `325.17 W / 350 W`, `73 C`.
- GPU1: `0%`, `2917 MiB / 24576 MiB`, `97.53 W / 350 W`, `41 C`.
- GPU0 compute process: PID `17488`, `python scripts\train_cxr.py --config configs\schema_sweep\frozen_lm_s1_state_only.yaml`.
- GPU1 compute process: PID `14692`, `H:\Xiyao_Wang\022_tooth9\Final\run_classification.py ... --mask-aware-training`, parent PID `21076` (`Final\run_20260626_meiwenti0626_workflow.py`).

### Metrics / Speed Diagnosis

| metric | value |
|---|---:|
| latest named checkpoint | `step_7000.pt` |
| current best checkpoint step | `6000` |
| current best_val_loss | `0.03637683048099279` |
| latest visible step during check | about `7157/10000` |
| post-validation step time | about `2.85-2.90s/it` |
| GPU0 memory | `23961 MiB / 24576 MiB` |
| GPU0 power | `325.17 W / 350 W` |
| GPU1 non-VIVID process | PID `14692`, tooth9 classification |

### Failure Reason / Boundary

- No runtime failure through step 7000.
- The temporary slow steps around validation recovery are consistent with earlier gates; speed returned to the same normal band.
- No evidence of an unknown malware-like GPU process from the inspected command lines.
- There is a separate GPU1 Python training job from `H:\Xiyao_Wang\022_tooth9`, so GPU1 should not be assumed free for VIVID until that process exits.
- This remains a running source job; no `step_10000.pt` or `final.pt` exists yet.

### Next Step

- Continue S1 source on GPU0.
- Next checkpoint/health target: step `8000`.
- Do not launch S1 LP until source completion and integration are recorded.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S1_SOURCE_GATED_RUN 8000-step record

### Plan

- Verify that the S1 frozen-LM source run reached the 8000-step checkpoint cleanly.
- Extract `best.pt` and `step_8000.pt` metadata, log speed, validation behavior, and GPU/process state.
- Continue the same single-source run if no failure or stall is observed.

### Commands / Inputs / Outputs / Stop Conditions

```powershell
Get-Content outputs\logs\schema_frozen_lm_s1_source_gpu0.log -Tail 90
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
python -c "from pathlib import Path; import torch; ckpt_dir=Path('outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints'); names=['best.pt','step_7500.pt','step_8000.pt','step_9000.pt','final.pt']; [print(name, 'MISSING' if not (ckpt_dir/name).exists() else {'global_step': torch.load(ckpt_dir/name, map_location='cpu').get('global_step'), 'best_val_loss': torch.load(ckpt_dir/name, map_location='cpu').get('best_val_loss')}) for name in names]"
```

Inputs:

- Config: `configs/schema_sweep/frozen_lm_s1_state_only.yaml`.
- Log: `outputs/logs/schema_frozen_lm_s1_source_gpu0.log`.
- Checkpoints: `outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/`.

Expected outputs:

- `step_8000.pt` exists and has `global_step=8000`.
- `best.pt` metadata indicates whether step 8000 improved over the prior best.
- GPU/process table confirms no unknown contention.

Stop conditions:

- Stop or intervene only on traceback, OOM, stalled log/checkpoint writes, missing 8000 checkpoint, or unknown high-utilization process contention.
- Otherwise continue to the 9000-step gate.

### Results

- S1 formal frozen-LM source passed the 8000-step checkpoint.
- Validation at step 8000 completed across `250/250` fixed-val batches.
- `outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/step_8000.pt` exists.
- `best.pt` was updated at step 8000.
- Training continued after validation and recovered normal speed.
- GPU1 is now idle, but no dependent S1 LP run is started before source completion and integration.

Outputs:

- `best.pt`: `global_step=8000`, `best_val_loss=0.035852200925350186`.
- `step_8000.pt`: `global_step=8000`, `best_val_loss=0.035852200925350186`.
- `step_7500.pt`: missing; expected because 7500 was an eval/best update, not a named save checkpoint.
- `step_9000.pt`: missing, as expected at this gate.
- `final.pt`: missing, as expected before completion.
- Latest visible log step during check: about `8043/10000`.
- Validation speed: about `4.3-4.5 it/s` for most validation batches, `250/250` in about `56s`.
- Post-validation recovered speed: about `2.81-3.0s/it`.
- GPU0: `88%`, `23961 MiB / 24576 MiB`, `336.44 W`, `73 C`.
- GPU1: `0%`, `0 MiB / 24576 MiB`, no compute app.
- GPU0 compute process: PID `17488`, `python scripts\train_cxr.py --config configs\schema_sweep\frozen_lm_s1_state_only.yaml`.

### Metrics / Speed Diagnosis

| metric | value |
|---|---:|
| latest named checkpoint | `step_8000.pt` |
| current best checkpoint step | `8000` |
| current best_val_loss | `0.035852200925350186` |
| prior best at step 7500 | `0.03601991929113865` |
| latest visible step during check | about `8043/10000` |
| post-validation step time | about `2.81-3.0s/it` |
| GPU0 memory | `23961 MiB / 24576 MiB` |
| GPU0 power | `336.44 W` |
| GPU1 state | idle |

### Failure Reason / Boundary

- No runtime failure through step 8000.
- The progress bar again showed temporary slow estimates immediately after validation; speed recovered normally.
- Step 8000 improved the best validation loss, so the run is still scientifically useful rather than merely burning compute.
- This remains a running source job; no `step_10000.pt` or `final.pt` exists yet.

### Next Step

- Continue S1 source on GPU0.
- Next checkpoint/health target: step `9000`.
- Do not launch S1 LP until source completion and integration are recorded.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S1_SOURCE_GATED_RUN 9000-step record

### Plan

- Verify that the S1 frozen-LM source run reached the 9000-step checkpoint cleanly.
- Extract checkpoint metadata, validation behavior, log speed, and GPU/process state.
- Continue to the final 10000-step completion gate if no failure is observed.

### Commands / Inputs / Outputs / Stop Conditions

```powershell
Get-Content outputs\logs\schema_frozen_lm_s1_source_gpu0.log -Tail 90
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
python -c "from pathlib import Path; import torch; ckpt_dir=Path('outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints'); names=['best.pt','step_9000.pt','step_10000.pt','final.pt']; [print(name, 'MISSING' if not (ckpt_dir/name).exists() else {'global_step': torch.load(ckpt_dir/name, map_location='cpu').get('global_step'), 'best_val_loss': torch.load(ckpt_dir/name, map_location='cpu').get('best_val_loss')}) for name in names]"
```

Inputs:

- Config: `configs/schema_sweep/frozen_lm_s1_state_only.yaml`.
- Log: `outputs/logs/schema_frozen_lm_s1_source_gpu0.log`.
- Checkpoints: `outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/`.

Expected outputs:

- `step_9000.pt` exists and has `global_step=9000`.
- `best.pt` metadata indicates whether step 9000 improved over the prior best.
- GPU/process table confirms no unknown contention.

Stop conditions:

- Stop or intervene only on traceback, OOM, stalled log/checkpoint writes, missing 9000 checkpoint, or unknown high-utilization process contention.
- Otherwise continue to the 10000-step completion gate.

### Results

- S1 formal frozen-LM source passed the 9000-step checkpoint.
- Validation at step 9000 completed across `250/250` fixed-val batches.
- `outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/step_9000.pt` exists.
- `best.pt` was updated at step 9000.
- Training continued after validation and recovered normal speed.
- GPU1 remains idle; no new dependent LP run was launched before source completion.

Outputs:

- `best.pt`: `global_step=9000`, `best_val_loss=0.035434496022760865`.
- `step_9000.pt`: `global_step=9000`, `best_val_loss=0.035434496022760865`.
- `step_10000.pt`: missing, as expected at this gate.
- `final.pt`: missing, as expected before completion.
- Latest visible log step during check: about `9048/10000`.
- Validation speed: about `4.5-4.6 it/s` for most validation batches, `250/250` in about `54s`.
- Post-validation recovered speed: about `2.79-3.0s/it`.
- GPU0: `91%`, `23961 MiB / 24576 MiB`, `336.95 W`, `73 C`.
- GPU1: `0%`, `0 MiB / 24576 MiB`, no compute app.
- GPU0 compute process: PID `17488`, `python scripts\train_cxr.py --config configs\schema_sweep\frozen_lm_s1_state_only.yaml`.

### Metrics / Speed Diagnosis

| metric | value |
|---|---:|
| latest named checkpoint | `step_9000.pt` |
| current best checkpoint step | `9000` |
| current best_val_loss | `0.035434496022760865` |
| prior best at step 8500 | `0.035842554092407226` |
| latest visible step during check | about `9048/10000` |
| post-validation step time | about `2.79-3.0s/it` |
| GPU0 memory | `23961 MiB / 24576 MiB` |
| GPU0 power | `336.95 W` |
| GPU1 state | idle |

### Failure Reason / Boundary

- No runtime failure through step 9000.
- The progress bar again showed temporary slow estimates immediately after validation; speed recovered normally.
- Step 9000 materially improved the best validation loss relative to step 8500.
- This remains a running source job; no `step_10000.pt` or `final.pt` exists yet.

### Next Step

- Continue S1 source on GPU0 through final step `10000`.
- At completion, verify `step_10000.pt`, `final.pt`, `best.pt`, wrapper exit code, and log tail.
- Then integrate S1 source into the frozen-LM source summary and schema/final-table scripts before launching S1 LP.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S1_SOURCE_GATED_RUN completion record

### Plan

- Confirm the S1 frozen-LM source run completed with exit code 0.
- Verify `best.pt`, `step_10000.pt`, `final.pt`, final log tail, and GPU release.
- Record the source result before any S1 LP launch.

### Commands / Inputs / Outputs / Stop Conditions

```powershell
Get-Content outputs\logs\schema_frozen_lm_s1_source_gpu0.log -Tail 140
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
python -c "from pathlib import Path; import torch; ckpt_dir=Path('outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints'); names=['best.pt','step_9000.pt','step_10000.pt','final.pt']; [print(name, {'global_step': torch.load(ckpt_dir/name, map_location='cpu').get('global_step'), 'best_val_loss': torch.load(ckpt_dir/name, map_location='cpu').get('best_val_loss')}) for name in names]"
```

Inputs:

- Config: `configs/schema_sweep/frozen_lm_s1_state_only.yaml`.
- Log: `outputs/logs/schema_frozen_lm_s1_source_gpu0.log`.
- Checkpoints: `outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/`.

Expected outputs:

- Log shows `Training completed!`, `EXITCODE 0`, and wrapper `END`.
- `step_10000.pt` and `final.pt` exist with `global_step=10000`.
- `best.pt` identifies the best source checkpoint.
- GPUs are released after process exit.

Stop conditions:

- Stop and diagnose if wrapper exit code is nonzero, final checkpoints are missing, or GPU/process state indicates the job is still running unexpectedly.
- Otherwise proceed to source summary integration.

### Results

- S1 formal frozen-LM source completed successfully.
- Wrapper log shows `EXITCODE 0`.
- Wrapper log end time: `2026/06/26 周五 22:18:27.25`.
- Final validation at step 10000 completed across `250/250` fixed-val batches.
- `outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/step_10000.pt` exists.
- `outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/final.pt` exists.
- `best.pt` remained at step 9000; step 10000 did not improve the best validation loss.
- Both GPUs were released after completion.

Outputs:

- `best.pt`: `global_step=9000`, `best_val_loss=0.035434496022760865`.
- `step_9000.pt`: `global_step=9000`, `best_val_loss=0.035434496022760865`.
- `step_10000.pt`: `global_step=10000`, `best_val_loss=0.035434496022760865`.
- `final.pt`: `global_step=10000`, `best_val_loss=0.035434496022760865`.
- Final log: `Step 10000: val_loss = 0.0354`.
- Final log: `Checkpoint saved: ...\step_10000.pt`.
- Final log: `Checkpoint saved: ...\final.pt`.
- Final log: `Training completed!`.
- GPU0 after completion: `0%`, `0 MiB / 24576 MiB`, `13.45 W`, `38 C`.
- GPU1 after completion: `0%`, `0 MiB / 24576 MiB`, `7.67 W`, `31 C`.

### Metrics / Source Result

| metric | value |
|---|---:|
| final checkpoint | `final.pt` |
| final global_step | `10000` |
| best checkpoint | `best.pt` |
| best checkpoint step | `9000` |
| best_val_loss | `0.035434496022760865` |
| final visible val_loss | `0.0354` |
| wrapper exit code | `0` |
| final GPU state | both GPUs idle |

### Failure Reason / Boundary

- No runtime failure in the completed S1 source run.
- Step 9500 and step 10000 did not improve over the step 9000 best, but final checkpoint metadata is complete.
- This record completes the S1 frozen-LM source row only.
- S1 LP remains unrun and must not be marked complete until a separate LP execution-before record, run, and integration are done.

### Next Step

- Integrate S1 source into `frozen_lm_source_training_summary.csv`, schema diagnostics, revision execution status, gap audit, and phase-4 synthesis.
- Verify summary scripts compile/run.
- Then create a separate execution-before record for S1 LP and launch it on an idle GPU.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S1_SOURCE_INTEGRATION result

### Plan

- Add the completed S1 frozen-LM source row to source-training summary and downstream schema/status/gap/synthesis tables.
- Preserve the boundary that S1 source is not S1 LP.
- Verify the scripts compile, regenerate final tables, and confirm no stale claim says S1 source is missing.

### Commands / Inputs / Outputs / Stop Conditions

```powershell
python -m py_compile scripts\summarize_frozen_lm_source_training.py scripts\summarize_schema_complexity_diagnostics.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py
python scripts\summarize_frozen_lm_source_training.py
python scripts\summarize_schema_complexity_diagnostics.py
python scripts\summarize_revision_execution_status.py
python scripts\summarize_revision_completion_gap_audit.py
python scripts\summarize_phase4_revision_synthesis.py
git diff --check -- scripts\summarize_frozen_lm_source_training.py scripts\summarize_schema_complexity_diagnostics.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py vivid_med_revision_execution_plan.md
```

Inputs:

- `outputs/logs/schema_frozen_lm_s1_source_gpu0.log`.
- `outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/best.pt`.
- `outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/final.pt`.
- `outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/step_10000.pt`.

Expected outputs:

- `outputs/final_tables/frozen_lm_source_training_summary.csv`.
- `outputs/final_tables/frozen_lm_source_val_loss_trace.csv`.
- `outputs/final_tables/schema_complexity_diagnostic_summary.csv`.
- `outputs/final_tables/revision_execution_status.csv`.
- `outputs/final_tables/revision_completion_gap_audit.csv`.
- `outputs/final_tables/llm_necessity.csv`.
- `outputs/final_tables/module_candidates.csv`.

Stop conditions:

- Stop if any summary script fails, if S1 source is marked failed, if S1 LP is marked complete without a run, or if formatting checks fail.

### Results

- `summarize_frozen_lm_source_training.py` now includes S1/S2/S3 frozen-LM source rows.
- Source completion logic now checks `final.pt` and `step_10000.pt` rather than requiring `best.pt` to be at step 10000; this prevents the valid S1 best-at-9000 run from being misclassified as failed.
- Summary scripts compiled and regenerated successfully.
- `frozen_lm_source_training_summary.csv`: 3 summary rows.
- `frozen_lm_source_val_loss_trace.csv`: 59 trace rows.
- `schema_complexity_diagnostic_summary.csv`: 17 rows.
- `revision_execution_status.csv`: 44 rows.
- `revision_completion_gap_audit.csv`: 12 rows.
- Phase 4 synthesis tables regenerated.
- `git diff --check` passed on the touched scripts and plan document.
- GPUs remained idle after integration verification.

### Metrics / Integrated Evidence

| artifact | S1 source state |
|---|---|
| `frozen_lm_source_training_summary.csv` | `status=completed`, `final_step=10000`, `checkpoint_best_step=9000`, `checkpoint_best_val_loss=0.035434`, `exitcode=0` |
| `schema_complexity_diagnostic_summary.csv` | `frozen_lm_s1_state_only_formal_source`, `primary_value=0.035434`, boundary says S1 LP missing |
| `revision_execution_status.csv` | `P1_SCHEMA_COMPLEXITY_FROZEN_LM_S1_FORMAL_SOURCE_RUN`, `completed_formal_source_run` |
| `revision_completion_gap_audit.csv` | schema sweep gap narrowed to frozen-LM S1 LP missing |
| `llm_necessity.csv` | schema-complexity status updated to S1/S2/S3 source + S2/S3 LP complete, S1 LP missing |

### Failure Reason / Boundary

- No integration failure.
- This integration does not complete the frozen-LM S1 LP row.
- The full matched frozen-LM/no-LM S1/S2/S3 source+LP comparison remains incomplete until S1 LP exists or is explicitly scoped out.

### Next Step

- Write a separate execution-before record for `P1_SCHEMA_COMPLEXITY_FROZEN_LM_S1_LP_FORMAL_RUN`.
- Launch S1 LP only after verifying the LP config, source `best.pt`, output path, and idle GPU state.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S1_LP_FORMAL_RUN execution-before

### Plan

- Run the dependent frozen-LM S1 LP after the S1 source run completed and was integrated.
- Use the matched fixed-split LP config with frozen S1 source ViT initialization.
- Launch only this LP row; do not start a new SPD variant or a broad matrix.

### Commands / Inputs / Outputs / Stop Conditions

Preflight commands:

```powershell
Get-Content configs\schema_sweep\lp_frozen_lm_s1_state_only.yaml
Test-Path outputs\schema_sweep\frozen_lm_s1_state_only\checkpoints\best.pt
if (Test-Path outputs\schema_sweep\lp_frozen_lm_s1_state_only) { Get-ChildItem outputs\schema_sweep\lp_frozen_lm_s1_state_only -Recurse -Depth 2 } else { 'lp_frozen_lm_s1_state_only output missing' }
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

Launch command:

```powershell
wmic process call create "cmd.exe /c H:\Xiyao_Wang\021_260129VIVID\scripts\run_schema_lp_frozen_lm_s1_gpu0.cmd"
```

Inputs:

- Config: `configs/schema_sweep/lp_frozen_lm_s1_state_only.yaml`.
- Source checkpoint: `outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/best.pt`.
- Wrapper: `scripts/run_schema_lp_frozen_lm_s1_gpu0.cmd`.

Expected outputs:

- Log: `outputs/logs/schema_lp_frozen_lm_s1_gpu0.log`.
- Output dir: `outputs/schema_sweep/lp_frozen_lm_s1_state_only`.
- Metrics: `metrics_final.json`, `metrics_step_*.json`.
- Checkpoints: `best.pt`, `final.pt`, `step_3000.pt`.

Stop conditions:

- Stop if config or source `best.pt` is missing.
- Stop if output dir already contains a completed `final.pt` from a previous run.
- Stop on nonzero wrapper exit code, traceback, OOM, stalled log/checkpoint writes, or unexpected GPU contention.
- On success, integrate S1 LP into schema diagnostics, status, gap audit, and Phase 4 synthesis.

### Preflight Results

- Config exists and points to `./outputs/schema_sweep/frozen_lm_s1_state_only/checkpoints/best.pt`.
- Source `best.pt` exists.
- LP output directory `outputs/schema_sweep/lp_frozen_lm_s1_state_only` did not exist before launch.
- GPU0: `0%`, `0 MiB / 24576 MiB`, idle.
- GPU1: `0%`, `0 MiB / 24576 MiB`, idle.
- Wrapper `scripts/run_schema_lp_frozen_lm_s1_gpu0.cmd` was created using the existing S2/S3 LP wrapper pattern.

### Boundary

- This run is a downstream binary LP run, not source training.
- S1 source loss is not a downstream performance metric.
- The full matched frozen-LM/no-LM S1/S2/S3 source+LP comparison remains incomplete until this LP finishes and is integrated.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S1_LP_FORMAL_RUN result

### Results

- S1 frozen-LM LP completed successfully.
- Wrapper log shows `EXITCODE 0`.
- Wrapper log end time: `2026/06/26 周五 22:33:24.83`.
- `outputs/schema_sweep/lp_frozen_lm_s1_state_only/metrics_final.json` exists.
- `outputs/schema_sweep/lp_frozen_lm_s1_state_only/metrics_step_3000.json` exists.
- `outputs/schema_sweep/lp_frozen_lm_s1_state_only/best.pt` exists.
- `outputs/schema_sweep/lp_frozen_lm_s1_state_only/final.pt` exists.
- `outputs/schema_sweep/lp_frozen_lm_s1_state_only/step_3000.pt` exists.
- Final validation completed across `63/63` batches.
- Both GPUs were idle after completion.

### Commands / Inputs / Outputs

```powershell
Get-Content outputs\logs\schema_lp_frozen_lm_s1_gpu0.log -Tail 140
python -c "from pathlib import Path; import json; out=Path('outputs/schema_sweep/lp_frozen_lm_s1_state_only'); print(json.loads((out/'metrics_final.json').read_text()))"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

Outputs:

- Final `val_loss=0.26912268925280797`.
- Final `macro_auc=0.7985421002758822`.
- Final `macro_f1=0.9107372982154678`.
- Final `micro_f1=0.8855707507675132`.
- Best-loss step: `2800`, `val_loss=0.26786478147620246`, `macro_auc=0.793706079134021`.
- Best-AUC step: `800`, `val_loss=0.2950224857481699`, `macro_auc=0.8113016852907395`.
- Final GPU0: `0%`, `0 MiB / 24576 MiB`.
- Final GPU1: `0%`, `0 MiB / 24576 MiB`.

### Metrics

| metric | value |
|---|---:|
| final macro_auc | `0.7985421002758822` |
| final macro_f1 | `0.9107372982154678` |
| final micro_f1 | `0.8855707507675132` |
| final val_loss | `0.26912268925280797` |
| best-loss step | `2800` |
| best-loss macro_auc | `0.793706079134021` |
| best-AUC step | `800` |
| best-AUC macro_auc | `0.8113016852907395` |
| wrapper exit code | `0` |

### Failure Reason / Boundary

- No runtime failure.
- This completes the frozen-LM S1 LP row only after the already completed S1 source row.
- Source validation loss and LP macro-AUC remain separate metric families.

### Next Step

- Integrate S1 LP into `schema_complexity_diagnostic_summary.csv`, `revision_execution_status.csv`, `revision_completion_gap_audit.csv`, and Phase 4 synthesis.
- Re-run validation scripts and check that the formal S1/S2/S3 frozen-LM/no-LM source+LP schema comparison is now marked complete only where all required rows exist.

## 2026-06-26 Phase 1 / P1_SCHEMA_COMPLEXITY_FROZEN_LM_S1_LP_INTEGRATION result

### Plan

- Add the completed S1 frozen-LM LP row to schema diagnostics, execution status, gap audit, and Phase 4 synthesis.
- Rebuild final tables and verify that the schema matrix is marked complete only for the fixed-split formal source+LP scope.
- Preserve metric-family boundaries: source loss is not LP macro-AUC.

### Commands / Inputs / Outputs / Stop Conditions

```powershell
python -m py_compile scripts\summarize_frozen_lm_source_training.py scripts\summarize_schema_complexity_diagnostics.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py
python scripts\summarize_frozen_lm_source_training.py
python scripts\summarize_schema_complexity_diagnostics.py
python scripts\summarize_revision_execution_status.py
python scripts\summarize_revision_completion_gap_audit.py
python scripts\summarize_phase4_revision_synthesis.py
git diff --check -- scripts\summarize_frozen_lm_source_training.py scripts\summarize_schema_complexity_diagnostics.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py scripts\run_schema_lp_frozen_lm_s1_gpu0.cmd vivid_med_revision_execution_plan.md
```

Inputs:

- `outputs/schema_sweep/lp_frozen_lm_s1_state_only/metrics_final.json`.
- `outputs/schema_sweep/lp_frozen_lm_s1_state_only/best.pt`.
- `outputs/schema_sweep/lp_frozen_lm_s1_state_only/final.pt`.
- `outputs/schema_sweep/lp_frozen_lm_s1_state_only/step_3000.pt`.
- `outputs/logs/schema_lp_frozen_lm_s1_gpu0.log`.

Expected outputs:

- `outputs/final_tables/schema_complexity_diagnostic_summary.csv`.
- `outputs/final_tables/revision_execution_status.csv`.
- `outputs/final_tables/revision_completion_gap_audit.csv`.
- `outputs/final_tables/llm_necessity.csv`.
- `outputs/final_tables/module_candidates.csv`.

Stop conditions:

- Stop if S1 LP row is missing, if schema sweep is still marked S1 LP missing, if source loss and LP macro-AUC are mixed, or if formatting checks fail.

### Results

- Summary scripts compiled and regenerated successfully.
- `schema_complexity_diagnostic_summary.csv`: 18 rows, including `frozen_lm_s1_state_only_formal_lp`.
- `revision_execution_status.csv`: 45 rows, including `P1_SCHEMA_COMPLEXITY_FROZEN_LM_S1_LP_FORMAL_RUN`.
- `revision_completion_gap_audit.csv`: `P1_SCHEMA_COMPLEXITY_SWEEP` is now `completed_formal_no_lm_and_frozen_lm_s1_s2_s3_source_lp_matrix`.
- `llm_necessity.csv`: schema-complexity status is now `completed_fixed_split_formal_schema_source_lp_matrix`.
- `module_candidates.csv`: `Hierarchical UMS Head` remains deferred because the evidence matrix is now complete and any module would be a new intervention.
- `git diff --check` passed.
- Both GPUs remained idle after integration verification.

### Integrated Metrics

| artifact | integrated S1 LP state |
|---|---|
| `schema_complexity_diagnostic_summary.csv` | `frozen_lm_s1_state_only_formal_lp`, `primary_value=0.798542`, `best_auc_step=800:0.811302` |
| `revision_execution_status.csv` | `P1_SCHEMA_COMPLEXITY_FROZEN_LM_S1_LP_FORMAL_RUN`, `completed_formal_lp_run` |
| `revision_completion_gap_audit.csv` | fixed-split formal S1/S2/S3 schema source+LP matrix complete |
| `llm_necessity.csv` | frozen S1 source val_loss `0.035434` / LP `0.798542` |

### Failure Reason / Boundary

- No integration failure.
- The schema matrix is complete for the fixed-split formal source+LP scope.
- Broader data-scaling remains a separate original-scope gap.
- Do not infer frozen-LM dominance without checking the per-level metrics.

### Next Step

- Use the refreshed gap audit to decide whether to stop the prioritized schema/answerability/serialization pass or open the lower-priority long-running data-scaling matrix separately.

## 2026-06-26 Prioritized pass consistency audit

### Plan

- Verify that no current final-table/script conclusion still says frozen-LM S1 source/LP is missing.
- Verify that schema-complexity gap audit is now complete for the fixed-split formal source+LP matrix.
- Verify GPUs and VIVID training processes are idle before deciding the next task.

### Commands / Inputs / Outputs / Stop Conditions

```powershell
rg -n "S1 source\\+LP is still missing|S1 source/LP rows are still missing|S1 LP is still missing|frozen-LM S1 matched formal rows are missing|frozen-LM S1 formal matched rows are missing|frozen-LM formal matched S1 source\\+LP is still missing" outputs\final_tables scripts -g "*.csv" -g "*.md" -g "*.py"
git diff --check -- scripts\summarize_frozen_lm_source_training.py scripts\summarize_schema_complexity_diagnostics.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py scripts\run_schema_lp_frozen_lm_s1_gpu0.cmd vivid_med_revision_execution_plan.md
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

Expected outputs:

- No stale current missing-S1 text in `outputs/final_tables` or active scripts.
- Formatting check passes.
- No active GPU compute process.

Stop conditions:

- Stop and repair if stale current missing-S1 text remains in final tables/scripts.
- Stop and repair if formatting check fails.
- Do not launch another run if a VIVID training process is still active unexpectedly.

### Results

- Stale current missing-S1 search returned no matches in `outputs/final_tables` or active scripts.
- `git diff --check` passed.
- `revision_completion_gap_audit.csv` now marks `P1_SCHEMA_COMPLEXITY_SWEEP` as `completed_formal_no_lm_and_frozen_lm_s1_s2_s3_source_lp_matrix`.
- Schema gap is now: `none for the fixed-split formal S1/S2/S3 schema source+LP matrix; broader data-scaling scope remains separate`.
- GPU0: `0%`, `0 MiB / 24576 MiB`.
- GPU1: `0%`, `0 MiB / 24576 MiB`.

### Boundary / Next Step

- The prioritized pass covering frozen-LM use case evidence, UMS/schema contribution, answerability semantics, and schema serialization/dependency is now closed for the current fixed-split evidence packet.
- At this gate the remaining original-scope gap still included the then-unfinished 3k matched branch plus larger-scale rows; later gates below supersede this after matched 3k no-LM/frozen-LM LP completed, leaving 10k/30k as the remaining full-matrix gap.
- If continuing the full original plan, the next safe task is to inspect data-scaling configs and run one matched 3k row at a time with a fresh execution-before record.

## 2026-06-26 Phase 1 / P1_DATA_SCALING_3K_NO_LM_UMS_SOURCE_RUN execution-before

### Plan

- Run exactly one independent data-scaling row: formal 3k no-LM UMS source training.
- Use GPU0 only, keep GPU1 idle for now, and do not launch the frozen-LM 3k row until this source run reaches a stop condition.
- Keep the task separate from schema-complexity S1/S2/S3 results; this is a lower-priority original-scope data-scaling gap.
- After completion, collect final/best-step metrics, inspect log/checkpoints/GPU state, and write the result back before deciding whether to run the matched LP.

### Commands

Preflight and launch:

```powershell
Get-Content configs\data_scaling\no_lm_ums_3k.yaml
if (Test-Path outputs\data_scaling\no_lm_ums_3k) { Get-ChildItem outputs\data_scaling\no_lm_ums_3k -Recurse -Depth 2 } else { 'no_lm_ums_3k output missing' }
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '021_260129VIVID|train_ums_classifier|train_cxr|schema|data_scaling' } | Select-Object ProcessId,Name,CommandLine
wmic process call create "cmd.exe /c H:\Xiyao_Wang\021_260129VIVID\scripts\run_data_scaling_no_lm_ums_3k_source_gpu0.cmd"
```

Monitoring:

```powershell
Get-Content outputs\logs\data_scaling_no_lm_ums_3k_source_gpu0.log -Tail 80
Get-ChildItem outputs\data_scaling\no_lm_ums_3k
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv
```

### Inputs

- Config: `configs/data_scaling/no_lm_ums_3k.yaml`.
- Train split: `data/splits/chexpert_train_3k.jsonl`.
- Validation split: `data/splits/chexpert_val_fixed.jsonl`.
- Trainer: `scripts/train_ums_classifier.py`.
- Wrapper: `scripts/run_data_scaling_no_lm_ums_3k_source_gpu0.cmd`.

### Expected Outputs

- Log: `outputs/logs/data_scaling_no_lm_ums_3k_source_gpu0.log`.
- Output dir: `outputs/data_scaling/no_lm_ums_3k`.
- Metrics: `metrics_final.json`, `metrics_step_*.json`.
- Checkpoints: `best.pt`, `final.pt`, `step_*.pt`.

### Stop Conditions

- Stop before launch if config/splits are missing, if `outputs/data_scaling/no_lm_ums_3k/final.pt` already exists, or if an unrelated VIVID training process is active.
- Stop during run on nonzero wrapper exit code, traceback, OOM, stalled log/checkpoint writes, or unexpected GPU contention.
- On success, do not claim a matched 3k data-scaling point until the dependent LP row also runs from this source checkpoint.

### Preflight Results

- Config exists and targets `./outputs/data_scaling/no_lm_ums_3k`.
- Train split exists: `data/splits/chexpert_train_3k.jsonl`.
- Validation split exists: `data/splits/chexpert_val_fixed.jsonl`.
- `outputs/data_scaling/no_lm_ums_3k` was absent before launch.
- GPU0: `0%`, `0 MiB / 24576 MiB`, idle.
- GPU1: `0%`, `0 MiB / 24576 MiB`, idle.
- No active VIVID training process was found.

## 2026-06-26 Phase 1 / P1_DATA_SCALING_3K_NO_LM_UMS_SOURCE_RUN result

### Results

- Formal 3k no-LM UMS source training completed successfully.
- Wrapper log shows `EXITCODE 0`.
- Wrapper log end time: `2026/06/26 23:28:32.20`.
- Main training progress: `10000/10000`.
- Overall logged runtime: `44:23` for the main tqdm training loop including periodic validation overhead.
- Final validation completed across `16/16` batches.
- `outputs/data_scaling/no_lm_ums_3k/metrics_final.json` exists.
- `outputs/data_scaling/no_lm_ums_3k/metrics_step_10000.json` exists.
- `outputs/data_scaling/no_lm_ums_3k/best.pt` exists.
- `outputs/data_scaling/no_lm_ums_3k/final.pt` exists.
- `outputs/data_scaling/no_lm_ums_3k/step_10000.pt` exists.
- Both GPUs were idle after completion.

### Commands / Inputs / Outputs

```powershell
Get-Content outputs\logs\data_scaling_no_lm_ums_3k_source_gpu0.log -Tail 40
python -c "import json, glob, pathlib; ..."
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '021_260129VIVID|train_ums_classifier|run_data_scaling_no_lm_ums_3k' }
```

Outputs:

- Final `val_loss=2.57381571829319`.
- Final `macro_auc=0.6668418742173546`.
- Final `macro_f1=0.4247953267026716`.
- Final `micro_f1=0.5831622176591376`.
- Best-loss step: `1000`, `val_loss=0.7477902360260487`, `macro_auc=0.6715223080302589`, `macro_f1=0.3546732056308515`, `micro_f1=0.5975359342915811`.
- Best-AUC step: `3000`, `val_loss=1.87629896402359`, `macro_auc=0.6829854150599323`, `macro_f1=0.41558367349909026`, `micro_f1=0.551481372836609`.
- Metric files: `20` step metrics plus `metrics_final.json`.
- Final GPU0: `0%`, `0 MiB / 24576 MiB`.
- Final GPU1: `0%`, `0 MiB / 24576 MiB`.

### Metrics

| metric | value |
|---|---:|
| final macro_auc | `0.6668418742173546` |
| final macro_f1 | `0.4247953267026716` |
| final micro_f1 | `0.5831622176591376` |
| final val_loss | `2.57381571829319` |
| best-loss step | `1000` |
| best-loss macro_auc | `0.6715223080302589` |
| best-AUC step | `3000` |
| best-AUC macro_auc | `0.6829854150599323` |
| wrapper exit code | `0` |

### Failure Reason / Boundary

- No runtime failure.
- Speed was normal for this machine and configuration: steady training around `3.8-4.0 it/s`, GPU0 about `4.3 GiB` VRAM and about `220 W`; GPU1 stayed idle.
- Later validation loss increased while train loss decreased, so this source run shows overfitting after the early best-loss checkpoint.
- This is a source-training row only; it is not a matched 3k data-scaling point until the dependent LP row is run from `outputs/data_scaling/no_lm_ums_3k/best.pt`.

### Next Step

- Integrate this completed 3k no-LM UMS source row into data-scaling/status/gap-audit artifacts.
- Then write a fresh execution-before record for the dependent 3k no-LM LP row, unless the plan is paused before lower-priority data scaling.

## 2026-06-26 Phase 1 / P1_DATA_SCALING_3K_NO_LM_UMS_SOURCE_INTEGRATION execution-before

### Plan

- Integrate the completed 3k no-LM UMS source row into the data-scaling progress artifacts.
- Keep old 3k BCE progress outputs intact for compatibility, and add a combined 3k source-progress table covering BCE and no-LM source rows.
- Update execution status and gap audit to say 3k no-LM source is complete, while matched 3k evidence is still incomplete until LP and frozen-LM rows run.
- Refresh config validation so the 3k no-LM LP dependency changes from blocked to ready if `outputs/data_scaling/no_lm_ums_3k/best.pt` is detected.

### Commands

```powershell
python -m py_compile scripts\summarize_data_scaling_3k_bce_progress.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py
python scripts\validate_data_scaling_configs.py
python scripts\summarize_data_scaling_3k_bce_progress.py
python scripts\summarize_revision_execution_status.py
python scripts\summarize_revision_completion_gap_audit.py
python scripts\summarize_phase4_revision_synthesis.py
git diff --check -- scripts\summarize_data_scaling_3k_bce_progress.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py vivid_med_revision_execution_plan.md
```

### Inputs

- `outputs/data_scaling/no_lm_ums_3k/metrics_final.json`.
- `outputs/data_scaling/no_lm_ums_3k/metrics_step_*.json`.
- `outputs/data_scaling/no_lm_ums_3k/best.pt`.
- `outputs/data_scaling/no_lm_ums_3k/final.pt`.
- `outputs/logs/data_scaling_no_lm_ums_3k_source_gpu0.log`.
- Existing 3k BCE outputs under `outputs/data_scaling/bce_3k`.

### Expected Outputs

- Existing compatibility outputs: `outputs/final_tables/data_scaling_3k_bce_progress.csv`, `.md`, and `data_scaling_3k_bce_trajectory.csv`.
- New combined outputs: `outputs/final_tables/data_scaling_3k_source_progress.csv`, `.md`, and `data_scaling_3k_source_trajectory.csv`.
- Updated `outputs/final_tables/data_scaling_config_validation.csv`.
- Updated `outputs/final_tables/revision_execution_status.csv`.
- Updated `outputs/final_tables/revision_completion_gap_audit.csv`.
- Updated Phase 4 synthesis tables if they mention data-scaling gaps.

### Stop Conditions

- Stop if the 3k no-LM row is absent from combined progress outputs.
- Stop if `lp_no_lm_ums_3k` remains blocked despite `outputs/data_scaling/no_lm_ums_3k/best.pt` existing.
- Stop if 3k no-LM source is described as a matched 3k result.
- Stop if old 3k BCE compatibility outputs disappear.

## 2026-06-26 Phase 1 / P1_DATA_SCALING_3K_NO_LM_UMS_SOURCE_INTEGRATION result

### Results

- Integration completed successfully.
- `scripts/summarize_data_scaling_3k_bce_progress.py` now writes both compatibility BCE-only outputs and combined 3k source-progress outputs.
- `outputs/final_tables/data_scaling_3k_source_progress.csv` includes both `bce_3k` and `no_lm_ums_3k`.
- `outputs/final_tables/data_scaling_3k_source_trajectory.csv` includes the full step trajectory for both source rows.
- `outputs/final_tables/data_scaling_3k_bce_progress.csv` and `.md` still exist for old references.
- `outputs/final_tables/revision_execution_status.csv` now has `46` rows and includes `P1_DATA_SCALING_3K_NO_LM_UMS_SOURCE_RUN`.
- `outputs/final_tables/revision_completion_gap_audit.csv` now marks data scaling as `partial_1k_matched_plus_3k_bce_and_no_lm_source`.
- `outputs/final_tables/llm_necessity.csv` now states that 3k BCE/no-LM source rows are complete but not a matched LP comparison.
- Stale text search for the old "3k BCE only" gap wording returned no matches.
- `git diff --check` passed for the touched scripts and plan document.

### Commands / Inputs / Outputs

```powershell
python -m py_compile scripts\summarize_data_scaling_3k_bce_progress.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py
python scripts\validate_data_scaling_configs.py
python scripts\summarize_data_scaling_3k_bce_progress.py
python scripts\summarize_revision_execution_status.py
python scripts\summarize_revision_completion_gap_audit.py
python scripts\summarize_phase4_revision_synthesis.py
rg -n "3k BCE source-control row is complete, but matched 3k no-LM|3k BCE source control and the formal schema|partial_1k_matched_plus_3k_bce_control|strong_for_1k_matched_lp_and_3k_bce_control_only" outputs\final_tables scripts -g "*.csv" -g "*.md" -g "*.py"
```

Outputs:

- `validate_data_scaling_configs.py`: `Validated 24 configs`, `Failures: 6`, `LP blocked until source checkpoints: 7`.
- `lp_no_lm_ums_3k`: `status=ok`, notes `LP source checkpoint already exists`.
- `no_lm_ums_3k`: `status=fail`, notes `would overwrite completed run`; this is the intended anti-overwrite guard, not a training failure.
- Config validation remaining blocked LP rows are frozen-LM 3k/10k/30k, no-LM 10k/30k, and random-LM 3k/30k dependencies.

### Integrated Metrics

| artifact | integrated state |
|---|---|
| `data_scaling_3k_source_progress.csv` | `no_lm_ums_3k` final macro_auc `0.666842`, best-loss step `1000`, best-AUC step `3000` |
| `data_scaling_config_validation.csv` | `lp_no_lm_ums_3k` is now `ok` because `outputs/data_scaling/no_lm_ums_3k/best.pt` exists |
| `revision_execution_status.csv` | `P1_DATA_SCALING_3K_NO_LM_UMS_SOURCE_RUN`, `completed_formal_source_run_source_only` |
| `revision_completion_gap_audit.csv` | data scaling is `partial_1k_matched_plus_3k_bce_and_no_lm_source` |
| `llm_necessity.csv` | low-data necessity remains unsupported by current matched LP evidence |

### Failure Reason / Boundary

- No integration failure.
- The six config-validation failures are completed-output overwrite guards, not failed experiments.
- The 3k no-LM source row is complete but source-only; matched 3k evidence still requires the dependent no-LM LP and frozen-LM 3k source/LP rows.
- Do not compare this 3k no-LM source row directly to 1k LP rows or to frozen-LM source loss.

### Next Step

- Write a fresh execution-before record for `P1_DATA_SCALING_3K_NO_LM_UMS_LP_FORMAL_RUN`.
- Because `lp_no_lm_ums_3k` is now unblocked and LP is low-memory, run it before considering the heavier frozen-LM 3k source row.

## 2026-06-26 Phase 1 / P1_DATA_SCALING_3K_NO_LM_UMS_LP_FORMAL_RUN execution-before

### Plan

- Run exactly one dependent LP row: formal 3k no-LM UMS linear probe from `outputs/data_scaling/no_lm_ums_3k/best.pt`.
- Use GPU0 only; keep GPU1 idle unless a later independent low-memory task is explicitly opened with its own execution-before record.
- Keep this LP separate from the 3k source metrics; only LP metrics can be compared with other LP rows.
- After completion, collect final/best-step metrics, inspect log/checkpoints/GPU state, and write the result back before any frozen-LM 3k run.

### Commands

Preflight and launch:

```powershell
Get-Content configs\data_scaling\lp_no_lm_ums_3k.yaml
if (Test-Path outputs\data_scaling\lp_no_lm_ums_3k) { Get-ChildItem outputs\data_scaling\lp_no_lm_ums_3k -Recurse -Depth 2 } else { 'lp_no_lm_ums_3k output missing' }
Get-Item outputs\data_scaling\no_lm_ums_3k\best.pt
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '021_260129VIVID|train_vit_baseline|train_ums_classifier|data_scaling' } | Select-Object ProcessId,Name,CommandLine
Start-Process -FilePath cmd.exe -ArgumentList '/c','H:\Xiyao_Wang\021_260129VIVID\scripts\run_data_scaling_lp_no_lm_ums_3k_gpu0.cmd' -WindowStyle Hidden -PassThru
```

Monitoring:

```powershell
Get-Content outputs\logs\data_scaling_lp_no_lm_ums_3k_gpu0.log -Tail 100
Get-ChildItem outputs\data_scaling\lp_no_lm_ums_3k
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv
```

### Inputs

- Config: `configs/data_scaling/lp_no_lm_ums_3k.yaml`.
- Source checkpoint: `outputs/data_scaling/no_lm_ums_3k/best.pt`.
- Train split: `data/splits/chexpert_train_3k.jsonl`.
- Validation split: `data/splits/chexpert_val_fixed.jsonl`.
- Trainer: `scripts/train_vit_baseline.py`.
- Wrapper: `scripts/run_data_scaling_lp_no_lm_ums_3k_gpu0.cmd`.

### Expected Outputs

- Log: `outputs/logs/data_scaling_lp_no_lm_ums_3k_gpu0.log`.
- Output dir: `outputs/data_scaling/lp_no_lm_ums_3k`.
- Metrics: `metrics_final.json`, `metrics_step_*.json`.
- Checkpoints: `best.pt`, `final.pt`, `step_*.pt`.

### Stop Conditions

- Stop before launch if the source `best.pt` is missing, if `outputs/data_scaling/lp_no_lm_ums_3k/final.pt` already exists, or if an unrelated VIVID training process is active.
- Stop during run on nonzero wrapper exit code, traceback, OOM, stalled log/checkpoint writes, or unexpected GPU contention.
- On success, still do not claim full matched 3k evidence until frozen-LM 3k source/LP exists.

### Preflight Results

- Config exists and points to `./outputs/data_scaling/no_lm_ums_3k/best.pt`.
- Source `best.pt` exists and was produced by the completed 3k no-LM source run.
- `outputs/data_scaling/lp_no_lm_ums_3k` was absent before launch.
- GPU0: `0%`, `0 MiB / 24576 MiB`, idle.
- GPU1: `0%`, `0 MiB / 24576 MiB`, idle.
- No active VIVID training process was found.

## 2026-06-26 Phase 1 / P1_DATA_SCALING_3K_NO_LM_UMS_LP_FORMAL_RUN result

### Results

- Formal 3k no-LM UMS LP completed successfully.
- Wrapper log shows `EXITCODE 0`.
- Wrapper log end time: `2026/06/26 23:46:17.31`.
- Source checkpoint loaded successfully from `outputs/data_scaling/no_lm_ums_3k/best.pt`.
- Log confirms `Loaded params: 150`, missing only `head.weight` and `head.bias`.
- Linear probe mode froze `150` backbone params and trained `10,766` head params.
- Main training progress: `3000/3000`.
- Main training loop runtime: `08:15`.
- Final validation completed across `63/63` batches.
- `outputs/data_scaling/lp_no_lm_ums_3k/metrics_final.json` exists.
- `outputs/data_scaling/lp_no_lm_ums_3k/metrics_step_3000.json` exists.
- `outputs/data_scaling/lp_no_lm_ums_3k/best.pt` exists.
- `outputs/data_scaling/lp_no_lm_ums_3k/final.pt` exists.
- `outputs/data_scaling/lp_no_lm_ums_3k/step_3000.pt` exists.
- Both GPUs were idle after completion.

### Commands / Inputs / Outputs

```powershell
Get-Content outputs\logs\data_scaling_lp_no_lm_ums_3k_gpu0.log -Tail 30
python -c "import json, glob, pathlib; ..."
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
```

Outputs:

- Final `val_loss=0.3352395702922155`.
- Final `macro_auc=0.7342345302322837`.
- Final `macro_f1=0.8974090495714947`.
- Final `micro_f1=0.8657549539492045`.
- Best-loss step: `200`, `val_loss=0.3101223527439057`, `macro_auc=0.760698267539557`, `macro_f1=0.9042584939220925`, `micro_f1=0.8699413898967345`.
- Best-AUC step: `200`, `macro_auc=0.760698267539557`.
- Best-F1 step: `200`, `macro_f1=0.9042584939220925`.
- Metric files: `15` step metrics plus `metrics_final.json`.
- Final GPU0: `0%`, `0 MiB / 24576 MiB`.
- Final GPU1: `0%`, `0 MiB / 24576 MiB`.

### Metrics

| metric | value |
|---|---:|
| final macro_auc | `0.7342345302322837` |
| final macro_f1 | `0.8974090495714947` |
| final micro_f1 | `0.8657549539492045` |
| final val_loss | `0.3352395702922155` |
| best-loss step | `200` |
| best-loss macro_auc | `0.760698267539557` |
| best-AUC step | `200` |
| best-AUC macro_auc | `0.760698267539557` |
| wrapper exit code | `0` |

### Failure Reason / Boundary

- No runtime failure.
- LP speed and resource use were normal: about `1.9 GiB` VRAM, low sustained power, and training around `7-8 it/s` after initialization.
- This completes the 3k no-LM source+LP branch, but not the full matched 3k data-scaling comparison because frozen-LM 3k source/LP is still missing.
- Do not compare source-run macro-AUC directly with LP macro-AUC; only LP rows are comparable for downstream binary classifier performance.

### Next Step

- Integrate this completed LP row into 3k data-scaling progress, execution status, gap audit, and Phase 4 synthesis.
- Then decide whether to open the heavier 3k frozen-LM source run with a fresh execution-before record.

## 2026-06-26 Phase 1 / P1_DATA_SCALING_3K_NO_LM_UMS_LP_INTEGRATION execution-before

### Plan

- Add the completed 3k no-LM LP row to final-table progress artifacts without mixing it into the source-only table.
- Keep old `data_scaling_3k_source_*` outputs as source-only and add LP/current-progress outputs for the no-LM LP endpoint.
- Update execution status and gap audit to say 3k no-LM source+LP is complete, while frozen-LM 3k source+LP and 10k/30k remain missing.
- Refresh config validation so completed LP output is treated as an anti-overwrite guard, not a failed experiment.

### Commands

```powershell
python -m py_compile scripts\summarize_data_scaling_3k_bce_progress.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py
python scripts\validate_data_scaling_configs.py
python scripts\summarize_data_scaling_3k_bce_progress.py
python scripts\summarize_revision_execution_status.py
python scripts\summarize_revision_completion_gap_audit.py
python scripts\summarize_phase4_revision_synthesis.py
git diff --check -- scripts\summarize_data_scaling_3k_bce_progress.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py scripts\run_data_scaling_lp_no_lm_ums_3k_gpu0.cmd vivid_med_revision_execution_plan.md
```

### Inputs

- `outputs/data_scaling/lp_no_lm_ums_3k/metrics_final.json`.
- `outputs/data_scaling/lp_no_lm_ums_3k/metrics_step_*.json`.
- `outputs/data_scaling/lp_no_lm_ums_3k/best.pt`.
- `outputs/data_scaling/lp_no_lm_ums_3k/final.pt`.
- `outputs/logs/data_scaling_lp_no_lm_ums_3k_gpu0.log`.
- Existing 3k BCE and no-LM source outputs.

### Expected Outputs

- New LP outputs: `outputs/final_tables/data_scaling_3k_no_lm_lp_progress.csv`, `.md`, and `data_scaling_3k_no_lm_lp_trajectory.csv`.
- New current-progress outputs: `outputs/final_tables/data_scaling_3k_current_progress.csv`, `.md`, and `data_scaling_3k_current_trajectory.csv`.
- Existing source-only outputs remain present: `data_scaling_3k_source_progress.csv`, `.md`, and `data_scaling_3k_source_trajectory.csv`.
- Updated `revision_execution_status.csv`, `revision_completion_gap_audit.csv`, and `llm_necessity.csv`.

### Stop Conditions

- Stop if the no-LM LP row is absent from LP/current-progress outputs.
- Stop if source-only outputs now contain LP rows.
- Stop if gap audit says the full matched 3k comparison is complete before frozen-LM 3k source/LP exists.
- Stop if config-validation anti-overwrite rows are described as experiment failures.

## 2026-06-26 Phase 1 / P1_DATA_SCALING_3K_NO_LM_UMS_LP_INTEGRATION result

### Results

- Integration completed successfully after one script fix.
- New LP outputs exist: `outputs/final_tables/data_scaling_3k_no_lm_lp_progress.csv`, `.md`, and `data_scaling_3k_no_lm_lp_trajectory.csv`.
- New current-progress outputs exist: `outputs/final_tables/data_scaling_3k_current_progress.csv`, `.md`, and `data_scaling_3k_current_trajectory.csv`.
- Existing source-only outputs remain present and source-only: `data_scaling_3k_source_progress.csv`, `.md`, and `data_scaling_3k_source_trajectory.csv`.
- `revision_execution_status.csv` now has `47` rows and includes `P1_DATA_SCALING_3K_NO_LM_UMS_LP_FORMAL_RUN`.
- `revision_completion_gap_audit.csv` now marks data scaling as `partial_1k_matched_plus_3k_bce_source_and_no_lm_source_lp`.
- At this earlier integration gate, `llm_necessity.csv` still stated that 3k no-LM source+LP was complete while 3k frozen-LM source/LP was missing; later gates below supersede that boundary after the frozen-LM source+LP rows completed.
- Stale text search for old 3k no-LM LP-missing/source-only wording returned no matches.
- `git diff --check` passed.

### Commands / Inputs / Outputs

```powershell
python -m py_compile scripts\summarize_data_scaling_3k_bce_progress.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py
python scripts\validate_data_scaling_configs.py
python scripts\summarize_data_scaling_3k_bce_progress.py
python scripts\summarize_revision_execution_status.py
python scripts\summarize_revision_completion_gap_audit.py
python scripts\summarize_phase4_revision_synthesis.py
rg -n "3k no-LM LP and frozen-LM source/LP rows are not yet run|3k BCE/no-LM source rows are complete but not yet|partial_1k_matched_plus_3k_bce_and_no_lm_source|strong_for_1k_matched_lp_and_3k_source_rows_only|until its matched LP runs" outputs\final_tables scripts -g "*.csv" -g "*.md" -g "*.py"
```

Outputs:

- `validate_data_scaling_configs.py`: `Validated 24 configs`, `Failures: 7`, `LP blocked until source checkpoints: 7`.
- `lp_no_lm_ums_3k`: `status=fail`, notes `would overwrite completed run; LP source checkpoint already exists`; this is the intended completed-output overwrite guard.
- `data_scaling_3k_no_lm_lp_progress.csv`: final macro_auc `0.734235`, best-step-200 macro_auc `0.760698`.
- `revision_execution_status.csv`: `P1_DATA_SCALING_3K_NO_LM_UMS_LP_FORMAL_RUN`, `completed_formal_lp_run`.
- `revision_completion_gap_audit.csv`: `evidence_present_count=11 / 11` for `P1_DATA_SCALING`.

### Integrated Metrics

| artifact | integrated state |
|---|---|
| `data_scaling_3k_no_lm_lp_progress.csv` | `lp_no_lm_ums_3k` final macro_auc `0.734235`; best-step-200 macro_auc `0.760698` |
| `data_scaling_3k_current_progress.csv` | 3k BCE source, no-LM source, and no-LM LP rows present with separate `stage` values |
| `revision_execution_status.csv` | `P1_DATA_SCALING_3K_NO_LM_UMS_LP_FORMAL_RUN`, `completed_formal_lp_run` |
| `revision_completion_gap_audit.csv` | 3k no-LM source+LP complete; 3k frozen-LM source+LP still missing |
| `llm_necessity.csv` | low-data necessity remains unsupported by current matched evidence |

### Failure Reason / Boundary

- Initial integration attempt failed with `NameError: name 'all_key_rows' is not defined` in `scripts/summarize_data_scaling_3k_bce_progress.py`.
- Root cause: after refactoring the 3k source-progress script from BCE-only to source+LP outputs, one call site still referenced the old variable names.
- Fix: replaced the old variables with `source_key_rows` and `source_trajectory`; reran compilation and full generation successfully.
- The seven config-validation failures are completed-output overwrite guards or missing-source dependency guards, not failed runs.
- The full matched 3k frozen-LM/no-LM comparison is still incomplete until frozen-LM 3k source and LP complete.

### Next Step

- If continuing the lower-priority full data-scaling branch, inspect `configs/data_scaling/frozen_lm_ums_3k.yaml` and write a fresh execution-before record for the heavier 3k frozen-LM source run.
- Use one GPU only for frozen-LM 3k source unless preflight proves it is light enough, because frozen-LM source rows are high-memory/high-power compared with LP.

## 2026-06-26 Phase 1 / P1_DATA_SCALING_3K_FROZEN_LM_UMS_SOURCE_RUN execution-before

### Plan

- Run exactly one heavy data-scaling row: formal 3k frozen-LM UMS source training.
- Use GPU0 only and do not launch another VIVID training job in parallel.
- Use the existing Qwen2.5-1.5B cache; do not treat model download/cache lookup time as training slowness unless logs show repeated network fetch failures.
- Expected runtime is long: same 10000-step frozen-LM source family previously took about `8h17m` on this host; 3k train size still uses 10000 steps and the same fixed validation size.
- After completion, collect final/best-step source loss metrics and only then open the dependent 3k frozen-LM LP row.

### Commands

Preflight and launch:

```powershell
Get-Content configs\data_scaling\frozen_lm_ums_3k.yaml
if (Test-Path outputs\data_scaling\frozen_lm_ums_3k) { Get-ChildItem outputs\data_scaling\frozen_lm_ums_3k -Recurse -Depth 2 } else { 'frozen_lm_ums_3k output missing' }
Get-ChildItem $env:USERPROFILE\.cache\huggingface\hub -Directory -Filter 'models--Qwen--Qwen2.5-1.5B-Instruct'
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '021_260129VIVID|train_cxr|train_vit_baseline|train_ums_classifier|data_scaling|schema' } | Select-Object ProcessId,Name,CommandLine
Start-Process -FilePath cmd.exe -ArgumentList '/c','H:\Xiyao_Wang\021_260129VIVID\scripts\run_data_scaling_frozen_lm_ums_3k_source_gpu0.cmd' -WindowStyle Hidden -PassThru
```

Monitoring:

```powershell
Get-Content outputs\logs\data_scaling_frozen_lm_ums_3k_source_gpu0.log -Tail 120
Get-ChildItem outputs\data_scaling\frozen_lm_ums_3k -Recurse -Depth 2
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv
```

### Inputs

- Config: `configs/data_scaling/frozen_lm_ums_3k.yaml`.
- Train split: `data/splits/chexpert_train_3k.jsonl`.
- Validation split: `data/splits/chexpert_val_fixed.jsonl`.
- LLM: `Qwen/Qwen2.5-1.5B-Instruct`.
- Local HF cache found: `$env:USERPROFILE/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct`.
- Trainer: `scripts/train_cxr.py`.
- Wrapper: `scripts/run_data_scaling_frozen_lm_ums_3k_source_gpu0.cmd`.

### Expected Outputs

- Log: `outputs/logs/data_scaling_frozen_lm_ums_3k_source_gpu0.log`.
- Output dir: `outputs/data_scaling/frozen_lm_ums_3k`.
- Checkpoints under `outputs/data_scaling/frozen_lm_ums_3k/checkpoints`: `best.pt`, `final.pt`, `step_*.pt`.
- Validation trace in `outputs/data_scaling/frozen_lm_ums_3k/training_history.json` if the trainer writes it.

### Stop Conditions

- Stop before launch if config/splits/cache are missing, if `outputs/data_scaling/frozen_lm_ums_3k/checkpoints/final.pt` already exists, or if another VIVID training process is active.
- Stop during run on nonzero wrapper exit code, traceback, OOM, repeated model download/cache failure, stalled log/checkpoint writes, or unexpected GPU contention.
- If externally interrupted after a valid `best.pt`, document early-stop provenance before using it for LP; do not call it a completed source run.

### Preflight Results

- Config exists and targets `./outputs/data_scaling/frozen_lm_ums_3k`.
- Train split exists: `data/splits/chexpert_train_3k.jsonl`.
- Validation split exists: `data/splits/chexpert_val_fixed.jsonl`.
- Output directory `outputs/data_scaling/frozen_lm_ums_3k` was absent before launch.
- Qwen HF cache directory exists locally.
- GPU0: `0%`, `0 MiB / 24576 MiB`, idle.
- GPU1: `0%`, `0 MiB / 24576 MiB`, idle.
- No active VIVID training process was found.

## 2026-06-26 Phase 1 / P1_DATA_SCALING_3K_FROZEN_LM_UMS_SOURCE_RUN runtime gate

### Results

- Run launched successfully with wrapper PID `20160`; Python child PID observed as `22028`.
- Model and dataloaders initialized successfully.
- Train split loaded: `3000` samples, `750` train batches.
- Validation split loaded: `1000` samples, `250` validation batches.
- Qwen2.5-1.5B loaded successfully from local ModelScope/HF cache path; log says target directory already exists.
- FlashAttention2 was unavailable and the model fell back to eager attention, matching earlier frozen-LM runs.
- LLM hidden size: `1536`; frozen LLM parameters: `1,543,714,304`.
- Trainable parameters: `89,349,888`.
- Training started and reached at least step `24/10000` during the first runtime gate.

### Runtime / Speed

- GPU0: about `24297 MiB / 24576 MiB`, about `319 W`, `99%` utilization.
- GPU1: idle.
- Observed training speed after warmup: about `2.9-3.1s/it`.
- Estimated duration remains about `8h` plus validation/checkpoint overhead, consistent with prior frozen-LM source runs.

### Failure Reason / Boundary

- No runtime failure at this gate.
- This is expected to occupy almost all GPU0 VRAM; do not co-schedule another GPU0 task.
- The slow speed is normal for Qwen2.5-1.5B frozen-LM source training on this host, not evidence of malware or unrelated automation.
- GPU1 being idle does not make this safe to duplicate on GPU0; any parallel run would need its own GPU and dependency boundary.

### Next Step

- Continue monitoring for first validation/checkpoint at step `500/1000`.
- Stop only if traceback, OOM, stalled log/checkpoint writes, or external GPU contention appears.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_3K_FROZEN_LM_UMS_SOURCE_RUN step500-1000 gate

### Results

- Run remained active and stable through the first two validation gates.
- Step `500`: `val_loss=0.0557`; `best.pt` saved.
- Step `1000`: `val_loss=0.0472`; `best.pt` updated and `step_1000.pt` saved.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or `EXITCODE` failure line was observed in the log scan.
- Current observed progress after this gate was about `1411/10000`.

### Commands / Inputs / Outputs

```powershell
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_3k_source_gpu0.log -Pattern 'Step [0-9]+: val_loss|Checkpoint saved|EXITCODE|Traceback|CUDA out of memory|RuntimeError'
Get-ChildItem outputs\data_scaling\frozen_lm_ums_3k\checkpoints
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
```

Outputs:

- `outputs/data_scaling/frozen_lm_ums_3k/checkpoints/best.pt` exists.
- `outputs/data_scaling/frozen_lm_ums_3k/checkpoints/step_1000.pt` exists.
- GPU0 remained the only active VIVID training GPU, with about `24.3 GiB` VRAM in use.
- GPU1 remained idle.

### Failure Reason / Boundary

- No runtime failure at this gate.
- Validation and checkpoint writes are occurring as expected.
- The run remains incomplete; do not use `best.pt` for LP as a completed-source substitute unless the run is later interrupted and explicitly documented as early-stop provenance.

### Next Step

- Continue monitoring the long run toward step `2000`.
- At step `2000`, record whether `best.pt` improves and whether `step_2000.pt` is saved.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_3K_FROZEN_LM_UMS_SOURCE_RUN step2000 gate

### Results

- Run remained active and stable through step `2000`.
- Step `2000`: `val_loss=0.0438`.
- `best.pt` updated at step `2000`.
- `step_2000.pt` saved.
- Current observed progress after this gate was about `2113/10000`.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or `EXITCODE` failure line was observed in the latest log window.

### Commands / Inputs / Outputs

```powershell
Get-Content outputs\logs\data_scaling_frozen_lm_ums_3k_source_gpu0.log -Tail 180
Get-ChildItem outputs\data_scaling\frozen_lm_ums_3k\checkpoints
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
```

Outputs:

- `outputs/data_scaling/frozen_lm_ums_3k/checkpoints/best.pt` exists and was updated at step `2000`.
- `outputs/data_scaling/frozen_lm_ums_3k/checkpoints/step_1000.pt` exists.
- `outputs/data_scaling/frozen_lm_ums_3k/checkpoints/step_2000.pt` exists.
- GPU0 remained around `24.3 GiB` VRAM and high utilization.
- GPU1 remained idle.

### Failure Reason / Boundary

- No runtime failure at this gate.
- This remains a running source experiment, not a completed source run.
- `best.pt` is valid as an in-progress checkpoint but should not feed LP unless the run completes or an early-stop rationale is explicitly recorded.

### Next Step

- Continue monitoring toward step `3000/4000`.
- At each checkpoint, record `val_loss`, `best.pt` update behavior, and saved `step_*.pt` files.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_3K_FROZEN_LM_UMS_SOURCE_RUN completion gate

### Results

- The 3k frozen-LM UMS source run completed successfully.
- Wrapper exit line: `EXITCODE 0`.
- End time: `2026/06/27 08:12:03.84`.
- Total training progress: `10000/10000` in about `8:17:26`, around `2.98s/it`.
- `final.pt` and `step_10000.pt` were saved.
- Best validation loss was reached at step `2500`: `val_loss=0.0410`, and `best.pt` was last updated there.
- Later validation losses rose, so the completed source run shows overfitting after the best checkpoint:
  - step `3000`: `0.0439`
  - step `4000`: `0.0418`
  - step `5000`: `0.0540`
  - step `6000`: `0.0647`
  - step `7000`: `0.0756`
  - step `8000`: `0.0872`
  - step `9000`: `0.0942`
  - step `10000`: `0.1011`

### Commands / Inputs / Outputs

```powershell
Get-Process -Id 20160,22028 -ErrorAction SilentlyContinue
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_3k_source_gpu0.log -Pattern 'Step [0-9]+: val_loss|Checkpoint saved|Final|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 40
Get-ChildItem outputs\data_scaling\frozen_lm_ums_3k\checkpoints
Get-Content outputs\logs\data_scaling_frozen_lm_ums_3k_source_gpu0.log -Tail 60
```

Outputs:

- No `cmd`/`python` process remained for PID `20160`/`22028`; the run exited.
- GPU0 and GPU1 were both idle after completion.
- `outputs/data_scaling/frozen_lm_ums_3k/checkpoints/best.pt` exists and was last updated at `2026/6/27 02:01:55`.
- `step_1000.pt`, `step_2000.pt`, `step_3000.pt`, `step_4000.pt`, `step_5000.pt`, `step_6000.pt`, `step_7000.pt`, `step_8000.pt`, `step_9000.pt`, `step_10000.pt`, and `final.pt` exist.

### Metrics

| step | val_loss | checkpoint behavior |
|---:|---:|---|
| 500 | 0.0557 | `best.pt` saved |
| 1000 | 0.0472 | `best.pt`, `step_1000.pt` saved |
| 1500 | 0.0439 | `best.pt` updated |
| 2000 | 0.0438 | `best.pt`, `step_2000.pt` saved |
| 2500 | 0.0410 | `best.pt` updated; best source checkpoint |
| 3000 | 0.0439 | `step_3000.pt` saved |
| 4000 | 0.0418 | `step_4000.pt` saved |
| 5000 | 0.0540 | `step_5000.pt` saved |
| 6000 | 0.0647 | `step_6000.pt` saved |
| 7000 | 0.0756 | `step_7000.pt` saved |
| 8000 | 0.0872 | `step_8000.pt` saved |
| 9000 | 0.0942 | `step_9000.pt` saved |
| 10000 | 0.1011 | `step_10000.pt`, `final.pt` saved |

### Failure Reason / Boundary

- No runtime failure, OOM, traceback, or nonzero exit occurred.
- The source row is complete, but LP performance is not implied by source loss; the dependent LP row must be run from `best.pt`.
- The speed was consistent with prior frozen-LM source runs on this host; no evidence of virus-like or unrelated automation slowdown was found in process/GPU checks.

### Next Step

- Launch the dependent 3k frozen-LM LP row from `outputs/data_scaling/frozen_lm_ums_3k/checkpoints/best.pt`.
- Keep it single-GPU unless another independent lightweight task is explicitly selected; LP is short and directly closes the current 3k matched comparison.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_3K_FROZEN_LM_UMS_LP_FORMAL_RUN execution-before

### Plan

- Run the dependent 3k frozen-LM LP classifier row after the source row completed successfully.
- Initialize ViT from `outputs/data_scaling/frozen_lm_ums_3k/checkpoints/best.pt`, freeze the backbone, and train only the LP head.
- Use GPU0 only; expected memory is low compared with frozen-LM source, but the task is short enough that parallel scheduling is unnecessary.
- After completion, integrate 3k frozen-LM source+LP metrics into the data-scaling progress tables, revision status, gap audit, LLM necessity table, and Phase4 synthesis.

### Commands

Preflight and launch:

```powershell
Get-Content configs\data_scaling\lp_frozen_lm_ums_3k.yaml
Test-Path outputs\data_scaling\frozen_lm_ums_3k\checkpoints\best.pt
Test-Path outputs\data_scaling\frozen_lm_ums_3k\checkpoints\final.pt
if (Test-Path outputs\data_scaling\lp_frozen_lm_ums_3k) { Get-ChildItem outputs\data_scaling\lp_frozen_lm_ums_3k -Recurse -Depth 2 } else { 'lp_frozen_lm_ums_3k output missing' }
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '021_260129VIVID|train_cxr|train_vit_baseline|train_ums_classifier|data_scaling|schema' } | Select-Object ProcessId,Name,CommandLine
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Start-Process -FilePath cmd.exe -ArgumentList '/c','H:\Xiyao_Wang\021_260129VIVID\scripts\run_data_scaling_lp_frozen_lm_ums_3k_gpu0.cmd' -WindowStyle Hidden -PassThru
```

Monitoring:

```powershell
Get-Content outputs\logs\data_scaling_lp_frozen_lm_ums_3k_gpu0.log -Tail 120
Get-ChildItem outputs\data_scaling\lp_frozen_lm_ums_3k -Recurse -Depth 2
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv
```

### Inputs

- Config: `configs/data_scaling/lp_frozen_lm_ums_3k.yaml`.
- Source checkpoint: `outputs/data_scaling/frozen_lm_ums_3k/checkpoints/best.pt`.
- Train split: `data/splits/chexpert_train_3k.jsonl`.
- Validation split: `data/splits/chexpert_val_fixed.jsonl`.
- Trainer: `scripts/train_vit_baseline.py`.
- Wrapper: `scripts/run_data_scaling_lp_frozen_lm_ums_3k_gpu0.cmd`.

### Expected Outputs

- Log: `outputs/logs/data_scaling_lp_frozen_lm_ums_3k_gpu0.log`.
- Output dir: `outputs/data_scaling/lp_frozen_lm_ums_3k`.
- Checkpoints under `outputs/data_scaling/lp_frozen_lm_ums_3k/checkpoints`: `best.pt`, `final.pt`, `step_*.pt`.
- LP metrics in the log and `training_history.json`.

### Stop Conditions

- Stop before launch if source `best.pt` is missing, LP output already has a completed `final.pt`, config points at the wrong checkpoint, or another VIVID training process is active.
- Stop during run on nonzero wrapper exit code, traceback, OOM, stalled log/checkpoint writes, or unexpected GPU contention.
- Do not integrate this as a completed matched comparator until the wrapper exits with `EXITCODE 0` and LP metrics are extracted.

### Preflight Results

- Config exists and points `transfer.init_vit_checkpoint` to `./outputs/data_scaling/frozen_lm_ums_3k/checkpoints/best.pt`.
- Source `best.pt` exists.
- Source `final.pt` exists.
- LP output directory was absent before launch.
- No active VIVID training process was found after excluding the preflight query itself.
- GPU0 and GPU1 were both idle.
- Added missing wrapper `scripts/run_data_scaling_lp_frozen_lm_ums_3k_gpu0.cmd` following the existing LP wrapper pattern.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_3K_FROZEN_LM_UMS_LP_FORMAL_RUN completion and integration gate

### Results

- The 3k frozen-LM UMS LP run completed successfully.
- Wrapper PID at launch: `22220`; Python child PID observed during runtime: `12788`.
- Wrapper exit line: `EXITCODE 0`.
- End time: `2026/06/27 11:54:07.39`.
- Training reached `3000/3000` in about `6:26`.
- The LP trainer loaded `150` ViT backbone parameters from `outputs/data_scaling/frozen_lm_ums_3k/checkpoints/best.pt`.
- Linear-probe mode froze `150` backbone params and trained `10,766` head params.
- GPU0 used about `1.8 GiB` VRAM during runtime; GPU1 stayed idle.
- No active VIVID training process remained after completion.

### Commands / Inputs / Outputs

```powershell
Start-Process -FilePath cmd.exe -ArgumentList '/c','H:\Xiyao_Wang\021_260129VIVID\scripts\run_data_scaling_lp_frozen_lm_ums_3k_gpu0.cmd' -WindowStyle Hidden -PassThru
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'data_scaling_lp_frozen_lm_ums_3k|lp_frozen_lm_ums_3k|train_vit_baseline' } | Select-Object ProcessId,Name,CommandLine
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_lp_frozen_lm_ums_3k_gpu0.log -Pattern 'Loaded|Frozen|Trainable|Step [0-9]+|val_loss|macro_auc|macro_f1|micro_f1|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError'
Get-ChildItem outputs\data_scaling\lp_frozen_lm_ums_3k
python -m py_compile scripts\summarize_data_scaling_3k_bce_progress.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_phase4_revision_synthesis.py
python scripts\summarize_data_scaling_3k_bce_progress.py
python scripts\summarize_revision_execution_status.py
python scripts\summarize_revision_completion_gap_audit.py
python scripts\summarize_phase4_revision_synthesis.py
rg -n <old-3k-frozen-missing-wording-patterns> scripts outputs\final_tables vivid_med_revision_execution_plan.md -g "*.py" -g "*.csv" -g "*.md"
git diff --check
```

Outputs:

- Log: `outputs/logs/data_scaling_lp_frozen_lm_ums_3k_gpu0.log`.
- LP output dir: `outputs/data_scaling/lp_frozen_lm_ums_3k`.
- LP artifacts: `best.pt`, `final.pt`, `step_600.pt`, `step_1200.pt`, `step_1800.pt`, `step_2400.pt`, `step_3000.pt`, `metrics_final.json`, and `metrics_step_*.json`.
- Updated summary outputs:
  - `outputs/final_tables/data_scaling_3k_source_progress.csv/md`
  - `outputs/final_tables/data_scaling_3k_source_trajectory.csv`
  - `outputs/final_tables/data_scaling_3k_current_progress.csv/md`
  - `outputs/final_tables/data_scaling_3k_current_trajectory.csv`
  - `outputs/final_tables/revision_execution_status.csv/md`
  - `outputs/final_tables/revision_completion_gap_audit.csv/md`
  - `outputs/final_tables/llm_necessity.csv/md`
  - `outputs/final_tables/phase4_writing_claim_checklist.md`
- `revision_execution_status.csv` now has `49` rows and includes both `P1_DATA_SCALING_3K_FROZEN_LM_UMS_SOURCE_RUN` and `P1_DATA_SCALING_3K_FROZEN_LM_UMS_LP_FORMAL_RUN`.
- Stale search for old "3k frozen-LM missing" wording returned no matches.
- `git diff --check` passed with only pre-existing CRLF normalization warnings.
- `python scripts\validate_data_scaling_configs.py` returned nonzero by design after completed outputs were present: `Validated 24 configs`, `Failures: 8`, `LP blocked until source checkpoints: 6`.
- In the refreshed config-validation table, `lp_frozen_lm_ums_3k` is now a completed-output overwrite guard (`would overwrite completed run; LP source checkpoint already exists`), not a missing-source dependency.

### Metrics

| row | policy | step | val_loss | macro_auc | macro_f1 | micro_f1 |
|---|---|---:|---:|---:|---:|---:|
| no-LM 3k LP | final | final | 0.335240 | 0.734235 | 0.897409 | 0.865755 |
| no-LM 3k LP | best val / best AUC / best F1 | 200 | 0.310122 | 0.760698 | 0.904258 | 0.869941 |
| frozen-LM 3k LP | final | final | 0.310675 | 0.744936 | 0.896224 | 0.868267 |
| frozen-LM 3k LP | best val | 2800 | 0.307326 | 0.744139 | 0.898435 | 0.872453 |
| frozen-LM 3k LP | best AUC | 1600 | 0.325496 | 0.750523 | 0.888709 | 0.863243 |
| frozen-LM 3k LP | best F1 | 1000 | 0.327296 | 0.748312 | 0.903649 | 0.867988 |

Derived matched 3k LP interpretation:

- Final macro-AUC: frozen-LM `0.744936` vs no-LM `0.734235`, delta `+0.010701`.
- Final macro-F1: frozen-LM `0.896224` vs no-LM `0.897409`, delta `-0.001185`.
- Best-AUC policy: frozen-LM `0.750523` vs no-LM `0.760698`, so no-LM remains higher under best-AUC selection.
- This supports only a small metric-policy-dependent 3k final-AUC frozen-LM gain, not a broad low-data frozen-LM necessity claim.

### Failure Reason / Boundary

- No runtime failure, OOM, traceback, or nonzero exit occurred.
- There is no `training_history.json`; this is normal for `train_vit_baseline.py`, which writes `metrics_step_*.json` and `metrics_final.json` directly under the output dir.
- Frozen-LM source rows write val-loss provenance in log/checkpoints rather than source `metrics_*.json`; the updated 3k summarizer parses the source log for frozen-LM source val-loss trajectory.
- Config validation's nonzero exit after integration is not a failed training result; it reflects completed-output overwrite guards for already finished rows plus still-blocked 10k/30k LP dependencies.
- The original full data-scaling scope still lacks 10k/30k fixed-split rows; do not declare the original full scaling matrix complete.
- Because 1k matched LP is negative for frozen-LM macro-AUC and 3k best-AUC still favors no-LM, do not claim `500x` or broad low-data frozen-LM necessity.

### Next Step

- Treat 3k matched LP as now complete for the current priority pass.
- Keep the paper-facing claim as mixed/metric-policy-dependent:
  - UMS/schema contribution is still the main supported mechanism.
  - Frozen-LM has selected in-domain/difficult-field support and a small 3k final-AUC gain.
  - Answerability semantics and schema serialization dependence remain explicit boundaries.
- Only open 10k/30k data-scaling rows if the revision truly requires the original full scaling matrix; otherwise continue with final consistency/audit work rather than launching more long runs.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_10K_30K_GO_NO_GO_AUDIT execution-before

### Plan

- Do not launch any 10k/30k training in this task.
- Audit the remaining original-scope data-scaling rows after the 1k and 3k matched LP branches completed.
- Use current config-validation outputs, split sizes, existing checkpoints, GPU state, and observed 3k frozen-LM runtime to decide whether opening 10k/30k is scientifically necessary for the current priority pass.
- Preserve the user priority: do not continue SPD variants; prioritize claim-safe frozen-LM use cases, UMS/schema contribution, answerability semantics, and schema serialization boundaries.

### Commands

```powershell
Import-Csv outputs\final_tables\revision_completion_gap_audit.csv | Where-Object { $_.requirement -eq 'P1_DATA_SCALING' -or $_.requirement -eq 'Phase 4 paper tables and writing checklist' } | Format-List
Import-Csv outputs\final_tables\data_scaling_config_validation.csv | Where-Object { $_.size -eq '10k' -or $_.size -eq '30k' } | Select-Object config,method,size,train_records,train_patients,output_status,checkpoint_dependency,status,notes | Format-Table -AutoSize
Import-Csv outputs\final_tables\data_scaling_3k_current_progress.csv | Where-Object { $_.stage -eq 'linear_probe' } | Select-Object run_id,metric_policy,step,val_loss,macro_auc,macro_f1,micro_f1 | Format-Table -AutoSize
Get-ChildItem outputs\data_scaling -Directory | Where-Object { $_.Name -match '10k|30k' } | Select-Object Name,LastWriteTime
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '021_260129VIVID|train_cxr|train_vit_baseline|train_ums_classifier|data_scaling|schema' } | Select-Object ProcessId,Name,CommandLine
```

### Inputs

- `outputs/final_tables/revision_completion_gap_audit.csv`.
- `outputs/final_tables/data_scaling_config_validation.csv`.
- `outputs/final_tables/data_scaling_3k_current_progress.csv`.
- `configs/data_scaling/*10k*.yaml` and `configs/data_scaling/*30k*.yaml` as represented in the validation table.
- Current GPU/process state.
- Runtime evidence from completed 3k frozen-LM source: about `8:17:26` for 10000 source steps on one RTX 3090.

### Expected Outputs

- A go/no-go recommendation for 10k/30k data-scaling rows.
- A remaining-row inventory separating:
  - source rows that are launchable,
  - LP rows blocked until source checkpoints exist,
  - completed-output overwrite guards,
  - optional random-LM rows.
- A claim boundary for whether current paper-facing tables can proceed without 10k/30k.

### Stop Conditions

- Stop and do not launch training if current evidence already supports a claim-safe revision boundary and 10k/30k would only complete the original exhaustive matrix.
- Stop if any 10k/30k row would require multiple long frozen-LM source runs without a paper-critical decision it can change.
- Stop if GPU/process checks reveal another active training job or unexpected contention.
- Stop if the audit finds a missing required artifact in the current priority packet; fix the artifact/table boundary instead of launching new training.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_10K_30K_GO_NO_GO_AUDIT result

### Results

- Recommendation: **NO-GO for launching 10k/30k training in the current priority pass**.
- The current claim-safe revision packet is complete enough for the prioritized questions:
  - UMS/schema contribution is supported.
  - Frozen-LM use cases are bounded to in-domain/difficult-field and small 3k final-AUC signals.
  - Answerability semantics and schema serialization dependence are explicitly documented.
  - P2/SPD new variants remain deferred/out of scope.
- The remaining 10k/30k rows would complete the original exhaustive data-scaling matrix, but they are not required to repair the current paper-facing claim boundary.
- GPU0 and GPU1 were idle during the audit, and no VIVID training process was active; the no-go decision is scientific/prioritization-based, not caused by resource contention.

### Commands / Inputs / Outputs

```powershell
Import-Csv outputs\final_tables\revision_completion_gap_audit.csv | Where-Object { $_.requirement -eq 'P1_DATA_SCALING' -or $_.requirement -eq 'Phase 4 paper tables and writing checklist' } | Format-List
Import-Csv outputs\final_tables\data_scaling_config_validation.csv | Where-Object { $_.size -eq '10k' -or $_.size -eq '30k' } | ForEach-Object { "$($_.method),$($_.size),$($_.output_status),$($_.status),$($_.notes),$($_.checkpoint_dependency)" }
Import-Csv outputs\final_tables\data_scaling_3k_current_progress.csv | Where-Object { $_.stage -eq 'linear_probe' } | Select-Object run_id,metric_policy,step,val_loss,macro_auc,macro_f1,micro_f1
Get-ChildItem outputs\data_scaling -Directory | Where-Object { $_.Name -match '10k|30k' } | Select-Object Name,LastWriteTime
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '021_260129VIVID|train_cxr|train_vit_baseline|train_ums_classifier|data_scaling|schema' } | Select-Object ProcessId,Name,CommandLine
```

Key outputs:

- `P1_DATA_SCALING` status remains `partial_1k_matched_plus_3k_bce_source_and_matched_no_lm_frozen_lp`.
- Evidence strength at this audit was `strong_for_1k_matched_lp_and_3k_matched_lp_branch`; later `bce_10k` completion below supersedes this to include a 10k BCE source-control row.
- Gap at this audit was `10k/30k fixed-split matrix rows are not run`; later `bce_10k` completion below narrows the gap to matched 10k no-LM/frozen-LM and 30k rows.
- Phase 4 packet status is `completed_claim_synthesis` with explicit 10k/30k boundary.
- 10k/30k source rows that are launchable:
  - `bce_10k`, `bce_30k`
  - `frozen_lm_ums_10k`, `frozen_lm_ums_30k`
  - `no_lm_ums_10k`, `no_lm_ums_30k`
  - `random_lm_ums_30k` (optional / not prioritized)
- 10k/30k LP rows blocked until source checkpoints exist:
  - `lp_frozen_lm_ums_10k` -> `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/best.pt`
  - `lp_frozen_lm_ums_30k` -> `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/best.pt`
  - `lp_no_lm_ums_10k` -> `outputs/data_scaling/no_lm_ums_10k/best.pt`
  - `lp_no_lm_ums_30k` -> `outputs/data_scaling/no_lm_ums_30k/best.pt`
  - `lp_random_lm_ums_30k` -> `outputs/data_scaling/random_lm_ums_30k/checkpoints/best.pt`
- Existing 3k matched LP interpretation remains:
  - no-LM final macro-AUC `0.734235`, best-AUC `0.760698`.
  - frozen-LM final macro-AUC `0.744936`, best-AUC `0.750523`.
  - Final-AUC favors frozen-LM by `+0.010701`; best-AUC favors no-LM.

### Metrics / Cost-Relevant Estimate

| remaining row family | launchable now | dependency | expected cost boundary |
|---|---:|---|---|
| 10k BCE source | yes | none | moderate source run + dependent LP not needed for BCE |
| 30k BCE source | yes | none | moderate/long source run |
| 10k no-LM source | yes | none | lighter than frozen-LM source, but LP still required for matched comparison |
| 30k no-LM source | yes | none | lighter than frozen-LM source, but long |
| 10k frozen-LM source | yes | none | likely at least comparable to the 3k frozen-LM source runtime because max_steps is still 10000; observed 3k source was about `8:17:26` |
| 30k frozen-LM source | yes | none | likely long/high-memory Qwen source run; would occupy almost all one RTX 3090 |
| 10k/30k LP rows | no | wait for source `best.pt` | short after source exists |
| random-LM 30k | yes | none | optional and not prioritized by current user instruction |

### Failure Reason / Boundary

- No command failed unexpectedly in the audit.
- `outputs/data_scaling` currently has no `10k` or `30k` output directories, so these rows are genuinely not run.
- The no-go decision is not a claim that the original `P1_DATA_SCALING` scope is complete; it is a documented decision to stop broad matrix expansion because the current prioritized revision packet has enough evidence and clear limitations.
- If the paper/reviewer specifically requires the original full matrix, open a separate long-running execution-before task starting with one source row, not a batch launch.

### Next Step

- Do not launch 10k/30k now.
- Continue final consistency and paper-claim audit:
  - make sure final tables do not hide the 10k/30k gap,
  - keep 3k result language mixed/metric-policy-dependent,
  - preserve cost-table missing fields instead of imputing resource numbers,
  - keep P2 modules deferred unless a concrete failure slice is selected.

## 2026-06-27 Phase 4 / CURRENT_PRIORITY_COMPLETION_AUDIT execution-before

### Plan

- Produce a final current-priority audit table from authoritative generated artifacts.
- Separate two notions explicitly:
  - **current priority packet complete**: frozen-LM use cases, UMS/schema contribution, answerability semantics, schema serialization dependence, schema matrix, 1k/3k matched LP evidence, Phase 4 claim controls.
  - **original exhaustive plan incomplete**: 10k/30k data-scaling rows remain no-go/not run.
- Do not launch training.
- Do not claim broad frozen-LM necessity, schema-agnostic robustness, or full data-scaling completion.

### Commands

```powershell
python scripts\summarize_current_priority_completion_audit.py
Import-Csv outputs\final_tables\current_priority_completion_audit.csv | Format-List
Get-Content outputs\final_tables\current_priority_completion_audit.md
rg -n "broad low-data|500x|schema-agnostic|full data-scaling complete|10k/30k rows complete" outputs\final_tables scripts vivid_med_revision_execution_plan.md -g "*.md" -g "*.csv" -g "*.py"
python -m py_compile scripts\summarize_current_priority_completion_audit.py
git diff --check
```

### Inputs

- `outputs/final_tables/revision_completion_gap_audit.csv`.
- `outputs/final_tables/revision_execution_status.csv`.
- `outputs/final_tables/llm_necessity.csv`.
- `outputs/final_tables/schema_complexity_diagnostic_summary.csv`.
- `outputs/final_tables/data_scaling_3k_current_progress.csv`.
- `outputs/final_tables/phase4_writing_claim_checklist.md`.

### Expected Outputs

- `outputs/final_tables/current_priority_completion_audit.csv`.
- `outputs/final_tables/current_priority_completion_audit.md`.
- A clear `goal_completion_decision` row stating that current priority packet is complete, but the original exhaustive data-scaling matrix is not complete.

### Stop Conditions

- Stop and fix if any required input artifact is missing.
- Stop if generated audit implies 10k/30k completion.
- Stop if generated audit hides metric-policy dependence of the 3k frozen-LM gain.
- Stop if stale overclaim wording remains in active final tables/scripts.

## 2026-06-27 Phase 4 / CURRENT_PRIORITY_COMPLETION_AUDIT result

### Results

- Generated a current-priority completion audit without launching training.
- Output files:
  - `outputs/final_tables/current_priority_completion_audit.csv`
  - `outputs/final_tables/current_priority_completion_audit.md`
- The audit separates current-priority completion from original exhaustive-scope completion.
- Current-priority packet is complete for paper-claim cleanup:
  - UMS/schema contribution is supported.
  - Frozen-LM use case is bounded to in-domain/difficult-field signals plus a small 3k final-AUC gain.
  - Low-data scaling is complete for 1k/3k priority evidence but not for 10k/30k full matrix.
  - Schema complexity fixed-split S1/S2/S3 no-LM/frozen-LM source+LP matrix is complete.
  - Answerability semantics is complete as a semantic/protocol boundary, with calibration limitations retained.
  - Schema serialization dependence is complete as a limitation.
  - Cost/resource table remains complete with missing fields explicitly preserved.
  - P2/SPD new variants remain deferred/no-go.
- Full original experiment plan is **not** proven complete because 10k/30k data-scaling rows remain no-go/not run.

### Commands / Inputs / Outputs

```powershell
python scripts\summarize_current_priority_completion_audit.py
python -m py_compile scripts\summarize_current_priority_completion_audit.py
Import-Csv outputs\final_tables\current_priority_completion_audit.csv | Select-Object audit_item,decision,evidence_strength,remaining_gap | Format-Table -AutoSize
rg -n "broad low-data|500x|schema-agnostic|full data-scaling complete|10k/30k rows complete|entire original experiment execution plan is fully complete" outputs\final_tables scripts vivid_med_revision_execution_plan.md -g "*.md" -g "*.csv" -g "*.py"
git diff --check
```

Outputs:

- `python scripts\summarize_current_priority_completion_audit.py`: wrote audit to `outputs/final_tables`.
- `py_compile`: passed.
- `git diff --check`: passed with only pre-existing CRLF normalization warnings.
- Overclaim/stale search returned hits only in guardrail/forbidden-claim contexts or historical plan text, not as active positive claims.

### Metrics / Audit Rows

| audit_item | decision | evidence_strength | remaining_gap |
|---|---|---|---|
| UMS/schema contribution | `complete_for_current_priority` | strong | none for current claim boundary |
| Frozen-LM use case | `complete_bounded` | strong_for_subgroup_signal | not universal; external/NIH dominance remains weakened |
| Low-data scaling | `complete_for_1k_3k_priority_not_full_matrix` | strong_for_1k_matched_lp_and_3k_matched_lp_branch at this audit; later includes 10k BCE source-control | at this audit 10k/30k rows were not run; later `bce_10k` completion narrows the gap to matched 10k no-LM/frozen-LM and 30k rows |
| Schema complexity | `complete_fixed_split_matrix` | strong_for_fixed_split_formal_schema_source_lp_matrix | none for fixed-split S1/S2/S3 matrix |
| Answerability semantics | `complete_semantic_boundary_limited_calibration` | moderate | sample-level probability/OOD calibration remains missing |
| Schema serialization dependency | `complete_as_limitation` | strong_for_limitation | mitigation not implemented |
| Cost/resource table | `complete_with_missing_fields` | moderate | peak memory and some resource fields remain missing where old logs do not prove them |
| Phase 2 / SPD variants | `deferred_no_new_spd` | supported_by_failure_mining_and_user_constraint | no new module training in current pass |
| Phase 4 claim packet | `complete_for_current_evidence_boundary` | strong_for_current_evidence_boundary | full 10k/30k data scaling remains incomplete by original scope |
| Goal completion decision | `do_not_mark_goal_complete` | strong_for_existing_tables | original exhaustive P1 data-scaling still has unrun 10k/30k rows |

### Failure Reason / Boundary

- No script/runtime failure occurred.
- The broad overclaim search intentionally matched forbidden-language cells such as `Do not write a 500x...` and limitation text such as `not schema-agnostic`; these are guardrails, not positive claims.
- The audit deliberately says `do_not_mark_goal_complete` because the original exhaustive data-scaling matrix is not complete.
- This does not block using the current-priority packet for paper rewriting; it only blocks claiming that the entire original execution plan is fully complete.

### Next Step

- Keep the thread goal active rather than marking it complete.
- If the user accepts the documented no-go boundary as final scope closure, the current-priority packet can be treated as ready.
- If the original exhaustive scope must be finished literally, open a separate execution-before task for 10k/30k data scaling, one source row at a time.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_10K_BCE_SOURCE_RUN execution-before

### Plan

- Re-open the original exhaustive `P1_DATA_SCALING` scope one row at a time because the active thread goal still asks to complete the plan.
- Run exactly one launchable low-risk source-control row: `bce_10k`.
- Use GPU0 only.
- Do not launch 30k, no-LM, frozen-LM, random-LM, or LP rows in parallel.
- Treat this as a source-control data-scaling row, not frozen-LM/no-LM matched evidence.

### Commands

Preflight and launch:

```powershell
Import-Csv outputs\final_tables\data_scaling_config_validation.csv | Where-Object { $_.method -eq 'bce_10k' } | Format-List
Get-Content configs\data_scaling\bce_10k.yaml
if (Test-Path outputs\data_scaling\bce_10k) { Get-ChildItem outputs\data_scaling\bce_10k -Recurse -Depth 2 } else { 'outputs/data_scaling/bce_10k missing' }
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '021_260129VIVID|train_cxr|train_vit_baseline|train_ums_classifier|data_scaling|schema' } | Select-Object ProcessId,Name,CommandLine
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Start-Process -FilePath cmd.exe -ArgumentList '/c','H:\Xiyao_Wang\021_260129VIVID\scripts\run_data_scaling_bce_10k_source_gpu0.cmd' -WindowStyle Hidden -PassThru
```

Monitoring:

```powershell
Get-Content outputs\logs\data_scaling_bce_10k_source_gpu0.log -Tail 120
Get-ChildItem outputs\data_scaling\bce_10k
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv
```

### Inputs

- Config: `configs/data_scaling/bce_10k.yaml`.
- Train split: `data/splits/chexpert_train_10k.jsonl`.
- Validation split: `data/splits/chexpert_val_fixed.jsonl`.
- Trainer: `scripts/train_vit_baseline.py`.
- Wrapper: `scripts/run_data_scaling_bce_10k_source_gpu0.cmd`.

### Expected Outputs

- Log: `outputs/logs/data_scaling_bce_10k_source_gpu0.log`.
- Output dir: `outputs/data_scaling/bce_10k`.
- Artifacts under output dir: `best.pt`, `final.pt`, `metrics_final.json`, `metrics_step_*.json`, and periodic `step_*.pt`.
- Later integration into data-scaling progress/status/gap tables.

### Stop Conditions

- Stop before launch if config/splits are missing, if `outputs/data_scaling/bce_10k/final.pt` already exists, or if another VIVID training process is active.
- Stop during run on nonzero wrapper exit code, traceback, OOM, stalled log/checkpoint writes, or unexpected GPU contention.
- Do not compare this BCE source row directly to frozen-LM/no-LM LP rows; it is a control row for the 10k size.

### Preflight Results

- `data_scaling_config_validation.csv` marks `bce_10k` as `available`, `ok`.
- Config exists and targets `./outputs/data_scaling/bce_10k`.
- Output directory was absent before launch.
- No active VIVID training process was found after excluding the preflight query itself.
- GPU0 and GPU1 were idle.
- Added wrapper `scripts/run_data_scaling_bce_10k_source_gpu0.cmd` following the existing baseline source wrapper pattern.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_10K_BCE_SOURCE_RUN runtime gate

### Results

- Run launched successfully with wrapper PID `7984`; Python child PID observed as `3540`.
- Dataloaders initialized successfully.
- Train split loaded: `10000` samples, `312` train batches.
- Validation split loaded: `1000` samples, `32` validation batches.
- Model initialized with pretrained ViT backbone and 14-label BCE head.
- Trainable parameter groups:
  - backbone params: `85,798,656`
  - head params: `10,766`
- Training reached at least step `41/10000` during the first runtime gate.

### Runtime / Speed

- GPU0: about `4015 MiB / 24576 MiB`, about `162 W`.
- GPU1: idle.
- Observed training speed after startup: about `1.4-1.7 it/s`, roughly `1.5-2 h` before validation/checkpoint overhead.
- This is normal for the 10k BCE ViT baseline on this host and much lighter than frozen-LM source rows.

### Commands / Inputs / Outputs

```powershell
Start-Process -FilePath cmd.exe -ArgumentList '/c','H:\Xiyao_Wang\021_260129VIVID\scripts\run_data_scaling_bce_10k_source_gpu0.cmd' -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 25
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'data_scaling_bce_10k|bce_10k|train_vit_baseline' } | Select-Object ProcessId,Name,CommandLine
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Get-Content outputs\logs\data_scaling_bce_10k_source_gpu0.log -Tail 120
```

### Failure Reason / Boundary

- No runtime failure at this gate.
- No traceback, OOM, nonzero exit, or unexpected GPU contention was observed.
- This remains an in-progress source-control row; no metrics should be integrated until validation/checkpoint artifacts exist.

### Next Step

- Continue monitoring to the first validation gate at step `500`.
- Stop only if traceback, OOM, stalled log/checkpoint writes, nonzero exit, or unexpected GPU contention appears.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_10K_BCE_SOURCE_RUN step500-1500 gate

### Results

- Run remained active and stable through step `1500`.
- Step `500`: `val_loss=0.3285`; `metrics_step_500.json` saved.
- Step `1000`: `val_loss=0.3204`; `metrics_step_1000.json` and `step_1000.pt` saved.
- Step `1500`: `val_loss=0.3059`; `metrics_step_1500.json` saved and `best.pt` updated.
- Current observed progress after this gate was about `1645/10000`.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or `EXITCODE` failure line was observed in the latest log scan.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 420
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'data_scaling_bce_10k|bce_10k|train_vit_baseline' } | Select-Object ProcessId,Name,CommandLine
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_bce_10k_source_gpu0.log -Pattern 'Step [0-9]+: val_loss|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError'
Get-ChildItem outputs\data_scaling\bce_10k
Get-Content outputs\logs\data_scaling_bce_10k_source_gpu0.log -Tail 80
```

Outputs:

- `outputs/data_scaling/bce_10k/metrics_step_500.json` exists.
- `outputs/data_scaling/bce_10k/metrics_step_1000.json` exists.
- `outputs/data_scaling/bce_10k/step_1000.pt` exists.
- `outputs/data_scaling/bce_10k/metrics_step_1500.json` exists.
- `outputs/data_scaling/bce_10k/best.pt` exists and was updated at step `1500`.
- GPU0 remained around `4.0 GiB` VRAM; GPU1 remained idle.

### Runtime / Speed

- After warmup, observed speed improved to about `5.0-5.6 it/s`.
- This is much faster than frozen-LM source rows and consistent with a lightweight BCE ViT source-control row.

### Failure Reason / Boundary

- No runtime failure at this gate.
- This remains an in-progress source-control row; do not integrate until `final.pt`, `metrics_final.json`, and wrapper `EXITCODE 0` exist.

### Next Step

- Continue monitoring toward completion.
- At completion, extract final, best-val, best-AUC, and best-F1 metrics and update data-scaling status/gap tables.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_10K_BCE_SOURCE_RUN completion and integration gate

### Results

- The 10k BCE source-control row completed successfully.
- Wrapper exit line: `EXITCODE 0`.
- End time: `2026/06/27 12:53:19.79`.
- Training reached `10000/10000` in about `36:20`, around `4.59 it/s`.
- `final.pt`, `metrics_final.json`, `step_10000.pt`, and `metrics_step_10000.json` were saved.
- Best validation loss occurred at step `2000`, after which validation loss rose sharply, so final checkpoint is overfit relative to best-val/best-AUC checkpoints.
- GPU0 and GPU1 were idle after completion; no training process remained.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 1800
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'data_scaling_bce_10k|bce_10k|train_vit_baseline' } | Select-Object ProcessId,Name,CommandLine
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_bce_10k_source_gpu0.log -Pattern 'Step [0-9]+: val_loss|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 80
Get-ChildItem outputs\data_scaling\bce_10k
Get-Content outputs\data_scaling\bce_10k\metrics_final.json
Get-Content outputs\data_scaling\bce_10k\metrics_step_2000.json
python scripts\summarize_data_scaling_10k_progress.py
python scripts\validate_data_scaling_configs.py
python scripts\summarize_revision_execution_status.py
python scripts\summarize_revision_completion_gap_audit.py
python scripts\summarize_current_priority_completion_audit.py
python -m py_compile scripts\summarize_data_scaling_10k_progress.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_current_priority_completion_audit.py
```

Outputs:

- Log: `outputs/logs/data_scaling_bce_10k_source_gpu0.log`.
- Run dir: `outputs/data_scaling/bce_10k`.
- Summary outputs:
  - `outputs/final_tables/data_scaling_10k_progress.csv`
  - `outputs/final_tables/data_scaling_10k_progress.md`
  - `outputs/final_tables/data_scaling_10k_trajectory.csv`
  - refreshed `outputs/final_tables/revision_execution_status.csv/md`
  - refreshed `outputs/final_tables/revision_completion_gap_audit.csv/md`
  - refreshed `outputs/final_tables/current_priority_completion_audit.csv/md`
- `revision_execution_status.csv` now has `50` rows and includes `P1_DATA_SCALING_10K_BCE_SOURCE_RUN`.
- `data_scaling_config_validation.csv` now marks `bce_10k` as `completed_run_exists`, `fail`, `would overwrite completed run`; this is the intended completed-output overwrite guard, not a failed run.

### Metrics

| policy | step | val_loss | macro_auc | macro_f1 | micro_f1 |
|---|---:|---:|---:|---:|---:|
| final checkpoint | final | 0.793052 | 0.709233 | 0.897106 | 0.873011 |
| best validation loss | 2000 | 0.297180 | 0.743417 | 0.902348 | 0.877477 |
| best macro-AUC | 3500 | 0.328254 | 0.757318 | 0.905356 | 0.872732 |
| best macro-F1 | 4000 | 0.304797 | 0.748656 | 0.907478 | 0.875523 |

Validation-loss trajectory boundary:

- step `500`: `0.3285`
- step `1000`: `0.3204`
- step `1500`: `0.3059`
- step `2000`: `0.2972`
- step `3000`: `0.3002`
- step `5000`: `0.3741`
- step `7000`: `0.6432`
- step `9000`: `0.7803`
- step `10000`: `0.7931`

### Failure Reason / Boundary

- No runtime failure, OOM, traceback, or nonzero exit occurred.
- The row is a 10k BCE source-control row only; it does not complete matched 10k no-LM/frozen-LM evidence.
- The final checkpoint is not the best validation checkpoint; any table should preserve final, best-val, best-AUC, and best-F1 policies separately.
- The original exhaustive data-scaling scope remains incomplete: matched 10k no-LM/frozen-LM source+LP rows and 30k rows are not run.

### Next Step

- Continue original `P1_DATA_SCALING` one row at a time.
- The next scientifically useful launchable row is `P1_DATA_SCALING_10K_NO_LM_UMS_SOURCE_RUN`, because it creates the source checkpoint needed by `lp_no_lm_ums_10k` and is lighter than frozen-LM source training.
- Do not launch frozen-LM 10k or any 30k row in parallel with the next source row.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_10K_NO_LM_UMS_SOURCE_RUN execution-before

### Plan

- Run exactly one next source row: formal 10k no-LM UMS source training.
- Use GPU0 only.
- Do not launch frozen-LM 10k, 30k, or dependent LP rows in parallel.
- This row is required before `lp_no_lm_ums_10k` can run.
- Keep source metrics separate from LP metrics and from historical P0 main-table rows.

### Commands

Preflight and launch:

```powershell
Import-Csv outputs\final_tables\data_scaling_config_validation.csv | Where-Object { $_.method -eq 'no_lm_ums_10k' } | Format-List
Get-Content configs\data_scaling\no_lm_ums_10k.yaml
if (Test-Path outputs\data_scaling\no_lm_ums_10k) { Get-ChildItem outputs\data_scaling\no_lm_ums_10k -Recurse -Depth 2 } else { 'outputs/data_scaling/no_lm_ums_10k missing' }
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '021_260129VIVID|train_cxr|train_vit_baseline|train_ums_classifier|data_scaling|schema' } | Select-Object ProcessId,Name,CommandLine
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Start-Process -FilePath cmd.exe -ArgumentList '/c','H:\Xiyao_Wang\021_260129VIVID\scripts\run_data_scaling_no_lm_ums_10k_source_gpu0.cmd' -WindowStyle Hidden -PassThru
```

Monitoring:

```powershell
Get-Content outputs\logs\data_scaling_no_lm_ums_10k_source_gpu0.log -Tail 120
Get-ChildItem outputs\data_scaling\no_lm_ums_10k
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv
```

### Inputs

- Config: `configs/data_scaling/no_lm_ums_10k.yaml`.
- Train split: `data/splits/chexpert_train_10k.jsonl`.
- Validation split: `data/splits/chexpert_val_fixed.jsonl`.
- Trainer: `scripts/train_ums_classifier.py`.
- Wrapper: `scripts/run_data_scaling_no_lm_ums_10k_source_gpu0.cmd`.

### Expected Outputs

- Log: `outputs/logs/data_scaling_no_lm_ums_10k_source_gpu0.log`.
- Output dir: `outputs/data_scaling/no_lm_ums_10k`.
- Expected artifacts: `best.pt`, `final.pt`, `metrics_final.json`, `metrics_step_*.json`, periodic `step_*.pt`.
- Source checkpoint for future LP: `outputs/data_scaling/no_lm_ums_10k/best.pt`.

### Stop Conditions

- Stop before launch if config/splits are missing, if `outputs/data_scaling/no_lm_ums_10k/final.pt` already exists, or if another VIVID training process is active.
- Stop during run on nonzero wrapper exit code, traceback, OOM, stalled log/checkpoint writes, or unexpected GPU contention.
- Do not run `lp_no_lm_ums_10k` until this source row completes with `EXITCODE 0` and `best.pt` exists.

### Preflight Results

- `data_scaling_config_validation.csv` marks `no_lm_ums_10k` as `available`, `ok`.
- Config exists and targets `./outputs/data_scaling/no_lm_ums_10k`.
- Output directory was absent before launch.
- No active VIVID training process was found after excluding the preflight query itself.
- GPU0 and GPU1 were idle.
- Added wrapper `scripts/run_data_scaling_no_lm_ums_10k_source_gpu0.cmd` following the existing no-LM source wrapper pattern.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_10K_NO_LM_UMS_SOURCE_RUN runtime and step500 gate

### Results

- The 10k no-LM UMS source run launched successfully on GPU0.
- Wrapper process: `cmd.exe` PID `22632`.
- Python training process: `python.exe` PID `22596`.
- Start time from wrapper/log: `2026/06/27 13:06:09`.
- Training reached the first validation gate at step `500`.
- Step `500`: `val_loss=0.7515`, `macro_auc=0.6575525389977871`.
- `metrics_step_500.json`, `progress.json`, and `best.pt` were written under `outputs/data_scaling/no_lm_ums_10k`.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or wrapper `EXITCODE` failure line was observed at this gate.

### Commands / Inputs / Outputs

```powershell
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'data_scaling_no_lm_ums_10k|no_lm_ums_10k|train_ums_classifier' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_no_lm_ums_10k_source_gpu0.log -Pattern 'Step [0-9]+: val_loss|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 100
Get-ChildItem outputs\data_scaling\no_lm_ums_10k
Get-Content outputs\logs\data_scaling_no_lm_ums_10k_source_gpu0.log -Tail 120
```

Outputs:

- Log: `outputs/logs/data_scaling_no_lm_ums_10k_source_gpu0.log`.
- Run dir: `outputs/data_scaling/no_lm_ums_10k`.
- First validation artifact: `outputs/data_scaling/no_lm_ums_10k/metrics_step_500.json`.
- First checkpoint artifact: `outputs/data_scaling/no_lm_ums_10k/best.pt`.

### Runtime / Speed

- GPU0 at the runtime gate: about `4.0-4.3 GiB / 24 GiB` VRAM.
- GPU1 was idle.
- Observed GPU power varied from about `78 W` to `217 W` around startup and validation, consistent with a light no-LM source classifier row.
- Observed training speed before the first validation was about `4.8-5.0 it/s`.
- This is normal for this host and this row; it is much lighter than frozen-LM source training and similar to the completed 10k BCE row.

### Failure Reason / Boundary

- No runtime failure at this gate.
- The low VRAM/power usage is expected for no-LM source training and is not evidence of a stalled run.
- This remains an in-progress source row; do not integrate final metrics until `final.pt`, `metrics_final.json`, and wrapper `EXITCODE 0` exist.
- Do not launch `lp_no_lm_ums_10k` until this source row completes successfully and `best.pt` is finalized.

### Next Step

- Continue monitoring through later validation/checkpoint gates.
- Stop only if traceback, OOM, stalled log/checkpoint writes, nonzero wrapper exit, or unexpected GPU contention appears.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_10K_NO_LM_UMS_SOURCE_RUN step1000-1500 gate

### Results

- Run remained active and stable through step `1500`.
- Step `1000`: `val_loss=0.7225`, `macro_auc=0.6997557074523909`.
- Step `1500`: `val_loss=0.7203`, `macro_auc=0.7057085029090314`.
- Current observed progress after this gate was around `1526/10000`.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or `EXITCODE` failure line was observed in the latest log scan.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 180
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'data_scaling_no_lm_ums_10k|no_lm_ums_10k|train_ums_classifier' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_no_lm_ums_10k_source_gpu0.log -Pattern 'Step [0-9]+: val_loss|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 80
Get-Content outputs\logs\data_scaling_no_lm_ums_10k_source_gpu0.log -Tail 60
```

Outputs:

- `outputs/data_scaling/no_lm_ums_10k/metrics_step_1000.json` exists.
- `outputs/data_scaling/no_lm_ums_10k/metrics_step_1500.json` exists.
- `outputs/data_scaling/no_lm_ums_10k/step_1000.pt` is expected from the configured save interval.
- `outputs/data_scaling/no_lm_ums_10k/best.pt` has been available since step `500` and should continue tracking validation improvements.

### Runtime / Speed

- GPU0 remained the active device, around `4.3 GiB / 24 GiB` VRAM and about `242 W` in the sampled check.
- GPU1 remained idle.
- Observed speed around this gate remained near `4.5-5.0 it/s` outside validation overhead.
- The slow-looking `2.15s/it` immediately after validation is tqdm smoothing after a validation pause, not evidence of persistent slowdown.

### Failure Reason / Boundary

- No runtime failure at this gate.
- This remains an in-progress source row; do not integrate final metrics until `final.pt`, `metrics_final.json`, and wrapper `EXITCODE 0` exist.

### Next Step

- Continue monitoring toward completion.
- At completion, extract final, best-val, best-AUC, and best-F1 metrics and refresh the 10k data-scaling tables.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_10K_NO_LM_UMS_SOURCE_RUN mid-run gate

### Results

- Run remained active and stable through at least step `4500`.
- Step `2000`: `val_loss=0.7150`, `macro_auc=0.7241980680575527`.
- Step `2500`: `val_loss=0.7051`, `macro_auc=0.7194912260103847`.
- Step `3000`: `val_loss=0.7100`, `macro_auc=0.717980440044114`.
- Step `3500`: `val_loss=0.7097`, `macro_auc=0.7131779552833334`.
- Step `4000`: `val_loss=0.7286`, `macro_auc=0.7055550789448346`.
- Step `4500`: `val_loss=0.7681`, `macro_auc=0.6843287274675497`.
- Current observed progress after this gate was around `4553/10000`.
- `metrics_step_*.json` and periodic `step_*.pt` artifacts were written as expected.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or `EXITCODE` failure line was observed in the latest log scan.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 600
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'data_scaling_no_lm_ums_10k|no_lm_ums_10k|train_ums_classifier' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_no_lm_ums_10k_source_gpu0.log -Pattern 'Step [0-9]+: val_loss|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 120
Get-ChildItem outputs\data_scaling\no_lm_ums_10k | Select-Object Name,Length,LastWriteTime | Sort-Object LastWriteTime | Format-Table -AutoSize
Get-Content outputs\logs\data_scaling_no_lm_ums_10k_source_gpu0.log -Tail 80
```

Outputs:

- Metrics through `metrics_step_4500.json` exist.
- Periodic checkpoints through `step_4000.pt` exist.
- `best.pt` was updated at step `2500`, consistent with best validation-loss policy so far.

### Runtime / Speed

- GPU0 remained active at about `4.3 GiB / 24 GiB` VRAM and about `249 W` in the sampled check.
- GPU1 remained idle.
- Training speed remained around `4.5-5.1 it/s` outside validation pauses.
- Current runtime behavior is normal for this host and row; there is no evidence of virus-like or unrelated automation contention from this run snapshot.

### Metrics / Interpretation

- Best validation loss so far is around step `2500`.
- Best macro-AUC so far is around step `2000`.
- Later steps show overfitting or metric degradation, so completion integration must preserve final, best-val, best-AUC, and best-F1 policies separately.

### Failure Reason / Boundary

- No runtime failure at this gate.
- The row is not complete; final metrics are not yet available.

### Next Step

- Continue monitoring to completion.
- On completion, extract the full metric-policy table and compare against `bce_10k`; do not overclaim from final-only metrics.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_10K_NO_LM_UMS_SOURCE_RUN completion and integration gate

### Results

- The 10k no-LM UMS source row completed successfully.
- Wrapper exit line: `EXITCODE 0`.
- End time: `2026/06/27 13:43:40.24`.
- Training reached `10000/10000` in about `37:14`, around `4.48 it/s` including validation overhead.
- `final.pt`, `metrics_final.json`, `step_10000.pt`, and `metrics_step_10000.json` were saved.
- `best.pt` exists and was last updated at step `2500`, consistent with best validation-loss policy.
- GPU0 and GPU1 were idle after completion; no training process remained.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 420
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'data_scaling_no_lm_ums_10k|no_lm_ums_10k|train_ums_classifier' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_no_lm_ums_10k_source_gpu0.log -Pattern 'Step [0-9]+: val_loss|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 200
Get-ChildItem outputs\data_scaling\no_lm_ums_10k | Select-Object Name,Length,LastWriteTime | Sort-Object LastWriteTime | Format-Table -AutoSize
Get-Content outputs\logs\data_scaling_no_lm_ums_10k_source_gpu0.log -Tail 120
python scripts\summarize_data_scaling_10k_progress.py
python scripts\summarize_revision_execution_status.py
python scripts\summarize_revision_completion_gap_audit.py
python scripts\summarize_current_priority_completion_audit.py
python -m py_compile scripts\summarize_data_scaling_10k_progress.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_current_priority_completion_audit.py
python scripts\validate_data_scaling_configs.py
```

Outputs:

- Log: `outputs/logs/data_scaling_no_lm_ums_10k_source_gpu0.log`.
- Run dir: `outputs/data_scaling/no_lm_ums_10k`.
- Summary outputs refreshed:
  - `outputs/final_tables/data_scaling_10k_progress.csv`
  - `outputs/final_tables/data_scaling_10k_progress.md`
  - `outputs/final_tables/data_scaling_10k_trajectory.csv`
  - `outputs/final_tables/revision_execution_status.csv`
  - `outputs/final_tables/revision_completion_gap_audit.csv`
  - `outputs/final_tables/current_priority_completion_audit.csv`
- `revision_execution_status.csv` now has `51` rows and includes `P1_DATA_SCALING_10K_NO_LM_UMS_SOURCE_RUN`.
- `data_scaling_config_validation.csv` now marks `no_lm_ums_10k` as `completed_run_exists`, `fail`, `would overwrite completed run`; this is the intended completed-output overwrite guard, not a failed training run.

### Metrics

| policy | step | val_loss | macro_auc | macro_f1 | micro_f1 |
|---|---:|---:|---:|---:|---:|
| final checkpoint | final | 1.587877 | 0.661644 | 0.464942 | 0.597243 |
| best validation loss | 2500 | 0.705094 | 0.719491 | 0.315222 | 0.565268 |
| best macro-AUC | 2000 | 0.714983 | 0.724198 | 0.316274 | 0.552068 |
| best macro-F1 | 6000 | 1.120205 | 0.684607 | 0.495529 | 0.611030 |

Validation-loss trajectory boundary:

- step `500`: `0.7515`
- step `1000`: `0.7225`
- step `1500`: `0.7203`
- step `2000`: `0.7150`
- step `2500`: `0.7051`
- step `3000`: `0.7100`
- step `4500`: `0.7681`
- step `6000`: `1.1202`
- step `8000`: `1.4468`
- step `10000`: `1.5879`

### Failure Reason / Boundary

- No runtime failure, OOM, traceback, or nonzero exit occurred.
- The row is a 10k no-LM UMS source row only; it creates the checkpoint required by `lp_no_lm_ums_10k`.
- It is not a completed LP result and must not be compared directly against LP macro-AUC rows.
- The final checkpoint strongly overfits relative to best-val and best-AUC checkpoints; tables must preserve metric policies separately.
- The original exhaustive data-scaling scope remains incomplete: 10k no-LM LP, 10k frozen-LM source+LP, and 30k rows are not run.

### Next Step

- Continue original `P1_DATA_SCALING` one row at a time.
- The next dependency-respecting launchable row is `P1_DATA_SCALING_10K_NO_LM_UMS_LP_RUN`, because `outputs/data_scaling/no_lm_ums_10k/best.pt` now exists.
- Do not launch frozen-LM 10k or any 30k row in parallel with this dependent LP row.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_10K_NO_LM_UMS_LP_RUN execution-before

### Plan

- Run exactly one dependent LP row: formal 10k no-LM UMS linear probe.
- Use GPU0 only.
- Initialize from `outputs/data_scaling/no_lm_ums_10k/best.pt`.
- Freeze the backbone and train only the LP head according to `configs/data_scaling/lp_no_lm_ums_10k.yaml`.
- Keep this LP metric family separate from source-classifier metrics.
- Do not launch frozen-LM 10k, 30k, or any SPD variant in parallel.

### Commands

Preflight and launch:

```powershell
Import-Csv outputs\final_tables\data_scaling_config_validation.csv | Where-Object { $_.method -eq 'lp_no_lm_ums_10k' } | Format-List
Get-Content configs\data_scaling\lp_no_lm_ums_10k.yaml
if (Test-Path outputs\data_scaling\lp_no_lm_ums_10k) { Get-ChildItem outputs\data_scaling\lp_no_lm_ums_10k -Recurse -Depth 2 } else { 'outputs/data_scaling/lp_no_lm_ums_10k missing' }
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '021_260129VIVID|train_cxr|train_vit_baseline|train_ums_classifier|data_scaling|schema' } | Select-Object ProcessId,Name,CommandLine
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Start-Process -FilePath cmd.exe -ArgumentList '/c','H:\Xiyao_Wang\021_260129VIVID\scripts\run_data_scaling_lp_no_lm_ums_10k_gpu0.cmd' -WindowStyle Hidden -PassThru
```

Monitoring:

```powershell
Get-Content outputs\logs\data_scaling_lp_no_lm_ums_10k_gpu0.log -Tail 120
Get-ChildItem outputs\data_scaling\lp_no_lm_ums_10k
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv
```

### Inputs

- Config: `configs/data_scaling/lp_no_lm_ums_10k.yaml`.
- Train split: `data/splits/chexpert_train_10k.jsonl`.
- Validation split: `data/splits/chexpert_val_fixed.jsonl`.
- Source checkpoint: `outputs/data_scaling/no_lm_ums_10k/best.pt`.
- Trainer: `scripts/train_vit_baseline.py`.
- Wrapper: `scripts/run_data_scaling_lp_no_lm_ums_10k_gpu0.cmd`.

### Expected Outputs

- Log: `outputs/logs/data_scaling_lp_no_lm_ums_10k_gpu0.log`.
- Output dir: `outputs/data_scaling/lp_no_lm_ums_10k`.
- Expected artifacts: `best.pt`, `final.pt`, `metrics_final.json`, `metrics_step_*.json`, periodic `step_*.pt`.
- Updated summaries after completion: `outputs/final_tables/data_scaling_10k_progress.csv/md`, `revision_execution_status.csv/md`, `revision_completion_gap_audit.csv/md`, and `current_priority_completion_audit.csv/md`.

### Stop Conditions

- Stop before launch if config/splits are missing, source checkpoint is absent, `outputs/data_scaling/lp_no_lm_ums_10k/final.pt` already exists, or another VIVID training process is active.
- Stop during run on nonzero wrapper exit code, traceback, OOM, stalled log/checkpoint writes, or unexpected GPU contention.
- Do not run frozen-LM 10k until this dependent LP row completes or fails with a documented reason.

### Preflight Results

- `data_scaling_config_validation.csv` marks `lp_no_lm_ums_10k` as `available`, `ok`.
- The checkpoint dependency `outputs/data_scaling/no_lm_ums_10k/best.pt` exists.
- Config exists and targets `./outputs/data_scaling/lp_no_lm_ums_10k`.
- Output directory was absent before launch.
- No active VIVID training process was found after excluding the preflight query itself.
- GPU0 and GPU1 were idle.
- Added wrapper `scripts/run_data_scaling_lp_no_lm_ums_10k_gpu0.cmd` following the existing 3k LP wrapper pattern.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_10K_NO_LM_UMS_LP_RUN runtime gate

### Results

- The 10k no-LM UMS LP run launched successfully on GPU0.
- Wrapper process: `cmd.exe` PID `21672`.
- Python training process: `python.exe` PID `16912`.
- Start time from wrapper/log: `2026/06/27 13:50:26.96`.
- The trainer loaded `outputs/data_scaling/no_lm_ums_10k/best.pt`.
- Loaded ViT backbone params: `150`; missing keys were only the LP head (`head.weight`, `head.bias`).
- Linear probe mode froze `150` params and left `10,766` trainable head parameters.
- Training reached at least step `400`; monitoring tail observed progress around step `563/3000`.
- Step `200`: `val_loss=0.2912`; `metrics_step_200.json` and `best.pt` were written.
- Step `400`: `val_loss=0.2947`; `metrics_step_400.json` was written.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or wrapper `EXITCODE` failure line was observed at this gate.

### Commands / Inputs / Outputs

```powershell
Start-Process -FilePath cmd.exe -ArgumentList '/c','H:\Xiyao_Wang\021_260129VIVID\scripts\run_data_scaling_lp_no_lm_ums_10k_gpu0.cmd' -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 30
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'data_scaling_lp_no_lm_ums_10k|lp_no_lm_ums_10k|train_vit_baseline' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Get-Content outputs\logs\data_scaling_lp_no_lm_ums_10k_gpu0.log -Tail 160
Start-Sleep -Seconds 80
Select-String -Path outputs\logs\data_scaling_lp_no_lm_ums_10k_gpu0.log -Pattern 'Step [0-9]+: val_loss|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 80
Get-ChildItem outputs\data_scaling\lp_no_lm_ums_10k
```

Outputs:

- Log: `outputs/logs/data_scaling_lp_no_lm_ums_10k_gpu0.log`.
- Run dir: `outputs/data_scaling/lp_no_lm_ums_10k`.
- First validation artifacts:
  - `outputs/data_scaling/lp_no_lm_ums_10k/metrics_step_200.json`
  - `outputs/data_scaling/lp_no_lm_ums_10k/metrics_step_400.json`
  - `outputs/data_scaling/lp_no_lm_ums_10k/best.pt`

### Runtime / Speed

- GPU0 at the runtime gate: about `1.8-1.9 GiB / 24 GiB` VRAM and about `84-88 W`.
- GPU1 was idle.
- Observed training speed after warmup was about `5-10 it/s`, with expected fluctuations from data loading and validation.
- This is normal for a frozen-backbone LP row and much lighter than source training.

### Failure Reason / Boundary

- No runtime failure at this gate.
- The low VRAM/power usage is expected for a linear-probe row and is not evidence of a stalled run.
- This remains an in-progress LP row; do not integrate final metrics until `final.pt`, `metrics_final.json`, and wrapper `EXITCODE 0` exist.

### Next Step

- Continue monitoring toward completion.
- Stop only if traceback, OOM, stalled log/checkpoint writes, nonzero wrapper exit, or unexpected GPU contention appears.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_10K_NO_LM_UMS_LP_RUN completion and integration gate

### Results

- The 10k no-LM UMS LP row completed successfully.
- Wrapper exit line: `EXITCODE 0`.
- End time: `2026/06/27 13:58:33.18`.
- Training reached `3000/3000` in about `7:44`, around `6.46 it/s` including validation overhead.
- `final.pt`, `metrics_final.json`, `step_3000.pt`, and `metrics_step_3000.json` were saved.
- `best.pt` was updated at step `2800`, consistent with best validation-loss policy.
- GPU0 and GPU1 were idle after completion; no training process remained.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 420
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'data_scaling_lp_no_lm_ums_10k|lp_no_lm_ums_10k|train_vit_baseline' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_lp_no_lm_ums_10k_gpu0.log -Pattern 'Step [0-9]+: val_loss|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 160
Get-ChildItem outputs\data_scaling\lp_no_lm_ums_10k
Get-Content outputs\logs\data_scaling_lp_no_lm_ums_10k_gpu0.log -Tail 120
python scripts\summarize_data_scaling_10k_progress.py
python scripts\summarize_revision_execution_status.py
python scripts\summarize_revision_completion_gap_audit.py
python scripts\summarize_current_priority_completion_audit.py
python -m py_compile scripts\summarize_data_scaling_10k_progress.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_current_priority_completion_audit.py
python scripts\validate_data_scaling_configs.py
```

Outputs:

- Log: `outputs/logs/data_scaling_lp_no_lm_ums_10k_gpu0.log`.
- Run dir: `outputs/data_scaling/lp_no_lm_ums_10k`.
- Summary outputs refreshed:
  - `outputs/final_tables/data_scaling_10k_progress.csv`
  - `outputs/final_tables/data_scaling_10k_progress.md`
  - `outputs/final_tables/data_scaling_10k_trajectory.csv`
  - `outputs/final_tables/revision_execution_status.csv`
  - `outputs/final_tables/revision_completion_gap_audit.csv`
  - `outputs/final_tables/current_priority_completion_audit.csv`
- `revision_execution_status.csv` now has `52` rows and includes `P1_DATA_SCALING_10K_NO_LM_UMS_LP_RUN`.
- `data_scaling_config_validation.csv` now marks `lp_no_lm_ums_10k` as `completed_run_exists`, `fail`, `would overwrite completed run; LP source checkpoint already exists`; this is the intended completed-output overwrite guard, not a failed training run.

### Metrics

| policy | step | val_loss | macro_auc | macro_f1 | micro_f1 |
|---|---:|---:|---:|---:|---:|
| final checkpoint | final | 0.278615 | 0.771488 | 0.907583 | 0.881105 |
| best validation loss | 2800 | 0.277527 | 0.772206 | 0.909461 | 0.884175 |
| best macro-AUC | 600 | 0.287460 | 0.786171 | 0.904262 | 0.875244 |
| best macro-F1 | 2600 | 0.278504 | 0.772228 | 0.911695 | 0.883896 |

Validation-loss trajectory boundary:

- step `200`: `0.2912`
- step `400`: `0.2947`
- step `600`: `0.2875`
- step `1000`: `0.2819`
- step `1400`: `0.2805`
- step `2000`: `0.2802`
- step `2400`: `0.2789`
- step `2600`: `0.2785`
- step `2800`: `0.2775`
- step `3000`: `0.2786`

### Failure Reason / Boundary

- No runtime failure, OOM, traceback, or nonzero exit occurred.
- This row completes the 10k no-LM LP branch.
- It does not complete matched 10k frozen-LM evidence; 10k frozen-LM source+LP remains required for a 10k frozen-vs-no-LM comparison.
- Keep LP metrics separate from source-classifier metrics.

### Next Step

- Since the user allows dual-GPU execution when dependencies do not conflict, run independent source rows in parallel:
  - GPU0: `P1_DATA_SCALING_10K_FROZEN_LM_UMS_SOURCE_RUN`.
  - GPU1: `P1_DATA_SCALING_30K_BCE_SOURCE_RUN`.
- Do not launch `lp_frozen_lm_ums_10k` until the frozen-LM 10k source row completes and `best.pt` exists.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_DUAL_SOURCE_BATCH_10K_FROZEN_30K_BCE execution-before

### Plan

- Run two independent source rows in parallel because their dependency order does not conflict:
  - GPU0: `P1_DATA_SCALING_10K_FROZEN_LM_UMS_SOURCE_RUN`.
  - GPU1: `P1_DATA_SCALING_30K_BCE_SOURCE_RUN`.
- Do not launch any dependent LP rows during this batch.
- Do not launch 10k frozen-LM LP until `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/best.pt` exists and the source wrapper exits with code `0`.
- Keep source metrics separate from LP metrics and from historical P0 rows.
- Treat `bce_30k` as a source-control row; its split has `29000` records despite the 30k label.

### Commands

Preflight:

```powershell
Import-Csv outputs\final_tables\data_scaling_config_validation.csv | Where-Object { $_.method -in @('frozen_lm_ums_10k','bce_30k') } | Format-List
Get-Content configs\data_scaling\frozen_lm_ums_10k.yaml
Get-Content configs\data_scaling\bce_30k.yaml
if (Test-Path outputs\data_scaling\frozen_lm_ums_10k) { Get-ChildItem outputs\data_scaling\frozen_lm_ums_10k -Recurse -Depth 2 } else { 'outputs/data_scaling/frozen_lm_ums_10k missing' }
if (Test-Path outputs\data_scaling\bce_30k) { Get-ChildItem outputs\data_scaling\bce_30k -Recurse -Depth 2 } else { 'outputs/data_scaling/bce_30k missing' }
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '021_260129VIVID|train_cxr|train_vit_baseline|train_ums_classifier|data_scaling|schema' } | Select-Object ProcessId,Name,CommandLine
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
```

Launch:

```powershell
Start-Process -FilePath cmd.exe -ArgumentList '/c','H:\Xiyao_Wang\021_260129VIVID\scripts\run_data_scaling_frozen_lm_ums_10k_source_gpu0.cmd' -WindowStyle Hidden -PassThru
Start-Process -FilePath cmd.exe -ArgumentList '/c','H:\Xiyao_Wang\021_260129VIVID\scripts\run_data_scaling_bce_30k_source_gpu1.cmd' -WindowStyle Hidden -PassThru
```

Monitoring:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|bce_30k|train_cxr|train_vit_baseline' } | Select-Object ProcessId,Name,CommandLine
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Get-Content outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Tail 120
Get-Content outputs\logs\data_scaling_bce_30k_source_gpu1.log -Tail 120
```

### Inputs

Frozen-LM 10k source:

- Config: `configs/data_scaling/frozen_lm_ums_10k.yaml`.
- Train split: `data/splits/chexpert_train_10k.jsonl` (`10000` records, `6548` patients).
- Validation split: `data/splits/chexpert_val_fixed.jsonl` (`1000` records, `661` patients).
- Trainer: `scripts/train_cxr.py`.
- Wrapper: `scripts/run_data_scaling_frozen_lm_ums_10k_source_gpu0.cmd`.

BCE 30k source-control:

- Config: `configs/data_scaling/bce_30k.yaml`.
- Train split: `data/splits/chexpert_train_30k.jsonl` (`29000` records, `19047` patients).
- Validation split: `data/splits/chexpert_val_fixed.jsonl` (`1000` records, `661` patients).
- Trainer: `scripts/train_vit_baseline.py`.
- Wrapper: `scripts/run_data_scaling_bce_30k_source_gpu1.cmd`.

### Expected Outputs

Frozen-LM 10k source:

- Log: `outputs/logs/data_scaling_frozen_lm_ums_10k_source_gpu0.log`.
- Output dir: `outputs/data_scaling/frozen_lm_ums_10k`.
- Expected artifacts: `checkpoints/best.pt`, `checkpoints/final.pt`, periodic checkpoints/metrics depending on trainer behavior.

BCE 30k source-control:

- Log: `outputs/logs/data_scaling_bce_30k_source_gpu1.log`.
- Output dir: `outputs/data_scaling/bce_30k`.
- Expected artifacts: `best.pt`, `final.pt`, `metrics_final.json`, `metrics_step_*.json`, periodic `step_*.pt`.

### Stop Conditions

- Stop before launch if either config/split is missing, either output directory already has a completed final checkpoint, or another VIVID training process is active.
- Stop during run on nonzero wrapper exit code, traceback, OOM, stalled log/checkpoint writes, or unexpected GPU contention.
- If only one row fails, keep the other running if it is healthy; record the failed row separately with failure layer and next action.
- Do not run `lp_frozen_lm_ums_10k` until the frozen-LM 10k source row completes successfully.

### Preflight Results

- `data_scaling_config_validation.csv` marks both `frozen_lm_ums_10k` and `bce_30k` as `available`, `ok`.
- Patient overlap is `0` for both rows.
- `outputs/data_scaling/frozen_lm_ums_10k` and `outputs/data_scaling/bce_30k` were absent before launch.
- No active VIVID training process was found after excluding the preflight query itself.
- GPU0 and GPU1 were idle.
- Added wrappers:
  - `scripts/run_data_scaling_frozen_lm_ums_10k_source_gpu0.cmd`
  - `scripts/run_data_scaling_bce_30k_source_gpu1.cmd`

## 2026-06-27 Phase 1 / P1_DATA_SCALING_DUAL_SOURCE_BATCH_10K_FROZEN_30K_BCE runtime gate

### Results

- Both independent source rows launched successfully.
- Frozen-LM 10k source:
  - Wrapper process: `cmd.exe` PID `6824`.
  - Python training process: `python.exe` PID `10424`.
  - Start time from wrapper/log: `2026/06/27 14:04:12.28`.
  - Loaded `Qwen/Qwen2.5-1.5B-Instruct` from local ModelScope cache after checking existing directory.
  - LLM hidden size: `1536`; frozen LLM params: `1,543,714,304`.
  - Trainable params: `89,349,888`; frozen params: `1,543,714,304`.
  - Training reached at least step `7/10000`.
- BCE 30k source-control:
  - Wrapper process: `cmd.exe` PID `21496`.
  - Python training process: `python.exe` PID `23844`.
  - Start time from wrapper/log: `2026/06/27 14:04:12.35`.
  - Loaded `29000` train samples and `1000` validation samples.
  - Training reached at least step `61/10000`.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or wrapper `EXITCODE` failure line was observed at this gate.

### Commands / Inputs / Outputs

```powershell
Start-Process -FilePath cmd.exe -ArgumentList '/c','H:\Xiyao_Wang\021_260129VIVID\scripts\run_data_scaling_frozen_lm_ums_10k_source_gpu0.cmd' -WindowStyle Hidden -PassThru
Start-Process -FilePath cmd.exe -ArgumentList '/c','H:\Xiyao_Wang\021_260129VIVID\scripts\run_data_scaling_bce_30k_source_gpu1.cmd' -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 45
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|bce_30k|train_cxr|train_vit_baseline' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Get-Content outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Tail 140
Get-Content outputs\logs\data_scaling_bce_30k_source_gpu1.log -Tail 140
```

Outputs:

- Frozen log: `outputs/logs/data_scaling_frozen_lm_ums_10k_source_gpu0.log`.
- BCE log: `outputs/logs/data_scaling_bce_30k_source_gpu1.log`.
- Frozen output dir: `outputs/data_scaling/frozen_lm_ums_10k`.
- BCE output dir: `outputs/data_scaling/bce_30k`.

### Runtime / Speed

- GPU0 frozen-LM 10k sample: about `17127 MiB / 24576 MiB`, `99%` utilization, about `332 W`, `69 C`.
- GPU1 BCE 30k sample: about `4015 MiB / 24576 MiB`, about `156 W`, `43 C`.
- Frozen-LM source speed after startup was about `2.9 s/it`, consistent with prior frozen-LM source rows and implying an hours-long run.
- BCE 30k source speed at startup was about `1.6-1.9 it/s`, slower than single-GPU BCE rows but acceptable because it is overlapped with the much longer frozen-LM source row.

### Failure Reason / Boundary

- No runtime failure at this gate.
- The ModelScope/HF messages show cache reuse and unauthenticated-HF warning, not a blocking download failure.
- The BCE split is named `30k` but currently contains `29000` records; report the actual count in tables.
- Do not treat this runtime gate as completed evidence; no metrics have been integrated yet.

### Next Step

- Continue monitoring:
  - BCE 30k should reach step `500` first.
  - Frozen-LM 10k first validation at step `500` will take much longer.
- If BCE 30k remains much slower but healthy, keep it running because overlap still saves wall-clock time relative to waiting for frozen-LM.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_DUAL_SOURCE_BATCH_10K_FROZEN_30K_BCE bce-step500 gate

### Results

- Both source rows remained active and stable.
- Frozen-LM 10k source:
  - Same wrapper/Python processes remained active.
  - Observed progress was around step `151/10000`.
  - No validation metric yet because the first validation gate is step `500`.
- BCE 30k source-control:
  - Same wrapper/Python processes remained active.
  - Step `500`: `val_loss=0.3477`.
  - `outputs/data_scaling/bce_30k/metrics_step_500.json` and `best.pt` were written.
  - Observed progress after the gate was around step `761/10000`.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or wrapper `EXITCODE` failure line was observed at this gate.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 360
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|bce_30k|train_cxr|train_vit_baseline' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 80
Get-Content outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Tail 80
Select-String -Path outputs\logs\data_scaling_bce_30k_source_gpu1.log -Pattern 'Step [0-9]+: val_loss|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 120
Get-ChildItem outputs\data_scaling\bce_30k
Get-Content outputs\logs\data_scaling_bce_30k_source_gpu1.log -Tail 100
```

Outputs:

- BCE first metric artifact: `outputs/data_scaling/bce_30k/metrics_step_500.json`.
- BCE first checkpoint artifact: `outputs/data_scaling/bce_30k/best.pt`.
- Frozen-LM log continues to update: `outputs/logs/data_scaling_frozen_lm_ums_10k_source_gpu0.log`.

### Runtime / Speed

- GPU0 frozen-LM 10k sample: about `23169 MiB / 24576 MiB`, `89%` utilization, about `333 W`, `76 C`.
- GPU1 BCE 30k sample: about `4015 MiB / 24576 MiB`, about `160 W`, `49 C`.
- Frozen-LM 10k stayed around `2.82-2.84 s/it`, consistent with a long frozen-LM source row.
- BCE 30k stayed around `1.5-1.8 it/s` while sharing the host with frozen-LM 10k. This is slower than single-GPU BCE rows but still expected to finish much earlier than frozen-LM 10k.

### Failure Reason / Boundary

- No runtime failure at this gate.
- GPU0 VRAM is high but still below capacity; continue monitoring for OOM.
- Do not integrate BCE 30k final metrics until `final.pt`, `metrics_final.json`, and wrapper `EXITCODE 0` exist.
- Do not launch frozen-LM 10k LP until frozen-LM source completes.

### Next Step

- Continue monitoring both rows.
- BCE 30k should reach later validation/checkpoint gates first; frozen-LM 10k first validation at step `500` is expected roughly tens of minutes after launch.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_DUAL_SOURCE_BATCH_10K_FROZEN_30K_BCE frozen-step500-bce-midrun gate

### Results

- Both source rows remained active and stable.
- Frozen-LM 10k source:
  - Step `500`: `val_loss=0.1890`.
  - `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/best.pt` was saved.
  - Observed progress after validation was around step `573/10000`.
- BCE 30k source-control:
  - Step `1000`: `val_loss=0.3060`.
  - Step `1500`: `val_loss=0.3234`.
  - Step `2000`: `val_loss=0.2967`.
  - Step `2500`: `val_loss=0.3040`.
  - Step `3000`: `val_loss=0.3036`.
  - Step `3500`: `val_loss=0.2971`.
  - Step `4000`: `val_loss=0.2908`.
  - Step `4500`: `val_loss=0.2860`.
  - Step `5000`: `val_loss=0.2837`.
  - Step `5500`: `val_loss=0.2807`; `best.pt` updated.
  - Step `6000`: `val_loss=0.2807`.
  - Step `6500`: `val_loss=0.2845`.
  - Observed progress after the gate was around step `6723/10000`.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or wrapper `EXITCODE` failure line was observed at this gate.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 1200
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|bce_30k|train_cxr|train_vit_baseline' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 120
Get-ChildItem outputs\data_scaling\frozen_lm_ums_10k -Recurse -Depth 2
Get-Content outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Tail 80
Select-String -Path outputs\logs\data_scaling_bce_30k_source_gpu1.log -Pattern 'Step [0-9]+: val_loss|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 160
Get-ChildItem outputs\data_scaling\bce_30k
Get-Content outputs\logs\data_scaling_bce_30k_source_gpu1.log -Tail 100
```

Outputs:

- Frozen-LM first checkpoint artifact: `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/best.pt`.
- BCE metrics through `metrics_step_6500.json` exist.
- BCE periodic checkpoints through `step_6000.pt` exist.

### Runtime / Speed

- GPU0 frozen-LM 10k sample: about `24299 MiB / 24576 MiB`, `90%` utilization, about `336 W`, `78 C`.
- GPU1 BCE 30k sample: about `4015 MiB / 24576 MiB`, `45%` utilization, about `238 W`, `65 C`.
- Frozen-LM 10k is very close to the 24 GB VRAM ceiling but remains healthy with no OOM.
- Frozen-LM speed returned to about `2.84-2.86 s/it` after validation smoothing.
- BCE 30k recovered to about `5.3 it/s` after early warmup, so dual-GPU execution is currently beneficial rather than persistently slowed.

### Failure Reason / Boundary

- No runtime failure at this gate.
- Continue watching GPU0 for OOM because VRAM headroom is small.
- Frozen-LM step500 source loss is not an LP metric; do not compare directly to LP macro-AUC/F1.
- Do not integrate BCE 30k final metrics until `final.pt`, `metrics_final.json`, and wrapper `EXITCODE 0` exist.

### Next Step

- Continue monitoring.
- BCE 30k should complete soon and then be integrated independently.
- Frozen-LM 10k should continue as the long-running row; do not launch its LP until the source row completes.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_30K_BCE_SOURCE_RUN completion and integration gate

### Results

- The 30k BCE source-control row completed successfully on GPU1 while frozen-LM 10k continued on GPU0.
- Wrapper exit line: `EXITCODE 0`.
- End time: `2026/06/27 14:43:27.97`.
- Training reached `10000/10000` in about `38:59`, around `4.27 it/s` including validation overhead.
- `final.pt`, `metrics_final.json`, `step_10000.pt`, and `metrics_step_10000.json` were saved.
- `best.pt` was updated at step `7000`, consistent with best validation-loss policy.
- GPU1 was idle after completion; frozen-LM 10k continued on GPU0 around step `905/10000`.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 900
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|bce_30k|train_cxr|train_vit_baseline' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_bce_30k_source_gpu1.log -Pattern 'Step [0-9]+: val_loss|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 200
Get-ChildItem outputs\data_scaling\bce_30k
Get-Content outputs\logs\data_scaling_bce_30k_source_gpu1.log -Tail 120
python scripts\summarize_data_scaling_30k_progress.py
python scripts\summarize_revision_execution_status.py
python scripts\summarize_revision_completion_gap_audit.py
python scripts\summarize_current_priority_completion_audit.py
python -m py_compile scripts\summarize_data_scaling_30k_progress.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_current_priority_completion_audit.py
python scripts\validate_data_scaling_configs.py
```

Outputs:

- Log: `outputs/logs/data_scaling_bce_30k_source_gpu1.log`.
- Run dir: `outputs/data_scaling/bce_30k`.
- Summary outputs:
  - `outputs/final_tables/data_scaling_30k_progress.csv`
  - `outputs/final_tables/data_scaling_30k_progress.md`
  - `outputs/final_tables/data_scaling_30k_trajectory.csv`
  - refreshed `outputs/final_tables/revision_execution_status.csv/md`
  - refreshed `outputs/final_tables/revision_completion_gap_audit.csv/md`
  - refreshed `outputs/final_tables/current_priority_completion_audit.csv/md`
- `revision_execution_status.csv` now has `53` rows and includes `P1_DATA_SCALING_30K_BCE_SOURCE_RUN`.
- `data_scaling_config_validation.csv` now marks `bce_30k` as `completed_run_exists`, `fail`, `would overwrite completed run`; this is the intended completed-output overwrite guard, not a failed run.

### Metrics

| policy | step | val_loss | macro_auc | macro_f1 | micro_f1 |
|---|---:|---:|---:|---:|---:|
| final checkpoint | final | 0.284332 | 0.789930 | 0.905414 | 0.879152 |
| best validation loss | 7000 | 0.272276 | 0.785833 | 0.909565 | 0.882780 |
| best macro-AUC | 7500 | 0.278824 | 0.799368 | 0.896900 | 0.879152 |
| best macro-F1 | 8500 | 0.282430 | 0.791749 | 0.910495 | 0.882501 |

Validation-loss trajectory boundary:

- step `500`: `0.3477`
- step `1000`: `0.3060`
- step `2000`: `0.2967`
- step `4000`: `0.2908`
- step `5500`: `0.2807`
- step `7000`: `0.2723`
- step `8500`: `0.2824`
- step `10000`: `0.2843`

### Failure Reason / Boundary

- No runtime failure, OOM, traceback, or nonzero exit occurred.
- This row is a 30k source-control row only; it does not complete matched 30k no-LM/frozen-LM evidence.
- The named 30k split contains `29000` train records; report the actual count.
- Keep source-control metrics separate from no-LM/frozen-LM source and LP metrics.

### Next Step

- Keep frozen-LM 10k source running on GPU0.
- Since GPU1 is idle and dependencies do not conflict, the next launchable row is `P1_DATA_SCALING_30K_NO_LM_UMS_SOURCE_RUN` on GPU1.
- Do not launch `lp_no_lm_ums_30k` until the no-LM 30k source row completes and `best.pt` exists.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_30K_NO_LM_UMS_SOURCE_RUN execution-before

### Plan

- Keep the long-running frozen-LM 10k source row on GPU0.
- Launch one independent source row on idle GPU1: formal 30k no-LM UMS source training.
- Do not launch `lp_no_lm_ums_30k` until this source row completes with `EXITCODE 0` and `best.pt` exists.
- Keep source metrics separate from LP metrics and from BCE source-control metrics.
- Treat the named 30k split as `29000` train records in all reporting.

### Commands

Preflight and launch:

```powershell
Import-Csv outputs\final_tables\data_scaling_config_validation.csv | Where-Object { $_.method -eq 'no_lm_ums_30k' } | Format-List
Get-Content configs\data_scaling\no_lm_ums_30k.yaml
if (Test-Path outputs\data_scaling\no_lm_ums_30k) { Get-ChildItem outputs\data_scaling\no_lm_ums_30k -Recurse -Depth 2 } else { 'outputs/data_scaling/no_lm_ums_30k missing' }
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|no_lm_ums_30k|bce_30k|train_cxr|train_vit_baseline|train_ums_classifier' } | Select-Object ProcessId,Name,CommandLine
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Start-Process -FilePath cmd.exe -ArgumentList '/c','H:\Xiyao_Wang\021_260129VIVID\scripts\run_data_scaling_no_lm_ums_30k_source_gpu1.cmd' -WindowStyle Hidden -PassThru
```

Monitoring:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|no_lm_ums_30k|train_cxr|train_ums_classifier' } | Select-Object ProcessId,Name,CommandLine
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Get-Content outputs\logs\data_scaling_no_lm_ums_30k_source_gpu1.log -Tail 120
Get-Content outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Tail 80
```

### Inputs

- Config: `configs/data_scaling/no_lm_ums_30k.yaml`.
- Train split: `data/splits/chexpert_train_30k.jsonl` (`29000` records, `19047` patients).
- Validation split: `data/splits/chexpert_val_fixed.jsonl` (`1000` records, `661` patients).
- Trainer: `scripts/train_ums_classifier.py`.
- Wrapper: `scripts/run_data_scaling_no_lm_ums_30k_source_gpu1.cmd`.

### Expected Outputs

- Log: `outputs/logs/data_scaling_no_lm_ums_30k_source_gpu1.log`.
- Output dir: `outputs/data_scaling/no_lm_ums_30k`.
- Expected artifacts: `best.pt`, `final.pt`, `metrics_final.json`, `metrics_step_*.json`, periodic `step_*.pt`.
- Source checkpoint for future LP: `outputs/data_scaling/no_lm_ums_30k/best.pt`.

### Stop Conditions

- Stop before launch if the config/splits are missing, if `outputs/data_scaling/no_lm_ums_30k/final.pt` already exists, or if GPU1 is not idle.
- Stop during run on nonzero wrapper exit code, traceback, OOM, stalled log/checkpoint writes, or unexpected GPU contention.
- If the GPU1 no-LM row fails, keep the healthy GPU0 frozen-LM 10k row running and document the GPU1 failure separately.
- Do not run `lp_no_lm_ums_30k` until this source row completes successfully.

### Preflight Results

- `data_scaling_config_validation.csv` marks `no_lm_ums_30k` as `available`, `ok`.
- Patient overlap is `0`.
- `outputs/data_scaling/no_lm_ums_30k` was absent before launch.
- GPU0 is occupied by healthy frozen-LM 10k source training.
- GPU1 was idle before launch.
- Added wrapper `scripts/run_data_scaling_no_lm_ums_30k_source_gpu1.cmd`.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_30K_NO_LM_UMS_SOURCE_RUN runtime gate

### Results

- The 30k no-LM UMS source row launched successfully on GPU1 while frozen-LM 10k continued on GPU0.
- no-LM 30k source:
  - Wrapper process: `cmd.exe` PID `23520`.
  - Python training process: `python.exe` PID `21436`.
  - Training reached at least step `126/10000`.
  - Config snapshot and progress files were written.
- Frozen-LM 10k source:
  - Same wrapper/Python processes remained active on GPU0.
  - Step `1000`: `val_loss=0.0676`.
  - `checkpoints/best.pt` and `checkpoints/step_1000.pt` were saved.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or wrapper `EXITCODE` failure line was observed at this gate.

### Commands / Inputs / Outputs

```powershell
Start-Process -FilePath cmd.exe -ArgumentList '/c','H:\Xiyao_Wang\021_260129VIVID\scripts\run_data_scaling_no_lm_ums_30k_source_gpu1.cmd' -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 35
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|no_lm_ums_30k|train_cxr|train_ums_classifier' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Get-Content outputs\logs\data_scaling_no_lm_ums_30k_source_gpu1.log -Tail 140
Get-Content outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Tail 50
Get-ChildItem outputs\data_scaling\no_lm_ums_30k
```

Outputs:

- no-LM 30k log: `outputs/logs/data_scaling_no_lm_ums_30k_source_gpu1.log`.
- no-LM 30k run dir: `outputs/data_scaling/no_lm_ums_30k`.
- no-LM 30k initial artifacts:
  - `outputs/data_scaling/no_lm_ums_30k/config_snapshot.json`
  - `outputs/data_scaling/no_lm_ums_30k/progress.json`
- Frozen-LM 10k artifacts now include:
  - `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/best.pt`
  - `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/step_1000.pt`

### Runtime / Speed

- GPU0 frozen-LM 10k sample: about `24299 MiB / 24576 MiB`, `87%` utilization, about `331 W`, `72 C`.
- GPU1 no-LM 30k sample: about `4017 MiB / 24576 MiB`, `47%` utilization, about `216 W`, `48 C`.
- no-LM 30k source speed after warmup was about `4.8-5.1 it/s`, normal for this host.
- Frozen-LM 10k speed remains about `2.8 s/it` outside validation smoothing.

### Failure Reason / Boundary

- No runtime failure at this gate.
- The low GPU1 memory usage is expected for no-LM source training.
- GPU0 remains near the memory ceiling but has not OOMed.
- Do not integrate no-LM 30k metrics until `final.pt`, `metrics_final.json`, and wrapper `EXITCODE 0` exist.
- Do not launch `lp_no_lm_ums_30k` until no-LM 30k source completes.

### Next Step

- Continue monitoring both rows.
- no-LM 30k should reach step `500` quickly.
- Frozen-LM 10k remains the long-running dependency for future 10k frozen-LM LP.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_30K_NO_LM_UMS_SOURCE_RUN mid-run gate

### Results

- no-LM 30k source remained active and stable through at least step `4500`.
- Step `500`: `val_loss=0.7400`, `macro_auc=0.6807877102811859`.
- Step `1000`: `val_loss=0.7307`, `macro_auc=0.690303183577884`.
- Step `1500`: `val_loss=0.7165`, `macro_auc=0.7201399984833875`.
- Step `2000`: `val_loss=0.6991`, `macro_auc=0.7228074944335033`.
- Step `2500`: `val_loss=0.6960`, `macro_auc=0.7173754966225028`.
- Step `3000`: `val_loss=0.7038`, `macro_auc=0.7068680110807031`.
- Step `3500`: `val_loss=0.6910`, `macro_auc=0.7333848899923386`.
- Step `4000`: `val_loss=0.6950`, `macro_auc=0.7275996898036059`.
- Step `4500`: `val_loss=0.6846`, `macro_auc=0.7194066298107402`; `best.pt` updated.
- Observed no-LM 30k progress after this gate was around `4707/10000`.
- Frozen-LM 10k remained active on GPU0, with observed progress around `1335/10000`.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or wrapper `EXITCODE` failure line was observed for either row at this gate.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 900
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|no_lm_ums_30k|train_cxr|train_ums_classifier' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_no_lm_ums_30k_source_gpu1.log -Pattern 'Step [0-9]+: val_loss|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 160
Get-ChildItem outputs\data_scaling\no_lm_ums_30k
Get-Content outputs\logs\data_scaling_no_lm_ums_30k_source_gpu1.log -Tail 100
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 120
Get-Content outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Tail 60
```

Outputs:

- no-LM 30k metrics through `metrics_step_4500.json` exist.
- no-LM 30k periodic checkpoints through `step_4000.pt` exist.
- no-LM 30k `best.pt` was updated at step `4500`.

### Runtime / Speed

- GPU0 frozen-LM 10k sample: about `24299 MiB / 24576 MiB`, `99%` utilization, about `335 W`, `78 C`.
- GPU1 no-LM 30k sample: about `4313 MiB / 24576 MiB`, `29%` utilization, about `233 W`, `62 C`.
- no-LM 30k speed remained around `5.2 it/s`, normal for this host.
- Frozen-LM 10k remained around `2.84-2.86 s/it`.

### Failure Reason / Boundary

- No runtime failure at this gate.
- This remains an in-progress source row; do not integrate final metrics until `final.pt`, `metrics_final.json`, and wrapper `EXITCODE 0` exist.
- Current no-LM 30k best validation-loss and best macro-AUC policies differ; preserve metric policies separately at completion.

### Next Step

- Continue monitoring no-LM 30k toward completion.
- Keep frozen-LM 10k running on GPU0; do not launch frozen-LM LP yet.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_30K_NO_LM_UMS_SOURCE_RUN completion and integration gate

### Results

- Formal no-LM UMS 30k source row completed with wrapper `EXITCODE 0`.
- The named 30k split is `29000` train records, not a full 30000; keep this explicit in all reporting.
- Final checkpoint policy:
  - `final_checkpoint`: step `final`, `val_loss=0.688614`, `macro_auc=0.714086`, `macro_f1=0.418470`, `micro_f1=0.613670`.
  - `best_val_loss`: step `7500`, `val_loss=0.674669`, `macro_auc=0.735255`, `macro_f1=0.396921`, `micro_f1=0.603109`.
  - `best_macro_auc`: step `5000`, `val_loss=0.681660`, `macro_auc=0.743815`, `macro_f1=0.338355`, `micro_f1=0.570842`.
  - `best_macro_f1`: step `10000`, `val_loss=0.688614`, `macro_auc=0.714086`, `macro_f1=0.418470`, `micro_f1=0.613670`.
- `outputs/data_scaling/no_lm_ums_30k/best.pt` exists and unlocks `lp_no_lm_ums_30k`.
- `revision_execution_status.csv` now has `54` rows and includes `P1_DATA_SCALING_30K_NO_LM_UMS_SOURCE_RUN`.
- `revision_completion_gap_audit.csv` now records P1 data scaling as complete for 1k matched LP, 3k matched LP, 10k BCE/no-LM source+LP, 30k BCE source, and 30k no-LM source; remaining rows are 10k frozen-LM source+LP, 30k no-LM LP, and 30k frozen-LM rows.

### Commands / Inputs / Outputs

```powershell
python scripts\summarize_data_scaling_30k_progress.py
python scripts\summarize_revision_execution_status.py
python scripts\summarize_revision_completion_gap_audit.py
python scripts\summarize_current_priority_completion_audit.py
python -m py_compile scripts\summarize_data_scaling_30k_progress.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_current_priority_completion_audit.py
python scripts\validate_data_scaling_configs.py
Import-Csv outputs\final_tables\data_scaling_30k_progress.csv | Format-Table -AutoSize
Import-Csv outputs\final_tables\revision_execution_status.csv | Where-Object { $_.task_id -match '30K_NO_LM|30K_BCE|10K_FROZEN|30K_LP_NO_LM' } | Format-List
Import-Csv outputs\final_tables\revision_completion_gap_audit.csv | Where-Object { $_.requirement -eq 'P1_DATA_SCALING' } | Format-List
Import-Csv outputs\final_tables\data_scaling_config_validation.csv | Where-Object { $_.method -match '30k|frozen_lm_ums_10k' } | Select-Object method,size,output_status,checkpoint_dependency,status,notes | Format-Table -AutoSize
```

Inputs:

- Source wrapper: `scripts/run_data_scaling_no_lm_ums_30k_source_gpu1.cmd`.
- Source log: `outputs/logs/data_scaling_no_lm_ums_30k_source_gpu1.log`.
- Source run dir: `outputs/data_scaling/no_lm_ums_30k`.
- Config: `configs/data_scaling/no_lm_ums_30k.yaml`.

Outputs:

- `outputs/final_tables/data_scaling_30k_progress.csv`
- `outputs/final_tables/data_scaling_30k_progress.md`
- `outputs/final_tables/revision_execution_status.csv`
- `outputs/final_tables/revision_completion_gap_audit.csv`
- `outputs/final_tables/current_priority_completion_audit.csv`
- `outputs/final_tables/data_scaling_config_validation.csv`

### Runtime / Speed

- no-LM 30k source completed faster than the frozen-LM row and was normal for this host.
- Frozen-LM 10k source is still active on GPU0; current observed speed remains about `2.8-2.9 s/it`.
- Current GPU state during this integration gate:
  - GPU0: frozen-LM 10k active, about `24299 MiB / 24576 MiB`, high utilization, no OOM observed.
  - GPU1: idle after no-LM 30k completion.

### Failure Reason / Boundary

- No experiment failure for no-LM 30k source.
- `python scripts\validate_data_scaling_configs.py` returned nonzero because completed rows such as `bce_30k` and `no_lm_ums_30k` are protected by `completed_run_exists` overwrite guards. Treat that as an expected safety failure, not a model/runtime failure.
- Source metrics must not be reported as LP metrics.
- The no-LM source row should only be used as the checkpoint dependency for `lp_no_lm_ums_30k`.

### Next Step

- Launch `P1_DATA_SCALING_30K_LP_NO_LM_UMS_RUN` on idle GPU1.
- Keep frozen-LM 10k source running on GPU0.
- Do not launch `lp_frozen_lm_ums_10k` until frozen-LM 10k source wrapper exits with code `0`.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_30K_LP_NO_LM_UMS_RUN execution-before

### Plan

- Use GPU1 for a light LP row while GPU0 continues the long frozen-LM 10k source row.
- Train `lp_no_lm_ums_30k` from the completed source checkpoint `outputs/data_scaling/no_lm_ums_30k/best.pt`.
- Keep the output isolated in `outputs/data_scaling/lp_no_lm_ums_30k`.
- Do not start any frozen-LM LP row yet.

### Commands

Preflight and launch:

```powershell
Import-Csv outputs\final_tables\data_scaling_config_validation.csv | Where-Object { $_.method -eq 'lp_no_lm_ums_30k' } | Format-List
Get-Content configs\data_scaling\lp_no_lm_ums_30k.yaml
if (Test-Path outputs\data_scaling\lp_no_lm_ums_30k) { Get-ChildItem outputs\data_scaling\lp_no_lm_ums_30k -Recurse -Depth 2 } else { 'outputs/data_scaling/lp_no_lm_ums_30k missing' }
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|lp_no_lm_ums_30k|train_cxr|train_vit_baseline|train_ums_classifier' } | Select-Object ProcessId,Name,CommandLine
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Start-Process -FilePath cmd.exe -ArgumentList '/c','H:\Xiyao_Wang\021_260129VIVID\scripts\run_data_scaling_lp_no_lm_ums_30k_gpu1.cmd' -WindowStyle Hidden -PassThru
```

Monitoring:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|lp_no_lm_ums_30k|train_cxr|train_vit_baseline|train_ums_classifier' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Get-Content outputs\logs\data_scaling_lp_no_lm_ums_30k_gpu1.log -Tail 120
Get-Content outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Tail 80
```

### Inputs

- Config: `configs/data_scaling/lp_no_lm_ums_30k.yaml`.
- Source checkpoint: `outputs/data_scaling/no_lm_ums_30k/best.pt`.
- Train split: `data/splits/chexpert_train_30k.jsonl` (`29000` records, `19047` patients).
- Validation split: `data/splits/chexpert_val_fixed.jsonl` (`1000` records, `661` patients).
- Wrapper: `scripts/run_data_scaling_lp_no_lm_ums_30k_gpu1.cmd`.

### Expected Outputs

- Log: `outputs/logs/data_scaling_lp_no_lm_ums_30k_gpu1.log`.
- Output dir: `outputs/data_scaling/lp_no_lm_ums_30k`.
- Expected artifacts: `best.pt`, `final.pt`, `metrics_final.json`, `metrics_step_*.json`, periodic `step_*.pt`.

### Stop Conditions

- Stop before launch if `outputs/data_scaling/no_lm_ums_30k/best.pt` is missing, if `outputs/data_scaling/lp_no_lm_ums_30k/final.pt` already exists, or if GPU1 is not idle.
- Stop during run on nonzero wrapper exit code, traceback, OOM, stalled log/checkpoint writes, or unexpected contention with frozen-LM 10k.
- If LP fails, keep the frozen-LM 10k source row running and document the LP failure separately.

### Preflight Results

- `data_scaling_config_validation.csv` marks `lp_no_lm_ums_30k` as `available`, `ok`.
- Source dependency `outputs/data_scaling/no_lm_ums_30k/best.pt` exists.
- `outputs/data_scaling/lp_no_lm_ums_30k` was absent before launch.
- GPU0 was occupied by frozen-LM 10k source.
- GPU1 was idle before launch.
- Added wrapper `scripts/run_data_scaling_lp_no_lm_ums_30k_gpu1.cmd`.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_30K_LP_NO_LM_UMS_RUN runtime gate

### Results

- LP no-LM 30k launched successfully on GPU1 while frozen-LM 10k continued on GPU0.
- LP no-LM 30k reached at least step `400/3000`.
- Step `400`: `val_loss=0.2718`.
- `outputs/data_scaling/lp_no_lm_ums_30k/best.pt` was written.
- `metrics_step_200.json` and `metrics_step_400.json` were written.
- Frozen-LM 10k source remained active on GPU0.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or wrapper `EXITCODE` failure line was observed at this gate.

### Commands / Inputs / Outputs

```powershell
Start-Process -FilePath cmd.exe -ArgumentList '/c','H:\Xiyao_Wang\021_260129VIVID\scripts\run_data_scaling_lp_no_lm_ums_30k_gpu1.cmd' -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 45
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|lp_no_lm_ums_30k|train_cxr|train_vit_baseline|train_ums_classifier' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Get-Content outputs\logs\data_scaling_lp_no_lm_ums_30k_gpu1.log -Tail 160
Get-ChildItem outputs\data_scaling\lp_no_lm_ums_30k
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 60
```

Outputs:

- LP log: `outputs/logs/data_scaling_lp_no_lm_ums_30k_gpu1.log`.
- LP run dir: `outputs/data_scaling/lp_no_lm_ums_30k`.
- Initial LP artifacts:
  - `outputs/data_scaling/lp_no_lm_ums_30k/best.pt`
  - `outputs/data_scaling/lp_no_lm_ums_30k/metrics_step_200.json`
  - `outputs/data_scaling/lp_no_lm_ums_30k/metrics_step_400.json`

### Runtime / Speed

- GPU0 frozen-LM 10k sample: about `24299 MiB / 24576 MiB`, `79%` utilization, about `331 W`, `72 C`.
- GPU1 LP no-LM 30k sample: about `1893 MiB / 24576 MiB`, `18%` utilization, about `130 W`, `44 C`.
- LP speed after warmup was around `10-12 it/s`, normal and light enough to coexist with GPU0 frozen-LM training.

### Failure Reason / Boundary

- No runtime failure at this gate.
- This is an in-progress LP row; do not integrate final metrics until `final.pt`, `metrics_final.json`, and wrapper `EXITCODE 0` exist.
- Keep this LP result separate from the source no-LM 30k row.

### Next Step

- Continue monitoring LP no-LM 30k to completion.
- Keep frozen-LM 10k running on GPU0.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_30K_LP_NO_LM_UMS_RUN completion and integration gate

### Results

- Formal no-LM UMS 30k LP row completed with wrapper `EXITCODE 0`.
- The named 30k split is `29000` train records; keep this explicit.
- Final checkpoint policy:
  - `final_checkpoint`: step `final`, `val_loss=0.263937`, `macro_auc=0.815050`, `macro_f1=0.911138`, `micro_f1=0.888083`.
  - `best_val_loss`: step `2000`, `val_loss=0.262425`, `macro_auc=0.810583`, `macro_f1=0.911280`, `micro_f1=0.887245`.
  - `best_macro_auc`: step `400`, `val_loss=0.271760`, `macro_auc=0.849124`, `macro_f1=0.907979`, `micro_f1=0.888083`.
  - `best_macro_f1`: step `200`, `val_loss=0.270297`, `macro_auc=0.841976`, `macro_f1=0.912467`, `micro_f1=0.883896`.
- `revision_execution_status.csv` now has `55` rows and includes `P1_DATA_SCALING_30K_NO_LM_UMS_LP_RUN`.
- `revision_completion_gap_audit.csv` now records P1 data scaling as complete for 1k matched LP, 3k matched LP, 10k BCE/no-LM source+LP, 30k BCE source, and 30k no-LM source+LP. Remaining P1 rows are 10k frozen-LM source+LP and 30k frozen-LM rows.

### Commands / Inputs / Outputs

```powershell
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|lp_no_lm_ums_30k|train_cxr|train_vit_baseline|train_ums_classifier' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_lp_no_lm_ums_30k_gpu1.log -Pattern 'Step [0-9]+:|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 120
Get-ChildItem outputs\data_scaling\lp_no_lm_ums_30k | Sort-Object Name
python scripts\summarize_data_scaling_30k_progress.py
python scripts\summarize_revision_execution_status.py
python scripts\summarize_revision_completion_gap_audit.py
python scripts\summarize_current_priority_completion_audit.py
python -m py_compile scripts\summarize_data_scaling_30k_progress.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_current_priority_completion_audit.py
Import-Csv outputs\final_tables\data_scaling_30k_progress.csv | Format-Table -AutoSize
Import-Csv outputs\final_tables\revision_execution_status.csv | Where-Object { $_.task_id -match '30K_NO_LM' } | Format-List
Import-Csv outputs\final_tables\revision_completion_gap_audit.csv | Where-Object { $_.requirement -eq 'P1_DATA_SCALING' } | Format-List
```

Inputs:

- Wrapper: `scripts/run_data_scaling_lp_no_lm_ums_30k_gpu1.cmd`.
- Log: `outputs/logs/data_scaling_lp_no_lm_ums_30k_gpu1.log`.
- Config: `configs/data_scaling/lp_no_lm_ums_30k.yaml`.
- Source checkpoint: `outputs/data_scaling/no_lm_ums_30k/best.pt`.

Outputs:

- `outputs/data_scaling/lp_no_lm_ums_30k/final.pt`
- `outputs/data_scaling/lp_no_lm_ums_30k/best.pt`
- `outputs/data_scaling/lp_no_lm_ums_30k/metrics_final.json`
- `outputs/final_tables/data_scaling_30k_progress.csv`
- `outputs/final_tables/data_scaling_30k_progress.md`
- `outputs/final_tables/revision_execution_status.csv`
- `outputs/final_tables/revision_completion_gap_audit.csv`
- `outputs/final_tables/current_priority_completion_audit.csv`

### Runtime / Speed

- GPU1 returned to idle after completion.
- LP runtime was normal for this host and used about `1893 MiB / 24576 MiB` during the run.
- Frozen-LM 10k source continued on GPU0 during the LP run, still using about `24299 MiB / 24576 MiB` with high utilization and no OOM signal.

### Failure Reason / Boundary

- No LP runtime failure.
- This completes 30k no-LM LP evidence only; it does not complete 30k frozen-LM evidence.
- The best macro-AUC policy (`step 400`) is much earlier than best validation-loss (`step 2000`), so reporting must preserve metric-policy boundaries.

### Next Step

- Check whether GPU1 can safely start independent `P1_DATA_SCALING_30K_FROZEN_LM_UMS_SOURCE_RUN` while GPU0 continues 10k frozen-LM source.
- Do not launch any LP from frozen-LM 10k until its source wrapper exits with code `0`.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_30K_FROZEN_LM_UMS_SOURCE_RUN execution-before

### Plan

- Keep the existing `frozen_lm_ums_10k` source row running on GPU0.
- Launch independent `frozen_lm_ums_30k` source training on GPU1 because it does not depend on the 10k source result.
- Use the unchanged formal config; do not reduce batch size unless the early runtime gate shows OOM or severe contention.
- Do not launch `lp_frozen_lm_ums_30k` until this source row completes with `EXITCODE 0` and `best.pt` exists.
- Do not launch `lp_frozen_lm_ums_10k` until the 10k source wrapper exits with `EXITCODE 0`.

### Commands

Preflight and launch:

```powershell
Import-Csv outputs\final_tables\data_scaling_config_validation.csv | Where-Object { $_.method -eq 'frozen_lm_ums_30k' } | Format-List
Get-Content configs\data_scaling\frozen_lm_ums_30k.yaml
if (Test-Path outputs\data_scaling\frozen_lm_ums_30k) { Get-ChildItem outputs\data_scaling\frozen_lm_ums_30k -Recurse -Depth 2 } else { 'outputs/data_scaling/frozen_lm_ums_30k missing' }
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|frozen_lm_ums_30k|train_cxr|train_vit_baseline|train_ums_classifier' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Get-CimInstance Win32_OperatingSystem | Select-Object @{Name='TotalGB';Expression={[math]::Round($_.TotalVisibleMemorySize/1MB,1)}},@{Name='FreeGB';Expression={[math]::Round($_.FreePhysicalMemory/1MB,1)}} | Format-List
Start-Process -FilePath cmd.exe -ArgumentList '/c','H:\Xiyao_Wang\021_260129VIVID\scripts\run_data_scaling_frozen_lm_ums_30k_source_gpu1.cmd' -WindowStyle Hidden -PassThru
```

Monitoring:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|frozen_lm_ums_30k|train_cxr' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Get-Content outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Tail 160
Get-Content outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Tail 80
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 120
```

### Inputs

- Config: `configs/data_scaling/frozen_lm_ums_30k.yaml`.
- Train split: `data/splits/chexpert_train_30k.jsonl` (`29000` records, `19047` patients).
- Validation split: `data/splits/chexpert_val_fixed.jsonl` (`1000` records, `661` patients).
- Frozen LM: `Qwen/Qwen2.5-1.5B-Instruct`.
- Wrapper: `scripts/run_data_scaling_frozen_lm_ums_30k_source_gpu1.cmd`.

### Expected Outputs

- Log: `outputs/logs/data_scaling_frozen_lm_ums_30k_source_gpu1.log`.
- Output dir: `outputs/data_scaling/frozen_lm_ums_30k`.
- Expected artifacts: `checkpoints/best.pt`, `checkpoints/final.pt`, `metrics_final.json` if exported by trainer, `progress.json`, `config_snapshot.json`, periodic `checkpoints/step_*.pt`.
- Source checkpoint for future LP: `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/best.pt`.

### Stop Conditions

- Stop before launch if `outputs/data_scaling/frozen_lm_ums_30k/final.pt` or `checkpoints/final.pt` already exists, if GPU1 is not idle, or if system memory is unexpectedly low.
- Stop during run on nonzero wrapper exit code, traceback, OOM, stalled log/checkpoint writes, or if running the second frozen-LM row causes severe contention or thermal throttling.
- If the 30k frozen-LM source fails, keep the 10k frozen-LM source row running and document the GPU1 failure separately.
- Do not run `lp_frozen_lm_ums_30k` until the source row completes successfully.

### Preflight Results

- `data_scaling_config_validation.csv` marks `frozen_lm_ums_30k` as `available`, `ok`.
- Patient overlap is `0`.
- `outputs/data_scaling/frozen_lm_ums_30k` was absent before launch.
- GPU0 is occupied by frozen-LM 10k source at about `24299 MiB / 24576 MiB`.
- GPU1 was idle before launch.
- System memory at preflight: `63.8 GB` total, `41.4 GB` free.
- Added wrapper `scripts/run_data_scaling_frozen_lm_ums_30k_source_gpu1.cmd`.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_30K_FROZEN_LM_UMS_SOURCE_RUN runtime gate

### Results

- Frozen-LM 30k source launched successfully on GPU1 while frozen-LM 10k source continued on GPU0.
- Frozen-LM 30k source reached at least step `26/10000`.
- Early speed stabilized around `2.8-2.9 s/it`.
- Frozen-LM 10k source remained active and reached step `2000`.
- Frozen-LM 10k step `2000`: `val_loss=0.0442`; `best.pt` and `step_2000.pt` were saved.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or wrapper `EXITCODE` failure line was observed at this gate.

### Commands / Inputs / Outputs

```powershell
Start-Process -FilePath cmd.exe -ArgumentList '/c','H:\Xiyao_Wang\021_260129VIVID\scripts\run_data_scaling_frozen_lm_ums_30k_source_gpu1.cmd' -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 75
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|frozen_lm_ums_30k|train_cxr' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Get-CimInstance Win32_OperatingSystem | Select-Object @{Name='TotalGB';Expression={[math]::Round($_.TotalVisibleMemorySize/1MB,1)}},@{Name='FreeGB';Expression={[math]::Round($_.FreePhysicalMemory/1MB,1)}} | Format-List
Get-Content outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Tail 180
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 80
```

Outputs:

- 30k frozen-LM source log: `outputs/logs/data_scaling_frozen_lm_ums_30k_source_gpu1.log`.
- 30k frozen-LM source run dir: `outputs/data_scaling/frozen_lm_ums_30k`.
- 10k frozen-LM artifacts now include:
  - `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/best.pt`
  - `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/step_2000.pt`

### Runtime / Speed

- GPU0 frozen-LM 10k sample: about `24299 MiB / 24576 MiB`, `99%` utilization, about `335 W`, `76 C`.
- GPU1 frozen-LM 30k sample: about `24309 MiB / 24576 MiB`, `96%` utilization, about `302 W`, `67 C`.
- System memory after launching both frozen-LM rows: `63.8 GB` total, `37.8 GB` free.
- Speed is normal for this configuration; there is no current evidence that a virus or unrelated automation process is slowing training.

### Failure Reason / Boundary

- No runtime failure at this gate.
- Both GPUs are close to the VRAM ceiling; any OOM, thermal issue, or major speed collapse should stop the newer GPU1 frozen-LM 30k row first and preserve GPU0 10k source.
- This is an in-progress source row; do not integrate metrics until `final.pt`/final checkpoint artifacts and wrapper `EXITCODE 0` exist.

### Next Step

- Continue monitoring both frozen-LM source rows.
- Do not launch either frozen-LM LP row until its corresponding source row exits with code `0`.

## 2026-06-27 Runtime health diagnostic execution-before

### Plan

- Check whether current slowness is explained by training workload versus unrelated GPU/CPU/memory consumers.
- Inspect high CPU and high memory processes.
- Inspect GPU process ownership through `nvidia-smi`.
- Inspect Microsoft Defender status without launching a full scan.
- Avoid intrusive scans or process termination while experiments are running.

### Commands

```powershell
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
nvidia-smi
Get-Process | Sort-Object CPU -Descending | Select-Object -First 15 Id,ProcessName,CPU,WorkingSet64,Path | Format-Table -AutoSize
Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 15 Id,ProcessName,CPU,WorkingSet64,Path | Format-Table -AutoSize
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'train_cxr|train_vit_baseline|python|cmd.exe' } | Select-Object ProcessId,Name,CommandLine | Format-List
Get-MpComputerStatus | Select-Object AMServiceEnabled,AntivirusEnabled,RealTimeProtectionEnabled,AntispywareEnabled,FullScanAge,QuickScanAge,AntivirusSignatureLastUpdated | Format-List
```

### Inputs

- Running Windows process table.
- NVIDIA process table and GPU telemetry.
- Microsoft Defender local status.

### Expected Outputs

- A short classification of whether the current runtime bottleneck is expected training load, unrelated automation, or suspicious/unknown resource use.
- Evidence paths remain the command output and this plan document.

### Stop Conditions

- Stop and preserve logs if a non-training process is consuming major GPU memory/compute.
- Stop and ask before killing any process.
- Do not run full malware scans during active training unless the diagnostics show a concrete reason.

## 2026-06-27 Runtime health diagnostic results

### Results

- `nvidia-smi` showed exactly two GPU compute processes:
  - GPU0: `python.exe` PID `10424`, `frozen_lm_ums_10k`.
  - GPU1: `python.exe` PID `19372`, `frozen_lm_ums_30k`.
- No non-training process was using GPU memory/compute.
- High working-set processes were expected local applications: the two training Python processes, PyCharm, Codex, VS Code, Explorer, ToDesk, Everything, and Microsoft Defender service.
- Microsoft Defender status:
  - `AMServiceEnabled=True`
  - `AntivirusEnabled=True`
  - `RealTimeProtectionEnabled=True`
  - `AntispywareEnabled=True`
  - `QuickScanAge=1`
  - `AntivirusSignatureLastUpdated=2026/6/25 10:52:49`
- Current speed remains consistent with frozen-LM training rather than an unrelated background process.

### Commands / Inputs / Outputs

```powershell
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
nvidia-smi
Get-Process | Sort-Object CPU -Descending | Select-Object -First 15 Id,ProcessName,CPU,@{Name='WorkingSetGB';Expression={[math]::Round($_.WorkingSet64/1GB,2)}},Path | Format-Table -AutoSize
Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 15 Id,ProcessName,CPU,@{Name='WorkingSetGB';Expression={[math]::Round($_.WorkingSet64/1GB,2)}},Path | Format-Table -AutoSize
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'train_cxr|train_vit_baseline|python|cmd.exe' } | Select-Object ProcessId,Name,CommandLine | Format-List
Get-MpComputerStatus | Select-Object AMServiceEnabled,AntivirusEnabled,RealTimeProtectionEnabled,AntispywareEnabled,FullScanAge,QuickScanAge,AntivirusSignatureLastUpdated | Format-List
```

Outputs:

- GPU telemetry at `2026-06-27 15:50:34 +08:00`:
  - GPU0: `24299 MiB / 24576 MiB`, `81%` utilization, `334 W`, `78 C`.
  - GPU1: `24309 MiB / 24576 MiB`, `82%` utilization, `310 W`, `77 C`.
- Top process list showed no unknown high-GPU consumer.
- Defender status was available and enabled.

### Failure Reason / Boundary

- No runtime health failure was found.
- This is not a full malware forensic audit; it is a nonintrusive training-safety check.
- Do not run full scans while both GPUs are training unless suspicious symptoms appear.

### Next Step

- Treat current speed as normal for two frozen-LM rows.
- Continue monitoring log progress, checkpoint writes, temperature, and OOM/Traceback signals.

## 2026-06-27 Dual frozen-LM source mid-run gate 1

### Results

- Both frozen-LM source rows remained active.
- GPU0 `frozen_lm_ums_10k` reached approximately step `2305/10000`.
- GPU1 `frozen_lm_ums_30k` reached approximately step `192/10000`.
- No new `Traceback`, `CUDA out of memory`, `RuntimeError`, or wrapper `EXITCODE` failure line was observed.
- GPU0 10k source latest integrated milestone remains step `2000`: `val_loss=0.0442`, with `best.pt` and `step_2000.pt` saved.
- GPU1 30k source has not reached the first evaluation interval yet.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 300
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|frozen_lm_ums_30k|train_cxr' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 80
Get-Content outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Tail 40
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 100
Get-Content outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Tail 40
```

Outputs:

- GPU0 log: `outputs/logs/data_scaling_frozen_lm_ums_10k_source_gpu0.log`.
- GPU1 log: `outputs/logs/data_scaling_frozen_lm_ums_30k_source_gpu1.log`.

### Runtime / Speed

- GPU0 sample: `24299 MiB / 24576 MiB`, `86%` utilization, `336 W`, `79 C`.
- GPU1 sample: `24309 MiB / 24576 MiB`, `99%` utilization, `312 W`, `81 C`.
- GPU0 training speed: about `2.83 s/it`.
- GPU1 training speed: about `2.88-2.89 s/it`.
- This is still normal for the frozen-LM configuration.

### Failure Reason / Boundary

- No failure.
- Both cards remain near the VRAM ceiling and warm; if a failure occurs, stop and diagnose the newer GPU1 30k row first.
- Do not integrate GPU1 30k source until first evaluation and final artifacts exist.

### Next Step

- Continue monitoring until the next evaluation/checkpoint gate.
- Expected next useful milestones: GPU0 10k step `2500`; GPU1 30k step `500`.

## 2026-06-27 Dual frozen-LM source mid-run gate 2

### Results

- Both frozen-LM source rows remained active.
- GPU0 `frozen_lm_ums_10k` reached step `2500`.
- GPU0 step `2500`: `val_loss=0.0490`.
- GPU1 `frozen_lm_ums_30k` reached approximately step `412/10000`, not yet first eval.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or wrapper `EXITCODE` failure line was observed.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 600
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|frozen_lm_ums_30k|train_cxr' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 120
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 120
Get-Content outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Tail 30
Get-Content outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Tail 30
```

Outputs:

- GPU0 log: `outputs/logs/data_scaling_frozen_lm_ums_10k_source_gpu0.log`.
- GPU1 log: `outputs/logs/data_scaling_frozen_lm_ums_30k_source_gpu1.log`.

### Runtime / Speed

- GPU0 sample: `24299 MiB / 24576 MiB`, `88%` utilization, `340 W`, `78 C`.
- GPU1 sample: `24309 MiB / 24576 MiB`, `86%` utilization, `307 W`, `81 C`.
- GPU0 speed after validation returned toward `2.9 s/it`.
- GPU1 speed remained about `2.87-2.88 s/it`.
- Training speed remains normal for this dual frozen-LM load.

### Failure Reason / Boundary

- No failure.
- GPU1 30k source has not produced validation metrics yet; do not report any 30k frozen-LM result.
- GPU0 10k source still lacks final artifacts; do not launch LP.

### Next Step

- Continue to the GPU1 first evaluation gate at step `500`.
- Keep both rows running if no OOM/thermal/runtime failure appears.

## 2026-06-27 Dual frozen-LM source first-eval gate

### Results

- Both frozen-LM source rows remained active.
- GPU1 `frozen_lm_ums_30k` reached first eval at step `500`.
- GPU1 step `500`: `val_loss=0.0503`.
- GPU1 wrote `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/best.pt`.
- GPU0 `frozen_lm_ums_10k` remained active; latest milestone remains step `2500`, `val_loss=0.0490`.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or wrapper `EXITCODE` failure line was observed.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 360
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|frozen_lm_ums_30k|train_cxr' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 120
Get-ChildItem outputs\data_scaling\frozen_lm_ums_30k -Recurse -Depth 2
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 120
```

Outputs:

- GPU1 first checkpoint: `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/best.pt`.
- GPU1 log: `outputs/logs/data_scaling_frozen_lm_ums_30k_source_gpu1.log`.
- GPU0 log: `outputs/logs/data_scaling_frozen_lm_ums_10k_source_gpu0.log`.

### Runtime / Speed

- GPU0 sample: `24299 MiB / 24576 MiB`, `85%` utilization, `337 W`, `79 C`.
- GPU1 sample: `24311 MiB / 24576 MiB`, `99%` utilization, `312 W`, `82 C`.
- Dual frozen-LM speed and telemetry remain normal for this host and configuration.

### Failure Reason / Boundary

- No failure.
- GPU1 now has a `best.pt`, but it is only an early source checkpoint; do not launch `lp_frozen_lm_ums_30k` until the source wrapper exits with code `0`.
- Do not treat first-eval source loss as final 30k frozen-LM evidence.

### Next Step

- Let both source rows continue.
- Next useful milestones: GPU0 10k step `3000`; GPU1 30k step `1000` and `step_1000.pt`.

## 2026-06-27 Dual frozen-LM source mid-run gate 4

### Results

- Both frozen-LM source rows remained active.
- GPU0 `frozen_lm_ums_10k` reached approximately step `2992/10000`; next eval at step `3000` is imminent.
- GPU1 `frozen_lm_ums_30k` reached approximately step `864/10000`; next eval/checkpoint at step `1000` is expected soon.
- Latest integrated GPU0 metric remains step `2500`: `val_loss=0.0490`.
- Latest integrated GPU1 metric remains step `500`: `val_loss=0.0503`, `best.pt` saved.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or wrapper `EXITCODE` failure line was observed.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 900
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|frozen_lm_ums_30k|train_cxr' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 140
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 140
Get-Content outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Tail 20
Get-Content outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Tail 20
```

Outputs:

- GPU0 log: `outputs/logs/data_scaling_frozen_lm_ums_10k_source_gpu0.log`.
- GPU1 log: `outputs/logs/data_scaling_frozen_lm_ums_30k_source_gpu1.log`.

### Runtime / Speed

- GPU0 sample: `24299 MiB / 24576 MiB`, `99%` utilization, `337 W`, `78 C`.
- GPU1 sample: `24311 MiB / 24576 MiB`, `99%` utilization, `312 W`, `80 C`.
- GPU0 speed remains around `2.83 s/it`.
- GPU1 speed remains around `2.87-2.90 s/it`.

### Failure Reason / Boundary

- No failure.
- Still no final source artifacts for either row.
- Do not launch LP rows before source wrapper success.

### Next Step

- Continue to GPU0 step `3000` and GPU1 step `1000` validation/checkpoint gates.

## 2026-06-27 Dual frozen-LM source checkpoint gate

### Results

- Both frozen-LM source rows remained active.
- GPU0 `frozen_lm_ums_10k` reached step `3000`.
- GPU0 step `3000`: `val_loss=0.0463`.
- GPU0 wrote `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/step_3000.pt`.
- GPU1 `frozen_lm_ums_30k` reached step `1000`.
- GPU1 step `1000`: `val_loss=0.0447`.
- GPU1 wrote:
  - `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/best.pt`
  - `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/step_1000.pt`
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or wrapper `EXITCODE` failure line was observed.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 500
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|frozen_lm_ums_30k|train_cxr' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 160
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 160
Get-ChildItem outputs\data_scaling\frozen_lm_ums_10k\checkpoints | Sort-Object Name
Get-ChildItem outputs\data_scaling\frozen_lm_ums_30k\checkpoints | Sort-Object Name
```

Outputs:

- GPU0 10k checkpoints:
  - `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/best.pt`
  - `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/step_1000.pt`
  - `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/step_2000.pt`
  - `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/step_3000.pt`
- GPU1 30k checkpoints:
  - `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/best.pt`
  - `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/step_1000.pt`

### Runtime / Speed

- GPU0 sample: `24299 MiB / 24576 MiB`, `99%` utilization, `337 W`, `79 C`.
- GPU1 sample: `24311 MiB / 24576 MiB`, `98%` utilization, `310 W`, `81 C`.
- Both rows remain at expected frozen-LM speed.

### Failure Reason / Boundary

- No failure.
- GPU0 10k and GPU1 30k are still source rows in progress; do not launch LP or integrate final metrics yet.
- GPU1 first `step_1000.pt` confirms the dual-GPU plan is stable enough to continue.

### Next Step

- Continue monitoring at larger intervals.
- Next useful milestones: GPU0 step `3500`; GPU1 step `1500`, then GPU0/30k final source completion later.

## 2026-06-27 Dual frozen-LM source long-run gate 1

### Results

- Both frozen-LM source rows remained active.
- GPU0 `frozen_lm_ums_10k` reached step `3500`.
- GPU0 step `3500`: `val_loss=0.0403`; `best.pt` was updated.
- GPU1 `frozen_lm_ums_30k` reached step `1500`.
- GPU1 step `1500`: `val_loss=0.0409`; `best.pt` was updated.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or wrapper `EXITCODE` failure line was observed.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 1800
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|frozen_lm_ums_30k|train_cxr' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 180
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 180
Get-ChildItem outputs\data_scaling\frozen_lm_ums_10k\checkpoints | Sort-Object Name
Get-ChildItem outputs\data_scaling\frozen_lm_ums_30k\checkpoints | Sort-Object Name
```

Outputs:

- GPU0 log: `outputs/logs/data_scaling_frozen_lm_ums_10k_source_gpu0.log`.
- GPU1 log: `outputs/logs/data_scaling_frozen_lm_ums_30k_source_gpu1.log`.
- GPU0 latest best checkpoint: `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/best.pt`.
- GPU1 latest best checkpoint: `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/best.pt`.

### Runtime / Speed

- GPU0 sample: `24299 MiB / 24576 MiB`, `92%` utilization, `339 W`, `79 C`.
- GPU1 sample: `24311 MiB / 24576 MiB`, `98%` utilization, `314 W`, `82 C`.
- Long-run dual frozen-LM speed and thermals remain acceptable.

### Failure Reason / Boundary

- No failure.
- These remain in-progress source rows; no LP or final result integration yet.
- The early `best.pt` checkpoints are useful for provenance, but final source completion is still required before LP launch under the current execution contract.

### Next Step

- Continue monitoring.
- Next useful checkpoints: GPU0 step `4000`; GPU1 step `2000`.

## 2026-06-27 Dual frozen-LM source long-run gate 2

### Results

- Both frozen-LM source rows remained active after the interrupted monitor window was rechecked from current state.
- GPU0 `frozen_lm_ums_10k` reached step `4000`.
- GPU0 step `4000`: `val_loss=0.0408`; `step_4000.pt` was saved.
- GPU1 `frozen_lm_ums_30k` reached step `2000`.
- GPU1 step `2000`: `val_loss=0.0413`; `step_2000.pt` was saved.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or wrapper `EXITCODE` failure line was observed.

### Commands / Inputs / Outputs

```powershell
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|frozen_lm_ums_30k|train_cxr' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 220
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 220
Get-ChildItem outputs\data_scaling\frozen_lm_ums_10k\checkpoints | Sort-Object Name
Get-ChildItem outputs\data_scaling\frozen_lm_ums_30k\checkpoints | Sort-Object Name
```

Outputs:

- GPU0 10k checkpoints now include:
  - `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/best.pt`
  - `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/step_1000.pt`
  - `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/step_2000.pt`
  - `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/step_3000.pt`
  - `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/step_4000.pt`
- GPU1 30k checkpoints now include:
  - `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/best.pt`
  - `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/step_1000.pt`
  - `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/step_2000.pt`

### Runtime / Speed

- GPU0 sample: `24299 MiB / 24576 MiB`, `99%` utilization, `325 W`, `78 C`.
- GPU1 sample: `24305 MiB / 24576 MiB`, `98%` utilization, `293 W`, `77 C`.
- Current speed and thermal profile remain normal.

### Failure Reason / Boundary

- No failure.
- The previous long monitor command was interrupted by continuation control, so this gate uses the refreshed current-state command outputs as authoritative evidence.
- These remain in-progress source rows; no LP launch yet.

### Next Step

- Continue monitoring.
- Next useful milestones: GPU0 step `4500`; GPU1 step `2500`.

## 2026-06-27 Dual frozen-LM source long-run gate 3

### Results

- Both frozen-LM source rows remained active.
- GPU0 `frozen_lm_ums_10k` reached step `4500`.
- GPU0 step `4500`: `val_loss=0.0401`; `best.pt` was updated.
- GPU0 had progressed to approximately step `4895/10000` by the tail sample.
- GPU1 `frozen_lm_ums_30k` reached step `2500`.
- GPU1 step `2500`: `val_loss=0.0412`.
- GPU1 had progressed to approximately step `2711/10000` by the tail sample.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or wrapper `EXITCODE` failure line was observed.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 1500
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|frozen_lm_ums_30k|train_cxr' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 220
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 220
Get-Content outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Tail 20
Get-Content outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Tail 20
```

Outputs:

- GPU0 log: `outputs/logs/data_scaling_frozen_lm_ums_10k_source_gpu0.log`.
- GPU1 log: `outputs/logs/data_scaling_frozen_lm_ums_30k_source_gpu1.log`.
- GPU0 latest best checkpoint: `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/best.pt`.

### Runtime / Speed

- GPU0 sample: `24299 MiB / 24576 MiB`, `77%` utilization, `331 W`, `78 C`.
- GPU1 sample: `24305 MiB / 24576 MiB`, `63%` utilization, `298 W`, `75 C`.
- GPU0 tail speed was about `2.9 s/it`.
- GPU1 tail speed briefly fluctuated around `3.1 s/it`, still within normal dual-load/validation/IO variance and without error signals.

### Failure Reason / Boundary

- No failure.
- Both rows are still source runs in progress; no LP launch yet.
- Do not interpret GPU1 step `2500` as final 30k frozen-LM evidence.

### Next Step

- Continue monitoring.
- Next useful milestones: GPU0 step `5000` and likely `step_5000.pt`; GPU1 step `3000`.

## 2026-06-27 Dual frozen-LM source long-run gate 4

### Results

- Both frozen-LM source rows remained active.
- GPU0 `frozen_lm_ums_10k` reached step `5000`.
- GPU0 step `5000`: `val_loss=0.0392`; `best.pt` and `step_5000.pt` were saved.
- GPU1 `frozen_lm_ums_30k` latest integrated metric remains step `2500`: `val_loss=0.0412`; it had not reached step `3000` at this gate.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or wrapper `EXITCODE` failure line was observed.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 480
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|frozen_lm_ums_30k|train_cxr' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 240
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 240
Get-ChildItem outputs\data_scaling\frozen_lm_ums_10k\checkpoints | Sort-Object Name
Get-ChildItem outputs\data_scaling\frozen_lm_ums_30k\checkpoints | Sort-Object Name
```

Outputs:

- GPU0 10k checkpoints now include `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/step_5000.pt`.
- GPU0 latest best checkpoint: `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/best.pt`.
- GPU1 latest source log: `outputs/logs/data_scaling_frozen_lm_ums_30k_source_gpu1.log`.

### Runtime / Speed

- GPU0 sample: `24299 MiB / 24576 MiB`, `86%` utilization, `333 W`, `78 C`.
- GPU1 sample: `24305 MiB / 24576 MiB`, `84%` utilization, `305 W`, `75 C`.
- Speed and thermals remain normal.

### Failure Reason / Boundary

- No failure.
- GPU0 10k is halfway through but still not complete; no LP launch yet.
- GPU1 30k remains in-progress source evidence only.

### Next Step

- Continue monitoring.
- Next useful milestones: GPU1 step `3000`; GPU0 step `5500`.

## 2026-06-27 Dual frozen-LM source long-run gate 5

### Results

- Both frozen-LM source rows remained active.
- GPU0 `frozen_lm_ums_10k` latest integrated metric remains step `5000`: `val_loss=0.0392`, with `best.pt` and `step_5000.pt` saved.
- GPU1 `frozen_lm_ums_30k` reached step `3000`.
- GPU1 step `3000`: `val_loss=0.0391`; `best.pt` and `step_3000.pt` were saved.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or wrapper `EXITCODE` failure line was observed.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 900
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|frozen_lm_ums_30k|train_cxr' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 260
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 260
Get-ChildItem outputs\data_scaling\frozen_lm_ums_10k\checkpoints | Sort-Object Name
Get-ChildItem outputs\data_scaling\frozen_lm_ums_30k\checkpoints | Sort-Object Name
```

Outputs:

- GPU1 30k checkpoints now include `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/step_3000.pt`.
- GPU1 latest best checkpoint: `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/best.pt`.

### Runtime / Speed

- GPU0 sample: `24299 MiB / 24576 MiB`, `70%` utilization, `330 W`, `79 C`.
- GPU1 sample: `24305 MiB / 24576 MiB`, `90%` utilization, `296 W`, `75 C`.
- No abnormal slow-down was observed.

### Failure Reason / Boundary

- No failure.
- Both source rows remain incomplete; no LP launch yet.
- GPU1 30k best checkpoint is still source-run provenance, not final performance evidence.

### Next Step

- Continue monitoring.
- Next useful milestones: GPU0 step `5500`; GPU1 step `3500`.

## 2026-06-27 Dual frozen-LM source long-run gate 6

### Results

- Both frozen-LM source rows remained active.
- GPU0 `frozen_lm_ums_10k` reached step `5500`.
- GPU0 step `5500`: `val_loss=0.0393`.
- GPU0 had progressed to approximately step `5869/10000` by the tail sample.
- GPU1 `frozen_lm_ums_30k` reached step `3500`.
- GPU1 step `3500`: `val_loss=0.0402`.
- GPU1 had progressed to approximately step `3641/10000` by the tail sample.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or wrapper `EXITCODE` failure line was observed.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 1500
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|frozen_lm_ums_30k|train_cxr' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 280
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 280
Get-Content outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Tail 20
Get-Content outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Tail 20
```

Outputs:

- GPU0 log: `outputs/logs/data_scaling_frozen_lm_ums_10k_source_gpu0.log`.
- GPU1 log: `outputs/logs/data_scaling_frozen_lm_ums_30k_source_gpu1.log`.

### Runtime / Speed

- GPU0 sample: `24299 MiB / 24576 MiB`, `59%` utilization, `292 W`, `77 C`.
- GPU1 sample: `24305 MiB / 24576 MiB`, `100%` utilization, `170 W`, `67 C`.
- Tail training speed was around `3.0 s/it`, normal for the current dual frozen-LM run.

### Failure Reason / Boundary

- No failure.
- GPU0 and GPU1 remain incomplete source rows; no LP launch or final metric integration yet.

### Next Step

- Continue monitoring.
- Next useful milestones: GPU0 step `6000` with `step_6000.pt`; GPU1 step `4000` with `step_4000.pt`.

## 2026-06-27 Dual frozen-LM source checkpoint gate 2

### Results

- Both frozen-LM source rows remained active.
- GPU0 `frozen_lm_ums_10k` reached step `6000`.
- GPU0 step `6000`: `val_loss=0.0393`; `step_6000.pt` was saved.
- GPU1 `frozen_lm_ums_30k` reached step `4000`.
- GPU1 step `4000`: `val_loss=0.0379`; `best.pt` and `step_4000.pt` were saved.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or wrapper `EXITCODE` failure line was observed.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 1200
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|frozen_lm_ums_30k|train_cxr' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 300
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 300
Get-ChildItem outputs\data_scaling\frozen_lm_ums_10k\checkpoints | Sort-Object Name
Get-ChildItem outputs\data_scaling\frozen_lm_ums_30k\checkpoints | Sort-Object Name
```

Outputs:

- GPU0 10k checkpoints now include `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/step_6000.pt`.
- GPU1 30k checkpoints now include `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/step_4000.pt`.
- GPU1 latest best checkpoint: `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/best.pt`.

### Runtime / Speed

- GPU0 sample: `24299 MiB / 24576 MiB`, `79%` utilization, `336 W`, `78 C`.
- GPU1 sample: `24305 MiB / 24576 MiB`, `72%` utilization, `309 W`, `80 C`.
- Runtime remains stable.

### Failure Reason / Boundary

- No failure.
- Both rows are still source rows in progress; no LP launch or final integration yet.

### Next Step

- Continue monitoring.
- Next useful milestones: GPU0 step `6500`; GPU1 step `4500`.

## 2026-06-27 Dual frozen-LM source long-run gate 7

### Results

- Both frozen-LM source rows remained active.
- GPU0 `frozen_lm_ums_10k` reached step `6500`.
- GPU0 step `6500`: `val_loss=0.0410`.
- GPU0 had progressed to approximately step `6795/10000` by the tail sample.
- GPU1 `frozen_lm_ums_30k` reached step `4500`.
- GPU1 step `4500`: `val_loss=0.0387`.
- GPU1 had progressed to approximately step `4551/10000` by the tail sample.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or wrapper `EXITCODE` failure line was observed.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 1500
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|frozen_lm_ums_30k|train_cxr' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 320
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 320
Get-Content outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Tail 20
Get-Content outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Tail 20
```

Outputs:

- GPU0 log: `outputs/logs/data_scaling_frozen_lm_ums_10k_source_gpu0.log`.
- GPU1 log: `outputs/logs/data_scaling_frozen_lm_ums_30k_source_gpu1.log`.

### Runtime / Speed

- GPU0 sample: `24299 MiB / 24576 MiB`, `86%` utilization, `329 W`, `78 C`.
- GPU1 sample: `24305 MiB / 24576 MiB`, `84%` utilization, `308 W`, `78 C`.
- Tail speeds were about `2.9 s/it`.
- Approximate remaining time from this gate: GPU0 about `2.5-2.7 h`, GPU1 about `4.4-4.6 h`.

### Failure Reason / Boundary

- No failure.
- No source row has completed yet; no LP launch yet.

### Next Step

- Continue monitoring.
- Next useful milestones: GPU0 step `7000` with `step_7000.pt`; GPU1 step `5000` with `step_5000.pt`.

## 2026-06-27 Dual frozen-LM source checkpoint gate 3

### Results

- Both frozen-LM source rows remained active.
- GPU0 `frozen_lm_ums_10k` reached step `7000`.
- GPU0 step `7000`: `val_loss=0.0406`; `step_7000.pt` was saved.
- GPU1 `frozen_lm_ums_30k` latest integrated metric remains step `4500`: `val_loss=0.0387`; it had not reached step `5000` at this gate.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or wrapper `EXITCODE` failure line was observed.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 1200
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|frozen_lm_ums_30k|train_cxr' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 340
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 340
Get-ChildItem outputs\data_scaling\frozen_lm_ums_10k\checkpoints | Sort-Object Name
Get-ChildItem outputs\data_scaling\frozen_lm_ums_30k\checkpoints | Sort-Object Name
```

Outputs:

- GPU0 10k checkpoints now include `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/step_7000.pt`.
- GPU1 latest 30k checkpoint remains `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/step_4000.pt`.

### Runtime / Speed

- GPU0 sample: `24299 MiB / 24576 MiB`, `95%` utilization, `325 W`, `78 C`.
- GPU1 sample: `24305 MiB / 24576 MiB`, `77%` utilization, `303 W`, `78 C`.
- Runtime remains stable.

### Failure Reason / Boundary

- No failure.
- No final source artifacts yet; no LP launch.

### Next Step

- Continue monitoring.
- Next useful milestones: GPU0 step `7500`; GPU1 step `5000` with `step_5000.pt`.

## 2026-06-27 Dual frozen-LM source long-run gate 8

### Results

- Both frozen-LM source rows remained active.
- GPU0 `frozen_lm_ums_10k` reached step `7500`.
- GPU0 step `7500`: `val_loss=0.0420`.
- GPU0 had progressed to approximately step `7714/10000` by the tail sample.
- GPU1 `frozen_lm_ums_30k` reached step `5000`.
- GPU1 step `5000`: `val_loss=0.0373`; `best.pt` and `step_5000.pt` were saved.
- GPU1 had progressed to approximately step `5467/10000` by the tail sample.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or wrapper `EXITCODE` failure line was observed.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 1500
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|frozen_lm_ums_30k|train_cxr' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 360
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 360
Get-Content outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Tail 20
Get-Content outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Tail 20
```

Outputs:

- GPU0 log: `outputs/logs/data_scaling_frozen_lm_ums_10k_source_gpu0.log`.
- GPU1 30k checkpoints now include `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/step_5000.pt`.
- GPU1 latest best checkpoint: `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/best.pt`.

### Runtime / Speed

- GPU0 sample: `24299 MiB / 24576 MiB`, `95%` utilization, `336 W`, `79 C`.
- GPU1 sample: `24305 MiB / 24576 MiB`, `80%` utilization, `307 W`, `76 C`.
- Tail speeds remained around `2.9-3.0 s/it`.

### Failure Reason / Boundary

- No failure.
- GPU0 10k is approaching completion but still has no final artifacts; no LP launch yet.
- GPU1 30k source remains in progress.

### Next Step

- Continue monitoring.
- Next useful milestones: GPU0 step `8000` with `step_8000.pt`; GPU1 step `5500`.

## 2026-06-27 Dual frozen-LM source checkpoint gate 4

### Results

- Both frozen-LM source rows remained active.
- GPU0 `frozen_lm_ums_10k` reached step `8000`.
- GPU0 step `8000`: `val_loss=0.0423`; `step_8000.pt` was saved.
- GPU1 `frozen_lm_ums_30k` reached step `5500`.
- GPU1 step `5500`: `val_loss=0.0384`.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or wrapper `EXITCODE` failure line was observed.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 1200
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|frozen_lm_ums_30k|train_cxr' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 380
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 380
Get-ChildItem outputs\data_scaling\frozen_lm_ums_10k\checkpoints | Sort-Object Name
Get-ChildItem outputs\data_scaling\frozen_lm_ums_30k\checkpoints | Sort-Object Name
```

Outputs:

- GPU0 10k checkpoints now include `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/step_8000.pt`.
- GPU1 latest 30k checkpoint remains `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/step_5000.pt`.

### Runtime / Speed

- GPU0 sample: `24299 MiB / 24576 MiB`, `73%` utilization, `327 W`, `78 C`.
- GPU1 sample: `24305 MiB / 24576 MiB`, `76%` utilization, `303 W`, `77 C`.
- Runtime remains stable.

### Failure Reason / Boundary

- No failure.
- GPU0 10k is close to source completion but still in progress; no LP launch yet.
- GPU1 30k remains in progress.

### Next Step

- Continue monitoring.
- Next useful milestones: GPU0 step `8500`; GPU1 step `6000` with `step_6000.pt`.

## 2026-06-27 Dual frozen-LM source checkpoint gate 5

### Results

- Both frozen-LM source rows remained active.
- GPU0 `frozen_lm_ums_10k` reached step `8500`.
- GPU0 step `8500`: `val_loss=0.0445`.
- GPU1 `frozen_lm_ums_30k` reached step `6000`.
- GPU1 step `6000`: `val_loss=0.0371`; `best.pt` and `step_6000.pt` were saved.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or wrapper `EXITCODE` failure line was observed.
- A transient `git hash-object` process was visible in the process list, likely from editor/Codex file state tracking; it was not a GPU compute process and did not consume GPU memory.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 1200
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|frozen_lm_ums_30k|train_cxr' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 400
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 400
Get-ChildItem outputs\data_scaling\frozen_lm_ums_10k\checkpoints | Sort-Object Name
Get-ChildItem outputs\data_scaling\frozen_lm_ums_30k\checkpoints | Sort-Object Name
```

Outputs:

- GPU1 30k checkpoints now include `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/step_6000.pt`.
- GPU1 latest best checkpoint: `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/best.pt`.
- GPU0 latest source log: `outputs/logs/data_scaling_frozen_lm_ums_10k_source_gpu0.log`.

### Runtime / Speed

- GPU0 sample: `24299 MiB / 24576 MiB`, `98%` utilization, `328 W`, `78 C`.
- GPU1 sample: `24305 MiB / 24576 MiB`, `98%` utilization, `301 W`, `79 C`.
- Runtime remains stable.

### Failure Reason / Boundary

- No failure.
- GPU0 10k remains in progress; no LP launch until wrapper `EXITCODE 0`.
- GPU1 30k remains in progress.

### Next Step

- Continue monitoring.
- Next useful milestones: GPU0 step `9000` with `step_9000.pt`; GPU1 step `6500`.

## 2026-06-27 Dual frozen-LM source long-run gate 9

### Results

- Both frozen-LM source rows remained active.
- GPU0 `frozen_lm_ums_10k` latest integrated metric remains step `8500`: `val_loss=0.0445`; it had not reached step `9000` at this gate.
- GPU1 `frozen_lm_ums_30k` reached step `6500`.
- GPU1 step `6500`: `val_loss=0.0367`; `best.pt` was updated.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or wrapper `EXITCODE` failure line was observed.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 1200
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|frozen_lm_ums_30k|train_cxr' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 420
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 420
Get-ChildItem outputs\data_scaling\frozen_lm_ums_10k\checkpoints | Sort-Object Name
Get-ChildItem outputs\data_scaling\frozen_lm_ums_30k\checkpoints | Sort-Object Name
```

Outputs:

- GPU1 latest best checkpoint: `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/best.pt`.
- GPU0 latest 10k checkpoint remains `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/step_8000.pt`.

### Runtime / Speed

- GPU0 sample: `24299 MiB / 24576 MiB`, `99%` utilization, `338 W`, `78 C`.
- GPU1 sample: `24305 MiB / 24576 MiB`, `70%` utilization, `295 W`, `77 C`.
- Runtime remains stable.

### Failure Reason / Boundary

- No failure.
- GPU0 10k is still in progress; no LP launch.
- GPU1 30k remains in progress.

### Next Step

- Continue monitoring.
- Next useful milestones: GPU0 step `9000` with `step_9000.pt`; GPU1 step `7000` with `step_7000.pt`.

## 2026-06-27 Dual frozen-LM source checkpoint gate 6

### Results

- Both frozen-LM source rows remained active.
- GPU0 `frozen_lm_ums_10k` reached step `9000`.
- GPU0 step `9000`: `val_loss=0.0460`; `step_9000.pt` was saved.
- GPU1 `frozen_lm_ums_30k` latest integrated metric remains step `6500`: `val_loss=0.0367`; `best.pt` was updated at that step.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or wrapper `EXITCODE` failure line was observed.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 900
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|frozen_lm_ums_30k|train_cxr' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 440
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 440
Get-ChildItem outputs\data_scaling\frozen_lm_ums_10k\checkpoints | Sort-Object Name
Get-ChildItem outputs\data_scaling\frozen_lm_ums_30k\checkpoints | Sort-Object Name
```

Outputs:

- GPU0 10k checkpoints now include `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/step_9000.pt`.
- GPU1 latest best checkpoint: `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/best.pt`.

### Runtime / Speed

- GPU0 sample: `24299 MiB / 24576 MiB`, `85%` utilization, `328 W`, `79 C`.
- GPU1 sample: `24305 MiB / 24576 MiB`, `83%` utilization, `304 W`, `78 C`.
- Runtime remains stable.

### Failure Reason / Boundary

- No failure.
- GPU0 10k source is near completion but still lacks final artifacts and wrapper `EXITCODE 0`.
- Do not launch `lp_frozen_lm_ums_10k` until source completion is verified.

### Next Step

- Continue monitoring GPU0 to source completion.
- Keep GPU1 30k source running.

## 2026-06-27 Dual frozen-LM source near-completion gate

### Results

- Both frozen-LM source rows remained active.
- GPU0 `frozen_lm_ums_10k` reached step `9500`.
- GPU0 step `9500`: `val_loss=0.0461`.
- GPU0 had progressed to approximately step `9853/10000` by the tail sample.
- GPU1 `frozen_lm_ums_30k` reached step `7500`.
- GPU1 step `7500`: `val_loss=0.0376`.
- GPU1 had progressed to approximately step `7550/10000` by the tail sample.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or wrapper `EXITCODE` failure line was observed.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 1800
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|frozen_lm_ums_30k|train_cxr' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 460
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 460
Get-Content outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Tail 80
Get-Content outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Tail 40
```

Outputs:

- GPU0 log: `outputs/logs/data_scaling_frozen_lm_ums_10k_source_gpu0.log`.
- GPU1 log: `outputs/logs/data_scaling_frozen_lm_ums_30k_source_gpu1.log`.

### Runtime / Speed

- GPU0 sample: `24299 MiB / 24576 MiB`, `86%` utilization, `333 W`, `79 C`.
- GPU1 sample: `24305 MiB / 24576 MiB`, `79%` utilization, `307 W`, `78 C`.
- GPU0 source was within about ten minutes of the terminal step at this gate, plus final evaluation/checkpoint overhead.

### Failure Reason / Boundary

- No failure.
- GPU0 source still lacks final artifacts and wrapper `EXITCODE 0`; do not launch LP yet.
- GPU1 30k source remains in progress.

### Next Step

- Monitor GPU0 to source completion.
- After GPU0 source completes, integrate 10k frozen-LM source evidence first, then launch `lp_frozen_lm_ums_10k` only after writing its execution-before gate.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_10K_FROZEN_LM_UMS_SOURCE_RUN completion and integration gate

### Results

- Formal frozen-LM UMS 10k source row completed with wrapper `EXITCODE 0`.
- GPU0 is now idle.
- GPU1 `frozen_lm_ums_30k` source remains active.
- Source validation-loss summary:
  - Latest/final source step: `10000`.
  - Final source `val_loss=0.046600`.
  - Best source `val_loss=0.039200` at step `5000`.
- Source artifacts:
  - `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/best.pt`
  - `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/final.pt`
  - `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/step_10000.pt`
- `revision_execution_status.csv` now has `56` rows and includes `P1_DATA_SCALING_10K_FROZEN_LM_UMS_SOURCE_RUN`.
- `revision_completion_gap_audit.csv` now records 10k frozen-LM source as complete, with 10k frozen-LM LP still missing.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 900
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_10k|frozen_lm_ums_30k|train_cxr' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log -Pattern 'Step [0-9]+:|Training completed|Final|final|Checkpoint saved|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 160
Get-ChildItem outputs\data_scaling\frozen_lm_ums_10k -Recurse -Depth 2 | Sort-Object FullName
python scripts\summarize_data_scaling_frozen_source_progress.py
python scripts\summarize_revision_execution_status.py
python scripts\summarize_revision_completion_gap_audit.py
python scripts\summarize_current_priority_completion_audit.py
python -m py_compile scripts\summarize_data_scaling_frozen_source_progress.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_current_priority_completion_audit.py
Import-Csv outputs\final_tables\data_scaling_frozen_source_progress.csv | Format-Table -AutoSize
Import-Csv outputs\final_tables\revision_execution_status.csv | Where-Object { $_.task_id -match '10K_FROZEN|30K_FROZEN' } | Format-List
Import-Csv outputs\final_tables\revision_completion_gap_audit.csv | Where-Object { $_.requirement -eq 'P1_DATA_SCALING' } | Format-List
```

Inputs:

- Wrapper: `scripts/run_data_scaling_frozen_lm_ums_10k_source_gpu0.cmd`.
- Log: `outputs/logs/data_scaling_frozen_lm_ums_10k_source_gpu0.log`.
- Config: `configs/data_scaling/frozen_lm_ums_10k.yaml`.

Outputs:

- `outputs/final_tables/data_scaling_frozen_source_progress.csv`
- `outputs/final_tables/data_scaling_frozen_source_progress.md`
- `outputs/final_tables/data_scaling_frozen_source_trajectory.csv`
- `outputs/final_tables/revision_execution_status.csv`
- `outputs/final_tables/revision_completion_gap_audit.csv`
- `outputs/final_tables/current_priority_completion_audit.csv`

### Runtime / Speed

- GPU0 was released after source completion.
- GPU1 30k source sample at integration gate: `24305 MiB / 24576 MiB`, `14%` utilization at sample time, `207 W`, `65 C`; it remains active.
- The completed GPU0 source row ran at normal frozen-LM speed throughout the monitored gates.

### Failure Reason / Boundary

- No source runtime failure.
- The completed 10k frozen-LM row is source-loss/checkpoint evidence only; it is not a downstream LP macro-AUC/F1 result.
- Do not compare source val_loss directly against no-LM/BCE LP macro-AUC.

### Next Step

- Launch `P1_DATA_SCALING_10K_FROZEN_LM_UMS_LP_RUN` on GPU0 from `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/best.pt`.
- Keep GPU1 30k source running.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_10K_FROZEN_LM_UMS_LP_RUN execution-before

### Plan

- Use idle GPU0 for the dependent 10k frozen-LM LP row.
- Keep GPU1 30k frozen-LM source running; the LP row is light and does not depend on the 30k source result.
- Initialize ViT from `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/best.pt`, freeze the backbone, and train the linear probe head.
- Keep 10k frozen-LM LP metrics separate from 10k frozen-LM source val_loss.

### Commands

Preflight and launch:

```powershell
Import-Csv outputs\final_tables\data_scaling_config_validation.csv | Where-Object { $_.method -eq 'lp_frozen_lm_ums_10k' } | Format-List
Get-Content configs\data_scaling\lp_frozen_lm_ums_10k.yaml
Test-Path outputs\data_scaling\frozen_lm_ums_10k\checkpoints\best.pt
if (Test-Path outputs\data_scaling\lp_frozen_lm_ums_10k) { Get-ChildItem outputs\data_scaling\lp_frozen_lm_ums_10k -Recurse -Depth 2 } else { 'outputs/data_scaling/lp_frozen_lm_ums_10k missing' }
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_30k|lp_frozen_lm_ums_10k|train_cxr|train_vit_baseline' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Start-Process -FilePath cmd.exe -ArgumentList '/c','H:\Xiyao_Wang\021_260129VIVID\scripts\run_data_scaling_lp_frozen_lm_ums_10k_gpu0.cmd' -WindowStyle Hidden -PassThru
```

Monitoring:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_30k|lp_frozen_lm_ums_10k|train_cxr|train_vit_baseline' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Get-Content outputs\logs\data_scaling_lp_frozen_lm_ums_10k_gpu0.log -Tail 120
Get-Content outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Tail 80
```

### Inputs

- Config: `configs/data_scaling/lp_frozen_lm_ums_10k.yaml`.
- Source checkpoint: `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/best.pt`.
- Train split: `data/splits/chexpert_train_10k.jsonl`.
- Validation split: `data/splits/chexpert_val_fixed.jsonl`.
- Wrapper: `scripts/run_data_scaling_lp_frozen_lm_ums_10k_gpu0.cmd`.

### Expected Outputs

- Log: `outputs/logs/data_scaling_lp_frozen_lm_ums_10k_gpu0.log`.
- Output dir: `outputs/data_scaling/lp_frozen_lm_ums_10k`.
- Expected artifacts: `best.pt`, `final.pt`, `metrics_final.json`, `metrics_step_*.json`, periodic `step_*.pt`.

### Stop Conditions

- Stop before launch if the source checkpoint is missing, if `outputs/data_scaling/lp_frozen_lm_ums_10k/final.pt` already exists, or if GPU0 is not idle.
- Stop during run on nonzero wrapper exit code, traceback, OOM, stalled log/checkpoint writes, or unexpected contention with GPU1 30k source.
- If the LP row fails, keep GPU1 30k source running and document the LP failure separately.

### Preflight Results

- `data_scaling_config_validation.csv` marks `lp_frozen_lm_ums_10k` as `available`, `ok`.
- Source checkpoint `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/best.pt` exists.
- `outputs/data_scaling/lp_frozen_lm_ums_10k` was absent before launch.
- GPU0 is idle before launch.
- GPU1 remains occupied by `frozen_lm_ums_30k` source.
- Added wrapper `scripts/run_data_scaling_lp_frozen_lm_ums_10k_gpu0.cmd`.

## 2026-06-27 Phase 1 / P1_DATA_SCALING_10K_FROZEN_LM_UMS_LP_RUN runtime gate

### Results

- Launched `lp_frozen_lm_ums_10k` on GPU0 with wrapper PID `22784` and Python PID `21636`.
- GPU1 `frozen_lm_ums_30k` source continued with wrapper PID `19432` and Python PID `19372`.
- The LP row entered training and validation successfully; tail sample reached step `200/3000`.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or wrapper `EXITCODE` failure line was observed at the runtime gate.
- 30k frozen-LM source remained in progress; latest integrated source metric was still step `7500`, `val_loss=0.0376`.

### Commands / Inputs / Outputs

```powershell
Start-Process -FilePath cmd.exe -ArgumentList '/c','H:\Xiyao_Wang\021_260129VIVID\scripts\run_data_scaling_lp_frozen_lm_ums_10k_gpu0.cmd' -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 45
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_30k|lp_frozen_lm_ums_10k|train_cxr|train_vit_baseline' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Get-Content outputs\logs\data_scaling_lp_frozen_lm_ums_10k_gpu0.log -Tail 160
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError' | Select-Object -Last 80
```

Outputs:

- LP log: `outputs/logs/data_scaling_lp_frozen_lm_ums_10k_gpu0.log`.
- LP output dir: `outputs/data_scaling/lp_frozen_lm_ums_10k`.
- 30k source log: `outputs/logs/data_scaling_frozen_lm_ums_30k_source_gpu1.log`.

### Runtime / Speed

- GPU0 LP sample: `1933 MiB / 24576 MiB`, `0%` instant utilization during validation sample, `47.99 W`, `52 C`; training tail showed roughly `5-9 it/s`.
- GPU1 source sample: `24305 MiB / 24576 MiB`, `81%` utilization, `305.88 W`, `83 C`.
- This confirms the safe mixed mode: one light LP on GPU0 plus one full frozen-LM source row on GPU1.

### Failure Reason / Boundary

- No failure at this gate.
- The LP row is still incomplete; do not integrate it into data-scaling metrics until wrapper `EXITCODE 0` and final artifacts are present.
- Do not launch 30k LP until 30k source wrapper exits successfully, even though an intermediate `best.pt` already exists.

### Next Step

- Monitor `lp_frozen_lm_ums_10k` to completion and then refresh the 10k data-scaling summary/status/gap tables.
- Continue monitoring GPU1 30k source independently.

## 2026-06-28 Phase 1 / P1_DATA_SCALING_10K_FROZEN_LM_UMS_LP_RUN completion and integration gate

### Results

- Formal frozen-LM UMS 10k LP row completed with wrapper `EXITCODE 0`.
- GPU0 was released after completion.
- GPU1 `frozen_lm_ums_30k` source remained active and continued to step `9500`.
- LP final metrics:
  - Final/best-val step: `3000`.
  - Final `val_loss=0.281845`.
  - Final `macro_auc=0.777809`.
  - Final `macro_f1=0.908396`.
  - Final `micro_f1=0.883059`.
- LP best metric policies:
  - Best validation loss: step `3000`, `val_loss=0.281845`, `macro_auc=0.777809`, `macro_f1=0.908396`, `micro_f1=0.883059`.
  - Best macro-AUC: step `1600`, `val_loss=0.301213`, `macro_auc=0.791607`, `macro_f1=0.904913`, `micro_f1=0.879152`.
  - Best macro-F1: step `1800`, `val_loss=0.323649`, `macro_auc=0.787068`, `macro_f1=0.910677`, `micro_f1=0.877477`.
- `revision_execution_status.csv` now has `57` rows and includes `P1_DATA_SCALING_10K_FROZEN_LM_UMS_LP_RUN`.
- `revision_completion_gap_audit.csv` now records 10k matched source+LP as complete; remaining original-scope data-scaling gap is `30k frozen-LM source+LP`.

### Commands / Inputs / Outputs

```powershell
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_30k|lp_frozen_lm_ums_10k|train_cxr|train_vit_baseline' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_lp_frozen_lm_ums_10k_gpu0.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError|Final' | Select-Object -Last 120
Get-ChildItem outputs\data_scaling\lp_frozen_lm_ums_10k | Sort-Object Name | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize
Get-Content outputs\data_scaling\lp_frozen_lm_ums_10k\metrics_final.json
python scripts\summarize_data_scaling_10k_progress.py
python scripts\summarize_data_scaling_frozen_source_progress.py
python scripts\summarize_revision_execution_status.py
python scripts\summarize_revision_completion_gap_audit.py
python scripts\summarize_current_priority_completion_audit.py
python -m py_compile scripts\summarize_data_scaling_10k_progress.py scripts\summarize_data_scaling_frozen_source_progress.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_current_priority_completion_audit.py
Import-Csv outputs\final_tables\data_scaling_10k_progress.csv | Where-Object { $_.run_id -eq 'lp_frozen_lm_ums_10k' } | Format-Table -AutoSize
Import-Csv outputs\final_tables\data_scaling_frozen_source_progress.csv | Format-Table -AutoSize
Import-Csv outputs\final_tables\revision_execution_status.csv | Where-Object { $_.task_id -match '10K_FROZEN|30K_FROZEN' } | Format-List
Import-Csv outputs\final_tables\revision_completion_gap_audit.csv | Where-Object { $_.requirement -eq 'P1_DATA_SCALING' } | Format-List
```

Inputs:

- Wrapper: `scripts/run_data_scaling_lp_frozen_lm_ums_10k_gpu0.cmd`.
- Config: `configs/data_scaling/lp_frozen_lm_ums_10k.yaml`.
- Source checkpoint: `outputs/data_scaling/frozen_lm_ums_10k/checkpoints/best.pt`.
- Log: `outputs/logs/data_scaling_lp_frozen_lm_ums_10k_gpu0.log`.

Outputs:

- `outputs/data_scaling/lp_frozen_lm_ums_10k/metrics_final.json`
- `outputs/data_scaling/lp_frozen_lm_ums_10k/metrics_step_1600.json`
- `outputs/data_scaling/lp_frozen_lm_ums_10k/best.pt`
- `outputs/data_scaling/lp_frozen_lm_ums_10k/final.pt`
- `outputs/final_tables/data_scaling_10k_progress.csv`
- `outputs/final_tables/data_scaling_10k_progress.md`
- `outputs/final_tables/data_scaling_10k_trajectory.csv`
- `outputs/final_tables/revision_execution_status.csv`
- `outputs/final_tables/revision_completion_gap_audit.csv`
- `outputs/final_tables/current_priority_completion_audit.csv`

### Runtime / Speed

- Completion check sample: GPU0 `0 MiB / 24576 MiB`, idle after LP completion.
- GPU1 30k source sample at the same gate: `24305 MiB / 24576 MiB`, `63-93%` utilization across samples, `280 W`, `79-80 C`.
- The LP completed quickly and did not interfere with the full frozen-LM source row.

### Failure Reason / Boundary

- No LP runtime failure.
- The 10k frozen-LM LP result is now a downstream classification/LP result and must be compared only against matched LP rows.
- The 10k frozen-LM source `val_loss` remains source-provenance evidence only.
- Do not generalize 10k matched results to 30k until the 30k frozen-LM source+LP branch completes.

### Next Step

- Continue monitoring `frozen_lm_ums_30k` source to wrapper `EXITCODE 0`.
- After 30k source completion, refresh frozen-source summaries, write a 30k source completion gate, then launch the dependent 30k frozen-LM LP.

## 2026-06-28 Phase 1 / P1_DATA_SCALING_30K_FROZEN_LM_UMS_SOURCE_RUN completion and integration gate

### Results

- Formal frozen-LM UMS 30k source row completed with wrapper `EXITCODE 0`.
- Both GPUs were idle after completion.
- Named 30k split contains `29000` train records and `19047` train patients.
- Source validation-loss summary:
  - Latest/final source step: `10000`.
  - Final source `val_loss=0.036400`.
  - Best source `val_loss=0.036100` at step `9500`.
- Source artifacts:
  - `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/best.pt`
  - `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/final.pt`
  - `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/step_10000.pt`
- `data_scaling_frozen_source_progress.csv` now marks both 10k and 30k frozen-LM source rows as `completed`.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 420
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_30k|lp_frozen_lm_ums_30k|train_cxr|train_vit_baseline' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_frozen_lm_ums_30k_source_gpu1.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError|Final' | Select-Object -Last 140
Get-ChildItem outputs\data_scaling\frozen_lm_ums_30k\checkpoints | Sort-Object Name | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize
python scripts\summarize_data_scaling_frozen_source_progress.py
Import-Csv outputs\final_tables\data_scaling_frozen_source_progress.csv | Format-Table -AutoSize
```

Inputs:

- Wrapper: `scripts/run_data_scaling_frozen_lm_ums_30k_source_gpu1.cmd`.
- Log: `outputs/logs/data_scaling_frozen_lm_ums_30k_source_gpu1.log`.
- Config: `configs/data_scaling/frozen_lm_ums_30k.yaml`.

Outputs:

- `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/best.pt`
- `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/final.pt`
- `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/step_10000.pt`
- `outputs/final_tables/data_scaling_frozen_source_progress.csv`
- `outputs/final_tables/data_scaling_frozen_source_progress.md`
- `outputs/final_tables/data_scaling_frozen_source_trajectory.csv`

### Runtime / Speed

- Final source completion released GPU1.
- Completion sample: GPU0 `0 MiB / 24576 MiB`, GPU1 `0 MiB / 24576 MiB`.
- The run sustained normal frozen-LM source behavior through completion; late-stage checkpoints were saved at steps `8000`, `9000`, `10000`, and final.

### Failure Reason / Boundary

- No source runtime failure.
- This row is source-loss/checkpoint evidence only; it is not the downstream LP/classification result.
- Do not compare source `val_loss` directly to 30k no-LM LP macro-AUC/F1.

### Next Step

- Launch `P1_DATA_SCALING_30K_FROZEN_LM_UMS_LP_RUN` from `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/best.pt`.

## 2026-06-28 Phase 1 / P1_DATA_SCALING_30K_FROZEN_LM_UMS_LP_RUN execution-before

### Plan

- Use GPU0 for the dependent 30k frozen-LM LP row.
- Initialize ViT from `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/best.pt`, freeze the backbone, and train the linear-probe head.
- Keep named split size explicit: this `30k` config has `29000` train records.
- Keep 30k frozen-LM LP metrics separate from 30k frozen-LM source val_loss.

### Commands

Preflight and launch:

```powershell
Import-Csv outputs\final_tables\data_scaling_config_validation.csv | Where-Object { $_.method -eq 'lp_frozen_lm_ums_30k' } | Format-List
Get-Content configs\data_scaling\lp_frozen_lm_ums_30k.yaml
Test-Path outputs\data_scaling\frozen_lm_ums_30k\checkpoints\best.pt
if (Test-Path outputs\data_scaling\lp_frozen_lm_ums_30k) { Get-ChildItem outputs\data_scaling\lp_frozen_lm_ums_30k -Recurse -Depth 2 } else { 'outputs/data_scaling/lp_frozen_lm_ums_30k missing' }
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'frozen_lm_ums_30k|lp_frozen_lm_ums_30k|train_cxr|train_vit_baseline' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Start-Process -FilePath cmd.exe -ArgumentList '/c','H:\Xiyao_Wang\021_260129VIVID\scripts\run_data_scaling_lp_frozen_lm_ums_30k_gpu0.cmd' -WindowStyle Hidden -PassThru
```

Monitoring:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'lp_frozen_lm_ums_30k|train_vit_baseline' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Get-Content outputs\logs\data_scaling_lp_frozen_lm_ums_30k_gpu0.log -Tail 120
```

### Inputs

- Config: `configs/data_scaling/lp_frozen_lm_ums_30k.yaml`.
- Source checkpoint: `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/best.pt`.
- Train split: `data/splits/chexpert_train_30k.jsonl` (`29000` records).
- Validation split: `data/splits/chexpert_val_fixed.jsonl`.
- Wrapper: `scripts/run_data_scaling_lp_frozen_lm_ums_30k_gpu0.cmd`.

### Expected Outputs

- Log: `outputs/logs/data_scaling_lp_frozen_lm_ums_30k_gpu0.log`.
- Output dir: `outputs/data_scaling/lp_frozen_lm_ums_30k`.
- Expected artifacts: `best.pt`, `final.pt`, `metrics_final.json`, `metrics_step_*.json`, periodic `step_*.pt`.

### Stop Conditions

- Stop before launch if the source checkpoint is missing, if `outputs/data_scaling/lp_frozen_lm_ums_30k/final.pt` already exists, or if GPU0 is not idle.
- Stop during run on nonzero wrapper exit code, traceback, OOM, stalled log/checkpoint writes, or unexpected GPU contention.
- If the LP row fails, document the failure separately and do not mark the full 30k matched branch complete.

### Preflight Results

- `data_scaling_config_validation.csv` marks `lp_frozen_lm_ums_30k` as `available`, `ok`.
- Source checkpoint `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/best.pt` exists.
- Both GPUs are idle before launch.
- Added wrapper `scripts/run_data_scaling_lp_frozen_lm_ums_30k_gpu0.cmd`.

## 2026-06-28 Phase 1 / P1_DATA_SCALING_30K_FROZEN_LM_UMS_LP_RUN runtime gate

### Results

- Launched `lp_frozen_lm_ums_30k` on GPU0 with wrapper PID `22744` and Python PID `4296`.
- The LP row entered training and validation successfully.
- Runtime gate reached step `200/3000`.
- Step `200`: `val_loss=0.3089`.
- No `Traceback`, `CUDA out of memory`, `RuntimeError`, or wrapper `EXITCODE` failure line was observed at this gate.

### Commands / Inputs / Outputs

```powershell
Start-Process -FilePath cmd.exe -ArgumentList '/c','H:\Xiyao_Wang\021_260129VIVID\scripts\run_data_scaling_lp_frozen_lm_ums_30k_gpu0.cmd' -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 45
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'lp_frozen_lm_ums_30k|train_vit_baseline' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Get-Content outputs\logs\data_scaling_lp_frozen_lm_ums_30k_gpu0.log -Tail 160
Get-ChildItem outputs\data_scaling\lp_frozen_lm_ums_30k | Sort-Object Name | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize
```

Outputs:

- LP log: `outputs/logs/data_scaling_lp_frozen_lm_ums_30k_gpu0.log`.
- LP output dir: `outputs/data_scaling/lp_frozen_lm_ums_30k`.
- Early artifact: `outputs/data_scaling/lp_frozen_lm_ums_30k/metrics_step_200.json`.

### Runtime / Speed

- GPU0 LP sample: `1933 MiB / 24576 MiB`, `40%` instant utilization, `88.84 W`, `53 C`.
- GPU1 sample: idle, `0 MiB / 24576 MiB`.
- Training tail showed roughly `6-9 it/s` outside validation/checkpoint pauses.

### Failure Reason / Boundary

- No failure at this gate.
- The LP row is still incomplete; do not integrate it into 30k matched metrics until wrapper `EXITCODE 0` and final artifacts are present.

### Next Step

- Monitor `lp_frozen_lm_ums_30k` to completion.
- After completion, refresh 30k progress/status/gap/current-priority tables and decide whether the original data-scaling matrix is now complete under the documented 1k early-stop caveat.

## 2026-06-28 Phase 1 / P1_DATA_SCALING_30K_FROZEN_LM_UMS_LP_RUN completion and final integration gate

### Results

- Formal frozen-LM UMS 30k LP row completed with wrapper `EXITCODE 0`.
- Both GPUs are idle after completion.
- Named 30k split size remains `29000` train records.
- LP final metrics:
  - Final step: `3000`.
  - Final `val_loss=0.267407`.
  - Final `macro_auc=0.801109`.
  - Final `macro_f1=0.911076`.
  - Final `micro_f1=0.885850`.
- LP best metric policies:
  - Best validation loss: step `2800`, `val_loss=0.265781`, `macro_auc=0.804373`, `macro_f1=0.913133`, `micro_f1=0.885292`.
  - Best macro-AUC: step `800`, `val_loss=0.308243`, `macro_auc=0.817147`, `macro_f1=0.891617`, `micro_f1=0.868825`.
  - Best macro-F1: step `2800`, `val_loss=0.265781`, `macro_auc=0.804373`, `macro_f1=0.913133`, `micro_f1=0.885292`.
- 30k matched LP comparison:
  - no-LM final macro-AUC `0.815050` vs frozen-LM final macro-AUC `0.801109`.
  - no-LM best-AUC `0.849124` vs frozen-LM best-AUC `0.817147`.
- `revision_execution_status.csv` now has `59` rows and includes `P1_DATA_SCALING_30K_FROZEN_LM_UMS_SOURCE_RUN` and `P1_DATA_SCALING_30K_FROZEN_LM_UMS_LP_RUN`.
- `revision_completion_gap_audit.csv` now records the 1k/3k/10k/30k matched LP matrix as complete, with a documented 1k frozen-LM source early-stop provenance caveat.
- `llm_necessity.csv` was refreshed: low-data frozen-LM necessity is `mixed_no_broad_necessity`, not a broad data-efficiency claim.

### Commands / Inputs / Outputs

```powershell
Start-Sleep -Seconds 420
Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'lp_frozen_lm_ums_30k|train_vit_baseline' } | Select-Object ProcessId,Name,CommandLine | Format-List
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
Select-String -Path outputs\logs\data_scaling_lp_frozen_lm_ums_30k_gpu0.log -Pattern 'Step [0-9]+:|Checkpoint saved|Training completed|EXITCODE|Traceback|CUDA out of memory|RuntimeError|Final' | Select-Object -Last 120
Get-ChildItem outputs\data_scaling\lp_frozen_lm_ums_30k | Sort-Object Name | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize
Get-Content outputs\data_scaling\lp_frozen_lm_ums_30k\metrics_final.json
python scripts\summarize_data_scaling_30k_progress.py
python scripts\summarize_data_scaling_frozen_source_progress.py
python scripts\summarize_phase4_revision_synthesis.py
python scripts\summarize_revision_execution_status.py
python scripts\summarize_revision_completion_gap_audit.py
python scripts\summarize_current_priority_completion_audit.py
python -m py_compile scripts\summarize_data_scaling_30k_progress.py scripts\summarize_data_scaling_frozen_source_progress.py scripts\summarize_phase4_revision_synthesis.py scripts\summarize_revision_execution_status.py scripts\summarize_revision_completion_gap_audit.py scripts\summarize_current_priority_completion_audit.py
Import-Csv outputs\final_tables\data_scaling_30k_progress.csv | Where-Object { $_.run_id -match 'lp_(no_lm|frozen_lm)_ums_30k' } | Format-Table -AutoSize
Import-Csv outputs\final_tables\revision_execution_status.csv | Where-Object { $_.task_id -match '30K_FROZEN' } | Format-List
Import-Csv outputs\final_tables\revision_completion_gap_audit.csv | Where-Object { $_.requirement -eq 'P1_DATA_SCALING' -or $_.requirement -eq 'Final required output set' -or $_.requirement -eq 'Phase 4 paper tables and writing checklist' } | Format-List
Import-Csv outputs\final_tables\llm_necessity.csv | Where-Object { $_.claim_area -eq 'Low-data frozen-LM necessity' } | Format-List
Import-Csv outputs\final_tables\current_priority_completion_audit.csv | Where-Object { $_.audit_item -eq 'Low-data scaling' -or $_.audit_item -eq 'Goal completion decision' } | Format-List
```

Inputs:

- Wrapper: `scripts/run_data_scaling_lp_frozen_lm_ums_30k_gpu0.cmd`.
- Config: `configs/data_scaling/lp_frozen_lm_ums_30k.yaml`.
- Source checkpoint: `outputs/data_scaling/frozen_lm_ums_30k/checkpoints/best.pt`.
- Log: `outputs/logs/data_scaling_lp_frozen_lm_ums_30k_gpu0.log`.

Outputs:

- `outputs/data_scaling/lp_frozen_lm_ums_30k/metrics_final.json`
- `outputs/data_scaling/lp_frozen_lm_ums_30k/metrics_step_800.json`
- `outputs/data_scaling/lp_frozen_lm_ums_30k/best.pt`
- `outputs/data_scaling/lp_frozen_lm_ums_30k/final.pt`
- `outputs/final_tables/data_scaling_30k_progress.csv`
- `outputs/final_tables/data_scaling_30k_progress.md`
- `outputs/final_tables/data_scaling_30k_trajectory.csv`
- `outputs/final_tables/llm_necessity.csv`
- `outputs/final_tables/llm_necessity.md`
- `outputs/final_tables/revision_execution_status.csv`
- `outputs/final_tables/revision_completion_gap_audit.csv`
- `outputs/final_tables/current_priority_completion_audit.csv`

### Runtime / Speed

- Completion check sample: GPU0 `0 MiB / 24576 MiB`, GPU1 `0 MiB / 24576 MiB`.
- The LP completed in minutes and did not produce GPU contention.

### Failure Reason / Boundary

- No LP runtime failure.
- The completed 30k LP row is downstream classification evidence and should be compared to 30k no-LM LP under explicit metric policies.
- The 30k frozen-LM source `val_loss` remains source-provenance evidence only.
- The remaining caveat is not a missing 30k row: it is the already documented 1k frozen-LM source early-stop provenance.

### Next Step

- Use the completed data-scaling matrix with metric-policy qualifiers.
- Preserve the final low-data interpretation: matched evidence is mixed and does not support broad frozen-LM low-data necessity.
- Do not start SPD variants or new P2 modules unless a later paper-specific need selects a concrete failure slice.
