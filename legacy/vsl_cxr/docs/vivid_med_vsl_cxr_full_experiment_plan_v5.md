# VIVID-Med 下一阶段完整实验计划 v5  
## VSL-CXR：Visual Sufficiency Learning for Report-Guided Chest X-ray Representation

> 版本：2026-07-06  
> 目标：在现有 SAMEQ / CCSH / CEQ / HNMB / Curriculum 结果基础上，把项目升级成一套面向 TMI / MIA 的完整医学影像方法，而不是一个单独训练 trick。  
> 核心主线：**Visual Sufficiency Learning** —— 让模型判断一张胸片是否提供了足够视觉证据来支持、反驳或无法判断一个临床陈述。  
> 最终目标：训练一个部署时不依赖 LLM 的胸片视觉编码器，并可选接入轻量 clinical consistency readout。

---

# Active Execution Status

> 接管时间：2026-07-07  
> 当前状态：VSL-CXR v5 已成为项目当前正式实验口径。旧的 SAMEQ-CVCP / CVCP-CCSH / next-stage / case-study / A800 / upload / revision 根级计划已归档到 `History/20260707_vsl_cxr_project_organization/`，不再作为当前实验入口。  
> 当前入口：`docs/README.md`、`task_plan.md`、`findings.md`、`progress.md`、`docs/vsl_cxr_requirement_ledger.md`。

执行规则：

```text
1. 本文档是 VSL-CXR 实验的 source of truth。
2. 所有实验必须按本文档的 support / contradict / uncertain / insufficient、SAMEQ、HNMB、CEQ、CCSH、AUCH、external、teacher comparison 和 locked final comparison 口径执行。
3. 旧实验产物只能在 requirement ledger 中被标记为 exact match 或 bounded evidence 后复用。
4. 稳定运行时按约每两小时检查一次；训练完成、postprocess handoff、partial row、错误、显存异常或进程异常时加密检查。
5. 最终完成必须写回本文档，并在最后一次编辑后重新跑 audit/summary 验证。
```

当前已落地的首个 gate：

| Gate | Artifact | Status |
|---|---|---|
| Readiness audit | `docs/vsl_cxr_readiness_audit.md` / `outputs/final_tables/vsl_cxr_readiness_audit.csv` | 通过；当前 `wrote_rows=67`，`script exact_exists=29`，`missing=1`；D0-D9 source manifest 已生成，D6/D9 VSL schema 均已生成，主 external 仍因标签/manifest 边界阻塞 |
| D6 VSL-4class data | `outputs/instruction_data/vsl_cxr/d6_vsl_4class_train.jsonl` / `d6_vsl_4class_val.jsonl` | 已生成 |
| D6 data quality | `outputs/final_tables/vsl_cxr_data_quality_summary.md` | 结构审计通过：train 11149/11149，val 1600/1600 accepted |
| D6 manual audit | `outputs/final_tables/vsl_cxr_d6_manual_audit_template.csv` | 模板已生成，人工 correctness 尚未填写 |
| VSL trainer smoke | `outputs/qwen3vl_instruction_runs/vsl_cxr_d6_vsl4_debug/metrics_final.json` | 1-step debug smoke 通过；不是正式实验结果 |
| D9 VSL-full data | `outputs/final_tables/vsl_cxr_d9_full_dataset_manifest.md` | 已生成：mixed instruction + CEQ/CCSH companion files |
| D9 data quality | `outputs/final_tables/vsl_cxr_d9_data_quality_summary.md` | 结构审计通过：train 18000/18000，val 2000/2000 accepted |
| D9 trainer smoke | `outputs/qwen3vl_instruction_runs/vsl_cxr_d9_full_debug/metrics_final.json` | 1-step debug smoke 通过；不是正式实验结果 |
| D0-D9 source manifest | `outputs/final_tables/vsl_cxr_data_source_manifest.md` | 已生成：13/13 source artifacts exist，覆盖 Basic-QA、CF-QA、SAMEQ、SAMEQ-CF、SAMEQ-K4、SAMEQ-HNMB、D6/D9/CEQ/CCSH sources |
| VSL sufficiency summary | `outputs/final_tables/vsl_cxr_sufficiency_summary.md` | 已生成：6 个 VSL label/loss formal rows |
| Calibration summary | `outputs/final_tables/vsl_cxr_calibration_summary.md` | 已生成：14 行 ECE/Brier/ECE summary；binned curve points 未导出 |
| B7 formal run | `F:/Xiyao_Wang/021_260129VIVID_vsl_cxr_outputs/qwen3vl/vsl_cxr_d6_vsl4/` | 已完成：`global_step=5000`，`best_val_loss=0.3948386136543704`，final metrics/runtime/final checkpoint/result-table row 已落盘 |
| Phase 2 VSL-2class/3class data | `outputs/final_tables/vsl_cxr_label_variant_manifest.md` | 已从 D6 派生并审计：2class 6000/800 accepted，3class 8149/1200 accepted |
| Phase 2 VSL-2class formal run | `F:/Xiyao_Wang/021_260129VIVID_vsl_cxr_outputs/qwen3vl/vsl_cxr_d6_vsl2/` | 已完成：`global_step=5000`，`best_val_loss=0.04671015161014566`，final metrics/runtime/final checkpoint/result-table row 已落盘 |
| Phase 2 VSL-3class formal run | `F:/Xiyao_Wang/021_260129VIVID_vsl_cxr_outputs/qwen3vl/vsl_cxr_d6_vsl3/` | 已完成：`global_step=5000`，`best_val_loss=0.14087300902791322`，final metrics/runtime/final checkpoint/result-table row 已落盘 |
| Phase 2 VSL-4class-balanced data | `outputs/final_tables/vsl_cxr_d6_vsl4_balanced_data_quality_summary.md` | 已从 D6 四类等量采样并审计：train 8596/8596 accepted，val 1600/1600 accepted |
| Phase 2 VSL-4class-balanced formal run | `F:/Xiyao_Wang/021_260129VIVID_vsl_cxr_outputs/qwen3vl/vsl_cxr_d6_vsl4_balanced/` | 已完成：`global_step=5000`，`best_val_loss=0.4522515568471281`，final metrics/runtime/final checkpoint/result-table row 已落盘 |
| Phase 2 VSL-4class-field-balanced data | `outputs/final_tables/vsl_cxr_d6_vsl4_field_balanced_data_quality_summary.md` | 已从 D6 finding 等量采样并审计：train 5382/5382 accepted，val 767/767 accepted |
| Phase 2 VSL-4class-field-balanced formal run | `F:/Xiyao_Wang/021_260129VIVID_vsl_cxr_outputs/qwen3vl/vsl_cxr_d6_vsl4_field_balanced/` | 已完成：`global_step=5000`，`best_val_loss=0.34942927536666346`，final metrics/runtime/final checkpoint/result-table row 已落盘 |
| Phase 2 VSL-hierarchical implementation | `scripts/train_qwen3vl_clinical_instruction.py` / `configs/qwen3vl_instruction/vsl_cxr/vsl_hierarchical.yaml` | 已实现并 debug smoke 通过：在 answer-token CE 之外加入 answerable、support/contradict、uncertain 三项层级损失 |
| Phase 2 VSL-hierarchical formal run | `F:/Xiyao_Wang/021_260129VIVID_vsl_cxr_outputs/qwen3vl/vsl_cxr_d6_vsl4_hierarchical/` | 已完成：`global_step=5000`，`best_val_loss=0.47355849220942764`，final metrics/runtime/final checkpoint/result-table row 已落盘 |
| Phase 1 B0 Raw-Vision LP readout | `F:/Xiyao_Wang/021_260129VIVID_vsl_cxr_outputs/qwen3vl/vsl_cxr_b0_raw_vision_lp/` | 已完成：原始 Qwen3-VL vision tower 冻结，`vision_checkpoint=null`，只训练 LP readout；`global_step=1000`，macro-AUC `0.6790032275900184`，macro-F1 `0.7323508931634031`，micro-F1 `0.697265625`，final metrics/final_probe/result-table row 已落盘 |
| Phase 1 B1 Basic-QA formal run | `F:/Xiyao_Wang/021_260129VIVID_vsl_cxr_outputs/qwen3vl/vsl_cxr_b1_basic_qa/` | 已完成：`global_step=5000`，`best_val_loss=0.023826396770775318`，final metrics/runtime/final checkpoint/result-table row 已落盘 |
| Phase 1 B2 CF-QA formal run | `F:/Xiyao_Wang/021_260129VIVID_vsl_cxr_outputs/qwen3vl/vsl_cxr_b2_cf_qa/` | 已完成：`global_step=5000`，`best_val_loss=0.12035670908632119`，final metrics/runtime/final checkpoint/result-table row 已落盘 |
| Phase 1 B3 SAMEQ formal run | `F:/Xiyao_Wang/021_260129VIVID_vsl_cxr_outputs/qwen3vl/vsl_cxr_b3_sameq/` | 已完成：`global_step=5000`，`best_val_loss=0.17672864127079244`，final metrics/runtime/final checkpoint/result-table row 已落盘 |
| Phase 1 B4 SAMEQ-CF formal run | `F:/Xiyao_Wang/021_260129VIVID_vsl_cxr_outputs/qwen3vl/vsl_cxr_b4_sameq_cf/` | 已完成：`global_step=5000`，`best_val_loss=0.21318705889705136`，final metrics/runtime/final checkpoint/result-table row 已落盘 |
| Phase 1 B5 SAMEQ-K4 formal run | `F:/Xiyao_Wang/021_260129VIVID_vsl_cxr_outputs/qwen3vl/vsl_cxr_b5_sameq_k4/` | 已完成：`global_step=5000`，`best_val_loss=0.12773469497652912`，final metrics/runtime/final checkpoint/result-table row 已落盘 |
| Phase 1 B6 SAMEQ-HNMB formal run | `F:/Xiyao_Wang/021_260129VIVID_vsl_cxr_outputs/qwen3vl/vsl_cxr_b6_sameq_hnmb/` | 已完成：`global_step=5000`，`best_val_loss=0.09585380152730649`，final metrics/runtime/final checkpoint/result-table row 已落盘 |
| Phase 3 CEQ quantitative runs | `outputs/final_tables/vsl_cxr_ceq_results.md` | 已完成 5/5：basic/diverse/sparse/region/statement；当前最强二分类 AUC 为 CEQ-region `0.8471731089704508`，state accuracy `0.716`，region accuracy `0.654` |
| Phase 4 CCSH/AUCH readout runs | `outputs/final_tables/vsl_cxr_ccsh_results.md`, `outputs/final_tables/vsl_cxr_auch_results.md` | 已完成 Phase 4 readout：CCSH/AUCH+CCSH `9/9` completed，AUCH-only `1/1` completed；当前最佳 binary AUC 为 CCSH-CEQ `0.9059760000000001`，最佳 AUPRC 为 AUCH-CEQ-CCSH `0.9005512099194461`，最佳 ECE 为 AUCH-VSL4 `0.11341642936132851` |
| Phase 5 integrated candidates | `outputs/final_tables/vsl_cxr_phase5_candidate_results.md` | 已生成：4 个 component-completed candidate，`VSL-Full` formal training completed，`VSL-Domain` blocked by external data；`VSL-Full` D9 mixed-instruction best val loss `0.19854170768998938` |
| Phase 6 CheXpert LP readouts | `F:/Xiyao_Wang/021_260129VIVID_vsl_cxr_outputs/qwen3vl/vsl_cxr_p6_lp_{sameq,vsl_core,vsl_ceq_backbone,vsl_full}/` | 已完成 SAMEQ、VSL-Core、VSL-CEQ backbone proxy、VSL-Full；Raw 使用 B0 raw-vision LP。当前 CheXpert LP macro-AUC 最强为 VSL-Full `0.7148588673744163` |
| Phase 6 NIH appendix/stress transfer | `outputs/final_tables/vsl_cxr_external_results.md` | 已完成 Raw、SAMEQ、VSL-Core、VSL-CEQ backbone proxy、VSL-Full 的 NIH-appendix-1k；最佳 NIH macro-AUC 为 SAMEQ `0.5932955434118374`，最佳 NIH macro-AUPRC 为 VSL-Core `0.15464047456578672`；该证据不能替代主 external |
| Phase 6 main external readiness | `docs/vindr_cxr_external_integration.md` / `outputs/final_tables/vsl_cxr_external_results.md` | 2026-07-16 integration in progress：官方 VinDr-CXR 1.0.0 的 18,000 DICOM 与 bbox/image-level labels 已到位；direct 7-label 主协议和 deterministic manifests 已生成；解压/完整性审计和五组 test-3k 推理 pending |
| Phase 7 teacher comparison | `outputs/final_tables/vsl_cxr_teacher_comparison_results.md` / `outputs/final_tables/vsl_cxr_teacher_model_audit.md` | bounded 完成：Qwen3-VL current-main smoke/full 有 v5 证据；InternVL、LLaVA/Mllama、Qwen3.5 text-only、Qwen-Coder rows 因缺 VSL-specific trainer/scaffold adapter 阻塞 |
| Phase 8 casebook / visualization | `outputs/final_tables/vsl_cxr_phase8_casebook.md` / `outputs/final_tables/vsl_cxr_phase8_visualization_manifest.md` | 已生成：33-row casebook 覆盖 9 类必做 casebook，7-row figure manifest 覆盖 Fig 1-Fig 7；casebook 需人工视觉复核，calibration curve 缺 binned curve points |
| Phase 9 locked final comparison | `outputs/final_tables/vsl_cxr_locked_final_comparison.md` | 已生成：8 个 finalist row；Integrated finalist = `VSL-Full`，Teacher finalist = `Qwen3-VL 2B`；所有 row 均为 single-seed，主 external 仍 blocked |
| Formal run result table | `outputs/final_tables/vsl_cxr_formal_run_results.md` | 已生成；当前 33 行，33 个 completed |

---

# 0. 一句话主线

## 0.1 论文唯一主线

```text
Radiology reports contain rich clinical language, but not every report-derived question is visually grounded. 
We formulate report-guided CXR representation learning as Visual Sufficiency Learning:
given an image and a clinical statement, the model learns whether the image provides sufficient evidence to support, contradict, or leave the statement uncertain/insufficient.
```

中文：

```text
胸片报告有丰富的临床语言，但报告里的问题不一定都真正依赖图像。我们把报告监督重新定义为“视觉充分性学习”：给定一张胸片和一个临床陈述，模型需要判断这张图是否提供了足够视觉证据来支持、反驳或无法判断该陈述。
```

---

## 0.2 方法名

推荐方法名：

```text
VSL-CXR
Visual Sufficiency Learning for Chest X-rays
```

完整写法：

```text
VSL-CXR: Visual Sufficiency Learning with VLM Teachers for Deployable Chest X-ray Encoders
```

---

## 0.3 最终方法由三层组成

```text
Layer 1: Visual Sufficiency Data Engine
         把报告变成可由图像验证的 clinical statements，并构造 SAMEQ / hard negatives / insufficient cases。

Layer 2: Evidence-Aware Vision Encoder
         用 VLM teacher 冻结语言 decoder，训练 vision tower；可加入 CEQ/HNMB 等模块。

Layer 3: Deployable Clinical Consistency Readout
         训练后丢掉 LLM，用 CCSH/AUCH 等轻量模块读出图像-陈述一致性、uncertainty 和 calibration。
```

---

# 1. 为什么要升级到 VSL-CXR？

## 1.1 仅靠 SAMEQ-CVCP 还不够

SAMEQ 是很好的核心训练信号：

```text
same question + different images + different answers
```

它能逼模型看图，但如果论文只讲 SAMEQ，会像一个 clever training design，还不够 TMI / MIA。

VSL-CXR 把 SAMEQ 放进更大的医学问题中：

```text
一个临床陈述到底能不能由这张胸片支持？
```

这样就自然包含：

- report-derived statement extraction；
- counterfactual statement generation；
- image-specific SAMEQ；
- hard negative mining；
- visual sufficiency labels；
- evidence-aware vision encoder；
- deployable consistency head；
- uncertainty / answerability / calibration；
- external validation；
- case study and error taxonomy。

---

## 1.2 VSL-CXR 和普通 QA 的区别

普通 QA：

```text
Q: Is there pleural effusion?
A: Yes.
```

问题：

- 可能靠先验；
- 可能靠模板；
- 可能不看图；
- 不能表达 insufficient / uncertain；
- 部署时不一定有可解释读出。

VSL-CXR：

```text
Statement: There is a left pleural effusion.
Image: chest X-ray.
Question: Does the image provide sufficient evidence to support this statement?
Answer: support / contradict / uncertain / insufficient.
```

优点：

- 直接建模 image-statement consistency；
- 可构造反事实；
- 可做 same-question image-specific training；
- 可部署为轻量 consistency head；
- 更符合临床判断逻辑。

---

# 2. 总体实验分期

| Phase | 名称 | 目的 |
|---|---|---|
| Phase 0 | 数据与审计准备 | 生成 clinical statements / SAMEQ / hard negatives / sufficiency labels，并严格过滤 leakage 和 false negatives |
| Phase 1 | Baseline 与主干确认 | 对比 Basic QA / CF / SAMEQ / SAMEQ-K / SAMEQ-HNMB，证明 VSL data engine 的价值 |
| Phase 2 | Visual Sufficiency Data Engine | 系统测试 support / contradict / uncertain / insufficient 四类监督 |
| Phase 3 | Evidence-Aware Encoder | 测 CEQ、HNMB、CEQ+HNMB，验证是否超过 global embedding |
| Phase 4 | Deployable Readout | 测 CCSH、AUCH、CCSH+AUCH，证明部署时不用 LLM |
| Phase 5 | Integrated VSL-CXR Candidates | 组合 VSL-Core / VSL-CEQ / VSL-HNMB / VSL-Full |
| Phase 6 | External Validation | 选择一个主 external 数据集，替代 NIH 主验证 |
| Phase 7 | Model Teacher Comparison | Qwen3-VL vs InternVL vs LLaVA/Llama-based VLM vs text-only scaffold |
| Phase 8 | Case Study and Visualization | 做 qualitative proof、false negative audit、CEQ attention、consistency examples |
| Phase 9 | Locked Final Comparison | 每个 family 只选一个 finalist，多 seed + external + calibration + cost |

---

# 3. Phase 0：数据生成与审计

## 3.1 输入数据

每个样本应包含：

| Field | Description |
|---|---|
| sample_id | 唯一 ID |
| image_path | CXR 图像路径 |
| report_text | 原始报告 |
| findings_section | 报告 findings |
| impression_section | 报告 impression |
| existing_labels | CheXpert/MIMIC/UMS 标签 |
| split | train/val/test |
| patient_id | 防止 patient leakage |
| view_position | AP/PA/lateral if available |

---

## 3.2 Visual Sufficiency Data Engine 输出

每条训练样本建议统一为 JSONL：

```json
{
  "sample_id": "xxx",
  "image_path": "xxx.jpg",
  "statement": "There is a left pleural effusion.",
  "question": "Does the chest X-ray provide sufficient evidence to support this statement?",
  "answer": "support",
  "answer_type": "support|contradict|uncertain|insufficient|choice",
  "finding": "pleural_effusion",
  "state": "present",
  "laterality": "left",
  "severity": "small",
  "evidence_span": "small left pleural effusion",
  "counterfactual_statement": "There is a right pleural effusion.",
  "sufficiency_label": "support",
  "visual_dependency": "high",
  "negative_image_path": "xxx_negative.jpg",
  "negative_type": "same_question_opposite_laterality",
  "source": "report_derived",
  "generation_model": "GLM/API",
  "validation_status": "validated"
}
```

---

## 3.3 四类视觉充分性标签

| Label | 中文 | 含义 | 示例 |
|---|---|---|---|
| support | 支持 | 图像足以支持该陈述 | left effusion statement + left effusion image |
| contradict | 反驳 | 图像证据与陈述冲突 | right effusion statement + left effusion image |
| uncertain | 不确定 | 陈述或图像证据不确定 | possible edema |
| insufficient | 视觉证据不足 | 报告未提及或图像无法可靠判断 | unmentioned fracture |

---

## 3.4 Statement 类型

| Statement type | Example | Visual dependency |
|---|---|---|
| finding-present | There is a pneumothorax. | high |
| finding-absent | There is no pneumothorax. | medium/high |
| laterality | There is a left pleural effusion. | high |
| severity | There is a small pleural effusion. | medium/high |
| uncertainty | There is possible edema. | medium |
| answerability | Fracture is not visually answerable from this study. | medium |
| support-device | A right PICC terminates in the SVC. | high |
| anatomy/location | There is right basilar opacity. | high |

---

## 3.5 数据版本

| Dataset ID | Name | Description | Priority |
|---|---|---|---|
| D0 | Basic-QA | 普通 report QA | baseline |
| D1 | CF-QA | 反事实 A/B 选择 | baseline |
| D2 | SAMEQ | same question / different image / different answer | core |
| D3 | SAMEQ-CF | SAMEQ + CF-compatible rows | high |
| D4 | SAMEQ-K | SAMEQ + K hard negative images | high |
| D5 | SAMEQ-HNMB | SAMEQ + memory-mined hard negatives | high |
| D6 | VSL-4class | support/contradict/uncertain/insufficient | core |
| D7 | VSL-CEQ | CEQ evidence query labels | module |
| D8 | VSL-CCSH | image-statement consistency pairs | module |
| D9 | VSL-full | SAMEQ + K + 4class + CEQ + CCSH | upper |

---

## 3.6 自动审计

| Check | Reject / Flag |
|---|---|
| statement not supported by report | reject |
| evidence_span not found in report when required | flag/reject |
| unmentioned finding converted to absent | reject |
| question leaks answer | reject |
| A/B answer imbalance | rebalance |
| option length imbalance | flag |
| hard negative same label/answer | reject |
| false hard negative suspected | manual audit |
| laterality generated without report support | reject |
| severity generated without explicit phrase | reject |
| uncertain converted to definite | reject |
| duplicate statement overrepresented | downsample |

---

## 3.7 数据质量表

| Dataset | N images | N statements | N pairs | QA/img | support % | contradict % | uncertain % | insufficient % | leakage % | false-neg % | accepted? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D0 Basic-QA |  |  |  |  |  |  |  |  |  |  |  |
| D1 CF-QA |  |  |  |  |  |  |  |  |  |  |  |
| D2 SAMEQ |  |  |  |  |  |  |  |  |  |  |  |
| D3 SAMEQ-CF |  |  |  |  |  |  |  |  |  |  |  |
| D4 SAMEQ-K |  |  |  |  |  |  |  |  |  |  |  |
| D5 SAMEQ-HNMB |  |  |  |  |  |  |  |  |  |  |  |
| D6 VSL-4class |  |  |  |  |  |  |  |  |  |  |  |

---

## 3.8 Manual audit 表

每个数据版本至少人工抽样 200 条。

| audit_id | dataset | sample_id | statement | answer | image | negative_image | evidence_span | finding | leakage? | correct? | hard_neg_valid? | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |  |  | yes/no | yes/no | yes/no |  |
| 2 |  |  |  |  |  |  |  |  | yes/no | yes/no | yes/no |  |

通过标准：

```text
manual correctness >= 90%
leakage <= 5%-10%
false hard negative <= 5%-8%
A/B balance 45%-55%
support/contradict balance not overly skewed
```

---

# 4. Phase 1：Baseline 与主干确认

## 4.1 目的

证明 VSL-CXR 的数据引擎比普通 QA / CF / fixed prompt 更有用。

---

## 4.2 必跑 baseline

| Run ID | Data | Model | Train policy | Purpose |
|---|---|---|---|---|
| B0 Raw-Vision | none | Qwen3-VL vision tower | no training | raw baseline |
| B1 Basic-QA | D0 | Qwen3-VL | freeze LLM, train vision+connector | ordinary QA baseline |
| B2 CF-QA | D1 | Qwen3-VL | same | counterfactual baseline |
| B3 SAMEQ | D2 | Qwen3-VL | same | core signal |
| B4 SAMEQ-CF | D3 | Qwen3-VL | same | sameq + CF gate |
| B5 SAMEQ-K4 | D4 | Qwen3-VL | same | multi-negative |
| B6 SAMEQ-HNMB | D5 | Qwen3-VL | same | mined negative |
| B7 VSL-4class | D6 | Qwen3-VL | same | visual sufficiency labels |

---

## 4.3 Baseline result table

| Run | CheXpert AUC | External AUC | Hard shuffle delta | SAMEQ acc | CF acc | VSL-4class acc | AUPRC | ECE | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B0 Raw-Vision | 0.679003 |  |  |  |  |  |  |  | raw Qwen3-VL vision tower + LP readout completed; no vision/VSL training |
| B1 Basic-QA |  |  |  |  |  |  |  |  |  |
| B2 CF-QA |  |  |  |  |  |  |  |  |  |
| B3 SAMEQ |  |  |  |  |  |  |  |  |  |
| B4 SAMEQ-CF |  |  |  |  |  |  |  |  |  |
| B5 SAMEQ-K4 |  |  |  |  |  |  |  |  |  |
| B6 SAMEQ-HNMB |  |  |  |  |  |  |  |  |  |
| B7 VSL-4class |  |  |  |  |  |  |  |  |  |

---

## 4.4 决策规则

| 结果 | 解释 |
|---|---|
| SAMEQ > Basic-QA | 同题换图确实比普通 QA 强 |
| SAMEQ-K4 > SAMEQ | 多负样本有用 |
| SAMEQ-HNMB > SAMEQ-K4 | 动态 hard negative 有用 |
| VSL-4class > SAMEQ | 视觉充分性标签有额外价值 |
| CF-QA 高但 hard shuffle 低 | CF 只学陈述真假，不够 image-specific |
| Basic-QA 接近 SAMEQ | 需要检查 SAMEQ 数据是否真的 hard |

---

# 5. Phase 2：Visual Sufficiency Data Engine

## 5.1 目的

证明新主线不是 SAMEQ 一个 trick，而是完整的视觉充分性监督体系。

---

## 5.2 VSL 标签版本

| Run ID | Label set | Description |
|---|---|---|
| VSL-2class | support vs contradict | 最简单 |
| VSL-3class | support / contradict / uncertain | 加不确定 |
| VSL-4class | support / contradict / uncertain / insufficient | 完整 |
| VSL-4class-balanced | class-balanced sampling | 防止 support dominates |
| VSL-4class-field-balanced | finding-balanced | 防止 frequent labels dominate |

---

## 5.3 VSL loss 设计

### Option 1: CE classification

```text
L = CE(support/contradict/uncertain/insufficient)
```

### Option 2: hierarchical loss

```text
L = BCE(answerable)
  + CE(support/contradict | answerable)
  + BCE(uncertain)
```

### Option 3: margin loss

```text
support score(image, true statement) > support score(image, counterfactual statement)
```

---

## 5.4 VSL result table

| Run | Label set | Loss | CheXpert AUC | External AUC | VSL acc | Support AUC | Contradict AUC | Uncertain F1 | Insufficient F1 | ECE | Decision |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| VSL-2class | support/contradict | CE |  |  |  |  |  |  |  |  |  |
| VSL-3class | +uncertain | CE |  |  |  |  |  |  |  |  |  |
| VSL-4class | full | CE |  |  |  |  |  |  |  |  |  |
| VSL-4class-balanced | full | class-balanced |  |  |  |  |  |  |  |  |  |
| VSL-hierarchical | full | hierarchical |  |  |  |  |  |  |  |  |  |

---

# 6. Phase 3：Evidence-Aware Vision Encoder

## 6.1 模块 A：CEQ

CEQ = Clinical Evidence Query。

每个 finding 一个 query：

```text
query_effusion
query_pneumothorax
query_cardiomegaly
...
```

它去 image patch tokens 里找 evidence。

---

## 6.2 CEQ 版本

| Run | Module | Description |
|---|---|---|
| CEQ-basic | finding queries | basic |
| CEQ-diverse | + query diversity loss | 防止 query collapse |
| CEQ-sparse | + attention sparsity | 更可解释 |
| CEQ-region | + laterality/location weak labels | 如果有 location |
| CEQ-statement | statement-conditioned query | 根据 statement 动态 query |

---

## 6.3 CEQ loss

| Loss | Purpose |
|---|---|
| finding state loss | finding-specific learning |
| VSL consistency loss | support/contradict |
| query diversity loss | 防止所有 query 看同一区域 |
| attention entropy/sparsity | 可解释 |
| laterality weak supervision | left/right consistency |

---

## 6.4 CEQ result table

| Run | Base data | Module | CheXpert AUC | External AUC | Hard shuffle | VSL acc | Attention quality | Per-field AUC | Decision |
|---|---|---|---:|---:|---:|---:|---|---:|---|
| CEQ-basic | SAMEQ | CEQ | 0.768056 |  |  | 0.512 | quantitative pending | 0.656821 | completed |
| CEQ-diverse | SAMEQ | CEQ+div | 0.690942 |  |  | 0.540 | quantitative pending | 0.514169 | completed |
| CEQ-sparse | SAMEQ | CEQ+sparse | 0.715837 |  |  | 0.646 | quantitative pending | 0.663323 | completed |
| CEQ-region | SAMEQ | CEQ+region | 0.847173 |  |  | 0.716 | region acc 0.654 | 0.801449 | best current CEQ variant |
| CEQ-statement | VSL | statement query | 0.720522 |  |  | 0.702 | quantitative pending | 0.716206 | completed on VSL-4class backbone |

---

## 6.5 CEQ 可视化表

| Sample | Finding | Expected region | CEQ attention correct? | Comment |
|---|---|---|---|---|
|  | Pleural Effusion | costophrenic angle |  |  |
|  | Pneumothorax | pleural line/apex |  |  |
|  | Cardiomegaly | cardiac silhouette |  |  |
|  | Edema | bilateral lungs |  |  |
|  | Support Device | catheter/tube |  |  |

---

# 7. Phase 4：Deployable Readout

## 7.1 模块 B：CCSH

CCSH = Clinical Consistency Scoring Head。

输入：

```text
image embedding + clinical statement embedding
```

输出：

```text
support / contradict / uncertain / insufficient
```

---

## 7.2 CCSH 必跑实验

| Run | Backbone | Readout | Purpose |
|---|---|---|---|
| CCSH-Raw | raw Qwen3-VL | CCSH | head-only control |
| CCSH-SAMEQ | SAMEQ backbone | CCSH | 主桥接 |
| CCSH-SAMEQ-K4 | SAMEQ-K4 backbone | CCSH | hard negative readout |
| CCSH-HNMB | SAMEQ-HNMB backbone | CCSH | mined negative readout |
| CCSH-CEQ | CEQ backbone | CCSH | evidence-aware readout |
| CCSH-VSL4 | VSL-4class backbone | CCSH | sufficiency label readout |

---

## 7.3 CCSH result table

| Run | Backbone | Binary AUC | 4-class acc | AUPRC | F1 | ECE | CheXpert AUC | External AUC | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| CCSH-Raw | raw | 0.881560 |  | 0.850778 | 0.828070 | 0.156402 |  |  | completed no-leak |
| CCSH-SAMEQ | SAMEQ | 0.895176 |  | 0.878471 | 0.848592 | 0.109215 |  |  | completed no-leak |
| CCSH-SAMEQ-K4 | K4 | 0.788360 |  | 0.712013 | 0.719665 | 0.186511 |  |  | completed |
| CCSH-HNMB | HNMB | 0.835360 |  | 0.794997 | 0.797048 | 0.199638 |  |  | completed |
| CCSH-CEQ | CEQ | 0.905976 |  | 0.885995 | 0.842105 | 0.119305 |  |  | best current CCSH row |
| CCSH-VSL4 | VSL4 | 0.880824 |  | 0.860066 | 0.784091 | 0.147463 |  |  | completed |

---

## 7.4 CCSH 决策规则

| Result | Interpretation |
|---|---|
| CCSH-SAMEQ > CCSH-Raw | SAMEQ training improves deployable readout |
| CCSH-CEQ > CCSH-SAMEQ | evidence-aware encoder helps |
| CCSH-HNMB > CCSH-SAMEQ-K4 | mined hard negatives help |
| CCSH-VSL4 best | visual sufficiency labels most useful |
| CCSH high but LP low | readout module useful, backbone not necessarily best |
| CCSH not better than raw | head itself drives result, weakens method claim |

---

## 7.5 模块 C：AUCH

AUCH = Answerability-Uncertainty Calibration Head。

作用：

```text
预测 answerability / uncertainty / insufficient。
```

| Run | Backbone | AUCH? | CCSH? |
|---|---|---|---|
| AUCH-SAMEQ | SAMEQ | yes | no |
| AUCH-CCSH-SAMEQ | SAMEQ | yes | yes |
| AUCH-CEQ-CCSH | CEQ | yes | yes |
| AUCH-VSL4 | VSL-4class | yes | yes |

### AUCH table

| Run | Answerability AUC | Uncertainty F1 | Insufficient F1 | ECE | Brier | CheXpert AUC | External AUC | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| AUCH-SAMEQ | 0.547668 | 0.000000 |  | 0.062070 |  |  |  | completed AUCH-only; answerability AUPRC 0.928481, uncertainty AUC 0.546642 |
| AUCH-CCSH-SAMEQ |  |  |  | 0.163020 |  |  |  | completed as AUCH+CCSH readout row: binary AUC 0.900720, AUPRC 0.893840 |
| AUCH-CEQ-CCSH |  |  |  | 0.133392 |  |  |  | completed as AUCH+CCSH readout row: binary AUC 0.888968, AUPRC 0.900551 |
| AUCH-VSL4 |  |  |  | 0.113416 |  |  |  | completed as AUCH+CCSH readout row: binary AUC 0.895976, AUPRC 0.870734 |

---

# 8. Phase 5：Integrated VSL-CXR Candidates

## 8.1 候选方法

| Candidate | Data Engine | Encoder Module | Readout | Description |
|---|---|---|---|---|
| VSL-Lite | SAMEQ | global | none/LP | 最简单主干 |
| VSL-Core | SAMEQ-K4 | global | CCSH | 推荐起点 |
| VSL-HNMB | SAMEQ-HNMB | global | CCSH | hard-negative mining |
| VSL-CEQ | SAMEQ | CEQ | CCSH | TMI 可解释 |
| VSL-Full | SAMEQ-HNMB + VSL-4class | CEQ | CCSH+AUCH | 最完整 |
| VSL-Domain | VSL-Core | optional DRA | CCSH | 外部增强 |

---

## 8.2 Integrated result table

| Candidate | CheXpert AUC | External AUC | VSL acc | CCSH AUC | Hard shuffle | AUPRC | ECE | Interpretability | Cost | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| VSL-Lite |  |  |  |  |  |  |  | low | 3638.210253 | component completed via SAMEQ global encoder; LP/CheXpert readout not yet available |
| VSL-Core |  |  |  | 0.788360 |  | 0.712013 | 0.186511 | medium | 4776.907956 | component completed via SAMEQ-K4 + CCSH |
| VSL-HNMB |  |  |  | 0.835360 |  | 0.794997 | 0.199638 | medium | 4992.569633 | component completed via SAMEQ-HNMB + CCSH |
| VSL-CEQ |  |  |  | 0.905976 |  | 0.885995 | 0.119305 | high | 3638.210253 | strongest current Phase 5 component candidate by CCSH binary AUC |
| VSL-Full |  |  |  | 0.888968 |  | 0.900551 | 0.133392 | high | 6891.429297 | D9 mixed-instruction formal training completed; best val loss 0.19854170768998938 |
| VSL-Domain |  |  |  | 0.788360 |  | 0.712013 | 0.186511 | medium |  | blocked: external dataset and label-manifest eligibility missing |

---

## 8.3 如何选择最终方法

| 情况 | 最终选择 |
|---|---|
| VSL-Lite 已经最强 | 主方法简化，模块放 appendix |
| VSL-Core 明显优于 VSL-Lite | 主打 SAMEQ-K + CCSH |
| VSL-HNMB 最强 | 主打 hard negative memory bank |
| VSL-CEQ 性能接近且解释性强 | 主打 TMI 架构 |
| VSL-Full 最强但复杂 | 要看 cost，否则谨慎 |
| VSL-Domain 提升 external | 加入 domain module |

---

# 9. Phase 6：External Validation

## 9.1 主外部数据集选择

| Dataset | Role | Condition |
|---|---|---|
| VinDr-CXR / VinBigData | preferred | label/bbox CSV available |
| PadChest | backup | label mapping reliable |
| MIMIC-CXR | conditional | only if not used in training |
| NIH | appendix/stress | not main |

---

## 9.2 External validation protocol

必须包括：

- label mapping audit；
- macro-AUC；
- macro-AUPRC；
- ECE/Brier；
- per-label analysis；
- failure case study；
- domain shift embedding visualization。

---

## 9.3 External table

当前 Phase 6 执行状态（2026-07-07）：

- 主 external 数据边界已于 2026-07-16 更新：官方 VinDr-CXR 1.0.0 的 15,000 train + 3,000 test DICOM、bbox 和 image-level labels 已到位；直接七标签映射/manifests 已生成，解压/完整性审计与五组 test-3k 正式推理正在执行。在结果落盘前，主 external 性能结论仍保持 pending。
- NIH 只作为 appendix/stress，不作为主 external claim。
- NIH-appendix-1k 已完成 Raw、SAMEQ、VSL-Core、VSL-CEQ backbone proxy、VSL-Full；完整表见 `outputs/final_tables/vsl_cxr_external_results.md`。

| Run | External dataset | Macro-AUC | Macro-AUPRC | ECE | Brier | Best labels | Worst labels | Failure cause |
|---|---|---:|---:|---:|---:|---|---|---|
| Raw | NIH-appendix-1k | 0.5737087050807025 | 0.14895860751532564 | 0.6914112159833312 | 0.6918081035837531 | Edema:0.707179; Pneumothorax:0.643168; Consolidation:0.632786 | Atelectasis:0.449987; Pneumonia:0.455926; No Finding:0.500000 | NIH is appendix/stress only; not main external |
| SAMEQ | NIH-appendix-1k | 0.5932955434118374 | 0.1486527095007633 | 0.5451605868618935 | 0.5306810569018126 | Edema:0.752856; Consolidation:0.674612; Pleural Effusion:0.639447 | Pneumothorax:0.493612; No Finding:0.500000; Atelectasis:0.508952 | NIH is appendix/stress only; not main external |
| VSL-Core | NIH-appendix-1k | 0.5872269642158319 | 0.15464047456578672 | 0.6556109379571862 | 0.6266131419688463 | Edema:0.777160; Consolidation:0.654936; Pleural Effusion:0.625496 | Atelectasis:0.482759; No Finding:0.500000; Pneumonia:0.502173 | NIH is appendix/stress only; not main external |
| VSL-CEQ | NIH-appendix-1k | 0.5742891859006891 | 0.147347753397811 | 0.7452516441009939 | 0.7273509986698627 | Edema:0.767057; Pleural Effusion:0.635760; Cardiomegaly:0.602987 | Atelectasis:0.478083; No Finding:0.500000; Pneumonia:0.500000 | CEQ readout is not a CheXpert classifier; row evaluates SAMEQ visual backbone proxy |
| VSL-Full | NIH-appendix-1k | 0.5815168249418435 | 0.14187687475042848 | 0.7153810005113482 | 0.6907003438100219 | Edema:0.716763; Consolidation:0.639325; Pleural Effusion:0.621760 | No Finding:0.500000; Atelectasis:0.500000; Pneumothorax:0.520284 | NIH is appendix/stress only; not main external |

---

## 9.4 External 决策规则

| Result | Interpretation |
|---|---|
| external improves with VSL-Core | core method generalizes |
| CEQ improves external | evidence-aware queries improve transfer |
| DRA needed | domain shift significant |
| no method improves external | report-derived representation may be in-domain only; write as limitation |
| per-label improves only in exact mappings | label mapping is limiting factor |

---

# 10. Phase 7：VLM Teacher 对比

## 10.1 对比模型

| Model | Type | Role |
|---|---|---|
| Qwen3-VL | VLM | current main |
| InternVL | VLM | strong comparator |
| LLaVA/Llama-based VLM | VLM | model family comparator |
| medical VLM if available | VLM | domain-specific |
| Qwen3.5 text-only | text scaffold | non-VLM control |
| Qwen-Coder | text scaffold | template bias control |

---

## 10.2 Smoke test

| Run | Steps | Goal |
|---|---:|---|
| Qwen3VL-VSL-smoke | 500 | confirm |
| InternVL-VSL-smoke | 500 | adapter |
| LLaVA-VSL-smoke | 500 | adapter |
| Qwen3.5-scaffold-smoke | 500 | negative control |

---

## 10.3 Full model comparison

| Run | Model | Data | Steps | CheXpert AUC | External AUC | VSL acc | Hard shuffle | CCSH AUC | Cost | Decision |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3VL-VSL | Qwen3-VL 2B | VSL-Core | 5000 | 0.6985809632677275 | 0.5872269642158319 |  |  | 0.7883600000000001 | 4776.907955646515 | completed_current_main_only；current-main evidence complete, cross-family comparison bounded |
| InternVL-VSL | InternVL | VSL-Core |  |  |  |  |  |  |  | blocked_until_smoke_adapter；requires InternVL-specific VSL trainer and smoke pass |
| LLaVA-VSL | LLaVA/Mllama | VSL-Core |  |  |  |  |  |  |  | blocked_until_smoke_adapter；requires Llama-vision VSL trainer and smoke pass |
| Qwen3.5-scaffold | text-only | VSL-Core |  |  |  |  |  |  |  | blocked_text_scaffold_trainer_missing；requires exact non-vision VSL scaffold trainer |
| Qwen-Coder-scaffold | text-only | VSL-Core |  |  |  |  |  |  |  | blocked_text_scaffold_trainer_missing；historical scripts are not exact v5 evidence |

Phase 7 smoke status（2026-07-07）：

| Run | Status | Evidence / blocker |
|---|---|---|
| Qwen3VL-VSL-smoke | completed_by_current_main | Completed `VSL-CXR-B5-SAMEQ-K4` formal run already exceeds 500-step smoke requirement |
| InternVL-VSL-smoke | blocked_adapter_missing | Local InternVL exists but current trainer is Qwen3-VL specific; processor audit also reports missing `start_image_token` |
| LLaVA-VSL-smoke | blocked_adapter_missing | Local Mllama exists and processor loads, but no Llama-vision VSL trainer adapter exists |
| Qwen3.5-scaffold-smoke | blocked_text_scaffold_trainer_missing | Text-only local model exists, but exact v5 scaffold trainer is missing |

---

## 10.4 解释规则

| Result | Claim |
|---|---|
| all VLMs > text-only scaffold | VLM-coupled teacher matters |
| Qwen3-VL only wins | current instantiation is Qwen-specific |
| text-only close to VLM | data engine is main contribution |
| larger VLM worse | stronger decoder may reduce visual pressure |
| medical VLM best | domain-specific teacher helps |

---

# 11. Phase 8：Case Study and Visualization

## 11.1 必做 casebook

| Casebook | Purpose |
|---|---|
| VSL support cases | 图像支持陈述 |
| VSL contradict cases | 图像反驳陈述 |
| uncertain cases | 不确定 |
| insufficient cases | 视觉证据不足 |
| SAMEQ pair cases | 同题换图 |
| false hard negatives | hard negative 质量 |
| CCSH success/failure | readout 解释 |
| CEQ attention | 证据区域 |
| external failures | 外部数据问题 |

---

## 11.2 Case study template

| Case ID | Dataset | Image | Statement | True label | Model output | Explanation | Failure type | Manual note |
|---|---|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |  |  |

---

## 11.3 图像展示建议

| Figure | Description |
|---|---|
| Fig 1 | VSL-CXR framework |
| Fig 2 | SAMEQ examples |
| Fig 3 | support vs contradict examples |
| Fig 4 | CEQ attention maps |
| Fig 5 | CCSH consistency readout |
| Fig 6 | external failure examples |
| Fig 7 | calibration curves |

Phase 8 当前执行状态（2026-07-07）：

| Artifact | Status | Evidence |
|---|---|---|
| VSL support / contradict / uncertain / insufficient cases | completed_needs_manual_review | `outputs/final_tables/vsl_cxr_phase8_casebook.md` |
| SAMEQ pair cases | completed_needs_manual_review | `outputs/final_tables/vsl_cxr_phase8_casebook.md` |
| false hard negatives | completed_needs_manual_review | `outputs/final_tables/vsl_cxr_phase8_casebook.md`; current val rows lack `negative_image_path`, so fallback rows require manual counterfactual-validity review |
| CCSH success/failure | completed_needs_manual_review | `outputs/final_tables/vsl_cxr_phase8_casebook.md` plus `outputs/final_tables/vsl_cxr_ccsh_results.md` |
| CEQ attention | completed_needs_manual_review | `outputs/final_tables/vsl_cxr_phase8_casebook.md`; attention assets indexed by `outputs/attention_maps/summary.json` |
| external failures | completed_appendix_only | NIH appendix/stress failures in `outputs/final_tables/vsl_cxr_phase8_casebook.md`; main external remains blocked |
| Fig 1-Fig 7 manifest | completed | `outputs/final_tables/vsl_cxr_phase8_visualization_manifest.md` |
| calibration curves | bounded_metric_summary_only | ECE/Brier are in `outputs/final_tables/vsl_cxr_external_results.md`; binned curve points are not exported |

---

# 12. Phase 9：Locked Final Comparison

## 12.1 每个 family 只选一个 finalist

| Family | Candidates | Finalist |
|---|---|---|
| Basic QA | Basic-QA / CF-QA | Basic-QA |
| SAMEQ | SAMEQ / SAMEQ-CF | SAMEQ |
| Hard Negative | SAMEQ-K / SAMEQ-HNMB | SAMEQ-HNMB |
| Evidence Encoder | CEQ variants | CEQ-region |
| Readout | CCSH / CCSH+AUCH | CCSH-CEQ |
| Integrated VSL | VSL-Lite/Core/HNMB/CEQ/Full | VSL-Full |
| Teacher model | Qwen/InternVL/LLaVA/text | Qwen3-VL 2B |

---

## 12.2 Locked metrics

| Metric | Role |
|---|---|
| CheXpert macro-AUC | primary |
| main external macro-AUC | primary external |
| VSL support/contradict AUC | primary sufficiency |
| hard-shuffle delta | primary grounding |
| CCSH binary AUC | primary readout |
| macro-AUPRC | secondary |
| ECE/Brier | calibration |
| false hard negative rate | data validity |
| leakage % | safety gate |
| cost | feasibility |

---

## 12.3 Locked final table

| Family | Finalist | Seeds | CheXpert AUC mean±std | External AUC mean±std | VSL AUC | Hard shuffle | CCSH AUC | ECE | Cost | Final role |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| Raw | Raw Qwen3-VL vision LP | 1 | 0.6790032275900184 single-seed | main external blocked; NIH appendix 0.5737087050807025 |  |  |  | 0.6914112159833312 | 317.5859773159027 | baseline |
| QA | Basic-QA | 1 |  | not evaluated |  |  |  |  | 2848.706378221512 | baseline |
| SAMEQ | SAMEQ | 1 | 0.6961128245416363 single-seed | main external blocked; NIH appendix 0.5932955434118374 |  |  | 0.895176 | 0.5451605868618935 | 3638.2102530002594 | core |
| HardNeg | SAMEQ-HNMB | 1 |  | main external blocked |  |  | 0.8353600000000001 | 0.19963750052824616 | 4992.569633245468 | hard-negative |
| CEQ | CEQ-region | 1 |  | not directly evaluated | 0.8471731089704508 |  | 0.9059760000000001 | 0.08269000466053303 |  | evidence |
| CCSH | CCSH-CEQ | 1 |  | not directly evaluated |  |  | 0.9059760000000001 | 0.11930484469234945 |  | readout |
| VSL Integrated | VSL-Full | 1 | 0.7148588673744163 single-seed | main external blocked; NIH appendix 0.5815168249418435 |  |  | 0.8889679999999999 | 0.7153810005113482 | 6891.429297447205 | final |

Phase 9 final decision（2026-07-07）：

- Locked integrated finalist: `VSL-Full`.
- Locked teacher finalist: `Qwen3-VL 2B`.
- Current evidence boundary: all locked rows are single-seed; no row has v5 multi-seed mean±std.
- External boundary: main external data and labels are now available, but the locked table still predates the pending VinDr test-3k inference rows; NIH remains appendix/stress only until those rows are completed and audited.
- Interpretation boundary: `VSL-Full` wins the current integrated slot by CheXpert LP and full-stack training completion, while SAMEQ/Core remain stronger on NIH appendix macro-AUC.

---

# 13. 优先执行队列

## 13.1 第一批：验证主方法成立

| Priority | Run | Why |
|---:|---|---|
| 1 | SAMEQ-full + CCSH | 桥接最强 SAMEQ 与 readout |
| 2 | SAMEQ-K4 + CCSH | hard-negative + readout |
| 3 | SAMEQ-HNMB + CCSH | mined negative + readout |
| 4 | VSL-4class | support/contradict/uncertain/insufficient |
| 5 | Raw + CCSH | 检查 CCSH head-only |
| 6 | SAMEQ + CEQ + CCSH | 可解释架构 |
| 7 | 外部数据 manifest audit | 解决 external |
| 8 | SAMEQ-full seed3/seed5 | 稳定性 |

---

## 13.2 第二批：增强方法性

| Priority | Run | Why |
|---:|---|---|
| 1 | CEQ-diverse + CCSH | 防 query collapse |
| 2 | HNMB-online + CCSH | dynamic hard negatives |
| 3 | VSL-hierarchical loss | 更符合 support/answerability |
| 4 | CCSH+AUCH | uncertainty/calibration |
| 5 | VSL-Full | 完整方法上限 |
| 6 | DRA on external | 如果外部差 |

---

## 13.3 第三批：模型泛化

| Priority | Run | Why |
|---:|---|---|
| 1 | InternVL-VSL-smoke | bounded：local model exists, but exact v5 row requires an InternVL-specific VSL trainer adapter |
| 2 | LLaVA-VSL-smoke | bounded：local Mllama model exists, but exact v5 row requires a Llama-vision VSL trainer adapter |
| 3 | Qwen3.5 text scaffold | bounded：non-VLM control requires an exact text-only VSL scaffold trainer |
| 4 | InternVL-VSL-full | blocked until smoke adapter exists and passes |
| 5 | LLaVA-VSL-full | blocked until smoke adapter exists and passes |

---

# 14. 各种情况的决策树

## 14.1 如果 SAMEQ+CCSH 很强

最终方法：

```text
VSL-CXR-Core = SAMEQ Data Engine + CCSH Readout
```

写法：

```text
Same-question visual exams provide the strongest image-specific supervision, and CCSH makes the learned consistency deployable.
```

---

## 14.2 如果 SAMEQ-K4+CCSH 更强

最终方法：

```text
VSL-CXR-HardNeg = SAMEQ + multi-negative + CCSH
```

写法：

```text
Multi-negative visual sufficiency exams improve discrimination among clinically similar images.
```

---

## 14.3 如果 HNMB+CCSH 更强

最终方法：

```text
VSL-CXR-HNMB = visual sufficiency with mined hard negatives
```

写法：

```text
Model-mined hard negatives provide a stronger clinical visual curriculum than static negatives.
```

---

## 14.4 如果 CEQ+CCSH 性能接近但解释性强

最终方法：

```text
VSL-CXR-CEQ = evidence-aware visual sufficiency learning
```

写法：

```text
Clinical evidence queries provide finding-specific evidence and interpretable consistency scoring.
```

---

## 14.5 如果 VSL-4class 最强

最终方法：

```text
VSL-CXR = support/contradict/uncertain/insufficient classification
```

写法：

```text
Visual sufficiency is best modeled as a four-way clinical evidence decision.
```

---

## 14.6 如果所有模块都不如 SAMEQ

最终方法：

```text
SAMEQ-CVCP remains final
```

写法：

```text
The data engine is the main contribution; additional modules are auxiliary.
```

---

## 14.7 如果 external 不涨

写法：

```text
The method improves image-specific visual sufficiency and in-domain representation. External label transfer is limited by dataset label mapping and domain shift.
```

不要硬吹 external generalization。

---

# 15. 论文结构建议

## Section 1：Introduction

核心问题：

```text
Report-derived language supervision may not be visually sufficient.
```

## Section 2：Visual Sufficiency Formulation

定义：

```text
support / contradict / uncertain / insufficient
```

## Section 3：VSL-CXR Method

三层：

```text
Data Engine
Evidence Encoder
Consistency Readout
```

## Section 4：Training with VLM Teacher

```text
Frozen decoder
Train vision tower
SAMEQ / hard negatives / VSL loss
```

## Section 5：Experiments

```text
CheXpert
external dataset
VSL diagnostics
hard shuffle
CCSH
calibration
case study
```

## Section 6：Ablations

```text
QA vs CF vs SAMEQ
K-negative
HNMB
CEQ
CCSH/AUCH
teacher model
```

## Section 7：Discussion

```text
why visual sufficiency matters
limitations
false hard negatives
external labels
deployment
```

---

# 16. Codex / 实验同学任务清单

## 16.1 Data scripts

```text
scripts/extract_clinical_statements.py
scripts/generate_counterfactual_statements.py
scripts/generate_sameq_pairs.py
scripts/generate_vsl_4class_labels.py
scripts/generate_hard_negative_pairs.py
scripts/mine_hard_negatives_memory_bank.py
scripts/audit_vsl_data_quality.py
scripts/audit_false_hard_negatives.py
```

## 16.2 Training scripts

```text
scripts/train_vsl_cxr.py
scripts/train_vsl_ceq.py
scripts/train_vsl_ccsh.py
scripts/train_vsl_hnmb.py
scripts/train_vsl_full.py
scripts/train_vlm_teacher_comparison.py
```

## 16.3 Evaluation scripts

```text
scripts/eval_chexpert_lp.py
scripts/eval_external_lp.py
scripts/eval_vsl_sufficiency.py
scripts/eval_ccsh_consistency.py
scripts/eval_hard_shuffle.py
scripts/eval_calibration.py
scripts/eval_ceq_attention.py
scripts/eval_casebook.py
scripts/eval_locked_final_comparison.py
```

## 16.4 Reporting scripts

```text
scripts/build_vsl_results_table.py
scripts/build_external_results_table.py
scripts/build_module_results_table.py
scripts/build_case_study_markdown.py
scripts/build_paper_figures.py
scripts/build_cost_table.py
```

---

# 17. 最终建议

## 17.1 最推荐短期主实验

```text
1. SAMEQ-full + CCSH
2. SAMEQ-K4 + CCSH
3. SAMEQ-HNMB + CCSH
4. VSL-4class
5. SAMEQ + CEQ + CCSH
6. Raw + CCSH
7. Main external dataset audit
```

## 17.2 最可能的最终方法

### Candidate A

```text
VSL-CXR-Core = SAMEQ + CCSH
```

### Candidate B

```text
VSL-CXR-HardNeg = SAMEQ-K/HNMB + CCSH
```

### Candidate C

```text
VSL-CXR-CEQ = SAMEQ + CEQ + CCSH
```

### Candidate D

```text
VSL-CXR-Full = SAMEQ-HNMB + CEQ + CCSH + AUCH
```

---

# 18. 最后一页总结

这份计划的重点不是继续堆实验，而是把项目升级为：

```text
VSL-CXR: Visual Sufficiency Learning for Chest X-rays
```

唯一主线：

```text
判断一张胸片是否提供足够视觉证据来支持一个临床陈述。
```

SAMEQ 是核心训练信号。  
HNMB/K-negative 是 hard negative 增强。  
CEQ 是证据查询模块。  
CCSH 是部署读出模块。  
AUCH 是 uncertainty/calibration 模块。  
External 是最终可信度验证。  
VLM teacher comparison 是泛化性验证。

最终希望得到一个清楚结论：

```text
VSL-CXR turns report-derived language supervision into visually sufficient clinical evidence learning, producing a deployable CXR vision encoder and consistency readout without requiring the LLM at deployment.
```
