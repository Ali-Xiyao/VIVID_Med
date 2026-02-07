# UMS Schema Coverage Table（v0.x）

> 关键原则：**schema 是统一的，但 coverage 不统一**。缺失字段必须显式写为 `null`（或 `answerability=false` + `uncertainty`），而不是省略。

| 数据集 | modality | anatomy | findings | study_view | laterality | geometry | measurements | answerability/uncertainty | provenance/verifier | 备注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| AMOS (CT) | ✅ | ✅ | ⚠️ | ❌ | ❌ | ✅ (mask/bbox) | ✅ | ✅ | ✅ | laterality/view 多为不可答，建议 `null` + abstain |
| CheXpert (CXR) | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | 30k 采样；anatomy 固定为 chest |
| NIH ChestX-ray14 (CXR) | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | external test-only；使用 8 标签共同子集 |
| KiTS21 (CT) | ✅ | ✅ | ❌/⚠️ | ❌ | ❌ | ✅ (mask/bbox) | ✅/⚠️ | ✅ | ✅ | transfer；measurement 字段名/口径待定 |
| LIDC-IDRI (CT) | ✅ | ✅ | ⚠️ | ❌ | ❌ | ✅/⚠️ | ✅ | ✅ | ✅ | 可选加分：结节测量/属性；字段映射待定 |
| AMOS (MRI 子集) | ✅ | ✅ | ⚠️ | ❌ | ❌ | ✅ (mask/bbox) | ✅ | ✅ | ✅ | 可选：低成本引入 MRI；是否纳入待定 |

---

## CheXpert 字段覆盖详情（30k 采样）

> 总体缺失率：30k * 14 findings 的 missing(blank/NaN)≈70.47%（来源：`data/dataset/processed/chexpert_sampled_30k.csv`；在 UMS 中表现为 `state=null + answerability=false`）。

**基础信息**
- 采样数量：30,000
- 采样策略：类均衡采样（稀有标签权重更高）
- 随机种子：42

**findings 字段覆盖率**

| 标签 | present | absent | uncertain | missing | 覆盖率 |
|------|---------|--------|-----------|---------|--------|
| No Finding | 2,551 | 0 | 0 | 27,449 | 8.5% |
| Enlarged Cardiomediastinum | 3,189 | 2,440 | 1,382 | 22,989 | 23.4% |
| Cardiomegaly | 5,086 | 1,279 | 824 | 22,811 | 24.0% |
| Lung Opacity | 15,028 | 763 | 659 | 13,550 | 54.8% |
| Lung Lesion | 3,027 | 144 | 183 | 26,646 | 11.2% |
| Edema | 7,593 | 2,461 | 1,396 | 18,550 | 38.2% |
| Consolidation | 3,680 | 3,337 | 2,933 | 20,050 | 33.2% |
| Pneumonia | 2,461 | 281 | 2,169 | 25,089 | 16.4% |
| Atelectasis | 5,690 | 132 | 3,754 | 20,424 | 31.9% |
| Pneumothorax | 3,906 | 6,806 | 309 | 18,979 | 36.7% |
| Pleural Effusion | 12,421 | 4,226 | 1,347 | 12,006 | 60.0% |
| Pleural Other | 1,975 | 37 | 253 | 27,735 | 7.6% |
| Fracture | 2,933 | 265 | 60 | 26,742 | 10.9% |
| Support Devices | 16,180 | 744 | 135 | 12,941 | 56.9% |

**其他字段**
- `study_view`：覆盖率高（AP/PA/LAT 均有元数据）
- `laterality`：固定为 `null`（CXR 不可判定左右侧）
- `geometry`：固定为 `null`（无 bbox/mask 标注）
- `measurements`：固定为空（无数值测量）

---

## NIH ChestX-ray14 字段覆盖详情（External Test）

**基础信息**
- 测试集数量：25,596（官方 test split）
- 用途：external test-only（不参与训练）

**findings 字段覆盖率（共同子集 8 标签）**

| 标签 | present | absent | 覆盖率 |
|------|---------|--------|--------|
| No Finding | 9,861 | 15,735 | 100% |
| Atelectasis | 3,279 | 22,317 | 100% |
| Cardiomegaly | 1,069 | 24,527 | 100% |
| Consolidation | 1,815 | 23,781 | 100% |
| Edema | 925 | 24,671 | 100% |
| Pleural Effusion | 4,658 | 20,938 | 100% |
| Pneumonia | 555 | 25,041 | 100% |
| Pneumothorax | 2,665 | 22,931 | 100% |

**注意**：NIH 没有 `uncertain` 标签，所有标签都是确定的 present/absent。

**其他字段**
- `study_view`：部分覆盖（View Position 字段）
- `laterality`：固定为 `null`
- `geometry`：固定为 `null`（BBox_List_2017.csv 仅有少量标注，不作为主要监督）
- `measurements`：固定为空

---

## NIH ↔ CheXpert 标签映射表

| CheXpert 标签 | NIH 标签 | 语义匹配 |
|---------------|----------|----------|
| Atelectasis | Atelectasis | exact |
| Cardiomegaly | Cardiomegaly | exact |
| Consolidation | Consolidation | exact |
| Edema | Edema | exact |
| Pleural Effusion | Effusion | equivalent |
| Pneumonia | Pneumonia | exact |
| Pneumothorax | Pneumothorax | exact |
| No Finding | No Finding | exact |

**CheXpert 独有标签**（不参与 external test 评测）：
- Enlarged Cardiomediastinum
- Lung Opacity
- Lung Lesion
- Pleural Other
- Fracture
- Support Devices

**NIH 独有标签**（不参与 external test 评测）：
- Emphysema
- Fibrosis
- Hernia
- Infiltration
- Mass
- Nodule
- Pleural_Thickening

---

## 生成的数据文件

数据文件位于 `vivid/data/dataset/processed/` 目录：

| 文件 | 说明 | 大小 |
|------|------|------|
| `chexpert_ums.jsonl` | CheXpert 30k UMS-JSON | ~92 MB |
| `chexpert_sampled_30k.csv` | CheXpert 采样 CSV | ~3.2 MB |
| `chexpert_sample_list.json` | 采样列表（用于复现） | ~2.4 MB |
| `nih_external_test_ums.jsonl` | NIH 测试集 UMS-JSON | ~51 MB |
| `nih_external_test.csv` | NIH 测试集 CSV | ~1.8 MB |
| `nih_chexpert_label_mapping.json` | 标签映射表 | ~0.5 KB |

---

## 待补（占位）

- ~~"findings"的具体字段集合（CheXpert 标签映射表）~~ ✅ 已完成
- ~~"study_view"的来源字段与缺失率统计~~ ✅ 已完成
- ~~laterality 推导规则与不可判定占比~~ ✅ 已完成（固定为 null）
- ~~NIH 的字段映射细则与最终统计~~ ✅ 已完成
- KiTS21 / LIDC 的字段映射细则与最终统计（待数据集准备后补充）
