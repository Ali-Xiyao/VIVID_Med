# VIVID-Med: Vision-Language Structured Alignment for Medical Imaging

用冻结 LLM 作为"结构化语义解码器/监督空间"，训练 ViT 学到可迁移、可验证的医学视觉表征。

## 核心思想

**LLM 是工具，ViT 是产品**：训练结束后丢掉 LLM，ViT-only 下游仍显著提升。

```
Image → ViT(train) → Projector(train) → Frozen LLM(forward) → JSON token logits
```

## 项目结构

```
021_260129VIVID/
├── configs/                 # 配置文件
│   └── cxr_chexpert.yaml   # CheXpert CXR 训练配置
├── data/                    # 数据加载模块
│   ├── dataset/            # 数据集文件
│   │   ├── CheXpert-v1.0-small/  # CheXpert 图像
│   │   ├── NIH Chest X-rays/     # NIH 图像
│   │   └── processed/            # 预处理后的 UMS-JSONL
│   ├── chexpert_dataset.py # CheXpert UMS 数据集
│   └── transforms.py       # 图像预处理
├── models/                  # 模型架构
│   ├── vit.py              # Vision Encoder (ViT)
│   ├── projector.py        # MLP + Prefix Tokens
│   └── vivid_model.py      # 整合模型
├── training/               # 训练模块
│   ├── losses.py           # L_tok, L_rank, L_vdep, L_ans
│   └── trainer.py          # 训练器
├── evaluation/             # 评估模块
│   ├── verifier.py         # UMS Verifier
│   └── metrics.py          # 评估指标
├── scripts/                # 运行脚本
│   ├── train_cxr.py        # CXR 训练脚本
│   └── test_pipeline.py    # 测试脚本
└── Profle/                 # 设计文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

数据集已放置在 `data/dataset/` 目录下：

```
data/dataset/
├── CheXpert-v1.0-small/    # CheXpert 图像数据
├── NIH Chest X-rays/       # NIH 图像数据
├── processed/              # 预处理后的 UMS-JSONL 文件
│   ├── chexpert_ums.jsonl
│   └── nih_external_test_ums.jsonl
├── prepare_chexpert.py     # CheXpert 预处理脚本
└── prepare_nih.py          # NIH 预处理脚本
```

如需重新生成 UMS 数据：

```bash
cd data/dataset
python prepare_chexpert.py
python prepare_nih.py
```

### 3. 测试流程

```bash
cd scripts
python test_pipeline.py
```

这将验证：
- 数据加载是否正常
- 模型是否能正常创建
- 前向传播是否正常
- Verifier 是否工作

### 4. 开始训练

Debug 模式（快速验证）：
```bash
python train_cxr.py --debug
```

完整训练：
```bash
python train_cxr.py --config ../configs/cxr_chexpert.yaml
```

## ViT Baseline（论文对比）

训练基线：
```bash
python train_vit_baseline.py --config ../configs/baseline_vit_chexpert.yaml
```

评估基线（指定 checkpoint）：
```bash
python eval_vit_baseline.py --config ../configs/baseline_vit_chexpert.yaml --checkpoint ../outputs/baseline_vit/best.pt
```

### 5. 恢复训练

```bash
python train_cxr.py --resume ./outputs/cxr_chexpert/checkpoints/step_5000.pt
```

## 配置说明

主要配置项（`configs/cxr_chexpert.yaml`）：

```yaml
model:
  vit_model_name: "vit_base_patch16_224"  # ViT 模型
  llm_model_name: "Qwen/Qwen2.5-1.5B-Instruct"  # 冻结 LLM
  num_prefix_tokens: 16  # 可学习前缀 token 数

training:
  learning_rate: 1.0e-4
  max_steps: 10000
  batch_size: 8
  gradient_accumulation_steps: 4  # 有效 batch = 32
```

## 训练目标

v1.0 默认只使用 `L_tok`（next-token prediction）：

```
L = L_tok + λ_rank * L_rank + λ_vdep * L_vdep + λ_ans * L_ans
```

- `L_tok`: 主监督，生成 UMS-JSON tokens
- `L_rank`: Hard-negative ranking（可选）
- `L_vdep`: Visual-dependence anti-shortcut（可选）
- `L_ans`: Answerability prediction（可选）

## 关键技术点

### 冻结 LLM 但保持可导

```python
# 冻结参数
for p in llm.parameters():
    p.requires_grad = False

# 但 forward 不能用 no_grad()！
# 否则梯度无法回传到 projector/ViT
outputs = llm(inputs_embeds=visual_embeds, ...)  # 不要包在 torch.no_grad() 里
```

### UMS-JSON 结构

```json
{
  "modality": "CXR",
  "findings": {
    "Pneumonia": {"state": "present", "score": 0.9},
    "Cardiomegaly": {"state": "absent", "score": null}
  },
  "study_view": "AP",
  "laterality": null,
  "answerability": {
    "Pneumonia": true,
    "laterality": false
  }
}
```

## 评估指标

- **分类指标**: AUC, F1, Precision, Recall
- **可靠性指标**: ECE, Brier Score, Risk-Coverage
- **临床硬错误**: laterality flip, unit error, scale error

## 参考文档

详细设计文档见 `Profle/` 目录：
- `00_方案总览.md` - 整体方案
- `02_方法.md` - 方法细节
- `03_数据集.md` - 数据集说明
- `04_模型.md` - 模型配置
- `05_UMS_与Verifier_与FailureTaxonomy.md` - UMS 规范
- `06_训练目标与SamplePolicy.md` - 训练目标
