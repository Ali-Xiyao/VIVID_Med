# VIVID-Med: Vision-Language Structured Alignment for Medical Imaging
"""
核心思想：用冻结 LLM 作为"结构化语义解码器/监督空间"，训练 ViT 学到可迁移、可验证的医学视觉表征。

架构：Image → ViT(train) → Projector(train) → Frozen LLM(forward) → JSON token logits
"""

__version__ = "0.1.0"
