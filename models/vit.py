"""
Vision Encoder (ViT)
使用 timm 库加载预训练的 ViT 模型
"""

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError("Please install timm: pip install timm")


def create_vit_encoder(
    model_name: str = "vit_base_patch16_224",
    pretrained: bool = True,
    num_classes: int = 0,  # 0 表示不要分类头，只要特征
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
) -> nn.Module:
    """
    创建 ViT 编码器

    Args:
        model_name: timm 模型名称
            - vit_base_patch16_224: ViT-B/16 (默认)
            - vit_large_patch16_224: ViT-L/16
            - vit_base_patch32_224: ViT-B/32
        pretrained: 是否使用 ImageNet 预训练权重
        num_classes: 分类头输出维度，0 表示移除分类头
        drop_rate: Dropout rate
        drop_path_rate: DropPath rate (stochastic depth)

    Returns:
        ViT 模型
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )

    return model


class ViTEncoder(nn.Module):
    """
    ViT 编码器封装类
    提供更灵活的特征提取接口
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        output_type: str = "cls",  # "cls", "mean", "all"
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        """
        Args:
            model_name: timm 模型名称
            pretrained: 是否使用预训练权重
            output_type: 输出类型
                - "cls": 只返回 [CLS] token
                - "mean": 返回所有 patch tokens 的平均
                - "all": 返回所有 tokens (包括 CLS)
            drop_rate: Dropout rate
            drop_path_rate: DropPath rate
        """
        super().__init__()

        self.output_type = output_type

        # 创建 ViT 模型
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # 移除分类头
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # 获取模型配置
        self.embed_dim = self.vit.embed_dim
        self.num_patches = self.vit.patch_embed.num_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像 (B, C, H, W)

        Returns:
            features: 视觉特征
                - output_type="cls": (B, embed_dim)
                - output_type="mean": (B, embed_dim)
                - output_type="all": (B, num_tokens, embed_dim)
        """
        features = self.vit.forward_features(x)  # (B, num_tokens, embed_dim)

        if self.output_type == "cls":
            return features[:, 0]  # (B, embed_dim)
        elif self.output_type == "mean":
            return features[:, 1:].mean(dim=1)  # (B, embed_dim)
        else:  # "all"
            return features  # (B, num_tokens, embed_dim)

    def get_num_tokens(self) -> int:
        """获取输出 token 数量"""
        if self.output_type in ["cls", "mean"]:
            return 1
        else:
            return self.num_patches + 1  # patches + CLS

    def get_embed_dim(self) -> int:
        """获取嵌入维度"""
        return self.embed_dim


def get_vit_config(model_name: str) -> dict:
    """
    获取 ViT 模型配置信息

    Returns:
        dict with keys: embed_dim, num_heads, depth, patch_size
    """
    configs = {
        "vit_base_patch16_224": {
            "embed_dim": 768,
            "num_heads": 12,
            "depth": 12,
            "patch_size": 16,
            "num_patches": 196,  # (224/16)^2
        },
        "vit_large_patch16_224": {
            "embed_dim": 1024,
            "num_heads": 16,
            "depth": 24,
            "patch_size": 16,
            "num_patches": 196,
        },
        "vit_base_patch32_224": {
            "embed_dim": 768,
            "num_heads": 12,
            "depth": 12,
            "patch_size": 32,
            "num_patches": 49,  # (224/32)^2
        },
    }

    if model_name not in configs:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(configs.keys())}")

    return configs[model_name]
