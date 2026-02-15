"""
Vision Encoder (ViT)
使用 timm 库加载预训练的 ViT 模型
支持 GFTM (Gradient-Focused Token Masking)
支持 SAR (Soft Attention Reweighting) — V10
支持 HFP (Hierarchical Feature Projection) 中间层特征提取 — V10
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

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
        gftm_enabled: bool = False,
        sar_enabled: bool = False,
        sar_alpha: float = 1.0,
        hfp_enabled: bool = False,
        hfp_layers: Optional[list] = None,
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
            gftm_enabled: 是否启用 GFTM attention hook
            sar_enabled: 是否启用 SAR (Soft Attention Reweighting)
            sar_alpha: SAR 加权强度 (可学习初始值)
            hfp_enabled: 是否启用 HFP (Hierarchical Feature Projection)
            hfp_layers: HFP 提取的中间层索引，默认 [3, 7, 11] for ViT-B/16
        """
        super().__init__()

        self.output_type = output_type
        self.gftm_enabled = gftm_enabled
        self.sar_enabled = sar_enabled
        self.hfp_enabled = hfp_enabled

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

        # GFTM / SAR: 注册 attention hook 捕获最后一个 block 的 attention weights
        self._last_attn_weights = None
        if gftm_enabled or sar_enabled:
            self._register_attn_hook()

        # SAR: 可学习的加权强度参数
        if sar_enabled:
            self.sar_alpha = nn.Parameter(torch.tensor(float(sar_alpha)))

        # HFP: 注册中间层 hook 捕获多层特征
        self._hfp_features = {}
        if hfp_enabled:
            num_blocks = len(self.vit.blocks)
            self.hfp_layers = hfp_layers if hfp_layers is not None else [num_blocks // 4 - 1, num_blocks // 2 - 1, num_blocks - 1]
            self._register_hfp_hooks()

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.0, mask_mode: str = "gftm") -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像 (B, C, H, W)
            mask_ratio: GFTM token masking 比例 (0.0 = no masking)
            mask_mode: "gftm" (attention-guided) or "random"

        Returns:
            features: 视觉特征
                - output_type="cls": (B, embed_dim)
                - output_type="mean": (B, embed_dim)
                - output_type="all": (B, num_tokens, embed_dim)
        """
        # 获取所有 tokens
        features = self.vit.forward_features(x)  # (B, num_tokens, embed_dim)

        # GFTM: 在 forward_features 之后做 token selection (hard masking)
        if mask_ratio > 0 and self.training and self.gftm_enabled:
            features = self._apply_gftm(features, mask_ratio, mask_mode)

        # SAR: Soft Attention Reweighting (不丢 token，软加权)
        # 训练和推理都启用，避免分布偏移
        if self.sar_enabled and self._last_attn_weights is not None:
            features = self._apply_sar(features)

        if self.output_type == "cls":
            return features[:, 0]  # (B, embed_dim)
        elif self.output_type == "mean":
            return features[:, 1:].mean(dim=1)  # (B, embed_dim)
        else:  # "all"
            return features  # (B, num_tokens, embed_dim)

    def _register_attn_hook(self):
        """注册 hook 捕获最后一个 block 的 attention weights"""
        last_block = self.vit.blocks[-1]

        def hook_fn(module, input, output):
            # timm Attention module: output is (B, N, C)
            # We need to access the attention weights before softmax dropout
            # timm stores attn weights in module.attn_drop if we override
            pass

        # 更好的方式：直接 monkey-patch Attention.forward 来保存 attn weights
        original_attn_forward = last_block.attn.forward

        def patched_attn_forward(x, **kwargs):
            B, N, C = x.shape
            qkv = last_block.attn.qkv(x).reshape(B, N, 3, last_block.attn.num_heads, C // last_block.attn.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * last_block.attn.scale
            attn = attn.softmax(dim=-1)
            self._last_attn_weights = attn.detach()  # (B, num_heads, N, N)
            attn = last_block.attn.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = last_block.attn.proj(x)
            x = last_block.attn.proj_drop(x)
            return x

        last_block.attn.forward = patched_attn_forward

    def _apply_gftm(self, features: torch.Tensor, mask_ratio: float, mask_mode: str) -> torch.Tensor:
        """
        GFTM: Gradient-Focused Token Masking
        在 forward_features 之后，根据 attention weights 选择性保留 tokens

        Args:
            features: (B, N, embed_dim) where N = 1 (CLS) + num_patches
            mask_ratio: 要 mask 掉的比例
            mask_mode: "gftm" or "random"

        Returns:
            masked_features: (B, 1 + num_keep, embed_dim)
        """
        B, N, D = features.shape
        num_patches = N - 1  # 去掉 CLS
        num_keep = max(1, int(num_patches * (1 - mask_ratio)))

        cls_token = features[:, :1, :]  # (B, 1, D)
        patch_tokens = features[:, 1:, :]  # (B, num_patches, D)

        if mask_mode == "gftm" and self._last_attn_weights is not None:
            # 使用 CLS-to-patch attention: 保留低注意力 tokens (mask 高注意力)
            cls_attn = self._last_attn_weights[:, :, 0, 1:].mean(dim=1)  # (B, num_patches)
            # topk with largest=False → 保留注意力最低的 tokens
            _, keep_indices = cls_attn.topk(num_keep, dim=1, largest=False)
        else:
            # Random masking (用于消融)
            keep_indices = torch.stack([
                torch.randperm(num_patches, device=features.device)[:num_keep]
                for _ in range(B)
            ])

        # Sort indices for consistent ordering
        keep_indices, _ = keep_indices.sort(dim=1)

        # Gather selected patch tokens
        keep_indices_expanded = keep_indices.unsqueeze(-1).expand(-1, -1, D)
        selected_patches = torch.gather(patch_tokens, 1, keep_indices_expanded)

        # Concatenate CLS + selected patches
        return torch.cat([cls_token, selected_patches], dim=1)

    def _apply_sar(self, features: torch.Tensor) -> torch.Tensor:
        """
        SAR: Soft Attention Reweighting
        用 CLS-to-patch attention 的倒数作为软权重，放大低注意力 tokens。
        不丢弃任何 token，避免 GFTM 的信息损失和过拟合问题。

        Args:
            features: (B, N, embed_dim) where N = 1 (CLS) + num_patches

        Returns:
            reweighted_features: (B, N, embed_dim) 同形状
        """
        B, N, D = features.shape
        num_patches = N - 1

        cls_token = features[:, :1, :]  # (B, 1, D)
        patch_tokens = features[:, 1:, :]  # (B, num_patches, D)

        # CLS-to-patch attention: (B, num_patches)
        cls_attn = self._last_attn_weights[:, :, 0, 1:].mean(dim=1)  # avg over heads

        # 倒数加权：低注意力 patch 获得更高权重
        inv_attn = 1.0 / (cls_attn + 1e-6)
        # 归一化到均值为 1（不改变整体 scale）
        inv_attn = inv_attn / (inv_attn.mean(dim=1, keepdim=True) + 1e-6)

        # 软加权：token_i = token_i * (1 + alpha * (w_i - 1))
        # alpha=0 时退化为原始特征，alpha>0 时放大低注意力 tokens
        weights = 1.0 + self.sar_alpha * (inv_attn - 1.0)  # (B, num_patches)
        weights = weights.unsqueeze(-1)  # (B, num_patches, 1)

        reweighted_patches = patch_tokens * weights
        return torch.cat([cls_token, reweighted_patches], dim=1)

    def _register_hfp_hooks(self):
        """注册 HFP hooks 捕获中间层特征"""
        for layer_idx in self.hfp_layers:
            block = self.vit.blocks[layer_idx]

            def make_hook(idx):
                def hook_fn(module, input, output):
                    self._hfp_features[idx] = output
                return hook_fn

            block.register_forward_hook(make_hook(layer_idx))

    def get_hfp_features(self) -> dict:
        """获取 HFP 中间层特征 (forward 之后调用)"""
        return self._hfp_features

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
