"""
Segmentation Transfer Head
ViT patch tokens → spatial feature map → upsampled segmentation mask
用于 KiTS21 分割 transfer evaluation
"""

import torch
import torch.nn as nn


class ViTSegHead(nn.Module):
    """
    将 ViT patch tokens reshape 为 spatial feature map，
    通过转置卷积上采样到原始分辨率
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 4,
        img_size: int = 224,
        patch_size: int = 16,
    ):
        super().__init__()
        self.grid_size = img_size // patch_size  # 14
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=2, stride=2),  # 14→28
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 28→56
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 56→112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2),  # 112→224
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, num_patches, embed_dim) — CLS token excluded

        Returns:
            logits: (B, num_classes, H, W)
        """
        B = patch_tokens.shape[0]
        x = patch_tokens.transpose(1, 2).reshape(B, self.embed_dim, self.grid_size, self.grid_size)
        return self.decoder(x)
