"""
OrganAMNIST Dataset
腹部 CT 器官 11 分类（MedMNIST v2）
用于 CT 分类 Linear Probe 评估
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .transforms import get_train_transforms, get_val_transforms


# OrganAMNIST 11 classes
ORGAN_CLASSES = [
    "bladder", "femur_left", "femur_right", "heart",
    "kidney_left", "kidney_right", "liver", "lung_left",
    "lung_right", "spleen", "stomach",
]


class OrganAMNISTDataset(Dataset):
    """
    OrganAMNIST 224×224 (腹部 CT 器官 11 分类)
    数据来源: organamnist_224.npz (MedMNIST v2)
    """

    NUM_CLASSES = 11

    def __init__(
        self,
        npz_path: str,
        split: str = "train",
        transform=None,
        image_size: int = 224,
    ):
        data = np.load(npz_path)
        self.images = data[f"{split}_images"]   # (N, 224, 224) uint8
        self.labels = data[f"{split}_labels"].squeeze()  # (N,)
        self.split = split

        if transform is None:
            is_train = (split == "train")
            self.transform = get_train_transforms(image_size) if is_train else get_val_transforms(image_size)
        else:
            self.transform = transform

        print(f"OrganAMNIST [{split}]: {len(self)} samples, {self.NUM_CLASSES} classes")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]  # (224, 224) uint8 grayscale
        label = int(self.labels[idx])

        # Grayscale → RGB PIL
        img = Image.fromarray(img).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return {"image": img, "label": label}
