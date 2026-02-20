"""
KiTS21 2.5D Dataset
加载预处理后的 KiTS21 2.5D slices + UMS-JSON 标签
用于 segmentation transfer evaluation
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .transforms import get_train_transforms, get_val_transforms


class KiTS2DDataset(Dataset):
    """
    KiTS21 2.5D CT 数据集 (用于分割 transfer evaluation)

    数据来源：
    - 图像：KITS21/processed_2d/{train,val}/*.png
    - Masks：KITS21/processed_2d/{train,val}/*_mask.png
    - 标签：processed/kits_ums_{train,val}.jsonl
    """

    LABEL_NAMES = ["kidney", "tumor", "cyst"]
    NUM_SEG_CLASSES = 4  # background + 3 classes

    def __init__(
        self,
        data_root: str,
        ums_jsonl_path: str,
        transform=None,
        is_train: bool = True,
        max_samples: Optional[int] = None,
        subset_fraction: Optional[float] = None,
        seed: int = 42,
    ):
        self.data_root = Path(data_root)
        self.is_train = is_train
        if transform is None:
            self.transform = get_train_transforms() if is_train else get_val_transforms()
        else:
            self.transform = transform

        split = "train" if is_train else "val"
        self.slice_dir = self.data_root / "KITS21" / "processed_2d" / split

        self.samples = self._load_ums_jsonl(ums_jsonl_path, max_samples)

        # Low-data subset support (10%, 25%, 50%)
        if subset_fraction is not None and 0 < subset_fraction < 1.0 and is_train:
            rng = np.random.RandomState(seed)
            n = max(1, int(len(self.samples) * subset_fraction))
            indices = rng.choice(len(self.samples), size=n, replace=False)
            self.samples = [self.samples[i] for i in sorted(indices)]
            print(f"KiTS2D: Using {subset_fraction*100:.0f}% subset = {len(self.samples)} samples")

        print(f"KiTS2D: Loaded {len(self.samples)} samples from {ums_jsonl_path}")

    def _load_ums_jsonl(self, jsonl_path: str, max_samples: Optional[int] = None) -> List[Dict]:
        samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                samples.append(json.loads(line.strip()))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        ext = sample["extensions"]
        slice_filename = ext["slice_filename"]
        mask_filename = ext.get("mask_filename", slice_filename.replace(".png", "_mask.png"))

        # Load image
        try:
            image = Image.open(self.slice_dir / slice_filename).convert("RGB")
        except Exception as e:
            print(f"Error loading {slice_filename}: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        # Load segmentation mask
        try:
            mask = Image.open(self.slice_dir / mask_filename)
            mask = np.array(mask, dtype=np.int64)
        except Exception:
            mask = np.zeros((224, 224), dtype=np.int64)

        if self.transform:
            image = self.transform(image)

        mask_tensor = torch.from_numpy(mask).long()  # (H, W)

        return {
            "image": image,
            "mask": mask_tensor,
            "sample_id": ext["sample_id"],
            "case_id": ext["case_id"],
            "slice_index": ext["slice_index"],
        }
