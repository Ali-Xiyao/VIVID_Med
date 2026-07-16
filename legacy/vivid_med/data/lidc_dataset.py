"""
LIDC-IDRI Dataset
胸部 CT 肺结节良恶性二分类
用于 CT 分类 Linear Probe 评估
"""

import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .transforms import get_train_transforms, get_val_transforms


class LIDCDataset(Dataset):
    """
    LIDC-IDRI 肺结节良恶性二分类

    数据来源:
    - 图像: LIDC-IDRI-slices/{patient_id}/nodule-{idx}/images/*.png
    - 标签: processed/lidc_nodule_labels.csv (malignancy → benign/malignant)

    每个 nodule 取中间 slice 作为代表（避免边缘 slice 信息不足）
    """

    NUM_CLASSES = 2
    CLASS_NAMES = ["benign", "malignant"]

    def __init__(
        self,
        data_root: str,
        label_csv: str,
        split: str = "train",
        transform=None,
        image_size: int = 224,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.data_root = Path(data_root)
        self.slices_root = self.data_root / "LIDC-IDRI-slices"
        self.split = split

        if transform is None:
            is_train = (split == "train")
            self.transform = get_train_transforms(image_size) if is_train else get_val_transforms(image_size)
        else:
            self.transform = transform

        # Load labels, exclude indeterminate (label == -1)
        all_samples = self._load_labels(label_csv)
        all_samples = [s for s in all_samples if s["label"] >= 0]

        # Patient-level split to avoid data leakage
        self.samples = self._split_by_patient(all_samples, split, val_ratio, test_ratio, seed)

        benign = sum(1 for s in self.samples if s["label"] == 0)
        malig = sum(1 for s in self.samples if s["label"] == 1)
        print(f"LIDC [{split}]: {len(self.samples)} nodules (benign={benign}, malignant={malig})")

    def _load_labels(self, csv_path: str):
        samples = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append({
                    "patient_id": row["patient_id"],
                    "nodule_idx": int(row["nodule_idx"]),
                    "label": int(row["label"]),
                    "num_slices": int(row["num_slices"]),
                })
        return samples

    def _split_by_patient(self, samples, split, val_ratio, test_ratio, seed):
        """Patient-level split to prevent data leakage"""
        patients = sorted(set(s["patient_id"] for s in samples))
        rng = np.random.RandomState(seed)
        rng.shuffle(patients)

        n = len(patients)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)

        test_patients = set(patients[:n_test])
        val_patients = set(patients[n_test:n_test + n_val])
        train_patients = set(patients[n_test + n_val:])

        if split == "train":
            target = train_patients
        elif split == "val":
            target = val_patients
        else:
            target = test_patients

        return [s for s in samples if s["patient_id"] in target]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        nodule_dir = self.slices_root / sample["patient_id"] / f"nodule-{sample['nodule_idx']}"
        img_dir = nodule_dir / "images"

        # Take middle slice as representative
        slices = sorted(img_dir.glob("*.png"))
        mid = len(slices) // 2
        img_path = slices[mid] if slices else None

        if img_path and img_path.exists():
            img = Image.open(img_path).convert("RGB")
        else:
            img = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        return {"image": img, "label": sample["label"]}
