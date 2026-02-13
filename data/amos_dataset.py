"""
AMOS22 CT 2.5D Dataset
加载预处理后的 AMOS 2.5D slices + UMS-JSON 标签
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from torch.utils.data import Dataset
from PIL import Image

from .transforms import get_train_transforms, get_val_transforms


class AMOS2DDataset(Dataset):
    """
    AMOS22 2.5D CT 数据集

    数据来源：
    - 图像：AMOS22/processed_2d/{train,val}/*.png
    - 标签：processed/amos_ums_{train,val}.jsonl
    """

    ORGAN_NAMES = [
        "spleen", "right_kidney", "left_kidney", "gall_bladder",
        "esophagus", "liver", "stomach", "aorta", "postcava",
        "pancreas", "right_adrenal_gland", "left_adrenal_gland",
        "duodenum", "bladder", "prostate_uterus",
    ]

    def __init__(
        self,
        data_root: str,
        ums_jsonl_path: str,
        transform=None,
        is_train: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.is_train = is_train
        self.label_names = self.ORGAN_NAMES
        self.num_labels = len(self.label_names)
        if transform is None:
            self.transform = get_train_transforms() if is_train else get_val_transforms()
        else:
            self.transform = transform

        self.samples = self._load_ums_jsonl(ums_jsonl_path, max_samples)
        split = "train" if is_train else "val"
        self.slice_dir = self.data_root / "AMOS22" / "processed_2d" / split
        print(f"AMOS2D: Loaded {len(self.samples)} samples from {ums_jsonl_path}")

    def _load_ums_jsonl(self, jsonl_path: str, max_samples: Optional[int] = None) -> List[Dict]:
        samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                samples.append(json.loads(line.strip()))
        return samples

    def _parse_findings_to_labels(self, findings: Dict) -> torch.Tensor:
        labels = torch.zeros(self.num_labels)
        for i, name in enumerate(self.label_names):
            if name in findings and findings[name].get("state") == "present":
                labels[i] = 1.0
        return labels

    def _create_ums_json_string(self, sample: Dict) -> str:
        output = {
            "modality": sample["modality"],
            "findings": {},
            "study_view": sample.get("study_view"),
        }
        for name in self.label_names:
            if name in sample["findings"]:
                output["findings"][name] = sample["findings"][name]
        return json.dumps(output, ensure_ascii=False)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        slice_filename = sample["extensions"]["slice_filename"]
        image_path = self.slice_dir / slice_filename

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        labels = self._parse_findings_to_labels(sample["findings"])
        answerable = torch.ones(self.num_labels, dtype=torch.bool)
        target_json = self._create_ums_json_string(sample)

        return {
            "image": image,
            "labels": labels,
            "answerable": answerable,
            "target_json": target_json,
            "study_view": sample.get("study_view"),
            "sample_id": sample["extensions"]["sample_id"],
            "original_path": f"AMOS22/processed_2d/{'train' if self.is_train else 'val'}/{slice_filename}",
            "prompt_text": None,
            "query_labels": None,
        }
