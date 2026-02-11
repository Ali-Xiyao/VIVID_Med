"""
CheXpert UMS Dataset
加载预处理后的 CheXpert UMS-JSONL 数据
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from torch.utils.data import Dataset
from PIL import Image

from .transforms import get_train_transforms, get_val_transforms


class CheXpertUMSDataset(Dataset):
    """
    CheXpert 数据集，使用 UMS-JSON 格式的标签

    数据来源：
    - 图像：CheXpert-v1.0-small/train/ 或 valid/
    - 标签：processed/chexpert_ums.jsonl
    """

    # CheXpert 的 14 个标签（与 UMS findings 对应）
    FINDING_NAMES = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
        "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
        "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
    ]

    # 用于 external test 的共同标签子集（与 NIH 对齐）
    COMMON_LABELS = [
        "No Finding", "Atelectasis", "Cardiomegaly", "Consolidation",
        "Edema", "Pleural Effusion", "Pneumonia", "Pneumothorax"
    ]

    def __init__(
        self,
        data_root: str,
        ums_jsonl_path: str,
        transform=None,
        is_train: bool = True,
        use_common_labels_only: bool = False,
        selected_labels: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        json_include_all_labels: bool = False,
        json_missing_state: Optional[str] = None,
        json_null_state: Optional[str] = None,
        dense_subset_top_k: Optional[int] = None,
        dense_subset_min_answerable: Optional[int] = None,
    ):
        """
        Args:
            data_root: 数据集根目录（包含 CheXpert-v1.0-small/）
            ums_jsonl_path: UMS JSONL 文件路径
            transform: 图像变换
            is_train: 是否为训练模式
            use_common_labels_only: 是否只使用与 NIH 共同的标签子集
            selected_labels: 指定自定义 findings 标签子集（优先级高于 use_common_labels_only）
            max_samples: 最大样本数（用于调试）
        """
        self.data_root = Path(data_root)
        self.is_train = is_train
        self.use_common_labels_only = use_common_labels_only
        self.selected_labels = selected_labels
        self.json_include_all_labels = json_include_all_labels
        self.json_missing_state = json_missing_state
        self.json_null_state = json_null_state
        self.dense_subset_top_k = dense_subset_top_k
        self.dense_subset_min_answerable = dense_subset_min_answerable

        # 设置 transform
        if transform is None:
            self.transform = get_train_transforms() if is_train else get_val_transforms()
        else:
            self.transform = transform

        if selected_labels:
            normalized_labels = []
            seen = set()
            for label in selected_labels:
                if not isinstance(label, str):
                    continue
                if label in seen:
                    continue
                normalized_labels.append(label)
                seen.add(label)
            if not normalized_labels:
                raise ValueError("selected_labels is provided but empty after normalization")
            invalid_labels = [label for label in normalized_labels if label not in self.FINDING_NAMES]
            if invalid_labels:
                raise ValueError(
                    f"selected_labels contains unknown labels: {invalid_labels}. "
                    f"Supported labels: {self.FINDING_NAMES}"
                )
            self.label_names = normalized_labels
        else:
            self.label_names = self.COMMON_LABELS if use_common_labels_only else self.FINDING_NAMES
        self.num_labels = len(self.label_names)

        # 加载 UMS 数据
        self.samples = self._load_ums_jsonl(ums_jsonl_path, max_samples)
        self.samples = self._apply_dense_subset(
            samples=self.samples,
            top_k=dense_subset_top_k,
            min_answerable=dense_subset_min_answerable,
        )

        print(f"Loaded {len(self.samples)} samples from {ums_jsonl_path}")
        print(f"Using {self.num_labels} labels: {self.label_names}")

    def _load_ums_jsonl(self, jsonl_path: str, max_samples: Optional[int] = None) -> List[Dict]:
        """加载 UMS JSONL 文件"""
        samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                sample = json.loads(line.strip())
                samples.append(sample)
        return samples

    def _count_answerable_findings(self, sample: Dict[str, Any]) -> int:
        answerability = sample.get("answerability", {})
        findings = sample.get("findings", {})
        count = 0

        for name in self.label_names:
            answerable_value = answerability.get(name)
            if answerable_value is None:
                finding = findings.get(name, {})
                if isinstance(finding, dict) and finding.get("state") is not None:
                    count += 1
                continue

            if bool(answerable_value):
                count += 1

        return count

    def _apply_dense_subset(
        self,
        samples: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        min_answerable: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not samples:
            return samples

        if min_answerable is None and top_k is None:
            return samples

        scored_samples = []
        for sample in samples:
            answerable_count = self._count_answerable_findings(sample)
            scored_samples.append((answerable_count, sample))

        if min_answerable is not None:
            min_answerable = int(min_answerable)
            scored_samples = [
                (answerable_count, sample)
                for answerable_count, sample in scored_samples
                if answerable_count >= min_answerable
            ]
            print(
                f"Applied dense subset min_answerable>={min_answerable}, "
                f"remaining samples: {len(scored_samples)}"
            )

        if top_k is not None:
            top_k = int(top_k)
            scored_samples.sort(key=lambda item: item[0], reverse=True)
            scored_samples = scored_samples[:top_k]
            if scored_samples:
                counts = [item[0] for item in scored_samples]
                print(
                    f"Applied dense subset top_k={top_k}, "
                    f"answerable findings range: [{min(counts)}, {max(counts)}], "
                    f"avg={sum(counts)/len(counts):.2f}"
                )
            else:
                print(f"Applied dense subset top_k={top_k}, no samples left")

        return [sample for _, sample in scored_samples]

    def _get_image_path(self, original_path: str) -> Path:
        """
        获取图像的完整路径
        original_path 格式: CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg
        """
        return self.data_root / original_path

    def _parse_findings_to_labels(self, findings: Dict) -> torch.Tensor:
        """
        将 UMS findings 转换为多标签向量

        Returns:
            labels: shape (num_labels,)
                - 1.0: present
                - 0.0: absent
                - -1.0: uncertain (用于 uncertain-aware loss)
                - nan: missing/unanswerable
        """
        labels = torch.full((self.num_labels,), float('nan'))

        for i, name in enumerate(self.label_names):
            if name in findings:
                state = findings[name].get("state")
                if state == "present":
                    labels[i] = 1.0
                elif state == "absent":
                    labels[i] = 0.0
                elif state == "uncertain":
                    labels[i] = -1.0
                # state == None 保持 nan

        return labels

    def _parse_answerability(self, answerability: Dict) -> torch.Tensor:
        """
        解析字段级可答性

        Returns:
            answerable: shape (num_labels,), bool tensor
        """
        answerable = torch.zeros(self.num_labels, dtype=torch.bool)

        for i, name in enumerate(self.label_names):
            if name in answerability:
                answerable[i] = answerability[name]

        return answerable

    def _normalize_state(self, state: Optional[str], missing: bool = False) -> Optional[str]:
        if state is None:
            if missing and self.json_missing_state is not None:
                return self.json_missing_state
            if not missing and self.json_null_state is not None:
                return self.json_null_state
        return state

    def _create_ums_json_string(self, sample: Dict) -> str:
        """
        创建用于训练的 UMS JSON 字符串（简化版，只包含 findings）
        这是 LLM 需要生成的目标序列
        """
        # 简化的 UMS 输出格式（用于训练）
        output = {
            "modality": sample["modality"],
            "findings": {},
            "study_view": sample.get("study_view"),
        }

        # 只包含使用的标签
        for name in self.label_names:
            if name in sample["findings"]:
                item = dict(sample["findings"][name])
                item["state"] = self._normalize_state(item.get("state"), missing=False)
                output["findings"][name] = item
            elif self.json_include_all_labels:
                output["findings"][name] = {
                    "state": self._normalize_state(None, missing=True),
                    "score": None,
                }

        return json.dumps(output, ensure_ascii=False)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # 加载图像
        image_path = self._get_image_path(sample["extensions"]["original_path"])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回一个黑色图像作为 fallback
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 解析标签
        labels = self._parse_findings_to_labels(sample["findings"])
        answerable = self._parse_answerability(sample["answerability"])

        # 创建目标 JSON 字符串
        target_json = self._create_ums_json_string(sample)

        # 获取 study_view
        study_view = sample.get("study_view")  # AP, PA, LAT, or None

        return {
            "image": image,
            "labels": labels,
            "answerable": answerable,
            "target_json": target_json,
            "study_view": study_view,
            "sample_id": sample["extensions"]["sample_id"],
            "original_path": sample["extensions"]["original_path"],
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    自定义 collate 函数
    """
    images = torch.stack([item["image"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    answerable = torch.stack([item["answerable"] for item in batch])

    return {
        "images": images,
        "labels": labels,
        "answerable": answerable,
        "target_jsons": [item["target_json"] for item in batch],
        "study_views": [item["study_view"] for item in batch],
        "sample_ids": [item["sample_id"] for item in batch],
        "original_paths": [item["original_path"] for item in batch],
    }
