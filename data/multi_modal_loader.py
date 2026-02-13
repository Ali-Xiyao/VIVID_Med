"""
Multi-Modal DataLoader
按比例从 CXR 和 CT 数据集中采样，支持混合 batch
"""

from typing import Dict, Any, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader, Sampler

import numpy as np


class MultiModalDataset(Dataset):
    """
    包装多个数据集，统一接口
    每个样本额外标记 modality 来源
    """

    def __init__(self, datasets: Dict[str, Dataset]):
        """
        Args:
            datasets: {"cxr": CheXpertUMSDataset, "ct": AMOS2DDataset, ...}
        """
        self.datasets = datasets
        self.modality_names = list(datasets.keys())

        # 构建全局索引 → (modality, local_idx) 映射
        self._index_map: List[tuple] = []
        for modality, ds in datasets.items():
            for i in range(len(ds)):
                self._index_map.append((modality, i))

        print(f"MultiModalDataset: {len(self)} total samples")
        for name, ds in datasets.items():
            print(f"  {name}: {len(ds)} samples")

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        modality, local_idx = self._index_map[idx]
        item = self.datasets[modality][local_idx]
        item["modality_source"] = modality
        return item
class MultiModalSampler(Sampler):
    """
    按比例从不同模态中采样
    例如 CXR:CT = 7:3 → 每个 epoch 中 70% 来自 CXR, 30% 来自 CT
    """

    def __init__(
        self,
        dataset: MultiModalDataset,
        ratios: Dict[str, float],
        total_samples_per_epoch: Optional[int] = None,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

        # 按模态分组索引
        self._modality_indices: Dict[str, List[int]] = {}
        for global_idx, (modality, _) in enumerate(dataset._index_map):
            if modality not in self._modality_indices:
                self._modality_indices[modality] = []
            self._modality_indices[modality].append(global_idx)

        # 归一化比例
        total_ratio = sum(ratios.get(m, 0) for m in self._modality_indices)
        self.ratios = {m: ratios.get(m, 0) / total_ratio for m in self._modality_indices}

        if total_samples_per_epoch is None:
            self.total_samples = len(dataset)
        else:
            self.total_samples = total_samples_per_epoch

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        return self.total_samples

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        indices = []

        for modality, ratio in self.ratios.items():
            n = int(self.total_samples * ratio)
            pool = self._modality_indices[modality]
            if n <= len(pool):
                chosen = rng.choice(pool, size=n, replace=False).tolist()
            else:
                chosen = rng.choice(pool, size=n, replace=True).tolist()
            indices.extend(chosen)

        rng.shuffle(indices)
        return iter(indices[:self.total_samples])


def _pad_and_stack(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Pad 1D tensors to the same length (with NaN) and stack"""
    max_len = max(t.shape[0] for t in tensors)
    padded = []
    for t in tensors:
        if t.shape[0] < max_len:
            pad = torch.full((max_len - t.shape[0],), float("nan"), dtype=t.dtype)
            t = torch.cat([t, pad])
        padded.append(t)
    return torch.stack(padded)


def multi_modal_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Multi-modal collate: 处理不同模态样本的混合 batch
    """
    images = torch.stack([item["image"] for item in batch])

    # labels 可能维度不同（CXR 7 labels vs CT 15 labels），用 target_json 统一
    return {
        "images": images,
        "target_jsons": [item["target_json"] for item in batch],
        "modality_sources": [item["modality_source"] for item in batch],
        "study_views": [item.get("study_view") for item in batch],
        "sample_ids": [item.get("sample_id") for item in batch],
        "original_paths": [item.get("original_path", "") for item in batch],
        "prompt_texts": [item.get("prompt_text") for item in batch],
        "query_labels": [item.get("query_labels") for item in batch],
        # 保留 labels/answerable 如果存在（用于 metrics 计算）
        # 不同模态 label 维度可能不同，pad 到最大长度
        "labels": _pad_and_stack([item["labels"] for item in batch]) if "labels" in batch[0] else None,
        "answerable": _pad_and_stack([item["answerable"] for item in batch]) if "answerable" in batch[0] else None,
    }
