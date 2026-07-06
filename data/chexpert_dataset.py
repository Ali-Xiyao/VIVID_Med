"""
CheXpert UMS Dataset
加载预处理后的 CheXpert UMS-JSONL 数据
"""

import json
import os
import random
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
        field_query_training: Optional[Dict[str, Any]] = None,
        target_format: str = "json",  # "json" (UMS) or "text" (free-text)
        schema_mode: str = "state_only",
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
        self._label_name_to_index = {name: idx for idx, name in enumerate(self.label_names)}
        self.field_query_training_cfg = self._build_field_query_training_config(field_query_training)
        _valid_formats = ("json", "text")
        if target_format not in _valid_formats:
            raise ValueError(
                f"target_format must be one of {_valid_formats}, got '{target_format}'"
            )
        self.target_format = target_format
        _valid_schema_modes = ("state_only", "state_answerability", "state_uncertainty")
        if schema_mode not in _valid_schema_modes:
            raise ValueError(
                f"schema_mode must be one of {_valid_schema_modes}, got '{schema_mode}'"
            )
        self.schema_mode = schema_mode

        # 加载 UMS 数据
        self.samples = self._load_ums_jsonl(ums_jsonl_path, max_samples)
        self.samples = self._apply_dense_subset(
            samples=self.samples,
            top_k=dense_subset_top_k,
            min_answerable=dense_subset_min_answerable,
        )

        print(f"Loaded {len(self.samples)} samples from {ums_jsonl_path}")
        print(f"Using {self.num_labels} labels: {self.label_names}")
        if self.field_query_training_cfg["enabled"]:
            print(
                "Field-query training enabled: "
                f"K={self.field_query_training_cfg['k_min']}..{self.field_query_training_cfg['k_max']}, "
                f"focus_labels={self.field_query_training_cfg['focus_labels']}"
            )

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

    def _build_field_query_training_config(
        self,
        field_query_training: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        cfg = dict(field_query_training or {})
        enabled = bool(cfg.get("enabled", False))

        k_min = max(int(cfg.get("k_min", 2)), 1)
        k_max = max(int(cfg.get("k_max", 3)), k_min)

        focus_labels_raw = cfg.get("focus_labels", []) or []
        focus_labels = [
            name for name in focus_labels_raw if isinstance(name, str) and name in self._label_name_to_index
        ]
        focus_probability = float(cfg.get("focus_probability", 0.7))
        focus_probability = min(max(focus_probability, 0.0), 1.0)

        sampling_weights_raw = cfg.get("label_sampling_weights", {}) or {}
        sampling_weights: Dict[str, float] = {}
        if isinstance(sampling_weights_raw, dict):
            for key, value in sampling_weights_raw.items():
                if key not in self._label_name_to_index:
                    continue
                try:
                    weight = float(value)
                except (TypeError, ValueError):
                    continue
                if weight > 0.0:
                    sampling_weights[key] = weight

        prompt_template = str(
            cfg.get(
                "prompt_template",
                (
                    "You are a medical imaging AI assistant.\n"
                    "Output ONLY one valid JSON object with keys: modality, findings, study_view.\n"
                    "modality must be \"CXR\".\n"
                    "findings must include ONLY these labels: {label_list}.\n"
                    "For each finding, output {\"state\": present|absent|uncertain|null, \"score\": null}.\n"
                    "Do not output any labels outside the list.\n"
                    "study_view is AP, PA, LAT, or null."
                ),
            )
        )

        return {
            "enabled": enabled,
            "k_min": k_min,
            "k_max": k_max,
            "focus_labels": focus_labels,
            "focus_probability": focus_probability,
            "label_sampling_weights": sampling_weights,
            "prompt_template": prompt_template,
        }

    def _sample_weighted_without_replacement(
        self,
        candidates: List[str],
        count: int,
        sampling_weights: Dict[str, float],
    ) -> List[str]:
        if count <= 0 or not candidates:
            return []
        if count >= len(candidates):
            return list(candidates)

        selected: List[str] = []
        remaining = list(candidates)
        for _ in range(count):
            if not remaining:
                break
            weights = torch.tensor(
                [sampling_weights.get(name, 1.0) for name in remaining], dtype=torch.float32
            )
            weights = torch.clamp(weights, min=1e-6)
            chosen_idx = int(torch.multinomial(weights, num_samples=1, replacement=False).item())
            selected_name = remaining.pop(chosen_idx)
            selected.append(selected_name)
        return selected

    def _sample_field_query_labels(self, answerable: torch.Tensor) -> List[str]:
        cfg = self.field_query_training_cfg
        if not cfg["enabled"]:
            return list(self.label_names)

        answerable_flags = answerable.tolist()
        candidates = [
            name
            for name in self.label_names
            if self._label_name_to_index[name] < len(answerable_flags)
            and bool(answerable_flags[self._label_name_to_index[name]])
        ]
        if not candidates:
            candidates = list(self.label_names)

        max_count = min(cfg["k_max"], len(candidates))
        min_count = min(cfg["k_min"], max_count)
        min_count = max(min_count, 1)
        max_count = max(max_count, min_count)
        query_count = int(torch.randint(min_count, max_count + 1, (1,)).item())

        focus_candidates = [name for name in candidates if name in cfg["focus_labels"]]
        selected: List[str] = []
        remaining = list(candidates)

        if focus_candidates and torch.rand(1).item() < cfg["focus_probability"]:
            focus_pick = self._sample_weighted_without_replacement(
                candidates=focus_candidates,
                count=1,
                sampling_weights=cfg["label_sampling_weights"],
            )
            if focus_pick:
                selected.extend(focus_pick)
                remaining = [name for name in remaining if name not in focus_pick]

        need = query_count - len(selected)
        if need > 0:
            selected.extend(
                self._sample_weighted_without_replacement(
                    candidates=remaining,
                    count=need,
                    sampling_weights=cfg["label_sampling_weights"],
                )
            )
        return selected

    def _create_field_query_prompt(self, query_labels: List[str]) -> str:
        label_list = ", ".join(f"\"{name}\"" for name in query_labels)
        template = self.field_query_training_cfg["prompt_template"]
        return (
            template
            .replace("{label_list}", label_list)
            .replace("{num_fields}", str(len(query_labels)))
        )

    def _create_ums_json_string(
        self,
        sample: Dict,
        include_labels: Optional[List[str]] = None,
        include_missing_for_selected: bool = False,
    ) -> str:
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
        target_labels = include_labels if include_labels is not None else self.label_names

        for name in target_labels:
            if name in sample["findings"]:
                item = dict(sample["findings"][name])
                item["state"] = self._normalize_state(item.get("state"), missing=False)
                self._augment_schema_item(sample, name, item)
                output["findings"][name] = item
            elif include_missing_for_selected or (
                include_labels is None and self.json_include_all_labels
            ):
                item = {
                    "state": self._normalize_state(None, missing=True),
                    "score": None,
                }
                self._augment_schema_item(sample, name, item)
                output["findings"][name] = item

        return json.dumps(output, ensure_ascii=False)

    def _augment_schema_item(self, sample: Dict, name: str, item: Dict[str, Any]) -> None:
        if self.schema_mode == "state_answerability":
            answerability = sample.get("answerability", {})
            item["answerable"] = bool(answerability.get(name, False))
        elif self.schema_mode == "state_uncertainty":
            uncertainty = sample.get("uncertainty", {})
            uncertain = uncertainty.get(name)
            if uncertain is None:
                item["uncertain"] = None
            else:
                item["uncertain"] = bool(uncertain)

    # --- free-text 模板 ---------------------------------------------------
    _FREETEXT_PRESENT = [
        "{name} is present.",
        "{name} is observed.",
        "{name} is identified.",
        "There is evidence of {name}.",
        "Findings consistent with {name}.",
        "The image shows {name}.",
    ]
    _FREETEXT_ABSENT = [
        "{name} is absent.",
        "No {name} is seen.",
        "No evidence of {name}.",
        "{name} is not identified.",
        "The image is negative for {name}.",
    ]
    _FREETEXT_UNCERTAIN = [
        "{name} is uncertain.",
        "{name} is equivocal.",
        "Possible {name}.",
        "{name} cannot be excluded.",
    ]
    _FREETEXT_MISSING = [
        "{name} is not observed.",
        "{name} is not assessed.",
        "No comment on {name}.",
    ]

    def _create_freetext_string(
        self,
        sample: Dict,
        include_labels: Optional[List[str]] = None,
    ) -> str:
        """
        创建自由文本格式的 target（用于消融实验：对比结构化 JSON vs 自由文本）。
        引入随机句式、随机顺序、随机省略 absent findings，
        使 LLM 预测不确定性高于结构化 JSON，从而产生更噪的梯度信号。
        """
        target_labels = list(
            include_labels if include_labels is not None else self.label_names
        )
        # 随机打乱 finding 顺序
        random.shuffle(target_labels)

        parts = []
        for name in target_labels:
            if name in sample["findings"]:
                item = sample["findings"][name]
                state = self._normalize_state(item.get("state"), missing=False)
                if state == "present":
                    tpl = random.choice(self._FREETEXT_PRESENT)
                elif state == "absent":
                    # 50% 概率省略 absent finding（真实报告不会列出所有阴性）
                    if random.random() < 0.5:
                        continue
                    tpl = random.choice(self._FREETEXT_ABSENT)
                elif state == "uncertain":
                    tpl = random.choice(self._FREETEXT_UNCERTAIN)
                else:
                    tpl = random.choice(self._FREETEXT_MISSING)
            else:
                # missing finding: 70% 概率省略
                if random.random() < 0.7:
                    continue
                tpl = random.choice(self._FREETEXT_MISSING)
            parts.append(tpl.format(name=name))

        if not parts:
            parts.append("No significant findings.")
        return " ".join(parts)

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

        # 创建目标字符串（JSON 或自由文本）
        prompt_text = None
        query_labels = None
        if self.is_train and self.field_query_training_cfg["enabled"]:
            query_labels = self._sample_field_query_labels(answerable)
            if self.target_format == "text":
                target_json = self._create_freetext_string(
                    sample=sample, include_labels=query_labels,
                )
            else:
                target_json = self._create_ums_json_string(
                    sample=sample,
                    include_labels=query_labels,
                    include_missing_for_selected=False,
                )
            prompt_text = self._create_field_query_prompt(query_labels)
        else:
            if self.target_format == "text":
                target_json = self._create_freetext_string(sample)
            else:
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
            "prompt_text": prompt_text,
            "query_labels": query_labels,
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
        "prompt_texts": [item.get("prompt_text") for item in batch],
        "query_labels": [item.get("query_labels") for item in batch],
    }
