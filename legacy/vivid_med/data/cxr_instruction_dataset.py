"""CXR instruction dataset for image + clinical question -> answer training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import get_train_transforms, get_val_transforms


class CXRInstructionDataset(Dataset):
    """Load one clinical instruction record per row."""

    def __init__(
        self,
        data_root: str,
        instruction_jsonl_path: str,
        transform=None,
        is_train: bool = True,
        image_size: int = 224,
        max_samples: int | None = None,
        prompt_template: str = "Question: {question}\nAnswer: ",
        append_eos: bool = False,
        eos_token: str = "",
    ):
        self.data_root = Path(data_root)
        self.instruction_jsonl_path = Path(instruction_jsonl_path)
        self.is_train = is_train
        self.prompt_template = prompt_template
        self.append_eos = append_eos
        self.eos_token = eos_token
        self.transform = transform or (
            get_train_transforms(image_size) if is_train else get_val_transforms(image_size)
        )
        self.records = self._load_jsonl(self.instruction_jsonl_path, max_samples)
        print(f"Loaded {len(self.records)} instruction records from {self.instruction_jsonl_path}")

    def _load_jsonl(self, path: Path, max_samples: int | None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if max_samples is not None and idx >= max_samples:
                    break
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def __len__(self) -> int:
        return len(self.records)

    def _image_path(self, record: dict[str, Any]) -> Path:
        image_path = Path(str(record.get("image_path") or ""))
        if image_path.is_absolute():
            return image_path
        return self.data_root / image_path

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        path = self._image_path(record)
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001 - data loader should keep runs moving.
            print(f"Error loading image {path}: {exc}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform:
            image = self.transform(image)

        question = str(record.get("question") or "")
        answer = str(record.get("answer") or "")
        if self.append_eos and self.eos_token and not answer.endswith(self.eos_token):
            answer = answer + self.eos_token
        prompt_text = self.prompt_template.format(question=question)

        return {
            "image": image,
            "prompt_text": prompt_text,
            "target_text": answer,
            "instruction_id": record.get("instruction_id"),
            "record_index": idx,
            "sample_id": record.get("sample_id"),
            "original_path": record.get("image_path"),
            "answer_type": record.get("answer_type"),
            "finding": record.get("finding"),
            "state": record.get("state"),
            "visual_dependency": record.get("visual_dependency"),
            "quality_flags": record.get("quality_flags", []),
            "token_weight_mask": None,
            "metadata": record,
        }


def instruction_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([item["image"] for item in batch])
    return {
        "images": images,
        "target_jsons": [item["target_text"] for item in batch],
        "prompt_texts": [item["prompt_text"] for item in batch],
        "instruction_ids": [item["instruction_id"] for item in batch],
        "record_indices": [item["record_index"] for item in batch],
        "sample_ids": [item["sample_id"] for item in batch],
        "original_paths": [item["original_path"] for item in batch],
        "answer_types": [item["answer_type"] for item in batch],
        "findings": [item["finding"] for item in batch],
        "states": [item["state"] for item in batch],
        "visual_dependencies": [item["visual_dependency"] for item in batch],
        "quality_flags": [item["quality_flags"] for item in batch],
        "token_weight_mask": [item["token_weight_mask"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }
