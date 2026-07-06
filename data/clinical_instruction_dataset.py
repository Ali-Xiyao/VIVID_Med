"""Qwen3-VL clinical instruction dataset and collator."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset


ANSWER_TYPE_ALIASES = {
    "report_consistency": "image_report_consistency",
    "image_report_consistency": "image_report_consistency",
    "laterality": "laterality_location",
    "location": "laterality_location",
}


def normalize_answer_type(value: Any) -> str:
    raw = str(value or "unknown").strip()
    return ANSWER_TYPE_ALIASES.get(raw, raw)


def first_non_empty(record: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in {None, ""}:
            return value
    return default


def read_jsonl(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if max_samples is not None and len(rows) >= max_samples:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


class ClinicalInstructionDataset(Dataset):
    """Load report-grounded clinical instruction rows for Qwen3-VL."""

    def __init__(
        self,
        data_root: str,
        instruction_jsonl_path: str,
        max_samples: int | None = None,
        image_fallback_size: int = 448,
    ) -> None:
        self.data_root = Path(data_root)
        self.instruction_jsonl_path = Path(instruction_jsonl_path)
        self.image_fallback_size = int(image_fallback_size)
        self.records = read_jsonl(self.instruction_jsonl_path, max_samples=max_samples)
        print(f"Loaded {len(self.records)} clinical instruction records from {self.instruction_jsonl_path}")

    def __len__(self) -> int:
        return len(self.records)

    def _image_path(self, record: dict[str, Any]) -> Path:
        image_path = Path(str(first_non_empty(record, ["image_path", "path", "original_path"], "")))
        if image_path.is_absolute():
            return image_path
        return self.data_root / image_path

    def _load_image(self, path: Path) -> Image.Image:
        try:
            return Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001 - keep long training runs moving while preserving metadata.
            print(f"Error loading image {path}: {exc}")
            size = self.image_fallback_size
            return Image.new("RGB", (size, size), (0, 0, 0))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = dict(self.records[idx])
        image_path = self._image_path(record)
        hard_negative_image_path_raw = first_non_empty(record, ["hard_negative_image_path"], "")
        hard_negative_paths_raw = record.get("hard_negative_image_paths") or record.get("hard_negative_paths") or []
        if isinstance(hard_negative_paths_raw, str):
            hard_negative_paths = [item.strip() for item in re.split(r"[|;,]", hard_negative_paths_raw) if item.strip()]
        else:
            hard_negative_paths = [str(item) for item in hard_negative_paths_raw if item]
        if hard_negative_image_path_raw:
            hard_negative_paths.insert(0, str(hard_negative_image_path_raw))

        hard_negative_images = []
        hard_negative_image_paths = []
        seen_negative_paths = set()
        for raw_path in hard_negative_paths:
            hard_negative_path = Path(str(raw_path))
            if not hard_negative_path.is_absolute():
                hard_negative_path = self.data_root / hard_negative_path
            normalized_path = str(hard_negative_path)
            if normalized_path in seen_negative_paths:
                continue
            seen_negative_paths.add(normalized_path)
            hard_negative_images.append(self._load_image(hard_negative_path))
            hard_negative_image_paths.append(normalized_path)
        hard_negative_image = hard_negative_images[0] if hard_negative_images else None
        hard_negative_image_path = hard_negative_image_paths[0] if hard_negative_image_paths else ""
        answer_type = normalize_answer_type(record.get("answer_type"))
        report_text = first_non_empty(record, ["report_text", "report"], "")
        evidence_span = first_non_empty(record, ["evidence_span", "evidence_phrase"], None)
        source = first_non_empty(record, ["source", "source_mode"], None)
        instruction_id = first_non_empty(record, ["instruction_id"], f"{record.get('sample_id', idx)}::{idx}")
        negative_answer = first_non_empty(record, ["negative_answer", "counterfactual_answer", "negative_option"], None)

        normalized = {
            **record,
            "instruction_id": instruction_id,
            "answer_type": answer_type,
            "report_text": report_text,
            "evidence_span": evidence_span,
            "source": source,
            "image_path": str(image_path),
        }

        return {
            "image": self._load_image(image_path),
            "question": str(record.get("question") or ""),
            "answer": str(first_non_empty(record, ["answer", "answer_short"], "")),
            "answer_short": str(record.get("answer_short") or ""),
            "hard_negative_image": hard_negative_image,
            "hard_negative_image_path": hard_negative_image_path,
            "hard_negative_images": hard_negative_images,
            "hard_negative_image_paths": hard_negative_image_paths,
            "hard_negative_expected_answer": record.get("hard_negative_expected_answer"),
            "negative_answer": negative_answer,
            "instruction_id": instruction_id,
            "sample_id": record.get("sample_id"),
            "image_path": str(image_path),
            "answer_type": answer_type,
            "finding": record.get("finding"),
            "state": record.get("state"),
            "visual_dependency": record.get("visual_dependency"),
            "quality_flags": list(record.get("quality_flags") or []),
            "curriculum_stage": record.get("curriculum_stage"),
            "curriculum_start_step": record.get("curriculum_start_step"),
            "curriculum_end_step": record.get("curriculum_end_step"),
            "metadata": normalized,
        }


class Qwen3VLInstructionCollator:
    """Build Qwen3-VL chat tensors with labels masked to assistant answer tokens."""

    def __init__(
        self,
        processor: Any,
        max_length: int | None = None,
        loss_weighting: dict[str, Any] | None = None,
        loss_masking: dict[str, Any] | None = None,
        in_batch_negative: dict[str, Any] | None = None,
    ) -> None:
        self.processor = processor
        self.max_length = max_length
        self.loss_weighting = loss_weighting or {}
        self.loss_masking = loss_masking or {}
        self.in_batch_negative = in_batch_negative or {}
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "right"

    def _messages(self, image: Image.Image, question: str, answer: str | None = None) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        if answer is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        return messages

    def _chat_text(self, item: dict[str, Any], include_answer: bool, answer_override: str | None = None) -> str:
        messages = self._messages(
            image=item["image"],
            question=item["question"],
            answer=(answer_override if answer_override is not None else item["answer"]) if include_answer else None,
        )
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not include_answer,
        )

    def _loss_mask_mode(self) -> str:
        return str(self.loss_masking.get("mode") or self.loss_masking.get("loss_mask") or "answer_only")

    def _token_sequences_for_value(self, value: str) -> list[list[int]]:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            return []
        variants = [value, f" {value}", f'"{value}"', f' "{value}"']
        sequences: list[list[int]] = []
        seen = set()
        for text in variants:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                continue
            key = tuple(int(token_id) for token_id in token_ids)
            if key in seen:
                continue
            seen.add(key)
            sequences.append(list(key))
        return sequences

    def _apply_loss_masking(
        self,
        input_ids_row: torch.Tensor,
        labels_row: torch.Tensor,
        loss_weights_row: torch.Tensor,
        prompt_len: int,
    ) -> None:
        mode = self._loss_mask_mode()
        if mode in {"", "answer_only", "assistant_answer"}:
            return

        active_mask = labels_row != -100
        if not bool(active_mask.any()):
            return

        tokenizer = getattr(self.processor, "tokenizer", None)
        if mode in {"json_value_only", "value_only", "state_token_only"}:
            values = self.loss_masking.get("values") or ["present", "absent", "uncertain", "null", "true", "false"]
            keep_mask = torch.zeros_like(active_mask, dtype=torch.bool)
            token_ids = [int(token_id) for token_id in input_ids_row.tolist()]
            for raw_value in values:
                for sequence in self._token_sequences_for_value(str(raw_value)):
                    seq_len = len(sequence)
                    if seq_len == 0 or seq_len > len(token_ids):
                        continue
                    for start in range(max(0, prompt_len - seq_len), len(token_ids) - seq_len + 1):
                        if token_ids[start:start + seq_len] != sequence:
                            continue
                        end = start + seq_len
                        if bool(active_mask[start:end].any()):
                            keep_mask[start:end] = True
            drop_mask = active_mask & ~keep_mask
            labels_row[drop_mask] = -100
            loss_weights_row[drop_mask] = 0.0
            return

        if mode in {"json_no_punct", "no_punct"}:
            if tokenizer is None:
                return
            for token_index in range(prompt_len, int(labels_row.shape[0])):
                if int(labels_row[token_index]) == -100:
                    continue
                piece = tokenizer.decode([int(input_ids_row[token_index])], skip_special_tokens=False)
                if not any(char.isalnum() for char in piece):
                    labels_row[token_index] = -100
                    loss_weights_row[token_index] = 0.0
            return

        raise ValueError(f"Unsupported loss_masking.mode: {mode}")

    def _prompt_length(self, item: dict[str, Any]) -> int:
        text = self._chat_text(item, include_answer=False)
        encoded = self.processor(text=[text], images=[item["image"]], return_tensors="pt")
        return int(encoded["input_ids"].shape[1])

    def _row_weight(self, item: dict[str, Any]) -> float:
        cfg = self.loss_weighting
        if not cfg or not bool(cfg.get("enabled", False)):
            return 1.0
        weight = float(cfg.get("base_weight", 1.0))
        answer_type_weights = cfg.get("answer_type_weights", {}) or {}
        visual_weights = cfg.get("visual_dependency_weights", {}) or {}
        flag_weights = cfg.get("quality_flag_weights", {}) or {}
        weight *= float(answer_type_weights.get(str(item.get("answer_type")), 1.0))
        weight *= float(visual_weights.get(str(item.get("visual_dependency")), 1.0))
        for flag in item.get("quality_flags", []) or []:
            weight *= float(flag_weights.get(str(flag), 1.0))
        return max(weight, 0.0)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        full_texts = [self._chat_text(item, include_answer=True) for item in batch]
        images = [item["image"] for item in batch]
        processor_kwargs: dict[str, Any] = {
            "text": full_texts,
            "images": images,
            "return_tensors": "pt",
            "padding": True,
        }
        if self.max_length is not None:
            processor_kwargs["max_length"] = int(self.max_length)
            processor_kwargs["truncation"] = True
        inputs = self.processor(**processor_kwargs)

        input_ids = inputs["input_ids"]
        labels = input_ids.clone()
        loss_weights = torch.ones_like(input_ids, dtype=torch.float32)
        pad_token_id = getattr(getattr(self.processor, "tokenizer", None), "pad_token_id", None)
        if pad_token_id is not None:
            labels[input_ids == pad_token_id] = -100
            loss_weights[input_ids == pad_token_id] = 0.0

        for row_idx, item in enumerate(batch):
            prompt_len = min(self._prompt_length(item), labels.shape[1])
            labels[row_idx, :prompt_len] = -100
            loss_weights[row_idx, :prompt_len] = 0.0
            self._apply_loss_masking(input_ids[row_idx], labels[row_idx], loss_weights[row_idx], prompt_len)
            row_weight = self._row_weight(item)
            answer_mask = labels[row_idx] != -100
            loss_weights[row_idx, answer_mask] *= row_weight

        inputs["labels"] = labels
        inputs["loss_weights"] = loss_weights
        inputs["instruction_ids"] = [item["instruction_id"] for item in batch]
        inputs["sample_ids"] = [item["sample_id"] for item in batch]
        inputs["image_paths"] = [item["image_path"] for item in batch]
        inputs["answer_types"] = [item["answer_type"] for item in batch]
        inputs["findings"] = [item["finding"] for item in batch]
        inputs["states"] = [item["state"] for item in batch]
        inputs["visual_dependencies"] = [item["visual_dependency"] for item in batch]
        inputs["curriculum_stages"] = [item.get("curriculum_stage") for item in batch]
        inputs["metadata"] = [item["metadata"] for item in batch]
        negative_items: list[dict[str, Any]] = []
        negative_images = []
        negative_paths = []
        for item in batch:
            item_negative_images = item.get("hard_negative_images") or []
            item_negative_paths = item.get("hard_negative_image_paths") or []
            for neg_idx, negative_image in enumerate(item_negative_images):
                negative_items.append(item)
                negative_images.append(negative_image)
                negative_paths.append(item_negative_paths[neg_idx] if neg_idx < len(item_negative_paths) else "")
        if bool(self.in_batch_negative.get("enabled", False)) and len(batch) > 1:
            offset = int(self.in_batch_negative.get("offset", 1))
            for idx, item in enumerate(batch):
                partner_idx = (idx + max(1, offset)) % len(batch)
                for _ in range(len(batch)):
                    if partner_idx != idx and batch[partner_idx].get("sample_id") != item.get("sample_id"):
                        break
                    partner_idx = (partner_idx + 1) % len(batch)
                if partner_idx == idx or batch[partner_idx].get("sample_id") == item.get("sample_id"):
                    continue
                partner = batch[partner_idx]
                negative_items.append(item)
                negative_images.append(partner["image"])
                negative_paths.append(partner["image_path"])
        hard_negative_available = [bool(item.get("hard_negative_images")) for item in batch]
        if negative_items:
            negative_texts = [self._chat_text(item, include_answer=True) for item in negative_items]
            negative_inputs = self.processor(
                text=negative_texts,
                images=negative_images,
                return_tensors="pt",
                padding=True,
                **({"max_length": int(self.max_length), "truncation": True} if self.max_length is not None else {}),
            )
            negative_labels = negative_inputs["input_ids"].clone()
            negative_loss_weights = torch.ones_like(negative_inputs["input_ids"], dtype=torch.float32)
            if pad_token_id is not None:
                negative_labels[negative_inputs["input_ids"] == pad_token_id] = -100
                negative_loss_weights[negative_inputs["input_ids"] == pad_token_id] = 0.0
            for row_idx, item in enumerate(negative_items):
                prompt_len = min(self._prompt_length(item), negative_labels.shape[1])
                negative_labels[row_idx, :prompt_len] = -100
                negative_loss_weights[row_idx, :prompt_len] = 0.0
                self._apply_loss_masking(
                    negative_inputs["input_ids"][row_idx],
                    negative_labels[row_idx],
                    negative_loss_weights[row_idx],
                    prompt_len,
                )
                row_weight = self._row_weight(item)
                answer_mask = negative_labels[row_idx] != -100
                negative_loss_weights[row_idx, answer_mask] *= row_weight
            for key, value in negative_inputs.items():
                inputs[f"negative_{key}"] = value
            inputs["negative_labels"] = negative_labels
            inputs["negative_loss_weights"] = negative_loss_weights
            inputs["hard_negative_available"] = torch.tensor(hard_negative_available, dtype=torch.bool)
            inputs["hard_negative_image_paths"] = [item.get("hard_negative_image_path", "") for item in batch]
            inputs["hard_negative_image_paths_flat"] = negative_paths
            inputs["hard_negative_expected_answers"] = [item.get("hard_negative_expected_answer") for item in batch]

        answer_negative_items = [item for item in batch if item.get("negative_answer") not in {None, ""}]
        if answer_negative_items:
            answer_negative_texts = [
                self._chat_text(item, include_answer=True, answer_override=str(item["negative_answer"]))
                for item in answer_negative_items
            ]
            answer_negative_inputs = self.processor(
                text=answer_negative_texts,
                images=[item["image"] for item in answer_negative_items],
                return_tensors="pt",
                padding=True,
                **({"max_length": int(self.max_length), "truncation": True} if self.max_length is not None else {}),
            )
            answer_negative_labels = answer_negative_inputs["input_ids"].clone()
            answer_negative_loss_weights = torch.ones_like(answer_negative_inputs["input_ids"], dtype=torch.float32)
            if pad_token_id is not None:
                answer_negative_labels[answer_negative_inputs["input_ids"] == pad_token_id] = -100
                answer_negative_loss_weights[answer_negative_inputs["input_ids"] == pad_token_id] = 0.0
            for row_idx, item in enumerate(answer_negative_items):
                prompt_len = min(self._prompt_length(item), answer_negative_labels.shape[1])
                answer_negative_labels[row_idx, :prompt_len] = -100
                answer_negative_loss_weights[row_idx, :prompt_len] = 0.0
                self._apply_loss_masking(
                    answer_negative_inputs["input_ids"][row_idx],
                    answer_negative_labels[row_idx],
                    answer_negative_loss_weights[row_idx],
                    prompt_len,
                )
                row_weight = self._row_weight(item)
                answer_mask = answer_negative_labels[row_idx] != -100
                answer_negative_loss_weights[row_idx, answer_mask] *= row_weight
            for key, value in answer_negative_inputs.items():
                inputs[f"answer_negative_{key}"] = value
            inputs["answer_negative_labels"] = answer_negative_labels
            inputs["answer_negative_loss_weights"] = answer_negative_loss_weights
            inputs["answer_negative_answers"] = [str(item["negative_answer"]) for item in answer_negative_items]
        return inputs
