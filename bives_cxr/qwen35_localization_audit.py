"""Frozen Qwen3.5 scoring helpers for localization-causality development gates."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


QWEN35_2B_SNAPSHOT_SHA256 = (
    "6b57c58ceb3c97d4199d4e8e32b54f6ff11d0cf3a8bde3a4e940d5775ae12120"
)
QWEN35_AUDIT_PROMPT = (
    "Is the following finding present in this chest radiograph: {statement} "
    "Answer only Yes or No."
)


def deterministic_top_cell_mask(
    sensitivity: np.ndarray,
    *,
    image_height: int,
    image_width: int,
) -> np.ndarray:
    """Expand the highest-sensitivity grid cell with stable row-major ties."""

    values = np.asarray(sensitivity, dtype=np.float64)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("sensitivity must be a non-empty finite 2D array")
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image geometry must be positive")
    rows, columns = values.shape
    flat_index = min(
        range(values.size),
        key=lambda index: (-float(values.flat[index]), index),
    )
    row, column = divmod(flat_index, columns)
    y_edges = np.linspace(0, image_height, rows + 1, dtype=np.int64)
    x_edges = np.linspace(0, image_width, columns + 1, dtype=np.int64)
    mask = np.zeros((image_height, image_width), dtype=bool)
    mask[y_edges[row] : y_edges[row + 1], x_edges[column] : x_edges[column + 1]] = True
    return mask


class Qwen35StatementScorer:
    """Read-only first-token Yes/No scorer with one frozen prompt contract."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str,
        dtype: str = "bf16",
        attention_implementation: str = "eager",
    ) -> None:
        from transformers import AutoProcessor
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5ForConditionalGeneration,
        )

        if dtype not in {"bf16", "fp16"}:
            raise ValueError("Qwen3.5 audit dtype must be bf16 or fp16")
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("the real Qwen3.5 audit scorer requires local CUDA")
        self.dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
        self.model_path = Path(model_path)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=False,
            local_files_only=True,
        )
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attention_implementation,
            local_files_only=True,
        ).to(self.device).eval()
        if self.model.config.model_type != "qwen3_5":
            raise ValueError("local audit model is not Qwen3.5")
        self.no_token_id = self._single_token_id("No")
        self.yes_token_id = self._single_token_id("Yes")
        if self.no_token_id == self.yes_token_id:
            raise ValueError("Yes/No audit tokens must be distinct")

    def _single_token_id(self, text: str) -> int:
        ids = self.processor.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"audit response {text!r} is not one tokenizer token: {ids}")
        return int(ids[0])

    def prompt(self, statement: str) -> str:
        normalized = str(statement).strip()
        if not normalized or "\n" in normalized or "\r" in normalized:
            raise ValueError("statement must be one non-empty line")
        question = QWEN35_AUDIT_PROMPT.format(statement=normalized)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.new("RGB", (1, 1))},
                    {"type": "text", "text": question},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def score(self, images: Sequence[Image.Image], statement: str) -> list[float]:
        if not images:
            raise ValueError("Qwen3.5 audit scoring batch is empty")
        prompt = self.prompt(statement)
        normalized_images = [image.convert("RGB") for image in images]
        batch = self.processor(
            text=[prompt] * len(normalized_images),
            images=normalized_images,
            padding=True,
            return_tensors="pt",
        )
        batch = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
        last_indices = batch["attention_mask"].sum(dim=1).long() - 1
        with torch.inference_mode():
            output = self.model(**batch, use_cache=False)
        row_indices = torch.arange(len(normalized_images), device=self.device)
        last_logits = output.logits[row_indices, last_indices].float()
        binary_logits = last_logits[:, [self.no_token_id, self.yes_token_id]]
        probabilities = torch.softmax(binary_logits, dim=1)[:, 1]
        if not bool(torch.isfinite(probabilities).all()):
            raise ValueError("Qwen3.5 audit score is non-finite")
        return [float(value) for value in probabilities.detach().cpu()]

    def identity(self) -> dict[str, Any]:
        return {
            "family": "Qwen3.5",
            "scale": "2B",
            "model_type": self.model.config.model_type,
            "snapshot_sha256": QWEN35_2B_SNAPSHOT_SHA256,
            "score": "softmax(No,Yes)[Yes] at first assistant token",
            "prompt_template": QWEN35_AUDIT_PROMPT,
            "no_token_id": self.no_token_id,
            "yes_token_id": self.yes_token_id,
        }
