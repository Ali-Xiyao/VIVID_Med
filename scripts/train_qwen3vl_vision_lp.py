"""Train a linear probe on a Qwen3-VL vision tower."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import compute_classification_metrics
from scripts.train_qwen3vl_clinical_instruction import DEFAULT_MODEL_PATH, choose_dtype


FINDING_NAMES = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
COMMON_LABELS = [
    "No Finding",
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if max_samples is not None and len(rows) >= max_samples:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


class UMSPILDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        ums_jsonl_path: str,
        label_names: list[str],
        max_samples: int | None = None,
        fallback_size: int = 448,
    ) -> None:
        self.data_root = Path(data_root)
        self.label_names = label_names
        self.samples = read_jsonl(Path(ums_jsonl_path), max_samples=max_samples)
        self.fallback_size = int(fallback_size)
        self._image_index_cache: dict[str, Path] = {}
        print(f"Loaded {len(self.samples)} UMS samples from {ums_jsonl_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _image_path(self, sample: dict[str, Any]) -> Path:
        extensions = sample.get("extensions") or {}
        raw = extensions.get("original_path") or sample.get("image_path") or ""
        path = Path(str(raw))
        if path.is_absolute():
            return path
        image_index = extensions.get("image_index")
        if not raw and image_index:
            image_index = str(image_index)
            cached = self._image_index_cache.get(image_index)
            if cached is not None:
                return cached
            nih_root = self.data_root / "NIH Chest X-rays"
            candidates = [self.data_root / image_index, self.data_root / "images" / image_index]
            candidates.extend(nih_root / f"images_{idx:03d}" / "images" / image_index for idx in range(1, 13))
            for candidate in candidates:
                if candidate.exists():
                    self._image_index_cache[image_index] = candidate
                    return candidate
            self._image_index_cache[image_index] = candidates[-1]
            return candidates[-1]
        return self.data_root / path

    def _load_image(self, path: Path) -> Image.Image:
        try:
            return Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001 - preserve run continuity.
            print(f"Error loading image {path}: {exc}")
            return Image.new("RGB", (self.fallback_size, self.fallback_size), (0, 0, 0))

    def _labels(self, sample: dict[str, Any]) -> torch.Tensor:
        labels = torch.full((len(self.label_names),), float("nan"))
        findings = sample.get("findings") or {}
        for idx, name in enumerate(self.label_names):
            item = findings.get(name) or {}
            state = item.get("state")
            if state == "present":
                labels[idx] = 1.0
            elif state == "absent":
                labels[idx] = 0.0
            elif state == "uncertain":
                labels[idx] = -1.0
        return labels

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        path = self._image_path(sample)
        return {
            "image": self._load_image(path),
            "labels": self._labels(sample),
            "sample_id": (sample.get("extensions") or {}).get("sample_id") or str(idx),
            "image_path": str(path),
        }


class Qwen3VLLPCollator:
    def __init__(self, processor: Any, prompt: str = "Classify the chest X-ray findings.") -> None:
        self.processor = processor
        self.prompt = prompt

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts = []
        images = []
        for item in batch:
            image = item["image"]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            images.append(image)
        encoded = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        encoded["labels"] = torch.stack([item["labels"] for item in batch])
        encoded["sample_ids"] = [item["sample_id"] for item in batch]
        encoded["image_paths"] = [item["image_path"] for item in batch]
        return encoded


class Qwen3VLVisionLinearProbe(nn.Module):
    def __init__(self, visual: nn.Module, feature_dim: int, num_labels: int, freeze_backbone: bool = True) -> None:
        super().__init__()
        self.visual = visual
        self.head = nn.Linear(feature_dim, num_labels)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.visual.parameters():
                param.requires_grad = False

    def _pool_features(self, visual_outputs: Any, grid_thw: torch.Tensor, batch_size: int) -> torch.Tensor:
        hidden = visual_outputs.last_hidden_state
        counts = grid_thw.prod(dim=1).long().tolist()
        if sum(counts) != hidden.shape[0]:
            chunks = torch.chunk(hidden, batch_size, dim=0)
        else:
            chunks = torch.split(hidden, counts, dim=0)
        return torch.stack([chunk.mean(dim=0) for chunk in chunks], dim=0)

    def extract_features(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.visual(pixel_values, grid_thw=image_grid_thw)
        else:
            outputs = self.visual(pixel_values, grid_thw=image_grid_thw)
        return self._pool_features(outputs, image_grid_thw, int(image_grid_thw.shape[0]))

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(pixel_values, image_grid_thw)
        return self.head(features.float())


def load_model_and_processor(config: dict[str, Any], device: torch.device) -> tuple[Any, Any]:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_cfg = config.get("model", {})
    model_path = str(model_cfg.get("model_path", DEFAULT_MODEL_PATH))
    dtype = choose_dtype(model_cfg.get("dtype", "bf16"))
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    checkpoint_path = model_cfg.get("vision_checkpoint")
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("trainable_state_dict", checkpoint)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(json.dumps({"vision_checkpoint": str(checkpoint_path), "missing_keys": len(missing), "unexpected_keys": len(unexpected)}))
    model.to(device)
    model.eval()
    return model, processor


def create_dataloaders(config: dict[str, Any], processor: Any) -> tuple[DataLoader, DataLoader, list[str]]:
    data_cfg = config["data"]
    if data_cfg.get("selected_labels"):
        label_names = list(data_cfg["selected_labels"])
    else:
        label_names = COMMON_LABELS if data_cfg.get("use_common_labels_only", False) else FINDING_NAMES
    train_dataset = UMSPILDataset(
        data_root=data_cfg["data_root"],
        ums_jsonl_path=data_cfg["train_ums_path"],
        label_names=label_names,
        max_samples=data_cfg.get("max_train_samples"),
    )
    val_dataset = UMSPILDataset(
        data_root=data_cfg["data_root"],
        ums_jsonl_path=data_cfg["val_ums_path"],
        label_names=label_names,
        max_samples=data_cfg.get("max_val_samples"),
    )
    collator = Qwen3VLLPCollator(processor=processor, prompt=data_cfg.get("processor_prompt", "Classify the chest X-ray findings."))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"].get("eval_batch_size", config["training"]["batch_size"])),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=collator,
        pin_memory=True,
    )
    return train_loader, val_loader, label_names


def move_tensors_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def prepare_labels_for_loss(labels: torch.Tensor, policy: str) -> tuple[torch.Tensor, torch.Tensor]:
    labels = labels.clone()
    mask = torch.isfinite(labels)
    if policy == "ignore":
        mask = mask & (labels != -1)
        labels[labels == -1] = 0.0
    elif policy == "positive":
        labels[labels == -1] = 1.0
    elif policy == "negative":
        labels[labels == -1] = 0.0
    else:
        raise ValueError(f"Unknown uncertain_policy: {policy}")
    labels = torch.nan_to_num(labels, nan=0.0)
    return labels, mask


def prepare_labels_for_metrics(labels: np.ndarray, policy: str) -> np.ndarray:
    labels = labels.copy()
    if policy == "ignore":
        labels[labels == -1] = np.nan
    elif policy == "positive":
        labels[labels == -1] = 1.0
    elif policy == "negative":
        labels[labels == -1] = 0.0
    else:
        raise ValueError(f"Unknown uncertain_policy: {policy}")
    return labels


def compute_loss(logits: torch.Tensor, labels: torch.Tensor, policy: str) -> torch.Tensor:
    labels, mask = prepare_labels_for_loss(labels, policy)
    loss = nn.BCEWithLogitsLoss(reduction="none")(logits, labels)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(model: Qwen3VLVisionLinearProbe, dataloader: DataLoader, device: torch.device, policy: str, label_names: list[str]) -> dict[str, Any]:
    model.eval()
    losses = []
    probs = []
    labels = []
    for batch in tqdm(dataloader, desc="Validating", leave=False):
        batch = move_tensors_to_device(batch, device)
        logits = model(pixel_values=batch["pixel_values"], image_grid_thw=batch["image_grid_thw"])
        loss = compute_loss(logits, batch["labels"], policy)
        losses.append(float(loss.detach().cpu()))
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(prepare_labels_for_metrics(batch["labels"].cpu().numpy(), policy))
    y_prob = np.concatenate(probs, axis=0)
    y_true = np.concatenate(labels, axis=0)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_classification_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob, label_names=label_names, threshold=0.5)
    return {"val_loss": float(np.mean(losses)) if losses else float("nan"), "metrics": metrics}


def infer_feature_dim(visual: nn.Module, dataloader: DataLoader, device: torch.device) -> int:
    batch = next(iter(dataloader))
    batch = move_tensors_to_device(batch, device)
    with torch.no_grad():
        outputs = visual(batch["pixel_values"], grid_thw=batch["image_grid_thw"])
    hidden = outputs.last_hidden_state
    return int(hidden.shape[-1])


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=to_jsonable), encoding="utf-8")


def apply_debug_overrides(config: dict[str, Any]) -> None:
    data_cfg = config["data"]
    train_cfg = config["training"]
    data_cfg["max_train_samples"] = min(int(data_cfg.get("max_train_samples") or 8), 8)
    data_cfg["max_val_samples"] = min(int(data_cfg.get("max_val_samples") or 4), 4)
    data_cfg["num_workers"] = 0
    train_cfg["batch_size"] = 1
    train_cfg["eval_batch_size"] = 1
    train_cfg["max_steps"] = min(int(train_cfg.get("max_steps", 2)), 2)
    train_cfg["eval_interval"] = 1
    train_cfg["save_interval"] = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.debug:
        print("Debug mode enabled")
        apply_debug_overrides(config)
    if args.device:
        config["device"] = args.device
    set_seed(int(config.get("seed", 42)))
    requested_device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    output_dir = Path(config["training"]["output_dir"])
    if (output_dir / "metrics_final.json").exists():
        raise SystemExit(f"{output_dir / 'metrics_final.json'} already exists; remove it manually to rerun.")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / "config.yaml")
    save_json(output_dir / "resolved_config.json", config)

    base_model, processor = load_model_and_processor(config, device)
    train_loader, val_loader, label_names = create_dataloaders(config, processor)
    feature_dim = infer_feature_dim(base_model.model.visual, train_loader, device)
    model = Qwen3VLVisionLinearProbe(
        visual=base_model.model.visual,
        feature_dim=feature_dim,
        num_labels=len(label_names),
        freeze_backbone=bool(config.get("model", {}).get("freeze_backbone", True)),
    ).to(device)
    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )

    policy = config["training"].get("uncertain_policy", "ignore")
    max_steps = int(config["training"]["max_steps"])
    eval_interval = int(config["training"].get("eval_interval", 100))
    save_interval = int(config["training"].get("save_interval", 500))
    global_step = 0
    best_val_loss = float("inf")
    started = time.time()
    model.train()

    while global_step < max_steps:
        for batch in tqdm(train_loader, desc="Training", leave=False):
            batch = move_tensors_to_device(batch, device)
            logits = model(pixel_values=batch["pixel_values"], image_grid_thw=batch["image_grid_thw"])
            loss = compute_loss(logits, batch["labels"], policy)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step % int(config["training"].get("log_interval", 10)) == 0:
                print(json.dumps({"global_step": global_step, "train_loss": float(loss.detach().cpu())}))
            if global_step % eval_interval == 0:
                result = evaluate(model, val_loader, device, policy, label_names)
                print(json.dumps({"global_step": global_step, "val_loss": result["val_loss"]}))
                if result["val_loss"] < best_val_loss:
                    best_val_loss = result["val_loss"]
                    torch.save({"head": model.head.state_dict(), "feature_dim": feature_dim, "label_names": label_names}, output_dir / "best_probe.pt")
            if global_step % save_interval == 0:
                torch.save({"head": model.head.state_dict(), "feature_dim": feature_dim, "label_names": label_names}, output_dir / f"probe_step_{global_step}.pt")
            if global_step >= max_steps:
                break

    final_result = evaluate(model, val_loader, device, policy, label_names)
    metrics = {
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "final_val_loss": final_result["val_loss"],
        "elapsed_seconds": time.time() - started,
        "train_records": len(train_loader.dataset),
        "val_records": len(val_loader.dataset),
        "feature_dim": feature_dim,
        "label_names": label_names,
        "metrics": final_result["metrics"],
    }
    torch.save({"head": model.head.state_dict(), "feature_dim": feature_dim, "label_names": label_names}, output_dir / "final_probe.pt")
    save_json(output_dir / "metrics_final.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
