"""Train the Qwen3.5-only BiVES-CXR evidence model."""

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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bives_cxr.backbones import Qwen35VisionAdapter, load_qwen35_model_and_processor
from bives_cxr.data import BiVESManifestDataset
from bives_cxr.losses import BiVESLoss, BiVESLossConfig
from bives_cxr.metrics import intervention_metrics
from bives_cxr.model import BiVESCXR, BiVESModelConfig


class BiVESExperiment(nn.Module):
    """Qwen3.5 patch encoder plus canonical statement table and BiVES head."""

    def __init__(
        self,
        backbone: Qwen35VisionAdapter,
        num_statements: int,
        statement_dim: int,
        head_config: BiVESModelConfig,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.statement_table = nn.Embedding(num_statements, statement_dim)
        self.head = BiVESCXR(head_config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        statement_indices: torch.Tensor,
        run_interventions: bool = True,
    ) -> tuple[dict[str, dict[str, torch.Tensor] | torch.Tensor], list[tuple[int, int]]]:
        patches = self.backbone(pixel_values, image_grid_thw)
        statement_embeddings = self.statement_table(statement_indices)
        outputs = self.head(
            patches.tokens,
            statement_embeddings,
            patches.valid_mask,
            run_interventions=run_interventions,
        )
        return outputs, patches.grid_hw


class Qwen35BiVESCollator:
    def __init__(self, processor: Any, image_size: int = 448) -> None:
        self.processor = processor
        self.image_size = int(image_size)

    def _letterbox(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        scale = min(self.image_size / image.width, self.image_size / image.height)
        resized = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.BICUBIC,
        )
        canvas = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
        left = (self.image_size - resized.width) // 2
        top = (self.image_size - resized.height) // 2
        canvas.paste(resized, (left, top))
        return canvas

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts = []
        images = []
        for item in batch:
            image = self._letterbox(item["image"])
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": item["statement_text"]},
                    ],
                }
            ]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
            images.append(image)
        encoded = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        encoded["statement_indices"] = torch.tensor([item["statement_index"] for item in batch], dtype=torch.long)
        encoded["targets"] = torch.tensor([item["state_index"] for item in batch], dtype=torch.long)
        encoded["sample_ids"] = [str(item["sample_id"]) for item in batch]
        return encoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Run the synthetic CPU core smoke without loading Qwen3.5.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def require_uniform_grid(grid_hw: list[tuple[int, int]]) -> tuple[int, int]:
    unique = set(grid_hw)
    if len(unique) != 1:
        raise ValueError(f"BiVES batches currently require one padded image grid; got {sorted(unique)}")
    return grid_hw[0]


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


@torch.no_grad()
def evaluate(
    experiment: BiVESExperiment,
    loader: DataLoader,
    loss_fn: BiVESLoss,
    device: torch.device,
) -> dict[str, float]:
    experiment.eval()
    totals: list[float] = []
    correct = 0
    samples = 0
    mechanism: dict[str, list[float]] = {}
    for batch in loader:
        batch = move_to_device(batch, device)
        outputs, grids = experiment(
            batch["pixel_values"],
            batch["image_grid_thw"],
            batch["statement_indices"],
            run_interventions=True,
        )
        losses = loss_fn(outputs, batch["targets"], require_uniform_grid(grids))
        totals.append(float(losses["total"].cpu()))
        probabilities = outputs["original"]["state_probs"]
        correct += int((probabilities.argmax(dim=-1) == batch["targets"]).sum().item())
        samples += int(batch["targets"].numel())
        for key, value in intervention_metrics(outputs, batch["targets"]).items():
            mechanism.setdefault(key, []).append(value)
    experiment.train()
    result = {
        "loss": float(np.mean(totals)) if totals else float("nan"),
        "accuracy": float(correct / max(samples, 1)),
    }
    result.update({key: float(np.mean(values)) for key, values in mechanism.items()})
    return result


def synthetic_smoke() -> None:
    from scripts.smoke_bives_cxr import main

    main()


def main() -> None:
    args = parse_args()
    if args.smoke:
        synthetic_smoke()
        return
    if args.config is None:
        raise SystemExit("--config is required unless --smoke is used")

    config = load_config(args.config)
    if str(config["model"]["family"]).lower() != "qwen3.5":
        raise ValueError("active BiVES-CXR configs are Qwen3.5-only")
    if args.debug:
        config["data"]["max_train_samples"] = min(int(config["data"].get("max_train_samples") or 8), 8)
        config["data"]["max_val_samples"] = min(int(config["data"].get("max_val_samples") or 4), 4)
        config["training"]["max_steps"] = min(int(config["training"].get("max_steps", 2)), 2)
        config["training"]["batch_size"] = 1
        config["training"]["eval_batch_size"] = 1
        config["training"]["eval_interval"] = 1
        config["training"]["output_dir"] = str(config["training"]["output_dir"]).rstrip("/\\") + "_debug"

    set_seed(int(config.get("seed", 17)))
    device = torch.device(str(config.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")))
    output_dir = Path(config["training"]["output_dir"])
    if (output_dir / "metrics_final.json").exists():
        raise SystemExit(f"{output_dir} already contains metrics_final.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / "config.yaml")

    train_dataset = BiVESManifestDataset(config["data"]["train_manifest"], config["data"]["data_root"])
    val_dataset = BiVESManifestDataset(
        config["data"]["val_manifest"],
        config["data"]["data_root"],
        statement_to_index=train_dataset.statement_to_index,
    )
    if args.debug:
        train_dataset.rows = train_dataset.rows[: config["data"]["max_train_samples"]]
        val_dataset.rows = val_dataset.rows[: config["data"]["max_val_samples"]]

    qwen_model, processor, qwen_config = load_qwen35_model_and_processor(
        config["model"]["path"],
        dtype=str(config["model"].get("dtype", "bf16")),
        device_map=None,
    )
    qwen_model.to(device)
    freeze_backbone = bool(config["model"].get("freeze_backbone", True))
    for parameter in qwen_model.parameters():
        parameter.requires_grad = not freeze_backbone
    visual_adapter = Qwen35VisionAdapter(
        qwen_model,
        spatial_merge_size=int(qwen_config["vision_config"]["spatial_merge_size"]),
    )
    visual_dim = int(qwen_config["vision_config"]["hidden_size"])
    statement_dim = int(config["model"].get("statement_dim", 512))
    head_config = BiVESModelConfig(
        visual_dim=visual_dim,
        statement_dim=statement_dim,
        fusion_dim=int(config["bives"].get("fusion_dim", 512)),
        evidence_max=float(config["bives"].get("evidence_max", 8.0)),
        gate_mode=str(config["bives"]["mask"].get("type", "soft_topk")),
        topk=int(config["bives"]["mask"].get("topk", 16)),
        gate_temperature=float(config["bives"]["mask"].get("temperature", 0.5)),
        tau_a=float(config["bives"]["decoder"].get("tau_a", 1.0)),
        tau_d=float(config["bives"]["decoder"].get("tau_d", 1.0)),
        tau_p=float(config["bives"]["decoder"].get("tau_p", 1.0)),
    )
    experiment = BiVESExperiment(
        visual_adapter,
        num_statements=len(train_dataset.statement_to_index),
        statement_dim=statement_dim,
        head_config=head_config,
    ).to(device)
    collator = Qwen35BiVESCollator(processor, image_size=int(config["data"].get("image_size", 448)))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 0)),
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"].get("eval_batch_size", config["training"]["batch_size"])),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
        collate_fn=collator,
    )
    loss_fn = BiVESLoss(BiVESLossConfig(**config.get("loss", {})))
    head_parameters = list(experiment.statement_table.parameters()) + list(experiment.head.parameters())
    parameter_groups = [
        {
            "params": head_parameters,
            "lr": float(config["training"].get("lr_new_layers", 1e-4)),
        }
    ]
    if not freeze_backbone:
        parameter_groups.append(
            {
                "params": [parameter for parameter in experiment.backbone.parameters() if parameter.requires_grad],
                "lr": float(config["training"].get("lr_backbone", 1e-5)),
            }
        )
    optimizer = AdamW(parameter_groups, weight_decay=float(config["training"].get("weight_decay", 0.05)))

    max_steps = int(config["training"]["max_steps"])
    eval_interval = int(config["training"].get("eval_interval", 100))
    max_grad_norm = float(config["training"].get("max_grad_norm", 1.0))
    step = 0
    events: list[dict[str, Any]] = []
    started = time.time()
    experiment.train()
    while step < max_steps:
        for batch in train_loader:
            batch = move_to_device(batch, device)
            outputs, grids = experiment(
                batch["pixel_values"],
                batch["image_grid_thw"],
                batch["statement_indices"],
                run_interventions=True,
            )
            losses = loss_fn(outputs, batch["targets"], require_uniform_grid(grids))
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(experiment.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % eval_interval == 0 or step == max_steps:
                validation = evaluate(experiment, val_loader, loss_fn, device)
                event = {
                    "step": step,
                    "train_loss": float(losses["total"].detach().cpu()),
                    **validation,
                }
                events.append(event)
                print(json.dumps(event, ensure_ascii=False))
                save_json(output_dir / f"metrics_step_{step}.json", event)
            if step >= max_steps:
                break

    checkpoint = {
        "step": step,
        "model_family": "Qwen3.5",
        "model_path": str(config["model"]["path"]),
        "statement_to_index": train_dataset.statement_to_index,
        "statement_table": experiment.statement_table.state_dict(),
        "bives_head": experiment.head.state_dict(),
        "config": config,
    }
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_dir / "checkpoints" / "final.pt")
    final_metrics = {
        "step": step,
        "elapsed_seconds": time.time() - started,
        "model_family": "Qwen3.5",
        "model_path": str(config["model"]["path"]),
        "train_records": len(train_dataset),
        "val_records": len(val_dataset),
        "events": events,
    }
    save_json(output_dir / "metrics_final.json", final_metrics)


if __name__ == "__main__":
    main()
