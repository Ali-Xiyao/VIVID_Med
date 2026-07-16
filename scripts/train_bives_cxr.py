"""Train the Qwen3.5-only BiVES-CXR evidence model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bives_cxr.audit import audit_manifests
from bives_cxr.backbones import Qwen35VisionAdapter, load_qwen35_visual_and_processor
from bives_cxr.data import (
    BiVESManifestDataset,
    SameStatementStateBatchSampler,
    build_group_loss_indices,
)
from bives_cxr.losses import BiVESLoss, BiVESLossConfig
from bives_cxr.metrics import finalize_intervention_metrics, intervention_metric_counts
from bives_cxr.model import BiVESCXR, BiVESModelConfig


class BiVESExperiment(nn.Module):
    """Qwen3.5 patch encoder plus canonical statement table and BiVES head."""

    def __init__(
        self,
        backbone: Qwen35VisionAdapter,
        num_statements: int,
        statement_dim: int,
        head_config: BiVESModelConfig,
        frozen_statement_embeddings: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        if frozen_statement_embeddings is None:
            self.statement_table = nn.Embedding(num_statements, statement_dim)
        else:
            if frozen_statement_embeddings.shape != (num_statements, statement_dim):
                raise ValueError(
                    "frozen statement embedding matrix must have shape "
                    f"{(num_statements, statement_dim)}"
                )
            self.statement_table = nn.Embedding.from_pretrained(
                frozen_statement_embeddings.float(),
                freeze=True,
            )
        self.head = BiVESCXR(head_config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        statement_indices: torch.Tensor,
        content_valid_mask: torch.Tensor | None = None,
        run_interventions: bool = True,
    ) -> tuple[dict[str, dict[str, torch.Tensor] | torch.Tensor], list[tuple[int, int]]]:
        first_backbone_parameter = next(self.backbone.parameters(), None)
        if first_backbone_parameter is not None and pixel_values.is_floating_point():
            pixel_values = pixel_values.to(dtype=first_backbone_parameter.dtype)
        if any(parameter.requires_grad for parameter in self.backbone.parameters()):
            patches = self.backbone(pixel_values, image_grid_thw)
        else:
            self.backbone.eval()
            with torch.no_grad():
                patches = self.backbone(pixel_values, image_grid_thw)
        head_dtype = self.head.visual_projection.weight.dtype
        patch_tokens = patches.tokens.to(dtype=head_dtype)
        statement_embeddings = self.statement_table(statement_indices).to(dtype=head_dtype)
        valid_mask = patches.valid_mask
        if content_valid_mask is not None:
            if content_valid_mask.shape != valid_mask.shape:
                raise ValueError(
                    f"content_valid_mask shape {tuple(content_valid_mask.shape)} "
                    f"does not match visual grid {tuple(valid_mask.shape)}"
                )
            valid_mask = valid_mask & content_valid_mask.bool()
        outputs = self.head(
            patch_tokens,
            statement_embeddings,
            valid_mask,
            run_interventions=run_interventions,
        )
        return outputs, patches.grid_hw


class Qwen35BiVESCollator:
    def __init__(self, processor: Any, image_size: int = 448) -> None:
        self.processor = processor
        self.image_size = int(image_size)

    def _letterbox(self, image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
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
        return canvas, (left, top, left + resized.width, top + resized.height)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts = []
        images = []
        content_boxes = []
        for item in batch:
            image, content_box = self._letterbox(item["image"])
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
            content_boxes.append(content_box)
        processor_output = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        encoded = {
            "pixel_values": processor_output["pixel_values"],
            "image_grid_thw": processor_output["image_grid_thw"],
        }
        masks: list[torch.Tensor] = []
        for grid, box in zip(encoded["image_grid_thw"], content_boxes):
            height, width = int(grid[1]), int(grid[2])
            left, top, right, bottom = box
            x_centers = (torch.arange(width, dtype=torch.float32) + 0.5) * self.image_size / width
            y_centers = (torch.arange(height, dtype=torch.float32) + 0.5) * self.image_size / height
            mask = (
                (y_centers[:, None] >= top)
                & (y_centers[:, None] < bottom)
                & (x_centers[None, :] >= left)
                & (x_centers[None, :] < right)
            ).reshape(-1)
            masks.append(mask)
        max_patches = max(mask.numel() for mask in masks)
        content_valid = torch.zeros((len(masks), max_patches), dtype=torch.bool)
        for index, mask in enumerate(masks):
            content_valid[index, : mask.numel()] = mask
        encoded["content_valid_mask"] = content_valid
        encoded["statement_indices"] = torch.tensor([item["statement_index"] for item in batch], dtype=torch.long)
        encoded["targets"] = torch.tensor([item["state_index"] for item in batch], dtype=torch.long)
        encoded["sample_ids"] = [str(item["sample_id"]) for item in batch]
        encoded.update(build_group_loss_indices(batch))
        return encoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Run the synthetic CPU core smoke without loading Qwen3.5.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", type=Path)
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


DEVICE_FIELDS = {
    "pixel_values",
    "image_grid_thw",
    "content_valid_mask",
    "statement_indices",
    "targets",
    "support_pair_indices",
    "contradict_pair_indices",
    "uncertain_indices",
}


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if key in DEVICE_FIELDS and torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def require_uniform_grid(grid_hw: list[tuple[int, int]]) -> tuple[int, int]:
    unique = set(grid_hw)
    if len(unique) != 1:
        raise ValueError(f"BiVES batches currently require one padded image grid; got {sorted(unique)}")
    return grid_hw[0]


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def limit_to_complete_groups(dataset: BiVESManifestDataset, max_records: int | None) -> None:
    if not max_records:
        return
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for row in dataset.rows:
        statement_id = str(row["canonical_statement_id"])
        groups.setdefault(statement_id, {}).setdefault(str(row["state"]), row)
    selected: list[dict[str, Any]] = []
    for statement_id in sorted(groups):
        state_rows = groups[statement_id]
        if all(state in state_rows for state in ("support", "contradict", "uncertain", "insufficient")):
            selected.extend(
                state_rows[state]
                for state in ("support", "contradict", "uncertain", "insufficient")
            )
        if len(selected) + 4 > max_records:
            break
    if not selected:
        raise ValueError(f"max_records={max_records} does not retain one complete S/C/U/I group")
    dataset.rows = selected


def load_frozen_statement_embeddings(
    path: str | Path,
    statement_to_index: dict[str, int],
) -> torch.Tensor:
    cache_path = Path(path)
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"formal BiVES config requires frozen Qwen3.5 statement embeddings: {cache_path}"
        )
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    mapping = payload.get("embeddings", payload) if isinstance(payload, dict) else payload
    if not isinstance(mapping, dict):
        raise ValueError("statement embedding cache must be a statement_id -> tensor mapping")
    rows: list[torch.Tensor] = []
    for statement_id, _ in sorted(statement_to_index.items(), key=lambda item: item[1]):
        if statement_id not in mapping:
            raise ValueError(f"statement embedding cache is missing {statement_id!r}")
        tensor = torch.as_tensor(mapping[statement_id]).float().reshape(-1)
        rows.append(tensor)
    dimensions = {int(row.numel()) for row in rows}
    if len(dimensions) != 1:
        raise ValueError(f"statement embedding dimensions are inconsistent: {sorted(dimensions)}")
    return torch.stack(rows)


@torch.no_grad()
def evaluate(
    experiment: BiVESExperiment,
    loader: DataLoader,
    loss_fn: BiVESLoss,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    experiment.eval()
    loss_sum = 0.0
    correct = 0
    samples = 0
    mechanism: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for batch in loader:
        batch = move_to_device(batch, device)
        outputs, grids = experiment(
            batch["pixel_values"],
            batch["image_grid_thw"],
            batch["statement_indices"],
            batch["content_valid_mask"],
            run_interventions=True,
        )
        losses = loss_fn(
            outputs,
            batch["targets"],
            require_uniform_grid(grids),
            batch["support_pair_indices"],
            batch["contradict_pair_indices"],
            batch["uncertain_indices"],
        )
        batch_size = int(batch["targets"].numel())
        loss_sum += float(losses["total"].cpu()) * batch_size
        probabilities = outputs["original"]["state_probs"]
        correct += int((probabilities.argmax(dim=-1) == batch["targets"]).sum().item())
        samples += batch_size
        for key, value in intervention_metric_counts(outputs, batch["targets"]).items():
            mechanism[key] = mechanism.get(key, 0.0) + value
        for index, sample_id in enumerate(batch["sample_ids"]):
            rows.append(
                {
                    "sample_id": sample_id,
                    "target": int(batch["targets"][index].item()),
                    "original_probs": probabilities[index].detach().float().cpu().tolist(),
                    "keep_probs": outputs["keep"]["state_probs"][index].detach().float().cpu().tolist(),
                    "drop_probs": outputs["drop"]["state_probs"][index].detach().float().cpu().tolist(),
                    "control_probs": outputs["control"]["state_probs"][index].detach().float().cpu().tolist(),
                }
            )
    experiment.train()
    result = {
        "loss": float(loss_sum / samples) if samples else float("nan"),
        "accuracy": float(correct / max(samples, 1)),
    }
    result.update(finalize_intervention_metrics(mechanism))
    return result, rows


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
        config.setdefault("sampling", {})["groups_per_batch"] = 1
        config["training"]["eval_interval"] = 1
        config["training"]["output_dir"] = str(config["training"]["output_dir"]).rstrip("/\\") + "_debug"

    set_seed(int(config.get("seed", 17)))
    device = torch.device(str(config.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")))
    output_dir = Path(config["training"]["output_dir"])
    if (output_dir / "metrics_final.json").exists():
        raise SystemExit(f"{output_dir} already contains metrics_final.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    manifest_paths = {
        split: config["data"][key]
        for split, key in (
            ("train", "train_manifest"),
            ("val", "val_manifest"),
            ("calibration", "calibration_manifest"),
            ("test", "test_manifest"),
        )
        if config["data"].get(key)
    }
    audit_config = config.get("audit", {})
    audit_report = audit_manifests(
        manifest_paths,
        data_root=config["data"]["data_root"],
        check_images=bool(audit_config.get("check_images", True)),
        require_complete_statements=bool(audit_config.get("require_complete_statements", True)),
        check_decodable=bool(audit_config.get("check_decodable", True)),
        reject_constant_images=bool(audit_config.get("reject_constant_images", True)),
        require_provenance=bool(audit_config.get("require_provenance", True)),
    )
    save_json(output_dir / "manifest_readiness_audit.json", audit_report)
    if audit_report["status"] != "pass":
        raise SystemExit(
            f"manifest readiness gate failed before Qwen3.5 load; "
            f"see {output_dir / 'manifest_readiness_audit.json'}"
        )

    train_dataset = BiVESManifestDataset(config["data"]["train_manifest"], config["data"]["data_root"])
    val_dataset = BiVESManifestDataset(
        config["data"]["val_manifest"],
        config["data"]["data_root"],
        statement_to_index=train_dataset.statement_to_index,
    )
    calibration_dataset = (
        BiVESManifestDataset(
            config["data"]["calibration_manifest"],
            config["data"]["data_root"],
            statement_to_index=train_dataset.statement_to_index,
        )
        if config["data"].get("calibration_manifest")
        else None
    )
    test_dataset = (
        BiVESManifestDataset(
            config["data"]["test_manifest"],
            config["data"]["data_root"],
            statement_to_index=train_dataset.statement_to_index,
        )
        if config["data"].get("test_manifest")
        else None
    )
    if args.debug:
        limit_to_complete_groups(train_dataset, config["data"].get("max_train_samples"))
        limit_to_complete_groups(val_dataset, config["data"].get("max_val_samples"))

    statement_embedding_config = config["model"].get(
        "statement_embeddings",
        {"mode": "learned_id"},
    )
    statement_embedding_mode = str(statement_embedding_config.get("mode", "learned_id"))
    frozen_statement_embeddings = None
    if statement_embedding_mode == "frozen_cached":
        frozen_statement_embeddings = load_frozen_statement_embeddings(
            statement_embedding_config["path"],
            train_dataset.statement_to_index,
        )
        statement_dim = int(frozen_statement_embeddings.shape[1])
    elif statement_embedding_mode == "learned_id":
        statement_dim = int(config["model"].get("statement_dim", 512))
    else:
        raise ValueError(f"unsupported statement embedding mode: {statement_embedding_mode}")

    visual_model, processor, qwen_config = load_qwen35_visual_and_processor(
        config["model"]["path"],
        dtype=str(config["model"].get("dtype", "bf16")),
    )
    freeze_backbone = bool(config["model"].get("freeze_backbone", True))
    for parameter in visual_model.parameters():
        parameter.requires_grad = not freeze_backbone
    visual_adapter = Qwen35VisionAdapter(
        visual_model,
        spatial_merge_size=int(qwen_config["vision_config"]["spatial_merge_size"]),
    ).to(device)
    visual_dim = int(qwen_config["vision_config"]["hidden_size"])
    intervention_config = config["bives"].get("interventions", {})
    if str(intervention_config.get("replacement", "zero")) != "zero":
        raise ValueError("active BiVES mainline currently supports only zero intervention replacement")
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
        num_controls=int(intervention_config.get("num_controls", 4)),
        control_mode=str(intervention_config.get("control_mode", "random_disjoint")),
    )
    experiment = BiVESExperiment(
        visual_adapter,
        num_statements=len(train_dataset.statement_to_index),
        statement_dim=statement_dim,
        head_config=head_config,
        frozen_statement_embeddings=frozen_statement_embeddings,
    ).to(device)
    collator = Qwen35BiVESCollator(processor, image_size=int(config["data"].get("image_size", 448)))
    sampling = config.get("sampling", {})
    if sampling.get("type") != "same_statement_state_group":
        raise ValueError("active BiVES training requires sampling.type=group")
    groups_per_batch = int(sampling.get("groups_per_batch", 1))
    sampler_kwargs = {
        "groups_per_batch": groups_per_batch,
        "states": tuple(sampling.get("states", ("support", "contradict", "uncertain", "insufficient"))),
        "seed": int(config.get("seed", 17)),
        "require_complete_groups": bool(sampling.get("require_complete_groups", True)),
    }
    train_batch_sampler = SameStatementStateBatchSampler(
        train_dataset,
        shuffle=True,
        **sampler_kwargs,
    )
    val_batch_sampler = SameStatementStateBatchSampler(
        val_dataset,
        shuffle=False,
        **sampler_kwargs,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=int(config["data"].get("num_workers", 0)),
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_batch_sampler,
        num_workers=int(config["data"].get("num_workers", 0)),
        collate_fn=collator,
    )
    def make_eval_loader(dataset: BiVESManifestDataset | None) -> DataLoader | None:
        if dataset is None:
            return None
        return DataLoader(
            dataset,
            batch_sampler=SameStatementStateBatchSampler(
                dataset,
                shuffle=False,
                **sampler_kwargs,
            ),
            num_workers=int(config["data"].get("num_workers", 0)),
            collate_fn=collator,
        )

    calibration_loader = make_eval_loader(calibration_dataset)
    test_loader = make_eval_loader(test_dataset)
    loss_fn = BiVESLoss(BiVESLossConfig(**config.get("loss", {})))
    head_parameters = [
        parameter
        for parameter in list(experiment.statement_table.parameters()) + list(experiment.head.parameters())
        if parameter.requires_grad
    ]
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
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max_steps,
        eta_min=float(config["training"].get("min_lr", 0.0)),
    )
    eval_interval = int(config["training"].get("eval_interval", 100))
    max_grad_norm = float(config["training"].get("max_grad_norm", 1.0))
    step = 0
    best_val_loss = float("inf")
    events: list[dict[str, Any]] = []
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        experiment.statement_table.load_state_dict(checkpoint["statement_table"])
        experiment.head.load_state_dict(checkpoint["bives_head"])
        if "backbone" in checkpoint:
            experiment.backbone.load_state_dict(checkpoint["backbone"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        step = int(checkpoint["step"])
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        events = list(checkpoint.get("events", []))
    started = time.time()
    experiment.train()
    epoch = step // max(len(train_loader), 1)
    while step < max_steps:
        train_batch_sampler.set_epoch(epoch)
        for batch in train_loader:
            batch = move_to_device(batch, device)
            outputs, grids = experiment(
                batch["pixel_values"],
                batch["image_grid_thw"],
                batch["statement_indices"],
                batch["content_valid_mask"],
                run_interventions=True,
            )
            losses = loss_fn(
                outputs,
                batch["targets"],
                require_uniform_grid(grids),
                batch["support_pair_indices"],
                batch["contradict_pair_indices"],
                batch["uncertain_indices"],
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(experiment.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % eval_interval == 0 or step == max_steps:
                validation, validation_rows = evaluate(experiment, val_loader, loss_fn, device)
                event = {
                    "step": step,
                    "train_loss": float(losses["total"].detach().cpu()),
                    **validation,
                }
                events.append(event)
                print(json.dumps(event, ensure_ascii=False))
                save_json(output_dir / f"metrics_step_{step}.json", event)
                with (output_dir / f"val_predictions_step_{step}.jsonl").open(
                    "w", encoding="utf-8"
                ) as handle:
                    for row in validation_rows:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                checkpoint = {
                    "step": step,
                    "best_val_loss": best_val_loss,
                    "events": events,
                    "model_family": "Qwen3.5",
                    "model_path": str(config["model"]["path"]),
                    "statement_to_index": train_dataset.statement_to_index,
                    "statement_table": experiment.statement_table.state_dict(),
                    "bives_head": experiment.head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": config,
                }
                if not freeze_backbone:
                    checkpoint["backbone"] = experiment.backbone.state_dict()
                (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, output_dir / "checkpoints" / "last.pt")
                if validation["loss"] < best_val_loss:
                    best_val_loss = validation["loss"]
                    checkpoint["best_val_loss"] = best_val_loss
                    torch.save(checkpoint, output_dir / "checkpoints" / "best.pt")
            if step >= max_steps:
                break
        epoch += 1

    checkpoint = {
        "step": step,
        "model_family": "Qwen3.5",
        "model_path": str(config["model"]["path"]),
        "statement_to_index": train_dataset.statement_to_index,
        "statement_table": experiment.statement_table.state_dict(),
        "bives_head": experiment.head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "events": events,
        "config": config,
    }
    if not freeze_backbone:
        checkpoint["backbone"] = experiment.backbone.state_dict()
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_dir / "checkpoints" / "final.pt")
    split_metrics: dict[str, Any] = {}
    for split, loader in (("calibration", calibration_loader), ("test", test_loader)):
        if loader is None:
            continue
        metrics, prediction_rows = evaluate(experiment, loader, loss_fn, device)
        split_metrics[split] = metrics
        with (output_dir / f"{split}_predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row in prediction_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    final_metrics = {
        "step": step,
        "elapsed_seconds": time.time() - started,
        "model_family": "Qwen3.5",
        "model_path": str(config["model"]["path"]),
        "train_records": len(train_dataset),
        "val_records": len(val_dataset),
        "best_val_loss": best_val_loss,
        "split_metrics": split_metrics,
        "manifest_sha256": {
            split: file_sha256(path)
            for split, path in manifest_paths.items()
        },
        "git_commit": git_commit(),
        "torch_version": torch.__version__,
        "package_versions": {
            package: importlib.metadata.version(package)
            for package in ("transformers", "safetensors", "numpy", "pillow")
        },
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "pid": os.getpid(),
        "events": events,
    }
    save_json(output_dir / "metrics_final.json", final_metrics)


if __name__ == "__main__":
    main()
