"""
ViT Baseline (CheXpert multi-label classification)

用法:
    python train_vit_baseline.py --config ../configs/baseline_vit_chexpert.yaml
"""

import argparse
import json
import random
import contextlib
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import yaml
import timm
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

# 添加项目根目录到 path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import CheXpertUMSDataset, get_train_transforms, get_val_transforms
from data.chexpert_dataset import collate_fn
from evaluation.metrics import compute_classification_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_base_dataset(dataset):
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def create_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, list]:
    data_cfg = config["data"]
    image_size = data_cfg["image_size"]
    data_root = data_cfg["data_root"]
    train_ums_path = data_cfg["train_ums_path"]
    val_ums_path = data_cfg.get("val_ums_path")
    use_common_labels_only = data_cfg.get("use_common_labels_only", False)
    selected_labels = data_cfg.get("selected_labels")
    max_train_samples = data_cfg.get("max_train_samples")
    max_val_samples = data_cfg.get("max_val_samples", 1000)
    train_dense_top_k = data_cfg.get("train_dense_top_k")
    train_dense_min_answerable = data_cfg.get("train_dense_min_answerable")
    val_dense_top_k = data_cfg.get("val_dense_top_k")
    val_dense_min_answerable = data_cfg.get("val_dense_min_answerable")

    train_dataset = CheXpertUMSDataset(
        data_root=data_root,
        ums_jsonl_path=train_ums_path,
        transform=get_train_transforms(image_size),
        is_train=True,
        use_common_labels_only=use_common_labels_only,
        selected_labels=selected_labels,
        max_samples=max_train_samples,
        dense_subset_top_k=train_dense_top_k,
        dense_subset_min_answerable=train_dense_min_answerable,
    )

    val_dataset = None
    if val_ums_path:
        val_dataset = CheXpertUMSDataset(
            data_root=data_root,
            ums_jsonl_path=val_ums_path,
            transform=get_val_transforms(image_size),
            is_train=False,
            use_common_labels_only=use_common_labels_only,
            selected_labels=selected_labels,
            max_samples=max_val_samples,
            dense_subset_top_k=val_dense_top_k,
            dense_subset_min_answerable=val_dense_min_answerable,
        )
    else:
        if max_val_samples and max_val_samples < len(train_dataset):
            generator = torch.Generator().manual_seed(config.get("seed", 42))
            indices = torch.randperm(len(train_dataset), generator=generator).tolist()
            val_indices = indices[:max_val_samples]
            train_indices = indices[max_val_samples:]

            train_dataset = Subset(train_dataset, train_indices)

            val_pool_max_samples = max_train_samples
            val_base_dataset = CheXpertUMSDataset(
                data_root=data_root,
                ums_jsonl_path=train_ums_path,
                transform=get_val_transforms(image_size),
                is_train=False,
                use_common_labels_only=use_common_labels_only,
                selected_labels=selected_labels,
                max_samples=val_pool_max_samples,
                dense_subset_top_k=val_dense_top_k,
                dense_subset_min_answerable=val_dense_min_answerable,
            )
            val_dataset = Subset(val_base_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=data_cfg.get("num_workers", 4),
            collate_fn=collate_fn,
            pin_memory=True,
        )

    label_names = get_base_dataset(train_dataset).label_names
    return train_loader, val_loader, label_names


def prepare_labels_for_loss(labels: torch.Tensor, policy: str) -> Tuple[torch.Tensor, torch.Tensor]:
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
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    loss = bce(logits, labels)
    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denom


@torch.no_grad()
def evaluate(model, dataloader, device, policy: str, threshold: float, label_names: list):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_probs = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)

        logits = model(images)
        loss = compute_loss(logits, labels, policy)
        total_loss += loss.item()
        num_batches += 1

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(prepare_labels_for_metrics(labels.cpu().numpy(), policy))

    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        label_names=label_names,
        threshold=threshold,
    )

    return {
        "val_loss": total_loss / max(num_batches, 1),
        "metrics": metrics,
    }


def save_json(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def create_optimizer_param_groups(
    model: torch.nn.Module,
    training_cfg: Dict[str, Any],
) -> Tuple[list, Dict[str, Any]]:
    base_lr = float(training_cfg["learning_rate"])
    backbone_lr = float(training_cfg.get("backbone_learning_rate", base_lr))
    head_lr = float(training_cfg.get("head_learning_rate", base_lr))
    weight_decay = float(training_cfg["weight_decay"])

    head_prefixes = ("head.", "fc_norm.")
    backbone_params = []
    head_params = []
    backbone_numel = 0
    head_numel = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(head_prefixes):
            head_params.append(parameter)
            head_numel += parameter.numel()
        else:
            backbone_params.append(parameter)
            backbone_numel += parameter.numel()

    # Fallback: if model has no explicit head prefix, keep single-group behavior.
    if not head_params:
        backbone_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        backbone_numel = sum(parameter.numel() for parameter in backbone_params)
        head_numel = 0

    param_groups = []
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": backbone_lr,
                "weight_decay": weight_decay,
                "group_name": "backbone",
            }
        )
    if head_params:
        param_groups.append(
            {
                "params": head_params,
                "lr": head_lr,
                "weight_decay": weight_decay,
                "group_name": "head",
            }
        )

    group_info = {
        "base_lr": base_lr,
        "backbone_lr": backbone_lr,
        "head_lr": head_lr,
        "weight_decay": weight_decay,
        "backbone_numel": backbone_numel,
        "head_numel": head_numel,
    }
    return param_groups, group_info


def _select_backbone_state_dict(raw_state: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw_state, dict):
        raise ValueError("Checkpoint state must be a dict")

    # VIVID checkpoint stores ViT under key 'vit'
    if "vit" in raw_state and isinstance(raw_state["vit"], dict):
        candidate = raw_state["vit"]
    # Baseline/full-model checkpoints may store weights under key 'model'
    elif "model" in raw_state and isinstance(raw_state["model"], dict):
        candidate = raw_state["model"]
    else:
        candidate = raw_state

    # ViTEncoder wraps timm model as self.vit, strip prefix for timm model loading
    vit_prefixed = [k for k in candidate.keys() if isinstance(k, str) and k.startswith("vit.")]
    if candidate and len(vit_prefixed) > len(candidate) // 2:
        normalized = {}
        for key, value in candidate.items():
            if isinstance(key, str) and key.startswith("vit."):
                normalized[key[len("vit."):]] = value
            # Skip non-vit keys (e.g. sar_alpha) — not part of timm model
    else:
        normalized = candidate

    return normalized


def load_vit_backbone_from_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: str) -> None:
    checkpoint_path = str(Path(checkpoint_path))
    state = torch.load(checkpoint_path, map_location=device)
    source_state = _select_backbone_state_dict(state)

    model_state = model.state_dict()
    filtered_state = {}
    ignored_prefixes = ("head.", "fc_norm.")

    for key, value in source_state.items():
        if not isinstance(key, str):
            continue
        if key.startswith(ignored_prefixes):
            continue
        if key not in model_state:
            continue
        if getattr(model_state[key], "shape", None) != getattr(value, "shape", None):
            continue
        filtered_state[key] = value

    load_result = model.load_state_dict(filtered_state, strict=False)
    loaded_count = len(filtered_state)
    print(f"Loaded ViT backbone from {checkpoint_path}")
    print(f"  Loaded params: {loaded_count}")
    if load_result.missing_keys:
        print(f"  Missing keys (first 10): {load_result.missing_keys[:10]}")
    if load_result.unexpected_keys:
        print(f"  Unexpected keys (first 10): {load_result.unexpected_keys[:10]}")


def main():
    parser = argparse.ArgumentParser(description="Train ViT baseline for CheXpert")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent.parent / "configs" / "baseline_vit_chexpert.yaml"),
    )
    parser.add_argument(
        "--init_vit_checkpoint",
        type=str,
        default=None,
        help="Initialize ViT backbone from VIVID/baseline checkpoint",
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.debug:
        print("Debug mode enabled")
        config["data"]["max_train_samples"] = 200
        config["data"]["max_val_samples"] = 50
        config["data"]["num_workers"] = 0
        config["training"]["batch_size"] = 8
        config["training"]["gradient_accumulation_steps"] = 1
        config["training"]["max_steps"] = 20
        config["training"]["log_interval"] = 2
        config["training"]["eval_interval"] = 5
        config["training"]["save_interval"] = 20
        config["training"]["bf16"] = False
        config["training"]["fp16"] = False

    seed = config.get("seed", 42)
    set_seed(seed)

    requested_device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(requested_device, str) and requested_device.startswith("cuda"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("CUDA not available, falling back to CPU")
    else:
        device = requested_device
    print(f"Using device: {device}")

    print("\nCreating dataloaders...")
    train_loader, val_loader, label_names = create_dataloaders(config)
    print(f"  Train batches: {len(train_loader)}")
    if val_loader:
        print(f"  Val batches: {len(val_loader)}")

    model_cfg = config["model"]
    num_labels = len(label_names)

    print("\nCreating model...")
    model = timm.create_model(
        model_cfg["vit_model_name"],
        pretrained=model_cfg.get("vit_pretrained", True),
        num_classes=num_labels,
        drop_rate=model_cfg.get("drop_rate", 0.0),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.1),
    ).to(device)

    transfer_cfg = config.get("transfer", {})
    init_vit_checkpoint = args.init_vit_checkpoint or transfer_cfg.get("init_vit_checkpoint")
    if init_vit_checkpoint:
        load_vit_backbone_from_checkpoint(model, init_vit_checkpoint, device)

    # Linear probe: freeze backbone, only train head
    freeze_backbone = config.get("transfer", {}).get("freeze_backbone", False)
    if freeze_backbone:
        head_prefixes = ("head.", "fc_norm.")
        frozen_count = 0
        for name, param in model.named_parameters():
            if not name.startswith(head_prefixes):
                param.requires_grad = False
                frozen_count += 1
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nLinear probe mode: froze {frozen_count} params, trainable: {trainable:,}")

    training_cfg = config["training"]
    eval_cfg = config.get("evaluation", {})
    uncertain_policy = eval_cfg.get("uncertain_policy", "ignore")
    threshold = float(eval_cfg.get("threshold", 0.5))

    param_groups, group_info = create_optimizer_param_groups(model, training_cfg)
    optimizer = AdamW(param_groups)

    print("\nOptimizer groups:")
    print(f"  base lr: {group_info['base_lr']:.2e}")
    print(f"  backbone lr: {group_info['backbone_lr']:.2e}, params: {group_info['backbone_numel']:,}")
    print(f"  head lr: {group_info['head_lr']:.2e}, params: {group_info['head_numel']:,}")
    print(f"  weight decay: {group_info['weight_decay']:.4f}")

    warmup_steps = training_cfg["warmup_steps"]
    max_steps = training_cfg["max_steps"]
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max_steps - warmup_steps, eta_min=training_cfg["learning_rate"] * 0.1)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    use_amp = device == "cuda" and (training_cfg.get("bf16", False) or training_cfg.get("fp16", False))
    use_bf16 = training_cfg.get("bf16", False) and device == "cuda"
    scaler = torch.cuda.amp.GradScaler() if training_cfg.get("fp16", False) and device == "cuda" else None

    output_dir = Path(training_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nStarting training...")
    print(f"  Max steps: {max_steps}")
    print(f"  Output dir: {output_dir}")

    train_iter = iter(train_loader)
    progress_bar = tqdm(total=max_steps, desc="Training")
    global_step = 0
    best_val_loss = float("inf")
    accumulated_loss = 0.0
    num_accumulated = 0

    while global_step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        images = batch["images"].to(device)
        labels = batch["labels"].to(device)

        if use_amp:
            autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
            autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=autocast_dtype)
        else:
            autocast_ctx = contextlib.nullcontext()

        with autocast_ctx:
            logits = model(images)
            loss = compute_loss(logits, labels, uncertain_policy)

        loss = loss / training_cfg["gradient_accumulation_steps"]

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accumulated_loss += loss.item()
        num_accumulated += 1

        if num_accumulated >= training_cfg["gradient_accumulation_steps"]:
            if training_cfg["max_grad_norm"] > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_cfg["max_grad_norm"])

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            progress_bar.update(1)

            if global_step % training_cfg["log_interval"] == 0:
                avg_loss = accumulated_loss / max(num_accumulated, 1)
                lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
                accumulated_loss = 0.0
                num_accumulated = 0

            if val_loader is not None and global_step % training_cfg["eval_interval"] == 0:
                result = evaluate(model, val_loader, device, uncertain_policy, threshold, label_names)
                val_loss = result["val_loss"]
                print(f"\nStep {global_step}: val_loss = {val_loss:.4f}")

                metrics_path = output_dir / f"metrics_step_{global_step}.json"
                save_json(metrics_path, result)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    ckpt_path = output_dir / "best.pt"
                    torch.save({"model": model.state_dict(), "step": global_step}, ckpt_path)

                model.train()

            if global_step % training_cfg["save_interval"] == 0:
                ckpt_path = output_dir / f"step_{global_step}.pt"
                torch.save({"model": model.state_dict(), "step": global_step}, ckpt_path)

    progress_bar.close()
    final_path = output_dir / "final.pt"
    torch.save({"model": model.state_dict(), "step": global_step}, final_path)

    if val_loader is not None:
        result = evaluate(model, val_loader, device, uncertain_policy, threshold, label_names)
        save_json(output_dir / "metrics_final.json", result)

    print("Training completed!")


if __name__ == "__main__":
    main()
