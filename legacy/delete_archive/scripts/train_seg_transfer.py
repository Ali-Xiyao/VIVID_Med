"""
ViT → Segmentation Transfer (KiTS21)

从 SVRL/baseline checkpoint 提取 ViT backbone，接 SegHead fine-tune 做分割

用法:
    python train_seg_transfer.py --config ../configs/transfer_seg_v9_kits.yaml
    python train_seg_transfer.py --config ../configs/transfer_seg_v9_kits.yaml --init_vit_checkpoint ../outputs/v9_svrl/checkpoints/best.pt
"""

import argparse
import contextlib
import json
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import timm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import KiTS2DDataset, get_train_transforms, get_val_transforms
from models.seg_head import ViTSegHead


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def seg_collate_fn(batch):
    """KiTS segmentation collate function"""
    images = torch.stack([b["image"] for b in batch])
    masks = torch.stack([b["mask"] for b in batch])
    return {"images": images, "masks": masks}


class DiceLoss(nn.Module):
    """Per-class Dice Loss for segmentation"""

    def __init__(self, num_classes: int, smooth: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W), targets: (B, H, W)
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        valid_mask = (targets != self.ignore_index)  # (B, H, W)
        targets_clean = targets.clone()
        targets_clean[~valid_mask] = 0
        one_hot = F.one_hot(targets_clean, self.num_classes)  # (B, H, W, C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        valid_mask = valid_mask.unsqueeze(1).float()  # (B, 1, H, W)

        probs = probs * valid_mask
        one_hot = one_hot * valid_mask

        dims = (0, 2, 3)  # sum over batch, H, W
        intersection = (probs * one_hot).sum(dims)
        cardinality = probs.sum(dims) + one_hot.sum(dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Exclude background (class 0) from dice loss
        return 1.0 - dice_per_class[1:].mean()


class CombinedSegLoss(nn.Module):
    """CE + Dice Loss with optional class weights"""

    def __init__(self, num_classes: int, class_weights: list = None,
                 dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.ce_loss = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=255)
        self.dice_loss = DiceLoss(num_classes=num_classes)

    def to(self, device):
        super().to(device)
        if self.ce_loss.weight is not None:
            self.ce_loss.weight = self.ce_loss.weight.to(device)
        return self

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice


def create_dataloaders(config: Dict[str, Any]):
    data_cfg = config["data"]
    image_size = data_cfg["image_size"]
    data_root = data_cfg["data_root"]
    subset_fraction = data_cfg.get("subset_fraction")
    seed = config.get("seed", 42)

    train_dataset = KiTS2DDataset(
        data_root=data_root,
        ums_jsonl_path=data_cfg["train_ums_path"],
        transform=get_train_transforms(image_size),
        is_train=True,
        max_samples=data_cfg.get("max_train_samples"),
        subset_fraction=subset_fraction,
        seed=seed,
    )

    val_dataset = KiTS2DDataset(
        data_root=data_root,
        ums_jsonl_path=data_cfg["val_ums_path"],
        transform=get_val_transforms(image_size),
        is_train=False,
        max_samples=data_cfg.get("max_val_samples"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=seg_collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=seg_collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader


class ViTWithSegHead(nn.Module):
    """ViT backbone + SegHead for segmentation transfer"""

    def __init__(self, vit_model_name: str, pretrained: bool, num_seg_classes: int):
        super().__init__()
        self.vit = timm.create_model(
            vit_model_name,
            pretrained=pretrained,
            num_classes=0,
        )
        embed_dim = self.vit.embed_dim
        self.seg_head = ViTSegHead(
            embed_dim=embed_dim,
            num_classes=num_seg_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.vit.forward_features(x)  # (B, 1+196, embed_dim)
        patch_tokens = features[:, 1:, :]  # exclude CLS
        return self.seg_head(patch_tokens)  # (B, num_classes, 224, 224)


def _select_backbone_state_dict(raw_state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract ViT backbone weights from various checkpoint formats"""
    if not isinstance(raw_state, dict):
        raise ValueError("Checkpoint state must be a dict")

    # VIVID checkpoint: ViT under key 'vit'
    if "vit" in raw_state and isinstance(raw_state["vit"], dict):
        candidate = raw_state["vit"]
    elif "model" in raw_state and isinstance(raw_state["model"], dict):
        candidate = raw_state["model"]
    else:
        candidate = raw_state

    # ViTEncoder wraps timm model as self.vit — strip prefix
    if candidate and all(isinstance(k, str) and k.startswith("vit.") for k in candidate.keys()):
        candidate = {k[len("vit."):]: v for k, v in candidate.items()}

    return candidate


def load_vit_backbone(model: ViTWithSegHead, checkpoint_path: str, device: str):
    """Load ViT backbone weights from SVRL/baseline checkpoint"""
    state = torch.load(checkpoint_path, map_location=device)
    source_state = _select_backbone_state_dict(state)

    vit_state = model.vit.state_dict()
    filtered = {}
    for k, v in source_state.items():
        if k in vit_state and vit_state[k].shape == v.shape:
            filtered[k] = v

    result = model.vit.load_state_dict(filtered, strict=False)
    print(f"Loaded ViT backbone from {checkpoint_path}")
    print(f"  Loaded: {len(filtered)} params")
    if result.missing_keys:
        print(f"  Missing (first 10): {result.missing_keys[:10]}")


def compute_dice(pred: np.ndarray, target: np.ndarray, num_classes: int) -> Dict[str, float]:
    """Compute per-class Dice score"""
    dice_scores = {}
    class_names = ["background", "kidney", "tumor", "cyst"]
    for c in range(num_classes):
        p = (pred == c).astype(np.float32)
        t = (target == c).astype(np.float32)
        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        if union == 0:
            dice = float("nan")  # class not present
        else:
            dice = 2.0 * intersection / union
        name = class_names[c] if c < len(class_names) else f"class_{c}"
        dice_scores[f"dice_{name}"] = dice
    return dice_scores


@torch.no_grad()
def evaluate(model, dataloader, device, num_classes: int):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_dice = []

    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=255)

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        images = batch["images"].to(device)
        masks = batch["masks"].to(device)

        logits = model(images)  # (B, C, H, W)

        # Resize mask to match logits if needed
        if logits.shape[-2:] != masks.shape[-2:]:
            masks = torch.nn.functional.interpolate(
                masks.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest"
            ).squeeze(1).long()

        loss = ce_loss_fn(logits, masks)
        total_loss += loss.item()
        num_batches += 1

        pred = logits.argmax(dim=1).cpu().numpy()
        target = masks.cpu().numpy()

        for i in range(pred.shape[0]):
            dice = compute_dice(pred[i], target[i], num_classes)
            all_dice.append(dice)

    # Aggregate dice scores
    avg_dice = {}
    if all_dice:
        keys = all_dice[0].keys()
        for k in keys:
            vals = [d[k] for d in all_dice if not np.isnan(d[k])]
            avg_dice[k] = float(np.mean(vals)) if vals else float("nan")

    # Mean dice (excluding background)
    fg_dice = [v for k, v in avg_dice.items() if "background" not in k and not np.isnan(v)]
    avg_dice["mean_dice_fg"] = float(np.mean(fg_dice)) if fg_dice else float("nan")

    return {
        "val_loss": total_loss / max(num_batches, 1),
        "dice": avg_dice,
    }


def main():
    parser = argparse.ArgumentParser(description="ViT Segmentation Transfer (KiTS21)")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent.parent / "configs" / "transfer_seg_v9_kits.yaml"),
    )
    parser.add_argument("--init_vit_checkpoint", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.debug:
        print("Debug mode enabled")
        config["data"]["max_train_samples"] = 100
        config["data"]["max_val_samples"] = 20
        config["data"]["num_workers"] = 0
        config["training"]["batch_size"] = 2
        config["training"]["gradient_accumulation_steps"] = 1
        config["training"]["max_steps"] = 10
        config["training"]["log_interval"] = 2
        config["training"]["eval_interval"] = 5
        config["training"]["save_interval"] = 10
        config["training"]["bf16"] = False
        config["training"]["fp16"] = False

    seed = config.get("seed", 42)
    set_seed(seed)

    requested_device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(requested_device, str) and requested_device.startswith("cuda"):
        if torch.cuda.is_available():
            device = requested_device
        else:
            device = "cpu"
            print("CUDA not available, falling back to CPU")
    else:
        device = requested_device
    print(f"Using device: {device}")

    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    model_cfg = config["model"]
    num_seg_classes = model_cfg.get("num_seg_classes", 4)

    print("\nCreating model...")
    model = ViTWithSegHead(
        vit_model_name=model_cfg["vit_model_name"],
        pretrained=model_cfg.get("vit_pretrained", True),
        num_seg_classes=num_seg_classes,
    ).to(device)

    # Load pretrained ViT backbone
    transfer_cfg = config.get("transfer", {})
    init_ckpt = args.init_vit_checkpoint or transfer_cfg.get("init_vit_checkpoint")
    if init_ckpt:
        load_vit_backbone(model, init_ckpt, device)

    training_cfg = config["training"]

    # Optimizer: separate lr for backbone vs seg_head
    backbone_lr = float(training_cfg.get("backbone_learning_rate", training_cfg["learning_rate"]))
    head_lr = float(training_cfg.get("head_learning_rate", training_cfg["learning_rate"]))
    weight_decay = float(training_cfg["weight_decay"])

    param_groups = [
        {"params": model.vit.parameters(), "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": model.seg_head.parameters(), "lr": head_lr, "weight_decay": weight_decay},
    ]
    optimizer = AdamW(param_groups)

    vit_params = sum(p.numel() for p in model.vit.parameters() if p.requires_grad)
    head_params = sum(p.numel() for p in model.seg_head.parameters() if p.requires_grad)
    print(f"  ViT params: {vit_params:,} (lr={backbone_lr:.2e})")
    print(f"  SegHead params: {head_params:,} (lr={head_lr:.2e})")

    warmup_steps = training_cfg["warmup_steps"]
    max_steps = training_cfg["max_steps"]
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    use_amp = device.startswith("cuda") and (training_cfg.get("bf16", False) or training_cfg.get("fp16", False))
    use_bf16 = training_cfg.get("bf16", False) and device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler() if training_cfg.get("fp16", False) and device.startswith("cuda") else None

    # Loss: CE + Dice with class weights
    # KiTS class distribution: bg=100%, kidney=100%, tumor=41.7%, cyst=12.2%
    # Inverse frequency weights (normalized): bg=1.0, kidney=1.0, tumor=2.4, cyst=8.2
    loss_cfg = config.get("loss", {})
    class_weights = loss_cfg.get("class_weights", [1.0, 1.0, 2.5, 8.0])
    dice_weight = loss_cfg.get("dice_weight", 0.5)
    ce_weight = loss_cfg.get("ce_weight", 0.5)
    loss_fn = CombinedSegLoss(
        num_classes=num_seg_classes,
        class_weights=class_weights,
        dice_weight=dice_weight,
        ce_weight=ce_weight,
    ).to(device)
    print(f"  Loss: CE(w={ce_weight}) + Dice(w={dice_weight}), class_weights={class_weights}")

    output_dir = Path(training_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting segmentation transfer training...")
    print(f"  Max steps: {max_steps}")
    print(f"  Output dir: {output_dir}")

    train_iter = iter(train_loader)
    progress_bar = tqdm(total=max_steps, desc="Training")
    global_step = 0
    best_mean_dice = 0.0
    accumulated_loss = 0.0
    num_accumulated = 0

    model.train()
    while global_step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        images = batch["images"].to(device)
        masks = batch["masks"].to(device)

        if use_amp:
            autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
            autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=autocast_dtype)
        else:
            autocast_ctx = contextlib.nullcontext()

        with autocast_ctx:
            logits = model(images)
            if logits.shape[-2:] != masks.shape[-2:]:
                masks = torch.nn.functional.interpolate(
                    masks.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest"
                ).squeeze(1).long()
            loss = loss_fn(logits, masks)

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

            if global_step % training_cfg["eval_interval"] == 0:
                result = evaluate(model, val_loader, device, num_seg_classes)
                val_loss = result["val_loss"]
                dice = result["dice"]
                mean_dice_fg = dice.get("mean_dice_fg", 0.0)
                print(f"\nStep {global_step}: val_loss={val_loss:.4f}, mean_dice_fg={mean_dice_fg:.4f}")
                for k, v in dice.items():
                    if k != "mean_dice_fg":
                        print(f"  {k}: {v:.4f}")

                save_json(output_dir / f"metrics_step_{global_step}.json", result)

                if mean_dice_fg > best_mean_dice and not np.isnan(mean_dice_fg):
                    best_mean_dice = mean_dice_fg
                    torch.save({"model": model.state_dict(), "step": global_step}, output_dir / "best.pt")
                    print(f"  New best mean_dice_fg: {best_mean_dice:.4f}")

                model.train()

            if global_step % training_cfg["save_interval"] == 0:
                torch.save({"model": model.state_dict(), "step": global_step}, output_dir / f"step_{global_step}.pt")

    progress_bar.close()
    torch.save({"model": model.state_dict(), "step": global_step}, output_dir / "final.pt")

    # Final evaluation
    result = evaluate(model, val_loader, device, num_seg_classes)
    save_json(output_dir / "metrics_final.json", result)
    print(f"\nFinal mean_dice_fg: {result['dice'].get('mean_dice_fg', 0.0):.4f}")
    print("Segmentation transfer training completed!")


if __name__ == "__main__":
    main()
