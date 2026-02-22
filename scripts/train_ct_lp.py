"""
CT Classification Linear Probe

冻结 ViT backbone，只训练线性分类头
支持 OrganAMNIST (11-class) 和 LIDC-IDRI (binary)

用法:
    python train_ct_lp.py --config ../configs/ct_lp_organ_imagenet.yaml
    python train_ct_lp.py --config ../configs/ct_lp_lidc_imagenet.yaml
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
import yaml
import timm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import OrganAMNISTDataset, LIDCDataset, get_train_transforms, get_val_transforms


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


def ct_collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"images": images, "labels": labels}


def create_dataloaders(config: Dict[str, Any]):
    data_cfg = config["data"]
    dataset_name = data_cfg["dataset"]
    image_size = data_cfg["image_size"]
    data_root = data_cfg["data_root"]
    bs = config["training"]["batch_size"]
    nw = data_cfg.get("num_workers", 4)

    if dataset_name == "organamnist":
        npz_path = data_cfg["npz_path"]
        train_ds = OrganAMNISTDataset(npz_path, split="train", image_size=image_size)
        val_ds = OrganAMNISTDataset(npz_path, split="val", image_size=image_size)
        test_ds = OrganAMNISTDataset(npz_path, split="test", image_size=image_size)
        num_classes = OrganAMNISTDataset.NUM_CLASSES
    elif dataset_name == "lidc":
        label_csv = data_cfg["label_csv"]
        seed = config.get("seed", 42)
        train_ds = LIDCDataset(data_root, label_csv, split="train", image_size=image_size, seed=seed)
        val_ds = LIDCDataset(data_root, label_csv, split="val", image_size=image_size, seed=seed)
        test_ds = LIDCDataset(data_root, label_csv, split="test", image_size=image_size, seed=seed)
        num_classes = LIDCDataset.NUM_CLASSES
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                              collate_fn=ct_collate_fn, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw,
                            collate_fn=ct_collate_fn, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw,
                             collate_fn=ct_collate_fn, pin_memory=True)

    return train_loader, val_loader, test_loader, num_classes


def _select_backbone_state_dict(raw_state: Dict[str, Any]) -> Dict[str, Any]:
    if "vit" in raw_state and isinstance(raw_state["vit"], dict):
        candidate = raw_state["vit"]
    elif "model" in raw_state and isinstance(raw_state["model"], dict):
        candidate = raw_state["model"]
    else:
        candidate = raw_state
    if candidate and all(isinstance(k, str) and k.startswith("vit.") for k in candidate.keys()):
        candidate = {k[len("vit."):]: v for k, v in candidate.items()}
    return candidate


def load_vit_backbone(model, checkpoint_path: str, device: str):
    state = torch.load(checkpoint_path, map_location=device)
    source_state = _select_backbone_state_dict(state)
    model_state = model.state_dict()
    filtered = {}
    for k, v in source_state.items():
        if k.startswith(("head.", "fc_norm.")):
            continue
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
    result = model.load_state_dict(filtered, strict=False)
    print(f"Loaded ViT backbone from {checkpoint_path}")
    print(f"  Loaded: {len(filtered)} params, Missing: {len(result.missing_keys)}")


@torch.no_grad()
def evaluate(model, dataloader, device, num_classes: int):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds, all_labels, all_probs = [], [], []
    ce_loss = nn.CrossEntropyLoss()

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        logits = model(images)
        loss = ce_loss(logits, labels)
        total_loss += loss.item()
        num_batches += 1

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # AUC: binary or multi-class
    try:
        if num_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    return {
        "val_loss": total_loss / max(num_batches, 1),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "macro_auc": float(auc),
    }

def main():
    parser = argparse.ArgumentParser(description="CT Classification Linear Probe")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--init_vit_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.seed is not None:
        config["seed"] = args.seed
        base_dir = config["training"]["output_dir"].rstrip("/")
        config["training"]["output_dir"] = f"{base_dir}_seed{args.seed}"

    if args.debug:
        config["data"]["num_workers"] = 0
        config["training"]["batch_size"] = 4
        config["training"]["max_steps"] = 20
        config["training"]["log_interval"] = 5
        config["training"]["eval_interval"] = 10
        config["training"]["save_interval"] = 20
        config["training"]["bf16"] = False

    seed = config.get("seed", 42)
    set_seed(seed)

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, falling back to CPU")
    print(f"Using device: {device}")

    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(config)
    print(f"  Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    model_cfg = config["model"]
    print("\nCreating model...")
    model = timm.create_model(
        model_cfg["vit_model_name"],
        pretrained=model_cfg.get("vit_pretrained", True),
        num_classes=num_classes,
        drop_rate=0.0,
        drop_path_rate=0.0,
    ).to(device)

    # Load pretrained backbone
    transfer_cfg = config.get("transfer", {})
    init_ckpt = args.init_vit_checkpoint or transfer_cfg.get("init_vit_checkpoint")
    if init_ckpt:
        load_vit_backbone(model, init_ckpt, device)

    # Freeze backbone (linear probe)
    if transfer_cfg.get("freeze_backbone", True):
        frozen = 0
        for name, param in model.named_parameters():
            if not name.startswith(("head.", "fc_norm.")):
                param.requires_grad = False
                frozen += 1
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Linear probe: froze {frozen} params, trainable: {trainable:,}")

    training_cfg = config["training"]
    head_lr = float(training_cfg.get("head_learning_rate", training_cfg["learning_rate"]))
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=head_lr, weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )
    warmup_steps = training_cfg["warmup_steps"]
    max_steps = training_cfg["max_steps"]
    warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_steps])

    use_amp = device.startswith("cuda") and training_cfg.get("bf16", False)
    ce_loss_fn = nn.CrossEntropyLoss()

    output_dir = Path(training_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting CT LP training...")
    print(f"  Max steps: {max_steps}, Output: {output_dir}")

    train_iter = iter(train_loader)
    progress_bar = tqdm(total=max_steps, desc="Training")
    global_step = 0
    best_val_loss = float("inf")
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
        labels = batch["labels"].to(device)

        if use_amp:
            ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            logits = model(images)
            loss = ce_loss_fn(logits, labels)

        loss = loss / training_cfg["gradient_accumulation_steps"]
        loss.backward()
        accumulated_loss += loss.item()
        num_accumulated += 1

        if num_accumulated >= training_cfg["gradient_accumulation_steps"]:
            if training_cfg.get("max_grad_norm", 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_cfg["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            progress_bar.update(1)

            if global_step % training_cfg["log_interval"] == 0:
                avg = accumulated_loss / max(num_accumulated, 1)
                lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr:.2e}")
                accumulated_loss = 0.0
                num_accumulated = 0

            if global_step % training_cfg["eval_interval"] == 0:
                result = evaluate(model, val_loader, device, num_classes)
                vl = result["val_loss"]
                acc = result["accuracy"]
                auc = result["macro_auc"]
                print(f"\nStep {global_step}: val_loss={vl:.4f} acc={acc:.4f} auc={auc:.4f}")
                save_json(output_dir / f"metrics_step_{global_step}.json", result)

                if vl < best_val_loss:
                    best_val_loss = vl
                    torch.save({"model": model.state_dict(), "step": global_step}, output_dir / "best.pt")
                    print(f"  New best val_loss: {best_val_loss:.4f}")
                model.train()

            if global_step % training_cfg["save_interval"] == 0:
                torch.save({"model": model.state_dict(), "step": global_step}, output_dir / f"step_{global_step}.pt")

    progress_bar.close()
    torch.save({"model": model.state_dict(), "step": global_step}, output_dir / "final.pt")

    # Final eval on val + test
    val_result = evaluate(model, val_loader, device, num_classes)
    test_result = evaluate(model, test_loader, device, num_classes)
    save_json(output_dir / "metrics_final.json", {"val": val_result, "test": test_result})
    print(f"\nFinal val: acc={val_result['accuracy']:.4f} auc={val_result['macro_auc']:.4f}")
    print(f"Final test: acc={test_result['accuracy']:.4f} auc={test_result['macro_auc']:.4f}")
    print("CT LP training completed!")


if __name__ == "__main__":
    main()
