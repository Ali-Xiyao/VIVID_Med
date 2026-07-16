"""
Train a no-LLM UMS field-state classifier.

This is the controlled baseline for the reviewer question:
does the UMS schema alone explain the gain, or is the frozen LM teacher needed?
The model trains a ViT backbone plus per-finding 4-state logits
for {null, absent, uncertain, present}; checkpoints save the ViT backbone
under the "vit" key so existing linear-probe code can reuse it.
"""

import argparse
import contextlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import CheXpertUMSDataset, get_train_transforms, get_val_transforms
from data.chexpert_dataset import collate_fn
from evaluation.metrics import compute_classification_metrics


STATE_TO_INDEX = {
    "null": 0,
    "absent": 1,
    "uncertain": 2,
    "present": 3,
}
AUXILIARY_TARGETS = ("answerability", "uncertainty")


def time_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def set_seed(seed: int) -> None:
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


def create_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, List[str]]:
    data_cfg = config["data"]
    image_size = data_cfg["image_size"]
    max_train_samples = data_cfg.get("max_train_samples")
    max_val_samples = data_cfg.get("max_val_samples", 1000)
    selected_labels = data_cfg.get("selected_labels")

    train_dataset = CheXpertUMSDataset(
        data_root=data_cfg["data_root"],
        ums_jsonl_path=data_cfg["train_ums_path"],
        transform=get_train_transforms(image_size),
        is_train=True,
        use_common_labels_only=data_cfg.get("use_common_labels_only", False),
        selected_labels=selected_labels,
        max_samples=max_train_samples,
        dense_subset_top_k=data_cfg.get("train_dense_top_k"),
        dense_subset_min_answerable=data_cfg.get("train_dense_min_answerable"),
        schema_mode=data_cfg.get("schema_mode", "state_only"),
    )

    if data_cfg.get("val_ums_path"):
        val_dataset = CheXpertUMSDataset(
            data_root=data_cfg["data_root"],
            ums_jsonl_path=data_cfg["val_ums_path"],
            transform=get_val_transforms(image_size),
            is_train=False,
            use_common_labels_only=data_cfg.get("use_common_labels_only", False),
            selected_labels=selected_labels,
            max_samples=max_val_samples,
            dense_subset_top_k=data_cfg.get("val_dense_top_k"),
            dense_subset_min_answerable=data_cfg.get("val_dense_min_answerable"),
            schema_mode=data_cfg.get("schema_mode", "state_only"),
        )
    else:
        if not max_val_samples or max_val_samples >= len(train_dataset):
            raise ValueError("val_ums_path is required unless max_val_samples splits the train set")
        generator = torch.Generator().manual_seed(config.get("seed", 42))
        indices = torch.randperm(len(train_dataset), generator=generator).tolist()
        val_indices = indices[:max_val_samples]
        train_indices = indices[max_val_samples:]
        train_dataset = Subset(train_dataset, train_indices)
        val_base = CheXpertUMSDataset(
            data_root=data_cfg["data_root"],
            ums_jsonl_path=data_cfg["train_ums_path"],
            transform=get_val_transforms(image_size),
            is_train=False,
            use_common_labels_only=data_cfg.get("use_common_labels_only", False),
            selected_labels=selected_labels,
            max_samples=max_train_samples,
            schema_mode=data_cfg.get("schema_mode", "state_only"),
        )
        val_dataset = Subset(val_base, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"].get("eval_batch_size", config["training"]["batch_size"]),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader, list(get_base_dataset(train_dataset).label_names)


def labels_to_state_targets(labels: torch.Tensor) -> torch.Tensor:
    targets = torch.zeros(labels.shape, dtype=torch.long, device=labels.device)
    finite = torch.isfinite(labels)
    targets[finite & (labels == 0)] = STATE_TO_INDEX["absent"]
    targets[finite & (labels == -1)] = STATE_TO_INDEX["uncertain"]
    targets[finite & (labels == 1)] = STATE_TO_INDEX["present"]
    return targets


def prepare_labels_for_metrics(labels: np.ndarray, policy: str) -> np.ndarray:
    labels = labels.copy()
    labels[np.isnan(labels)] = np.nan
    if policy == "ignore":
        labels[labels == -1] = np.nan
    elif policy == "positive":
        labels[labels == -1] = 1.0
    elif policy == "negative":
        labels[labels == -1] = 0.0
    else:
        raise ValueError(f"Unknown uncertain_policy: {policy}")
    return labels


class UMSStateClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        num_labels: int,
        drop_rate: float,
        drop_path_rate: float,
        auxiliary_targets: List[str] | None = None,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        embed_dim = getattr(self.backbone, "num_features", None) or self.backbone.embed_dim
        self.head = nn.Linear(embed_dim, num_labels * len(STATE_TO_INDEX))
        self.auxiliary_targets = list(auxiliary_targets or [])
        self.auxiliary_heads = nn.ModuleDict(
            {target: nn.Linear(embed_dim, num_labels) for target in self.auxiliary_targets}
        )
        self.num_labels = num_labels

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return self.state_logits_from_features(features, images.shape[0])

    def state_logits_from_features(self, features: torch.Tensor, batch_size: int) -> torch.Tensor:
        logits = self.head(features)
        return logits.view(batch_size, self.num_labels, len(STATE_TO_INDEX))

    def forward_with_aux(self, images: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        features = self.backbone(images)
        state_logits = self.state_logits_from_features(features, images.shape[0])
        auxiliary_logits = {
            target: head(features)
            for target, head in self.auxiliary_heads.items()
        }
        return state_logits, auxiliary_logits


def get_schema_auxiliary_targets(config: Dict[str, Any]) -> List[str]:
    raw = config.get("model", {}).get("schema_auxiliary_targets", [])
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    targets = list(dict.fromkeys(raw))
    invalid = [target for target in targets if target not in AUXILIARY_TARGETS]
    if invalid:
        raise ValueError(f"Unknown schema_auxiliary_targets: {invalid}")
    return targets


def build_state_weights(config: Dict[str, Any], device: torch.device) -> torch.Tensor:
    raw = config.get("training", {}).get("state_class_weights", {})
    weights = [
        float(raw.get("null", 1.0)),
        float(raw.get("absent", 1.0)),
        float(raw.get("uncertain", 1.0)),
        float(raw.get("present", 1.0)),
    ]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_auxiliary_loss_weights(config: Dict[str, Any]) -> Dict[str, float]:
    raw = config.get("training", {}).get("schema_auxiliary_loss_weights", {})
    return {target: float(raw.get(target, 1.0)) for target in AUXILIARY_TARGETS}


def labels_to_auxiliary_targets(
    labels: torch.Tensor,
    answerable: torch.Tensor,
    auxiliary_targets: List[str],
) -> Dict[str, torch.Tensor]:
    targets: Dict[str, torch.Tensor] = {}
    if "answerability" in auxiliary_targets:
        targets["answerability"] = answerable.float()
    if "uncertainty" in auxiliary_targets:
        targets["uncertainty"] = (torch.isfinite(labels) & (labels == -1)).float()
    return targets


def compute_loss_and_logits(
    model: UMSStateClassifier,
    images: torch.Tensor,
    labels: torch.Tensor,
    answerable: torch.Tensor,
    state_criterion: nn.Module,
    auxiliary_criterion: nn.Module,
    auxiliary_targets: List[str],
    auxiliary_loss_weights: Dict[str, float],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, float]]:
    if auxiliary_targets:
        logits, auxiliary_logits = model.forward_with_aux(images)
    else:
        logits = model(images)
        auxiliary_logits = {}

    state_targets = labels_to_state_targets(labels)
    state_loss = state_criterion(logits.reshape(-1, len(STATE_TO_INDEX)), state_targets.reshape(-1))
    loss = state_loss
    loss_parts = {"state_loss": float(state_loss.detach().item())}

    target_tensors = labels_to_auxiliary_targets(labels, answerable, auxiliary_targets)
    for target in auxiliary_targets:
        target_loss = auxiliary_criterion(auxiliary_logits[target], target_tensors[target])
        loss = loss + auxiliary_loss_weights.get(target, 1.0) * target_loss
        loss_parts[f"{target}_loss"] = float(target_loss.detach().item())

    return loss, logits, auxiliary_logits, loss_parts


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    state_criterion: nn.Module,
    auxiliary_criterion: nn.Module,
    device: torch.device,
    label_names: List[str],
    uncertain_policy: str,
    threshold: float,
    auxiliary_targets: List[str],
    auxiliary_loss_weights: Dict[str, float],
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_present_prob = []
    all_labels = []
    all_state_targets = []
    all_state_preds = []
    all_auxiliary_targets = {target: [] for target in auxiliary_targets}
    all_auxiliary_probs = {target: [] for target in auxiliary_targets}

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        answerable = batch["answerable"].to(device)
        targets = labels_to_state_targets(labels)
        loss, logits, auxiliary_logits, _ = compute_loss_and_logits(
            model=model,
            images=images,
            labels=labels,
            answerable=answerable,
            state_criterion=state_criterion,
            auxiliary_criterion=auxiliary_criterion,
            auxiliary_targets=auxiliary_targets,
            auxiliary_loss_weights=auxiliary_loss_weights,
        )
        total_loss += float(loss.item())
        num_batches += 1

        probs = torch.softmax(logits, dim=-1)
        all_present_prob.append(probs[..., STATE_TO_INDEX["present"]].cpu().numpy())
        all_labels.append(prepare_labels_for_metrics(labels.cpu().numpy(), uncertain_policy))
        all_state_targets.append(targets.cpu().numpy())
        all_state_preds.append(logits.argmax(dim=-1).cpu().numpy())
        auxiliary_target_tensors = labels_to_auxiliary_targets(labels, answerable, auxiliary_targets)
        for target in auxiliary_targets:
            all_auxiliary_targets[target].append(auxiliary_target_tensors[target].cpu().numpy())
            all_auxiliary_probs[target].append(torch.sigmoid(auxiliary_logits[target]).cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_present_prob, axis=0)
    y_pred = (y_prob >= threshold).astype(int)
    state_true = np.concatenate(all_state_targets, axis=0)
    state_pred = np.concatenate(all_state_preds, axis=0)

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        label_names=label_names,
        threshold=threshold,
    )
    metrics["state_accuracy_all_fields"] = float((state_true == state_pred).mean())
    answerable_mask = ~np.isnan(y_true)
    if answerable_mask.any():
        metrics["state_accuracy_answerable_fields"] = float((state_true[answerable_mask] == state_pred[answerable_mask]).mean())

    for target in auxiliary_targets:
        aux_true = np.concatenate(all_auxiliary_targets[target], axis=0)
        aux_prob = np.concatenate(all_auxiliary_probs[target], axis=0)
        aux_pred = (aux_prob >= threshold).astype(int)
        aux_metrics = compute_classification_metrics(
            y_true=aux_true,
            y_pred=aux_pred,
            y_prob=aux_prob,
            label_names=label_names,
            threshold=threshold,
        )
        metrics[f"{target}_support_positive"] = int(aux_true.sum())
        metrics[f"{target}_prevalence"] = float(aux_true.mean())
        metrics[f"{target}_pred_rate"] = float(aux_pred.mean())
        metrics[f"{target}_accuracy"] = float((aux_pred == aux_true).mean())
        metrics[f"{target}_macro_auc"] = aux_metrics.get("macro_auc")
        metrics[f"{target}_macro_f1"] = aux_metrics.get("macro_f1")
        metrics[f"{target}_micro_f1"] = aux_metrics.get("micro_f1")

    return {
        "val_loss": total_loss / max(num_batches, 1),
        "metrics": metrics,
    }


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_checkpoint(path: Path, model: UMSStateClassifier, optimizer, scheduler, step: int, best_val_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vit": model.backbone.state_dict(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "best_val_loss": best_val_loss,
            "state_to_index": STATE_TO_INDEX,
            "schema_auxiliary_targets": model.auxiliary_targets,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train no-LLM UMS state classifier")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
        base_dir = config["training"]["output_dir"].rstrip("/")
        config["training"]["output_dir"] = f"{base_dir}_seed{args.seed}"

    if args.debug:
        config["data"]["max_train_samples"] = 200
        config["data"]["max_val_samples"] = 50
        config["data"]["num_workers"] = 0
        config["training"]["batch_size"] = 8
        config["training"]["eval_batch_size"] = 8
        config["training"]["gradient_accumulation_steps"] = 1
        config["training"]["max_steps"] = 20
        config["training"]["warmup_steps"] = 2
        config["training"]["log_interval"] = 2
        config["training"]["eval_interval"] = 5
        config["training"]["save_interval"] = 20
        config["training"]["bf16"] = False
        config["training"]["fp16"] = False

    output_dir = Path(config["training"]["output_dir"])
    if (output_dir / "metrics_final.json").exists():
        print(f"ERROR: {output_dir / 'metrics_final.json'} already exists. Delete manually to re-run.")
        sys.exit(1)
    if args.resume is not None:
        config["resume_from"] = args.resume
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config_snapshot.json", config)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    requested_device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA not available, falling back to CPU")
    elif requested_device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device(requested_device)
    print(f"Using device: {device}")

    print("\nCreating dataloaders...")
    train_loader, val_loader, label_names = create_dataloaders(config)
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Labels: {label_names}")

    model_cfg = config["model"]
    auxiliary_targets = get_schema_auxiliary_targets(config)
    model = UMSStateClassifier(
        model_name=model_cfg["vit_model_name"],
        pretrained=bool(model_cfg.get("vit_pretrained", True)),
        num_labels=len(label_names),
        drop_rate=float(model_cfg.get("drop_rate", 0.0)),
        drop_path_rate=float(model_cfg.get("drop_path_rate", 0.1)),
        auxiliary_targets=auxiliary_targets,
    ).to(device)
    if auxiliary_targets:
        print(f"  Schema auxiliary targets: {auxiliary_targets}")

    training_cfg = config["training"]
    state_criterion = nn.CrossEntropyLoss(weight=build_state_weights(config, device))
    auxiliary_criterion = nn.BCEWithLogitsLoss()
    auxiliary_loss_weights = build_auxiliary_loss_weights(config)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    warmup_steps = int(training_cfg["warmup_steps"])
    max_steps = int(training_cfg["max_steps"])
    if warmup_steps >= max_steps:
        warmup_steps = max(1, max_steps // 10)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=max_steps - warmup_steps, eta_min=float(training_cfg["learning_rate"]) * 0.1),
        ],
        milestones=[warmup_steps],
    )

    best_val_loss = float("inf")
    global_step = 0
    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(resume_path)
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = int(checkpoint.get("step", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        if global_step <= 0:
            raise RuntimeError(f"Resume checkpoint {resume_path} has invalid step={global_step}")
        if global_step >= max_steps:
            raise RuntimeError(
                f"Resume checkpoint step={global_step} is already >= max_steps={max_steps}"
            )
        print(f"Resumed from {resume_path} at step {global_step}, best_val_loss={best_val_loss:.6f}")

    use_amp = device.type == "cuda" and (training_cfg.get("bf16", False) or training_cfg.get("fp16", False))
    use_bf16 = bool(training_cfg.get("bf16", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if training_cfg.get("fp16", False) and device.type == "cuda" else None

    gradient_accumulation_steps = int(training_cfg["gradient_accumulation_steps"])
    accumulated_loss = 0.0
    num_accumulated = 0
    train_iter = iter(train_loader)

    print("\nStarting no-LLM UMS classifier training...")
    print(f"  Max steps: {max_steps}")
    print(f"  Output dir: {output_dir}")
    progress_bar = tqdm(total=max_steps, initial=global_step, desc="Training")
    save_json(
        output_dir / "progress.json",
        {
            "status": "running",
            "global_step": global_step,
            "max_steps": max_steps,
            "updated_at": time_stamp(),
        },
    )

    while global_step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        model.train()
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)

        if use_amp:
            autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
            autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=autocast_dtype)
        else:
            autocast_ctx = contextlib.nullcontext()

        with autocast_ctx:
            loss, _, _, _ = compute_loss_and_logits(
                model=model,
                images=images,
                labels=labels,
                answerable=batch["answerable"].to(device),
                state_criterion=state_criterion,
                auxiliary_criterion=auxiliary_criterion,
                auxiliary_targets=auxiliary_targets,
                auxiliary_loss_weights=auxiliary_loss_weights,
            )

        scaled_loss = loss / gradient_accumulation_steps
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        accumulated_loss += float(loss.item())
        num_accumulated += 1

        if num_accumulated >= gradient_accumulation_steps:
            if float(training_cfg["max_grad_norm"]) > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_cfg["max_grad_norm"]))

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            progress_bar.update(1)

            if global_step % int(training_cfg["log_interval"]) == 0:
                avg_loss = accumulated_loss / max(num_accumulated, 1)
                progress_bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
                save_json(
                    output_dir / "progress.json",
                    {
                        "status": "running",
                        "global_step": global_step,
                        "max_steps": max_steps,
                        "train_loss": avg_loss,
                        "lr": scheduler.get_last_lr()[0],
                        "updated_at": time_stamp(),
                    },
                )
                accumulated_loss = 0.0
                num_accumulated = 0

            if global_step % int(training_cfg["eval_interval"]) == 0:
                eval_cfg = config.get("evaluation", {})
                result = evaluate(
                    model=model,
                    dataloader=val_loader,
                    state_criterion=state_criterion,
                    auxiliary_criterion=auxiliary_criterion,
                    device=device,
                    label_names=label_names,
                    uncertain_policy=eval_cfg.get("uncertain_policy", "ignore"),
                    threshold=float(eval_cfg.get("threshold", 0.5)),
                    auxiliary_targets=auxiliary_targets,
                    auxiliary_loss_weights=auxiliary_loss_weights,
                )
                save_json(output_dir / f"metrics_step_{global_step}.json", result)
                val_loss = float(result["val_loss"])
                print(f"\nStep {global_step}: val_loss = {val_loss:.4f}, macro_auc = {result['metrics'].get('macro_auc')}")
                save_json(
                    output_dir / "progress.json",
                    {
                        "status": "running",
                        "global_step": global_step,
                        "max_steps": max_steps,
                        "val_loss": val_loss,
                        "macro_auc": result["metrics"].get("macro_auc"),
                        "updated_at": time_stamp(),
                    },
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, global_step, best_val_loss)

            if global_step % int(training_cfg["save_interval"]) == 0:
                save_checkpoint(output_dir / f"step_{global_step}.pt", model, optimizer, scheduler, global_step, best_val_loss)

    progress_bar.close()
    save_checkpoint(output_dir / "final.pt", model, optimizer, scheduler, global_step, best_val_loss)

    eval_cfg = config.get("evaluation", {})
    final_result = evaluate(
        model=model,
        dataloader=val_loader,
        state_criterion=state_criterion,
        auxiliary_criterion=auxiliary_criterion,
        device=device,
        label_names=label_names,
        uncertain_policy=eval_cfg.get("uncertain_policy", "ignore"),
        threshold=float(eval_cfg.get("threshold", 0.5)),
        auxiliary_targets=auxiliary_targets,
        auxiliary_loss_weights=auxiliary_loss_weights,
    )
    save_json(output_dir / "metrics_final.json", final_result)
    save_json(
        output_dir / "progress.json",
        {
            "status": "completed",
            "global_step": global_step,
            "max_steps": max_steps,
            "val_loss": final_result.get("val_loss"),
            "macro_auc": final_result.get("metrics", {}).get("macro_auc"),
            "updated_at": time_stamp(),
        },
    )
    print("Training completed!")


if __name__ == "__main__":
    main()
