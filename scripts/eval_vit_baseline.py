"""
Evaluate ViT baseline on a dataset split

用法:
    python eval_vit_baseline.py --config ../configs/baseline_vit_chexpert.yaml --checkpoint ../outputs/baseline_vit/best.pt
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import yaml
import timm
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# 添加项目根目录到 path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import CheXpertUMSDataset, get_val_transforms
from data.chexpert_dataset import collate_fn
from evaluation.metrics import compute_classification_metrics


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_base_dataset(dataset):
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def create_val_loader(config: Dict[str, Any]):
    data_cfg = config["data"]
    image_size = data_cfg["image_size"]
    data_root = data_cfg["data_root"]
    val_ums_path = data_cfg.get("val_ums_path") or data_cfg["train_ums_path"]
    use_common_labels_only = data_cfg.get("use_common_labels_only", False)
    max_val_samples = data_cfg.get("max_val_samples", 1000)

    val_dataset = CheXpertUMSDataset(
        data_root=data_root,
        ums_jsonl_path=val_ums_path,
        transform=get_val_transforms(image_size),
        is_train=False,
        use_common_labels_only=use_common_labels_only,
        selected_labels=data_cfg.get("selected_labels"),
        max_samples=max_val_samples,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    label_names = get_base_dataset(val_dataset).label_names
    return val_loader, label_names


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


@torch.no_grad()
def evaluate(model, dataloader, device, policy: str, threshold: float, label_names: list):
    model.eval()
    all_probs = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)

        logits = model(images)
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

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate ViT baseline")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent.parent / "configs" / "baseline_vit_chexpert.yaml"),
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Path to metrics json")
    args = parser.parse_args()

    config = load_config(args.config)
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    val_loader, label_names = create_val_loader(config)

    model_cfg = config["model"]
    num_labels = len(label_names)
    model = timm.create_model(
        model_cfg["vit_model_name"],
        pretrained=False,
        num_classes=num_labels,
        drop_rate=model_cfg.get("drop_rate", 0.0),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.1),
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"], strict=True)

    eval_cfg = config.get("evaluation", {})
    uncertain_policy = eval_cfg.get("uncertain_policy", "ignore")
    threshold = float(eval_cfg.get("threshold", 0.5))

    metrics = evaluate(model, val_loader, device, uncertain_policy, threshold, label_names)

    output_path = Path(args.output) if args.output else Path(args.checkpoint).with_suffix(".metrics.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Metrics saved to {output_path}")


if __name__ == "__main__":
    main()
