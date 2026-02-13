"""
NIH External Test — 用 CheXpert 训练的分类模型直接在 NIH 上评估

用法:
    python eval_nih_external.py --config ../configs/transfer_cls_v9_nih.yaml

前置条件:
    1. 已完成 CheXpert transfer 训练 (best.pt 存在)
    2. NIH UMS JSONL 已生成
"""

import argparse
import json
import contextlib
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import yaml
import timm
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import CheXpertUMSDataset, get_val_transforms
from data.chexpert_dataset import collate_fn
from evaluation.metrics import compute_classification_metrics


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_labels_for_metrics(labels: np.ndarray) -> np.ndarray:
    labels = labels.copy()
    labels[labels == -1] = np.nan
    return labels


def main():
    parser = argparse.ArgumentParser(description="NIH External Test")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    model_cfg = config["model"]
    transfer_cfg = config.get("transfer", {})
    checkpoint_path = transfer_cfg["init_checkpoint"]
    threshold = float(eval_cfg.get("threshold", 0.5))
    output_dir = Path(eval_cfg.get("output_dir", "./outputs/transfer_cls_v9_nih"))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    print(f"Device: {device}")

    # 加载 NIH 测试集
    selected_labels = data_cfg["selected_labels"]
    test_dataset = CheXpertUMSDataset(
        data_root=data_cfg["data_root"],
        ums_jsonl_path=data_cfg["test_ums_path"],
        transform=get_val_transforms(data_cfg["image_size"]),
        is_train=False,
        selected_labels=selected_labels,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )
    print(f"NIH test samples: {len(test_dataset)}")

    # 创建模型并加载 checkpoint
    num_labels = len(selected_labels)
    model = timm.create_model(
        model_cfg["vit_model_name"],
        pretrained=False,
        num_classes=num_labels,
        drop_rate=model_cfg.get("drop_rate", 0.0),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.1),
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    model_state = state.get("model", state)
    model.load_state_dict(model_state, strict=True)
    print(f"Loaded checkpoint from {checkpoint_path}")

    # 评估
    model.eval()
    all_probs, all_labels = [], []

    use_bf16 = device == "cuda"
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()

    with torch.no_grad(), autocast_ctx:
        for batch in tqdm(test_loader, desc="Evaluating on NIH"):
            images = batch["images"].to(device)
            labels = batch["labels"]
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(prepare_labels_for_metrics(labels.numpy()))

    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_classification_metrics(
        y_true=y_true, y_pred=y_pred, y_prob=y_prob,
        label_names=selected_labels, threshold=threshold,
    )

    result = {"dataset": "NIH_external", "checkpoint": checkpoint_path, "metrics": metrics}
    result_path = output_dir / "nih_external_metrics.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {result_path}")
    print(f"  macro_F1: {metrics.get('macro_f1', 'N/A')}")
    print(f"  macro_AUC: {metrics.get('macro_auc', 'N/A')}")
    for label in selected_labels:
        f1 = metrics.get("per_label", {}).get(label, {}).get("f1", "N/A")
        auc = metrics.get("per_label", {}).get(label, {}).get("auc", "N/A")
        print(f"  {label}: F1={f1}, AUC={auc}")


if __name__ == "__main__":
    main()
