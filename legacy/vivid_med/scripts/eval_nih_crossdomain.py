"""
NIH ChestX-ray14 Cross-Domain Evaluation

用 CheXpert 训练的 linear probe checkpoint 直接在 NIH 上测试（8 个共同标签）。
不需要重新训练，只做 inference + metrics。

用法:
    python scripts/eval_nih_crossdomain.py \
        --checkpoint ./outputs/linear_probe_v10.1_spd_full14/best.pt \
        --label V10.1_SPD
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import timm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.transforms import get_val_transforms
from evaluation.metrics import compute_classification_metrics

# CheXpert 14 labels (same order as training)
CHEXPERT_14_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]

# 8 common labels between CheXpert and NIH (CheXpert naming)
COMMON_LABELS = [
    "No Finding", "Atelectasis", "Cardiomegaly", "Consolidation",
    "Edema", "Pleural Effusion", "Pneumonia", "Pneumothorax"
]


class NIHChestXrayDataset(Dataset):
    """NIH ChestX-ray14 dataset, loads from preprocessed UMS JSONL."""

    NIH_DATA_DIR = "NIH Chest X-rays"

    def __init__(self, data_root: str, ums_jsonl_path: str, transform=None,
                 max_samples=None):
        self.data_root = Path(data_root)
        self.transform = transform or get_val_transforms(224)
        self.label_names = COMMON_LABELS
        self.num_labels = len(self.label_names)

        self.samples = []
        with open(ums_jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.samples.append(json.loads(line.strip()))
        print(f"Loaded {len(self.samples)} NIH samples from {ums_jsonl_path}")

    def _find_image(self, image_index: str) -> Path:
        """Search images_001..012/images/ for the given filename."""
        for i in range(1, 13):
            candidate = self.data_root / self.NIH_DATA_DIR / f"images_{i:03d}" / "images" / image_index
            if candidate.exists():
                return candidate
        return self.data_root / self.NIH_DATA_DIR / "images_001" / "images" / image_index

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_index = sample["extensions"]["image_index"]

        try:
            image = Image.open(self._find_image(image_index)).convert("RGB")
        except Exception as e:
            print(f"Error loading {image_index}: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        labels = torch.full((self.num_labels,), float("nan"))
        findings = sample.get("findings", {})
        for i, name in enumerate(self.label_names):
            if name in findings:
                state = findings[name].get("state")
                if state == "present":
                    labels[i] = 1.0
                elif state == "absent":
                    labels[i] = 0.0

        return {"image": image, "labels": labels}


def nih_collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"images": images, "labels": labels}


def get_common_label_indices(train_labels: List[str], common_labels: List[str]) -> List[int]:
    """Get indices of common labels within the 14-label training head."""
    indices = []
    for name in common_labels:
        if name in train_labels:
            indices.append(train_labels.index(name))
    return indices


@torch.no_grad()
def evaluate_nih(model, dataloader, common_indices, device, threshold=0.5):
    """Evaluate on NIH, selecting only common label logits from 14-label head."""
    model.eval()
    all_probs = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating NIH"):
        images = batch["images"].to(device)
        labels = batch["labels"].cpu().numpy()

        logits = model(images)  # (B, 14)
        # Select only the 8 common label columns
        logits_common = logits[:, common_indices]  # (B, 8)
        probs = torch.sigmoid(logits_common).cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels)

    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        label_names=COMMON_LABELS,
        threshold=threshold,
    )
    return {"num_samples": len(y_true), "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description="NIH ChestX-ray14 Cross-Domain Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Linear probe checkpoint (best.pt)")
    parser.add_argument("--data_root", type=str, default="./data/dataset")
    parser.add_argument("--nih_ums_path", type=str,
                        default="./data/dataset/processed/nih_external_test_ums.jsonl")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--label", type=str, default="unknown",
                        help="Experiment label for display")
    parser.add_argument("--vit_model_name", type=str, default="vit_base_patch16_224",
                        help="timm model name (e.g. vit_base_patch16_dinov3)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Experiment: {args.label}")

    # Load model (14-label head, same as training)
    model = timm.create_model(
        args.vit_model_name, pretrained=False,
        num_classes=len(CHEXPERT_14_LABELS),
        drop_rate=0.0, drop_path_rate=0.0,
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    model_state = state["model"] if "model" in state else state
    model.load_state_dict(model_state)
    print(f"Loaded checkpoint (step {state.get('step', '?')})")

    # Common label index mapping: 14-label head → 8 common labels
    common_indices = get_common_label_indices(CHEXPERT_14_LABELS, COMMON_LABELS)
    print(f"Common label indices in 14-label head: {common_indices}")
    for idx, name in zip(common_indices, COMMON_LABELS):
        print(f"  [{idx}] {name}")

    # Load NIH dataset
    dataset = NIHChestXrayDataset(
        data_root=args.data_root,
        ums_jsonl_path=args.nih_ums_path,
        transform=get_val_transforms(224),
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=nih_collate_fn, pin_memory=True,
    )

    # Evaluate
    result = evaluate_nih(model, loader, common_indices, device)

    # Print results
    m = result["metrics"]
    print(f"\n{'='*60}")
    print(f"NIH Cross-Domain Results [{args.label}]")
    print(f"{'='*60}")
    print(f"Samples: {result['num_samples']}")
    print(f"macro_F1:  {m['macro_f1']:.4f}")
    print(f"macro_AUC: {m['macro_auc']:.4f}" if m.get("macro_auc") else "macro_AUC: N/A")
    print(f"micro_F1:  {m['micro_f1']:.4f}")
    print(f"\nPer-label:")
    for name, lm in m.get("per_label", {}).items():
        auc_str = f"AUC={lm['auc']:.4f}" if lm.get("auc") is not None else "AUC=N/A"
        print(f"  {name:25s}  F1={lm['f1']:.4f}  {auc_str}  support={lm['support']}")

    # Save
    output_path = Path(args.output) if args.output else Path(args.checkpoint).parent / "nih_crossdomain.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result["experiment"] = args.label
    result["checkpoint"] = args.checkpoint
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
