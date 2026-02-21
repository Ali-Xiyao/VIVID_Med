"""
t-SNE visualization: compare CLS embeddings from different pretrained ViTs.
Compares ImageNet supervised / BiomedCLIP / VIVID-Med (SPD) on CheXpert val set.
"""

import argparse
import json
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import CheXpertUMSDataset, get_val_transforms
from data.chexpert_dataset import collate_fn

LABELS_12 = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Fracture", "Support Devices",
]


def extract_cls_embeddings(model, dataloader, device, max_samples=2000):
    """Extract CLS token embeddings from ViT backbone."""
    all_emb, all_labels = [], []
    n = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting"):
            images = batch["images"].to(device)
            labels = batch["labels"]
            feat = model.forward_features(images)  # (B, tokens, D)
            cls = feat[:, 0, :]  # (B, D)
            all_emb.append(cls.cpu().numpy())
            all_labels.append(labels.numpy())
            n += images.shape[0]
            if max_samples and n >= max_samples:
                break
    emb = np.concatenate(all_emb)[:max_samples]
    lab = np.concatenate(all_labels)[:max_samples]
    return emb, lab


def load_model(checkpoint_path, vit_model_name, num_classes, device):
    """Create timm model and load checkpoint."""
    model = timm.create_model(
        vit_model_name, pretrained=False,
        num_classes=num_classes, drop_rate=0.0, drop_path_rate=0.0,
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model_state = state["model"] if "model" in state else state
    model.load_state_dict(model_state)
    return model


def main():
    parser = argparse.ArgumentParser(description="t-SNE visualization of CLS embeddings")
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--perplexity", type=int, default=30)
    parser.add_argument("--output", type=str, default="outputs/tsne_comparison.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset (val set)
    dataset = CheXpertUMSDataset(
        data_root="./data/dataset",
        ums_jsonl_path="./data/dataset/processed/chexpert_ums_val.jsonl",
        transform=get_val_transforms(224),
        is_train=False,
        use_common_labels_only=False,
        max_samples=args.max_samples,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False,
                        num_workers=4, collate_fn=collate_fn, pin_memory=True)

    # Models to compare
    models_cfg = [
        ("ImageNet sup.", "./outputs/linear_probe_imagenet_full14/best.pt", "vit_base_patch16_224"),
        ("BiomedCLIP", "./outputs/lp_biomedclip_baseline_seed0/best.pt", "vit_base_patch16_224"),
        ("VIVID-Med (SPD)", "./outputs/lp_A_ums_spd_12label/best.pt", "vit_base_patch16_224"),
    ]

    embeddings_list = []
    labels_list = []
    names = []

    for name, ckpt, vit_name in models_cfg:
        if not os.path.exists(ckpt):
            print(f"SKIP {name}: {ckpt} not found")
            continue
        print(f"\n=== {name} ===")
        model = load_model(ckpt, vit_name, num_classes=14, device=device)
        emb, lab = extract_cls_embeddings(model, loader, device, args.max_samples)
        embeddings_list.append(emb)
        labels_list.append(lab)
        names.append(name)
        print(f"  Extracted {emb.shape[0]} samples, dim={emb.shape[1]}")
        del model
        torch.cuda.empty_cache()

    if len(names) < 2:
        print("Need at least 2 models. Check checkpoint paths.")
        return

    # Run t-SNE on concatenated embeddings
    all_emb = np.concatenate(embeddings_list)
    print(f"\nRunning t-SNE on {all_emb.shape[0]} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=args.perplexity, max_iter=1000)
    coords = tsne.fit_transform(all_emb)

    # Split back
    sizes = [e.shape[0] for e in embeddings_list]
    coords_split = np.split(coords, np.cumsum(sizes)[:-1])

    # --- Plot: one subplot per model, colored by dominant finding ---
    label_names = dataset.label_names
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5.5), dpi=150)
    if len(names) == 1:
        axes = [axes]

    # Color by dominant positive finding (first positive label index)
    cmap = plt.cm.get_cmap("tab10", 10)

    for ax, name, c2d, lab in zip(axes, names, coords_split, labels_list):
        # Assign color: dominant finding index (first positive)
        dominant = np.full(lab.shape[0], -1)
        for i in range(lab.shape[0]):
            pos = np.where(lab[i] == 1)[0]
            if len(pos) > 0:
                dominant[i] = pos[0] % 10
        mask_pos = dominant >= 0
        mask_neg = ~mask_pos

        ax.scatter(c2d[mask_neg, 0], c2d[mask_neg, 1], c="lightgray", s=8, alpha=0.3, label="No finding")
        for ci in range(10):
            m = dominant == ci
            if m.sum() > 0:
                ax.scatter(c2d[m, 0], c2d[m, 1], c=[cmap(ci)], s=10, alpha=0.5)
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("t-SNE of CLS Embeddings (colored by dominant finding)", fontsize=14, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"\nSaved to {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
