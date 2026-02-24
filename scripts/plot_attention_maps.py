"""
SPD Attention Map Visualization
展示 SPD 3 个 group 分别关注的解剖区域，证明正交分解的有效性。
"""

import argparse
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.functional import cosine_similarity, interpolate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import CheXpertUMSDataset, get_val_transforms
from data.chexpert_dataset import collate_fn
from models.vivid_model import VIVIDModel
from models.spd import SPDProjector

import yaml
def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_vivid_model(config_path: str, checkpoint_path: str, device: str):
    """Load VIVID model with SPD projector from pretraining checkpoint."""
    config = load_config(config_path)
    model_cfg = config["model"]
    spd_cfg = model_cfg.get("spd", {})
    model = VIVIDModel(
        vit_model_name=model_cfg.get("vit_model_name", "vit_base_patch16_224"),
        vit_pretrained=False,
        vit_output_type=model_cfg.get("vit_output_type", "all"),
        projector_mlp_hidden_dim=model_cfg.get("projector_mlp_hidden_dim"),
        projector_dropout=model_cfg.get("projector_dropout", 0.1),
        llm_model_name=model_cfg.get("llm_model_name", "Qwen/Qwen3-1.7B"),
        load_llm=False,
        spd_enabled=spd_cfg.get("enabled", True),
        spd_num_groups=spd_cfg.get("num_groups", 3),
        spd_tokens_per_group=spd_cfg.get("tokens_per_group", 2),
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    # Pretraining checkpoint has separate 'vit' and 'projector' keys
    if "vit" in state and "projector" in state:
        model.vit.load_state_dict(state["vit"], strict=False)
        model.projector.load_state_dict(state["projector"], strict=False)
    else:
        model_state = state["model"] if "model" in state else state
        model.load_state_dict(model_state, strict=False)
    model.eval()
    return model


def extract_attention_maps(model, images, device):
    """Run forward pass and extract SPD cross-attention weights."""
    projector = model.projector
    assert isinstance(projector, SPDProjector), "Model must use SPD projector"

    # Clear previous attention weights
    projector._attn_weights = []

    with torch.no_grad():
        # Encode image through ViT
        vit_features = model.vit(images.to(device))  # (B, N, D)
        # Forward through projector to populate _attn_weights
        _ = projector(vit_features)

    # attn_weights[group_idx]: (B, tokens_per_group, num_vit_tokens)
    attn_weights = projector._attn_weights
    return attn_weights, vit_features


def visualize_sample_separate(images, attn_weights, sample_idx, output_dir,
                              num_groups=3, grid_size=14):
    """Save original image and each group attention map as separate files."""
    img = images[sample_idx].cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    # Save original image
    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
    ax.imshow(img, cmap="gray" if img.shape[2] == 1 else None)
    ax.axis("off")
    p = os.path.join(output_dir, f"sample{sample_idx}_input.png")
    fig.savefig(p, dpi=200, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print(f"Saved: {p}")

    # Save each group attention map
    for g in range(num_groups):
        attn = attn_weights[g][sample_idx].mean(dim=0)
        attn_spatial = attn[1:].reshape(grid_size, grid_size).cpu().numpy()
        attn_up = np.array(
            interpolate(
                torch.tensor(attn_spatial).unsqueeze(0).unsqueeze(0).float(),
                size=(224, 224), mode="bilinear", align_corners=False,
            ).squeeze()
        )
        fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
        ax.imshow(img, cmap="gray" if img.shape[2] == 1 else None)
        ax.imshow(attn_up, cmap="jet", alpha=0.5, vmin=0)
        ax.axis("off")
        p = os.path.join(output_dir, f"sample{sample_idx}_group{g+1}.png")
        fig.savefig(p, dpi=200, bbox_inches="tight", pad_inches=0.02, facecolor="white")
        plt.close(fig)
        print(f"Saved: {p}")


def compute_orthogonality(attn_weights):
    """Compute pairwise cosine similarity between group attention patterns."""
    group_attns = []
    for attn_w in attn_weights:
        avg_attn = attn_w.mean(dim=1)  # (B, N)
        group_attns.append(avg_attn)

    n = len(group_attns)
    sims = {}
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(group_attns[i], group_attns[j], dim=-1)
            sims[f"G{i+1}-G{j+1}"] = sim.mean().item()
    return sims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="configs/ablation_A_ums_spd_12label.yaml")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/ablation_A_ums_spd_12label/checkpoints/best.pt")
    parser.add_argument("--num_samples", type=int, default=8,
                        help="Number of samples to visualize")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/attention_maps")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("Loading VIVID model with SPD projector...")
    model = load_vivid_model(args.config, args.checkpoint, device)
    num_groups = model.projector.num_groups
    print(f"  SPD groups: {num_groups}")

    # Load val dataset
    dataset = CheXpertUMSDataset(
        data_root="./data/dataset",
        ums_jsonl_path="./data/dataset/processed/chexpert_ums_val.jsonl",
        transform=get_val_transforms(224),
        is_train=False,
        use_common_labels_only=False,
        max_samples=args.num_samples,
    )
    loader = DataLoader(dataset, batch_size=args.num_samples, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)

    batch = next(iter(loader))
    images = batch["images"].to(device)
    print(f"Loaded {images.shape[0]} samples")

    # Extract attention maps
    attn_weights, _ = extract_attention_maps(model, images, device)

    # Visualize each sample
    for i in range(min(args.num_samples, images.shape[0])):
        out_path = os.path.join(args.output_dir, f"attn_sample_{i}.png")
        visualize_sample(images, attn_weights, i, out_path, num_groups)

    # Compute orthogonality
    sims = compute_orthogonality(attn_weights)
    print(f"\nOrthogonality (cosine similarity between groups):")
    for pair, sim in sims.items():
        print(f"  {pair}: {sim:.4f}")

    # Save summary
    import json
    summary = {"orthogonality": sims, "num_samples": images.shape[0]}
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
