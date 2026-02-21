"""Convert BiomedCLIP open_clip weights to timm ViT-B/16 format.

Extracts visual.trunk.* keys, strips prefix, saves as standard checkpoint.
"""
import torch
import argparse


def convert_biomedclip_to_timm(src_path: str, dst_path: str):
    print(f"Loading BiomedCLIP weights from {src_path}")
    state = torch.load(src_path, map_location="cpu")

    # Extract visual trunk keys and strip prefix
    prefix = "visual.trunk."
    vit_state = {}
    skipped = []
    for k, v in state.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            vit_state[new_key] = v
        else:
            skipped.append(k)

    # Also skip the CLIP projection head (visual.head.*)
    print(f"Extracted {len(vit_state)} ViT keys, skipped {len(skipped)} non-visual keys")
    print(f"Sample keys: {list(vit_state.keys())[:5]}")

    # Save in the format expected by load_vit_backbone_from_checkpoint
    torch.save({"model": vit_state}, dst_path)
    print(f"Saved to {dst_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="H:/Xiyao_Wang/001_models/biomedclip/open_clip_pytorch_model.bin")
    parser.add_argument("--dst", default="./pretrained/biomedclip_vit_base.pt")
    args = parser.parse_args()
    convert_biomedclip_to_timm(args.src, args.dst)
