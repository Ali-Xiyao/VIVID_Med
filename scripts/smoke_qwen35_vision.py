"""Read-only Qwen3.5 vision-only loading and patch-grid smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bives_cxr.backbones import Qwen35VisionAdapter, load_qwen35_visual_and_processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    visual, processor, config = load_qwen35_visual_and_processor(
        args.model_path,
        dtype=args.dtype,
    )
    language_parameters = sum(
        parameter.numel()
        for name, parameter in visual.named_parameters()
        if "language" in name or "embed_tokens" in name or "lm_head" in name
    )
    if language_parameters:
        raise RuntimeError(f"vision-only loader retained {language_parameters} language parameters")
    visual = visual.to(device).eval()

    if args.image is None:
        image = Image.new("RGB", (448, 448), (127, 127, 127))
    else:
        image = Image.open(args.image).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "A right pleural effusion is present."},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    batch = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    adapter = Qwen35VisionAdapter(
        visual,
        spatial_merge_size=int(config["vision_config"]["spatial_merge_size"]),
    )
    with torch.no_grad():
        patches = adapter(
            batch["pixel_values"].to(device=device, dtype=next(visual.parameters()).dtype),
            batch["image_grid_thw"].to(device),
        )
    payload = {
        "model_type": config["model_type"],
        "visual_class": type(visual).__name__,
        "visual_parameters": sum(parameter.numel() for parameter in visual.parameters()),
        "language_parameters": language_parameters,
        "grid_hw": patches.grid_hw,
        "token_shape": list(patches.tokens.shape),
        "token_dtype": str(patches.tokens.dtype),
        "valid_patches": int(patches.valid_mask.sum().item()),
        "finite": bool(torch.isfinite(patches.tokens).all()),
    }
    print(json.dumps(payload, indent=2))
    if not payload["finite"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
