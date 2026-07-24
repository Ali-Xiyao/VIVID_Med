"""Read-only attention-group collapse diagnostic for trained SPD checkpoints."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vivid_spd_clean.io import load_jsonl, sha256_file  # noqa: E402
from vivid_spd_clean.model import build_projector  # noqa: E402


class ValidationImages(Dataset):
    def __init__(
        self,
        *,
        manifest: Path,
        image_root: Path,
        max_rows: int,
    ) -> None:
        rows = [
            row for row in load_jsonl(manifest) if row["split"] == "validate"
        ]
        self.rows = sorted(rows, key=lambda row: str(row["row_id"]))[:max_rows]
        if len(self.rows) != max_rows:
            raise ValueError("locked attention diagnostic row count unavailable")
        self.image_root = image_root
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        row = self.rows[index]
        path = self.image_root / str(row["image_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            pixels = self.transform(image.convert("RGB"))
        return pixels, str(row["row_id"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("ums_spd4x2", "ums_spd4x2_no_ortho"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--hard-manifest", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-rows", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.manual_seed(0)
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload.get("arm") != args.arm:
        raise ValueError("checkpoint arm mismatch")
    vision = payload["vision"]
    backbone = timm.create_model(
        "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        pretrained=False,
        num_classes=0,
        drop_rate=0.0,
        drop_path_rate=0.0,
    )
    incompatible = backbone.load_state_dict(vision["backbone"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("backbone checkpoint mismatch")
    output_dim = int(vision["projector"]["mlp.3.weight"].shape[0])
    projector = build_projector(
        args.arm,
        vision_dim=int(backbone.embed_dim),
        output_dim=output_dim,
    )
    incompatible = projector.load_state_dict(vision["projector"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("projector checkpoint mismatch")
    backbone.to(device).eval()
    projector.to(device).eval()
    dataset = ValidationImages(
        manifest=args.hard_manifest,
        image_root=args.image_root,
        max_rows=args.max_rows,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )
    pair_values: dict[str, list[float]] = {
        f"{left}-{right}": []
        for left, right in itertools.combinations(range(4), 2)
    }
    entropies: list[list[float]] = [[] for _ in range(4)]
    row_ids: list[str] = []
    with torch.inference_mode():
        for images, batch_ids in loader:
            tokens = backbone.forward_features(images.to(device))
            projector(tokens)
            maps = [weights.mean(dim=1) for weights in projector._attention_maps]
            for left, right in itertools.combinations(range(4), 2):
                values = F.cosine_similarity(
                    maps[left], maps[right], dim=-1
                )
                pair_values[f"{left}-{right}"].extend(
                    values.detach().cpu().tolist()
                )
            token_count = maps[0].shape[-1]
            for group, weights in enumerate(maps):
                entropy = -(
                    weights.clamp_min(1e-12)
                    * weights.clamp_min(1e-12).log()
                ).sum(dim=-1) / math.log(token_count)
                entropies[group].extend(entropy.detach().cpu().tolist())
            row_ids.extend(batch_ids)
    pair_summary = {
        key: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        for key, values in pair_values.items()
    }
    all_cosines = [value for values in pair_values.values() for value in values]
    result = {
        "schema_version": 1,
        "artifact": "strict_vivid_spd_attention_group_diagnostic",
        "arm": args.arm,
        "promotion_effect": "none",
        "rows": len(dataset),
        "selection": "lexicographically_first_validation_row_id",
        "pairwise_attention_cosine": pair_summary,
        "overall_attention_cosine": {
            "mean": float(np.mean(all_cosines)),
            "std": float(np.std(all_cosines)),
            "max": float(np.max(all_cosines)),
        },
        "normalized_attention_entropy": {
            str(group): {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
            for group, values in enumerate(entropies)
        },
        "hashes": {
            "checkpoint": sha256_file(args.checkpoint),
            "hard_manifest": sha256_file(args.hard_manifest),
        },
        "first_row_id": row_ids[0],
        "last_row_id": row_ids[-1],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
