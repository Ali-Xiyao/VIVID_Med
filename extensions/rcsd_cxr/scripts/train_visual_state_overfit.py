"""Fail-closed 256-study overfit gate for SPD and field-anchored variants."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from rcsd_cxr.gold_mapping import FINDINGS
from rcsd_cxr.models.visual_state import VisualStateModel


VALUE_TO_STATE = {"0": 0, "1": 1, "-1": 2}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class StateDataset(Dataset):
    def __init__(self, manifest: Path, image_root: Path) -> None:
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"image_path", *FINDINGS}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"manifest missing fields: {sorted(missing)}")
            self.rows = list(reader)
        if not self.rows:
            raise ValueError("empty training manifest")
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        path = self.image_root / row["image_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            pixels = self.transform(image.convert("RGB"))
        states = torch.full((len(FINDINGS),), -100, dtype=torch.long)
        for finding_index, finding in enumerate(FINDINGS):
            value = str(row[finding] or "").strip()
            if value in VALUE_TO_STATE:
                states[finding_index] = VALUE_TO_STATE[value]
        if not (states != -100).any():
            raise ValueError(f"row {index} has no observed targets")
        return pixels, states


def masked_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
    )


@torch.inference_mode()
def evaluate(
    model: VisualStateModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    observed = 0
    correct = 0
    state_seen: set[int] = set()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)["logits"]
        mask = targets != -100
        count = int(mask.sum())
        loss_sum += float(masked_loss(logits, targets)) * count
        observed += count
        predictions = logits.argmax(dim=-1)
        correct += int((predictions[mask] == targets[mask]).sum())
        state_seen.update(int(value) for value in targets[mask].unique())
    return {
        "loss": loss_sum / observed,
        "accuracy": correct / observed,
        "observed_targets": observed,
        "states_seen": len(state_seen),
    }


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("spd", "field_anchor"), required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--backbone-weights", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    args = parser.parse_args()
    set_deterministic(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("overfit gate requires CUDA")
    device = torch.device("cuda:0")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = StateDataset(args.manifest, args.image_root)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    eval_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    model = VisualStateModel(
        args.variant, backbone_weights=args.backbone_weights
    ).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.backbone_lr},
            {
                "params": [
                    *model.projector.parameters(),
                    *model.state_head.parameters(),
                ],
                "lr": args.head_lr,
            },
        ],
        weight_decay=0.05,
    )
    initial = evaluate(model, eval_loader, device)
    records = [{"step": 0, **initial}]
    started = time.time()
    iterator = iter(loader)
    passed = False
    for step in range(1, args.max_steps + 1):
        try:
            images, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, targets = next(iterator)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images)["logits"]
            loss = masked_loss(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.max_steps:
            metrics = evaluate(model, eval_loader, device)
            metrics["step"] = step
            records.append(metrics)
            print(json.dumps(metrics), flush=True)
            loss_reduction = 1.0 - metrics["loss"] / initial["loss"]
            if metrics["accuracy"] >= 0.98 and loss_reduction >= 0.80:
                passed = True
                break

    final = records[-1]
    loss_reduction = 1.0 - final["loss"] / initial["loss"]
    summary = {
        "schema_version": 1,
        "artifact": "visual_state_overfit",
        "variant": args.variant,
        "pass": passed,
        "thresholds": {"accuracy": 0.98, "loss_reduction": 0.80},
        "initial": initial,
        "final": final,
        "loss_reduction": loss_reduction,
        "steps_run": int(final["step"]),
        "elapsed_seconds": time.time() - started,
        "trainable_counts": model.trainable_counts(),
        "max_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "hashes": {
            "manifest": sha256_file(args.manifest),
            "backbone_weights": sha256_file(args.backbone_weights),
        },
        "records": records,
    }
    torch.save(
        {
            "model": model.state_dict(),
            "summary": summary,
            "optimizer": optimizer.state_dict(),
        },
        args.output_dir / "final.pt",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
