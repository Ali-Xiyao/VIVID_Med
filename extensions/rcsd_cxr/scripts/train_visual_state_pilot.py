"""One-seed 20k pilot for unanchored versus field-anchored distillation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from rcsd_cxr.gold_mapping import FINDINGS
from rcsd_cxr.models.visual_state import VisualStateModel
from train_visual_state_overfit import VALUE_TO_STATE, masked_loss


PERMUTATIONS = torch.tensor(
    list(itertools.permutations(range(4))), dtype=torch.long
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PilotDataset(Dataset):
    def __init__(
        self, manifest: Path, image_root: Path, split: str, train: bool
    ) -> None:
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            split_rows = [
                row for row in csv.DictReader(handle) if row["split"] == split
            ]
        rows = [
            row
            for row in split_rows
            if any(str(row[finding] or "").strip() in VALUE_TO_STATE for finding in FINDINGS)
        ]
        if not rows:
            raise ValueError(f"no rows for split {split}")
        self.rows = rows
        self.original_rows = len(split_rows)
        self.excluded_all_missing = len(split_rows) - len(rows)
        self.image_root = image_root
        common = [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
        self.transform = (
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        224, scale=(0.90, 1.0), ratio=(0.95, 1.05)
                    ),
                    transforms.RandomRotation(5),
                    transforms.ColorJitter(
                        brightness=0.05, contrast=0.05
                    ),
                    *common,
                ]
            )
            if train
            else transforms.Compose([transforms.Resize((224, 224)), *common])
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
            raise RuntimeError("all-missing row escaped initialization filter")
        return pixels, states


def load_prototypes(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if tuple(payload["findings"]) != FINDINGS:
        raise ValueError("prototype finding order mismatch")
    if tuple(payload["states"]) != ("absent", "present", "uncertain"):
        raise ValueError("prototype state order mismatch")
    return {
        field: payload[field].float()
        for field in ("observation", "assertion", "anatomy", "global")
    }


def aggregate_targets(
    states: torch.Tensor, prototypes: dict[str, torch.Tensor]
) -> torch.Tensor:
    mask = states != -100
    safe_states = states.clamp_min(0)
    batch, findings = states.shape
    finding_index = (
        torch.arange(findings, device=states.device)
        .unsqueeze(0)
        .expand(batch, -1)
    )
    values = []
    for field in ("observation", "assertion", "anatomy", "global"):
        prototype = prototypes[field].to(states.device)
        if prototype.ndim == 2:
            atom = prototype.unsqueeze(0).expand(batch, -1, -1)
        else:
            atom = prototype[finding_index, safe_states]
        weighted = atom * mask.unsqueeze(-1)
        target = weighted.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        values.append(torch.nn.functional.normalize(target, dim=-1))
    return torch.stack(values, dim=1)


def semantic_loss(
    predicted: torch.Tensor, targets: torch.Tensor, variant: str
) -> torch.Tensor:
    predicted = torch.nn.functional.normalize(predicted.float(), dim=-1)
    targets = torch.nn.functional.normalize(targets.float(), dim=-1)
    similarity = torch.einsum("bgd,bhd->bgh", predicted, targets)
    if variant == "field_anchor":
        diagonal = similarity.diagonal(dim1=1, dim2=2)
        return 1.0 - diagonal.mean()
    permutations = PERMUTATIONS.to(similarity.device)
    group = torch.arange(4, device=similarity.device)
    scores = torch.stack(
        [
            similarity[:, group, permutation].mean(dim=1)
            for permutation in permutations
        ],
        dim=1,
    )
    return 1.0 - scores.max(dim=1).values.mean()


def ece_score(
    truth: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == truth
    value = 0.0
    for low in np.linspace(0.0, 1.0, bins, endpoint=False):
        high = low + 1.0 / bins
        mask = (confidence >= low) & (
            confidence < high if high < 1.0 else confidence <= high
        )
        if mask.any():
            value += mask.mean() * abs(
                float(correct[mask].mean())
                - float(confidence[mask].mean())
            )
    return float(value)


@torch.inference_mode()
def evaluate(
    model: VisualStateModel,
    loader: DataLoader,
    prototypes: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    truths = []
    probabilities = []
    finding_ids = []
    state_loss_sum = 0.0
    semantic_sum = 0.0
    observed = 0
    samples = 0
    for images, states in loader:
        images = images.to(device, non_blocking=True)
        states = states.to(device, non_blocking=True)
        output = model(images)
        batch_targets = aggregate_targets(states, prototypes)
        batch_semantic = semantic_loss(
            output["fields"], batch_targets, model.variant
        )
        mask = states != -100
        count = int(mask.sum())
        state_loss_sum += float(masked_loss(output["logits"], states)) * count
        semantic_sum += float(batch_semantic) * images.shape[0]
        observed += count
        samples += images.shape[0]
        probs = output["logits"].softmax(dim=-1)
        truths.extend(states[mask].cpu().tolist())
        probabilities.extend(probs[mask].cpu().tolist())
        finding_grid = (
            torch.arange(len(FINDINGS), device=device)
            .unsqueeze(0)
            .expand(states.shape[0], -1)
        )
        finding_ids.extend(finding_grid[mask].cpu().tolist())
    truth = np.asarray(truths, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    prediction = probs.argmax(axis=1)
    per_finding = {}
    for index, finding in enumerate(FINDINGS):
        mask = np.asarray(finding_ids) == index
        per_finding[finding] = {
            "n": int(mask.sum()),
            "macro_f1": float(
                f1_score(
                    truth[mask],
                    prediction[mask],
                    labels=[0, 1, 2],
                    average="macro",
                    zero_division=0,
                )
            ),
        }
    return {
        "nll": state_loss_sum / observed,
        "semantic_loss": semantic_sum / samples,
        "accuracy": float((prediction == truth).mean()),
        "macro_f1": float(
            f1_score(
                truth,
                prediction,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
        "ece": ece_score(truth, probs),
        "observed_targets": observed,
        "per_finding": per_finding,
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
    parser.add_argument("--prototypes", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accumulation", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--field-weight", type=float, default=1.0)
    args = parser.parse_args()
    set_deterministic(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("pilot requires CUDA")
    device = torch.device("cuda:0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prototypes = load_prototypes(args.prototypes)
    train_dataset = PilotDataset(
        args.manifest, args.image_root, "train", train=True
    )
    validation_dataset = PilotDataset(
        args.manifest, args.image_root, "validate", train=False
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
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

    def lr_factor(step: int) -> float:
        warmup = 50
        if step < warmup:
            return max(step, 1) / warmup
        progress = (step - warmup) / max(args.max_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    records = []
    best_nll = float("inf")
    best_step = None
    started = time.time()
    iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.max_steps + 1):
        accumulated_state = 0.0
        accumulated_semantic = 0.0
        for _ in range(args.grad_accumulation):
            try:
                images, states = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                images, states = next(iterator)
            images = images.to(device, non_blocking=True)
            states = states.to(device, non_blocking=True)
            model.train()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(images)
                state_value = masked_loss(output["logits"], states)
                targets = aggregate_targets(states, prototypes)
                field_value = semantic_loss(
                    output["fields"], targets, args.variant
                )
                loss = (
                    state_value + args.field_weight * field_value
                ) / args.grad_accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward()
            accumulated_state += float(state_value)
            accumulated_semantic += float(field_value)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        if step % 20 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "train_state_loss": accumulated_state
                        / args.grad_accumulation,
                        "train_semantic_loss": accumulated_semantic
                        / args.grad_accumulation,
                        "lr": [group["lr"] for group in optimizer.param_groups],
                    }
                ),
                flush=True,
            )
        if step % args.eval_every == 0 or step == args.max_steps:
            metrics = evaluate(
                model, validation_loader, prototypes, device
            )
            record = {"step": step, **metrics}
            records.append(record)
            print(json.dumps(record), flush=True)
            if float(metrics["nll"]) < best_nll:
                best_nll = float(metrics["nll"])
                best_step = step
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step": step,
                        "validation": metrics,
                    },
                    args.output_dir / "best.pt",
                )
    summary = {
        "schema_version": 1,
        "artifact": "visual_state_20k_pilot",
        "variant": args.variant,
        "pass": True,
        "checkpoint_selection": "strictly lower validation structured NLL",
        "best_step": best_step,
        "best_nll": best_nll,
        "best_validation": next(
            record for record in records if record["step"] == best_step
        ),
        "records": records,
        "elapsed_seconds": time.time() - started,
        "train_rows": len(train_dataset),
        "validation_rows": len(validation_dataset),
        "excluded_all_missing": {
            "train": train_dataset.excluded_all_missing,
            "validate": validation_dataset.excluded_all_missing,
        },
        "effective_batch_size": args.batch_size * args.grad_accumulation,
        "trainable_counts": model.trainable_counts(),
        "max_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "hashes": {
            "manifest": sha256_file(args.manifest),
            "backbone_weights": sha256_file(args.backbone_weights),
            "prototypes": sha256_file(args.prototypes),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
