"""Sequential-arm trainer for the locked D0-CP versus D1 token objective."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from rcsd_cxr.models.token_distillation import D0D1TokenModel
from rcsd_cxr.token_objective import (
    prepare_token_batch,
    token_accuracy,
    token_cross_entropy,
)


PROMPT = "Generate a structured medical report:\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def teacher_weight_authority(path: Path) -> dict[str, object]:
    """Hash the exact single-file weight or shard index plus referenced shards."""

    index = path / "model.safetensors.index.json"
    single = path / "model.safetensors"
    if index.is_file():
        payload = json.loads(index.read_text(encoding="utf-8"))
        names = sorted(set(payload["weight_map"].values()))
        files = [index, *(path / name for name in names)]
    elif single.is_file():
        files = [single]
    else:
        raise FileNotFoundError(f"no safetensors authority under {path}")
    missing = [file for file in files if not file.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return {
        "files": {
            file.name: {
                "bytes": file.stat().st_size,
                "sha256": sha256_file(file),
            }
            for file in files
        }
    }


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class D0D1Dataset(Dataset):
    def __init__(
        self,
        *,
        hard_manifest: Path,
        reliability_manifest: Path,
        image_root: Path,
        split: str,
        train: bool,
        row_ids_path: Path | None = None,
    ) -> None:
        hard = load_jsonl(hard_manifest)
        reliability_rows = load_jsonl(reliability_manifest)
        reliability = {row["row_id"]: row for row in reliability_rows}
        if len(reliability) != len(reliability_rows):
            raise ValueError("duplicate reliability row ids")
        selected_ids = None
        if row_ids_path is not None:
            payload = json.loads(row_ids_path.read_text(encoding="utf-8"))
            selected_ids = set(payload["row_ids"])
        rows = []
        for row in hard:
            if row["split"] != split:
                continue
            if selected_ids is not None and row["row_id"] not in selected_ids:
                continue
            if row["row_id"] not in reliability:
                raise ValueError(f"missing reliability row: {row['row_id']}")
            rows.append(
                {
                    **row,
                    "finding_weights": reliability[row["row_id"]][
                        "finding_weights"
                    ],
                }
            )
        if selected_ids is not None and len(rows) != len(selected_ids):
            raise ValueError("overfit row-id lock does not match hard manifest")
        if not rows:
            raise ValueError(f"no rows selected for split={split}")
        self.rows = rows
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
                    transforms.ColorJitter(brightness=0.05, contrast=0.05),
                    *common,
                ]
            )
            if train and row_ids_path is None
            else transforms.Compose([transforms.Resize((224, 224)), *common])
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        path = self.image_root / str(row["image_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            pixels = self.transform(image.convert("RGB"))
        return {
            "image": pixels,
            "target": row["target"],
            "finding_weights": row["finding_weights"],
            "row_id": row["row_id"],
        }


class TokenCollator:
    def __init__(self, tokenizer, variant: str) -> None:
        self.tokenizer = tokenizer
        self.variant = variant

    def __call__(self, rows: list[dict[str, object]]) -> dict[str, object]:
        tokens = prepare_token_batch(
            self.tokenizer,
            prompt=PROMPT,
            targets=[str(row["target"]) for row in rows],
            finding_weights=[
                dict(row["finding_weights"]) for row in rows
            ],
            variant=self.variant,
        )
        return {
            "images": torch.stack([row["image"] for row in rows]),
            **tokens,
            "row_ids": [row["row_id"] for row in rows],
        }


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)


def move_batch(
    batch: dict[str, object], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }


@torch.inference_mode()
def evaluate(
    model: D0D1TokenModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    observed = 0
    batches = 0
    for raw in loader:
        batch = move_batch(raw, device)
        output = model(
            batch["images"],
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )
        loss = token_cross_entropy(output["logits"], output["labels"])
        batch_correct, batch_observed = token_accuracy(
            output["logits"], output["labels"]
        )
        loss_sum += float(loss) * batch_observed
        correct += batch_correct
        observed += batch_observed
        batches += 1
    return {
        "token_nll": loss_sum / observed,
        "token_accuracy": correct / observed,
        "observed_tokens": observed,
        "batches": batches,
    }


def gradient_audit(model: D0D1TokenModel) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, parameters in model.trainable_parameter_groups().items():
        gradients = [
            parameter.grad
            for parameter in parameters
            if parameter.grad is not None
        ]
        finite = all(bool(torch.isfinite(value).all()) for value in gradients)
        nonzero = sum(int(torch.count_nonzero(value)) for value in gradients)
        result[name] = {
            "gradient_tensors": len(gradients),
            "finite": finite,
            "nonzero_values": nonzero,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("d0", "d1"), required=True)
    parser.add_argument("--mode", choices=("overfit", "pilot"), required=True)
    parser.add_argument("--hard-manifest", required=True, type=Path)
    parser.add_argument("--reliability-manifest", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--teacher-path", required=True, type=Path)
    parser.add_argument("--backbone-weights", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overfit-ids", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "overfit" and args.overfit_ids is None:
        raise ValueError("overfit mode requires --overfit-ids")
    max_steps = args.max_steps or (500 if args.mode == "overfit" else 3000)
    eval_interval = args.eval_interval or (
        50 if args.mode == "overfit" else 500
    )
    set_deterministic(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("D0/D1 token training requires CUDA")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = D0D1TokenModel(
        teacher_path=args.teacher_path,
        backbone_weights=args.backbone_weights,
    ).to(device)
    train_dataset = D0D1Dataset(
        hard_manifest=args.hard_manifest,
        reliability_manifest=args.reliability_manifest,
        image_root=args.image_root,
        split="train",
        train=True,
        row_ids_path=args.overfit_ids if args.mode == "overfit" else None,
    )
    validation_dataset = (
        train_dataset
        if args.mode == "overfit"
        else D0D1Dataset(
            hard_manifest=args.hard_manifest,
            reliability_manifest=args.reliability_manifest,
            image_root=args.image_root,
            split="validate",
            train=False,
        )
    )
    collator = TokenCollator(model.tokenizer, args.variant)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    groups = model.trainable_parameter_groups()
    optimizer = torch.optim.AdamW(
        [
            {"params": groups["backbone"], "lr": 2e-5},
            {"params": groups["projector"], "lr": 1e-4},
        ],
        weight_decay=0.01,
    )

    def lr_factor(step: int) -> float:
        if step < 500:
            return max(step, 1) / 500
        progress = (step - 500) / max(max_steps - 500, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    initial = evaluate(model, validation_loader, device)
    records: list[dict[str, object]] = [{"step": 0, **initial}]
    best_nll = float(initial["token_nll"])
    best_step = 0
    iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    first_gradient_audit = None
    started = time.time()
    passed_overfit = False
    for step in range(1, max_steps + 1):
        accumulated = 0.0
        for _ in range(args.gradient_accumulation):
            try:
                raw = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                raw = next(iterator)
            batch = move_batch(raw, device)
            model.train()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(
                    batch["images"],
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["labels"],
                )
                visual_tokens = int(output["visual_tokens"])
                visual_weights = torch.ones(
                    output["labels"].shape[0],
                    visual_tokens,
                    device=device,
                )
                full_weights = torch.cat(
                    [visual_weights, batch["token_weights"]], dim=1
                )
                token_loss = token_cross_entropy(
                    output["logits"],
                    output["labels"],
                    token_weights=(
                        full_weights if args.variant == "d1" else None
                    ),
                )
                loss = (
                    token_loss + 0.02 * output["orthogonality"]
                ) / args.gradient_accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward()
            accumulated += float(token_loss)
        if first_gradient_audit is None:
            first_gradient_audit = gradient_audit(model)
        torch.nn.utils.clip_grad_norm_(
            [*groups["backbone"], *groups["projector"]], 1.0
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        if step % 10 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "train_token_loss": (
                            accumulated / args.gradient_accumulation
                        ),
                        "gpu_memory_bytes": torch.cuda.max_memory_allocated(
                            device
                        ),
                    }
                ),
                flush=True,
            )
        if step % eval_interval == 0 or step == max_steps:
            metrics = evaluate(model, validation_loader, device)
            record = {"step": step, **metrics}
            records.append(record)
            print(json.dumps(record), flush=True)
            if float(metrics["token_nll"]) < best_nll:
                best_nll = float(metrics["token_nll"])
                best_step = step
                torch.save(
                    {
                        "vision": model.vision_state_dict(),
                        "step": step,
                        "validation": metrics,
                        "variant": args.variant,
                    },
                    args.output_dir / "best.pt",
                )
            if args.mode == "overfit":
                reduction = 1.0 - (
                    float(metrics["token_nll"]) / initial["token_nll"]
                )
                if metrics["token_accuracy"] >= 0.98 and reduction >= 0.80:
                    passed_overfit = True
                    break
    best_record = next(
        record for record in records if record["step"] == best_step
    )
    final = records[-1]
    summary = {
        "schema_version": 1,
        "artifact": "d0_d1_token_training",
        "variant": args.variant,
        "mode": args.mode,
        "pass": passed_overfit if args.mode == "overfit" else True,
        "initial": initial,
        "final": final,
        "best_step": best_step,
        "best_validation": best_record,
        "overfit_loss_reduction": (
            1.0 - float(final["token_nll"]) / initial["token_nll"]
            if args.mode == "overfit"
            else None
        ),
        "gradient_audit": first_gradient_audit,
        "rows": {
            "train": len(train_dataset),
            "validation": len(validation_dataset),
        },
        "budget": {
            "max_steps": max_steps,
            "steps_run": int(final["step"]),
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_batch_size": (
                args.batch_size * args.gradient_accumulation
            ),
            "seed": args.seed,
        },
        "checkpoint_rule": "strictly lower unweighted validation token NLL",
        "prompt": PROMPT,
        "elapsed_seconds": time.time() - started,
        "max_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "hashes": {
            "hard_manifest": sha256_file(args.hard_manifest),
            "reliability_manifest": sha256_file(
                args.reliability_manifest
            ),
            "overfit_ids": (
                sha256_file(args.overfit_ids)
                if args.overfit_ids is not None
                else None
            ),
            "backbone_weights": sha256_file(args.backbone_weights),
            "teacher_weights": teacher_weight_authority(args.teacher_path),
        },
        "records": records,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)
    return 0 if summary["pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
