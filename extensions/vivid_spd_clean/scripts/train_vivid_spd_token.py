"""Train one strict hard-UMS prefix/SPD arm."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# PyTorch's fail-closed deterministic mode requires this CUDA >=10.2 cuBLAS
# workspace contract to be present before torch initializes CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vivid_spd_clean.io import (  # noqa: E402
    load_jsonl,
    sha256_file,
    teacher_weight_authority,
)
from vivid_spd_clean.model import VividSPDTokenModel  # noqa: E402
from vivid_spd_clean.objective import (  # noqa: E402
    prepare_token_batch,
    token_accuracy,
    token_cross_entropy,
)


PROMPT = "Generate a structured medical report:\n"
ARMS = (
    "ums_prefix4",
    "ums_spd4x2",
    "ums_prefix8",
    "ums_spd4x2_no_ortho",
)


class HardUMSDataset(Dataset):
    def __init__(
        self,
        *,
        manifest: Path,
        image_root: Path,
        split: str,
        train: bool,
        row_ids_path: Path | None = None,
    ) -> None:
        selected_ids = None
        if row_ids_path is not None:
            selected_ids = set(
                json.loads(row_ids_path.read_text(encoding="utf-8"))["row_ids"]
            )
        self.rows = [
            row
            for row in load_jsonl(manifest)
            if row["split"] == split
            and (selected_ids is None or row["row_id"] in selected_ids)
        ]
        if selected_ids is not None and len(self.rows) != len(selected_ids):
            raise ValueError("overfit row-id lock does not match hard manifest")
        if not self.rows:
            raise ValueError(f"no rows selected for split={split}")
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
            "target": str(row["target"]),
            "row_id": str(row["row_id"]),
        }


class TokenCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, rows: list[dict[str, object]]) -> dict[str, object]:
        tokens = prepare_token_batch(
            self.tokenizer,
            prompt=PROMPT,
            targets=[str(row["target"]) for row in rows],
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
    model: VividSPDTokenModel,
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


def gradient_audit(model: VividSPDTokenModel) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, parameters in model.trainable_parameter_groups().items():
        gradients = [
            parameter.grad for parameter in parameters if parameter.grad is not None
        ]
        result[name] = {
            "gradient_tensors": len(gradients),
            "finite": all(bool(torch.isfinite(value).all()) for value in gradients),
            "nonzero_values": sum(
                int(torch.count_nonzero(value)) for value in gradients
            ),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--mode", choices=("overfit", "pilot"), required=True)
    parser.add_argument("--hard-manifest", required=True, type=Path)
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
        raise RuntimeError("strict VIVID/SPD token training requires CUDA")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = VividSPDTokenModel(
        arm=args.arm,
        teacher_path=args.teacher_path,
        backbone_weights=args.backbone_weights,
    ).to(device)
    train_dataset = HardUMSDataset(
        manifest=args.hard_manifest,
        image_root=args.image_root,
        split="train",
        train=True,
        row_ids_path=args.overfit_ids if args.mode == "overfit" else None,
    )
    validation_dataset = (
        train_dataset
        if args.mode == "overfit"
        else HardUMSDataset(
            manifest=args.hard_manifest,
            image_root=args.image_root,
            split="validate",
            train=False,
        )
    )
    collator = TokenCollator(model.tokenizer)
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
                token_loss = token_cross_entropy(
                    output["logits"], output["labels"]
                )
                loss = (
                    token_loss
                    + model.orthogonality_weight * output["orthogonality"]
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
                        "arm": args.arm,
                    },
                    args.output_dir / "best.pt",
                )
            if args.mode == "overfit":
                reduction = 1.0 - (
                    float(metrics["token_nll"]) / initial["token_nll"]
                )
                if metrics["token_accuracy"] >= 0.98 and reduction >= 0.80:
                    break
    best_record = next(
        record for record in records if record["step"] == best_step
    )
    final = records[-1]
    reduction = 1.0 - float(final["token_nll"]) / initial["token_nll"]
    gradients_pass = bool(
        first_gradient_audit
        and all(
            row["finite"]
            and row["gradient_tensors"] > 0
            and row["nonzero_values"] > 0
            for row in first_gradient_audit.values()
        )
    )
    passed = gradients_pass and (
        (
            float(final["token_accuracy"]) >= 0.98
            and reduction >= 0.80
        )
        if args.mode == "overfit"
        else reduction >= 0.20
    )
    summary = {
        "schema_version": 1,
        "artifact": "strict_vivid_spd_token_training",
        "arm": args.arm,
        "mode": args.mode,
        "pass": passed,
        "initial": initial,
        "final": final,
        "best_step": best_step,
        "best_validation": best_record,
        "loss_reduction": reduction,
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
        "identity": {
            "prompt": PROMPT,
            "orthogonality_weight": model.orthogonality_weight,
            "query_tokens": model.projector.num_query_tokens,
            "checkpoint_rule": (
                "strictly lower unweighted validation token NLL"
            ),
        },
        "elapsed_seconds": time.time() - started,
        "max_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "hashes": {
            "hard_manifest": sha256_file(args.hard_manifest),
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
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
