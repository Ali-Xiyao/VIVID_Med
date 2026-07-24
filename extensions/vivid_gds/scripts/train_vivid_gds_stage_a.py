"""Train one frozen VIVID-GDS Stage-A arm."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vivid_gds.contracts import parse_ums_target, render_free_text  # noqa: E402
from vivid_gds.io import (  # noqa: E402
    load_jsonl,
    sha256_file,
    teacher_weight_authority,
)
from vivid_gds.model import (  # noqa: E402
    ARMS,
    GENERATIVE_ARMS,
    SCHEMA_ARMS,
    VividGDSModel,
)
from vivid_gds.objective import (  # noqa: E402
    masked_schema_cross_entropy,
    prepare_token_batch,
    schema_accuracy,
    token_accuracy,
    token_cross_entropy,
)


UMS_PROMPT = "Generate a structured medical report:\n"
FREE_TEXT_PROMPT = "Describe the findings in this chest X-ray:\n"


class StageADataset(Dataset):
    def __init__(
        self,
        *,
        manifest: Path,
        image_root: Path,
        split: str,
        arm: str,
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
        self.arm = arm
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
        target = str(row["target"])
        states, _ = parse_ums_target(target)
        generation_target = (
            render_free_text(target, str(row["row_id"]))
            if self.arm == "A1_freetext"
            else target
        )
        return {
            "image": pixels,
            "generation_target": generation_target,
            "schema_states": torch.tensor(states, dtype=torch.long),
            "row_id": str(row["row_id"]),
        }


class StageACollator:
    def __init__(self, arm: str, tokenizer=None) -> None:
        self.arm = arm
        self.tokenizer = tokenizer

    def __call__(self, rows: list[dict[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {
            "images": torch.stack([row["image"] for row in rows]),
            "schema_states": torch.stack(
                [row["schema_states"] for row in rows]
            ),
            "row_ids": [row["row_id"] for row in rows],
        }
        if self.arm in GENERATIVE_ARMS:
            if self.tokenizer is None:
                raise ValueError("generative collator requires tokenizer")
            prompt = (
                FREE_TEXT_PROMPT if self.arm == "A1_freetext" else UMS_PROMPT
            )
            result.update(
                prepare_token_batch(
                    self.tokenizer,
                    prompt=prompt,
                    targets=[
                        str(row["generation_target"]) for row in rows
                    ],
                )
            )
        return result


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
    model: VividGDSModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    token_loss_sum = 0.0
    token_correct = 0
    token_observed = 0
    schema_sums = torch.zeros(12, dtype=torch.float64)
    schema_counts = torch.zeros(12, dtype=torch.long)
    schema_correct = 0
    schema_observed = 0
    batches = 0
    for raw in loader:
        batch = move_batch(raw, device)
        output = model(
            batch["images"],
            batch.get("input_ids"),
            batch.get("attention_mask"),
            batch.get("labels"),
        )
        if model.arm in GENERATIVE_ARMS:
            loss = token_cross_entropy(output["logits"], output["labels"])
            correct, observed = token_accuracy(
                output["logits"], output["labels"]
            )
            token_loss_sum += float(loss) * observed
            token_correct += correct
            token_observed += observed
        if model.arm in SCHEMA_ARMS:
            states = batch["schema_states"]
            logits = output["schema_logits"]
            correct, observed = schema_accuracy(logits, states)
            schema_correct += correct
            schema_observed += observed
            for finding in range(states.shape[1]):
                valid = states[:, finding] != -100
                if bool(valid.any()):
                    schema_sums[finding] += float(
                        F.cross_entropy(
                            logits[valid, finding],
                            states[valid, finding],
                            reduction="sum",
                        )
                    )
                    schema_counts[finding] += int(valid.sum())
        batches += 1
    result: dict[str, float] = {"batches": float(batches)}
    if model.arm in GENERATIVE_ARMS:
        result.update(
            token_nll=token_loss_sum / token_observed,
            token_accuracy=token_correct / token_observed,
            observed_tokens=float(token_observed),
        )
    if model.arm in SCHEMA_ARMS:
        active = schema_counts > 0
        result.update(
            schema_nll=float(
                (schema_sums[active] / schema_counts[active]).mean()
            ),
            schema_accuracy=schema_correct / schema_observed,
            observed_fields=float(schema_observed),
        )
    return result


def gradient_audit(model: VividGDSModel) -> dict[str, object]:
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


def reduction(initial: dict[str, float], final: dict[str, float], key: str) -> float:
    return 1.0 - float(final[key]) / float(initial[key])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--mode", choices=("overfit", "pilot"), required=True)
    parser.add_argument("--hard-manifest", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--teacher-path", type=Path)
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
    if args.arm in GENERATIVE_ARMS and args.teacher_path is None:
        raise ValueError("generative arm requires --teacher-path")
    max_steps = args.max_steps or (500 if args.mode == "overfit" else 3000)
    eval_interval = args.eval_interval or (
        50 if args.mode == "overfit" else 500
    )
    set_deterministic(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("VIVID-GDS Stage A requires CUDA")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = VividGDSModel(
        arm=args.arm,
        teacher_path=args.teacher_path,
        backbone_weights=args.backbone_weights,
    ).to(device)
    train_dataset = StageADataset(
        manifest=args.hard_manifest,
        image_root=args.image_root,
        split="train",
        arm=args.arm,
        train=True,
        row_ids_path=args.overfit_ids if args.mode == "overfit" else None,
    )
    validation_dataset = (
        train_dataset
        if args.mode == "overfit"
        else StageADataset(
            manifest=args.hard_manifest,
            image_root=args.image_root,
            split="validate",
            arm=args.arm,
            train=False,
        )
    )
    collator = StageACollator(args.arm, model.tokenizer)
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
    optimizer_rows = [{"params": groups["backbone"], "lr": 2e-5}]
    for name in ("projector", "schema_head"):
        if name in groups:
            optimizer_rows.append({"params": groups[name], "lr": 1e-4})
    optimizer = torch.optim.AdamW(
        optimizer_rows,
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
    primary_key = "schema_nll" if args.arm == "A0_direct" else "token_nll"
    best_nll = float(initial[primary_key])
    best_step = 0
    iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    first_gradient_audit = None
    started = time.time()
    for step in range(1, max_steps + 1):
        accumulated = {"token": 0.0, "schema": 0.0}
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
                    batch.get("input_ids"),
                    batch.get("attention_mask"),
                    batch.get("labels"),
                )
                loss = torch.zeros((), device=device)
                if args.arm in GENERATIVE_ARMS:
                    token_loss = token_cross_entropy(
                        output["logits"], output["labels"]
                    )
                    loss = loss + token_loss
                    accumulated["token"] += float(token_loss)
                if args.arm in SCHEMA_ARMS:
                    schema_loss = masked_schema_cross_entropy(
                        output["schema_logits"], batch["schema_states"]
                    )
                    weight = (
                        0.5 * min(1.0, step / 500)
                        if args.arm == "A3_gds"
                        else 1.0
                    )
                    loss = loss + weight * schema_loss
                    accumulated["schema"] += float(schema_loss)
                loss = loss / args.gradient_accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward()
        if first_gradient_audit is None:
            first_gradient_audit = gradient_audit(model)
        torch.nn.utils.clip_grad_norm_(
            [parameter for values in groups.values() for parameter in values],
            1.0,
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
                            accumulated["token"] / args.gradient_accumulation
                            if args.arm in GENERATIVE_ARMS else None
                        ),
                        "train_schema_loss": (
                            accumulated["schema"] / args.gradient_accumulation
                            if args.arm in SCHEMA_ARMS else None
                        ),
                        "lambda_schema": (
                            0.5 * min(1.0, step / 500)
                            if args.arm == "A3_gds" else None
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
            if float(metrics[primary_key]) < best_nll:
                best_nll = float(metrics[primary_key])
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
                generation_pass = (
                    args.arm not in GENERATIVE_ARMS
                    or (
                        metrics["token_accuracy"] >= 0.98
                        and reduction(initial, metrics, "token_nll") >= 0.80
                    )
                )
                schema_pass = (
                    args.arm not in SCHEMA_ARMS
                    or (
                        metrics["schema_accuracy"] >= 0.98
                        and reduction(initial, metrics, "schema_nll") >= 0.80
                    )
                )
                if generation_pass and schema_pass:
                    break
    final = records[-1]
    reductions = {
        key: reduction(initial, final, key)
        for key in ("token_nll", "schema_nll")
        if key in initial
    }
    gradients_pass = bool(
        first_gradient_audit
        and all(
            row["finite"]
            and row["gradient_tensors"] > 0
            and row["nonzero_values"] > 0
            for row in first_gradient_audit.values()
        )
    )
    if args.mode == "overfit":
        generation_pass = (
            args.arm not in GENERATIVE_ARMS
            or (
                float(final["token_accuracy"]) >= 0.98
                and reductions["token_nll"] >= 0.80
            )
        )
        schema_pass = (
            args.arm not in SCHEMA_ARMS
            or (
                float(final["schema_accuracy"]) >= 0.98
                and reductions["schema_nll"] >= 0.80
            )
        )
        passed = gradients_pass and generation_pass and schema_pass
    else:
        passed = gradients_pass and reductions[primary_key] >= 0.20
        if args.arm == "A3_gds":
            passed = passed and reductions["schema_nll"] >= 0.20
    summary = {
        "schema_version": 1,
        "artifact": "vivid_gds_stage_a_training",
        "arm": args.arm,
        "mode": args.mode,
        "pass": passed,
        "initial": initial,
        "final": final,
        "best_step": best_step,
        "primary_checkpoint_metric": primary_key,
        "loss_reduction": reductions,
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
            "generation_prompt": (
                FREE_TEXT_PROMPT
                if args.arm == "A1_freetext"
                else UMS_PROMPT if args.arm in GENERATIVE_ARMS else None
            ),
            "lambda_schema": 0.5 if args.arm == "A3_gds" else None,
            "lambda_ramp_steps": 500 if args.arm == "A3_gds" else None,
            "checkpoint_rule": f"strictly lower validation {primary_key}",
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
            "teacher_weights": (
                teacher_weight_authority(args.teacher_path)
                if args.teacher_path is not None else None
            ),
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
