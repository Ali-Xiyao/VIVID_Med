"""Train Qwen3-VL vision tower and visual connector on clinical instructions."""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.clinical_instruction_dataset import ClinicalInstructionDataset, Qwen3VLInstructionCollator


DEFAULT_MODEL_PATH = r"H:\Xiyao_Wang\001_models\qwen3-vl-2b-thinking-new"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def choose_dtype(name: str) -> torch.dtype:
    lowered = str(name).lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16"}:
        return torch.float16
    if lowered in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def module_name_matches(name: str, keywords: list[str]) -> bool:
    lowered = name.lower()
    return any(keyword in lowered for keyword in keywords)


def classify_parameter(name: str) -> str:
    lowered = name.lower()
    connector_keywords = ["merger", "merge", "projector", "connector", "mm_projector", "adapter"]
    vision_keywords = ["visual", "vision", "image", "vit"]
    language_keywords = ["language", "llm", "lm_head", "embed_tokens", "model.layers", "model.norm"]
    if module_name_matches(lowered, connector_keywords):
        return "visual_connector"
    if module_name_matches(lowered, vision_keywords):
        return "vision_tower"
    if module_name_matches(lowered, language_keywords):
        return "language_decoder"
    return "other"


def apply_freeze_plan(model: torch.nn.Module, train_groups: set[str], trainable_vision_last_n: int | None = None) -> None:
    vision_block_indices = set()
    if trainable_vision_last_n:
        for name, _ in model.named_parameters():
            match = re.search(r"\bvisual\.blocks\.(\d+)\.", name)
            if match:
                vision_block_indices.add(int(match.group(1)))
        keep_from = max(vision_block_indices) - int(trainable_vision_last_n) + 1 if vision_block_indices else None
    else:
        keep_from = None
    for name, param in model.named_parameters():
        group = classify_parameter(name)
        param.requires_grad = group in train_groups
        if keep_from is not None and group == "vision_tower":
            match = re.search(r"\bvisual\.blocks\.(\d+)\.", name)
            param.requires_grad = bool(match and int(match.group(1)) >= keep_from)


def count_parameters(model: torch.nn.Module) -> dict[str, dict[str, int]]:
    groups: dict[str, dict[str, int]] = {
        "vision_tower": {"parameters": 0, "trainable": 0},
        "visual_connector": {"parameters": 0, "trainable": 0},
        "language_decoder": {"parameters": 0, "trainable": 0},
        "other": {"parameters": 0, "trainable": 0},
    }
    for name, param in model.named_parameters():
        group = classify_parameter(name)
        count = int(param.numel())
        groups[group]["parameters"] += count
        if param.requires_grad:
            groups[group]["trainable"] += count
    groups["total"] = {
        "parameters": sum(item["parameters"] for item in groups.values()),
        "trainable": sum(item["trainable"] for item in groups.values()),
    }
    return groups


def move_tensors_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def model_inputs(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    skip = {
        "labels",
        "loss_weights",
        "instruction_ids",
        "sample_ids",
        "image_paths",
        "answer_types",
        "findings",
        "states",
        "visual_dependencies",
        "hard_negative_available",
        "metadata",
    }
    return {
        key: value
        for key, value in batch.items()
        if key not in skip
        and not key.startswith("negative_")
        and not key.startswith("answer_negative_")
        and torch.is_tensor(value)
    }


def negative_model_inputs(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    skip = {"negative_labels", "negative_loss_weights"}
    inputs: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if not key.startswith("negative_") or key in skip or not torch.is_tensor(value):
            continue
        inputs[key.removeprefix("negative_")] = value
    return inputs


def answer_negative_model_inputs(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    skip = {"answer_negative_labels", "answer_negative_loss_weights"}
    inputs: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if not key.startswith("answer_negative_") or key in skip or not torch.is_tensor(value):
            continue
        inputs[key.removeprefix("answer_negative_")] = value
    return inputs


def compute_weighted_lm_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_weights = weights[:, 1:].contiguous().to(dtype=torch.float32)
    vocab_size = shift_logits.shape[-1]
    token_loss = F.cross_entropy(
        shift_logits.float().view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)
    mask = shift_labels != -100
    weighted = token_loss * shift_weights * mask.float()
    denom = (shift_weights * mask.float()).sum().clamp_min(1.0)
    return weighted.sum() / denom


def compute_training_loss(model: torch.nn.Module, batch: dict[str, Any], config: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    outputs = model(**model_inputs(batch), use_cache=False)
    ce_loss = compute_weighted_lm_loss(outputs.logits, batch["labels"], batch["loss_weights"])
    metrics = {"ce_loss": float(ce_loss.detach().cpu())}
    total = ce_loss
    margin_cfg = (config.get("training", {}) or {}).get("image_shuffle_margin", {}) or {}
    if bool(margin_cfg.get("enabled", False)) and "negative_input_ids" in batch:
        negative_inputs = negative_model_inputs(batch)
        if negative_inputs:
            negative_outputs = model(**negative_inputs, use_cache=False)
            negative_loss = compute_weighted_lm_loss(
                negative_outputs.logits,
                batch["negative_labels"],
                batch["negative_loss_weights"],
            )
            margin = float(margin_cfg.get("margin", 0.2))
            weight = float(margin_cfg.get("weight", 0.1))
            margin_loss = torch.relu(ce_loss - negative_loss + ce_loss.new_tensor(margin))
            total = total + weight * margin_loss
            metrics.update(
                {
                    "negative_loss": float(negative_loss.detach().cpu()),
                    "shuffle_margin_loss": float(margin_loss.detach().cpu()),
                }
            )
    answer_margin_cfg = (config.get("training", {}) or {}).get("answer_margin", {}) or {}
    if bool(answer_margin_cfg.get("enabled", False)) and "answer_negative_input_ids" in batch:
        answer_negative_inputs = answer_negative_model_inputs(batch)
        if answer_negative_inputs:
            answer_negative_outputs = model(**answer_negative_inputs, use_cache=False)
            answer_negative_loss = compute_weighted_lm_loss(
                answer_negative_outputs.logits,
                batch["answer_negative_labels"],
                batch["answer_negative_loss_weights"],
            )
            margin = float(answer_margin_cfg.get("margin", 0.2))
            weight = float(answer_margin_cfg.get("weight", 0.1))
            answer_margin_loss = torch.relu(ce_loss - answer_negative_loss + ce_loss.new_tensor(margin))
            total = total + weight * answer_margin_loss
            metrics.update(
                {
                    "answer_negative_loss": float(answer_negative_loss.detach().cpu()),
                    "answer_margin_loss": float(answer_margin_loss.detach().cpu()),
                }
            )
    metrics["total_loss"] = float(total.detach().cpu())
    return total, metrics


def create_dataloaders(
    config: dict[str, Any],
    processor: Any,
) -> tuple[DataLoader, DataLoader | None]:
    data_cfg = config["data"]
    train_dataset = ClinicalInstructionDataset(
        data_root=data_cfg.get("data_root", "."),
        instruction_jsonl_path=data_cfg["train_instruction_path"],
        max_samples=data_cfg.get("max_train_samples"),
    )
    collator = Qwen3VLInstructionCollator(
        processor=processor,
        max_length=data_cfg.get("max_length"),
        loss_weighting=config.get("training", {}).get("loss_weighting"),
        loss_masking=config.get("training", {}).get("loss_masking"),
        in_batch_negative=config.get("training", {}).get("in_batch_negative"),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = None
    if data_cfg.get("val_instruction_path"):
        val_dataset = ClinicalInstructionDataset(
            data_root=data_cfg.get("data_root", "."),
            instruction_jsonl_path=data_cfg["val_instruction_path"],
            max_samples=data_cfg.get("max_val_samples"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(config["training"].get("eval_batch_size", config["training"]["batch_size"])),
            shuffle=False,
            num_workers=int(data_cfg.get("num_workers", 0)),
            collate_fn=collator,
            pin_memory=True,
        )
    return train_loader, val_loader


class CurriculumWindowDataset(Dataset):
    def __init__(self, base: ClinicalInstructionDataset) -> None:
        self.base = base
        self.indices_by_stage: dict[tuple[int, int], list[int]] = {}
        self.all_indices = list(range(len(base.records)))
        for idx, record in enumerate(base.records):
            start = record.get("curriculum_start_step")
            end = record.get("curriculum_end_step")
            if start is None or end is None:
                continue
            self.indices_by_stage.setdefault((int(start), int(end)), []).append(idx)
        self.active_indices = self.all_indices
        self._cursor = 0

    def set_step(self, global_step: int) -> None:
        active: list[int] = []
        for (start, end), indices in self.indices_by_stage.items():
            if int(start) <= global_step < int(end):
                active.extend(indices)
        self.active_indices = active or self.all_indices
        if self._cursor >= len(self.active_indices):
            self._cursor = 0

    def __len__(self) -> int:
        return max(1, len(self.active_indices))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if not self.active_indices:
            raise IndexError("No active curriculum samples")
        real_idx = self.active_indices[idx % len(self.active_indices)]
        return self.base[real_idx]

    def sample(self, rng: random.Random) -> dict[str, Any]:
        if not self.active_indices:
            self.set_step(0)
        return self.base[rng.choice(self.active_indices)]


def create_curriculum_dataset(config: dict[str, Any]) -> CurriculumWindowDataset | None:
    if not config.get("training", {}).get("curriculum_schedule"):
        return None
    data_cfg = config["data"]
    dataset = ClinicalInstructionDataset(
        data_root=data_cfg.get("data_root", "."),
        instruction_jsonl_path=data_cfg["train_instruction_path"],
        max_samples=data_cfg.get("max_train_samples"),
    )
    if not any("curriculum_start_step" in record and "curriculum_end_step" in record for record in dataset.records):
        return None
    return CurriculumWindowDataset(dataset)


def load_model_and_processor(config: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, Any]:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_cfg = config.get("model", {})
    model_path = str(model_cfg.get("model_path", DEFAULT_MODEL_PATH))
    dtype = choose_dtype(model_cfg.get("dtype", "bf16"))
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    train_groups = set(model_cfg.get("trainable_groups", ["vision_tower", "visual_connector"]))
    apply_freeze_plan(
        model,
        train_groups=train_groups,
        trainable_vision_last_n=model_cfg.get("trainable_vision_last_n"),
    )
    model.to(device)
    return model, processor


def create_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> AdamW:
    train_cfg = config["training"]
    base_lr = float(train_cfg["learning_rate"])
    vision_lr = float(train_cfg.get("vision_learning_rate", base_lr))
    connector_lr = float(train_cfg.get("connector_learning_rate", base_lr))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    groups: dict[str, list[torch.nn.Parameter]] = {
        "vision_tower": [],
        "visual_connector": [],
        "language_decoder": [],
        "other": [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        groups.setdefault(classify_parameter(name), []).append(param)
    param_groups = []
    if groups.get("vision_tower"):
        param_groups.append({"params": groups["vision_tower"], "lr": vision_lr, "weight_decay": weight_decay})
    if groups.get("visual_connector"):
        param_groups.append({"params": groups["visual_connector"], "lr": connector_lr, "weight_decay": weight_decay})
    if groups.get("language_decoder"):
        param_groups.append({"params": groups["language_decoder"], "lr": float(train_cfg.get("llm_learning_rate", base_lr)), "weight_decay": weight_decay})
    if groups.get("other"):
        param_groups.append({"params": groups["other"], "lr": base_lr, "weight_decay": weight_decay})
    if not param_groups:
        raise ValueError("No trainable parameters after freeze plan")
    return AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)


def trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def save_checkpoint(path: Path, model: torch.nn.Module, config: dict[str, Any], global_step: int, best_val_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "model_path": str(config.get("model", {}).get("model_path", DEFAULT_MODEL_PATH)),
            "trainable_state_dict": trainable_state_dict(model),
            "parameter_groups": count_parameters(model),
        },
        path,
    )


def load_trainable_checkpoint(path: Path, model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("trainable_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(device)
    print(
        json.dumps(
            {
                "resume": str(path),
                "missing_keys": len(missing),
                "unexpected_keys": len(unexpected),
                "global_step": checkpoint.get("global_step"),
            },
            indent=2,
        )
    )
    return checkpoint


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    for batch in tqdm(dataloader, desc="Validating", leave=False):
        batch = move_tensors_to_device(batch, device)
        loss, _ = compute_training_loss(model, batch, {"training": {}})
        losses.append(float(loss.detach().cpu()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def apply_debug_overrides(config: dict[str, Any]) -> None:
    data_cfg = config["data"]
    train_cfg = config["training"]
    data_cfg["max_train_samples"] = min(int(data_cfg.get("max_train_samples") or 4), 4)
    data_cfg["max_val_samples"] = min(int(data_cfg.get("max_val_samples") or 2), 2)
    data_cfg["num_workers"] = 0
    train_cfg["batch_size"] = 1
    train_cfg["eval_batch_size"] = 1
    train_cfg["gradient_accumulation_steps"] = 1
    train_cfg["max_steps"] = min(int(train_cfg.get("max_steps", 1)), 1)
    train_cfg["eval_interval"] = 1
    train_cfg["save_interval"] = 1
    train_cfg["save_checkpoints"] = False
    base_output = str(train_cfg.get("output_dir", "./outputs/qwen3vl_instruction_runs/debug")).rstrip("/\\")
    if not base_output.endswith("_debug"):
        train_cfg["output_dir"] = f"{base_output}_debug"
    config.setdefault("model", {})["dtype"] = config.get("model", {}).get("dtype", "bf16")


def save_run_config(config: dict[str, Any], config_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    snapshot = {"source_config": str(config_path), "resolved_config": config}
    (output_dir / "config_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")


def write_progress(output_dir: Path, events: list[dict[str, Any]]) -> None:
    (output_dir / "progress.json").write_text(json.dumps({"events": events}, indent=2), encoding="utf-8")
    lines = [json.dumps(event, ensure_ascii=False, sort_keys=True) for event in events]
    (output_dir / "training_log.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.debug:
        print("Debug mode enabled")
        apply_debug_overrides(config)
    if args.seed is not None:
        config["seed"] = args.seed

    set_seed(int(config.get("seed", 42)))
    requested_device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    output_dir = Path(config["training"]["output_dir"])
    if (output_dir / "metrics_final.json").exists():
        raise SystemExit(f"{output_dir / 'metrics_final.json'} already exists; remove it manually to rerun.")
    save_run_config(config, args.config, output_dir)

    model, processor = load_model_and_processor(config, device)
    print(json.dumps(count_parameters(model), indent=2))
    resume_checkpoint: dict[str, Any] | None = None
    if args.resume:
        resume_checkpoint = load_trainable_checkpoint(args.resume, model, device)
    train_loader, val_loader = create_dataloaders(config, processor)
    optimizer = create_optimizer(model, config)

    train_cfg = config["training"]
    max_steps = int(train_cfg["max_steps"])
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    log_interval = int(train_cfg.get("log_interval", 10))
    eval_interval = int(train_cfg.get("eval_interval", 100))
    save_interval = int(train_cfg.get("save_interval", 500))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    save_checkpoints = bool(train_cfg.get("save_checkpoints", True))
    save_best_checkpoint = bool(train_cfg.get("save_best_checkpoint", save_checkpoints))
    save_final_checkpoint = bool(train_cfg.get("save_final_checkpoint", save_checkpoints))

    global_step = int((resume_checkpoint or {}).get("global_step") or 0)
    best_val_loss = float((resume_checkpoint or {}).get("best_val_loss") or float("inf"))
    optimizer.zero_grad(set_to_none=True)
    model.train()
    started = time.time()
    progress_events: list[dict[str, Any]] = []
    progress_path = output_dir / "progress.json"
    if progress_path.exists() and args.resume:
        try:
            progress_events = json.loads(progress_path.read_text(encoding="utf-8")).get("events", [])
        except (json.JSONDecodeError, OSError):
            progress_events = []
    progress_events.append({"event": "start", "time": started, "output_dir": str(output_dir), "max_steps": max_steps})
    if args.resume:
        progress_events.append(
            {
                "event": "resume",
                "time": started,
                "checkpoint": str(args.resume),
                "global_step": global_step,
                "best_val_loss": best_val_loss,
            }
        )
    curriculum_dataset = create_curriculum_dataset(config)
    curriculum_rng = random.Random(int(config.get("seed", 42)))
    if curriculum_dataset is not None:
        progress_events.append(
            {
                "event": "curriculum_enabled",
                "stages": len(curriculum_dataset.indices_by_stage),
                "records": len(curriculum_dataset.base.records),
                "schedule": config.get("training", {}).get("curriculum_schedule"),
            }
        )
    write_progress(output_dir, progress_events)

    def train_one_batch(batch: dict[str, Any]) -> bool:
        nonlocal global_step, best_val_loss
        batch = move_tensors_to_device(batch, device)
        loss, loss_metrics = compute_training_loss(model, batch, config)
        (loss / grad_accum).backward()
        if (global_step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [param for param in model.parameters() if param.requires_grad],
                max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if global_step % log_interval == 0:
            event = {"event": "train_log", "global_step": global_step, "train_loss": float(loss.detach().cpu()), **loss_metrics}
            if curriculum_dataset is not None:
                curriculum_dataset.set_step(global_step)
                event["curriculum_active_records"] = len(curriculum_dataset.active_indices)
            progress_events.append(event)
            write_progress(output_dir, progress_events)
            print(json.dumps(event))

        if val_loader is not None and global_step % eval_interval == 0:
            val_loss = evaluate(model, val_loader, device)
            event = {"event": "eval", "global_step": global_step, "val_loss": val_loss}
            progress_events.append(event)
            (output_dir / f"metrics_step_{global_step}.json").write_text(json.dumps(event, indent=2), encoding="utf-8")
            write_progress(output_dir, progress_events)
            print(json.dumps(event))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_best_checkpoint:
                    save_checkpoint(output_dir / "checkpoints" / "best.pt", model, config, global_step, best_val_loss)

        if save_checkpoints and global_step % save_interval == 0:
            save_checkpoint(output_dir / "checkpoints" / f"step_{global_step}.pt", model, config, global_step, best_val_loss)

        return global_step >= max_steps

    while global_step < max_steps:
        if curriculum_dataset is not None:
            curriculum_dataset.set_step(global_step)
            items = [
                curriculum_dataset.sample(curriculum_rng)
                for _ in range(int(config["training"]["batch_size"]))
            ]
            batch = train_loader.collate_fn(items)
            if train_one_batch(batch):
                break
            continue
        for batch in tqdm(train_loader, desc="Training", leave=False):
            if train_one_batch(batch):
                break

    if val_loader is not None and not np.isfinite(best_val_loss):
        best_val_loss = evaluate(model, val_loader, device)
    if save_final_checkpoint:
        save_checkpoint(output_dir / "checkpoints" / "final.pt", model, config, global_step, best_val_loss)

    metrics = {
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "elapsed_seconds": time.time() - started,
        "train_records": len(train_loader.dataset),
        "val_records": len(val_loader.dataset) if val_loader is not None else 0,
        "parameter_groups": count_parameters(model),
    }
    (output_dir / "metrics_final.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "runtime_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    progress_events.append({"event": "complete", **metrics})
    write_progress(output_dir, progress_events)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
