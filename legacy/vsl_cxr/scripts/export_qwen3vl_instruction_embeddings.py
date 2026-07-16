"""Export Qwen3-VL vision embeddings for instruction rows."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.clinical_instruction_dataset import ClinicalInstructionDataset
from scripts.train_qwen3vl_clinical_instruction import (
    load_model_and_processor,
    load_trainable_checkpoint,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(raw: str, root: Path) -> str:
    path = Path(str(raw))
    if path.is_absolute():
        return str(path)
    return str(root / path)


def pool_visual_outputs(visual_outputs: Any, grid_thw: torch.Tensor, batch_size: int) -> torch.Tensor:
    hidden = getattr(visual_outputs, "last_hidden_state", visual_outputs)
    counts = grid_thw.prod(dim=1).long().tolist()
    if sum(counts) == int(hidden.shape[0]):
        chunks = torch.split(hidden, counts, dim=0)
    else:
        chunks = torch.chunk(hidden, batch_size, dim=0)
    return torch.stack([chunk.mean(dim=0) for chunk in chunks], dim=0)


def get_visual_module(model: torch.nn.Module) -> torch.nn.Module:
    for path in ("visual", "model.visual"):
        module: Any = model
        found = True
        for part in path.split("."):
            if not hasattr(module, part):
                found = False
                break
            module = getattr(module, part)
        if found:
            return module
    candidates = [
        name for name, _ in model.named_modules()
        if name and name.lower().endswith("visual")
    ]
    raise AttributeError(
        "Could not locate Qwen3-VL visual module. "
        f"Tried visual/model.visual; candidates={candidates[:10]}"
    )


def summarize_checkpoint_meta(checkpoint: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("global_step", "best_val_loss", "model_path"):
        if key in checkpoint:
            value = checkpoint.get(key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                summary[key] = value
            else:
                summary[key] = str(value)
    parameter_groups = checkpoint.get("parameter_groups")
    if isinstance(parameter_groups, dict):
        summary["parameter_groups"] = parameter_groups
    state = checkpoint.get("trainable_state_dict")
    if isinstance(state, dict):
        summary["trainable_state_keys"] = len(state)
    return summary


def metadata_from_item(item: dict[str, Any]) -> dict[str, Any]:
    meta = dict(item.get("metadata") or {})
    keep = {
        "instruction_id": item.get("instruction_id"),
        "sample_id": item.get("sample_id"),
        "image_path": item.get("image_path"),
        "question": item.get("question"),
        "answer": item.get("answer"),
        "answer_short": item.get("answer_short"),
        "answer_type": item.get("answer_type"),
        "finding": item.get("finding"),
        "state": item.get("state"),
        "laterality": meta.get("laterality"),
        "location": meta.get("location"),
        "severity": meta.get("severity"),
        "visual_dependency": item.get("visual_dependency"),
        "quality_flags": item.get("quality_flags") or [],
        "hard_negative_image_path": item.get("hard_negative_image_path"),
        "hard_negative_sample_id": meta.get("hard_negative_sample_id"),
        "hard_negative_reason": meta.get("hard_negative_reason"),
        "negative_answer": item.get("negative_answer"),
    }
    return keep


def collate_embeddings(batch: list[dict[str, Any]], processor: Any, prompt: str) -> dict[str, Any]:
    texts = []
    images = []
    metadata = []
    for item in batch:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item["image"]},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        images.append(item["image"])
        metadata.append(metadata_from_item(item))
    encoded = processor(text=texts, images=images, return_tensors="pt", padding=True)
    encoded["metadata"] = metadata
    return encoded


@torch.no_grad()
def export_embeddings(
    model: torch.nn.Module,
    processor: Any,
    dataset: ClinicalInstructionDataset,
    output_npz: Path,
    metadata_jsonl: Path,
    batch_size: int,
    device: torch.device,
    prompt: str,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate_embeddings(batch, processor, prompt),
    )
    embeddings: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    model.eval()
    visual = get_visual_module(model)
    use_bf16 = str(getattr(next(model.parameters()), "dtype", "")).endswith("bfloat16")
    for batch in tqdm(loader, desc="Exporting embeddings"):
        metadata = batch.pop("metadata")
        pixel_values = batch["pixel_values"].to(device)
        image_grid_thw = batch["image_grid_thw"].to(device)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16 and str(device).startswith("cuda")
            else torch.no_grad()
        )
        with autocast_ctx:
            visual_outputs = visual(pixel_values, grid_thw=image_grid_thw)
            features = pool_visual_outputs(visual_outputs, image_grid_thw, len(metadata))
        embeddings.append(features.float().cpu().numpy())
        rows.extend(metadata)

    matrix = np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 0), dtype=np.float32)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    metadata_jsonl.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, embeddings=matrix)
    with metadata_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "rows": len(rows),
        "embedding_shape": list(matrix.shape),
        "output_npz": str(output_npz),
        "metadata_jsonl": str(metadata_jsonl),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--instruction-jsonl", type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--metadata-jsonl", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device")
    parser.add_argument("--prompt", default="Represent this chest X-ray for finding-level retrieval.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    requested_device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    data_cfg = config["data"]
    instruction_path = args.instruction_jsonl or Path(str(data_cfg["train_instruction_path"]))
    dataset = ClinicalInstructionDataset(
        data_root=str(data_cfg.get("data_root", ".")),
        instruction_jsonl_path=str(instruction_path),
        max_samples=args.max_samples,
    )
    model, processor = load_model_and_processor(config, device)
    checkpoint_meta: dict[str, Any] = {
        "global_step": 0,
        "best_val_loss": None,
        "model_path": str(config.get("model", {}).get("model_path", "")),
    }
    if args.checkpoint:
        checkpoint_meta = load_trainable_checkpoint(args.checkpoint, model, device)
    result = export_embeddings(
        model=model,
        processor=processor,
        dataset=dataset,
        output_npz=args.output_npz,
        metadata_jsonl=args.metadata_jsonl,
        batch_size=args.batch_size,
        device=device,
        prompt=args.prompt,
    )
    manifest = {
        **result,
        "config": str(args.config),
        "checkpoint": str(args.checkpoint) if args.checkpoint else "",
        "instruction_jsonl": str(instruction_path),
        "checkpoint_meta": summarize_checkpoint_meta(checkpoint_meta),
    }
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
