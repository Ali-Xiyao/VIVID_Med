"""Export Qwen3-VL vision embeddings for UMS JSONL rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_qwen3vl_instruction_embeddings import get_visual_module, pool_visual_outputs
from scripts.train_qwen3vl_clinical_instruction import load_model_and_processor, load_trainable_checkpoint, set_seed
from scripts.train_qwen3vl_vision_lp import COMMON_LABELS, FINDING_NAMES, Qwen3VLLPCollator, UMSPILDataset


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def labels_to_metadata(labels: torch.Tensor, label_names: list[str]) -> dict[str, Any]:
    states: dict[str, str] = {}
    for idx, value in enumerate(labels.tolist()):
        if not np.isfinite(value):
            continue
        if value == 1.0:
            states[label_names[idx]] = "present"
        elif value == 0.0:
            states[label_names[idx]] = "absent"
        elif value == -1.0:
            states[label_names[idx]] = "uncertain"
    return states


@torch.no_grad()
def export_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    output_npz: Path,
    metadata_jsonl: Path,
    label_names: list[str],
    device: torch.device,
) -> dict[str, Any]:
    visual = get_visual_module(model)
    embeddings: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    model.eval()
    use_bf16 = str(getattr(next(model.parameters()), "dtype", "")).endswith("bfloat16")
    for batch in tqdm(loader, desc="Exporting UMS embeddings"):
        sample_ids = batch.pop("sample_ids")
        image_paths = batch.pop("image_paths")
        labels = batch.pop("labels")
        pixel_values = batch["pixel_values"].to(device)
        image_grid_thw = batch["image_grid_thw"].to(device)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16 and str(device).startswith("cuda")
            else torch.no_grad()
        )
        with autocast_ctx:
            visual_outputs = visual(pixel_values, grid_thw=image_grid_thw)
            features = pool_visual_outputs(visual_outputs, image_grid_thw, len(sample_ids))
        embeddings.append(features.float().cpu().numpy())
        for idx, sample_id in enumerate(sample_ids):
            metadata.append(
                {
                    "sample_id": sample_id,
                    "image_path": image_paths[idx],
                    "label_states": labels_to_metadata(labels[idx], label_names),
                }
            )
    matrix = np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 0), dtype=np.float32)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    metadata_jsonl.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, embeddings=matrix)
    with metadata_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for row in metadata:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "rows": len(metadata),
        "embedding_shape": list(matrix.shape),
        "output_npz": str(output_npz),
        "metadata_jsonl": str(metadata_jsonl),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--ums-jsonl", required=True, type=Path)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--metadata-jsonl", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--max-samples", type=int, default=0, help="Use 0 or a negative value for all available rows.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device")
    parser.add_argument("--use-common-labels-only", action="store_true")
    parser.add_argument("--prompt", default="Represent this chest X-ray for domain-shift analysis.")
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
    label_names = COMMON_LABELS if args.use_common_labels_only else FINDING_NAMES
    dataset = UMSPILDataset(
        data_root=args.data_root,
        ums_jsonl_path=str(args.ums_jsonl),
        label_names=label_names,
        max_samples=None if args.max_samples <= 0 else args.max_samples,
    )
    model, processor = load_model_and_processor(config, device)
    checkpoint_meta = load_trainable_checkpoint(args.checkpoint, model, device)
    collator = Qwen3VLLPCollator(processor=processor, prompt=args.prompt)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collator)
    result = export_embeddings(model, loader, args.output_npz, args.metadata_jsonl, label_names, device)
    manifest = {
        **result,
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "ums_jsonl": str(args.ums_jsonl),
        "data_root": args.data_root,
        "max_samples": "all_available" if args.max_samples <= 0 else args.max_samples,
        "checkpoint_meta": {
            "global_step": checkpoint_meta.get("global_step"),
            "best_val_loss": checkpoint_meta.get("best_val_loss"),
        },
    }
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
