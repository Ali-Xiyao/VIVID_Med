"""Cache frozen Qwen3.5 patch tokens for locked weak S/C manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bives_cxr.backbones import Qwen35VisionAdapter, load_qwen35_visual_and_processor  # noqa: E402
from bives_cxr.expert_sc import file_sha256, read_expert_sc_manifest  # noqa: E402
from bives_cxr.qwen35_preprocessing import (  # noqa: E402
    QWEN35_IMAGE_PREPROCESS_VERSION,
    content_mask_for_grid,
    letterbox_image,
)


CACHE_FORMAT_VERSION = "bives_qwen35_patch_cache_v1"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def snapshot_files(model_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    processor_names = {
        "chat_template.jinja",
        "config.json",
        "merges.txt",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
    }
    model_names = {"config.json", "model.safetensors.index.json"}
    model_names.update(path.name for path in model_root.glob("*.safetensors"))
    processor = {
        name: file_sha256(model_root / name)
        for name in sorted(processor_names)
        if (model_root / name).is_file()
    }
    model = {
        name: file_sha256(model_root / name)
        for name in sorted(model_names)
        if (model_root / name).is_file()
    }
    return processor, model


def read_image(path: Path) -> Image.Image:
    if path.suffix.lower() in {".dicom", ".dcm"}:
        from bives_cxr.dicom import load_cxr_dicom

        image, _ = load_cxr_dicom(path)
        return image
    with Image.open(path) as source:
        return source.convert("RGB").copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--image-size", type=int, default=448)
    args = parser.parse_args()

    manifests = {
        "train": read_expert_sc_manifest(args.train_manifest),
        "val": read_expert_sc_manifest(args.val_manifest),
    }
    manifest_hashes = {
        "train": file_sha256(args.train_manifest),
        "val": file_sha256(args.val_manifest),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items_dir = args.output_dir / "items"
    items_dir.mkdir(exist_ok=True)
    progress_path = args.output_dir / "progress.json"

    processor_files, model_files = snapshot_files(args.model_path)
    processor_snapshot_sha256 = canonical_sha256(processor_files)
    model_snapshot_sha256 = canonical_sha256(model_files)
    identity = {
        "format_version": CACHE_FORMAT_VERSION,
        "manifest_sha256": manifest_hashes,
        "model_snapshot_sha256": model_snapshot_sha256,
        "processor_snapshot_sha256": processor_snapshot_sha256,
        "image_preprocess_version": QWEN35_IMAGE_PREPROCESS_VERSION,
        "image_size": int(args.image_size),
        "dtype": str(args.dtype),
    }
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != identity:
            raise ValueError("existing patch-cache progress belongs to a different identity")
    else:
        progress = {"identity": identity, "completed_images": {}}

    unique_images: dict[str, dict[str, Any]] = {}
    row_image_hash: dict[str, str] = {}
    for rows in manifests.values():
        for row in rows:
            image_path = Path(row["image_path"])
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            image_hash = str(row.get("image_sha256") or "")
            if not image_hash:
                image_hash = file_sha256(image_path)
            row_image_hash[str(row["sample_id"])] = image_hash
            previous = unique_images.get(image_hash)
            if previous is not None and Path(previous["image_path"]) != image_path:
                if file_sha256(previous["image_path"]) != image_hash:
                    raise ValueError(f"image SHA collision or stale manifest: {image_hash}")
            unique_images[image_hash] = {
                "image_path": str(image_path),
                "statement_text": str(row["statement_text"]),
            }

    device = torch.device(args.device)
    visual, processor, qwen_config = load_qwen35_visual_and_processor(
        args.model_path,
        dtype=args.dtype,
        attention_implementation="eager",
    )
    visual = visual.to(device).eval()
    adapter = Qwen35VisionAdapter(
        visual,
        spatial_merge_size=int(qwen_config["vision_config"]["spatial_merge_size"]),
    ).to(device).eval()
    completed: dict[str, Any] = dict(progress.get("completed_images", {}))
    for image_index, (image_hash, source) in enumerate(sorted(unique_images.items()), start=1):
        target = items_dir / f"{image_hash}.pt"
        existing = completed.get(image_hash)
        if existing and target.is_file() and file_sha256(target) == existing["cache_file_sha256"]:
            continue
        image = read_image(Path(source["image_path"]))
        image, content_box = letterbox_image(image, args.image_size)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": source["statement_text"]},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        batch = processor(text=[text], images=[image], return_tensors="pt", padding=True)
        pixel_values = batch["pixel_values"].to(
            device=device,
            dtype=next(visual.parameters()).dtype,
        )
        grid = batch["image_grid_thw"].to(device)
        with torch.no_grad():
            patches = adapter(pixel_values, grid)
        content_mask = content_mask_for_grid(
            batch["image_grid_thw"][0],
            content_box,
            args.image_size,
        )
        valid_mask = patches.valid_mask[0].detach().cpu() & content_mask
        payload = {
            "format_version": CACHE_FORMAT_VERSION,
            "image_sha256": image_hash,
            "image_path": source["image_path"],
            "patch_tokens": patches.tokens[0].detach().to(device="cpu", dtype=torch.bfloat16),
            "valid_mask": valid_mask,
            "grid_hw": list(patches.grid_hw[0]),
            "identity": identity,
        }
        temporary = target.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, target)
        completed[image_hash] = {
            "cache_file": str(target.relative_to(args.output_dir).as_posix()),
            "cache_file_sha256": file_sha256(target),
            "grid_hw": list(patches.grid_hw[0]),
            "valid_patches": int(valid_mask.sum()),
            "token_shape": list(payload["patch_tokens"].shape),
        }
        progress = {
            "identity": identity,
            "status": "in_progress",
            "completed_images": completed,
            "completed_count": len(completed),
            "total_unique_images": len(unique_images),
        }
        write_json_atomic(progress_path, progress)
        print(json.dumps({"cached": image_index, "total": len(unique_images), "image_sha256": image_hash}))

    for split, rows in manifests.items():
        index_path = args.output_dir / f"{split}_index.jsonl"
        with index_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                image_hash = row_image_hash[str(row["sample_id"])]
                handle.write(
                    json.dumps(
                        {
                            "sample_id": str(row["sample_id"]),
                            "unit_id": str(row["unit_id"]),
                            "patient_id": row.get("patient_id"),
                            "canonical_statement_id": str(row["canonical_statement_id"]),
                            "statement_text": str(row["statement_text"]),
                            "binary_label": int(row["binary_label"]),
                            "image_sha256": image_hash,
                            "cache_file": completed[image_hash]["cache_file"],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
    lock = {
        **identity,
        "status": "complete",
        "model_path": str(args.model_path.resolve()),
        "processor_files": processor_files,
        "model_files": model_files,
        "unique_images": len(unique_images),
        "train_records": len(manifests["train"]),
        "val_records": len(manifests["val"]),
        "train_index_sha256": file_sha256(args.output_dir / "train_index.jsonl"),
        "val_index_sha256": file_sha256(args.output_dir / "val_index.jsonl"),
    }
    write_json_atomic(args.output_dir / "cache_lock.json", lock)
    progress.update({"status": "complete", "lock": "cache_lock.json"})
    write_json_atomic(progress_path, progress)
    print(json.dumps(lock, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
