"""Cache frozen Qwen3.5-2B patch tokens for the VICER V0 manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bives_cxr.backbones import Qwen35VisionAdapter, load_qwen35_visual_and_processor  # noqa: E402
from bives_cxr.qwen35_preprocessing import (  # noqa: E402
    QWEN35_IMAGE_PREPROCESS_VERSION,
    content_mask_for_grid,
    letterbox_image,
)
from scripts.cache_qwen35_patch_tokens import read_image, snapshot_files  # noqa: E402
from vicer_cxr.validity import canonical_sha256, file_sha256, validate_v0_manifest  # noqa: E402


FORMAT_VERSION = "vicer-v0-qwen35-patch-cache-v1"


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-lock", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--image-size", type=int, default=448)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    validate_v0_manifest(rows)
    data_lock = json.loads(args.data_lock.read_text(encoding="utf-8"))
    if data_lock.get("status") != "complete" or data_lock.get("manifest_sha256") != file_sha256(args.manifest):
        raise ValueError("VICER V0 data lock does not bind the manifest")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    items_dir = args.output_dir / "items"
    items_dir.mkdir(exist_ok=True)
    progress_path = args.output_dir / "progress.json"
    processor_files, model_files = snapshot_files(args.model_path)
    identity = {
        "format_version": FORMAT_VERSION,
        "manifest_sha256": file_sha256(args.manifest),
        "data_lock_canonical_sha256": data_lock["canonical_sha256"],
        "model_snapshot_sha256": canonical_sha256(model_files),
        "processor_snapshot_sha256": canonical_sha256(processor_files),
        "image_preprocess_version": QWEN35_IMAGE_PREPROCESS_VERSION,
        "image_size": int(args.image_size),
        "dtype": str(args.dtype),
    }
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != identity:
            raise ValueError("existing VICER cache progress has another identity")
    else:
        progress = {"identity": identity, "completed_images": {}}

    unique = {str(row["image_sha256"]): row for row in rows}
    device = torch.device(args.device)
    visual, processor, config = load_qwen35_visual_and_processor(
        args.model_path, dtype=args.dtype, attention_implementation="eager"
    )
    visual = visual.to(device).eval()
    adapter = Qwen35VisionAdapter(
        visual,
        spatial_merge_size=int(config["vision_config"]["spatial_merge_size"]),
    ).to(device).eval()
    completed = dict(progress.get("completed_images", {}))
    for image_index, (image_hash, row) in enumerate(sorted(unique.items()), start=1):
        target = items_dir / f"{image_hash}.pt"
        existing = completed.get(image_hash)
        if existing and target.is_file() and file_sha256(target) == existing["cache_file_sha256"]:
            continue
        source = Path(row["image_path"])
        if file_sha256(source) != image_hash:
            raise ValueError("VICER source image hash changed")
        image = read_image(source)
        letterboxed, content_box = letterbox_image(image, args.image_size)
        messages = [{"role": "user", "content": [{"type": "image", "image": letterboxed}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        batch = processor(text=[text], images=[letterboxed], return_tensors="pt", padding=True)
        pixel_values = batch["pixel_values"].to(device=device, dtype=next(visual.parameters()).dtype)
        grid = batch["image_grid_thw"].to(device)
        with torch.no_grad():
            patches = adapter(pixel_values, grid)
        content_mask = content_mask_for_grid(batch["image_grid_thw"][0], content_box, args.image_size)
        valid_mask = patches.valid_mask[0].detach().cpu() & content_mask
        payload = {
            "format_version": FORMAT_VERSION,
            "image_sha256": image_hash,
            "image_path": str(source),
            "patch_tokens": patches.tokens[0].detach().to(device="cpu", dtype=torch.bfloat16),
            "valid_mask": valid_mask,
            "grid_hw": list(patches.grid_hw[0]),
            "content_box": list(map(int, content_box)),
            "native_size": [int(image.width), int(image.height)],
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
        }
        progress = {
            "identity": identity,
            "status": "in_progress",
            "completed_images": completed,
            "completed_count": len(completed),
            "total_unique_images": len(unique),
        }
        write_json_atomic(progress_path, progress)
        print(json.dumps({"cached": image_index, "total": len(unique), "image_sha256": image_hash}))

    index_path = args.output_dir / "index.jsonl"
    index_path.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": row["sample_id"],
                    "image_sha256": row["image_sha256"],
                    "cache_file": completed[row["image_sha256"]]["cache_file"],
                },
                sort_keys=True,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )
    lock = {
        **identity,
        "status": "complete",
        "model_path": str(args.model_path.resolve()),
        "processor_files": processor_files,
        "model_files": model_files,
        "unique_images": len(unique),
        "records": len(rows),
        "index_sha256": file_sha256(index_path),
    }
    lock["canonical_sha256"] = canonical_sha256(lock)
    write_json_atomic(args.output_dir / "cache_lock.json", lock)
    progress.update({"status": "complete", "lock": "cache_lock.json"})
    write_json_atomic(progress_path, progress)
    print(json.dumps(lock, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
