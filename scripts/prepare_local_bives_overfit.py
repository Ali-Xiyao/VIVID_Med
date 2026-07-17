"""Create an ignored, non-clinical BiVES local-overfit input from one local image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


STATES = ("support", "contradict", "uncertain", "insufficient")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def balanced_uncertain(image: Image.Image) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Build an equal-area support/contradict spatial mixture for local-only U.

    Posterization is a third texture, not an engineering definition of balanced
    bipolar evidence.  This fixture deliberately exposes equal support-like and
    contradict-like regions at a scale much larger than one vision patch.
    """

    support = ImageOps.autocontrast(image.convert("L"))
    contradict = ImageOps.invert(support)
    width, height = support.size
    midpoint_x, midpoint_y = width // 2, height // 2
    output = Image.new("L", (width, height))
    positive = Image.new("L", (width, height), color=0)
    negative = Image.new("L", (width, height), color=0)
    tiles = (
        (0, 0, midpoint_x, midpoint_y, support, positive),
        (midpoint_x, 0, width, midpoint_y, contradict, negative),
        (0, midpoint_y, midpoint_x, height, contradict, negative),
        (midpoint_x, midpoint_y, width, height, support, positive),
    )
    for left, top, right, bottom, source, mask in tiles:
        box = (left, top, right, bottom)
        output.paste(source.crop(box), box)
        mask.paste(255, box)
    return output, positive, negative


def transformed_with_masks(
    image: Image.Image, index: int
) -> tuple[Image.Image, Image.Image | None, Image.Image | None]:
    image = image.convert("L")
    state_index = index % len(STATES)
    validation_variant = index >= len(STATES)
    if validation_variant:
        fill = int(np.median(np.asarray(image)))
        image = image.rotate(
            1.0,
            resample=Image.Resampling.BICUBIC,
            fillcolor=fill,
        )
    positive_mask: Image.Image | None = None
    negative_mask: Image.Image | None = None
    if state_index == 0:
        image = ImageOps.autocontrast(image)
    elif state_index == 1:
        image = ImageOps.invert(ImageOps.autocontrast(image))
    elif state_index == 2:
        image, positive_mask, negative_mask = balanced_uncertain(image)
    else:
        image = image.filter(ImageFilter.GaussianBlur(radius=8.0))
        image = ImageEnhance.Brightness(image).enhance(0.55)
    if validation_variant:
        image = ImageEnhance.Contrast(image).enhance(0.92)
    return image.convert("RGB"), positive_mask, negative_mask


def transformed(image: Image.Image, index: int) -> Image.Image:
    return transformed_with_masks(image, index)[0]


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_rows(output_dir: Path, split: str, image: Image.Image) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, state in enumerate(STATES):
        name = f"{split}_{state}.png"
        target = output_dir / name
        transformed_image, positive_mask, negative_mask = transformed_with_masks(
            image, index + (0 if split == "train" else 4)
        )
        transformed_image.save(target)
        if state == "uncertain":
            assert positive_mask is not None and negative_mask is not None
            positive_mask.save(output_dir / f"{split}_uncertain_positive_mask.png")
            negative_mask.save(output_dir / f"{split}_uncertain_negative_mask.png")
        rows.append(
            {
                "sample_id": f"local-overfit-{split}-{state}",
                "patient_id": f"local-overfit-{split}-patient-{index}",
                "image_path": str(target.as_posix()),
                "group_id": f"local-overfit-{split}-quartet",
                "canonical_statement_id": "local-overfit-synthetic-statement",
                "statement_text": "Synthetic local mechanism statement.",
                "state": state,
                "source_dataset": "local_mechanism_gate",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if not args.source_image.is_file():
        raise FileNotFoundError(args.source_image)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(args.source_image) as source:
        image = source.copy()
    write_jsonl(args.output_dir / "train.jsonl", make_rows(args.output_dir, "train", image))
    write_jsonl(args.output_dir / "val.jsonl", make_rows(args.output_dir, "val", image))
    print(json.dumps({"formal_result": False, "train": str(args.output_dir / "train.jsonl"), "val": str(args.output_dir / "val.jsonl")}, indent=2))


if __name__ == "__main__":
    main()
