"""Create an ignored, non-clinical BiVES local-overfit input from one local image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps


STATES = ("support", "contradict", "uncertain", "insufficient")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def transformed(image: Image.Image, index: int) -> Image.Image:
    image = image.convert("L")
    image = ImageEnhance.Brightness(image).enhance(0.82 + 0.06 * index)
    image = ImageEnhance.Contrast(image).enhance(0.90 + 0.05 * index)
    if index % 2:
        image = ImageOps.mirror(image)
    return image.convert("RGB")


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_rows(output_dir: Path, split: str, image: Image.Image) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, state in enumerate(STATES):
        name = f"{split}_{state}.png"
        target = output_dir / name
        transformed(image, index + (0 if split == "train" else 4)).save(target)
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
