"""Generate qualitative positive/negative image casebooks for next-stage data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and len(rows) >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_path(raw: Any, data_root: Path) -> Path:
    path = Path(str(raw or ""))
    if path.is_absolute():
        return path
    return data_root / path


def load_thumb(path: Path, size: int) -> Image.Image:
    try:
        image = Image.open(path).convert("RGB")
    except Exception:
        image = Image.new("RGB", (size, size), (0, 0, 0))
    image.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), (245, 245, 245))
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def make_pair_image(pos_path: Path, neg_path: Path, output: Path, size: int = 320) -> None:
    left = load_thumb(pos_path, size)
    right = load_thumb(neg_path, size)
    gap = 16
    header = 28
    canvas = Image.new("RGB", (size * 2 + gap, size + header), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), "positive image", fill=(0, 0, 0))
    draw.text((size + gap + 8, 6), "hard negative", fill=(0, 0, 0))
    canvas.paste(left, (0, header))
    canvas.paste(right, (size + gap, header))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-md", default=ROOT / "outputs/final_tables/next_stage_qualitative_cases.md", type=Path)
    parser.add_argument("--asset-dir", default=ROOT / "outputs/final_tables/next_stage_qualitative_assets", type=Path)
    parser.add_argument("--data-root", default=ROOT, type=Path)
    parser.add_argument("--limit", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [row for row in read_jsonl(args.input) if row.get("hard_negative_image_path")]
    selected = rows[: args.limit]
    lines = [
        "# Next-Stage Qualitative Hard-Negative Cases",
        "",
        f"- Source: `{args.input}`",
        f"- Cases: `{len(selected)}`",
        "- Boundary: side-by-side casebook for qualitative inspection; not a Grad-CAM or attention attribution map.",
        "",
    ]
    for idx, row in enumerate(selected, start=1):
        pos = resolve_path(row.get("image_path"), args.data_root)
        neg = resolve_path(row.get("hard_negative_image_path"), args.data_root)
        asset = args.asset_dir / f"case_{idx:03d}.jpg"
        make_pair_image(pos, neg, asset)
        lines.extend(
            [
                f"## Case {idx:03d}",
                "",
                f"![case {idx:03d}]({rel(asset)})",
                "",
                f"- sample_id: `{row.get('sample_id')}`",
                f"- finding: `{row.get('finding')}`",
                f"- state: `{row.get('state')}`",
                f"- answer_type: `{row.get('answer_type')}`",
                f"- hard_negative_reason: `{row.get('hard_negative_reason')}`",
                f"- answer: `{row.get('answer_short') or row.get('answer')}`",
                f"- question: {str(row.get('question') or '').replace(chr(10), ' ')}",
                "",
            ]
        )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"source": str(args.input), "cases": len(selected), "output_md": str(args.output_md)}, indent=2))


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    main()
