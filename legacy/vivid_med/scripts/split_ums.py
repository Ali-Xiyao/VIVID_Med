"""
Split a UMS JSONL file into train/val JSONL files with a fixed seed.

Usage:
    python split_ums.py --input ../data/dataset/processed/chexpert_ums.jsonl \
        --train_output ../data/dataset/processed/chexpert_ums_train.jsonl \
        --val_output ../data/dataset/processed/chexpert_ums_val.jsonl \
        --val_size 1000 --seed 42
"""

import argparse
import json
import random
from pathlib import Path
from typing import List


def read_jsonl(path: Path) -> List[str]:
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # basic validation
            json.loads(line)
            lines.append(line)
    return lines


def write_jsonl(path: Path, lines: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(description="Split UMS JSONL into train/val")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL path")
    parser.add_argument("--train_output", type=str, required=True, help="Train JSONL output")
    parser.add_argument("--val_output", type=str, required=True, help="Val JSONL output")
    parser.add_argument("--val_size", type=int, default=1000, help="Number of val samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before split (recommended)")
    args = parser.parse_args()

    input_path = Path(args.input)
    train_output = Path(args.train_output)
    val_output = Path(args.val_output)

    lines = read_jsonl(input_path)
    if args.val_size <= 0 or args.val_size >= len(lines):
        raise ValueError(f"val_size must be in (0, {len(lines)-1}]")

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(lines)

    val_lines = lines[: args.val_size]
    train_lines = lines[args.val_size :]

    write_jsonl(train_output, train_lines)
    write_jsonl(val_output, val_lines)

    print(f"Total: {len(lines)}")
    print(f"Train: {len(train_lines)} -> {train_output}")
    print(f"Val:   {len(val_lines)} -> {val_output}")


if __name__ == "__main__":
    main()
