"""Train VSL-CXR instruction models using the existing Qwen3-VL trainer."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = Path(__file__).with_name("train_qwen3vl_clinical_instruction.py")
    forwarded = [str(target), "--config", str(args.config)]
    if args.resume:
        forwarded.extend(["--resume", str(args.resume)])
    if args.debug:
        forwarded.append("--debug")
    if args.seed is not None:
        forwarded.extend(["--seed", str(args.seed)])
    sys.argv = forwarded
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
