"""Train or resume the VSL-CXR HNMB backbone row."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qwen3vl_instruction/vsl_cxr/phase1_baselines/b6_sameq_hnmb.yaml"),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    target = Path(__file__).with_name("train_vsl_cxr.py")
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
