"""Write a temporary evaluation config with overridden validation JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--val-instruction-path", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    cfg.setdefault("data", {})["val_instruction_path"] = args.val_instruction_path
    cfg["data"]["max_val_samples"] = cfg["data"].get("max_val_samples", 1000)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(str(args.output))


if __name__ == "__main__":
    main()
