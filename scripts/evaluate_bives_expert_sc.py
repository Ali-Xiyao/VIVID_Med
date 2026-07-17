"""Evaluate locked external expert S/C prediction scores without model selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bives_cxr.expert_sc import (  # noqa: E402
    evaluate_expert_sc_predictions,
    read_expert_sc_manifest,
)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--locked-thresholds",
        type=Path,
        required=True,
        help="Development-locked thresholds; never derive these from VinDr test.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    args = parser.parse_args()
    thresholds = json.loads(args.locked_thresholds.read_text(encoding="utf-8"))
    result = evaluate_expert_sc_predictions(
        read_expert_sc_manifest(args.manifest),
        read_jsonl(args.predictions),
        thresholds,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
