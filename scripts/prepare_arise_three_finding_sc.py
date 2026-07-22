#!/usr/bin/env python
"""Prepare locked ARISE weak S/C manifests for the three-finding gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arise_cxr.weak_sc import prepare_arise_weak_sc  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / "local_runs/bives_cxr/p0_intake/mimic_parser_candidates_5k.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "local_runs/arise_cxr/weak_sc_three_finding_v1",
    )
    parser.add_argument("--skip-image-hash", action="store_true")
    args = parser.parse_args()
    result = prepare_arise_weak_sc(
        args.candidates,
        args.output_dir,
        findings=("consolidation", "pleural_effusion", "pulmonary_edema"),
        verify_images=not args.skip_image_hash,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
