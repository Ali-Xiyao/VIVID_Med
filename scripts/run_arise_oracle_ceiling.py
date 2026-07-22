#!/usr/bin/env python
"""Run the ARISE-CXR expert-mask oracle ceiling on locked development rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arise_cxr.oracle_ceiling import (  # noqa: E402
    evaluate_oracle_ceiling,
    load_locked_development_rows,
)


DEFAULT_CONFIG = ROOT / "configs/arise_cxr/oracle_ceiling_phase_h.yaml"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = config["source"]
    identity = config["identity"]
    gate = config["gate"]
    rows, merged = load_locked_development_rows(
        rows_path=ROOT / source["rows_path"],
        merged_result_path=ROOT / source["merged_result_path"],
        expected_input_lock_sha256=source["input_lock_canonical_sha256"],
        expected_model_id=identity["model_id"],
        expected_explanation_id=identity["explanation_id"],
        expected_pathologies=identity["pathologies"],
        expected_operators=identity["operators"],
    )
    result = evaluate_oracle_ceiling(
        rows,
        required_pathologies=identity["pathologies"],
        required_operators=identity["operators"],
        minimum_passing_pathologies=gate["minimum_passing_pathologies"],
        bootstrap_replicates=gate["bootstrap_replicates"],
        bootstrap_seed=gate["bootstrap_seed"],
        ci_lower_threshold=gate["ci_lower_threshold"],
    )
    result["source_rows_sha256"] = merged["rows_sha256"]
    result["source_result_canonical_sha256"] = merged["canonical_sha256"]
    from bives_cxr.provenance import canonical_json_sha256

    result.pop("canonical_sha256", None)
    result["canonical_sha256"] = canonical_json_sha256(result)
    output = args.output or (ROOT / config["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
