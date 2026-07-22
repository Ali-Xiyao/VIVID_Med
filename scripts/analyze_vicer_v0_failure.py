#!/usr/bin/env python
"""Create a deterministic aggregate-only VICER V0 failure case study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vicer_cxr.case_study import analyze_v0_failure
from vicer_cxr.validity import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    if file_sha256(args.rows) != result.get("rows_sha256"):
        raise ValueError("VICER V0 row file hash does not match the frozen result")
    rows = [
        json.loads(line)
        for line in args.rows.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = analyze_v0_failure(rows, result)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# VICER-CXR V0 Failure Case Study",
        "",
        f"- Result hash: `{report['v0_result_canonical_sha256']}`",
        f"- Rows hash: `{report['v0_rows_sha256']}`",
        f"- Cells passed: `{report['cells_passed']}/{report['cells_total']}`",
        "- Surviving operator families: `none`",
        "",
        "| Operator family | Finding | Pass | Failed components | Valid fraction | Mean valid gap |",
        "| --- | --- | ---: | --- | ---: | ---: |",
    ]
    for key, cell in report["cells"].items():
        family, finding = key.split("|", 1)
        failures = ", ".join(cell["failed_components"]) or "none"
        lines.append(
            f"| {family} | {finding} | {str(cell['pass']).lower()} | {failures} | "
            f"{cell['valid_fraction']:.4f} | {cell['mean_valid_target_control_gap']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
            report["diagnosis"]["interpretation"],
            "",
            "## Decision",
            "",
            report["diagnosis"]["repair_decision"],
            "",
            "No threshold changed, no model was rescored, no test split was opened, and no selector started.",
        ]
    )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
