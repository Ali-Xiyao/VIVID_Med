#!/usr/bin/env python
"""Create an aggregate case study from a completed ARISE dense-oracle run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arise_cxr.case_study import analyze_oracle_failure
from bives_cxr.provenance import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_path = args.run_dir / "result.json"
    rows_path = args.run_dir / "audit_rows.jsonl"
    if not result_path.is_file() or not rows_path.is_file():
        raise FileNotFoundError("completed ARISE result or rows are missing")
    source = json.loads(result_path.read_text(encoding="utf-8"))
    if source.get("status") != "complete_development":
        raise ValueError("ARISE case study requires a complete development run")
    if source.get("formal_result") is not False or source.get("test_opened") is not False:
        raise ValueError("ARISE case study source boundary changed")
    if source.get("rows_sha256") != file_sha256(rows_path):
        raise ValueError("ARISE dense-oracle row hash changed")
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
    report = analyze_oracle_failure(rows)
    report["source_result_canonical_sha256"] = source["canonical_sha256"]
    from bives_cxr.provenance import canonical_json_sha256

    report.pop("canonical_sha256")
    report["canonical_sha256"] = canonical_json_sha256(report)
    output = args.output or args.run_dir / "case_study.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
