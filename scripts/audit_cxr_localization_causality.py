"""Summarize precomputed localization-causality development rows.

This entrypoint is intentionally unable to open a locked test or load a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bives_cxr.localization_causality import summarize_audit_rows
from bives_cxr.provenance import canonical_json_sha256, file_sha256


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit model-free, precomputed development rows; test roles fail closed."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260719)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_jsonl.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    rows = _read_jsonl(input_path)
    summary = summarize_audit_rows(
        rows,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "audit_summary.json"
    _write_json(summary_path, summary)
    lock = {
        "status": "development_complete_nonformal",
        "formal_result": False,
        "test_opened": False,
        "input_jsonl": input_path.as_posix(),
        "input_jsonl_sha256": file_sha256(input_path),
        "input_rows_canonical_sha256": canonical_json_sha256(rows),
        "summary_canonical_sha256": canonical_json_sha256(summary),
        "module_sha256": file_sha256(
            REPO_ROOT / "bives_cxr" / "localization_causality.py"
        ),
        "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "bootstrap_seed": int(args.bootstrap_seed),
    }
    lock["canonical_sha256"] = canonical_json_sha256(lock)
    lock_path = output_dir / "development_lock.json"
    _write_json(lock_path, lock)
    print(
        json.dumps(
            {
                "status": lock["status"],
                "rows": summary["rows"],
                "patients": summary["patients"],
                "groups": len(summary["groups"]),
                "summary": summary_path.as_posix(),
                "lock": lock_path.as_posix(),
                "lock_canonical_sha256": lock["canonical_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
