"""Combine JSONL instruction shards with duplicate protection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("instruction_id") or ""),
        str(row.get("sample_id") or ""),
        str(row.get("question") or ""),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    input_counts: dict[str, int] = {}
    duplicate_count = 0

    for path in args.inputs:
        count = 0
        if not path.exists():
            input_counts[str(path)] = 0
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                count += 1
                key = row_key(row)
                if key in seen:
                    duplicate_count += 1
                    continue
                seen.add(key)
                rows.append(row)
        input_counts[str(path)] = count

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    payload = {
        "output": str(args.output),
        "rows": len(rows),
        "duplicates_dropped": duplicate_count,
        "input_counts": input_counts,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
