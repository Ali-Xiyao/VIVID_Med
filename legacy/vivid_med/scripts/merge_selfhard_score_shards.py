"""Merge SelfHard-SHUF score shards into final JSONL/CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def file_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime": stat.st_mtime}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, nargs="+", type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    existing = [path for path in args.inputs if path.exists()]
    if len(existing) != len(args.inputs) and not args.allow_missing:
        missing = [str(path) for path in args.inputs if not path.exists()]
        raise SystemExit(f"Missing shard files: {missing}")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates = 0
    for path in existing:
        for row in read_jsonl(path):
            key = str(row.get("instruction_id") or "")
            if key and key in seen:
                duplicates += 1
                continue
            if key:
                seen.add(key)
            rows.append(row)
    rows.sort(key=lambda row: float(row.get("hard_negative_nll") or float("inf")))
    write_jsonl(args.output_jsonl, rows)
    if args.output_csv:
        write_csv(args.output_csv, rows)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": [file_info(path) for path in existing],
        "missing_inputs": [str(path) for path in args.inputs if not path.exists()],
        "output_jsonl": str(args.output_jsonl),
        "output_csv": str(args.output_csv) if args.output_csv else None,
        "rows": len(rows),
        "duplicates_dropped": duplicates,
    }
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
