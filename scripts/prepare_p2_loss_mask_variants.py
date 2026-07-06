"""Prepare P2 fixed-JSON diagnostic variants.

The value-only and no-punctuation variants can reuse the original D0 JSONL
with `training.loss_masking.mode` set in the config. This script materializes
the compact schema and field-query variants needed by the next-stage plan.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def read_jsonl(path: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if max_rows is not None and len(rows) >= max_rows:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def field_id(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def state_value(value: Any) -> str:
    if value in {"present", "absent", "uncertain"}:
        return str(value)
    return "null"


def parsed_target(row: dict[str, Any]) -> dict[str, Any]:
    answer = row.get("answer")
    if isinstance(answer, str):
        return json.loads(answer)
    if isinstance(answer, dict):
        return answer
    return {}


def base_from(row: dict[str, Any], version: str) -> dict[str, Any]:
    out = dict(row)
    out["source_version"] = version
    out["source_mode"] = "p2_diagnostic_programmatic"
    flags = [flag for flag in (out.get("quality_flags") or []) if not str(flag).startswith("d0_")]
    for flag in [version, "p2_diagnostic"]:
        if flag not in flags:
            flags.append(flag)
    out["quality_flags"] = flags
    return out


def compact_record(row: dict[str, Any], idx: int, version: str) -> dict[str, Any]:
    target = parsed_target(row)
    findings = target.get("findings") or {}
    lines = [f"{field_id(name)}:{state_value((payload or {}).get('state'))}" for name, payload in sorted(findings.items())]
    out = base_from(row, version)
    out["instruction_id"] = f"{row.get('sample_id')}_{version}_{idx:06d}"
    out["question"] = "Return compact finding states as field_id:state lines for this chest X-ray."
    out["answer"] = "\n".join(lines)
    out["answer_short"] = None
    out["answer_type"] = "p2_state_only_compact"
    out["finding"] = "global"
    out["state"] = "not_applicable"
    return out


def field_query_records(row: dict[str, Any], idx: int, version: str) -> list[dict[str, Any]]:
    target = parsed_target(row)
    findings = target.get("findings") or {}
    records: list[dict[str, Any]] = []
    for field_index, (name, payload) in enumerate(sorted(findings.items())):
        value = state_value((payload or {}).get("state"))
        out = base_from(row, version)
        out["instruction_id"] = f"{row.get('sample_id')}_{version}_{idx:06d}_{field_index:02d}"
        out["question"] = f"What is the status of {name}: present, absent, uncertain, or null?"
        out["answer"] = value
        out["answer_short"] = value
        out["answer_type"] = "p2_field_query"
        out["finding"] = name
        out["state"] = value
        records.append(out)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--compact-output", type=Path)
    parser.add_argument("--field-query-output", type=Path)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--compact-version", default="p2_state_only_compact")
    parser.add_argument("--field-query-version", default="p2_field_query")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input, max_rows=args.max_rows)
    summary: dict[str, Any] = {"input": str(args.input), "source_records": len(rows)}
    if args.compact_output:
        compact = [compact_record(row, idx, args.compact_version) for idx, row in enumerate(rows)]
        write_jsonl(args.compact_output, compact)
        summary["compact_records"] = len(compact)
        summary["compact_output"] = str(args.compact_output)
    if args.field_query_output:
        field_rows: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            field_rows.extend(field_query_records(row, idx, args.field_query_version))
        write_jsonl(args.field_query_output, field_rows)
        summary["field_query_records"] = len(field_rows)
        summary["field_query_output"] = str(args.field_query_output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
