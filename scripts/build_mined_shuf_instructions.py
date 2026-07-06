"""Build mined/self-hard SHUF instruction JSONL from base rows and mined pairs."""

from __future__ import annotations

import argparse
import csv
import json
import random
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


def row_key(row: dict[str, Any]) -> str:
    instruction_id = row.get("instruction_id")
    if instruction_id:
        return str(instruction_id)
    return "|".join(
        str(row.get(key) or "")
        for key in ["sample_id", "image_path", "finding", "state", "answer", "answer_short"]
    )


def index_pairs(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("instruction_id") or "")
        if not key:
            key = "|".join(
                str(row.get(name) or "")
                for name in ["sample_id", "image_path", "finding", "state", "answer"]
            )
        old = indexed.get(key)
        if old is None or float(row.get("rank") or 9999) < float(old.get("rank") or 9999):
            indexed[key] = row
    return indexed


def patch_row(row: dict[str, Any], pair: dict[str, Any], tag: str, index: int) -> dict[str, Any]:
    new = dict(row)
    new["instruction_id"] = f"{row.get('instruction_id', row.get('sample_id', index))}_{tag}_{index:06d}"
    new["hard_negative_image_path"] = pair.get("negative_image_path")
    new["hard_negative_sample_id"] = pair.get("negative_sample_id")
    new["hard_negative_expected_answer"] = pair.get("negative_answer")
    new["hard_negative_reason"] = tag
    flags = list(new.get("quality_flags") or [])
    for flag in ["hard_image_shuffle", tag]:
        if flag not in flags:
            flags.append(flag)
    new["quality_flags"] = flags
    metadata = dict(new.get("metadata") or {})
    metadata[tag] = {
        "negative_instruction_id": pair.get("negative_instruction_id"),
        "negative_sample_id": pair.get("negative_sample_id"),
        "negative_state": pair.get("negative_state"),
        "negative_answer": pair.get("negative_answer"),
        "negative_laterality": pair.get("negative_laterality"),
        "negative_location": pair.get("negative_location"),
        "cosine_similarity": pair.get("cosine_similarity"),
        "rank": pair.get("rank"),
    }
    new["metadata"] = metadata
    return new


def build_rows(
    base_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    tag: str,
    max_rows: int | None,
    seed: int,
    keep_unmatched: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    pair_by_key = index_pairs(pair_rows)
    output: list[dict[str, Any]] = []
    matched = 0
    unmatched = 0
    for index, row in enumerate(base_rows):
        pair = pair_by_key.get(row_key(row))
        if pair is None:
            unmatched += 1
            if keep_unmatched:
                output.append(dict(row))
            continue
        matched += 1
        output.append(patch_row(row, pair, tag, index))
    rng.shuffle(output)
    if max_rows is not None:
        output = output[:max_rows]
    summary = {
        "base_rows": len(base_rows),
        "pair_rows": len(pair_rows),
        "matched_rows": matched,
        "unmatched_rows": unmatched,
        "output_rows": len(output),
        "tag": tag,
    }
    return output, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-jsonl", required=True, type=Path)
    parser.add_argument("--mined-pairs", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--tag", default="embedding_mined_hard_negative")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-unmatched", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary = build_rows(
        base_rows=read_jsonl(args.base_jsonl),
        pair_rows=read_jsonl(args.mined_pairs),
        tag=args.tag,
        max_rows=args.max_rows,
        seed=args.seed,
        keep_unmatched=args.keep_unmatched,
    )
    write_jsonl(args.output_jsonl, rows)
    summary["output_jsonl"] = str(args.output_jsonl)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.summary_csv:
        write_csv(args.summary_csv, [summary])
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
