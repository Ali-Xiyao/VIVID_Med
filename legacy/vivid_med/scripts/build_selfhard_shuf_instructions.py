"""Build SelfHard-SHUF rows by oversampling low wrong-image-NLL examples."""

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


def index_scores(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("instruction_id")): row for row in rows if row.get("instruction_id")}


def build_rows(
    base_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    hard_fraction: float,
    repeat_hard: int,
    max_rows: int | None,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    score_by_id = index_scores(score_rows)
    scored = [
        row for row in base_rows
        if row.get("instruction_id") in score_by_id and row.get("hard_negative_image_path")
    ]
    scored.sort(key=lambda row: float(score_by_id[str(row["instruction_id"])].get("hard_negative_nll") or float("inf")))
    hard_n = max(1, int(len(scored) * hard_fraction)) if scored else 0
    hard_ids = {str(row.get("instruction_id")) for row in scored[:hard_n]}
    output: list[dict[str, Any]] = []
    for index, row in enumerate(base_rows):
        new = dict(row)
        flags = list(new.get("quality_flags") or [])
        if str(row.get("instruction_id")) in hard_ids:
            if "selfhard_shuf" not in flags:
                flags.append("selfhard_shuf")
            score = score_by_id[str(row["instruction_id"])]
            metadata = dict(new.get("metadata") or {})
            metadata["selfhard_shuf"] = {"hard_negative_nll": score.get("hard_negative_nll"), "token_count": score.get("token_count")}
            new["metadata"] = metadata
            for repeat in range(max(1, repeat_hard)):
                repeated = dict(new)
                repeated["instruction_id"] = f"{row.get('instruction_id', index)}_selfhard_{repeat:02d}"
                repeated["quality_flags"] = flags
                output.append(repeated)
        else:
            output.append(new)
    rng.shuffle(output)
    if max_rows is not None:
        output = output[:max_rows]
    summary = {
        "base_rows": len(base_rows),
        "score_rows": len(score_rows),
        "scored_rows": len(scored),
        "hard_rows": hard_n,
        "hard_fraction": hard_fraction,
        "repeat_hard": repeat_hard,
        "output_rows": len(output),
    }
    return output, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-jsonl", required=True, type=Path)
    parser.add_argument("--scores-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--hard-fraction", type=float, default=0.25)
    parser.add_argument("--repeat-hard", type=int, default=3)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary = build_rows(
        base_rows=read_jsonl(args.base_jsonl),
        score_rows=read_jsonl(args.scores_jsonl),
        hard_fraction=args.hard_fraction,
        repeat_hard=args.repeat_hard,
        max_rows=args.max_rows,
        seed=args.seed,
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
