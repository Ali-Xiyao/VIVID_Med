"""Attach K hard negative images to SHUF-style instruction rows."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
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


def key_candidates(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        finding = str(row.get("finding") or "")
        state = str(row.get("state") or "")
        answer = str(row.get("answer_short") or row.get("answer") or "")
        location = str(row.get("location") or "")
        if finding and state:
            index[(finding, f"state:{state}")].append(row)
        if finding and answer:
            index[(finding, f"answer:{answer}")].append(row)
        if finding and location:
            index[(finding, f"location:{location}")].append(row)
    return index


def negative_keys(row: dict[str, Any]) -> list[tuple[str, str, str]]:
    finding = str(row.get("finding") or "")
    state = str(row.get("state") or "")
    answer = str(row.get("answer_short") or row.get("answer") or "")
    location = str(row.get("location") or "")
    keys: list[tuple[str, str, str]] = []
    opposite_state = {"present": "absent", "absent": "present", "uncertain": "present"}.get(state)
    if finding and opposite_state:
        keys.append((finding, f"state:{opposite_state}", "same_finding_opposite_state"))
    opposite_answer = {"A": "B", "B": "A", "Yes": "No", "No": "Yes"}.get(answer)
    if finding and opposite_answer:
        keys.append((finding, f"answer:{opposite_answer}", "same_finding_opposite_answer"))
    opposite_location = {"left": "right", "right": "left"}.get(location)
    if finding and opposite_location:
        keys.append((finding, f"location:{opposite_location}", "same_finding_opposite_laterality"))
    return keys


def attach_k_negatives(rows: list[dict[str, Any]], pool_rows: list[dict[str, Any]], k: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    index = key_candidates(pool_rows)
    output: list[dict[str, Any]] = []
    for row in rows:
        candidates: list[tuple[str, dict[str, Any]]] = []
        for finding, key, reason in negative_keys(row):
            for candidate in index.get((finding, key), []):
                if str(candidate.get("sample_id")) == str(row.get("sample_id")):
                    continue
                if not candidate.get("image_path"):
                    continue
                candidates.append((reason, candidate))
        rng.shuffle(candidates)
        chosen_paths: list[str] = []
        chosen_ids: list[str] = []
        chosen_reasons: list[str] = []
        for reason, candidate in candidates:
            path = str(candidate.get("image_path") or "")
            if not path or path in chosen_paths:
                continue
            chosen_paths.append(path)
            chosen_ids.append(str(candidate.get("sample_id") or ""))
            chosen_reasons.append(reason)
            if len(chosen_paths) >= k:
                break
        new = dict(row)
        new["hard_negative_image_paths"] = chosen_paths
        new["hard_negative_sample_ids"] = chosen_ids
        new["hard_negative_reasons"] = chosen_reasons
        if chosen_paths:
            new["hard_negative_image_path"] = chosen_paths[0]
            new["hard_negative_sample_id"] = chosen_ids[0]
            new["hard_negative_reason"] = chosen_reasons[0]
            flags = list(new.get("quality_flags") or [])
            for flag in ["hard_image_shuffle", f"multi_negative_k{k}"]:
                if flag not in flags:
                    flags.append(flag)
            new["quality_flags"] = flags
        else:
            new["hard_negative_image_path"] = ""
            new["hard_negative_sample_id"] = ""
            new["hard_negative_reason"] = "no_candidate"
        output.append(new)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pool", type=Path, help="Optional candidate pool JSONL. Defaults to --input.")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input, max_rows=args.max_rows)
    pool_rows = read_jsonl(args.pool or args.input)
    output = attach_k_negatives(rows, pool_rows, args.k, args.seed)
    write_jsonl(args.output, output)
    covered = sum(1 for row in output if row.get("hard_negative_image_paths"))
    print(json.dumps({"input": str(args.input), "output": str(args.output), "records": len(output), "covered": covered, "k": args.k}, indent=2))


if __name__ == "__main__":
    main()
