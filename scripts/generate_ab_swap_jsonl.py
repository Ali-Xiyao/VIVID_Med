"""Create A/B-swapped diagnostic JSONL for option-bias evaluation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


OPTION_RE = re.compile(r"(?ms)^\s*([AB])([\.)])\s*(.+?)(?=^\s*[AB][\.)]\s*|\Z)")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def swap_answer(answer: Any) -> str:
    value = str(answer or "").strip()
    if value.upper() == "A":
        return "B"
    if value.upper() == "B":
        return "A"
    return value


def swap_question(question: str, option_a: str | None, option_b: str | None) -> str | None:
    if option_a and option_b:
        stem = re.split(r"(?m)^\s*A[\.)]\s*", question or "", maxsplit=1)[0].rstrip()
        return f"{stem}\nA. {option_b}\nB. {option_a}".strip()
    matches = list(OPTION_RE.finditer(question or ""))
    if len(matches) < 2:
        return None
    a = matches[0].group(3).strip()
    b = matches[1].group(3).strip()
    stem = question[: matches[0].start()].rstrip()
    return f"{stem}\nA. {b}\nB. {a}".strip()


def swap_row(row: dict[str, Any]) -> dict[str, Any] | None:
    answer = str(row.get("answer_short") or row.get("answer") or "").strip().upper()
    if answer not in {"A", "B"}:
        return None
    option_a = row.get("option_a")
    option_b = row.get("option_b")
    question = swap_question(str(row.get("question") or ""), str(option_a) if option_a else None, str(option_b) if option_b else None)
    if not question:
        return None
    out = dict(row)
    out["instruction_id"] = f"{row.get('instruction_id')}_ab_swap"
    out["question"] = question
    out["answer"] = swap_answer(row.get("answer"))
    out["answer_short"] = swap_answer(row.get("answer_short") or row.get("answer"))
    if option_a or option_b:
        out["option_a"] = option_b
        out["option_b"] = option_a
    out["ab_swap_source_instruction_id"] = row.get("instruction_id")
    flags = list(out.get("quality_flags") or [])
    if "ab_swap_diagnostic" not in flags:
        flags.append("ab_swap_diagnostic")
    out["quality_flags"] = flags
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-records", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    swapped = []
    for row in read_jsonl(args.input):
        item = swap_row(row)
        if item is None:
            continue
        swapped.append(item)
        if args.max_records is not None and len(swapped) >= args.max_records:
            break
    write_jsonl(args.output, swapped)
    print(json.dumps({"input": str(args.input), "output": str(args.output), "swapped": len(swapped)}, indent=2))


if __name__ == "__main__":
    main()
