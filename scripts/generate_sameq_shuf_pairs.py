"""Generate same-question different-answer SHUF instruction rows."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_p4v2_d6_d7 import clean_facts, finding_text, read_jsonl, write_jsonl  # noqa: E402


def fact_rows_by_key(facts_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]]:
    groups: dict[tuple[str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row in facts_rows:
        for fact in clean_facts(row):
            finding = str(fact.get("finding") or "")
            state = str(fact.get("state") or "")
            location = str(fact.get("location") or "")
            if state in {"present", "absent"}:
                groups[("state", finding, state)].append((row, fact))
            if location in {"left", "right"}:
                groups[("laterality", finding, location)].append((row, fact))
    return groups


def base_record(
    row: dict[str, Any],
    fact: dict[str, Any],
    source_version: str,
    idx: int,
    question: str,
    answer: str,
    negative: tuple[dict[str, Any], dict[str, Any]],
    sameq_type: str,
) -> dict[str, Any]:
    sid = str(row.get("sample_id"))
    negative_row, negative_fact = negative
    return {
        "instruction_id": f"{sid}_{source_version}_{sameq_type}_{idx:06d}",
        "sample_id": sid,
        "image_path": row.get("image_path"),
        "report": row.get("report"),
        "report_text": row.get("report"),
        "question": question,
        "answer": answer,
        "answer_short": answer,
        "finding": fact.get("finding"),
        "state": fact.get("state"),
        "answer_type": "same_question_different_answer",
        "visual_dependency": "very_high",
        "evidence_span": fact.get("evidence_span"),
        "location": fact.get("location"),
        "laterality": fact.get("location") if fact.get("location") in {"left", "right", "bilateral"} else None,
        "severity": fact.get("severity"),
        "certainty": fact.get("certainty"),
        "sameq_type": sameq_type,
        "hard_negative_image_path": negative_row.get("image_path"),
        "hard_negative_sample_id": negative_row.get("sample_id"),
        "hard_negative_expected_answer": "B" if answer == "A" else "A",
        "hard_negative_reason": f"same_question_{sameq_type}",
        "negative_answer": "B" if answer == "A" else "A",
        "negative_fact_state": negative_fact.get("state"),
        "source_version": source_version,
        "source_mode": "sameq_shuf_programmatic",
        "quality_flags": [source_version, "same_question", "hard_image_shuffle", "standardized_ab"],
        "metadata": {"sameq_type": sameq_type},
    }


def state_question(finding: str) -> str:
    text = finding_text(finding)
    return f"Which statement is better supported by this chest X-ray?\nA. There is {text}.\nB. There is no {text}."


def laterality_question(finding: str) -> str:
    text = finding_text(finding)
    return f"Which statement is better supported by this chest X-ray?\nA. There is left {text}.\nB. There is right {text}."


def build_sameq_rows(facts_rows: list[dict[str, Any]], source_version: str, max_pairs: int | None, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups = fact_rows_by_key(facts_rows)
    rows: list[dict[str, Any]] = []
    pair_count = 0

    findings = sorted({key[1] for key in groups if key[0] == "state"})
    for finding in findings:
        present = list(groups.get(("state", finding, "present"), []))
        absent = list(groups.get(("state", finding, "absent"), []))
        rng.shuffle(present)
        rng.shuffle(absent)
        for item_a, item_b in zip(present, absent):
            if max_pairs is not None and pair_count >= max_pairs:
                return rows
            question = state_question(finding)
            rows.append(base_record(item_a[0], item_a[1], source_version, len(rows), question, "A", item_b, "state"))
            rows.append(base_record(item_b[0], item_b[1], source_version, len(rows), question, "B", item_a, "state"))
            pair_count += 1

    findings = sorted({key[1] for key in groups if key[0] == "laterality"})
    for finding in findings:
        left = list(groups.get(("laterality", finding, "left"), []))
        right = list(groups.get(("laterality", finding, "right"), []))
        rng.shuffle(left)
        rng.shuffle(right)
        for item_a, item_b in zip(left, right):
            if max_pairs is not None and pair_count >= max_pairs:
                return rows
            question = laterality_question(finding)
            rows.append(base_record(item_a[0], item_a[1], source_version, len(rows), question, "A", item_b, "laterality"))
            rows.append(base_record(item_b[0], item_b[1], source_version, len(rows), question, "B", item_a, "laterality"))
            pair_count += 1
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-version", default="sameq_shuf")
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    facts_rows = read_jsonl(args.facts, max_samples=args.max_samples)
    rows = build_sameq_rows(facts_rows, args.source_version, args.max_pairs, args.seed)
    write_jsonl(args.output, rows)
    print(json.dumps({"facts": str(args.facts), "output": str(args.output), "records": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
