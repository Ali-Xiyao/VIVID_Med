"""Generate rich QA mixture instruction JSONL from P4-v2 fact rows."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_p4v2_d6_d7 import (  # noqa: E402
    attach_hard_negative,
    base_record,
    clean_facts,
    finding_text,
    make_laterality_cf,
    make_state_cf,
    make_uncertainty,
    make_yes_no,
    read_jsonl,
    write_csv,
    write_jsonl,
)


MIX_PRESETS = {
    "balanced": {"basic": 20, "location": 20, "uncertainty_answerability": 20, "cf": 25, "shuf": 15},
    "cf-heavy": {"basic": 10, "location": 15, "uncertainty_answerability": 15, "cf": 45, "shuf": 15},
    "shuf-heavy": {"basic": 10, "location": 15, "uncertainty_answerability": 10, "cf": 25, "shuf": 40},
    "clinical-rich": {"basic": 15, "location": 25, "uncertainty_answerability": 25, "cf": 20, "shuf": 15},
    "story": {"basic": 10, "location": 20, "uncertainty_answerability": 15, "cf": 30, "shuf": 25},
}


def weighted_slots(mix: dict[str, int], count: int, rng: random.Random) -> list[str]:
    names = list(mix)
    weights = [int(mix[name]) for name in names]
    slots = rng.choices(names, weights=weights, k=count)
    if count >= len(names):
        for name in names:
            if name not in slots and mix[name] > 0:
                slots[rng.randrange(count)] = name
    return slots


def add_mixture_metadata(row: dict[str, Any], source_version: str, group: str, slot: int) -> dict[str, Any]:
    out = dict(row)
    sid = str(out.get("sample_id"))
    out["instruction_id"] = f"{sid}_{source_version}_{group}_{slot:03d}"
    out["source_version"] = source_version
    out["mixture_group"] = group
    out["answer_short"] = out.get("answer_short") or out.get("answer")
    flags = list(out.get("quality_flags") or [])
    for flag in [source_version, f"mixture_{group}"]:
        if flag not in flags:
            flags.append(flag)
    out["quality_flags"] = flags
    metadata = dict(out.get("metadata") or {})
    metadata["mixture_group"] = group
    out["metadata"] = metadata
    return out


def make_location(row: dict[str, Any], fact: dict[str, Any], idx: int, source_version: str) -> dict[str, Any] | None:
    location = fact.get("location")
    if not location:
        return None
    rec = base_record(row, fact, idx, source_version, "laterality_location")
    rec["question"] = f"Where is the {finding_text(str(fact.get('finding')))} best localized on this chest X-ray?"
    rec["answer"] = str(location)
    rec["answer_short"] = str(location)
    rec["visual_dependency"] = "high"
    return rec


def make_evidence(row: dict[str, Any], fact: dict[str, Any], idx: int, source_version: str) -> dict[str, Any] | None:
    evidence = fact.get("evidence_span")
    if not evidence:
        return None
    rec = base_record(row, fact, idx, source_version, "evidence_phrase")
    rec["question"] = f"What visual observation supports {finding_text(str(fact.get('finding')))}?"
    rec["answer"] = str(evidence)
    rec["answer_short"] = str(evidence)
    rec["visual_dependency"] = fact.get("visual_dependency") or "medium"
    return rec


def make_answerability(row: dict[str, Any], fact: dict[str, Any], idx: int, source_version: str) -> dict[str, Any]:
    rec = base_record(row, fact, idx, source_version, "answerability")
    rec["question"] = f"Is {finding_text(str(fact.get('finding')))} answerable from this image-report pair?"
    rec["answer"] = "Yes"
    rec["answer_short"] = "Yes"
    rec["visual_dependency"] = "low"
    return rec


def choose_record(
    group: str,
    source: dict[str, Any],
    facts: list[dict[str, Any]],
    idx: int,
    source_version: str,
    answer_counter: Counter[str],
    rng: random.Random,
) -> dict[str, Any] | None:
    if not facts:
        return None
    shuffled = list(facts)
    rng.shuffle(shuffled)
    if group == "basic":
        return make_yes_no(source, shuffled[0], idx, source_version)
    if group == "location":
        for fact in shuffled:
            rec = make_location(source, fact, idx, source_version)
            if rec is not None:
                return rec
        return make_evidence(source, shuffled[0], idx, source_version)
    if group == "uncertainty_answerability":
        if rng.random() < 0.5:
            return make_uncertainty(source, shuffled[0], idx, source_version)
        return make_answerability(source, shuffled[0], idx, source_version)
    if group in {"cf", "shuf"}:
        for fact in shuffled:
            rec = make_laterality_cf(source, fact, idx, source_version, answer_counter)
            if rec is not None:
                return rec
            rec = make_state_cf(source, fact, idx, source_version, answer_counter)
            if rec is not None:
                return rec
    return make_yes_no(source, shuffled[0], idx, source_version)


def distribution(rows: list[dict[str, Any]], dataset_name: str) -> list[dict[str, Any]]:
    counters = {
        "mixture_group": Counter(str(row.get("mixture_group") or "") for row in rows),
        "answer_type": Counter(str(row.get("answer_type") or "") for row in rows),
        "answer": Counter(str(row.get("answer_short") or row.get("answer") or "") for row in rows),
        "hard_negative_reason": Counter(str(row.get("hard_negative_reason") or "none") for row in rows),
    }
    out: list[dict[str, Any]] = []
    for field, counter in counters.items():
        for value, count in counter.most_common():
            out.append({"dataset": dataset_name, "field": field, "value": value, "count": count})
    return out


def parse_mix(args: argparse.Namespace) -> dict[str, int]:
    if args.mix_json:
        payload = json.loads(args.mix_json)
        return {str(key): int(value) for key, value in payload.items()}
    return dict(MIX_PRESETS[args.mix])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--distribution-csv", type=Path)
    parser.add_argument("--mix", choices=sorted(MIX_PRESETS), default="story")
    parser.add_argument("--mix-json", help="Optional JSON mapping overriding --mix, e.g. '{\"basic\":10,...}'.")
    parser.add_argument("--source-version", default=None)
    parser.add_argument("--qa-per-image", type=int, default=8)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    facts_rows = read_jsonl(args.facts, max_samples=args.max_samples)
    mix = parse_mix(args)
    source_version = args.source_version or f"{args.mix}_qa{args.qa_per_image}"
    answer_counter: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for source in facts_rows:
        facts = clean_facts(source)
        slots = weighted_slots(mix, args.qa_per_image, rng)
        for slot_idx, group in enumerate(slots):
            rec = choose_record(group, source, facts, slot_idx, source_version, answer_counter, rng)
            if rec is None:
                continue
            rows.append(add_mixture_metadata(rec, source_version, group, slot_idx))

    with_negatives = attach_hard_negative(rows, facts_rows)
    final_rows: list[dict[str, Any]] = []
    for row in with_negatives:
        if row.get("mixture_group") != "shuf":
            row.pop("hard_negative_image_path", None)
            row.pop("hard_negative_sample_id", None)
            row.pop("hard_negative_reason", None)
            row.pop("hard_negative_expected_answer", None)
            row["quality_flags"] = [flag for flag in row.get("quality_flags", []) if flag != "hard_image_shuffle"]
        final_rows.append(row)

    write_jsonl(args.output, final_rows)
    if args.distribution_csv:
        write_csv(args.distribution_csv, distribution(final_rows, args.output.stem))
    print(
        json.dumps(
            {
                "facts": str(args.facts),
                "output": str(args.output),
                "records": len(final_rows),
                "mix": mix,
                "qa_per_image": args.qa_per_image,
                "images": len(facts_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
