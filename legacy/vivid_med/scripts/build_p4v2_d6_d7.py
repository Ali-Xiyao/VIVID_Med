"""Build standardized P4-v2 D6/D7 instruction JSONL from extracted facts."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STATE_ORDER = {"present": 0, "absent": 1, "uncertain": 2}
OPPOSITE_LATERALITY = {"left": "right", "right": "left"}


def read_jsonl(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if max_samples is not None and len(rows) >= max_samples:
                break
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
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finding_text(finding: str) -> str:
    return str(finding).replace("_", " ").lower()


def fact_sort_key(fact: dict[str, Any]) -> tuple[int, str, str]:
    visual_rank = {"high": 0, "medium": 1, "low": 2}
    return (
        visual_rank.get(str(fact.get("visual_dependency") or "medium"), 1),
        str(fact.get("finding") or ""),
        str(fact.get("evidence_span") or ""),
    )


def clean_facts(row: dict[str, Any]) -> list[dict[str, Any]]:
    facts = [fact for fact in row.get("facts") or [] if fact.get("finding") and fact.get("state")]
    return sorted(facts, key=fact_sort_key)


def statement(fact: dict[str, Any], *, state: str | None = None, location: str | None = None, severity: str | None = None) -> str:
    finding = finding_text(fact["finding"])
    state = state or str(fact.get("state") or "present")
    location = location if location is not None else fact.get("location")
    severity = severity if severity is not None else fact.get("severity")
    parts = []
    if severity:
        parts.append(str(severity))
    if location:
        parts.append(str(location))
    parts.append(finding)
    finding_phrase = " ".join(parts)
    if state == "absent":
        return f"There is no {finding}."
    if state == "uncertain":
        return f"The presence of {finding_phrase} is uncertain."
    return f"There is {finding_phrase}."


def choose_answer(a_is_true: bool) -> str:
    return "A" if a_is_true else "B"


def ab_question(true_statement: str, false_statement: str, make_true_a: bool) -> tuple[str, str, str, str]:
    if make_true_a:
        a_text, b_text = true_statement, false_statement
        answer = "A"
    else:
        a_text, b_text = false_statement, true_statement
        answer = "B"
    question = f"Which statement is better supported by the chest X-ray?\nA. {a_text}\nB. {b_text}"
    return question, answer, a_text, b_text


def base_record(row: dict[str, Any], fact: dict[str, Any], idx: int, source_version: str, answer_type: str) -> dict[str, Any]:
    sid = str(row.get("sample_id"))
    return {
        "instruction_id": f"{sid}_{source_version}_{idx:03d}",
        "sample_id": sid,
        "image_path": row.get("image_path"),
        "report": row.get("report"),
        "question": "",
        "answer": "",
        "answer_short": "",
        "finding": fact.get("finding"),
        "state": fact.get("state"),
        "answer_type": answer_type,
        "evidence_span": fact.get("evidence_span"),
        "location": fact.get("location"),
        "laterality": fact.get("location") if fact.get("location") in {"left", "right", "bilateral"} else None,
        "severity": fact.get("severity"),
        "certainty": fact.get("certainty"),
        "visual_dependency": fact.get("visual_dependency") or "medium",
        "counterfactual_type": None,
        "negative_option_source": None,
        "source_version": source_version,
        "source_mode": "p4v2_fact_programmatic",
        "quality_flags": [source_version, "p4v2", "standardized_ab" if answer_type == "counterfactual_choice" else "non_ab"],
        "metadata": {
            "fact_source": row.get("source"),
            "fact_model": row.get("model"),
        },
    }


def target_true_a(counter: Counter[str]) -> bool:
    return counter["A"] <= counter["B"]


def make_state_cf(row: dict[str, Any], fact: dict[str, Any], idx: int, source_version: str, answer_counter: Counter[str]) -> dict[str, Any] | None:
    state = str(fact.get("state") or "")
    if state == "present":
        true_statement = statement(fact, state="present")
        false_statement = statement(fact, state="absent")
        cf_type = "state_flip_present_absent"
    elif state == "absent":
        true_statement = statement(fact, state="absent")
        false_statement = statement(fact, state="present")
        cf_type = "state_flip_absent_present"
    elif state == "uncertain":
        true_statement = statement(fact, state="uncertain")
        false_statement = statement(fact, state="present")
        cf_type = "uncertainty_flip"
    else:
        return None
    true_a = target_true_a(answer_counter)
    question, answer, a_text, b_text = ab_question(true_statement, false_statement, true_a)
    answer_counter[answer] += 1
    rec = base_record(row, fact, idx, source_version, "counterfactual_choice")
    rec.update(
        {
            "question": question,
            "answer": answer,
            "answer_short": answer,
            "counterfactual_type": cf_type,
            "positive_option": answer,
            "negative_option": "B" if answer == "A" else "A",
            "option_a": a_text,
            "option_b": b_text,
            "negative_option_source": "state_flip",
        }
    )
    return rec


def make_laterality_cf(row: dict[str, Any], fact: dict[str, Any], idx: int, source_version: str, answer_counter: Counter[str]) -> dict[str, Any] | None:
    location = fact.get("location")
    opposite = OPPOSITE_LATERALITY.get(str(location))
    if not opposite:
        return None
    true_statement = statement(fact, location=str(location))
    false_statement = statement(fact, location=opposite)
    true_a = target_true_a(answer_counter)
    question, answer, a_text, b_text = ab_question(true_statement, false_statement, true_a)
    answer_counter[answer] += 1
    rec = base_record(row, fact, idx, source_version, "counterfactual_choice")
    rec.update(
        {
            "question": question,
            "answer": answer,
            "answer_short": answer,
            "counterfactual_type": "laterality_flip",
            "positive_option": answer,
            "negative_option": "B" if answer == "A" else "A",
            "option_a": a_text,
            "option_b": b_text,
            "negative_option_source": "laterality_flip",
        }
    )
    return rec


def make_yes_no(row: dict[str, Any], fact: dict[str, Any], idx: int, source_version: str) -> dict[str, Any]:
    rec = base_record(row, fact, idx, source_version, "finding_verification")
    state = str(fact.get("state") or "")
    rec["question"] = f"Does this chest X-ray support {finding_text(str(fact.get('finding')))}?"
    if state == "present":
        rec["answer"] = "Yes"
    elif state == "absent":
        rec["answer"] = "No"
    else:
        rec["answer"] = "Uncertain"
    rec["answer_short"] = rec["answer"]
    return rec


def make_uncertainty(row: dict[str, Any], fact: dict[str, Any], idx: int, source_version: str) -> dict[str, Any]:
    rec = base_record(row, fact, idx, source_version, "uncertainty")
    rec["question"] = f"Is {finding_text(str(fact.get('finding')))} definite, uncertain, or not answerable on this chest X-ray?"
    rec["answer"] = "uncertain" if fact.get("state") == "uncertain" or fact.get("certainty") == "uncertain" else "definite"
    rec["answer_short"] = rec["answer"]
    return rec


def build_d6_rows(facts_rows: list[dict[str, Any]], source_version: str, qa_per_image: int, seed: int) -> list[dict[str, Any]]:
    random.seed(seed)
    rows: list[dict[str, Any]] = []
    answer_counter: Counter[str] = Counter()
    for source in facts_rows:
        facts = clean_facts(source)
        if not facts:
            continue
        sample_records: list[dict[str, Any]] = []
        for fact in facts:
            if len(sample_records) >= qa_per_image:
                break
            laterality = make_laterality_cf(source, fact, len(sample_records), source_version, answer_counter)
            if laterality is not None:
                sample_records.append(laterality)
            if len(sample_records) >= qa_per_image:
                break
            state_cf = make_state_cf(source, fact, len(sample_records), source_version, answer_counter)
            if state_cf is not None:
                sample_records.append(state_cf)
        fact_idx = 0
        while len(sample_records) < qa_per_image and fact_idx < len(facts):
            sample_records.append(make_yes_no(source, facts[fact_idx], len(sample_records), source_version))
            if len(sample_records) < qa_per_image:
                sample_records.append(make_uncertainty(source, facts[fact_idx], len(sample_records), source_version))
            fact_idx += 1
        rows.extend(sample_records[:qa_per_image])
    return rows


def build_negative_index(facts_rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in facts_rows:
        for fact in clean_facts(row):
            index[(str(fact.get("finding")), str(fact.get("state")))].append({"row": row, "fact": fact})
            location = fact.get("location")
            if location:
                index[(str(fact.get("finding")), f"location:{location}")].append({"row": row, "fact": fact})
    return index


def attach_hard_negative(rows: list[dict[str, Any]], facts_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = build_negative_index(facts_rows)
    output: list[dict[str, Any]] = []
    for rec in rows:
        new = dict(rec)
        finding = str(rec.get("finding"))
        state = str(rec.get("state"))
        location = rec.get("location")
        candidates: list[tuple[str, list[dict[str, Any]]]] = []
        if location in OPPOSITE_LATERALITY:
            candidates.append((f"same_finding_opposite_laterality", index.get((finding, f"location:{OPPOSITE_LATERALITY[str(location)]}"), [])))
        opposite_state = {"present": "absent", "absent": "present", "uncertain": "present"}.get(state)
        if opposite_state:
            candidates.append(("same_finding_opposite_state", index.get((finding, opposite_state), [])))
        chosen = None
        reason = ""
        for candidate_reason, values in candidates:
            for item in values:
                if str(item["row"].get("sample_id")) != str(rec.get("sample_id")):
                    chosen = item
                    reason = candidate_reason
                    break
            if chosen is not None:
                break
        if chosen is not None:
            new["hard_negative_image_path"] = chosen["row"].get("image_path")
            new["hard_negative_sample_id"] = chosen["row"].get("sample_id")
            new["hard_negative_reason"] = reason
            new["hard_negative_expected_answer"] = rec.get("negative_option") or ("No" if rec.get("answer") == "Yes" else "Yes")
            flags = list(new.get("quality_flags") or [])
            if "hard_image_shuffle" not in flags:
                flags.append("hard_image_shuffle")
            new["quality_flags"] = flags
        else:
            new["hard_negative_image_path"] = ""
            new["hard_negative_sample_id"] = ""
            new["hard_negative_reason"] = "no_candidate"
            new["hard_negative_expected_answer"] = ""
        output.append(new)
    return output


def distribution_rows(rows: list[dict[str, Any]], dataset: str) -> list[dict[str, Any]]:
    counters = {
        "answer_type": Counter(str(row.get("answer_type")) for row in rows),
        "counterfactual_type": Counter(str(row.get("counterfactual_type") or "none") for row in rows),
        "answer_short": Counter(str(row.get("answer_short") or row.get("answer")) for row in rows),
        "finding": Counter(str(row.get("finding")) for row in rows),
        "hard_negative_reason": Counter(str(row.get("hard_negative_reason") or "none") for row in rows),
    }
    out: list[dict[str, Any]] = []
    for field, counter in counters.items():
        for value, count in counter.most_common():
            out.append({"dataset": dataset, "field": field, "value": value, "count": count})
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", required=True, type=Path)
    parser.add_argument("--d6-output", required=True, type=Path)
    parser.add_argument("--d7-output", type=Path)
    parser.add_argument("--distribution-csv", type=Path)
    parser.add_argument("--source-version", default="d6_hard_cf")
    parser.add_argument("--qa-per-image", type=int, default=5)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    facts_rows = read_jsonl(args.facts, max_samples=args.max_samples)
    d6_rows = build_d6_rows(facts_rows, source_version=args.source_version, qa_per_image=args.qa_per_image, seed=args.seed)
    write_jsonl(args.d6_output, d6_rows)
    output_summary = {
        "facts": str(args.facts),
        "d6_output": str(args.d6_output),
        "fact_samples": len(facts_rows),
        "d6_records": len(d6_rows),
    }
    all_distribution_rows = distribution_rows(d6_rows, args.d6_output.stem)
    if args.d7_output:
        d7_rows = attach_hard_negative(d6_rows, facts_rows)
        write_jsonl(args.d7_output, d7_rows)
        output_summary["d7_output"] = str(args.d7_output)
        output_summary["d7_records"] = len(d7_rows)
        output_summary["d7_hard_negative_attached"] = sum(1 for row in d7_rows if row.get("hard_negative_image_path"))
        all_distribution_rows.extend(distribution_rows(d7_rows, args.d7_output.stem))
    if args.distribution_csv:
        write_csv(args.distribution_csv, all_distribution_rows)
    print(json.dumps(output_summary, indent=2))


if __name__ == "__main__":
    main()
