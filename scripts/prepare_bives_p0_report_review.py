"""Create frozen-rule MIMIC report candidates and blinded P0 review packets.

The parser is intentionally conservative and produces review candidates only.
It must never be treated as a four-state labeler or as a source for a formal
BiVES manifest.  The emitted review CSV deliberately omits parser states and
matched report cues so clinical reviewers are blind to the heuristic proposal.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PARSER_VERSION = "bives_mimic_rule_candidates_v1"
REVIEW_FIELDS = (
    "candidate_id",
    "source_dataset",
    "patient_id",
    "study_id",
    "image_id",
    "image_path",
    "report_path",
    "canonical_statement_id",
    "statement_text",
    "review_track",
    "reviewer_1_id",
    "reviewer_1_state",
    "reviewer_1_notes",
    "reviewer_2_id",
    "reviewer_2_state",
    "reviewer_2_notes",
    "adjudicator_id",
    "adjudicated_state",
    "adjudication_notes",
)

# All text patterns are frozen into the provenance hash written with every
# candidate. They are a screening heuristic, never a clinical ground truth.
FINDING_RULES: dict[str, dict[str, Any]] = {
    "pleural_effusion": {
        "statement_text": "Pleural effusion is present.",
        "entities": (r"pleural effusions?", r"pleural fluid"),
    },
    "pneumothorax": {
        "statement_text": "Pneumothorax is present.",
        "entities": (r"pneumothorax",),
    },
    "cardiomegaly": {
        "statement_text": "Cardiomegaly is present.",
        "entities": (r"cardiomegaly", r"enlarged cardiac silhouette"),
    },
    "pulmonary_edema": {
        "statement_text": "Pulmonary edema is present.",
        "entities": (r"pulmonary edema", r"interstitial edema", r"vascular congestion"),
    },
    "atelectasis": {
        "statement_text": "Atelectasis is present.",
        "entities": (r"atelectasis", r"volume loss"),
    },
    "consolidation": {
        "statement_text": "Focal air-space consolidation is present.",
        "entities": (r"consolidation", r"airspace opacity", r"air-space opacity"),
    },
}
NEGATION = re.compile(r"\b(no|without|negative for|free of|absent)\b", flags=re.IGNORECASE)
UNCERTAINTY = re.compile(
    r"\b(possible|possibly|may represent|may reflect|cannot exclude|questionable|question of|suggestive of)\b",
    flags=re.IGNORECASE,
)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|[\r\n]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--parsed-output", type=Path, required=True)
    parser.add_argument("--review-packet", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--per-bucket",
        type=int,
        default=25,
        help="Maximum blinded review rows for each finding/state candidate bucket.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rules_sha256() -> str:
    payload = json.dumps(FINDING_RULES, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"candidate_id", "report_path", "image_path", "patient_id", "study_id", "image_id"}
            missing = required - set(row)
            if missing:
                raise ValueError(f"{path}:{line_number} missing candidate fields: {sorted(missing)}")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty candidate index: {path}")
    return rows


def _sentences(report: str) -> list[str]:
    return [item.strip() for item in SENTENCE_SPLIT.split(report) if item.strip()]


def _state_for_sentence(sentence: str, entities: tuple[str, ...]) -> tuple[str, str] | None:
    lowered = sentence.lower()
    entity = next((pattern for pattern in entities if re.search(pattern, lowered, flags=re.IGNORECASE)), None)
    if entity is None:
        return None
    if NEGATION.search(lowered):
        return "contradict", entity
    if UNCERTAINTY.search(lowered):
        return "uncertain", entity
    return "support", entity


def parse_candidate(row: dict[str, Any], rules_hash: str) -> list[dict[str, Any]]:
    report_path = Path(str(row["report_path"]))
    if not report_path.is_file():
        raise FileNotFoundError(f"candidate report is missing: {report_path}")
    report = report_path.read_text(encoding="utf-8", errors="replace")
    report_hash = hashlib.sha256(report.encode("utf-8")).hexdigest()
    parsed: list[dict[str, Any]] = []
    for finding, rule in FINDING_RULES.items():
        matches: list[tuple[str, str]] = []
        for sentence in _sentences(report):
            result = _state_for_sentence(sentence, tuple(rule["entities"]))
            if result is not None:
                matches.append(result)
        if not matches:
            continue
        states = sorted({state for state, _ in matches})
        status = "candidate" if len(states) == 1 else "requires_review_conflict"
        state = states[0] if len(states) == 1 else None
        cue = ";".join(sorted({cue for _, cue in matches}))
        parsed.append(
            {
                **{key: row[key] for key in ("candidate_id", "source_dataset", "patient_id", "study_id", "image_id", "image_path", "report_path") if key in row},
                "canonical_statement_id": finding,
                "statement_text": rule["statement_text"],
                "parser_version": PARSER_VERSION,
                "parser_rules_sha256": rules_hash,
                "report_sha256": report_hash,
                "parser_state_candidate": state,
                "parser_cue": cue,
                "parser_status": status,
                "labeling_claim": "none",
                "review_track": "p0_1_explicit_positive_negative" if state in {"support", "contradict"} else "p0_2_uncertain_insufficient",
            }
        )
    return parsed


def review_row(row: dict[str, Any]) -> dict[str, str]:
    # Parser state/cue/hash are intentionally absent from the reviewer packet.
    output = {field: "" for field in REVIEW_FIELDS}
    for field in (
        "candidate_id",
        "source_dataset",
        "patient_id",
        "study_id",
        "image_id",
        "image_path",
        "report_path",
        "canonical_statement_id",
        "statement_text",
        "review_track",
    ):
        output[field] = str(row.get(field, ""))
    return output


def main() -> None:
    args = parse_args()
    if args.per_bucket <= 0:
        raise ValueError("--per-bucket must be positive")
    source_rows = read_jsonl(args.candidates)
    rules_hash = rules_sha256()
    parsed_rows = [parsed for row in source_rows for parsed in parse_candidate(row, rules_hash)]
    parsed_rows.sort(key=lambda row: (str(row["canonical_statement_id"]), str(row["parser_state_candidate"]), str(row["candidate_id"])))
    args.parsed_output.parent.mkdir(parents=True, exist_ok=True)
    with args.parsed_output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in parsed_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    selected: list[dict[str, Any]] = []
    bucket_counts: dict[str, int] = defaultdict(int)
    for row in parsed_rows:
        bucket = f"{row['canonical_statement_id']}::{row.get('parser_state_candidate') or row['parser_status']}"
        if bucket_counts[bucket] >= args.per_bucket:
            continue
        bucket_counts[bucket] += 1
        selected.append(row)
    args.review_packet.parent.mkdir(parents=True, exist_ok=True)
    with args.review_packet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(review_row(row) for row in selected)

    summary = {
        "status": "review_required",
        "labeling_claim": "none",
        "parser_version": PARSER_VERSION,
        "parser_rules_sha256": rules_hash,
        "candidate_index_sha256": file_sha256(args.candidates),
        "input_candidates": len(source_rows),
        "parsed_candidates": len(parsed_rows),
        "review_packet_rows": len(selected),
        "parser_state_counts": dict(Counter(str(row.get("parser_state_candidate")) for row in parsed_rows)),
        "parser_status_counts": dict(Counter(str(row["parser_status"]) for row in parsed_rows)),
        "review_bucket_counts": dict(sorted(bucket_counts.items())),
        "parsed_output": str(args.parsed_output),
        "review_packet": str(args.review_packet),
        "reviewer_fields": list(REVIEW_FIELDS[10:]),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
