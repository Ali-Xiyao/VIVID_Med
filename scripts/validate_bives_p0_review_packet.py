"""Fail closed unless a BiVES-CXR P0 blinded review packet is complete."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bives_cxr.decoder import STATE_NAMES


REQUIRED_FIELDS = (
    "candidate_id",
    "reviewer_1_id",
    "reviewer_1_state",
    "reviewer_2_id",
    "reviewer_2_state",
    "adjudicator_id",
    "adjudicated_state",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-packet", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.review_packet.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = set(REQUIRED_FIELDS) - fields
        if missing:
            raise ValueError(f"review packet missing fields: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("review packet has no rows")

    errors: list[str] = []
    agreement = 0
    state_counts: Counter[str] = Counter()
    for line_number, row in enumerate(rows, start=2):
        values = {field: str(row.get(field, "")).strip() for field in REQUIRED_FIELDS}
        missing_values = [field for field, value in values.items() if not value]
        if missing_values:
            errors.append(f"line {line_number} missing values: {missing_values}")
            continue
        for field in ("reviewer_1_state", "reviewer_2_state", "adjudicated_state"):
            if values[field] not in STATE_NAMES:
                errors.append(f"line {line_number} invalid {field}: {values[field]!r}")
        if values["reviewer_1_id"] == values["reviewer_2_id"]:
            errors.append(f"line {line_number} uses the same reviewer twice")
        agreement += int(values["reviewer_1_state"] == values["reviewer_2_state"])
        state_counts[values["adjudicated_state"]] += 1
    report = {
        "status": "pass" if not errors else "fail",
        "rows": len(rows),
        "agreement_rate": agreement / len(rows),
        "adjudicated_state_counts": dict(state_counts),
        "error_count": len(errors),
        "errors_preview": errors[:50],
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
