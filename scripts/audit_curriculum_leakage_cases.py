"""Mine curriculum leakage and stage-failure cases from instruction audits."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from case_study_modules_common import FINAL_DIR, read_csv_rows, rel, write_csv_rows, write_md_table


COLUMNS = ["case_study", "case_id", "run_id", "sample_id", "question", "answer", "leakage_pattern", "fix", "source"]


def select_curriculum_audit_rows(limit: int) -> list[dict[str, Any]]:
    audit_dir = FINAL_DIR / "next_stage_audits"
    rows: list[dict[str, Any]] = []
    for path in sorted(audit_dir.glob("*leakage_detail.csv")):
        name = path.name
        if not any(token in name for token in ["cur_", "prog_mix", "progressive_hardneg"]):
            continue
        for row in read_csv_rows(path):
            flags = " ".join(str(row.get(key, "")) for key in row)
            if "flag" not in flags.lower() and "leak" not in flags.lower() and "duplicate" not in flags.lower():
                continue
            rows.append(
                {
                    "case_study": "CS4_curriculum_leakage",
                    "case_id": len(rows) + 1,
                    "run_id": name.replace("_leakage_detail.csv", ""),
                    "sample_id": row.get("sample_id", ""),
                    "question": row.get("question", "")[:220],
                    "answer": row.get("answer", row.get("answer_short", "")),
                    "leakage_pattern": row.get("flags", row.get("flag_reasons", "audit_flag")),
                    "fix": "deduplicate per image/question, reduce repeated stage replay, and rerun leakage audit before training",
                    "source": rel(path),
                }
            )
            if len(rows) >= limit:
                return rows
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--output-csv", type=Path, default=FINAL_DIR / "case_study_curriculum_failure.csv")
    parser.add_argument("--output-md", type=Path, default=FINAL_DIR / "case_study_curriculum_failure.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = select_curriculum_audit_rows(args.limit)
    note = "Rows are mined from leakage audit details for curriculum/progressive runs; they are review candidates, not final radiologist labels."
    write_csv_rows(args.output_csv, rows, COLUMNS)
    write_md_table(args.output_md, "Curriculum Leakage / Failure Case Study", rows, COLUMNS, note)


if __name__ == "__main__":
    main()

