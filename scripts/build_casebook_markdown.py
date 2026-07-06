"""Build combined case-study markdown and CSV artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from case_study_modules_common import FINAL_DIR, read_csv_rows, write_csv_rows, write_md_sections, write_md_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="*",
        type=Path,
        default=[
            FINAL_DIR / "case_study_shuf_tw_vs_shuf.csv",
            FINAL_DIR / "case_study_nih_transfer.csv",
            FINAL_DIR / "case_study_curriculum_failure.csv",
            FINAL_DIR / "case_study_hard_negative_quality.csv",
        ],
    )
    parser.add_argument("--output-csv", type=Path, default=FINAL_DIR / "case_study_summary.csv")
    parser.add_argument("--output-md", type=Path, default=FINAL_DIR / "case_study_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combined = []
    sections = []
    for path in args.inputs:
        rows = read_csv_rows(path)
        combined.extend(rows)
        if rows:
            preview = rows[:12]
            columns = list(preview[0].keys())
            lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
            for row in preview:
                lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
            sections.append((path.stem, "\n".join(lines)))
        else:
            sections.append((path.stem, "No mined rows. Check source artifact availability or run the relevant audit script."))
    write_csv_rows(args.output_csv, combined)
    write_md_sections(args.output_md, "Case Study Summary", sections)
    # Also refresh a compact table when downstream tooling expects a simple markdown table.
    write_md_table(FINAL_DIR / "case_study_summary_table.md", "Case Study Summary Table", combined[:40])


if __name__ == "__main__":
    main()
