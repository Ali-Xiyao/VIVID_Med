"""Build refined curriculum-v2 progressive replay schedules."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from case_study_modules_common import FINAL_DIR, write_csv_rows, write_json, write_md_table


DEFAULT_STAGES = [
    {"stage": "s1_basic", "start_pct": 0.0, "end_pct": 0.2, "basic_qa": 35, "location": 25, "uncertainty": 20, "cf": 20, "shuf": 0, "sameq": 0, "replay": 0},
    {"stage": "s2_cf_intro", "start_pct": 0.2, "end_pct": 0.5, "basic_qa": 20, "location": 20, "uncertainty": 15, "cf": 35, "shuf": 10, "sameq": 0, "replay": 20},
    {"stage": "s3_grounding", "start_pct": 0.5, "end_pct": 0.8, "basic_qa": 10, "location": 15, "uncertainty": 10, "cf": 30, "shuf": 30, "sameq": 5, "replay": 15},
    {"stage": "s4_sameq_final", "start_pct": 0.8, "end_pct": 1.0, "basic_qa": 5, "location": 10, "uncertainty": 5, "cf": 20, "shuf": 35, "sameq": 25, "replay": 10},
]


def materialize(stages: list[dict[str, Any]], max_steps: int) -> list[dict[str, Any]]:
    rows = []
    for item in stages:
        out = dict(item)
        out["start_step"] = int(round(float(out["start_pct"]) * max_steps))
        out["end_step"] = int(round(float(out["end_pct"]) * max_steps))
        rows.append(out)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=12000)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/next_stage_manifests/curriculum_v2_progressive_replay_schedule.json"))
    parser.add_argument("--output-csv", type=Path, default=FINAL_DIR / "curriculum_v2_schedule.csv")
    parser.add_argument("--output-md", type=Path, default=FINAL_DIR / "curriculum_v2_schedule.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = materialize(DEFAULT_STAGES, args.max_steps)
    write_json(args.output_json, {"max_steps": args.max_steps, "stages": rows})
    columns = ["stage", "start_step", "end_step", "basic_qa", "location", "uncertainty", "cf", "shuf", "sameq", "replay"]
    write_csv_rows(args.output_csv, rows, columns)
    write_md_table(args.output_md, "Curriculum v2 Progressive Replay Schedule", rows, columns)


if __name__ == "__main__":
    main()

