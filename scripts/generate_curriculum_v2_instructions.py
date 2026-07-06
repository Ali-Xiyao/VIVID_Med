"""Materialize curriculum-v2 instruction JSONL from existing instruction pools."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from case_study_modules_common import read_json, read_jsonl, write_jsonl, write_json, write_md_table


def sample_rows(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if not rows or count <= 0:
        return []
    if count <= len(rows):
        return rng.sample(rows, count)
    return [rng.choice(rows) for _ in range(count)]


def tag_rows(rows: list[dict[str, Any]], stage: dict[str, Any], source: str, weight: float) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        new = dict(row)
        new["curriculum_stage"] = stage["stage"]
        new["curriculum_source"] = source
        new["curriculum_start_step"] = stage["start_step"]
        new["curriculum_end_step"] = stage["end_step"]
        new["curriculum_sampling_weight"] = weight
        output.append(new)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, default=Path("outputs/next_stage_manifests/curriculum_v2_progressive_replay_schedule.json"))
    parser.add_argument("--qa", type=Path, default=Path("outputs/instruction_data/next_stage/storymix_qa8_train.jsonl"))
    parser.add_argument("--cf", type=Path, default=Path("outputs/instruction_data/glm_validated/d6_hard_cf_3k.jsonl"))
    parser.add_argument("--shuf", type=Path, default=Path("outputs/instruction_data/glm_validated/d7_hard_shuffle_3k.jsonl"))
    parser.add_argument("--sameq", type=Path, default=Path("outputs/instruction_data/next_stage/sameq_shuf_3k_train.jsonl"))
    parser.add_argument("--rows-per-stage", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/instruction_data/next_stage/cur_v2_progressive_replay_train.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/next_stage_manifests/curriculum_v2_generation_manifest.json"))
    parser.add_argument("--report-md", type=Path, default=Path("outputs/final_tables/curriculum_v2_generation.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    schedule = read_json(args.schedule) or {"stages": []}
    pools = {"qa": read_jsonl(args.qa), "cf": read_jsonl(args.cf), "shuf": read_jsonl(args.shuf), "sameq": read_jsonl(args.sameq)}
    output: list[dict[str, Any]] = []
    summary = []
    for stage in schedule.get("stages", []):
        weights = {
            "qa": float(stage.get("basic_qa", 0)) + float(stage.get("location", 0)) + float(stage.get("uncertainty", 0)),
            "cf": float(stage.get("cf", 0)),
            "shuf": float(stage.get("shuf", 0)),
            "sameq": float(stage.get("sameq", 0)),
        }
        total = sum(weights.values()) or 1.0
        stage_count = 0
        for name, weight in weights.items():
            count = int(round(args.rows_per_stage * weight / total))
            selected = sample_rows(pools[name], count, rng)
            output.extend(tag_rows(selected, stage, name, weight / total))
            stage_count += len(selected)
        summary.append({"stage": stage["stage"], "rows": stage_count, **{f"{key}_pool": len(value) for key, value in pools.items()}})
    write_jsonl(args.output, output)
    write_json(args.manifest, {"output": args.output.as_posix(), "rows": len(output), "summary": summary})
    write_md_table(args.report_md, "Curriculum v2 Generation", summary)


if __name__ == "__main__":
    main()

