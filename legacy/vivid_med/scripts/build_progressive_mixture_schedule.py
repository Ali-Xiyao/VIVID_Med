"""Build curriculum/progressive mixture schedules and optional materialized JSONL."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


PRESETS: dict[str, list[dict[str, Any]]] = {
    "cur-p3-cf-shuf": [
        {"stage": "p3_rich_qa", "start_step": 0, "end_step": 1600, "ratios": {"p3": 100}},
        {"stage": "hard_cf", "start_step": 1600, "end_step": 4000, "ratios": {"cf": 100}},
        {"stage": "hard_shuf", "start_step": 4000, "end_step": 8000, "ratios": {"shuf": 100}},
    ],
    "prog-mix": [
        {"stage": "clinical_warmup", "start_step": 0, "end_step": 2000, "ratios": {"basic": 35, "location": 25, "uncertainty": 20, "cf": 20, "shuf": 0}},
        {"stage": "cf_bridge", "start_step": 2000, "end_step": 5000, "ratios": {"basic": 20, "location": 20, "uncertainty": 15, "cf": 35, "shuf": 10}},
        {"stage": "shuf_grounding", "start_step": 5000, "end_step": 8000, "ratios": {"basic": 10, "location": 15, "uncertainty": 10, "cf": 30, "shuf": 35}},
    ],
    "prog-mix-tw": [
        {"stage": "clinical_warmup_tw", "start_step": 0, "end_step": 2000, "ratios": {"basic": 35, "location": 25, "uncertainty": 20, "cf": 20, "shuf": 0}, "token_weighting": "tw_visual"},
        {"stage": "cf_bridge_tw", "start_step": 2000, "end_step": 5000, "ratios": {"basic": 20, "location": 20, "uncertainty": 15, "cf": 35, "shuf": 10}, "token_weighting": "tw_visual"},
        {"stage": "shuf_grounding_tw", "start_step": 5000, "end_step": 8000, "ratios": {"basic": 10, "location": 15, "uncertainty": 10, "cf": 30, "shuf": 35}, "token_weighting": "tw_visual"},
    ],
    "progressive-hardneg": [
        {"stage": "random_negative", "start_step": 0, "end_step": 1600, "ratios": {"random": 100}},
        {"stage": "same_finding_opposite_state", "start_step": 1600, "end_step": 4000, "ratios": {"same_finding": 100}},
        {"stage": "same_finding_opposite_laterality", "start_step": 4000, "end_step": 6400, "ratios": {"laterality": 100}},
        {"stage": "mined_or_sameq", "start_step": 6400, "end_step": 8000, "ratios": {"mined": 60, "sameq": 40}},
    ],
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def write_md(path: Path, manifest: dict[str, Any]) -> None:
    lines = [f"# {manifest['run_id']} Curriculum Schedule", "", f"- preset: `{manifest['preset']}`", f"- total_steps: `{manifest['total_steps']}`", ""]
    lines.append("| stage | start_step | end_step | ratios | token_weighting |")
    lines.append("| --- | ---: | ---: | --- | --- |")
    for stage in manifest["stages"]:
        lines.append(
            "| {stage} | {start} | {end} | {ratios} | {tw} |".format(
                stage=stage["stage"],
                start=stage["start_step"],
                end=stage["end_step"],
                ratios=json.dumps(stage.get("ratios", {}), sort_keys=True),
                tw=stage.get("token_weighting", ""),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_stage_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--stage-input must use name=path")
    name, path = value.split("=", 1)
    return name.strip(), Path(path)


def sample_rows(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    shuffled = list(rows)
    rng.shuffle(shuffled)
    if count <= len(shuffled):
        return shuffled[:count]
    output = []
    while len(output) < count:
        cycle = list(rows)
        rng.shuffle(cycle)
        output.extend(cycle)
    return output[:count]


def materialize(stages: list[dict[str, Any]], stage_inputs: dict[str, Path], rows_per_stage: int | None, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    output: list[dict[str, Any]] = []
    source_cache = {key: read_jsonl(path) for key, path in stage_inputs.items()}
    for stage in stages:
        stage_name = str(stage["stage"])
        ratios = stage.get("ratios") or {}
        ratio_keys = list(ratios.keys())
        candidate_rows: list[dict[str, Any]] = []
        if ratios:
            limit = rows_per_stage or sum(len(source_cache.get(key, [])) for key in ratio_keys)
            total_ratio = sum(float(value) for value in ratios.values()) or 1.0
            allocated = 0
            for pos, key in enumerate(ratio_keys):
                if key not in source_cache:
                    continue
                if pos == len(ratio_keys) - 1:
                    count = max(0, limit - allocated)
                else:
                    count = int(round(limit * float(ratios[key]) / total_ratio))
                    allocated += count
                candidate_rows.extend(sample_rows(source_cache[key], count, rng))
        elif stage_name in source_cache:
            candidate_rows.extend(source_cache[stage_name])
        if not candidate_rows:
            continue
        rng.shuffle(candidate_rows)
        limit = rows_per_stage or len(candidate_rows)
        for index, row in enumerate(candidate_rows[:limit]):
            new = dict(row)
            new["curriculum_stage"] = stage_name
            new["curriculum_start_step"] = stage["start_step"]
            new["curriculum_end_step"] = stage["end_step"]
            new["curriculum_ratios"] = stage.get("ratios", {})
            new["instruction_id"] = f"{row.get('instruction_id', row.get('sample_id', index))}_{stage_name}_{index:06d}"
            flags = list(new.get("quality_flags") or [])
            if stage_name not in flags:
                flags.append(stage_name)
            new["quality_flags"] = flags
            output.append(new)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="prog-mix")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--materialized-output", type=Path)
    parser.add_argument("--stage-input", action="append", default=[], help="Stage data source as name=path.")
    parser.add_argument("--rows-per-stage", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stages = PRESETS[args.preset]
    manifest = {
        "run_id": args.run_id or args.preset.upper(),
        "preset": args.preset,
        "total_steps": max(int(stage["end_step"]) for stage in stages),
        "stages": stages,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.output_md:
        write_md(args.output_md, manifest)
    if args.materialized_output:
        stage_inputs = dict(parse_stage_input(item) for item in args.stage_input)
        rows = materialize(stages, stage_inputs, args.rows_per_stage, args.seed)
        write_jsonl(args.materialized_output, rows)
    print(json.dumps({"output_json": str(args.output_json), "preset": args.preset, "stages": len(stages)}, indent=2))


if __name__ == "__main__":
    main()
