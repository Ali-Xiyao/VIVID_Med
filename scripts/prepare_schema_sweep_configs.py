"""Prepare frozen-LM schema complexity sweep configs without launching training."""

from __future__ import annotations

import csv
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "schema_sweep"
FINAL_DIR = ROOT / "outputs" / "final_tables"

BASE_SOURCE = ROOT / "configs" / "ablation_A_ums_12label.yaml"
BASE_LP = ROOT / "configs" / "lp_A_ums_12label.yaml"
TRAIN_SPLIT = ROOT / "data" / "splits" / "chexpert_train_30k.jsonl"
VAL_SPLIT = ROOT / "data" / "splits" / "chexpert_val_fixed.jsonl"

SCHEMA_LEVELS = [
    ("S1", "state_only"),
    ("S2", "state_answerability"),
    ("S3", "state_uncertainty"),
]


def rel(path: Path) -> str:
    return "./" + path.relative_to(ROOT).as_posix()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping YAML in {path}")
    return loaded


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def patient_id(record: dict[str, Any]) -> str | None:
    original_path = str(record.get("extensions", {}).get("original_path", ""))
    match = re.search(r"patient(\d+)", original_path)
    if not match:
        return None
    return f"patient{match.group(1)}"


def jsonl_stats(path: Path) -> tuple[int, set[str]]:
    count = 0
    patients: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            count += 1
            pid = patient_id(json.loads(line))
            if pid is not None:
                patients.add(pid)
    return count, patients


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(row.get(column, "") for column in columns) + " |")
    return "\n".join([header, divider, *body]) + "\n"


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare_source_config(
    base: dict[str, Any],
    schema_level: str,
    schema_mode: str,
) -> tuple[Path, str]:
    config = deepcopy(base)
    output_name = f"frozen_lm_{schema_level.lower()}_{schema_mode}"
    config["data"]["train_ums_path"] = rel(TRAIN_SPLIT)
    config["data"]["val_ums_path"] = rel(VAL_SPLIT)
    config["data"]["max_val_samples"] = 1000
    config["data"]["num_workers"] = 0
    config["data"]["target_format"] = "json"
    config["data"]["schema_mode"] = schema_mode
    config["model"]["debug_llm_model_name"] = config["model"]["llm_model_name"]
    config["training"]["output_dir"] = f"./outputs/schema_sweep/{output_name}"
    path = CONFIG_DIR / f"{output_name}.yaml"
    write_yaml(path, config)
    return path, config["training"]["output_dir"]


def prepare_lp_config(
    base: dict[str, Any],
    schema_level: str,
    schema_mode: str,
    source_output_dir: str,
) -> tuple[Path, str]:
    config = deepcopy(base)
    output_name = f"lp_frozen_lm_{schema_level.lower()}_{schema_mode}"
    config["data"]["train_ums_path"] = rel(TRAIN_SPLIT)
    config["data"]["val_ums_path"] = rel(VAL_SPLIT)
    config["data"]["max_val_samples"] = 1000
    config["data"]["num_workers"] = 0
    config["transfer"]["init_vit_checkpoint"] = (
        f"{source_output_dir}/checkpoints/best.pt"
    )
    config["training"]["output_dir"] = f"./outputs/schema_sweep/{output_name}"
    path = CONFIG_DIR / f"{output_name}.yaml"
    write_yaml(path, config)
    return path, config["training"]["output_dir"]


def main() -> None:
    for path in [BASE_SOURCE, BASE_LP, TRAIN_SPLIT, VAL_SPLIT]:
        if not path.exists():
            raise FileNotFoundError(path)

    train_records, train_patients = jsonl_stats(TRAIN_SPLIT)
    val_records, val_patients = jsonl_stats(VAL_SPLIT)
    overlap = train_patients & val_patients
    if overlap:
        raise RuntimeError(f"Train/val patient overlap detected: {len(overlap)}")

    source_base = load_yaml(BASE_SOURCE)
    lp_base = load_yaml(BASE_LP)
    rows: list[dict[str, str]] = []

    for schema_level, schema_mode in SCHEMA_LEVELS:
        source_path, source_output = prepare_source_config(
            source_base, schema_level, schema_mode
        )
        lp_path, lp_output = prepare_lp_config(
            lp_base, schema_level, schema_mode, source_output
        )
        rows.append(
            {
                "schema_level": schema_level,
                "schema_mode": schema_mode,
                "source_config": rel(source_path),
                "lp_config": rel(lp_path),
                "train_path": rel(TRAIN_SPLIT),
                "train_records": str(train_records),
                "train_patients": str(len(train_patients)),
                "val_path": rel(VAL_SPLIT),
                "val_records": str(val_records),
                "val_patients": str(len(val_patients)),
                "train_val_patient_overlap": str(len(overlap)),
                "source_output_dir": source_output,
                "lp_output_dir": lp_output,
                "run_status": "config_ready_but_runtime_import_blocked",
                "boundary": "frozen-LM only; no-LM S2/S3 head not implemented",
            }
        )

    columns = [
        "schema_level",
        "schema_mode",
        "source_config",
        "lp_config",
        "train_path",
        "train_records",
        "train_patients",
        "val_path",
        "val_records",
        "val_patients",
        "train_val_patient_overlap",
        "source_output_dir",
        "lp_output_dir",
        "run_status",
        "boundary",
    ]
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(FINAL_DIR / "schema_sweep_config_manifest.csv", rows, columns)

    text = "# Schema Sweep Config Prep\n\n"
    text += "No training, model import, or GPU job was launched.\n\n"
    text += "## Patient Split Check\n\n"
    text += f"- train: `{rel(TRAIN_SPLIT)}` ({train_records} records, {len(train_patients)} patients)\n"
    text += f"- val: `{rel(VAL_SPLIT)}` ({val_records} records, {len(val_patients)} patients)\n"
    text += f"- train/val patient overlap: {len(overlap)}\n\n"
    text += "## Config Manifest\n\n"
    text += markdown_table(rows, columns)
    text += "\n## Run Order After Runtime Recovery\n\n"
    for row in rows:
        text += (
            f"1. Source: `python scripts/train_cxr.py --config "
            f"{row['source_config']}`; then LP: `python scripts/train_vit_baseline.py "
            f"--config {row['lp_config']}`.\n"
        )
    text += "\n## Boundary\n\n"
    text += (
        "- This prep only covers frozen-LM S1/S2/S3 schema serialization.\n"
        "- no-LM S2/S3 is intentionally excluded until a separate answerability/"
        "uncertainty target or head is implemented.\n"
        "- These configs use the strict fixed split, so future numbers should be "
        "reported separately from historical P0 metrics that used older train/val "
        "artifacts.\n"
    )
    (FINAL_DIR / "schema_sweep_config_prep.md").write_text(text, encoding="utf-8")
    print(f"Wrote {len(rows)} schema sweep source/LP config pairs.")
    print(f"Manifest: {rel(FINAL_DIR / 'schema_sweep_config_manifest.csv')}")


if __name__ == "__main__":
    main()
