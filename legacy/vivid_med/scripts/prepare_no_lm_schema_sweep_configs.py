"""Prepare no-LM schema complexity configs with explicit auxiliary heads."""

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

BASE_CONFIG = ROOT / "configs" / "ums_classifier_no_llm_12label.yaml"
BASE_LP_CONFIG = ROOT / "configs" / "lp_ums_classifier_no_llm_12label.yaml"
TRAIN_SPLIT = ROOT / "data" / "splits" / "chexpert_train_30k.jsonl"
VAL_SPLIT = ROOT / "data" / "splits" / "chexpert_val_fixed.jsonl"

SCHEMA_LEVELS = [
    ("S1", "state_only", []),
    ("S2", "state_answerability", ["answerability"]),
    ("S3", "state_uncertainty", ["uncertainty"]),
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
    body = ["| " + " | ".join(row.get(column, "") for column in columns) + " |" for row in rows]
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
    auxiliary_targets: list[str],
) -> tuple[Path, str]:
    config = deepcopy(base)
    output_name = f"no_lm_{schema_level.lower()}_{schema_mode}"
    config["data"]["train_ums_path"] = rel(TRAIN_SPLIT)
    config["data"]["val_ums_path"] = rel(VAL_SPLIT)
    config["data"]["max_val_samples"] = 1000
    config["data"]["num_workers"] = 0
    config["data"]["schema_mode"] = schema_mode
    config["model"]["schema_auxiliary_targets"] = auxiliary_targets
    config["training"]["output_dir"] = f"./outputs/schema_sweep/{output_name}"
    config["training"]["schema_auxiliary_loss_weights"] = {
        target: 1.0 for target in auxiliary_targets
    }
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
    output_name = f"lp_no_lm_{schema_level.lower()}_{schema_mode}"
    config["data"]["train_ums_path"] = rel(TRAIN_SPLIT)
    config["data"]["val_ums_path"] = rel(VAL_SPLIT)
    config["data"]["max_val_samples"] = 1000
    config["data"]["num_workers"] = 0
    config["transfer"]["init_vit_checkpoint"] = f"{source_output_dir}/best.pt"
    config["training"]["output_dir"] = f"./outputs/schema_sweep/{output_name}"
    path = CONFIG_DIR / f"{output_name}.yaml"
    write_yaml(path, config)
    return path, config["training"]["output_dir"]


def main() -> None:
    for path in [BASE_CONFIG, BASE_LP_CONFIG, TRAIN_SPLIT, VAL_SPLIT]:
        if not path.exists():
            raise FileNotFoundError(path)

    train_records, train_patients = jsonl_stats(TRAIN_SPLIT)
    val_records, val_patients = jsonl_stats(VAL_SPLIT)
    overlap = train_patients & val_patients
    if overlap:
        raise RuntimeError(f"Train/val patient overlap detected: {len(overlap)}")

    source_base = load_yaml(BASE_CONFIG)
    lp_base = load_yaml(BASE_LP_CONFIG)
    rows: list[dict[str, str]] = []
    for schema_level, schema_mode, auxiliary_targets in SCHEMA_LEVELS:
        source_path, source_output_dir = prepare_source_config(
            base=source_base,
            schema_level=schema_level,
            schema_mode=schema_mode,
            auxiliary_targets=auxiliary_targets,
        )
        lp_path, lp_output_dir = prepare_lp_config(
            base=lp_base,
            schema_level=schema_level,
            schema_mode=schema_mode,
            source_output_dir=source_output_dir,
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
                "auxiliary_targets": ",".join(auxiliary_targets) or "none",
                "source_output_dir": source_output_dir,
                "lp_output_dir": lp_output_dir,
                "run_status": "config_ready_not_formal_run",
                "boundary": "no-LM explicit-head comparator config; debug/formal runs must be recorded separately",
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
        "auxiliary_targets",
        "source_output_dir",
        "lp_output_dir",
        "run_status",
        "boundary",
    ]
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(FINAL_DIR / "no_lm_schema_sweep_config_manifest.csv", rows, columns)

    text = "# no-LM Schema Sweep Config Prep\n\n"
    text += "No training, model import, or GPU job was launched.\n\n"
    text += "## Patient Split Check\n\n"
    text += f"- train: `{rel(TRAIN_SPLIT)}` ({train_records} records, {len(train_patients)} patients)\n"
    text += f"- val: `{rel(VAL_SPLIT)}` ({val_records} records, {len(val_patients)} patients)\n"
    text += f"- train/val patient overlap: {len(overlap)}\n\n"
    text += "## Config Manifest\n\n"
    text += markdown_table(rows, columns)
    text += "\n## Run Order\n\n"
    for row in rows:
        text += (
            f"1. Source: `python scripts/train_ums_classifier.py --config {row['source_config']}`; "
            f"then LP: `python scripts/train_vit_baseline.py --config {row['lp_config']}`.\n"
        )
    text += "\n## Boundary\n\n"
    text += (
        "- S1 is the original no-LM 4-state classifier.\n"
        "- S2 adds an explicit answerability head and loss.\n"
        "- S3 adds an explicit uncertainty head and loss.\n"
        "- These configs are no-LM only and do not add SPD or any frozen-LM variant.\n"
    )
    (FINAL_DIR / "no_lm_schema_sweep_config_prep.md").write_text(text, encoding="utf-8")
    print(f"Wrote {len(rows)} no-LM schema configs.")
    print(f"Manifest: {rel(FINAL_DIR / 'no_lm_schema_sweep_config_manifest.csv')}")


if __name__ == "__main__":
    main()
