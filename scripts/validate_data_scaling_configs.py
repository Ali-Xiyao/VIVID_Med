"""Validate generated P1 data scaling configs without launching training."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "data_scaling"
FINAL_DIR = ROOT / "outputs" / "final_tables"
JSONL_META_CACHE: dict[Path, tuple[int, set[str]]] = {}


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resolve(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path_text
    return path


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not parse to a YAML mapping")
    return payload


def read_jsonl_meta(path: Path | None) -> tuple[int, set[str]] | None:
    if path is None or not path.exists():
        return None
    resolved = path.resolve()
    if resolved in JSONL_META_CACHE:
        return JSONL_META_CACHE[resolved]
    count = 0
    patients: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
                patient = patient_id_from_record(json.loads(line))
                if patient is not None:
                    patients.add(patient)
    JSONL_META_CACHE[resolved] = (count, patients)
    return count, patients


def patient_id_from_record(record: dict[str, Any]) -> str | None:
    original_path = str(record.get("extensions", {}).get("original_path", ""))
    match = re.search(r"patient(\d+)", original_path)
    if not match:
        return None
    return f"patient{match.group(1)}"


def patients_jsonl(path: Path | None) -> set[str] | None:
    meta = read_jsonl_meta(path)
    if meta is None:
        return None
    return meta[1]


def count_jsonl(path: Path | None) -> int | None:
    meta = read_jsonl_meta(path)
    if meta is None:
        return None
    return meta[0]


def command_for(config: Path) -> str:
    name = config.name
    if name.startswith("no_lm_ums_"):
        return f"python scripts/train_ums_classifier.py --config {rel(config)}"
    return f"python scripts/train_vit_baseline.py --config {rel(config)}" if name.startswith(("bce_", "lp_")) else f"python scripts/train_cxr.py --config {rel(config)}"


def is_lp(name: str) -> bool:
    return name.startswith("lp_")


def method_group(name: str) -> str:
    return name.removesuffix(".yaml")


def size_from_name(name: str) -> str:
    stem = name.removesuffix(".yaml")
    for size in ["1k", "3k", "10k", "30k"]:
        if stem.endswith(f"_{size}") or stem == f"bce_{size}":
            return size
    return ""


def validate_config(path: Path) -> dict[str, str]:
    row = {
        "config": rel(path),
        "method": method_group(path.name),
        "size": size_from_name(path.name),
        "command": command_for(path),
        "train_ums_path": "",
        "train_records": "",
        "train_patients": "",
        "val_ums_path": "",
        "val_records": "",
        "val_patients": "",
        "train_val_patient_overlap": "",
        "output_dir": "",
        "output_status": "",
        "checkpoint_dependency": "",
        "status": "ok",
        "notes": "",
    }
    notes: list[str] = []

    try:
        config = read_yaml(path)
    except Exception as exc:  # noqa: BLE001 - validation report should capture any parse error.
        row["status"] = "fail"
        row["notes"] = f"yaml_parse_error: {exc}"
        return row

    data = config.get("data", {})
    training = config.get("training", {})
    transfer = config.get("transfer", {})
    train_path = resolve(data.get("train_ums_path"))
    val_path = resolve(data.get("val_ums_path"))
    output_dir = resolve(training.get("output_dir"))
    row["train_ums_path"] = rel(train_path) if train_path else ""
    row["val_ums_path"] = rel(val_path) if val_path else ""
    row["output_dir"] = rel(output_dir) if output_dir else ""

    train_count = count_jsonl(train_path)
    val_count = count_jsonl(val_path)
    train_patients = patients_jsonl(train_path)
    val_patients = patients_jsonl(val_path)
    row["train_records"] = str(train_count) if train_count is not None else ""
    row["val_records"] = str(val_count) if val_count is not None else ""
    row["train_patients"] = str(len(train_patients)) if train_patients is not None else ""
    row["val_patients"] = str(len(val_patients)) if val_patients is not None else ""

    if train_count is None:
        row["status"] = "fail"
        notes.append("train_ums_path missing or unreadable")
    if val_count is None:
        row["status"] = "fail"
        notes.append("val_ums_path missing or unreadable")
    if train_patients is not None and val_patients is not None:
        overlap = train_patients & val_patients
        row["train_val_patient_overlap"] = str(len(overlap))
        if overlap:
            row["status"] = "fail"
            notes.append(f"train/val patient overlap detected: {len(overlap)}")

    if output_dir is None:
        row["status"] = "fail"
        notes.append("training.output_dir missing")
    elif (output_dir / "metrics_final.json").exists():
        row["status"] = "fail"
        row["output_status"] = "completed_run_exists"
        notes.append("would overwrite completed run")
    elif output_dir.exists():
        row["output_status"] = "directory_exists_no_metrics"
    else:
        row["output_status"] = "available"

    if is_lp(path.name):
        ckpt = resolve(transfer.get("init_vit_checkpoint"))
        row["checkpoint_dependency"] = rel(ckpt) if ckpt else ""
        if ckpt is None:
            row["status"] = "fail"
            notes.append("LP config missing transfer.init_vit_checkpoint")
        elif ckpt.exists():
            notes.append("LP source checkpoint already exists")
        else:
            row["status"] = "blocked_until_source_run" if row["status"] == "ok" else row["status"]
            notes.append("LP waits for source training checkpoint")

    row["notes"] = "; ".join(notes)
    return row


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = [row.get(column, "").replace("\n", " ") for column in columns]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body]) + "\n"


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_order(rows: list[dict[str, str]]) -> str:
    text = "# Data Scaling Run Order\n\n"
    text += "Run source models before their LP configs. Do not run all sizes at once.\n\n"
    for size in ["1k", "3k", "10k", "30k"]:
        subset = [row for row in rows if row["size"] == size]
        source = [row for row in subset if not row["method"].startswith("lp_")]
        lp_rows = [row for row in subset if row["method"].startswith("lp_")]
        text += f"## {size}\n\n"
        text += "Source runs:\n\n"
        for row in source:
            text += f"- `{row['command']}`\n"
        text += "\nLP runs after source checkpoints exist:\n\n"
        for row in lp_rows:
            text += f"- `{row['command']}` (requires `{row['checkpoint_dependency']}`)\n"
        text += "\n"
    text += "Boundary: random-LM source/LP is only generated for 3k and 30k.\n"
    return text


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    rows = [validate_config(path) for path in sorted(CONFIG_DIR.glob("*.yaml"))]
    columns = [
        "config",
        "method",
        "size",
        "command",
        "train_ums_path",
        "train_records",
        "train_patients",
        "val_ums_path",
        "val_records",
        "val_patients",
        "train_val_patient_overlap",
        "output_dir",
        "output_status",
        "checkpoint_dependency",
        "status",
        "notes",
    ]
    write_csv(FINAL_DIR / "data_scaling_config_validation.csv", rows, columns)
    (FINAL_DIR / "data_scaling_config_validation.md").write_text(
        "# Data Scaling Config Validation\n\n" + markdown_table(rows, columns),
        encoding="utf-8",
    )
    (FINAL_DIR / "data_scaling_run_order.md").write_text(run_order(rows), encoding="utf-8")
    failures = [row for row in rows if row["status"] == "fail"]
    blocked = [row for row in rows if row["status"] == "blocked_until_source_run"]
    print(f"Validated {len(rows)} configs")
    print(f"Failures: {len(failures)}")
    print(f"LP blocked until source checkpoints: {len(blocked)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
