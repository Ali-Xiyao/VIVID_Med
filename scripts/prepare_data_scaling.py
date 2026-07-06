"""Prepare deterministic CheXpert subsets and configs for P1 data scaling."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_JSONL = ROOT / "data" / "dataset" / "processed" / "chexpert_ums.jsonl"
SPLIT_DIR = ROOT / "data" / "splits"
CONFIG_DIR = ROOT / "configs" / "data_scaling"
FINAL_DIR = ROOT / "outputs" / "final_tables"
SEED = 42
VAL_TARGET = 1000


SELECTED_LABELS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Fracture",
    "Support Devices",
]

TARGETS = {
    "1k": 1000,
    "3k": 3000,
    "10k": 10000,
    "30k": None,
}


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def patient_id(record: dict[str, Any]) -> str:
    original_path = str(record.get("extensions", {}).get("original_path", ""))
    match = re.search(r"patient(\d+)", original_path)
    if not match:
        raise ValueError(f"Cannot recover patient id from original_path={original_path!r}")
    return f"patient{match.group(1)}"


def stable_patient_order(patient_ids: list[str]) -> list[str]:
    def score(pid: str) -> str:
        return hashlib.sha256(f"{SEED}:{pid}".encode("utf-8")).hexdigest()

    return sorted(patient_ids, key=score)


def select_patient_records(
    patients: dict[str, list[dict[str, Any]]],
    ordered_patients: list[str],
    target: int | None,
) -> list[dict[str, Any]]:
    if target is None:
        selected_patients = set(ordered_patients)
    else:
        selected_patients: set[str] = set()
        count = 0
        for pid in ordered_patients:
            group_size = len(patients[pid])
            if count + group_size <= target:
                selected_patients.add(pid)
                count += group_size
        if not selected_patients:
            raise ValueError(f"No patients selected for target={target}")

    selected = []
    for pid in ordered_patients:
        if pid in selected_patients:
            selected.extend(patients[pid])
    return selected


def select_fixed_val_patients(
    patients: dict[str, list[dict[str, Any]]],
    ordered_patients: list[str],
    target: int,
) -> set[str]:
    selected: set[str] = set()
    count = 0
    for pid in ordered_patients:
        group_size = len(patients[pid])
        if count + group_size <= target:
            selected.add(pid)
            count += group_size
        if count == target:
            break
    if not selected:
        raise ValueError("Could not select a fixed validation patient set")
    return selected


def state_counts(record: dict[str, Any]) -> dict[str, int]:
    counts = {"present": 0, "absent": 0, "uncertain": 0, "null": 0, "answerable": 0}
    findings = record.get("findings", {})
    answerability = record.get("answerability", {})
    uncertainty = record.get("uncertainty", {})
    for label in SELECTED_LABELS:
        state = findings.get(label, {}).get("state")
        if state == "present":
            counts["present"] += 1
        elif state == "absent":
            counts["absent"] += 1
        elif state == "uncertain" or uncertainty.get(label) is True:
            counts["uncertain"] += 1
        else:
            counts["null"] += 1
        if answerability.get(label) is True:
            counts["answerable"] += 1
    return counts


def write_split_csv(path: Path, records: list[dict[str, Any]]) -> None:
    columns = [
        "sample_id",
        "patient_id",
        "original_path",
        "present_count",
        "absent_count",
        "uncertain_count",
        "null_count",
        "answerable_count",
        "present_labels",
        "uncertain_labels",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            counts = state_counts(record)
            findings = record.get("findings", {})
            uncertainty = record.get("uncertainty", {})
            present_labels = [
                label
                for label in SELECTED_LABELS
                if findings.get(label, {}).get("state") == "present"
            ]
            uncertain_labels = [
                label
                for label in SELECTED_LABELS
                if findings.get(label, {}).get("state") == "uncertain"
                or uncertainty.get(label) is True
            ]
            writer.writerow(
                {
                    "sample_id": record.get("extensions", {}).get("sample_id", ""),
                    "patient_id": patient_id(record),
                    "original_path": record.get("extensions", {}).get("original_path", ""),
                    "present_count": counts["present"],
                    "absent_count": counts["absent"],
                    "uncertain_count": counts["uncertain"],
                    "null_count": counts["null"],
                    "answerable_count": counts["answerable"],
                    "present_labels": ";".join(present_labels),
                    "uncertain_labels": ";".join(uncertain_labels),
                }
            )


def split_summary(name: str, records: list[dict[str, Any]]) -> dict[str, str]:
    counts = [state_counts(record) for record in records]
    n = len(records)
    patients = {patient_id(record) for record in records}
    slots = n * len(SELECTED_LABELS)
    totals = {key: sum(item[key] for item in counts) for key in counts[0]}
    row = {
        "split": name,
        "records": str(n),
        "patients": str(len(patients)),
        "label_slots": str(slots),
        "present_rate": f"{totals['present'] / slots:.6f}",
        "absent_rate": f"{totals['absent'] / slots:.6f}",
        "uncertain_rate": f"{totals['uncertain'] / slots:.6f}",
        "null_rate": f"{totals['null'] / slots:.6f}",
        "answerable_rate": f"{totals['answerable'] / slots:.6f}",
        "mean_present_per_image": f"{mean(item['present'] for item in counts):.3f}",
        "mean_null_per_image": f"{mean(item['null'] for item in counts):.3f}",
    }
    for label in SELECTED_LABELS:
        present = sum(
            1
            for record in records
            if record.get("findings", {}).get(label, {}).get("state") == "present"
        )
        row[f"{label}_present_rate"] = f"{present / n:.6f}"
    return row


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def dump_yaml(path: Path, config: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def baseline_config(split_path: Path, val_path: Path, size: str) -> dict[str, Any]:
    return {
        "data": {
            "data_root": "./data/dataset",
            "train_ums_path": f"./{rel(split_path)}",
            "val_ums_path": f"./{rel(val_path)}",
            "image_size": 224,
            "use_common_labels_only": False,
            "max_val_samples": 1000,
            "num_workers": 0,
        },
        "model": {
            "vit_model_name": "vit_base_patch16_224",
            "vit_pretrained": True,
            "drop_rate": 0.0,
            "drop_path_rate": 0.0,
        },
        "training": {
            "learning_rate": 1.0e-4,
            "weight_decay": 0.01,
            "warmup_steps": 500,
            "max_steps": 10000,
            "batch_size": 32,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "fp16": False,
            "bf16": True,
            "log_interval": 10,
            "eval_interval": 500,
            "save_interval": 1000,
            "output_dir": f"./outputs/data_scaling/bce_{size}",
        },
        "evaluation": {"threshold": 0.5, "uncertain_policy": "ignore"},
        "seed": SEED,
        "device": "cuda",
    }


def derived_config(
    base: dict[str, Any],
    split_path: Path,
    val_path: Path,
    method: str,
    size: str,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config.setdefault("data", {})["train_ums_path"] = f"./{rel(split_path)}"
    config["data"]["val_ums_path"] = f"./{rel(val_path)}"
    config["data"]["num_workers"] = 0
    config.setdefault("training", {})["output_dir"] = f"./outputs/data_scaling/{method}_{size}"
    config["seed"] = SEED
    return config


def write_configs(split_paths: dict[str, Path]) -> list[dict[str, str]]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    frozen_base = load_yaml(ROOT / "configs" / "ablation_A_ums_12label.yaml")
    no_lm_base = load_yaml(ROOT / "configs" / "ums_classifier_no_llm_12label.yaml")
    random_lm_base = load_yaml(ROOT / "configs" / "ablation_ums_random_lm_12label.yaml")
    lp_frozen_base = load_yaml(ROOT / "configs" / "lp_A_ums_12label.yaml")
    lp_no_lm_base = load_yaml(ROOT / "configs" / "lp_ums_classifier_no_llm_12label.yaml")
    lp_random_lm_base = load_yaml(ROOT / "configs" / "lp_ums_random_lm_12label.yaml")
    manifest: list[dict[str, str]] = []
    val_path = SPLIT_DIR / "chexpert_val_fixed.jsonl"

    for size, split_path in split_paths.items():
        no_lm_source_ckpt = f"./outputs/data_scaling/no_lm_ums_{size}/best.pt"
        frozen_source_ckpt = f"./outputs/data_scaling/frozen_lm_ums_{size}/checkpoints/best.pt"
        random_source_ckpt = f"./outputs/data_scaling/random_lm_ums_{size}/checkpoints/best.pt"
        configs = {
            f"bce_{size}.yaml": (
                baseline_config(split_path, val_path, size),
                "python scripts/train_vit_baseline.py",
                "new minimal BCE template",
            ),
            f"no_lm_ums_{size}.yaml": (
                derived_config(no_lm_base, split_path, val_path, "no_lm_ums", size),
                "python scripts/train_ums_classifier.py",
                "derived from configs/ums_classifier_no_llm_12label.yaml",
            ),
            f"frozen_lm_ums_{size}.yaml": (
                derived_config(frozen_base, split_path, val_path, "frozen_lm_ums", size),
                "python scripts/train_cxr.py",
                "derived from configs/ablation_A_ums_12label.yaml",
            ),
            f"lp_no_lm_ums_{size}.yaml": (
                lp_config(
                    lp_no_lm_base,
                    split_path,
                    val_path,
                    no_lm_source_ckpt,
                    f"lp_no_lm_ums_{size}",
                ),
                "python scripts/train_vit_baseline.py",
                "linear probe derived from configs/lp_ums_classifier_no_llm_12label.yaml",
            ),
            f"lp_frozen_lm_ums_{size}.yaml": (
                lp_config(
                    lp_frozen_base,
                    split_path,
                    val_path,
                    frozen_source_ckpt,
                    f"lp_frozen_lm_ums_{size}",
                ),
                "python scripts/train_vit_baseline.py",
                "linear probe derived from configs/lp_A_ums_12label.yaml",
            ),
        }
        if size in {"3k", "30k"}:
            configs[f"random_lm_ums_{size}.yaml"] = (
                derived_config(random_lm_base, split_path, val_path, "random_lm_ums", size),
                "python scripts/train_cxr.py",
                "derived from configs/ablation_ums_random_lm_12label.yaml",
            )
            configs[f"lp_random_lm_ums_{size}.yaml"] = (
                lp_config(
                    lp_random_lm_base,
                    split_path,
                    val_path,
                    random_source_ckpt,
                    f"lp_random_lm_ums_{size}",
                ),
                "python scripts/train_vit_baseline.py",
                "linear probe derived from configs/lp_ums_random_lm_12label.yaml",
            )
        for filename, (config, command, provenance) in configs.items():
            path = CONFIG_DIR / filename
            dump_yaml(path, config)
            manifest.append(
                {
                    "config": rel(path),
                    "method": filename.replace(f"_{size}.yaml", ""),
                    "size": size,
                    "command": f"{command} --config {rel(path)}",
                    "output_dir": config["training"]["output_dir"],
                    "provenance": provenance,
                }
            )
    return manifest


def lp_config(
    base: dict[str, Any],
    split_path: Path,
    val_path: Path,
    init_vit_checkpoint: str,
    output_name: str,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config.setdefault("data", {})["train_ums_path"] = f"./{rel(split_path)}"
    config["data"]["val_ums_path"] = f"./{rel(val_path)}"
    config["data"]["num_workers"] = 0
    config.setdefault("transfer", {})["init_vit_checkpoint"] = init_vit_checkpoint
    config["transfer"]["freeze_backbone"] = True
    config.setdefault("training", {})["output_dir"] = f"./outputs/data_scaling/{output_name}"
    config["seed"] = SEED
    return config


def write_summary(rows: list[dict[str, str]], configs: list[dict[str, str]]) -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    text = "# Data Scaling Prep Summary\n\n"
    text += (
        "Prepared deterministic patient-level CheXpert UMS JSONL subsets and "
        "config drafts. No training was launched.\n\n"
    )
    text += "## Splits\n\n"
    text += markdown_table(
        rows,
        [
            "split",
            "records",
            "patients",
            "present_rate",
            "null_rate",
            "uncertain_rate",
            "answerable_rate",
            "mean_present_per_image",
            "mean_null_per_image",
        ],
    )
    text += "\n## Config Queue\n\n"
    text += markdown_table(configs, ["config", "size", "command", "output_dir", "provenance"])
    text += (
        "\n## Boundary Notes\n\n"
        "- Splits are patient-level by `patientXXXXX` parsed from CheXpert `original_path`.\n"
        "- `chexpert_val_fixed.jsonl` is patient-disjoint from every generated train subset.\n"
        "- JSONL files are the training inputs; CSV files are provenance/distribution audits.\n"
        "- `30k` means the full generated training pool after holding out fixed-val patients, not exactly 30,000.\n"
        "- Generated configs use `num_workers: 0` for conservative Windows stability.\n"
        "- No SPD configs were generated.\n"
    )
    (FINAL_DIR / "data_scaling_prep_summary.md").write_text(text, encoding="utf-8")


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


def main() -> None:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(SOURCE_JSONL)
    patients: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        patients[patient_id(record)].append(record)
    ordered_patients = stable_patient_order(list(patients))
    val_patients = select_fixed_val_patients(patients, ordered_patients, VAL_TARGET)
    train_patients = [pid for pid in ordered_patients if pid not in val_patients]
    val_records = [
        record
        for pid in ordered_patients
        if pid in val_patients
        for record in patients[pid]
    ]
    write_jsonl(SPLIT_DIR / "chexpert_val_fixed.jsonl", val_records)
    write_split_csv(SPLIT_DIR / "chexpert_val_fixed.csv", val_records)

    split_paths: dict[str, Path] = {}
    summary_rows: list[dict[str, str]] = [split_summary("val_fixed", val_records)]
    for name, target in TARGETS.items():
        selected = select_patient_records(patients, train_patients, target)
        jsonl_path = SPLIT_DIR / f"chexpert_train_{name}.jsonl"
        csv_path = SPLIT_DIR / f"chexpert_train_{name}.csv"
        write_jsonl(jsonl_path, selected)
        write_split_csv(csv_path, selected)
        split_paths[name] = jsonl_path
        summary_rows.append(split_summary(name, selected))

    summary_columns = list(summary_rows[0].keys())
    write_csv(SPLIT_DIR / "data_scaling_split_summary.csv", summary_rows, summary_columns)
    (SPLIT_DIR / "data_scaling_split_summary.md").write_text(
        "# Data Scaling Split Summary\n\n"
        + markdown_table(
            summary_rows,
            [
                "split",
                "records",
                "patients",
                "present_rate",
                "null_rate",
                "uncertain_rate",
                "answerable_rate",
                "mean_present_per_image",
                "mean_null_per_image",
            ],
        ),
        encoding="utf-8",
    )

    config_manifest = write_configs(split_paths)
    write_summary(summary_rows, config_manifest)
    print(f"Prepared {len(summary_rows) - 1} data scaling train splits from {len(records)} records")
    print(f"Prepared fixed validation split with {len(val_records)} records")
    print(f"Prepared {len(config_manifest)} config drafts in {rel(CONFIG_DIR)}")


if __name__ == "__main__":
    main()
