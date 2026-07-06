"""Audit whether existing historical S1-like runs cover schema-sweep S1."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs" / "final_tables"


COLUMNS = [
    "audit_id",
    "pathway",
    "schema_sweep_config",
    "historical_config",
    "schema_train_path",
    "historical_train_path",
    "schema_val_path",
    "historical_val_path",
    "schema_train_count",
    "historical_train_count",
    "schema_val_count",
    "historical_val_count",
    "train_hash_match",
    "val_hash_match",
    "config_match_summary",
    "artifact_summary",
    "coverage_decision",
    "claim_boundary",
    "evidence_paths",
]


def read_yaml(rel_path: str) -> dict[str, Any]:
    path = ROOT / rel_path
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def norm_path(path: str | None) -> str:
    if not path:
        return ""
    return path.replace("\\", "/").lstrip("./")


def resolve_data_path(path: str | None) -> Path:
    if not path:
        raise ValueError("missing data path")
    return ROOT / norm_path(path)


def file_count_and_hash(path: Path) -> tuple[int, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            count += 1
            digest.update(line.rstrip(b"\r\n"))
            digest.update(b"\n")
    return count, digest.hexdigest()


def exists(rel_path: str) -> bool:
    return (ROOT / rel_path).exists()


def artifact_summary(paths: list[str]) -> str:
    return "; ".join(f"{path}={'present' if exists(path) else 'missing'}" for path in paths)


def effective_batch(config: dict[str, Any]) -> int:
    training = config["training"]
    return int(training["batch_size"]) * int(training.get("gradient_accumulation_steps", 1))


def selected_labels(config: dict[str, Any]) -> list[str]:
    return list(config.get("data", {}).get("selected_labels") or [])


def compare_pair(
    audit_id: str,
    pathway: str,
    schema_config_path: str,
    historical_config_path: str,
    required_artifacts: list[str],
) -> dict[str, str]:
    schema_config = read_yaml(schema_config_path)
    historical_config = read_yaml(historical_config_path)

    schema_train = resolve_data_path(schema_config["data"]["train_ums_path"])
    schema_val = resolve_data_path(schema_config["data"]["val_ums_path"])
    historical_train = resolve_data_path(historical_config["data"]["train_ums_path"])
    historical_val = resolve_data_path(historical_config["data"]["val_ums_path"])

    schema_train_count, schema_train_hash = file_count_and_hash(schema_train)
    schema_val_count, schema_val_hash = file_count_and_hash(schema_val)
    historical_train_count, historical_train_hash = file_count_and_hash(historical_train)
    historical_val_count, historical_val_hash = file_count_and_hash(historical_val)

    train_match = schema_train_hash == historical_train_hash
    val_match = schema_val_hash == historical_val_hash
    label_match = selected_labels(schema_config) == selected_labels(historical_config)
    steps_match = int(schema_config["training"]["max_steps"]) == int(historical_config["training"]["max_steps"])
    batch_match = effective_batch(schema_config) == effective_batch(historical_config)
    schema_mode_match = schema_config["data"].get("schema_mode", "state_only") == historical_config["data"].get(
        "schema_mode", "state_only"
    )
    target_format_match = schema_config["data"].get("target_format", "json") == historical_config["data"].get(
        "target_format", "json"
    )
    artifacts_present = all(exists(path) for path in required_artifacts)

    config_bits = {
        "labels": label_match,
        "max_steps": steps_match,
        "effective_batch": batch_match,
        "schema_mode": schema_mode_match,
        "target_format": target_format_match,
    }
    core_match = train_match and val_match and all(config_bits.values()) and artifacts_present
    decision = "covered_by_historical_formal_row" if core_match else "not_covered_fixed_split_mismatch_or_missing"

    return {
        "audit_id": audit_id,
        "pathway": pathway,
        "schema_sweep_config": schema_config_path,
        "historical_config": historical_config_path,
        "schema_train_path": norm_path(schema_config["data"]["train_ums_path"]),
        "historical_train_path": norm_path(historical_config["data"]["train_ums_path"]),
        "schema_val_path": norm_path(schema_config["data"]["val_ums_path"]),
        "historical_val_path": norm_path(historical_config["data"]["val_ums_path"]),
        "schema_train_count": str(schema_train_count),
        "historical_train_count": str(historical_train_count),
        "schema_val_count": str(schema_val_count),
        "historical_val_count": str(historical_val_count),
        "train_hash_match": str(train_match).lower(),
        "val_hash_match": str(val_match).lower(),
        "config_match_summary": "; ".join(f"{key}={str(value).lower()}" for key, value in config_bits.items()),
        "artifact_summary": artifact_summary(required_artifacts),
        "coverage_decision": decision,
        "claim_boundary": (
            "Historical row can cover formal S1 only if fixed train/val files, core config fields, and artifacts match."
        ),
        "evidence_paths": "; ".join([schema_config_path, historical_config_path, *required_artifacts]),
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    header = "| " + " | ".join(COLUMNS) + " |"
    divider = "| " + " | ".join("---" for _ in COLUMNS) + " |"
    body = ["| " + " | ".join(row.get(column, "") for column in COLUMNS) + " |" for row in rows]
    return "\n".join([header, divider, *body]) + "\n"


def main() -> None:
    rows = [
        compare_pair(
            "no_lm_s1_historical_coverage",
            "no-LM UMS state_only",
            "configs/schema_sweep/no_lm_s1_state_only.yaml",
            "configs/ums_classifier_no_llm_12label.yaml",
            [
                "outputs/ums_classifier_no_llm_12label_full/metrics_final.json",
                "outputs/ums_classifier_no_llm_12label_full/best.pt",
                "outputs/ums_classifier_no_llm_12label_full/final.pt",
                "outputs/lp_ums_classifier_no_llm_12label_full/metrics_final.json",
                "outputs/lp_ums_classifier_no_llm_12label_full/best.pt",
                "outputs/lp_ums_classifier_no_llm_12label_full/final.pt",
            ],
        ),
        compare_pair(
            "frozen_lm_s1_historical_coverage",
            "frozen-LM UMS state_only",
            "configs/schema_sweep/frozen_lm_s1_state_only.yaml",
            "configs/ablation_A_ums_12label.yaml",
            [
                "outputs/ablation_A_ums_12label/checkpoints/best.pt",
                "outputs/ablation_A_ums_12label/checkpoints/final.pt",
                "outputs/lp_A_ums_12label/metrics_final.json",
                "outputs/lp_A_ums_12label/best.pt",
                "outputs/lp_A_ums_12label/final.pt",
            ],
        ),
    ]

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(FINAL_DIR / "schema_s1_coverage_audit.csv", rows)

    text = "# Schema S1 Coverage Audit\n\n"
    text += (
        "This audit checks whether historical main-table S1-like runs can cover the formal "
        "schema-sweep S1 fixed-split requirement. Coverage requires matching train/val JSONL "
        "content, core config fields, and source/LP artifacts.\n\n"
    )
    text += markdown_table(rows)
    text += "\n## Decision\n\n"
    if all(row["coverage_decision"] == "covered_by_historical_formal_row" for row in rows):
        text += "- Historical main-table rows cover S1 under the formal schema-sweep protocol.\n"
    else:
        text += (
            "- Historical main-table rows do not fully cover formal fixed-split S1. "
            "Do not use them as exact S1 schema-sweep evidence without the mismatch caveat.\n"
        )
    (FINAL_DIR / "schema_s1_coverage_audit.md").write_text(text, encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {FINAL_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
