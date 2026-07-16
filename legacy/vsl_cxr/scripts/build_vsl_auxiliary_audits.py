"""Auxiliary VSL-CXR v5 manifests and bounded evaluation summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs" / "final_tables"


def repo_rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_jsonl_sample(path: Path, limit: int = 200) -> tuple[int, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    count = 0
    if not path.exists():
        return 0, rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            count += 1
            if len(rows) < limit:
                rows.append(json.loads(line))
    return count, rows


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def write_md(path: Path, title: str, rows: list[dict[str, Any]], columns: list[str], note: str = "") -> None:
    lines = [f"# {title}", ""]
    if note:
        lines.extend([note, ""])
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        values = [str(row.get(key, "")).replace("|", "\\|").replace("\n", " ") for key in columns]
        lines.append("| " + " | ".join(values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_jsonl(label: str, path: str, role: str) -> dict[str, Any]:
    full = ROOT / path
    count, sample = read_jsonl_sample(full)
    keys = sorted({key for row in sample for key in row})
    answer_counts: dict[str, int] = {}
    for row in sample:
        value = str(row.get("answer_type") or row.get("sufficiency_label") or row.get("answer") or "")
        answer_counts[value] = answer_counts.get(value, 0) + 1
    return {
        "artifact": label,
        "role": role,
        "path": path,
        "status": "exists" if full.exists() else "missing",
        "rows": count,
        "sample_keys": ",".join(keys[:40]),
        "sample_label_counts": "; ".join(f"{key}:{value}" for key, value in sorted(answer_counts.items())[:12]),
    }


def build_data_source_manifest() -> None:
    sources = [
        ("D0 Basic-QA", "outputs/instruction_data/glm_validated/d0_train_validated.jsonl", "clinical_statement_source"),
        ("D1 CF-QA", "outputs/instruction_data/next_stage/cf_10k_train.jsonl", "counterfactual_statement_source"),
        ("D1 CF-QA validated", "outputs/instruction_data/glm_validated/d6_hard_cf_3k.jsonl", "counterfactual_statement_source"),
        ("D2 SAMEQ", "outputs/instruction_data/next_stage/sameq_shuf_3k_train.jsonl", "sameq_pair_source"),
        ("D2 SAMEQ full", "outputs/instruction_data/cvcp_ccsh/cvcp_v1_sameq_full_train.jsonl", "sameq_pair_source"),
        ("D3 SAMEQ-CF", "outputs/instruction_data/cvcp_ccsh/sameq_cf_20_train.jsonl", "sameq_cf_source"),
        ("D4 SAMEQ-K4", "outputs/instruction_data/next_stage/shuf_k4_train.jsonl", "hard_negative_source"),
        ("D5 SAMEQ-HNMB mined", "outputs/instruction_data/next_stage/mined_shuf_train.jsonl", "memory_bank_hard_negative_source"),
        ("D5 SAMEQ-HNMB selfhard", "outputs/instruction_data/next_stage/selfhard_shuf_train.jsonl", "memory_bank_hard_negative_source"),
        ("D6 VSL-4class", "outputs/instruction_data/vsl_cxr/d6_vsl_4class_train.jsonl", "vsl_sufficiency_source"),
        ("D9 VSL-full", "outputs/instruction_data/vsl_cxr/d9_vsl_full_train.jsonl", "vsl_full_source"),
        ("D9 CEQ targets", "outputs/instruction_data/vsl_cxr/d9_ceq_targets_train.jsonl", "ceq_source"),
        ("D9 CCSH pairs", "outputs/instruction_data/vsl_cxr/d9_ccsh_pairs_train.jsonl", "ccsh_source"),
    ]
    rows = [summarize_jsonl(*source) for source in sources]
    columns = ["artifact", "role", "path", "status", "rows", "sample_keys", "sample_label_counts"]
    write_csv(FINAL_DIR / "vsl_cxr_data_source_manifest.csv", rows, columns)
    write_md(
        FINAL_DIR / "vsl_cxr_data_source_manifest.md",
        "VSL-CXR Data Source Manifest",
        rows,
        columns,
        "This table audits current D0-D9 source artifacts used or bounded by the v5 plan.",
    )
    print(json.dumps({"data_source_rows": len(rows), "existing": sum(1 for row in rows if row["status"] == "exists")}, indent=2))


def build_sufficiency_summary() -> None:
    formal = read_csv(FINAL_DIR / "vsl_cxr_formal_run_results.csv")
    keep = [
        "VSL-CXR-D6-VSL2",
        "VSL-CXR-D6-VSL3",
        "VSL-CXR-D6-VSL4",
        "VSL-CXR-D6-VSL4-BALANCED",
        "VSL-CXR-D6-VSL4-FIELD-BALANCED",
        "VSL-CXR-D6-VSL4-HIERARCHICAL",
    ]
    rows = []
    for row in formal:
        if row.get("run_id") in keep:
            rows.append(
                {
                    "run_id": row.get("run_id", ""),
                    "data_version": row.get("data_version", ""),
                    "status": row.get("status", ""),
                    "global_step": row.get("global_step", ""),
                    "best_val_loss": row.get("best_val_loss", ""),
                    "train_records": row.get("train_records", ""),
                    "val_records": row.get("val_records", ""),
                    "evidence": row.get("metrics_final", ""),
                }
            )
    columns = ["run_id", "data_version", "status", "global_step", "best_val_loss", "train_records", "val_records", "evidence"]
    write_csv(FINAL_DIR / "vsl_cxr_sufficiency_summary.csv", rows, columns)
    write_md(FINAL_DIR / "vsl_cxr_sufficiency_summary.md", "VSL-CXR Sufficiency Summary", rows, columns)
    print(json.dumps({"sufficiency_rows": len(rows)}, indent=2))


def build_calibration_summary() -> None:
    rows: list[dict[str, Any]] = []
    for row in read_csv(FINAL_DIR / "vsl_cxr_external_results.csv"):
        if row.get("ece") or row.get("brier"):
            rows.append(
                {
                    "source": "phase6_external",
                    "run": row.get("run", ""),
                    "dataset": row.get("external_dataset", ""),
                    "status": row.get("status", ""),
                    "ece": row.get("ece", ""),
                    "brier": row.get("brier", ""),
                    "evidence": row.get("evidence", ""),
                    "boundary": row.get("failure_cause", ""),
                }
            )
    for row in read_csv(FINAL_DIR / "vsl_cxr_ccsh_results.csv"):
        if row.get("binary_ece"):
            rows.append(
                {
                    "source": "phase4_ccsh",
                    "run": row.get("run_id", ""),
                    "dataset": "D9 CCSH validation",
                    "status": row.get("status", ""),
                    "ece": row.get("binary_ece", ""),
                    "brier": "",
                    "evidence": row.get("metrics_final", ""),
                    "boundary": "CCSH ECE; no binned calibration curve points exported.",
                }
            )
    columns = ["source", "run", "dataset", "status", "ece", "brier", "evidence", "boundary"]
    write_csv(FINAL_DIR / "vsl_cxr_calibration_summary.csv", rows, columns)
    write_md(
        FINAL_DIR / "vsl_cxr_calibration_summary.md",
        "VSL-CXR Calibration Summary",
        rows,
        columns,
        "ECE/Brier are current v5 calibration evidence. Binned calibration curve points are not exported.",
    )
    print(json.dumps({"calibration_rows": len(rows)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=["data_sources", "sufficiency", "calibration", "all"],
        nargs="?",
        default="all",
    )
    args = parser.parse_args()
    if args.mode in {"data_sources", "all"}:
        build_data_source_manifest()
    if args.mode in {"sufficiency", "all"}:
        build_sufficiency_summary()
    if args.mode in {"calibration", "all"}:
        build_calibration_summary()


if __name__ == "__main__":
    main()
