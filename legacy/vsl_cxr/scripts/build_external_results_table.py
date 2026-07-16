"""Build VSL-CXR Phase 6 external validation tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs" / "final_tables"
OUT_CSV = FINAL_DIR / "vsl_cxr_external_results.csv"
OUT_MD = FINAL_DIR / "vsl_cxr_external_results.md"
VINDR_AUDIT = FINAL_DIR / "vindr_cxr_data_quality_audit.json"
MIMIC_MANIFEST = Path("H:/Xiyao_Wang/000_Public Dataset/mimic_cxr_other/mimic-cxr-2.0.0-chexpert.csv.gz")


RUNS = [
    {
        "run": "Raw",
        "candidate": "Raw",
        "dataset": "NIH-appendix-1k",
        "status": "completed_appendix_stress",
        "path": ROOT / "outputs/vsl_cxr_phase6_external/nih1k_raw/transfer_metrics.json",
        "failure_cause": "NIH is appendix/stress only; not main external.",
    },
    {
        "run": "SAMEQ",
        "candidate": "SAMEQ",
        "dataset": "NIH-appendix-1k",
        "status": "completed_appendix_stress",
        "path": ROOT / "outputs/vsl_cxr_phase6_external/nih1k_sameq/transfer_metrics.json",
        "failure_cause": "NIH is appendix/stress only; not main external.",
    },
    {
        "run": "VSL-Core",
        "candidate": "VSL-Core",
        "dataset": "NIH-appendix-1k",
        "status": "completed_appendix_stress",
        "path": ROOT / "outputs/vsl_cxr_phase6_external/nih1k_vsl_core/transfer_metrics.json",
        "failure_cause": "NIH is appendix/stress only; not main external.",
    },
    {
        "run": "VSL-CEQ",
        "candidate": "VSL-CEQ backbone proxy",
        "dataset": "NIH-appendix-1k",
        "status": "completed_appendix_stress_proxy",
        "path": ROOT / "outputs/vsl_cxr_phase6_external/nih1k_vsl_ceq_backbone/transfer_metrics.json",
        "failure_cause": "CEQ readout is not a CheXpert classifier; row evaluates SAMEQ visual backbone proxy.",
    },
    {
        "run": "VSL-Full",
        "candidate": "VSL-Full",
        "dataset": "NIH-appendix-1k",
        "status": "completed_appendix_stress",
        "path": ROOT / "outputs/vsl_cxr_phase6_external/nih1k_vsl_full/transfer_metrics.json",
        "failure_cause": "NIH is appendix/stress only; not main external.",
    },
]

VINDR_RUNS = [
    {
        "run": "Raw",
        "candidate": "Raw",
        "dataset": "VinDr-CXR-test-3k",
        "status": "completed_main_external",
        "path": ROOT / "outputs/vsl_cxr_phase6_external/vindr_raw/transfer_metrics.json",
        "failure_cause": "Primary metrics use seven direct label mappings; Edema is retained but excluded because the official test split has zero positives.",
        "metric_key": "primary_metrics",
    },
    {
        "run": "SAMEQ",
        "candidate": "SAMEQ",
        "dataset": "VinDr-CXR-test-3k",
        "status": "completed_main_external",
        "path": ROOT / "outputs/vsl_cxr_phase6_external/vindr_sameq/transfer_metrics.json",
        "failure_cause": "Primary metrics use seven direct label mappings; Edema is retained but excluded because the official test split has zero positives.",
        "metric_key": "primary_metrics",
    },
    {
        "run": "VSL-Core",
        "candidate": "VSL-Core",
        "dataset": "VinDr-CXR-test-3k",
        "status": "completed_main_external",
        "path": ROOT / "outputs/vsl_cxr_phase6_external/vindr_vsl_core/transfer_metrics.json",
        "failure_cause": "Primary metrics use seven direct label mappings; Edema is retained but excluded because the official test split has zero positives.",
        "metric_key": "primary_metrics",
    },
    {
        "run": "VSL-CEQ",
        "candidate": "VSL-CEQ backbone proxy",
        "dataset": "VinDr-CXR-test-3k",
        "status": "completed_main_external_proxy",
        "path": ROOT / "outputs/vsl_cxr_phase6_external/vindr_vsl_ceq_backbone/transfer_metrics.json",
        "failure_cause": "CEQ readout is not a CheXpert classifier; this evaluates the SAMEQ visual-backbone proxy on the direct seven-label protocol.",
        "metric_key": "primary_metrics",
    },
    {
        "run": "VSL-Full",
        "candidate": "VSL-Full",
        "dataset": "VinDr-CXR-test-3k",
        "status": "completed_main_external",
        "path": ROOT / "outputs/vsl_cxr_phase6_external/vindr_vsl_full/transfer_metrics.json",
        "failure_cause": "Primary metrics use seven direct label mappings; Edema is retained but excluded because the official test split has zero positives.",
        "metric_key": "primary_metrics",
    },
]


READINESS_ROWS = [
    {
        "run": "Main external backup",
        "candidate": "PadChest",
        "dataset": "PadChest",
        "status": "missing",
        "failure_cause": "No local PadChest directory found at audited paths.",
    },
    {
        "run": "Conditional external",
        "candidate": "MIMIC-CXR",
        "dataset": "MIMIC-CXR",
        "status": "exists_label_manifest_overlap_audit_pending",
        "failure_cause": "Official CheXpert/metadata/split manifests exist; MIMIC remains conditional because training-overlap eligibility must be audited.",
    },
]


COLUMNS = [
    "run",
    "candidate",
    "external_dataset",
    "status",
    "records",
    "macro_auc",
    "macro_auprc",
    "ece",
    "brier",
    "best_labels",
    "worst_labels",
    "failure_cause",
    "evidence",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main_external_readiness_row() -> dict[str, Any]:
    audit = read_json(VINDR_AUDIT)
    status = str(audit.get("status") or "missing_audit")
    if status == "ready":
        readiness = "data_ready_evaluation_pending"
        cause = "VinDr-CXR official test split and direct 7-label mapping are ready; formal candidate inference has not yet been completed."
    elif status == "manifest_ready_extraction_incomplete":
        readiness = "extracting_manifest_ready"
        cause = "VinDr-CXR labels and deterministic manifests are ready while the 18,000-DICOM extraction/CRC pass is still running."
    else:
        readiness = "data_audit_pending"
        cause = "VinDr-CXR data-quality audit has not reached ready status."
    return {
        "run": "Main external",
        "candidate": "VinDr-CXR",
        "external_dataset": "VinDr-CXR-test-3k",
        "status": readiness,
        "records": audit.get("test_images", ""),
        "failure_cause": cause,
        "evidence": repo_rel(VINDR_AUDIT) if audit else "",
    }


def repo_rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def label_rank(per_label: dict[str, Any], reverse: bool) -> str:
    values = []
    for label, metrics in per_label.items():
        auc = metrics.get("auc")
        if isinstance(auc, (int, float)):
            values.append((str(label), float(auc)))
    values.sort(key=lambda item: item[1], reverse=reverse)
    return "; ".join(f"{label}:{auc:.6f}" for label, auc in values[:3])


def row_from_metrics(spec: dict[str, Any]) -> dict[str, Any]:
    payload = read_json(spec["path"])
    metrics = payload.get(spec.get("metric_key", "metrics")) or {}
    per_label = metrics.get("per_label") or {}
    return {
        "run": spec["run"],
        "candidate": spec["candidate"],
        "external_dataset": spec["dataset"],
        "status": spec["status"] if payload else (
            "pending_main_external" if spec.get("metric_key") == "primary_metrics" else "missing_metrics"
        ),
        "records": payload.get("evaluated_records", ""),
        "macro_auc": metrics.get("macro_auc", ""),
        "macro_auprc": metrics.get("macro_auprc", ""),
        "ece": metrics.get("macro_ece", ""),
        "brier": metrics.get("macro_brier", ""),
        "best_labels": label_rank(per_label, reverse=True),
        "worst_labels": label_rank(per_label, reverse=False),
        "failure_cause": spec["failure_cause"],
        "evidence": repo_rel(spec["path"]) if payload else "",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in COLUMNS})


def md_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| " + " | ".join(COLUMNS) + " |", "| " + " | ".join("---" for _ in COLUMNS) + " |"]
    for row in rows:
        values = [str(row.get(column, "")).replace("|", "\\|").replace("\n", " ") for column in COLUMNS]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    rows = [main_external_readiness_row()]
    for blocked in READINESS_ROWS:
        rows.append(
            {
                "run": blocked["run"],
                "candidate": blocked["candidate"],
                "external_dataset": blocked["dataset"],
                "status": blocked["status"],
                "failure_cause": blocked["failure_cause"],
                "evidence": str(MIMIC_MANIFEST) if blocked["candidate"] == "MIMIC-CXR" and MIMIC_MANIFEST.exists() else "",
            }
        )
    rows.extend(row_from_metrics(spec) for spec in VINDR_RUNS)
    rows.extend(row_from_metrics(spec) for spec in RUNS)
    write_csv(OUT_CSV, rows)
    OUT_MD.write_text(
        "# VSL-CXR Phase 6 External Results\n\n"
        "Generated from current VinDr-CXR main-external readiness/results and NIH appendix/stress transfer metrics. "
        "NIH rows remain appendix/stress evidence only.\n\n"
        + md_table(rows)
        + "\n",
        encoding="utf-8",
    )
    completed_main = sum(
        1
        for row in rows
        if str(row.get("status", "")).startswith("completed_main_external")
    )
    completed_appendix = sum(
        1
        for row in rows
        if str(row.get("status", "")).startswith("completed_appendix")
    )
    print(f"rows={len(rows)}")
    print(f"completed_main={completed_main}")
    print(f"completed_appendix={completed_appendix}")
    print(f"csv={repo_rel(OUT_CSV)}")
    print(f"md={repo_rel(OUT_MD)}")


if __name__ == "__main__":
    main()
