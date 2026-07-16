"""Build VSL-CXR Phase 5 integrated candidate evidence tables."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs" / "final_tables"
FORMAL_CSV = FINAL_DIR / "vsl_cxr_formal_run_results.csv"
CCSH_CSV = FINAL_DIR / "vsl_cxr_ccsh_results.csv"
AUCH_CSV = FINAL_DIR / "vsl_cxr_auch_results.csv"
OUT_CSV = FINAL_DIR / "vsl_cxr_phase5_candidate_results.csv"
OUT_MD = FINAL_DIR / "vsl_cxr_phase5_candidate_results.md"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def by_run_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row.get("run_id", ""): row for row in rows}


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = [str(row.get(column, "")).replace("|", "\\|").replace("\n", " ") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def metric(row: dict[str, str] | None, key: str) -> str:
    return "" if not row else row.get(key, "")


def completed(*rows: dict[str, str] | None) -> bool:
    return all(row and row.get("status") == "completed" for row in rows)


def build_rows() -> list[dict[str, Any]]:
    formal = by_run_id(read_rows(FORMAL_CSV))
    ccsh = by_run_id(read_rows(CCSH_CSV))
    auch = by_run_id(read_rows(AUCH_CSV))

    b3 = formal.get("VSL-CXR-B3-SAMEQ")
    b5 = formal.get("VSL-CXR-B5-SAMEQ-K4")
    b6 = formal.get("VSL-CXR-B6-SAMEQ-HNMB")
    full = formal.get("VSL-CXR-P5-VSL-FULL")

    sameq_ccsh = ccsh.get("VSL-CXR-CCSH-SAMEQ")
    k4_ccsh = ccsh.get("VSL-CXR-CCSH-SAMEQ-K4")
    hnmb_ccsh = ccsh.get("VSL-CXR-CCSH-HNMB")
    ceq_ccsh = ccsh.get("VSL-CXR-CCSH-CEQ")
    full_readout = ccsh.get("VSL-CXR-AUCH-CEQ-CCSH")
    auch_sameq = auch.get("VSL-CXR-AUCH-SAMEQ")

    full_completed = completed(full, full_readout)
    rows: list[dict[str, Any]] = [
        {
            "candidate": "VSL-Lite",
            "data_engine": "SAMEQ",
            "encoder_module": "global",
            "readout": "none/LP",
            "status": "component_completed" if completed(b3) else "pending",
            "backbone_run_id": "VSL-CXR-B3-SAMEQ",
            "backbone_best_val_loss": metric(b3, "best_val_loss"),
            "ccsh_auc": "",
            "auprc": "",
            "ece": "",
            "answerability_auc": "",
            "interpretability": "low",
            "cost_proxy_seconds": metric(b3, "elapsed_seconds"),
            "evidence": metric(b3, "checkpoint"),
            "decision": "SAMEQ global encoder baseline; LP/CheXpert readout not yet available.",
        },
        {
            "candidate": "VSL-Core",
            "data_engine": "SAMEQ-K4",
            "encoder_module": "global",
            "readout": "CCSH",
            "status": "component_completed" if completed(b5, k4_ccsh) else "pending",
            "backbone_run_id": "VSL-CXR-B5-SAMEQ-K4",
            "backbone_best_val_loss": metric(b5, "best_val_loss"),
            "ccsh_auc": metric(k4_ccsh, "binary_auc"),
            "auprc": metric(k4_ccsh, "binary_auprc"),
            "ece": metric(k4_ccsh, "binary_ece"),
            "answerability_auc": "",
            "interpretability": "medium",
            "cost_proxy_seconds": metric(b5, "elapsed_seconds"),
            "evidence": metric(k4_ccsh, "run_dir"),
            "decision": "Completed as SAMEQ-K4 backbone plus CCSH readout.",
        },
        {
            "candidate": "VSL-HNMB",
            "data_engine": "SAMEQ-HNMB",
            "encoder_module": "global",
            "readout": "CCSH",
            "status": "component_completed" if completed(b6, hnmb_ccsh) else "pending",
            "backbone_run_id": "VSL-CXR-B6-SAMEQ-HNMB",
            "backbone_best_val_loss": metric(b6, "best_val_loss"),
            "ccsh_auc": metric(hnmb_ccsh, "binary_auc"),
            "auprc": metric(hnmb_ccsh, "binary_auprc"),
            "ece": metric(hnmb_ccsh, "binary_ece"),
            "answerability_auc": "",
            "interpretability": "medium",
            "cost_proxy_seconds": metric(b6, "elapsed_seconds"),
            "evidence": metric(hnmb_ccsh, "run_dir"),
            "decision": "Completed as SAMEQ-HNMB backbone plus CCSH readout.",
        },
        {
            "candidate": "VSL-CEQ",
            "data_engine": "SAMEQ",
            "encoder_module": "CEQ",
            "readout": "CCSH",
            "status": "component_completed" if completed(b3, ceq_ccsh) else "pending",
            "backbone_run_id": "VSL-CXR-B3-SAMEQ",
            "backbone_best_val_loss": metric(b3, "best_val_loss"),
            "ccsh_auc": metric(ceq_ccsh, "binary_auc"),
            "auprc": metric(ceq_ccsh, "binary_auprc"),
            "ece": metric(ceq_ccsh, "binary_ece"),
            "answerability_auc": "",
            "interpretability": "high",
            "cost_proxy_seconds": metric(b3, "elapsed_seconds"),
            "evidence": metric(ceq_ccsh, "run_dir"),
            "decision": "Current strongest Phase 5 component candidate by CCSH binary AUC.",
        },
        {
            "candidate": "VSL-Full",
            "data_engine": "SAMEQ-HNMB + VSL-4class",
            "encoder_module": "CEQ",
            "readout": "CCSH+AUCH",
            "status": "formal_training_completed" if full_completed else "needs_formal_training",
            "backbone_run_id": "VSL-CXR-P5-VSL-FULL",
            "backbone_best_val_loss": metric(full, "best_val_loss"),
            "ccsh_auc": metric(full_readout, "binary_auc"),
            "auprc": metric(full_readout, "binary_auprc"),
            "ece": metric(full_readout, "binary_ece"),
            "answerability_auc": metric(auch_sameq, "answerability_auc"),
            "interpretability": "high",
            "cost_proxy_seconds": metric(full, "elapsed_seconds"),
            "evidence": metric(full, "checkpoint") or metric(full_readout, "run_dir"),
            "decision": (
                "D9 mixed-instruction formal training completed; external and locked-final evidence still pending."
                if full_completed
                else "Requires D9 mixed-instruction formal training before locked-final claims."
            ),
        },
        {
            "candidate": "VSL-Domain",
            "data_engine": "VSL-Core",
            "encoder_module": "optional DRA",
            "readout": "CCSH",
            "status": "blocked_external_data",
            "backbone_run_id": "VSL-CXR-B5-SAMEQ-K4",
            "backbone_best_val_loss": metric(b5, "best_val_loss"),
            "ccsh_auc": metric(k4_ccsh, "binary_auc"),
            "auprc": metric(k4_ccsh, "binary_auprc"),
            "ece": metric(k4_ccsh, "binary_ece"),
            "answerability_auc": "",
            "interpretability": "medium",
            "cost_proxy_seconds": "",
            "evidence": "readiness audit missing PadChest/MIMIC-CXR/NIH local external data directories",
            "decision": "Blocked until external dataset and label-manifest eligibility are resolved.",
        },
    ]
    return rows


def main() -> None:
    rows = build_rows()
    columns = [
        "candidate",
        "data_engine",
        "encoder_module",
        "readout",
        "status",
        "backbone_run_id",
        "backbone_best_val_loss",
        "ccsh_auc",
        "auprc",
        "ece",
        "answerability_auc",
        "interpretability",
        "cost_proxy_seconds",
        "evidence",
        "decision",
    ]
    write_csv(OUT_CSV, rows, columns)
    OUT_MD.write_text(
        "# VSL-CXR Phase 5 Candidate Results\n\n"
        "Generated from formal run, CCSH/AUCH readout, and AUCH-only tables. "
        "`component_completed` means the candidate's v5-specified backbone/readout composition is available from completed exact rows; "
        "it does not imply external validation or locked-final multi-seed closure.\n\n"
        + md_table(rows, columns)
        + "\n",
        encoding="utf-8",
    )
    print(f"rows={len(rows)}")
    print(f"component_completed={sum(1 for row in rows if row.get('status') == 'component_completed')}")
    print(f"formal_training_completed={sum(1 for row in rows if row.get('status') == 'formal_training_completed')}")
    print(f"blocked_external_data={sum(1 for row in rows if row.get('status') == 'blocked_external_data')}")
    print(f"csv={OUT_CSV.relative_to(ROOT).as_posix()}")
    print(f"md={OUT_MD.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
