"""Build the VSL-CXR v5 locked final comparison table."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs" / "final_tables"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    return {}


def best(rows: list[dict[str, str]], key: str, metric: str, reverse: bool = True) -> dict[str, str]:
    candidates = [row for row in rows if row.get(key) and row.get(metric)]
    if not candidates:
        return {}
    return sorted(candidates, key=lambda row: float(row.get(metric) or 0), reverse=reverse)[0]


def fmt_single(value: str) -> str:
    return f"{value} (single seed)" if value else ""


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def write_md_table(path: Path, title: str, rows: list[dict[str, Any]], columns: list[str], note: str = "") -> None:
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


def main() -> None:
    formal = read_csv(FINAL_DIR / "vsl_cxr_formal_run_results.csv")
    ceq = read_csv(FINAL_DIR / "vsl_cxr_ceq_results.csv")
    ccsh = read_csv(FINAL_DIR / "vsl_cxr_ccsh_results.csv")
    candidates = read_csv(FINAL_DIR / "vsl_cxr_phase5_candidate_results.csv")
    external = read_csv(FINAL_DIR / "vsl_cxr_external_results.csv")
    teacher = read_csv(FINAL_DIR / "vsl_cxr_teacher_comparison_results.csv")

    b0 = find(formal, "run_id", "VSL-CXR-B0-RAW-VISION")
    basic = find(formal, "run_id", "VSL-CXR-B1-BASIC-QA")
    sameq = find(formal, "run_id", "VSL-CXR-B3-SAMEQ")
    hnmb = find(formal, "run_id", "VSL-CXR-B6-SAMEQ-HNMB")
    p6_sameq = find(formal, "run_id", "VSL-CXR-P6-LP-SAMEQ")
    p6_full = find(formal, "run_id", "VSL-CXR-P6-LP-VSL-FULL")
    vsl_full = find(formal, "run_id", "VSL-CXR-P5-VSL-FULL")

    ceq_region = best(ceq, "variant", "binary_auc")
    ccsh_ceq = find(ccsh, "variant", "ccsh_ceq") or best(ccsh, "variant", "binary_auc")
    integrated_full = find(candidates, "candidate", "VSL-Full")
    teacher_qwen = find(teacher, "run", "Qwen3VL-VSL")

    ext_raw = find(external, "run", "Raw")
    ext_sameq = find(external, "run", "SAMEQ")
    ext_full = find(external, "run", "VSL-Full")

    rows: list[dict[str, Any]] = [
        {
            "family": "Raw",
            "candidates": "Raw Qwen3-VL",
            "finalist": "Raw Qwen3-VL vision LP",
            "seeds": 1,
            "chexpert_auc_mean_std": fmt_single(b0.get("macro_auc", "")),
            "external_auc_mean_std": "main external blocked",
            "nih_appendix_auc": ext_raw.get("macro_auc", ""),
            "vsl_auc": "",
            "hard_shuffle": "",
            "ccsh_auc": "",
            "ece": ext_raw.get("ece", ""),
            "cost_seconds": b0.get("elapsed_seconds", ""),
            "final_role": "baseline",
            "status": "locked_single_seed",
            "evidence": b0.get("metrics_final", ""),
            "limitations": "Raw baseline has LP evidence but no VSL sufficiency or CCSH readout.",
        },
        {
            "family": "QA",
            "candidates": "Basic-QA / CF-QA",
            "finalist": "Basic-QA",
            "seeds": 1,
            "chexpert_auc_mean_std": "",
            "external_auc_mean_std": "not evaluated",
            "nih_appendix_auc": "",
            "vsl_auc": "",
            "hard_shuffle": "",
            "ccsh_auc": "",
            "ece": "",
            "cost_seconds": basic.get("elapsed_seconds", ""),
            "final_role": "baseline",
            "status": "locked_training_loss_only",
            "evidence": basic.get("metrics_final", ""),
            "limitations": "Selected by lower v5 instruction best-val loss; no deployable LP/external/readout row was run for QA.",
        },
        {
            "family": "SAMEQ",
            "candidates": "SAMEQ / SAMEQ-CF",
            "finalist": "SAMEQ",
            "seeds": 1,
            "chexpert_auc_mean_std": fmt_single(p6_sameq.get("macro_auc", "")),
            "external_auc_mean_std": "main external blocked",
            "nih_appendix_auc": ext_sameq.get("macro_auc", ""),
            "vsl_auc": "",
            "hard_shuffle": "",
            "ccsh_auc": find(ccsh, "variant", "ccsh_sameq").get("binary_auc", ""),
            "ece": ext_sameq.get("ece", ""),
            "cost_seconds": sameq.get("elapsed_seconds", ""),
            "final_role": "core",
            "status": "locked_single_seed",
            "evidence": sameq.get("metrics_final", ""),
            "limitations": "Main external blocked; hard-shuffle delta not exported in current v5 table.",
        },
        {
            "family": "Hard Negative",
            "candidates": "SAMEQ-K / SAMEQ-HNMB",
            "finalist": "SAMEQ-HNMB",
            "seeds": 1,
            "chexpert_auc_mean_std": "",
            "external_auc_mean_std": "main external blocked",
            "nih_appendix_auc": "",
            "vsl_auc": "",
            "hard_shuffle": "",
            "ccsh_auc": find(ccsh, "variant", "ccsh_hnmb").get("binary_auc", ""),
            "ece": find(ccsh, "variant", "ccsh_hnmb").get("binary_ece", ""),
            "cost_seconds": hnmb.get("elapsed_seconds", ""),
            "final_role": "hard-negative",
            "status": "locked_single_seed",
            "evidence": hnmb.get("metrics_final", ""),
            "limitations": "Selected over SAMEQ-K by lower training loss and stronger CCSH AUC; Phase 6 LP used VSL-Core/SAMEQ-K rather than HNMB.",
        },
        {
            "family": "CEQ",
            "candidates": "CEQ variants",
            "finalist": "CEQ-region",
            "seeds": 1,
            "chexpert_auc_mean_std": "",
            "external_auc_mean_std": "not directly evaluated",
            "nih_appendix_auc": "",
            "vsl_auc": ceq_region.get("binary_auc", ""),
            "hard_shuffle": "",
            "ccsh_auc": ccsh_ceq.get("binary_auc", ""),
            "ece": ceq_region.get("binary_ece", ""),
            "cost_seconds": ceq_region.get("elapsed_seconds", ""),
            "final_role": "evidence",
            "status": "locked_single_seed",
            "evidence": ceq_region.get("metrics_final", ""),
            "limitations": "CEQ external row is a backbone proxy, not a CEQ readout classifier.",
        },
        {
            "family": "CCSH",
            "candidates": "CCSH / CCSH+AUCH",
            "finalist": "CCSH-CEQ",
            "seeds": 1,
            "chexpert_auc_mean_std": "",
            "external_auc_mean_std": "not directly evaluated",
            "nih_appendix_auc": "",
            "vsl_auc": "",
            "hard_shuffle": "",
            "ccsh_auc": ccsh_ceq.get("binary_auc", ""),
            "ece": ccsh_ceq.get("binary_ece", ""),
            "cost_seconds": ccsh_ceq.get("elapsed_seconds", ""),
            "final_role": "readout",
            "status": "locked_single_seed",
            "evidence": ccsh_ceq.get("metrics_final", ""),
            "limitations": "Selected by primary CCSH binary AUC; AUCH-CEQ-CCSH remains best AUPRC row.",
        },
        {
            "family": "VSL Integrated",
            "candidates": "VSL-Lite/Core/HNMB/CEQ/Full",
            "finalist": "VSL-Full",
            "seeds": 1,
            "chexpert_auc_mean_std": fmt_single(p6_full.get("macro_auc", "")),
            "external_auc_mean_std": "main external blocked",
            "nih_appendix_auc": ext_full.get("macro_auc", ""),
            "vsl_auc": "",
            "hard_shuffle": "",
            "ccsh_auc": integrated_full.get("ccsh_auc", ""),
            "ece": ext_full.get("ece", ""),
            "cost_seconds": vsl_full.get("elapsed_seconds", ""),
            "final_role": "final",
            "status": "locked_single_seed_with_external_boundary",
            "evidence": integrated_full.get("evidence", ""),
            "limitations": "Best CheXpert LP among Phase 6 LP rows, but NIH appendix macro-AUC favors SAMEQ/Core and main external remains blocked.",
        },
        {
            "family": "Teacher model",
            "candidates": "Qwen/InternVL/LLaVA/text",
            "finalist": "Qwen3-VL 2B",
            "seeds": 1,
            "chexpert_auc_mean_std": teacher_qwen.get("chexpert_auc", ""),
            "external_auc_mean_std": "main external blocked",
            "nih_appendix_auc": teacher_qwen.get("external_auc", ""),
            "vsl_auc": "",
            "hard_shuffle": "",
            "ccsh_auc": teacher_qwen.get("ccsh_auc", ""),
            "ece": "",
            "cost_seconds": teacher_qwen.get("cost_seconds", ""),
            "final_role": "teacher",
            "status": "locked_current_main_bounded",
            "evidence": teacher_qwen.get("evidence", ""),
            "limitations": "Cross-family rows are blocked by missing InternVL/LLaVA/text-scaffold VSL trainers.",
        },
    ]

    columns = [
        "family",
        "candidates",
        "finalist",
        "seeds",
        "chexpert_auc_mean_std",
        "external_auc_mean_std",
        "nih_appendix_auc",
        "vsl_auc",
        "hard_shuffle",
        "ccsh_auc",
        "ece",
        "cost_seconds",
        "final_role",
        "status",
        "evidence",
        "limitations",
    ]
    note = (
        "Locked finalists are selected from current v5 evidence only. All numeric means are single-seed unless noted; "
        "the v5 main-external requirement remains blocked, so NIH values are appendix/stress evidence."
    )
    write_csv(FINAL_DIR / "vsl_cxr_locked_final_comparison.csv", rows, columns)
    write_md_table(FINAL_DIR / "vsl_cxr_locked_final_comparison.md", "VSL-CXR Locked Final Comparison", rows, columns, note)

    summary = {
        "rows": len(rows),
        "single_seed_rows": sum(1 for row in rows if str(row["seeds"]) == "1"),
        "main_external_blocked_rows": sum(1 for row in rows if "blocked" in str(row["external_auc_mean_std"])),
        "final_integrated": "VSL-Full",
        "teacher_finalist": "Qwen3-VL 2B",
    }
    (FINAL_DIR / "vsl_cxr_locked_final_comparison.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
