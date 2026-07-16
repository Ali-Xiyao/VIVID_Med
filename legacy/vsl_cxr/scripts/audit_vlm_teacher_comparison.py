"""Build the VSL-CXR v5 Phase 7 teacher-comparison audit table.

This script is intentionally audit-only. Phase 7 requires model-family
comparisons, but the current runnable training stack is Qwen3-VL specific.
Rows for InternVL, LLaVA/Mllama, and text-only scaffolds are therefore bounded
unless a matching VSL trainer/adapter exists.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path("H:/Xiyao_Wang/001_models")
FINAL_DIR = ROOT / "outputs" / "final_tables"


MODEL_SPECS = [
    ("Qwen3VL-VSL-2B", "Qwen3-VL", "VLM", MODEL_ROOT / "qwen3-vl-2b-thinking-new"),
    ("Qwen3VL-VSL-4B", "Qwen3-VL", "VLM", MODEL_ROOT / "Qwen3-VL-4B-Instruct"),
    ("Qwen3VL-VSL-8B", "Qwen3-VL", "VLM", MODEL_ROOT / "Qwen3-VL-8B-Instruct"),
    ("InternVL-VSL-1B", "InternVL", "VLM", MODEL_ROOT / "InternVL3_5-1B"),
    ("InternVL-VSL-2B", "InternVL", "VLM", MODEL_ROOT / "InternVL3_5-2B"),
    ("LLaVA-VSL", "LLaVA/Llama-based VLM", "VLM", MODEL_ROOT / "Llama-3.2-11B-Vision-Instruct"),
    ("Medical-VLM", "medical VLM", "VLM", MODEL_ROOT / "medical-vlm-not-found"),
    ("Qwen3.5-scaffold-2B", "Qwen3.5 text-only", "text scaffold", MODEL_ROOT / "Qwen3.5-2B"),
    ("Qwen3.5-scaffold-4B", "Qwen3.5 text-only", "text scaffold", MODEL_ROOT / "Qwen3.5-4B"),
    ("Qwen-Coder-scaffold", "Qwen-Coder", "text scaffold", MODEL_ROOT / "Qwen2.5-Coder-7B-Instruct"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_row(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    return {}


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


def audit_model(run_id: str, family: str, model_type: str, path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": run_id,
        "family": family,
        "model_family_type": model_type,
        "model_path": path.as_posix(),
        "path_exists": path.exists(),
        "hf_model_type": "",
        "architectures": "",
        "config_status": "missing_path",
        "processor_status": "not_attempted",
        "current_vsl_trainer_status": "not_supported",
        "adapter_boundary": "",
    }
    if not path.exists():
        row["adapter_boundary"] = "model_path_missing"
        return row

    try:
        from transformers import AutoConfig, AutoProcessor

        config = AutoConfig.from_pretrained(str(path), trust_remote_code=True)
        row["hf_model_type"] = str(getattr(config, "model_type", ""))
        row["architectures"] = ";".join(str(item) for item in (getattr(config, "architectures", None) or []))
        row["config_status"] = "ok"
        try:
            processor = AutoProcessor.from_pretrained(str(path), trust_remote_code=True)
            row["processor_status"] = f"ok:{type(processor).__name__}"
        except Exception as exc:  # noqa: BLE001 - compatibility evidence belongs in table.
            row["processor_status"] = f"failed:{type(exc).__name__}:{exc}"
    except Exception as exc:  # noqa: BLE001 - compatibility evidence belongs in table.
        row["config_status"] = f"failed:{type(exc).__name__}:{exc}"
        row["adapter_boundary"] = "config_load_failed"
        return row

    hf_type = row["hf_model_type"].lower()
    if hf_type == "qwen3_vl":
        row["current_vsl_trainer_status"] = "supported_by_qwen3vl_vsl_stack"
        row["adapter_boundary"] = "current_main_family"
    elif "intern" in hf_type:
        row["current_vsl_trainer_status"] = "requires_internvl_specific_vsl_trainer"
        row["adapter_boundary"] = "blocked_adapter_missing"
    elif "llama" in hf_type or "mllama" in hf_type:
        row["current_vsl_trainer_status"] = "requires_llama_vision_specific_vsl_trainer"
        row["adapter_boundary"] = "blocked_adapter_missing"
    elif model_type == "text scaffold":
        row["current_vsl_trainer_status"] = "requires_text_only_vsl_scaffold_trainer"
        row["adapter_boundary"] = "blocked_text_scaffold_trainer_missing"
    else:
        row["adapter_boundary"] = "blocked_unmapped_model_family"
    return row


def build_phase7_rows(model_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formal_rows = read_csv(FINAL_DIR / "vsl_cxr_formal_run_results.csv")
    external_rows = read_csv(FINAL_DIR / "vsl_cxr_external_results.csv")
    ccsh_rows = read_csv(FINAL_DIR / "vsl_cxr_ccsh_results.csv")

    qwen_core = find_row(formal_rows, "run_id", "VSL-CXR-B5-SAMEQ-K4")
    qwen_core_lp = find_row(formal_rows, "run_id", "VSL-CXR-P6-LP-VSL-CORE")
    external_core = find_row(external_rows, "run", "VSL-Core")
    ccsh_core = find_row(ccsh_rows, "variant", "ccsh_sameq_k4")

    model_by_run = {row["run_id"]: row for row in model_rows}
    qwen2b = model_by_run.get("Qwen3VL-VSL-2B", {})
    intern = model_by_run.get("InternVL-VSL-1B", {})
    llava = model_by_run.get("LLaVA-VSL", {})
    qwen35 = model_by_run.get("Qwen3.5-scaffold-2B", {})
    coder = model_by_run.get("Qwen-Coder-scaffold", {})

    rows: list[dict[str, Any]] = [
        {
            "run": "Qwen3VL-VSL-smoke",
            "model": "Qwen3-VL 2B",
            "phase7_stage": "smoke",
            "status": "completed_by_current_main",
            "steps": qwen_core.get("global_step", ""),
            "chexpert_auc": "",
            "external_auc": "",
            "vsl_acc": "",
            "hard_shuffle": "",
            "ccsh_auc": "",
            "cost_seconds": qwen_core.get("elapsed_seconds", ""),
            "decision": "Qwen3-VL VSL stack is already confirmed by completed v5 formal VSL-Core run.",
            "evidence": qwen_core.get("metrics_final", ""),
            "adapter_boundary": qwen2b.get("adapter_boundary", ""),
        },
        {
            "run": "InternVL-VSL-smoke",
            "model": "InternVL",
            "phase7_stage": "smoke",
            "status": "blocked_adapter_missing",
            "decision": "Local model exists, but the current v5 trainer is Qwen3-VL specific and InternVL processor/trainer adapter is missing.",
            "evidence": intern.get("processor_status", ""),
            "adapter_boundary": intern.get("adapter_boundary", ""),
        },
        {
            "run": "LLaVA-VSL-smoke",
            "model": "LLaVA/Llama-based VLM",
            "phase7_stage": "smoke",
            "status": "blocked_adapter_missing",
            "decision": "Local Mllama model exists, but a Llama-vision VSL trainer adapter is missing.",
            "evidence": llava.get("processor_status", ""),
            "adapter_boundary": llava.get("adapter_boundary", ""),
        },
        {
            "run": "Qwen3.5-scaffold-smoke",
            "model": "Qwen3.5 text-only",
            "phase7_stage": "smoke",
            "status": "blocked_text_scaffold_trainer_missing",
            "decision": "Text-only model exists, but v5 scaffold control requires a separate non-vision VSL trainer.",
            "evidence": qwen35.get("processor_status", ""),
            "adapter_boundary": qwen35.get("adapter_boundary", ""),
        },
        {
            "run": "Qwen3VL-VSL",
            "model": "Qwen3-VL 2B",
            "phase7_stage": "full",
            "status": "completed_current_main_only",
            "steps": qwen_core.get("global_step", ""),
            "chexpert_auc": qwen_core_lp.get("macro_auc", ""),
            "external_auc": external_core.get("macro_auc", ""),
            "vsl_acc": "",
            "hard_shuffle": "",
            "ccsh_auc": ccsh_core.get("binary_auc", ""),
            "cost_seconds": qwen_core.get("elapsed_seconds", ""),
            "decision": "Current main VSL-Core evidence is complete; cross-family comparison remains bounded until other trainers exist.",
            "evidence": "; ".join(
                value
                for value in [
                    qwen_core.get("metrics_final", ""),
                    qwen_core_lp.get("metrics_final", ""),
                    external_core.get("evidence", ""),
                    ccsh_core.get("metrics_final", ""),
                ]
                if value
            ),
            "adapter_boundary": qwen2b.get("adapter_boundary", ""),
        },
        {
            "run": "InternVL-VSL",
            "model": "InternVL",
            "phase7_stage": "full",
            "status": "blocked_until_smoke_adapter",
            "decision": "Full comparison depends on an InternVL-specific VSL trainer and smoke pass.",
            "evidence": intern.get("processor_status", ""),
            "adapter_boundary": intern.get("adapter_boundary", ""),
        },
        {
            "run": "LLaVA-VSL",
            "model": "LLaVA/Llama-based VLM",
            "phase7_stage": "full",
            "status": "blocked_until_smoke_adapter",
            "decision": "Full comparison depends on a Llama-vision VSL trainer and smoke pass.",
            "evidence": llava.get("processor_status", ""),
            "adapter_boundary": llava.get("adapter_boundary", ""),
        },
        {
            "run": "Qwen3.5-scaffold",
            "model": "Qwen3.5 text-only",
            "phase7_stage": "full",
            "status": "blocked_text_scaffold_trainer_missing",
            "decision": "VSL-Core text-only scaffold needs a separate trainer because the current stack expects image inputs and Qwen3-VL processors.",
            "evidence": qwen35.get("processor_status", ""),
            "adapter_boundary": qwen35.get("adapter_boundary", ""),
        },
        {
            "run": "Qwen-Coder-scaffold",
            "model": "Qwen-Coder text-only",
            "phase7_stage": "full",
            "status": "blocked_text_scaffold_trainer_missing",
            "decision": "Historical Qwen-Coder scripts are not exact v5 VSL-Core scaffold evidence.",
            "evidence": coder.get("processor_status", ""),
            "adapter_boundary": coder.get("adapter_boundary", ""),
        },
    ]
    return rows


def main() -> None:
    model_rows = [audit_model(*spec) for spec in MODEL_SPECS]
    model_columns = [
        "run_id",
        "family",
        "model_family_type",
        "model_path",
        "path_exists",
        "hf_model_type",
        "architectures",
        "config_status",
        "processor_status",
        "current_vsl_trainer_status",
        "adapter_boundary",
    ]
    phase7_rows = build_phase7_rows(model_rows)
    phase7_columns = [
        "run",
        "model",
        "phase7_stage",
        "status",
        "steps",
        "chexpert_auc",
        "external_auc",
        "vsl_acc",
        "hard_shuffle",
        "ccsh_auc",
        "cost_seconds",
        "decision",
        "evidence",
        "adapter_boundary",
    ]

    write_csv(FINAL_DIR / "vsl_cxr_teacher_model_audit.csv", model_rows, model_columns)
    write_md_table(
        FINAL_DIR / "vsl_cxr_teacher_model_audit.md",
        "VSL-CXR Phase 7 Teacher Model Audit",
        model_rows,
        model_columns,
    )
    write_csv(FINAL_DIR / "vsl_cxr_teacher_comparison_results.csv", phase7_rows, phase7_columns)
    write_md_table(
        FINAL_DIR / "vsl_cxr_teacher_comparison_results.md",
        "VSL-CXR Phase 7 Teacher Comparison Results",
        phase7_rows,
        phase7_columns,
        note="Rows are formal v5 Phase 7 evidence. Blocked rows are bounded by missing VSL-specific trainer adapters, not by missing local model directories.",
    )
    (FINAL_DIR / "vsl_cxr_teacher_comparison_results.json").write_text(
        json.dumps({"model_audit": model_rows, "phase7_results": phase7_rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "model_rows": len(model_rows),
                "phase7_rows": len(phase7_rows),
                "completed_or_current_main": sum(1 for row in phase7_rows if row["status"].startswith("completed")),
                "blocked": sum(1 for row in phase7_rows if row["status"].startswith("blocked")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
