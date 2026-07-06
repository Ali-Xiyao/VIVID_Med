"""Audit local teacher-model compatibility for the CVCP/CCSH comparison plan."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path("H:/Xiyao_Wang/001_models")
FINAL_DIR = ROOT / "outputs" / "final_tables"

MODEL_SPECS = [
    ("Qwen3VL-CVCP-2B", "Qwen3VL", MODEL_ROOT / "qwen3-vl-2b-thinking-new", "formal_qwen3vl_trainer_supported"),
    ("Qwen3VL-CVCP-4B", "Qwen3VL", MODEL_ROOT / "Qwen3-VL-4B-Instruct", "same_architecture_needs_vram_smoke"),
    ("Qwen3VL-CVCP-8B", "Qwen3VL", MODEL_ROOT / "Qwen3-VL-8B-Instruct", "same_architecture_needs_vram_smoke"),
    ("InternVL-CVCP-1B", "InternVL", MODEL_ROOT / "InternVL3_5-1B", "requires_internvl_specific_trainer"),
    ("InternVL-CVCP-2B", "InternVL", MODEL_ROOT / "InternVL3_5-2B", "requires_internvl_specific_trainer"),
    ("LLaVA-CVCP", "LLaVA/Llama", MODEL_ROOT / "Llama-3.2-11B-Vision-Instruct", "requires_llama_vision_specific_trainer"),
    ("Qwen3.5-scaffold-2B", "Text scaffold", MODEL_ROOT / "Qwen3.5-2B", "text_only_no_vision_tower"),
    ("Qwen3.5-scaffold-4B", "Text scaffold", MODEL_ROOT / "Qwen3.5-4B", "text_only_no_vision_tower"),
    ("Qwen-Coder-scaffold", "Text scaffold", MODEL_ROOT / "Qwen2.5-Coder-7B-Instruct", "text_only_no_vision_tower"),
]


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def write_md_table(path: Path, title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        values = [str(row.get(key, "")).replace("|", "\\|").replace("\n", " ") for key in columns]
        lines.append("| " + " | ".join(values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_one(run_id: str, family: str, path: Path, protocol_note: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": run_id,
        "family": family,
        "model_path": path.as_posix(),
        "exists": path.exists(),
        "model_type": "",
        "architectures": "",
        "config_status": "missing_path",
        "processor_status": "not_attempted",
        "current_trainer_status": "not_supported",
        "decision": protocol_note,
    }
    if not path.exists():
        return row
    try:
        from transformers import AutoConfig, AutoProcessor

        config = AutoConfig.from_pretrained(str(path), trust_remote_code=True)
        row["model_type"] = str(getattr(config, "model_type", ""))
        row["architectures"] = ";".join(str(item) for item in (getattr(config, "architectures", None) or []))
        row["config_status"] = "ok"
        try:
            processor = AutoProcessor.from_pretrained(str(path), trust_remote_code=True)
            row["processor_status"] = f"ok:{type(processor).__name__}"
        except Exception as exc:  # noqa: BLE001 - preserve compatibility failure.
            row["processor_status"] = f"failed:{type(exc).__name__}:{exc}"
    except Exception as exc:  # noqa: BLE001 - preserve compatibility failure.
        row["config_status"] = f"failed:{type(exc).__name__}:{exc}"
        return row

    if row["model_type"] == "qwen3_vl":
        row["current_trainer_status"] = "supported_by_train_qwen3vl_clinical_instruction"
        if "4B" in run_id or "8B" in run_id:
            row["decision"] = protocol_note
        else:
            row["decision"] = "formal_2b_cvcp_queue_running"
    elif "qwen2_5_vl" in row["model_type"] or "qwen2_vl" in row["model_type"]:
        row["current_trainer_status"] = "nearby_vl_architecture_requires_adapter_audit"
    elif "intern" in row["model_type"].lower():
        row["current_trainer_status"] = "requires_internvl_specific_trainer"
    elif "llama" in row["model_type"].lower() or "mllama" in row["model_type"].lower():
        row["current_trainer_status"] = "requires_llama_vision_specific_trainer"
    elif row["processor_status"].startswith("ok") and "vision" not in row["model_type"].lower():
        row["current_trainer_status"] = "text_only_scaffold_not_vlm_teacher"
    return row


def main() -> None:
    rows = [audit_one(*spec) for spec in MODEL_SPECS]
    columns = [
        "run_id",
        "family",
        "model_path",
        "exists",
        "model_type",
        "architectures",
        "config_status",
        "processor_status",
        "current_trainer_status",
        "decision",
    ]
    write_csv(FINAL_DIR / "model_comparison_results.csv", rows, columns)
    write_md_table(FINAL_DIR / "model_comparison_results.md", "Model Comparison Results", rows, columns)
    (FINAL_DIR / "model_comparison_results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "supported": sum(1 for row in rows if "supported" in row["current_trainer_status"])}, indent=2))


if __name__ == "__main__":
    main()
