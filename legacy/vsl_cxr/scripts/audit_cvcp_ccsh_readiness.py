"""Audit repository readiness for the CVCP/CCSH full experiment plan."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "outputs" / "final_tables" / "cvcp_ccsh_requirement_ledger.csv"
DEFAULT_MD = ROOT / "docs" / "cvcp_ccsh_readiness_audit.md"
DEFAULT_CSV = ROOT / "outputs" / "final_tables" / "cvcp_ccsh_readiness_audit.csv"
MODEL_ROOT = Path("H:/Xiyao_Wang/001_models")

COLUMNS = ["area", "item", "status", "evidence", "notes"]

SCRIPT_ANALOGS: dict[str, list[str]] = {
    "scripts/generate_cvcp_curriculum.py": [
        "scripts/build_curriculum_v2_schedule.py",
        "scripts/generate_curriculum_v2_instructions.py",
        "scripts/build_progressive_mixture_schedule.py",
    ],
    "scripts/generate_sameq_cf_compatible.py": [
        "scripts/generate_sameq_shuf_pairs.py",
        "scripts/generate_ab_swap_jsonl.py",
    ],
    "scripts/generate_shuf_k_cf_compatible.py": [
        "scripts/generate_multi_negative_shuf.py",
        "scripts/generate_ab_swap_jsonl.py",
        "scripts/build_mined_shuf_instructions.py",
    ],
    "scripts/generate_ccsh_statements.py": [
        "models/clinical_consistency_head.py",
        "scripts/train_case_study_module_ablation.py",
    ],
    "scripts/generate_ceq_targets.py": [
        "models/clinical_evidence_query.py",
        "scripts/smoke_case_study_modules.py",
    ],
    "scripts/audit_instruction_leakage_v3.py": [
        "scripts/audit_instruction_leakage_v2.py",
        "scripts/audit_curriculum_leakage_cases.py",
    ],
    "scripts/audit_false_hard_negatives.py": [
        "scripts/audit_hard_negative_quality.py",
        "scripts/mine_hard_negatives_from_embeddings.py",
    ],
    "scripts/train_qwen3vl_cvcp.py": [
        "scripts/train_qwen3vl_curriculum_v2.py",
        "scripts/train_qwen3vl_clinical_instruction.py",
    ],
    "scripts/train_qwen3vl_sameq_ccsh.py": [
        "scripts/train_qwen3vl_clinical_instruction.py",
        "scripts/train_case_study_module_ablation.py",
        "models/clinical_consistency_head.py",
    ],
    "scripts/train_qwen3vl_shufk_ccsh.py": [
        "scripts/train_qwen3vl_clinical_instruction.py",
        "scripts/train_case_study_module_ablation.py",
        "models/clinical_consistency_head.py",
    ],
    "scripts/train_ceq_ccsh.py": [
        "scripts/train_case_study_module_ablation.py",
        "models/clinical_evidence_query.py",
        "models/clinical_consistency_head.py",
    ],
    "scripts/train_hnmb_ccsh.py": [
        "scripts/train_case_study_module_ablation.py",
        "models/hard_negative_memory_bank.py",
        "models/clinical_consistency_head.py",
    ],
    "scripts/train_vlm_teacher_comparison.py": [
        "scripts/train_qwen3vl_clinical_instruction.py",
    ],
    "scripts/eval_locked_final_suite.py": [
        "scripts/summarize_case_study_full_execution.py",
        "scripts/summarize_next_stage_results.py",
    ],
    "scripts/eval_external_dataset.py": [
        "scripts/eval_nih_external.py",
        "scripts/evaluate_qwen3vl_lp_transfer.py",
        "scripts/run_nih_full_transfer.py",
    ],
    "scripts/eval_ccsh_consistency.py": [
        "scripts/train_case_study_module_ablation.py",
        "outputs/final_tables/module_ablation_results.csv",
    ],
    "scripts/eval_ceq_attention.py": [
        "models/clinical_evidence_query.py",
        "outputs/final_tables/module_ablation_results.csv",
    ],
    "scripts/eval_ab_swap.py": [
        "scripts/generate_ab_swap_jsonl.py",
        "scripts/evaluate_qwen3vl_counterfactual_diagnostics.py",
    ],
    "scripts/eval_hard_shuffle.py": [
        "scripts/evaluate_qwen3vl_visual_dependence.py",
    ],
    "scripts/eval_calibration_auprc.py": [
        "evaluation/metrics.py",
        "outputs/final_tables/next_stage_calibration_auprc.csv",
    ],
    "scripts/bootstrap_locked_comparison.py": [
        "scripts/bootstrap_auc_ci.py",
        "scripts/paired_bootstrap_method_delta.py",
    ],
}

MODEL_ITEMS = {
    "Qwen3VL current main": ["qwen3-vl-2b-thinking-new", "Qwen3-VL-4B-Instruct", "Qwen3-VL-8B-Instruct"],
    "InternVL comparator": ["InternVL2_5-1B", "InternVL3_5-1B", "InternVL3_5-2B", "InternVL3_5-4B", "InternVL3_5-8B"],
    "LLaVA/Llama vision comparator": ["Llama-3.2-11B-Vision-Instruct"],
    "Qwen3.5 text scaffold": ["Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B"],
    "Qwen-Coder scaffold": ["Qwen2.5-Coder-7B-Instruct"],
}


def repo_rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def add(rows: list[dict[str, str]], area: str, item: str, status: str, evidence: str = "", notes: str = "") -> None:
    rows.append({"area": area, "item": item, "status": status, "evidence": evidence, "notes": notes})


def audit_scripts(rows: list[dict[str, str]], ledger_rows: list[dict[str, str]]) -> None:
    scripts = [row["name"] for row in ledger_rows if row.get("type") == "script"]
    for script in scripts:
        exact = ROOT / script
        if exact.exists():
            add(rows, "target_script", script, "exact_exists", script, "Exact target-plan entry point exists.")
            continue
        analogs = [path for path in SCRIPT_ANALOGS.get(script, []) if (ROOT / path).exists()]
        if analogs:
            add(
                rows,
                "target_script",
                script,
                "missing_exact_reusable_analogs",
                "; ".join(analogs),
                "Create target-named wrapper or adapt analogs; analog existence is not protocol completion.",
            )
        else:
            add(rows, "target_script", script, "missing_no_known_analog", "", "Needs new implementation.")


def audit_models(rows: list[dict[str, str]]) -> None:
    for item, candidates in MODEL_ITEMS.items():
        found = [name for name in candidates if (MODEL_ROOT / name).exists()]
        status = "available" if found else "missing"
        add(
            rows,
            "model",
            item,
            status,
            "; ".join((MODEL_ROOT / name).as_posix() for name in found),
            "Local model-directory audit only; each model still needs a component/GPU smoke before formal comparison.",
        )


def audit_data(rows: list[dict[str, str]], ledger_rows: list[dict[str, str]]) -> None:
    for row in ledger_rows:
        if row.get("type") == "dataset":
            add(rows, "dataset", row["name"], row.get("status", ""), row.get("evidence", ""), row.get("notes", ""))


def audit_reusable_artifacts(rows: list[dict[str, str]]) -> None:
    full_status = read_csv(ROOT / "outputs" / "final_tables" / "case_study_full_execution_status.csv")
    if full_status:
        completed = [row for row in full_status if row.get("train_status") == "completed" and not row.get("missing_required")]
        families = sorted({row.get("family", "") for row in completed})
        add(
            rows,
            "reusable_artifact",
            "case-study multiseed stability/downstream",
            "complete_for_prior_protocol",
            "outputs/final_tables/case_study_full_execution_status.csv",
            f"{len(completed)}/{len(full_status)} rows complete for families: {', '.join(families)}. Reusable for baselines, not module-combo completion.",
        )
    extra_status = read_csv(ROOT / "outputs" / "final_tables" / "case_study_extra_execution_status.csv")
    if extra_status:
        complete = [row for row in extra_status if row.get("status") in {"completed", "completed_existing"}]
        add(
            rows,
            "reusable_artifact",
            "case-study extra curriculum/embedding/module queue",
            "complete_for_prior_protocol",
            "outputs/final_tables/case_study_extra_execution_status.csv",
            f"{len(complete)}/{len(extra_status)} rows complete; includes curriculum-v2 long training and embedding exports.",
        )
    modules = read_csv(ROOT / "outputs" / "final_tables" / "module_ablation_results.csv")
    if modules:
        complete = [row for row in modules if row.get("status") == "completed"]
        add(
            rows,
            "reusable_artifact",
            "module ablation head evidence",
            "complete_embedding_level",
            "outputs/final_tables/module_ablation_results.csv",
            f"{len(complete)}/{len(modules)} CEQ/AUCH/HNMB/DRA/CCSH/CDCS rows complete; not full end-to-end module-combo training.",
        )
    next_audit = read_csv(ROOT / "outputs" / "final_tables" / "next_stage_completion_audit.csv")
    if next_audit:
        status_counts = Counter(row.get("status", "") for row in next_audit)
        add(
            rows,
            "reusable_artifact",
            "next-stage 39-run package audit",
            "complete_for_prior_protocol" if status_counts.get("completed") == len(next_audit) else "mixed",
            "outputs/final_tables/next_stage_completion_audit.csv",
            ", ".join(f"{key}={value}" for key, value in sorted(status_counts.items())),
        )
    package_status = read_json(ROOT / "outputs" / "final_tables" / "next_stage_run_package_status.json")
    if package_status:
        add(
            rows,
            "reusable_artifact",
            "next-stage run package status",
            "available",
            "outputs/final_tables/next_stage_run_package_status.json",
            "Use for old run package provenance while creating CVCP-specific summary tables.",
        )


def audit_outputs(rows: list[dict[str, str]], ledger_rows: list[dict[str, str]]) -> None:
    for row in ledger_rows:
        if row.get("type") == "output":
            add(rows, "target_output", row["name"], row.get("status", ""), row.get("evidence", ""), row.get("notes", ""))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in COLUMNS})


def md_table(rows: list[dict[str, str]]) -> list[str]:
    lines = ["| " + " | ".join(COLUMNS) + " |", "| " + " | ".join("---" for _ in COLUMNS) + " |"]
    for row in rows:
        values = [str(row.get(key, "")).replace("|", "\\|").replace("\n", " ") for key in COLUMNS]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_md(path: Path, rows: list[dict[str, str]], csv_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter((row["area"], row["status"]) for row in rows)
    lines = [
        "# VIVID-Med CVCP/CCSH Readiness Audit",
        "",
        f"Machine-readable CSV: `{repo_rel(csv_path)}`",
        "",
        "This audit turns the generated requirement ledger into execution categories. It does not mark CVCP/CCSH experiments complete unless the status explicitly says so.",
        "",
        "## Status Counts",
        "",
        "| Area | Status | Count |",
        "| --- | --- | ---: |",
    ]
    for (area, status), count in sorted(counts.items()):
        lines.append(f"| {area} | {status} | {count} |")
    lines.extend(["", "## Audit Rows", ""])
    lines.extend(md_table(rows))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ledger_rows = read_csv(args.ledger)
    rows: list[dict[str, str]] = []
    audit_scripts(rows, ledger_rows)
    audit_outputs(rows, ledger_rows)
    audit_data(rows, ledger_rows)
    audit_models(rows)
    audit_reusable_artifacts(rows)
    write_csv(args.csv, rows)
    write_md(args.md, rows, args.csv)
    counts = Counter(row["status"] for row in rows)
    print(f"wrote_rows={len(rows)}")
    for status, count in sorted(counts.items()):
        print(f"{status}={count}")
    print(f"md={repo_rel(args.md)}")
    print(f"csv={repo_rel(args.csv)}")


if __name__ == "__main__":
    main()
