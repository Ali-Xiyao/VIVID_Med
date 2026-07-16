"""Build VSL-CXR v5 Phase 8 casebook and visualization manifests."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs" / "final_tables"
VSL_DIR = ROOT / "outputs" / "instruction_data" / "vsl_cxr"


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


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


def first_by_label(rows: list[dict[str, Any]], label: str, n: int) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    for row in rows:
        if (row.get("sufficiency_label") or row.get("answer_type") or row.get("answer")) == label:
            picked.append(row)
            if len(picked) >= n:
                break
    return picked


def case_from_vsl(row: dict[str, Any], case_type: str, idx: int) -> dict[str, Any]:
    label = row.get("sufficiency_label") or row.get("answer_type") or row.get("answer", "")
    return {
        "case_id": f"{case_type}-{idx:03d}",
        "casebook": case_type,
        "dataset": row.get("source_version") or row.get("source") or "D6/D9 VSL",
        "image": row.get("image_path", ""),
        "statement": row.get("counterfactual_statement") or row.get("question") or row.get("evidence_query", ""),
        "true_label": label,
        "model_output": "casebook_input_only",
        "explanation": row.get("evidence_span") or row.get("finding") or "",
        "failure_type": "",
        "manual_note": "Requires human-facing image review before publication.",
        "evidence": row.get("instruction_id") or row.get("sample_id") or "",
    }


def build_casebook() -> list[dict[str, Any]]:
    d6_val = read_jsonl(VSL_DIR / "d6_vsl_4class_val.jsonl")
    d9_ceq_val = read_jsonl(VSL_DIR / "d9_ceq_targets_val.jsonl")
    d9_ccsh_val = read_jsonl(VSL_DIR / "d9_ccsh_pairs_val.jsonl")
    sameq_val = read_jsonl(ROOT / "outputs" / "instruction_data" / "next_stage" / "sameq_shuf_val.jsonl")
    external_rows = read_csv(FINAL_DIR / "vsl_cxr_external_results.csv")

    cases: list[dict[str, Any]] = []
    idx = 1
    for label, case_type in [
        ("support", "vsl_support"),
        ("contradict", "vsl_contradict"),
        ("uncertain", "vsl_uncertain"),
        ("insufficient", "vsl_insufficient"),
    ]:
        for row in first_by_label(d6_val, label, 3):
            cases.append(case_from_vsl(row, case_type, idx))
            idx += 1

    for row in sameq_val[:4]:
        cases.append(
            {
                "case_id": f"sameq_pair-{idx:03d}",
                "casebook": "sameq_pair",
                "dataset": row.get("source_version") or "SAMEQ validation",
                "image": row.get("image_path", ""),
                "statement": row.get("question", ""),
                "true_label": row.get("answer", ""),
                "model_output": "casebook_input_only",
                "explanation": f"negative_image={row.get('negative_image_path', '')}; finding={row.get('finding', '')}",
                "failure_type": "",
                "manual_note": "Verify same-question/different-image visual contrast before publication.",
                "evidence": row.get("instruction_id", ""),
            }
        )
        idx += 1

    for row in d6_val:
        if row.get("negative_image_path"):
            cases.append(case_from_vsl(row, "false_hard_negative_review", idx))
            idx += 1
            if sum(1 for case in cases if case["casebook"] == "false_hard_negative_review") >= 4:
                break
    if not any(case["casebook"] == "false_hard_negative_review" for case in cases):
        for row in first_by_label(d6_val, "contradict", 4):
            case = case_from_vsl(row, "false_hard_negative_review", idx)
            case["failure_type"] = "counterfactual_or_hard_negative_review"
            case["manual_note"] = "Fallback review case: current val row has no negative_image_path, so manually audit counterfactual validity from statement/image evidence."
            cases.append(case)
            idx += 1

    for row in d9_ccsh_val[:4]:
        cases.append(
            {
                "case_id": f"ccsh-{idx:03d}",
                "casebook": "ccsh_success_failure",
                "dataset": row.get("source_version") or "D9 CCSH validation",
                "image": row.get("image_path", ""),
                "statement": row.get("statement", ""),
                "true_label": row.get("binary_label", ""),
                "model_output": "readout_case_input",
                "explanation": f"finding={row.get('finding', '')}; state={row.get('state', '')}",
                "failure_type": "",
                "manual_note": "Pair with CCSH metrics before publication.",
                "evidence": row.get("statement_id") or row.get("sample_id") or "",
            }
        )
        idx += 1

    for row in d9_ceq_val[:4]:
        cases.append(
            {
                "case_id": f"ceq-attention-{idx:03d}",
                "casebook": "ceq_attention",
                "dataset": row.get("source_version") or "D9 CEQ validation",
                "image": row.get("image_path", ""),
                "statement": row.get("evidence_query", ""),
                "true_label": row.get("state", ""),
                "model_output": "attention_case_input",
                "explanation": f"expected_region={row.get('expected_region', '')}; finding={row.get('finding', '')}",
                "failure_type": "",
                "manual_note": "Attention assets exist under outputs/attention_maps when linked by sample.",
                "evidence": row.get("target_id") or row.get("sample_id") or "",
            }
        )
        idx += 1

    for row in external_rows:
        if row.get("external_dataset") == "NIH-appendix-1k" and row.get("worst_labels"):
            cases.append(
                {
                    "case_id": f"external-failure-{idx:03d}",
                    "casebook": "external_failure",
                    "dataset": row.get("external_dataset", ""),
                    "image": "",
                    "statement": f"{row.get('run', '')} transfer worst labels",
                    "true_label": row.get("worst_labels", ""),
                    "model_output": f"macro_auc={row.get('macro_auc', '')}; ece={row.get('ece', '')}",
                    "explanation": row.get("failure_cause", ""),
                    "failure_type": "domain_shift_or_label_mapping",
                    "manual_note": "NIH is appendix/stress only, not main external.",
                    "evidence": row.get("evidence", ""),
                }
            )
            idx += 1
    return cases


def build_visual_manifest(case_count: int) -> list[dict[str, Any]]:
    external_rows = read_csv(FINAL_DIR / "vsl_cxr_external_results.csv")
    ceq_rows = read_csv(FINAL_DIR / "vsl_cxr_ceq_results.csv")
    ccsh_rows = read_csv(FINAL_DIR / "vsl_cxr_ccsh_results.csv")
    attention_summary = ROOT / "outputs" / "attention_maps" / "summary.json"

    best_ceq = max(ceq_rows, key=lambda row: float(row.get("binary_auc") or 0), default={})
    best_ccsh = max(ccsh_rows, key=lambda row: float(row.get("binary_auc") or 0), default={})
    external_core = next((row for row in external_rows if row.get("run") == "VSL-Core"), {})

    return [
        {
            "figure": "Fig 1",
            "description": "VSL-CXR framework",
            "status": "figure_spec_ready",
            "evidence": "vivid_med_vsl_cxr_full_experiment_plan_v5.md",
            "notes": "Use D0-D9/CEQ/CCSH/AUCH pipeline from the source plan.",
        },
        {
            "figure": "Fig 2",
            "description": "SAMEQ examples",
            "status": "casebook_ready",
            "evidence": "outputs/final_tables/vsl_cxr_phase8_casebook.md",
            "notes": f"Casebook rows total={case_count}.",
        },
        {
            "figure": "Fig 3",
            "description": "support vs contradict examples",
            "status": "casebook_ready",
            "evidence": "outputs/final_tables/vsl_cxr_phase8_casebook.md",
            "notes": "D6 support/contradict/uncertain/insufficient rows sampled separately.",
        },
        {
            "figure": "Fig 4",
            "description": "CEQ attention maps",
            "status": "attention_assets_available" if attention_summary.exists() else "attention_casebook_only",
            "evidence": attention_summary.as_posix() if attention_summary.exists() else "outputs/final_tables/vsl_cxr_phase8_casebook.md",
            "notes": f"Best CEQ variant by binary AUC: {best_ceq.get('variant', '')} ({best_ceq.get('binary_auc', '')}).",
        },
        {
            "figure": "Fig 5",
            "description": "CCSH consistency readout",
            "status": "metrics_and_cases_ready",
            "evidence": "outputs/final_tables/vsl_cxr_ccsh_results.md; outputs/final_tables/vsl_cxr_phase8_casebook.md",
            "notes": f"Best CCSH binary AUC row: {best_ccsh.get('variant', '')} ({best_ccsh.get('binary_auc', '')}).",
        },
        {
            "figure": "Fig 6",
            "description": "external failure examples",
            "status": "appendix_stress_cases_ready",
            "evidence": "outputs/final_tables/vsl_cxr_external_results.md; outputs/final_tables/vsl_cxr_phase8_casebook.md",
            "notes": f"VSL-Core NIH macro-AUC={external_core.get('macro_auc', '')}; main external remains blocked.",
        },
        {
            "figure": "Fig 7",
            "description": "calibration curves",
            "status": "metric_table_ready_curve_points_missing",
            "evidence": "outputs/final_tables/vsl_cxr_external_results.md",
            "notes": "ECE/Brier are available; binned calibration curve points are not yet exported.",
        },
    ]


def main() -> None:
    cases = build_casebook()
    case_columns = [
        "case_id",
        "casebook",
        "dataset",
        "image",
        "statement",
        "true_label",
        "model_output",
        "explanation",
        "failure_type",
        "manual_note",
        "evidence",
    ]
    write_csv(FINAL_DIR / "vsl_cxr_phase8_casebook.csv", cases, case_columns)
    write_md_table(
        FINAL_DIR / "vsl_cxr_phase8_casebook.md",
        "VSL-CXR Phase 8 Casebook",
        cases,
        case_columns,
        note="Generated from current v5 VSL-CXR data/evidence. Rows with image paths require manual visual review before publication.",
    )

    visual_rows = build_visual_manifest(len(cases))
    visual_columns = ["figure", "description", "status", "evidence", "notes"]
    write_csv(FINAL_DIR / "vsl_cxr_phase8_visualization_manifest.csv", visual_rows, visual_columns)
    write_md_table(
        FINAL_DIR / "vsl_cxr_phase8_visualization_manifest.md",
        "VSL-CXR Phase 8 Visualization Manifest",
        visual_rows,
        visual_columns,
    )

    summary_rows = [
        {
            "artifact": "vsl_cxr_phase8_casebook",
            "status": "completed_needs_manual_review",
            "rows": len(cases),
            "evidence": "outputs/final_tables/vsl_cxr_phase8_casebook.md",
        },
        {
            "artifact": "vsl_cxr_phase8_visualization_manifest",
            "status": "completed",
            "rows": len(visual_rows),
            "evidence": "outputs/final_tables/vsl_cxr_phase8_visualization_manifest.md",
        },
    ]
    summary_columns = ["artifact", "status", "rows", "evidence"]
    write_csv(FINAL_DIR / "vsl_cxr_phase8_artifact_summary.csv", summary_rows, summary_columns)
    write_md_table(
        FINAL_DIR / "vsl_cxr_phase8_artifact_summary.md",
        "VSL-CXR Phase 8 Artifact Summary",
        summary_rows,
        summary_columns,
    )
    print(json.dumps({"casebook_rows": len(cases), "visual_rows": len(visual_rows)}, indent=2))


if __name__ == "__main__":
    main()
