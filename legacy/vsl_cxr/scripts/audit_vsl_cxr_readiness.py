"""Audit repository readiness for the VSL-CXR v5 experiment plan."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs" / "final_tables"
DEFAULT_MD = ROOT / "docs" / "vsl_cxr_readiness_audit.md"
DEFAULT_CSV = FINAL_DIR / "vsl_cxr_readiness_audit.csv"
MODEL_ROOT = Path("H:/Xiyao_Wang/001_models")
PUBLIC_DATA_ROOT = Path("H:/Xiyao_Wang/000_Public Dataset")

COLUMNS = ["area", "item", "status", "evidence", "notes"]

TARGET_SCRIPTS: dict[str, list[str]] = {
    "scripts/extract_clinical_statements.py": [
        "scripts/generate_ums_facts.py",
        "scripts/generate_p4v2_facts_with_glm.py",
        "scripts/prepare_mimic_report_manifest.py",
    ],
    "scripts/generate_counterfactual_statements.py": [
        "scripts/generate_ab_swap_jsonl.py",
        "scripts/generate_sameq_cf_compatible.py",
        "scripts/generate_shuf_k_cf_compatible.py",
    ],
    "scripts/generate_sameq_pairs.py": [
        "scripts/generate_sameq_shuf_pairs.py",
        "scripts/generate_sameq_cf_compatible.py",
    ],
    "scripts/generate_vsl_4class_labels.py": [
        "scripts/analyze_answerability_semantics.py",
        "scripts/analyze_schema_answerability.py",
        "scripts/validate_clinical_instruction_jsonl.py",
    ],
    "scripts/generate_hard_negative_pairs.py": [
        "scripts/generate_multi_negative_shuf.py",
        "scripts/generate_shuf_k_cf_compatible.py",
        "scripts/build_mined_shuf_instructions.py",
    ],
    "scripts/mine_hard_negatives_memory_bank.py": [
        "scripts/mine_hard_negatives_from_embeddings.py",
        "scripts/build_selfhard_shuf_instructions.py",
        "models/hard_negative_memory_bank.py",
    ],
    "scripts/audit_vsl_data_quality.py": [
        "scripts/audit_instruction_leakage_v3.py",
        "scripts/audit_instruction_leakage_v2.py",
        "scripts/audit_p4v2_instruction_quality.py",
        "scripts/validate_clinical_instruction_jsonl.py",
    ],
    "scripts/audit_false_hard_negatives.py": [
        "scripts/audit_hard_negative_quality.py",
    ],
    "scripts/train_vsl_cxr.py": [
        "scripts/train_qwen3vl_cvcp.py",
        "scripts/train_qwen3vl_clinical_instruction.py",
    ],
    "scripts/train_vsl_ceq.py": [
        "scripts/train_ceq_ccsh.py",
        "models/clinical_evidence_query.py",
    ],
    "scripts/train_vsl_ccsh.py": [
        "scripts/train_qwen3vl_sameq_ccsh.py",
        "scripts/train_qwen3vl_shufk_ccsh.py",
        "models/clinical_consistency_head.py",
    ],
    "scripts/train_vsl_hnmb.py": [
        "scripts/train_hnmb_ccsh.py",
        "models/hard_negative_memory_bank.py",
    ],
    "scripts/train_vsl_full.py": [
        "scripts/train_qwen3vl_cvcp.py",
        "scripts/train_ceq_ccsh.py",
        "scripts/train_hnmb_ccsh.py",
    ],
    "scripts/train_vlm_teacher_comparison.py": [
        "scripts/train_qwen3vl_clinical_instruction.py",
    ],
    "scripts/eval_chexpert_lp.py": [
        "scripts/train_qwen3vl_vision_lp.py",
        "scripts/evaluate_qwen3vl_lp_transfer.py",
    ],
    "scripts/eval_external_lp.py": [
        "scripts/eval_external_dataset.py",
        "scripts/eval_nih_external.py",
        "scripts/run_nih_full_transfer.py",
    ],
    "scripts/eval_vsl_sufficiency.py": [
        "scripts/evaluate_qwen3vl_counterfactual_diagnostics.py",
        "scripts/evaluate_instruction_counterfactual_diagnostics.py",
    ],
    "scripts/eval_ccsh_consistency.py": [
        "scripts/train_case_study_module_ablation.py",
        "outputs/final_tables/module_ablation_results.csv",
    ],
    "scripts/eval_hard_shuffle.py": [
        "scripts/evaluate_qwen3vl_visual_dependence.py",
    ],
    "scripts/eval_calibration.py": [
        "scripts/eval_calibration_auprc.py",
        "scripts/analyze_answerability_semantics.py",
    ],
    "scripts/eval_ceq_attention.py": [
        "scripts/plot_attention_maps.py",
        "models/clinical_evidence_query.py",
    ],
    "scripts/eval_casebook.py": [
        "scripts/build_casebook_markdown.py",
        "scripts/mine_pairwise_case_studies.py",
    ],
    "scripts/eval_locked_final_comparison.py": [
        "scripts/eval_locked_final_suite.py",
        "scripts/bootstrap_locked_comparison.py",
        "scripts/paired_bootstrap_method_delta.py",
    ],
    "scripts/build_vsl_results_table.py": [
        "scripts/summarize_cvcp_ccsh_results.py",
        "scripts/summarize_next_stage_results.py",
    ],
    "scripts/build_external_results_table.py": [
        "scripts/eval_external_dataset.py",
        "scripts/audit_label_mapping_nih.py",
    ],
    "scripts/build_module_results_table.py": [
        "scripts/summarize_cvcp_ccsh_module_combos.py",
        "scripts/summarize_case_study_modules_results.py",
    ],
    "scripts/build_case_study_markdown.py": [
        "scripts/build_casebook_markdown.py",
    ],
    "scripts/build_paper_figures.py": [
        "scripts/plot_attention_maps.py",
        "scripts/plot_dataset_embedding_umap.py",
        "scripts/plot_tsne.py",
    ],
    "scripts/build_cost_table.py": [
        "scripts/consolidate_cost_table.py",
    ],
}

DATA_CANDIDATES = {
    "D0 Basic-QA": ["outputs/instruction_data/glm_validated/d0_train_validated.jsonl"],
    "D1 CF-QA": [
        "outputs/instruction_data/glm_validated/d6_hard_cf_3k.jsonl",
        "outputs/instruction_data/next_stage/cf_10k_train.jsonl",
    ],
    "D2 SAMEQ": [
        "outputs/instruction_data/next_stage/sameq_shuf_3k_train.jsonl",
        "outputs/instruction_data/cvcp_ccsh/cvcp_v1_sameq_full_train.jsonl",
    ],
    "D3 SAMEQ-CF": [
        "outputs/instruction_data/cvcp_ccsh/sameq_cf_20_train.jsonl",
        "outputs/final_tables/sameq_cf_compatible_manifest.csv",
    ],
    "D4 SAMEQ-K": [
        "outputs/instruction_data/next_stage/shuf_k4_train.jsonl",
        "outputs/instruction_data/cvcp_ccsh/cvcp_v2_shuf_k4_train.jsonl",
    ],
    "D5 SAMEQ-HNMB": [
        "outputs/instruction_data/next_stage/mined_shuf_train.jsonl",
        "outputs/instruction_data/next_stage/selfhard_shuf_train.jsonl",
    ],
    "D6 VSL-4class": [
        "outputs/instruction_data/vsl_cxr/d6_vsl_4class_train.jsonl",
        "outputs/instruction_data/vsl_cxr/d6_vsl_4class_val.jsonl",
    ],
    "D7 VSL-CEQ": ["outputs/instruction_data/cvcp_ccsh/ceq_targets_train.jsonl"],
    "D8 VSL-CCSH": ["outputs/instruction_data/cvcp_ccsh/ccsh_statements_train.jsonl"],
    "D9 VSL-full": [
        "outputs/instruction_data/vsl_cxr/d9_vsl_full_train.jsonl",
        "outputs/instruction_data/vsl_cxr/d9_vsl_full_val.jsonl",
        "outputs/instruction_data/vsl_cxr/d9_ceq_targets_train.jsonl",
        "outputs/instruction_data/vsl_cxr/d9_ccsh_pairs_train.jsonl",
        "outputs/final_tables/vsl_cxr_d9_full_dataset_manifest.csv",
    ],
}

FINAL_TABLE_CANDIDATES = {
    "training/backbone results": [
        "outputs/final_tables/cvcp_training_results.csv",
        "outputs/final_tables/next_stage_training_results.csv",
        "outputs/final_tables/sameq_v4_multiseed_training_results.csv",
    ],
    "module/readout results": [
        "outputs/final_tables/module_combo_results.csv",
        "outputs/final_tables/module_ablation_results.csv",
        "outputs/final_tables/ccsh_consistency_results.csv",
        "outputs/final_tables/ceq_consistency_results.csv",
    ],
    "external results": [
        "outputs/final_tables/external_eval_results.csv",
        "outputs/final_tables/sameq_v4_multiseed_external_eval_results.csv",
        "outputs/final_tables/nih_available_transfer_status.csv",
    ],
    "calibration": [
        "outputs/final_tables/cvcp_calibration_auprc.csv",
        "outputs/final_tables/next_stage_calibration_auprc.csv",
        "outputs/final_tables/sameq_v4_multiseed_calibration_auprc.csv",
    ],
    "hard shuffle": [
        "outputs/final_tables/cvcp_hard_shuffle_results.csv",
        "outputs/final_tables/sameq_v4_multiseed_hard_shuffle_results.csv",
        "outputs/final_tables/visual_dependence_results.csv",
    ],
    "casebook": [
        "outputs/final_tables/casebook.md",
        "outputs/final_tables/case_study_summary.md",
        "outputs/final_tables/next_stage_qualitative_cases.md",
    ],
    "cost": [
        "outputs/final_tables/cost_table.csv",
        "outputs/final_tables/qwen3vl_cost_table.csv",
        "outputs/final_tables/sameq_v4_multiseed_cost_table.csv",
    ],
    "locked comparison": [
        "outputs/final_tables/locked_final_comparison.csv",
        "outputs/final_tables/sameq_v4_multiseed_locked_final_comparison.csv",
        "outputs/final_tables/bootstrap_locked_comparison.csv",
    ],
}

MODEL_ITEMS = {
    "Qwen3-VL": ["qwen3-vl-2b-thinking-new", "Qwen3-VL-4B-Instruct", "Qwen3-VL-8B-Instruct"],
    "InternVL": ["InternVL3_5-1B", "InternVL3_5-2B", "InternVL3_5-4B", "InternVL3_5-8B"],
    "LLaVA/Llama-based VLM": ["Llama-3.2-11B-Vision-Instruct"],
    "Qwen3.5 text scaffold": ["Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B"],
    "Qwen-Coder scaffold": ["Qwen2.5-Coder-7B-Instruct"],
}


def repo_rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def add(rows: list[dict[str, str]], area: str, item: str, status: str, evidence: str = "", notes: str = "") -> None:
    rows.append({"area": area, "item": item, "status": status, "evidence": evidence, "notes": notes})


def read_jsonl_sample(path: Path, max_rows: int = 200) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) >= max_rows:
                    break
    return rows


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def classify_dataset_candidate(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "missing", ""
    rows = read_jsonl_sample(path)
    keys = set()
    answer_types = Counter()
    sufficiency = Counter()
    for row in rows:
        keys.update(row.keys())
        answer_types[str(row.get("answer_type", ""))] += 1
        sufficiency[str(row.get("sufficiency_label", ""))] += 1
    notes = [
        f"rows={count_jsonl(path)}",
        f"sample_keys={','.join(sorted(k for k in keys if k)[:20])}",
    ]
    if answer_types:
        notes.append("answer_type=" + ",".join(f"{k}:{v}" for k, v in answer_types.most_common(6)))
    if sufficiency and any(k for k in sufficiency):
        notes.append("sufficiency_label=" + ",".join(f"{k}:{v}" for k, v in sufficiency.most_common(6)))
    required_vsl = {"statement", "sufficiency_label"}
    if required_vsl.issubset(keys):
        return "candidate_vsl_schema", "; ".join(notes)
    if {"question", "answer"}.issubset(keys):
        return "candidate_qa_schema", "; ".join(notes)
    return "exists_schema_review_needed", "; ".join(notes)


def audit_scripts(rows: list[dict[str, str]]) -> None:
    for target, analogs in TARGET_SCRIPTS.items():
        target_path = ROOT / target
        if target_path.exists():
            add(rows, "script", target, "exact_exists", target, "Exact v5-named entry point exists.")
            continue
        existing_analogs = [path for path in analogs if (ROOT / path).exists()]
        if existing_analogs:
            add(
                rows,
                "script",
                target,
                "missing_exact_analogs_exist",
                "; ".join(existing_analogs),
                "Analog existence is implementation help, not v5 completion evidence.",
            )
        else:
            add(rows, "script", target, "missing_no_known_analog", "", "Needs implementation.")


def audit_data(rows: list[dict[str, str]]) -> None:
    for item, candidates in DATA_CANDIDATES.items():
        if not candidates:
            add(rows, "dataset", item, "missing_no_candidate", "", "No obvious current artifact found.")
            continue
        found_any = False
        for candidate in candidates:
            path = ROOT / candidate
            if path.exists():
                found_any = True
                status, notes = classify_dataset_candidate(path) if path.suffix == ".jsonl" else ("artifact_exists", "")
                add(rows, "dataset", item, status, candidate, notes)
        if not found_any:
            add(rows, "dataset", item, "missing_candidate_paths", "; ".join(candidates), "Expected candidate paths not found.")


def audit_outputs(rows: list[dict[str, str]]) -> None:
    for item, candidates in FINAL_TABLE_CANDIDATES.items():
        found = [path for path in candidates if (ROOT / path).exists()]
        status = "historical_artifacts_exist" if found else "missing"
        add(rows, "final_table", item, status, "; ".join(found), "Must be remapped to v5 rows before reuse." if found else "")


def audit_models(rows: list[dict[str, str]]) -> None:
    for item, candidates in MODEL_ITEMS.items():
        found = [name for name in candidates if (MODEL_ROOT / name).exists()]
        status = "available_needs_smoke" if found else "missing"
        evidence = "; ".join((MODEL_ROOT / name).as_posix() for name in found)
        add(rows, "model", item, status, evidence, "Directory presence only; formal teacher rows require adapter/GPU smoke.")


def audit_external_data(rows: list[dict[str, str]]) -> None:
    vindr_root = PUBLIC_DATA_ROOT / "vindr-cxr-an-open-dataset-of-chest-x-rays-with-radiologists-annotations-1.0.0"
    vindr_archive = PUBLIC_DATA_ROOT / "vindr-cxr-an-open-dataset-of-chest-x-rays-with-radiologist-annotations-1.0.0.zip"
    vindr_audit = FINAL_DIR / "vindr_cxr_data_quality_audit.json"
    checks = {
        "VinBigData/VinDr derived images": [
            ROOT / "data/dataset/vinbigdata_xhlulu_512png",
            vindr_root,
            vindr_archive,
        ],
        "PadChest": [
            ROOT / "data/dataset/PadChest",
            PUBLIC_DATA_ROOT / "PadChest",
        ],
        "MIMIC-CXR": [
            ROOT / "data/dataset/mimic-cxr",
            PUBLIC_DATA_ROOT / "mimic-cxr",
        ],
        "NIH": [
            ROOT / "data/dataset/NIH",
            ROOT / "data/dataset/NIH Chest X-rays",
            PUBLIC_DATA_ROOT / "NIH Chest X-rays",
        ],
        "CheXpert": [
            ROOT / "data/dataset/CheXpert-v1.0-small",
            PUBLIC_DATA_ROOT / "CheXpert-v1.0-small",
        ],
    }
    label_hints = {
        "VinBigData/VinDr derived images": [
            ROOT / "data/dataset/vinbigdata_xhlulu_512png/train.csv",
            ROOT / "data/dataset/vinbigdata_xhlulu_512png/train_meta.csv",
            vindr_root / "annotations/image_labels_train.csv",
            vindr_root / "annotations/image_labels_test.csv",
            vindr_root / "annotations/annotations_train.csv",
            vindr_root / "annotations/annotations_test.csv",
            ROOT / "data/dataset/processed/vindr_cxr_external_test_ums.jsonl",
        ],
        "NIH": [
            ROOT / "data/dataset/NIH Chest X-rays/Data_Entry_2017.csv",
            PUBLIC_DATA_ROOT / "NIH Chest X-rays/Data_Entry_2017.csv",
        ],
        "MIMIC-CXR": [
            PUBLIC_DATA_ROOT / "mimic-cxr/mimic-cxr/mimic-cxr-2.0.0-metadata.csv",
            PUBLIC_DATA_ROOT / "mimic-cxr/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv",
            PUBLIC_DATA_ROOT / "mimic_cxr_other/mimic-cxr-2.0.0-metadata.csv.gz",
            PUBLIC_DATA_ROOT / "mimic_cxr_other/mimic-cxr-2.0.0-chexpert.csv.gz",
            PUBLIC_DATA_ROOT / "mimic_cxr_other/mimic-cxr-2.0.0-split.csv.gz",
        ],
        "CheXpert": [
            ROOT / "data/dataset/CheXpert-v1.0-small/valid.csv",
            PUBLIC_DATA_ROOT / "CheXpert-v1.0-small/valid.csv",
        ],
        "PadChest": [],
    }
    for item, paths in checks.items():
        existing_paths = [path for path in paths if path.exists()]
        existing_labels = [path for path in label_hints.get(item, []) if path.exists()]
        if not existing_paths:
            add(rows, "external_data", item, "missing", "; ".join(repo_rel(path) for path in paths), "No local directory found at audited paths.")
            continue
        status = "exists_needs_label_audit" if existing_labels else "exists_missing_label_manifest"
        notes = "v5 main-external eligibility depends on labels/manifest and training-overlap audit."
        if item == "VinBigData/VinDr derived images":
            extraction_complete = (vindr_root / "_extraction_complete.json").exists()
            audit_payload = read_json(vindr_audit) if vindr_audit.exists() else {}
            if audit_payload.get("status") == "ready" and extraction_complete:
                status = "ready_main_external_evaluation"
                notes = "Official VinDr-CXR test labels and DICOMs are complete; direct 7-label primary mapping is ready for formal evaluation."
            elif any(path.name == "image_labels_test.csv" for path in existing_labels):
                status = "extracting_labels_available"
                notes = "Official VinDr-CXR image labels are available and the deterministic manifest exists; full DICOM extraction/validation is still running."
        if item == "MIMIC-CXR" and existing_labels:
            status = "exists_label_manifest_overlap_audit_pending"
            notes = "Official MIMIC-CXR CheXpert/metadata/split manifests exist as .csv.gz; eligibility still requires training-overlap audit."
        add(
            rows,
            "external_data",
            item,
            status,
            "; ".join(repo_rel(path) for path in existing_paths + existing_labels),
            notes,
        )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in COLUMNS})


def write_md(path: Path, rows: list[dict[str, str]], csv_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter((row["area"], row["status"]) for row in rows)
    lines = [
        "# VSL-CXR v5 Readiness Audit",
        "",
        f"Machine-readable CSV: `{repo_rel(csv_path)}`",
        "",
        "This audit is read-only. It classifies current scripts, data artifacts, final tables, models, and external-data buckets for the active VSL-CXR v5 plan.",
        "",
        "## Status Counts",
        "",
        "| Area | Status | Count |",
        "| --- | --- | ---: |",
    ]
    for (area, status), count in sorted(counts.items()):
        lines.append(f"| {area} | {status} | {count} |")
    lines.extend(["", "## Audit Rows", "", "| " + " | ".join(COLUMNS) + " |", "| " + " | ".join("---" for _ in COLUMNS) + " |"])
    for row in rows:
        values = [str(row.get(key, "")).replace("|", "\\|").replace("\n", " ") for key in COLUMNS]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str]] = []
    audit_scripts(rows)
    audit_data(rows)
    audit_outputs(rows)
    audit_models(rows)
    audit_external_data(rows)
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
