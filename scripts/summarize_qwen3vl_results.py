"""Summarize Qwen3-VL instruction, LP, and extraction artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs" / "final_tables"

INSTRUCTION_RUNS = [
    {
        "pilot_id": "P2",
        "run": "p2_d0_fixed_json_schema",
        "data": "D0 fixed JSON schema",
        "path": ROOT / "outputs/qwen3vl_instruction_runs/p2_d0_fixed_json_schema/metrics_final.json",
    },
    {
        "pilot_id": "P3",
        "run": "p3_d2_report_grounded_qa",
        "data": "D2 report-grounded QA",
        "path": ROOT / "outputs/qwen3vl_instruction_runs/p3_d2_report_grounded_qa/metrics_final.json",
    },
    {
        "pilot_id": "P4",
        "run": "p4_d3_report_grounded_counterfactual",
        "data": "D3 report-grounded QA + counterfactual",
        "path": ROOT / "outputs/qwen3vl_instruction_runs/p4_d3_report_grounded_counterfactual/metrics_final.json",
    },
    {
        "pilot_id": "P5",
        "run": "p5_d4_counterfactual_weighted",
        "data": "D4 counterfactual weighted",
        "path": ROOT / "outputs/qwen3vl_instruction_runs/p5_d4_counterfactual_weighted/metrics_final.json",
    },
]

LP_RUNS = [
    {
        "pilot_id": "Base",
        "run": "base_chexpert_1k",
        "path": ROOT / "outputs/qwen3vl_lp_runs/base_chexpert_1k/metrics_final.json",
        "checkpoint_source": "base Qwen3-VL vision tower",
    },
    {
        "pilot_id": "P2",
        "run": "p2_d0_fixed_json_schema_chexpert_1k",
        "path": ROOT / "outputs/qwen3vl_lp_runs/p2_d0_fixed_json_schema_chexpert_1k/metrics_final.json",
        "checkpoint_source": "P2 best checkpoint",
    },
    {
        "pilot_id": "P3",
        "run": "p3_d2_report_grounded_qa_chexpert_1k",
        "path": ROOT / "outputs/qwen3vl_lp_runs/p3_d2_report_grounded_qa_chexpert_1k/metrics_final.json",
        "checkpoint_source": "P3 best checkpoint",
    },
    {
        "pilot_id": "P4",
        "run": "p4_d3_report_grounded_counterfactual_chexpert_1k",
        "path": ROOT / "outputs/qwen3vl_lp_runs/p4_d3_report_grounded_counterfactual_chexpert_1k/metrics_final.json",
        "checkpoint_source": "P4 best checkpoint",
    },
    {
        "pilot_id": "P5",
        "run": "p5_d4_counterfactual_weighted_chexpert_1k",
        "path": ROOT / "outputs/qwen3vl_lp_runs/p5_d4_counterfactual_weighted_chexpert_1k/metrics_final.json",
        "checkpoint_source": "P5 best checkpoint",
    },
    {
        "pilot_id": "P6",
        "run": "p6_data_only_no_lm_chexpert_1k",
        "path": ROOT / "outputs/qwen3vl_lp_runs/p6_data_only_no_lm_chexpert_1k/metrics_final.json",
        "checkpoint_source": "data-only no-LM trainable Qwen3-VL vision tower",
    },
]

TRANSFER_RUNS = [
    {
        "pilot_id": "Base",
        "run": "base_nih_1k",
        "path": ROOT / "outputs/qwen3vl_transfer/base_nih_1k/transfer_metrics.json",
    },
    {
        "pilot_id": "P2",
        "run": "p2_nih_1k",
        "path": ROOT / "outputs/qwen3vl_transfer/p2_nih_1k/transfer_metrics.json",
    },
    {
        "pilot_id": "P3",
        "run": "p3_nih_1k",
        "path": ROOT / "outputs/qwen3vl_transfer/p3_nih_1k/transfer_metrics.json",
    },
    {
        "pilot_id": "P4",
        "run": "p4_nih_1k",
        "path": ROOT / "outputs/qwen3vl_transfer/p4_nih_1k/transfer_metrics.json",
    },
    {
        "pilot_id": "P5",
        "run": "p5_nih_1k",
        "path": ROOT / "outputs/qwen3vl_transfer/p5_nih_1k/transfer_metrics.json",
    },
    {
        "pilot_id": "P6",
        "run": "p6_nih_1k",
        "path": ROOT / "outputs/qwen3vl_transfer/p6_nih_1k/transfer_metrics.json",
    },
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def write_md(path: Path, title: str, rows: list[dict[str, Any]], columns: list[str], readout: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            f"# {title}",
            "",
            markdown_table(rows, columns),
            "",
            "## Readout",
            "",
            readout,
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def instruction_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in INSTRUCTION_RUNS:
        metrics = read_json(spec["path"])
        params = metrics["parameter_groups"]
        rows.append(
            {
                "pilot_id": spec["pilot_id"],
                "run": spec["run"],
                "data": spec["data"],
                "global_step": metrics["global_step"],
                "best_val_loss": fmt(metrics["best_val_loss"]),
                "train_records": metrics["train_records"],
                "val_records": metrics["val_records"],
                "elapsed_seconds": fmt(metrics["elapsed_seconds"]),
                "vision_trainable": params["vision_tower"]["trainable"],
                "connector_trainable": params["visual_connector"]["trainable"],
                "language_decoder_trainable": params["language_decoder"]["trainable"],
                "total_trainable": params["total"]["trainable"],
                "metrics_path": rel(spec["path"]),
            }
        )
    return rows


def lp_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in LP_RUNS:
        metrics = read_json(spec["path"])
        cls = metrics["metrics"]
        rows.append(
            {
                "pilot_id": spec["pilot_id"],
                "run": spec["run"],
                "checkpoint_source": spec["checkpoint_source"],
                "global_step": metrics["global_step"],
                "best_val_loss": fmt(metrics["best_val_loss"]),
                "final_val_loss": fmt(metrics["final_val_loss"]),
                "macro_auc": fmt(cls["macro_auc"]),
                "macro_f1": fmt(cls["macro_f1"]),
                "micro_f1": fmt(cls["micro_f1"]),
                "train_records": metrics["train_records"],
                "val_records": metrics["val_records"],
                "feature_dim": metrics["feature_dim"],
                "elapsed_seconds": fmt(metrics["elapsed_seconds"]),
                "metrics_path": rel(spec["path"]),
            }
        )
    return rows


def extraction_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in INSTRUCTION_RUNS:
        manifest_path = ROOT / "outputs/qwen3vl_extracted" / spec["run"] / "manifest.json"
        manifest = read_json(manifest_path)
        params = manifest["parameter_groups"]
        load_info = manifest["load_info"]
        rows.append(
            {
                "pilot_id": spec["pilot_id"],
                "run": spec["run"],
                "checkpoint": manifest["checkpoint"],
                "global_step": load_info.get("global_step"),
                "best_val_loss": fmt(load_info.get("best_val_loss")),
                "missing_keys": load_info.get("missing_keys"),
                "unexpected_keys": load_info.get("unexpected_keys"),
                "vision_state": manifest["vision_tower_state"],
                "connector_state": manifest["visual_connector_state"],
                "combined_state": manifest["combined_vision_side_state"],
                "vision_trainable": params["vision_tower"]["trainable"],
                "connector_trainable": params["visual_connector"]["trainable"],
                "language_decoder_trainable": params["language_decoder"]["trainable"],
                "total_trainable": params["total"]["trainable"],
                "manifest_path": rel(manifest_path),
            }
        )
    return rows


def visual_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in INSTRUCTION_RUNS:
        diag_path = ROOT / "outputs/qwen3vl_diagnostics" / f"{spec['pilot_id'].lower()}_visual_dependence.json"
        if not diag_path.exists():
            continue
        payload = read_json(diag_path)
        modes = {row["mode"]: row for row in payload["modes"]}
        normal = modes.get("normal", {})
        question_only = modes.get("question_only", {})
        image_shuffle = modes.get("image_shuffle", {})
        rows.append(
            {
                "pilot_id": spec["pilot_id"],
                "run": spec["run"],
                "examples": normal.get("examples"),
                "normal_loss": fmt(normal.get("loss")),
                "question_only_loss": fmt(question_only.get("loss")),
                "question_only_delta": fmt(question_only.get("delta_vs_normal")),
                "image_shuffle_loss": fmt(image_shuffle.get("loss")),
                "image_shuffle_delta": fmt(image_shuffle.get("delta_vs_normal")),
                "diagnostic_path": rel(diag_path),
                "boundary": "positive question-only delta, small image-shuffle delta",
            }
        )
    return rows


def counterfactual_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    overall_rows: list[dict[str, Any]] = []
    type_rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    for spec in INSTRUCTION_RUNS:
        if spec["pilot_id"] not in {"P4", "P5"}:
            continue
        diag_path = ROOT / "outputs/qwen3vl_diagnostics" / f"{spec['pilot_id'].lower()}_counterfactual_diagnostics.json"
        if not diag_path.exists():
            continue
        payload = read_json(diag_path)
        overall = payload["summary"]["counterfactual_option_nll"]["overall"]
        overall_rows.append(
            {
                "pilot_id": spec["pilot_id"],
                "run": spec["run"],
                "total_counterfactual_records": overall.get("total_records"),
                "option_formatted_records": overall.get("option_formatted_records"),
                "no_option_records": overall.get("no_option_records"),
                "correct_letter_failures": overall.get("correct_letter_failures"),
                "pairwise_accuracy": fmt(overall.get("pairwise_accuracy")),
                "mean_best_negative_minus_correct_nll": fmt(overall.get("mean_best_negative_minus_correct_nll")),
                "median_best_negative_minus_correct_nll": fmt(overall.get("median_best_negative_minus_correct_nll")),
                "diagnostic_path": rel(diag_path),
                "boundary": "pairwise metric applies only to option-formatted records",
            }
        )
        for counterfactual_type, row in payload["summary"]["counterfactual_option_nll"]["by_counterfactual_type"].items():
            type_rows.append(
                {
                    "pilot_id": spec["pilot_id"],
                    "run": spec["run"],
                    "counterfactual_type": counterfactual_type,
                    "total_records": row.get("total_records"),
                    "option_formatted_records": row.get("option_formatted_records"),
                    "no_option_records": row.get("no_option_records"),
                    "correct_letter_failures": row.get("correct_letter_failures"),
                    "pairwise_accuracy": fmt(row.get("pairwise_accuracy")),
                    "mean_best_negative_minus_correct_nll": fmt(row.get("mean_best_negative_minus_correct_nll")),
                    "median_best_negative_minus_correct_nll": fmt(row.get("median_best_negative_minus_correct_nll")),
                }
            )
        for answer_type, row in payload["summary"]["answer_nll"]["by_answer_type"].items():
            answer_rows.append(
                {
                    "pilot_id": spec["pilot_id"],
                    "run": spec["run"],
                    "answer_type": answer_type,
                    "n": row.get("n"),
                    "mean_nll": fmt(row.get("mean")),
                    "median_nll": fmt(row.get("median")),
                    "min_nll": fmt(row.get("min")),
                    "max_nll": fmt(row.get("max")),
                }
            )
    return overall_rows, type_rows, answer_rows


def paraphrase_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    overall_rows: list[dict[str, Any]] = []
    answer_type_rows: list[dict[str, Any]] = []
    for spec in INSTRUCTION_RUNS:
        diag_path = ROOT / "outputs/qwen3vl_diagnostics" / f"{spec['pilot_id'].lower()}_paraphrase_robustness.json"
        if not diag_path.exists():
            continue
        payload = read_json(diag_path)
        for variant, row in payload["summary"]["overall"].items():
            overall_rows.append(
                {
                    "pilot_id": spec["pilot_id"],
                    "run": spec["run"],
                    "variant": variant,
                    "n": row.get("n"),
                    "original_nll_mean": fmt(row.get("original_nll_mean")),
                    "variant_nll_mean": fmt(row.get("variant_nll_mean")),
                    "mean_delta_vs_original": fmt(row.get("mean_delta_vs_original")),
                    "median_delta_vs_original": fmt(row.get("median_delta_vs_original")),
                    "relative_delta_vs_original": fmt(row.get("relative_delta_vs_original")),
                    "variant_worse_rate": fmt(row.get("variant_worse_rate")),
                    "diagnostic_path": rel(diag_path),
                    "boundary": "question template/paraphrase sensitivity; not a semantic accuracy metric",
                }
            )
        for answer_type, variants in payload["summary"]["by_answer_type"].items():
            for variant, row in variants.items():
                answer_type_rows.append(
                    {
                        "pilot_id": spec["pilot_id"],
                        "run": spec["run"],
                        "answer_type": answer_type,
                        "variant": variant,
                        "n": row.get("n"),
                        "original_nll_mean": fmt(row.get("original_nll_mean")),
                        "variant_nll_mean": fmt(row.get("variant_nll_mean")),
                        "mean_delta_vs_original": fmt(row.get("mean_delta_vs_original")),
                        "median_delta_vs_original": fmt(row.get("median_delta_vs_original")),
                        "relative_delta_vs_original": fmt(row.get("relative_delta_vs_original")),
                        "variant_worse_rate": fmt(row.get("variant_worse_rate")),
                    }
                )
    return overall_rows, answer_type_rows


def transfer_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in TRANSFER_RUNS:
        if not spec["path"].exists():
            continue
        payload = read_json(spec["path"])
        metrics = payload["metrics"]
        rows.append(
            {
                "pilot_id": spec["pilot_id"],
                "run": spec["run"],
                "evaluated_records": payload.get("evaluated_records"),
                "missing_images": (payload.get("image_audit") or {}).get("missing_count"),
                "val_loss": fmt(payload.get("val_loss")),
                "macro_auc": fmt(metrics.get("macro_auc")),
                "macro_f1": fmt(metrics.get("macro_f1")),
                "micro_f1": fmt(metrics.get("micro_f1")),
                "elapsed_seconds": fmt(payload.get("elapsed_seconds")),
                "metrics_path": rel(spec["path"]),
                "boundary": "NIH external 1k subset transfer; not full NIH external test",
            }
        )
    return rows


def label_distribution(path: Path, label_names: list[str]) -> dict[str, dict[str, int]]:
    stats = {label: {"present": 0, "absent": 0, "uncertain": 0, "null": 0, "total": 0} for label in label_names}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            findings = row.get("findings") or {}
            for label in label_names:
                item = findings.get(label) or {}
                state = item.get("state")
                stats[label]["total"] += 1
                if state in {"present", "absent", "uncertain"}:
                    stats[label][state] += 1
                else:
                    stats[label]["null"] += 1
    return stats


def mean_metric(per_label: dict[str, Any], labels: list[str], metric: str) -> str:
    values = []
    for label in labels:
        value = (per_label.get(label) or {}).get(metric)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return fmt(sum(values) / len(values)) if values else ""


def subgroup_rows(lp: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    val_path = ROOT / "data/splits/chexpert_val_fixed.jsonl"
    for spec in LP_RUNS:
        if not spec["path"].exists():
            continue
        payload = read_json(spec["path"])
        label_names = list(payload.get("label_names") or [])
        per_label = payload["metrics"].get("per_label") or {}
        dist = label_distribution(val_path, label_names)
        rare_labels = [
            label
            for label, stats in dist.items()
            if stats["present"] > 0 and (stats["present"] <= 50 or stats["present"] / max(stats["total"], 1) <= 0.05)
        ]
        high_null_labels = [label for label, stats in dist.items() if stats["null"] / max(stats["total"], 1) >= 0.20]
        uncertain_heavy_labels = [label for label, stats in dist.items() if stats["uncertain"] / max(stats["total"], 1) >= 0.05]
        support_device_auc = ""
        support_note = ""
        if "Support Devices" in per_label:
            support_device_auc = fmt((per_label["Support Devices"] or {}).get("auc"))
        else:
            support_note = "Support Devices not evaluated because LP uses the common 8-label subset"
        rows.append(
            {
                "pilot_id": spec["pilot_id"],
                "run": spec["run"],
                "common_auc": mean_metric(per_label, label_names, "auc"),
                "rare_auc": mean_metric(per_label, rare_labels, "auc"),
                "rare_labels": ", ".join(rare_labels),
                "high_null_auc": mean_metric(per_label, high_null_labels, "auc"),
                "high_null_labels": ", ".join(high_null_labels) if high_null_labels else "none at >=20% null",
                "uncertain_heavy_auc": mean_metric(per_label, uncertain_heavy_labels, "auc"),
                "uncertain_heavy_labels": ", ".join(uncertain_heavy_labels) if uncertain_heavy_labels else "none at >=5% uncertain",
                "support_device_auc": support_device_auc,
                "notes": support_note,
            }
        )
    return rows


def pilot_rows(instruction: list[dict[str, Any]], lp: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pilot = {row["pilot_id"]: row for row in instruction}
    lp_by_pilot = {row["pilot_id"]: row for row in lp}
    rows = [
        {
            "pilot_id": "P0",
            "route": "old scaffold baseline",
            "status": "closed historical baseline",
            "training": "see old MIMIC closure",
            "lp_macro_auc": "",
            "diagnostics": "old visual/counterfactual tables available",
            "boundary": "do not extend old document line unless explicitly reopened",
        },
        {
            "pilot_id": "P1",
            "route": "old scaffold control",
            "status": "closed historical control",
            "training": "old V3/V4 scaffold evidence",
            "lp_macro_auc": "",
            "diagnostics": "old visual/counterfactual tables available",
            "boundary": "baseline only",
        },
    ]
    for pilot_id in ["P2", "P3", "P4", "P5"]:
        train = by_pilot[pilot_id]
        probe = lp_by_pilot[pilot_id]
        rows.append(
            {
                "pilot_id": pilot_id,
                "route": "Qwen3-VL coupled",
                "status": (
                    "training+LP+extraction+visual+counterfactual+paraphrase+NIH transfer complete"
                    if pilot_id in {"P4", "P5"}
                    else "training+LP+extraction+visual+paraphrase+NIH transfer complete"
                ),
                "training": f"step {train['global_step']}, best val loss {train['best_val_loss']}",
                "lp_macro_auc": probe["macro_auc"],
                "diagnostics": "visual-dependence/paraphrase complete; counterfactual complete for P4/P5",
                "boundary": "LP alone does not prove visual grounding",
            }
        )
    p6 = lp_by_pilot.get("P6")
    rows.append(
        {
            "pilot_id": "P6",
            "route": "data-only no-LM",
            "status": "CheXpert LP + NIH transfer complete" if p6 else "pending",
            "training": f"step {p6['global_step']}, best val loss {p6['best_val_loss']}" if p6 else "",
            "lp_macro_auc": p6["macro_auc"] if p6 else "",
            "diagnostics": "direct label-head training without instruction decoder" if p6 else "",
            "boundary": "uses CheXpert UMS label heads, not GLM D3 instruction heads",
        }
    )
    rows.append(
        {
            "pilot_id": "P7",
            "route": "optional LoRA upper bound",
            "status": "optional",
            "training": "",
            "lp_macro_auc": "",
            "diagnostics": "",
            "boundary": "not required before P2-P6 completion",
        }
    )
    return rows


def cost_rows(instruction: list[dict[str, Any]], lp: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lp_by_pilot = {row["pilot_id"]: row for row in lp}
    rows.append(
        {
            "run": "base_chexpert_1k",
            "model": "Qwen3-VL base vision tower",
            "trainable_params": "LP head only",
            "frozen_params": "Qwen3-VL vision tower",
            "peak_vram": "not captured",
            "gpu_hours": fmt(float(lp_by_pilot["Base"]["elapsed_seconds"]) / 3600.0),
            "steps_per_sec": fmt(float(lp_by_pilot["Base"]["global_step"]) / float(lp_by_pilot["Base"]["elapsed_seconds"])),
            "deployment_model": "Qwen3-VL vision tower + LP head",
            "deployment_llm": "no",
        }
    )
    for row in instruction:
        lp_row = lp_by_pilot[row["pilot_id"]]
        gpu_hours = (float(row["elapsed_seconds"]) + float(lp_row["elapsed_seconds"])) / 3600.0
        rows.append(
            {
                "run": row["run"],
                "model": "Qwen3-VL-2B vision tower + connector",
                "trainable_params": row["total_trainable"],
                "frozen_params": 2127532032 - int(row["total_trainable"]),
                "peak_vram": "not captured",
                "gpu_hours": fmt(gpu_hours),
                "steps_per_sec": fmt(float(row["global_step"]) / float(row["elapsed_seconds"])),
                "deployment_model": "Qwen3-VL vision tower",
                "deployment_llm": "no",
            }
        )
    if "P6" in lp_by_pilot:
        p6 = lp_by_pilot["P6"]
        rows.append(
            {
                "run": p6["run"],
                "model": "Qwen3-VL vision tower + linear head",
                "trainable_params": 306242560 + 1024 * 8 + 8,
                "frozen_params": "language decoder and connector unused by no-LM objective",
                "peak_vram": "not captured",
                "gpu_hours": fmt(float(p6["elapsed_seconds"]) / 3600.0),
                "steps_per_sec": fmt(float(p6["global_step"]) / float(p6["elapsed_seconds"])),
                "deployment_model": "Qwen3-VL vision tower + LP head",
                "deployment_llm": "no",
            }
        )
    return rows


def final_audit_rows() -> list[dict[str, Any]]:
    checks = [
        {
            "requirement": "Preserve old document as baseline only",
            "status": "completed",
            "evidence": "task_plan.md active scope; qwen3vl_pilot_matrix P0/P1 closed historical baseline/control",
            "boundary": "Old MIMIC line was not extended after user stop instruction.",
        },
        {
            "requirement": "Create Qwen3-VL v2 proposal document",
            "status": "completed" if (ROOT / "vivid_med_clinical_instruction_proposal_v2_qwen_vlm.md").exists() else "missing",
            "evidence": "vivid_med_clinical_instruction_proposal_v2_qwen_vlm.md",
            "boundary": "",
        },
        {
            "requirement": "Audit local Qwen3-VL components and freeze plan",
            "status": "completed" if (FINAL_DIR / "qwen3vl_component_audit.json").exists() else "missing",
            "evidence": "outputs/final_tables/qwen3vl_component_audit.{json,md}",
            "boundary": "",
        },
        {
            "requirement": "Generate/validate D0-D4 instruction data",
            "status": "completed" if (FINAL_DIR / "instruction_data_audit.md").exists() else "missing",
            "evidence": "outputs/final_tables/instruction_data_audit.{csv,md}; outputs/instruction_data/glm_validated",
            "boundary": "Rejected rows are preserved as data-quality evidence.",
        },
        {
            "requirement": "Run P2-P5 Qwen3-VL instruction training with frozen language decoder",
            "status": "completed",
            "evidence": "outputs/final_tables/qwen3vl_instruction_training_results.{csv,md}",
            "boundary": "All four runs reached 1000 steps; language_decoder_trainable is 0.",
        },
        {
            "requirement": "Extract trained vision-side checkpoints",
            "status": "completed",
            "evidence": "outputs/final_tables/qwen3vl_extraction_manifest.{csv,md}; outputs/qwen3vl_extracted/*/manifest.json",
            "boundary": "Manifests export vision tower, connector, and combined vision-side states.",
        },
        {
            "requirement": "Run CheXpert LP for Base/P2-P6",
            "status": "completed",
            "evidence": "outputs/final_tables/qwen3vl_vision_lp_results.{csv,md}",
            "boundary": "P6 is a no-LM label-head control, not a GLM D3 instruction-head equivalent.",
        },
        {
            "requirement": "Run NIH transfer",
            "status": "completed",
            "evidence": "outputs/final_tables/qwen3vl_nih_transfer_results.{csv,md}",
            "boundary": "Evaluated NIH external 1k subset with 0 missing images, not the full NIH external test.",
        },
        {
            "requirement": "Run visual-dependence diagnostics",
            "status": "completed",
            "evidence": "outputs/final_tables/qwen3vl_visual_dependence_results.{csv,md}",
            "boundary": "Question-only deltas are large; image-shuffle deltas remain small, so strong image-specific grounding is not validated.",
        },
        {
            "requirement": "Run counterfactual diagnostics",
            "status": "completed",
            "evidence": "outputs/final_tables/qwen3vl_counterfactual_results*.{csv,md}",
            "boundary": "Applies to P4/P5 option-formatted subset; most counterfactual_choice rows are not explicit A/B/C/D options.",
        },
        {
            "requirement": "Run paraphrase/template diagnostics",
            "status": "completed",
            "evidence": "outputs/final_tables/qwen3vl_paraphrase_robustness_results*.{csv,md}",
            "boundary": "Style rewrites are consistently harder; robustness is measured but not fully solved.",
        },
        {
            "requirement": "Run clinical subgroup summary",
            "status": "completed",
            "evidence": "outputs/final_tables/qwen3vl_subgroup_results.{csv,md}",
            "boundary": "Support Devices is not evaluated because the active LP configs use the common 8-label subset.",
        },
        {
            "requirement": "Run cost/runtime summary",
            "status": "completed",
            "evidence": "outputs/final_tables/qwen3vl_cost_table.{csv,md}",
            "boundary": "GPU-hours and steps/sec are recorded; peak VRAM was not captured by training metrics.",
        },
        {
            "requirement": "Evaluate optional P7 LoRA upper bound",
            "status": "optional_skipped",
            "evidence": "qwen3vl_pilot_matrix marks P7 optional",
            "boundary": "Not required for P2-P6 artifact-backed completion.",
        },
        {
            "requirement": "State final scientific claim boundary",
            "status": "completed",
            "evidence": "qwen3vl_visual_dependence_results; qwen3vl_nih_transfer_results; qwen3vl_vision_lp_results",
            "boundary": "P4 is strongest by CheXpert/NIH macro-AUC among instruction runs, but image-shuffle evidence does not support a strong image-specific grounding claim.",
        },
    ]
    return checks


def main() -> None:
    instruction = instruction_rows()
    lp = lp_rows()
    extraction = extraction_rows()
    visual = visual_rows()
    counterfactual, counterfactual_by_type, answer_type = counterfactual_rows()
    paraphrase, paraphrase_by_answer_type = paraphrase_rows()
    transfer = transfer_rows()
    subgroups = subgroup_rows(lp)
    pilots = pilot_rows(instruction, lp)
    costs = cost_rows(instruction, lp)
    final_audit = final_audit_rows()

    instruction_cols = [
        "pilot_id",
        "run",
        "data",
        "global_step",
        "best_val_loss",
        "train_records",
        "val_records",
        "elapsed_seconds",
        "vision_trainable",
        "connector_trainable",
        "language_decoder_trainable",
        "total_trainable",
        "metrics_path",
    ]
    lp_cols = [
        "pilot_id",
        "run",
        "checkpoint_source",
        "global_step",
        "best_val_loss",
        "final_val_loss",
        "macro_auc",
        "macro_f1",
        "micro_f1",
        "train_records",
        "val_records",
        "feature_dim",
        "elapsed_seconds",
        "metrics_path",
    ]
    extraction_cols = [
        "pilot_id",
        "run",
        "checkpoint",
        "global_step",
        "best_val_loss",
        "missing_keys",
        "unexpected_keys",
        "vision_state",
        "connector_state",
        "combined_state",
        "vision_trainable",
        "connector_trainable",
        "language_decoder_trainable",
        "total_trainable",
        "manifest_path",
    ]
    visual_cols = [
        "pilot_id",
        "run",
        "examples",
        "normal_loss",
        "question_only_loss",
        "question_only_delta",
        "image_shuffle_loss",
        "image_shuffle_delta",
        "diagnostic_path",
        "boundary",
    ]
    counterfactual_cols = [
        "pilot_id",
        "run",
        "total_counterfactual_records",
        "option_formatted_records",
        "no_option_records",
        "correct_letter_failures",
        "pairwise_accuracy",
        "mean_best_negative_minus_correct_nll",
        "median_best_negative_minus_correct_nll",
        "diagnostic_path",
        "boundary",
    ]
    counterfactual_type_cols = [
        "pilot_id",
        "run",
        "counterfactual_type",
        "total_records",
        "option_formatted_records",
        "no_option_records",
        "correct_letter_failures",
        "pairwise_accuracy",
        "mean_best_negative_minus_correct_nll",
        "median_best_negative_minus_correct_nll",
    ]
    answer_type_cols = ["pilot_id", "run", "answer_type", "n", "mean_nll", "median_nll", "min_nll", "max_nll"]
    paraphrase_cols = [
        "pilot_id",
        "run",
        "variant",
        "n",
        "original_nll_mean",
        "variant_nll_mean",
        "mean_delta_vs_original",
        "median_delta_vs_original",
        "relative_delta_vs_original",
        "variant_worse_rate",
        "diagnostic_path",
        "boundary",
    ]
    paraphrase_answer_type_cols = [
        "pilot_id",
        "run",
        "answer_type",
        "variant",
        "n",
        "original_nll_mean",
        "variant_nll_mean",
        "mean_delta_vs_original",
        "median_delta_vs_original",
        "relative_delta_vs_original",
        "variant_worse_rate",
    ]
    transfer_cols = [
        "pilot_id",
        "run",
        "evaluated_records",
        "missing_images",
        "val_loss",
        "macro_auc",
        "macro_f1",
        "micro_f1",
        "elapsed_seconds",
        "metrics_path",
        "boundary",
    ]
    subgroup_cols = [
        "pilot_id",
        "run",
        "common_auc",
        "rare_auc",
        "rare_labels",
        "high_null_auc",
        "high_null_labels",
        "uncertain_heavy_auc",
        "uncertain_heavy_labels",
        "support_device_auc",
        "notes",
    ]
    pilot_cols = ["pilot_id", "route", "status", "training", "lp_macro_auc", "diagnostics", "boundary"]
    cost_cols = [
        "run",
        "model",
        "trainable_params",
        "frozen_params",
        "peak_vram",
        "gpu_hours",
        "steps_per_sec",
        "deployment_model",
        "deployment_llm",
    ]
    final_audit_cols = ["requirement", "status", "evidence", "boundary"]

    write_csv(FINAL_DIR / "qwen3vl_instruction_training_results.csv", instruction, instruction_cols)
    write_md(
        FINAL_DIR / "qwen3vl_instruction_training_results.md",
        "Qwen3-VL Instruction Training Results",
        instruction,
        instruction_cols,
        "P2-P5 all reached 1000 steps with the language decoder frozen and only the vision tower plus visual connector trainable.",
    )

    write_csv(FINAL_DIR / "qwen3vl_vision_lp_results.csv", lp, lp_cols)
    write_md(
        FINAL_DIR / "qwen3vl_vision_lp_results.md",
        "Qwen3-VL Vision LP Results",
        lp,
        lp_cols,
        "CheXpert 1k LP is complete for base and P2-P5. These are representation metrics, not visual-grounding proof.",
    )

    write_csv(FINAL_DIR / "qwen3vl_extraction_manifest.csv", extraction, extraction_cols)
    write_md(
        FINAL_DIR / "qwen3vl_extraction_manifest.md",
        "Qwen3-VL Extraction Manifest",
        extraction,
        extraction_cols,
        "All refreshed manifests report language decoder trainable count 0 and export vision tower, connector, and combined vision-side states.",
    )

    write_csv(FINAL_DIR / "qwen3vl_visual_dependence_results.csv", visual, visual_cols)
    write_md(
        FINAL_DIR / "qwen3vl_visual_dependence_results.md",
        "Qwen3-VL Visual Dependence Results",
        visual,
        visual_cols,
        "All P2-P5 runs show large loss increases for black-image question-only evaluation, while image-shuffle deltas remain small. This supports image presence sensitivity but not strong image-specific grounding.",
    )

    write_csv(FINAL_DIR / "qwen3vl_counterfactual_results.csv", counterfactual, counterfactual_cols)
    write_md(
        FINAL_DIR / "qwen3vl_counterfactual_results.md",
        "Qwen3-VL Counterfactual Results",
        counterfactual,
        counterfactual_cols,
        "P4 and P5 both prefer the correct option on the option-formatted subset, but most counterfactual_choice rows are not explicit A/B/C/D option records.",
    )

    write_csv(FINAL_DIR / "qwen3vl_counterfactual_results_by_type.csv", counterfactual_by_type, counterfactual_type_cols)
    write_md(
        FINAL_DIR / "qwen3vl_counterfactual_results_by_type.md",
        "Qwen3-VL Counterfactual Results By Type",
        counterfactual_by_type,
        counterfactual_type_cols,
        "State-choice rows carry most of the reliable pairwise signal; laterality/location/device subsets are weaker or sparsely formatted.",
    )

    write_csv(FINAL_DIR / "qwen3vl_answer_type_diagnostics.csv", answer_type, answer_type_cols)
    write_md(
        FINAL_DIR / "qwen3vl_answer_type_diagnostics.md",
        "Qwen3-VL Answer Type Diagnostics",
        answer_type,
        answer_type_cols,
        "Counterfactual-choice dominates P4/P5 validation. Sparse answer types should not be overinterpreted.",
    )

    write_csv(FINAL_DIR / "qwen3vl_paraphrase_robustness_results.csv", paraphrase, paraphrase_cols)
    write_md(
        FINAL_DIR / "qwen3vl_paraphrase_robustness_results.md",
        "Qwen3-VL Paraphrase Robustness Results",
        paraphrase,
        paraphrase_cols,
        "All P2-P5 paraphrase/template diagnostics completed. Style rewrites are consistently harder than clinical rewrites, so template sensitivity remains a measured limitation rather than a solved property.",
    )

    write_csv(FINAL_DIR / "qwen3vl_paraphrase_robustness_by_answer_type.csv", paraphrase_by_answer_type, paraphrase_answer_type_cols)
    write_md(
        FINAL_DIR / "qwen3vl_paraphrase_robustness_by_answer_type.md",
        "Qwen3-VL Paraphrase Robustness By Answer Type",
        paraphrase_by_answer_type,
        paraphrase_answer_type_cols,
        "Per-answer-type paraphrase rows are diagnostic only; sparse categories should be read as qualitative failure probes.",
    )

    write_csv(FINAL_DIR / "qwen3vl_nih_transfer_results.csv", transfer, transfer_cols)
    write_md(
        FINAL_DIR / "qwen3vl_nih_transfer_results.md",
        "Qwen3-VL NIH Transfer Results",
        transfer,
        transfer_cols,
        "NIH transfer was evaluated on a 1000-image external subset with image verification. P4 is the best Qwen3-VL instruction run by NIH macro-AUC, but the margin over the base probe is small.",
    )

    write_csv(FINAL_DIR / "qwen3vl_subgroup_results.csv", subgroups, subgroup_cols)
    write_md(
        FINAL_DIR / "qwen3vl_subgroup_results.md",
        "Qwen3-VL Subgroup Results",
        subgroups,
        subgroup_cols,
        "Subgroups are derived from CheXpert validation label prevalence and uncertainty. Support-device behavior is not evaluated because the common 8-label LP configs exclude Support Devices.",
    )

    write_csv(FINAL_DIR / "qwen3vl_pilot_matrix.csv", pilots, pilot_cols)
    write_md(
        FINAL_DIR / "qwen3vl_pilot_matrix.md",
        "Qwen3-VL Pilot Matrix",
        pilots,
        pilot_cols,
        "P2-P5 have training, LP, extraction, visual-dependence, paraphrase, and NIH transfer evidence. P4/P5 additionally have counterfactual diagnostics; P6 is a no-LM label-head control.",
    )

    write_csv(FINAL_DIR / "qwen3vl_cost_table.csv", costs, cost_cols)
    write_md(
        FINAL_DIR / "qwen3vl_cost_table.md",
        "Qwen3-VL Cost Table",
        costs,
        cost_cols,
        "Runtime-derived GPU-hours and steps/sec are available, but peak VRAM was not captured in the training metrics.",
    )

    write_csv(FINAL_DIR / "qwen3vl_final_requirement_audit.csv", final_audit, final_audit_cols)
    write_md(
        FINAL_DIR / "qwen3vl_final_requirement_audit.md",
        "Qwen3-VL Final Requirement Audit",
        final_audit,
        final_audit_cols,
        "Every required Qwen3-VL v2 item now has a current artifact-backed status or an explicit stop boundary. The main scientific boundary is that image-shuffle deltas remain small.",
    )

    print(
        json.dumps(
            {
                "instruction_rows": len(instruction),
                "lp_rows": len(lp),
                "extraction_rows": len(extraction),
                "visual_rows": len(visual),
                "counterfactual_rows": len(counterfactual),
                "counterfactual_type_rows": len(counterfactual_by_type),
                "answer_type_rows": len(answer_type),
                "paraphrase_rows": len(paraphrase),
                "paraphrase_answer_type_rows": len(paraphrase_by_answer_type),
                "transfer_rows": len(transfer),
                "subgroup_rows": len(subgroups),
                "pilot_rows": len(pilots),
                "cost_rows": len(costs),
                "final_audit_rows": len(final_audit),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
