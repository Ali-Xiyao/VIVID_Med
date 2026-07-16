"""Summarize P4-v2 / scale experiment artifacts into final tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs/final_tables"


RUNS = [
    {
        "run_id": "S-P4-1k",
        "data": "old P4 / D3",
        "train_dir": ROOT / "outputs/qwen3vl_instruction_runs/p4_d3_report_grounded_counterfactual",
        "lp_dir": ROOT / "outputs/qwen3vl_lp_runs/p4_d3_report_grounded_counterfactual_chexpert_1k",
        "transfer_dir": ROOT / "outputs/qwen3vl_transfer/p4_nih_1k",
        "diag_prefix": "p4",
        "legacy_diag": True,
    },
    {
        "run_id": "S-P4-3k",
        "data": "old P4 / D3",
        "train_dir": ROOT / "outputs/qwen3vl_instruction/s_p4_3k",
        "lp_dir": ROOT / "outputs/qwen3vl_p4v2_lp_runs/s_p4_3k_chexpert_1k",
        "transfer_dir": ROOT / "outputs/qwen3vl_p4v2_transfer/s_p4_3k_nih_1k",
        "diag_prefix": "s_p4_3k",
    },
    {
        "run_id": "S-P4-5k",
        "data": "old P4 / D3",
        "train_dir": ROOT / "outputs/qwen3vl_instruction/s_p4_5k",
        "lp_dir": ROOT / "outputs/qwen3vl_p4v2_lp_runs/s_p4_5k_chexpert_1k",
        "transfer_dir": ROOT / "outputs/qwen3vl_p4v2_transfer/s_p4_5k_nih_1k",
        "diag_prefix": "s_p4_5k",
    },
    {
        "run_id": "S-P4-8k",
        "data": "old P4 / D3",
        "train_dir": ROOT / "outputs/qwen3vl_instruction/s_p4_8k",
        "lp_dir": ROOT / "outputs/qwen3vl_p4v2_lp_runs/s_p4_8k_chexpert_1k",
        "transfer_dir": ROOT / "outputs/qwen3vl_p4v2_transfer/s_p4_8k_nih_1k",
        "diag_prefix": "s_p4_8k",
    },
    {
        "run_id": "CF-1k-3k",
        "data": "D6 hard CF 1k",
        "train_dir": ROOT / "outputs/qwen3vl_instruction/cf_1k_3k",
        "lp_dir": ROOT / "outputs/qwen3vl_p4v2_lp_runs/cf_1k_3k_chexpert_1k",
        "transfer_dir": ROOT / "outputs/qwen3vl_p4v2_transfer/cf_1k_3k_nih_1k",
        "diag_prefix": "cf_1k_3k",
    },
    {
        "run_id": "CF-3k-5k",
        "data": "D6 hard CF 3k",
        "train_dir": ROOT / "outputs/qwen3vl_instruction/cf_3k_5k",
        "lp_dir": ROOT / "outputs/qwen3vl_p4v2_lp_runs/cf_3k_5k_chexpert_1k",
        "transfer_dir": ROOT / "outputs/qwen3vl_p4v2_transfer/cf_3k_5k_nih_1k",
        "diag_prefix": "cf_3k_5k",
    },
    {
        "run_id": "SHUF-3k",
        "data": "D7 hard shuffle 3k",
        "train_dir": ROOT / "outputs/qwen3vl_instruction/shuf_3k_5k",
        "lp_dir": ROOT / "outputs/qwen3vl_p4v2_lp_runs/shuf_3k_5k_chexpert_1k",
        "transfer_dir": ROOT / "outputs/qwen3vl_p4v2_transfer/shuf_3k_5k_nih_1k",
        "diag_prefix": "shuf_3k_5k",
    },
    {
        "run_id": "CF-3k-8k",
        "data": "D6 hard CF 3k",
        "train_dir": ROOT / "outputs/qwen3vl_instruction/cf_3k_8k",
        "lp_dir": ROOT / "outputs/qwen3vl_p4v2_lp_runs/cf_3k_8k_chexpert_1k",
        "transfer_dir": ROOT / "outputs/qwen3vl_p4v2_transfer/cf_3k_8k_nih_1k",
        "diag_prefix": "cf_3k_8k",
    },
    {
        "run_id": "QA8-3k",
        "data": "D6 hard CF 3k / QA8",
        "train_dir": ROOT / "outputs/qwen3vl_instruction/qa8_3k_5k",
        "lp_dir": ROOT / "outputs/qwen3vl_p4v2_lp_runs/qa8_3k_5k_chexpert_1k",
        "transfer_dir": ROOT / "outputs/qwen3vl_p4v2_transfer/qa8_3k_5k_nih_1k",
        "diag_prefix": "qa8_3k_5k",
    },
]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def nested(payload: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_md(path: Path, title: str, rows: list[dict[str, Any]], columns: list[str], note: str = "") -> None:
    lines = [f"# {title}", ""]
    if note:
        lines.extend([note, ""])
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_rows() -> list[dict[str, Any]]:
    rows = []
    for spec in RUNS:
        metrics = read_json(spec["train_dir"] / "metrics_final.json")
        rows.append(
            {
                "run_id": spec["run_id"],
                "data": spec["data"],
                "status": "completed" if metrics else "missing",
                "global_step": fmt(nested(metrics, "global_step")),
                "best_val_loss": fmt(nested(metrics, "best_val_loss")),
                "elapsed_seconds": fmt(nested(metrics, "elapsed_seconds")),
                "train_records": fmt(nested(metrics, "train_records")),
                "val_records": fmt(nested(metrics, "val_records")),
                "language_decoder_trainable": fmt(nested(metrics, "parameter_groups", "language_decoder", "trainable")),
                "path": str(spec["train_dir"] / "metrics_final.json"),
            }
        )
    return rows


def lp_rows() -> list[dict[str, Any]]:
    rows = []
    for spec in RUNS:
        metrics = read_json(spec["lp_dir"] / "metrics_final.json")
        transfer = read_json(spec["transfer_dir"] / "transfer_metrics.json")
        rows.append(
            {
                "run_id": spec["run_id"],
                "data": spec["data"],
                "lp_status": "completed" if metrics else "missing",
                "chexpert_auc": fmt(nested(metrics, "metrics", "macro_auc")),
                "chexpert_f1": fmt(nested(metrics, "metrics", "macro_f1")),
                "nih_auc": fmt(nested(transfer, "metrics", "macro_auc")),
                "nih_f1": fmt(nested(transfer, "metrics", "macro_f1")),
                "lp_elapsed_seconds": fmt(nested(metrics, "elapsed_seconds")),
                "lp_path": str(spec["lp_dir"] / "metrics_final.json"),
                "nih_path": str(spec["transfer_dir"] / "transfer_metrics.json"),
            }
        )
    return rows


def visual_rows() -> list[dict[str, Any]]:
    rows = []
    for spec in RUNS:
        diag_root = ROOT / ("outputs/qwen3vl_diagnostics" if spec.get("legacy_diag") else "outputs/qwen3vl_p4v2_diagnostics")
        payload = read_json(diag_root / f"{spec['diag_prefix']}_visual_dependence.json")
        modes = {row.get("mode"): row for row in (payload or {}).get("modes", [])}
        rows.append(
            {
                "run_id": spec["run_id"],
                "status": "completed" if payload else "missing",
                "correct_image_loss": fmt(nested(modes.get("normal"), "loss")),
                "blank_image_loss": fmt(nested(modes.get("question_only"), "loss")),
                "random_shuffle_loss": fmt(nested(modes.get("image_shuffle"), "loss")),
                "hard_shuffle_loss": fmt(nested(modes.get("hard_shuffle"), "loss")),
                "question_only_delta": fmt(nested(modes.get("question_only"), "delta_vs_normal")),
                "random_shuffle_delta": fmt(nested(modes.get("image_shuffle"), "delta_vs_normal")),
                "hard_shuffle_delta": fmt(nested(modes.get("hard_shuffle"), "delta_vs_normal")),
                "path": str(diag_root / f"{spec['diag_prefix']}_visual_dependence.json"),
            }
        )
    return rows


def counterfactual_rows() -> list[dict[str, Any]]:
    rows = []
    for spec in RUNS:
        diag_root = ROOT / ("outputs/qwen3vl_diagnostics" if spec.get("legacy_diag") else "outputs/qwen3vl_p4v2_diagnostics")
        payload = read_json(diag_root / f"{spec['diag_prefix']}_counterfactual_diagnostics.json")
        overall = nested(payload, "summary", "counterfactual_option_nll", "overall") or {}
        rows.append(
            {
                "run_id": spec["run_id"],
                "status": "completed" if payload else "missing",
                "total_cf_examples": fmt(overall.get("total_records")),
                "valid_ab_examples": fmt(overall.get("option_formatted_records")),
                "cf_acc": fmt(overall.get("pairwise_accuracy")),
                "mean_best_negative_minus_correct_nll": fmt(overall.get("mean_best_negative_minus_correct_nll")),
                "path": str(diag_root / f"{spec['diag_prefix']}_counterfactual_diagnostics.json"),
            }
        )
    return rows


def paraphrase_rows() -> list[dict[str, Any]]:
    rows = []
    for spec in RUNS:
        diag_root = ROOT / ("outputs/qwen3vl_diagnostics" if spec.get("legacy_diag") else "outputs/qwen3vl_p4v2_diagnostics")
        payload = read_json(diag_root / f"{spec['diag_prefix']}_paraphrase_robustness.json")
        overall = (payload or {}).get("summary", {}).get("overall", {})
        clinical = overall.get("clinical_rewrite", {})
        style = overall.get("style_rewrite", {})
        rows.append(
            {
                "run_id": spec["run_id"],
                "status": "completed" if payload else "missing",
                "clinical_delta": fmt(clinical.get("mean_delta_vs_original")),
                "style_delta": fmt(style.get("mean_delta_vs_original")),
                "clinical_worse_rate": fmt(clinical.get("variant_worse_rate")),
                "style_worse_rate": fmt(style.get("variant_worse_rate")),
                "path": str(diag_root / f"{spec['diag_prefix']}_paraphrase_robustness.json"),
            }
        )
    return rows


def subgroup_rows(lp: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for spec in RUNS:
        metrics = read_json(spec["lp_dir"] / "metrics_final.json")
        per_label = nested(metrics, "metrics", "per_label") or {}
        aucs = [float(item["auc"]) for item in per_label.values() if isinstance(item, dict) and item.get("auc") is not None]
        rare = [
            float(item["auc"])
            for item in per_label.values()
            if isinstance(item, dict) and item.get("auc") is not None and float(item.get("support") or 0) < 150
        ]
        rows.append(
            {
                "run_id": spec["run_id"],
                "status": "completed" if metrics else "missing",
                "common_auc": fmt(sum(aucs) / len(aucs) if aucs else None),
                "rare_auc": fmt(sum(rare) / len(rare) if rare else None),
                "high_null_auc": "",
                "uncertain_heavy_auc": "",
                "location_related_auc": "",
                "support_devices_auc": "",
                "notes": "Derived from available CheXpert LP per-label AUC; specialized subgroup splits not yet present.",
            }
        )
    return rows


def cost_rows(train: list[dict[str, Any]], lp: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lp_by_run = {row["run_id"]: row for row in lp}
    rows = []
    for row in train:
        lp_row = lp_by_run.get(row["run_id"], {})
        train_s = float(row["elapsed_seconds"] or 0)
        lp_s = float(lp_row.get("lp_elapsed_seconds") or 0)
        rows.append(
            {
                "run_id": row["run_id"],
                "train_seconds": fmt(train_s or None),
                "lp_seconds": fmt(lp_s or None),
                "gpu_hours": fmt((train_s + lp_s) / 3600 if train_s or lp_s else None),
                "peak_vram": "",
                "notes": "Peak VRAM is not captured unless recorded in logs.",
            }
        )
    return rows


def decision_rows(lp: list[dict[str, Any]], visual: list[dict[str, Any]], cf: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_visual = {row["run_id"]: row for row in visual}
    by_cf = {row["run_id"]: row for row in cf}
    baseline_auc = next((float(row["chexpert_auc"]) for row in lp if row["run_id"] == "S-P4-1k" and row["chexpert_auc"] != ""), None)
    rows = []
    for row in lp:
        auc = float(row["chexpert_auc"]) if row["chexpert_auc"] != "" else None
        hard_delta = by_visual.get(row["run_id"], {}).get("hard_shuffle_delta", "")
        cf_acc = by_cf.get(row["run_id"], {}).get("cf_acc", "")
        decision = "pending"
        if auc is not None and baseline_auc is not None:
            if auc >= baseline_auc + 0.01 and hard_delta not in {"", None} and float(hard_delta) > 0.05:
                decision = "strong_success"
            elif auc >= baseline_auc:
                decision = "partial_success"
            else:
                decision = "negative_or_no_gain"
        rows.append(
            {
                "run_id": row["run_id"],
                "chexpert_auc": row["chexpert_auc"],
                "nih_auc": row["nih_auc"],
                "hard_shuffle_delta": hard_delta,
                "cf_acc": cf_acc,
                "decision": decision,
            }
        )
    return rows


def main() -> None:
    train = train_rows()
    lp = lp_rows()
    visual = visual_rows()
    cf = counterfactual_rows()
    paraphrase = paraphrase_rows()
    subgroup = subgroup_rows(lp)
    cost = cost_rows(train, lp)
    decision = decision_rows(lp, visual, cf)

    tables = [
        ("qwen3vl_p4v2_training_results", "Qwen3-VL P4-v2 Training Results", train, ["run_id", "data", "status", "global_step", "best_val_loss", "elapsed_seconds", "train_records", "val_records", "language_decoder_trainable", "path"], ""),
        ("qwen3vl_p4v2_lp_results", "Qwen3-VL P4-v2 LP Results", lp, ["run_id", "data", "lp_status", "chexpert_auc", "chexpert_f1", "nih_auc", "nih_f1", "lp_elapsed_seconds", "lp_path", "nih_path"], ""),
        ("qwen3vl_p4v2_visual_dependence", "Qwen3-VL P4-v2 Visual Dependence", visual, ["run_id", "status", "correct_image_loss", "blank_image_loss", "random_shuffle_loss", "hard_shuffle_loss", "question_only_delta", "random_shuffle_delta", "hard_shuffle_delta", "path"], "Positive deltas mean perturbations increased teacher-forced answer loss."),
        ("qwen3vl_p4v2_counterfactual", "Qwen3-VL P4-v2 Counterfactual", cf, ["run_id", "status", "total_cf_examples", "valid_ab_examples", "cf_acc", "mean_best_negative_minus_correct_nll", "path"], ""),
        ("qwen3vl_p4v2_paraphrase", "Qwen3-VL P4-v2 Paraphrase", paraphrase, ["run_id", "status", "clinical_delta", "style_delta", "clinical_worse_rate", "style_worse_rate", "path"], ""),
        ("qwen3vl_p4v2_subgroup", "Qwen3-VL P4-v2 Subgroup", subgroup, ["run_id", "status", "common_auc", "rare_auc", "high_null_auc", "uncertain_heavy_auc", "location_related_auc", "support_devices_auc", "notes"], ""),
        ("qwen3vl_p4v2_cost_table", "Qwen3-VL P4-v2 Cost Table", cost, ["run_id", "train_seconds", "lp_seconds", "gpu_hours", "peak_vram", "notes"], ""),
        ("qwen3vl_p4v2_decision_summary", "Qwen3-VL P4-v2 Decision Summary", decision, ["run_id", "chexpert_auc", "nih_auc", "hard_shuffle_delta", "cf_acc", "decision"], ""),
    ]
    for stem, title, rows, columns, note in tables:
        write_csv(FINAL_DIR / f"{stem}.csv", rows, columns)
        write_md(FINAL_DIR / f"{stem}.md", title, rows, columns, note)
    print(json.dumps({"tables": [stem for stem, *_ in tables]}, indent=2))


if __name__ == "__main__":
    main()
