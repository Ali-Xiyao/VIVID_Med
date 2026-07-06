"""Summarize next-stage VIVID-Med experiment artifacts into final tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs/final_tables"
DIAG_DIR = ROOT / "outputs/qwen3vl_next_stage_diagnostics"
TRANSFER_DIR = ROOT / "outputs/qwen3vl_next_stage_transfer"
AB_SWAP_ROOT = ROOT / "outputs/instruction_data/next_stage"


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_first(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return next(reader, None)


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def fmt(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def nested(payload: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def jsonl_row_count(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def ab_swap_input_path(run: dict[str, Any]) -> Path | None:
    stem = Path(str(run.get("val") or "")).stem
    if not stem:
        return None
    candidate = AB_SWAP_ROOT / f"{stem}_ab_swap.jsonl"
    return candidate if candidate.exists() else None


def ab_swap_status(payload: dict[str, Any] | None, input_rows: int | None) -> str:
    if payload:
        return "completed"
    if input_rows == 0:
        return "not_applicable_no_ab_rows"
    return "pending"


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, title: str, rows: list[dict[str, Any]], columns: list[str], note: str = "") -> None:
    lines = [f"# {title}", ""]
    if note:
        lines.extend([note, ""])
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_manifests() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    config_manifest = read_json(ROOT / "outputs/next_stage_manifests/config_manifest.json") or {"configs": []}
    lp_manifest = read_json(ROOT / "outputs/next_stage_manifests/lp_config_manifest.json") or {"configs": []}
    lp_lookup = {str(row.get("id")): row for row in lp_manifest.get("configs", [])}
    return list(config_manifest.get("configs", [])), lp_lookup


def train_dir(run: dict[str, Any]) -> Path:
    config = read_json(Path(str(run["config"])).with_suffix(".json"))
    if config and nested(config, "training", "output_dir"):
        return ROOT / str(nested(config, "training", "output_dir")).replace("./", "")
    return ROOT / "outputs/qwen3vl_instruction/next_stage" / str(run["id"])


def audit_summary_for(run: dict[str, Any]) -> dict[str, Any] | None:
    stem = Path(str(run.get("train") or "")).stem
    return read_csv_first(ROOT / "outputs/final_tables/next_stage_audits" / f"{stem}_leakage_summary.csv")


def training_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        out_dir = train_dir(run)
        metrics = read_json(out_dir / "metrics_final.json")
        rows.append(
            {
                "run_id": run.get("run_id"),
                "id": run.get("id"),
                "status": "completed" if metrics else "pending",
                "global_step": fmt(nested(metrics, "global_step")),
                "best_val_loss": fmt(nested(metrics, "best_val_loss")),
                "elapsed_seconds": fmt(nested(metrics, "elapsed_seconds")),
                "train_records": fmt(nested(metrics, "train_records")),
                "val_records": fmt(nested(metrics, "val_records")),
                "language_decoder_trainable": fmt(nested(metrics, "parameter_groups", "language_decoder", "trainable")),
                "path": rel(out_dir / "metrics_final.json"),
            }
        )
    return rows


def lp_rows(runs: list[dict[str, Any]], lp_lookup: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        lp_spec = lp_lookup.get(str(run.get("id")), {})
        lp_dir = ROOT / str(lp_spec.get("output_dir", "")).replace("./", "")
        lp = read_json(lp_dir / "metrics_final.json")
        transfer = read_json(TRANSFER_DIR / f"{run['id']}_nih_1k" / "transfer_metrics.json")
        rows.append(
            {
                "run_id": run.get("run_id"),
                "id": run.get("id"),
                "lp_status": "completed" if lp else "pending",
                "chexpert_auc": fmt(nested(lp, "metrics", "macro_auc")),
                "chexpert_auprc": fmt(nested(lp, "metrics", "macro_auprc")),
                "chexpert_f1": fmt(nested(lp, "metrics", "macro_f1")),
                "chexpert_ece": fmt(nested(lp, "metrics", "macro_ece")),
                "nih_status": "completed" if transfer else "pending",
                "nih_auc": fmt(nested(transfer, "metrics", "macro_auc")),
                "nih_auprc": fmt(nested(transfer, "metrics", "macro_auprc")),
                "nih_f1": fmt(nested(transfer, "metrics", "macro_f1")),
                "nih_missing_images": fmt(nested(transfer, "image_audit", "missing_count")),
                "lp_path": rel(lp_dir / "metrics_final.json"),
                "nih_path": rel(TRANSFER_DIR / f"{run['id']}_nih_1k" / "transfer_metrics.json"),
            }
        )
    return rows


def mean_per_label_metric(payload: dict[str, Any] | None, labels: set[str], metric: str) -> float | None:
    per_label = nested(payload, "metrics", "per_label") or {}
    values = []
    for label, row in per_label.items():
        if str(label) in labels and isinstance(row, dict):
            value = row.get(metric)
            if isinstance(value, (int, float)):
                values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def calibration_rows(runs: list[dict[str, Any]], lp_lookup: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    rare_labels = {"Consolidation", "Pneumonia", "Pneumothorax"}
    for run in runs:
        lp_spec = lp_lookup.get(str(run.get("id")), {})
        lp_dir = ROOT / str(lp_spec.get("output_dir", "")).replace("./", "")
        lp = read_json(lp_dir / "metrics_final.json")
        transfer = read_json(TRANSFER_DIR / f"{run['id']}_nih_1k" / "transfer_metrics.json")
        rows.append(
            {
                "run_id": run.get("run_id"),
                "id": run.get("id"),
                "status": "completed" if lp else "pending",
                "threshold_policy": "fixed_0.5",
                "chexpert_macro_auprc": fmt(nested(lp, "metrics", "macro_auprc")),
                "chexpert_macro_ece": fmt(nested(lp, "metrics", "macro_ece")),
                "chexpert_macro_brier": fmt(nested(lp, "metrics", "macro_brier")),
                "chexpert_rare_auprc": fmt(mean_per_label_metric(lp, rare_labels, "auprc")),
                "nih_macro_auprc": fmt(nested(transfer, "metrics", "macro_auprc")),
                "nih_macro_ece": fmt(nested(transfer, "metrics", "macro_ece")),
                "nih_macro_brier": fmt(nested(transfer, "metrics", "macro_brier")),
                "nih_rare_auprc": fmt(mean_per_label_metric(transfer, rare_labels, "auprc")),
                "high_null_ece": "",
                "boundary": "high-null ECE requires sample-level null stratification; aggregate LP metrics support macro/rare-label calibration only",
                "lp_path": rel(lp_dir / "metrics_final.json"),
                "nih_path": rel(TRANSFER_DIR / f"{run['id']}_nih_1k" / "transfer_metrics.json"),
            }
        )
    return rows


def visual_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        payload = read_json(DIAG_DIR / f"{run['id']}_visual_dependence.json")
        modes = {row.get("mode"): row for row in (payload or {}).get("modes", [])}
        rows.append(
            {
                "run_id": run.get("run_id"),
                "id": run.get("id"),
                "status": "completed" if payload else "pending",
                "normal_loss": fmt(nested(modes.get("normal"), "loss")),
                "question_only_delta": fmt(nested(modes.get("question_only"), "delta_vs_normal")),
                "image_shuffle_delta": fmt(nested(modes.get("image_shuffle"), "delta_vs_normal")),
                "hard_shuffle_delta": fmt(nested(modes.get("hard_shuffle"), "delta_vs_normal")),
                "path": rel(DIAG_DIR / f"{run['id']}_visual_dependence.json"),
            }
        )
    return rows


def counterfactual_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        payload = read_json(DIAG_DIR / f"{run['id']}_counterfactual_diagnostics.json")
        overall = nested(payload, "summary", "counterfactual_option_nll", "overall") or {}
        rows.append(
            {
                "run_id": run.get("run_id"),
                "id": run.get("id"),
                "status": "completed" if payload else "pending",
                "total_records": fmt(overall.get("total_records")),
                "option_formatted_records": fmt(overall.get("option_formatted_records")),
                "cf_acc": fmt(overall.get("pairwise_accuracy")),
                "mean_best_negative_minus_correct_nll": fmt(overall.get("mean_best_negative_minus_correct_nll")),
                "path": rel(DIAG_DIR / f"{run['id']}_counterfactual_diagnostics.json"),
            }
        )
    return rows


def ab_swap_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        payload = read_json(DIAG_DIR / f"{run['id']}_ab_swap_counterfactual_diagnostics.json")
        overall = nested(payload, "summary", "counterfactual_option_nll", "overall") or {}
        input_path = ab_swap_input_path(run)
        input_rows = jsonl_row_count(input_path)
        status = ab_swap_status(payload, input_rows)
        rows.append(
            {
                "run_id": run.get("run_id"),
                "id": run.get("id"),
                "status": status,
                "ab_swap_input_rows": fmt(input_rows),
                "total_records": fmt(overall.get("total_records")),
                "option_formatted_records": fmt(overall.get("option_formatted_records")),
                "ab_swap_acc": fmt(overall.get("pairwise_accuracy")),
                "mean_best_negative_minus_correct_nll": fmt(overall.get("mean_best_negative_minus_correct_nll")),
                "path": rel(DIAG_DIR / f"{run['id']}_ab_swap_counterfactual_diagnostics.json"),
                "input_path": rel(input_path) if input_path else "",
            }
        )
    return rows


def paraphrase_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        payload = read_json(DIAG_DIR / f"{run['id']}_paraphrase_robustness.json")
        overall = (payload or {}).get("summary", {}).get("overall", {})
        clinical = overall.get("clinical_rewrite", {})
        style = overall.get("style_rewrite", {})
        rows.append(
            {
                "run_id": run.get("run_id"),
                "id": run.get("id"),
                "status": "completed" if payload else "pending",
                "clinical_delta": fmt(clinical.get("mean_delta_vs_original")),
                "style_delta": fmt(style.get("mean_delta_vs_original")),
                "clinical_worse_rate": fmt(clinical.get("variant_worse_rate")),
                "style_worse_rate": fmt(style.get("variant_worse_rate")),
                "path": rel(DIAG_DIR / f"{run['id']}_paraphrase_robustness.json"),
            }
        )
    return rows


def audit_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        audit = audit_summary_for(run)
        rows.append(
            {
                "run_id": run.get("run_id"),
                "id": run.get("id"),
                "status": "completed" if audit else "pending",
                "instructions": fmt((audit or {}).get("instructions")),
                "accepted_pct": fmt((audit or {}).get("accepted_pct")),
                "leakage_or_flag_pct": fmt((audit or {}).get("leakage_or_flag_pct")),
                "answer_a_pct": fmt((audit or {}).get("answer_a_pct")),
                "path": rel(ROOT / "outputs/final_tables/next_stage_audits" / f"{Path(str(run.get('train') or '')).stem}_leakage_summary.csv"),
            }
        )
    return rows


def decision_rows(runs: list[dict[str, Any]], lp: list[dict[str, Any]], visual: list[dict[str, Any]], cf: list[dict[str, Any]], ab_swap: list[dict[str, Any]], audits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lp_by_id = {row["id"]: row for row in lp}
    visual_by_id = {row["id"]: row for row in visual}
    cf_by_id = {row["id"]: row for row in cf}
    ab_swap_by_id = {row["id"]: row for row in ab_swap}
    audit_by_id = {row["id"]: row for row in audits}
    baseline = {
        "chexpert_auc": 0.7267088289073564,
        "nih_auc": 0.5680449234338893,
        "hard_shuffle_delta": 0.08067435596721262,
        "cf_acc": 0.8707482993197279,
    }
    rows = []
    for run in runs:
        rid = str(run.get("id"))
        lp_row = lp_by_id.get(rid, {})
        visual_row = visual_by_id.get(rid, {})
        cf_row = cf_by_id.get(rid, {})
        ab_swap_row = ab_swap_by_id.get(rid, {})
        audit_row = audit_by_id.get(rid, {})
        try_float = lambda value: float(value) if value not in {None, ""} else None
        chexpert = try_float(lp_row.get("chexpert_auc"))
        nih = try_float(lp_row.get("nih_auc"))
        hard_delta = try_float(visual_row.get("hard_shuffle_delta"))
        cf_acc = try_float(cf_row.get("cf_acc"))
        ab_acc = try_float(ab_swap_row.get("ab_swap_acc"))
        leakage = try_float(audit_row.get("leakage_or_flag_pct"))
        a_pct = try_float(audit_row.get("answer_a_pct"))
        passed = (
            chexpert is not None
            and chexpert >= baseline["chexpert_auc"] - 0.003
            and nih is not None
            and nih >= baseline["nih_auc"] - 0.005
            and hard_delta is not None
            and hard_delta >= 0.05
            and cf_acc is not None
            and cf_acc >= 0.85
            and ab_acc is not None
            and ab_acc >= 0.85
            and leakage is not None
            and leakage <= 10.0
            and (a_pct is None or 45.0 <= a_pct <= 55.0)
        )
        rows.append(
            {
                "run_id": run.get("run_id"),
                "id": rid,
                "chexpert_auc": fmt(chexpert),
                "nih_auc": fmt(nih),
                "hard_shuffle_delta": fmt(hard_delta),
                "cf_acc": fmt(cf_acc),
                "ab_swap_acc": fmt(ab_acc),
                "leakage_or_flag_pct": fmt(leakage),
                "answer_a_pct": fmt(a_pct),
                "decision": "candidate" if passed else "pending_or_ablation",
            }
        )
    return rows


def main() -> None:
    runs, lp_lookup = load_manifests()
    train = training_rows(runs)
    lp = lp_rows(runs, lp_lookup)
    visual = visual_rows(runs)
    cf = counterfactual_rows(runs)
    ab_swap = ab_swap_rows(runs)
    paraphrase = paraphrase_rows(runs)
    audits = audit_rows(runs)
    calibration = calibration_rows(runs, lp_lookup)
    decision = decision_rows(runs, lp, visual, cf, ab_swap, audits)
    tables = [
        ("next_stage_training_results", "Next-Stage Training Results", train, ["run_id", "id", "status", "global_step", "best_val_loss", "elapsed_seconds", "train_records", "val_records", "language_decoder_trainable", "path"], ""),
        ("next_stage_lp_transfer_results", "Next-Stage LP and Transfer Results", lp, ["run_id", "id", "lp_status", "chexpert_auc", "chexpert_auprc", "chexpert_f1", "chexpert_ece", "nih_status", "nih_auc", "nih_auprc", "nih_f1", "nih_missing_images", "lp_path", "nih_path"], ""),
        ("next_stage_visual_dependence", "Next-Stage Visual Dependence", visual, ["run_id", "id", "status", "normal_loss", "question_only_delta", "image_shuffle_delta", "hard_shuffle_delta", "path"], "Positive deltas mean perturbations increased teacher-forced answer loss."),
        ("next_stage_counterfactual", "Next-Stage Counterfactual Diagnostics", cf, ["run_id", "id", "status", "total_records", "option_formatted_records", "cf_acc", "mean_best_negative_minus_correct_nll", "path"], ""),
        ("next_stage_ab_swap_counterfactual", "Next-Stage A/B-Swap Counterfactual Diagnostics", ab_swap, ["run_id", "id", "status", "ab_swap_input_rows", "total_records", "option_formatted_records", "ab_swap_acc", "mean_best_negative_minus_correct_nll", "path", "input_path"], "A/B-swap rows swap option order to test option-position bias."),
        ("next_stage_paraphrase", "Next-Stage Paraphrase Robustness", paraphrase, ["run_id", "id", "status", "clinical_delta", "style_delta", "clinical_worse_rate", "style_worse_rate", "path"], ""),
        ("next_stage_instruction_audit", "Next-Stage Instruction Audit", audits, ["run_id", "id", "status", "instructions", "accepted_pct", "leakage_or_flag_pct", "answer_a_pct", "path"], ""),
        ("next_stage_calibration_auprc", "Next-Stage Calibration and AUPRC", calibration, ["run_id", "id", "status", "threshold_policy", "chexpert_macro_auprc", "chexpert_macro_ece", "chexpert_macro_brier", "chexpert_rare_auprc", "nih_macro_auprc", "nih_macro_ece", "nih_macro_brier", "nih_rare_auprc", "high_null_ece", "boundary", "lp_path", "nih_path"], "Calibration metrics are computed from saved aggregate LP/transfer metrics. High-null calibration remains a boundary without sample-level null-stratified probabilities."),
        ("next_stage_decision_summary", "Next-Stage Decision Summary", decision, ["run_id", "id", "chexpert_auc", "nih_auc", "hard_shuffle_delta", "cf_acc", "ab_swap_acc", "leakage_or_flag_pct", "answer_a_pct", "decision"], "Baseline thresholds use completed SHUF-3k P4-v2 evidence."),
    ]
    for stem, title, rows, columns, note in tables:
        write_csv(FINAL_DIR / f"{stem}.csv", rows, columns)
        write_md(FINAL_DIR / f"{stem}.md", title, rows, columns, note)
    print(json.dumps({"tables": [stem for stem, *_ in tables], "runs": len(runs)}, indent=2))


if __name__ == "__main__":
    main()
