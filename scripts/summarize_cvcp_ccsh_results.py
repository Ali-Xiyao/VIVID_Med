"""Summarize CVCP/CCSH formal training and postprocess artifacts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs" / "final_tables"
TRAIN_MANIFEST = FINAL_DIR / "cvcp_ccsh_training_manifest.csv"
POST_MANIFEST = FINAL_DIR / "cvcp_ccsh_postprocess_manifest.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


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
        values = [format_value(row.get(key, "")).replace("|", "\\|").replace("\n", " ") for key in columns]
        lines.append("| " + " | ".join(values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def nested(payload: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def metric(payload: dict[str, Any] | None, *keys: str) -> float | None:
    value = nested(payload, *keys)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def visual_delta(payload: dict[str, Any] | None, mode: str) -> float | None:
    for row in (payload or {}).get("modes", []):
        if row.get("mode") == mode:
            try:
                return float(row.get("delta_vs_normal"))
            except (TypeError, ValueError):
                return None
    return None


def cf_metric(payload: dict[str, Any] | None, key: str) -> float | None:
    return metric(payload, "summary", "counterfactual_option_nll", "overall", key)


def paraphrase_delta(payload: dict[str, Any] | None) -> float | None:
    if not payload:
        return None
    for path in [
        ("summary", "overall", "mean_delta_vs_original"),
        ("summary", "mean_delta_vs_original"),
        ("overall", "mean_delta_vs_original"),
    ]:
        value = metric(payload, *path)
        if value is not None:
            return value
    rows = payload.get("modes") or payload.get("paraphrases") or []
    values = []
    for row in rows:
        try:
            values.append(float(row.get("delta_vs_original")))
        except (AttributeError, TypeError, ValueError):
            pass
    return sum(values) / len(values) if values else None


def status_for(paths: list[str]) -> str:
    existing = sum(1 for path in paths if path and Path(path).exists())
    if existing == len([p for p in paths if p]):
        return "complete"
    if existing:
        return "partial"
    return "pending"


def main() -> None:
    train_rows = {row["id"]: row for row in read_csv(TRAIN_MANIFEST)}
    post_rows = {row["id"]: row for row in read_csv(POST_MANIFEST)}
    rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []

    for run_id, train in train_rows.items():
        post = post_rows.get(run_id, {})
        train_metrics = read_json(Path(train["train_output_dir"]) / "metrics_final.json")
        lp_metrics = read_json(Path(post.get("lp_output_dir", "")) / "metrics_final.json") if post else None
        nih_metrics = read_json(Path(post.get("nih_output_dir", "")) / "transfer_metrics.json") if post else None
        visual = read_json(post.get("visual_output", "")) if post else None
        cf = read_json(post.get("counterfactual_output", "")) if post else None
        ab = read_json(post.get("ab_swap_output", "")) if post else None
        para = read_json(post.get("paraphrase_output", "")) if post else None
        paths = [
            str(Path(train["train_output_dir"]) / "metrics_final.json"),
            post.get("lp_output_dir", "") and str(Path(post["lp_output_dir"]) / "metrics_final.json"),
            post.get("nih_output_dir", "") and str(Path(post["nih_output_dir"]) / "transfer_metrics.json"),
            post.get("visual_output", ""),
            post.get("counterfactual_output", ""),
            post.get("paraphrase_output", ""),
        ]
        status_rows.append(
            {
                "run_id": run_id,
                "family": train.get("family", ""),
                "status": status_for([p for p in paths if p]),
                "training": "complete" if train_metrics else "pending",
                "lp": "complete" if lp_metrics else "pending",
                "nih_appendix": "complete" if nih_metrics else "pending",
                "visual": "complete" if visual else "pending",
                "counterfactual": "complete" if cf else "pending",
                "ab_swap": "complete" if ab else ("not_applicable_or_pending" if post.get("ab_swap_output") else ""),
                "paraphrase": "complete" if para else "pending",
            }
        )
        rows.append(
            {
                "run_id": run_id,
                "label": train.get("run_id", run_id),
                "family": train.get("family", ""),
                "seed": train.get("seed", ""),
                "planned_steps": train.get("steps", ""),
                "global_step": metric(train_metrics, "global_step"),
                "best_val_loss": metric(train_metrics, "best_val_loss"),
                "train_records": metric(train_metrics, "train_records") or train.get("train_records", ""),
                "val_records": metric(train_metrics, "val_records"),
                "chexpert_auc": metric(lp_metrics, "metrics", "macro_auc"),
                "chexpert_auprc": metric(lp_metrics, "metrics", "macro_auprc"),
                "chexpert_ece": metric(lp_metrics, "metrics", "ece"),
                "nih_appendix_auc_1k": metric(nih_metrics, "metrics", "macro_auc"),
                "nih_appendix_auprc_1k": metric(nih_metrics, "metrics", "macro_auprc"),
                "image_shuffle_delta": visual_delta(visual, "image_shuffle"),
                "hard_shuffle_delta": visual_delta(visual, "hard_shuffle"),
                "cf_acc": cf_metric(cf, "pairwise_accuracy"),
                "cf_option_records": cf_metric(cf, "option_formatted_records"),
                "ab_swap_acc": cf_metric(ab, "pairwise_accuracy"),
                "paraphrase_delta": paraphrase_delta(para),
                "train_elapsed_seconds": metric(train_metrics, "elapsed_seconds"),
                "status": status_rows[-1]["status"],
            }
        )

    result_columns = [
        "run_id",
        "label",
        "family",
        "seed",
        "planned_steps",
        "global_step",
        "best_val_loss",
        "train_records",
        "val_records",
        "chexpert_auc",
        "chexpert_auprc",
        "chexpert_ece",
        "nih_appendix_auc_1k",
        "nih_appendix_auprc_1k",
        "image_shuffle_delta",
        "hard_shuffle_delta",
        "cf_acc",
        "cf_option_records",
        "ab_swap_acc",
        "paraphrase_delta",
        "train_elapsed_seconds",
        "status",
    ]
    write_csv(FINAL_DIR / "cvcp_training_results.csv", rows, result_columns)
    write_md_table(FINAL_DIR / "cvcp_training_results.md", "CVCP Training Results", rows, result_columns)

    status_columns = ["run_id", "family", "status", "training", "lp", "nih_appendix", "visual", "counterfactual", "ab_swap", "paraphrase"]
    write_csv(FINAL_DIR / "cvcp_ccsh_postprocess_status.csv", status_rows, status_columns)
    write_md_table(FINAL_DIR / "cvcp_ccsh_postprocess_status.md", "CVCP/CCSH Postprocess Status", status_rows, status_columns)

    external_rows = [
        {"dataset": "VinDr-CXR", "status": "partial_image_only", "run_id": "", "macro_auc": "", "macro_auprc": "", "notes": "Image package exists, but no label/bbox CSV is available locally; cannot be main external."},
        {"dataset": "PadChest", "status": "missing", "run_id": "", "macro_auc": "", "macro_auprc": "", "notes": "No local PadChest data found."},
        {"dataset": "NIH", "status": "appendix_stress_test", "run_id": "", "macro_auc": "", "macro_auprc": "", "notes": "NIH is not promoted to main external under the target document."},
    ]
    for row in rows:
        if row.get("nih_appendix_auc_1k") is not None:
            external_rows.append(
                {
                    "dataset": "NIH-appendix-1k",
                    "status": "complete",
                    "run_id": row["run_id"],
                    "macro_auc": row.get("nih_appendix_auc_1k"),
                    "macro_auprc": row.get("nih_appendix_auprc_1k"),
                    "notes": "Appendix/stress only.",
                }
            )
    external_columns = ["dataset", "status", "run_id", "macro_auc", "macro_auprc", "notes"]
    write_csv(FINAL_DIR / "external_eval_results.csv", external_rows, external_columns)
    write_md_table(FINAL_DIR / "external_eval_results.md", "External Evaluation Results", external_rows, external_columns)

    write_csv(FINAL_DIR / "cvcp_hard_shuffle_results.csv", rows, ["run_id", "family", "image_shuffle_delta", "hard_shuffle_delta", "status"])
    write_md_table(FINAL_DIR / "cvcp_hard_shuffle_results.md", "CVCP Hard-Shuffle Results", rows, ["run_id", "family", "image_shuffle_delta", "hard_shuffle_delta", "status"])
    write_csv(FINAL_DIR / "cvcp_ab_swap_results.csv", rows, ["run_id", "family", "cf_acc", "cf_option_records", "ab_swap_acc", "status"])
    write_md_table(FINAL_DIR / "cvcp_ab_swap_results.md", "CVCP A/B-Swap Results", rows, ["run_id", "family", "cf_acc", "cf_option_records", "ab_swap_acc", "status"])
    write_csv(FINAL_DIR / "cvcp_calibration_auprc.csv", rows, ["run_id", "family", "chexpert_auprc", "chexpert_ece", "nih_appendix_auprc_1k", "status"])
    write_md_table(FINAL_DIR / "cvcp_calibration_auprc.md", "CVCP Calibration/AUPRC", rows, ["run_id", "family", "chexpert_auprc", "chexpert_ece", "nih_appendix_auprc_1k", "status"])

    cost_rows = [
        {
            "run_id": row["run_id"],
            "model": "Qwen3-VL-2B",
            "gpu": "RTX 3090",
            "hours": (row["train_elapsed_seconds"] / 3600.0) if isinstance(row.get("train_elapsed_seconds"), float) else "",
            "steps": row.get("global_step"),
            "cost_acceptable": "yes" if row.get("global_step") else "pending",
        }
        for row in rows
    ]
    cost_columns = ["run_id", "model", "gpu", "hours", "steps", "cost_acceptable"]
    write_csv(FINAL_DIR / "cost_table.csv", cost_rows, cost_columns)
    write_md_table(FINAL_DIR / "cost_table.md", "Training Cost Table", cost_rows, cost_columns)

    completed = [row for row in rows if row.get("status") == "complete"]
    locked = []
    for family in sorted({row["family"] for row in rows}):
        fam_rows = [row for row in completed if row["family"] == family]
        if not fam_rows:
            locked.append({"family": family, "finalist": "", "seeds": 0, "chexpert_auc": "", "nih_appendix_auc_1k": "", "hard_shuffle_delta": "", "cf_acc": "", "ece": "", "final_role": "pending"})
            continue
        finalist = max(fam_rows, key=lambda item: item.get("chexpert_auc") or -1.0)
        locked.append(
            {
                "family": family,
                "finalist": finalist["run_id"],
                "seeds": len(fam_rows),
                "chexpert_auc": finalist.get("chexpert_auc"),
                "nih_appendix_auc_1k": finalist.get("nih_appendix_auc_1k"),
                "hard_shuffle_delta": finalist.get("hard_shuffle_delta") or finalist.get("image_shuffle_delta"),
                "cf_acc": finalist.get("cf_acc"),
                "ece": finalist.get("chexpert_ece"),
                "final_role": "candidate_after_formal_cvcp_queue",
            }
        )
    locked_columns = ["family", "finalist", "seeds", "chexpert_auc", "nih_appendix_auc_1k", "hard_shuffle_delta", "cf_acc", "ece", "final_role"]
    write_csv(FINAL_DIR / "locked_final_comparison.csv", locked, locked_columns)
    write_md_table(FINAL_DIR / "locked_final_comparison.md", "Locked Final Comparison", locked, locked_columns)
    print(f"rows={len(rows)} complete={sum(1 for row in rows if row.get('status') == 'complete')}")


if __name__ == "__main__":
    main()
