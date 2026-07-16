"""Summarize real case-study execution artifacts.

This script is intentionally evidence-first: a row is marked complete only when
the artifact named in the execution manifest exists and can be parsed.
"""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path
from typing import Any

from case_study_modules_common import (
    FINAL_DIR,
    ROOT,
    fmt,
    read_csv_rows,
    read_json,
    root_path,
    to_float,
    write_csv_rows,
    write_md_table,
)


FULL_COLUMNS = [
    "run_id",
    "id",
    "family",
    "seed",
    "train_status",
    "train_step",
    "train_best_val_loss",
    "chexpert_macro_auc",
    "chexpert_macro_auprc",
    "chexpert_macro_f1",
    "nih_subset",
    "nih_records",
    "nih_macro_auc",
    "nih_macro_auprc",
    "nih_macro_f1",
    "nih_macro_ece",
    "hard_shuffle_delta",
    "cf_acc",
    "ab_swap_acc",
    "paraphrase_mean_delta",
    "missing_required",
    "evidence",
]

STABILITY_METRICS = [
    ("chexpert_macro_auc", "CheXpert AUC"),
    ("nih_macro_auc", "NIH AUC"),
    ("hard_shuffle_delta", "Hard shuffle delta"),
    ("cf_acc", "CF acc"),
    ("ab_swap_acc", "A/B-swap acc"),
]

EXTRA_COLUMNS = ["step_id", "step_type", "module", "status", "exists", "force_run", "success_path", "notes"]
MODULE_COLUMNS = [
    "module",
    "status",
    "global_step",
    "best_val_loss",
    "state_accuracy",
    "binary_auc",
    "binary_auprc",
    "binary_f1",
    "binary_ece",
    "train_records",
    "val_records",
    "embedding_dim",
    "evidence",
]

NIH_COLUMNS = [
    "run_id",
    "id",
    "family",
    "seed",
    "subset",
    "records",
    "macro_auc",
    "macro_auprc",
    "macro_f1",
    "macro_ece",
    "status",
    "evidence",
]


def path_or_none(raw: Any) -> Path | None:
    if raw in (None, ""):
        return None
    return root_path(str(raw))


def exists(raw: Any) -> bool:
    path = path_or_none(raw)
    return bool(path and path.exists())


def is_fresh(raw: Any, reference: Path) -> bool:
    path = path_or_none(raw)
    if not path or not path.exists() or not reference.exists():
        return False
    return path.stat().st_mtime >= reference.stat().st_mtime


def family_from_run_id(run_id: str, run_key: str) -> str:
    if "-seed" in run_id:
        return run_id.rsplit("-seed", 1)[0]
    if "_seed" in run_key:
        return run_key.rsplit("_seed", 1)[0]
    return run_id or run_key


def latest_progress(train_dir: Path) -> dict[str, Any]:
    payload = read_json(train_dir / "progress.json")
    if not payload:
        return {}
    events = payload.get("events") or []
    logs = [event for event in events if isinstance(event, dict) and event.get("event") == "train_log"]
    if logs:
        return logs[-1]
    return events[-1] if events and isinstance(events[-1], dict) else {}


def train_status(train_dir: Path) -> tuple[str, dict[str, Any]]:
    metrics = read_json(train_dir / "metrics_final.json")
    if metrics:
        return "completed", metrics
    progress = latest_progress(train_dir)
    if progress:
        return "running", progress
    if train_dir.exists():
        return "created", {}
    return "pending", {}


def nested(payload: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def metric(payload: dict[str, Any] | None, *keys: str) -> str:
    return fmt(nested(payload, *keys))


def visual_delta(payload: dict[str, Any] | None, preferred: str) -> str:
    if not payload:
        return ""
    rows = payload.get("modes") or []
    for row in rows:
        if isinstance(row, dict) and row.get("mode") == preferred:
            return fmt(row.get("delta_vs_normal"))
    return ""


def paraphrase_delta(payload: dict[str, Any] | None) -> str:
    overall = nested(payload, "summary", "overall")
    if not isinstance(overall, dict):
        return ""
    values = []
    for item in overall.values():
        if isinstance(item, dict):
            value = to_float(item.get("mean_delta_vs_original"))
            if value is not None:
                values.append(value)
    if not values:
        return ""
    return fmt(sum(values) / len(values))


def summarize_full_rows(manifest: Path) -> list[dict[str, Any]]:
    rows = []
    for item in read_csv_rows(manifest):
        run_id = item.get("run_id", "")
        run_key = item.get("id", "")
        train_dir = root_path(item["train_output_dir"])
        status, train_payload = train_status(train_dir)
        train_metrics = read_json(train_dir / "metrics_final.json")
        lp_payload = read_json(root_path(item["lp_output_dir"]) / "metrics_final.json")
        nih_payload = read_json(root_path(item["nih_output_dir"]) / "transfer_metrics.json")
        visual_payload = read_json(item.get("visual_output", ""))
        cf_payload = read_json(item.get("counterfactual_output", ""))
        ab_payload = read_json(item.get("ab_swap_output", ""))
        para_payload = read_json(item.get("paraphrase_output", ""))

        required = {
            "train": bool(train_metrics),
            "lp": bool(lp_payload),
            "nih_available": bool(nih_payload),
            "visual": bool(visual_payload),
            "counterfactual": bool(cf_payload),
            "ab_swap": bool(ab_payload),
        }
        missing = [name for name, ok in required.items() if not ok]
        train_step = train_payload.get("global_step") or train_payload.get("step")
        if train_metrics:
            train_step = train_metrics.get("global_step", train_step)
        evidence_bits = []
        for key in ["train_output_dir", "lp_output_dir", "nih_output_dir", "visual_output", "counterfactual_output", "ab_swap_output"]:
            raw = item.get(key)
            if raw:
                evidence_bits.append(str(raw).replace("\\", "/"))

        rows.append(
            {
                "run_id": run_id,
                "id": run_key,
                "family": family_from_run_id(run_id, run_key),
                "seed": item.get("seed", ""),
                "train_status": status,
                "train_step": train_step or "",
                "train_best_val_loss": fmt(train_metrics.get("best_val_loss") if train_metrics else None),
                "chexpert_macro_auc": metric(lp_payload, "metrics", "macro_auc"),
                "chexpert_macro_auprc": metric(lp_payload, "metrics", "macro_auprc"),
                "chexpert_macro_f1": metric(lp_payload, "metrics", "macro_f1"),
                "nih_subset": nih_payload.get("max_samples", "") if nih_payload else "",
                "nih_records": nih_payload.get("evaluated_records", "") if nih_payload else "",
                "nih_macro_auc": metric(nih_payload, "metrics", "macro_auc"),
                "nih_macro_auprc": metric(nih_payload, "metrics", "macro_auprc"),
                "nih_macro_f1": metric(nih_payload, "metrics", "macro_f1"),
                "nih_macro_ece": metric(nih_payload, "metrics", "macro_ece"),
                "hard_shuffle_delta": visual_delta(visual_payload, "hard_shuffle") or visual_delta(visual_payload, "image_shuffle"),
                "cf_acc": metric(cf_payload, "summary", "counterfactual_option_nll", "overall", "pairwise_accuracy"),
                "ab_swap_acc": metric(ab_payload, "summary", "counterfactual_option_nll", "overall", "pairwise_accuracy"),
                "paraphrase_mean_delta": paraphrase_delta(para_payload),
                "missing_required": ";".join(missing),
                "evidence": ";".join(evidence_bits),
            }
        )
    return rows


def mean_std(values: list[float]) -> tuple[str, str]:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return "", ""
    if len(clean) == 1:
        return fmt(clean[0]), ""
    return fmt(statistics.mean(clean)), fmt(statistics.stdev(clean))


def summarize_stability(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    families = sorted({str(row.get("family") or "") for row in rows})
    out = []
    for family in families:
        family_rows = [row for row in rows if row.get("family") == family]
        for key, label in STABILITY_METRICS:
            values = [to_float(row.get(key)) for row in family_rows]
            clean = [value for value in values if value is not None]
            mean, std = mean_std(clean)
            out.append(
                {
                    "family": family,
                    "metric": label,
                    "mean": mean,
                    "std": std,
                    "completed_seeds": len(clean),
                    "expected_seeds": len(family_rows),
                    "status": "complete" if len(clean) == len(family_rows) and family_rows else "pending",
                }
            )
    return out


def summarize_nih_rows(manifest: Path) -> list[dict[str, Any]]:
    rows = []
    for item in read_csv_rows(manifest):
        path = root_path(item["nih_output_dir"]) / "transfer_metrics.json"
        payload = read_json(path)
        run_id = item.get("run_id", "")
        family = family_from_run_id(run_id, item.get("id", ""))
        if not payload:
            rows.append(
                {
                    "run_id": run_id,
                    "id": item.get("id", ""),
                    "family": family,
                    "seed": item.get("seed", ""),
                    "subset": "all_available",
                    "records": "",
                    "macro_auc": "",
                    "macro_auprc": "",
                    "macro_f1": "",
                    "macro_ece": "",
                    "status": "pending",
                    "evidence": path.as_posix(),
                }
            )
            continue
        subsets = payload.get("subset_metrics")
        if isinstance(subsets, dict):
            for name, subset in subsets.items():
                metrics = subset.get("metrics", {}) if isinstance(subset, dict) else {}
                rows.append(
                    {
                        "run_id": run_id,
                        "id": item.get("id", ""),
                        "family": family,
                        "seed": item.get("seed", ""),
                        "subset": name,
                        "records": subset.get("records", "") if isinstance(subset, dict) else "",
                        "macro_auc": fmt(metrics.get("macro_auc")),
                        "macro_auprc": fmt(metrics.get("macro_auprc")),
                        "macro_f1": fmt(metrics.get("macro_f1")),
                        "macro_ece": fmt(metrics.get("macro_ece")),
                        "status": "completed" if metrics else subset.get("status", "pending"),
                        "evidence": path.as_posix(),
                    }
                )
        else:
            metrics = payload.get("metrics", {})
            rows.append(
                {
                    "run_id": run_id,
                    "id": item.get("id", ""),
                    "family": family,
                    "seed": item.get("seed", ""),
                    "subset": payload.get("max_samples", ""),
                    "records": payload.get("evaluated_records", ""),
                    "macro_auc": fmt(metrics.get("macro_auc")),
                    "macro_auprc": fmt(metrics.get("macro_auprc")),
                    "macro_f1": fmt(metrics.get("macro_f1")),
                    "macro_ece": fmt(metrics.get("macro_ece")),
                    "status": "completed",
                    "evidence": path.as_posix(),
                }
            )
    return rows


def summarize_extra_rows(manifest: Path) -> list[dict[str, Any]]:
    rows = []
    for item in read_csv_rows(manifest):
        success_path = item.get("success_path", "")
        force_run = str(item.get("force_run", "")).strip() in {"1", "true", "True", "yes"}
        exists_now = exists(success_path)
        if force_run and exists_now and not is_fresh(success_path, manifest):
            status = "rerun_required"
        else:
            status = "completed" if exists_now else "pending"
        rows.append(
            {
                "step_id": item.get("step_id", ""),
                "step_type": item.get("step_type", ""),
                "module": item.get("module", ""),
                "status": status,
                "exists": "1" if exists_now else "0",
                "force_run": "1" if force_run else "0",
                "success_path": success_path,
                "notes": item.get("notes", ""),
            }
        )
    return rows


def summarize_module_rows(extra_manifest: Path) -> list[dict[str, Any]]:
    rows = []
    for item in read_csv_rows(extra_manifest):
        if item.get("step_type") != "module_ablation":
            continue
        payload = read_json(item.get("success_path", ""))
        final = payload.get("final", {}) if payload else {}
        binary = final.get("binary", {}) if isinstance(final, dict) else {}
        rows.append(
            {
                "module": item.get("module", ""),
                "status": "completed" if payload else "pending",
                "global_step": payload.get("global_step", "") if payload else "",
                "best_val_loss": fmt(payload.get("best_val_loss") if payload else None),
                "state_accuracy": fmt(final.get("state_accuracy") if isinstance(final, dict) else None),
                "binary_auc": fmt(binary.get("auc") if isinstance(binary, dict) else None),
                "binary_auprc": fmt(binary.get("auprc") if isinstance(binary, dict) else None),
                "binary_f1": fmt(binary.get("f1") if isinstance(binary, dict) else None),
                "binary_ece": fmt(binary.get("ece") if isinstance(binary, dict) else None),
                "train_records": payload.get("train_records", "") if payload else "",
                "val_records": payload.get("val_records", "") if payload else "",
                "embedding_dim": payload.get("embedding_dim", "") if payload else "",
                "evidence": item.get("success_path", ""),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-manifest", type=Path, default=FINAL_DIR / "case_study_full_execution_manifest.csv")
    parser.add_argument("--extra-manifest", type=Path, default=FINAL_DIR / "case_study_extra_execution_manifest.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_rows = summarize_full_rows(root_path(args.full_manifest))
    write_csv_rows(FINAL_DIR / "case_study_full_execution_status.csv", full_rows, FULL_COLUMNS)
    write_md_table(
        FINAL_DIR / "case_study_full_execution_status.md",
        "Case Study Full Execution Status",
        full_rows,
        FULL_COLUMNS,
        "A row is complete only when train, LP, NIH available transfer, visual, CF, and A/B-swap artifacts are present.",
    )
    write_csv_rows(FINAL_DIR / "multiseed_stability.csv", full_rows, FULL_COLUMNS)
    write_md_table(
        FINAL_DIR / "multiseed_stability.md",
        "Multi-Seed Stability",
        full_rows,
        FULL_COLUMNS,
        "This table is generated from real execution artifacts; missing cells are pending evidence, not zero-valued results.",
    )
    stability = summarize_stability(full_rows)
    stability_columns = ["family", "metric", "mean", "std", "completed_seeds", "expected_seeds", "status"]
    write_csv_rows(FINAL_DIR / "multiseed_stability_summary.csv", stability, stability_columns)
    write_md_table(FINAL_DIR / "multiseed_stability_summary.md", "Multi-Seed Stability Summary", stability, stability_columns)
    nih_rows = summarize_nih_rows(root_path(args.full_manifest))
    write_csv_rows(FINAL_DIR / "nih_available_transfer_status.csv", nih_rows, NIH_COLUMNS)
    write_md_table(
        FINAL_DIR / "nih_available_transfer_status.md",
        "NIH Available Transfer Status",
        nih_rows,
        NIH_COLUMNS,
        "Full transfer JSONs produced by the current evaluator include NIH-1k, NIH-5k, and all_available subset metrics from one pass.",
    )

    extra_rows = summarize_extra_rows(root_path(args.extra_manifest))
    write_csv_rows(FINAL_DIR / "case_study_extra_execution_status.csv", extra_rows, EXTRA_COLUMNS)
    write_md_table(FINAL_DIR / "case_study_extra_execution_status.md", "Case Study Extra Execution Status", extra_rows, EXTRA_COLUMNS)
    module_rows = summarize_module_rows(root_path(args.extra_manifest))
    write_csv_rows(FINAL_DIR / "module_ablation_results.csv", module_rows, MODULE_COLUMNS)
    write_md_table(
        FINAL_DIR / "module_ablation_results.md",
        "Formal Module Ablation Results",
        module_rows,
        MODULE_COLUMNS,
        "CEQ/AUCH/HNMB/DRA/CCSH/CDCS rows require metrics_final.json from formal module training, not smoke tests.",
    )
    print(
        {
            "full_rows": len(full_rows),
            "extra_rows": len(extra_rows),
            "module_rows": len(module_rows),
            "output_dir": str(FINAL_DIR.relative_to(ROOT)),
        }
    )


if __name__ == "__main__":
    main()
