"""Write per-run G3 markdown packages for next-stage experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "outputs/final_tables/next_stage_audits"
DIAG_DIR = ROOT / "outputs/qwen3vl_next_stage_diagnostics"
TRANSFER_ROOT = ROOT / "outputs/qwen3vl_next_stage_transfer"
AB_SWAP_ROOT = ROOT / "outputs/instruction_data/next_stage"


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def fmt(value: Any) -> str:
    if value is None:
        return "pending"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


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


def table(rows: list[tuple[str, Any]]) -> str:
    lines = ["| Field | Value |", "| --- | --- |"]
    for key, value in rows:
        lines.append(f"| {key} | {fmt(value)} |")
    return "\n".join(lines)


def write_md(path: Path, title: str, rows: list[tuple[str, Any]], note: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", table(rows)]
    if note:
        lines.extend(["", note])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_paths(train_path: str, val_path: str) -> list[Path]:
    paths = []
    for raw in [train_path, val_path]:
        stem = Path(raw).stem
        candidate = AUDIT_DIR / f"{stem}_leakage_summary.md"
        if candidate.exists():
            paths.append(candidate)
    return paths


def read_first_audit_rate(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if "accepted_rate" in line or "flagged_rate" in line or "leakage" in line:
            return line.strip()
    return "audit summary present"


def package_run(run: dict[str, Any], lp_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    config = read_yaml(Path(str(run["config"])))
    train_dir = ROOT / Path(str(config["training"]["output_dir"]))
    train_metrics = read_json(train_dir / "metrics_final.json")
    lp_spec = lp_lookup.get(str(run["id"]), {})
    lp_dir_raw = lp_spec.get("output_dir")
    lp_dir = ROOT / Path(str(lp_dir_raw).replace("./", "")) if lp_dir_raw else ROOT / "missing_lp_dir"
    lp_metrics = read_json(lp_dir / "metrics_final.json")
    transfer_dir = TRANSFER_ROOT / f"{run['id']}_nih_1k"
    transfer = read_json(transfer_dir / "transfer_metrics.json")
    visual = read_json(DIAG_DIR / f"{run['id']}_visual_dependence.json")
    cf = read_json(DIAG_DIR / f"{run['id']}_counterfactual_diagnostics.json")
    ab_swap = read_json(DIAG_DIR / f"{run['id']}_ab_swap_counterfactual_diagnostics.json")
    ab_input = ab_swap_input_path(run)
    ab_input_rows = jsonl_row_count(ab_input)
    ab_status = ab_swap_status(ab_swap, ab_input_rows)
    paraphrase = read_json(DIAG_DIR / f"{run['id']}_paraphrase_robustness.json")
    export_manifest = train_dir / "vision_export_manifest.json"

    write_md(
        train_dir / "lp_results.md",
        f"{run['run_id']} LP and Transfer Results",
        [
            ("lp_status", "completed" if lp_metrics else "pending"),
            ("lp_metrics_path", rel(lp_dir / "metrics_final.json")),
            ("chexpert_macro_auc", nested(lp_metrics, "metrics", "macro_auc")),
            ("chexpert_macro_auprc", nested(lp_metrics, "metrics", "macro_auprc")),
            ("chexpert_macro_f1", nested(lp_metrics, "metrics", "macro_f1")),
            ("chexpert_macro_ece", nested(lp_metrics, "metrics", "macro_ece")),
            ("transfer_status", "completed" if transfer else "pending"),
            ("nih_metrics_path", rel(transfer_dir / "transfer_metrics.json")),
            ("nih_macro_auc", nested(transfer, "metrics", "macro_auc")),
            ("nih_macro_auprc", nested(transfer, "metrics", "macro_auprc")),
            ("nih_macro_f1", nested(transfer, "metrics", "macro_f1")),
            ("nih_missing_images", nested(transfer, "image_audit", "missing_count")),
        ],
    )

    modes = {row.get("mode"): row for row in (visual or {}).get("modes", [])}
    write_md(
        train_dir / "visual_dependence_results.md",
        f"{run['run_id']} Visual Dependence",
        [
            ("status", "completed" if visual else "pending"),
            ("path", rel(DIAG_DIR / f"{run['id']}_visual_dependence.json")),
            ("normal_loss", nested(modes.get("normal"), "loss")),
            ("question_only_delta", nested(modes.get("question_only"), "delta_vs_normal")),
            ("image_shuffle_delta", nested(modes.get("image_shuffle"), "delta_vs_normal")),
            ("hard_shuffle_delta", nested(modes.get("hard_shuffle"), "delta_vs_normal")),
        ],
        "Positive deltas mean visual perturbation increased teacher-forced answer loss.",
    )

    cf_overall = nested(cf, "summary", "counterfactual_option_nll", "overall") or {}
    write_md(
        train_dir / "counterfactual_results.md",
        f"{run['run_id']} Counterfactual Diagnostics",
        [
            ("status", "completed" if cf else "pending"),
            ("path", rel(DIAG_DIR / f"{run['id']}_counterfactual_diagnostics.json")),
            ("total_records", cf_overall.get("total_records")),
            ("option_formatted_records", cf_overall.get("option_formatted_records")),
            ("pairwise_accuracy", cf_overall.get("pairwise_accuracy")),
            ("mean_best_negative_minus_correct_nll", cf_overall.get("mean_best_negative_minus_correct_nll")),
        ],
    )

    ab_overall = nested(ab_swap, "summary", "counterfactual_option_nll", "overall") or {}
    write_md(
        train_dir / "ab_swap_results.md",
        f"{run['run_id']} A/B-Swap Counterfactual Diagnostics",
        [
            ("status", ab_status),
            ("path", rel(DIAG_DIR / f"{run['id']}_ab_swap_counterfactual_diagnostics.json")),
            ("ab_swap_input_path", rel(ab_input) if ab_input else "missing"),
            ("ab_swap_input_rows", ab_input_rows),
            ("total_records", ab_overall.get("total_records")),
            ("option_formatted_records", ab_overall.get("option_formatted_records")),
            ("pairwise_accuracy", ab_overall.get("pairwise_accuracy")),
            ("mean_best_negative_minus_correct_nll", ab_overall.get("mean_best_negative_minus_correct_nll")),
        ],
        "This diagnostic swaps A/B option order to test option-position bias.",
    )

    overall = (paraphrase or {}).get("summary", {}).get("overall", {})
    clinical = overall.get("clinical_rewrite", {})
    style = overall.get("style_rewrite", {})
    write_md(
        train_dir / "paraphrase_results.md",
        f"{run['run_id']} Paraphrase Robustness",
        [
            ("status", "completed" if paraphrase else "pending"),
            ("path", rel(DIAG_DIR / f"{run['id']}_paraphrase_robustness.json")),
            ("clinical_delta", clinical.get("mean_delta_vs_original")),
            ("clinical_worse_rate", clinical.get("variant_worse_rate")),
            ("style_delta", style.get("mean_delta_vs_original")),
            ("style_worse_rate", style.get("variant_worse_rate")),
        ],
    )

    audits = audit_paths(str(run.get("train", "")), str(run.get("val", "")))
    audit_rows: list[tuple[str, Any]] = [
        ("status", "completed" if audits else "pending"),
        ("train_instruction_path", run.get("train")),
        ("val_instruction_path", run.get("val")),
    ]
    for idx, path in enumerate(audits, start=1):
        audit_rows.append((f"audit_{idx}", rel(path)))
        audit_rows.append((f"audit_{idx}_note", read_first_audit_rate(path)))
    write_md(train_dir / "instruction_audit.md", f"{run['run_id']} Instruction Audit", audit_rows)

    train_seconds = nested(train_metrics, "elapsed_seconds")
    lp_seconds = nested(lp_metrics, "elapsed_seconds")
    transfer_seconds = nested(transfer, "elapsed_seconds")
    total_seconds = sum(float(value or 0.0) for value in [train_seconds, lp_seconds, transfer_seconds])
    write_md(
        train_dir / "cost_table.md",
        f"{run['run_id']} Cost Table",
        [
            ("train_status", "completed" if train_metrics else "pending"),
            ("global_step", nested(train_metrics, "global_step")),
            ("train_seconds", train_seconds),
            ("lp_seconds", lp_seconds),
            ("nih_transfer_seconds", transfer_seconds),
            ("known_gpu_hours", total_seconds / 3600.0 if total_seconds else None),
            ("peak_vram", "not_captured"),
            ("vision_export_manifest", rel(export_manifest) if export_manifest.exists() else "pending"),
        ],
    )

    return {
        "id": run["id"],
        "run_id": run.get("run_id"),
        "train_dir": rel(train_dir),
        "train_done": bool(train_metrics),
        "lp_done": bool(lp_metrics),
        "transfer_done": bool(transfer),
        "visual_done": bool(visual),
        "counterfactual_done": bool(cf),
        "ab_swap_done": bool(ab_swap) or ab_status == "not_applicable_no_ab_rows",
        "ab_swap_status": ab_status,
        "paraphrase_done": bool(paraphrase),
        "audit_done": bool(audits),
        "export_done": export_manifest.exists(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=ROOT / "outputs/next_stage_manifests/config_manifest.json", type=Path)
    parser.add_argument("--lp-manifest", default=ROOT / "outputs/next_stage_manifests/lp_config_manifest.json", type=Path)
    parser.add_argument("--run-id", help="Internal id, e.g. storymix_qa8. Omit to package all runs.")
    parser.add_argument("--output", default=ROOT / "outputs/final_tables/next_stage_run_package_status.json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest) or {"configs": []}
    lp_manifest = read_json(args.lp_manifest) or {"configs": []}
    lp_lookup = {str(row["id"]): row for row in lp_manifest.get("configs", [])}
    rows = []
    for run in manifest.get("configs", []):
        if args.run_id and str(run.get("id")) != args.run_id:
            continue
        rows.append(package_run(run, lp_lookup))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"runs": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"packaged": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
