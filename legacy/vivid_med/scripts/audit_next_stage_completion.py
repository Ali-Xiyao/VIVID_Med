"""Audit next-stage VIVID-Med completion against current artifacts.

The script is intentionally read-only. It turns the broad next-stage plan into
an artifact checklist so interim and final states use the same evidence rules.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs/final_tables"
AUDIT_DIR = FINAL_DIR / "next_stage_audits"
DIAG_DIR = ROOT / "outputs/qwen3vl_next_stage_diagnostics"
TRANSFER_DIR = ROOT / "outputs/qwen3vl_next_stage_transfer"
AB_SWAP_ROOT = ROOT / "outputs/instruction_data/next_stage"


G1_SCRIPTS = [
    "scripts/generate_storymix_instructions.py",
    "scripts/generate_sameq_shuf_pairs.py",
    "scripts/generate_multi_negative_shuf.py",
    "scripts/audit_instruction_leakage_v2.py",
    "scripts/build_progressive_mixture_schedule.py",
    "scripts/build_token_weight_map.py",
    "scripts/mine_hard_negatives_from_embeddings.py",
]

SUPPORT_SCRIPTS = [
    "scripts/prepare_p2_loss_mask_variants.py",
    "scripts/prepare_next_stage_configs.py",
    "scripts/prepare_next_stage_lp_configs.py",
    "scripts/package_next_stage_run_outputs.py",
    "scripts/summarize_next_stage_results.py",
    "scripts/generate_next_stage_qualitative_report.py",
    "scripts/export_qwen3vl_instruction_embeddings.py",
    "scripts/build_mined_shuf_instructions.py",
    "scripts/score_qwen3vl_hard_negative_loss.py",
    "scripts/build_selfhard_shuf_instructions.py",
]

GLOBAL_ARTIFACTS = [
    ("plan", "target plan", "vivid_med_next_stage_comprehensive_experiment_plan.md"),
    ("plan", "execution plan", "task_plan.md"),
    ("plan", "findings log", "findings.md"),
    ("plan", "progress log", "progress.md"),
    ("plan", "requirement ledger", "docs/next_stage_requirement_ledger.md"),
    ("manifest", "training config manifest", "outputs/next_stage_manifests/config_manifest.json"),
    ("manifest", "LP config manifest", "outputs/next_stage_manifests/lp_config_manifest.json"),
    ("extension", "external/model availability boundary", "docs/next_stage_external_model_availability.md"),
    ("extension", "qualitative casebook", "outputs/final_tables/next_stage_qualitative_cases.md"),
]

FINAL_TABLES = [
    "outputs/final_tables/next_stage_training_results.csv",
    "outputs/final_tables/next_stage_lp_transfer_results.csv",
    "outputs/final_tables/next_stage_visual_dependence.csv",
    "outputs/final_tables/next_stage_counterfactual.csv",
    "outputs/final_tables/next_stage_ab_swap_counterfactual.csv",
    "outputs/final_tables/next_stage_paraphrase.csv",
    "outputs/final_tables/next_stage_instruction_audit.csv",
    "outputs/final_tables/next_stage_calibration_auprc.csv",
    "outputs/final_tables/next_stage_decision_summary.csv",
]

RUN_REQUIRED_BASE = [
    "config_snapshot.json",
    "metrics_final.json",
    "progress.json",
    "training_log.txt",
    "runtime_summary.json",
]

RUN_REQUIRED_POSTPROCESS = [
    "vision_export_manifest.json",
    "lp_results.md",
    "visual_dependence_results.md",
    "counterfactual_results.md",
    "ab_swap_results.md",
    "paraphrase_results.md",
    "instruction_audit.md",
    "cost_table.md",
]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resolve(pathish: str | Path | None) -> Path:
    raw = Path(str(pathish or ""))
    if raw.is_absolute():
        return raw
    return ROOT / raw


def exists(pathish: str | Path | None) -> bool:
    return resolve(pathish).exists()


def add(rows: list[dict[str, str]], group: str, item: str, status: str, evidence: str, note: str = "") -> None:
    rows.append(
        {
            "group": group,
            "item": item,
            "status": status,
            "evidence": evidence,
            "note": note,
        }
    )


def status_for(pathish: str | Path | None, pending_if: bool = False) -> str:
    if exists(pathish):
        return "completed"
    return "pending" if pending_if else "missing"


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


def ab_swap_diagnostic_status(run: dict[str, Any], train_done: bool) -> tuple[str, Path, str]:
    run_id = str(run.get("id"))
    diag_path = DIAG_DIR / f"{run_id}_ab_swap_counterfactual_diagnostics.json"
    if diag_path.exists():
        return "completed", diag_path, ""
    if not train_done:
        return "pending", diag_path, "waiting for training completion"
    input_path = ab_swap_input_path(run)
    input_rows = jsonl_row_count(input_path)
    if input_rows == 0 and input_path is not None:
        return "completed", input_path, "not_applicable_no_ab_rows"
    if input_path is not None:
        return "missing", diag_path, f"ab-swap input rows={input_rows}"
    return "missing", diag_path, "ab-swap input jsonl missing"


def load_manifests() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    config_manifest = read_json(ROOT / "outputs/next_stage_manifests/config_manifest.json") or {"configs": []}
    lp_manifest = read_json(ROOT / "outputs/next_stage_manifests/lp_config_manifest.json") or {"configs": []}
    lp_lookup = {str(row.get("id")): row for row in lp_manifest.get("configs", [])}
    return list(config_manifest.get("configs", [])), lp_lookup


def output_dir_for(run: dict[str, Any]) -> Path:
    config_path = resolve(str(run.get("config", "")))
    config = read_yaml(config_path) or {}
    output_dir = ((config.get("training") or {}).get("output_dir") or f"outputs/qwen3vl_instruction/next_stage/{run.get('id')}")
    return resolve(str(output_dir).replace("./", ""))


def audit_for_data_path(data_path: str) -> Path:
    return AUDIT_DIR / f"{Path(data_path).stem}_leakage_summary.csv"


def metrics_step_exists(out_dir: Path) -> bool:
    return any(out_dir.glob("metrics_step_*.json"))


def audit_global(rows: list[dict[str, str]]) -> None:
    for group, item, path in GLOBAL_ARTIFACTS:
        add(rows, group, item, status_for(path), path)
    for path in G1_SCRIPTS:
        add(rows, "G1 scripts", Path(path).name, status_for(path), path)
    for path in SUPPORT_SCRIPTS:
        add(rows, "support scripts", Path(path).name, status_for(path), path)
    for path in FINAL_TABLES:
        add(rows, "final tables", Path(path).name, status_for(path, pending_if=True), path)


def audit_data_and_configs(rows: list[dict[str, str]], runs: list[dict[str, Any]], lp_lookup: dict[str, dict[str, Any]]) -> None:
    for run in runs:
        run_id = str(run.get("id"))
        config_path = str(run.get("config", ""))
        add(rows, "run config", run_id, status_for(config_path), rel(resolve(config_path)))
        for split in ["train", "val"]:
            data_path = str(run.get(split, ""))
            add(rows, f"{split} data", run_id, status_for(data_path), data_path)
            audit_path = audit_for_data_path(data_path)
            add(rows, f"{split} leakage audit", run_id, status_for(audit_path, pending_if=not exists(data_path)), rel(audit_path))
        lp_spec = lp_lookup.get(run_id)
        if lp_spec:
            lp_config = str(lp_spec.get("config", ""))
            add(rows, "LP config", run_id, status_for(lp_config), rel(resolve(lp_config)))
        else:
            add(rows, "LP config", run_id, "missing", "outputs/next_stage_manifests/lp_config_manifest.json", "run absent from LP manifest")


def audit_run_outputs(rows: list[dict[str, str]], runs: list[dict[str, Any]], lp_lookup: dict[str, dict[str, Any]]) -> None:
    for run in runs:
        run_id = str(run.get("id"))
        out_dir = output_dir_for(run)
        train_done = (out_dir / "metrics_final.json").exists()
        for filename in RUN_REQUIRED_BASE:
            add(rows, "training package", f"{run_id}/{filename}", status_for(out_dir / filename, pending_if=not train_done), rel(out_dir / filename))
        add(
            rows,
            "training package",
            f"{run_id}/metrics_step_*.json",
            "completed" if metrics_step_exists(out_dir) else ("pending" if not train_done else "missing"),
            rel(out_dir / "metrics_step_*.json"),
        )
        for filename in RUN_REQUIRED_POSTPROCESS:
            add(rows, "postprocess package", f"{run_id}/{filename}", status_for(out_dir / filename, pending_if=not train_done), rel(out_dir / filename))

        lp_spec = lp_lookup.get(run_id, {})
        lp_dir = resolve(str(lp_spec.get("output_dir", f"outputs/qwen3vl_next_stage_lp/{run_id}")).replace("./", ""))
        add(rows, "LP metrics", run_id, status_for(lp_dir / "metrics_final.json", pending_if=not train_done), rel(lp_dir / "metrics_final.json"))
        add(
            rows,
            "NIH transfer",
            run_id,
            status_for(TRANSFER_DIR / f"{run_id}_nih_1k" / "transfer_metrics.json", pending_if=not train_done),
            rel(TRANSFER_DIR / f"{run_id}_nih_1k" / "transfer_metrics.json"),
        )
        for name, path in [
            ("visual dependence", DIAG_DIR / f"{run_id}_visual_dependence.json"),
            ("counterfactual", DIAG_DIR / f"{run_id}_counterfactual_diagnostics.json"),
            ("paraphrase", DIAG_DIR / f"{run_id}_paraphrase_robustness.json"),
        ]:
            add(rows, name, run_id, status_for(path, pending_if=not train_done), rel(path))
        ab_status, ab_evidence, ab_note = ab_swap_diagnostic_status(run, train_done)
        add(rows, "A/B-swap counterfactual", run_id, ab_status, rel(ab_evidence), ab_note)


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    lines = ["# Next-Stage Completion Audit", ""]
    lines.append("| status | count |")
    lines.append("| --- | --- |")
    for status in sorted(counts):
        lines.append(f"| {status} | {counts[status]} |")
    lines.extend(["", "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"])
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, default=FINAL_DIR / "next_stage_completion_audit.csv")
    parser.add_argument("--output-md", type=Path, default=FINAL_DIR / "next_stage_completion_audit.md")
    parser.add_argument("--fail-on-open", action="store_true", help="Exit nonzero when missing/pending rows remain.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs, lp_lookup = load_manifests()
    rows: list[dict[str, str]] = []
    audit_global(rows)
    audit_data_and_configs(rows, runs, lp_lookup)
    audit_run_outputs(rows, runs, lp_lookup)
    columns = ["group", "item", "status", "evidence", "note"]
    write_csv(args.output_csv, rows, columns)
    write_md(args.output_md, rows, columns)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(json.dumps({"rows": len(rows), "counts": counts, "output_csv": rel(args.output_csv), "output_md": rel(args.output_md)}, indent=2))
    if args.fail_on_open and any(row["status"] in {"missing", "pending"} for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
