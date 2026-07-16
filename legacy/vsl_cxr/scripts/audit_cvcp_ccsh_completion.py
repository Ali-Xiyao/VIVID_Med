"""Audit CVCP/CCSH final completion against current artifacts."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs" / "final_tables"
TARGET_DOC = ROOT / "vivid_med_cvcp_ccsh_full_next_experiment_plan.md"
CLOSURE_MARKER = "CVCP_CCSH_FINAL_EXECUTION_CLOSURE_20260706"

REQUIRED_FINAL_FILES = [
    "cvcp_training_results.csv",
    "cvcp_training_results.md",
    "cvcp_ccsh_postprocess_status.csv",
    "module_combo_results.csv",
    "module_combo_results.md",
    "model_comparison_results.csv",
    "external_eval_results.csv",
    "locked_final_comparison.csv",
    "locked_final_comparison.md",
    "cost_table.csv",
    "cost_table.md",
    "casebook.md",
    "cvcp_ccsh_requirement_ledger.csv",
    "cvcp_ccsh_readiness_audit.csv",
]

QUEUE_LOGS = [
    ROOT / "outputs" / "logs" / "cvcp_ccsh" / "training_gpu0_lane0_of2.log",
    ROOT / "outputs" / "logs" / "cvcp_ccsh" / "training_gpu1_lane1_of2.log",
    ROOT / "outputs" / "logs" / "cvcp_ccsh_postprocess" / "postprocess_gpu0_lane0_of2.log",
    ROOT / "outputs" / "logs" / "cvcp_ccsh_postprocess" / "postprocess_gpu1_lane1_of2.log",
    ROOT / "outputs" / "logs" / "cvcp_ccsh_module_combos" / "module_combo_gpu0_lane0_of2.log",
    ROOT / "outputs" / "logs" / "cvcp_ccsh_module_combos" / "module_combo_gpu1_lane1_of2.log",
]

COLUMNS = ["group", "item", "status", "evidence", "note"]


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def add(rows: list[dict[str, str]], group: str, item: str, status: str, evidence: str, note: str = "") -> None:
    rows.append({"group": group, "item": item, "status": status, "evidence": evidence, "note": note})


def audit_counts(rows: list[dict[str, str]]) -> None:
    train = read_csv(FINAL_DIR / "cvcp_training_results.csv")
    complete_train = [row for row in train if row.get("status") == "complete"]
    add(
        rows,
        "experiment_rows",
        "cvcp_training_results",
        "completed" if len(train) == 27 and len(complete_train) == 27 else "open",
        "outputs/final_tables/cvcp_training_results.csv",
        f"complete={len(complete_train)}/{len(train)}",
    )

    modules = read_csv(FINAL_DIR / "module_combo_results.csv")
    complete_modules = [row for row in modules if row.get("status") == "complete"]
    add(
        rows,
        "module_rows",
        "module_combo_results",
        "completed" if len(modules) == 18 and len(complete_modules) == 18 else "open",
        "outputs/final_tables/module_combo_results.csv",
        f"complete={len(complete_modules)}/{len(modules)}",
    )


def audit_files(rows: list[dict[str, str]]) -> None:
    for name in REQUIRED_FINAL_FILES:
        path = FINAL_DIR / name
        add(
            rows,
            "final_artifact",
            name,
            "completed" if path.exists() and path.stat().st_size > 0 else "missing",
            rel(path),
            f"bytes={path.stat().st_size}" if path.exists() else "",
        )


def audit_logs(rows: list[dict[str, str]]) -> None:
    for path in QUEUE_LOGS:
        if not path.exists():
            add(rows, "queue_log", rel(path), "missing", rel(path), "")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        add(
            rows,
            "queue_log",
            rel(path),
            "completed" if "QUEUE_DONE" in text else "open",
            rel(path),
            "contains QUEUE_DONE" if "QUEUE_DONE" in text else "QUEUE_DONE not found",
        )


def audit_gpu(rows: list[dict[str, str]]) -> None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        add(rows, "gpu", "nvidia-smi", "unknown", "nvidia-smi", str(exc))
        return
    if result.returncode != 0:
        add(rows, "gpu", "nvidia-smi", "unknown", "nvidia-smi", result.stderr.strip())
        return
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    idle = True
    for line in lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            idle = idle and parts[1] == "0" and parts[2] == "0"
    add(rows, "gpu", "two_local_3090_state", "completed" if idle else "open", "nvidia-smi", "; ".join(lines))


def audit_doc(rows: list[dict[str, str]]) -> None:
    text = TARGET_DOC.read_text(encoding="utf-8", errors="replace") if TARGET_DOC.exists() else ""
    add(
        rows,
        "source_doc",
        TARGET_DOC.name,
        "completed" if CLOSURE_MARKER in text else "missing",
        rel(TARGET_DOC),
        f"marker={CLOSURE_MARKER}",
    )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, str]]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    lines = ["# CVCP/CCSH Completion Audit", ""]
    lines.append("| status | count |")
    lines.append("| --- | ---: |")
    for status in sorted(counts):
        lines.append(f"| {status} | {counts[status]} |")
    lines.extend(["", "| " + " | ".join(COLUMNS) + " |", "| " + " | ".join("---" for _ in COLUMNS) + " |"])
    for row in rows:
        values = [str(row.get(column, "")).replace("|", "\\|").replace("\n", " ") for column in COLUMNS]
        lines.append("| " + " | ".join(values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, default=FINAL_DIR / "cvcp_ccsh_completion_audit.csv")
    parser.add_argument("--output-md", type=Path, default=FINAL_DIR / "cvcp_ccsh_completion_audit.md")
    parser.add_argument("--fail-on-open", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str]] = []
    audit_counts(rows)
    audit_files(rows)
    audit_logs(rows)
    audit_gpu(rows)
    audit_doc(rows)
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print({"rows": len(rows), "counts": counts, "output_csv": rel(args.output_csv), "output_md": rel(args.output_md)})
    if args.fail_on_open and any(row["status"] != "completed" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
