"""Summarize CVCP/CCSH module-combo outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs" / "final_tables"
MANIFEST = FINAL_DIR / "cvcp_ccsh_module_combo_manifest.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def write_md_table(path: Path, title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        values = [str(row.get(key, "")).replace("|", "\\|").replace("\n", " ") for key in columns]
        lines.append("| " + " | ".join(values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric(payload: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return ""
        current = current.get(key)
    return "" if current is None else current


def main() -> None:
    rows = []
    for spec in read_csv(MANIFEST):
        module_rows = []
        for module in [item for item in spec["modules"].split(";") if item]:
            metrics_path = Path(spec["output_root"]) / module.lower() / "metrics_final.json"
            metrics = read_json(metrics_path)
            module_rows.append(
                {
                    "module": module,
                    "status": "complete" if metrics else "pending",
                    "state_accuracy": metric(metrics, "final", "state_accuracy"),
                    "binary_auc": metric(metrics, "final", "binary", "auc"),
                    "binary_auprc": metric(metrics, "final", "binary", "auprc"),
                    "best_val_loss": metric(metrics, "best_val_loss"),
                    "evidence": metrics_path.as_posix(),
                }
            )
        complete = [row for row in module_rows if row["status"] == "complete"]
        aucs = [float(row["binary_auc"]) for row in complete if row["binary_auc"] not in ("", None)]
        accs = [float(row["state_accuracy"]) for row in complete if row["state_accuracy"] not in ("", None)]
        rows.append(
            {
                "combo_id": spec["combo_id"],
                "label": spec["label"],
                "backbone_run_id": spec["backbone_run_id"],
                "modules": spec["modules"],
                "status": "complete" if len(complete) == len(module_rows) else ("partial" if complete else "pending"),
                "best_binary_auc": max(aucs) if aucs else "",
                "best_state_accuracy": max(accs) if accs else "",
                "evidence_root": spec["output_root"],
                "notes": spec["notes"],
                "module_metrics": json.dumps(module_rows, ensure_ascii=False),
            }
        )
    columns = ["combo_id", "label", "backbone_run_id", "modules", "status", "best_binary_auc", "best_state_accuracy", "evidence_root", "notes", "module_metrics"]
    write_csv(FINAL_DIR / "module_combo_results.csv", rows, columns)
    write_md_table(FINAL_DIR / "module_combo_results.md", "Module Combo Results", rows, columns)
    print(f"rows={len(rows)} complete={sum(1 for row in rows if row['status'] == 'complete')}")


if __name__ == "__main__":
    main()
