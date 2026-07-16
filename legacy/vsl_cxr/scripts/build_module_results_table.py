"""Build VSL-CXR module result tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = Path("F:/Xiyao_Wang/021_260129VIVID_vsl_cxr_outputs/qwen3vl")
FINAL_DIR = ROOT / "outputs" / "final_tables"


def repo_rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = [str(row.get(column, "")).replace("|", "\\|").replace("\n", " ") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def collect_ceq(run_root: Path) -> list[dict[str, Any]]:
    rows = []
    for run_dir in sorted(run_root.glob("vsl_cxr_ceq_*")):
        if not run_dir.is_dir() or run_dir.name.endswith("_debug"):
            continue
        metrics = read_json(run_dir / "metrics_final.json")
        config = read_json(run_dir / "config_snapshot.json").get("resolved_config") or {}
        experiment = config.get("experiment") or {}
        ceq = config.get("ceq") or {}
        model = config.get("model") or {}
        final = metrics.get("final") or {}
        binary = final.get("binary") or {}
        checkpoint = run_dir / "checkpoints" / "final.pt"
        rows.append(
            {
                "run_id": experiment.get("run_id", run_dir.name),
                "variant": metrics.get("variant", ceq.get("variant", "")),
                "backbone_checkpoint": model.get("vision_checkpoint", ""),
                "status": "completed" if metrics and checkpoint.exists() else "in_progress",
                "global_step": metrics.get("global_step", ""),
                "best_val_loss": metrics.get("best_val_loss", ""),
                "final_val_loss": metrics.get("final_val_loss", ""),
                "state_accuracy": final.get("state_accuracy", ""),
                "binary_auc": binary.get("auc", ""),
                "binary_auprc": binary.get("auprc", ""),
                "binary_f1": binary.get("f1", ""),
                "binary_ece": binary.get("ece", ""),
                "region_accuracy": final.get("region_accuracy", ""),
                "train_records": metrics.get("train_records", ""),
                "val_records": metrics.get("val_records", ""),
                "checkpoint": checkpoint.as_posix() if checkpoint.exists() else "",
                "metrics_final": (run_dir / "metrics_final.json").as_posix() if metrics else "",
                "run_dir": run_dir.as_posix(),
            }
        )
    rows.sort(key=lambda row: str(row["run_id"]))
    return rows


def collect_ccsh(run_root: Path) -> list[dict[str, Any]]:
    rows = []
    patterns = ["vsl_cxr_ccsh_*", "vsl_cxr_auch_ceq_ccsh", "vsl_cxr_auch_ccsh_*", "vsl_cxr_auch_vsl4"]
    seen: set[Path] = set()
    for pattern in patterns:
        for run_dir in sorted(run_root.glob(pattern)):
            if not run_dir.is_dir() or run_dir.name.endswith("_debug") or run_dir in seen:
                continue
            seen.add(run_dir)
            metrics = read_json(run_dir / "metrics_final.json")
            config = read_json(run_dir / "config_snapshot.json").get("resolved_config") or {}
            experiment = config.get("experiment") or {}
            ccsh = config.get("ccsh") or {}
            model = config.get("model") or {}
            final = metrics.get("final") or {}
            binary = final.get("binary") or {}
            checkpoint = run_dir / "checkpoints" / "final.pt"
            rows.append(
                {
                    "run_id": experiment.get("run_id", run_dir.name),
                    "variant": metrics.get("variant", ccsh.get("variant", "")),
                    "backbone_checkpoint": model.get("vision_checkpoint", ""),
                    "ceq_checkpoint": ccsh.get("ceq_checkpoint", ""),
                    "status": "completed" if metrics and checkpoint.exists() else "in_progress",
                    "global_step": metrics.get("global_step", ""),
                    "best_val_loss": metrics.get("best_val_loss", ""),
                    "final_val_loss": metrics.get("final_val_loss", ""),
                    "binary_auc": binary.get("auc", ""),
                    "binary_auprc": binary.get("auprc", ""),
                    "binary_f1": binary.get("f1", ""),
                    "binary_accuracy": binary.get("accuracy", ""),
                    "binary_ece": binary.get("ece", ""),
                    "train_records": metrics.get("train_records", ""),
                    "val_records": metrics.get("val_records", ""),
                    "checkpoint": checkpoint.as_posix() if checkpoint.exists() else "",
                    "metrics_final": (run_dir / "metrics_final.json").as_posix() if metrics else "",
                    "run_dir": run_dir.as_posix(),
                }
            )
    rows.sort(key=lambda row: str(row["run_id"]))
    return rows


def collect_auch(run_root: Path) -> list[dict[str, Any]]:
    rows = []
    for run_dir in sorted(run_root.glob("vsl_cxr_auch_sameq")):
        if not run_dir.is_dir() or run_dir.name.endswith("_debug"):
            continue
        metrics = read_json(run_dir / "metrics_final.json")
        config = read_json(run_dir / "config_snapshot.json").get("resolved_config") or {}
        experiment = config.get("experiment") or {}
        auch = config.get("auch") or {}
        model = config.get("model") or {}
        final = metrics.get("final") or {}
        answerability = final.get("answerability") or {}
        uncertainty = final.get("uncertainty") or {}
        checkpoint = run_dir / "checkpoints" / "final.pt"
        rows.append(
            {
                "run_id": experiment.get("run_id", run_dir.name),
                "variant": metrics.get("variant", auch.get("variant", "")),
                "backbone_checkpoint": model.get("vision_checkpoint", ""),
                "status": "completed" if metrics and checkpoint.exists() else "in_progress",
                "global_step": metrics.get("global_step", ""),
                "best_val_loss": metrics.get("best_val_loss", ""),
                "final_val_loss": metrics.get("final_val_loss", ""),
                "state_accuracy": final.get("state_accuracy", ""),
                "answerability_auc": answerability.get("auc", ""),
                "answerability_auprc": answerability.get("auprc", ""),
                "answerability_f1": answerability.get("f1", ""),
                "answerability_accuracy": answerability.get("accuracy", ""),
                "answerability_ece": answerability.get("ece", ""),
                "uncertainty_auc": uncertainty.get("auc", ""),
                "uncertainty_auprc": uncertainty.get("auprc", ""),
                "uncertainty_f1": uncertainty.get("f1", ""),
                "uncertainty_accuracy": uncertainty.get("accuracy", ""),
                "uncertainty_ece": uncertainty.get("ece", ""),
                "train_records": metrics.get("train_records", ""),
                "val_records": metrics.get("val_records", ""),
                "checkpoint": checkpoint.as_posix() if checkpoint.exists() else "",
                "metrics_final": (run_dir / "metrics_final.json").as_posix() if metrics else "",
                "run_dir": run_dir.as_posix(),
            }
        )
    rows.sort(key=lambda row: str(row["run_id"]))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-csv", type=Path, default=FINAL_DIR / "vsl_cxr_ceq_results.csv")
    parser.add_argument("--out-md", type=Path, default=FINAL_DIR / "vsl_cxr_ceq_results.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_ceq(args.run_root)
    columns = [
        "run_id",
        "variant",
        "status",
        "global_step",
        "best_val_loss",
        "final_val_loss",
        "state_accuracy",
        "binary_auc",
        "binary_auprc",
        "binary_f1",
        "binary_ece",
        "region_accuracy",
        "train_records",
        "val_records",
        "checkpoint",
        "metrics_final",
        "backbone_checkpoint",
        "run_dir",
    ]
    write_csv(args.out_csv, rows, columns)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(
        "# VSL-CXR CEQ Results\n\n"
        "Generated from v5-named CEQ patch-token runs. These rows use frozen Qwen3-VL backbones and train only the CEQ head/readout.\n\n"
        + md_table(rows, columns)
        + "\n",
        encoding="utf-8",
    )
    print(f"rows={len(rows)}")
    print(f"completed={sum(1 for row in rows if row.get('status') == 'completed')}")
    print(f"csv={repo_rel(args.out_csv)}")
    print(f"md={repo_rel(args.out_md)}")

    ccsh_rows = collect_ccsh(args.run_root)
    ccsh_columns = [
        "run_id",
        "variant",
        "status",
        "global_step",
        "best_val_loss",
        "final_val_loss",
        "binary_auc",
        "binary_auprc",
        "binary_f1",
        "binary_accuracy",
        "binary_ece",
        "train_records",
        "val_records",
        "checkpoint",
        "metrics_final",
        "backbone_checkpoint",
        "ceq_checkpoint",
        "run_dir",
    ]
    ccsh_csv = FINAL_DIR / "vsl_cxr_ccsh_results.csv"
    ccsh_md = FINAL_DIR / "vsl_cxr_ccsh_results.md"
    write_csv(ccsh_csv, ccsh_rows, ccsh_columns)
    ccsh_md.write_text(
        "# VSL-CXR CCSH/AUCH Readout Results\n\n"
        "Generated from v5-named CCSH/AUCH deployable readout runs. Rows use frozen Qwen3-VL backbones and train only the readout head stack.\n\n"
        + md_table(ccsh_rows, ccsh_columns)
        + "\n",
        encoding="utf-8",
    )
    print(f"ccsh_rows={len(ccsh_rows)}")
    print(f"ccsh_completed={sum(1 for row in ccsh_rows if row.get('status') == 'completed')}")
    print(f"ccsh_csv={repo_rel(ccsh_csv)}")
    print(f"ccsh_md={repo_rel(ccsh_md)}")

    auch_rows = collect_auch(args.run_root)
    auch_columns = [
        "run_id",
        "variant",
        "status",
        "global_step",
        "best_val_loss",
        "final_val_loss",
        "state_accuracy",
        "answerability_auc",
        "answerability_auprc",
        "answerability_f1",
        "answerability_accuracy",
        "answerability_ece",
        "uncertainty_auc",
        "uncertainty_auprc",
        "uncertainty_f1",
        "uncertainty_accuracy",
        "uncertainty_ece",
        "train_records",
        "val_records",
        "checkpoint",
        "metrics_final",
        "backbone_checkpoint",
        "run_dir",
    ]
    auch_csv = FINAL_DIR / "vsl_cxr_auch_results.csv"
    auch_md = FINAL_DIR / "vsl_cxr_auch_results.md"
    write_csv(auch_csv, auch_rows, auch_columns)
    auch_md.write_text(
        "# VSL-CXR AUCH Readout Results\n\n"
        "Generated from v5-named AUCH-only readout runs. Rows use frozen Qwen3-VL backbones and train only the AUCH head.\n\n"
        + md_table(auch_rows, auch_columns)
        + "\n",
        encoding="utf-8",
    )
    print(f"auch_rows={len(auch_rows)}")
    print(f"auch_completed={sum(1 for row in auch_rows if row.get('status') == 'completed')}")
    print(f"auch_csv={repo_rel(auch_csv)}")
    print(f"auch_md={repo_rel(auch_md)}")


if __name__ == "__main__":
    main()
