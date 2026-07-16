"""Consolidate VIVID-Med revision metrics into final tables.

This script is intentionally read-only with respect to experiment artifacts. It
collects completed metrics, records missing artifacts, and writes paper-facing
tables under outputs/final_tables/.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "final_tables"


@dataclass(frozen=True)
class MethodSpec:
    method: str
    data: str
    supervision: str
    llm: str
    module: str
    output_dir: str
    config_path: str | None
    claim_supported: str
    claim_weakened: str
    interpretation: str


METHODS = [
    MethodSpec(
        method="Data-matched BCE ViT-B",
        data="CheXpert 30k",
        supervision="labels",
        llm="no",
        module="none",
        output_dir="baseline_vit_full14",
        config_path=None,
        claim_supported="data-matched baseline",
        claim_weakened="-",
        interpretation="ordinary supervised ViT baseline",
    ),
    MethodSpec(
        method="Frozen-LM UMS / no-SPD",
        data="CheXpert 30k",
        supervision="UMS JSON",
        llm="pretrained frozen",
        module="no-SPD",
        output_dir="lp_A_ums_12label",
        config_path="configs/lp_A_ums_12label.yaml",
        claim_supported="frozen-LM in-domain; UMS/schema",
        claim_weakened="LLM external dominance",
        interpretation="strongest current CheXpert controlled result",
    ),
    MethodSpec(
        method="no-LM UMS state classifier",
        data="CheXpert 30k",
        supervision="UMS state",
        llm="no",
        module="classifier",
        output_dir="lp_ums_classifier_no_llm_12label_full",
        config_path="configs/lp_ums_classifier_no_llm_12label.yaml",
        claim_supported="UMS/schema contribution",
        claim_weakened="LLM dominant-source claim",
        interpretation="schema supervision remains strong without LM",
    ),
    MethodSpec(
        method="Frozen-LM UMS + SPD default",
        data="CheXpert 30k",
        supervision="UMS JSON",
        llm="pretrained frozen",
        module="SPD",
        output_dir="lp_A_ums_spd_12label",
        config_path="configs/lp_A_ums_spd_12label.yaml",
        claim_supported="weak external signal",
        claim_weakened="SPD stable-gain claim",
        interpretation="historical baseline / negative sensitivity",
    ),
    MethodSpec(
        method="Frozen-LM UMS + SPD G=2",
        data="CheXpert 30k",
        supervision="UMS JSON",
        llm="pretrained frozen",
        module="SPD G=2",
        output_dir="lp_spd_g2_12label",
        config_path="configs/lp_spd_g2_12label.yaml",
        claim_supported="SPD sensitivity",
        claim_weakened="SPD stable-gain claim",
        interpretation="historical sensitivity; not a new SPD variant",
    ),
    MethodSpec(
        method="Frozen-LM free-text target",
        data="CheXpert 30k",
        supervision="free text",
        llm="pretrained frozen",
        module="no-SPD",
        output_dir="lp_A_freetext_12label",
        config_path="configs/lp_A_freetext_12label.yaml",
        claim_supported="UMS stronger than free text",
        claim_weakened="free-text target sufficiency",
        interpretation="text-target baseline",
    ),
    MethodSpec(
        method="Random-mask proxy",
        data="CheXpert 30k",
        supervision="UMS JSON with random mask",
        llm="pretrained frozen",
        module="random mask",
        output_dir="lp_random_mask_12label",
        config_path="configs/lp_random_mask_12label.yaml",
        claim_supported="not explained by random mask",
        claim_weakened="mask-only explanation",
        interpretation="mask proxy control",
    ),
    MethodSpec(
        method="BiomedCLIP baseline",
        data="CheXpert 30k",
        supervision="external vision-language pretraining",
        llm="no training-time LM in this run",
        module="linear probe",
        output_dir="lp_biomedclip_baseline_seed0",
        config_path="configs/lp_biomedclip_baseline.yaml",
        claim_supported="external pretrained baseline",
        claim_weakened="-",
        interpretation="comparison baseline",
    ),
    MethodSpec(
        method="Frozen-LM UMS + answerability mask",
        data="CheXpert 30k",
        supervision="UMS JSON answerable fields",
        llm="pretrained frozen",
        module="answerability mask",
        output_dir="lp_ums_ansmask_12label",
        config_path="configs/lp_ums_ansmask_12label.yaml",
        claim_supported="missingness-faithful objective",
        claim_weakened="AUC-only answerability claim",
        interpretation="answerability semantics baseline",
    ),
    MethodSpec(
        method="Frozen-LM UMS + null-as-negative",
        data="CheXpert 30k",
        supervision="UMS JSON null-as-negative",
        llm="pretrained frozen",
        module="dense null-negative objective",
        output_dir="lp_ums_null_as_negative_12label",
        config_path="configs/lp_ums_null_as_negative_12label.yaml",
        claim_supported="dense classification objective",
        claim_weakened="null means clinically absent",
        interpretation="classification-oriented baseline",
    ),
    MethodSpec(
        method="Random-LM same-architecture UMS",
        data="CheXpert 30k",
        supervision="UMS JSON",
        llm="random frozen",
        module="no-SPD",
        output_dir="lp_ums_random_lm_12label",
        config_path="configs/lp_ums_random_lm_12label.yaml",
        claim_supported="pretrained LM stronger than random decoder",
        claim_weakened="architecture-only explanation",
        interpretation="random-LM control",
    ),
]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_value(payload: dict[str, Any] | None, key: str) -> float | None:
    if payload is None:
        return None
    metrics = payload.get("metrics", payload)
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def find_checkpoint(run_dir: Path) -> str:
    priorities = ["best.pt", "final.pt"]
    for name in priorities:
        candidate = run_dir / name
        if candidate.exists():
            return rel(candidate)
    step_checkpoints = sorted(
        run_dir.glob("step_*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if step_checkpoints:
        return rel(step_checkpoints[0])
    return ""


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = [row.get(column, "").replace("\n", " ") for column in columns]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body]) + "\n"


def collect_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []

    for spec in METHODS:
        run_dir = ROOT / "outputs" / spec.output_dir
        metrics_path = run_dir / "metrics_final.json"
        nih_path = run_dir / "nih_crossdomain.json"
        config_path = ROOT / spec.config_path if spec.config_path else None

        metrics_payload = read_json(metrics_path)
        nih_payload = read_json(nih_path)
        checkpoint = find_checkpoint(run_dir) if run_dir.exists() else ""

        if not run_dir.exists():
            missing.append(
                {
                    "method": spec.method,
                    "artifact": rel(run_dir),
                    "severity": "required",
                    "reason": "run directory missing",
                }
            )
        if metrics_payload is None:
            missing.append(
                {
                    "method": spec.method,
                    "artifact": rel(metrics_path),
                    "severity": "required",
                    "reason": "P0 input metrics_final.json missing",
                }
            )
        if nih_payload is None:
            missing.append(
                {
                    "method": spec.method,
                    "artifact": rel(nih_path),
                    "severity": "expected-if-available",
                    "reason": "NIH/cross-domain metric unavailable for this run",
                }
            )
        if not checkpoint:
            missing.append(
                {
                    "method": spec.method,
                    "artifact": rel(run_dir / "{best.pt,final.pt,step_*.pt}"),
                    "severity": "required",
                    "reason": "no checkpoint found",
                }
            )
        if config_path is None or not config_path.exists():
            missing.append(
                {
                    "method": spec.method,
                    "artifact": spec.config_path or "configs/<unknown baseline config>",
                    "severity": "provenance",
                    "reason": "config source not found or not identified",
                }
            )

        rows.append(
            {
                "Method": spec.method,
                "Data": spec.data,
                "Supervision": spec.supervision,
                "LLM": spec.llm,
                "Module": spec.module,
                "Metrics path": rel(metrics_path) if metrics_path.exists() else "",
                "Config path": spec.config_path or "",
                "Checkpoint path": checkpoint,
                "CheXpert AUC": fmt_float(metric_value(metrics_payload, "macro_auc")),
                "CheXpert F1": fmt_float(metric_value(metrics_payload, "macro_f1")),
                "NIH AUC": fmt_float(metric_value(nih_payload, "macro_auc")),
                "NIH F1": fmt_float(metric_value(nih_payload, "macro_f1")),
                "Claim supported": spec.claim_supported,
                "Claim weakened": spec.claim_weakened,
                "Interpretation": spec.interpretation,
            }
        )

    return rows, missing


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_claim_matrix(path: Path, rows: list[dict[str, str]]) -> None:
    matrix_rows = [
        {
            "Method": row["Method"],
            "Supports": row["Claim supported"],
            "Weakens": row["Claim weakened"],
            "Evidence": (
                f"CheXpert AUC={row['CheXpert AUC'] or 'missing'}, "
                f"F1={row['CheXpert F1'] or 'missing'}; "
                f"NIH AUC={row['NIH AUC'] or 'missing'}"
            ),
            "Interpretation": row["Interpretation"],
        }
        for row in rows
    ]
    text = "# Claim Support Matrix\n\n"
    text += markdown_table(
        matrix_rows,
        ["Method", "Supports", "Weakens", "Evidence", "Interpretation"],
    )
    path.write_text(text, encoding="utf-8")


def write_missing(path: Path, missing: list[dict[str, str]]) -> None:
    text = "# Missing Artifacts\n\n"
    if missing:
        text += markdown_table(missing, ["method", "artifact", "severity", "reason"])
    else:
        text += "No missing artifacts detected for P0_RESULT_CONSOLIDATION.\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, missing = collect_rows()

    columns = [
        "Method",
        "Data",
        "Supervision",
        "LLM",
        "Module",
        "Metrics path",
        "Config path",
        "Checkpoint path",
        "CheXpert AUC",
        "CheXpert F1",
        "NIH AUC",
        "NIH F1",
        "Claim supported",
        "Claim weakened",
        "Interpretation",
    ]

    write_csv(OUTPUT_DIR / "main_controlled_results.csv", rows, columns)
    (OUTPUT_DIR / "main_controlled_results.md").write_text(
        "# Main Controlled Results\n\n" + markdown_table(rows, columns),
        encoding="utf-8",
    )
    write_claim_matrix(OUTPUT_DIR / "claim_support_matrix.md", rows)
    write_missing(OUTPUT_DIR / "missing_artifacts.md", missing)

    print(f"Wrote {len(rows)} method rows to {rel(OUTPUT_DIR)}")
    print(f"Recorded {len(missing)} missing/provenance issues")


if __name__ == "__main__":
    main()
