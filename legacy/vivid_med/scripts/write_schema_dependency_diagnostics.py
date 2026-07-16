"""Convert schema robustness diagnostics into paper-ready tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs" / "final_tables"


SOURCES = [
    {
        "path": ROOT / "outputs" / "schema_key_robustness_A_ums_12label_128.json",
        "source": "schema_key_robustness",
        "variants": [
            "reversed_order",
            "shuffled_order",
            "clinical_key_shift",
            "generic_keys",
        ],
    },
    {
        "path": ROOT / "outputs" / "field_paraphrase_robustness_A_ums_12label_128.json",
        "source": "field_paraphrase_robustness",
        "variants": ["clinical_paraphrase", "lay_paraphrase"],
    },
]


INTERPRETATIONS = {
    "reversed_order": "order-sensitive; fixed serialization order is part of the learned interface",
    "shuffled_order": "order-sensitive; random field order substantially hurts NLL",
    "clinical_key_shift": "key-sensitive; clinical synonyms are not interchangeable without adaptation",
    "generic_keys": "field-name-sensitive; generic keys break the learned clinical schema",
    "clinical_paraphrase": "not clinical-paraphrase robust under current fixed schema",
    "lay_paraphrase": "not lay-paraphrase robust under current fixed schema",
}


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def build_rows() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    case_rows: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []

    for source in SOURCES:
        path = source["path"]
        payload = read_json(path)
        if payload is None:
            missing.append(
                {
                    "artifact": rel(path),
                    "field": "json",
                    "reason": "missing or unparsable diagnostic file",
                }
            )
            continue
        summary = payload.get("summary", {})
        for variant in source["variants"]:
            stats = summary.get(variant)
            if not isinstance(stats, dict):
                missing.append(
                    {
                        "artifact": rel(path),
                        "field": variant,
                        "reason": "variant summary missing",
                    }
                )
                continue
            rows.append(
                {
                    "source": source["source"],
                    "variant": variant,
                    "n": fmt(stats.get("n")),
                    "original_nll": fmt(stats.get("original_nll_mean")),
                    "variant_nll": fmt(stats.get("variant_nll_mean")),
                    "margin": fmt(stats.get("mean_margin")),
                    "median_margin": fmt(stats.get("median_margin")),
                    "original_better_rate": fmt(stats.get("original_better_rate")),
                    "relative_delta_vs_original": fmt(stats.get("relative_delta_vs_original")),
                    "interpretation": INTERPRETATIONS.get(variant, "fixed-schema dependency"),
                    "artifact": rel(path),
                }
            )

        for item in payload.get("rows", [])[:24]:
            if not isinstance(item, dict):
                continue
            case_rows.append(
                {
                    "source": source["source"],
                    "variant": str(item.get("variant", "")),
                    "sample_id": str(item.get("sample_id", "")),
                    "original_path": str(item.get("original_path", "")),
                    "original_nll": fmt(item.get("original_nll")),
                    "variant_nll": fmt(item.get("variant_nll")),
                    "margin": fmt(item.get("margin")),
                }
            )
    return rows, case_rows, missing


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = [row.get(column, "").replace("\n", " ") for column in columns]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body]) + "\n"


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_case_study(case_rows: list[dict[str, str]], diagnostics: list[dict[str, str]]) -> None:
    columns = [
        "source",
        "variant",
        "sample_id",
        "original_path",
        "original_nll",
        "variant_nll",
        "margin",
    ]
    strongest = sorted(
        case_rows,
        key=lambda row: float(row["margin"]) if row.get("margin") else -1.0,
        reverse=True,
    )[:12]
    text = "# Schema Dependency Case Study\n\n"
    text += (
        "The current objective is schema-supervised rather than schema-agnostic. "
        "The model was trained against a fixed clinical serialization, and the "
        "diagnostics below show that reordered, renamed, or paraphrased schemas "
        "increase teacher-forcing NLL.\n\n"
    )
    text += "## Paper-Ready Limitation\n\n"
    text += (
        "Our objective should be interpreted as a fixed-schema interface for CXR "
        "representation learning. It should not be described as arbitrary "
        "natural-language schema understanding or paraphrase-robust schema "
        "following.\n\n"
    )
    text += "## Strongest Example Margins\n\n"
    text += markdown_table(strongest, columns)
    text += "\n## Compact Diagnostic Summary\n\n"
    text += markdown_table(
        diagnostics,
        [
            "variant",
            "original_nll",
            "variant_nll",
            "margin",
            "original_better_rate",
            "interpretation",
        ],
    )
    (FINAL_DIR / "schema_dependency_case_study.md").write_text(text, encoding="utf-8")


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    rows, case_rows, missing = build_rows()
    diagnostic_columns = [
        "source",
        "variant",
        "n",
        "original_nll",
        "variant_nll",
        "margin",
        "median_margin",
        "original_better_rate",
        "relative_delta_vs_original",
        "interpretation",
        "artifact",
    ]
    missing_columns = ["artifact", "field", "reason"]
    write_csv(FINAL_DIR / "schema_dependency_diagnostics.csv", rows, diagnostic_columns)
    (FINAL_DIR / "schema_dependency_diagnostics.md").write_text(
        "# Schema Dependency Diagnostics\n\n"
        + markdown_table(rows, diagnostic_columns),
        encoding="utf-8",
    )
    write_case_study(case_rows, rows)
    (FINAL_DIR / "schema_dependency_missing_artifacts.md").write_text(
        "# Schema Dependency Missing Artifacts\n\n"
        + (markdown_table(missing, missing_columns) if missing else "No missing artifacts.\n"),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} schema dependency diagnostic rows")
    print(f"Wrote {len(case_rows)} case-study rows")
    print(f"Recorded {len(missing)} missing/boundary issues")


if __name__ == "__main__":
    main()
