"""Write the no-LM schema-comparator design table.

This script is intentionally dependency-light: it does not import torch and
does not load checkpoints. It documents what can be derived from the existing
4-state no-LM classifier before any formal S2/S3 schema sweep is launched.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs" / "final_tables"


COLUMNS = [
    "schema_level",
    "frozen_lm_target",
    "no_lm_current_path",
    "derived_no_lm_signal",
    "supported_without_retraining",
    "current_metric_export",
    "fair_comparison_scope",
    "required_next_step",
    "claim_boundary",
]


ROWS = [
    {
        "schema_level": "S1 state_only",
        "frozen_lm_target": "JSON finding state only: null/absent/uncertain/present.",
        "no_lm_current_path": "scripts/train_ums_classifier.py trains per-label 4-state logits for null/absent/uncertain/present.",
        "derived_no_lm_signal": "Direct 4-state prediction; binary present probability uses p_present.",
        "supported_without_retraining": "yes",
        "current_metric_export": "Existing metrics_final.json exports present-vs-nonpresent metrics plus state_accuracy_all_fields and state_accuracy_answerable_fields.",
        "fair_comparison_scope": "Can compare as state-only UMS/schema baseline if frozen-LM S1 is run under matched split/config.",
        "required_next_step": "No new head needed; only run matched frozen-LM S1 if a formal S1 table is needed.",
        "claim_boundary": "This is the current no-LM UMS baseline, not evidence for richer answerability/uncertainty serialization.",
    },
    {
        "schema_level": "S2 state_answerability",
        "frozen_lm_target": "JSON finding state plus explicit answerable boolean.",
        "no_lm_current_path": "scripts/train_ums_classifier.py now supports an optional answerability auxiliary head and BCE loss.",
        "derived_no_lm_signal": "Legacy diagnostic remains available as answerable_target = state != null; p_answerable = 1 - p_null.",
        "supported_without_retraining": "explicit_head_requires_training; debug_ready",
        "current_metric_export": "Debug/future formal explicit-head runs export answerability AUROC/F1/accuracy fields in metrics_final.json.",
        "fair_comparison_scope": "Can become a matched S2 comparator after formal no-LM S2 source and LP runs finish.",
        "required_next_step": "Run formal no-LM S2 source, then matched LP, only after a separate execution-before record.",
        "claim_boundary": "Current explicit-head evidence is debug/runtime support, not formal S2 performance.",
    },
    {
        "schema_level": "S3 state_uncertainty",
        "frozen_lm_target": "JSON finding state plus explicit answerable and uncertain fields.",
        "no_lm_current_path": "scripts/train_ums_classifier.py now supports an optional uncertainty auxiliary head and BCE loss.",
        "derived_no_lm_signal": "Legacy diagnostic remains available as uncertain_target = state == uncertain; p_uncertain = softmax probability for the uncertain class.",
        "supported_without_retraining": "explicit_head_requires_training; debug_ready",
        "current_metric_export": "Debug/future formal explicit-head runs export uncertainty AUROC/F1/accuracy fields in metrics_final.json.",
        "fair_comparison_scope": "Can become a matched S3 comparator after formal no-LM S3 source and LP runs finish.",
        "required_next_step": "Run formal no-LM S3 source, then matched LP, only after a separate execution-before record.",
        "claim_boundary": "Current explicit-head evidence is debug/runtime support, not formal S3 performance.",
    },
    {
        "schema_level": "S2/S3 multi_head no-LM",
        "frozen_lm_target": "Explicit schema fields in serialized language target.",
        "no_lm_current_path": "Implemented as optional answerability and uncertainty heads in scripts/train_ums_classifier.py.",
        "derived_no_lm_signal": "Not a derived-only path when auxiliary heads are enabled; uses explicit binary targets/losses.",
        "supported_without_retraining": "no; source training required",
        "current_metric_export": "debug metrics available for S2/S3; formal metrics not yet run",
        "fair_comparison_scope": "Would be the closest architectural analogue to frozen-LM richer schema supervision, but it is a new model variant.",
        "required_next_step": "Treat as the formal no-LM comparator path, not as a paper result until full source/LP runs complete.",
        "claim_boundary": "Debug success proves runtime support only; do not report as completed schema-complexity sweep.",
    },
]


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row.get(column, "") for column in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body]) + "\n"


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(FINAL_DIR / "no_lm_schema_comparator_design.csv", ROWS, COLUMNS)

    text = "# no-LM Schema Comparator Design\n\n"
    text += (
        "**Superseded status (2026-06-27):** This is an earlier design artifact. "
        "The formal fixed-split no-LM and frozen-LM S1/S2/S3 source+LP schema matrix has since completed; "
        "use `outputs/final_tables/schema_complexity_diagnostic_summary.md` for current evidence and treat the recommendations below as historical execution guidance.\n\n"
    )
    text += (
        "This table documents what the existing no-LM 4-state classifier can support "
        "before any formal schema-complexity comparison is launched. It is a design "
        "artifact, not a training result.\n\n"
    )
    text += markdown_table(ROWS, COLUMNS)
    text += "\n## Historical Recommendation\n\n"
    text += (
        "- Keep legacy derived diagnostics as a baseline boundary: answerability from `1 - p_null` and uncertainty from `p_uncertain`.\n"
        "- Treat new S2/S3 no-LM explicit heads as debug-ready but not formal evidence until source and LP runs finish.\n"
        "- Do not use frozen-LM-only formal schema sweeps as comparative evidence; at the time, matched no-LM source/LP rows had not yet completed.\n"
        "- The next executable step, if schema sweep remains required, should be one formal no-LM S2 or S3 source run after a separate execution-before record.\n"
        "\nCurrent replacement: the formal no-LM and frozen-LM S1/S2/S3 source+LP rows are complete under the fixed split. "
        "Do not use this historical recommendation as a current missing-row checklist.\n"
    )
    (FINAL_DIR / "no_lm_schema_comparator_design.md").write_text(text, encoding="utf-8")
    print(f"Wrote {len(ROWS)} rows to {FINAL_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
