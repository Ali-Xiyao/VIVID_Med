"""Audit current code support for P1 schema complexity sweep."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs" / "final_tables"


SCHEMA_LEVELS = [
    {
        "schema_level": "S1",
        "schema_mode": "state_only",
        "target_content": "finding -> state",
        "frozen_lm_status": "supported_explicit_schema_mode",
        "no_lm_status": "supported_as_current_4_state_classifier",
        "random_lm_status": "supported_via_llm_random_init",
        "implementation_gap": "none for frozen-LM S1; no-LM remains current 4-state classifier",
    },
    {
        "schema_level": "S2",
        "schema_mode": "state_answerability",
        "target_content": "finding -> state + answerability",
        "frozen_lm_status": "minimal_serializer_supported",
        "no_lm_status": "supported_via_optional_answerability_head",
        "random_lm_status": "supported_via_same_serializer_with_llm_random_init",
        "implementation_gap": "formal no-LM S2 source/LP runs remain unrun; debug evidence is not paper performance",
    },
    {
        "schema_level": "S3",
        "schema_mode": "state_uncertainty",
        "target_content": "finding -> state + uncertainty",
        "frozen_lm_status": "minimal_serializer_supported",
        "no_lm_status": "supported_via_optional_uncertainty_head",
        "random_lm_status": "supported_via_same_serializer_with_llm_random_init",
        "implementation_gap": "formal no-LM S3 source/LP runs remain unrun; debug evidence is not paper performance",
    },
    {
        "schema_level": "S4",
        "schema_mode": "state_uncertainty_location_severity",
        "target_content": "finding -> state + uncertainty + location/severity",
        "frozen_lm_status": "not_supported_by_current_chexpert_fields",
        "no_lm_status": "not_supported",
        "random_lm_status": "not_supported",
        "implementation_gap": "requires enriched UMS fields beyond current CheXpert label-derived schema",
    },
    {
        "schema_level": "S5",
        "schema_mode": "compositional_full",
        "target_content": "finding-anatomy-state-severity-temporality",
        "frozen_lm_status": "not_supported",
        "no_lm_status": "not_supported",
        "random_lm_status": "not_supported",
        "implementation_gap": "requires new schema/data construction and model target design",
    },
]


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def has_text(path: Path, needle: str) -> bool:
    return needle in path.read_text(encoding="utf-8", errors="ignore")


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


def code_evidence() -> list[str]:
    dataset = ROOT / "data" / "chexpert_dataset.py"
    train_cxr = ROOT / "scripts" / "train_cxr.py"
    no_lm = ROOT / "scripts" / "train_ums_classifier.py"
    evidence = []
    evidence.append(
        f"`{rel(dataset)}` contains `target_format` support: {has_text(dataset, 'target_format')}"
    )
    evidence.append(
        f"`{rel(dataset)}` contains explicit `schema_mode`: {has_text(dataset, 'schema_mode')}"
    )
    evidence.append(
        f"`{rel(dataset)}` serializes `answerability`: {has_text(dataset, 'answerability')}"
    )
    evidence.append(
        f"`{rel(dataset)}` serializes `uncertainty`: {has_text(dataset, 'uncertainty')}"
    )
    evidence.append(
        f"`{rel(train_cxr)}` passes `json_null_state`: {has_text(train_cxr, 'json_null_state')}"
    )
    evidence.append(
        f"`{rel(train_cxr)}` passes `answerability_mask`: {has_text(train_cxr, 'answerability_mask')}"
    )
    evidence.append(
        f"`{rel(no_lm)}` has fixed `STATE_TO_INDEX`: {has_text(no_lm, 'STATE_TO_INDEX')}"
    )
    evidence.append(
        f"`{rel(no_lm)}` has optional schema auxiliary heads: {has_text(no_lm, 'schema_auxiliary_targets')}"
    )
    return evidence


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    columns = [
        "schema_level",
        "schema_mode",
        "target_content",
        "frozen_lm_status",
        "no_lm_status",
        "random_lm_status",
        "implementation_gap",
    ]
    write_csv(FINAL_DIR / "schema_complexity_support_matrix.csv", SCHEMA_LEVELS, columns)

    text = "# Schema Complexity Sweep Prep\n\n"
    text += "No training or model import was performed.\n\n"
    text += "## Support Matrix\n\n"
    text += markdown_table(SCHEMA_LEVELS, columns)
    text += "\n## Code Evidence\n\n"
    for item in code_evidence():
        text += f"- {item}\n"
    text += "\n## Implemented Minimal Path\n\n"
    text += (
        "1. `CheXpertUMSDataset` now accepts `schema_mode`, and "
        "`scripts/train_cxr.py` passes `data.schema_mode` through.\n"
        "2. `state_only` remains the default and preserves current JSON "
        "serialization, so S1 stays backward-compatible.\n"
        "3. Frozen-LM serializers are available for `state_answerability` and "
        "`state_uncertainty` using existing UMS `answerability` and `uncertainty` "
        "fields.\n"
        "4. no-LM S2/S3 can now use optional answerability/uncertainty heads, "
        "but current evidence is debug-only until formal source and LP runs finish.\n"
        "5. Defer S4/S5 until enriched location/severity/temporality fields exist; "
        "do not fabricate those fields from current CheXpert labels.\n"
    )
    text += "\n## Safe Direct Runs\n\n"
    text += (
        "- S1/current equivalent can use existing frozen-LM and no-LM UMS configs "
        "after runtime import is fixed.\n"
        "- S2/S3 frozen-LM configs can now be generated by setting "
        "`data.schema_mode` to `state_answerability` or `state_uncertainty`.\n"
        "- S2/S3 no-LM source configs are ready for explicit-head debug/formal runs; "
        "S4/S5 are out of scope for current label-derived UMS artifacts.\n"
    )
    (FINAL_DIR / "schema_complexity_prep.md").write_text(text, encoding="utf-8")
    print(f"Wrote schema complexity prep to {rel(FINAL_DIR)}")


if __name__ == "__main__":
    main()
