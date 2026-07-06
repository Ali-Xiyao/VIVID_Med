"""
Audit whether local schema sources contain report text fields.

This is a lightweight EMNLP P1 diagnostic: if local data only contains labels
and paths, report-to-schema extraction cannot be completed from this workspace
without adding a report-text dataset such as MIMIC-CXR reports.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


TEXT_FIELD_HINTS = (
    "report",
    "impression",
    "history",
    "indication",
    "comparison",
    "caption",
    "text",
    "note",
    "section",
)

STRUCTURED_SCHEMA_KEYS = {"findings", "answerability", "uncertainty", "provenance"}


def is_text_like_field(name: str) -> bool:
    lower = name.lower()
    if lower in STRUCTURED_SCHEMA_KEYS:
        return False
    return any(hint in lower for hint in TEXT_FIELD_HINTS)


def audit_csv(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        first_rows = []
        for _, row in zip(range(3), reader):
            first_rows.append(row)

    candidate_fields = [name for name in fieldnames if is_text_like_field(name)]
    non_empty_candidates = []
    for name in candidate_fields:
        values = [str(row.get(name, "") or "").strip() for row in first_rows]
        if any(values):
            non_empty_candidates.append(name)

    return {
        "path": str(path),
        "format": "csv",
        "field_count": len(fieldnames),
        "fields": fieldnames,
        "candidate_text_fields": candidate_fields,
        "non_empty_candidate_text_fields_in_first_rows": non_empty_candidates,
        "first_row_keys": list(first_rows[0].keys()) if first_rows else fieldnames,
    }


def audit_jsonl(path: Path) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for _, line in zip(range(3), f):
            line = line.strip()
            if line:
                records.append(json.loads(line))

    top_level_keys = sorted({key for record in records for key in record.keys()})
    extension_keys = sorted({
        key
        for record in records
        for key in (record.get("extensions", {}) or {}).keys()
    })
    candidate_top = [name for name in top_level_keys if is_text_like_field(name)]
    candidate_ext = [name for name in extension_keys if is_text_like_field(name)]

    return {
        "path": str(path),
        "format": "jsonl",
        "top_level_keys": top_level_keys,
        "extension_keys": extension_keys,
        "candidate_text_fields": {
            "top_level": candidate_top,
            "extensions": candidate_ext,
        },
    }


def iter_default_sources(root: Path) -> Iterable[Path]:
    candidates = [
        root / "data/dataset/CheXpert-v1.0-small/train.csv",
        root / "data/dataset/CheXpert-v1.0-small/valid.csv",
        root / "data/dataset/processed/chexpert_sampled_30k.csv",
        root / "data/dataset/processed/chexpert_ums_train.jsonl",
        root / "data/dataset/processed/chexpert_ums_val.jsonl",
        root / "data/dataset/processed/nih_external_test.csv",
        root / "data/dataset/processed/nih_external_test_ums.jsonl",
    ]
    return [path for path in candidates if path.exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local schema source fields")
    parser.add_argument("--output", type=str, default="outputs/schema_source_field_audit.json")
    args = parser.parse_args()

    root = Path.cwd()
    sources = list(iter_default_sources(root))
    audited = []
    for path in sources:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            audited.append(audit_csv(path))
        elif suffix == ".jsonl":
            audited.append(audit_jsonl(path))

    has_report_text = False
    for item in audited:
        if item["format"] == "csv":
            if item.get("non_empty_candidate_text_fields_in_first_rows"):
                has_report_text = True
        elif item["format"] == "jsonl":
            fields = item.get("candidate_text_fields", {})
            if fields.get("top_level") or fields.get("extensions"):
                has_report_text = True

    result = {
        "sources_checked": [str(path) for path in sources],
        "has_local_report_text_field": has_report_text,
        "conclusion": (
            "Local sources expose report-like text fields."
            if has_report_text
            else "Local sources checked here expose labels, paths, metadata, and UMS schemas, but no raw report text field; report-to-schema extraction needs an additional report-text source."
        ),
        "audits": audited,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: result[k] for k in ("has_local_report_text_field", "conclusion")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
