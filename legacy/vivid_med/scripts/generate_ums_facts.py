"""Convert UMS JSONL splits into fact rows for scale experiments.

These facts are structured-label derived, not GLM report-derived. They are used
for scale/control experiments where a local 10k split is available but report
fact extraction has not been run for every image.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if max_samples is not None and len(rows) >= max_samples:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def resolve_image_path(sample: dict[str, Any], data_root: Path) -> str:
    extensions = sample.get("extensions") or {}
    raw = extensions.get("original_path") or sample.get("image_path") or sample.get("path") or ""
    path = Path(str(raw))
    if path.is_absolute():
        return str(path)
    return str(data_root / path)


def fact_from_finding(finding: str, payload: dict[str, Any], uncertainty: Any) -> dict[str, Any] | None:
    state = payload.get("state")
    if state not in {"present", "absent", "uncertain"}:
        return None
    certainty = "uncertain" if state == "uncertain" or uncertainty is True else "definite"
    visual_dependency = "high" if state in {"present", "uncertain"} else "medium"
    if finding in {"Support Devices", "Fracture", "Pneumothorax", "Pleural Effusion"}:
        visual_dependency = "high"
    return {
        "certainty": certainty,
        "evidence_span": "",
        "finding": finding,
        "location": None,
        "severity": None,
        "state": state,
        "visual_dependency": visual_dependency,
    }


def convert_sample(sample: dict[str, Any], data_root: Path, source: str) -> dict[str, Any] | None:
    findings = sample.get("findings") or {}
    uncertainty = sample.get("uncertainty") or {}
    facts = []
    for finding in sorted(findings):
        fact = fact_from_finding(str(finding), findings[finding] or {}, uncertainty.get(finding))
        if fact is not None:
            facts.append(fact)
    if not facts:
        return None
    extensions = sample.get("extensions") or {}
    sample_id = extensions.get("sample_id") or sample.get("sample_id") or len(facts)
    return {
        "facts": facts,
        "image_path": resolve_image_path(sample, data_root),
        "model": "ums_structured_labels",
        "report": "",
        "sample_id": f"chexpert_ums_{sample_id}",
        "source": source,
        "unmentioned_findings": [str(key) for key, value in findings.items() if (value or {}).get("state") is None],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--data-root", default="H:/Xiyao_Wang/000_Public Dataset", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--source", default="ums_structured_fact_extraction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = read_jsonl(args.input, max_samples=args.max_samples)
    rows = []
    state_counter: Counter[str] = Counter()
    finding_counter: Counter[str] = Counter()
    for sample in samples:
        row = convert_sample(sample, args.data_root, args.source)
        if row is None:
            continue
        rows.append(row)
        for fact in row["facts"]:
            state_counter[str(fact["state"])] += 1
            finding_counter[str(fact["finding"])] += 1
    write_jsonl(args.output, rows)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "source": args.source,
        "samples_read": len(samples),
        "fact_rows": len(rows),
        "facts": sum(len(row["facts"]) for row in rows),
        "state_counts": dict(state_counter),
        "finding_counts": dict(finding_counter),
        "boundary": "structured UMS label facts; no report text or GLM evidence spans",
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
