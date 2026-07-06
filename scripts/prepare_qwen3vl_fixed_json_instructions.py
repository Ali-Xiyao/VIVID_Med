"""Prepare fixed-JSON D0 clinical instruction records for Qwen3-VL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


def read_jsonl(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if max_samples is not None and len(rows) >= max_samples:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sample_id(sample: dict[str, Any], idx: int) -> str:
    extensions = sample.get("extensions") or {}
    return str(extensions.get("sample_id") or extensions.get("original_path") or idx)


def image_path(sample: dict[str, Any]) -> str:
    return str((sample.get("extensions") or {}).get("original_path") or sample.get("image_path") or "")


def normalize_state(value: Any) -> str | None:
    if value in {"present", "absent", "uncertain"}:
        return str(value)
    return None


def make_target(sample: dict[str, Any], labels: list[str]) -> dict[str, Any]:
    findings = sample.get("findings") or {}
    answerability = sample.get("answerability") or {}
    uncertainty = sample.get("uncertainty") or {}
    target = {
        "modality": "CXR",
        "study_view": sample.get("study_view"),
        "findings": {},
        "answerability": {},
        "uncertainty": {},
    }
    for label in labels:
        item = findings.get(label) or {}
        state = normalize_state(item.get("state"))
        target["findings"][label] = {"state": state}
        target["answerability"][label] = bool(answerability.get(label, state is not None))
        value = uncertainty.get(label)
        if value is True:
            target["uncertainty"][label] = "uncertain"
        elif value is False:
            target["uncertainty"][label] = "definite"
        else:
            target["uncertainty"][label] = None
    return target


def make_record(sample: dict[str, Any], idx: int, labels: list[str], version: str) -> dict[str, Any]:
    sid = sample_id(sample, idx)
    target = make_target(sample, labels)
    return {
        "instruction_id": f"{sid}_{version}_{idx:06d}",
        "sample_id": sid,
        "image_path": image_path(sample),
        "report": None,
        "report_text": None,
        "question": (
            "Return the fixed UMS JSON for this chest X-ray. "
            "Use only the provided schema fields and set unmentioned findings to null."
        ),
        "answer": json.dumps(target, ensure_ascii=False, sort_keys=True),
        "answer_short": None,
        "finding": "global",
        "state": "not_applicable",
        "answerability": "answerable",
        "uncertainty": None,
        "laterality": None,
        "location": None,
        "severity": None,
        "evidence_phrase": None,
        "evidence_span": None,
        "evidence_source": "structured_label",
        "answer_type": "fixed_json_schema",
        "visual_dependency": "medium",
        "counterfactual_type": None,
        "quality_flags": [version, "fixed_json_schema", "no_report_text"],
        "source_version": version,
        "source_mode": "local",
        "source": "local_ums_schema",
        "generation_model": None,
        "validation_status": "raw",
        "reject_reason": None,
        "metadata": {
            "patient_age": (sample.get("extensions") or {}).get("patient_age"),
            "patient_sex": (sample.get("extensions") or {}).get("patient_sex"),
            "study_view": sample.get("study_view"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--version", default="d0_fixed_json_schema")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--labels", nargs="*", default=DEFAULT_LABELS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = read_jsonl(args.input, max_samples=args.max_samples)
    rows = [make_record(sample, idx, list(args.labels), args.version) for idx, sample in enumerate(samples)]
    write_jsonl(args.output, rows)
    print(json.dumps({"input": str(args.input), "output": str(args.output), "records": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
