"""Build frozen CheXbert targets for the deterministic 20k MIMIC pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import torch

from run_chexbert_lunguage_source import (
    RCSD_FINDINGS,
    load_model,
    run_inference,
    sha256_file,
)


SECTION_PATTERN = re.compile(
    r"(?ims)^\s*(FINDINGS?|IMPRESSION)\s*:\s*(.*?)"
    r"(?=^\s*[A-Z][A-Z0-9 /(),._+-]{2,}\s*:\s*|\Z)"
)


def stable_key(row: dict[str, str]) -> str:
    token = f"{row['patient_id']}|{row['study_id']}"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def select_rows(
    rows: list[dict[str, str]], train_count: int
) -> tuple[list[dict[str, str]], dict[str, int]]:
    train = sorted(
        (row for row in rows if row["split"] == "train"), key=stable_key
    )
    validate = sorted(
        (row for row in rows if row["split"] == "validate"), key=stable_key
    )
    if len(train) < train_count:
        raise ValueError(
            f"requested {train_count} train studies but found {len(train)}"
        )
    selected = train[:train_count] + validate
    return selected, {
        "train": train_count,
        "validate": len(validate),
        "total": len(selected),
    }


def extract_chexbert_text(text: str) -> tuple[str, str]:
    sections: dict[str, list[str]] = {"find": [], "impr": []}
    for match in SECTION_PATTERN.finditer(text):
        key = "find" if match.group(1).upper().startswith("FIND") else "impr"
        value = re.sub(r"\s+", " ", match.group(2)).strip()
        if value:
            sections[key].append(value)
    ordered = sections["find"] + sections["impr"]
    if ordered:
        pattern = (
            "find+impr"
            if sections["find"] and sections["impr"]
            else "find"
            if sections["find"]
            else "impr"
        )
        return "\n".join(ordered), pattern
    fallback = re.sub(r"\s+", " ", text).strip()
    if not fallback:
        raise ValueError("empty MIMIC report")
    return fallback, "full_report_fallback"


def load_canonical_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "patient_id",
            "study_id",
            "image_id",
            "split",
            "image_path",
            "report_path",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"canonical manifest missing: {sorted(missing)}")
        rows = list(reader)
    if any(row["split"] not in {"train", "validate"} for row in rows):
        raise ValueError("canonical manifest contains a forbidden split")
    return rows


def write_manifest(
    path: Path,
    selected: list[dict[str, str]],
    labels: list[dict[str, str]],
) -> None:
    label_by_study = {row["study_id"]: row for row in labels}
    if len(label_by_study) != len(selected):
        raise ValueError("CheXbert output is not one-to-one with selected studies")
    fields = list(selected[0]) + list(RCSD_FINDINGS)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            source = label_by_study[row["study_id"]]
            writer.writerow(
                {
                    **row,
                    **{finding: source[finding] for finding in RCSD_FINDINGS},
                }
            )


def write_overfit_subset(
    path: Path, pilot_path: Path, rows: int = 256
) -> dict[str, object]:
    with pilot_path.open("r", encoding="utf-8", newline="") as handle:
        values = list(csv.DictReader(handle))
        fields = list(values[0])
    candidates = [
        row
        for row in values
        if row["split"] == "train"
        and sum(bool(row[finding]) for finding in RCSD_FINDINGS) >= 2
    ]
    candidates.sort(key=stable_key)
    if len(candidates) < rows:
        raise ValueError(f"only {len(candidates)} overfit candidates")
    chosen = candidates[:rows]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(chosen)
    return {
        "rows": rows,
        "patients": len({row["patient_id"] for row in chosen}),
        "state_counts": {
            finding: dict(
                Counter(
                    row[finding] if row[finding] else "missing"
                    for row in chosen
                )
            )
            for finding in RCSD_FINDINGS
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-manifest", required=True, type=Path)
    parser.add_argument("--report-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--auxiliary-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overfit-output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--train-count", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    canonical = load_canonical_manifest(args.canonical_manifest)
    selected, split_counts = select_rows(canonical, args.train_count)
    reports = []
    section_counts: Counter[str] = Counter()
    for row in selected:
        report_path = args.report_root / row["report_path"]
        if not report_path.is_file():
            raise FileNotFoundError(report_path)
        text, pattern = extract_chexbert_text(
            report_path.read_text(encoding="utf-8", errors="strict")
        )
        reports.append(
            {
                "study_id": row["study_id"],
                "subject_id": row["patient_id"],
                "report": text,
            }
        )
        section_counts[pattern] += 1

    device = torch.device(args.device)
    model, tokenizer, model_audit = load_model(
        args.checkpoint, args.auxiliary_dir, device
    )
    labels, inference_audit = run_inference(
        reports, model, tokenizer, device, args.batch_size
    )
    write_manifest(args.output, selected, labels)
    overfit_audit = write_overfit_subset(
        args.overfit_output, args.output, rows=256
    )
    audit = {
        "schema_version": 1,
        "artifact": "mimic_chexbert_pilot",
        "pass": True,
        "selection": {
            "method": "ascending SHA256(patient_id|study_id)",
            "train_requested": args.train_count,
            "all_canonical_validation_included": True,
            "split_counts": split_counts,
        },
        "text_contract": {
            "primary_sections": ["FINDINGS", "IMPRESSION"],
            "section_order": ["FINDINGS", "IMPRESSION"],
            "fallback": "whitespace-normalized full report only when both absent",
            "section_counts": dict(section_counts),
        },
        "model_audit": model_audit,
        "inference_audit": inference_audit,
        "overfit_subset": overfit_audit,
        "hashes": {
            "canonical_manifest": sha256_file(args.canonical_manifest),
            "checkpoint": sha256_file(args.checkpoint),
            "pilot_manifest": sha256_file(args.output),
            "overfit_manifest": sha256_file(args.overfit_output),
        },
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
