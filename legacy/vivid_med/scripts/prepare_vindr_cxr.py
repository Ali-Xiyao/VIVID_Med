"""Build deterministic VinDr-CXR UMS manifests and a data-quality audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = Path(r"H:\Xiyao_Wang\000_Public Dataset")
DATASET_DIRNAME = "vindr-cxr-an-open-dataset-of-chest-x-rays-with-radiologists-annotations-1.0.0"
DEFAULT_DATASET_ROOT = PUBLIC_ROOT / DATASET_DIRNAME
DEFAULT_MAPPING = ROOT / "configs/qwen3vl_instruction/vsl_cxr/phase6_external/vindr_chexpert_label_mapping.json"
DEFAULT_OUTPUT_DIR = ROOT / "data/dataset/processed"
DEFAULT_AUDIT_JSON = ROOT / "outputs/final_tables/vindr_cxr_data_quality_audit.json"
DEFAULT_AUDIT_MD = ROOT / "outputs/final_tables/vindr_cxr_data_quality_audit.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT_JSON)
    parser.add_argument("--audit-md", type=Path, default=DEFAULT_AUDIT_MD)
    parser.add_argument("--allow-missing-images", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def binary(value: str) -> int:
    return 1 if str(value).strip() == "1" else 0


def ums_record(
    image_id: str,
    split: str,
    source_values: dict[str, int],
    source_to_chexpert: dict[str, str],
) -> dict[str, Any]:
    findings = {
        target: {"state": "present" if source_values[source] else "absent", "score": None}
        for source, target in source_to_chexpert.items()
    }
    positive_original = sorted(source for source, value in source_values.items() if value)
    relative_path = f"{DATASET_DIRNAME}/{split}/{image_id}.dicom"
    return {
        "modality": "CXR",
        "anatomy": ["chest"],
        "findings": findings,
        "laterality": None,
        "study_view": None,
        "geometry": {"bbox": None, "mask": None, "keypoints": None},
        "measurements": {},
        "answerability": {**{target: True for target in findings}, "study_view": False, "laterality": False},
        "uncertainty": {target: False for target in findings},
        "provenance": {
            target: {
                "label_source": "VinDr-CXR image-level annotation",
                "verifier_version": "vindr_chexpert_direct_v1",
                "failure_type": None,
            }
            for target in findings
        },
        "verifier": {"pass": True, "failure_type": None, "confidence": 1.0, "role": "external_label"},
        "extensions": {
            "dataset": "VinDr-CXR-1.0.0",
            "split": split,
            "image_id": image_id,
            "sample_id": f"vindr_{split}_{image_id}",
            "original_path": relative_path,
            "original_positive_labels": positive_original,
            "label_mapping_version": "vindr_chexpert_direct_v1",
        },
    }


def aggregate_train(rows: list[dict[str, str]], source_labels: list[str]) -> dict[str, dict[str, int]]:
    votes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    readers: Counter[str] = Counter()
    for row in rows:
        image_id = row["image_id"]
        readers[image_id] += 1
        for label in source_labels:
            votes[image_id][label] += binary(row[label])
    bad_reader_counts = {image_id: count for image_id, count in readers.items() if count != 3}
    if bad_reader_counts:
        examples = list(bad_reader_counts.items())[:10]
        raise ValueError(f"Expected exactly 3 train readers per image; examples={examples}")
    return {
        image_id: {label: int(label_votes[label] >= 2) for label in source_labels}
        for image_id, label_votes in votes.items()
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def label_counts(rows: list[dict[str, Any]], labels: list[str]) -> dict[str, int]:
    return {
        label: sum(row["findings"][label]["state"] == "present" for row in rows)
        for label in labels
    }


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    mapping_payload = json.loads(args.mapping.read_text(encoding="utf-8"))
    source_to_chexpert: dict[str, str] = mapping_payload["source_to_chexpert"]
    source_labels = list(source_to_chexpert)
    target_labels = list(source_to_chexpert.values())
    annotations = dataset_root / "annotations"

    train_raw = read_csv(annotations / "image_labels_train.csv")
    test_raw = read_csv(annotations / "image_labels_test.csv")
    train_aggregated = aggregate_train(train_raw, source_labels)
    test_aggregated = {
        row["image_id"]: {label: binary(row[label]) for label in source_labels}
        for row in test_raw
    }
    overlap = sorted(set(train_aggregated) & set(test_aggregated))
    if overlap:
        raise ValueError(f"Official train/test image-id overlap detected: {overlap[:10]}")

    train_rows = [
        ums_record(image_id, "train", values, source_to_chexpert)
        for image_id, values in sorted(train_aggregated.items())
    ]
    test_rows = [
        ums_record(image_id, "test", values, source_to_chexpert)
        for image_id, values in sorted(test_aggregated.items())
    ]

    missing = {"train": [], "test": []}
    for split, rows in (("train", train_rows), ("test", test_rows)):
        for row in rows:
            relative = row["extensions"]["original_path"]
            path = args.dataset_root.parent / relative
            if not path.is_file():
                missing[split].append(relative)
    if not args.allow_missing_images and any(missing.values()):
        counts = {split: len(values) for split, values in missing.items()}
        raise FileNotFoundError(f"VinDr extraction is incomplete; missing image counts={counts}")

    train_path = args.output_dir / "vindr_cxr_external_train_majority_ums.jsonl"
    test_path = args.output_dir / "vindr_cxr_external_test_ums.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(test_path, test_rows)

    no_finding_conflicts = {}
    for split, rows in (("train", train_rows), ("test", test_rows)):
        conflicts = 0
        for row in rows:
            findings = row["findings"]
            if findings["No Finding"]["state"] == "present" and any(
                findings[label]["state"] == "present" for label in target_labels if label != "No Finding"
            ):
                conflicts += 1
        no_finding_conflicts[split] = conflicts

    audit = {
        "dataset_root": str(dataset_root),
        "mapping_path": str(args.mapping.resolve()),
        "protocol": mapping_payload["protocol"],
        "train_source_rows": len(train_raw),
        "train_images": len(train_rows),
        "test_source_rows": len(test_raw),
        "test_images": len(test_rows),
        "train_test_image_id_overlap": len(overlap),
        "missing_images": {split: len(values) for split, values in missing.items()},
        "missing_image_examples": {split: values[:10] for split, values in missing.items()},
        "mapped_labels": target_labels,
        "primary_external_labels": mapping_payload["primary_external_labels"],
        "train_positive_counts": label_counts(train_rows, target_labels),
        "test_positive_counts": label_counts(test_rows, target_labels),
        "no_finding_conflicts": no_finding_conflicts,
        "outputs": {"train": str(train_path), "test": str(test_path)},
        "status": "ready" if not any(missing.values()) else "manifest_ready_extraction_incomplete",
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.audit_md.write_text(
        "# VinDr-CXR Data Quality Audit\n\n"
        f"- Status: `{audit['status']}`\n"
        f"- Dataset root: `{dataset_root}`\n"
        f"- Official split: {len(train_rows)} train images / {len(test_rows)} test images; overlap={len(overlap)}.\n"
        f"- Missing images: train={len(missing['train'])}, test={len(missing['test'])}.\n"
        f"- Primary comparable labels: {', '.join(mapping_payload['primary_external_labels'])}.\n"
        "- Edema is retained in the manifest but excluded from the primary VinDr macro-AUC because the official test table has zero positive Edema rows.\n"
        f"- Train positive counts: `{json.dumps(audit['train_positive_counts'], ensure_ascii=False)}`\n"
        f"- Test positive counts: `{json.dumps(audit['test_positive_counts'], ensure_ascii=False)}`\n"
        f"- No-Finding conflicts: `{json.dumps(no_finding_conflicts, ensure_ascii=False)}`\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
