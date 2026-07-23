"""Aggregate the Chest ImaGenome gold report-label coverage without extracting PHI."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile


GOLD_MEMBER = (
    "chest-imagenome-dataset-1.0.0/gold_dataset/"
    "gold_attributes_relations_500pts_500studies1st.txt"
)
DIRECT_TERMS = {
    "Cardiomegaly": {"cardiomegaly", "enlarged cardiac silhouette"},
    "Lung Opacity": {"lung opacity"},
    "Lung Lesion": {"lung lesion"},
    "Edema": {"pulmonary edema", "edema"},
    "Consolidation": {"consolidation"},
    "Pneumonia": {"pneumonia"},
    "Atelectasis": {"atelectasis"},
    "Pneumothorax": {"pneumothorax"},
    "Pleural Effusion": {"pleural effusion"},
    "Fracture": {"fracture", "rib fracture"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_gold(archive: Path) -> dict[str, object]:
    with ZipFile(archive) as package:
        names = set(package.namelist())
        if GOLD_MEMBER not in names:
            raise FileNotFoundError(f"gold member missing from archive: {GOLD_MEMBER}")
        text = io.TextIOWrapper(package.open(GOLD_MEMBER), encoding="utf-8")
        reader = csv.DictReader(text, delimiter="\t")
        required = {"patient_id", "study_id", "relation", "label_name", "categoryID"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"gold file missing columns: {sorted(missing)}")
        patients: set[str] = set()
        studies: set[str] = set()
        relation_counts: dict[str, int] = {}
        per_finding: dict[str, dict[str, object]] = {
            name: {"rows": 0, "studies": set(), "positive": 0, "negative": 0}
            for name in DIRECT_TERMS
        }
        support_device = {"rows": 0, "studies": set(), "positive": 0, "negative": 0}
        total = 0
        for row in reader:
            total += 1
            patients.add(row["patient_id"])
            studies.add(row["study_id"])
            relation = str(row["relation"]).strip()
            relation_counts[relation] = relation_counts.get(relation, 0) + 1
            label = str(row["label_name"]).strip().lower()
            for name, terms in DIRECT_TERMS.items():
                if label in terms:
                    record = per_finding[name]
                    record["rows"] += 1
                    record["studies"].add(row["study_id"])
                    if relation == "1":
                        record["positive"] += 1
                    elif relation == "0":
                        record["negative"] += 1
            if str(row["categoryID"]).strip().lower() in {"tubesandlines", "device"}:
                support_device["rows"] += 1
                support_device["studies"].add(row["study_id"])
                if relation == "1":
                    support_device["positive"] += 1
                elif relation == "0":
                    support_device["negative"] += 1
    per_finding["Support Devices"] = support_device
    serializable: dict[str, dict[str, int]] = {}
    for name, raw in per_finding.items():
        serializable[name] = {
            "rows": int(raw["rows"]),
            "studies": len(raw["studies"]),
            "positive": int(raw["positive"]),
            "negative": int(raw["negative"]),
        }
    eligible = sorted(
        name
        for name, raw in serializable.items()
        if raw["positive"] > 0 and raw["negative"] > 0
    )
    return {
        "schema_version": 1,
        "artifact": "chest_imagenome_gold_coverage_audit",
        "pass": True,
        "archive_sha256": sha256_file(archive),
        "gold_member": GOLD_MEMBER,
        "rows": total,
        "patients": len(patients),
        "studies": len(studies),
        "relation_counts": relation_counts,
        "per_finding": serializable,
        "binary_quality_eligible_findings": eligible,
        "binary_quality_eligible_count": len(eligible),
        "uncertain_gold_available": False,
        "decision": (
            "auxiliary present/absent quality audit only; cannot unlock the "
            "three-state posterior gate without LUNGUAGE or equivalent gold"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = audit_gold(args.archive)
    except Exception as error:
        result = {
            "schema_version": 1,
            "artifact": "chest_imagenome_gold_coverage_audit",
            "pass": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
