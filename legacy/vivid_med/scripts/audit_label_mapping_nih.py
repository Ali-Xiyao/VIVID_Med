"""Audit CheXpert/MIMIC to NIH label mapping confidence and risks."""

from __future__ import annotations

import argparse
from pathlib import Path

from case_study_modules_common import FINAL_DIR, write_csv_rows, write_md_table


DEFAULT_MAPPING = [
    {
        "chexpert_mimic_field": "Lung Opacity",
        "nih_label": "Infiltration / opacity",
        "mapping_confidence": "medium_low",
        "risk": "definition_mismatch",
        "note": "NIH Infiltration is not a clean synonym for CheXpert Lung Opacity.",
    },
    {"chexpert_mimic_field": "Pleural Effusion", "nih_label": "Pleural Effusion", "mapping_confidence": "high", "risk": "low", "note": "Closest direct mapping."},
    {"chexpert_mimic_field": "Pneumothorax", "nih_label": "Pneumothorax", "mapping_confidence": "high", "risk": "low", "note": "Closest direct mapping."},
    {"chexpert_mimic_field": "Cardiomegaly", "nih_label": "Cardiomegaly", "mapping_confidence": "high", "risk": "low", "note": "Closest direct mapping."},
    {"chexpert_mimic_field": "Edema", "nih_label": "Edema", "mapping_confidence": "medium", "risk": "label_noise", "note": "Clinical definition may differ by labeler and report policy."},
    {"chexpert_mimic_field": "Consolidation", "nih_label": "Consolidation", "mapping_confidence": "medium", "risk": "overlap_with_opacity", "note": "Can overlap with opacity/infiltration categories."},
    {"chexpert_mimic_field": "Atelectasis", "nih_label": "Atelectasis", "mapping_confidence": "medium_high", "risk": "view_quality_and_report_policy", "note": "Common label but radiographic/report policy varies."},
    {"chexpert_mimic_field": "Pneumonia", "nih_label": "Pneumonia", "mapping_confidence": "medium_low", "risk": "diagnosis_vs_finding", "note": "Often a clinical diagnosis rather than purely visual finding."},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, default=FINAL_DIR / "nih_label_mapping_audit.csv")
    parser.add_argument("--output-md", type=Path, default=FINAL_DIR / "nih_label_mapping_audit.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    columns = ["chexpert_mimic_field", "nih_label", "mapping_confidence", "risk", "note"]
    note = "Static mapping audit used by the NIH/domain diagnosis report. It should be updated if a different NIH label manifest is used."
    write_csv_rows(args.output_csv, DEFAULT_MAPPING, columns)
    write_md_table(args.output_md, "NIH Label Mapping Audit", DEFAULT_MAPPING, columns, note)


if __name__ == "__main__":
    main()

