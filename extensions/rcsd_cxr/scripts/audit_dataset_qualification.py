"""Audit the fail-closed RCSD dataset qualification ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


REQUIRED_FIELDS = {
    "version",
    "parent",
    "roles",
    "access",
    "license_status",
    "eligibility",
    "test_exposure",
    "evidence",
}
ALLOWED_ELIGIBILITY = {"qualified", "conditional", "excluded"}
EXPECTED_PARENT = {
    "mimic_cxr_metadata": "mimic_cxr",
    "ms_cxr": "mimic_cxr",
    "chest_imagenome": "mimic_cxr",
    "chexpert_plus": "chexpert",
    "chexlocalize_validation": "chexpert",
    "chexlocalize_test": "chexpert",
}


def audit_qualification(
    qualification: dict[str, object],
    *,
    chexpert_lineage: dict[str, object],
) -> dict[str, object]:
    datasets = qualification.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("qualification ledger has no datasets mapping")
    errors: list[str] = []
    warnings: list[str] = []
    for name, raw in datasets.items():
        if not isinstance(raw, dict):
            errors.append(f"{name}: record is not a mapping")
            continue
        missing = REQUIRED_FIELDS - set(raw)
        if missing:
            errors.append(f"{name}: missing fields {sorted(missing)}")
        eligibility = raw.get("eligibility")
        if eligibility not in ALLOWED_ELIGIBILITY:
            errors.append(f"{name}: invalid eligibility {eligibility}")
        if eligibility == "conditional":
            warnings.append(f"{name}: conditional and locked from execution")
        expected_parent = EXPECTED_PARENT.get(str(name))
        if expected_parent is not None and raw.get("parent") != expected_parent:
            errors.append(
                f"{name}: parent {raw.get('parent')} != expected {expected_parent}"
            )
        evidence = raw.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"{name}: evidence must be a non-empty list")

    for required in ("mimic_cxr", "mimic_cxr_metadata"):
        record = datasets.get(required, {})
        if not isinstance(record, dict) or record.get("eligibility") != "qualified":
            errors.append(f"{required}: Track A requires qualified eligibility")

    sealed = datasets.get("chexlocalize_test", {})
    if not isinstance(sealed, dict) or sealed.get("eligibility") != "excluded":
        errors.append("chexlocalize_test must be excluded")
    if isinstance(sealed, dict) and sealed.get("test_exposure") != "sealed_absent":
        errors.append("chexlocalize_test must be recorded as sealed_absent")

    lineage_summary = chexpert_lineage.get("summary", {})
    lineage_decision = chexpert_lineage.get("decision", {})
    if lineage_summary.get("multi_split_patients") != 0:
        errors.append("CheXpert-Plus has cross-split patients")
    if lineage_summary.get("valid_path_symmetric_difference") != 0:
        errors.append("CheXpert-Plus valid paths do not equal official CheXpert valid")
    if not lineage_decision.get("chexpert_plus_valid_is_official_chexpert_valid"):
        errors.append("CheXpert-Plus valid identity decision is false")

    return {
        "schema_version": 1,
        "gate": "G0_lineage_overlap_license",
        "pass": not errors,
        "scope": "Track A execution qualification; conditional external datasets stay locked",
        "errors": errors,
        "warnings": warnings,
        "qualified": sorted(
            name
            for name, raw in datasets.items()
            if isinstance(raw, dict) and raw.get("eligibility") == "qualified"
        ),
        "conditional": sorted(
            name
            for name, raw in datasets.items()
            if isinstance(raw, dict) and raw.get("eligibility") == "conditional"
        ),
        "excluded": sorted(
            name
            for name, raw in datasets.items()
            if isinstance(raw, dict) and raw.get("eligibility") == "excluded"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualification", required=True, type=Path)
    parser.add_argument("--chexpert-lineage", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    qualification = yaml.safe_load(args.qualification.read_text(encoding="utf-8"))
    chexpert_lineage = json.loads(
        args.chexpert_lineage.read_text(encoding="utf-8")
    )
    try:
        result = audit_qualification(
            qualification, chexpert_lineage=chexpert_lineage
        )
    except Exception as error:
        result = {
            "schema_version": 1,
            "gate": "G0_lineage_overlap_license",
            "pass": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
