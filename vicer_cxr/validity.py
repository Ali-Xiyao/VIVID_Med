"""Contracts and statistics for VICER-CXR intervention validity V0."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr


VALIDITY_FINDINGS = (
    "pneumothorax",
    "consolidation",
    "pleural_effusion",
    "cardiomegaly",
)
VALIDITY_ROLES = {
    "critic_train": 10,
    "critic_calibration": 5,
    "verifier_train": 10,
    "verifier_calibration": 6,
    "validity_eval": 8,
}
V0_SCHEMA_VERSION = "vicer-v0-validity-dose-response-v1"
THRESHOLD_ATOL = 1e-12


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def meets_minimum(value: float, minimum: float) -> bool:
    """Compare a measured value to a frozen threshold without binary-float artifacts."""
    return bool(np.isfinite(value) and value + THRESHOLD_ATOL >= minimum)


def stable_rank(seed: int, *parts: str) -> int:
    payload = ":".join((str(seed), *map(str, parts)))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def validate_v0_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("VICER V0 manifest is empty")
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("VICER V0 sample_id values are not unique")
    image_roles: dict[str, set[str]] = {}
    for row in rows:
        finding = str(row.get("canonical_statement_id"))
        role = str(row.get("v0_role"))
        state = str(row.get("state"))
        if finding not in VALIDITY_FINDINGS or role not in VALIDITY_ROLES:
            raise ValueError("VICER V0 finding/role boundary changed")
        if row.get("source_split") != "train" or row.get("patient_level_claim") is not False:
            raise ValueError("VICER V0 must remain VinDr-train image-level development")
        if row.get("arise_identity_excluded") is not True:
            raise ValueError("VICER V0 row did not prove ARISE identity exclusion")
        if state == "support":
            if int(row.get("positive_vote_count", 0)) < 1 or not row.get("roi_boxes"):
                raise ValueError("VICER V0 support row lacks reader box evidence")
        elif state == "contradict":
            if int(row.get("positive_vote_count", -1)) != 0:
                raise ValueError("VICER V0 negative row is not 0-of-3")
            if role.startswith("critic_") and not row.get("roi_boxes"):
                raise ValueError("VICER validity critic negative lacks a frozen ROI template")
        else:
            raise ValueError("VICER V0 is support/contradict only")
        image_id = str(row["image_id"])
        image_roles.setdefault(image_id, set()).add(role)
    if any(len(roles) != 1 for roles in image_roles.values()):
        raise ValueError("VICER V0 image leaked across critic/verifier/evaluation roles")

    counts = Counter(
        (str(row["canonical_statement_id"]), str(row["v0_role"]), str(row["state"]))
        for row in rows
    )
    for finding in VALIDITY_FINDINGS:
        for role, positive_count in VALIDITY_ROLES.items():
            if counts[(finding, role, "support")] != positive_count:
                raise ValueError(f"VICER V0 support count changed: {finding}/{role}")
            expected_negative = 0 if role == "validity_eval" else positive_count
            if counts[(finding, role, "contradict")] != expected_negative:
                raise ValueError(f"VICER V0 negative count changed: {finding}/{role}")
    return {
        "records": len(rows),
        "unique_images": len(image_roles),
        "counts": {
            "|".join(key): value for key, value in sorted(counts.items())
        },
    }


def linear_margin(features: np.ndarray, model: dict[str, np.ndarray | float]) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    mean = np.asarray(model["scaler_mean"], dtype=np.float64)
    scale = np.asarray(model["scaler_scale"], dtype=np.float64)
    weight = np.asarray(model["weight"], dtype=np.float64)
    intercept = float(model["intercept"])
    if values.shape[-1] != mean.size or mean.shape != scale.shape or weight.shape != mean.shape:
        raise ValueError("VICER linear head dimensions changed")
    normalized = (values - mean) / np.maximum(scale, np.finfo(np.float64).eps)
    return normalized @ weight + intercept


def summarize_v0_rows(
    rows: Iterable[dict[str, Any]],
    *,
    minimum_critic_auroc: float,
    minimum_verifier_auroc: float,
    minimum_monotonic_spearman: float,
    minimum_preservation: float,
    minimum_realism: float,
    minimum_valid_fraction: float,
) -> dict[str, Any]:
    records = list(rows)
    if not records:
        raise ValueError("VICER V0 result rows are empty")
    identities = {
        (str(row["sample_id"]), str(row["operator_family"]), float(row["strength"]))
        for row in records
    }
    if len(identities) != len(records):
        raise ValueError("VICER V0 result identities are not unique")

    per_family: dict[str, Any] = {}
    families = sorted({str(row["operator_family"]) for row in records})
    for family in families:
        subset = [row for row in records if row["operator_family"] == family]
        per_finding: dict[str, Any] = {}
        family_passes = []
        for finding in VALIDITY_FINDINGS:
            finding_rows = [row for row in subset if row["canonical_statement_id"] == finding]
            strengths = sorted({float(row["strength"]) for row in finding_rows})
            if len(strengths) < 3:
                raise ValueError("VICER V0 needs at least three strengths per family/finding")
            med_remove = []
            med_preserve = []
            med_realism = []
            med_gap = []
            valid_flags = []
            for strength in strengths:
                level = [row for row in finding_rows if float(row["strength"]) == strength]
                med_remove.append(float(np.median([float(row["q_remove"]) for row in level])))
                med_preserve.append(float(np.median([float(row["q_preserve"]) for row in level])))
                med_realism.append(float(np.median([float(row["q_realism"]) for row in level])))
                med_gap.append(float(np.median([float(row["target_control_gap"]) for row in level])))
                valid_flags.extend(bool(row["valid_intervention"]) for row in level)
            correlation = float(spearmanr(strengths, med_remove).statistic)
            strongest_preservation = med_preserve[-1]
            strongest_realism = med_realism[-1]
            valid_fraction = float(np.mean(valid_flags))
            positive_valid_gaps = [
                float(row["target_control_gap"])
                for row in finding_rows
                if bool(row["valid_intervention"])
            ]
            valid_gap_mean = float(np.mean(positive_valid_gaps)) if positive_valid_gaps else float("nan")
            passed = bool(
                meets_minimum(correlation, minimum_monotonic_spearman)
                and meets_minimum(strongest_preservation, minimum_preservation)
                and meets_minimum(strongest_realism, minimum_realism)
                and meets_minimum(valid_fraction, minimum_valid_fraction)
                and positive_valid_gaps
                and valid_gap_mean > 0.0
            )
            family_passes.append(passed)
            per_finding[finding] = {
                "strengths": strengths,
                "median_q_remove": med_remove,
                "median_q_preserve": med_preserve,
                "median_q_realism": med_realism,
                "median_target_control_gap": med_gap,
                "q_remove_strength_spearman": correlation,
                "valid_fraction": valid_fraction,
                "mean_valid_target_control_gap": valid_gap_mean,
                "pass": passed,
            }
        per_family[family] = {
            "per_finding": per_finding,
            "all_findings_pass": all(family_passes),
        }
    critic_auc = min(float(row["critic_calibration_auroc"]) for row in records)
    verifier_auc = min(float(row["verifier_calibration_auroc"]) for row in records)
    heads_pass = meets_minimum(critic_auc, minimum_critic_auroc) and meets_minimum(
        verifier_auc, minimum_verifier_auroc
    )
    surviving = [family for family, value in per_family.items() if value["all_findings_pass"]]
    result = {
        "schema_version": V0_SCHEMA_VERSION,
        "records": len(records),
        "findings": list(VALIDITY_FINDINGS),
        "minimum_calibration_auroc": {
            "critic": critic_auc,
            "verifier": verifier_auc,
        },
        "head_gate_pass": heads_pass,
        "per_operator_family": per_family,
        "surviving_operator_families": surviving,
        "v0_pass": bool(heads_pass and surviving),
        "v1_authorized": bool(heads_pass and surviving),
        "thresholds": {
            "minimum_critic_auroc": minimum_critic_auroc,
            "minimum_verifier_auroc": minimum_verifier_auroc,
            "minimum_monotonic_spearman": minimum_monotonic_spearman,
            "minimum_preservation": minimum_preservation,
            "minimum_realism": minimum_realism,
            "minimum_valid_fraction": minimum_valid_fraction,
        },
    }
    result["canonical_sha256"] = canonical_sha256(result)
    return result
