"""Fail-closed validator for the D0-versus-D1 review lock."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def validate(payload: dict[str, object]) -> list[str]:
    errors: list[str] = []

    if payload.get("status") != "PREPARED_NOT_APPROVED":
        errors.append("review status must remain PREPARED_NOT_APPROVED")
    if payload.get("execution_authorized") is not False:
        errors.append("execution must remain unauthorized")
    if payload.get("training_jobs_allowed") != 0:
        errors.append("training_jobs_allowed must remain zero")
    if payload.get("test_sets_opened") != []:
        errors.append("test sets must remain unopened")

    candidate = payload["candidate_d1"]
    if candidate["only_delta_from_d0_cp"] != (
        "entropy agreement weight on hard-target finding token spans"
    ):
        errors.append("D1 single-factor identity drifted")
    if candidate["hard_target_source"] != "chexbert":
        errors.append("D1 hard target source drifted")
    if candidate["weight_formula"] != (
        "m_ic * (1 - H(q_bar_ic) / log(3))"
    ):
        errors.append("D1 weight formula drifted")
    for forbidden in (
        "target_replacement_allowed",
        "learned_label_model_allowed",
        "posterior_fusion_allowed",
        "field_anchor_allowed",
    ):
        if candidate[forbidden] is not False:
            errors.append(f"{forbidden} must remain false")

    controlled = payload["controlled_d0"]
    method = controlled["method_contract"]
    training = controlled["training_contract"]
    if method["teacher"] != "Qwen/Qwen2.5-1.5B-Instruct":
        errors.append("historical teacher identity drifted")
    if method["spd_groups"] != 4 or method["tokens_per_group"] != 2:
        errors.append("SPD must remain exactly 4x2")
    if method["spd_orthogonality_weight"] != 0.02:
        errors.append("SPD orthogonality weight drifted")
    if training["checkpoint_rule"] != (
        "strictly lower unweighted validation token NLL"
    ):
        errors.append("checkpoint rule drifted")
    if training["max_steps"] != 3000 or training["seed"] != 0:
        errors.append("paired diagnostic budget or seed drifted")

    for key in (
        "canonical_g0_manifest_sha256",
        "row_lock_sha256",
    ):
        value = controlled["data_contract"][key]
        if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
            errors.append(f"invalid frozen hash: {key}")

    gate = payload["promotion_gate"]
    expected_gate = {
        "validation_token_nll_relative_change_at_most": -0.03,
        "expert_dev_macro_auroc_gain_at_least": 0.005,
        "expert_dev_ece_change_at_most": 0.01,
        "findings_below_minus_0_02_at_most": 2,
        "positive_prevalence_tiers_at_least": 2,
        "positive_findings_at_least": 3,
        "threshold_changes_allowed": False,
        "all_conditions_required": True,
    }
    if gate != expected_gate:
        errors.append("promotion gate drifted")

    prerequisites = payload["prerequisites"]
    if any(value is not False for value in prerequisites.values()):
        errors.append("review prerequisites must remain incomplete in this branch")
    return errors


def validate_source_contract(
    *,
    repo_root: Path,
    source_contract: Path,
) -> list[str]:
    errors: list[str] = []
    with source_contract.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 8:
        errors.append("D0 source contract must contain exactly eight rows")
        return errors

    seen_paths: set[str] = set()
    for row in rows:
        relative_path = row["path"]
        if relative_path in seen_paths:
            errors.append(f"duplicate D0 source path: {relative_path}")
            continue
        seen_paths.add(relative_path)
        source_path = repo_root / relative_path
        if not source_path.is_file():
            errors.append(f"missing D0 source file: {relative_path}")
            continue
        text = source_path.read_text(encoding="utf-8")
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        actual = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if actual != row["lf_normalized_sha256"]:
            errors.append(f"D0 normalized hash drifted: {relative_path}")
        if not re.fullmatch(r"[0-9a-f]{40}", row["git_blob_sha1"]):
            errors.append(f"invalid Git blob identity: {relative_path}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lock",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "audit"
            / "rcsd_d0_d1_review_lock.json"
        ),
    )
    parser.add_argument(
        "--source-contract",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "audit"
            / "tables"
            / "rcsd_d0_source_contract.csv"
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    args = parser.parse_args()
    payload = json.loads(args.lock.read_text(encoding="utf-8"))
    errors = validate(payload)
    errors.extend(
        validate_source_contract(
            repo_root=args.repo_root,
            source_contract=args.source_contract,
        )
    )
    result = {
        "artifact": "rcsd_d0_d1_review_integrity",
        "audit_completed": True,
        "pass": not errors,
        "errors": errors,
        "execution_authorized": payload["execution_authorized"],
        "training_jobs_allowed": payload["training_jobs_allowed"],
    }
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
