"""Fail-closed validator for the D0-versus-D1 review lock."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def default_repo_root() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    if (project_root / "legacy" / "vivid_med").is_dir():
        return project_root
    return Path(__file__).resolve().parents[3]


def validate(payload: dict[str, object]) -> list[str]:
    errors: list[str] = []

    status = payload.get("status")
    if status not in {
        "PREREQUISITES_IN_PROGRESS",
        "OVERFIT_AUTHORIZED",
        "PILOT_AUTHORIZED",
        "TERMINAL",
    }:
        errors.append("invalid review state")
    if payload.get("test_sets_opened") != []:
        errors.append("test sets must remain unopened")
    target = payload.get("execution_target", {})
    if target.get("kind") != "sues_hpc_slurm":
        errors.append("execution target must remain the frozen SUES Slurm allocation")
    if target.get("host") != "sues-hpc":
        errors.append("remote host identity drifted")
    if target.get("node") != "gpu01":
        errors.append("remote node identity drifted")
    if target.get("allocation_id") != 3066:
        errors.append("remote allocation identity drifted")
    if target.get("preferred_gpu") != 0:
        errors.append("job-local GPU identity must remain zero")
    if target.get("server_allowed") is not True:
        errors.append("server execution authorization is missing")
    if target.get("slurm_allowed") is not True:
        errors.append("Slurm execution authorization is missing")
    if target.get("ownership_check_required_before_launch") is not True:
        errors.append("GPU ownership check must remain required")

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
    if method["teacher"] != "Qwen/Qwen3.5-2B":
        errors.append("frozen primary teacher identity drifted")
    for key in (
        "teacher_config_sha256",
        "teacher_weight_index_sha256",
        "teacher_weight_shard_sha256",
    ):
        value = method.get(key)
        if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
            errors.append(f"invalid frozen teacher hash: {key}")
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
    if any(not isinstance(value, bool) for value in prerequisites.values()):
        errors.append("all prerequisite flags must be boolean")
    provenance_resolved = (
        prerequisites.get("historical_checkpoint_hash_imported") is True
        or prerequisites.get("historical_checkpoint_unavailable_recorded") is True
    )
    readiness_keys = {
        key
        for key in prerequisites
        if key not in {
            "historical_checkpoint_hash_imported",
            "historical_checkpoint_unavailable_recorded",
        }
    }
    ready = (
        provenance_resolved
        and all(prerequisites[key] is True for key in readiness_keys)
    )
    execution_authorized = payload.get("execution_authorized")
    jobs_allowed = payload.get("training_jobs_allowed")
    if status == "PREREQUISITES_IN_PROGRESS":
        if execution_authorized is not False or jobs_allowed != 0:
            errors.append("prerequisite state cannot authorize training")
    elif status == "OVERFIT_AUTHORIZED":
        if not ready:
            errors.append("overfit cannot be authorized before prerequisites pass")
        if execution_authorized is not True or jobs_allowed != 2:
            errors.append("overfit state must authorize exactly two sequential arms")
    elif status == "PILOT_AUTHORIZED":
        if not ready:
            errors.append("pilot cannot be authorized before prerequisites pass")
        if execution_authorized is not True or jobs_allowed != 2:
            errors.append("pilot state must authorize exactly two sequential arms")
    elif status == "TERMINAL":
        if execution_authorized is not False or jobs_allowed != 0:
            errors.append("terminal state must authorize zero jobs")
        terminal = payload.get("terminal_decision")
        if not isinstance(terminal, dict):
            errors.append("terminal state requires a terminal decision")
        else:
            if terminal.get("decision") != "NO_GO":
                errors.append("terminal decision must remain NO_GO")
            if terminal.get("first_failed_condition") != (
                "validation_token_nll_relative_change"
            ):
                errors.append("terminal first failed condition drifted")
            if terminal.get("required_at_most") != -0.03:
                errors.append("terminal NLL threshold drifted")
            if terminal.get("repair_preregistered") is not False:
                errors.append("terminal state cannot invent a repair")
    return errors


def validate_terminal_result(
    *,
    lock: dict[str, object],
    result: dict[str, object],
) -> list[str]:
    errors: list[str] = []
    if result.get("artifact") != "rcsd_d0_d1_qwen35_2b_terminal_result":
        errors.append("invalid terminal result artifact")
    if result.get("decision") != "TERMINAL_NO_GO":
        errors.append("terminal result decision drifted")
    execution = result.get("execution", {})
    if execution.get("test_sets_opened") != []:
        errors.append("terminal result opened a test set")
    if execution.get("d0_returncode") != 0 or execution.get("d1_returncode") != 0:
        errors.append("terminal result requires two completed arms")

    d0 = result.get("d0", {})
    d1 = result.get("d1", {})
    try:
        observed = (float(d1["token_nll"]) - float(d0["token_nll"])) / float(
            d0["token_nll"]
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        errors.append("terminal result has invalid token NLL values")
        observed = None
    gate = result.get("primary_gate", {})
    if observed is not None:
        if not math.isclose(
            observed,
            float(gate.get("observed", float("nan"))),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            errors.append("terminal relative NLL arithmetic drifted")
        if observed <= -0.03:
            errors.append("terminal NO-GO contradicts the frozen NLL gate")
    if gate.get("required_at_most") != -0.03 or gate.get("passed") is not False:
        errors.append("terminal primary gate drifted")

    terminal = lock.get("terminal_decision", {})
    if terminal.get("observed") != gate.get("observed"):
        errors.append("lock and terminal result NLL values disagree")
    if terminal.get("required_at_most") != gate.get("required_at_most"):
        errors.append("lock and terminal result thresholds disagree")

    for arm in (d0, d1):
        for key in ("summary_sha256", "checkpoint_sha256"):
            value = arm.get(key)
            if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
                errors.append(f"invalid terminal artifact hash: {key}")
    hashes = result.get("hashes", {})
    for key, value in hashes.items():
        if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
            errors.append(f"invalid terminal evidence hash: {key}")

    consequences = result.get("consequences", {})
    for key in (
        "expert_development_probe_run",
        "repair_authorized",
        "teacher_scaling_authorized",
        "full_data_scaling_authorized",
        "multi_seed_authorized",
        "external_test_authorized",
    ):
        if consequences.get(key) is not False:
            errors.append(f"terminal consequence must remain false: {key}")
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
        "--terminal-result",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "audit"
            / "rcsd_d0_d1_qwen35_2b_terminal_result.json"
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
        default=default_repo_root(),
    )
    args = parser.parse_args()
    payload = json.loads(args.lock.read_text(encoding="utf-8"))
    errors = validate(payload)
    if payload.get("status") == "TERMINAL":
        terminal_result = json.loads(
            args.terminal_result.read_text(encoding="utf-8")
        )
        errors.extend(
            validate_terminal_result(lock=payload, result=terminal_result)
        )
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
