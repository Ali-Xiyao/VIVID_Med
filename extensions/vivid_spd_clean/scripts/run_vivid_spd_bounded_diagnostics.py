"""Run the two preregistered strict VIVID/SPD diagnostics sequentially."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


ARMS = ("ums_prefix8", "ums_spd4x2_no_ortho")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    return process.returncode


def write_state(path: Path, state: dict[str, object]) -> None:
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--strict-root", required=True, type=Path)
    parser.add_argument("--hard-manifest", required=True, type=Path)
    parser.add_argument("--overfit-ids", required=True, type=Path)
    parser.add_argument("--mimic-image-root", required=True, type=Path)
    parser.add_argument("--teacher-path", required=True, type=Path)
    parser.add_argument("--backbone-weights", required=True, type=Path)
    parser.add_argument("--probe-train-manifest", required=True, type=Path)
    parser.add_argument("--expert-manifest", required=True, type=Path)
    parser.add_argument("--chexpert-root", required=True, type=Path)
    parser.add_argument("--strict-lock", required=True, type=Path)
    parser.add_argument("--diagnostic-lock", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    args.run_root.mkdir(parents=True, exist_ok=False)
    state_path = args.run_root / "queue_state.json"
    state: dict[str, object] = {
        "schema_version": 1,
        "artifact": "strict_vivid_spd_bounded_diagnostic_queue",
        "started_at_unix": time.time(),
        "status": "running",
        "stage": "authority_check",
        "arms": [],
    }
    write_state(state_path, state)
    strict_verdict = json.loads(
        (args.strict_root / "s3_verdict.json").read_text(encoding="utf-8")
    )
    diagnostic_lock = json.loads(
        args.diagnostic_lock.read_text(encoding="utf-8")
    )
    authority = diagnostic_lock["strict_authority"]
    authority_paths = {
        "s3_verdict_sha256": args.strict_root / "s3_verdict.json",
        "prefix_s3_summary_sha256": (
            args.strict_root / "s3" / "ums_prefix4" / "summary.json"
        ),
        "spd_s3_summary_sha256": (
            args.strict_root / "s3" / "ums_spd4x2" / "summary.json"
        ),
        "prefix_s2_checkpoint_sha256": (
            args.strict_root / "s2" / "ums_prefix4" / "best.pt"
        ),
        "spd_s2_checkpoint_sha256": (
            args.strict_root / "s2" / "ums_spd4x2" / "best.pt"
        ),
    }
    observed_hashes = {
        key: sha256_file(path) for key, path in authority_paths.items()
    }
    hash_pass = all(
        observed_hashes[key] == authority[key] for key in authority_paths
    )
    if strict_verdict["verdict"] != "STRICT_NO_GO_DIAGNOSTIC_OPEN":
        state.update(
            status="blocked",
            terminal_reason="strict_verdict_does_not_authorize_diagnostics",
            finished_at_unix=time.time(),
        )
        write_state(state_path, state)
        return 20
    if not hash_pass:
        state.update(
            status="blocked",
            terminal_reason="strict_authority_hash_mismatch",
            observed_hashes=observed_hashes,
            finished_at_unix=time.time(),
        )
        write_state(state_path, state)
        return 21
    state["strict_authority_hashes"] = observed_hashes
    write_state(state_path, state)

    active = set(ARMS)
    for stage, mode in (("S1", "overfit"), ("S2", "pilot")):
        state["stage"] = stage
        write_state(state_path, state)
        for arm in ARMS:
            if arm not in active:
                continue
            output = args.run_root / stage.lower() / arm
            command = [
                sys.executable,
                str(scripts / "train_vivid_spd_token.py"),
                "--arm",
                arm,
                "--mode",
                mode,
                "--hard-manifest",
                str(args.hard_manifest),
                "--image-root",
                str(args.mimic_image_root),
                "--teacher-path",
                str(args.teacher_path),
                "--backbone-weights",
                str(args.backbone_weights),
                "--output-dir",
                str(output),
                "--device",
                args.device,
            ]
            if mode == "overfit":
                command.extend(["--overfit-ids", str(args.overfit_ids)])
            arm_state = {
                "stage": stage,
                "arm": arm,
                "status": "running",
                "command": command,
                "started_at_unix": time.time(),
            }
            state["arms"].append(arm_state)
            write_state(state_path, state)
            returncode = run(
                command,
                log_path=args.run_root / stage.lower() / f"{arm}.log",
            )
            arm_state.update(
                status="passed" if returncode == 0 else "diagnostic_failed",
                returncode=returncode,
                finished_at_unix=time.time(),
            )
            if returncode != 0:
                active.remove(arm)
            write_state(state_path, state)

    state["stage"] = "S3"
    write_state(state_path, state)
    for arm in ARMS:
        if arm not in active:
            continue
        output = args.run_root / "s3" / arm
        command = [
            sys.executable,
            str(scripts / "train_chexpert_probe.py"),
            "--arm",
            arm,
            "--vision-checkpoint",
            str(args.run_root / "s2" / arm / "best.pt"),
            "--train-manifest",
            str(args.probe_train_manifest),
            "--expert-manifest",
            str(args.expert_manifest),
            "--image-root",
            str(args.chexpert_root),
            "--output-dir",
            str(output),
            "--device",
            args.device,
        ]
        arm_state = {
            "stage": "S3",
            "arm": arm,
            "status": "running",
            "command": command,
            "started_at_unix": time.time(),
        }
        state["arms"].append(arm_state)
        write_state(state_path, state)
        returncode = run(
            command, log_path=args.run_root / "s3" / f"{arm}.log"
        )
        arm_state.update(
            status="passed" if returncode == 0 else "implementation_failed",
            returncode=returncode,
            finished_at_unix=time.time(),
        )
        if returncode != 0:
            active.remove(arm)
        write_state(state_path, state)

    if active != set(ARMS):
        state.update(
            status="terminal_no_go",
            stage="diagnostic_verdict",
            verdict="TERMINAL_NO_GO",
            terminal_reason="one_or_more_diagnostic_arms_failed",
            finished_at_unix=time.time(),
        )
        write_state(state_path, state)
        return 0

    state["stage"] = "diagnostic_verdict"
    write_state(state_path, state)
    verdict_path = args.run_root / "diagnostic_verdict.json"
    command = [
        sys.executable,
        str(scripts / "summarize_vivid_spd_diagnostics.py"),
        "--strict-root",
        str(args.strict_root),
        "--diagnostic-root",
        str(args.run_root),
        "--strict-lock",
        str(args.strict_lock),
        "--diagnostic-lock",
        str(args.diagnostic_lock),
        "--output",
        str(verdict_path),
    ]
    returncode = run(
        command, log_path=args.run_root / "diagnostic_verdict.log"
    )
    if returncode != 0:
        state.update(
            status="blocked",
            terminal_reason="diagnostic_verdict_implementation_failed",
            returncode=returncode,
            finished_at_unix=time.time(),
        )
        write_state(state_path, state)
        return returncode
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    state.update(
        status=(
            "repair_nominated"
            if verdict["verdict"] == "REPAIR_NOMINATED"
            else "terminal_no_go"
        ),
        verdict=verdict["verdict"],
        returncode=0,
        finished_at_unix=time.time(),
    )
    write_state(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
