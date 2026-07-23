"""Sequential S0-S3 queue for the strict VIVID/SPD route."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ARMS = ("ums_prefix4", "ums_spd4x2")


def run(
    command: list[str],
    *,
    log_path: Path,
) -> int:
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
    parser.add_argument("--hard-manifest", required=True, type=Path)
    parser.add_argument("--overfit-ids", required=True, type=Path)
    parser.add_argument("--mimic-image-root", required=True, type=Path)
    parser.add_argument("--teacher-path", required=True, type=Path)
    parser.add_argument("--backbone-weights", required=True, type=Path)
    parser.add_argument("--probe-train-manifest", required=True, type=Path)
    parser.add_argument("--expert-manifest", required=True, type=Path)
    parser.add_argument("--chexpert-root", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    args.run_root.mkdir(parents=True, exist_ok=False)
    state_path = args.run_root / "queue_state.json"
    state: dict[str, object] = {
        "schema_version": 1,
        "artifact": "strict_vivid_spd_queue",
        "started_at_unix": time.time(),
        "status": "running",
        "stage": "S0",
        "arms": [],
    }
    write_state(state_path, state)
    readiness = args.run_root / "readiness.json"
    command = [
        sys.executable,
        str(scripts / "audit_server_readiness.py"),
        "--hard-manifest",
        str(args.hard_manifest),
        "--overfit-ids",
        str(args.overfit_ids),
        "--image-root",
        str(args.mimic_image_root),
        "--teacher-path",
        str(args.teacher_path),
        "--backbone-weights",
        str(args.backbone_weights),
        "--probe-train-manifest",
        str(args.probe_train_manifest),
        "--expert-manifest",
        str(args.expert_manifest),
        "--chexpert-root",
        str(args.chexpert_root),
        "--output",
        str(readiness),
    ]
    returncode = run(command, log_path=args.run_root / "s0.log")
    if returncode != 0:
        state.update(
            status="blocked", stage="S0", returncode=returncode,
            finished_at_unix=time.time()
        )
        write_state(state_path, state)
        return returncode
    for stage, mode in (("S1", "overfit"), ("S2", "pilot")):
        state["stage"] = stage
        write_state(state_path, state)
        for arm in ARMS:
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
                status="passed" if returncode == 0 else "failed",
                returncode=returncode,
                finished_at_unix=time.time(),
            )
            write_state(state_path, state)
            if returncode != 0:
                state.update(
                    status="no_go",
                    stage=stage,
                    terminal_reason=f"{arm}_{stage}_failed",
                    finished_at_unix=time.time(),
                )
                write_state(state_path, state)
                return returncode
    state["stage"] = "S3"
    write_state(state_path, state)
    for arm in ARMS:
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
            status="passed" if returncode == 0 else "failed",
            returncode=returncode,
            finished_at_unix=time.time(),
        )
        write_state(state_path, state)
        if returncode != 0:
            state.update(
                status="blocked",
                terminal_reason=f"{arm}_S3_implementation_failed",
                finished_at_unix=time.time(),
            )
            write_state(state_path, state)
            return returncode
    verdict_path = args.run_root / "s3_verdict.json"
    command = [
        sys.executable,
        str(scripts / "apply_spd_promotion_gate.py"),
        "--prefix-summary",
        str(args.run_root / "s3" / "ums_prefix4" / "summary.json"),
        "--spd-summary",
        str(args.run_root / "s3" / "ums_spd4x2" / "summary.json"),
        "--lock",
        str(args.lock),
        "--output",
        str(verdict_path),
    ]
    returncode = run(command, log_path=args.run_root / "s3_verdict.log")
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    state.update(
        status="strict_pass" if verdict["pass"] else "diagnostic_open",
        stage="S3",
        verdict=verdict["verdict"],
        returncode=returncode,
        finished_at_unix=time.time(),
    )
    write_state(state_path, state)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
