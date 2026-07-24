"""Sequential server queue for the locked VIVID-GDS fresh replication."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ARMS = ("A0_direct", "A2_ums", "A3_gds")
SEEDS = (0, 1, 2)


def write_state(path: Path, state: dict[str, object]) -> None:
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def run_record(
    state: dict[str, object],
    state_path: Path,
    *,
    stage: str,
    identity: str,
    command: list[str],
    log_path: Path,
) -> int:
    row = {
        "stage": stage,
        "identity": identity,
        "status": "running",
        "command": command,
        "started_at_unix": time.time(),
    }
    state["runs"].append(row)
    state["stage"] = stage
    write_state(state_path, state)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    row.update(
        status="passed" if process.returncode == 0 else "failed",
        returncode=process.returncode,
        finished_at_unix=time.time(),
    )
    write_state(state_path, state)
    return process.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--split-audit", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--hard-manifest", required=True, type=Path)
    parser.add_argument("--mimic-image-root", required=True, type=Path)
    parser.add_argument("--chexpert-image-root", required=True, type=Path)
    parser.add_argument("--teacher-path", required=True, type=Path)
    parser.add_argument("--backbone-weights", required=True, type=Path)
    parser.add_argument("--training-script", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    scripts = Path(__file__).resolve().parent
    args.run_root.mkdir(parents=True, exist_ok=False)
    state_path = args.run_root / "queue_state.json"
    state: dict[str, object] = {
        "schema_version": 1,
        "artifact": "vivid_gds_fresh_replication_queue",
        "status": "running",
        "stage": "R0",
        "started_at_unix": time.time(),
        "runs": [],
    }
    write_state(state_path, state)

    readiness = args.run_root / "readiness.json"
    command = [
        sys.executable,
        str(scripts / "audit_replication_readiness.py"),
        "--lock",
        str(args.lock),
        "--split-audit",
        str(args.split_audit),
        "--split-dir",
        str(args.split_dir),
        "--hard-manifest",
        str(args.hard_manifest),
        "--mimic-image-root",
        str(args.mimic_image_root),
        "--chexpert-image-root",
        str(args.chexpert_image_root),
        "--teacher-path",
        str(args.teacher_path),
        "--backbone-weights",
        str(args.backbone_weights),
        "--training-script",
        str(args.training_script),
        "--probe-script",
        str(scripts / "train_fresh_chexpert_probe.py"),
        "--gate-script",
        str(scripts / "apply_fresh_replication_gate.py"),
        "--output",
        str(readiness),
    ]
    returncode = run_record(
        state,
        state_path,
        stage="R0",
        identity="readiness",
        command=command,
        log_path=args.run_root / "r0.log",
    )
    if returncode != 0:
        state.update(
            status="blocked",
            terminal_reason="readiness_failed",
            returncode=returncode,
            finished_at_unix=time.time(),
        )
        write_state(state_path, state)
        return returncode

    for seed in SEEDS:
        for arm in ARMS:
            output = args.run_root / "training" / f"seed{seed}" / arm
            command = [
                sys.executable,
                str(args.training_script),
                "--arm",
                arm,
                "--mode",
                "pilot",
                "--hard-manifest",
                str(args.hard_manifest),
                "--image-root",
                str(args.mimic_image_root),
                "--backbone-weights",
                str(args.backbone_weights),
                "--output-dir",
                str(output),
                "--device",
                args.device,
                "--seed",
                str(seed),
                "--max-steps",
                "3000",
            ]
            if arm != "A0_direct":
                command.extend(["--teacher-path", str(args.teacher_path)])
            identity = f"seed{seed}_{arm}"
            returncode = run_record(
                state,
                state_path,
                stage="R1",
                identity=identity,
                command=command,
                log_path=args.run_root / "training" / f"{identity}.log",
            )
            if returncode != 0:
                state.update(
                    status="terminal_no_go",
                    terminal_reason=f"{identity}_pilot_failed",
                    returncode=returncode,
                    finished_at_unix=time.time(),
                )
                write_state(state_path, state)
                return returncode

    for seed in SEEDS:
        for arm in ARMS:
            output = args.run_root / "probes" / f"seed{seed}" / arm
            command = [
                sys.executable,
                str(scripts / "train_fresh_chexpert_probe.py"),
                "--arm",
                arm,
                "--vision-checkpoint",
                str(
                    args.run_root
                    / "training"
                    / f"seed{seed}"
                    / arm
                    / "best.pt"
                ),
                "--train-manifest",
                str(args.split_dir / "probe_train.csv"),
                "--validation-manifest",
                str(args.split_dir / "probe_validation.csv"),
                "--development-manifest",
                str(args.split_dir / "fresh_development.csv"),
                "--image-root",
                str(args.chexpert_image_root),
                "--output-dir",
                str(output),
                "--device",
                args.device,
                "--seed",
                str(seed),
            ]
            identity = f"seed{seed}_{arm}"
            returncode = run_record(
                state,
                state_path,
                stage="R2",
                identity=identity,
                command=command,
                log_path=args.run_root / "probes" / f"{identity}.log",
            )
            if returncode != 0:
                state.update(
                    status="blocked",
                    terminal_reason=f"{identity}_probe_failed",
                    returncode=returncode,
                    finished_at_unix=time.time(),
                )
                write_state(state_path, state)
                return returncode

    verdict_path = args.run_root / "fresh_replication_verdict.json"
    command = [
        sys.executable,
        str(scripts / "apply_fresh_replication_gate.py"),
        "--probe-root",
        str(args.run_root / "probes"),
        "--lock",
        str(args.lock),
        "--output",
        str(verdict_path),
    ]
    returncode = run_record(
        state,
        state_path,
        stage="R3",
        identity="frozen_gate",
        command=command,
        log_path=args.run_root / "r3.log",
    )
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    state.update(
        status=(
            "replication_pass"
            if verdict["verdict"] == "REPLICATION_PASS"
            else "terminal_no_go"
        ),
        verdict=verdict["verdict"],
        returncode=returncode,
        finished_at_unix=time.time(),
    )
    write_state(state_path, state)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())

