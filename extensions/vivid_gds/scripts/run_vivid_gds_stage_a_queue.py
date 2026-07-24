"""Sequential G0-G4 server queue for VIVID-GDS Stage A."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


NEW_ARMS = ("A0_direct", "A1_freetext", "A3_gds")


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


def record_run(
    state: dict[str, object],
    state_path: Path,
    *,
    stage: str,
    arm: str,
    command: list[str],
    log_path: Path,
) -> int:
    row = {
        "stage": stage,
        "arm": arm,
        "status": "running",
        "command": command,
        "started_at_unix": time.time(),
    }
    state["runs"].append(row)
    write_state(state_path, state)
    returncode = run(command, log_path=log_path)
    row.update(
        status="passed" if returncode == 0 else "failed",
        returncode=returncode,
        finished_at_unix=time.time(),
    )
    write_state(state_path, state)
    return returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--hard-manifest", required=True, type=Path)
    parser.add_argument("--overfit-ids", required=True, type=Path)
    parser.add_argument("--mimic-image-root", required=True, type=Path)
    parser.add_argument("--teacher-path", required=True, type=Path)
    parser.add_argument("--backbone-weights", required=True, type=Path)
    parser.add_argument("--probe-script", required=True, type=Path)
    parser.add_argument("--probe-train-manifest", required=True, type=Path)
    parser.add_argument("--expert-manifest", required=True, type=Path)
    parser.add_argument("--chexpert-root", required=True, type=Path)
    parser.add_argument("--a2-training-summary", required=True, type=Path)
    parser.add_argument("--a2-checkpoint", required=True, type=Path)
    parser.add_argument("--a2-probe-summary", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    args.run_root.mkdir(parents=True, exist_ok=False)
    state_path = args.run_root / "queue_state.json"
    state: dict[str, object] = {
        "schema_version": 1,
        "artifact": "vivid_gds_stage_a_queue",
        "started_at_unix": time.time(),
        "status": "running",
        "stage": "G0",
        "runs": [],
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
        "--a2-summary",
        str(args.a2_training_summary),
        "--a2-checkpoint",
        str(args.a2_checkpoint),
        "--output",
        str(readiness),
    ]
    returncode = record_run(
        state,
        state_path,
        stage="G0",
        arm="all",
        command=command,
        log_path=args.run_root / "g0.log",
    )
    if returncode != 0:
        state.update(
            status="blocked",
            stage="G0",
            terminal_reason="readiness_failed",
            returncode=returncode,
            finished_at_unix=time.time(),
        )
        write_state(state_path, state)
        return returncode

    for stage, mode in (("G1", "overfit"), ("G2", "pilot")):
        state["stage"] = stage
        write_state(state_path, state)
        for arm in NEW_ARMS:
            output = args.run_root / stage.lower() / arm
            command = [
                sys.executable,
                str(scripts / "train_vivid_gds_stage_a.py"),
                "--arm",
                arm,
                "--mode",
                mode,
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
            ]
            if arm != "A0_direct":
                command.extend(["--teacher-path", str(args.teacher_path)])
            if mode == "overfit":
                command.extend(["--overfit-ids", str(args.overfit_ids)])
            returncode = record_run(
                state,
                state_path,
                stage=stage,
                arm=arm,
                command=command,
                log_path=args.run_root / stage.lower() / f"{arm}.log",
            )
            if returncode != 0:
                state.update(
                    status="gate_failed",
                    stage=stage,
                    terminal_reason=f"{arm}_{stage}_failed",
                    returncode=returncode,
                    finished_at_unix=time.time(),
                )
                write_state(state_path, state)
                return returncode

    state["stage"] = "G3"
    write_state(state_path, state)
    probe_summaries = {"A2_ums": args.a2_probe_summary}
    for arm in NEW_ARMS:
        output = args.run_root / "g3" / arm
        command = [
            sys.executable,
            str(args.probe_script),
            "--arm",
            arm,
            "--vision-checkpoint",
            str(args.run_root / "g2" / arm / "best.pt"),
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
        returncode = record_run(
            state,
            state_path,
            stage="G3",
            arm=arm,
            command=command,
            log_path=args.run_root / "g3" / f"{arm}.log",
        )
        if returncode != 0:
            state.update(
                status="blocked",
                stage="G3",
                terminal_reason=f"{arm}_probe_failed",
                returncode=returncode,
                finished_at_unix=time.time(),
            )
            write_state(state_path, state)
            return returncode
        probe_summaries[arm] = output / "summary.json"

    state["stage"] = "G4"
    write_state(state_path, state)
    verdict_path = args.run_root / "stage_a_verdict.json"
    command = [
        sys.executable,
        str(scripts / "apply_stage_a_gate.py"),
        "--A0-direct",
        str(probe_summaries["A0_direct"]),
        "--A1-freetext",
        str(probe_summaries["A1_freetext"]),
        "--A2-ums",
        str(probe_summaries["A2_ums"]),
        "--A3-gds",
        str(probe_summaries["A3_gds"]),
        "--lock",
        str(args.lock),
        "--output",
        str(verdict_path),
    ]
    returncode = record_run(
        state,
        state_path,
        stage="G4",
        arm="all",
        command=command,
        log_path=args.run_root / "g4.log",
    )
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    state.update(
        status="stage_a_pass" if verdict["pass"] else "stage_a_no_go",
        stage="G4",
        verdict=verdict["verdict"],
        returncode=returncode,
        finished_at_unix=time.time(),
    )
    write_state(state_path, state)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
