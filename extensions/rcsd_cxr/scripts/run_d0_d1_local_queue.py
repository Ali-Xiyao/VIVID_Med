"""Fail-closed sequential queue for the authorized D0/D1 execution target."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def gpu_compute_processes(index: int) -> list[dict[str, str]]:
    uuid_result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    gpu_uuid = uuid_result.stdout.strip()
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) == 3 and parts[1] == gpu_uuid:
            rows.append(
                {"pid": parts[0], "gpu_uuid": parts[1], "used_mib": parts[2]}
            )
    return rows


def write_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("overfit", "pilot"), required=True)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--hard-manifest", required=True, type=Path)
    parser.add_argument("--reliability-manifest", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--teacher-path", required=True, type=Path)
    parser.add_argument("--backbone-weights", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--overfit-ids", type=Path)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    expected = (
        "OVERFIT_AUTHORIZED"
        if args.mode == "overfit"
        else "PILOT_AUTHORIZED"
    )
    if lock.get("status") != expected:
        raise PermissionError(f"lock status must be {expected}")
    if lock.get("execution_authorized") is not True:
        raise PermissionError("machine lock does not authorize execution")
    if lock.get("training_jobs_allowed") != 2:
        raise PermissionError("queue requires exactly two authorized arms")
    target = lock["execution_target"]
    kind = target["kind"]
    if kind == "local_workstation":
        if target["server_allowed"] or target["slurm_allowed"]:
            raise PermissionError("local target cannot enable remote execution")
    elif kind == "sues_hpc_slurm":
        if target["server_allowed"] is not True or target["slurm_allowed"] is not True:
            raise PermissionError("SUES target requires server and Slurm authorization")
        expected_job = str(target["allocation_id"])
        actual_job = os.environ.get("SLURM_JOB_ID", "").split(".", 1)[0]
        if actual_job != expected_job:
            raise PermissionError(
                f"queue must run inside Slurm allocation {expected_job}; got {actual_job!r}"
            )
        if socket.gethostname().split(".", 1)[0] != target["node"]:
            raise PermissionError(
                f"queue must run on {target['node']}; got {socket.gethostname()}"
            )
    else:
        raise PermissionError(f"unsupported execution target: {kind}")
    gpu = int(target["preferred_gpu"])
    processes = gpu_compute_processes(gpu)
    if processes:
        raise RuntimeError(f"GPU {gpu} has active compute processes: {processes}")
    if args.mode == "overfit" and args.overfit_ids is None:
        raise ValueError("overfit queue requires --overfit-ids")
    state_path = args.output_root / "queue_state.json"
    state: dict[str, object] = {
        "schema_version": 1,
        "artifact": "d0_d1_local_queue",
        "mode": args.mode,
        "execution_target": kind,
        "allocation_id": target.get("allocation_id"),
        "node": socket.gethostname(),
        "gpu": gpu,
        "started_at_unix": time.time(),
        "arms": [],
        "pass": False,
    }
    write_state(state_path, state)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    trainer = Path(__file__).with_name("train_d0_d1_token.py")
    for variant in ("d0", "d1"):
        arm_output = args.output_root / variant
        if arm_output.exists():
            raise FileExistsError(f"refusing to overwrite {arm_output}")
        command = [
            sys.executable,
            str(trainer),
            "--variant",
            variant,
            "--mode",
            args.mode,
            "--hard-manifest",
            str(args.hard_manifest),
            "--reliability-manifest",
            str(args.reliability_manifest),
            "--image-root",
            str(args.image_root),
            "--teacher-path",
            str(args.teacher_path),
            "--backbone-weights",
            str(args.backbone_weights),
            "--output-dir",
            str(arm_output),
            "--device",
            "cuda:0",
        ]
        if args.overfit_ids is not None:
            command.extend(["--overfit-ids", str(args.overfit_ids)])
        arm = {
            "variant": variant,
            "status": "running",
            "command": command,
            "started_at_unix": time.time(),
        }
        state["arms"].append(arm)
        write_state(state_path, state)
        log_path = args.output_root / f"{variant}.log"
        args.output_root.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        arm["returncode"] = completed.returncode
        arm["finished_at_unix"] = time.time()
        arm["status"] = "passed" if completed.returncode == 0 else "failed"
        write_state(state_path, state)
        if completed.returncode != 0:
            state["terminal_reason"] = f"{variant}_{args.mode}_failed"
            write_state(state_path, state)
            return completed.returncode
    state["pass"] = True
    state["finished_at_unix"] = time.time()
    write_state(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
