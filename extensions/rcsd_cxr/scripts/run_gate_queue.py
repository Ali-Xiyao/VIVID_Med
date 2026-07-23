"""Sequential, fail-closed experiment queue for a retained Slurm allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time
from typing import Any

import yaml


def load_queue(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("queue requires schema_version: 1")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("queue requires a non-empty tasks list")
    seen: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("id"), str):
            raise ValueError("every task requires a string id")
        if task["id"] in seen:
            raise ValueError(f"duplicate task id: {task['id']}")
        seen.add(task["id"])
    return payload


def query_gpu() -> dict[str, int]:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True).strip().splitlines()
    if len(output) != 1:
        raise RuntimeError(f"expected one visible GPU, got {len(output)}")
    used, total, utilization = (int(item.strip()) for item in output[0].split(","))
    return {
        "used_mib": used,
        "total_mib": total,
        "free_mib": total - used,
        "utilization_percent": utilization,
    }


def wait_for_gpu(task: dict[str, Any], event_log: Path) -> dict[str, int]:
    min_free = int(task.get("min_free_mib", 50000))
    max_util = int(task.get("max_utilization_percent", 85))
    poll = int(task.get("gpu_poll_seconds", 60))
    while True:
        state = query_gpu()
        event = {
            "event": "gpu_guard",
            "task": task["id"],
            "time": time.time(),
            **state,
            "min_free_mib": min_free,
            "max_utilization_percent": max_util,
        }
        with event_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        if state["free_mib"] >= min_free and state["utilization_percent"] <= max_util:
            return state
        time.sleep(poll)


def command_digest(command: str) -> str:
    return hashlib.sha256(command.encode("utf-8")).hexdigest()


def run_task(
    task: dict[str, Any],
    *,
    project_root: Path,
    state_dir: Path,
    event_log: Path,
) -> int:
    task_id = task["id"]
    command = task.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError(f"enabled task {task_id} requires a command")
    requires = task.get("requires", [])
    if not isinstance(requires, list) or not all(isinstance(x, str) for x in requires):
        raise ValueError(f"task {task_id} requires must be a list of task ids")
    missing = [item for item in requires if not (state_dir / f"{item}.done.json").is_file()]
    if missing:
        raise RuntimeError(f"task {task_id} is locked; missing markers: {missing}")

    gpu_state = None
    if task.get("resource") == "gpu":
        gpu_state = wait_for_gpu(task, event_log)

    log_path = state_dir / f"{task_id}.log"
    started = time.time()
    start_record = {
        "event": "task_start",
        "task": task_id,
        "time": started,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "resource": task.get("resource", "cpu"),
        "command_sha256": command_digest(command),
        "gpu_state": gpu_state,
    }
    with event_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(start_record, sort_keys=True) + "\n")

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"COMMAND: {command}\n")
        log.flush()
        process = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=project_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return_code = process.wait()

    finished = time.time()
    record = {
        "schema_version": 1,
        "task": task_id,
        "status": "pass" if return_code == 0 else "fail",
        "return_code": return_code,
        "started_unix": started,
        "finished_unix": finished,
        "elapsed_seconds": finished - started,
        "hostname": socket.gethostname(),
        "command": command,
        "command_sha256": command_digest(command),
        "log": str(log_path.resolve()),
    }
    suffix = "done" if return_code == 0 else "failed"
    marker = state_dir / f"{task_id}.{suffix}.json"
    marker.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with event_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "task_finish", **record}, sort_keys=True) + "\n")
    return return_code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--from-task")
    args = parser.parse_args()

    payload = load_queue(args.queue)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    event_log = args.state_dir / "events.jsonl"
    queue_hash = hashlib.sha256(args.queue.read_bytes()).hexdigest()
    (args.state_dir / "queue_identity.json").write_text(
        json.dumps(
            {"queue": str(args.queue.resolve()), "sha256": queue_hash},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    started = args.from_task is None
    for task in payload["tasks"]:
        if args.from_task and task["id"] == args.from_task:
            started = True
        if not started:
            continue
        if not bool(task.get("enabled", False)):
            with event_log.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "event": "task_locked",
                            "task": task["id"],
                            "reason": task.get("locked_reason", "disabled by protocol"),
                            "time": time.time(),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            continue
        done = args.state_dir / f"{task['id']}.done.json"
        if done.is_file():
            continue
        return_code = run_task(
            task,
            project_root=args.project_root.resolve(),
            state_dir=args.state_dir.resolve(),
            event_log=event_log,
        )
        if return_code != 0:
            print(f"STOP: {task['id']} failed; downstream queue remains locked")
            return return_code
    print("QUEUE_COMPLETE_OR_LOCKED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

