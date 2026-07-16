"""Validate VinDr-CXR and run the five main-external rows on local or remote hosts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIRNAME = "vindr-cxr-an-open-dataset-of-chest-x-rays-with-radiologists-annotations-1.0.0"
MANIFEST = ROOT / "data/dataset/processed/vindr_cxr_external_test_ums.jsonl"
LOG_DIR = ROOT / "outputs/logs"
OUTPUT_ROOT = ROOT / "outputs/vsl_cxr_phase6_external"
PRIMARY_LABELS = [
    "No Finding",
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
]
RUNS = [
    {
        "name": "raw",
        "config": ROOT / "configs/qwen3vl_instruction/vsl_cxr/phase1_baselines/b0_raw_vision_lp.yaml",
        "probe_run": "vsl_cxr_b0_raw_vision_lp",
        "checkpoint_run": None,
    },
    {
        "name": "sameq",
        "config": ROOT / "configs/qwen3vl_instruction/vsl_cxr/phase6_external/lp_sameq.yaml",
        "probe_run": "vsl_cxr_p6_lp_sameq",
        "checkpoint_run": "vsl_cxr_b3_sameq",
    },
    {
        "name": "vsl_core",
        "config": ROOT / "configs/qwen3vl_instruction/vsl_cxr/phase6_external/lp_vsl_core.yaml",
        "probe_run": "vsl_cxr_p6_lp_vsl_core",
        "checkpoint_run": "vsl_cxr_b5_sameq_k4",
    },
    {
        "name": "vsl_ceq_backbone",
        "config": ROOT / "configs/qwen3vl_instruction/vsl_cxr/phase6_external/lp_vsl_ceq_backbone.yaml",
        "probe_run": "vsl_cxr_p6_lp_vsl_ceq_backbone",
        "checkpoint_run": "vsl_cxr_b3_sameq",
    },
    {
        "name": "vsl_full",
        "config": ROOT / "configs/qwen3vl_instruction/vsl_cxr/phase6_external/lp_vsl_full.yaml",
        "probe_run": "vsl_cxr_p6_lp_vsl_full",
        "checkpoint_run": "vsl_cxr_p5_vsl_full",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-hours", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("VSL_CXR_DATA_ROOT", ROOT / "data/dataset")),
        help="Directory containing the extracted VinDr-CXR dataset directory.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(os.environ.get("VSL_CXR_MODEL_PATH", ROOT / "model/qwen3-vl-2b-thinking-new")),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(os.environ.get("VSL_CXR_ARTIFACT_ROOT", ROOT / "outputs/qwen3vl_external_sources")),
        help="Root containing the five LP probe runs and three source-backbone checkpoints.",
    )
    return parser.parse_args()


def run_checked(command: list[str]) -> None:
    print(json.dumps({"event": "run", "command": command}, ensure_ascii=False), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def wait_for_extraction(data_root: Path, poll_seconds: int, max_wait_hours: float) -> None:
    marker = data_root / DATASET_DIRNAME / "_extraction_complete.json"
    deadline = time.time() + max_wait_hours * 3600
    while not marker.is_file():
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for {marker}")
        progress_path = LOG_DIR / "vindr_cxr_extract_progress.json"
        progress = {}
        if progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "event": "waiting_for_extraction",
                    "completed_files": progress.get("completed_files"),
                    "total_files": progress.get("total_files"),
                    "completed_bytes": progress.get("completed_bytes"),
                    "total_bytes": progress.get("total_uncompressed_bytes"),
                }
            ),
            flush=True,
        )
        time.sleep(poll_seconds)


def launch(
    spec: dict[str, object],
    gpu: int,
    batch_size: int,
    data_root: Path,
    model_path: Path,
    artifact_root: Path,
) -> tuple[subprocess.Popen[bytes], object, object]:
    name = str(spec["name"])
    output_dir = OUTPUT_ROOT / f"vindr_{name}"
    if (output_dir / "transfer_metrics.json").exists():
        print(json.dumps({"event": "skip_complete", "name": name}), flush=True)
        return None, None, None  # type: ignore[return-value]
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_handle = (LOG_DIR / f"vindr_external_{name}.out.log").open("wb")
    stderr_handle = (LOG_DIR / f"vindr_external_{name}.err.log").open("wb")
    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/evaluate_qwen3vl_lp_transfer.py"),
        "--lp-config",
        str(spec["config"]),
        "--probe-dir",
        str(artifact_root / str(spec["probe_run"])),
        "--model-path",
        str(model_path),
        "--val-ums-path",
        str(MANIFEST),
        "--data-root",
        str(data_root),
        "--output-dir",
        str(output_dir),
        "--max-samples",
        "0",
        "--batch-size",
        str(batch_size),
        "--device",
        "cuda:0",
        "--verify-images",
        "--primary-labels",
        *PRIMARY_LABELS,
    ]
    checkpoint_run = spec.get("checkpoint_run")
    if checkpoint_run:
        command.extend(
            [
                "--vision-checkpoint",
                str(artifact_root / str(checkpoint_run) / "checkpoints/final.pt"),
            ]
        )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout_handle, stderr=stderr_handle)
    print(json.dumps({"event": "launched", "name": name, "gpu": gpu, "pid": process.pid}), flush=True)
    return process, stdout_handle, stderr_handle


def main() -> None:
    args = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    wait_for_extraction(args.data_root, args.poll_seconds, args.max_wait_hours)
    dataset_root = args.data_root / DATASET_DIRNAME
    run_checked(
        [
            sys.executable,
            "scripts/audit_vindr_cxr_integrity.py",
            "--dataset-root",
            str(dataset_root),
        ]
    )
    run_checked(
        [
            sys.executable,
            "scripts/prepare_vindr_cxr.py",
            "--dataset-root",
            str(dataset_root),
        ]
    )

    pending = list(RUNS)
    active: dict[int, tuple[dict[str, object], subprocess.Popen[bytes], object, object]] = {}
    while pending or active:
        for gpu in (0, 1):
            if gpu in active or not pending:
                continue
            spec = pending.pop(0)
            process, stdout_handle, stderr_handle = launch(
                spec,
                gpu,
                args.batch_size,
                args.data_root,
                args.model_path,
                args.artifact_root,
            )
            if process is not None:
                active[gpu] = (spec, process, stdout_handle, stderr_handle)
        if not active:
            continue
        time.sleep(10)
        for gpu, (spec, process, stdout_handle, stderr_handle) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            stdout_handle.close()
            stderr_handle.close()
            name = str(spec["name"])
            print(json.dumps({"event": "finished", "name": name, "gpu": gpu, "exit_code": code}), flush=True)
            del active[gpu]
            if code != 0:
                raise subprocess.CalledProcessError(code, ["VinDr external run", name])

    run_checked([sys.executable, "scripts/build_external_results_table.py"])
    run_checked([sys.executable, "scripts/audit_vsl_cxr_readiness.py"])
    print(json.dumps({"event": "suite_complete", "runs": [spec["name"] for spec in RUNS]}), flush=True)


if __name__ == "__main__":
    main()
