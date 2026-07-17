"""Run the frozen C3 local timing/replay gate on VinDr-train design rows only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

# PyTorch requires this before the CUDA context is created when deterministic
# algorithms are enforced for the cuBLAS matmuls used by Qwen3.5 vision.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bives_cxr.backbones import (  # noqa: E402
    Qwen35VisionAdapter,
    load_qwen35_visual_and_processor,
)
from bives_cxr.dicom import DICOM_PREPROCESS_VERSION, load_cxr_dicom  # noqa: E402
from bives_cxr.polarity_runtime import (  # noqa: E402
    extract_qwen35_patches,
    load_locked_polarity_checkpoint,
    score_statements,
)
from bives_cxr.qwen35_preprocessing import (  # noqa: E402
    QWEN35_IMAGE_PREPROCESS_VERSION,
)


FORMAT_VERSION = "bives_connected_control_c3_timing_replay_v1"
EXPECTED_MANIFEST_SHA256 = (
    "bd84cd7ca5384afbcb6228c49331b028a9641dd3dd9011157c2cec75b1f6514f"
)
EXPECTED_DATA_LOCK_SHA256 = (
    "4251027b3069b21fb6fb5acd6bc02bf003206fbcfffb6d045abd2289ea2ac409"
)
EXPECTED_GEOMETRY_ROWS_SHA256 = (
    "b94b77bc528033ea44681fce33f868b87c2806d19450a66ff023628fc2e039d9"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "09c2f77313027ca313f4b03c5553f90d3d7d57436e960888466d2712e9705480"
)
EXPECTED_CACHE_LOCK_SHA256 = (
    "503ead96a0de948b56361b8097bd14cc1ba0942878b56cb3aa0ba2e39d3ec0f2"
)
EXPECTED_CONFIG_SHA256 = (
    "248d57d9a62e77acf36c6c0809428e17c1a3b37bb00741383b94db51ab4d395f"
)
EXPECTED_MODEL_SNAPSHOT_SHA256 = (
    "6b57c58ceb3c97d4199d4e8e32b54f6ff11d0cf3a8bde3a4e940d5775ae12120"
)
FINDINGS = ("consolidation", "pleural_effusion")
SAMPLES_PER_FINDING = 8
IMAGE_SIZE = 448
REPLAY_ATOL = 1e-6
C4_VARIANTS_PER_ROW = 5
C4_COMPUTE_CAP_HOURS = 4.0
C4_TIMING_MULTIPLIER = 1.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("local_runs/bives_cxr/vindr_rescue_dev/vindr_train_rescue_dev.jsonl"),
    )
    parser.add_argument(
        "--data-lock",
        type=Path,
        default=Path("local_runs/bives_cxr/vindr_rescue_dev/vindr_train_rescue_dev_lock.json"),
    )
    parser.add_argument(
        "--geometry-rows",
        type=Path,
        default=Path(
            "local_runs/bives_cxr/vindr_connected_control_geometry/connected_geometry_rows.jsonl"
        ),
    )
    parser.add_argument(
        "--geometry-lock",
        type=Path,
        default=Path(
            "local_runs/bives_cxr/vindr_connected_control_geometry/connected_geometry_lock.json"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("local_runs/bives_cxr/qwen35_2b_sc_b2_sparse_k16_seed17/best.pt"),
    )
    parser.add_argument(
        "--training-cache-lock",
        type=Path,
        default=Path("local_runs/bives_cxr/qwen35_2b_weak_sc_cache/cache_lock.json"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/bives_cxr/qwen35_2b_sc_b2_sparse_k16.yaml"),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("H:/Xiyao_Wang/001_models/Qwen3.5-2B"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("local_runs/bives_cxr/connected_control_c3_timing_replay"),
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def snapshot_model_files(model_root: Path) -> str:
    names = {"config.json", "model.safetensors.index.json"}
    names.update(path.name for path in model_root.glob("*.safetensors"))
    files = {
        name: file_sha256(model_root / name)
        for name in sorted(names)
        if (model_root / name).is_file()
    }
    return canonical_sha256(files)


def select_c3_rows(
    manifest_rows: list[dict[str, Any]],
    geometry_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select 8+8 unique images by the frozen within-finding sample-id order."""

    manifest_by_sample = {str(row["sample_id"]): row for row in manifest_rows}
    feasible = [
        row
        for row in geometry_rows
        if bool(row.get("feasible"))
        and row.get("rescue_split") == "protocol_design"
        and row.get("source_split") == "train"
    ]
    selected: list[dict[str, Any]] = []
    selected_units: set[str] = set()
    for finding in FINDINGS:
        candidates = sorted(
            (
                row
                for row in feasible
                if str(row["canonical_statement_id"]) == finding
            ),
            key=lambda row: str(row["sample_id"]),
        )
        finding_rows = []
        for geometry_row in candidates:
            sample_id = str(geometry_row["sample_id"])
            source = manifest_by_sample.get(sample_id)
            if source is None:
                raise ValueError(f"C3 geometry row is absent from its manifest: {sample_id}")
            unit_id = str(source["unit_id"])
            if unit_id in selected_units:
                continue
            if source.get("rescue_split") != "protocol_design":
                raise ValueError("C3 selected a non-design manifest row")
            image_path = Path(str(source["image_path"]))
            if source.get("source_split") != "train" or "test" in {
                part.lower() for part in image_path.parts
            }:
                raise ValueError(f"VinDr-test path is forbidden in C3: {image_path}")
            finding_rows.append(source)
            selected_units.add(unit_id)
            if len(finding_rows) == SAMPLES_PER_FINDING:
                break
        if len(finding_rows) != SAMPLES_PER_FINDING:
            raise ValueError(f"C3 cannot select {SAMPLES_PER_FINDING} unique {finding} images")
        selected.extend(finding_rows)
    if len(selected) != len({str(row["unit_id"]) for row in selected}):
        raise AssertionError("C3 selection must contain 16 unique images")
    return selected


def estimate_c4_hours(
    pass_seconds: list[float],
    *,
    sample_count: int,
    eligible_rows: int,
    geometry_seconds: float,
) -> float:
    if sample_count <= 0 or eligible_rows <= 0 or not pass_seconds:
        raise ValueError("C3 estimate inputs must be positive")
    per_forward = max(float(value) for value in pass_seconds) / sample_count
    visual_seconds = (
        per_forward
        * eligible_rows
        * C4_VARIANTS_PER_ROW
        * C4_TIMING_MULTIPLIER
    )
    return float((visual_seconds + geometry_seconds) / 3600.0)


def score_one(
    image,
    row: dict[str, Any],
    *,
    polarity_model,
    statement_to_index: dict[str, int],
    adapter,
    visual,
    processor,
    device: torch.device,
) -> dict[str, Any]:
    finding = str(row["canonical_statement_id"])
    extracted = extract_qwen35_patches(
        image,
        str(row["statement_text"]),
        adapter=adapter,
        visual=visual,
        processor=processor,
        device=device,
        image_size=IMAGE_SIZE,
    )
    output = score_statements(
        polarity_model,
        extracted["patch_tokens"],
        extracted["valid_mask"],
        [statement_to_index[finding]],
    )
    return {
        "support_probability": float(output["support_probability"][0].cpu()),
        "signed_evidence": float(output["signed_evidence"][0].cpu()),
        "evidence_pos": float(output["evidence_pos"][0].cpu()),
        "evidence_neg": float(output["evidence_neg"][0].cpu()),
        "topk_indices": torch.where(output["gate"][0].detach().cpu() > 0.5)[0].tolist(),
        "grid_hw": list(extracted["grid_hw"]),
    }


def run_timed_pass(
    decoded: list[tuple[dict[str, Any], Any, dict[str, Any]]],
    **score_kwargs: Any,
) -> tuple[list[dict[str, Any]], float]:
    device = score_kwargs["device"]
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    results = []
    for row, image, dicom_record in decoded:
        item_start = time.perf_counter()
        score = score_one(image, row, **score_kwargs)
        torch.cuda.synchronize(device)
        results.append(
            {
                "sample_id": str(row["sample_id"]),
                "unit_id": str(row["unit_id"]),
                "canonical_statement_id": str(row["canonical_statement_id"]),
                "dicom_rgb_sha256": dicom_record["rgb_sha256"],
                "score": score,
                "elapsed_seconds": float(time.perf_counter() - item_start),
            }
        )
    torch.cuda.synchronize(device)
    return results, float(time.perf_counter() - start)


def main() -> None:
    args = parse_args()
    expected_hashes = {
        args.manifest: EXPECTED_MANIFEST_SHA256,
        args.data_lock: EXPECTED_DATA_LOCK_SHA256,
        args.geometry_rows: EXPECTED_GEOMETRY_ROWS_SHA256,
        args.checkpoint: EXPECTED_CHECKPOINT_SHA256,
        args.training_cache_lock: EXPECTED_CACHE_LOCK_SHA256,
        args.config: EXPECTED_CONFIG_SHA256,
    }
    observed_hashes = {str(path): file_sha256(path) for path in expected_hashes}
    for path, expected in expected_hashes.items():
        if observed_hashes[str(path)] != expected:
            raise ValueError(f"C3 frozen identity mismatch: {path}")

    data_lock = json.loads(args.data_lock.read_text(encoding="utf-8"))
    geometry_lock = json.loads(args.geometry_lock.read_text(encoding="utf-8"))
    cache_lock = json.loads(args.training_cache_lock.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if data_lock.get("status") != "pass" or data_lock.get("source_split") != "train_only":
        raise ValueError("C3 requires the passing train-only R001 lock")
    if geometry_lock.get("status") != "pass" or geometry_lock.get("scores_accessed"):
        raise ValueError("C3 requires the score-blind passing C2 geometry lock")
    if geometry_lock.get("geometry_rows_sha256") != EXPECTED_GEOMETRY_ROWS_SHA256:
        raise ValueError("C3 geometry rows do not match the C2 lock")
    if config.get("variant") != "B2_sparse_exact_k" or int(config.get("topk", 0)) != 16:
        raise ValueError("C3 config is not the frozen B2 exact-K=16 config")
    if str(config.get("model", {}).get("family")) != "Qwen3.5":
        raise ValueError("C3 config is not Qwen3.5-only")
    if cache_lock.get("status") != "complete":
        raise ValueError("C3 training cache lock is incomplete")
    if snapshot_model_files(args.model_path) != EXPECTED_MODEL_SNAPSHOT_SHA256:
        raise ValueError("C3 local Qwen3.5-2B snapshot differs from the frozen cache")

    manifest_rows = read_jsonl(args.manifest)
    geometry_rows = read_jsonl(args.geometry_rows)
    selected = select_c3_rows(manifest_rows, geometry_rows)
    selection_counts = Counter(str(row["canonical_statement_id"]) for row in selected)
    if selection_counts != Counter({finding: SAMPLES_PER_FINDING for finding in FINDINGS}):
        raise AssertionError("C3 selection must be balanced 8+8")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("C3 requires an available local CUDA device")
    torch.manual_seed(20260718)
    torch.cuda.manual_seed_all(20260718)
    torch.use_deterministic_algorithms(True)

    polarity_model, statement_to_index, checkpoint = load_locked_polarity_checkpoint(
        args.checkpoint, device
    )
    if checkpoint.get("variant") != "B2_sparse_exact_k" or int(checkpoint.get("step", -1)) != 450:
        raise ValueError("C3 checkpoint is not frozen B2 step 450")
    if checkpoint.get("cache_lock_sha256") != EXPECTED_CACHE_LOCK_SHA256:
        raise ValueError("C3 checkpoint/cache binding failed")
    if set(statement_to_index) != set(FINDINGS):
        raise ValueError("C3 checkpoint statement vocabulary changed")

    visual, processor, qwen_config = load_qwen35_visual_and_processor(
        args.model_path,
        dtype=args.dtype,
        attention_implementation="eager",
    )
    visual = visual.to(device).eval()
    adapter = Qwen35VisionAdapter(
        visual,
        spatial_merge_size=int(qwen_config["vision_config"]["spatial_merge_size"]),
    ).to(device).eval()

    decoded = []
    for row in selected:
        image, record = load_cxr_dicom(Path(str(row["image_path"])))
        decoded.append((row, image, record.to_dict()))
    score_kwargs = {
        "polarity_model": polarity_model,
        "statement_to_index": statement_to_index,
        "adapter": adapter,
        "visual": visual,
        "processor": processor,
        "device": device,
    }
    # One unmeasured warmup stabilizes lazy CUDA/runtime initialization.
    score_one(decoded[0][1], decoded[0][0], **score_kwargs)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    first, first_seconds = run_timed_pass(decoded, **score_kwargs)
    second, second_seconds = run_timed_pass(decoded, **score_kwargs)

    rows = []
    max_abs_diff = 0.0
    topk_mismatch_count = 0
    for left, right in zip(first, second, strict=True):
        if left["sample_id"] != right["sample_id"]:
            raise AssertionError("C3 replay ordering changed")
        numeric_diffs = {
            key: abs(float(left["score"][key]) - float(right["score"][key]))
            for key in ("support_probability", "signed_evidence", "evidence_pos", "evidence_neg")
        }
        row_max = max(numeric_diffs.values())
        max_abs_diff = max(max_abs_diff, row_max)
        topk_equal = left["score"]["topk_indices"] == right["score"]["topk_indices"]
        topk_mismatch_count += int(not topk_equal)
        rows.append(
            {
                "sample_id": left["sample_id"],
                "unit_id": left["unit_id"],
                "canonical_statement_id": left["canonical_statement_id"],
                "dicom_rgb_sha256": left["dicom_rgb_sha256"],
                "first": left,
                "replay": right,
                "absolute_differences": numeric_diffs,
                "max_absolute_difference": row_max,
                "topk_indices_equal": topk_equal,
            }
        )

    eligible_rows = int(geometry_lock["eligible"])
    estimated_c4_hours = estimate_c4_hours(
        [first_seconds, second_seconds],
        sample_count=len(selected),
        eligible_rows=eligible_rows,
        geometry_seconds=float(geometry_lock.get("wall_seconds", 0.0)),
    )
    replay_pass = max_abs_diff <= REPLAY_ATOL and topk_mismatch_count == 0
    compute_pass = estimated_c4_hours <= C4_COMPUTE_CAP_HOURS
    status = "pass" if replay_pass and compute_pass else "fail"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "timing_replay_rows.jsonl"
    temporary = rows_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(rows_path)
    script_path = Path(__file__)
    lock = {
        "format_version": FORMAT_VERSION,
        "status": status,
        "formal_result": False,
        "evaluation_only": True,
        "training_performed": False,
        "source_split": "vindr_train_protocol_design_positive_only",
        "forbidden_test_path_accessed": False,
        "rescue_confirm_rows_used": 0,
        "patient_level_claim": False,
        "selection_rule": (
            "first_8_feasible_consolidation_then_first_8_feasible_pleural_effusion_"
            "excluding_prior_units_by_sample_id"
        ),
        "selected_images": len(selected),
        "selected_unique_units": len({str(row["unit_id"]) for row in selected}),
        "selected_per_finding": dict(sorted(selection_counts.items())),
        "identity": {
            "git_head": git_head(),
            "source_hashes": observed_hashes,
            "geometry_lock_sha256": file_sha256(args.geometry_lock),
            "model_snapshot_sha256": EXPECTED_MODEL_SNAPSHOT_SHA256,
            "checkpoint_step": int(checkpoint["step"]),
            "checkpoint_variant": str(checkpoint["variant"]),
            "image_size": IMAGE_SIZE,
            "dtype": args.dtype,
            "device": str(device),
            "dicom_preprocess_version": DICOM_PREPROCESS_VERSION,
            "image_preprocess_version": QWEN35_IMAGE_PREPROCESS_VERSION,
            "script_sha256": file_sha256(script_path),
            "backbones_sha256": file_sha256(ROOT / "bives_cxr/backbones.py"),
            "polarity_runtime_sha256": file_sha256(ROOT / "bives_cxr/polarity_runtime.py"),
        },
        "replay": {
            "atol": REPLAY_ATOL,
            "max_absolute_difference": max_abs_diff,
            "topk_mismatch_count": topk_mismatch_count,
            "pass": replay_pass,
        },
        "timing": {
            "pass_seconds": [first_seconds, second_seconds],
            "seconds_per_image_conservative": max(first_seconds, second_seconds) / len(selected),
            "c4_eligible_rows": eligible_rows,
            "c4_variants_per_row": C4_VARIANTS_PER_ROW,
            "c4_visual_forwards": eligible_rows * C4_VARIANTS_PER_ROW,
            "timing_multiplier": C4_TIMING_MULTIPLIER,
            "geometry_seconds_included": float(geometry_lock.get("wall_seconds", 0.0)),
            "estimated_c4_hours": estimated_c4_hours,
            "compute_cap_hours": C4_COMPUTE_CAP_HOURS,
            "pass": compute_pass,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "rows_sha256": file_sha256(rows_path),
    }
    write_json(args.output_dir / "timing_replay_lock.json", lock)
    print(json.dumps(lock, indent=2, ensure_ascii=False))
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
