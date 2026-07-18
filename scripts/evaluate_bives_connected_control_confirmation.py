"""Open the frozen image-disjoint C5 confirmation split exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bives_cxr.backbones import (  # noqa: E402
    Qwen35VisionAdapter,
    load_qwen35_visual_and_processor,
)
from bives_cxr.dicom import load_cxr_dicom  # noqa: E402
from bives_cxr.pixel_interventions import (  # noqa: E402
    LOCAL_MEAN_RING_WIDTH,
    MASKED_GAUSSIAN_SIGMA,
    MASKED_GAUSSIAN_TRUNCATE,
    deterministic_random_mask,
    patch_gate_to_pixel_mask,
    replace_with_local_ring_mean,
    replace_with_masked_gaussian_blur,
    transform_mask_to_letterbox,
    union_box_mask,
)
from bives_cxr.polarity_runtime import (  # noqa: E402
    load_locked_polarity_checkpoint,
)
from bives_cxr.rescue_protocol import (  # noqa: E402
    COORDINATE_ZONE_CONTROL_VERSION,
    deterministic_coordinate_zone_connected_control_mask,
)
from scripts.evaluate_bives_connected_control_mechanism import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    FINDINGS,
    IMAGE_SIZE,
    OPERATORS,
    content_geometry,
    evaluate_c4_gate,
    file_sha256,
    read_jsonl,
    score_original,
    score_variants,
    snapshot_model_files,
    summarize_c4_operator,
    write_json,
)


FORMAT_VERSION = "bives_connected_control_c5_confirmation_v1"
MINIMUM_OVERALL_FINDING_FEASIBILITY = 0.95
MINIMUM_FINDING_AREA_QUARTILE_FEASIBILITY = 0.90
EXPECTED_MANIFEST_SHA256 = (
    "bd84cd7ca5384afbcb6228c49331b028a9641dd3dd9011157c2cec75b1f6514f"
)
EXPECTED_DATA_LOCK_SHA256 = (
    "4251027b3069b21fb6fb5acd6bc02bf003206fbcfffb6d045abd2289ea2ac409"
)
EXPECTED_C4_ROWS_SHA256 = (
    "268d2cc6f758d719ef7112399da38dd3ca60b1069ad12af7175afce93993dbdd"
)
EXPECTED_C4_METRICS_SHA256 = (
    "072128051b9266bb771f9c6c95a21dcbfd96ed324609b6f0758e850d3dab931c"
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
EXPECTED_CONFIRM_ROWS = 756
EXPECTED_CONFIRM_POSITIVES = 378


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
        "--c4-dir",
        type=Path,
        default=Path("local_runs/bives_cxr/connected_control_c4_mechanism"),
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
        "--b0-dir",
        type=Path,
        default=Path("local_runs/bives_cxr/qwen35_2b_sc_b0_pooled_seed17"),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("H:/Xiyao_Wang/001_models/Qwen3.5-2B"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("local_runs/bives_cxr/connected_control_c5_confirmation"),
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--geometry-workers", type=int, default=8)
    return parser.parse_args()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def require_tracked_tree_clean() -> None:
    for command in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            raise ValueError("C5 requires all tracked code/protocol changes committed")


def prepare_confirmation_opening(
    output_dir: Path,
    identity: dict[str, Any],
) -> tuple[Path, bool]:
    """Create or validate the one-time opening marker; completed runs never reopen."""
    marker_path = output_dir / "CONFIRMATION_OPENED.json"
    metrics_path = output_dir / "metrics_final.json"
    if metrics_path.exists():
        raise ValueError("C5 confirmation already has a final result and cannot rerun")
    if output_dir.exists() and any(output_dir.iterdir()) and not marker_path.exists():
        raise ValueError("nonempty C5 output lacks the one-time opening marker")
    output_dir.mkdir(parents=True, exist_ok=True)
    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("identity") != identity:
            raise ValueError("existing C5 opening marker has a different frozen identity")
        return marker_path, False
    write_json(
        marker_path,
        {
            "format_version": FORMAT_VERSION,
            "opened_once": True,
            "opened_at_unix": time.time(),
            "identity": identity,
            "post_outcome_changes_permitted": False,
            "rerun_permitted": False,
        },
    )
    return marker_path, True


def prepare_confirm_mask_one(payload: tuple[dict[str, Any], str]) -> dict[str, Any]:
    row, cache_dir_string = payload
    if row.get("rescue_split") != "rescue_confirm" or int(row["binary_label"]) != 1:
        raise ValueError("C5 geometry worker received a non-confirm/non-positive row")
    if row.get("source_split") != "train":
        raise ValueError("C5 geometry worker received a non-train row")
    image_path = Path(str(row["image_path"]))
    if "test" in {part.lower() for part in image_path.parts}:
        raise ValueError(f"VinDr-test path is forbidden in C5: {image_path}")
    cache_dir = Path(cache_dir_string)
    sample_id = str(row["sample_id"])
    width = int(row["native_columns"])
    height = int(row["native_rows"])
    letterbox, content = content_geometry(width, height)
    target = transform_mask_to_letterbox(
        union_box_mask(width, height, row["bounding_boxes"]),
        letterbox,
        IMAGE_SIZE,
    ) & content
    base = {
        "sample_id": sample_id,
        "unit_id": str(row["unit_id"]),
        "canonical_statement_id": str(row["canonical_statement_id"]),
        "reader_consensus": str(row["reader_consensus"]),
        "box_area_quartile": int(row["box_area_quartile"]),
    }
    try:
        control, geometry = deterministic_coordinate_zone_connected_control_mask(
            target,
            content,
            seed_key=f"{sample_id}:{COORDINATE_ZONE_CONTROL_VERSION}",
        )
    except ValueError as error:
        return {
            **base,
            "feasible": False,
            "exclusion_reason": str(error),
            "contracts_pass": False,
        }
    contracts = {
        "area_equal": int(target.sum()) == int(control.sum()),
        "contained_in_content": not bool((control & ~content).any()),
        "target_disjoint": not bool((control & target).any()),
        "single_4_connected_component": int(geometry["control"]["component_count"]) == 1,
        "horizontal_zone_equal": (
            geometry["target_zone"]["horizontal"]
            == geometry["control_zone"]["horizontal"]
        ),
        "vertical_zone_equal": (
            geometry["target_zone"]["vertical"]
            == geometry["control_zone"]["vertical"]
        ),
        "not_true_anatomy_claim": not bool(geometry["true_anatomy_segmentation"]),
    }
    if not all(contracts.values()):
        raise AssertionError(f"C5 connected-control contract failure: {contracts}")
    mask_name = hashlib.sha256(sample_id.encode("utf-8")).hexdigest() + ".npz"
    mask_path = cache_dir / mask_name
    temporary = mask_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        target_mask=target,
        control_mask=control,
        content_mask=content,
    )
    temporary.replace(mask_path)
    return {
        **base,
        "feasible": True,
        "exclusion_reason": None,
        "contracts_pass": True,
        "mask_file": mask_name,
        "mask_sha256": file_sha256(mask_path),
        "target_area_pixels": int(target.sum()),
        "control_area_pixels": int(control.sum()),
        "control_geometry": geometry,
    }


def summarize_geometry_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != EXPECTED_CONFIRM_POSITIVES:
        raise ValueError("C5 geometry requires every frozen confirmation positive")

    def summarize(fields: tuple[str, ...], minimum: float) -> dict[str, Any]:
        groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[tuple(str(row[field]) for field in fields)].append(row)
        result = {}
        for key, subset in sorted(groups.items()):
            eligible = sum(bool(row["feasible"]) for row in subset)
            rate = eligible / len(subset)
            result["|".join(key)] = {
                "total": len(subset),
                "eligible": eligible,
                "excluded": len(subset) - eligible,
                "feasibility": float(rate),
                "minimum": float(minimum),
                "pass": bool(rate >= minimum),
            }
        return result

    eligible = sum(bool(row["feasible"]) for row in rows)
    overall_rate = eligible / len(rows)
    per_finding = summarize(
        ("canonical_statement_id",), MINIMUM_OVERALL_FINDING_FEASIBILITY
    )
    per_finding_area = summarize(
        ("canonical_statement_id", "box_area_quartile"),
        MINIMUM_FINDING_AREA_QUARTILE_FEASIBILITY,
    )
    invariant_failures = sum(
        bool(row["feasible"]) and not bool(row["contracts_pass"]) for row in rows
    )
    passed = (
        overall_rate >= MINIMUM_OVERALL_FINDING_FEASIBILITY
        and all(point["pass"] for point in per_finding.values())
        and all(point["pass"] for point in per_finding_area.values())
        and invariant_failures == 0
    )
    return {
        "status": "pass" if passed else "fail",
        "total": len(rows),
        "eligible": eligible,
        "excluded": len(rows) - eligible,
        "overall_feasibility": float(overall_rate),
        "overall_minimum": MINIMUM_OVERALL_FINDING_FEASIBILITY,
        "per_finding": per_finding,
        "per_finding_area_quartile": per_finding_area,
        "invariant_failures": invariant_failures,
        "pass": passed,
    }


def stable_sigmoid(value: float) -> float:
    if value >= 0:
        return float(1.0 / (1.0 + np.exp(-value)))
    exp_value = np.exp(value)
    return float(exp_value / (1.0 + exp_value))


def b0_probability(pooled: np.ndarray, model: Any) -> float:
    standardized = (pooled - model["scaler_mean"]) / model["scaler_scale"]
    logit = float(np.dot(model["coefficient"][0], standardized) + model["intercept"][0])
    return stable_sigmoid(logit)


def summarize_polarity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for finding in FINDINGS:
        subset = [row for row in rows if row["canonical_statement_id"] == finding]
        labels = np.asarray([int(row["binary_label"]) for row in subset], dtype=int)
        if len(subset) == 0 or labels.sum() * 2 != len(labels):
            raise ValueError(f"C5 polarity rows are not exactly balanced: {finding}")
        b0 = np.asarray([float(row["b0_support_probability"]) for row in subset])
        b2 = np.asarray([float(row["b2_support_probability"]) for row in subset])
        b0_auroc = float(roc_auc_score(labels, b0))
        b0_auprc = float(average_precision_score(labels, b0))
        b2_auroc = float(roc_auc_score(labels, b2))
        b2_auprc = float(average_precision_score(labels, b2))
        result[finding] = {
            "records": len(subset),
            "positives": int(labels.sum()),
            "negatives": int((1 - labels).sum()),
            "b0_auroc": b0_auroc,
            "b0_auprc": b0_auprc,
            "b2_auroc": b2_auroc,
            "b2_auprc": b2_auprc,
            "b2_auroc_not_below_b0": b2_auroc >= b0_auroc,
            "b2_auprc_not_below_b0": b2_auprc >= b0_auprc,
        }
    passed = all(
        point["b2_auroc_not_below_b0"] and point["b2_auprc_not_below_b0"]
        for point in result.values()
    )
    return {
        "per_finding": result,
        "pass": passed,
        "status": "pass" if passed else "fail",
    }


def main() -> None:
    process_start = time.perf_counter()
    args = parse_args()
    if args.geometry_workers <= 0:
        raise ValueError("geometry-workers must be positive")
    require_tracked_tree_clean()

    c4_rows_path = args.c4_dir / "mechanism_rows.jsonl"
    c4_metrics_path = args.c4_dir / "metrics_final.json"
    expected_hashes = {
        args.manifest: EXPECTED_MANIFEST_SHA256,
        args.data_lock: EXPECTED_DATA_LOCK_SHA256,
        c4_rows_path: EXPECTED_C4_ROWS_SHA256,
        c4_metrics_path: EXPECTED_C4_METRICS_SHA256,
        args.checkpoint: EXPECTED_CHECKPOINT_SHA256,
        args.training_cache_lock: EXPECTED_CACHE_LOCK_SHA256,
        args.config: EXPECTED_CONFIG_SHA256,
    }
    observed_hashes = {str(path): file_sha256(path) for path in expected_hashes}
    for path, expected in expected_hashes.items():
        if observed_hashes[str(path)] != expected:
            raise ValueError(f"C5 frozen identity mismatch: {path}")

    data_lock = json.loads(args.data_lock.read_text(encoding="utf-8"))
    c4_metrics = json.loads(c4_metrics_path.read_text(encoding="utf-8"))
    cache_lock = json.loads(args.training_cache_lock.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if data_lock.get("status") != "pass" or data_lock.get("source_split") != "train_only":
        raise ValueError("C5 requires passing train-only R001")
    if c4_metrics.get("status") != "pass" or not c4_metrics.get("gate", {}).get("pass"):
        raise ValueError("C5 requires the complete frozen C4 pass")
    if c4_metrics.get("mechanism_rows_sha256") != EXPECTED_C4_ROWS_SHA256:
        raise ValueError("C5 C4 rows binding failed")
    if config.get("variant") != "B2_sparse_exact_k" or int(config.get("topk", 0)) != 16:
        raise ValueError("C5 config is not frozen B2 exact-K=16")
    if cache_lock.get("status") != "complete":
        raise ValueError("C5 training cache lock is incomplete")
    if snapshot_model_files(args.model_path) != EXPECTED_MODEL_SNAPSHOT_SHA256:
        raise ValueError("C5 Qwen3.5-2B snapshot changed")
    c4_script_path = ROOT / "scripts/evaluate_bives_connected_control_mechanism.py"
    c4_pixel_path = ROOT / "bives_cxr/pixel_interventions.py"
    if file_sha256(c4_script_path) != c4_metrics["identity"]["script_sha256"]:
        raise ValueError("C5 cannot run after changing the frozen C4 evaluator")
    if file_sha256(c4_pixel_path) != c4_metrics["identity"]["pixel_interventions_sha256"]:
        raise ValueError("C5 cannot run after changing the frozen C4 operators")
    if (
        int(c4_metrics["identity"]["local_mean_ring_width"]) != LOCAL_MEAN_RING_WIDTH
        or float(c4_metrics["identity"]["masked_gaussian_sigma"])
        != MASKED_GAUSSIAN_SIGMA
        or float(c4_metrics["identity"]["masked_gaussian_truncate"])
        != MASKED_GAUSSIAN_TRUNCATE
    ):
        raise ValueError("C5 operator constants differ from C4")

    b0_metrics_path = args.b0_dir / "metrics_final.json"
    b0_metrics = json.loads(b0_metrics_path.read_text(encoding="utf-8"))
    if b0_metrics.get("cache_lock_sha256") != EXPECTED_CACHE_LOCK_SHA256:
        raise ValueError("C5 B0 is not bound to the frozen cache")
    b0_hashes = {str(b0_metrics_path): file_sha256(b0_metrics_path)}
    b0_models = {}
    for finding in FINDINGS:
        record = b0_metrics["models"][finding]
        path = args.b0_dir / str(record["file"])
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"C5 B0 model hash mismatch: {finding}")
        b0_hashes[str(path)] = file_sha256(path)
        b0_models[finding] = np.load(path)

    plan_path = ROOT / "refine-logs/CONNECTED_CONTROL_RESCUE_PLAN.md"
    tracker_path = ROOT / "refine-logs/CONNECTED_CONTROL_RESCUE_TRACKER.md"
    identity = {
        "format_version": FORMAT_VERSION,
        "git_head": git_head(),
        "script_sha256": file_sha256(Path(__file__)),
        "source_hashes": observed_hashes,
        "b0_hashes": b0_hashes,
        "plan_sha256": file_sha256(plan_path),
        "tracker_sha256": file_sha256(tracker_path),
        "model_snapshot_sha256": EXPECTED_MODEL_SNAPSHOT_SHA256,
        "control_version": COORDINATE_ZONE_CONTROL_VERSION,
        "local_mean_ring_width": LOCAL_MEAN_RING_WIDTH,
        "masked_gaussian_sigma": MASKED_GAUSSIAN_SIGMA,
        "masked_gaussian_truncate": MASKED_GAUSSIAN_TRUNCATE,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "device": args.device,
        "dtype": args.dtype,
    }
    opening_marker, opened_now = prepare_confirmation_opening(args.output_dir, identity)

    all_rows = read_jsonl(args.manifest)
    rows = sorted(
        (row for row in all_rows if row.get("rescue_split") == "rescue_confirm"),
        key=lambda row: str(row["sample_id"]),
    )
    if len(rows) != EXPECTED_CONFIRM_ROWS:
        raise ValueError(f"C5 expected {EXPECTED_CONFIRM_ROWS} confirmation rows")
    if any(row.get("source_split") != "train" for row in rows):
        raise ValueError("C5 selected a non-train row")
    if any("test" in {part.lower() for part in Path(str(row["image_path"])).parts} for row in rows):
        raise ValueError("C5 encountered a forbidden VinDr-test path")
    positives = [row for row in rows if int(row["binary_label"]) == 1]
    if len(positives) != EXPECTED_CONFIRM_POSITIVES:
        raise ValueError("C5 confirmation positive count changed")

    mask_dir = args.output_dir / "geometry_masks"
    mask_dir.mkdir(exist_ok=True)
    geometry_path = args.output_dir / "geometry_rows.jsonl"
    geometry_lock_path = args.output_dir / "geometry_lock.json"
    geometry_start = time.perf_counter()
    geometry_rows = []
    if geometry_lock_path.exists():
        lock = json.loads(geometry_lock_path.read_text(encoding="utf-8"))
        if lock.get("identity") != identity or lock.get("status") != "complete":
            raise ValueError("existing C5 geometry lock has a different identity")
        geometry_rows = read_jsonl(geometry_path)
    else:
        with ProcessPoolExecutor(max_workers=args.geometry_workers) as executor:
            futures = {
                executor.submit(prepare_confirm_mask_one, (row, str(mask_dir))): row
                for row in positives
            }
            for index, future in enumerate(as_completed(futures), start=1):
                geometry_rows.append(future.result())
                if index % 10 == 0 or index == len(positives):
                    print(f"C5_GEOMETRY {index}/{len(positives)}", flush=True)
        geometry_rows.sort(key=lambda row: str(row["sample_id"]))
        temporary = geometry_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in geometry_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary.replace(geometry_path)
        geometry_gate = summarize_geometry_gate(geometry_rows)
        write_json(
            geometry_lock_path,
            {
                "format_version": FORMAT_VERSION,
                "status": "complete",
                "identity": identity,
                "geometry_gate": geometry_gate,
                "rows_sha256": file_sha256(geometry_path),
            },
        )
    geometry_seconds = float(time.perf_counter() - geometry_start)
    geometry_gate = summarize_geometry_gate(geometry_rows)
    if file_sha256(geometry_path) != json.loads(
        geometry_lock_path.read_text(encoding="utf-8")
    )["rows_sha256"]:
        raise ValueError("C5 geometry rows do not match their lock")
    if not geometry_gate["pass"]:
        result = {
            "format_version": FORMAT_VERSION,
            "status": "fail_final_stop",
            "formal_result": False,
            "patient_level_claim": False,
            "training_performed": False,
            "opened_confirmation_once": True,
            "opened_now": opened_now,
            "opening_marker_sha256": file_sha256(opening_marker),
            "identity": identity,
            "geometry_gate": geometry_gate,
            "model_loaded": False,
            "scores_accessed": False,
            "forbidden_test_path_accessed": False,
            "decision": "C5 final stop at geometry; no rerun permitted",
        }
        write_json(args.output_dir / "metrics_final.json", result)
        raise SystemExit(2)

    geometry_by_sample = {str(row["sample_id"]): row for row in geometry_rows}
    progress_path = args.output_dir / "progress.json"
    completed: dict[str, dict[str, Any]] = {}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != identity or progress.get("status") != "in_progress":
            raise ValueError("existing C5 progress cannot be resumed")
        completed = dict(progress.get("completed", {}))

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("C5 requires an available local CUDA device")
    torch.manual_seed(20260718)
    torch.cuda.manual_seed_all(20260718)
    torch.use_deterministic_algorithms(True)
    polarity_model, statement_to_index, checkpoint = load_locked_polarity_checkpoint(
        args.checkpoint, device
    )
    if checkpoint.get("variant") != "B2_sparse_exact_k" or int(checkpoint.get("step", -1)) != 450:
        raise ValueError("C5 checkpoint is not frozen B2 step 450")
    if checkpoint.get("cache_lock_sha256") != EXPECTED_CACHE_LOCK_SHA256:
        raise ValueError("C5 checkpoint/cache binding failed")
    visual, processor, qwen_config = load_qwen35_visual_and_processor(
        args.model_path, dtype=args.dtype, attention_implementation="eager"
    )
    visual = visual.to(device).eval()
    adapter = Qwen35VisionAdapter(
        visual,
        spatial_merge_size=int(qwen_config["vision_config"]["spatial_merge_size"]),
    ).to(device).eval()
    score_kwargs = {
        "polarity_model": polarity_model,
        "statement_to_index": statement_to_index,
        "adapter": adapter,
        "visual": visual,
        "processor": processor,
        "device": device,
    }
    score_start = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        sample_id = str(row["sample_id"])
        if sample_id in completed:
            continue
        image, dicom_record = load_cxr_dicom(Path(str(row["image_path"])))
        original, extracted = score_original(image, row, **score_kwargs)
        finding = str(row["canonical_statement_id"])
        pooled = (
            extracted["patch_tokens"][extracted["valid_mask"]]
            .mean(dim=0)
            .detach()
            .cpu()
            .numpy()
        )
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "unit_id": str(row["unit_id"]),
            "canonical_statement_id": finding,
            "binary_label": int(row["binary_label"]),
            "state": str(row["state"]),
            "source_split": "train",
            "rescue_split": "rescue_confirm",
            "b2_support_probability": float(original["support_probability"]),
            "b0_support_probability": b0_probability(pooled, b0_models[finding]),
            "b2_signed_evidence": float(original["signed_evidence"]),
            "b2_evidence_pos": float(original["evidence_pos"]),
            "b2_evidence_neg": float(original["evidence_neg"]),
            "b2_topk_indices": list(original["topk_indices"]),
            "dicom_rgb_sha256": dicom_record.rgb_sha256,
            "mechanism_eligible": False,
        }
        geometry = geometry_by_sample.get(sample_id)
        if int(row["binary_label"]) == 1 and geometry and bool(geometry["feasible"]):
            mask_path = mask_dir / str(geometry["mask_file"])
            if file_sha256(mask_path) != geometry["mask_sha256"]:
                raise ValueError(f"C5 geometry mask hash mismatch: {sample_id}")
            with np.load(mask_path, allow_pickle=False) as payload:
                target = payload["target_mask"].astype(bool)
                control = payload["control_mask"].astype(bool)
                content = payload["content_mask"].astype(bool)
            letterboxed = extracted["letterboxed_image"]
            mean_target, mean_control, blur_target, blur_control = score_variants(
                [
                    replace_with_local_ring_mean(letterboxed, target, content),
                    replace_with_local_ring_mean(letterboxed, control, content),
                    replace_with_masked_gaussian_blur(letterboxed, target, content),
                    replace_with_masked_gaussian_blur(letterboxed, control, content),
                ],
                row,
                **score_kwargs,
            )
            evidence_mask = patch_gate_to_pixel_mask(
                extracted["gate"], extracted["grid_hw"], IMAGE_SIZE
            ) & content
            random_mask = deterministic_random_mask(
                int(evidence_mask.sum()),
                content,
                seed_key=f"{sample_id}::c5-localization::{BOOTSTRAP_SEED}",
            )
            target_area = int(target.sum())
            record.update(
                {
                    "mechanism_eligible": True,
                    "reader_consensus": str(row["reader_consensus"]),
                    "box_area_quartile": int(row["box_area_quartile"]),
                    "target_area_pixels": target_area,
                    "control_area_pixels": int(control.sum()),
                    "topk_target_coverage": float((evidence_mask & target).sum() / target_area),
                    "random_target_coverage": float((random_mask & target).sum() / target_area),
                    "local_mean": {
                        "target_score": mean_target,
                        "control_score": mean_control,
                        "target_effect": float(original["support_probability"] - mean_target),
                        "control_effect": float(original["support_probability"] - mean_control),
                        "tcig": float(mean_control - mean_target),
                    },
                    "masked_gaussian_blur": {
                        "target_score": blur_target,
                        "control_score": blur_control,
                        "target_effect": float(original["support_probability"] - blur_target),
                        "control_effect": float(original["support_probability"] - blur_control),
                        "tcig": float(blur_control - blur_target),
                    },
                    "geometry_mask_file": str(geometry["mask_file"]),
                    "geometry_mask_sha256": str(geometry["mask_sha256"]),
                }
            )
        completed[sample_id] = record
        write_json(
            progress_path,
            {
                "identity": identity,
                "status": "in_progress",
                "completed_count": len(completed),
                "total": len(rows),
                "completed": completed,
            },
        )
        if index % 10 == 0 or index == len(rows):
            print(f"C5_SCORE {len(completed)}/{len(rows)}", flush=True)
    score_seconds = float(time.perf_counter() - score_start)

    final_rows = [completed[key] for key in sorted(completed)]
    if len(final_rows) != EXPECTED_CONFIRM_ROWS:
        raise ValueError("C5 score table is incomplete")
    mechanism_rows = [row for row in final_rows if row["mechanism_eligible"]]
    if len(mechanism_rows) != int(geometry_gate["eligible"]):
        raise ValueError("C5 mechanism rows do not match geometry eligibility")
    operator_results = {
        operator: summarize_c4_operator(mechanism_rows, operator)
        for operator in OPERATORS
    }
    mechanism_gate = evaluate_c4_gate(operator_results)
    polarity = summarize_polarity(final_rows)
    passed = bool(geometry_gate["pass"] and mechanism_gate["pass"] and polarity["pass"])

    rows_path = args.output_dir / "confirmation_rows.jsonl"
    temporary = rows_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in final_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(rows_path)
    c4_seconds = float(c4_metrics["runtime"]["geometry_seconds_this_process"]) + float(
        c4_metrics["runtime"]["score_seconds_this_process"]
    )
    c5_process_seconds = float(time.perf_counter() - process_start)
    c3_conservative_upper_seconds = 0.24607786347588798 * 3600.0
    total_c3_c5_hours_upper = (
        c3_conservative_upper_seconds + c4_seconds + c5_process_seconds
    ) / 3600.0
    result = {
        "format_version": FORMAT_VERSION,
        "status": "pass_internal_image_disjoint_only" if passed else "fail_final_stop",
        "formal_result": False,
        "patient_level_claim": False,
        "evaluation_only": True,
        "training_performed": False,
        "opened_confirmation_once": True,
        "opened_now": opened_now,
        "rerun_permitted": False,
        "post_outcome_changes_permitted": False,
        "forbidden_test_path_accessed": False,
        "identity": identity,
        "opening_marker_sha256": file_sha256(opening_marker),
        "rows": len(final_rows),
        "mechanism_rows": len(mechanism_rows),
        "geometry_gate": geometry_gate,
        "polarity_gate": polarity,
        "operators": operator_results,
        "mechanism_gate": mechanism_gate,
        "complete_c5_gate_pass": passed,
        "runtime": {
            "geometry_seconds_this_process": geometry_seconds,
            "score_seconds_this_process": score_seconds,
            "c5_process_seconds": c5_process_seconds,
            "c3_conservative_upper_seconds": c3_conservative_upper_seconds,
            "c4_observed_geometry_plus_score_seconds": c4_seconds,
            "c3_c5_conservative_total_hours": total_c3_c5_hours_upper,
            "six_hour_cap_pass": total_c3_c5_hours_upper <= 6.0,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "confirmation_rows_sha256": file_sha256(rows_path),
        "geometry_rows_sha256": file_sha256(geometry_path),
        "decision": (
            "internal image-disjoint confirmation passed; independent patient-grouped final remains blocked"
            if passed
            else "C5 final stop; no result-driven change or rerun is permitted"
        ),
    }
    if not result["runtime"]["six_hour_cap_pass"]:
        result["status"] = "fail_final_stop"
        result["complete_c5_gate_pass"] = False
        result["decision"] = "C5 final stop at the frozen six-hour cap"
        passed = False
    write_json(args.output_dir / "metrics_final.json", result)
    write_json(
        progress_path,
        {
            "identity": identity,
            "status": "complete",
            "completed_count": len(completed),
            "total": len(rows),
            "metrics_final": "metrics_final.json",
        },
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
