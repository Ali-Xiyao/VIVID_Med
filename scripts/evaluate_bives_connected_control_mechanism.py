"""Run the frozen C4 connected-control mechanism gate on VinDr train only."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bives_cxr.backbones import (  # noqa: E402
    Qwen35VisionAdapter,
    load_qwen35_visual_and_processor,
)
from bives_cxr.dicom import DICOM_PREPROCESS_VERSION, load_cxr_dicom  # noqa: E402
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
    extract_qwen35_patch_batch,
    extract_qwen35_patches,
    load_locked_polarity_checkpoint,
    score_statements,
)
from bives_cxr.qwen35_preprocessing import QWEN35_IMAGE_PREPROCESS_VERSION  # noqa: E402
from bives_cxr.rescue_protocol import (  # noqa: E402
    COORDINATE_ZONE_CONTROL_VERSION,
    deterministic_coordinate_zone_connected_control_mask,
)


FORMAT_VERSION = "bives_connected_control_c4_mechanism_v1"
IMAGE_SIZE = 448
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 17
ORIGINAL_SCORE_ATOL = 1e-6
FINDINGS = ("consolidation", "pleural_effusion")
OPERATORS = ("local_mean", "masked_gaussian_blur")
EXPECTED_MANIFEST_SHA256 = (
    "bd84cd7ca5384afbcb6228c49331b028a9641dd3dd9011157c2cec75b1f6514f"
)
EXPECTED_DATA_LOCK_SHA256 = (
    "4251027b3069b21fb6fb5acd6bc02bf003206fbcfffb6d045abd2289ea2ac409"
)
EXPECTED_GEOMETRY_ROWS_SHA256 = (
    "b94b77bc528033ea44681fce33f868b87c2806d19450a66ff023628fc2e039d9"
)
EXPECTED_GEOMETRY_LOCK_SHA256 = (
    "91bc558f85cae38d143562ca9b08b9d71c1821569df5cbb35d98eb24f68af71b"
)
EXPECTED_C3_LOCK_SHA256 = (
    "e7047f605604def320079006d486f5f55645951de3566c9675fc8308510fd4d6"
)
EXPECTED_C3_ROWS_SHA256 = (
    "5d916fb9f86e45fc5fdee5e31fec1e1c4c849d861bfeb8f6d25c4ad4e6166da6"
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
        default=Path("local_runs/bives_cxr/vindr_connected_control_geometry/connected_geometry_rows.jsonl"),
    )
    parser.add_argument(
        "--geometry-lock",
        type=Path,
        default=Path("local_runs/bives_cxr/vindr_connected_control_geometry/connected_geometry_lock.json"),
    )
    parser.add_argument(
        "--c3-dir",
        type=Path,
        default=Path("local_runs/bives_cxr/connected_control_c3_timing_replay"),
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
        default=Path("local_runs/bives_cxr/connected_control_c4_mechanism"),
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--geometry-workers", type=int, default=8)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
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
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def snapshot_model_files(model_root: Path) -> str:
    names = {"config.json", "model.safetensors.index.json"}
    names.update(path.name for path in model_root.glob("*.safetensors"))
    files = {
        name: file_sha256(model_root / name)
        for name in sorted(names)
        if (model_root / name).is_file()
    }
    return canonical_sha256(files)


def content_geometry(
    width: int,
    height: int,
) -> tuple[tuple[int, int, int, int], np.ndarray]:
    scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    left = (IMAGE_SIZE - resized_width) // 2
    top = (IMAGE_SIZE - resized_height) // 2
    box = (left, top, left + resized_width, top + resized_height)
    content = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    content[top : top + resized_height, left : left + resized_width] = True
    return box, content


def prepare_mask_cache_one(payload: tuple[dict[str, Any], str]) -> dict[str, Any]:
    row, cache_dir_string = payload
    cache_dir = Path(cache_dir_string)
    sample_id = str(row["sample_id"])
    mask_name = hashlib.sha256(sample_id.encode("utf-8")).hexdigest() + ".npz"
    mask_path = cache_dir / mask_name
    width = int(row["native_columns"])
    height = int(row["native_rows"])
    letterbox, content = content_geometry(width, height)
    target = transform_mask_to_letterbox(
        union_box_mask(width, height, row["bounding_boxes"]),
        letterbox,
        IMAGE_SIZE,
    ) & content
    control, geometry = deterministic_coordinate_zone_connected_control_mask(
        target,
        content,
        seed_key=f"{sample_id}:{COORDINATE_ZONE_CONTROL_VERSION}",
    )
    temporary = mask_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        target_mask=target,
        control_mask=control,
        content_mask=content,
    )
    temporary.replace(mask_path)
    return {
        "sample_id": sample_id,
        "mask_file": mask_name,
        "mask_sha256": file_sha256(mask_path),
        "target_area_pixels": int(target.sum()),
        "control_area_pixels": int(control.sum()),
        "control_geometry": geometry,
    }


def localization_quartile(value: float, boundaries: list[float]) -> int:
    if len(boundaries) != 5:
        raise ValueError("localization quartiles require five boundaries")
    return int(np.searchsorted(np.asarray(boundaries[1:-1]), value, side="right") + 1)


def summarize_c4_operator(
    rows: list[dict[str, Any]],
    operator: str,
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if operator not in OPERATORS or not rows:
        raise ValueError("C4 operator/rows are invalid")
    per_finding: dict[str, Any] = {}
    rng = np.random.default_rng(bootstrap_seed)
    for finding in FINDINGS:
        subset = [row for row in rows if row["canonical_statement_id"] == finding]
        if not subset:
            raise ValueError(f"C4 has no rows for {finding}")
        tcig = np.asarray([float(row[operator]["tcig"]) for row in subset])
        target_effect = np.asarray(
            [float(row[operator]["target_effect"]) for row in subset]
        )
        control_effect = np.asarray(
            [float(row[operator]["control_effect"]) for row in subset]
        )
        units = sorted({str(row["unit_id"]) for row in subset})
        by_unit = {
            unit: [row for row in subset if str(row["unit_id"]) == unit]
            for unit in units
        }
        replicates = []
        for _ in range(int(bootstrap_replicates)):
            sampled_units = rng.choice(units, size=len(units), replace=True)
            sampled_rows = [row for unit in sampled_units for row in by_unit[str(unit)]]
            replicates.append(
                float(np.mean([float(row[operator]["tcig"]) for row in sampled_rows]))
            )
        area_strata = {}
        for quartile in range(1, 5):
            stratum = [row for row in subset if int(row["box_area_quartile"]) == quartile]
            area_strata[str(quartile)] = {
                "records": len(stratum),
                "mean_tcig": float(
                    np.mean([float(row[operator]["tcig"]) for row in stratum])
                ),
            }
        consensus_strata = {}
        for consensus in sorted({str(row["reader_consensus"]) for row in subset}):
            stratum = [row for row in subset if str(row["reader_consensus"]) == consensus]
            values = np.asarray([float(row[operator]["tcig"]) for row in stratum])
            consensus_strata[consensus] = {
                "records": len(stratum),
                "mean_tcig": float(values.mean()),
                "positive_fraction": float((values > 0).mean()),
            }
        coverage = np.asarray([float(row["topk_target_coverage"]) for row in subset])
        boundaries = np.quantile(coverage, [0.0, 0.25, 0.5, 0.75, 1.0]).tolist()
        localization_strata = {}
        for quartile in range(1, 5):
            stratum = [
                row
                for row in subset
                if localization_quartile(float(row["topk_target_coverage"]), boundaries)
                == quartile
            ]
            localization_strata[str(quartile)] = {
                "records": len(stratum),
                "mean_tcig": float(
                    np.mean([float(row[operator]["tcig"]) for row in stratum])
                )
                if stratum
                else None,
            }
        per_finding[finding] = {
            "records": len(subset),
            "mean_target_effect": float(target_effect.mean()),
            "mean_control_effect": float(control_effect.mean()),
            "mean_tcig": float(tcig.mean()),
            "bootstrap_95ci": {
                "lower": float(np.percentile(replicates, 2.5)),
                "upper": float(np.percentile(replicates, 97.5)),
            },
            "positive_image_fraction": float((tcig > 0).mean()),
            "area_quartiles": area_strata,
            "reader_consensus": consensus_strata,
            "localization_quartile_boundaries": boundaries,
            "localization_quartiles": localization_strata,
        }
    return {
        "operator": operator,
        "per_finding": per_finding,
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed": int(bootstrap_seed),
        "patient_level_confidence_interval": False,
        "confidence_interval_unit": "image_level_cluster_by_unit_id",
    }


def evaluate_c4_gate(operator_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for operator in OPERATORS:
        result = operator_results[operator]
        for finding in FINDINGS:
            point = result["per_finding"][finding]
            checks[f"{operator}|{finding}"] = {
                "mean_tcig_positive": float(point["mean_tcig"]) > 0.0,
                "highest_area_quartile_nonnegative": (
                    float(point["area_quartiles"]["4"]["mean_tcig"]) >= 0.0
                ),
                "positive_image_fraction_at_least_0_60": (
                    float(point["positive_image_fraction"]) >= 0.60
                ),
            }
    ci_checks = {
        finding: any(
            float(operator_results[operator]["per_finding"][finding]["bootstrap_95ci"]["lower"])
            > 0.0
            for operator in OPERATORS
        )
        for finding in FINDINGS
    }
    passed = all(all(values.values()) for values in checks.values()) and all(
        ci_checks.values()
    )
    return {
        "status": "pass" if passed else "fail",
        "per_operator_finding": checks,
        "at_least_one_operator_ci_lower_positive_per_finding": ci_checks,
        "pass": passed,
    }


def score_original(
    image,
    row: dict[str, Any],
    *,
    polarity_model,
    statement_to_index: dict[str, int],
    adapter,
    visual,
    processor,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
    score = {
        "support_probability": float(output["support_probability"][0].cpu()),
        "signed_evidence": float(output["signed_evidence"][0].cpu()),
        "evidence_pos": float(output["evidence_pos"][0].cpu()),
        "evidence_neg": float(output["evidence_neg"][0].cpu()),
        "topk_indices": torch.where(output["gate"][0].detach().cpu() > 0.5)[0].tolist(),
    }
    return score, {**extracted, "gate": output["gate"][0].detach().cpu() > 0.5}


def score_variants(
    images: list[Any],
    row: dict[str, Any],
    *,
    polarity_model,
    statement_to_index: dict[str, int],
    adapter,
    visual,
    processor,
    device: torch.device,
) -> list[float]:
    finding = str(row["canonical_statement_id"])
    extracted_batch = extract_qwen35_patch_batch(
        images,
        [str(row["statement_text"])] * len(images),
        adapter=adapter,
        visual=visual,
        processor=processor,
        device=device,
        image_size=IMAGE_SIZE,
    )
    scores = []
    for extracted in extracted_batch:
        output = score_statements(
            polarity_model,
            extracted["patch_tokens"],
            extracted["valid_mask"],
            [statement_to_index[finding]],
        )
        scores.append(float(output["support_probability"][0].cpu()))
    return scores


def main() -> None:
    args = parse_args()
    if args.geometry_workers <= 0:
        raise ValueError("geometry-workers must be positive")
    c3_lock_path = args.c3_dir / "timing_replay_lock.json"
    c3_rows_path = args.c3_dir / "timing_replay_rows.jsonl"
    expected_hashes = {
        args.manifest: EXPECTED_MANIFEST_SHA256,
        args.data_lock: EXPECTED_DATA_LOCK_SHA256,
        args.geometry_rows: EXPECTED_GEOMETRY_ROWS_SHA256,
        args.geometry_lock: EXPECTED_GEOMETRY_LOCK_SHA256,
        c3_lock_path: EXPECTED_C3_LOCK_SHA256,
        c3_rows_path: EXPECTED_C3_ROWS_SHA256,
        args.checkpoint: EXPECTED_CHECKPOINT_SHA256,
        args.training_cache_lock: EXPECTED_CACHE_LOCK_SHA256,
        args.config: EXPECTED_CONFIG_SHA256,
    }
    observed_hashes = {str(path): file_sha256(path) for path in expected_hashes}
    for path, expected in expected_hashes.items():
        if observed_hashes[str(path)] != expected:
            raise ValueError(f"C4 frozen identity mismatch: {path}")
    data_lock = json.loads(args.data_lock.read_text(encoding="utf-8"))
    geometry_lock = json.loads(args.geometry_lock.read_text(encoding="utf-8"))
    c3_lock = json.loads(c3_lock_path.read_text(encoding="utf-8"))
    cache_lock = json.loads(args.training_cache_lock.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if data_lock.get("status") != "pass" or data_lock.get("source_split") != "train_only":
        raise ValueError("C4 requires passing train-only R001")
    if geometry_lock.get("status") != "pass" or int(geometry_lock.get("eligible", 0)) != 375:
        raise ValueError("C4 requires passing 375-row C2 geometry")
    if c3_lock.get("status") != "pass" or not c3_lock.get("replay", {}).get("pass"):
        raise ValueError("C4 requires passing C3 replay")
    if float(c3_lock.get("timing", {}).get("estimated_c4_hours", 99.0)) > 4.0:
        raise ValueError("C4 compute estimate exceeds the frozen cap")
    if config.get("variant") != "B2_sparse_exact_k" or int(config.get("topk", 0)) != 16:
        raise ValueError("C4 config is not frozen B2 exact-K=16")
    if cache_lock.get("status") != "complete":
        raise ValueError("C4 training cache lock is incomplete")
    if snapshot_model_files(args.model_path) != EXPECTED_MODEL_SNAPSHOT_SHA256:
        raise ValueError("C4 Qwen3.5-2B snapshot changed")

    manifest_rows = {str(row["sample_id"]): row for row in read_jsonl(args.manifest)}
    geometry_rows = read_jsonl(args.geometry_rows)
    eligible_geometry = sorted(
        (row for row in geometry_rows if bool(row.get("feasible"))),
        key=lambda row: str(row["sample_id"]),
    )
    if len(eligible_geometry) != 375:
        raise ValueError("C4 expected exactly 375 feasible C2 rows")
    rows = []
    for geometry_row in eligible_geometry:
        sample_id = str(geometry_row["sample_id"])
        row = manifest_rows.get(sample_id)
        if row is None:
            raise ValueError(f"C4 sample absent from R001 manifest: {sample_id}")
        image_path = Path(str(row["image_path"]))
        if row.get("rescue_split") != "protocol_design" or row.get("source_split") != "train":
            raise ValueError("C4 selected a non-design/non-train row")
        if "test" in {part.lower() for part in image_path.parts}:
            raise ValueError(f"VinDr-test path is forbidden in C4: {image_path}")
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.output_dir / "geometry_masks"
    mask_dir.mkdir(exist_ok=True)
    mask_lock_path = args.output_dir / "geometry_mask_cache_lock.json"
    mask_identity = {
        "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "geometry_rows_sha256": EXPECTED_GEOMETRY_ROWS_SHA256,
        "control_version": COORDINATE_ZONE_CONTROL_VERSION,
        "image_size": IMAGE_SIZE,
    }
    mask_records: dict[str, dict[str, Any]] = {}
    if mask_lock_path.is_file():
        mask_lock = json.loads(mask_lock_path.read_text(encoding="utf-8"))
        if mask_lock.get("identity") != mask_identity:
            raise ValueError("existing C4 geometry-mask cache has another identity")
        mask_records = dict(mask_lock.get("records", {}))
    pending = [row for row in rows if str(row["sample_id"]) not in mask_records]
    geometry_start = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=args.geometry_workers) as executor:
            futures = {
                executor.submit(prepare_mask_cache_one, (row, str(mask_dir))): row
                for row in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                mask_records[record["sample_id"]] = record
                if index % 10 == 0 or index == len(pending):
                    print(f"C4_GEOMETRY {index}/{len(pending)}", flush=True)
    geometry_seconds = float(time.perf_counter() - geometry_start)
    if len(mask_records) != 375:
        raise ValueError("C4 geometry-mask cache is incomplete")
    write_json(
        mask_lock_path,
        {
            "format_version": "bives_connected_control_c4_mask_cache_v1",
            "status": "complete",
            "identity": mask_identity,
            "records": dict(sorted(mask_records.items())),
        },
    )

    c3_reference = {}
    for row in read_jsonl(c3_rows_path):
        c3_reference[str(row["sample_id"])] = {
            "support_probability": float(row["first"]["score"]["support_probability"]),
            "topk_indices": list(row["first"]["score"]["topk_indices"]),
        }
    identity = {
        "format_version": FORMAT_VERSION,
        "source_hashes": observed_hashes,
        "model_snapshot_sha256": EXPECTED_MODEL_SNAPSHOT_SHA256,
        "control_version": COORDINATE_ZONE_CONTROL_VERSION,
        "local_mean_ring_width": LOCAL_MEAN_RING_WIDTH,
        "masked_gaussian_sigma": MASKED_GAUSSIAN_SIGMA,
        "masked_gaussian_truncate": MASKED_GAUSSIAN_TRUNCATE,
        "image_size": IMAGE_SIZE,
        "dtype": args.dtype,
        "device": args.device,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "original_score_atol": ORIGINAL_SCORE_ATOL,
        "script_sha256": file_sha256(Path(__file__)),
        "pixel_interventions_sha256": file_sha256(ROOT / "bives_cxr/pixel_interventions.py"),
        "rescue_protocol_sha256": file_sha256(ROOT / "bives_cxr/rescue_protocol.py"),
        "geometry_mask_cache_lock_sha256": file_sha256(mask_lock_path),
    }
    progress_path = args.output_dir / "progress.json"
    completed: dict[str, dict[str, Any]] = {}
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != identity:
            raise ValueError("existing C4 progress has a different identity")
        completed = dict(progress.get("completed", {}))

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("C4 requires an available local CUDA device")
    torch.manual_seed(20260718)
    torch.cuda.manual_seed_all(20260718)
    torch.use_deterministic_algorithms(True)
    polarity_model, statement_to_index, checkpoint = load_locked_polarity_checkpoint(
        args.checkpoint, device
    )
    if checkpoint.get("variant") != "B2_sparse_exact_k" or int(checkpoint.get("step", -1)) != 450:
        raise ValueError("C4 checkpoint is not B2 step 450")
    if checkpoint.get("cache_lock_sha256") != EXPECTED_CACHE_LOCK_SHA256:
        raise ValueError("C4 checkpoint/cache binding failed")
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
        mask_record = mask_records[sample_id]
        mask_path = mask_dir / str(mask_record["mask_file"])
        if file_sha256(mask_path) != mask_record["mask_sha256"]:
            raise ValueError(f"C4 geometry mask hash mismatch: {sample_id}")
        with np.load(mask_path, allow_pickle=False) as payload:
            target = payload["target_mask"].astype(bool)
            control = payload["control_mask"].astype(bool)
            content = payload["content_mask"].astype(bool)
        letterboxed = extracted["letterboxed_image"]
        variant_scores = score_variants(
            [
                replace_with_local_ring_mean(letterboxed, target, content),
                replace_with_local_ring_mean(letterboxed, control, content),
                replace_with_masked_gaussian_blur(letterboxed, target, content),
                replace_with_masked_gaussian_blur(letterboxed, control, content),
            ],
            row,
            **score_kwargs,
        )
        mean_target, mean_control, blur_target, blur_control = variant_scores
        evidence_mask = patch_gate_to_pixel_mask(
            extracted["gate"], extracted["grid_hw"], IMAGE_SIZE
        ) & content
        random_mask = deterministic_random_mask(
            int(evidence_mask.sum()),
            content,
            seed_key=f"{sample_id}::c4-localization::{BOOTSTRAP_SEED}",
        )
        target_area = int(target.sum())
        reference = c3_reference.get(sample_id)
        replay_diff = None
        replay_topk_equal = None
        if reference is not None:
            replay_diff = abs(
                float(original["support_probability"])
                - float(reference["support_probability"])
            )
            replay_topk_equal = original["topk_indices"] == reference["topk_indices"]
            if replay_diff > ORIGINAL_SCORE_ATOL or not replay_topk_equal:
                raise ValueError(f"C4 original-score replay failed: {sample_id}")
        completed[sample_id] = {
            "sample_id": sample_id,
            "unit_id": str(row["unit_id"]),
            "canonical_statement_id": str(row["canonical_statement_id"]),
            "reader_consensus": str(row["reader_consensus"]),
            "box_area_quartile": int(row["box_area_quartile"]),
            "source_split": "train",
            "rescue_split": "protocol_design",
            "dicom_rgb_sha256": dicom_record.rgb_sha256,
            "original": original,
            "c3_original_score_abs_diff": replay_diff,
            "c3_topk_indices_equal": replay_topk_equal,
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
            "geometry_mask_file": str(mask_record["mask_file"]),
            "geometry_mask_sha256": str(mask_record["mask_sha256"]),
        }
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
            print(f"C4_SCORE {len(completed)}/{len(rows)}", flush=True)
    score_seconds = float(time.perf_counter() - score_start)

    final_rows = [completed[key] for key in sorted(completed)]
    if len(final_rows) != 375:
        raise ValueError("C4 score table is incomplete")
    replay_rows = [row for row in final_rows if row["c3_original_score_abs_diff"] is not None]
    if len(replay_rows) != 16:
        raise ValueError("C4 did not replay every C3 reference image")
    max_replay_diff = max(float(row["c3_original_score_abs_diff"]) for row in replay_rows)
    replay_pass = max_replay_diff <= ORIGINAL_SCORE_ATOL and all(
        row["c3_topk_indices_equal"] for row in replay_rows
    )
    operator_results = {
        operator: summarize_c4_operator(final_rows, operator) for operator in OPERATORS
    }
    gate = evaluate_c4_gate(operator_results)
    gate["original_score_replay_pass"] = replay_pass
    gate["pass"] = bool(gate["pass"] and replay_pass)
    gate["status"] = "pass" if gate["pass"] else "fail"

    rows_path = args.output_dir / "mechanism_rows.jsonl"
    temporary = rows_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in final_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(rows_path)
    result = {
        "format_version": FORMAT_VERSION,
        "status": gate["status"],
        "formal_result": False,
        "evaluation_only": True,
        "training_performed": False,
        "source_split": "vindr_train_protocol_design_positive_only",
        "forbidden_test_path_accessed": False,
        "rescue_confirm_rows_used": 0,
        "patient_level_claim": False,
        "identity": identity,
        "rows": len(final_rows),
        "per_finding_counts": {
            finding: sum(row["canonical_statement_id"] == finding for row in final_rows)
            for finding in FINDINGS
        },
        "original_score_replay": {
            "rows": len(replay_rows),
            "atol": ORIGINAL_SCORE_ATOL,
            "max_absolute_difference": max_replay_diff,
            "topk_mismatch_count": sum(
                not row["c3_topk_indices_equal"] for row in replay_rows
            ),
            "pass": replay_pass,
        },
        "operators": operator_results,
        "gate": gate,
        "runtime": {
            "geometry_seconds_this_process": geometry_seconds,
            "score_seconds_this_process": score_seconds,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "mechanism_rows_sha256": file_sha256(rows_path),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    write_json(args.output_dir / "metrics_final.json", result)
    write_json(
        progress_path,
        {
            "identity": identity,
            "status": "complete",
            "completed_count": len(completed),
            "total": len(rows),
            "completed": completed,
            "metrics_final": "metrics_final.json",
        },
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not gate["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
