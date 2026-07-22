#!/usr/bin/env python
"""Run the trained Qwen3.5 dense-verifier expert-mask oracle locally."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arise_cxr.dense_verifier import (  # noqa: E402
    DenseVerifierScorer,
    MILVerifierScorer,
    PooledLogisticVerifierScorer,
    dense_oracle_progress_identity_matches,
    oracle_model_id_for_head,
    reconstruct_phase_h_explanation_mask,
)
from arise_cxr.oracle_ceiling import evaluate_oracle_ceiling  # noqa: E402
from arise_cxr.matched_controls import (  # noqa: E402
    STAT_MATCHED_CONTROL_VERSION,
    deterministic_stat_matched_connected_control_mask,
)
from bives_cxr.localization_causality import (  # noqa: E402
    build_precomputed_audit_row,
    build_target_specific_controls,
    intervention_strength_metrics,
)
from bives_cxr.pixel_interventions import (  # noqa: E402
    replace_with_local_ring_mean,
    replace_with_masked_gaussian_blur,
)
from bives_cxr.provenance import canonical_json_sha256, file_sha256  # noqa: E402
from bives_cxr.qwen35_preprocessing import letterbox_image  # noqa: E402
from bives_cxr.rescue_protocol import mask_geometry  # noqa: E402


DEFAULT_PHASE_H = ROOT / "local_runs/cxr_localization_causality/chexlocalize_qwen35_development"
DEFAULT_OUTPUT = ROOT / "local_runs/arise_cxr/dense_oracle_phase_h"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-h-dir", type=Path, default=DEFAULT_PHASE_H)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path, default=Path(r"H:\Xiyao_Wang\001_models\Qwen3.5-2B"))
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "local_runs/bives_cxr/qwen35_2b_sc_b1_dense_seed17/best.pt")
    parser.add_argument("--pooled-model-dir", type=Path, default=ROOT / "local_runs/bives_cxr/qwen35_2b_sc_b0_pooled_seed17")
    parser.add_argument("--mil-checkpoint", type=Path, default=ROOT / "local_runs/arise_cxr/mil_dense_sc_seed20260722/best.pt")
    parser.add_argument("--mil-box-checkpoint", type=Path, default=ROOT / "local_runs/arise_cxr/mil_vindr_box_overlap_seed20260722/best.pt")
    parser.add_argument("--mil-box-train-cache-lock", type=Path, default=ROOT / "local_runs/arise_cxr/vindr_box_sc_cache_train/cache_lock.json")
    parser.add_argument("--mil-box-val-cache-lock", type=Path, default=ROOT / "local_runs/arise_cxr/vindr_box_sc_cache_val/cache_lock.json")
    parser.add_argument("--cache-lock", type=Path, default=ROOT / "local_runs/bives_cxr/qwen35_2b_weak_sc_cache/cache_lock.json")
    parser.add_argument("--backend", choices=("b1_dense", "pooled_logistic", "mil_dense", "mil_vindr_box"), default="b1_dense")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--control-family",
        choices=("frozen_geometry", "stat_matched_connected"),
        default="frozen_geometry",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def geometry_matches(actual: np.ndarray, expected: dict[str, Any], content: np.ndarray) -> bool:
    observed = mask_geometry(actual, content)
    exact_keys = ("area_pixels", "bbox", "component_count", "perimeter_edges", "vertical_band")
    return all(observed[key] == expected[key] for key in exact_keys)


def main() -> int:
    args = parse_args()
    phase_h_result = json.loads((args.phase_h_dir / "merged/merged_result.json").read_text(encoding="utf-8"))
    phase_h_rows_path = args.phase_h_dir / "merged/audit_rows.jsonl"
    if file_sha256(phase_h_rows_path) != phase_h_result["rows_sha256"]:
        raise ValueError("Phase-H source row hash changed")
    phase_h_rows = read_jsonl(phase_h_rows_path)
    reference = {
        (row["image_id"], row["pathology_id"]): row
        for row in phase_h_rows
        if row["operator_id"] == "local_mean_ring8"
    }
    manifest = read_jsonl(args.phase_h_dir / "development_manifest.jsonl")
    selected = [
        row for row in manifest
        if (row["image_id_hash"], row["canonical_statement_id"]) in reference
    ]
    if len(selected) != 99:
        raise ValueError("ARISE dense oracle accepted Phase-H pair count changed")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid ARISE oracle shard identity")
    if args.num_shards > 1:
        patients = sorted({str(row["patient_id_hash"]) for row in selected})
        patient_shard = {
            patient: index % args.num_shards
            for index, patient in enumerate(patients)
        }
        selected = [
            row
            for row in selected
            if patient_shard[str(row["patient_id_hash"])] == args.shard_index
        ]
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("max-samples must be positive")
        selected = selected[: args.max_samples]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "progress.json"
    rows_path = args.output_dir / "audit_rows.jsonl"
    identity = {
        "schema_version": "arise-dense-oracle-run-v1",
        "phase_h_rows_sha256": phase_h_result["rows_sha256"],
        "phase_h_result_canonical_sha256": phase_h_result["canonical_sha256"],
        "backend": args.backend,
        "checkpoint_sha256": file_sha256(args.checkpoint) if args.backend == "b1_dense" else None,
        "mil_checkpoint_sha256": file_sha256(args.mil_checkpoint) if args.backend == "mil_dense" else None,
        "mil_box_checkpoint_sha256": file_sha256(args.mil_box_checkpoint) if args.backend == "mil_vindr_box" else None,
        "mil_box_train_cache_lock_sha256": file_sha256(args.mil_box_train_cache_lock) if args.backend == "mil_vindr_box" else None,
        "mil_box_val_cache_lock_sha256": file_sha256(args.mil_box_val_cache_lock) if args.backend == "mil_vindr_box" else None,
        "pooled_model_sha256": (
            {
                finding: file_sha256(args.pooled_model_dir / f"{finding}_model.npz")
                for finding in ("consolidation", "pleural_effusion")
            }
            if args.backend == "pooled_logistic"
            else None
        ),
        "cache_lock_sha256": file_sha256(args.cache_lock) if args.backend != "mil_vindr_box" else None,
        "model_path": str(args.model_path.resolve()),
        "device": args.device,
        "dtype": args.dtype,
        "max_samples": args.max_samples,
        "test_opened": False,
        "control_family": args.control_family,
    }
    if args.control_family == "stat_matched_connected":
        identity.update(
            {
                "control_family_version": STAT_MATCHED_CONTROL_VERSION,
                "matched_controls_sha256": file_sha256(
                    ROOT / "arise_cxr/matched_controls.py"
                ),
            }
        )
    if args.num_shards > 1:
        identity.update(
            {
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
                "shard_partition": "sorted_patient_id_hash_round_robin_v1",
            }
        )
    completed: dict[str, list[dict[str, Any]]] = {}
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if not dense_oracle_progress_identity_matches(progress.get("identity", {}), identity):
            raise ValueError("existing ARISE dense-oracle progress identity changed")
        completed = dict(progress.get("completed", {}))

    torch.manual_seed(20260722)
    torch.cuda.manual_seed_all(20260722)
    torch.use_deterministic_algorithms(True)
    if args.backend == "b1_dense":
        scorer = DenseVerifierScorer(
            model_path=args.model_path,
            checkpoint_path=args.checkpoint,
            cache_lock_path=args.cache_lock,
            device=args.device,
            dtype=args.dtype,
        )
    elif args.backend == "pooled_logistic":
        scorer = PooledLogisticVerifierScorer(
            model_path=args.model_path,
            pooled_model_dir=args.pooled_model_dir,
            cache_lock_path=args.cache_lock,
            device=args.device,
            dtype=args.dtype,
        )
    elif args.backend == "mil_dense":
        scorer = MILVerifierScorer(
            model_path=args.model_path,
            checkpoint_path=args.mil_checkpoint,
            cache_lock_path=args.cache_lock,
            device=args.device,
            dtype=args.dtype,
        )
    else:
        scorer = MILVerifierScorer(
            model_path=args.model_path,
            checkpoint_path=args.mil_box_checkpoint,
            cache_lock_path=None,
            train_cache_lock_path=args.mil_box_train_cache_lock,
            val_cache_lock_path=args.mil_box_val_cache_lock,
            checkpoint_identity="vindr_box_v2",
            device=args.device,
            dtype=args.dtype,
        )
    for index, source in enumerate(selected, start=1):
        sample_id = str(source["sample_id"])
        if sample_id in completed:
            continue
        rows = evaluate_sample(
            source,
            reference[(source["image_id_hash"], source["canonical_statement_id"])],
            scorer,
            args.phase_h_dir,
            control_family=args.control_family,
        )
        completed[sample_id] = rows
        progress = {
            "identity": identity,
            "status": "in_progress",
            "completed_samples": len(completed),
            "total_samples": len(selected),
            "completed": completed,
        }
        write_json(progress_path, progress)
        print(json.dumps({"completed": len(completed), "total": len(selected), "sample": index}), flush=True)

    audit_rows = sorted(
        [row for sample_rows in completed.values() for row in sample_rows],
        key=lambda row: row["row_id"],
    )
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in audit_rows),
        encoding="utf-8",
    )
    pathologies = sorted({row["pathology_id"] for row in audit_rows})
    operators = sorted({row["operator_id"] for row in audit_rows})
    gate = evaluate_oracle_ceiling(
        audit_rows,
        required_pathologies=pathologies,
        required_operators=operators,
        minimum_passing_pathologies=3,
        bootstrap_replicates=2000,
        bootstrap_seed=20260722,
    )
    result = {
        "schema_version": "arise-dense-oracle-result-v1",
        "status": "complete_development" if len(completed) == len(selected) else "partial",
        "formal_result": False,
        "confirmatory_evidence": False,
        "test_opened": False,
        "new_model_scores_created": True,
        "identity": identity,
        "model": scorer.identity(),
        "source_samples": len(selected),
        "audit_rows": len(audit_rows),
        "rows_sha256": file_sha256(rows_path),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(scorer.device)),
        "oracle_gate": gate,
    }
    result["canonical_sha256"] = canonical_json_sha256(result)
    write_json(args.output_dir / "result.json", result)
    write_json(progress_path, {**progress, "status": "complete", "result_canonical_sha256": result["canonical_sha256"]})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def evaluate_sample(
    source: dict[str, Any],
    reference: dict[str, Any],
    scorer: DenseVerifierScorer,
    phase_h_dir: Path,
    *,
    control_family: str = "frozen_geometry",
) -> list[dict[str, Any]]:
    image_path = Path(source["image_path"])
    if file_sha256(image_path) != source["official_image_sha256"]:
        raise ValueError("ARISE dense oracle source image hash changed")
    with Image.open(image_path) as loaded:
        image = loaded.convert("RGB")
    letterboxed, content_box = letterbox_image(image, 448)
    if list(content_box) != source["score_free_audit"]["content_box"]:
        raise ValueError("ARISE dense oracle preprocessing geometry changed")
    mask_path = phase_h_dir / "expert_masks" / source["mask_file"]
    if file_sha256(mask_path) != source["mask_sha256"]:
        raise ValueError("ARISE dense oracle expert mask hash changed")
    with np.load(mask_path, allow_pickle=False) as payload:
        expert = payload["expert_mask"].astype(bool)
        content = payload["content_mask"].astype(bool)
    explanation = reconstruct_phase_h_explanation_mask(
        reference["geometry"]["E_vs_C_E"]["target"],
        content_mask=content,
    )
    control_certificates: dict[str, Any]
    if control_family == "frozen_geometry":
        controls, control_certificates = build_target_specific_controls(
            expert,
            explanation,
            content_mask=content,
            seed_key=f"chexlocalize-qwen35-development:{source['sample_id']}",
        )
        if not geometry_matches(controls["C_X"], reference["geometry"]["X_vs_C_X"]["control"], content):
            raise ValueError("ARISE dense oracle failed to reconstruct frozen C_X")
        if not geometry_matches(controls["C_E"], reference["geometry"]["E_vs_C_E"]["control"], content):
            raise ValueError("ARISE dense oracle failed to reconstruct frozen C_E")
    elif control_family == "stat_matched_connected":
        grayscale = np.asarray(letterboxed.convert("L"), dtype=np.float64)
        control_x, certificate_x = deterministic_stat_matched_connected_control_mask(
            grayscale,
            expert,
            content,
            seed_key=f"arise-stat-matched:X:{source['sample_id']}",
            forbidden_mask=explanation,
        )
        control_e, certificate_e = deterministic_stat_matched_connected_control_mask(
            grayscale,
            explanation,
            content,
            seed_key=f"arise-stat-matched:E:{source['sample_id']}",
            forbidden_mask=expert,
        )
        controls = {"C_X": control_x, "C_E": control_e}
        control_certificates = {"C_X": certificate_x, "C_E": certificate_e}
    else:
        raise ValueError("unknown ARISE control family")
    masks = {"X": expert, "C_X": controls["C_X"], "E": explanation, "C_E": controls["C_E"]}
    operators: dict[str, Callable[[Image.Image, np.ndarray, np.ndarray], Image.Image]] = {
        "local_mean_ring8": lambda src, mask, valid: replace_with_local_ring_mean(src, mask, valid, ring_width=8),
        "masked_gaussian_blur_sigma8": lambda src, mask, valid: replace_with_masked_gaussian_blur(src, mask, valid, sigma=8.0, truncate=3.0),
    }
    thresholds = {key: 1.0 for key in ("max_normalized_centroid_distance", "max_log_perimeter_ratio", "max_masked_l1_difference", "max_masked_rms_difference", "max_ssim_difference", "max_edge_difference")}
    statement_id = source["canonical_statement_id"]
    statement_text = source["statement_text"]
    original_score = scorer.score([letterboxed], statement_id=statement_id, statement_text=statement_text)[0]
    original_array = np.asarray(letterboxed.convert("RGB"))
    output = []
    for operator_id, operator in operators.items():
        variants = {role: operator(letterboxed, mask, content) for role, mask in masks.items()}
        variant_scores = scorer.score(
            [variants[role] for role in ("X", "C_X", "E", "C_E")],
            statement_id=statement_id,
            statement_text=statement_text,
        )
        strength = {
            role: intervention_strength_metrics(
                original_array,
                np.asarray(variants[role]),
                intervention_mask=masks[role],
                content_mask=content,
            )
            for role in masks
        }
        row = build_precomputed_audit_row(
            identity={
                "row_id": f"arise-dense-{control_family}-{source['sample_id'][:20]}-{operator_id}",
                "dataset_role": "development",
                "patient_id": source["patient_id_hash"],
                "image_id": source["image_id_hash"],
                "pathology_id": statement_id,
                "model_id": oracle_model_id_for_head(scorer.identity()["head"]),
                "explanation_id": "phase_h_occlusion_top1_reconstructed",
                "operator_id": operator_id,
            },
            scores={"s0": original_score, **dict(zip(("sX", "sCX", "sE", "sCE"), variant_scores, strict=True))},
            expert_mask=expert,
            explanation_mask=explanation,
            expert_control_mask=controls["C_X"],
            explanation_control_mask=controls["C_E"],
            content_mask=content,
            strength_metrics=strength,
            strength_thresholds=thresholds,
            explanation_map=explanation.astype(np.float32),
        )
        row["patient_level_claim"] = True
        row["cluster_unit_type"] = "patient_id_hash"
        row["source_split"] = "publisher_validation_prior_exposed_development"
        row["control_family"] = control_family
        row["control_family_version"] = (
            STAT_MATCHED_CONTROL_VERSION
            if control_family == "stat_matched_connected"
            else "phase_h_frozen_geometry_control_v1"
        )
        row["control_certificates"] = control_certificates
        output.append(row)
    return output


if __name__ == "__main__":
    raise SystemExit(main())
