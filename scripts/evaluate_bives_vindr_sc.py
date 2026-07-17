"""Run locked B2 expert S/C inference on VinDr test without model selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bives_cxr.backbones import Qwen35VisionAdapter, load_qwen35_visual_and_processor  # noqa: E402
from bives_cxr.dicom import DICOM_PREPROCESS_VERSION, load_cxr_dicom  # noqa: E402
from bives_cxr.expert_sc import (  # noqa: E402
    evaluate_expert_sc_predictions,
    file_sha256,
    read_expert_sc_manifest,
)
from bives_cxr.polarity_runtime import (  # noqa: E402
    extract_qwen35_patch_batch,
    load_locked_polarity_checkpoint,
    score_statements,
)
from bives_cxr.qwen35_preprocessing import QWEN35_IMAGE_PREPROCESS_VERSION  # noqa: E402


def canonical_sha256(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def snapshot_model_files(model_root: Path) -> str:
    names = {"config.json", "model.safetensors.index.json"}
    names.update(path.name for path in model_root.glob("*.safetensors"))
    files = {
        name: file_sha256(model_root / name)
        for name in sorted(names)
        if (model_root / name).is_file()
    }
    return canonical_sha256(files)


def require_integrity_gate(path: Path) -> dict[str, Any]:
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("status") != "pass":
        raise ValueError("VinDr integrity audit is not pass")
    if int(audit.get("official_manifest_entries", 0)) != 18006:
        raise ValueError("VinDr integrity audit does not cover 18,006 official entries")
    if int(audit.get("sha256_checked", 0)) != 18006 or audit.get("sha256_mismatches"):
        raise ValueError("VinDr full SHA-256 gate is incomplete or failed")
    if int(audit.get("dicom_decode_checked", 0)) < 3000 or audit.get("dicom_decode_failures"):
        raise ValueError("VinDr full test-DICOM decode gate is incomplete or failed")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-lock", type=Path, required=True)
    parser.add_argument("--integrity-audit", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-cache-lock", type=Path, required=True)
    parser.add_argument("--locked-thresholds", type=Path, required=True)
    parser.add_argument("--b0-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    args = parser.parse_args()

    integrity = require_integrity_gate(args.integrity_audit)
    manifest_lock = json.loads(args.manifest_lock.read_text(encoding="utf-8"))
    if manifest_lock.get("manifest_sha256") != file_sha256(args.manifest):
        raise ValueError("VinDr expert manifest does not match its lock")
    if manifest_lock.get("source_split") != "test_consensus_only":
        raise ValueError("expert evaluation requires the locked consensus test split")
    training_lock = json.loads(args.training_cache_lock.read_text(encoding="utf-8"))
    if training_lock.get("status") != "complete":
        raise ValueError("training patch-token cache lock is not complete")

    device = torch.device(args.device)
    polarity_model, statement_to_index, checkpoint = load_locked_polarity_checkpoint(
        args.checkpoint, device
    )
    if checkpoint.get("variant") != "B2_sparse_exact_k":
        raise ValueError("VinDr expert evaluation is locked to B2 sparse exact-K")
    if checkpoint.get("cache_lock_sha256") != file_sha256(args.training_cache_lock):
        raise ValueError("B2 checkpoint is not bound to the supplied training cache lock")
    thresholds = json.loads(args.locked_thresholds.read_text(encoding="utf-8"))
    for finding in statement_to_index:
        if thresholds.get(finding, {}).get("source") != "weak_sc_validation_only":
            raise ValueError("thresholds must be locked on weak S/C validation only")
    b0_metrics_path = args.b0_dir / "metrics_final.json"
    b0_metrics_lock = json.loads(b0_metrics_path.read_text(encoding="utf-8"))
    if b0_metrics_lock.get("cache_lock_sha256") != file_sha256(args.training_cache_lock):
        raise ValueError("B0 models are not bound to the supplied training cache lock")
    b0_thresholds_path = args.b0_dir / "locked_thresholds.json"
    b0_thresholds = json.loads(b0_thresholds_path.read_text(encoding="utf-8"))
    b0_models = {}
    for finding in statement_to_index:
        model_record = b0_metrics_lock["models"][finding]
        model_path = args.b0_dir / model_record["file"]
        if file_sha256(model_path) != model_record["sha256"]:
            raise ValueError(f"B0 model hash mismatch: {finding}")
        b0_models[finding] = np.load(model_path)
    if snapshot_model_files(args.model_path) != training_lock.get("model_snapshot_sha256"):
        raise ValueError("Qwen3.5 model snapshot differs from the frozen training cache")

    rows = read_expert_sc_manifest(args.manifest)
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_unit[str(row["unit_id"])].append(row)
    if set(statement_to_index) != {str(row["canonical_statement_id"]) for row in rows}:
        raise ValueError("VinDr findings do not exactly match the locked B2 vocabulary")

    identity = {
        "manifest_sha256": file_sha256(args.manifest),
        "manifest_lock_sha256": file_sha256(args.manifest_lock),
        "integrity_audit_sha256": file_sha256(args.integrity_audit),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "training_cache_lock_sha256": file_sha256(args.training_cache_lock),
        "locked_thresholds_sha256": file_sha256(args.locked_thresholds),
        "b0_metrics_sha256": file_sha256(b0_metrics_path),
        "b0_locked_thresholds_sha256": file_sha256(b0_thresholds_path),
        "model_snapshot_sha256": training_lock["model_snapshot_sha256"],
        "dicom_preprocess_version": DICOM_PREPROCESS_VERSION,
        "image_preprocess_version": QWEN35_IMAGE_PREPROCESS_VERSION,
        "image_size": int(args.image_size),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "progress.json"
    completed: dict[str, list[dict[str, Any]]] = {}
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != identity:
            raise ValueError("existing VinDr inference progress has a different identity")
        completed = dict(progress.get("completed_units", {}))

    visual, processor, qwen_config = load_qwen35_visual_and_processor(
        args.model_path, dtype=args.dtype, attention_implementation="eager"
    )
    visual = visual.to(device).eval()
    adapter = Qwen35VisionAdapter(
        visual,
        spatial_merge_size=int(qwen_config["vision_config"]["spatial_merge_size"]),
    ).to(device).eval()
    pending_units = [unit_id for unit_id in sorted(by_unit) if unit_id not in completed]
    batch_size = max(1, int(args.batch_size))
    for batch_start in range(0, len(pending_units), batch_size):
        unit_ids = pending_units[batch_start : batch_start + batch_size]
        decoded = []
        for unit_id in unit_ids:
            unit_rows = sorted(by_unit[unit_id], key=lambda row: str(row["canonical_statement_id"]))
            image_path = Path(unit_rows[0]["image_path"])
            declared = str(unit_rows[0].get("official_image_sha256") or "")
            if not declared:
                raise ValueError(f"VinDr official image SHA is missing: {unit_id}")
            image, dicom_record = load_cxr_dicom(image_path)
            decoded.append((unit_id, unit_rows, image, dicom_record))
        extracted_batch = extract_qwen35_patch_batch(
            [item[2] for item in decoded],
            [str(item[1][0]["statement_text"]) for item in decoded],
            adapter=adapter,
            visual=visual,
            processor=processor,
            device=device,
            image_size=args.image_size,
        )
        for (unit_id, unit_rows, _, dicom_record), extracted in zip(
            decoded, extracted_batch, strict=True
        ):
            indices = [statement_to_index[str(row["canonical_statement_id"])] for row in unit_rows]
            output = score_statements(
                polarity_model,
                extracted["patch_tokens"],
                extracted["valid_mask"],
                indices,
            )
            pooled = extracted["patch_tokens"][extracted["valid_mask"]].mean(dim=0).cpu().numpy()
            predicted = []
            for row_index, row in enumerate(unit_rows):
                finding = str(row["canonical_statement_id"])
                b0 = b0_models[finding]
                standardized = (pooled - b0["scaler_mean"]) / b0["scaler_scale"]
                b0_logit = float(np.dot(b0["coefficient"][0], standardized) + b0["intercept"][0])
                if b0_logit >= 0:
                    b0_probability = float(1.0 / (1.0 + np.exp(-b0_logit)))
                else:
                    exp_logit = np.exp(b0_logit)
                    b0_probability = float(exp_logit / (1.0 + exp_logit))
                predicted.append(
                    {
                        "sample_id": str(row["sample_id"]),
                        "unit_id": unit_id,
                        "canonical_statement_id": finding,
                        "support_probability": float(output["support_probability"][row_index].cpu()),
                        "b0_support_probability": b0_probability,
                        "signed_evidence": float(output["signed_evidence"][row_index].cpu()),
                        "evidence_pos": float(output["evidence_pos"][row_index].cpu()),
                        "evidence_neg": float(output["evidence_neg"][row_index].cpu()),
                        "evidence_topk_indices": torch.where(output["gate"][row_index].cpu() > 0.5)[0].tolist(),
                        "grid_hw": list(extracted["grid_hw"]),
                        "dicom_rgb_sha256": dicom_record.rgb_sha256,
                    }
                )
            completed[unit_id] = predicted
        write_json_atomic(
            progress_path,
            {
                "identity": identity,
                "status": "in_progress",
                "completed_count": len(completed),
                "total_units": len(by_unit),
                "completed_units": completed,
            },
        )
        print(json.dumps({"completed_units": len(completed), "total_units": len(by_unit)}))

    prediction_rows = [row for unit_id in sorted(completed) for row in completed[unit_id]]
    predictions_path = args.output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    b2_metrics = evaluate_expert_sc_predictions(
        rows,
        prediction_rows,
        thresholds,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    b0_prediction_rows = [
        {**row, "support_probability": row["b0_support_probability"]}
        for row in prediction_rows
    ]
    b0_metrics = evaluate_expert_sc_predictions(
        rows,
        b0_prediction_rows,
        b0_thresholds,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    findings = sorted(statement_to_index)
    comparison = {
        finding: {
            "b0_auroc": b0_metrics["per_finding"][finding]["auroc"],
            "b2_auroc": b2_metrics["per_finding"][finding]["auroc"],
            "b0_auprc": b0_metrics["per_finding"][finding]["auprc"],
            "b2_auprc": b2_metrics["per_finding"][finding]["auprc"],
            "b2_not_lower_on_both": bool(
                b2_metrics["per_finding"][finding]["auroc"]
                >= b0_metrics["per_finding"][finding]["auroc"]
                and b2_metrics["per_finding"][finding]["auprc"]
                >= b0_metrics["per_finding"][finding]["auprc"]
            ),
        }
        for finding in findings
    }
    metrics = {
        "evaluation_axis": "expert_statement_polarity_sc",
        "b0_frozen_pooled": b0_metrics,
        "b2_sparse_exact_k": b2_metrics,
        "b2_vs_b0_per_finding": comparison,
        "b2_not_lower_than_b0_all_findings": all(
            row["b2_not_lower_on_both"] for row in comparison.values()
        ),
    }
    metrics.update(
        {
            "formal_result": False,
            "evaluation_only": True,
            "external_test_used_for_selection": False,
            "threshold_source": "weak_sc_validation_only",
            "checkpoint_step": int(checkpoint["step"]),
            "identity": identity,
            "integrity_status": integrity["status"],
            "predictions_sha256": file_sha256(predictions_path),
        }
    )
    write_json_atomic(args.output_dir / "metrics_final.json", metrics)
    write_json_atomic(
        progress_path,
        {
            "identity": identity,
            "status": "complete",
            "completed_count": len(completed),
            "total_units": len(by_unit),
            "completed_units": completed,
            "metrics_final": "metrics_final.json",
        },
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
