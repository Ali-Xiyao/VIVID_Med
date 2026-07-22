"""Fit independent Qwen3.5-2B validity critics and global V0 verifiers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arise_cxr.box_supervision import boxes_to_patch_mask  # noqa: E402
from vicer_cxr.validity import (  # noqa: E402
    VALIDITY_FINDINGS,
    canonical_sha256,
    file_sha256,
    linear_margin,
    validate_v0_manifest,
)


def normalized_tokens(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    tokens = payload["patch_tokens"].float()
    tokens = torch.nn.functional.layer_norm(tokens, (tokens.shape[-1],))
    return tokens.numpy().astype(np.float64), payload["valid_mask"].bool().numpy()


def extract_feature(row: dict[str, Any], payload: dict[str, Any], *, local: bool) -> np.ndarray:
    tokens, valid = normalized_tokens(payload)
    if local:
        mask = boxes_to_patch_mask(
            row["roi_boxes"],
            native_width=int(row["native_columns"]),
            native_height=int(row["native_rows"]),
            grid_hw=tuple(payload["grid_hw"]),
            valid_mask=payload["valid_mask"],
        ).numpy()
    else:
        mask = valid
    selected = tokens[mask & valid]
    if selected.size == 0:
        raise ValueError("VICER head feature has no valid patch token")
    return selected.mean(axis=0)


def fit_head(features: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray | float]:
    scaler = StandardScaler().fit(features)
    normalized = scaler.transform(features)
    estimator = LogisticRegression(
        C=0.1,
        solver="liblinear",
        class_weight="balanced",
        random_state=20260722,
        max_iter=2000,
    ).fit(normalized, labels)
    return {
        "scaler_mean": scaler.mean_.astype(np.float64),
        "scaler_scale": scaler.scale_.astype(np.float64),
        "weight": estimator.coef_[0].astype(np.float64),
        "intercept": float(estimator.intercept_[0]),
    }


def save_head(path: Path, model: dict[str, np.ndarray | float], metadata: dict[str, Any]) -> None:
    np.savez_compressed(
        path,
        scaler_mean=model["scaler_mean"],
        scaler_scale=model["scaler_scale"],
        weight=model["weight"],
        intercept=np.asarray([model["intercept"]], dtype=np.float64),
        metadata_json=np.asarray([json.dumps(metadata, sort_keys=True, ensure_ascii=False)]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-lock", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-critic-auroc", type=float, default=0.60)
    parser.add_argument("--minimum-verifier-auroc", type=float, default=0.60)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    validate_v0_manifest(rows)
    data_lock = json.loads(args.data_lock.read_text(encoding="utf-8"))
    cache_lock_path = args.cache_dir / "cache_lock.json"
    cache_lock = json.loads(cache_lock_path.read_text(encoding="utf-8"))
    if data_lock["manifest_sha256"] != file_sha256(args.manifest):
        raise ValueError("VICER head manifest/data lock mismatch")
    if cache_lock.get("status") != "complete" or cache_lock.get("manifest_sha256") != file_sha256(args.manifest):
        raise ValueError("VICER head cache lock mismatch")
    index_rows = [
        json.loads(line)
        for line in (args.cache_dir / "index.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    index = {str(row["sample_id"]): row for row in index_rows}
    if set(index) != {str(row["sample_id"]) for row in rows}:
        raise ValueError("VICER head cache index identity mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload_cache: dict[str, dict[str, Any]] = {}

    def payload_for(row: dict[str, Any]) -> dict[str, Any]:
        image_hash = str(row["image_sha256"])
        if image_hash not in payload_cache:
            item = args.cache_dir / index[str(row["sample_id"])]["cache_file"]
            payload = torch.load(item, map_location="cpu", weights_only=False)
            if payload.get("image_sha256") != image_hash:
                raise ValueError("VICER cached item image identity mismatch")
            payload_cache[image_hash] = payload
        return payload_cache[image_hash]

    results: dict[str, Any] = {}
    for finding in VALIDITY_FINDINGS:
        results[finding] = {}
        for head_name, train_role, calibration_role, local in (
            ("critic", "critic_train", "critic_calibration", True),
            ("verifier", "verifier_train", "verifier_calibration", False),
        ):
            train_rows = [
                row for row in rows
                if row["canonical_statement_id"] == finding and row["v0_role"] == train_role
            ]
            calibration_rows = [
                row for row in rows
                if row["canonical_statement_id"] == finding and row["v0_role"] == calibration_role
            ]
            train_features = np.stack([extract_feature(row, payload_for(row), local=local) for row in train_rows])
            train_labels = np.asarray([int(row["binary_label"]) for row in train_rows])
            calibration_features = np.stack(
                [extract_feature(row, payload_for(row), local=local) for row in calibration_rows]
            )
            calibration_labels = np.asarray([int(row["binary_label"]) for row in calibration_rows])
            model = fit_head(train_features, train_labels)
            calibration_margin = linear_margin(calibration_features, model)
            auroc = float(roc_auc_score(calibration_labels, calibration_margin))
            path = args.output_dir / f"{finding}_{head_name}.npz"
            metadata = {
                "schema_version": "vicer-v0-independent-head-v1",
                "finding": finding,
                "head": head_name,
                "feature": "qwen35_2b_layernorm_roi_mean" if local else "qwen35_2b_layernorm_global_mean",
                "train_role": train_role,
                "calibration_role": calibration_role,
                "train_records": len(train_rows),
                "calibration_records": len(calibration_rows),
                "calibration_auroc": auroc,
                "target_effect_used_for_training": False,
                "data_lock_canonical_sha256": data_lock["canonical_sha256"],
                "cache_lock_canonical_sha256": cache_lock["canonical_sha256"],
            }
            save_head(path, model, metadata)
            results[finding][head_name] = {**metadata, "file": path.name}

    realism_rows = [row for row in rows if row["v0_role"] != "validity_eval"]
    realism_features = np.stack(
        [extract_feature(row, payload_for(row), local=False) for row in realism_rows]
    )
    components = min(16, len(realism_rows) - 1, realism_features.shape[1])
    pca = PCA(n_components=components, svd_solver="full", random_state=20260722)
    projected = pca.fit_transform(realism_features)
    scales = np.sqrt(np.maximum(pca.explained_variance_, np.finfo(np.float64).eps))
    distances = np.sqrt(np.mean((projected / scales) ** 2, axis=1))
    realism_path = args.output_dir / "realism_reference.npz"
    np.savez_compressed(
        realism_path,
        mean=pca.mean_.astype(np.float64),
        components=pca.components_.astype(np.float64),
        scales=scales.astype(np.float64),
        distance_q95=np.asarray([np.quantile(distances, 0.95)], dtype=np.float64),
        metadata_json=np.asarray([
            json.dumps(
                {
                    "schema_version": "vicer-v0-realism-reference-v1",
                    "records": len(realism_rows),
                    "components": components,
                    "evaluation_rows_used": False,
                    "target_verifier_effect_used": False,
                },
                sort_keys=True,
            )
        ]),
    )

    minimum_critic = min(results[finding]["critic"]["calibration_auroc"] for finding in VALIDITY_FINDINGS)
    minimum_verifier = min(results[finding]["verifier"]["calibration_auroc"] for finding in VALIDITY_FINDINGS)
    lock = {
        "schema_version": "vicer-v0-independent-head-lock-v1",
        "status": "complete",
        "data_lock_canonical_sha256": data_lock["canonical_sha256"],
        "cache_lock_canonical_sha256": cache_lock["canonical_sha256"],
        "findings": results,
        "minimum_calibration_auroc": {
            "critic": float(minimum_critic),
            "verifier": float(minimum_verifier),
        },
        "thresholds": {
            "minimum_critic_auroc": float(args.minimum_critic_auroc),
            "minimum_verifier_auroc": float(args.minimum_verifier_auroc),
        },
        "head_gate_pass": bool(
            minimum_critic >= args.minimum_critic_auroc
            and minimum_verifier >= args.minimum_verifier_auroc
        ),
        "validity_independence": {
            "critic_role_disjoint_from_verifier": True,
            "evaluation_role_disjoint_from_both": True,
            "critic_uses_target_verifier_effect": False,
        },
        "realism_reference": {
            "file": realism_path.name,
            "file_sha256": file_sha256(realism_path),
            "training_records": len(realism_rows),
            "evaluation_rows_used": False,
        },
    }
    for finding in VALIDITY_FINDINGS:
        for head_name in ("critic", "verifier"):
            entry = lock["findings"][finding][head_name]
            entry["file_sha256"] = file_sha256(args.output_dir / entry["file"])
    lock["canonical_sha256"] = canonical_sha256(lock)
    (args.output_dir / "head_lock.json").write_text(
        json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(lock, indent=2, ensure_ascii=False))
    if not lock["head_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
