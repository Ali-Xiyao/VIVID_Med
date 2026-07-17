"""Diagnose nonclinical proxy support/contradict data with frozen Qwen3.5 vision features."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bives_cxr.backbones import Qwen35VisionAdapter, load_qwen35_visual_and_processor


DEFAULT_FINDINGS = ("atelectasis", "consolidation", "pulmonary_edema")
SC_STATES = ("support", "contradict")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-candidates", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--feature-cache-input", type=Path)
    parser.add_argument("--findings", nargs="+", default=list(DEFAULT_FINDINGS))
    parser.add_argument("--max-patients-per-state", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--attention-implementation", choices=("eager", "sdpa"), default="eager")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def stable_key(row: dict[str, Any], seed: int) -> str:
    token = f"{seed}|{row['candidate_id']}"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def summarize_candidates(
    rows: list[dict[str, Any]], findings: list[str]
) -> dict[str, Any]:
    candidate_ids = [str(row.get("candidate_id", "")) for row in rows]
    summary: dict[str, Any] = {
        "records": len(rows),
        "unique_candidate_ids": len(set(candidate_ids)),
        "duplicate_candidate_ids": len(candidate_ids) - len(set(candidate_ids)),
        "findings": {},
    }
    for finding in findings:
        finding_rows = [
            row
            for row in rows
            if row.get("canonical_statement_id") == finding
            and row.get("parser_status") == "candidate"
            and row.get("parser_state_candidate") in SC_STATES
        ]
        state_payload: dict[str, Any] = {}
        patient_sets: dict[str, set[str]] = {}
        for state in SC_STATES:
            state_rows = [
                row for row in finding_rows if row.get("parser_state_candidate") == state
            ]
            patients = {str(row["patient_id"]) for row in state_rows}
            patient_sets[state] = patients
            state_payload[state] = {
                "records": len(state_rows),
                "patients": len(patients),
                "studies": len({str(row["study_id"]) for row in state_rows}),
                "images": len({str(row["image_path"]) for row in state_rows}),
                "reports": len({str(row["report_sha256"]) for row in state_rows}),
                "rows_per_report": (
                    len(state_rows)
                    / max(1, len({str(row["report_sha256"]) for row in state_rows}))
                ),
                "cue_counts": dict(
                    sorted(Counter(str(row["parser_cue"]) for row in state_rows).items())
                ),
            }
        summary["findings"][finding] = {
            "states": state_payload,
            "patients_with_both_sc_across_studies": len(
                patient_sets["support"] & patient_sets["contradict"]
            ),
        }
    return summary


def summarize_manifests(
    train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]], findings: list[str]
) -> dict[str, Any]:
    payload: dict[str, Any] = {"splits": {}}
    for split, rows in (("train", train_rows), ("val", val_rows)):
        payload["splits"][split] = {
            "records": len(rows),
            "patients": len({str(row["patient_id"]) for row in rows}),
            "groups": len({str(row["group_id"]) for row in rows}),
            "by_finding_state": {
                finding: {
                    state: sum(
                        row.get("canonical_statement_id") == finding
                        and row.get("state") == state
                        for row in rows
                    )
                    for state in ("support", "contradict", "uncertain", "insufficient")
                }
                for finding in findings
            },
        }
    train_patients = {str(row["patient_id"]) for row in train_rows}
    val_patients = {str(row["patient_id"]) for row in val_rows}
    train_hashes = {str(row["image_sha256"]) for row in train_rows}
    val_hashes = {str(row["image_sha256"]) for row in val_rows}
    payload["patient_overlap"] = len(train_patients & val_patients)
    payload["image_hash_overlap"] = len(train_hashes & val_hashes)
    payload["validation_sc_examples_per_finding"] = {
        finding: sum(
            row.get("canonical_statement_id") == finding and row.get("state") in SC_STATES
            for row in val_rows
        )
        for finding in findings
    }
    return payload


def _one_row_per_patient(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: stable_key(item, seed)):
        selected.setdefault(str(row["patient_id"]), row)
    return list(selected.values())


def select_balanced_sc(
    rows: list[dict[str, Any]],
    findings: list[str],
    max_patients_per_state: int,
    seed: int,
) -> list[dict[str, Any]]:
    if max_patients_per_state < 2:
        raise ValueError("max_patients_per_state must be at least 2")
    selected: list[dict[str, Any]] = []
    for finding in findings:
        pools = {
            state: _one_row_per_patient(
                [
                    row
                    for row in rows
                    if row.get("canonical_statement_id") == finding
                    and row.get("parser_status") == "candidate"
                    and row.get("parser_state_candidate") == state
                ],
                seed,
            )
            for state in SC_STATES
        }
        contradict = sorted(pools["contradict"], key=lambda row: stable_key(row, seed))[
            :max_patients_per_state
        ]
        contradict_patients = {str(row["patient_id"]) for row in contradict}
        support = [
            row
            for row in sorted(pools["support"], key=lambda row: stable_key(row, seed))
            if str(row["patient_id"]) not in contradict_patients
        ][:max_patients_per_state]
        count = min(len(support), len(contradict))
        if count < 2:
            raise ValueError(f"{finding} has only {count} patient-disjoint S/C pairs")
        selected.extend(support[:count])
        selected.extend(contradict[:count])
    return sorted(
        selected,
        key=lambda row: (
            str(row["canonical_statement_id"]),
            str(row["parser_state_candidate"]),
            stable_key(row, seed),
        ),
    )


def leave_one_out_centroid_metrics(
    features: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("features/labels shape mismatch")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("labels must contain both contradict=0 and support=1")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / np.clip(norms, 1e-12, None)
    scores: list[float] = []
    nearest_correct: list[float] = []
    for index, feature in enumerate(normalized):
        keep = np.arange(len(labels)) != index
        support_centroid = normalized[keep & (labels == 1)].mean(axis=0)
        contradict_centroid = normalized[keep & (labels == 0)].mean(axis=0)
        support_centroid /= max(float(np.linalg.norm(support_centroid)), 1e-12)
        contradict_centroid /= max(float(np.linalg.norm(contradict_centroid)), 1e-12)
        scores.append(float(feature @ support_centroid - feature @ contradict_centroid))
        similarities = normalized[keep] @ feature
        neighbor_index = np.arange(len(labels))[keep][int(np.argmax(similarities))]
        nearest_correct.append(float(labels[neighbor_index] == labels[index]))
    score_array = np.asarray(scores)
    return {
        "loo_centroid_auroc": float(roc_auc_score(labels, score_array)),
        "loo_centroid_accuracy": float(np.mean((score_array >= 0) == labels)),
        "nearest_neighbor_same_state": float(np.mean(nearest_correct)),
    }


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    value = 0.0
    for index in range(bins):
        if index + 1 == bins:
            selected = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            selected = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        if not selected.any():
            continue
        value += float(selected.mean()) * abs(
            float(labels[selected].mean()) - float(probabilities[selected].mean())
        )
    return float(value)


def logistic_probe_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    patient_ids: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    """Fixed patient-group-disjoint logistic probe without model selection."""

    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("features/labels shape mismatch")
    if patient_ids.shape != labels.shape:
        raise ValueError("patient_ids/labels shape mismatch")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("labels must contain both contradict=0 and support=1")
    unique_groups_by_class = [
        len(set(patient_ids[labels == class_index].tolist())) for class_index in (0, 1)
    ]
    folds = min(5, *unique_groups_by_class)
    if folds < 2:
        raise ValueError("logistic probe requires at least two patient groups per class")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    probabilities = np.full(len(labels), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(features, labels, groups=patient_ids)
    ):
        if set(patient_ids[train_indices]) & set(patient_ids[test_indices]):
            raise RuntimeError("patient leakage in logistic probe")
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, solver="liblinear", random_state=seed),
        )
        estimator.fit(features[train_indices], labels[train_indices])
        fold_probabilities = estimator.predict_proba(features[test_indices])[:, 1]
        probabilities[test_indices] = fold_probabilities
        fold_rows.append(
            {
                "fold": fold,
                "train_records": len(train_indices),
                "test_records": len(test_indices),
                "train_patients": len(set(patient_ids[train_indices].tolist())),
                "test_patients": len(set(patient_ids[test_indices].tolist())),
            }
        )
    if not np.isfinite(probabilities).all():
        raise RuntimeError("logistic probe did not produce complete out-of-fold predictions")
    predictions = (probabilities >= 0.5).astype(np.int64)
    full_estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="liblinear", random_state=seed),
    )
    full_estimator.fit(features, labels)
    logistic = full_estimator.named_steps["logisticregression"]
    return {
        "protocol": "fixed_c1_standardized_stratified_patient_group_cv",
        "folds": folds,
        "auroc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "nll": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece_10bin": expected_calibration_error(probabilities, labels, bins=10),
        "full_fit_intercept": float(logistic.intercept_[0]),
        "full_fit_coefficient_l2": float(np.linalg.norm(logistic.coef_)),
        "fold_details": fold_rows,
    }


def load_cached_features(
    path: Path, rows: list[dict[str, Any]]
) -> np.ndarray:
    cache = np.load(path, allow_pickle=False)
    required = {"features", "candidate_ids", "findings", "states", "patient_ids"}
    missing = required - set(cache.files)
    if missing:
        raise ValueError(f"feature cache missing keys: {sorted(missing)}")
    candidate_ids = [str(value) for value in cache["candidate_ids"].tolist()]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("feature cache has duplicate candidate IDs")
    index_by_id = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
    selected_features: list[np.ndarray] = []
    for row in rows:
        candidate_id = str(row["candidate_id"])
        if candidate_id not in index_by_id:
            raise ValueError(f"feature cache missing selected candidate {candidate_id}")
        index = index_by_id[candidate_id]
        expected = (
            str(row["canonical_statement_id"]),
            str(row["parser_state_candidate"]),
            str(row["patient_id"]),
        )
        actual = (
            str(cache["findings"][index]),
            str(cache["states"][index]),
            str(cache["patient_ids"][index]),
        )
        if actual != expected:
            raise ValueError(
                f"feature cache metadata mismatch for {candidate_id}: {actual} != {expected}"
            )
        selected_features.append(cache["features"][index].astype(np.float32, copy=False))
    return np.stack(selected_features)


def extract_features(
    rows: list[dict[str, Any]],
    model_path: Path,
    device: torch.device,
    dtype: str,
    attention_implementation: str,
) -> np.ndarray:
    visual, processor, config = load_qwen35_visual_and_processor(
        model_path,
        dtype=dtype,
        attention_implementation=attention_implementation,
    )
    visual = visual.to(device).eval()
    adapter = Qwen35VisionAdapter(
        visual,
        spatial_merge_size=int(config["vision_config"]["spatial_merge_size"]),
    )
    features: list[np.ndarray] = []
    try:
        for row in rows:
            with Image.open(Path(str(row["image_path"]))) as loaded:
                image = ImageOps.exif_transpose(loaded).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": str(row["statement_text"])},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            batch = processor(text=[text], images=[image], return_tensors="pt", padding=True)
            with torch.inference_mode():
                patches = adapter(
                    batch["pixel_values"].to(
                        device=device, dtype=next(visual.parameters()).dtype
                    ),
                    batch["image_grid_thw"].to(device),
                )
                valid = patches.valid_mask[0]
                pooled = patches.tokens[0, valid].float().mean(dim=0)
            if not bool(torch.isfinite(pooled).all()):
                raise RuntimeError(f"non-finite feature for {row['candidate_id']}")
            features.append(pooled.cpu().numpy())
    finally:
        del adapter, visual
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.stack(features)


def feature_diagnostic(
    rows: list[dict[str, Any]], features: np.ndarray, findings: list[str], seed: int
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for finding in findings:
        indices = [
            index
            for index, row in enumerate(rows)
            if row["canonical_statement_id"] == finding
        ]
        finding_features = features[indices]
        labels = np.asarray(
            [1 if rows[index]["parser_state_candidate"] == "support" else 0 for index in indices]
        )
        patient_ids = np.asarray([str(rows[index]["patient_id"]) for index in indices])
        metrics = leave_one_out_centroid_metrics(finding_features, labels)
        metrics["logistic_probe"] = logistic_probe_metrics(
            finding_features, labels, patient_ids, seed=seed
        )
        metrics.update(
            {
                "samples": len(indices),
                "support_patients": int(labels.sum()),
                "contradict_patients": int((labels == 0).sum()),
            }
        )
        payload[finding] = metrics
    return payload


def global_logistic_probe(
    rows: list[dict[str, Any]], features: np.ndarray, seed: int
) -> dict[str, Any]:
    labels = np.asarray(
        [1 if row["parser_state_candidate"] == "support" else 0 for row in rows]
    )
    patient_ids = np.asarray([str(row["patient_id"]) for row in rows])
    return logistic_probe_metrics(features, labels, patient_ids, seed)


def root_cause_decision(
    candidate_summary: dict[str, Any],
    manifest_summary: dict[str, Any],
    feature_metrics: dict[str, Any],
) -> dict[str, Any]:
    validation_counts = manifest_summary["validation_sc_examples_per_finding"]
    tiny_validation = any(count < 10 for count in validation_counts.values())
    aurocs = {
        finding: metrics["loo_centroid_auroc"]
        for finding, metrics in feature_metrics.items()
    }
    logistic_aurocs = {
        finding: metrics["logistic_probe"]["auroc"]
        for finding, metrics in feature_metrics.items()
    }
    low_feature_separability = [finding for finding, value in aurocs.items() if value < 0.65]
    candidate_imbalance = {
        finding: (
            details["states"]["support"]["patients"],
            details["states"]["contradict"]["patients"],
        )
        for finding, details in candidate_summary["findings"].items()
    }
    logistic_separable = all(value >= 0.75 for value in logistic_aurocs.values())
    if logistic_separable and tiny_validation:
        classification = "frozen_features_separable_but_proxy_validation_tiny"
    elif tiny_validation and low_feature_separability:
        classification = "tiny_split_plus_weak_label_or_visual_mismatch"
    elif tiny_validation:
        classification = "tiny_split_dominant"
    elif low_feature_separability:
        classification = "weak_label_or_visual_mismatch_dominant"
    else:
        classification = "proxy_sc_separable"
    return {
        "classification": classification,
        "tiny_validation": tiny_validation,
        "low_feature_separability_findings": low_feature_separability,
        "logistic_probe_aurocs": logistic_aurocs,
        "logistic_probe_separable": logistic_separable,
        "candidate_patient_balance_support_contradict": candidate_imbalance,
        "next_step": (
            "Run only the predeclared frozen-Qwen3.5-2B optimization-identifiability "
            "gate: fixed-budget state-only first, then the matched full objective only if "
            "state-only fits the 48-row train proxy. Do not scale to 4B/9B or sweep weights."
            if logistic_separable
            else "Do not scale or sweep losses; revisit proxy eligibility before another training run."
        ),
    }


def main() -> None:
    args = parse_args()
    if (
        args.feature_cache_input is None
        and not torch.cuda.is_available()
        and str(args.device).startswith("cuda")
    ):
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    candidates = read_jsonl(args.parsed_candidates)
    train_rows = read_jsonl(args.train_manifest)
    val_rows = read_jsonl(args.val_manifest)
    selected = select_balanced_sc(
        candidates,
        findings=args.findings,
        max_patients_per_state=args.max_patients_per_state,
        seed=args.seed,
    )
    device = torch.device(args.device)
    if args.feature_cache_input is not None:
        features = load_cached_features(args.feature_cache_input, selected)
        feature_source = {"kind": "reused_cache", "path": str(args.feature_cache_input)}
    else:
        if args.model_path is None:
            raise ValueError("--model-path is required without --feature-cache-input")
        features = extract_features(
            selected,
            model_path=args.model_path,
            device=device,
            dtype=args.dtype,
            attention_implementation=args.attention_implementation,
        )
        feature_source = {"kind": "qwen35_vision", "path": str(args.model_path)}
    candidate_summary = summarize_candidates(candidates, args.findings)
    manifest_summary = summarize_manifests(train_rows, val_rows, args.findings)
    feature_metrics = feature_diagnostic(selected, features, args.findings, args.seed)
    payload = {
        "status": "complete_nonclinical_read_only",
        "formal_result": False,
        "clinical_ground_truth": False,
        "model_path": str(args.model_path) if args.model_path is not None else None,
        "feature_source": feature_source,
        "device": str(device),
        "seed": args.seed,
        "selection": {
            "max_patients_per_state": args.max_patients_per_state,
            "records": len(selected),
            "patients": len({str(row["patient_id"]) for row in selected}),
            "candidate_ids_sha256": hashlib.sha256(
                "\n".join(str(row["candidate_id"]) for row in selected).encode("utf-8")
            ).hexdigest(),
        },
        "candidate_summary": candidate_summary,
        "manifest_summary": manifest_summary,
        "frozen_feature_metrics": feature_metrics,
        "global_logistic_probe": global_logistic_probe(selected, features, args.seed),
        "root_cause_decision": root_cause_decision(
            candidate_summary, manifest_summary, feature_metrics
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.feature_cache is not None:
        args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.feature_cache,
            features=features,
            candidate_ids=np.asarray([str(row["candidate_id"]) for row in selected]),
            findings=np.asarray([str(row["canonical_statement_id"]) for row in selected]),
            states=np.asarray([str(row["parser_state_candidate"]) for row in selected]),
            patient_ids=np.asarray([str(row["patient_id"]) for row in selected]),
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
