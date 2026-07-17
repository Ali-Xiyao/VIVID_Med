"""Run the frozen pooled-token logistic B0 baseline on a locked cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bives_cxr.expert_sc import file_sha256  # noqa: E402
from bives_cxr.polarity import CachedSCDataset  # noqa: E402
from bives_cxr.polarity_metrics import polarity_metrics  # noqa: E402


def pooled_features(dataset: CachedSCDataset) -> tuple[np.ndarray, list[dict]]:
    features = []
    rows = []
    for index in range(len(dataset)):
        item = dataset[index]
        features.append(
            item["patch_tokens"][item["valid_mask"]].mean(dim=0).numpy()
        )
        rows.append(item)
    return np.asarray(features, dtype=np.float32), rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--target-specificity", type=float, default=0.9)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "metrics_final.json").exists():
        raise FileExistsError(args.output_dir / "metrics_final.json")
    train_dataset = CachedSCDataset(args.cache_dir, "train")
    val_dataset = CachedSCDataset(args.cache_dir, "val")
    train_x, train_rows = pooled_features(train_dataset)
    val_x, val_rows = pooled_features(val_dataset)
    findings = sorted({row["canonical_statement_id"] for row in train_rows})
    predictions: list[dict] = []
    models: dict[str, dict] = {}
    for finding in findings:
        train_indices = [i for i, row in enumerate(train_rows) if row["canonical_statement_id"] == finding]
        val_indices = [i for i, row in enumerate(val_rows) if row["canonical_statement_id"] == finding]
        scaler = StandardScaler().fit(train_x[train_indices])
        model = LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver="lbfgs",
            random_state=args.seed,
        ).fit(
            scaler.transform(train_x[train_indices]),
            np.asarray([train_rows[i]["binary_label"] for i in train_indices]),
        )
        scores = model.predict_proba(scaler.transform(val_x[val_indices]))[:, 1]
        for index, score in zip(val_indices, scores, strict=True):
            predictions.append(
                {
                    "sample_id": val_rows[index]["sample_id"],
                    "unit_id": val_rows[index]["unit_id"],
                    "patient_id": val_rows[index]["patient_id"],
                    "canonical_statement_id": finding,
                    "binary_label": val_rows[index]["binary_label"],
                    "support_probability": float(score),
                }
            )
        model_path = args.output_dir / f"{finding}_model.npz"
        np.savez_compressed(
            model_path,
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
            coefficient=model.coef_,
            intercept=model.intercept_,
        )
        models[finding] = {"file": model_path.name, "sha256": file_sha256(model_path)}
    metrics, thresholds = polarity_metrics(
        predictions,
        target_specificity=args.target_specificity,
    )
    with (args.output_dir / "val_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in sorted(predictions, key=lambda item: item["sample_id"]):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    threshold_payload = {
        finding: {
            "support_probability_threshold": threshold,
            "target_specificity": args.target_specificity,
            "source": "weak_sc_validation_only",
        }
        for finding, threshold in thresholds.items()
    }
    (args.output_dir / "locked_thresholds.json").write_text(
        json.dumps(threshold_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    result = {
        "variant": "B0_frozen_pooled_logistic",
        "formal_result": False,
        "cache_lock_sha256": file_sha256(args.cache_dir / "cache_lock.json"),
        "seed": args.seed,
        "models": models,
        "validation": metrics,
        "threshold_source": "weak_sc_validation_only",
        "external_test_used_for_selection": False,
    }
    (args.output_dir / "metrics_final.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
