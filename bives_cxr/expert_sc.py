"""Independent expert support/contradict dataset and evaluation contracts."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from torch.utils.data import Dataset

from .dicom import load_cxr_dicom


EXPERT_SC_REQUIRED_FIELDS = {
    "sample_id",
    "unit_id",
    "image_path",
    "canonical_statement_id",
    "statement_text",
    "binary_label",
    "state",
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_expert_sc_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("expert S/C manifest is empty")
    sample_ids: set[str] = set()
    unit_finding: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        missing = EXPERT_SC_REQUIRED_FIELDS - set(row)
        if missing:
            raise ValueError(f"expert S/C row {index} is missing {sorted(missing)}")
        sample_id = str(row["sample_id"])
        key = (str(row["unit_id"]), str(row["canonical_statement_id"]))
        if not sample_id or sample_id in sample_ids:
            raise ValueError(f"expert S/C sample_id must be unique and non-empty: {sample_id!r}")
        if key in unit_finding:
            raise ValueError(f"duplicate expert S/C unit/finding pair: {key}")
        label = int(row["binary_label"])
        expected_state = "support" if label == 1 else "contradict"
        if label not in {0, 1} or str(row["state"]) != expected_state:
            raise ValueError(f"invalid expert S/C label/state for {sample_id}")
        sample_ids.add(sample_id)
        unit_finding.add(key)
    return rows


class ExpertSCDataset(Dataset):
    """Expert binary polarity rows with no quartet, U/I, or patient contract."""

    def __init__(self, manifest_path: str | Path, *, verify_sha256: bool = False) -> None:
        self.manifest_path = Path(manifest_path)
        self.rows = read_expert_sc_manifest(self.manifest_path)
        self.verify_sha256 = bool(verify_sha256)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = Path(row["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if self.verify_sha256:
            expected = str(row.get("official_image_sha256", ""))
            if not expected or file_sha256(image_path) != expected:
                raise ValueError(f"expert S/C image SHA-256 mismatch: {image_path}")
        if image_path.suffix.lower() in {".dicom", ".dcm"}:
            image, preprocess = load_cxr_dicom(image_path)
            preprocess_record: dict[str, Any] | None = preprocess.to_dict()
        else:
            with Image.open(image_path) as source:
                image = source.convert("RGB").copy()
            preprocess_record = None
        return {
            "sample_id": str(row["sample_id"]),
            "unit_id": str(row["unit_id"]),
            "canonical_statement_id": str(row["canonical_statement_id"]),
            "statement_text": str(row["statement_text"]),
            "binary_label": int(row["binary_label"]),
            "bounding_boxes": list(row.get("bounding_boxes", [])),
            "image": image,
            "image_path": str(image_path),
            "dicom_preprocess": preprocess_record,
        }


def _metric_block(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("expert S/C metrics require both support and contradict labels")
    clipped = np.clip(scores.astype(np.float64), 1e-7, 1.0 - 1e-7)
    predictions = clipped >= float(threshold)
    positives = labels == 1
    negatives = labels == 0
    return {
        "auroc": float(roc_auc_score(labels, clipped)),
        "auprc": float(average_precision_score(labels, clipped)),
        "nll": float(log_loss(labels, clipped, labels=[0, 1])),
        "brier": float(np.mean((clipped - labels) ** 2)),
        "prevalence": float(labels.mean()),
        "support_probability_threshold": float(threshold),
        "sensitivity_at_locked_threshold": float(predictions[positives].mean()),
        "specificity_at_locked_threshold": float((~predictions[negatives]).mean()),
    }


def _percentile_interval(values: list[float]) -> dict[str, float] | None:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return None
    lower, upper = np.percentile(np.asarray(finite), [2.5, 97.5])
    return {"lower": float(lower), "upper": float(upper)}


def evaluate_expert_sc_predictions(
    manifest_rows: Iterable[dict[str, Any]],
    prediction_rows: Iterable[dict[str, Any]],
    thresholds: dict[str, dict[str, float]],
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 17,
) -> dict[str, Any]:
    """Evaluate locked external S/C scores with image-clustered bootstrap CIs."""

    manifest = list(manifest_rows)
    predictions = list(prediction_rows)
    expected = {str(row["sample_id"]): row for row in manifest}
    observed: dict[str, dict[str, Any]] = {}
    for row in predictions:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in observed:
            raise ValueError(f"prediction sample_id must be unique and non-empty: {sample_id!r}")
        score = float(row["support_probability"])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"support_probability must be finite in [0, 1]: {sample_id}")
        observed[sample_id] = row
    if set(observed) != set(expected):
        raise ValueError(
            "expert S/C predictions must have exact manifest coverage; "
            f"missing={len(set(expected) - set(observed))}, "
            f"unexpected={len(set(observed) - set(expected))}"
        )

    aligned = [
        {
            "sample_id": sample_id,
            "unit_id": str(row["unit_id"]),
            "finding": str(row["canonical_statement_id"]),
            "label": int(row["binary_label"]),
            "score": float(observed[sample_id]["support_probability"]),
        }
        for sample_id, row in expected.items()
    ]
    findings = sorted({row["finding"] for row in aligned})
    missing_thresholds = set(findings) - set(thresholds)
    if missing_thresholds:
        raise ValueError(
            f"locked development thresholds are missing for {sorted(missing_thresholds)}"
        )

    def metrics_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
        per_finding: dict[str, dict[str, float]] = {}
        for finding in findings:
            subset = [row for row in rows if row["finding"] == finding]
            threshold = float(thresholds[finding]["support_probability_threshold"])
            block = _metric_block(
                np.asarray([row["label"] for row in subset], dtype=np.int64),
                np.asarray([row["score"] for row in subset], dtype=np.float64),
                threshold,
            )
            block["records"] = len(subset)
            block["target_specificity"] = float(thresholds[finding]["target_specificity"])
            per_finding[finding] = block
        macro_keys = ("auroc", "auprc", "nll", "brier")
        macro = {
            key: float(np.mean([per_finding[finding][key] for finding in findings]))
            for key in macro_keys
        }
        return {"per_finding": per_finding, "macro": macro}

    point = metrics_for_rows(aligned)
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aligned:
        by_unit[row["unit_id"]].append(row)
    unit_ids = sorted(by_unit)
    rng = np.random.default_rng(bootstrap_seed)
    replicates: list[dict[str, Any]] = []
    for _ in range(int(bootstrap_replicates)):
        sampled_units = rng.choice(unit_ids, size=len(unit_ids), replace=True)
        sampled = [row for unit_id in sampled_units for row in by_unit[str(unit_id)]]
        try:
            replicates.append(metrics_for_rows(sampled))
        except ValueError:
            continue
    ci: dict[str, Any] = {"per_finding": {}, "macro": {}}
    for finding in findings:
        ci["per_finding"][finding] = {
            key: _percentile_interval(
                [replicate["per_finding"][finding][key] for replicate in replicates]
            )
            for key in ("auroc", "auprc", "nll", "brier", "sensitivity_at_locked_threshold")
        }
    ci["macro"] = {
        key: _percentile_interval([replicate["macro"][key] for replicate in replicates])
        for key in ("auroc", "auprc", "nll", "brier")
    }
    return {
        "evaluation_axis": "expert_statement_polarity_sc",
        "confidence_interval_unit": "image_level_cluster_by_unit_id",
        "patient_level_confidence_interval": False,
        "records": len(aligned),
        "units": len(unit_ids),
        "findings": findings,
        **point,
        "image_cluster_bootstrap_95ci": ci,
        "bootstrap_requested_replicates": int(bootstrap_replicates),
        "bootstrap_effective_replicates": len(replicates),
        "bootstrap_seed": int(bootstrap_seed),
    }
