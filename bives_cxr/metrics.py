"""Primary, calibration, and mechanism metrics for BiVES-CXR."""

from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .decoder import STATE_NAMES


def intervention_metric_counts(
    outputs: dict[str, dict[str, torch.Tensor] | torch.Tensor],
    targets: torch.Tensor,
) -> dict[str, float]:
    """Return additive numerators/denominators for dataset-level aggregation."""

    original = outputs["original"]
    keep = outputs["keep"]
    drop = outputs["drop"]
    control = outputs["control"]
    assert isinstance(original, dict) and isinstance(keep, dict)
    assert isinstance(drop, dict) and isinstance(control, dict)

    answerable = targets != 3
    original_pred = original["state_probs"].argmax(dim=-1)
    keep_pred = keep["state_probs"].argmax(dim=-1)
    drop_pred = drop["state_probs"].argmax(dim=-1)
    control_pred = control["state_probs"].argmax(dim=-1)
    original_correct = original_pred == targets
    eligible = original_correct & answerable
    eos_eligible = eligible
    eri_eligible = eligible
    target_change = (drop["state_probs"] - original["state_probs"]).abs().sum(dim=-1)
    control_branches = outputs.get("controls")
    if isinstance(control_branches, list) and control_branches:
        control_probabilities = torch.stack(
            [branch["state_probs"] for branch in control_branches],
            dim=1,
        )
    else:
        control_probabilities = control["state_probs"].unsqueeze(1)
    control_changes = (
        control_probabilities - original["state_probs"].unsqueeze(1)
    ).abs().sum(dim=-1)
    control_predictions = control_probabilities.argmax(dim=-1)
    control_stable = control_predictions == original_pred.unsqueeze(1)
    mean_control_change = control_changes.mean(dim=1)
    worst_control_change = control_changes.max(dim=1).values
    return {
        "eos_numerator": float(((keep_pred == targets) & eos_eligible).sum().item()),
        "eos_denominator": float(eos_eligible.sum().item()),
        "eri_numerator": float(((drop_pred == 3) & eri_eligible).sum().item()),
        "eri_denominator": float(eri_eligible.sum().item()),
        "iis_numerator": float((control_pred == original_pred).sum().item()),
        "iis_denominator": float(targets.numel()),
        "control_stability_mean_sum": float(control_stable.float().mean(dim=1).sum().item()),
        "control_stability_worst_sum": float(control_stable.all(dim=1).float().sum().item()),
        "control_effect_mean_sum": float(mean_control_change.sum().item()),
        "control_effect_worst_sum": float(worst_control_change.sum().item()),
        "tcig_sum": float((target_change - mean_control_change).sum().item()),
        "tcig_worst_sum": float((target_change - worst_control_change).sum().item()),
        "tcig_denominator": float(targets.numel()),
        "eligible_denominator": float(eligible.sum().item()),
        "eligible_control_stability_mean_sum": float(
            (control_stable.float().mean(dim=1) * eligible.float()).sum().item()
        ),
        "eligible_control_stability_worst_sum": float(
            (control_stable.all(dim=1).float() * eligible.float()).sum().item()
        ),
        "eligible_control_effect_mean_sum": float(
            (mean_control_change * eligible.float()).sum().item()
        ),
        "eligible_control_effect_worst_sum": float(
            (worst_control_change * eligible.float()).sum().item()
        ),
        "eligible_tcig_sum": float(
            ((target_change - mean_control_change) * eligible.float()).sum().item()
        ),
        "eligible_tcig_worst_sum": float(
            ((target_change - worst_control_change) * eligible.float()).sum().item()
        ),
    }


def finalize_intervention_metrics(counts: dict[str, float]) -> dict[str, float]:
    def ratio(numerator: str, denominator: str) -> float:
        value = counts.get(denominator, 0.0)
        return float(counts.get(numerator, 0.0) / value) if value > 0 else float("nan")

    return {
        "evidence_only_sufficiency": ratio("eos_numerator", "eos_denominator"),
        "evidence_removal_insufficient": ratio("eri_numerator", "eri_denominator"),
        "irrelevant_stability": ratio("iis_numerator", "iis_denominator"),
        "irrelevant_stability_mean": ratio(
            "control_stability_mean_sum", "iis_denominator"
        ),
        "irrelevant_stability_worst_case": ratio(
            "control_stability_worst_sum", "iis_denominator"
        ),
        "control_effect_l1_mean": ratio("control_effect_mean_sum", "iis_denominator"),
        "control_effect_l1_worst_case": ratio(
            "control_effect_worst_sum", "iis_denominator"
        ),
        "target_control_gap": ratio("tcig_sum", "tcig_denominator"),
        "target_control_gap_worst_case": ratio("tcig_worst_sum", "tcig_denominator"),
        "irrelevant_stability_eligible_mean": ratio(
            "eligible_control_stability_mean_sum", "eligible_denominator"
        ),
        "irrelevant_stability_eligible_worst_case": ratio(
            "eligible_control_stability_worst_sum", "eligible_denominator"
        ),
        "control_effect_l1_eligible_mean": ratio(
            "eligible_control_effect_mean_sum", "eligible_denominator"
        ),
        "control_effect_l1_eligible_worst_case": ratio(
            "eligible_control_effect_worst_sum", "eligible_denominator"
        ),
        "target_control_gap_eligible": ratio(
            "eligible_tcig_sum", "eligible_denominator"
        ),
        "target_control_gap_eligible_worst_case": ratio(
            "eligible_tcig_worst_sum", "eligible_denominator"
        ),
    }


def intervention_metrics(
    outputs: dict[str, dict[str, torch.Tensor] | torch.Tensor],
    targets: torch.Tensor,
) -> dict[str, float]:
    return finalize_intervention_metrics(intervention_metric_counts(outputs, targets))


def expected_calibration_error(
    probabilities: np.ndarray,
    targets: np.ndarray,
    bins: int = 15,
) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == targets
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (
            (confidence >= lower)
            & (confidence < upper if index < bins - 1 else confidence <= upper)
        )
        if selected.any():
            ece += selected.mean() * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return float(ece)


def classwise_calibration_error(
    probabilities: np.ndarray,
    targets: np.ndarray,
    bins: int = 15,
) -> dict[str, float]:
    result: dict[str, float] = {}
    edges = np.linspace(0.0, 1.0, bins + 1)
    for class_index, state_name in enumerate(STATE_NAMES):
        confidence = probabilities[:, class_index]
        observed = targets == class_index
        error = 0.0
        for index in range(bins):
            lower, upper = edges[index], edges[index + 1]
            selected = (
                (confidence >= lower)
                & (confidence < upper if index < bins - 1 else confidence <= upper)
            )
            if selected.any():
                error += selected.mean() * abs(
                    float(observed[selected].mean()) - float(confidence[selected].mean())
                )
        result[state_name] = float(error)
    return result


def risk_coverage(probabilities: np.ndarray, targets: np.ndarray) -> tuple[list[dict[str, float]], float]:
    confidence = probabilities.max(axis=1)
    errors = (probabilities.argmax(axis=1) != targets).astype(np.float64)
    order = np.argsort(-confidence, kind="stable")
    ordered_errors = errors[order]
    cumulative_risk = np.cumsum(ordered_errors) / np.arange(1, len(targets) + 1)
    coverage = np.arange(1, len(targets) + 1) / len(targets)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    aurc = float(integrate(cumulative_risk, coverage))
    curve = [
        {
            "coverage": float(coverage[index]),
            "risk": float(cumulative_risk[index]),
        }
        for index in range(len(targets))
    ]
    return curve, aurc


def _safe_binary_metrics(targets: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if np.unique(targets).size < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {
        "auroc": float(roc_auc_score(targets, scores)),
        "auprc": float(average_precision_score(targets, scores)),
    }


def classification_metrics(
    probabilities: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    calibration_bins: int = 15,
) -> dict[str, object]:
    probabilities = np.asarray(
        probabilities.detach().cpu().numpy()
        if torch.is_tensor(probabilities)
        else probabilities,
        dtype=np.float64,
    )
    targets = np.asarray(
        targets.detach().cpu().numpy() if torch.is_tensor(targets) else targets,
        dtype=np.int64,
    )
    if probabilities.shape != (targets.size, len(STATE_NAMES)):
        raise ValueError("probabilities must have shape [N,4]")
    if targets.size == 0:
        raise ValueError("classification metrics require at least one sample")
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True).clip(min=1e-12)
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=np.arange(len(STATE_NAMES)),
        zero_division=0,
    )
    one_hot = np.eye(len(STATE_NAMES), dtype=np.float64)[targets]
    nll = -np.log(probabilities[np.arange(targets.size), targets].clip(min=1e-12)).mean()
    brier = np.square(probabilities - one_hot).sum(axis=1).mean()
    sc_selected = np.isin(targets, [0, 1])
    ui_selected = np.isin(targets, [2, 3])
    sc_score = probabilities[sc_selected, 0] / (
        probabilities[sc_selected, 0] + probabilities[sc_selected, 1]
    ).clip(min=1e-12)
    ui_score = probabilities[ui_selected, 2] / (
        probabilities[ui_selected, 2] + probabilities[ui_selected, 3]
    ).clip(min=1e-12)
    curve, aurc = risk_coverage(probabilities, targets)
    return {
        "accuracy": float((predictions == targets).mean()),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=np.arange(len(STATE_NAMES)),
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            np.mean(
                [
                    float((predictions[targets == class_index] == class_index).mean())
                    if bool((targets == class_index).any())
                    else 0.0
                    for class_index in range(len(STATE_NAMES))
                ]
            )
        ),
        "per_state": {
            state_name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, state_name in enumerate(STATE_NAMES)
        },
        "confusion_matrix": confusion_matrix(
            targets,
            predictions,
            labels=np.arange(len(STATE_NAMES)),
        ).tolist(),
        "support_vs_contradict": _safe_binary_metrics(
            (targets[sc_selected] == 0).astype(np.int64),
            sc_score,
        ),
        "uncertain_vs_insufficient": _safe_binary_metrics(
            (targets[ui_selected] == 2).astype(np.int64),
            ui_score,
        ),
        "nll": float(nll),
        "brier": float(brier),
        "ece": expected_calibration_error(probabilities, targets, calibration_bins),
        "classwise_ece": classwise_calibration_error(
            probabilities,
            targets,
            calibration_bins,
        ),
        "risk_coverage": curve,
        "aurc": aurc,
    }


def patient_bootstrap_confidence_intervals(
    probabilities: np.ndarray,
    targets: np.ndarray,
    patient_ids: list[str],
    replicates: int = 1000,
    seed: int = 17,
) -> dict[str, object]:
    if len(patient_ids) != len(targets):
        raise ValueError("patient_ids and targets must have equal length")
    unique_patients = sorted(set(patient_ids))
    if not unique_patients or replicates <= 0:
        return {}
    patient_rows = {
        patient: np.asarray(
            [index for index, value in enumerate(patient_ids) if value == patient],
            dtype=np.int64,
        )
        for patient in unique_patients
    }
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "accuracy": [],
        "macro_f1": [],
        "balanced_accuracy": [],
    }
    missing_class_replicates = 0
    for _ in range(replicates):
        sampled_patients = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        sampled_rows = np.concatenate([patient_rows[str(patient)] for patient in sampled_patients])
        predictions = probabilities[sampled_rows].argmax(axis=1)
        sampled_targets = targets[sampled_rows]
        values["accuracy"].append(float((predictions == sampled_targets).mean()))
        values["macro_f1"].append(
            float(
                f1_score(
                    sampled_targets,
                    predictions,
                    labels=np.arange(len(STATE_NAMES)),
                    average="macro",
                    zero_division=0,
                )
            )
        )
        recalls: list[float] = []
        for class_index in range(len(STATE_NAMES)):
            selected = sampled_targets == class_index
            if not bool(selected.any()):
                recalls = []
                missing_class_replicates += 1
                break
            recalls.append(float((predictions[selected] == class_index).mean()))
        values["balanced_accuracy"].append(
            float(np.mean(recalls)) if recalls else float("nan")
        )
    intervals = {
        key: {
            "lower_95": float(np.nanquantile(metric_values, 0.025)),
            "median": float(np.nanquantile(metric_values, 0.5)),
            "upper_95": float(np.nanquantile(metric_values, 0.975)),
            "requested_replicates": int(replicates),
            "valid_replicates": int(
                sum(not math.isnan(value) for value in metric_values)
            ),
        }
        for key, metric_values in values.items()
        if not all(math.isnan(value) for value in metric_values)
    }
    return {
        **intervals,
        "_metadata": {
            "class_labels": list(range(len(STATE_NAMES))),
            "sampling_unit": "patient",
            "requested_replicates": int(replicates),
            "balanced_accuracy_missing_class_replicates": int(
                missing_class_replicates
            ),
        },
    }
