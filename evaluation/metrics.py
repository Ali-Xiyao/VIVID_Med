"""
评估指标

包含：
- 分类指标：AUC, F1, Precision, Recall
- 可靠性指标：ECE, Brier Score, Risk-Coverage
- 临床硬错误：laterality/unit/×10 等
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    计算分类指标

    Args:
        y_true: (N, C) 真实标签
        y_pred: (N, C) 预测标签（二值）或 (N,) 单标签
        y_prob: (N, C) 预测概率（用于 AUC）
        label_names: 标签名称列表
        threshold: 二值化阈值

    Returns:
        包含各项指标的字典
    """
    metrics = {}

    # 处理多标签情况
    if y_true.ndim == 2:
        num_labels = y_true.shape[1]

        # 过滤掉全 NaN 的标签
        valid_labels = []
        for i in range(num_labels):
            mask = ~np.isnan(y_true[:, i])
            if mask.sum() > 0:
                valid_labels.append(i)

        # 每个标签的指标
        per_label_metrics = {}
        aucs = []
        f1s = []

        for i in valid_labels:
            mask = ~np.isnan(y_true[:, i])
            y_t = y_true[mask, i]
            y_p = (y_pred[mask, i] > threshold).astype(int) if y_pred.ndim == 2 else y_pred[mask]

            label_name = label_names[i] if label_names else f"label_{i}"

            # 基础指标
            label_metrics = {
                "accuracy": accuracy_score(y_t, y_p),
                "f1": f1_score(y_t, y_p, zero_division=0),
                "precision": precision_score(y_t, y_p, zero_division=0),
                "recall": recall_score(y_t, y_p, zero_division=0),
                "support": int(mask.sum()),
            }

            # AUC（需要概率）
            if y_prob is not None and len(np.unique(y_t)) > 1:
                try:
                    label_metrics["auc"] = roc_auc_score(y_t, y_prob[mask, i])
                    aucs.append(label_metrics["auc"])
                except:
                    label_metrics["auc"] = None

            f1s.append(label_metrics["f1"])
            per_label_metrics[label_name] = label_metrics

        metrics["per_label"] = per_label_metrics
        metrics["macro_f1"] = np.mean(f1s) if f1s else 0
        metrics["macro_auc"] = np.mean(aucs) if aucs else None

    else:
        # 单标签情况
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)

        if y_prob is not None:
            try:
                metrics["auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
            except:
                metrics["auc"] = None

    return metrics


def compute_reliability_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    计算可靠性指标

    Args:
        y_true: (N,) 或 (N, C) 真实标签
        y_prob: (N,) 或 (N, C) 预测概率
        n_bins: ECE 的 bin 数量

    Returns:
        包含 ECE, Brier Score 等的字典
    """
    metrics = {}

    # 展平处理
    if y_true.ndim == 2:
        # 过滤 NaN
        mask = ~np.isnan(y_true)
        y_true_flat = y_true[mask]
        y_prob_flat = y_prob[mask]
    else:
        y_true_flat = y_true
        y_prob_flat = y_prob

    # Brier Score
    metrics["brier_score"] = np.mean((y_prob_flat - y_true_flat) ** 2)

    # ECE (Expected Calibration Error)
    ece = compute_ece(y_true_flat, y_prob_flat, n_bins)
    metrics["ece"] = ece

    # MCE (Maximum Calibration Error)
    mce = compute_mce(y_true_flat, y_prob_flat, n_bins)
    metrics["mce"] = mce

    return metrics


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    计算 Expected Calibration Error (ECE)

    ECE = Σ (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Args:
        y_true: (N,) 真实标签
        y_prob: (N,) 预测概率
        n_bins: bin 数量

    Returns:
        ECE 值
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        # 找到落在这个 bin 的样本
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.sum() / total_samples

        if prop_in_bin > 0:
            # 计算这个 bin 的准确率和平均置信度
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()

            ece += prop_in_bin * abs(accuracy_in_bin - avg_confidence_in_bin)

    return ece


def compute_mce(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    计算 Maximum Calibration Error (MCE)

    MCE = max_m |acc(B_m) - conf(B_m)|
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    mce = 0.0

    for i in range(n_bins):
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])

        if in_bin.sum() > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            mce = max(mce, abs(accuracy_in_bin - avg_confidence_in_bin))

    return mce


def compute_risk_coverage_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_confidence: np.ndarray,
    n_points: int = 100,
) -> Dict[str, np.ndarray]:
    """
    计算 Risk-Coverage 曲线

    随着 coverage 增加（接受更多低置信度预测），risk（错误率）如何变化

    Args:
        y_true: (N,) 真实标签
        y_prob: (N,) 预测概率
        y_confidence: (N,) 置信度分数（用于排序）
        n_points: 曲线点数

    Returns:
        包含 coverage, risk, thresholds 的字典
    """
    # 按置信度排序（从高到低）
    sorted_indices = np.argsort(-y_confidence)
    y_true_sorted = y_true[sorted_indices]
    y_prob_sorted = y_prob[sorted_indices]

    # 二值化预测
    y_pred_sorted = (y_prob_sorted > 0.5).astype(int)

    # 计算累积错误
    errors = (y_pred_sorted != y_true_sorted).astype(float)
    cumulative_errors = np.cumsum(errors)

    # 计算不同 coverage 下的 risk
    n_samples = len(y_true)
    coverages = np.linspace(0.01, 1.0, n_points)
    risks = []

    for cov in coverages:
        n_covered = int(cov * n_samples)
        if n_covered > 0:
            risk = cumulative_errors[n_covered - 1] / n_covered
        else:
            risk = 0
        risks.append(risk)

    return {
        "coverage": coverages,
        "risk": np.array(risks),
        "auc": np.trapz(risks, coverages),  # Area under risk-coverage curve
    }


def compute_clinical_hard_errors(
    predictions: List[Dict],
    ground_truths: List[Dict],
) -> Dict[str, Any]:
    """
    计算临床硬错误

    硬错误类型：
    - laterality_flip: left ↔ right 错误
    - unit_error: 单位错误 (mm ↔ cm)
    - scale_error: 数值 ×10 或 ÷10 错误
    - finding_confusion: 混淆相似疾病

    Args:
        predictions: 预测的 UMS JSON 列表
        ground_truths: 真实的 UMS JSON 列表

    Returns:
        包含各类硬错误统计的字典
    """
    errors = {
        "laterality_flip": 0,
        "unit_error": 0,
        "scale_error": 0,
        "finding_confusion": 0,
        "total_samples": len(predictions),
    }

    for pred, gt in zip(predictions, ground_truths):
        # Laterality 错误
        pred_lat = pred.get("laterality")
        gt_lat = gt.get("laterality")
        if pred_lat is not None and gt_lat is not None:
            if (pred_lat == "left" and gt_lat == "right") or \
               (pred_lat == "right" and gt_lat == "left"):
                errors["laterality_flip"] += 1

        # Measurement 错误
        pred_measurements = pred.get("measurements", {})
        gt_measurements = gt.get("measurements", {})

        for name in gt_measurements:
            if name in pred_measurements:
                pred_m = pred_measurements[name]
                gt_m = gt_measurements[name]

                if isinstance(pred_m, dict) and isinstance(gt_m, dict):
                    pred_val = pred_m.get("value")
                    gt_val = gt_m.get("value")
                    pred_unit = pred_m.get("unit")
                    gt_unit = gt_m.get("unit")

                    # Unit 错误
                    if pred_unit != gt_unit and pred_unit is not None and gt_unit is not None:
                        errors["unit_error"] += 1

                    # Scale 错误 (×10 或 ÷10)
                    if pred_val is not None and gt_val is not None and gt_val != 0:
                        ratio = pred_val / gt_val
                        if abs(ratio - 10) < 0.1 or abs(ratio - 0.1) < 0.01:
                            errors["scale_error"] += 1

        # Finding confusion（简化版：检查 present/absent 翻转）
        pred_findings = pred.get("findings", {})
        gt_findings = gt.get("findings", {})

        for name in gt_findings:
            if name in pred_findings:
                pred_state = pred_findings[name].get("state") if isinstance(pred_findings[name], dict) else None
                gt_state = gt_findings[name].get("state") if isinstance(gt_findings[name], dict) else None

                if pred_state is not None and gt_state is not None:
                    if (pred_state == "present" and gt_state == "absent") or \
                       (pred_state == "absent" and gt_state == "present"):
                        errors["finding_confusion"] += 1

    # 计算错误率
    total = errors["total_samples"]
    if total > 0:
        errors["laterality_flip_rate"] = errors["laterality_flip"] / total
        errors["unit_error_rate"] = errors["unit_error"] / total
        errors["scale_error_rate"] = errors["scale_error"] / total
        errors["finding_confusion_rate"] = errors["finding_confusion"] / total
        errors["total_hard_error_rate"] = (
            errors["laterality_flip"] + errors["unit_error"] +
            errors["scale_error"] + errors["finding_confusion"]
        ) / total

    return errors
