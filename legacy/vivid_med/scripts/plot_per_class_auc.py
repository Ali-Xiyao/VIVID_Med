"""
Per-class AUC breakdown: SPD vs Q-Former proxy
生成 per-class AUC 对比图 + CSV，用于论文分析 F1/AUC dissociation
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 12-label (excluding No Finding and Pleural Other from display if AUC unstable)
LABELS_14 = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]

LABELS_12 = [l for l in LABELS_14 if l not in ("No Finding", "Pleural Other")]


def load_per_class_auc(output_dir: str, seeds=(0, 1, 2)):
    """Load per-class AUC from metrics_final.json across seeds."""
    per_class = {label: [] for label in LABELS_14}
    seed_dirs = []
    for s in seeds:
        if s == 0:
            d = Path(output_dir)
        else:
            d = Path(f"{output_dir}_seed{s}")
        seed_dirs.append(d)

    for d in seed_dirs:
        mf = d / "metrics_final.json"
        if not mf.exists():
            print(f"WARNING: {mf} not found, skipping")
            continue
        with open(mf) as f:
            data = json.load(f)
        auc_per_class = {}
        metrics = data.get("metrics", data)
        per_label = metrics.get("per_label", {})
        if per_label:
            for label, vals in per_label.items():
                if isinstance(vals, dict) and "auc" in vals:
                    auc_per_class[label] = vals["auc"]
        else:
            auc_per_class = data.get("auc_per_class", {})
        for label in LABELS_14:
            val = auc_per_class.get(label)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                per_class[label].append(val)

    # Compute mean ± std for 12-label only
    result = {}
    for label in LABELS_14:
        vals = per_class[label]
        if vals:
            result[label] = {"mean": np.mean(vals), "std": np.std(vals, ddof=1) if len(vals) > 1 else 0.0, "values": vals}
    return result


def main():
    base = Path("outputs")

    experiments = {
        "SPD (ours)": str(base / "lp_A_ums_spd_12label"),
        "Q-Former proxy": str(base / "lp_qformer_proxy_12label"),
    }

    all_data = {}
    for name, path in experiments.items():
        all_data[name] = load_per_class_auc(path)

    # Print table
    print(f"\n{'Label':<30} {'SPD mean':>10} {'SPD std':>10} {'QF mean':>10} {'QF std':>10} {'Diff':>10}")
    print("-" * 80)
    for label in LABELS_12:
        spd = all_data["SPD (ours)"].get(label, {})
        qf = all_data["Q-Former proxy"].get(label, {})
        sm, ss = spd.get("mean", 0), spd.get("std", 0)
        qm, qs = qf.get("mean", 0), qf.get("std", 0)
        diff = sm - qm
        marker = "<<<" if abs(diff) > 0.03 else ""
        print(f"{label:<30} {sm:>10.4f} {ss:>10.4f} {qm:>10.4f} {qs:>10.4f} {diff:>+10.4f} {marker}")

    # Also show Pleural Other (the key dissociation driver)
    print(f"\n--- Excluded from 12-label but informative ---")
    for label in ["No Finding", "Pleural Other"]:
        spd = all_data["SPD (ours)"].get(label, {})
        qf = all_data["Q-Former proxy"].get(label, {})
        sm = spd.get("mean", float("nan"))
        qm = qf.get("mean", float("nan"))
        print(f"{label:<30} SPD={sm:.4f}  QF={qm:.4f}  diff={sm-qm:+.4f}")

    # --- Bar chart ---
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(LABELS_12))
    width = 0.35

    spd_means = [all_data["SPD (ours)"].get(l, {}).get("mean", 0) for l in LABELS_12]
    spd_stds = [all_data["SPD (ours)"].get(l, {}).get("std", 0) for l in LABELS_12]
    qf_means = [all_data["Q-Former proxy"].get(l, {}).get("mean", 0) for l in LABELS_12]
    qf_stds = [all_data["Q-Former proxy"].get(l, {}).get("std", 0) for l in LABELS_12]

    bars1 = ax.bar(x - width / 2, spd_means, width, yerr=spd_stds, label="SPD (ours)",
                   color="#2196F3", capsize=3, alpha=0.85)
    bars2 = ax.bar(x + width / 2, qf_means, width, yerr=qf_stds, label="Q-Former proxy",
                   color="#FF9800", capsize=3, alpha=0.85)

    ax.set_ylabel("AUC", fontsize=13)
    ax.set_title("Per-class AUC: SPD vs Q-Former Proxy (CheXpert 12-label)", fontsize=14)
    ax.set_xticks(x)
    short_labels = [l.replace("Enlarged Cardiomediastinum", "Enl. Cardio.")
                     .replace("Pleural Effusion", "Pl. Effusion")
                     .replace("Support Devices", "Sup. Devices")
                    for l in LABELS_12]
    ax.set_xticklabels(short_labels, rotation=35, ha="right", fontsize=10)
    ax.set_ylim(0.5, 1.05)
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = "outputs/per_class_auc_spd_vs_qformer.png"
    fig.savefig(out_path, dpi=200)
    print(f"\nSaved figure to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
