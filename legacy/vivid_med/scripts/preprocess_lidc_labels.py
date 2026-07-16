"""
LIDC-IDRI 肺结节良恶性标签预处理

从 pylidc 提取 malignancy score，生成 CSV 标签文件。
malignancy 1-2 → benign (0), 4-5 → malignant (1), 3 → 排除

用法:
    pip install pylidc
    python scripts/preprocess_lidc_labels.py

输出: data/dataset/processed/lidc_nodule_labels.csv
"""

import csv
import os
from pathlib import Path

import numpy as np
# pylidc uses deprecated np.int, monkey-patch for numpy >= 1.24 compatibility
if not hasattr(np, 'int'):
    np.int = int


def main():
    try:
        import pylidc as pl
    except ImportError:
        print("pylidc not installed. Run: pip install pylidc")
        print("Also need the LIDC-IDRI DICOM data configured in ~/.pylidcrc")
        return

    output_path = Path(__file__).parent.parent / "data" / "dataset" / "processed" / "lidc_nodule_labels.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    slices_root = Path(__file__).parent.parent / "data" / "dataset" / "LIDC-IDRI-slices"

    rows = []
    scans = pl.query(pl.Scan).all()
    print(f"Found {len(scans)} scans")

    for scan in scans:
        patient_id = scan.patient_id  # e.g. LIDC-IDRI-0001
        case_dir = slices_root / patient_id
        if not case_dir.exists():
            continue

        nodules = scan.cluster_annotations()
        for nod_idx, anns in enumerate(nodules):
            nodule_dir = case_dir / f"nodule-{nod_idx}"
            if not nodule_dir.exists():
                continue

            # Average malignancy score across annotators
            scores = [a.malignancy for a in anns]
            avg_score = sum(scores) / len(scores)

            if avg_score <= 2.0:
                label = 0  # benign
            elif avg_score >= 4.0:
                label = 1  # malignant
            else:
                label = -1  # indeterminate, exclude

            num_slices = 0
            img_dir = nodule_dir / "images"
            if img_dir.exists():
                num_slices = len(list(img_dir.glob("*.png")))

            rows.append({
                "patient_id": patient_id,
                "nodule_idx": nod_idx,
                "avg_malignancy": round(avg_score, 2),
                "label": label,
                "num_slices": num_slices,
            })

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "nodule_idx", "avg_malignancy", "label", "num_slices"])
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    benign = sum(1 for r in rows if r["label"] == 0)
    malignant = sum(1 for r in rows if r["label"] == 1)
    excluded = sum(1 for r in rows if r["label"] == -1)
    print(f"Saved {total} nodules to {output_path}")
    print(f"  Benign: {benign}, Malignant: {malignant}, Excluded: {excluded}")


if __name__ == "__main__":
    main()
