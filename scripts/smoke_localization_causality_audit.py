"""Deterministic model-free smoke for the localization-causality audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bives_cxr.localization_causality import (
    build_precomputed_audit_row,
    intervention_strength_metrics,
    summarize_audit_rows,
)
from bives_cxr.pixel_interventions import (
    replace_with_local_ring_mean,
    replace_with_masked_gaussian_blur,
)


SYNTHETIC_THRESHOLDS = {
    "max_normalized_centroid_distance": 0.60,
    "max_log_perimeter_ratio": 0.01,
    "max_masked_l1_difference": 1.0,
    "max_masked_rms_difference": 1.0,
    "max_ssim_difference": 1.0,
    "max_edge_difference": 2.0,
}


def _mask(top: int, left: int, size: int = 6) -> np.ndarray:
    value = np.zeros((48, 48), dtype=bool)
    value[top : top + size, left : left + size] = True
    return value


def _image(patient_index: int) -> np.ndarray:
    y, x = np.mgrid[:48, :48]
    return np.stack(
        [
            (x * 4 + patient_index * 3) % 256,
            (y * 5 + patient_index * 7) % 256,
            ((x + y) * 3 + patient_index * 11) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)


def _perturb(
    operator: str,
    image: np.ndarray,
    mask: np.ndarray,
    content: np.ndarray,
) -> np.ndarray:
    pil = Image.fromarray(image, mode="RGB")
    if operator == "local_mean":
        result = replace_with_local_ring_mean(pil, mask, content, ring_width=4)
    elif operator == "masked_gaussian_blur":
        result = replace_with_masked_gaussian_blur(
            pil,
            mask,
            content,
            sigma=2.0,
            truncate=3.0,
        )
    else:
        raise ValueError(operator)
    return np.asarray(result)


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    content = np.ones((48, 48), dtype=bool)
    for patient_index in range(8):
        pathology = (
            "synthetic_positive_relation"
            if patient_index < 4
            else "synthetic_inverse_relation"
        )
        local_index = patient_index % 4
        expert = _mask(8, 8)
        explanation = _mask(8, 8 + local_index)
        expert_control = _mask(30, 6)
        explanation_control = _mask(30, 28)
        explanation_map = explanation.astype(np.float64)
        explanation_map += np.linspace(0.0, 1e-6, 48)[None, :]
        image = _image(patient_index)
        intersection = float((expert & explanation).sum())
        union = float((expert | explanation).sum())
        iou = intersection / union

        for operator in ("local_mean", "masked_gaussian_blur"):
            masks = {
                "X": expert,
                "C_X": expert_control,
                "E": explanation,
                "C_E": explanation_control,
            }
            strength = {
                role: intervention_strength_metrics(
                    image,
                    _perturb(operator, image, mask, content),
                    intervention_mask=mask,
                    content_mask=content,
                )
                for role, mask in masks.items()
            }
            cs_e = (
                0.05 + 0.25 * iou
                if pathology == "synthetic_positive_relation"
                else 0.12 - 0.22 * iou
            )
            if operator == "masked_gaussian_blur":
                cs_e -= 0.02
            s0 = 0.80
            d_cx = 0.03
            d_x = 0.25
            d_ce = 0.04
            d_e = d_ce + cs_e
            row = build_precomputed_audit_row(
                identity={
                    "row_id": f"synthetic::{patient_index:02d}::{operator}",
                    "patient_id": f"synthetic-patient-{patient_index:02d}",
                    "image_id": f"synthetic-image-{patient_index:02d}",
                    "pathology_id": pathology,
                    "model_id": "synthetic_score_fixture",
                    "explanation_id": "synthetic_square_map",
                    "operator_id": operator,
                    "dataset_role": "synthetic_development",
                },
                scores={
                    "s0": s0,
                    "sX": s0 - d_x,
                    "sCX": s0 - d_cx,
                    "sE": s0 - d_e,
                    "sCE": s0 - d_ce,
                },
                expert_mask=expert,
                explanation_mask=explanation,
                expert_control_mask=expert_control,
                explanation_control_mask=explanation_control,
                content_mask=content,
                strength_metrics=strength,
                strength_thresholds=SYNTHETIC_THRESHOLDS,
                explanation_map=explanation_map,
            )
            row["synthetic_expected_relation"] = (
                "positive" if patient_index < 4 else "inverse"
            )
            rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("local_runs/cxr_localization_causality/synthetic_smoke"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows()
    summary = summarize_audit_rows(rows, bootstrap_replicates=400)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "development_rows.jsonl"
    rows_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    summary_path = output_dir / "smoke_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "formal_result": summary["formal_result"],
                "test_opened": summary["test_opened"],
                "rows": summary["rows"],
                "patients": summary["patients"],
                "groups": len(summary["groups"]),
                "rows_path": rows_path.as_posix(),
                "summary_path": summary_path.as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
