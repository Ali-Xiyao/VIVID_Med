from __future__ import annotations

import copy
import unittest

import numpy as np

from arise_cxr.case_study import analyze_oracle_failure
from bives_cxr.localization_causality import build_precomputed_audit_row


def make_row(*, pathology: str, operator: str, suffix: str, d_x: float, d_cx: float):
    expert = np.zeros((8, 8), dtype=bool)
    expert[1:3, 1:3] = True
    control = np.zeros((8, 8), dtype=bool)
    control[5:7, 5:7] = True
    explanation = np.zeros((8, 8), dtype=bool)
    explanation[1:3, 5:7] = True
    explanation_control = np.zeros((8, 8), dtype=bool)
    explanation_control[5:7, 1:3] = True
    zeros = {
        "masked_l1": 0.0,
        "masked_rms": 0.0,
        "ssim": 1.0,
        "masked_edge_energy_change": 0.0,
    }
    row = build_precomputed_audit_row(
        identity={
            "row_id": f"row-{suffix}",
            "dataset_role": "development",
            "patient_id": f"patient-{suffix}",
            "image_id": f"image-{suffix}",
            "pathology_id": pathology,
            "model_id": "model",
            "explanation_id": "explanation",
            "operator_id": operator,
        },
        scores={"s0": 2.0, "sX": 2.0 - d_x, "sCX": 2.0 - d_cx, "sE": 2.0, "sCE": 2.0},
        expert_mask=expert,
        explanation_mask=explanation,
        expert_control_mask=control,
        explanation_control_mask=explanation_control,
        content_mask=np.ones((8, 8), dtype=bool),
        strength_metrics={name: zeros for name in ("X", "C_X", "E", "C_E")},
        strength_thresholds={
            "max_normalized_centroid_distance": 1.0,
            "max_log_perimeter_ratio": 1.0,
            "max_masked_l1_difference": 1.0,
            "max_masked_rms_difference": 1.0,
            "max_ssim_difference": 1.0,
            "max_edge_difference": 1.0,
        },
        explanation_map=explanation.astype(np.float32),
    )
    row["test_opened"] = False
    return row


class AriseCaseStudyTests(unittest.TestCase):
    def test_diagnoses_control_excess_and_operator_reversal(self) -> None:
        rows = [
            make_row(pathology="finding", operator="mean", suffix="a", d_x=0.2, d_cx=0.1),
            make_row(pathology="finding", operator="blur", suffix="b", d_x=0.1, d_cx=0.3),
        ]
        result = analyze_oracle_failure(rows)
        self.assertIn("matched_control_effect_exceeds_expert_target", result["cells"]["finding|blur"]["diagnoses"])
        self.assertIn("operator_sign_reversal", result["pathologies"]["finding"]["diagnoses"])
        self.assertTrue(result["identifier_free"])

    def test_rejects_test_rows(self) -> None:
        row = make_row(pathology="finding", operator="mean", suffix="a", d_x=0.2, d_cx=0.1)
        opened = copy.deepcopy(row)
        opened["test_opened"] = True
        with self.assertRaisesRegex(ValueError, "test-closed|opened test"):
            analyze_oracle_failure([opened])


if __name__ == "__main__":
    unittest.main()
