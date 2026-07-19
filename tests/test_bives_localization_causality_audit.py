from __future__ import annotations

import unittest

import numpy as np

from bives_cxr.localization_causality import (
    AUDIT_SCHEMA_VERSION,
    build_precomputed_audit_row,
    build_target_specific_controls,
    localization_metrics,
    score_contrasts,
    strength_match_diagnostics,
    summarize_audit_rows,
    validate_precomputed_rows,
    validate_target_control_pair,
)


class LocalizationCausalityAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.content = np.ones((24, 24), dtype=bool)
        self.expert = self._mask(3, 3, 4)
        self.explanation = self._mask(3, 5, 4)
        self.control_x = self._mask(15, 2, 4)
        self.control_e = self._mask(15, 16, 4)
        self.metrics = {
            role: {
                "masked_l1": 0.10,
                "masked_rms": 0.12,
                "ssim": 0.95,
                "masked_edge_energy_change": 0.04,
            }
            for role in ("X", "C_X", "E", "C_E")
        }
        self.thresholds = {
            "max_normalized_centroid_distance": 1.0,
            "max_log_perimeter_ratio": 0.01,
            "max_masked_l1_difference": 0.01,
            "max_masked_rms_difference": 0.01,
            "max_ssim_difference": 0.01,
            "max_edge_difference": 0.01,
        }

    @staticmethod
    def _mask(top: int, left: int, size: int) -> np.ndarray:
        value = np.zeros((24, 24), dtype=bool)
        value[top : top + size, left : left + size] = True
        return value

    def _row(self, patient: int, operator: str, cs_e: float) -> dict[str, object]:
        iou = localization_metrics(
            self.expert,
            self.explanation,
            content_mask=self.content,
        )["iou"]
        row = build_precomputed_audit_row(
            identity={
                "row_id": f"row-{patient}-{operator}",
                "patient_id": f"patient-{patient}",
                "image_id": f"image-{patient}",
                "pathology_id": "finding",
                "model_id": "model",
                "explanation_id": "explanation",
                "operator_id": operator,
                "dataset_role": "synthetic_development",
            },
            scores={"s0": 0.8, "sX": 0.5, "sCX": 0.75, "sE": 0.8 - 0.04 - cs_e, "sCE": 0.76},
            expert_mask=self.expert,
            explanation_mask=self.explanation,
            expert_control_mask=self.control_x,
            explanation_control_mask=self.control_e,
            content_mask=self.content,
            strength_metrics=self.metrics,
            strength_thresholds=self.thresholds,
        )
        row["test_iou"] = iou
        return row

    def test_localization_and_score_contrasts_are_separate(self) -> None:
        localization = localization_metrics(
            self.expert,
            self.explanation,
            content_mask=self.content,
        )
        self.assertAlmostEqual(localization["iou"], 8 / 24)
        scores = score_contrasts(
            {"s0": 0.8, "sX": 0.5, "sCX": 0.75, "sE": 0.6, "sCE": 0.76}
        )
        self.assertAlmostEqual(scores["CS_X"], 0.25)
        self.assertAlmostEqual(scores["CS_E"], 0.16)

    def test_one_control_cannot_match_different_target_areas(self) -> None:
        smaller = self._mask(3, 5, 3)
        validate_target_control_pair(
            self.expert, self.control_x, content_mask=self.content
        )
        with self.assertRaisesRegex(ValueError, "exact equal area"):
            validate_target_control_pair(
                smaller, self.control_x, content_mask=self.content
            )

    def test_target_specific_control_builder_excludes_both_targets(self) -> None:
        controls, certificates = build_target_specific_controls(
            self.expert,
            self.explanation,
            content_mask=self.content,
            seed_key="unit-test",
        )
        for role in ("C_X", "C_E"):
            self.assertFalse((controls[role] & (self.expert | self.explanation)).any())
            self.assertEqual(certificates[role]["control"]["component_count"], 1)
        self.assertEqual(controls["C_X"].sum(), self.expert.sum())
        self.assertEqual(controls["C_E"].sum(), self.explanation.sum())

    def test_strength_gate_fails_a_material_mismatch(self) -> None:
        geometry = validate_target_control_pair(
            self.expert, self.control_x, content_mask=self.content
        )
        bad = dict(self.metrics["C_X"])
        bad["masked_l1"] = 0.5
        result = strength_match_diagnostics(
            self.metrics["X"],
            bad,
            pair_geometry=geometry,
            thresholds=self.thresholds,
        )
        self.assertFalse(result["pass"])
        self.assertFalse(result["checks"]["masked_l1"])
        invalid_thresholds = dict(self.thresholds)
        invalid_thresholds["max_edge_difference"] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            strength_match_diagnostics(
                self.metrics["X"],
                self.metrics["C_X"],
                pair_geometry=geometry,
                thresholds=invalid_thresholds,
            )

    def test_development_validator_rejects_test_or_duplicate_rows(self) -> None:
        row = self._row(0, "blur", 0.1)
        self.assertEqual(row["schema_version"], AUDIT_SCHEMA_VERSION)
        bad = dict(row)
        bad["dataset_role"] = "chexlocalize_test"
        with self.assertRaisesRegex(ValueError, "only synthetic/development"):
            validate_precomputed_rows([bad])
        with self.assertRaisesRegex(ValueError, "duplicate audit row_id"):
            validate_precomputed_rows([row, row])
        bad_identity = dict(row)
        bad_identity["model_id"] = "bad|model"
        with self.assertRaisesRegex(ValueError, "delimiters"):
            validate_precomputed_rows([bad_identity])

    def test_patient_cluster_summary_and_operator_worst_case(self) -> None:
        rows = []
        for patient in range(4):
            rows.append(self._row(patient, "local_mean", 0.10 + 0.01 * patient))
            rows.append(self._row(patient, "blur", -0.02 + 0.01 * patient))
        first = summarize_audit_rows(rows, bootstrap_replicates=100, bootstrap_seed=9)
        second = summarize_audit_rows(rows, bootstrap_replicates=100, bootstrap_seed=9)
        self.assertEqual(first, second)
        cross = first["cross_operator"]["model|explanation|finding"]
        self.assertFalse(cross["all_positive"])
        self.assertFalse(cross["sign_agreement"])
        self.assertLess(cross["worst_mean_CS_E"], 0.0)


if __name__ == "__main__":
    unittest.main()
