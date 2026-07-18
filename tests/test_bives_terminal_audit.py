from __future__ import annotations

import unittest

import numpy as np

from bives_cxr.terminal_audit import (
    classify_effect_pair,
    descriptive_associations,
    image_perturbation_metrics,
    summarize_effect_rows,
)


class TerminalAuditContractTests(unittest.TestCase):
    def test_effect_taxonomy_is_fixed_and_mutually_exclusive(self) -> None:
        self.assertEqual(
            classify_effect_pair(0.2, 0.1), "target_dominant_positive"
        )
        self.assertEqual(classify_effect_pair(-0.1, -0.2), "target_sign_reversal")
        self.assertEqual(
            classify_effect_pair(0.0, 0.1), "control_dominant_or_target_inert"
        )
        self.assertEqual(
            classify_effect_pair(0.0, -0.1), "both_nonpositive_or_tied"
        )

    def test_associations_are_descriptive_and_handle_missing_features(self) -> None:
        rows = [
            {"tcig": float(i), "target_area_pixels": float(i), "box_count": None}
            for i in range(1, 5)
        ]
        result = descriptive_associations(rows)
        self.assertAlmostEqual(result["target_area_pixels"]["pearson_r"], 1.0)
        self.assertAlmostEqual(result["target_area_pixels"]["spearman_rho"], 1.0)
        self.assertNotIn("box_count", result)

    def test_image_metrics_detect_only_the_intervention_region(self) -> None:
        original = np.full((16, 16, 3), 128, dtype=np.uint8)
        perturbed = original.copy()
        mask = np.zeros((16, 16), dtype=bool)
        mask[4:8, 4:8] = True
        content = np.ones((16, 16), dtype=bool)
        perturbed[mask] = 0
        metrics = image_perturbation_metrics(original, perturbed, mask, content)
        self.assertEqual(metrics["mask_area_pixels"], 16)
        self.assertGreater(metrics["masked_l1"], metrics["global_l1"])
        self.assertLess(metrics["ssim"], 1.0)

    def test_summary_reports_target_and_control_separately(self) -> None:
        rows = [
            {
                "source": "c6i",
                "operator": "local_mean",
                "canonical_statement_id": "consolidation",
                "target_effect": 0.2,
                "control_effect": 0.1,
                "tcig": 0.1,
                "failure_category": "target_dominant_positive",
                "box_area_quartile": 1,
                "target_area_pixels": 100,
            },
            {
                "source": "c6i",
                "operator": "local_mean",
                "canonical_statement_id": "consolidation",
                "target_effect": -0.1,
                "control_effect": 0.1,
                "tcig": -0.2,
                "failure_category": "target_sign_reversal",
                "box_area_quartile": 2,
                "target_area_pixels": 200,
            },
        ]
        summary = summarize_effect_rows(rows)
        group = summary["groups"]["c6i|local_mean|consolidation"]
        self.assertAlmostEqual(group["mean_target_effect"], 0.05)
        self.assertAlmostEqual(group["mean_control_effect"], 0.1)
        self.assertEqual(group["taxonomy"]["target_sign_reversal"], 1)


if __name__ == "__main__":
    unittest.main()
