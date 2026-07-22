import unittest

import numpy as np

from arise_cxr.matched_controls import (
    deterministic_stat_matched_connected_control_mask,
)
from bives_cxr.localization_causality import validate_target_control_pair


class StatMatchedControlTests(unittest.TestCase):
    def setUp(self):
        yy, xx = np.mgrid[:48, :48]
        self.image = (0.7 * xx + 0.3 * yy + 5.0 * np.sin(xx / 5.0)).astype(
            np.float64
        )
        self.content = np.ones((48, 48), dtype=bool)
        self.target = np.zeros((48, 48), dtype=bool)
        self.target[10:16, 8:16] = True
        self.forbidden = np.zeros((48, 48), dtype=bool)
        self.forbidden[5:9, 4:12] = True

    def test_deterministic_exact_area_connected_and_disjoint(self):
        first, first_certificate = deterministic_stat_matched_connected_control_mask(
            self.image,
            self.target,
            self.content,
            seed_key="unit-case",
            forbidden_mask=self.forbidden,
            candidate_limit=12,
            seed_attempt_limit=256,
        )
        second, second_certificate = deterministic_stat_matched_connected_control_mask(
            self.image,
            self.target,
            self.content,
            seed_key="unit-case",
            forbidden_mask=self.forbidden,
            candidate_limit=12,
            seed_attempt_limit=256,
        )
        np.testing.assert_array_equal(first, second)
        self.assertFalse(bool((first & (self.target | self.forbidden)).any()))
        pair = validate_target_control_pair(
            self.target, first, content_mask=self.content
        )
        self.assertTrue(pair["pass"])
        self.assertLessEqual(pair["log_perimeter_ratio"], 1.0)
        self.assertEqual(int(first.sum()), int(self.target.sum()))
        self.assertTrue(first_certificate["selection_is_result_blind"])
        self.assertFalse(first_certificate["model_scores_used"])
        self.assertEqual(first_certificate["max_log_perimeter_ratio"], 1.0)
        self.assertEqual(
            first_certificate["selected"]["mask_sha256"],
            second_certificate["selected"]["mask_sha256"],
        )
        self.assertLessEqual(
            first_certificate["selected"]["objective"],
            first_certificate["runner_up_objective"],
        )

    def test_rejects_nonfinite_content(self):
        image = self.image.copy()
        image[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            deterministic_stat_matched_connected_control_mask(
                image,
                self.target,
                self.content,
                seed_key="bad-image",
            )

    def test_fails_closed_when_no_disjoint_area_exists(self):
        content = self.target.copy()
        with self.assertRaisesRegex(ValueError, "insufficient"):
            deterministic_stat_matched_connected_control_mask(
                self.image,
                self.target,
                content,
                seed_key="no-control",
            )


if __name__ == "__main__":
    unittest.main()
