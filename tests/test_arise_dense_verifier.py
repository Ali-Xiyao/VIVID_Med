from __future__ import annotations

import unittest

import numpy as np

from arise_cxr.dense_verifier import (
    dense_oracle_progress_identity_matches,
    oracle_model_id_for_head,
    pooled_logistic_margin,
    reconstruct_phase_h_explanation_mask,
)


class AriseDenseVerifierTests(unittest.TestCase):
    def test_box_supervised_head_has_a_frozen_audit_model_id(self) -> None:
        self.assertEqual(
            oracle_model_id_for_head("arise_patch_mil_vindr_box_overlap_v2"),
            "arise_dense_qwen35_2b_patch_mil_vindr_box_step200",
        )
        with self.assertRaisesRegex(ValueError, "unregistered"):
            oracle_model_id_for_head("future_unlocked_head")

    def test_reconstructs_content_clipped_top_cell(self) -> None:
        content = np.zeros((448, 448), dtype=bool)
        content[40:420, :] = True
        expected = {
            "bbox": [0, 40, 112, 112],
            "area_pixels": 112 * 72,
        }
        mask = reconstruct_phase_h_explanation_mask(expected, content_mask=content)
        self.assertEqual(int(mask.sum()), expected["area_pixels"])
        self.assertTrue(mask[40, 0])
        self.assertFalse(mask[112, 0])

    def test_rejects_geometry_drift(self) -> None:
        content = np.ones((448, 448), dtype=bool)
        with self.assertRaisesRegex(ValueError, "area changed"):
            reconstruct_phase_h_explanation_mask(
                {"bbox": [112, 112, 224, 224], "area_pixels": 7},
                content_mask=content,
            )

    def test_pooled_logistic_margin_matches_standardized_linear_head(self) -> None:
        value = pooled_logistic_margin(
            np.array([3.0, 5.0]),
            scaler_mean=np.array([1.0, 1.0]),
            scaler_scale=np.array([2.0, 4.0]),
            coefficient=np.array([[2.0, -1.0]]),
            intercept=np.array([0.5]),
        )
        self.assertAlmostEqual(value, 1.5)

    def test_pooled_logistic_margin_rejects_nonpositive_scale(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-positive"):
            pooled_logistic_margin(
                np.array([1.0]),
                scaler_mean=np.array([0.0]),
                scaler_scale=np.array([0.0]),
                coefficient=np.array([[1.0]]),
                intercept=np.array([0.0]),
            )

    def test_allows_only_exact_legacy_b1_progress_identity(self) -> None:
        current = {
            "schema_version": "arise-dense-oracle-run-v1",
            "backend": "b1_dense",
            "checkpoint_sha256": "checkpoint",
            "pooled_model_sha256": None,
            "mil_checkpoint_sha256": None,
            "test_opened": False,
        }
        legacy = {
            key: value
            for key, value in current.items()
            if key not in {"backend", "pooled_model_sha256", "mil_checkpoint_sha256"}
        }
        self.assertTrue(dense_oracle_progress_identity_matches(legacy, current))
        self.assertFalse(
            dense_oracle_progress_identity_matches(
                dict(legacy, checkpoint_sha256="changed"), current
            )
        )
        pooled = dict(
            current,
            backend="pooled_logistic",
            pooled_model_sha256={"finding": "hash"},
        )
        self.assertFalse(dense_oracle_progress_identity_matches(legacy, pooled))
        pre_mil = dict(current)
        pre_mil.pop("mil_checkpoint_sha256")
        self.assertTrue(dense_oracle_progress_identity_matches(pre_mil, current))


if __name__ == "__main__":
    unittest.main()
