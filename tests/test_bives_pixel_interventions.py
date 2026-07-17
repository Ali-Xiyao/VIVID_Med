from __future__ import annotations

import unittest

import numpy as np
import torch
from PIL import Image

from bives_cxr.pixel_interventions import (
    delete_pixels,
    deterministic_disjoint_control_mask,
    deterministic_random_mask,
    paired_intervention_metrics,
    patch_gate_to_pixel_mask,
    retain_pixels,
    transform_mask_to_letterbox,
    union_box_mask,
)


class PixelInterventionTests(unittest.TestCase):
    def test_target_and_control_are_exact_area_disjoint_and_reproducible(self) -> None:
        target = union_box_mask(20, 10, [{"x_min": 2, "y_min": 2, "x_max": 7, "y_max": 6}])
        content = np.ones_like(target)
        first = deterministic_disjoint_control_mask(target, content, seed_key="sample-a")
        second = deterministic_disjoint_control_mask(target, content, seed_key="sample-a")
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(int(first.sum()), int(target.sum()))
        self.assertFalse(bool((first & target).any()))
        random_mask = deterministic_random_mask(17, content, seed_key="random-a")
        self.assertEqual(int(random_mask.sum()), 17)

    def test_letterbox_and_patch_masks_preserve_geometry(self) -> None:
        source = np.zeros((10, 20), dtype=bool)
        source[2:6, 3:8] = True
        letterboxed = transform_mask_to_letterbox(source, (0, 5, 20, 15), 20)
        self.assertEqual(letterboxed.shape, (20, 20))
        gate = torch.tensor([1, 0, 0, 1], dtype=torch.bool)
        evidence = patch_gate_to_pixel_mask(gate, (2, 2), 20)
        self.assertEqual(int(evidence.sum()), 200)
        image = Image.new("RGB", (20, 20), (255, 255, 255))
        self.assertEqual(np.asarray(delete_pixels(image, evidence))[0, 0].sum(), 0)
        self.assertEqual(np.asarray(retain_pixels(image, evidence))[0, 19].sum(), 0)

    def test_paired_metrics_compute_tcig(self) -> None:
        result = paired_intervention_metrics(
            [
                {
                    "canonical_statement_id": "consolidation",
                    "original_score": 0.9,
                    "target_drop_score": 0.4,
                    "control_drop_score": 0.8,
                    "keep_score": 0.7,
                },
                {
                    "canonical_statement_id": "consolidation",
                    "original_score": 0.8,
                    "target_drop_score": 0.5,
                    "control_drop_score": 0.7,
                    "keep_score": 0.6,
                },
            ]
        )
        self.assertAlmostEqual(result["macro_mean_tcig"], 0.3)


if __name__ == "__main__":
    unittest.main()
