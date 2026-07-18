from __future__ import annotations

import unittest

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from bives_cxr.pixel_interventions import (
    LOCAL_MEAN_RING_WIDTH,
    MASKED_GAUSSIAN_SIGMA,
    MASKED_GAUSSIAN_TRUNCATE,
    delete_pixels,
    deterministic_disjoint_control_mask,
    deterministic_random_mask,
    paired_intervention_metrics,
    patch_gate_to_pixel_mask,
    replace_with_local_ring_mean,
    replace_with_masked_gaussian_blur,
    retain_pixels,
    transform_mask_to_letterbox,
    union_box_mask,
)


class PixelInterventionTests(unittest.TestCase):
    def test_local_mean_replaces_only_mask_from_fixed_exterior_ring(self) -> None:
        grid = np.arange(20 * 20 * 3, dtype=np.uint8).reshape(20, 20, 3)
        image = Image.fromarray(grid, mode="RGB")
        content = np.ones((20, 20), dtype=bool)
        mask = np.zeros((20, 20), dtype=bool)
        mask[8:12, 8:12] = True
        output = np.asarray(replace_with_local_ring_mean(image, mask, content))
        ring = (
            ndimage.binary_dilation(mask, iterations=LOCAL_MEAN_RING_WIDTH)
            & content
            & ~mask
        )
        expected = np.rint(grid[ring].astype(np.float64).mean(axis=0)).astype(np.uint8)
        self.assertTrue(np.array_equal(output[~mask], grid[~mask]))
        self.assertTrue(
            np.array_equal(output[mask], np.broadcast_to(expected, output[mask].shape))
        )

    def test_masked_gaussian_is_deterministic_and_changes_only_mask(self) -> None:
        grid = np.zeros((32, 32, 3), dtype=np.uint8)
        grid[8:24, 8:24] = (220, 120, 40)
        grid[14:18, 14:18] = (0, 255, 255)
        image = Image.fromarray(grid, mode="RGB")
        content = np.zeros((32, 32), dtype=bool)
        content[4:28, 4:28] = True
        mask = np.zeros((32, 32), dtype=bool)
        mask[12:20, 12:20] = True
        first = np.asarray(replace_with_masked_gaussian_blur(image, mask, content))
        second = np.asarray(replace_with_masked_gaussian_blur(image, mask, content))
        self.assertEqual(MASKED_GAUSSIAN_SIGMA, 8.0)
        self.assertEqual(MASKED_GAUSSIAN_TRUNCATE, 3.0)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.array_equal(first[~mask], grid[~mask]))
        self.assertFalse(np.array_equal(first[mask], grid[mask]))

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
