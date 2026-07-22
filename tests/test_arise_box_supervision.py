import unittest

import torch

from arise_cxr.box_supervision import box_ranking_loss, boxes_to_patch_mask


class AriseBoxSupervisionTests(unittest.TestCase):
    def test_maps_native_box_to_expected_intersecting_patches(self):
        mask = boxes_to_patch_mask(
            [{"x_min": 25, "y_min": 25, "x_max": 75, "y_max": 75}],
            native_width=100,
            native_height=100,
            grid_hw=(4, 4),
            valid_mask=torch.ones(16, dtype=torch.bool),
            image_size=100,
        )
        self.assertEqual(mask.reshape(4, 4).nonzero().tolist(), [
            [1, 1], [1, 2], [2, 1], [2, 2]
        ])

    def test_small_box_between_patch_centers_is_not_discarded(self):
        mask = boxes_to_patch_mask(
            [{"x_min": 24, "y_min": 24, "x_max": 26, "y_max": 26}],
            native_width=100,
            native_height=100,
            grid_hw=(4, 4),
            valid_mask=torch.ones(16, dtype=torch.bool),
            image_size=100,
        )
        self.assertEqual(mask.reshape(4, 4).nonzero().tolist(), [
            [0, 0], [0, 1], [1, 0], [1, 1]
        ])

    def test_ranking_loss_prefers_higher_inside_logits(self):
        valid = torch.ones((1, 4), dtype=torch.bool)
        box = torch.tensor([[True, True, False, False]])
        available = torch.tensor([True])
        good = box_ranking_loss(
            torch.tensor([[2.0, 2.0, -1.0, -1.0]]), valid, box, available
        )
        bad = box_ranking_loss(
            torch.tensor([[-1.0, -1.0, 2.0, 2.0]]), valid, box, available
        )
        self.assertLess(good.item(), bad.item())

    def test_ranking_loss_is_zero_without_box_rows(self):
        logits = torch.randn(2, 3, requires_grad=True)
        loss = box_ranking_loss(
            logits,
            torch.ones((2, 3), dtype=torch.bool),
            torch.zeros((2, 3), dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
        )
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
