import unittest

import torch

from rcsd_cxr.models.token_distillation import ExactSPDProjector


class ExactSPDProjectorTests(unittest.TestCase):
    def test_four_by_two_layout_and_historical_output_surface(self):
        module = ExactSPDProjector(
            vision_dim=16,
            output_dim=32,
            num_heads=4,
            dropout=0.0,
        )
        value = module(torch.randn(2, 5, 16))
        self.assertEqual(tuple(value.shape), (2, 13, 32))
        self.assertEqual(module.num_groups, 4)
        self.assertEqual(module.tokens_per_group, 2)
        self.assertEqual(module.num_query_tokens, 8)
        self.assertTrue(torch.isfinite(module.orthogonality_loss()))

    def test_nonhistorical_layout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "four groups"):
            ExactSPDProjector(num_groups=3)


if __name__ == "__main__":
    unittest.main()
