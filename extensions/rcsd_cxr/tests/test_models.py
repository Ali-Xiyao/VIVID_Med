import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class ModelContractTest(unittest.TestCase):
    def test_spd_is_frozen_at_four_by_two(self) -> None:
        from rcsd_cxr.models import SPDProjector

        model = SPDProjector(vision_dim=16, output_dim=24, num_heads=4, dropout=0.0)
        tokens = torch.randn(2, 9, 16)
        output = model(tokens)
        self.assertEqual(tuple(output.shape), (2, 8, 24))
        self.assertTrue(torch.isfinite(model.decorrelation_loss()))
        with self.assertRaises(ValueError):
            SPDProjector(vision_dim=16, output_dim=24, num_groups=3)

    def test_field_anchor_matches_token_budget(self) -> None:
        from rcsd_cxr.models import FieldAnchoredProjector, SPDProjector

        model = FieldAnchoredProjector(vision_dim=16, output_dim=24, num_heads=4, dropout=0.0)
        fields = model(torch.randn(2, 9, 16))
        flat = model.flatten(fields)
        self.assertEqual(tuple(flat.shape), (2, 8, 24))
        baseline = SPDProjector(
            vision_dim=16, output_dim=24, num_heads=4, dropout=0.0
        )
        baseline_count = sum(
            parameter.numel() for parameter in baseline.parameters()
        )
        anchored_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        self.assertEqual(baseline_count, anchored_count)
        self.assertNotIn("field_embeddings", dict(model.named_parameters()))
