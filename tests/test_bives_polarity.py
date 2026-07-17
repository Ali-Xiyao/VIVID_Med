from __future__ import annotations

import unittest

import torch

from bives_cxr.polarity import BipolarPolarityModel, PolarityModelConfig, polarity_loss


class BipolarPolarityTests(unittest.TestCase):
    def test_dense_uses_every_valid_patch_without_selector(self) -> None:
        model = BipolarPolarityModel(
            PolarityModelConfig(visual_dim=8, num_statements=2, fusion_dim=16, mode="dense")
        )
        valid = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
        output = model(torch.randn(2, 4, 8), valid, torch.tensor([0, 1]))
        self.assertIsNone(model.gate_head)
        self.assertTrue(torch.equal(output["gate"].bool(), valid))
        self.assertFalse(model.has_flat_binary_head)

    def test_sparse_has_exact_k_gate_and_polarity_backward(self) -> None:
        model = BipolarPolarityModel(
            PolarityModelConfig(
                visual_dim=8,
                num_statements=2,
                fusion_dim=16,
                mode="sparse_exact_k",
                topk=2,
            )
        )
        output = model(
            torch.randn(4, 6, 8),
            torch.ones(4, 6, dtype=torch.bool),
            torch.tensor([0, 0, 1, 1]),
        )
        self.assertTrue(torch.equal(output["gate"].detach().sum(dim=1), torch.tensor([2.0] * 4)))
        loss = polarity_loss(output["signed_evidence"], torch.tensor([1, 0, 1, 0]))
        loss.backward()
        self.assertIsNotNone(model.evidence_head.weight.grad)
        self.assertIsNotNone(model.gate_head.weight.grad)

    def test_polarity_loss_prefers_correct_signed_evidence(self) -> None:
        labels = torch.tensor([1, 0])
        correct = polarity_loss(torch.tensor([3.0, -3.0]), labels)
        wrong = polarity_loss(torch.tensor([-3.0, 3.0]), labels)
        self.assertLess(float(correct), float(wrong))


if __name__ == "__main__":
    unittest.main()
