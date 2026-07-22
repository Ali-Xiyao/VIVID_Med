from __future__ import annotations

import unittest

import torch

from arise_cxr.mil_verifier import MILVerifierConfig, PatchMILVerifier, mil_binary_loss


class AriseMILVerifierTests(unittest.TestCase):
    def test_is_permutation_invariant_over_valid_patches(self) -> None:
        model = PatchMILVerifier(MILVerifierConfig(visual_dim=4, num_statements=1))
        tokens = torch.tensor([[[1.0, 2.0, 0.0, -1.0], [0.0, 1.0, 3.0, -2.0]]])
        valid = torch.ones((1, 2), dtype=torch.bool)
        statement = torch.zeros(1, dtype=torch.long)
        first = model(tokens, valid, statement)["margin"]
        second = model(tokens[:, [1, 0]], valid, statement)["margin"]
        torch.testing.assert_close(first, second)

    def test_invalid_patch_does_not_change_margin(self) -> None:
        model = PatchMILVerifier(MILVerifierConfig(visual_dim=4, num_statements=1))
        tokens = torch.tensor([[[1.0, 2.0, 0.0, -1.0], [100.0, -100.0, 50.0, -50.0]]])
        valid = torch.tensor([[True, False]])
        statement = torch.zeros(1, dtype=torch.long)
        first = model(tokens, valid, statement)["margin"]
        tokens[:, 1] = torch.tensor([-7.0, 9.0, 11.0, -13.0])
        second = model(tokens, valid, statement)["margin"]
        torch.testing.assert_close(first, second)

    def test_binary_loss_prefers_correct_margin(self) -> None:
        correct = mil_binary_loss(torch.tensor([3.0, -3.0]), torch.tensor([1, 0]))
        wrong = mil_binary_loss(torch.tensor([-3.0, 3.0]), torch.tensor([1, 0]))
        self.assertLess(float(correct), float(wrong))


if __name__ == "__main__":
    unittest.main()
