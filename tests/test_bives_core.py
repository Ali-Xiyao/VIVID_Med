"""CPU tests for the proposal-level BiVES-CXR contracts."""

from __future__ import annotations

import unittest

import torch

from bives_cxr.decoder import EvidenceStateDecoder
from bives_cxr.interventions import delete_evidence, retain_evidence
from bives_cxr.losses import BiVESLoss, nll_from_probs, total_variation
from bives_cxr.model import BiVESCXR, BiVESModelConfig


class DecoderTests(unittest.TestCase):
    def test_probabilities_match_closed_form_and_normalize(self) -> None:
        decoder = EvidenceStateDecoder(tau_a=2.0, tau_d=3.0, tau_p=4.0)
        positive = torch.tensor([0.0, 8.0, 1.0, 5.0])
        negative = torch.tensor([0.0, 1.0, 8.0, 5.0])
        output = decoder(positive, negative)
        availability = 1.0 - torch.exp(-(positive + negative) / 2.0)
        decisiveness = 1.0 - torch.exp(-torch.abs(positive - negative) / 3.0)
        polarity = torch.sigmoid((positive - negative) / 4.0)
        expected = torch.stack(
            (
                availability * decisiveness * polarity,
                availability * decisiveness * (1.0 - polarity),
                availability * (1.0 - decisiveness),
                1.0 - availability,
            ),
            dim=-1,
        )
        self.assertTrue(torch.allclose(output["state_probs"], expected))
        self.assertTrue(torch.allclose(output["state_probs"].sum(dim=-1), torch.ones(4)))

    def test_state_semantics_and_polarity_symmetry(self) -> None:
        decoder = EvidenceStateDecoder()
        positive = torch.tensor([0.0, 9.0, 1.0, 8.0])
        negative = torch.tensor([0.0, 1.0, 9.0, 8.0])
        probabilities = decoder(positive, negative)["state_probs"]
        self.assertEqual(int(probabilities[0].argmax()), 3)
        self.assertEqual(int(probabilities[1].argmax()), 0)
        self.assertEqual(int(probabilities[2].argmax()), 1)
        self.assertEqual(int(probabilities[3].argmax()), 2)
        swapped = decoder(negative, positive)["state_probs"]
        self.assertTrue(torch.allclose(probabilities[:, 0], swapped[:, 1], atol=1e-7, rtol=1e-6))
        self.assertTrue(torch.allclose(probabilities[:, 1], swapped[:, 0], atol=1e-7, rtol=1e-6))
        self.assertTrue(torch.allclose(probabilities[:, 2:], swapped[:, 2:], atol=1e-7, rtol=1e-6))

    def test_uncertain_and_insufficient_are_operationally_distinct(self) -> None:
        decoder = EvidenceStateDecoder()
        low_total = decoder(torch.tensor([0.1]), torch.tensor([0.1]))["state_probs"]
        high_total = decoder(torch.tensor([5.0]), torch.tensor([5.0]))["state_probs"]
        decisive = decoder(torch.tensor([9.0]), torch.tensor([1.0]))["state_probs"]
        less_decisive = decoder(torch.tensor([6.0]), torch.tensor([4.0]))["state_probs"]
        self.assertGreater(float(low_total[0, 3]), float(high_total[0, 3]))
        self.assertGreater(float(high_total[0, 2]), float(low_total[0, 2]))
        self.assertGreater(float(less_decisive[0, 2]), float(decisive[0, 2]))

    def test_insufficient_nll_has_evidence_magnitude_gradient(self) -> None:
        positive = torch.tensor([0.7], requires_grad=True)
        negative = torch.tensor([0.2], requires_grad=True)
        probabilities = EvidenceStateDecoder(tau_a=2.0)(positive, negative)["state_probs"]
        loss = nll_from_probs(probabilities, torch.tensor([3])).sum()
        loss.backward()
        self.assertAlmostEqual(float(positive.grad), 0.5, places=5)
        self.assertAlmostEqual(float(negative.grad), 0.5, places=5)


class InterventionTests(unittest.TestCase):
    def test_keep_drop_algebra(self) -> None:
        tokens = torch.randn(2, 4, 3)
        mask = torch.tensor([[1.0, 0.0, 0.5, 1.0], [0.0, 1.0, 0.25, 0.75]])
        mask_token = torch.randn(3)
        keep = retain_evidence(tokens, mask, mask_token)
        drop = delete_evidence(tokens, mask, mask_token)
        expected = tokens + mask_token.view(1, 1, -1)
        self.assertTrue(torch.allclose(keep + drop, expected))
        self.assertEqual(tuple(keep.shape), (2, 4, 3))

    def test_total_variation(self) -> None:
        constant = torch.ones(1, 9)
        self.assertEqual(float(total_variation(constant, (3, 3))), 0.0)
        center = torch.zeros(1, 9)
        center[0, 4] = 1.0
        self.assertEqual(float(total_variation(center, (3, 3))), 4.0)
        with self.assertRaises(ValueError):
            total_variation(center, (2, 4))


class ModelContractTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.model = BiVESCXR(
            BiVESModelConfig(
                visual_dim=8,
                statement_dim=6,
                fusion_dim=16,
                gate_mode="sigmoid",
                topk=2,
            )
        )

    def test_architecture_has_no_flat_state_head(self) -> None:
        self.assertEqual(self.model.decoder_kind, "bipolar_closed_form")
        self.assertFalse(self.model.has_flat_state_head)
        self.assertFalse(hasattr(self.model, "state_head"))
        self.assertFalse(hasattr(self.model, "four_class_head"))
        self.assertFalse(hasattr(self.model, "classifier"))
        self.assertEqual(sum(parameter.numel() for parameter in self.model.decoder.parameters()), 0)

    def test_forward_loss_backward_and_optimizer_step(self) -> None:
        patches = torch.randn(4, 4, 8)
        statements = torch.randn(4, 6)
        valid = torch.ones(4, 4, dtype=torch.bool)
        targets = torch.tensor([0, 1, 2, 3])
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        before = self.model.evidence_head.weight.detach().clone()
        outputs = self.model(patches, statements, valid)
        original = outputs["original"]
        self.assertEqual(tuple(original["state_probs"].shape), (4, 4))
        self.assertEqual(tuple(original["evidence_maps"].shape), (4, 4, 2))
        self.assertTrue(bool((original["evidence_pm"] >= 0).all()))
        self.assertTrue(bool(((original["gate"] >= 0) & (original["gate"] <= 1)).all()))
        losses = BiVESLoss()(outputs, targets, (2, 2))
        self.assertTrue(bool(torch.isfinite(losses["total"])))
        losses["total"].backward()
        optimizer.step()
        self.assertFalse(torch.allclose(before, self.model.evidence_head.weight.detach()))
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))


if __name__ == "__main__":
    unittest.main()
