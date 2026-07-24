import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vivid_gds.objective import masked_schema_cross_entropy, schema_accuracy


class ObjectiveTests(unittest.TestCase):
    def test_masked_loss_ignores_missing(self) -> None:
        logits = torch.zeros(2, 12, 3, requires_grad=True)
        states = torch.full((2, 12), -100, dtype=torch.long)
        states[0, 1] = 0
        states[1, 1] = 1
        states[0, 4] = 2
        loss = masked_schema_cross_entropy(logits, states)
        self.assertAlmostEqual(float(loss), 1.0986123, places=5)
        loss.backward()
        self.assertEqual(int(torch.count_nonzero(logits.grad[:, 0])), 0)
        self.assertGreater(int(torch.count_nonzero(logits.grad[:, 1])), 0)

    def test_schema_accuracy_counts_only_observed(self) -> None:
        logits = torch.zeros(1, 12, 3)
        states = torch.full((1, 12), -100, dtype=torch.long)
        states[0, 0] = 0
        states[0, 1] = 2
        logits[0, 0, 0] = 2
        logits[0, 1, 1] = 2
        self.assertEqual(schema_accuracy(logits, states), (1, 2))
