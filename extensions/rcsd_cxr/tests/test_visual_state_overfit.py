import importlib.util
import sys
import unittest
from pathlib import Path

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "visual_state_overfit", SCRIPTS / "train_visual_state_overfit.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class VisualStateOverfitTests(unittest.TestCase):
    def test_masked_loss_ignores_missing(self):
        logits = torch.tensor(
            [[[0.0, 3.0, 0.0], [3.0, 0.0, 0.0]]]
        )
        targets = torch.tensor([[1, -100]])
        loss = MODULE.masked_loss(logits, targets)
        expected = torch.nn.functional.cross_entropy(
            logits[:, :1, :].reshape(-1, 3), torch.tensor([1])
        )
        self.assertTrue(torch.allclose(loss, expected))


if __name__ == "__main__":
    unittest.main()
