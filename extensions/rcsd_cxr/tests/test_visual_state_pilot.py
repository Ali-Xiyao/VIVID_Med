import importlib.util
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "visual_state_pilot", SCRIPTS / "train_visual_state_pilot.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class VisualStatePilotTests(unittest.TestCase):
    def prototypes(self):
        dimension = 8
        return {
            "observation": torch.randn(12, dimension),
            "assertion": torch.randn(12, 3, dimension),
            "anatomy": torch.randn(12, dimension),
            "global": torch.randn(12, 3, dimension),
        }

    def test_targets_have_four_fields(self):
        states = torch.tensor([[0, 1, 2] + [-100] * 9])
        targets = MODULE.aggregate_targets(states, self.prototypes())
        self.assertEqual(tuple(targets.shape), (1, 4, 8))
        self.assertTrue(torch.isfinite(targets).all())

    def test_unanchored_loss_is_permutation_invariant(self):
        targets = torch.nn.functional.normalize(
            torch.randn(2, 4, 8), dim=-1
        )
        loss_a = MODULE.semantic_loss(targets, targets, "spd")
        loss_b = MODULE.semantic_loss(
            targets[:, [2, 0, 3, 1]], targets, "spd"
        )
        self.assertTrue(torch.allclose(loss_a, loss_b, atol=1e-6))

    def test_anchor_loss_keeps_field_identity(self):
        targets = torch.nn.functional.normalize(
            torch.randn(2, 4, 8), dim=-1
        )
        aligned = MODULE.semantic_loss(targets, targets, "field_anchor")
        shuffled = MODULE.semantic_loss(
            targets[:, [2, 0, 3, 1]], targets, "field_anchor"
        )
        self.assertLess(float(aligned), float(shuffled))

    def test_all_missing_rows_are_excluded_not_relabelled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            fields = [
                "split",
                "image_path",
                *MODULE.FINDINGS,
            ]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "split": "train",
                        "image_path": "missing.jpg",
                        **{finding: "" for finding in MODULE.FINDINGS},
                    }
                )
                writer.writerow(
                    {
                        "split": "train",
                        "image_path": "observed.jpg",
                        **{
                            finding: "1" if index == 0 else ""
                            for index, finding in enumerate(MODULE.FINDINGS)
                        },
                    }
                )
            dataset = MODULE.PilotDataset(
                path, Path(directory), "train", train=False
            )
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.excluded_all_missing, 1)


if __name__ == "__main__":
    unittest.main()
