from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vivid_spd_clean.model import (  # noqa: E402
    HistoricalPrefixProjector,
    HistoricalSPDProjector,
    build_projector,
)


class ProjectorContractTests(unittest.TestCase):
    def test_historical_prefix_shape(self) -> None:
        model = HistoricalPrefixProjector(dropout=0.0)
        output = model(torch.randn(2, 197, 768))
        self.assertEqual(tuple(output.shape), (2, 201, 1536))
        self.assertEqual(model.num_query_tokens, 4)
        self.assertEqual(float(model.orthogonality_loss()), 0.0)

    def test_historical_spd_shape_and_loss(self) -> None:
        model = HistoricalSPDProjector(dropout=0.0)
        output = model(torch.randn(2, 197, 768))
        self.assertEqual(tuple(output.shape), (2, 205, 1536))
        self.assertEqual(model.num_query_tokens, 8)
        self.assertTrue(bool(torch.isfinite(model.orthogonality_loss())))

    def test_diagnostics_are_explicit(self) -> None:
        prefix = build_projector(
            "ums_prefix8", vision_dim=768, output_dim=1536
        )
        spd = build_projector(
            "ums_spd4x2_no_ortho", vision_dim=768, output_dim=1536
        )
        self.assertEqual(prefix.num_query_tokens, 8)
        self.assertEqual(spd.num_query_tokens, 8)

    def test_unknown_arm_fails(self) -> None:
        with self.assertRaises(ValueError):
            build_projector("unknown", vision_dim=768, output_dim=1536)


if __name__ == "__main__":
    unittest.main()
