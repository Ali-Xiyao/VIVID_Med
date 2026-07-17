from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from bives_cxr.polarity import BipolarPolarityModel, PolarityModelConfig
from bives_cxr.polarity_runtime import load_locked_polarity_checkpoint, score_statements


class PolarityRuntimeTests(unittest.TestCase):
    def test_locked_checkpoint_round_trip_and_statement_scoring(self) -> None:
        config = PolarityModelConfig(
            visual_dim=8,
            num_statements=2,
            statement_dim=4,
            fusion_dim=8,
            contextual_heads=2,
            mode="sparse_exact_k",
            topk=2,
        )
        source = BipolarPolarityModel(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            torch.save(
                {
                    "variant": "B2_sparse_exact_k",
                    "model_config": config.__dict__,
                    "model": source.state_dict(),
                    "statement_to_index": {"consolidation": 0, "pleural_effusion": 1},
                    "step": 25,
                    "cache_lock_sha256": "lock",
                },
                path,
            )
            model, vocabulary, payload = load_locked_polarity_checkpoint(path, torch.device("cpu"))
            output = score_statements(
                model,
                torch.randn(5, 8),
                torch.ones(5, dtype=torch.bool),
                [vocabulary["consolidation"], vocabulary["pleural_effusion"]],
            )
        self.assertEqual(payload["step"], 25)
        self.assertEqual(tuple(output["support_probability"].shape), (2,))
        self.assertTrue(torch.isfinite(output["support_probability"]).all())


if __name__ == "__main__":
    unittest.main()
