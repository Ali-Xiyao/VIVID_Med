import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LockTests(unittest.TestCase):
    def test_primary_identity_and_gates_are_frozen(self) -> None:
        lock = json.loads(
            (ROOT / "audit" / "vivid_gds_stage_a_lock.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(lock["method"]["name"], "VIVID-GDS")
        self.assertEqual(lock["method"]["teacher"], "Qwen3.5-2B")
        self.assertEqual(lock["method"]["lambda_schema"], 0.5)
        self.assertEqual(
            lock["promotion"]["A3_minus_A2"]["macro_auroc_at_least"],
            0.005,
        )
        self.assertIn("CheXlocalize_test", lock["forbidden"])
        self.assertIn("VinDr_test", lock["forbidden"])
