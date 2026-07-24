import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FreshReplicationLockTests(unittest.TestCase):
    def test_primary_identity_and_gate_are_frozen(self) -> None:
        lock = json.loads(
            (
                ROOT / "audit" / "vivid_gds_fresh_replication_lock.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            lock["pretraining"]["arms"],
            ["A0_direct", "A2_ums", "A3_gds"],
        )
        self.assertEqual(lock["pretraining"]["seeds"], [0, 1, 2])
        self.assertEqual(lock["pretraining"]["teacher"], "Qwen3.5-2B")
        self.assertEqual(lock["pretraining"]["pilot_steps"], 3000)
        self.assertEqual(lock["pretraining"]["lambda_schema"], 0.5)
        self.assertEqual(
            lock["promotion"]["mean_macro_auroc_delta_min"],
            {"A3_minus_A0": 0.005, "A3_minus_A2": 0.005},
        )
        self.assertEqual(
            lock["promotion"]["primary_bootstrap"]["replicates"], 10000
        )
        self.assertIn("CheXlocalize_test", lock["forbidden"])
        self.assertIn("same_surface_repair", lock["forbidden"])

    def test_split_partition_is_complete_and_disjoint(self) -> None:
        lock = json.loads(
            (
                ROOT / "audit" / "vivid_gds_fresh_replication_lock.json"
            ).read_text(encoding="utf-8")
        )
        data = lock["data"]
        groups = [
            set(data["fresh_development_buckets"]),
            set(data["probe_validation_buckets"]),
            set(data["probe_train_buckets"]),
        ]
        self.assertEqual(set().union(*groups), set(range(20)))
        self.assertEqual(sum(map(len, groups)), 20)


if __name__ == "__main__":
    unittest.main()
