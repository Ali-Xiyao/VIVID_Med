from __future__ import annotations

import unittest


class PromotionPolicyTests(unittest.TestCase):
    def test_primary_gate_has_five_expert_findings(self) -> None:
        import json
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        lock = json.loads(
            (root / "audit" / "vivid_spd_clean_lock.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(lock["promotion"]["primary_findings"]), 5)
        self.assertEqual(
            lock["promotion"]["nonnegative_finding_count_min"], 4
        )
        self.assertEqual(lock["promotion"]["large_decline_count_max"], 1)


if __name__ == "__main__":
    unittest.main()
