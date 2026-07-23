from __future__ import annotations

import json
import unittest
from pathlib import Path


class LockTests(unittest.TestCase):
    def test_primary_identity_is_hard_and_paired(self) -> None:
        root = Path(__file__).resolve().parents[1]
        lock = json.loads(
            (root / "audit" / "vivid_spd_clean_lock.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(lock["target"]["reliability_weighting"])
        self.assertFalse(lock["target"]["soft_posterior"])
        self.assertEqual(lock["arms"]["ums_prefix4"]["prefix_tokens"], 4)
        self.assertEqual(lock["arms"]["ums_spd4x2"]["groups"], 4)
        self.assertEqual(lock["arms"]["ums_spd4x2"]["tokens_per_group"], 2)
        self.assertEqual(
            lock["arms"]["ums_spd4x2"]["orthogonality_weight"], 0.02
        )

    def test_protected_surfaces_are_frozen(self) -> None:
        root = Path(__file__).resolve().parents[1]
        lock = json.loads(
            (root / "audit" / "vivid_spd_clean_lock.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            set(lock["protected_surfaces"]),
            {"CheXlocalize_test", "VinDr_test"},
        )


if __name__ == "__main__":
    unittest.main()
