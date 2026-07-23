from __future__ import annotations

import json
import unittest
from pathlib import Path


class BoundedDiagnosticPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.lock = json.loads(
            (
                self.root
                / "audit"
                / "vivid_spd_bounded_diagnostic_lock.json"
            ).read_text(encoding="utf-8")
        )

    def test_only_two_diagnostic_arms_are_authorized(self) -> None:
        self.assertEqual(
            set(self.lock["diagnostic_arms"]),
            {"ums_prefix8", "ums_spd4x2_no_ortho"},
        )
        self.assertFalse(
            self.lock["diagnostic_arms"]["ums_prefix8"][
                "nomination_eligible"
            ]
        )

    def test_only_no_ortho_can_be_nominated(self) -> None:
        policy = self.lock["repair_nomination"]
        self.assertEqual(policy["candidate"], "ums_spd4x2_no_ortho")
        self.assertTrue(policy["must_pass_original_s3_gate"])
        self.assertTrue(policy["must_exceed_historical_spd_macro_auroc"])

    def test_protected_surfaces_remain_closed(self) -> None:
        self.assertEqual(
            set(self.lock["protected_surfaces"]),
            {"CheXlocalize_test", "VinDr_test"},
        )


if __name__ == "__main__":
    unittest.main()
