from __future__ import annotations

import unittest

from scripts.audit_lunguage_gold import _normalize_id


class LunguageAuditTest(unittest.TestCase):
    def test_normalize_prefixed_ids(self) -> None:
        self.assertEqual(_normalize_id("p10274145"), "10274145")
        self.assertEqual(_normalize_id("s53183707"), "53183707")

    def test_empty_id_stays_empty(self) -> None:
        self.assertEqual(_normalize_id(""), "")


if __name__ == "__main__":
    unittest.main()
