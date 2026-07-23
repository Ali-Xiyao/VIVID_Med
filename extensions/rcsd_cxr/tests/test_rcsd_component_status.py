import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_rcsd_component_status.py"
STATUS = ROOT / "audit" / "rcsd_component_status.json"
SPEC = importlib.util.spec_from_file_location("component_status", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class RCSDComponentStatusTests(unittest.TestCase):
    def test_frozen_status_is_internally_consistent(self):
        payload = json.loads(STATUS.read_text(encoding="utf-8"))
        self.assertEqual(MODULE.validate(payload), [])

    def test_reopening_d1_is_rejected(self):
        payload = json.loads(STATUS.read_text(encoding="utf-8"))
        payload["arms"]["D1"]["status"] = "PASS"
        self.assertIn(
            "D1 must remain untested until separately authorized",
            MODULE.validate(payload),
        )

    def test_opening_a_test_set_is_rejected(self):
        payload = json.loads(STATUS.read_text(encoding="utf-8"))
        payload["test_sets_opened"] = ["CheXlocalize test"]
        self.assertIn(
            "test_sets_opened must remain empty",
            MODULE.validate(payload),
        )


if __name__ == "__main__":
    unittest.main()
