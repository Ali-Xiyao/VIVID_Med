import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_rcsd_d0_d1_review.py"
LOCK = ROOT / "audit" / "rcsd_d0_d1_review_lock.json"
SPEC = importlib.util.spec_from_file_location("d0_d1_review", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class D0D1ReviewTests(unittest.TestCase):
    def load_lock(self):
        return json.loads(LOCK.read_text(encoding="utf-8"))

    def test_current_review_lock_is_valid(self):
        self.assertEqual(MODULE.validate(self.load_lock()), [])
        self.assertEqual(
            MODULE.validate_source_contract(
                repo_root=ROOT.parents[1],
                source_contract=(
                    ROOT / "audit" / "tables" / "rcsd_d0_source_contract.csv"
                ),
            ),
            [],
        )

    def test_training_authorization_is_rejected(self):
        payload = self.load_lock()
        payload["execution_authorized"] = True
        payload["training_jobs_allowed"] = 2
        errors = MODULE.validate(payload)
        self.assertIn("execution must remain unauthorized", errors)
        self.assertIn("training_jobs_allowed must remain zero", errors)

    def test_teacher_substitution_is_rejected(self):
        payload = self.load_lock()
        payload["controlled_d0"]["method_contract"]["teacher"] = (
            "Qwen/Qwen3.5-2B"
        )
        self.assertIn(
            "historical teacher identity drifted",
            MODULE.validate(payload),
        )

    def test_external_test_opening_is_rejected(self):
        payload = self.load_lock()
        payload["test_sets_opened"] = ["CheXlocalize test"]
        self.assertIn("test sets must remain unopened", MODULE.validate(payload))


if __name__ == "__main__":
    unittest.main()
