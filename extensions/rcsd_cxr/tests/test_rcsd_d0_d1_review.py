import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_rcsd_d0_d1_review.py"
LOCK = ROOT / "audit" / "rcsd_d0_d1_review_lock.json"
TERMINAL_RESULT = (
    ROOT / "audit" / "rcsd_d0_d1_qwen35_2b_terminal_result.json"
)
SPEC = importlib.util.spec_from_file_location("d0_d1_review", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class D0D1ReviewTests(unittest.TestCase):
    def load_lock(self):
        return json.loads(LOCK.read_text(encoding="utf-8"))

    def load_terminal_result(self):
        return json.loads(TERMINAL_RESULT.read_text(encoding="utf-8"))

    def test_current_review_lock_is_valid(self):
        repo_root = ROOT if (ROOT / "legacy").is_dir() else ROOT.parents[1]
        self.assertEqual(MODULE.default_repo_root(), repo_root)
        self.assertEqual(MODULE.validate(self.load_lock()), [])
        self.assertEqual(
            MODULE.validate_terminal_result(
                lock=self.load_lock(),
                result=self.load_terminal_result(),
            ),
            [],
        )
        self.assertEqual(
            MODULE.validate_source_contract(
                repo_root=repo_root,
                source_contract=(
                    ROOT / "audit" / "tables" / "rcsd_d0_source_contract.csv"
                ),
            ),
            [],
        )

    def test_training_authorization_is_rejected_before_prerequisites(self):
        payload = self.load_lock()
        payload["status"] = "PREREQUISITES_IN_PROGRESS"
        payload["prerequisites"]["hard_ums_manifest_frozen"] = False
        payload["execution_authorized"] = True
        payload["training_jobs_allowed"] = 2
        errors = MODULE.validate(payload)
        self.assertIn("prerequisite state cannot authorize training", errors)

    def test_remote_allocation_drift_is_rejected(self):
        payload = self.load_lock()
        payload["execution_target"]["allocation_id"] = 4161
        self.assertIn(
            "remote allocation identity drifted",
            MODULE.validate(payload),
        )

    def test_teacher_substitution_is_rejected(self):
        payload = self.load_lock()
        payload["controlled_d0"]["method_contract"]["teacher"] = (
            "Qwen/Qwen2.5-1.5B-Instruct"
        )
        self.assertIn(
            "frozen primary teacher identity drifted",
            MODULE.validate(payload),
        )

    def test_external_test_opening_is_rejected(self):
        payload = self.load_lock()
        payload["test_sets_opened"] = ["CheXlocalize test"]
        self.assertIn("test sets must remain unopened", MODULE.validate(payload))

    def test_terminal_nll_arithmetic_drift_is_rejected(self):
        result = self.load_terminal_result()
        result["primary_gate"]["observed"] = -0.04
        errors = MODULE.validate_terminal_result(
            lock=self.load_lock(),
            result=result,
        )
        self.assertIn("terminal relative NLL arithmetic drifted", errors)


if __name__ == "__main__":
    unittest.main()
