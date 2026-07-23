import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "mimic_chexbert_pilot", SCRIPTS / "build_mimic_chexbert_pilot.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class MimicChexbertPilotTests(unittest.TestCase):
    def test_extracts_findings_then_impression(self):
        text = """FINAL REPORT
INDICATION: cough
FINDINGS:
No focal opacity.
IMPRESSION:
No acute process.
"""
        value, pattern = MODULE.extract_chexbert_text(text)
        self.assertEqual(value, "No focal opacity.\nNo acute process.")
        self.assertEqual(pattern, "find+impr")
        self.assertNotIn("cough", value)

    def test_full_report_fallback_is_explicit(self):
        value, pattern = MODULE.extract_chexbert_text("Unstructured report.")
        self.assertEqual(value, "Unstructured report.")
        self.assertEqual(pattern, "full_report_fallback")

    def test_stable_selection_keeps_all_validation(self):
        rows = [
            {
                "patient_id": str(index),
                "study_id": str(index),
                "split": "train",
            }
            for index in range(4)
        ] + [
            {"patient_id": "v", "study_id": "9", "split": "validate"}
        ]
        selected, counts = MODULE.select_rows(rows, 2)
        self.assertEqual(counts, {"train": 2, "validate": 1, "total": 3})
        self.assertIn("validate", {row["split"] for row in selected})


if __name__ == "__main__":
    unittest.main()
