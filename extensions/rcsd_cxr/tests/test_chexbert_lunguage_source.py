import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_chexbert_lunguage_source.py"
)
SPEC = importlib.util.spec_from_file_location("chexbert_lunguage", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class CheXbertLunguageSourceTests(unittest.TestCase):
    def write_rows(self, path: Path, rows: list[dict[str, str]]) -> None:
        fields = ["subject_id", "study_id", "section", "section_report"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_findings_then_impression_and_history_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gold.csv"
            self.write_rows(
                path,
                [
                    {
                        "subject_id": "p2",
                        "study_id": "s20",
                        "section": "impr",
                        "section_report": "Impression text.",
                    },
                    {
                        "subject_id": "p2",
                        "study_id": "s20",
                        "section": "find",
                        "section_report": "Findings text.",
                    },
                    {
                        "subject_id": "p2",
                        "study_id": "s20",
                        "section": "hist",
                        "section_report": "Private history.",
                    },
                ],
            )
            reports, audit = MODULE.build_reports(path)
            self.assertEqual(reports[0]["report"], "Findings text.\nImpression text.")
            self.assertNotIn("Private history", reports[0]["report"])
            self.assertTrue(audit["history_excluded"])

    def test_conflicting_section_text_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gold.csv"
            self.write_rows(
                path,
                [
                    {
                        "subject_id": "p2",
                        "study_id": "s20",
                        "section": "find",
                        "section_report": "First.",
                    },
                    {
                        "subject_id": "p2",
                        "study_id": "s20",
                        "section": "find",
                        "section_report": "Second.",
                    },
                ],
            )
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                MODULE.build_reports(path)

    def test_source_value_mapping_preserves_blank(self):
        self.assertEqual(MODULE.RAW_TO_VALUE[0], "")
        self.assertEqual(MODULE.RAW_TO_VALUE[1], "1")
        self.assertEqual(MODULE.RAW_TO_VALUE[2], "0")
        self.assertEqual(MODULE.RAW_TO_VALUE[3], "-1")
        self.assertNotIn("Pleural Other", MODULE.RCSD_FINDINGS)


if __name__ == "__main__":
    unittest.main()
