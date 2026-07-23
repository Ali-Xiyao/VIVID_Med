import csv
from pathlib import Path
import tempfile
import unittest

from scripts.audit_mimic_gate0 import audit_rows


class Gate0AuditTest(unittest.TestCase):
    def test_forbidden_test_split_fails(self) -> None:
        rows = [
            {
                "patient_id": "1",
                "study_id": "2",
                "image_id": "3",
                "view_position": "PA",
                "split": "test",
                "image_path": "a.jpg",
                "report_path": "a.txt",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "forbidden split"):
                audit_rows(rows, image_root=root, report_root=root, check_paths=False)

    def test_valid_rows_are_counted(self) -> None:
        rows = [
            {
                "patient_id": "1",
                "study_id": "2",
                "image_id": "3",
                "view_position": "PA",
                "split": "train",
                "image_path": "a.jpg",
                "report_path": "a.txt",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = audit_rows(
                rows, image_root=root, report_root=root, check_paths=False
            )
        self.assertEqual(result["rows"], 1)
        self.assertEqual(result["test_rows"], 0)


if __name__ == "__main__":
    unittest.main()

