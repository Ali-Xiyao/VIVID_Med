import csv
import tempfile
import unittest
from pathlib import Path

from scripts.build_mimic_source_labels import build_source_manifest, normalize_label, slug


class SourceLabelTest(unittest.TestCase):
    def test_normalize_label(self) -> None:
        self.assertEqual(normalize_label("1.0"), "1")
        self.assertEqual(normalize_label("-1"), "-1")
        self.assertEqual(normalize_label("nan"), "")

    def test_invalid_label_fails(self) -> None:
        with self.assertRaises(ValueError):
            normalize_label("2")

    def test_missing_source_row_stays_missing(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            canonical = root / "canonical.csv"
            canonical.write_text("study_id,split\n10,train\n", encoding="utf-8")
            headers = ["subject_id", "study_id", *(
                "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
                "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
                "Atelectasis", "Pneumothorax", "Pleural Effusion", "Fracture",
                "Support Devices",
            )]
            for name in ("chexpert.csv", "negbio.csv"):
                with (root / name).open("w", encoding="utf-8", newline="") as handle:
                    csv.DictWriter(handle, fieldnames=headers).writeheader()
            output = root / "output.csv"
            audit = build_source_manifest(
                canonical, root / "chexpert.csv", root / "negbio.csv", output
            )
            self.assertTrue(audit["pass"])
            self.assertEqual(audit["all_missing_rows"], 1)
            self.assertEqual(audit["source_rows_missing"]["chexpert"], 1)
            with output.open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["chexpert__cardiomegaly"], "")

    def test_slug(self) -> None:
        self.assertEqual(slug("Pleural Effusion"), "pleural_effusion")


if __name__ == "__main__":
    unittest.main()
