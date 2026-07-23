from pathlib import Path
import tempfile
import unittest

from rcsd_cxr.data import ManifestImageDataset


class ManifestDatasetTest(unittest.TestCase):
    def test_missing_image_fails_instead_of_black_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            manifest.write_text(
                "patient_id,study_id,image_id,image_path,split\n"
                "p1,s1,i1,missing.jpg,train\n",
                encoding="utf-8",
            )
            dataset = ManifestImageDataset(manifest, image_root=root)
            with self.assertRaises(FileNotFoundError):
                _ = dataset[0]

    def test_duplicate_image_identity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            manifest.write_text(
                "patient_id,study_id,image_id,image_path,split\n"
                "p1,s1,i1,a.jpg,train\n"
                "p2,s2,i1,b.jpg,train\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate image_id"):
                ManifestImageDataset(manifest)
