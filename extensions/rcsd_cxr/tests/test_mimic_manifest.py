import unittest

from scripts.build_mimic_frontal_manifest import validate_patient_disjoint


class MimicManifestTest(unittest.TestCase):
    def test_patient_overlap_fails(self) -> None:
        rows = [
            {"patient_id": "p1", "split": "train"},
            {"patient_id": "p1", "split": "validate"},
        ]
        with self.assertRaisesRegex(ValueError, "patient split overlap"):
            validate_patient_disjoint(rows)

    def test_same_split_is_allowed(self) -> None:
        rows = [
            {"patient_id": "p1", "split": "train"},
            {"patient_id": "p1", "split": "train"},
        ]
        validate_patient_disjoint(rows)
