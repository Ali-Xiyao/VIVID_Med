import unittest

from scripts.audit_dataset_qualification import audit_qualification
from scripts.build_chexpert_lineage_manifest import normalize_image_path


class DatasetQualificationTest(unittest.TestCase):
    def test_normalize_chexpert_path(self) -> None:
        self.assertEqual(
            normalize_image_path(
                "CheXpert-v1.0-small/valid/patient1/study1/view1_frontal.jpg"
            ),
            "valid/patient1/study1/view1_frontal.jpg",
        )

    def test_track_a_and_seal_pass_with_conditional_external(self) -> None:
        base = {
            "version": "v",
            "parent": None,
            "roles": ["track_a_train"],
            "access": "local",
            "license_status": "recorded",
            "eligibility": "qualified",
            "test_exposure": "train_only",
            "evidence": ["e"],
        }
        datasets = {
            "mimic_cxr": dict(base),
            "mimic_cxr_metadata": {**base, "parent": "mimic_cxr"},
            "chexpert": {**base, "eligibility": "conditional"},
            "chexpert_plus": {
                **base,
                "parent": "chexpert",
                "eligibility": "conditional",
            },
            "chexlocalize_validation": {
                **base,
                "parent": "chexpert",
                "eligibility": "excluded",
            },
            "chexlocalize_test": {
                **base,
                "parent": "chexpert",
                "eligibility": "excluded",
                "test_exposure": "sealed_absent",
            },
        }
        lineage = {
            "summary": {
                "multi_split_patients": 0,
                "valid_path_symmetric_difference": 0,
            },
            "decision": {"chexpert_plus_valid_is_official_chexpert_valid": True},
        }
        result = audit_qualification(
            {"datasets": datasets}, chexpert_lineage=lineage
        )
        self.assertTrue(result["pass"])
        self.assertIn("chexpert", result["conditional"])


if __name__ == "__main__":
    unittest.main()
