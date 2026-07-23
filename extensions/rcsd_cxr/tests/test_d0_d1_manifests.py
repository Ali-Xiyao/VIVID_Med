import importlib.util
import unittest
from pathlib import Path

from rcsd_cxr.gold_mapping import FINDINGS


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_d0_d1_manifests.py"
SPEC = importlib.util.spec_from_file_location("build_d0_d1_manifests", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class D0D1ManifestTests(unittest.TestCase):
    def test_hard_target_is_not_changed_by_disagreement(self):
        base = {
            "patient_id": "p1",
            "study_id": "s1",
            "split": "train",
            "image_path": "files/p1/s1/i1.jpg",
            **{finding: "" for finding in FINDINGS},
        }
        base["Cardiomegaly"] = "1"
        base["Edema"] = "0"
        official = {
            "study_id": "s1",
            **{
                f"{source}__{MODULE.slug(finding)}": ""
                for source in ("chexpert", "negbio")
                for finding in FINDINGS
            },
        }
        official["chexpert__cardiomegaly"] = "0"
        official["negbio__cardiomegaly"] = "-1"
        official["chexpert__edema"] = "0"
        official["negbio__edema"] = "0"
        hard, sources, reliability, audit = MODULE.build_rows(
            [base], [official]
        )
        self.assertIn('"Cardiomegaly": {"state": "present"', hard[0]["target"])
        self.assertEqual(
            sources[0]["sources"]["Cardiomegaly"]["chexbert"], "present"
        )
        self.assertAlmostEqual(
            reliability[0]["finding_weights"]["Cardiomegaly"], 0.0, places=12
        )
        self.assertEqual(reliability[0]["finding_weights"]["Edema"], 1.0)
        self.assertEqual(audit["rows"], 1)

    def test_missing_chexbert_is_not_rendered(self):
        base = {
            "patient_id": "p1",
            "study_id": "s1",
            "split": "train",
            "image_path": "i.jpg",
            **{finding: "" for finding in FINDINGS},
        }
        base["Edema"] = "1"
        official = {
            "study_id": "s1",
            **{
                f"{source}__{MODULE.slug(finding)}": ""
                for source in ("chexpert", "negbio")
                for finding in FINDINGS
            },
        }
        hard, _, _, _ = MODULE.build_rows([base], [official])
        self.assertNotIn("Cardiomegaly", hard[0]["target"])
        self.assertIn("Edema", hard[0]["target"])


if __name__ == "__main__":
    unittest.main()
