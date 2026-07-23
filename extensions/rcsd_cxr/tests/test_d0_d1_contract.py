import json
import math
import unittest

from rcsd_cxr.d0_d1_contract import (
    entropy_agreement_weight,
    render_hard_ums_target,
)


class D0D1ContractTests(unittest.TestCase):
    def test_all_sources_agree(self):
        result = entropy_agreement_weight(
            {
                "chexpert": "present",
                "negbio": "present",
                "chexbert": "present",
            }
        )
        self.assertEqual(result.hard_target, "present")
        self.assertEqual(result.weight, 1.0)
        self.assertEqual(result.observed_sources, 3)

    def test_disagreement_uses_normalized_entropy(self):
        result = entropy_agreement_weight(
            {
                "chexpert": "present",
                "negbio": "absent",
                "chexbert": "present",
            }
        )
        expected_entropy = -(
            (2.0 / 3.0) * math.log(2.0 / 3.0)
            + (1.0 / 3.0) * math.log(1.0 / 3.0)
        )
        expected_weight = 1.0 - expected_entropy / math.log(3.0)
        self.assertAlmostEqual(result.weight, expected_weight)

    def test_three_way_disagreement_has_zero_weight(self):
        result = entropy_agreement_weight(
            {
                "chexpert": "present",
                "negbio": "absent",
                "chexbert": "uncertain",
            }
        )
        self.assertAlmostEqual(result.weight, 0.0)

    def test_missing_reference_is_masked(self):
        result = entropy_agreement_weight(
            {
                "chexpert": "present",
                "negbio": "absent",
                "chexbert": None,
            }
        )
        self.assertIsNone(result.hard_target)
        self.assertEqual(result.weight, 0.0)

    def test_missing_corroboration_is_not_disagreement(self):
        result = entropy_agreement_weight(
            {"chexpert": None, "negbio": None, "chexbert": "absent"}
        )
        self.assertEqual(result.weight, 1.0)

    def test_hard_ums_renderer_omits_missing_without_reordering(self):
        rendered = render_hard_ums_target(
            {
                "Edema": "present",
                "Pneumonia": None,
                "Pleural Effusion": "absent",
            },
            study_view="PA",
        )
        payload = json.loads(rendered)
        self.assertEqual(list(payload["findings"]), ["Edema", "Pleural Effusion"])
        self.assertEqual(payload["findings"]["Edema"]["score"], None)
        self.assertEqual(payload["study_view"], "PA")

    def test_invalid_state_fails_closed(self):
        with self.assertRaises(ValueError):
            entropy_agreement_weight({"chexbert": "insufficient"})


if __name__ == "__main__":
    unittest.main()
