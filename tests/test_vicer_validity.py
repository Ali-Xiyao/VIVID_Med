from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from vicer_cxr.intervention_bank import apply_v0_intervention
from vicer_cxr.validity import (
    VALIDITY_FINDINGS,
    VALIDITY_ROLES,
    summarize_v0_rows,
    validate_v0_manifest,
)


class VicerValidityContracts(unittest.TestCase):
    def _manifest(self) -> list[dict]:
        rows = []
        for finding in VALIDITY_FINDINGS:
            for role, positives in VALIDITY_ROLES.items():
                for state, count in (
                    ("support", positives),
                    ("contradict", 0 if role == "validity_eval" else positives),
                ):
                    for index in range(count):
                        image_id = f"{finding}-{role}-{state}-{index}"
                        rows.append(
                            {
                                "sample_id": image_id,
                                "image_id": image_id,
                                "canonical_statement_id": finding,
                                "v0_role": role,
                                "state": state,
                                "source_split": "train",
                                "patient_level_claim": False,
                                "arise_identity_excluded": True,
                                "positive_vote_count": 1 if state == "support" else 0,
                                "roi_boxes": [{"x_min": 1, "y_min": 1, "x_max": 2, "y_max": 2}],
                            }
                        )
        return rows

    def test_manifest_requires_exact_disjoint_roles(self) -> None:
        summary = validate_v0_manifest(self._manifest())
        self.assertEqual(summary["records"], 280)
        bad = self._manifest()
        bad[-1]["image_id"] = bad[0]["image_id"]
        with self.assertRaisesRegex(ValueError, "leaked"):
            validate_v0_manifest(bad)

    def test_strength_controlled_interventions_only_modify_mask(self) -> None:
        array = np.zeros((32, 32, 3), dtype=np.uint8)
        array[:, :, 0] = np.arange(32, dtype=np.uint8)[None, :] * 8
        image = Image.fromarray(array)
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:24, 8:24] = True
        content = np.ones_like(mask)
        for family, strength in (
            ("masked_gaussian_blur", 4.0),
            ("local_ring_mean", 0.5),
            ("low_frequency_replacement", 0.75),
        ):
            changed, audit = apply_v0_intervention(
                image, mask, content, family=family, strength=strength
            )
            output = np.asarray(changed)
            self.assertTrue(np.array_equal(output[~mask], array[~mask]))
            self.assertEqual(audit["family"], family)

    def test_summary_unlocks_v1_only_for_all_finding_family(self) -> None:
        rows = []
        for finding in VALIDITY_FINDINGS:
            for sample in range(2):
                for strength, q_remove in zip((0.25, 0.5, 0.75, 1.0), (0.03, 0.05, 0.08, 0.11)):
                    rows.append(
                        {
                            "sample_id": f"{finding}-{sample}",
                            "operator_family": "local_ring_mean",
                            "strength": strength,
                            "canonical_statement_id": finding,
                            "q_remove": q_remove,
                            "q_preserve": 0.995,
                            "q_realism": 0.9,
                            "target_control_gap": 0.1,
                            "valid_intervention": True,
                            "critic_calibration_auroc": 0.8,
                            "verifier_calibration_auroc": 0.8,
                        }
                    )
        result = summarize_v0_rows(
            rows,
            minimum_critic_auroc=0.6,
            minimum_verifier_auroc=0.6,
            minimum_monotonic_spearman=0.8,
            minimum_preservation=0.98,
            minimum_realism=0.5,
            minimum_valid_fraction=0.5,
        )
        self.assertTrue(result["v0_pass"])
        rows[0]["critic_calibration_auroc"] = 0.4
        self.assertFalse(
            summarize_v0_rows(
                rows,
                minimum_critic_auroc=0.6,
                minimum_verifier_auroc=0.6,
                minimum_monotonic_spearman=0.8,
                minimum_preservation=0.98,
                minimum_realism=0.5,
                minimum_valid_fraction=0.5,
            )["v1_authorized"]
        )


if __name__ == "__main__":
    unittest.main()
