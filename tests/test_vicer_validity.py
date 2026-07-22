from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from vicer_cxr.intervention_bank import apply_v0_intervention
from vicer_cxr.case_study import analyze_v0_failure
from vicer_cxr.validity import (
    VALIDITY_FINDINGS,
    VALIDITY_ROLES,
    meets_minimum,
    summarize_v0_rows,
    validate_v0_manifest,
)
from vicer_cxr.matched_controls import deterministic_connected_statistics_control


class VicerValidityContracts(unittest.TestCase):
    def test_frozen_threshold_comparison_ignores_binary_float_artifact(self) -> None:
        self.assertTrue(meets_minimum(0.7999999999999999, 0.8))
        self.assertFalse(meets_minimum(0.799, 0.8))

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

    def test_connected_statistics_fallback_is_exact_and_deterministic(self) -> None:
        image = np.tile(np.linspace(0.0, 1.0, 32), (32, 1))
        content = np.ones((32, 32), dtype=bool)
        target = np.zeros_like(content)
        target[8:24, 8:24] = True
        first, audit = deterministic_connected_statistics_control(
            image, target, content, seed_key="unit"
        )
        second, _ = deterministic_connected_statistics_control(
            image, target, content, seed_key="unit"
        )
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(int(first.sum()), int(target.sum()))
        self.assertFalse(bool((first & target).any()))
        self.assertEqual(audit["connected_component_requirement"], 1)

    def test_failure_case_study_cannot_unlock_v1(self) -> None:
        rows = []
        per_finding = {}
        for finding in VALIDITY_FINDINGS:
            for strength in (0.25, 0.5, 0.75, 1.0):
                rows.append(
                    {
                        "sample_id": f"{finding}-{strength}",
                        "operator_family": "local_ring_mean",
                        "strength": strength,
                        "canonical_statement_id": finding,
                        "q_remove": 0.0,
                        "q_preserve": 0.995,
                        "q_realism": 0.9,
                        "target_control_gap": 0.1,
                        "valid_intervention": False,
                    }
                )
            per_finding[finding] = {
                "strengths": [0.25, 0.5, 0.75, 1.0],
                "median_q_remove": [0.0, 0.0, 0.0, 0.0],
                "median_q_preserve": [0.995] * 4,
                "median_q_realism": [0.9] * 4,
                "median_target_control_gap": [0.1] * 4,
                "q_remove_strength_spearman": 0.0,
                "valid_fraction": 0.0,
                "mean_valid_target_control_gap": 0.1,
                "pass": False,
            }
        result = {
            "schema_version": "vicer-v0-validity-dose-response-v1",
            "records": len(rows),
            "head_gate_pass": True,
            "per_operator_family": {
                "local_ring_mean": {
                    "per_finding": per_finding,
                    "all_findings_pass": False,
                }
            },
            "surviving_operator_families": [],
            "v0_pass": False,
            "v1_authorized": False,
            "thresholds": {
                "minimum_monotonic_spearman": 0.8,
                "minimum_preservation": 0.98,
                "minimum_realism": 0.5,
                "minimum_valid_fraction": 0.5,
            },
            "run_identity": {"thresholds": {"minimum_q_remove": 0.02}},
            "canonical_sha256": "result",
            "rows_sha256": "rows",
        }
        report = analyze_v0_failure(rows, result)
        self.assertEqual(report["cells_passed"], 0)
        self.assertEqual(
            report["diagnosis"]["dominant_invalid_row_component"], "removal"
        )
        self.assertFalse(report["boundaries"]["v1_authorized"])


if __name__ == "__main__":
    unittest.main()
