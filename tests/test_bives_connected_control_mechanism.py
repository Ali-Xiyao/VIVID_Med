from __future__ import annotations

import unittest

from scripts.evaluate_bives_connected_control_mechanism import (
    evaluate_c4_gate,
    localization_quartile,
    summarize_c4_operator,
)


class ConnectedControlMechanismTests(unittest.TestCase):
    def _rows(self, positive: bool = True) -> list[dict]:
        rows = []
        for finding in ("consolidation", "pleural_effusion"):
            for index in range(16):
                tcig = 0.2 if positive else -0.2
                rows.append(
                    {
                        "sample_id": f"{finding}-{index}",
                        "unit_id": f"{finding}-{index}",
                        "canonical_statement_id": finding,
                        "reader_consensus": "3_of_3" if index % 2 else "2_of_3",
                        "box_area_quartile": index % 4 + 1,
                        "topk_target_coverage": index / 16.0,
                        "local_mean": {
                            "target_effect": 0.3,
                            "control_effect": 0.3 - tcig,
                            "tcig": tcig,
                        },
                        "masked_gaussian_blur": {
                            "target_effect": 0.3,
                            "control_effect": 0.3 - tcig,
                            "tcig": tcig,
                        },
                    }
                )
        return rows

    def test_operator_summary_and_gate_pass_all_frozen_conditions(self) -> None:
        rows = self._rows(True)
        results = {
            operator: summarize_c4_operator(
                rows, operator, bootstrap_replicates=100, bootstrap_seed=17
            )
            for operator in ("local_mean", "masked_gaussian_blur")
        }
        gate = evaluate_c4_gate(results)
        self.assertTrue(gate["pass"])
        self.assertEqual(gate["status"], "pass")

    def test_gate_fails_negative_high_area_and_positive_fraction(self) -> None:
        rows = self._rows(False)
        results = {
            operator: summarize_c4_operator(
                rows, operator, bootstrap_replicates=100, bootstrap_seed=17
            )
            for operator in ("local_mean", "masked_gaussian_blur")
        }
        gate = evaluate_c4_gate(results)
        self.assertFalse(gate["pass"])
        self.assertEqual(gate["status"], "fail")

    def test_localization_quartile_boundaries_are_fixed(self) -> None:
        boundaries = [0.0, 0.25, 0.5, 0.75, 1.0]
        self.assertEqual(localization_quartile(0.1, boundaries), 1)
        self.assertEqual(localization_quartile(0.25, boundaries), 2)
        self.assertEqual(localization_quartile(0.5, boundaries), 3)
        self.assertEqual(localization_quartile(0.9, boundaries), 4)


if __name__ == "__main__":
    unittest.main()
