from __future__ import annotations

import unittest

from arise_cxr.oracle_ceiling import evaluate_oracle_ceiling


def _row(patient: int, pathology: str, operator: str, cs_x: float) -> dict[str, object]:
    return {
        "schema_version": "cxr_localization_causality_audit_v1",
        "row_id": f"{patient}-{pathology}-{operator}",
        "patient_id": f"patient-{patient}",
        "image_id": f"image-{patient}",
        "pathology_id": pathology,
        "model_id": "model",
        "explanation_id": "explanation",
        "operator_id": operator,
        "dataset_role": "development",
        "formal_result": False,
        "test_opened": False,
        "score_direction": "higher_is_more_support",
        "localization": {"iou": 0.25, "point_hit": 1},
        "scores": {"CS_X": cs_x, "CS_E": 0.05},
        "geometry": {"X_vs_C_X": {"pass": True}, "E_vs_C_E": {"pass": True}},
        "strength": {"X_vs_C_X": {"pass": True}, "E_vs_C_E": {"pass": True}},
    }


class AriseOracleCeilingTests(unittest.TestCase):
    def _rows(self, values: dict[str, float]) -> list[dict[str, object]]:
        rows = []
        for pathology, value in values.items():
            for patient in range(6):
                rows.append(_row(patient, pathology, "blur", value + patient * 0.001))
                rows.append(_row(patient, pathology, "local_mean", value + patient * 0.001))
        return rows

    def test_gate_unlocks_only_when_three_pathologies_pass(self) -> None:
        result = evaluate_oracle_ceiling(
            self._rows({"a": 0.2, "b": 0.15, "c": 0.1}),
            required_pathologies=("a", "b", "c"),
            required_operators=("blur", "local_mean"),
            minimum_passing_pathologies=3,
            bootstrap_replicates=100,
            bootstrap_seed=7,
        )
        self.assertEqual(result["status"], "pass_unlock_sc_selector")
        self.assertTrue(result["selector_training_unlocked"])

    def test_gate_stops_on_failed_pathology_and_insufficient_coverage(self) -> None:
        result = evaluate_oracle_ceiling(
            self._rows({"a": 0.2, "b": -0.1}),
            required_pathologies=("a", "b"),
            required_operators=("blur", "local_mean"),
            minimum_passing_pathologies=3,
            bootstrap_replicates=100,
            bootstrap_seed=7,
        )
        self.assertEqual(result["status"], "fail_stop_before_selector")
        self.assertFalse(result["coverage_pass"])
        self.assertFalse(result["pathologies"]["b"]["pass"])
        self.assertFalse(result["selector_training_unlocked"])

    def test_gate_rejects_an_unexpected_operator_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "operator set changed"):
            evaluate_oracle_ceiling(
                self._rows({"a": 0.2}),
                required_pathologies=("a",),
                required_operators=("blur",),
                minimum_passing_pathologies=1,
                bootstrap_replicates=10,
                bootstrap_seed=7,
            )


if __name__ == "__main__":
    unittest.main()
