from __future__ import annotations

import unittest

from rcsd_cxr.gold_mapping import aggregate_study_gold, entity_state, match_findings


class GoldMappingTest(unittest.TestCase):
    def test_tentative_takes_precedence(self) -> None:
        self.assertEqual(
            entity_state({"dx_status": "positive", "dx_certainty": "tentative"}),
            "uncertain",
        )

    def test_unmentioned_is_not_absent(self) -> None:
        self.assertIsNone(entity_state({"dx_status": "", "dx_certainty": ""}))

    def test_pericardial_effusion_is_not_pleural(self) -> None:
        row = {"cat": "pf", "normed_ent": "pericardial effusion"}
        self.assertNotIn("Pleural Effusion", match_findings(row))

    def test_non_radiographic_fracture_is_excluded(self) -> None:
        row = {"cat": "ncd", "normed_ent": "tibial plateau fracture"}
        self.assertNotIn("Fracture", match_findings(row))

    def test_study_aggregation_priority(self) -> None:
        base = {"subject_id": "p1", "study_id": "s2", "cat": "pf"}
        rows = [
            {**base, "normed_ent": "pneumothorax", "dx_status": "negative", "dx_certainty": "definitive"},
            {**base, "normed_ent": "pneumothorax", "dx_status": "positive", "dx_certainty": "tentative"},
        ]
        labels, _ = aggregate_study_gold(rows)
        self.assertEqual(labels[("2", "Pneumothorax")], 2)


if __name__ == "__main__":
    unittest.main()
