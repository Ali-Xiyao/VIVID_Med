from __future__ import annotations

import unittest

from bives_cxr.vindr_qwen35_development import select_development_rows, shard_rows
from scripts.merge_vindr_qwen35_development import relabel_summary_as_image_level


def _row(finding: str, quartile: int, index: int, consensus: str = "2_of_3") -> dict:
    return {
        "sample_id": f"{finding}-{quartile}-{index}",
        "image_id": f"image-{finding}-{quartile}-{index}",
        "canonical_statement_id": finding,
        "box_area_quartile": quartile,
        "reader_consensus": consensus,
        "binary_label": 1,
        "rescue_split": "protocol_design",
    }


class VinDrQwen35DevelopmentContractTests(unittest.TestCase):
    def test_selection_is_four_unique_rows_and_prefers_consensus(self) -> None:
        rows = []
        for finding in ("consolidation", "pleural_effusion"):
            for quartile in (1, 4):
                rows.append(_row(finding, quartile, 0))
                rows.append(_row(finding, quartile, 1, "3_of_3"))
        selected = select_development_rows(rows)
        self.assertEqual(len(selected), 4)
        self.assertEqual(len({row["image_id"] for row in selected}), 4)
        self.assertTrue(all(row["reader_consensus"] == "3_of_3" for row in selected))

    def test_selection_rejects_missing_stratum(self) -> None:
        with self.assertRaisesRegex(ValueError, "no unique VinDr row"):
            select_development_rows([_row("consolidation", 1, 0)])

    def test_two_shards_are_disjoint_and_complete(self) -> None:
        rows = [{"id": index} for index in range(4)]
        left = shard_rows(rows, 0, 2)
        right = shard_rows(rows, 1, 2)
        self.assertEqual(left, [rows[0], rows[2]])
        self.assertEqual(right, [rows[1], rows[3]])
        self.assertFalse({id(row) for row in left} & {id(row) for row in right})

    def test_summary_relabels_patient_terms_as_image_units(self) -> None:
        summary = {
            "patients": 4,
            "groups": {
                "cell": {
                    "patients": 2,
                    "patient_cluster_bootstrap_95ci": {"CS_E": {"lower": 0.0, "upper": 1.0}},
                }
            },
        }
        result = relabel_summary_as_image_level(summary)
        self.assertEqual(result["image_units"], 4)
        self.assertFalse(result["patient_level_claim"])
        self.assertEqual(result["groups"]["cell"]["image_units"], 2)
        self.assertNotIn("patients", result)
        self.assertNotIn("patient_cluster_bootstrap_95ci", result["groups"]["cell"])


if __name__ == "__main__":
    unittest.main()
