from __future__ import annotations

import unittest

from bives_cxr.vindr_qwen35_expanded_development import select_expanded_rows


def _row(finding: str, quartile: int, index: int, consensus: str) -> dict:
    return {
        "sample_id": f"{finding}-{quartile}-{index}",
        "image_id": f"image-{finding}-{quartile}-{index}",
        "canonical_statement_id": finding,
        "box_area_quartile": quartile,
        "reader_consensus": consensus,
        "binary_label": 1,
        "rescue_split": "protocol_design",
    }


class ExpandedVinDrQwen35DevelopmentTests(unittest.TestCase):
    def test_selects_four_per_stratum_and_excludes_pilot(self) -> None:
        rows = []
        for finding in ("consolidation", "pleural_effusion"):
            for quartile in (1, 2, 3, 4):
                for index in range(6):
                    rows.append(_row(finding, quartile, index, "3_of_3" if index > 0 else "2_of_3"))
        excluded = {rows[1]["image_id"]}
        selected = select_expanded_rows(rows, excluded_image_ids=excluded)
        self.assertEqual(len(selected), 32)
        self.assertFalse(excluded & {row["image_id"] for row in selected})
        self.assertEqual(len({row["image_id"] for row in selected}), 32)
        for finding in ("consolidation", "pleural_effusion"):
            for quartile in (1, 2, 3, 4):
                self.assertEqual(
                    sum(row["canonical_statement_id"] == finding and row["box_area_quartile"] == quartile for row in selected),
                    4,
                )

    def test_fails_closed_when_one_stratum_is_short(self) -> None:
        rows = [_row("consolidation", 1, index, "3_of_3") for index in range(3)]
        with self.assertRaisesRegex(ValueError, "insufficient unique VinDr rows"):
            select_expanded_rows(rows, excluded_image_ids=set())


if __name__ == "__main__":
    unittest.main()
