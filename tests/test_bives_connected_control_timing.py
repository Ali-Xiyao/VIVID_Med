from __future__ import annotations

import unittest

from scripts.audit_bives_connected_control_timing_replay import (
    C4_TIMING_MULTIPLIER,
    C4_VARIANTS_PER_ROW,
    estimate_c4_hours,
    select_c3_rows,
)


class ConnectedControlTimingTests(unittest.TestCase):
    def test_c3_selection_is_balanced_unique_and_stable(self) -> None:
        manifest = []
        geometry = []
        for finding in ("consolidation", "pleural_effusion"):
            for index in range(12):
                unit = f"shared-{index}" if finding == "pleural_effusion" and index < 3 else f"{finding}-{index}"
                sample = f"{finding}-{index:02d}"
                manifest.append(
                    {
                        "sample_id": sample,
                        "unit_id": unit,
                        "canonical_statement_id": finding,
                        "rescue_split": "protocol_design",
                        "source_split": "train",
                        "image_path": f"H:/dataset/train/{unit}.dicom",
                    }
                )
                geometry.append(
                    {
                        "sample_id": sample,
                        "canonical_statement_id": finding,
                        "rescue_split": "protocol_design",
                        "source_split": "train",
                        "feasible": True,
                    }
                )
        first = select_c3_rows(manifest, geometry)
        second = select_c3_rows(list(reversed(manifest)), list(reversed(geometry)))
        self.assertEqual(
            [row["sample_id"] for row in first],
            [row["sample_id"] for row in second],
        )
        self.assertEqual(len(first), 16)
        self.assertEqual(len({row["unit_id"] for row in first}), 16)
        self.assertEqual(
            [row["canonical_statement_id"] for row in first].count("consolidation"),
            8,
        )
        self.assertEqual(
            [row["canonical_statement_id"] for row in first].count("pleural_effusion"),
            8,
        )

    def test_c4_estimate_uses_slower_pass_and_fixed_overhead(self) -> None:
        estimate = estimate_c4_hours(
            [16.0, 32.0],
            sample_count=16,
            eligible_rows=100,
            geometry_seconds=60.0,
        )
        expected_seconds = (
            2.0 * 100 * C4_VARIANTS_PER_ROW * C4_TIMING_MULTIPLIER + 60.0
        )
        self.assertAlmostEqual(estimate, expected_seconds / 3600.0)

    def test_c3_selection_rejects_test_paths(self) -> None:
        manifest = []
        geometry = []
        for finding in ("consolidation", "pleural_effusion"):
            for index in range(8):
                sample = f"{finding}-{index:02d}"
                manifest.append(
                    {
                        "sample_id": sample,
                        "unit_id": sample,
                        "canonical_statement_id": finding,
                        "rescue_split": "protocol_design",
                        "source_split": "train",
                        "image_path": f"H:/dataset/{'test' if index == 0 else 'train'}/{sample}.dicom",
                    }
                )
                geometry.append(
                    {
                        "sample_id": sample,
                        "canonical_statement_id": finding,
                        "rescue_split": "protocol_design",
                        "source_split": "train",
                        "feasible": True,
                    }
                )
        with self.assertRaisesRegex(ValueError, "VinDr-test path is forbidden"):
            select_c3_rows(manifest, geometry)


if __name__ == "__main__":
    unittest.main()
