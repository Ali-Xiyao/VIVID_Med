from __future__ import annotations

import unittest

from bives_cxr.intervention_diagnostics import summarize_intervention_failures


def make_row(
    index: int,
    *,
    finding: str,
    dilation: float,
    tcig: float,
    area: int,
    localization_gain: float,
) -> dict:
    control_effect = 0.1 + area / 100_000
    target_effect = control_effect + tcig
    return {
        "unit_id": f"image-{finding}-{index}",
        "canonical_statement_id": finding,
        "dilation_fraction": dilation,
        "original_score": 0.6 + index / 100,
        "keep_score": 0.2,
        "target_deletion_effect": target_effect,
        "control_deletion_effect": control_effect,
        "tcig": tcig,
        "target_area_pixels": area,
        "control_area_pixels": area,
        "topk_localization_gain": localization_gain,
    }


class InterventionDiagnosticTests(unittest.TestCase):
    def test_summary_localizes_robust_negative_and_localization_slices(self) -> None:
        rows = []
        for dilation in (0.0, 0.1):
            for index in range(8):
                rows.append(
                    make_row(
                        index,
                        finding="pleural_effusion",
                        dilation=dilation,
                        tcig=-0.08 + 0.01 * index,
                        area=1_000 + 500 * index,
                        localization_gain=0.01 * index,
                    )
                )
        result = summarize_intervention_failures(
            rows, bootstrap_replicates=200, bootstrap_seed=3
        )
        primary = result["results_by_dilation"]["0"]["pleural_effusion"]
        self.assertEqual(result["row_count"], 16)
        self.assertTrue(primary["robustness"]["all_leave_one_out_means_negative"])
        self.assertTrue(primary["diagnosis_flags"]["negative_after_10pct_trim"])
        self.assertTrue(
            primary["diagnosis_flags"]["high_localization_slice_better_than_low"]
        )
        self.assertEqual(result["dilation_stability"]["pleural_effusion"]["n"], 8)

    def test_rejects_area_mismatch_and_duplicate_tasks(self) -> None:
        row = make_row(
            0,
            finding="consolidation",
            dilation=0.0,
            tcig=0.1,
            area=1_000,
            localization_gain=0.2,
        )
        mismatch = dict(row, control_area_pixels=999)
        with self.assertRaisesRegex(ValueError, "area mismatch"):
            summarize_intervention_failures([mismatch], bootstrap_replicates=10)
        with self.assertRaisesRegex(ValueError, "duplicate intervention task"):
            summarize_intervention_failures([row, dict(row)], bootstrap_replicates=10)


if __name__ == "__main__":
    unittest.main()
