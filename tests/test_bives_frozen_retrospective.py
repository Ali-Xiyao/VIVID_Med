from __future__ import annotations

import copy
import unittest

from bives_cxr.frozen_retrospective import (
    aggregate_frozen_rows,
    build_retrospective_summary,
    normalize_frozen_rows,
)


def _row(*, sample_id: str, gain: float, tcig: float) -> dict:
    return {
        "sample_id": sample_id,
        "unit_id": sample_id,
        "canonical_statement_id": "consolidation",
        "binary_label": 1,
        "mechanism_eligible": True,
        "topk_target_coverage": 0.2 + gain,
        "random_target_coverage": 0.2,
        "topk_localization_gain": gain,
        "local_mean": {
            "target_effect": tcig + 0.1,
            "control_effect": 0.1,
            "tcig": tcig,
        },
        "masked_gaussian_blur": {
            "target_effect": tcig + 0.05,
            "control_effect": 0.05,
            "tcig": tcig,
        },
    }


class FrozenRetrospectiveContractTests(unittest.TestCase):
    def test_normalization_derives_gain_and_keeps_legacy_boundary(self) -> None:
        rows = normalize_frozen_rows([_row(sample_id="a", gain=0.1, tcig=0.2)], source="c5")
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]["localization_gain"], 0.1)
        self.assertNotIn("expert_region", rows[0])
        self.assertNotIn("explanation_region", rows[0])

    def test_normalization_rejects_mutated_gain_or_tcig(self) -> None:
        bad_gain = _row(sample_id="a", gain=0.1, tcig=0.2)
        bad_gain["topk_localization_gain"] = 0.5
        with self.assertRaisesRegex(ValueError, "localization gain"):
            normalize_frozen_rows([bad_gain], source="c6i")
        bad_tcig = _row(sample_id="b", gain=0.1, tcig=0.2)
        bad_tcig["local_mean"]["tcig"] = 0.9
        with self.assertRaisesRegex(ValueError, "TCIG arithmetic"):
            normalize_frozen_rows([bad_tcig], source="c5")

    def test_aggregate_is_identifier_free_and_records_unit_boundary(self) -> None:
        source = [
            _row(sample_id="a", gain=0.1, tcig=0.05),
            _row(sample_id="b", gain=0.2, tcig=0.10),
            _row(sample_id="c", gain=0.3, tcig=0.15),
        ]
        aggregate = aggregate_frozen_rows(normalize_frozen_rows(source, source="c5"))
        self.assertEqual(len(aggregate), 2)
        self.assertEqual(aggregate[0]["unique_units"], 3)
        self.assertFalse(aggregate[0]["patient_level_claim"])
        self.assertNotIn("sample_id", aggregate[0])
        self.assertAlmostEqual(aggregate[0]["localization_tcig_spearman_rho"], 1.0)

    def test_summary_is_deterministic_and_fail_closed_about_schema(self) -> None:
        aggregate = aggregate_frozen_rows(
            normalize_frozen_rows(
                [
                    _row(sample_id="a", gain=0.1, tcig=0.05),
                    _row(sample_id="b", gain=0.2, tcig=0.10),
                    _row(sample_id="c", gain=0.3, tcig=0.15),
                ],
                source="c6i",
            )
        )
        kwargs = {
            "source_sha256": {"b": "2", "a": "1"},
            "source_record_counts": {"c6i": 3, "c5": 756},
        }
        first = build_retrospective_summary(copy.deepcopy(aggregate), **kwargs)
        second = build_retrospective_summary(copy.deepcopy(aggregate), **kwargs)
        self.assertEqual(first, second)
        self.assertFalse(first["formal_result"])
        self.assertFalse(first["schema_boundary"]["compatible_with_primary_schema"])
        self.assertFalse(first["model_loaded"])
        self.assertFalse(first["scores_computed"])


if __name__ == "__main__":
    unittest.main()
