from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from bives_cxr.c6g_geometry import (
    CONTROL_VERSION,
    ControlSearchFailure,
    deterministic_continuous_location_connected_control_mask,
)
from bives_cxr.rescue_protocol import coordinate_zone, mask_geometry


class C6GGeometryContracts(unittest.TestCase):
    def _thresholds(self, location: float = 0.35) -> dict:
        return {
            "format_version": "bives_c6g_geometry_thresholds_v1",
            "control_version": CONTROL_VERSION,
            "accepted_rows": 752,
            "max_location_distance": location,
            "max_log_perimeter_ratio": 1.0,
            "log_perimeter_ratio_weight": 0.10,
            "source_rows": [],
        }

    def test_v2_is_exact_connected_disjoint_and_deterministic(self) -> None:
        content = np.zeros((48, 64), dtype=bool)
        content[3:45, 4:60] = True
        target = np.zeros_like(content)
        target[17:30, 8:24] = True
        first, audit = deterministic_continuous_location_connected_control_mask(
            target,
            content,
            seed_key="c6g-contract",
            thresholds=self._thresholds(),
        )
        second, second_audit = deterministic_continuous_location_connected_control_mask(
            target,
            content,
            seed_key="c6g-contract",
            thresholds=self._thresholds(),
        )
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(audit, second_audit)
        self.assertEqual(audit["version"], CONTROL_VERSION)
        self.assertFalse(bool((first & target).any()))
        self.assertFalse(bool((first & ~content).any()))
        self.assertEqual(int(first.sum()), int(target.sum()))
        self.assertEqual(mask_geometry(first, content)["component_count"], 1)
        self.assertFalse(audit["model_evaluation_authorized"])
        self.assertFalse(audit["gpu_authorized"])
        self.assertFalse(audit["image_decode_authorized"])
        self.assertFalse(audit["scores_accessed"])

    def test_continuous_gate_can_cross_categorical_zone_boundary(self) -> None:
        content = np.ones((32, 40), dtype=bool)
        target = np.zeros_like(content)
        target[:, :16] = True
        control, audit = deterministic_continuous_location_connected_control_mask(
            target,
            content,
            seed_key="zone-boundary",
            thresholds=self._thresholds(location=0.45),
        )
        target_zone = coordinate_zone(target, content)
        control_zone = coordinate_zone(control, content)
        self.assertLessEqual(
            audit["selected_candidate"]["location_distance"],
            0.45,
        )
        self.assertEqual(audit["qualifying_candidate_count"] > 0, True)
        self.assertEqual(mask_geometry(control, content)["component_count"], 1)
        self.assertEqual(target_zone["horizontal"], "left")
        self.assertEqual(control_zone["horizontal"], "central")

    def test_failure_returns_complete_candidate_certificate(self) -> None:
        content = np.ones((24, 24), dtype=bool)
        target = np.zeros_like(content)
        target[8:16, 8:16] = True
        with self.assertRaises(ControlSearchFailure) as caught:
            deterministic_continuous_location_connected_control_mask(
                target,
                content,
                seed_key="forced-failure",
                thresholds=self._thresholds(location=0.0),
            )
        certificate = caught.exception.certificate
        self.assertIn("target", certificate)
        self.assertIn("valid_component_sizes", certificate)
        self.assertIn("candidate_counts", certificate)
        self.assertIn("nearest_candidate", certificate)
        self.assertTrue(certificate["nearest_candidate"]["rejection_reasons"])
        self.assertEqual(certificate["qualifying_candidate_count"], 0)

    def test_objective_matches_frozen_definition(self) -> None:
        content = np.ones((32, 40), dtype=bool)
        target = np.zeros_like(content)
        target[12:20, 5:13] = True
        _, audit = deterministic_continuous_location_connected_control_mask(
            target,
            content,
            seed_key="objective",
            thresholds=self._thresholds(location=1.0),
        )
        selected = audit["selected_candidate"]
        expected = selected["location_distance"] + 0.10 * selected["log_perimeter_ratio"]
        self.assertTrue(math.isclose(selected["objective"], expected, abs_tol=1e-12))


if __name__ == "__main__":
    unittest.main()
