from __future__ import annotations

import unittest

import numpy as np

from bives_cxr.rescue_protocol import (
    COORDINATE_ZONE_CONTROL_VERSION,
    coordinate_zone,
    deterministic_coordinate_zone_connected_control_mask,
    deterministic_multilabel_half_split,
    deterministic_translated_control_mask,
    mask_geometry,
)


class RescueProtocolTests(unittest.TestCase):
    def test_multilabel_split_is_deterministic_grouped_and_balanced(self) -> None:
        labels = {
            f"image-{index:02d}": {
                f"finding-a|q{index % 4}",
                *( [f"finding-b|q{index % 3}"] if index % 5 == 0 else [] ),
            }
            for index in range(24)
        }
        first, audit = deterministic_multilabel_half_split(labels, seed=20260718)
        second, second_audit = deterministic_multilabel_half_split(labels, seed=20260718)
        self.assertEqual(first, second)
        self.assertEqual(audit, second_audit)
        self.assertEqual(set(first), set(labels))
        self.assertEqual(sum(value == "protocol_design" for value in first.values()), 12)
        self.assertTrue(
            all(row["absolute_half_deviation"] <= 1.0 for row in audit["strata"].values())
        )

    def test_translated_control_preserves_exact_topology_and_band(self) -> None:
        content = np.zeros((48, 64), dtype=bool)
        content[4:44, 2:62] = True
        target = np.zeros_like(content)
        target[12:18, 24:31] = True
        target[18:21, 27:30] = True
        first, audit = deterministic_translated_control_mask(
            target, content, seed_key="sample-a"
        )
        second, second_audit = deterministic_translated_control_mask(
            target, content, seed_key="sample-a"
        )
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(audit, second_audit)
        self.assertFalse(bool((first & target).any()))
        target_geometry = mask_geometry(target, content)
        control_geometry = mask_geometry(first, content)
        for key in (
            "area_pixels",
            "component_count",
            "perimeter_edges",
            "compactness",
            "bbox_aspect_ratio",
            "vertical_band",
        ):
            self.assertEqual(target_geometry[key], control_geometry[key])

    def test_translated_control_fails_closed_when_no_horizontal_space_exists(self) -> None:
        content = np.ones((20, 20), dtype=bool)
        target = np.zeros_like(content)
        target[5:15, 0:20] = True
        with self.assertRaisesRegex(ValueError, "no disjoint same-band"):
            deterministic_translated_control_mask(target, content, seed_key="too-wide")

    def test_translated_control_allows_overlapping_bounding_boxes(self) -> None:
        content = np.ones((30, 30), dtype=bool)
        target = np.zeros_like(content)
        target[11:14, 3:6] = True
        target[11:14, 23:26] = True
        control, audit = deterministic_translated_control_mask(
            target, content, seed_key="overlapping-bboxes"
        )
        target_bbox = audit["target"]["bbox"]
        control_bbox = audit["control"]["bbox"]
        bbox_overlap = not (
            control_bbox[2] <= target_bbox[0]
            or control_bbox[0] >= target_bbox[2]
            or control_bbox[3] <= target_bbox[1]
            or control_bbox[1] >= target_bbox[3]
        )
        self.assertTrue(bbox_overlap)
        self.assertFalse(bool((control & target).any()))

    def test_coordinate_zone_connected_control_is_exact_and_deterministic(self) -> None:
        content = np.zeros((32, 40), dtype=bool)
        content[2:30, 3:37] = True
        target = np.zeros_like(content)
        target[20:25, 8:15] = True
        target_before = target.copy()
        content_before = content.copy()
        first, audit = deterministic_coordinate_zone_connected_control_mask(
            target,
            content,
            seed_key="connected-a",
        )
        second, second_audit = deterministic_coordinate_zone_connected_control_mask(
            target,
            content,
            seed_key="connected-a",
        )
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(audit, second_audit)
        self.assertTrue(np.array_equal(target, target_before))
        self.assertTrue(np.array_equal(content, content_before))
        self.assertFalse(bool((first & target).any()))
        self.assertTrue(bool((first & ~content).sum() == 0))
        target_geometry = mask_geometry(target, content)
        control_geometry = mask_geometry(first, content)
        self.assertEqual(target_geometry["area_pixels"], control_geometry["area_pixels"])
        self.assertEqual(control_geometry["component_count"], 1)
        self.assertEqual(
            coordinate_zone(target, content)["horizontal"],
            coordinate_zone(first, content)["horizontal"],
        )
        self.assertEqual(
            coordinate_zone(target, content)["vertical"],
            coordinate_zone(first, content)["vertical"],
        )
        self.assertEqual(audit["version"], COORDINATE_ZONE_CONTROL_VERSION)
        self.assertFalse(audit["true_anatomy_segmentation"])

    def test_coordinate_zone_control_does_not_claim_target_topology_matching(self) -> None:
        content = np.ones((36, 48), dtype=bool)
        target = np.zeros_like(content)
        target[23:26, 6:10] = True
        target[23:26, 16:20] = True
        control, audit = deterministic_coordinate_zone_connected_control_mask(
            target,
            content,
            seed_key="multi-component-target",
        )
        self.assertEqual(mask_geometry(target, content)["component_count"], 2)
        self.assertEqual(mask_geometry(control, content)["component_count"], 1)
        self.assertEqual(audit["connected_component_requirement"], 1)

    def test_coordinate_zone_control_fails_when_disjoint_area_is_too_small(self) -> None:
        content = np.ones((10, 10), dtype=bool)
        target = np.zeros_like(content)
        target[:6, :] = True
        with self.assertRaisesRegex(ValueError, "insufficient target-disjoint content"):
            deterministic_coordinate_zone_connected_control_mask(
                target,
                content,
                seed_key="insufficient-content",
            )

    def test_coordinate_zone_thresholds_are_frozen(self) -> None:
        content = np.ones((100, 100), dtype=bool)
        left_upper = np.zeros_like(content)
        left_upper[9:12, 9:12] = True
        central_middle = np.zeros_like(content)
        central_middle[49:52, 49:52] = True
        right_lower = np.zeros_like(content)
        right_lower[79:82, 79:82] = True
        self.assertEqual(
            (coordinate_zone(left_upper, content)["horizontal"], coordinate_zone(left_upper, content)["vertical"]),
            ("left", "upper"),
        )
        self.assertEqual(
            (
                coordinate_zone(central_middle, content)["horizontal"],
                coordinate_zone(central_middle, content)["vertical"],
            ),
            ("central", "middle"),
        )
        self.assertEqual(
            (coordinate_zone(right_lower, content)["horizontal"], coordinate_zone(right_lower, content)["vertical"]),
            ("right", "lower"),
        )


if __name__ == "__main__":
    unittest.main()
