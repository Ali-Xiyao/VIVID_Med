from __future__ import annotations

import unittest

import numpy as np

from bives_cxr.rescue_protocol import (
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


if __name__ == "__main__":
    unittest.main()
