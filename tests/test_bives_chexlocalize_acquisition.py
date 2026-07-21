import unittest

from bives_cxr.chexlocalize_acquisition import (
    ALLOWED_EXACT_PATHS,
    is_allowed_validation_path,
    select_validation_files,
)


class CheXlocalizeAcquisitionTests(unittest.TestCase):
    def test_test_paths_are_never_allowed(self) -> None:
        self.assertFalse(is_allowed_validation_path("gt_annotations_test.json"))
        self.assertFalse(is_allowed_validation_path("test/image.png"))
        self.assertTrue(is_allowed_validation_path("gt_annotations_val.json"))
        self.assertTrue(is_allowed_validation_path("gradcam_maps_val/a/b.npy"))

    def test_selection_is_fail_closed(self) -> None:
        rows = [
            {"path": path, "size": 1, "md5": "0" * 32, "uri": f"/{path}"}
            for path in sorted(ALLOWED_EXACT_PATHS)
        ]
        rows.append(
            {
                "path": "gt_annotations_test.json",
                "size": 100,
                "md5": "1" * 32,
                "uri": "/test",
            }
        )
        selected = select_validation_files(
            rows, expected_count=3, expected_total_bytes=3
        )
        self.assertEqual(len(selected), 3)
        self.assertFalse(any("test" in row["path"] for row in selected))
        with self.assertRaisesRegex(ValueError, "size drift"):
            select_validation_files(
                rows, expected_count=3, expected_total_bytes=4
            )


if __name__ == "__main__":
    unittest.main()
