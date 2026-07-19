import unittest

import numpy as np

from bives_cxr.qwen35_localization_audit import deterministic_top_cell_mask


class Qwen35LocalizationAuditTests(unittest.TestCase):
    def test_top_cell_is_deterministic_under_ties(self) -> None:
        sensitivity = np.zeros((2, 2), dtype=np.float64)
        mask = deterministic_top_cell_mask(
            sensitivity,
            image_height=8,
            image_width=8,
        )
        self.assertEqual(int(mask.sum()), 16)
        self.assertTrue(mask[:4, :4].all())
        self.assertFalse(mask[4:, :].any())

    def test_top_cell_rejects_nonfinite_map(self) -> None:
        with self.assertRaises(ValueError):
            deterministic_top_cell_mask(
                np.asarray([[np.inf]]),
                image_height=8,
                image_width=8,
            )


if __name__ == "__main__":
    unittest.main()
