from __future__ import annotations

import unittest

import torch
from PIL import Image

from bives_cxr.qwen35_preprocessing import content_mask_for_grid, letterbox_image


class Qwen35PreprocessingTests(unittest.TestCase):
    def test_letterbox_and_content_mask_are_deterministic(self) -> None:
        source = Image.new("RGB", (800, 400), (10, 20, 30))
        first, first_box = letterbox_image(source, 448)
        second, second_box = letterbox_image(source, 448)
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertEqual(first_box, (0, 112, 448, 336))
        self.assertEqual(first_box, second_box)
        grid = torch.tensor([1, 28, 28])
        mask = content_mask_for_grid(grid, first_box, 448)
        self.assertEqual(tuple(mask.shape), (784,))
        self.assertEqual(int(mask.sum()), 392)


if __name__ == "__main__":
    unittest.main()
