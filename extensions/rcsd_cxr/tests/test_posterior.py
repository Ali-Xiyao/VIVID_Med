import unittest

import numpy as np

from rcsd_cxr.posterior import fuse_log_opinion_pool


class PosteriorTest(unittest.TestCase):
    def test_all_missing_stays_missing(self) -> None:
        self.assertIsNone(fuse_log_opinion_pool([None, None], [0.5, 0.5]))

    def test_agreement_is_high_reliability(self) -> None:
        result = fuse_log_opinion_pool(
            [[0.98, 0.01, 0.01], [0.95, 0.03, 0.02]],
            [0.5, 0.5],
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertGreater(float(result.probabilities[0]), 0.94)
        self.assertGreater(result.reliability, 0.7)
        self.assertAlmostEqual(float(result.probabilities.sum()), 1.0, places=6)

    def test_weight_changes_posterior(self) -> None:
        sources = [[0.98, 0.01, 0.01], [0.01, 0.98, 0.01]]
        left = fuse_log_opinion_pool(sources, [0.9, 0.1])
        right = fuse_log_opinion_pool(sources, [0.1, 0.9])
        assert left is not None and right is not None
        self.assertGreater(left.probabilities[0], left.probabilities[1])
        self.assertGreater(right.probabilities[1], right.probabilities[0])

    def test_invalid_probability_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fuse_log_opinion_pool([[np.nan, 0.5, 0.5]], [1.0])
