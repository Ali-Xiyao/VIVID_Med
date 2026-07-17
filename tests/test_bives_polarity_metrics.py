from __future__ import annotations

import unittest

from bives_cxr.polarity_metrics import polarity_metrics


class PolarityMetricTests(unittest.TestCase):
    def test_metrics_lock_threshold_per_finding(self) -> None:
        rows = []
        for finding in ("effusion", "consolidation"):
            for index, (label, score) in enumerate(((0, 0.1), (0, 0.2), (1, 0.8), (1, 0.9))):
                rows.append(
                    {
                        "sample_id": f"{finding}-{index}",
                        "canonical_statement_id": finding,
                        "binary_label": label,
                        "support_probability": score,
                    }
                )
        metrics, thresholds = polarity_metrics(rows, target_specificity=0.9)
        self.assertEqual(metrics["macro"]["auroc"], 1.0)
        self.assertEqual(set(thresholds), {"effusion", "consolidation"})
        self.assertTrue(all(value > 0.2 for value in thresholds.values()))
        self.assertTrue(
            all(
                row["specificity_at_locked_threshold"] == 1.0
                for row in metrics["per_finding"].values()
            )
        )


if __name__ == "__main__":
    unittest.main()
