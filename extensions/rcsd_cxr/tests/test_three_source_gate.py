import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "three_source_gate", SCRIPTS / "run_lunguage_three_source_gate.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def metric(f1, nll, ece, reliability=0.8, gap=0.2):
    return {
        "macro_f1": f1,
        "nll": nll,
        "ece": ece,
        "reliability_correctness_auroc": reliability,
        "top_bottom_confidence_quartile_accuracy_gap": gap,
        "per_finding": {str(index): {} for index in range(11)},
        "predicted_state_counts": {0: 1, 1: 1, 2: 1},
    }


class ThreeSourceGateTests(unittest.TestCase):
    def test_all_thresholds_required(self):
        metrics = {
            "chexpert": metric(0.70, 0.50, 0.05),
            "negbio": metric(0.72, 0.48, 0.04),
            "chexbert": metric(0.71, 0.49, 0.045),
            "fusion": metric(0.74, 0.45, 0.035),
        }
        result = MODULE.evaluate_gate(metrics)
        self.assertTrue(result["g2_pass"])
        self.assertEqual(result["reference_sources"]["macro_f1"], "negbio")

    def test_f1_only_improvement_does_not_pass(self):
        metrics = {
            "chexpert": metric(0.70, 0.50, 0.05),
            "negbio": metric(0.72, 0.48, 0.04),
            "chexbert": metric(0.71, 0.49, 0.045),
            "fusion": metric(0.74, 0.49, 0.035),
        }
        result = MODULE.evaluate_gate(metrics)
        self.assertFalse(result["g2_pass"])
        self.assertFalse(result["checks"]["nll"])
        self.assertIn("drop posterior", result["failure_action"])


if __name__ == "__main__":
    unittest.main()
