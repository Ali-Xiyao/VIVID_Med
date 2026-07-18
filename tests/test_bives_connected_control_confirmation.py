import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_bives_connected_control_confirmation.py"
)
SPEC = importlib.util.spec_from_file_location("bives_c5_confirmation", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ConnectedControlConfirmationTests(unittest.TestCase):
    def test_geometry_gate_preserves_all_denominators(self):
        rows = []
        counts = {"consolidation": 59, "pleural_effusion": 319}
        for finding, count in counts.items():
            for index in range(count):
                rows.append(
                    {
                        "canonical_statement_id": finding,
                        "box_area_quartile": index % 4 + 1,
                        "feasible": True,
                        "contracts_pass": True,
                    }
                )
        result = MODULE.summarize_geometry_gate(rows)
        self.assertTrue(result["pass"])
        self.assertEqual(result["total"], 378)
        rows[0]["feasible"] = False
        self.assertEqual(MODULE.summarize_geometry_gate(rows)["total"], 378)

    def test_polarity_requires_each_b2_metric_not_below_b0(self):
        rows = []
        for finding in MODULE.FINDINGS:
            for label, b0, b2 in (
                (0, 0.1, 0.05),
                (0, 0.2, 0.15),
                (1, 0.8, 0.85),
                (1, 0.9, 0.95),
            ):
                rows.append(
                    {
                        "canonical_statement_id": finding,
                        "binary_label": label,
                        "b0_support_probability": b0,
                        "b2_support_probability": b2,
                    }
                )
        result = MODULE.summarize_polarity(rows)
        self.assertTrue(result["pass"])
        rows[-1]["b2_support_probability"] = 0.0
        self.assertFalse(MODULE.summarize_polarity(rows)["pass"])

    def test_confirmation_marker_is_single_identity_and_final_is_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "c5"
            identity = {"script": "abc", "manifest": "def"}
            marker, opened = MODULE.prepare_confirmation_opening(output, identity)
            self.assertTrue(opened)
            self.assertTrue(marker.is_file())
            _, opened_again = MODULE.prepare_confirmation_opening(output, identity)
            self.assertFalse(opened_again)
            with self.assertRaises(ValueError):
                MODULE.prepare_confirmation_opening(output, {"script": "changed"})
            (output / "metrics_final.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                MODULE.prepare_confirmation_opening(output, identity)


if __name__ == "__main__":
    unittest.main()
