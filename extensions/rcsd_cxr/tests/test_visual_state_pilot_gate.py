import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_visual_state_pilot.py"
)
SPEC = importlib.util.spec_from_file_location("pilot_gate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def summary(nll, f1, ece):
    validation = {
        "nll": nll,
        "macro_f1": f1,
        "ece": ece,
        "per_finding": {
            str(index): {"macro_f1": f1} for index in range(12)
        },
    }
    return {
        "best_validation": validation,
        "trainable_counts": {"all": 1},
        "hashes": {
            "manifest": "m",
            "backbone_weights": "b",
            "prototypes": "p",
        },
    }


class VisualStatePilotGateTests(unittest.TestCase):
    def test_gate_requires_all_metrics(self):
        result = MODULE.compare(
            summary(0.5, 0.70, 0.04), summary(0.48, 0.71, 0.045)
        )
        self.assertTrue(result["g3_pass"])

    def test_nll_failure_stops_scale(self):
        result = MODULE.compare(
            summary(0.5, 0.70, 0.04), summary(0.50, 0.71, 0.04)
        )
        self.assertFalse(result["g3_pass"])

    def test_cli_top_level_pass_matches_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spd = root / "spd.json"
            anchored = root / "anchored.json"
            output = root / "gate.json"
            spd.write_text(
                json.dumps(summary(0.5, 0.70, 0.04)), encoding="utf-8"
            )
            anchored.write_text(
                json.dumps(summary(0.50, 0.71, 0.04)), encoding="utf-8"
            )
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--spd",
                    str(spd),
                    "--field-anchor",
                    str(anchored),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(result["audit_completed"])
            self.assertFalse(result["pass"])


if __name__ == "__main__":
    unittest.main()
