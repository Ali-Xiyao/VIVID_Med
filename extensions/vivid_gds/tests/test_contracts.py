import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vivid_gds.contracts import FINDINGS, parse_ums_target, render_free_text


class ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = json.dumps(
            {
                "modality": "CXR",
                "findings": {
                    "Cardiomegaly": {"state": "present", "score": None},
                    "Edema": {"state": "uncertain", "score": None},
                },
                "study_view": None,
            }
        )

    def test_missing_is_mask_not_class(self) -> None:
        states, mask = parse_ums_target(self.target)
        self.assertEqual(states[FINDINGS.index("Cardiomegaly")], 0)
        self.assertEqual(states[FINDINGS.index("Edema")], 2)
        self.assertEqual(states[FINDINGS.index("Atelectasis")], -100)
        self.assertEqual(sum(mask), 2)

    def test_free_text_is_deterministic_and_field_preserving(self) -> None:
        left = render_free_text(self.target, "row-1")
        right = render_free_text(self.target, "row-1")
        self.assertEqual(left, right)
        self.assertIn("Cardiomegaly", left)
        self.assertIn("Edema", left)
        self.assertNotIn("Atelectasis", left)

    def test_unknown_finding_fails_closed(self) -> None:
        payload = json.loads(self.target)
        payload["findings"]["Unknown"] = {"state": "present"}
        with self.assertRaises(ValueError):
            parse_ums_target(json.dumps(payload))
