import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_qwen35_field_prototypes.py"
)
SPEC = importlib.util.spec_from_file_location("field_prototypes", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class FieldPrototypeTests(unittest.TestCase):
    def test_templates_cover_fixed_fields_and_states(self):
        values = MODULE.templates()
        texts, spans = MODULE.flatten_templates(values)
        self.assertEqual(set(values), set(MODULE.FIELD_NAMES))
        self.assertEqual(len(texts), 96)
        self.assertEqual(spans["observation"], (0, 12))
        self.assertEqual(spans["assertion"], (12, 48))
        self.assertIn("pleural space", values["anatomy"][9])


if __name__ == "__main__":
    unittest.main()
