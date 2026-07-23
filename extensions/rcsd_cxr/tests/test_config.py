from pathlib import Path
import tempfile
import unittest

from rcsd_cxr.config import DatasetRegistry


class DatasetRegistryTest(unittest.TestCase):
    def test_missing_parent_fails(self) -> None:
        text = """schema_version: 1
datasets:
  child:
    path: null
    kind: add_on
    parent: missing
    roles: [secondary_only]
    status: absent
    test_exposure: none
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.yaml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown parents"):
                DatasetRegistry.load(path)

    def test_forbidden_paper_one_role_fails_closed(self) -> None:
        text = """schema_version: 1
datasets:
  sealed:
    path: null
    kind: add_on
    parent: null
    roles: [paper2_reserved_test]
    status: absent
    test_exposure: sealed
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.yaml"
            path.write_text(text, encoding="utf-8")
            registry = DatasetRegistry.load(path)
            errors = registry.validate_paths(["paper2_reserved_test"], paper1=True)
        self.assertTrue(any("forbidden" in error for error in errors))
