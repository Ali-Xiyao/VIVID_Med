from pathlib import Path
import tempfile
import unittest

from scripts.run_gate_queue import load_queue


class GateQueueTest(unittest.TestCase):
    def test_duplicate_task_ids_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queue.yaml"
            path.write_text(
                "schema_version: 1\ntasks:\n"
                "  - {id: same}\n"
                "  - {id: same}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate task id"):
                load_queue(path)


if __name__ == "__main__":
    unittest.main()

