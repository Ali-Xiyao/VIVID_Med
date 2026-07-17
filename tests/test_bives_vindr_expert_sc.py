from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_bives_vindr_expert_sc import prepare_intake


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class VinDrExpertScTests(unittest.TestCase):
    def _fixture(self, root: Path, *, omit_positive_box: bool = False) -> Path:
        (root / "annotations").mkdir(parents=True)
        (root / "test").mkdir()
        (root / "_extraction_complete.json").write_text("{}\n", encoding="utf-8")
        image_ids = ("image-a", "image-b")
        for image_id in image_ids:
            (root / "test" / f"{image_id}.dicom").write_bytes(image_id.encode("ascii"))
        fields = ["image_id", "Pleural effusion", "Consolidation", "Edema"]
        with (root / "annotations" / "image_labels_test.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(
                {"image_id": "image-a", "Pleural effusion": 1, "Consolidation": 0, "Edema": 0}
            )
            writer.writerow(
                {"image_id": "image-b", "Pleural effusion": 0, "Consolidation": 1, "Edema": 0}
            )
        with (root / "annotations" / "annotations_test.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            fields = ["image_id", "class_name", "x_min", "y_min", "x_max", "y_max"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            if not omit_positive_box:
                writer.writerow(
                    {
                        "image_id": "image-a",
                        "class_name": "Pleural effusion",
                        "x_min": 1,
                        "y_min": 2,
                        "x_max": 10,
                        "y_max": 20,
                    }
                )
            writer.writerow(
                {
                    "image_id": "image-b",
                    "class_name": "Consolidation",
                    "x_min": 3,
                    "y_min": 4,
                    "x_max": 11,
                    "y_max": 22,
                }
            )
        paths = [
            root / "annotations" / "image_labels_test.csv",
            root / "annotations" / "annotations_test.csv",
            *(root / "test").glob("*.dicom"),
        ]
        with (root / "SHA256SUMS.txt").open("w", encoding="utf-8", newline="\n") as handle:
            for path in paths:
                handle.write(f"{sha256(path)}  {path.relative_to(root).as_posix()}\n")
        return root

    def test_builds_consensus_sc_and_excludes_degenerate_finding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fixture(Path(tmp) / "vindr")
            output = Path(tmp) / "output"
            summary = prepare_intake(root, output, expected_test_count=2)
            self.assertEqual(summary["records"], 4)
            self.assertEqual(summary["eligible_findings"], ["pleural_effusion", "consolidation"])
            self.assertEqual(summary["ineligible_findings"], ["pulmonary_edema"])
            self.assertFalse(summary["patient_level_ci_ready"])
            rows = [
                json.loads(line)
                for line in (output / "vindr_test_expert_sc.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual({row["state"] for row in rows}, {"support", "contradict"})
            self.assertTrue(all(row["patient_id"] is None for row in rows))
            support = [row for row in rows if row["state"] == "support"]
            self.assertTrue(all(row["bounding_boxes"] for row in support))

    def test_fails_when_positive_label_has_no_box(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fixture(Path(tmp) / "vindr", omit_positive_box=True)
            with self.assertRaisesRegex(ValueError, "positive_without_box=1"):
                prepare_intake(root, Path(tmp) / "output", expected_test_count=2)


if __name__ == "__main__":
    unittest.main()
