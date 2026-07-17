from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from bives_cxr.expert_sc import (
    ExpertSCDataset,
    evaluate_expert_sc_predictions,
    read_expert_sc_manifest,
)


class ExpertSCTests(unittest.TestCase):
    def _manifest(self, root: Path) -> Path:
        rows = []
        for unit_index in range(6):
            image = root / f"image-{unit_index}.png"
            Image.fromarray(np.full((8, 8), unit_index * 10, dtype=np.uint8)).save(image)
            for finding in ("effusion", "consolidation"):
                label = unit_index % 2
                rows.append(
                    {
                        "sample_id": f"{unit_index}::{finding}",
                        "unit_id": f"unit-{unit_index}",
                        "image_path": str(image),
                        "canonical_statement_id": finding,
                        "statement_text": f"{finding} is present.",
                        "binary_label": label,
                        "state": "support" if label else "contradict",
                        "bounding_boxes": [],
                    }
                )
        path = root / "manifest.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        return path

    def test_dataset_has_no_quartet_or_patient_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._manifest(Path(tmp))
            dataset = ExpertSCDataset(path)
            item = dataset[0]
            self.assertEqual(len(dataset), 12)
            self.assertNotIn("patient_id", item)
            self.assertNotIn("group_id", item)
            self.assertNotIn("state_index", item)
            self.assertEqual(item["image"].mode, "RGB")

    def test_evaluator_uses_image_cluster_and_locked_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._manifest(Path(tmp))
            manifest = read_expert_sc_manifest(path)
            predictions = [
                {
                    "sample_id": row["sample_id"],
                    "support_probability": 0.9 if row["binary_label"] else 0.1,
                }
                for row in manifest
            ]
            thresholds = {
                finding: {
                    "support_probability_threshold": 0.5,
                    "target_specificity": 0.9,
                }
                for finding in ("effusion", "consolidation")
            }
            result = evaluate_expert_sc_predictions(
                manifest,
                predictions,
                thresholds,
                bootstrap_replicates=50,
                bootstrap_seed=7,
            )
            self.assertEqual(result["units"], 6)
            self.assertEqual(
                result["confidence_interval_unit"],
                "image_level_cluster_by_unit_id",
            )
            self.assertFalse(result["patient_level_confidence_interval"])
            self.assertEqual(result["macro"]["auroc"], 1.0)

    def test_evaluator_rejects_missing_locked_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = read_expert_sc_manifest(self._manifest(Path(tmp)))
            predictions = [
                {"sample_id": row["sample_id"], "support_probability": 0.5}
                for row in manifest
            ]
            with self.assertRaisesRegex(ValueError, "thresholds are missing"):
                evaluate_expert_sc_predictions(
                    manifest,
                    predictions,
                    {},
                    bootstrap_replicates=0,
                )


if __name__ == "__main__":
    unittest.main()
