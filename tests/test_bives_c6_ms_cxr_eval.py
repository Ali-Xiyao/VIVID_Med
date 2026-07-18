from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from bives_cxr.c6_ms_cxr_eval import (
    EXPECTED_BOXES,
    EXPECTED_ROWS,
    build_ms_cxr_manifest,
    evaluate_survival_gate,
    summarize_operator,
    validate_ms_cxr_manifest,
)


class MsCxrEvaluationContracts(unittest.TestCase):
    def _release(self, root: Path) -> tuple[Path, Path, Path]:
        image_root = root / "images"
        annotations = []
        images = []
        metadata_rows = []
        annotation_id = 1
        image_id = 1
        specifications = (("Consolidation", 4, 15, 10), ("Pleural Effusion", 8, 14, 6))
        for finding, category_id, count, double_box_rows in specifications:
            for index in range(count):
                patient = f"p{10000000 + image_id}"
                study = f"s{20000000 + image_id}"
                dicom = f"00000000-00000000-00000000-00000000-{image_id:08x}"
                relative = Path("files", patient[:3], patient, study, f"{dicom}.jpg")
                image_path = image_root / relative
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(f"image-{image_id}".encode("ascii"))
                images.append(
                    {
                        "id": image_id,
                        "file_name": f"{dicom}.jpg",
                        "width": 100,
                        "height": 100,
                        "path": relative.as_posix(),
                    }
                )
                metadata_rows.append(
                    {"dicom_id": dicom, "subject_id": patient[1:], "study_id": study[1:]}
                )
                box_count = 2 if index < double_box_rows else 1
                for box_index in range(box_count):
                    annotations.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": [10 + box_index * 15, 10, 10, 10],
                            "label_text": f"publisher sentence {finding} {index}",
                            "split": "test",
                        }
                    )
                    annotation_id += 1
                image_id += 1
        release = {
            "info": {"version": "1.1.0"},
            "categories": [
                {"id": 4, "name": "Consolidation"},
                {"id": 8, "name": "Pleural Effusion"},
            ],
            "images": images,
            "annotations": annotations,
        }
        annotations_path = root / "release.json"
        annotations_path.write_text(json.dumps(release), encoding="utf-8")
        metadata_path = root / "metadata.csv"
        with metadata_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("dicom_id", "subject_id", "study_id"))
            writer.writeheader()
            writer.writerows(metadata_rows)
        return annotations_path, metadata_path, image_root

    def test_manifest_groups_boxes_and_uses_canonical_statements(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            annotations, metadata, image_root = self._release(Path(directory))
            rows = build_ms_cxr_manifest(
                annotations_path=annotations,
                mimic_metadata_path=metadata,
                mimic_images_root=image_root,
            )
            summary = validate_ms_cxr_manifest(rows)
            self.assertEqual(summary["per_finding"], EXPECTED_ROWS)
            self.assertEqual(summary["per_finding_boxes"], EXPECTED_BOXES)
            self.assertEqual(summary["patients"], 29)
            self.assertNotIn("publisher sentence", " ".join(row["statement_text"] for row in rows))
            self.assertTrue(all(len(row["patient_sha256"]) == 64 for row in rows))

    def test_manifest_rejects_changed_image_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            annotations, metadata, image_root = self._release(Path(directory))
            rows = build_ms_cxr_manifest(
                annotations_path=annotations,
                mimic_metadata_path=metadata,
                mimic_images_root=image_root,
            )
            Path(rows[0]["image_path"]).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "image hash changed"):
                validate_ms_cxr_manifest(rows)

    def _result_rows(self, value: float) -> list[dict]:
        rows = []
        for finding, count in EXPECTED_ROWS.items():
            for index in range(count):
                rows.append(
                    {
                        "sample_id": f"{finding}-{index}",
                        "unit_id": f"patient-{finding}-{index}",
                        "canonical_statement_id": finding,
                        "box_area_quartile": index % 4 + 1,
                        "topk_localization_gain": 0.1,
                        "local_mean": {"tcig": value},
                        "masked_gaussian_blur": {"tcig": value},
                    }
                )
        return rows

    def test_frozen_survival_gate_passes_positive_patient_effects(self) -> None:
        rows = self._result_rows(0.2)
        results = {
            operator: summarize_operator(
                rows, operator, bootstrap_replicates=100, bootstrap_seed=17
            )
            for operator in ("local_mean", "masked_gaussian_blur")
        }
        gate = evaluate_survival_gate(results)
        self.assertTrue(gate["pass"])
        self.assertEqual(gate["status"], "pass")

    def test_frozen_survival_gate_stops_negative_effects(self) -> None:
        rows = self._result_rows(-0.2)
        results = {
            operator: summarize_operator(
                rows, operator, bootstrap_replicates=100, bootstrap_seed=17
            )
            for operator in ("local_mean", "masked_gaussian_blur")
        }
        gate = evaluate_survival_gate(results)
        self.assertFalse(gate["pass"])
        self.assertEqual(gate["status"], "fail_final_stop")


if __name__ == "__main__":
    unittest.main()
