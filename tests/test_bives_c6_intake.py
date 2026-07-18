from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from bives_cxr.c6_intake import (
    audit_chexlocalize_test_release,
    build_chexpert_prior_access_registry,
    parse_chexlocalize_key,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class C6IntakeTests(unittest.TestCase):
    def _fixture(self, root: Path, *, overlap: bool = False, include_effusion: bool = True) -> tuple[Path, Path, Path]:
        dataset = root / "CheXlocalize"
        rows = [
            {"Path": "CheXpert/test/patient1/study1/view1_frontal.jpg"},
            {"Path": "CheXpert/test/patient2/study1/view1_frontal.jpg"},
        ]
        _write_csv(dataset / "CheXpert" / "test_labels.csv", rows)
        for index in (1, 2):
            image = dataset / "CheXpert" / "test" / f"patient{index}" / "study1" / "view1_frontal.jpg"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(f"image-{index}".encode("ascii"))
        annotations = {
            "patient1_study1_view1_frontal": {
                "img_size": [100, 120],
                "Consolidation": [[[10, 10], [20, 10], [20, 20]]],
            },
            "patient2_study1_view1_frontal": {
                "img_size": [100, 120],
                **(
                    {"Pleural Effusion": [[[30, 30], [40, 30], [40, 40]]]}
                    if include_effusion
                    else {}
                ),
            },
        }
        annotations_root = dataset / "CheXlocalize"
        annotations_root.mkdir(parents=True, exist_ok=True)
        (annotations_root / "gt_annotations_test.json").write_text(
            json.dumps(annotations), encoding="utf-8"
        )
        (annotations_root / "gt_segmentations_test.json").write_text("{}", encoding="utf-8")
        license_record = root / "license.json"
        license_record.write_text(
            json.dumps(
                {
                    "dataset_name": "CheXlocalize",
                    "release_version": "fixture",
                    "source_url": "https://aimi.stanford.edu/datasets/chexlocalize",
                    "terms_url": "https://example.test/terms",
                    "retrieved_at": "2026-07-18",
                    "package_sha256": "a" * 64,
                    "terms_accepted_by_user": True,
                    "access_secret_not_persisted": True,
                }
            ),
            encoding="utf-8",
        )
        prior_csv = root / "prior_valid.csv"
        prior_patient = 1 if overlap else 999
        _write_csv(
            prior_csv,
            [{"Path": f"CheXpert/valid/patient{prior_patient}/study1/view1_frontal.jpg"}],
        )
        registry = build_chexpert_prior_access_registry(prior_csv)
        prior_registry = root / "prior_registry.json"
        prior_registry.write_text(json.dumps(registry), encoding="utf-8")
        return dataset, license_record, prior_registry

    def test_parses_official_annotation_key(self) -> None:
        identity = parse_chexlocalize_key("patient64622_study1_view1_frontal")
        self.assertEqual(identity["patient"], "patient64622")
        self.assertEqual(identity["study"], "patient64622_study1")

    def test_builds_hashed_prior_registry_without_raw_identifiers(self) -> None:
        with TemporaryDirectory() as directory:
            csv_path = Path(directory) / "valid.csv"
            _write_csv(
                csv_path,
                [{"Path": "CheXpert/valid/patient123/study4/view1_frontal.jpg"}],
            )
            payload = build_chexpert_prior_access_registry(csv_path)
            rendered = json.dumps(payload, sort_keys=True)
            self.assertNotIn("patient123", rendered)
            self.assertEqual(payload["counts"], {"patient": 1, "study": 1, "image": 1})

    def test_audits_test_only_release_without_emitting_raw_ids(self) -> None:
        with TemporaryDirectory() as directory:
            dataset, license_record, registry = self._fixture(Path(directory))
            payload = audit_chexlocalize_test_release(
                dataset,
                license_record=license_record,
                prior_registry=registry,
                expected_test_images=2,
                expected_test_patients=2,
            )
            self.assertEqual(payload["status"], "metadata_intake_ready_no_model_authority")
            self.assertEqual(payload["targets"]["Consolidation"]["patients"], 1)
            self.assertEqual(payload["targets"]["Pleural Effusion"]["patients"], 1)
            rendered = json.dumps(payload, sort_keys=True)
            self.assertNotIn("patient1", rendered)
            self.assertFalse(payload["model_evaluation_authorized"])
            self.assertEqual(len(payload["canonical_artifact_sha256"]), 64)

    def test_rejects_validation_path(self) -> None:
        with TemporaryDirectory() as directory:
            dataset, license_record, registry = self._fixture(Path(directory))
            labels = dataset / "CheXpert" / "test_labels.csv"
            _write_csv(
                labels,
                [{"Path": "CheXpert/valid/patient1/study1/view1_frontal.jpg"}],
            )
            with self.assertRaisesRegex(ValueError, "non-test"):
                audit_chexlocalize_test_release(
                    dataset,
                    license_record=license_record,
                    prior_registry=registry,
                    expected_test_images=1,
                    expected_test_patients=1,
                )

    def test_rejects_image_path_escape(self) -> None:
        with TemporaryDirectory() as directory:
            dataset, license_record, registry = self._fixture(Path(directory))
            labels = dataset / "CheXpert" / "test_labels.csv"
            _write_csv(
                labels,
                [{"Path": "CheXpert/test/../patient1/study1/view1_frontal.jpg"}],
            )
            with self.assertRaisesRegex(ValueError, "escapes test root"):
                audit_chexlocalize_test_release(
                    dataset,
                    license_record=license_record,
                    prior_registry=registry,
                    expected_test_images=1,
                    expected_test_patients=1,
                )

    def test_rejects_prior_patient_overlap(self) -> None:
        with TemporaryDirectory() as directory:
            dataset, license_record, registry = self._fixture(Path(directory), overlap=True)
            with self.assertRaisesRegex(ValueError, "overlaps prior access"):
                audit_chexlocalize_test_release(
                    dataset,
                    license_record=license_record,
                    prior_registry=registry,
                    expected_test_images=2,
                    expected_test_patients=2,
                )

    def test_rejects_missing_target_regions(self) -> None:
        with TemporaryDirectory() as directory:
            dataset, license_record, registry = self._fixture(
                Path(directory), include_effusion=False
            )
            with self.assertRaisesRegex(ValueError, "missing target regions"):
                audit_chexlocalize_test_release(
                    dataset,
                    license_record=license_record,
                    prior_registry=registry,
                    expected_test_images=2,
                    expected_test_patients=2,
                )


if __name__ == "__main__":
    unittest.main()
