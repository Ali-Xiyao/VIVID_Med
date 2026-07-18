from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from bives_cxr.c6_ms_cxr import (
    audit_ms_cxr_test_release,
    build_mimic_prior_access_registry,
)


DICOM_ONE = "11111111-11111111-11111111-11111111-11111111"
DICOM_TWO = "22222222-22222222-22222222-22222222-22222222"


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _write_metadata(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["dicom_id", "subject_id", "study_id"]
        )
        writer.writeheader()
        writer.writerows(rows)


class C6MsCxrTests(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
        *,
        overlap: bool = False,
        second_split: str = "test",
        invalid_box: bool = False,
        include_second_metadata: bool = True,
        include_second_image: bool = True,
        license_ok: bool = True,
    ) -> tuple[Path, Path, Path, Path, Path]:
        dataset = root / "MS-CXR"
        dataset.mkdir(parents=True)
        release = {
            "info": {"version": "1.1.0"},
            "categories": [
                {"id": 1, "name": "Consolidation"},
                {"id": 2, "name": "Pleural Effusion"},
            ],
            "images": [
                {"id": 11, "file_name": f"{DICOM_ONE}.jpg", "width": 100, "height": 80},
                {"id": 12, "file_name": f"{DICOM_TWO}.jpg", "width": 120, "height": 90},
            ],
            "annotations": [
                {
                    "id": 21,
                    "image_id": 11,
                    "category_id": 1,
                    "bbox": [10, 10, 20, 20],
                    "sentence": "Consolidation.",
                    "split": "test",
                },
                {
                    "id": 22,
                    "image_id": 12,
                    "category_id": 2,
                    "bbox": [110, 10, 20, 20] if invalid_box else [30, 20, 25, 25],
                    "sentence": "Pleural effusion.",
                    "split": second_split,
                },
            ],
        }
        annotations = dataset / "MS_CXR_Local_Alignment_v1.1.0.json"
        annotations.write_text(json.dumps(release), encoding="utf-8")

        metadata = root / "mimic-cxr-2.0.0-metadata.csv.gz"
        metadata_rows: list[dict[str, object]] = [
            {"dicom_id": DICOM_ONE, "subject_id": 10000001, "study_id": 50000001}
        ]
        if include_second_metadata:
            metadata_rows.append(
                {"dicom_id": DICOM_TWO, "subject_id": 10000002, "study_id": 50000002}
            )
        _write_metadata(metadata, metadata_rows)

        images = root / "mimic-images"
        first_image = images / "files" / "p10" / "p10000001" / "s50000001" / f"{DICOM_ONE}.jpg"
        first_image.parent.mkdir(parents=True)
        first_image.write_bytes(b"first image")
        if include_second_image:
            second_image = images / "files" / "p10" / "p10000002" / "s50000002" / f"{DICOM_TWO}.jpg"
            second_image.parent.mkdir(parents=True)
            second_image.write_bytes(b"second image")

        prior_manifest = root / "prior.jsonl"
        prior_patient = "p10000001" if overlap else "p19999999"
        _write_jsonl(
            prior_manifest,
            [{"patient_id": prior_patient, "study_id": "s59999999"}],
        )
        registry_payload = build_mimic_prior_access_registry(
            [prior_manifest], enforce_frozen_identity=False
        )
        registry = root / "prior_registry.json"
        registry.write_text(json.dumps(registry_payload), encoding="utf-8")

        license_record = root / "license.json"
        license_record.write_text(
            json.dumps(
                {
                    "dataset_name": "MS-CXR",
                    "release_version": "1.1.0",
                    "source_url": "https://physionet.org/content/ms-cxr/1.1.0/",
                    "terms_url": "https://physionet.org/about/licenses/physionet-credentialed-health-data-license-150/",
                    "retrieved_at": "2026-07-18",
                    "package_sha256": "a" * 64,
                    "credentialed_access_confirmed": license_ok,
                    "citi_training_confirmed": True,
                    "dua_signed_by_user": True,
                    "access_secret_not_persisted": True,
                }
            ),
            encoding="utf-8",
        )
        return dataset, metadata, images, license_record, registry

    def _audit(self, fixture: tuple[Path, Path, Path, Path, Path]) -> dict[str, object]:
        dataset, metadata, images, license_record, registry = fixture
        return audit_ms_cxr_test_release(
            dataset,
            mimic_metadata=metadata,
            mimic_images_root=images,
            license_record=license_record,
            prior_registry=registry,
            expected_test_pairs={"Consolidation": 1, "Pleural Effusion": 1},
            expected_test_subjects={"Consolidation": 1, "Pleural Effusion": 1},
            enforce_frozen_prior_identity=False,
        )

    def test_builds_frozen_namespace_registry_without_raw_ids(self) -> None:
        with TemporaryDirectory() as directory:
            manifest = Path(directory) / "prior.jsonl"
            _write_jsonl(
                manifest,
                [{"patient_id": "p10000001", "study_id": "s50000001-proxy-i"}],
            )
            payload = build_mimic_prior_access_registry(
                [manifest], enforce_frozen_identity=False
            )
            rendered = json.dumps(payload, sort_keys=True)
            self.assertNotIn("p10000001", rendered)
            self.assertNotIn("s50000001", rendered)
            self.assertEqual(payload["counts"], {"patient": 1, "study": 1})

    def test_audits_official_test_without_emitting_raw_ids(self) -> None:
        with TemporaryDirectory() as directory:
            payload = self._audit(self._fixture(Path(directory)))
            self.assertEqual(payload["status"], "metadata_intake_ready_no_model_authority")
            self.assertEqual(payload["counts"]["target_test_pairs"], 2)
            self.assertEqual(payload["targets"]["Consolidation"]["patients"], 1)
            self.assertEqual(payload["targets"]["Pleural Effusion"]["patients"], 1)
            self.assertFalse(payload["model_evaluation_authorized"])
            rendered = json.dumps(payload, sort_keys=True)
            self.assertNotIn("p10000001", rendered)
            self.assertNotIn(DICOM_ONE, rendered)

    def test_rejects_prior_patient_overlap(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "overlaps prior MIMIC"):
                self._audit(self._fixture(Path(directory), overlap=True))

    def test_rejects_non_test_target_pair_by_official_count(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Pleural Effusion has 0 pairs"):
                self._audit(self._fixture(Path(directory), second_split="val"))

    def test_rejects_out_of_bounds_box(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "out of image bounds"):
                self._audit(self._fixture(Path(directory), invalid_box=True))

    def test_rejects_missing_mimic_metadata_binding(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "absent from MIMIC metadata"):
                self._audit(
                    self._fixture(Path(directory), include_second_metadata=False)
                )

    def test_rejects_missing_mimic_image(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                self._audit(self._fixture(Path(directory), include_second_image=False))

    def test_rejects_unconfirmed_user_access_record(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "credentialed_access_confirmed"):
                self._audit(self._fixture(Path(directory), license_ok=False))

    def test_rejects_unknown_category_reference(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            annotations = fixture[0] / "MS_CXR_Local_Alignment_v1.1.0.json"
            payload = json.loads(annotations.read_text(encoding="utf-8"))
            payload["annotations"][0]["category_id"] = 999
            annotations.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown category"):
                self._audit(fixture)

    def test_default_audit_rejects_nonfrozen_registry(self) -> None:
        with TemporaryDirectory() as directory:
            dataset, metadata, images, license_record, registry = self._fixture(
                Path(directory)
            )
            with self.assertRaisesRegex(ValueError, "frozen MIMIC prior-access"):
                audit_ms_cxr_test_release(
                    dataset,
                    mimic_metadata=metadata,
                    mimic_images_root=images,
                    license_record=license_record,
                    prior_registry=registry,
                    expected_test_pairs={"Consolidation": 1, "Pleural Effusion": 1},
                    expected_test_subjects={"Consolidation": 1, "Pleural Effusion": 1},
                )


if __name__ == "__main__":
    unittest.main()
