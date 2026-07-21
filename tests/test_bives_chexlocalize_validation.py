import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from bives_cxr.provenance import canonical_json_sha256, file_sha256
from bives_cxr.chexlocalize_validation import (
    bind_annotation_identities,
    patient_disjoint_shard,
    prepare_letterboxed_masks,
    rasterize_contours,
)
from scripts.freeze_chexlocalize_qwen35_development import snapshot_model_files


class CheXlocalizeValidationTests(unittest.TestCase):
    def test_annotation_binding_is_subset_based_and_fail_closed(self) -> None:
        validation = {
            "patient1_study2_view1_frontal": {"identity": {}, "image_path": Path("x")}
        }
        annotations = {
            "patient1_study2_view1_frontal": {"img_size": [10, 12]}
        }
        self.assertEqual(len(bind_annotation_identities(annotations, validation)), 1)
        with self.assertRaisesRegex(ValueError, "absent from valid.csv"):
            bind_annotation_identities(
                {"patient9_study2_view1_frontal": {"img_size": [10, 12]}},
                validation,
            )

    def test_contours_and_letterbox_geometry(self) -> None:
        contours = [[[2, 1], [8, 1], [8, 6], [2, 6]]]
        native = rasterize_contours(contours, [8, 10])
        self.assertEqual(native.shape, (8, 10))
        self.assertTrue(native.any())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.jpg"
            Image.new("RGB", (10, 8), (32, 32, 32)).save(path)
            expert, content, audit = prepare_letterboxed_masks(
                path, contours, [8, 10]
            )
        self.assertEqual(expert.shape, (448, 448))
        self.assertEqual(content.shape, (448, 448))
        self.assertTrue(np.all(~expert | content))
        self.assertGreater(audit["expert_area_pixels"], 0)

    def test_full_resolution_contours_scale_to_small_release(self) -> None:
        contours = [[[280, 230], [1400, 230], [1400, 1160], [280, 1160]]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.jpg"
            Image.new("RGB", (390, 320), (32, 32, 32)).save(path)
            expert, content, audit = prepare_letterboxed_masks(
                path, contours, [2320, 2828]
            )
        self.assertTrue(expert.any())
        self.assertTrue(np.all(~expert | content))
        self.assertLess(audit["aspect_ratio_delta"], 0.01)
        self.assertEqual(audit["local_image_size"], [320, 390])

    def test_patient_shards_are_disjoint_and_complete(self) -> None:
        rows = [
            {"patient_id_hash": patient, "sample_id": f"{patient}-{index}"}
            for patient, count in (("a", 3), ("b", 2), ("c", 1), ("d", 4))
            for index in range(count)
        ]
        shards = [patient_disjoint_shard(rows, index, 2) for index in range(2)]
        self.assertEqual(sum(map(len, shards)), len(rows))
        patients = [{row["patient_id_hash"] for row in shard} for shard in shards]
        self.assertFalse(patients[0] & patients[1])
        self.assertEqual({row["sample_id"] for shard in shards for row in shard}, {row["sample_id"] for row in rows})

    def test_model_snapshot_uses_stable_frozen_payload_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, payload in {
                "config.json": b"config",
                "model.safetensors.index.json": b"index",
                "model-00001-of-00001.safetensors": b"weights",
                "tokenizer.json": b"tokenizer-v1",
                ".download-metadata": b"metadata-v1",
            }.items():
                (root / name).write_bytes(payload)
            expected = canonical_json_sha256(
                {
                    name: file_sha256(root / name)
                    for name in (
                        "config.json",
                        "model-00001-of-00001.safetensors",
                        "model.safetensors.index.json",
                    )
                }
            )
            self.assertEqual(snapshot_model_files(root), expected)
            (root / "tokenizer.json").write_bytes(b"tokenizer-v2")
            (root / ".download-metadata").write_bytes(b"metadata-v2")
            self.assertEqual(snapshot_model_files(root), expected)


if __name__ == "__main__":
    unittest.main()
