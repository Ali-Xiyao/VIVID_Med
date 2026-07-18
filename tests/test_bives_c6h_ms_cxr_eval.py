from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bives_cxr.c6h_ms_cxr_eval import (
    C6G_CONTROL_VERSION,
    C6G_LOCK_FORMAT_VERSION,
    EXPECTED_C6G_CANONICAL_SHA256,
    validate_c6g_geometry_release,
    validate_c6h_protocol,
)
from bives_cxr.provenance import canonical_json_sha256, file_sha256


class C6HContracts(unittest.TestCase):
    def _config(self) -> dict:
        return {
            "authorization": {
                "model_evaluation_authorized": True,
                "one_time_execution_authorized": True,
                "local_only": True,
            },
            "model": {"family": "Qwen3.5", "scale": "2B"},
            "checkpoint": {"variant": "B2_sparse_exact_k", "step": 450, "topk": 16},
            "geometry": {
                "control_version": C6G_CONTROL_VERSION,
                "lock_canonical_sha256": EXPECTED_C6G_CANONICAL_SHA256,
            },
            "intervention": {
                "image_size": 448,
                "local_mean_ring_width": 8,
                "masked_gaussian_sigma": 8.0,
                "masked_gaussian_truncate": 3.0,
                "dilation_fraction": 0.0,
            },
            "evaluation": {
                "bootstrap_replicates": 2000,
                "bootstrap_seed": 17,
                "allow_classification_metrics": False,
                "terminal_after_completion": True,
            },
            "scale": {"qwen35_4b_authorized": False, "qwen35_9b_authorized": False},
        }

    def test_protocol_accepts_only_frozen_one_time_2b(self) -> None:
        validate_c6h_protocol(self._config())
        config = self._config()
        config["model"]["scale"] = "4B"
        with self.assertRaisesRegex(ValueError, "only Qwen3.5-2B"):
            validate_c6h_protocol(config)

    def test_protocol_rejects_classification_and_scale(self) -> None:
        config = self._config()
        config["evaluation"]["allow_classification_metrics"] = True
        with self.assertRaisesRegex(ValueError, "evaluation boundary"):
            validate_c6h_protocol(config)
        config = self._config()
        config["scale"]["qwen35_9b_authorized"] = True
        with self.assertRaisesRegex(ValueError, "4B/9B"):
            validate_c6h_protocol(config)

    def _geometry_fixture(self, root: Path) -> tuple[Path, Path, Path, Path, Path]:
        manifest = root / "manifest.jsonl"
        rows_path = root / "rows.jsonl"
        certificates_path = root / "certificates.jsonl"
        mask_dir = root / "masks"
        mask_dir.mkdir()
        manifest_rows = []
        rows = []
        certificates = []
        for index in range(29):
            sample_id = f"sample-{index:02d}"
            manifest_rows.append({"sample_id": sample_id})
            target = np.zeros((448, 448), dtype=np.uint8)
            control = np.zeros((448, 448), dtype=np.uint8)
            content = np.ones((448, 448), dtype=np.uint8)
            target[0, 0] = 1
            control[2, 2] = 1
            mask_path = mask_dir / f"{sample_id}.npz"
            np.savez_compressed(
                mask_path,
                target_mask=target,
                control_mask=control,
                content_mask=content,
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "feasible": True,
                    "model_evaluation_authorized": False,
                    "gpu_authorized": False,
                    "image_decode_authorized": False,
                    "scores_accessed": False,
                    "mask_file": mask_path.name,
                    "mask_sha256": file_sha256(mask_path),
                    "target_area_pixels": 1,
                    "control_area_pixels": 1,
                }
            )
            certificates.append({"sample_id": sample_id, "feasible": True})
        manifest.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows),
            encoding="utf-8",
        )
        rows_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        certificates_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in certificates),
            encoding="utf-8",
        )
        summary = {
            "format_version": C6G_LOCK_FORMAT_VERSION,
            "status": "pass",
            "rows": 29,
            "eligible": 29,
            "infeasible": 0,
            "image_size": 448,
            "per_finding": {"consolidation": 15, "pleural_effusion": 14},
            "denominator_exclusions": 0,
            "invariant_failures": 0,
            "evaluation_gate_open_geometry": True,
            "model_evaluation_authorized": False,
            "gpu_authorized": False,
            "image_decode_authorized": False,
            "scores_accessed": False,
            "control_version": C6G_CONTROL_VERSION,
            "thresholds": {
                "max_location_distance": 0.3,
                "max_log_perimeter_ratio": 0.9,
            },
        }
        summary["canonical_sha256"] = canonical_json_sha256(summary)
        lock = {
            **summary,
            "geometry_rows_sha256": file_sha256(rows_path),
            "candidate_certificates_sha256": file_sha256(certificates_path),
            "source_manifest_sha256": file_sha256(manifest),
        }
        lock["canonical_sha256"] = canonical_json_sha256(lock)
        lock_path = root / "lock.json"
        lock_path.write_text(json.dumps(lock), encoding="utf-8")
        return lock_path, rows_path, certificates_path, manifest, mask_dir

    def test_c6g_release_validates_all_29_masks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._geometry_fixture(Path(directory))
            lock = validate_c6g_geometry_release(
                lock_path=paths[0],
                rows_path=paths[1],
                certificates_path=paths[2],
                manifest_path=paths[3],
                mask_dir=paths[4],
            )
            self.assertEqual(lock["eligible"], 29)

    def test_c6g_release_rejects_mask_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._geometry_fixture(Path(directory))
            first_mask = sorted(paths[4].glob("*.npz"))[0]
            first_mask.write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError, "mask identity changed"):
                validate_c6g_geometry_release(
                    lock_path=paths[0],
                    rows_path=paths[1],
                    certificates_path=paths[2],
                    manifest_path=paths[3],
                    mask_dir=paths[4],
                )


if __name__ == "__main__":
    unittest.main()
