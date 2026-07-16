from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

from bives_cxr.audit import audit_manifests
from bives_cxr.backbones import validate_qwen35_model_path
from bives_cxr.data import BiVESManifestDataset, SameStatementStateBatchSampler
from scripts.train_bives_cxr import assert_full_sample_coverage


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class BiVESReadinessTest(unittest.TestCase):
    def test_active_configs_are_qwen35_only(self) -> None:
        config_root = Path(__file__).resolve().parents[1] / "configs" / "bives_cxr"
        configs = sorted(config_root.glob("*.yaml"))
        self.assertTrue(configs)
        for path in configs:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            model = payload["model"]
            self.assertEqual(str(model["family"]).lower(), "qwen3.5")
            self.assertIn("Qwen3.5-", str(model["path"]))
            self.assertGreater(float(payload["loss"].get("lambda_pair", 0.0)), 0.0)
            self.assertGreater(float(payload["loss"].get("lambda_u_pol", 0.0)), 0.0)
            self.assertEqual(payload["sampling"]["type"], "same_statement_state_group")
            self.assertEqual(payload["bives"]["mask"]["type"], "soft_topk")
            self.assertEqual(float(payload["loss"]["lambda_min"]), 0.0)
            self.assertGreaterEqual(int(payload["bives"]["contextual_layers"]), 1)
            self.assertTrue(payload["audit"]["require_complete_statements"])
            self.assertTrue(payload["audit"]["verify_image_sha256"])
            self.assertFalse(payload["evaluation"]["run_test"])
            self.assertEqual(payload["evaluation"]["selection_metric"], "nll")
        main = yaml.safe_load((config_root / "qwen35_4b_main.yaml").read_text(encoding="utf-8"))
        self.assertIn("train_locked.jsonl", main["data"]["train_manifest"])
        self.assertIn("calibration_locked.jsonl", main["data"]["calibration_manifest"])
        self.assertIn("test_locked.jsonl", main["data"]["test_manifest"])
        self.assertEqual(main["model"]["statement_embeddings"]["mode"], "frozen_cached")
        scale = yaml.safe_load((config_root / "qwen35_9b_scale.yaml").read_text(encoding="utf-8"))
        self.assertEqual(scale["model"]["statement_embeddings"]["mode"], "frozen_cached")
        self.assertEqual(scale["training"]["max_steps"], main["training"]["max_steps"])
        self.assertEqual(scale["training"]["gradient_accumulation_steps"], 2)

    def test_qwen35_path_guard_accepts_only_multimodal_qwen35(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            good = root / "good"
            good.mkdir()
            (good / "config.json").write_text(
                json.dumps({"model_type": "qwen3_5", "vision_config": {"hidden_size": 16}}),
                encoding="utf-8",
            )
            self.assertEqual(validate_qwen35_model_path(good)["model_type"], "qwen3_5")

            old = root / "old"
            old.mkdir()
            (old / "config.json").write_text(
                json.dumps({"model_type": "qwen3_vl", "vision_config": {"hidden_size": 16}}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                validate_qwen35_model_path(old)

    def test_manifest_audit_detects_patient_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [
                {
                    "sample_id": f"sample-{index}",
                    "patient_id": "shared-patient",
                    "image_path": "unused.png",
                    "group_id": "effusion-quartet",
                    "canonical_statement_id": "effusion-right",
                    "statement_text": "A right pleural effusion is present.",
                    "state": state,
                }
                for index, state in enumerate(
                    ("support", "contradict", "uncertain", "insufficient")
                )
            ]
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            write_jsonl(train, rows)
            write_jsonl(
                val,
                [{**row, "sample_id": f"val-{index}"} for index, row in enumerate(rows)],
            )
            report = audit_manifests({"train": train, "val": val})
            self.assertEqual(report["status"], "fail")
            self.assertTrue(any("patient leakage" in error for error in report["errors"]))

    def test_manifest_audit_passes_isolated_complete_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifests: dict[str, Path] = {}
            for split in ("train", "val"):
                rows = [
                    {
                        "sample_id": f"{split}-{index}",
                        "patient_id": f"{split}-patient-{index}",
                        "image_path": f"{split}-{index}.png",
                        "group_id": f"{split}-effusion-quartet",
                        "canonical_statement_id": "effusion-right",
                        "statement_text": "A right pleural effusion is present.",
                        "state": state,
                    }
                    for index, state in enumerate(
                        ("support", "contradict", "uncertain", "insufficient")
                    )
                ]
                path = root / f"{split}.jsonl"
                write_jsonl(path, rows)
                manifests[split] = path
            report = audit_manifests(
                manifests,
                require_complete_statements=True,
            )
            self.assertEqual(report["status"], "pass")

    def test_group_sampler_yields_ordered_complete_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for statement in ("effusion-right", "opacity-left"):
                for index, state in enumerate(
                    ("support", "contradict", "uncertain", "insufficient")
                ):
                    rows.append(
                        {
                            "sample_id": f"{statement}-{state}",
                            "patient_id": f"{statement}-patient-{index}",
                            "image_path": f"{statement}-{state}.png",
                            "group_id": f"{statement}-quartet",
                            "canonical_statement_id": statement,
                            "statement_text": statement,
                            "state": state,
                        }
                    )
            manifest = root / "groups.jsonl"
            write_jsonl(manifest, rows)
            dataset = BiVESManifestDataset(manifest, root)
            sampler = SameStatementStateBatchSampler(
                dataset,
                groups_per_batch=2,
                shuffle=False,
            )
            batches = list(sampler)
            self.assertEqual(len(batches), 1)
            sampled_states = [dataset.rows[index]["state"] for index in batches[0]]
            self.assertEqual(
                sampled_states,
                [
                    "support",
                    "contradict",
                    "uncertain",
                    "insufficient",
                    "support",
                    "contradict",
                    "uncertain",
                    "insufficient",
                ],
            )

    def test_primary_evaluation_coverage_contract_rejects_subsets_and_duplicates(self) -> None:
        expected = [f"sample-{index}" for index in range(24)]
        assert_full_sample_coverage(expected, list(expected))
        with self.assertRaises(RuntimeError):
            assert_full_sample_coverage(expected, expected[:8])
        with self.assertRaises(RuntimeError):
            assert_full_sample_coverage(expected, expected[:-1] + [expected[0]])

    def test_strict_audit_rejects_semantic_and_label_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for index, state in enumerate(
                ("support", "contradict", "uncertain", "insufficient")
            ):
                rows.append(
                    {
                        "sample_id": f"sample-{index}",
                        "patient_id": f"patient-{index}",
                        "image_path": "same.png" if index < 2 else f"image-{index}.png",
                        "group_id": "conflict-quartet",
                        "canonical_statement_id": "effusion-right",
                        "statement_text": "Different wording" if index == 3 else "Effusion present.",
                        "state": state,
                    }
                )
            path = root / "conflicts.jsonl"
            write_jsonl(path, rows)
            report = audit_manifests({"train": path}, require_complete_statements=True)
            self.assertEqual(report["status"], "fail")
            self.assertTrue(any("inconsistent text" in error for error in report["errors"]))
            self.assertTrue(any("conflicting labels" in error for error in report["errors"]))

    def test_strict_audit_passes_complete_provenance_and_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifests: dict[str, Path] = {}
            for split in ("train", "val"):
                rows = []
                for statement_index, statement_id in enumerate(("effusion-right", "opacity-left")):
                    for state_index, state in enumerate(
                        ("support", "contradict", "uncertain", "insufficient")
                    ):
                        image_name = f"{split}-{statement_index}-{state_index}.png"
                        image = Image.new("L", (8, 8))
                        split_offset = 0 if split == "train" else 80
                        image.putdata(
                            [
                                (pixel + state_index + split_offset) % 255
                                for pixel in range(64)
                            ]
                        )
                        image_path = root / image_name
                        image.save(image_path)
                        image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
                        rows.append(
                            {
                                "sample_id": f"{split}-{statement_id}-{state}",
                                "patient_id": f"{split}-patient-{statement_index}-{state_index}",
                                "study_id": f"{split}-study-{statement_index}-{state_index}",
                                "image_path": image_name,
                                "image_sha256": image_hash,
                                "group_id": f"{split}-{statement_id}",
                                "canonical_statement_id": statement_id,
                                "statement_text": statement_id,
                                "state": state,
                                "label_source": "expert",
                                "annotation_status": "expert_reviewed",
                                "insufficient_kind": (
                                    "natural" if statement_index == 0 else "synthetic"
                                )
                                if state == "insufficient"
                                else None,
                            }
                        )
                path = root / f"{split}.jsonl"
                write_jsonl(path, rows)
                manifests[split] = path
            report = audit_manifests(
                manifests,
                data_root=root,
                check_images=True,
                check_decodable=True,
                reject_constant_images=True,
                require_complete_statements=True,
                require_provenance=True,
            )
            self.assertEqual(report["status"], "pass", report["errors"])
            self.assertGreater(report["splits"]["train"]["verified_image_hash_count"], 0)

    def test_audit_rejects_declared_image_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for index, state in enumerate(
                ("support", "contradict", "uncertain", "insufficient")
            ):
                image_path = root / f"{index}.png"
                Image.new("L", (8, 8), color=20 + index).save(image_path)
                rows.append(
                    {
                        "sample_id": f"sample-{index}",
                        "patient_id": f"patient-{index}",
                        "image_path": image_path.name,
                        "image_sha256": "0" * 64,
                        "group_id": "quartet-1",
                        "canonical_statement_id": "effusion-right",
                        "statement_text": "Effusion right",
                        "state": state,
                    }
                )
            manifest = root / "manifest.jsonl"
            write_jsonl(manifest, rows)
            report = audit_manifests(
                {"train": manifest},
                data_root=root,
                verify_image_sha256=True,
            )
            self.assertEqual(report["status"], "fail")
            self.assertTrue(
                any("image_sha256 mismatch" in error for error in report["errors"])
            )

    def test_group_sampler_does_not_merge_two_quartets_with_same_statement(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for group_id in ("quartet-a", "quartet-b"):
                for index, state in enumerate(
                    ("support", "contradict", "uncertain", "insufficient")
                ):
                    rows.append(
                        {
                            "sample_id": f"{group_id}-{state}",
                            "patient_id": f"{group_id}-patient-{index}",
                            "image_path": f"{group_id}-{state}.png",
                            "group_id": group_id,
                            "canonical_statement_id": "effusion-right",
                            "statement_text": "Effusion right",
                            "state": state,
                        }
                    )
            manifest = root / "groups.jsonl"
            write_jsonl(manifest, rows)
            dataset = BiVESManifestDataset(manifest, root)
            batches = list(
                SameStatementStateBatchSampler(
                    dataset,
                    groups_per_batch=1,
                    shuffle=False,
                )
            )
            self.assertEqual(len(batches), 2)
            for batch in batches:
                self.assertEqual(
                    len({dataset.rows[index]["group_id"] for index in batch}),
                    1,
                )


if __name__ == "__main__":
    unittest.main()
