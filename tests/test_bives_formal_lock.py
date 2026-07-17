from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from bives_cxr.interventions import bives_control_seed
from bives_cxr.provenance import (
    build_run_lock,
    canonical_json_sha256,
    file_sha256,
    validate_calibrated_release_chain,
)
from bives_cxr.statement_cache import (
    build_statement_cache_payload,
    load_statement_embedding_matrix,
    validate_ontology_subset,
)
from scripts.train_bives_cxr import capture_rng_state, restore_rng_state


def encoder_provenance(model_name: str = "Qwen3.5-4B") -> dict[str, object]:
    return {
        "model_name_or_path": model_name,
        "revision": "locked",
        "tokenizer_revision": "locked",
        "tokenizer_class": "DummyTokenizer",
        "pooling": "input_embedding_mean",
        "normalize": True,
        "dtype": "float32",
    }


class FormalArtifactLockTests(unittest.TestCase):
    def test_rng_state_round_trip_is_exact(self) -> None:
        torch.manual_seed(41)
        state = capture_rng_state()
        expected = torch.rand(8)
        _ = torch.rand(16)
        restore_rng_state(state)
        self.assertTrue(torch.equal(torch.rand(8), expected))

    def test_release_chain_rejects_tampered_test_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "Qwen3.5-4B"
            model.mkdir()
            (model / "config.json").write_text('{"model_type":"qwen3_5"}', encoding="utf-8")
            (model / "model.safetensors.index.json").write_text(
                '{"weight_map":{}}', encoding="utf-8"
            )
            manifests = {}
            for split in ("train", "val", "calibration", "test"):
                path = root / f"{split}.jsonl"
                path.write_text(f'{{"split":"{split}"}}\n', encoding="utf-8")
                manifests[split] = path
            cache_path = root / "statements.pt"
            cache = build_statement_cache_payload(
                {"s1": torch.tensor([1.0, 0.0])},
                {"s1": "right effusion"},
                encoder_provenance(),
            )
            torch.save(cache, cache_path)
            config = {
                "model": {
                    "family": "Qwen3.5",
                    "path": str(model),
                    "statement_embeddings": {
                        "mode": "frozen_cached",
                        "path": str(cache_path),
                        "expected_sha256": file_sha256(cache_path),
                        "expected_vocabulary_sha256": cache["ontology"]["vocabulary_sha256"],
                        "expected_pooling": "input_embedding_mean",
                    },
                },
                "data": {
                    "train_manifest": str(manifests["train"]),
                    "val_manifest": str(manifests["val"]),
                    "calibration_manifest": str(manifests["calibration"]),
                    "test_manifest": str(manifests["test"]),
                },
                "evaluation": {"control_seed": 20260717},
            }
            lock = build_run_lock(config, git_commit="abc123")
            checkpoint = {
                "step": 7,
                "config": config,
                "run_lock": lock,
                "run_lock_sha256": canonical_json_sha256(lock),
            }
            checkpoint_path = root / "best.pt"
            torch.save(checkpoint, checkpoint_path)
            calibration = {
                "base_checkpoint_sha256": file_sha256(checkpoint_path),
                "run_lock_sha256": checkpoint["run_lock_sha256"],
                "statement_cache_sha256": lock["statement_cache_sha256"],
                "statement_vocabulary_sha256": lock["statement_vocabulary_sha256"],
                "selected_best_step": 7,
            }
            validated = validate_calibrated_release_chain(
                checkpoint_path=checkpoint_path,
                checkpoint=checkpoint,
                calibration=calibration,
                statement_cache_path=cache_path,
                test_manifest_path=manifests["test"],
                current_git_commit="abc123",
            )
            self.assertEqual(validated, lock)
            manifests["test"].write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "locked test manifest"):
                validate_calibrated_release_chain(
                    checkpoint_path=checkpoint_path,
                    checkpoint=checkpoint,
                    calibration=calibration,
                    statement_cache_path=cache_path,
                    test_manifest_path=manifests["test"],
                    current_git_commit="abc123",
                )

    def test_cache_expectations_and_test_subset_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cache.pt"
            texts = {"s1": "right effusion", "s2": "left opacity"}
            payload = build_statement_cache_payload(
                {
                    "s1": torch.tensor([1.0, 0.0]),
                    "s2": torch.tensor([0.0, 1.0]),
                },
                texts,
                encoder_provenance(),
            )
            torch.save(payload, path)
            expected = {
                "expected_sha256": file_sha256(path),
                "expected_vocabulary_sha256": payload["ontology"]["vocabulary_sha256"],
                "expected_pooling": "input_embedding_mean",
            }
            matrix = load_statement_embedding_matrix(
                path,
                {"s1": 0, "s2": 1},
                texts,
                expected_cache=expected,
            )
            self.assertEqual(tuple(matrix.shape), (2, 2))
            validate_ontology_subset(texts, {"s1": " RIGHT   EFFUSION "})
            with self.assertRaises(ValueError):
                validate_ontology_subset(texts, {"s3": "unseen"})
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_statement_embedding_matrix(
                    path,
                    {"s1": 0, "s2": 1},
                    texts,
                    expected_cache={**expected, "expected_sha256": "0" * 64},
                )

    def test_evaluation_control_seed_is_independent_of_training_seed(self) -> None:
        first = bives_control_seed(
            split="val",
            sample_id="sample-1",
            training_seed=17,
            evaluation_control_seed=20260717,
        )
        second = bives_control_seed(
            split="val",
            sample_id="sample-1",
            training_seed=23,
            evaluation_control_seed=20260717,
        )
        self.assertEqual(first, second)
        train_first = bives_control_seed(
            split="train",
            sample_id="sample-1",
            training_seed=17,
            evaluation_control_seed=20260717,
            epoch=2,
        )
        train_second = bives_control_seed(
            split="train",
            sample_id="sample-1",
            training_seed=23,
            evaluation_control_seed=20260717,
            epoch=2,
        )
        self.assertNotEqual(train_first, train_second)


if __name__ == "__main__":
    unittest.main()
