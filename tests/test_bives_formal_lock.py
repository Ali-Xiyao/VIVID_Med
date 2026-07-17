from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from bives_cxr.dataset_lock import build_dataset_lock, validate_dataset_lock
from bives_cxr.interventions import bives_control_seed
from bives_cxr.provenance import (
    _calibration_prediction_nll,
    build_run_lock,
    build_source_snapshot,
    canonical_json_sha256,
    file_sha256,
    validate_calibrated_release_chain,
    validate_checkpoint_run_lock,
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


def build_release_fixture(root: Path, *, git_commit: str) -> dict[str, object]:
    """Minimal complete artifact chain for validator and CLI integration tests."""

    model = root / "Qwen3.5-4B"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3_5"}', encoding="utf-8")
    (model / "model.safetensors.index.json").write_text(
        '{"weight_map":{"x":"model-00001-of-00001.safetensors"}}',
        encoding="utf-8",
    )
    (model / "model-00001-of-00001.safetensors").write_bytes(b"weight")
    manifests: dict[str, Path] = {}
    for split in ("train", "val", "calibration", "test"):
        rows = [
            {
                "sample_id": f"{split}-{state}",
                "patient_id": f"{split}-patient-{state}",
                "image_path": f"{split}-{state}.png",
                "image_sha256": (f"{split}-{state}".encode("utf-8").hex() + "0" * 64)[:64],
                "study_id": f"{split}-study-{state}",
                "group_id": f"{split}-group",
                "canonical_statement_id": "s1",
                "statement_text": "right effusion",
                "state": state,
            }
            for state in ("support", "contradict", "uncertain", "insufficient")
        ]
        path = root / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        manifests[split] = path
    audit_options = {
        "check_images": False,
        "require_complete_statements": True,
        "check_decodable": False,
        "reject_constant_images": False,
        "require_provenance": False,
        "verify_image_sha256": False,
        "require_matching_protocol": False,
    }
    dataset_lock = build_dataset_lock(
        manifests, data_root=root, audit_options=audit_options
    )
    dataset_lock_path = root / "dataset_lock.json"
    dataset_lock_path.write_text(json.dumps(dataset_lock), encoding="utf-8")
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
            "data_root": str(root),
            **{f"{split}_manifest": str(path) for split, path in manifests.items()},
        },
        "evaluation": {"control_seed": 20260717},
    }
    run_lock = build_run_lock(config, git_commit=git_commit, dataset_lock=dataset_lock)
    checkpoint = {
        "step": 7,
        "config": config,
        "run_lock": run_lock,
        "run_lock_sha256": canonical_json_sha256(run_lock),
        "bives_head": {
            "decoder.tau_a": torch.tensor(1.0),
            "decoder.tau_p": torch.tensor(1.0),
            "decoder.uncertainty_mass": torch.tensor(1.0),
        },
    }
    checkpoint_path = root / "best.pt"
    torch.save(checkpoint, checkpoint_path)
    predictions = root / "calibration_pre_predictions.jsonl"
    predictions.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"evidence_pos": 1.2, "evidence_neg": 0.2, "target": 0},
                {"evidence_pos": 0.2, "evidence_neg": 1.2, "target": 1},
                {"evidence_pos": 0.4, "evidence_neg": 0.4, "target": 2},
                {"evidence_pos": 0.0, "evidence_neg": 0.0, "target": 3},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    decoder_parameters = {"tau_a": 1.0, "tau_p": 1.0, "uncertainty_mass": 1.0}
    expected_nll = _calibration_prediction_nll(predictions, decoder_parameters)
    calibration = {
        "base_checkpoint_sha256": file_sha256(checkpoint_path),
        "run_lock_sha256": checkpoint["run_lock_sha256"],
        "statement_cache_sha256": run_lock["statement_cache_sha256"],
        "statement_vocabulary_sha256": run_lock["statement_vocabulary_sha256"],
        "selected_best_step": 7,
        "format_version": 3,
        "calibration_algorithm": "monotone_decoder_lbfgs_v1",
        "calibration_manifest_sha256": run_lock["manifest_sha256"]["calibration"],
        "control_protocol_version": run_lock["control_protocol_version"],
        "evaluation_control_seed": 20260717,
        "uncalibrated_decoder_parameters": decoder_parameters,
        "calibrated_decoder_parameters": decoder_parameters,
        "calibration_predictions_file": str(predictions),
        "calibration_predictions_sha256": file_sha256(predictions),
        "calibration_pre_nll": expected_nll,
        "calibration_post_nll": expected_nll,
    }
    calibration["canonical_artifact_sha256"] = canonical_json_sha256(calibration)
    calibration_path = root / "calibration_artifact.json"
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    return {
        "model": model,
        "manifests": manifests,
        "dataset_lock": dataset_lock,
        "dataset_lock_path": dataset_lock_path,
        "cache_path": cache_path,
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "predictions": predictions,
        "calibration": calibration,
        "calibration_path": calibration_path,
        "run_lock": run_lock,
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
                '{"weight_map":{"x":"model-00001-of-00001.safetensors"}}', encoding="utf-8"
            )
            (model / "model-00001-of-00001.safetensors").write_bytes(b"weight")
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
            lock = build_run_lock(
                config,
                git_commit="abc123",
                dataset_lock={"status": "pass", "format_version": 1},
            )
            checkpoint = {
                "step": 7,
                "config": config,
                "run_lock": lock,
                "run_lock_sha256": canonical_json_sha256(lock),
                "bives_head": {
                    "decoder.tau_a": torch.tensor(1.0),
                    "decoder.tau_p": torch.tensor(1.0),
                    "decoder.uncertainty_mass": torch.tensor(1.0),
                },
            }
            checkpoint_path = root / "best.pt"
            torch.save(checkpoint, checkpoint_path)
            prediction_path = root / "calibration_pre_predictions.jsonl"
            prediction_path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {"evidence_pos": 1.2, "evidence_neg": 0.2, "target": 0},
                        {"evidence_pos": 0.1, "evidence_neg": 1.3, "target": 1},
                        {"evidence_pos": 0.4, "evidence_neg": 0.4, "target": 2},
                        {"evidence_pos": 0.0, "evidence_neg": 0.0, "target": 3},
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            decoder_parameters = {
                "tau_a": 1.0,
                "tau_p": 1.0,
                "uncertainty_mass": 1.0,
            }
            expected_nll = _calibration_prediction_nll(
                prediction_path, decoder_parameters
            )
            calibration = {
                "base_checkpoint_sha256": file_sha256(checkpoint_path),
                "run_lock_sha256": checkpoint["run_lock_sha256"],
                "statement_cache_sha256": lock["statement_cache_sha256"],
                "statement_vocabulary_sha256": lock["statement_vocabulary_sha256"],
                "selected_best_step": 7,
                "format_version": 3,
                "calibration_algorithm": "monotone_decoder_lbfgs_v1",
                "calibration_manifest_sha256": lock["manifest_sha256"]["calibration"],
                "control_protocol_version": lock["control_protocol_version"],
                "evaluation_control_seed": 20260717,
                "uncalibrated_decoder_parameters": decoder_parameters,
                "calibrated_decoder_parameters": decoder_parameters,
                "calibration_predictions_file": str(prediction_path),
                "calibration_predictions_sha256": file_sha256(prediction_path),
                "calibration_pre_nll": expected_nll,
                "calibration_post_nll": expected_nll,
            }
            calibration["canonical_artifact_sha256"] = canonical_json_sha256(calibration)
            validated = validate_calibrated_release_chain(
                checkpoint_path=checkpoint_path,
                checkpoint=checkpoint,
                calibration=calibration,
                statement_cache_path=cache_path,
                test_manifest_path=manifests["test"],
                current_git_commit="abc123",
            )
            self.assertEqual(validated, lock)
            (model / "model-00001-of-00001.safetensors").write_bytes(b"tampered-weight")
            with self.assertRaisesRegex(ValueError, "base model snapshot"):
                validate_checkpoint_run_lock(checkpoint, current_git_commit="abc123")
            (model / "model-00001-of-00001.safetensors").write_bytes(b"weight")
            invalid_calibration = dict(calibration)
            invalid_calibration["calibrated_decoder_parameters"] = {
                "tau_a": float("nan"),
                "tau_p": 1.0,
                "uncertainty_mass": 1.0,
            }
            invalid_calibration["canonical_artifact_sha256"] = canonical_json_sha256(
                {key: value for key, value in invalid_calibration.items() if key != "canonical_artifact_sha256"}
            )
            with self.assertRaisesRegex(ValueError, "finite and bounded"):
                validate_calibrated_release_chain(
                    checkpoint_path=checkpoint_path,
                    checkpoint=checkpoint,
                    calibration=invalid_calibration,
                    statement_cache_path=cache_path,
                    test_manifest_path=manifests["test"],
                    current_git_commit="abc123",
                )
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

    def test_source_only_snapshot_rejects_unlisted_active_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package = root / "bives_cxr"
            package.mkdir()
            visible = package / "visible.py"
            visible.write_text("VALUE = 1\n", encoding="utf-8")
            files = {"bives_cxr/visible.py": file_sha256(visible)}
            (root / ".bives_source_manifest.json").write_text(
                json.dumps(
                    {
                        "kind": "source_archive",
                        "files": files,
                        "tree_sha256": canonical_json_sha256(files),
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                build_source_snapshot(root=root, require_clean=True)["files"], files
            )
            (package / "injected.py").write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unlisted active files"):
                build_source_snapshot(root=root, require_clean=True)
            (package / "injected.py").unlink()
            (root / "sitecustomize.py").write_text("raise RuntimeError\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unlisted active files"):
                build_source_snapshot(root=root, require_clean=True)

    def test_release_chain_validates_four_split_lock_and_prediction_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = build_release_fixture(root, git_commit="abc123")
            validated = validate_calibrated_release_chain(
                checkpoint_path=fixture["checkpoint_path"],
                checkpoint=fixture["checkpoint"],
                calibration=fixture["calibration"],
                statement_cache_path=fixture["cache_path"],
                test_manifest_path=fixture["manifests"]["test"],
                current_git_commit="abc123",
                calibration_artifact_path=fixture["calibration_path"],
                dataset_lock_path=fixture["dataset_lock_path"],
                data_root=root,
                dataset_manifests=fixture["manifests"],
            )
            self.assertEqual(validated, fixture["run_lock"])
            missing_config = dict(fixture["checkpoint"])
            missing_config.pop("config")
            with self.assertRaisesRegex(ValueError, "missing resolved config"):
                validate_calibrated_release_chain(
                    checkpoint_path=fixture["checkpoint_path"],
                    checkpoint=missing_config,
                    calibration=fixture["calibration"],
                    statement_cache_path=fixture["cache_path"],
                    test_manifest_path=fixture["manifests"]["test"],
                    current_git_commit="abc123",
                )
            with self.assertRaisesRegex(ValueError, "missing manifests: val"):
                validate_calibrated_release_chain(
                    checkpoint_path=fixture["checkpoint_path"],
                    checkpoint=fixture["checkpoint"],
                    calibration=fixture["calibration"],
                    statement_cache_path=fixture["cache_path"],
                    test_manifest_path=fixture["manifests"]["test"],
                    current_git_commit="abc123",
                    dataset_lock_path=fixture["dataset_lock_path"],
                    data_root=root,
                    dataset_manifests={
                        "train": fixture["manifests"]["train"],
                        "calibration": fixture["manifests"]["calibration"],
                        "test": fixture["manifests"]["test"],
                    },
                )
            broken_lock = dict(fixture["dataset_lock"])
            broken_lock["status"] = "fail"
            fixture["dataset_lock_path"].write_text(json.dumps(broken_lock), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a passing"):
                validate_calibrated_release_chain(
                    checkpoint_path=fixture["checkpoint_path"],
                    checkpoint=fixture["checkpoint"],
                    calibration=fixture["calibration"],
                    statement_cache_path=fixture["cache_path"],
                    test_manifest_path=fixture["manifests"]["test"],
                    current_git_commit="abc123",
                    dataset_lock_path=fixture["dataset_lock_path"],
                    data_root=root,
                    dataset_manifests=fixture["manifests"],
                )
            fixture["dataset_lock_path"].write_text(
                json.dumps(fixture["dataset_lock"]), encoding="utf-8"
            )
            train_rows = [
                json.loads(line)
                for line in fixture["manifests"]["train"].read_text(encoding="utf-8").splitlines()
            ]
            train_rows[0]["sample_id"] = "train-support-rewritten"
            fixture["manifests"]["train"].write_text(
                "".join(json.dumps(row) + "\n" for row in train_rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "dataset lock mismatch: manifest_sha256"):
                validate_calibrated_release_chain(
                    checkpoint_path=fixture["checkpoint_path"],
                    checkpoint=fixture["checkpoint"],
                    calibration=fixture["calibration"],
                    statement_cache_path=fixture["cache_path"],
                    test_manifest_path=fixture["manifests"]["test"],
                    current_git_commit="abc123",
                    dataset_lock_path=fixture["dataset_lock_path"],
                    data_root=root,
                    dataset_manifests=fixture["manifests"],
                )
            fixture["manifests"]["train"].write_text(
                "".join(
                    json.dumps({**row, "sample_id": f"train-{state}"}) + "\n"
                    for row, state in zip(
                        train_rows,
                        ("support", "contradict", "uncertain", "insufficient"),
                    )
                ),
                encoding="utf-8",
            )
            tampered = dict(fixture["calibration"])
            tampered["calibration_post_nll"] -= 0.1
            tampered["canonical_artifact_sha256"] = canonical_json_sha256(
                {key: value for key, value in tampered.items() if key != "canonical_artifact_sha256"}
            )
            with self.assertRaisesRegex(ValueError, "does not match locked calibration prediction evidence"):
                validate_calibrated_release_chain(
                    checkpoint_path=fixture["checkpoint_path"],
                    checkpoint=fixture["checkpoint"],
                    calibration=tampered,
                    statement_cache_path=fixture["cache_path"],
                    test_manifest_path=fixture["manifests"]["test"],
                    current_git_commit="abc123",
                    calibration_artifact_path=fixture["calibration_path"],
                )

    def test_final_evaluator_cli_release_chain_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            current_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            fixture = build_release_fixture(root, git_commit=current_commit)
            command = [
                sys.executable,
                "scripts/evaluate_bives_final.py",
                "--checkpoint", str(fixture["checkpoint_path"]),
                "--calibration-artifact", str(fixture["calibration_path"]),
                "--train-manifest", str(fixture["manifests"]["train"]),
                "--val-manifest", str(fixture["manifests"]["val"]),
                "--calibration-manifest", str(fixture["manifests"]["calibration"]),
                "--test-manifest", str(fixture["manifests"]["test"]),
                "--statement-cache", str(fixture["cache_path"]),
                "--dataset-lock", str(fixture["dataset_lock_path"]),
                "--data-root", str(root),
                "--output-dir", str(root / "release_output"),
                "--validate-release-chain-only",
            ]
            passed = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(passed.returncode, 0, passed.stderr)
            self.assertIn('"status": "pass"', passed.stdout)
            checkpoint = torch.load(fixture["checkpoint_path"], map_location="cpu", weights_only=False)
            checkpoint.pop("config")
            missing_config = root / "missing_config.pt"
            torch.save(checkpoint, missing_config)
            command[3] = str(missing_config)
            failed = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("checkpoint is missing resolved config", failed.stderr)
    def test_joint_dataset_lock_rejects_patient_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifests: dict[str, Path] = {}
            for split in ("train", "val", "calibration", "test"):
                rows = []
                for state in ("support", "contradict", "uncertain", "insufficient"):
                    rows.append({
                        "sample_id": f"{split}-{state}",
                        "patient_id": f"{split}-patient-{state}",
                        "image_path": f"{split}-{state}.png",
                        "image_sha256": (f"{split}-{state}".encode("utf-8").hex() + "0" * 64)[:64],
                        "study_id": f"{split}-study-{state}",
                        "group_id": f"{split}-group",
                        "canonical_statement_id": "s1",
                        "statement_text": "right effusion",
                        "state": state,
                    })
                path = root / f"{split}.jsonl"
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                manifests[split] = path
            options = {
                "check_images": False, "require_complete_statements": True,
                "check_decodable": False, "reject_constant_images": False,
                "require_provenance": False, "verify_image_sha256": False,
                "require_matching_protocol": False,
            }
            lock = build_dataset_lock(manifests, data_root=root, audit_options=options)
            lock_path = root / "dataset_lock.json"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            self.assertEqual(
                validate_dataset_lock(lock_path, manifests, data_root=root), lock
            )
            leaked = json.loads(manifests["test"].read_text(encoding="utf-8").splitlines()[0])
            leaked["patient_id"] = "train-patient-support"
            rows = [leaked] + [json.loads(line) for line in manifests["test"].read_text(encoding="utf-8").splitlines()[1:]]
            manifests["test"].write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "joint four-split dataset audit failed"):
                validate_dataset_lock(lock_path, manifests, data_root=root)

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
