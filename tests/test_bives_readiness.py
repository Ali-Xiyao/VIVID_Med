from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from bives_cxr.audit import audit_manifests
from bives_cxr.backbones import validate_qwen35_model_path


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
            self.assertEqual(float(payload["loss"].get("lambda_pair", 0.0)), 0.0)

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
                        "image_path": "unused.png",
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


if __name__ == "__main__":
    unittest.main()
