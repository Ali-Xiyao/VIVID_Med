from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from bives_cxr.audit import audit_manifests
from scripts.build_bives_proxy_p0 import build_proxy_manifests
from scripts.diagnose_bives_proxy_sc import (
    leave_one_out_centroid_metrics,
    select_balanced_sc,
    summarize_candidates,
)


class ProxyP0Tests(unittest.TestCase):
    def test_proxy_builder_is_patient_disjoint_and_explicitly_nonclinical(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate_path = root / "candidates.jsonl"
            rows = []
            states = ("support", "contradict", "uncertain", None)
            counter = 0
            for group in range(2):
                for state in states:
                    image_path = root / f"image-{counter}.png"
                    pixels = [((value * 3) + counter * 11) % 256 for value in range(32 * 32)]
                    image = Image.new("L", (32, 32))
                    image.putdata(pixels)
                    image.save(image_path)
                    rows.append(
                        {
                            "candidate_id": f"candidate-{counter}",
                            "canonical_statement_id": "atelectasis",
                            "statement_text": "Atelectasis is present.",
                            "image_id": f"image-{counter}",
                            "image_path": str(image_path),
                            "patient_id": f"patient-{counter}",
                            "study_id": f"study-{counter}",
                            "source_dataset": "MIMIC-CXR-JPG",
                            "parser_version": "test-parser-v1",
                            "parser_rules_sha256": "a" * 64,
                            "report_sha256": f"{counter:064x}",
                            "parser_state_candidate": state,
                            "parser_status": "candidate" if state else "requires_review_conflict",
                            "parser_cue": "must-not-leak",
                            "report_path": "must-not-leak.txt",
                        }
                    )
                    counter += 1
            candidate_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            output_dir = root / "proxy"
            summary = build_proxy_manifests(
                candidate_path,
                output_dir,
                ["atelectasis"],
                train_groups_per_finding=1,
                val_groups_per_finding=1,
                seed=7,
            )
            self.assertFalse(summary["formal_result"])
            self.assertFalse(summary["clinical_ground_truth"])
            lock = json.loads((output_dir / "proxy_dataset_lock.json").read_text(encoding="utf-8"))
            self.assertEqual(lock["kind"], "bives_weak_proxy_dataset_lock")
            self.assertFalse(lock["formal_result"])
            self.assertFalse(lock["clinical_ground_truth"])
            manifests = {
                split: output_dir / f"{split}_proxy.jsonl" for split in ("train", "val")
            }
            audit = audit_manifests(
                manifests,
                data_root=root,
                check_images=True,
                check_decodable=True,
                reject_constant_images=True,
                require_complete_statements=True,
                require_provenance=True,
                verify_image_sha256=True,
                require_matching_protocol=True,
                require_both_insufficient_kinds=False,
            )
            self.assertEqual(audit["status"], "pass", audit["errors"])
            train_rows = [json.loads(line) for line in manifests["train"].read_text(encoding="utf-8").splitlines()]
            val_rows = [json.loads(line) for line in manifests["val"].read_text(encoding="utf-8").splitlines()]
            self.assertTrue({row["patient_id"] for row in train_rows}.isdisjoint({row["patient_id"] for row in val_rows}))
            for row in train_rows + val_rows:
                self.assertEqual(row["annotation_status"], "weak_proxy_unreviewed")
                self.assertEqual(row["weak_label_claim"], "proxy_only_not_clinical_ground_truth")
                self.assertNotIn("report_path", row)
                self.assertNotIn("parser_cue", row)
            strict_audit = audit_manifests(
                manifests,
                data_root=root,
                require_provenance=True,
                verify_image_sha256=True,
                require_matching_protocol=True,
            )
            self.assertEqual(strict_audit["status"], "fail")
            self.assertTrue(any("both natural and synthetic" in error for error in strict_audit["errors"]))

    def test_proxy_builder_rejects_cross_finding_candidate_id_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image_path = root / "image.png"
            Image.new("L", (32, 32), 127).save(image_path)
            common = {
                "candidate_id": "duplicate",
                "source_image_candidate_id": "image-source",
                "source_dataset": "MIMIC-CXR-JPG",
                "patient_id": "patient-1",
                "study_id": "study-1",
                "image_id": "image-1",
                "image_path": str(image_path),
                "statement_text": "Atelectasis is present.",
                "parser_version": "test-parser-v2",
                "parser_rules_sha256": "a" * 64,
                "report_sha256": "b" * 64,
                "parser_state_candidate": "support",
                "parser_status": "candidate",
            }
            rows = [
                {**common, "canonical_statement_id": "atelectasis"},
                {**common, "canonical_statement_id": "consolidation"},
            ]
            candidate_path = root / "candidates.jsonl"
            candidate_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "globally unique"):
                build_proxy_manifests(
                    candidate_path,
                    root / "proxy",
                    ["atelectasis"],
                    train_groups_per_finding=1,
                    val_groups_per_finding=1,
                    seed=7,
                )

    def test_sc_diagnostic_selects_patient_disjoint_balanced_rows(self) -> None:
        rows = []
        for finding in ("atelectasis", "consolidation"):
            for state in ("support", "contradict"):
                for index in range(4):
                    rows.append(
                        {
                            "candidate_id": f"{finding}-{state}-{index}",
                            "canonical_statement_id": finding,
                            "parser_status": "candidate",
                            "parser_state_candidate": state,
                            "patient_id": f"{finding}-{state}-patient-{index}",
                            "study_id": f"study-{finding}-{state}-{index}",
                            "image_path": f"image-{finding}-{state}-{index}.png",
                            "report_sha256": f"report-{finding}-{state}-{index}",
                            "parser_cue": finding,
                        }
                    )
        selected = select_balanced_sc(rows, ["atelectasis", "consolidation"], 3, 7)
        self.assertEqual(len(selected), 12)
        self.assertEqual(len({row["patient_id"] for row in selected}), 12)
        summary = summarize_candidates(rows, ["atelectasis", "consolidation"])
        self.assertEqual(summary["duplicate_candidate_ids"], 0)
        self.assertEqual(summary["findings"]["atelectasis"]["states"]["support"]["patients"], 4)

    def test_sc_centroid_metric_detects_separable_features(self) -> None:
        features = np.asarray(
            [[-2.0, 0.0], [-1.0, 0.1], [-1.5, -0.1], [1.0, 0.0], [2.0, 0.1], [1.5, -0.1]],
            dtype=np.float32,
        )
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        metrics = leave_one_out_centroid_metrics(features, labels)
        self.assertEqual(metrics["loo_centroid_auroc"], 1.0)
        self.assertEqual(metrics["loo_centroid_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
