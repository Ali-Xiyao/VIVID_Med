from __future__ import annotations

import json
import unittest

import torch

from morph_cxr.case_study import analyze_morph_result
from morph_cxr.data import boundary_from_mask, mask_moments
from morph_cxr.experts import (
    EXPERT_TYPES,
    MorphologyConceptExpert,
    MorphologyExpertConfig,
    concept_monotonicity_deltas,
)
from morph_cxr.protocol import MORPH_FINDINGS, validate_manifest


class MorphExpertTests(unittest.TestCase):
    def test_all_experts_are_concept_only_and_shape_stable(self) -> None:
        tokens = torch.randn(2, 16, 8)
        valid = torch.ones(2, 16, dtype=torch.bool)
        statements = torch.tensor([0, 1])
        for expert_type in EXPERT_TYPES:
            if expert_type == "generic":
                continue
            model = MorphologyConceptExpert(
                MorphologyExpertConfig(
                    visual_dim=8,
                    num_statements=2,
                    expert_type=expert_type,
                )
            )
            output = model(tokens, valid, statements, grid_hw=(4, 4))
            self.assertFalse(model.has_flat_four_class_head)
            self.assertEqual(tuple(output["margin"].shape), (2,))
            self.assertEqual(tuple(output["patch_logits"].shape), (2, 16))
            self.assertEqual(tuple(output["concepts"].shape), (2, 7))
            deltas = concept_monotonicity_deltas(
                output["concepts"], output["concept_weights"]
            )
            self.assertTrue(bool((deltas >= 0).all()))

    def test_boundary_and_moments_are_deterministic(self) -> None:
        mask = torch.zeros(16, dtype=torch.bool)
        mask.reshape(4, 4)[1:3, 1:3] = True
        boundary = boundary_from_mask(mask, (4, 4))
        self.assertTrue(torch.equal(boundary, mask))
        moments = mask_moments(mask, (4, 4))
        self.assertTrue(torch.allclose(moments[:2], torch.zeros(2), atol=1e-6))
        self.assertTrue(bool((moments[2:] > 0).all()))


class MorphProtocolTests(unittest.TestCase):
    @staticmethod
    def _rows() -> list[dict]:
        rows = []
        counter = 0
        for split, count in (("train", 3), ("validation", 3)):
            for finding in MORPH_FINDINGS:
                for label in (0, 1):
                    for _ in range(count):
                        counter += 1
                        rows.append(
                            {
                                "sample_id": f"sample-{counter}",
                                "patient_sha256": f"patient-{counter}",
                                "image_sha256": f"image-{counter}",
                                "image_path": f"image-{counter}.jpg",
                                "canonical_statement_id": finding,
                                "binary_label": label,
                                "split": split,
                                "bounding_boxes": (
                                    [{"x_min": 1, "y_min": 1, "x_max": 4, "y_max": 4}]
                                    if label
                                    else []
                                ),
                                "native_columns": 8,
                                "native_rows": 8,
                                "source_dataset": "Chest-ImaGenome-gold-v1.0.0",
                                "source_split": "development_gold",
                            }
                        )
        return rows

    def test_manifest_enforces_global_patient_disjointness(self) -> None:
        rows = self._rows()
        summary = validate_manifest(rows)
        self.assertEqual(summary["records"], 48)
        self.assertEqual(summary["patient_overlap"], 0)
        validation = next(row for row in rows if row["split"] == "validation")
        validation["patient_sha256"] = rows[0]["patient_sha256"]
        with self.assertRaisesRegex(ValueError, "patients overlap"):
            validate_manifest(rows)


class MorphCaseStudyTests(unittest.TestCase):
    def test_case_study_is_identifier_free_and_fail_closed(self) -> None:
        runs = {}
        for expert in ("generic", "region", "boundary", "geometry", "distribution"):
            expert_runs = []
            for seed in (1, 2, 3):
                rows = []
                metrics = {}
                for finding in MORPH_FINDINGS:
                    metrics[finding] = {
                        "auroc": 0.6 if expert == "generic" else 0.7,
                        "spatial_hit": 0.4 if expert == "generic" else 0.5,
                        "boundary_hit": 0.3 if expert == "generic" else 0.4,
                        "geometry_iou": 0.2 if expert == "generic" else 0.3,
                        "moment_score": 0.5 if expert == "generic" else 0.6,
                    }
                    for index, label in enumerate((0, 0, 0, 1, 1, 1)):
                        rows.append(
                            {
                                "sample_id": f"{finding}-{index}",
                                "finding": finding,
                                "label": label,
                                "margin": float(label) + (0.01 if expert == "region" else 0.0),
                            }
                        )
                expert_runs.append(
                    {
                        "seed": seed,
                        "checkpoint_sha256": f"{expert}-{seed}",
                        "events": [{"loss": 1.0}, {"loss": 0.5}],
                        "metrics": metrics,
                        "validation_rows": rows,
                    }
                )
            runs[expert] = expert_runs
        per_finding = {
            "pneumothorax": {
                "prescribed_expert": "boundary",
                "discrimination_gain": 0.1,
                "spatial_gain": 0.1,
                "pass": True,
            },
            "consolidation": {
                "prescribed_expert": "region",
                "discrimination_gain": 0.1,
                "spatial_gain": 0.1,
                "pass": True,
            },
            "pleural_effusion": {
                "prescribed_expert": "region_boundary",
                "discrimination_gain": 0.1,
                "spatial_gain": 0.1,
                "pass": True,
            },
            "cardiomegaly": {
                "prescribed_expert": "geometry",
                "discrimination_gain": 0.1,
                "spatial_gain": 0.1,
                "pass": True,
            },
        }
        result = {
            "schema_version": "morph-separability-result-v1",
            "canonical_sha256": "source-lock",
            "status": "pass_open_full_proposal",
            "gate_pass": True,
            "passed_findings": 4,
            "minimum_passing_findings": 3,
            "per_finding": per_finding,
            "runs": runs,
            "chexlocalize_test_opened": False,
            "vindr_test_opened": False,
            "monotonic_min_delta": 0.0,
            "concept_direct_median_auroc_gap": 0.1,
        }
        analysis = analyze_morph_result(result)
        self.assertEqual(analysis["execution_integrity"]["completed_runs"], 15)
        self.assertEqual(analysis["passed_findings"], 4)
        self.assertNotIn("validation_rows", json.dumps(analysis))
        result["chexlocalize_test_opened"] = True
        with self.assertRaisesRegex(ValueError, "opened test split"):
            analyze_morph_result(result)


if __name__ == "__main__":
    unittest.main()
