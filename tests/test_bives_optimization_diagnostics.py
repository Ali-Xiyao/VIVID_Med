"""Tests for the bounded BiVES optimization-identifiability diagnostics."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from bives_cxr.losses import (
    BiVESLoss,
    BiVESLossConfig,
    requires_intervention_branches,
)
from bives_cxr.model import BiVESCXR, BiVESModelConfig
from bives_cxr.optimization_audit import (
    aggregate_optimization_audits,
    fixed_batch_optimization_audit,
    summarize_clipping_history,
    summarize_prediction_evidence,
    weighted_loss_components,
)
from scripts.diagnose_bives_proxy_sc import logistic_probe_metrics


class OptimizationAuditTests(unittest.TestCase):
    def test_intervention_requirement_follows_effective_ies_weight(self) -> None:
        self.assertFalse(
            requires_intervention_branches(BiVESLossConfig(lambda_ies=0.0))
        )
        self.assertFalse(
            requires_intervention_branches(
                BiVESLossConfig(lambda_ies=1.0),
                auxiliary_weight=0.0,
            )
        )
        self.assertTrue(
            requires_intervention_branches(BiVESLossConfig(lambda_ies=0.5))
        )

    def test_fixed_batch_audit_records_gradients_and_correct_state_direction(self) -> None:
        torch.manual_seed(9)
        model = BiVESCXR(
            BiVESModelConfig(
                visual_dim=8,
                statement_dim=6,
                fusion_dim=16,
                topk=2,
                contextual_heads=4,
                num_controls=2,
            )
        )
        targets = torch.tensor([0, 1, 2, 3])
        outputs = model(
            torch.randn(4, 8, 8),
            torch.randn(4, 6),
            torch.ones(4, 8, dtype=torch.bool),
            control_seeds=torch.tensor([1, 2, 3, 4]),
        )
        loss_config = BiVESLossConfig(lambda_pair=0.1, lambda_u_pol=0.1)
        losses = BiVESLoss(loss_config)(
            outputs,
            targets,
            (2, 4),
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([2]),
        )
        batch = {
            "targets": targets,
            "sample_ids": [f"sample-{index}" for index in range(4)],
            "patient_ids": [f"patient-{index}" for index in range(4)],
            "canonical_statement_ids": ["finding-a"] * 4,
        }
        payload = fixed_batch_optimization_audit(
            model,
            batch,
            outputs,
            losses,
            loss_config,
            step=0,
            auxiliary_weight=1.0,
        )
        self.assertIn("state", payload["gradient_norms"])
        self.assertGreater(payload["gradient_norms"]["state"]["all"], 0.0)
        directions = payload["state_nll_signed_evidence_direction"]
        self.assertLess(directions["support"]["mean_gradient_wrt_signed_evidence"], 0.0)
        self.assertGreater(directions["contradict"]["mean_gradient_wrt_signed_evidence"], 0.0)

    def test_weighted_components_follow_configured_training_objective(self) -> None:
        base = torch.tensor(1.0, requires_grad=True)
        losses = {
            "state": base,
            "ies": base * 2,
            "pair": base * 3,
            "uncertain_polarity": base * 4,
            "insufficient_magnitude": base * 5,
            "evidence_fraction": base * 6,
            "tv": base * 7,
            "total": base * 8,
        }
        config = BiVESLossConfig(
            lambda_ies=0.5,
            lambda_pair=0.1,
            lambda_u_pol=0.2,
            lambda_i_mag=0.3,
            lambda_min=0.0,
            lambda_tv=0.01,
        )
        components = weighted_loss_components(losses, config, auxiliary_weight=0.5)
        self.assertEqual(
            set(components),
            {"state", "ies", "pair", "uncertain_polarity", "insufficient_magnitude", "tv", "total"},
        )
        self.assertAlmostEqual(float(components["ies"]), 0.5)
        self.assertAlmostEqual(float(components["pair"]), 0.15)

    def test_prediction_evidence_summary_preserves_state_finding_groups(self) -> None:
        rows = [
            {
                "target": 0,
                "canonical_statement_id": "finding-a",
                "evidence_pos": 3.0,
                "evidence_neg": 1.0,
                "original_probs": [0.7, 0.1, 0.1, 0.1],
            },
            {
                "target": 0,
                "canonical_statement_id": "finding-a",
                "evidence_pos": 5.0,
                "evidence_neg": 1.0,
                "original_probs": [0.8, 0.05, 0.1, 0.05],
            },
        ]
        summary = summarize_prediction_evidence(rows)
        group = summary["groups"]["finding-a|support"]
        self.assertEqual(group["count"], 2)
        self.assertEqual(group["mean_evidence_pos"], 4.0)
        self.assertEqual(group["mean_signed_evidence"], 3.0)

    def test_audit_aggregation_preserves_all_quartets(self) -> None:
        audit = {
            "gradient_norms": {"state": {"all": 2.0, "fusion": 1.0}},
            "state_vs_auxiliary_gradient_cosines": {"tv": 0.25},
            "state_nll_signed_evidence_direction": {
                state: {
                    "count": 1,
                    "mean_gradient_wrt_signed_evidence": float(index),
                }
                for index, state in enumerate(("support", "contradict", "uncertain", "insufficient"))
            },
            "samples": [{"sample_id": "a"}],
        }
        second = {
            **audit,
            "gradient_norms": {"state": {"all": 4.0, "fusion": 3.0}},
            "samples": [{"sample_id": "b"}],
        }
        payload = aggregate_optimization_audits([audit, second])
        self.assertEqual(payload["audit_scope"], "all_train_quartets")
        self.assertEqual(payload["quartet_count"], 2)
        self.assertEqual(payload["sample_count"], 2)
        self.assertEqual(
            payload["gradient_norm_summary"]["state"]["all"]["mean"],
            3.0,
        )

    def test_clipping_summary_records_frequency_and_preclip_norms(self) -> None:
        summary = summarize_clipping_history(
            [
                {
                    "preclip_total_norm": 0.5,
                    "preclip_group_norms": {"all": 0.5, "fusion": 0.4},
                    "clip_coefficient": 1.0,
                    "clipped": False,
                },
                {
                    "preclip_total_norm": 2.0,
                    "preclip_group_norms": {"all": 2.0, "fusion": 1.5},
                    "clip_coefficient": 0.5,
                    "clipped": True,
                },
            ]
        )
        self.assertEqual(summary["optimizer_steps"], 2)
        self.assertEqual(summary["clipped_steps"], 1)
        self.assertEqual(summary["clipped_fraction"], 0.5)
        self.assertEqual(summary["preclip_total_norm"]["maximum"], 2.0)


class LogisticProbeTests(unittest.TestCase):
    def test_probe_is_patient_group_disjoint_and_reports_calibration(self) -> None:
        rng = np.random.default_rng(17)
        labels = np.asarray([0, 1] * 10, dtype=np.int64)
        patient_ids = np.asarray([f"patient-{index:02d}" for index in range(20)])
        features = rng.normal(size=(20, 4)).astype(np.float32)
        features[:, 0] += labels * 6.0
        metrics = logistic_probe_metrics(features, labels, patient_ids, seed=17)
        self.assertEqual(metrics["folds"], 5)
        self.assertGreater(metrics["auroc"], 0.8)
        self.assertIn("full_fit_intercept", metrics)
        self.assertIn("ece_10bin", metrics)
        self.assertEqual(sum(row["test_records"] for row in metrics["fold_details"]), 20)


if __name__ == "__main__":
    unittest.main()
