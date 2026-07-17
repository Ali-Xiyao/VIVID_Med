"""CPU tests for the proposal-level BiVES-CXR contracts."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from bives_cxr.backbones import PatchBatch, Qwen35VisionAdapter, restore_qwen35_row_major
from bives_cxr.calibration import fit_decoder_parameters, probabilities_from_evidence
from bives_cxr.gates import EvidenceGate
from bives_cxr.decoder import EvidenceStateDecoder
from bives_cxr.interventions import (
    build_matched_control_masks,
    delete_evidence,
    retain_evidence,
    stable_control_seed,
)
from bives_cxr.losses import (
    BiVESLoss,
    BiVESLossConfig,
    jensen_shannon,
    nll_from_probs,
    total_variation,
)
from bives_cxr.metrics import (
    classification_metrics,
    finalize_intervention_metrics,
    intervention_metric_counts,
    patient_bootstrap_confidence_intervals,
)
from bives_cxr.model import BiVESCXR, BiVESModelConfig
from bives_cxr.statement_cache import (
    build_statement_cache_payload,
    load_statement_embedding_matrix,
)
from scripts.train_bives_cxr import BiVESExperiment, load_checkpoint_model_state
from legacy.bives_cxr.legacy_abs_exp_decoder import legacy_abs_exp_probabilities


class DecoderTests(unittest.TestCase):
    def test_probabilities_match_closed_form_and_normalize(self) -> None:
        decoder = EvidenceStateDecoder(tau_a=2.0, tau_p=4.0, uncertainty_mass=1.5)
        positive = torch.tensor([0.0, 8.0, 1.0, 5.0])
        negative = torch.tensor([0.0, 1.0, 8.0, 5.0])
        output = decoder(positive, negative)
        availability = 1.0 - torch.exp(-(positive + negative) / 2.0)
        signed_logit = (positive - negative) / 8.0
        conditional = torch.softmax(
            torch.stack(
                (signed_logit, -signed_logit, torch.log(torch.tensor(3.0)).expand_as(signed_logit)),
                dim=-1,
            ),
            dim=-1,
        )
        expected = torch.stack(
            (
                availability * conditional[:, 0],
                availability * conditional[:, 1],
                availability * conditional[:, 2],
                1.0 - availability,
            ),
            dim=-1,
        )
        self.assertTrue(torch.allclose(output["state_probs"], expected))
        self.assertTrue(torch.allclose(output["state_probs"].sum(dim=-1), torch.ones(4)))

    def test_state_semantics_and_polarity_symmetry(self) -> None:
        decoder = EvidenceStateDecoder()
        positive = torch.tensor([0.0, 9.0, 1.0, 8.0])
        negative = torch.tensor([0.0, 1.0, 9.0, 8.0])
        probabilities = decoder(positive, negative)["state_probs"]
        self.assertEqual(int(probabilities[0].argmax()), 3)
        self.assertEqual(int(probabilities[1].argmax()), 0)
        self.assertEqual(int(probabilities[2].argmax()), 1)
        self.assertEqual(int(probabilities[3].argmax()), 2)
        swapped = decoder(negative, positive)["state_probs"]
        self.assertTrue(torch.allclose(probabilities[:, 0], swapped[:, 1], atol=1e-7, rtol=1e-6))
        self.assertTrue(torch.allclose(probabilities[:, 1], swapped[:, 0], atol=1e-7, rtol=1e-6))
        self.assertTrue(torch.allclose(probabilities[:, 2:], swapped[:, 2:], atol=1e-7, rtol=1e-6))

    def test_support_and_contradict_are_strictly_monotone_in_signed_evidence(self) -> None:
        decoder = EvidenceStateDecoder().double()
        delta = torch.linspace(-6.0, 6.0, 1001, dtype=torch.float64)
        positive = 4.0 + delta / 2.0
        negative = 4.0 - delta / 2.0
        probabilities = decoder(positive, negative)["state_probs"]
        self.assertTrue(bool((torch.diff(probabilities[:, 0]) > 0).all()))
        self.assertTrue(bool((torch.diff(probabilities[:, 1]) < 0).all()))
        self.assertTrue(torch.allclose(probabilities[:, 2], probabilities.flip(0)[:, 2]))
        self.assertEqual(int(probabilities[:, 2].argmax()), delta.numel() // 2)

    def test_state_nll_gradients_push_polarity_in_the_correct_direction(self) -> None:
        support_delta = torch.linspace(-6.0, -0.01, 101, dtype=torch.float64).requires_grad_()
        support_probs = EvidenceStateDecoder().double()(
            4.0 + support_delta / 2.0,
            4.0 - support_delta / 2.0,
        )["state_probs"]
        (-support_probs[:, 0].log().sum()).backward()
        self.assertTrue(bool((support_delta.grad < 0.0).all()))

        contradict_delta = torch.linspace(0.01, 6.0, 101, dtype=torch.float64).requires_grad_()
        contradict_probs = EvidenceStateDecoder().double()(
            4.0 + contradict_delta / 2.0,
            4.0 - contradict_delta / 2.0,
        )["state_probs"]
        (-contradict_probs[:, 1].log().sum()).backward()
        self.assertTrue(bool((contradict_delta.grad > 0.0).all()))

    def test_legacy_decoder_has_the_documented_wrong_half_axis_stationary_point(self) -> None:
        delta_star = -torch.asinh(torch.tensor(1.0, dtype=torch.float64))
        delta = delta_star.detach().clone().requires_grad_()
        probabilities = legacy_abs_exp_probabilities(
            4.0 + delta / 2.0,
            4.0 - delta / 2.0,
        )
        (-probabilities[0].log()).backward()
        self.assertLess(float(delta), 0.0)
        self.assertAlmostEqual(float(delta.grad), 0.0, places=8)

    def test_uncertain_and_insufficient_are_operationally_distinct(self) -> None:
        decoder = EvidenceStateDecoder()
        low_total = decoder(torch.tensor([0.1]), torch.tensor([0.1]))["state_probs"]
        high_total = decoder(torch.tensor([5.0]), torch.tensor([5.0]))["state_probs"]
        decisive = decoder(torch.tensor([9.0]), torch.tensor([1.0]))["state_probs"]
        less_decisive = decoder(torch.tensor([6.0]), torch.tensor([4.0]))["state_probs"]
        self.assertGreater(float(low_total[0, 3]), float(high_total[0, 3]))
        self.assertGreater(float(high_total[0, 2]), float(low_total[0, 2]))
        self.assertGreater(float(less_decisive[0, 2]), float(decisive[0, 2]))

    def test_insufficient_nll_has_evidence_magnitude_gradient(self) -> None:
        positive = torch.tensor([0.7], requires_grad=True)
        negative = torch.tensor([0.2], requires_grad=True)
        probabilities = EvidenceStateDecoder(tau_a=2.0)(positive, negative)["state_probs"]
        loss = nll_from_probs(probabilities, torch.tensor([3])).sum()
        loss.backward()
        self.assertAlmostEqual(float(positive.grad), 0.5, places=5)
        self.assertAlmostEqual(float(negative.grad), 0.5, places=5)


class InterventionTests(unittest.TestCase):
    def test_keep_drop_algebra(self) -> None:
        tokens = torch.randn(2, 4, 3)
        mask = torch.tensor([[1.0, 0.0, 0.5, 1.0], [0.0, 1.0, 0.25, 0.75]])
        keep = retain_evidence(tokens, mask)
        drop = delete_evidence(tokens, mask)
        self.assertTrue(torch.allclose(keep + drop, tokens))
        self.assertEqual(tuple(keep.shape), (2, 4, 3))

    def test_exact_k_gate_and_random_disjoint_controls(self) -> None:
        logits = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0, -1.0]])
        valid = torch.ones_like(logits, dtype=torch.bool)
        gate = EvidenceGate(mode="soft_topk", topk=2)(logits, valid)
        self.assertEqual(int((gate.detach() > 0.5).sum()), 2)
        controls = build_matched_control_masks(
            (gate.detach() > 0.5).float(),
            valid,
            topk=2,
            num_controls=4,
            generator=torch.Generator().manual_seed(7),
        )
        self.assertEqual(tuple(controls.shape), (1, 4, 6))
        self.assertTrue(bool((controls.sum(dim=-1) == 2).all()))
        self.assertFalse(bool(((controls > 0.5) & (gate.detach() > 0.5).unsqueeze(1)).any()))

    def test_seeded_controls_are_reproducible_and_sample_specific(self) -> None:
        evidence = torch.tensor(
            [[1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]],
            dtype=torch.float32,
        )
        valid = torch.ones_like(evidence, dtype=torch.bool)
        seeds = [
            stable_control_seed("val", "sample-a"),
            stable_control_seed("val", "sample-b"),
        ]
        first = build_matched_control_masks(
            evidence,
            valid,
            topk=2,
            num_controls=4,
            sample_seeds=seeds,
        )
        second = build_matched_control_masks(
            evidence,
            valid,
            topk=2,
            num_controls=4,
            sample_seeds=seeds,
        )
        self.assertTrue(torch.equal(first, second))
        changed = build_matched_control_masks(
            evidence,
            valid,
            topk=2,
            num_controls=4,
            sample_seeds=[seeds[0] + 1, seeds[1] + 1],
        )
        self.assertFalse(torch.equal(first, changed))

    def test_total_variation(self) -> None:
        constant = torch.ones(1, 9)
        self.assertEqual(float(total_variation(constant, (3, 3))), 0.0)
        center = torch.zeros(1, 9)
        center[0, 4] = 1.0
        self.assertEqual(float(total_variation(center, (3, 3))), 4.0)
        with self.assertRaises(ValueError):
            total_variation(center, (2, 4))


class ModelContractTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.model = BiVESCXR(
            BiVESModelConfig(
                visual_dim=8,
                statement_dim=6,
                fusion_dim=16,
                gate_mode="sigmoid",
                topk=2,
            )
        )

    def test_architecture_has_no_flat_state_head(self) -> None:
        self.assertEqual(self.model.decoder_kind, "monotone_bipolar_conditional")
        self.assertFalse(self.model.has_flat_state_head)
        self.assertFalse(hasattr(self.model, "state_head"))
        self.assertFalse(hasattr(self.model, "four_class_head"))
        self.assertFalse(hasattr(self.model, "classifier"))
        self.assertEqual(sum(parameter.numel() for parameter in self.model.decoder.parameters()), 0)

    def test_forward_loss_backward_and_optimizer_step(self) -> None:
        patches = torch.randn(4, 4, 8)
        statements = torch.randn(4, 6)
        valid = torch.ones(4, 4, dtype=torch.bool)
        targets = torch.tensor([0, 1, 2, 3])
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        before = self.model.evidence_head.weight.detach().clone()
        outputs = self.model(patches, statements, valid)
        original = outputs["original"]
        self.assertEqual(tuple(original["state_probs"].shape), (4, 4))
        self.assertEqual(tuple(original["evidence_maps"].shape), (4, 4, 2))
        self.assertTrue(bool((original["evidence_pm"] >= 0).all()))
        self.assertTrue(bool(((original["gate"] >= 0) & (original["gate"] <= 1)).all()))
        self.assertTrue(bool((outputs["evidence_hard_mask"].sum(dim=-1) == 2).all()))
        self.assertTrue(
            torch.equal(outputs["keep_valid_mask"], outputs["evidence_hard_mask"])
        )
        self.assertFalse(
            bool((outputs["drop_valid_mask"] & outputs["evidence_hard_mask"]).any())
        )
        self.assertTrue(bool((outputs["control_masks"].sum(dim=-1) == 2).all()))
        self.assertFalse(
            bool(
                (
                    (outputs["control_masks"] > 0.5)
                    & outputs["evidence_hard_mask"].unsqueeze(1)
                ).any()
            )
        )
        losses = BiVESLoss()(outputs, targets, (2, 2))
        self.assertTrue(bool(torch.isfinite(losses["total"])))
        losses["total"].backward()
        optimizer.step()
        self.assertFalse(torch.allclose(before, self.model.evidence_head.weight.detach()))
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_pair_and_uncertain_losses_are_mandatory_when_enabled(self) -> None:
        patches = torch.randn(4, 8, 8)
        statements = torch.randn(4, 6)
        valid = torch.ones(4, 8, dtype=torch.bool)
        targets = torch.tensor([0, 1, 2, 3])
        model = BiVESCXR(
            BiVESModelConfig(
                visual_dim=8,
                statement_dim=6,
                fusion_dim=16,
                gate_mode="soft_topk",
                topk=2,
                num_controls=2,
            )
        )
        outputs = model(patches, statements, valid)
        loss_fn = BiVESLoss(BiVESLossConfig(lambda_pair=0.1, lambda_u_pol=0.1))
        with self.assertRaises(ValueError):
            loss_fn(outputs, targets, (2, 4))
        losses = loss_fn(
            outputs,
            targets,
            (2, 4),
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([2]),
        )
        self.assertIn("pair", losses)
        self.assertIn("uncertain_polarity", losses)

    def test_zero_auxiliary_weight_keeps_only_state_objective(self) -> None:
        patches = torch.randn(4, 8, 8)
        statements = torch.randn(4, 6)
        valid = torch.ones(4, 8, dtype=torch.bool)
        targets = torch.tensor([0, 1, 2, 3])
        outputs = self.model(patches[:, :4], statements, valid[:, :4])
        loss_fn = BiVESLoss(BiVESLossConfig(lambda_pair=0.1, lambda_u_pol=0.1))
        losses = loss_fn(
            outputs,
            targets,
            (2, 2),
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([2]),
            auxiliary_weight=0.0,
        )
        self.assertEqual(float(losses["auxiliary_weight"]), 0.0)
        self.assertTrue(torch.allclose(losses["total"], losses["state"]))

    def test_contextual_interventions_are_not_algebraic_identities(self) -> None:
        torch.manual_seed(29)
        model = BiVESCXR(
            BiVESModelConfig(
                visual_dim=8,
                statement_dim=6,
                fusion_dim=16,
                gate_mode="soft_topk",
                topk=2,
                num_controls=3,
                contextual_layers=1,
                contextual_heads=4,
                contextual_dropout=0.0,
            )
        ).eval()
        outputs = model(
            torch.randn(4, 8, 8),
            torch.randn(4, 6),
            torch.ones(4, 8, dtype=torch.bool),
        )
        original_probs = outputs["original"]["state_probs"]
        keep_difference = (outputs["keep"]["state_probs"] - original_probs).abs().max()
        control_differences = torch.stack(
            [
                (branch["state_probs"] - original_probs).abs().max()
                for branch in outputs["controls"]
            ]
        )
        self.assertGreater(float(keep_difference.detach()), 1e-7)
        self.assertGreater(float(control_differences.max().detach()), 1e-7)

        control_loss = torch.stack(
            [
                jensen_shannon(original_probs, branch["state_probs"]).mean()
                for branch in outputs["controls"]
            ]
        ).mean()
        self.assertGreater(float(control_loss.detach()), 0.0)
        model.zero_grad(set_to_none=True)
        control_loss.backward()
        gradient = model.contextual_evidence.layers[0].self_attn.in_proj_weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)


class BackboneContractTests(unittest.TestCase):
    def test_inverse_unshuffle_restores_row_major(self) -> None:
        block_major = torch.tensor(
            [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15],
            dtype=torch.float32,
        ).unsqueeze(-1)
        restored = restore_qwen35_row_major(block_major, 4, 4, 2)
        self.assertEqual(restored.squeeze(-1).tolist(), list(range(16)))

    def test_qwen35_adapter_isolates_each_packed_image_forward(self) -> None:
        class PackedSensitiveVisual(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.patch_embed = nn.Identity()
                self.blocks = nn.ModuleList()
                self.merger = nn.Identity()
                self.calls: list[tuple[int, tuple[int, ...]]] = []

            def forward(
                self,
                pixel_values: torch.Tensor,
                *,
                grid_thw: torch.Tensor,
                return_dict: bool,
            ) -> torch.Tensor:
                self.calls.append((int(pixel_values.shape[0]), tuple(grid_thw.shape)))
                self.assert_single_grid(grid_thw)
                self.assert_return_dict(return_dict)
                return pixel_values

            @staticmethod
            def assert_single_grid(grid_thw: torch.Tensor) -> None:
                if tuple(grid_thw.shape) != (1, 3):
                    raise AssertionError(f"expected one image grid, got {tuple(grid_thw.shape)}")

            @staticmethod
            def assert_return_dict(return_dict: bool) -> None:
                if not return_dict:
                    raise AssertionError("adapter must request structured visual output")

        visual = PackedSensitiveVisual()
        adapter = Qwen35VisionAdapter(visual, spatial_merge_size=1)
        pixels = torch.arange(24, dtype=torch.float32).view(8, 3)
        grid = torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.long)
        output = adapter(pixels, grid)
        self.assertEqual(visual.calls, [(4, (1, 3)), (4, (1, 3))])
        self.assertTrue(torch.equal(output.tokens[0], pixels[:4]))
        self.assertTrue(torch.equal(output.tokens[1], pixels[4:]))
        self.assertTrue(bool(output.valid_mask.all()))
        self.assertEqual(output.grid_hw, [(2, 2), (2, 2)])

    def test_bf16_backbone_tokens_are_cast_to_fp32_head(self) -> None:
        class DummyBackbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros((), dtype=torch.bfloat16), requires_grad=False)

            def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> PatchBatch:
                del image_grid_thw
                if pixel_values.dtype != torch.bfloat16:
                    raise AssertionError(f"expected BF16 backbone input, got {pixel_values.dtype}")
                return PatchBatch(
                    tokens=torch.randn(4, 8, 8, dtype=torch.bfloat16),
                    valid_mask=torch.ones(4, 8, dtype=torch.bool),
                    grid_hw=[(2, 4)] * 4,
                )

        experiment = BiVESExperiment(
            DummyBackbone(),
            num_statements=1,
            statement_dim=6,
            head_config=BiVESModelConfig(
                visual_dim=8,
                statement_dim=6,
                fusion_dim=16,
                gate_mode="soft_topk",
                topk=2,
                num_controls=1,
            ),
        )
        outputs, _ = experiment(
            torch.empty(0, dtype=torch.float32),
            torch.empty(0, dtype=torch.long),
            torch.zeros(4, dtype=torch.long),
            torch.ones(4, 8, dtype=torch.bool),
        )
        self.assertEqual(outputs["original"]["state_probs"].dtype, torch.float32)

    def test_validation_selected_checkpoint_state_is_reloaded(self) -> None:
        class DummyBackbone(nn.Module):
            def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> PatchBatch:
                del pixel_values, image_grid_thw
                return PatchBatch(
                    tokens=torch.zeros(4, 8, 8),
                    valid_mask=torch.ones(4, 8, dtype=torch.bool),
                    grid_hw=[(2, 4)] * 4,
                )

        experiment = BiVESExperiment(
            DummyBackbone(),
            num_statements=1,
            statement_dim=6,
            head_config=BiVESModelConfig(
                visual_dim=8,
                statement_dim=6,
                fusion_dim=16,
                gate_mode="soft_topk",
                topk=2,
                num_controls=1,
            ),
        )
        with torch.no_grad():
            experiment.head.evidence_head.bias.fill_(1.0)
        selected = {
            "statement_table": {
                key: value.clone()
                for key, value in experiment.statement_table.state_dict().items()
            },
            "bives_head": {
                key: value.clone()
                for key, value in experiment.head.state_dict().items()
            },
        }
        with torch.no_grad():
            experiment.head.evidence_head.bias.fill_(2.0)
        load_checkpoint_model_state(experiment, selected)
        self.assertTrue(
            torch.equal(
                experiment.head.evidence_head.bias,
                torch.ones_like(experiment.head.evidence_head.bias),
            )
        )


class MetricContractTests(unittest.TestCase):
    def test_eos_and_eri_condition_on_original_correct(self) -> None:
        def branch(predictions: list[int]) -> dict[str, torch.Tensor]:
            probs = torch.zeros(3, 4)
            probs[torch.arange(3), torch.tensor(predictions)] = 1.0
            return {"state_probs": probs}

        outputs = {
            "original": branch([0, 0, 2]),
            "keep": branch([0, 1, 1]),
            "drop": branch([3, 3, 3]),
            "control": branch([0, 0, 2]),
        }
        targets = torch.tensor([0, 1, 2])
        metrics = finalize_intervention_metrics(intervention_metric_counts(outputs, targets))
        self.assertEqual(metrics["evidence_only_sufficiency"], 0.5)
        self.assertEqual(metrics["evidence_removal_insufficient"], 1.0)

    def test_eos_excludes_insufficient_targets(self) -> None:
        def branch(predictions: list[int]) -> dict[str, torch.Tensor]:
            probs = torch.zeros(len(predictions), 4)
            probs[torch.arange(len(predictions)), torch.tensor(predictions)] = 1.0
            return {"state_probs": probs}

        outputs = {
            "original": branch([0, 3]),
            "keep": branch([0, 0]),
            "drop": branch([3, 3]),
            "control": branch([0, 3]),
            "controls": [branch([0, 3]), branch([0, 0])],
        }
        counts = intervention_metric_counts(outputs, torch.tensor([0, 3]))
        self.assertEqual(counts["eos_denominator"], 1.0)
        metrics = finalize_intervention_metrics(counts)
        self.assertEqual(metrics["evidence_only_sufficiency"], 1.0)
        self.assertEqual(metrics["irrelevant_stability_worst_case"], 0.5)
        self.assertEqual(metrics["irrelevant_stability_eligible_worst_case"], 1.0)

    def test_primary_and_calibration_metrics(self) -> None:
        probabilities = torch.tensor(
            [
                [0.8, 0.1, 0.05, 0.05],
                [0.1, 0.8, 0.05, 0.05],
                [0.05, 0.05, 0.8, 0.1],
                [0.05, 0.05, 0.1, 0.8],
            ]
        )
        metrics = classification_metrics(probabilities, torch.arange(4))
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(
            metrics["confusion_matrix"],
            torch.eye(4, dtype=torch.int64).tolist(),
        )
        self.assertIn("aurc", metrics)

    def test_patient_bootstrap_uses_fixed_four_class_contract(self) -> None:
        probabilities = torch.eye(4).repeat(2, 1).numpy()
        targets = torch.arange(4).repeat(2).numpy()
        patient_ids = [f"patient-{index}" for index in range(8)]
        result = patient_bootstrap_confidence_intervals(
            probabilities,
            targets,
            patient_ids,
            replicates=50,
            seed=5,
        )
        self.assertEqual(result["_metadata"]["class_labels"], [0, 1, 2, 3])
        self.assertEqual(result["macro_f1"]["requested_replicates"], 50)
        self.assertLessEqual(result["balanced_accuracy"]["valid_replicates"], 50)
        self.assertIn("support_vs_contradict_auroc", result)
        self.assertIn("uncertain_vs_insufficient_auprc", result)

    def test_patient_bootstrap_includes_intervention_endpoints(self) -> None:
        probabilities = torch.eye(4).repeat(2, 1).numpy()
        targets = torch.arange(4).repeat(2).numpy()
        rows = []
        for index, target in enumerate(targets):
            original = probabilities[index].tolist()
            keep = list(original)
            drop = [0.0, 0.0, 0.0, 1.0]
            controls = [list(original), list(original)]
            rows.append(
                {
                    "target": int(target),
                    "original_probs": original,
                    "keep_probs": keep,
                    "drop_probs": drop,
                    "control_branch_probs": controls,
                }
            )
        result = patient_bootstrap_confidence_intervals(
            probabilities,
            targets,
            [f"patient-{index}" for index in range(8)],
            replicates=50,
            seed=7,
            prediction_rows=rows,
        )
        self.assertIn("evidence_only_sufficiency", result)
        self.assertIn("evidence_removal_insufficient", result)
        self.assertIn("target_control_gap_eligible", result)
        self.assertIn("target_control_gap_eligible_worst_case", result)


class StatementCacheTests(unittest.TestCase):
    def test_cache_binds_embeddings_to_exact_normalized_text(self) -> None:
        texts = {"stmt-a": "  Right   effusion PRESENT. "}
        payload = build_statement_cache_payload(
            {"stmt-a": torch.tensor([3.0, 4.0]) / 5.0},
            texts,
            {
                "model_name_or_path": "Qwen3.5-4B",
                "revision": "abc",
                "tokenizer_revision": "abc",
                "tokenizer_class": "DummyTokenizer",
                "pooling": "input_embedding_mean",
                "normalize": True,
                "dtype": "float32",
            },
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cache.pt"
            torch.save(payload, path)
            matrix = load_statement_embedding_matrix(
                path,
                {"stmt-a": 0},
                {"stmt-a": "right effusion present."},
            )
            self.assertEqual(tuple(matrix.shape), (1, 2))
            with self.assertRaises(ValueError):
                load_statement_embedding_matrix(
                    path,
                    {"stmt-a": 0},
                    {"stmt-a": "left effusion present."},
                )

    def test_decoder_parameter_fit_is_positive_and_improves_nll(self) -> None:
        positive = torch.tensor([8.0, 1.0, 5.0, 0.1] * 4)
        negative = torch.tensor([1.0, 8.0, 5.0, 0.1] * 4)
        targets = torch.tensor([0, 1, 2, 3] * 4)
        initial = (5.0, 5.0, 5.0)
        before = nll_from_probs(
            probabilities_from_evidence(
                positive,
                negative,
                *(torch.tensor(value) for value in initial),
            ),
            targets,
        ).mean()
        fitted = fit_decoder_parameters(
            positive,
            negative,
            targets,
            initial=initial,
            max_iter=50,
        )
        after = nll_from_probs(
            probabilities_from_evidence(
                positive,
                negative,
                torch.tensor(fitted["tau_a"]),
                torch.tensor(fitted["tau_p"]),
                torch.tensor(fitted["uncertainty_mass"]),
            ),
            targets,
        ).mean()
        self.assertTrue(all(value > 0 for value in fitted.values()))
        self.assertLess(float(after), float(before))


if __name__ == "__main__":
    unittest.main()
