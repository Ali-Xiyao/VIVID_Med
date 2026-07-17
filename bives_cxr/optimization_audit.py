"""Optimization-identifiability diagnostics for bounded local BiVES runs."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Iterable

import torch

from .decoder import STATE_NAMES
from .losses import BiVESLossConfig, nll_from_probs


def _parameter_group(name: str) -> str:
    if name.startswith("statement_table."):
        return "statement_table"
    if name.startswith(
        (
            "head.visual_projection.",
            "head.statement_projection.",
            "head.fusion.",
        )
    ):
        return "fusion"
    if name.startswith(("head.contextual_evidence.", "head.contextual_norm.")):
        return "context"
    if name.startswith("head.evidence_head."):
        return "evidence_head"
    if name.startswith("head.gate_head."):
        return "gate_head"
    return "other"


def _gradient_vector(
    value: torch.Tensor,
    parameters: list[tuple[str, torch.nn.Parameter]],
    *,
    retain_graph: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    gradients = torch.autograd.grad(
        value,
        [parameter for _, parameter in parameters],
        retain_graph=retain_graph,
        allow_unused=True,
    )
    vectors: list[torch.Tensor] = []
    squared_by_group: dict[str, float] = defaultdict(float)
    for (name, parameter), gradient in zip(parameters, gradients):
        if gradient is None:
            vector = torch.zeros_like(parameter, memory_format=torch.preserve_format).reshape(-1)
        else:
            vector = gradient.detach().float().reshape(-1)
        vectors.append(vector)
        squared_by_group[_parameter_group(name)] += float(torch.dot(vector, vector).cpu())
    full = torch.cat(vectors) if vectors else value.new_zeros((0,), dtype=torch.float32)
    norms = {
        group: math.sqrt(max(0.0, squared))
        for group, squared in sorted(squared_by_group.items())
    }
    norms["all"] = float(torch.linalg.vector_norm(full).cpu()) if full.numel() else 0.0
    return full, norms


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 0.0:
        return None
    return float(torch.dot(left, right).div(denominator).cpu())


def weighted_loss_components(
    losses: dict[str, torch.Tensor],
    config: BiVESLossConfig,
    auxiliary_weight: float,
) -> dict[str, torch.Tensor]:
    """Return the exact independently differentiable terms used by the objective."""

    auxiliary = float(auxiliary_weight)
    components = {"state": losses["state"]}
    weighted = (
        ("ies", config.lambda_ies),
        ("pair", config.lambda_pair),
        ("uncertain_polarity", config.lambda_u_pol),
        ("insufficient_magnitude", config.lambda_i_mag),
        ("evidence_fraction", config.lambda_min),
        ("tv", config.lambda_tv),
    )
    for name, coefficient in weighted:
        if name in losses and float(coefficient) != 0.0 and auxiliary != 0.0:
            components[name] = losses[name] * (auxiliary * float(coefficient))
    components["total"] = losses["total"]
    return components


def fixed_batch_optimization_audit(
    experiment: torch.nn.Module,
    batch: dict[str, Any],
    outputs: dict[str, Any],
    losses: dict[str, torch.Tensor],
    loss_config: BiVESLossConfig,
    *,
    step: int,
    auxiliary_weight: float,
) -> dict[str, Any]:
    """Audit evidence, dense selector logits, and per-loss gradient geometry."""

    parameters = [
        (name, parameter)
        for name, parameter in experiment.named_parameters()
        if parameter.requires_grad
    ]
    original = outputs["original"]
    targets = batch["targets"].long()
    per_sample_nll = nll_from_probs(original["state_probs"], targets, loss_config.eps)
    decoder = (
        experiment.head.decoder
        if hasattr(experiment, "head")
        else experiment.decoder
    )
    detached_pos = original["evidence_pos"].detach().float().requires_grad_(True)
    detached_neg = original["evidence_neg"].detach().float().requires_grad_(True)
    decoder_probabilities = decoder(detached_pos, detached_neg)["state_probs"]
    decoder_nll = nll_from_probs(decoder_probabilities, targets, loss_config.eps)
    gradient_pos, gradient_neg = torch.autograd.grad(
        decoder_nll.sum(),
        (detached_pos, detached_neg),
        allow_unused=False,
    )
    delta_gradients = 0.5 * (gradient_pos - gradient_neg)

    components = weighted_loss_components(losses, loss_config, auxiliary_weight)
    gradient_vectors: dict[str, torch.Tensor] = {}
    gradient_norms: dict[str, dict[str, float]] = {}
    component_items = list(components.items())
    for index, (name, value) in enumerate(component_items):
        vector, norms = _gradient_vector(
            value,
            parameters,
            retain_graph=index + 1 < len(component_items),
        )
        gradient_vectors[name] = vector
        gradient_norms[name] = norms

    state_vector = gradient_vectors["state"]
    gradient_cosines = {
        name: _cosine(state_vector, vector)
        for name, vector in gradient_vectors.items()
        if name not in {"state", "total"}
    }

    rows: list[dict[str, Any]] = []
    for index, sample_id in enumerate(batch["sample_ids"]):
        valid = original["valid_mask"][index].bool()
        hard_indices = torch.where(outputs["evidence_hard_mask"][index])[0]
        evidence_pos = float(original["evidence_pos"][index].detach().float().cpu())
        evidence_neg = float(original["evidence_neg"][index].detach().float().cpu())
        total = evidence_pos + evidence_neg
        rows.append(
            {
                "sample_id": str(sample_id),
                "patient_id": str(batch["patient_ids"][index]),
                "canonical_statement_id": str(batch["canonical_statement_ids"][index]),
                "target": int(targets[index].detach().cpu()),
                "state": STATE_NAMES[int(targets[index].detach().cpu())],
                "evidence_pos": evidence_pos,
                "evidence_neg": evidence_neg,
                "total_evidence": total,
                "signed_evidence": evidence_pos - evidence_neg,
                "rho": (evidence_pos - evidence_neg) / max(total, loss_config.eps),
                "state_probs": original["state_probs"][index].detach().float().cpu().tolist(),
                "state_nll": float(per_sample_nll[index].detach().float().cpu()),
                "state_nll_gradient_wrt_signed_evidence": float(
                    delta_gradients[index].detach().float().cpu()
                ),
                "gate_logits_valid": original["gate_logits"][index, valid]
                .detach()
                .float()
                .cpu()
                .tolist(),
                "evidence_topk_indices": hard_indices.detach().cpu().tolist(),
            }
        )

    state_direction = {
        state: {
            "count": sum(row["state"] == state for row in rows),
            "mean_gradient_wrt_signed_evidence": float(
                sum(
                    row["state_nll_gradient_wrt_signed_evidence"]
                    for row in rows
                    if row["state"] == state
                )
                / max(1, sum(row["state"] == state for row in rows))
            ),
        }
        for state in STATE_NAMES
    }
    return {
        "step": int(step),
        "auxiliary_weight": float(auxiliary_weight),
        "loss_values": {
            name: float(value.detach().float().cpu()) for name, value in losses.items()
        },
        "weighted_component_values": {
            name: float(value.detach().float().cpu()) for name, value in components.items()
        },
        "gradient_norms": gradient_norms,
        "state_vs_auxiliary_gradient_cosines": gradient_cosines,
        "state_nll_signed_evidence_direction": state_direction,
        "samples": rows,
    }


def aggregate_optimization_audits(
    audits: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate deterministic per-quartet audits without hiding any quartet."""

    materialized = list(audits)
    if not materialized:
        raise ValueError("at least one quartet audit is required")

    def summarize(values: list[float]) -> dict[str, float | int]:
        return {
            "count": len(values),
            "mean": float(statistics.fmean(values)),
            "median": float(statistics.median(values)),
            "maximum": float(max(values)),
            "minimum": float(min(values)),
        }

    norm_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    cosine_values: dict[str, list[float]] = defaultdict(list)
    state_direction_values: dict[str, list[tuple[int, float]]] = defaultdict(list)
    samples: list[dict[str, Any]] = []
    for audit in materialized:
        for component, groups in audit["gradient_norms"].items():
            for group, value in groups.items():
                norm_values[component][group].append(float(value))
        for component, value in audit["state_vs_auxiliary_gradient_cosines"].items():
            if value is not None:
                cosine_values[component].append(float(value))
        for state, row in audit["state_nll_signed_evidence_direction"].items():
            state_direction_values[state].append(
                (int(row["count"]), float(row["mean_gradient_wrt_signed_evidence"]))
            )
        samples.extend(audit["samples"])

    direction_summary: dict[str, dict[str, float | int]] = {}
    for state in STATE_NAMES:
        rows = state_direction_values.get(state, [])
        count = sum(row_count for row_count, _ in rows)
        weighted = sum(row_count * value for row_count, value in rows)
        direction_summary[state] = {
            "count": count,
            "mean_gradient_wrt_signed_evidence": (
                float(weighted / count) if count else float("nan")
            ),
        }

    return {
        "audit_scope": "all_train_quartets",
        "quartet_count": len(materialized),
        "sample_count": len(samples),
        "gradient_norm_summary": {
            component: {
                group: summarize(values)
                for group, values in sorted(groups.items())
            }
            for component, groups in sorted(norm_values.items())
        },
        "state_vs_auxiliary_gradient_cosine_summary": {
            component: summarize(values)
            for component, values in sorted(cosine_values.items())
        },
        "state_nll_signed_evidence_direction": direction_summary,
        "samples": samples,
        "quartet_audits": materialized,
    }


def trainable_gradient_norms(module: torch.nn.Module) -> dict[str, float]:
    """Measure current accumulated gradients before clipping, by model group."""

    squared_by_group: dict[str, float] = defaultdict(float)
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        vector = parameter.grad.detach().float().reshape(-1)
        squared_by_group[_parameter_group(name)] += float(torch.dot(vector, vector).cpu())
    norms = {
        group: math.sqrt(max(0.0, squared))
        for group, squared in sorted(squared_by_group.items())
    }
    norms["all"] = math.sqrt(max(0.0, sum(squared_by_group.values())))
    return norms


def summarize_clipping_history(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize observed pre-clip norms and clipping frequency."""

    materialized = list(rows)
    if not materialized:
        return {
            "optimizer_steps": 0,
            "clipped_steps": 0,
            "clipped_fraction": 0.0,
            "preclip_total_norm": None,
            "clip_coefficient": None,
            "preclip_group_norms": {},
        }

    def summarize(values: list[float]) -> dict[str, float]:
        return {
            "mean": float(statistics.fmean(values)),
            "median": float(statistics.median(values)),
            "maximum": float(max(values)),
            "minimum": float(min(values)),
        }

    group_values: dict[str, list[float]] = defaultdict(list)
    for row in materialized:
        for group, value in row["preclip_group_norms"].items():
            group_values[group].append(float(value))
    clipped_steps = sum(bool(row["clipped"]) for row in materialized)
    return {
        "optimizer_steps": len(materialized),
        "clipped_steps": clipped_steps,
        "clipped_fraction": float(clipped_steps / len(materialized)),
        "preclip_total_norm": summarize(
            [float(row["preclip_total_norm"]) for row in materialized]
        ),
        "clip_coefficient": summarize(
            [float(row["clip_coefficient"]) for row in materialized]
        ),
        "preclip_group_norms": {
            group: summarize(values) for group, values in sorted(group_values.items())
        },
    }


def summarize_prediction_evidence(
    rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize E+, E-, T, delta, rho and probabilities by state and finding."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    materialized = list(rows)
    for row in materialized:
        grouped[(STATE_NAMES[int(row["target"])], str(row["canonical_statement_id"]))].append(row)

    payload: dict[str, Any] = {"records": len(materialized), "groups": {}}
    for (state, finding), group_rows in sorted(grouped.items()):
        evidence_pos = [float(row["evidence_pos"]) for row in group_rows]
        evidence_neg = [float(row["evidence_neg"]) for row in group_rows]
        totals = [pos + neg for pos, neg in zip(evidence_pos, evidence_neg)]
        deltas = [pos - neg for pos, neg in zip(evidence_pos, evidence_neg)]
        probabilities = np_array([row["original_probs"] for row in group_rows])
        key = f"{finding}|{state}"
        payload["groups"][key] = {
            "canonical_statement_id": finding,
            "state": state,
            "count": len(group_rows),
            "mean_evidence_pos": sum(evidence_pos) / len(group_rows),
            "mean_evidence_neg": sum(evidence_neg) / len(group_rows),
            "mean_total_evidence": sum(totals) / len(group_rows),
            "mean_signed_evidence": sum(deltas) / len(group_rows),
            "mean_rho": sum(
                delta / max(total, 1e-8) for delta, total in zip(deltas, totals)
            )
            / len(group_rows),
            "mean_state_probs": probabilities,
        }
    return payload


def np_array(rows: list[list[float]]) -> list[float]:
    if not rows:
        return []
    width = len(rows[0])
    return [sum(float(row[index]) for row in rows) / len(rows) for index in range(width)]
