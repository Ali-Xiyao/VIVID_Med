#!/usr/bin/env python
"""Run the fixed-seed MORPH-CXR morphology separability survival gate."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arise_cxr.box_supervision import box_ranking_loss  # noqa: E402
from arise_cxr.mil_verifier import MILVerifierConfig, PatchMILVerifier, mil_binary_loss  # noqa: E402
from morph_cxr.data import MorphCachedDataset, collate_morph  # noqa: E402
from morph_cxr.experts import (  # noqa: E402
    EXPERT_TYPES,
    FINDING_TO_EXPERT,
    MorphologyConceptExpert,
    MorphologyExpertConfig,
    concept_monotonicity_deltas,
)
from morph_cxr.protocol import canonical_sha256, file_sha256  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _bbox_iou_from_scores(scores: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> float:
    target = target.bool() & valid.bool()
    if not bool(target.any()):
        return float("nan")
    k = int(target.sum())
    masked = scores.masked_fill(~valid.bool(), float("-inf"))
    selected = torch.zeros_like(target)
    selected[masked.topk(k).indices] = True
    height = width = int(round(target.numel() ** 0.5))
    if height * width != target.numel():
        raise ValueError("MORPH bbox IoU requires a square frozen grid")

    def box(mask: torch.Tensor) -> tuple[int, int, int, int]:
        y, x = torch.where(mask.reshape(height, width))
        return int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1

    ax1, ay1, ax2, ay2 = box(selected)
    bx1, by1, bx2, by2 = box(target)
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = ix * iy
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return float(intersection / union) if union else 0.0


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    expert_type: str,
    loader: DataLoader,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        tokens = batch["tokens"].to(device)
        valid = batch["valid"].to(device)
        statements = batch["statement_indices"].to(device)
        if expert_type == "generic":
            output = model(tokens, valid, statements)
            attention = torch.sigmoid(output["patch_logits"])
            concepts = weights = None
        else:
            output = model(tokens, valid, statements, grid_hw=batch["grid_hw"])
            attention = output["attention"]
            concepts = output["concepts"]
            weights = output["concept_weights"]
        logits = output["patch_logits"].detach().cpu()
        attention_cpu = attention.detach().cpu()
        valid_cpu = batch["valid"].bool()
        for index, finding in enumerate(batch["findings"]):
            spatial_hit = boundary_hit = geometry_iou = moment_score = None
            if bool(batch["spatial_available"][index]):
                maximum = int(logits[index].masked_fill(~valid_cpu[index], float("-inf")).argmax())
                spatial_hit = int(batch["spatial"][index, maximum])
                boundary_hit = int(batch["boundary"][index, maximum])
                geometry_iou = _bbox_iou_from_scores(
                    attention_cpu[index], batch["spatial"][index], valid_cpu[index]
                )
                if expert_type != "generic":
                    predicted = torch.cat((output["centroid"][index], output["spread"][index])).cpu()
                    moment_score = float((1.0 - 0.25 * (predicted - batch["moments"][index]).abs().sum()).clamp(0, 1))
            monotonic_min = None
            if concepts is not None and weights is not None:
                monotonic_min = float(
                    concept_monotonicity_deltas(concepts[index : index + 1], weights[index : index + 1]).min().cpu()
                )
            rows.append(
                {
                    "sample_id": batch["sample_ids"][index],
                    "patient_sha256": batch["patients"][index],
                    "finding": finding,
                    "label": int(batch["labels"][index]),
                    "margin": float(output["margin"][index].cpu()),
                    "spatial_hit": spatial_hit,
                    "boundary_hit": boundary_hit,
                    "geometry_iou": geometry_iou,
                    "moment_score": moment_score,
                    "monotonic_min_delta": monotonic_min,
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for finding in sorted({str(row["finding"]) for row in rows}):
        subset = [row for row in rows if row["finding"] == finding]
        labels = np.asarray([row["label"] for row in subset])
        margins = np.asarray([row["margin"] for row in subset])
        positives = [row for row in subset if row["label"] == 1]
        block: dict[str, Any] = {
            "records": len(subset),
            "auroc": float(roc_auc_score(labels, margins)),
            "auprc": float(average_precision_score(labels, margins)),
        }
        for metric in ("spatial_hit", "boundary_hit", "geometry_iou", "moment_score"):
            values = [float(row[metric]) for row in positives if row[metric] is not None]
            block[metric] = float(np.mean(values)) if values else None
        deltas = [row["monotonic_min_delta"] for row in subset if row["monotonic_min_delta"] is not None]
        block["monotonic_min_delta"] = float(min(deltas)) if deltas else None
        result[finding] = block
    return result


def train_one(
    expert_type: str,
    seed: int,
    train: MorphCachedDataset,
    validation: MorphCachedDataset,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    set_seed(seed)
    visual_dim = int(train[0]["tokens"].shape[-1])
    num_statements = len(train.statement_to_index)
    if expert_type == "generic":
        model: torch.nn.Module = PatchMILVerifier(
            MILVerifierConfig(
                visual_dim=visual_dim,
                num_statements=num_statements,
                temperature=float(config["temperature"]),
                max_pool_weight=float(config["generic_max_pool_weight"]),
            )
        ).to(device)
    else:
        model = MorphologyConceptExpert(
            MorphologyExpertConfig(
                visual_dim=visual_dim,
                num_statements=num_statements,
                expert_type=expert_type,
                temperature=float(config["temperature"]),
            )
        ).to(device)
    strata = [(row["canonical_statement_id"], int(row["binary_label"])) for row in train.rows]
    counts = {key: strata.count(key) for key in set(strata)}
    weights = torch.tensor([1.0 / counts[key] for key in strata], dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=int(config["batch_size"]) * int(config["max_steps"]),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        train,
        batch_size=int(config["batch_size"]),
        sampler=sampler,
        collate_fn=collate_morph,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        collate_fn=collate_morph,
        num_workers=0,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    events = []
    for step, batch in enumerate(loader, start=1):
        tokens = batch["tokens"].to(device)
        valid = batch["valid"].to(device)
        statements = batch["statement_indices"].to(device)
        labels = batch["labels"].to(device)
        if expert_type == "generic":
            output = model(tokens, valid, statements)
            target_mask = batch["spatial"].to(device)
        else:
            output = model(tokens, valid, statements, grid_hw=batch["grid_hw"])
            target_mask = (
                batch["boundary"] if expert_type == "boundary" else batch["spatial"]
            ).to(device)
        classification = mil_binary_loss(output["margin"], labels)
        ranking = box_ranking_loss(
            output["patch_logits"],
            output["valid_mask"],
            target_mask,
            batch["spatial_available"].to(device),
            margin=float(config["spatial_rank_margin"]),
            temperature=float(config["temperature"]),
        )
        geometry = output["margin"].sum() * 0.0
        if expert_type in ("geometry", "distribution"):
            available = batch["spatial_available"].to(device).bool()
            if bool(available.any()):
                predicted = torch.cat((output["centroid"], output["spread"]), dim=1)
                geometry = F.smooth_l1_loss(
                    predicted[available], batch["moments"].to(device)[available]
                )
        loss = (
            classification
            + float(config["spatial_loss_weight"]) * ranking
            + float(config["geometry_loss_weight"]) * geometry
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step in {1, int(config["max_steps"])}:
            events.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "classification_loss": float(classification.detach().cpu()),
                    "spatial_loss": float(ranking.detach().cpu()),
                    "geometry_loss": float(geometry.detach().cpu()),
                }
            )
        if step >= int(config["max_steps"]):
            break
    rows = evaluate_model(model, expert_type, validation_loader, device)
    return {"events": events, "metrics": summarize(rows), "validation_rows": rows}, {
        "schema_version": "morph-expert-checkpoint-v1",
        "expert_type": expert_type,
        "seed": seed,
        "statement_to_index": train.statement_to_index,
        "model": model.state_dict(),
    }


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/morph_cxr/separability_v0.yaml"
    )
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "morph-separability-config-v1":
        raise ValueError("unexpected MORPH config")
    data_dir = ROOT / config["data_dir"]
    cache_dir = ROOT / config["cache_dir"]
    output_dir = args.output_dir or ROOT / config["output_dir"]
    if (output_dir / "result.json").exists():
        raise FileExistsError(output_dir / "result.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    train = MorphCachedDataset(cache_dir, data_dir / "morph_separability_manifest.jsonl", "train")
    validation = MorphCachedDataset(
        cache_dir, data_dir / "morph_separability_manifest.jsonl", "validation"
    )
    if train.statement_to_index != validation.statement_to_index:
        raise ValueError("MORPH statement vocabularies differ")
    device = torch.device(args.device or config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("MORPH separability gate requires local CUDA")

    runs: dict[str, list[dict[str, Any]]] = {expert: [] for expert in EXPERT_TYPES}
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    for seed in map(int, config["seeds"]):
        for expert in EXPERT_TYPES:
            run, checkpoint = train_one(expert, seed, train, validation, config, device)
            path = checkpoints / f"{expert}_seed{seed}.pt"
            torch.save(checkpoint, path)
            run["checkpoint_sha256"] = file_sha256(path)
            run["seed"] = seed
            runs[expert].append(run)
            print(json.dumps({"expert": expert, "seed": seed, **run["metrics"]}, sort_keys=True), flush=True)

    per_finding: dict[str, Any] = {}
    for finding, prescribed in FINDING_TO_EXPERT.items():
        generic_runs = runs["generic"]
        if prescribed == "region_boundary":
            expert_aurocs = []
            expert_spatial = []
            generic_spatial = []
            for region_run, boundary_run, generic_run in zip(
                runs["region"], runs["boundary"], generic_runs, strict=True
            ):
                region_rows = {
                    row["sample_id"]: row
                    for row in region_run["validation_rows"]
                    if row["finding"] == finding
                }
                boundary_rows = {
                    row["sample_id"]: row
                    for row in boundary_run["validation_rows"]
                    if row["finding"] == finding
                }
                if set(region_rows) != set(boundary_rows):
                    raise ValueError("MORPH region/boundary ensemble identities differ")
                labels = np.asarray(
                    [region_rows[key]["label"] for key in sorted(region_rows)]
                )
                margins = np.asarray(
                    [
                        0.5
                        * (
                            region_rows[key]["margin"]
                            + boundary_rows[key]["margin"]
                        )
                        for key in sorted(region_rows)
                    ]
                )
                expert_aurocs.append(float(roc_auc_score(labels, margins)))
                expert_spatial.append(
                    0.5
                    * (
                        region_run["metrics"][finding]["spatial_hit"]
                        + boundary_run["metrics"][finding]["boundary_hit"]
                    )
                )
                generic_spatial.append(
                    0.5
                    * (
                        generic_run["metrics"][finding]["spatial_hit"]
                        + generic_run["metrics"][finding]["boundary_hit"]
                    )
                )
        else:
            metric = {
                "boundary": "boundary_hit",
                "region": "spatial_hit",
                "geometry": "geometry_iou",
                "distribution": "moment_score",
            }[prescribed]
            expert_aurocs = [run["metrics"][finding]["auroc"] for run in runs[prescribed]]
            expert_spatial = [run["metrics"][finding][metric] for run in runs[prescribed]]
            generic_spatial = [run["metrics"][finding][metric] for run in generic_runs]
        generic_aurocs = [run["metrics"][finding]["auroc"] for run in generic_runs]
        discrimination_gain = _median(expert_aurocs) - _median(generic_aurocs)
        spatial_gain = _median(expert_spatial) - _median(generic_spatial)
        per_finding[finding] = {
            "prescribed_expert": prescribed,
            "generic_median_auroc": _median(generic_aurocs),
            "expert_median_auroc": _median(expert_aurocs),
            "discrimination_gain": discrimination_gain,
            "generic_median_spatial": _median(generic_spatial),
            "expert_median_spatial": _median(expert_spatial),
            "spatial_gain": spatial_gain,
            "pass": bool(
                discrimination_gain > float(config["minimum_discrimination_gain"])
                and spatial_gain > float(config["minimum_spatial_gain"])
            ),
        }

    passed_findings = sum(int(block["pass"]) for block in per_finding.values())
    prescribed_macro = _median(
        [block["expert_median_auroc"] for block in per_finding.values()]
    )
    generic_macro = _median(
        [block["generic_median_auroc"] for block in per_finding.values()]
    )
    noninferiority_gap = prescribed_macro - generic_macro
    monotonic_min = min(
        run["metrics"][finding]["monotonic_min_delta"]
        for expert, expert_runs in runs.items()
        if expert != "generic"
        for run in expert_runs
        for finding in run["metrics"]
    )
    gate_pass = bool(
        passed_findings >= int(config["minimum_passing_findings"])
        and noninferiority_gap >= -float(config["concept_auroc_noninferiority"])
        and monotonic_min >= -float(config["monotonic_tolerance"])
    )
    result = {
        "schema_version": "morph-separability-result-v1",
        "status": "pass_open_full_proposal" if gate_pass else "fail_stop_before_full_morph",
        "formal_result": False,
        "confirmatory_evidence": False,
        "source_role": "exposed_development_only",
        "data_lock_sha256": file_sha256(data_dir / "data_lock.json"),
        "cache_lock_sha256": file_sha256(cache_dir / "cache_lock.json"),
        "config_sha256": file_sha256(args.config),
        "per_finding": per_finding,
        "passed_findings": passed_findings,
        "minimum_passing_findings": int(config["minimum_passing_findings"]),
        "concept_direct_median_auroc_gap": noninferiority_gap,
        "concept_auroc_noninferiority": float(config["concept_auroc_noninferiority"]),
        "monotonic_min_delta": monotonic_min,
        "gate_pass": gate_pass,
        "runs": runs,
        "chexlocalize_test_opened": False,
        "vindr_test_opened": False,
        "qwen35_4b_9b_used": False,
        "pixel_intervention_used": False,
    }
    result["canonical_sha256"] = canonical_sha256(result)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
