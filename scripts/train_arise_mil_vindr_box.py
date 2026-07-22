#!/usr/bin/env python
"""Fine-tune ARISE patch-MIL with image-disjoint VinDr-train boxes."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arise_cxr.box_supervision import (  # noqa: E402
    VinDrBoxCachedDataset,
    box_ranking_loss,
    collate_vindr_box,
)
from arise_cxr.mil_verifier import PatchMILVerifier, mil_binary_loss  # noqa: E402
from bives_cxr.provenance import canonical_json_sha256, file_sha256  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


@torch.no_grad()
def evaluate(model: PatchMILVerifier, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    rows = []
    for batch in loader:
        output = model(
            batch["patch_tokens"].to(device),
            batch["valid_mask"].to(device),
            batch["statement_indices"].to(device),
        )
        logits = output["patch_logits"].cpu()
        valid = batch["valid_mask"].bool()
        boxes = batch["box_mask"].bool()
        for index, (finding, label, margin) in enumerate(
            zip(
                batch["canonical_statement_ids"],
                batch["binary_labels"].tolist(),
                output["margin"].cpu().tolist(),
                strict=True,
            )
        ):
            pointing_hit = None
            box_rank_success = None
            if bool(batch["box_available"][index]):
                masked_logits = logits[index].masked_fill(~valid[index], float("-inf"))
                maximum = int(masked_logits.argmax())
                pointing_hit = int(boxes[index, maximum])
                inside_max = float(logits[index, boxes[index] & valid[index]].max())
                outside_max = float(logits[index, ~boxes[index] & valid[index]].max())
                box_rank_success = int(inside_max > outside_max)
            rows.append(
                {
                    "finding": finding,
                    "label": int(label),
                    "margin": float(margin),
                    "pointing_hit": pointing_hit,
                    "box_rank_success": box_rank_success,
                }
            )
    per_finding = {}
    for finding in sorted({row["finding"] for row in rows}):
        subset = [row for row in rows if row["finding"] == finding]
        labels = np.asarray([row["label"] for row in subset], dtype=np.int64)
        margins = np.asarray([row["margin"] for row in subset], dtype=np.float64)
        positives = [row for row in subset if row["pointing_hit"] is not None]
        per_finding[finding] = {
            "records": len(subset),
            "auroc": float(roc_auc_score(labels, margins)),
            "auprc": float(average_precision_score(labels, margins)),
            "margin_std": float(margins.std()),
            "pointing_hit": float(np.mean([row["pointing_hit"] for row in positives])),
            "box_rank_success": float(
                np.mean([row["box_rank_success"] for row in positives])
            ),
            "box_records": len(positives),
        }
    macro = {
        metric: float(np.mean([block[metric] for block in per_finding.values()]))
        for metric in ("auroc", "auprc", "margin_std", "pointing_hit", "box_rank_success")
    }
    return {"per_finding": per_finding, "macro": macro}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/arise_cxr/mil_vindr_box_finetune.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("variant") != "arise_patch_mil_vindr_box_overlap_v2":
        raise ValueError("unexpected ARISE VinDr-box variant")
    train_cache_dir = ROOT / config["train_cache_dir"]
    val_cache_dir = ROOT / config["val_cache_dir"]
    data_dir = ROOT / config["data_dir"]
    initial_path = ROOT / config["initial_checkpoint"]
    output_dir = args.output_dir or ROOT / config["output_dir"]
    if (output_dir / "result.json").exists():
        raise FileExistsError(output_dir / "result.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(config["seed"]))
    device = torch.device(args.device or config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ARISE VinDr-box fine-tuning requires local CUDA")
    train = VinDrBoxCachedDataset(
        train_cache_dir, data_dir / "vindr_box_sc_train.jsonl", "train"
    )
    val = VinDrBoxCachedDataset(
        val_cache_dir, data_dir / "vindr_box_sc_val.jsonl", "val"
    )
    if train.statement_to_index != val.statement_to_index:
        raise ValueError("VinDr box train/val statement vocabularies differ")
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    if initial.get("schema_version") != "arise-patch-mil-checkpoint-v1":
        raise ValueError("unexpected initial ARISE MIL checkpoint")
    if initial.get("statement_to_index") != train.statement_to_index:
        raise ValueError("initial/VinDr statement vocabularies differ")
    from arise_cxr.mil_verifier import MILVerifierConfig

    model = PatchMILVerifier(MILVerifierConfig(**initial["model_config"])).to(device)
    model.load_state_dict(initial["model"], strict=True)
    strata = [(str(row["canonical_statement_id"]), int(row["binary_label"])) for row in train.rows]
    counts = {key: strata.count(key) for key in set(strata)}
    weights = torch.tensor([1.0 / counts[key] for key in strata], dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights,
        len(train),
        replacement=True,
        generator=torch.Generator().manual_seed(int(config["seed"])),
    )
    train_loader = DataLoader(
        train,
        batch_size=int(config["batch_size"]),
        sampler=sampler,
        collate_fn=collate_vindr_box,
        num_workers=0,
    )
    val_loader = DataLoader(
        val,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        collate_fn=collate_vindr_box,
        num_workers=0,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    initial_metrics = evaluate(model, val_loader, device)
    floor = initial_metrics["macro"]["auroc"] - float(
        config["classification_auroc_noninferiority"]
    )
    events = [{"step": 0, "validation": initial_metrics}]
    best = None

    def consider(step: int, metrics: dict) -> None:
        noninferior = metrics["macro"]["auroc"] >= floor
        key = (
            int(noninferior),
            metrics["macro"]["pointing_hit"] if noninferior else metrics["macro"]["auroc"],
            metrics["macro"]["auroc"],
            -step,
        )
        nonlocal best
        if best is None or key > best["key"]:
            checkpoint = {
                "schema_version": "arise-patch-mil-vindr-box-checkpoint-v2",
                "variant": config["variant"],
                "model_config": initial["model_config"],
                "model": model.state_dict(),
                "statement_to_index": train.statement_to_index,
                "initial_checkpoint_sha256": file_sha256(initial_path),
                "train_cache_lock_sha256": file_sha256(
                    train_cache_dir / "cache_lock.json"
                ),
                "val_cache_lock_sha256": file_sha256(
                    val_cache_dir / "cache_lock.json"
                ),
                "data_lock_sha256": file_sha256(data_dir / "data_lock.json"),
                "box_supervision_sha256": file_sha256(
                    ROOT / "arise_cxr/box_supervision.py"
                ),
                "step": step,
                "validation": metrics,
                "test_opened": False,
            }
            torch.save(checkpoint, output_dir / "best.pt")
            best = {"key": key, "step": step, "validation": metrics}

    consider(0, initial_metrics)
    step = 0
    max_steps = int(args.max_steps or config["max_steps"])
    while step < max_steps:
        model.train()
        for batch in train_loader:
            output = model(
                batch["patch_tokens"].to(device),
                batch["valid_mask"].to(device),
                batch["statement_indices"].to(device),
            )
            classification = mil_binary_loss(
                output["margin"], batch["binary_labels"].to(device)
            )
            box = box_ranking_loss(
                output["patch_logits"],
                output["valid_mask"],
                batch["box_mask"].to(device),
                batch["box_available"].to(device),
                margin=float(config["box_rank_margin"]),
                temperature=float(config["box_rank_temperature"]),
            )
            loss = classification + float(config["box_loss_weight"]) * box
            loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["max_grad_norm"])
                )
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % int(config["eval_interval"]) == 0 or step == max_steps:
                metrics = evaluate(model, val_loader, device)
                event = {
                    "step": step,
                    "train_loss": float(loss.detach().cpu()),
                    "classification_loss": float(classification.detach().cpu()),
                    "box_loss": float(box.detach().cpu()),
                    "preclip_grad_norm": grad_norm,
                    "validation": metrics,
                }
                events.append(event)
                print(json.dumps(event, sort_keys=True), flush=True)
                consider(step, metrics)
                model.train()
            if step >= max_steps:
                break
    assert best is not None
    pointing_improvement = (
        best["validation"]["macro"]["pointing_hit"]
        - initial_metrics["macro"]["pointing_hit"]
    )
    noninferior = best["validation"]["macro"]["auroc"] >= floor
    localization_pass = pointing_improvement >= float(
        config["minimum_pointing_hit_improvement"]
    )
    result = {
        "schema_version": "arise-patch-mil-vindr-box-training-result-v2",
        "status": "ready_for_oracle" if noninferior and localization_pass else "fail_stop_before_oracle",
        "formal_result": False,
        "confirmatory_evidence": False,
        "test_opened": False,
        "initial_checkpoint_sha256": file_sha256(initial_path),
        "train_cache_lock_sha256": file_sha256(
            train_cache_dir / "cache_lock.json"
        ),
        "val_cache_lock_sha256": file_sha256(val_cache_dir / "cache_lock.json"),
        "data_lock_sha256": file_sha256(data_dir / "data_lock.json"),
        "box_supervision_sha256": file_sha256(
            ROOT / "arise_cxr/box_supervision.py"
        ),
        "config_sha256": file_sha256(args.config),
        "checkpoint_sha256": file_sha256(output_dir / "best.pt"),
        "initial_validation": initial_metrics,
        "best_step": best["step"],
        "best_validation": best["validation"],
        "classification_noninferiority_floor": floor,
        "classification_noninferiority_pass": noninferior,
        "pointing_hit_improvement": pointing_improvement,
        "minimum_pointing_hit_improvement": float(config["minimum_pointing_hit_improvement"]),
        "localization_improvement_pass": localization_pass,
        "events": events,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    result["canonical_sha256"] = canonical_json_sha256(result)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
