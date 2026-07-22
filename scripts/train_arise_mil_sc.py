#!/usr/bin/env python
"""Train the local Qwen3.5 patch-MIL dense S/C verifier."""

from __future__ import annotations

import argparse
import json
import math
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

from arise_cxr.mil_verifier import MILVerifierConfig, PatchMILVerifier, mil_binary_loss
from bives_cxr.polarity import CachedSCDataset, collate_cached_sc
from bives_cxr.provenance import canonical_json_sha256, file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/arise_cxr/mil_dense_sc.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


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
        for finding, label, margin in zip(
            batch["canonical_statement_ids"],
            batch["binary_labels"].tolist(),
            output["margin"].cpu().tolist(),
            strict=True,
        ):
            rows.append({"finding": finding, "label": int(label), "margin": float(margin)})
    per_finding = {}
    for finding in sorted({row["finding"] for row in rows}):
        subset = [row for row in rows if row["finding"] == finding]
        labels = np.asarray([row["label"] for row in subset], dtype=np.int64)
        margins = np.asarray([row["margin"] for row in subset], dtype=np.float64)
        probabilities = 1.0 / (1.0 + np.exp(-margins))
        per_finding[finding] = {
            "records": len(subset),
            "auroc": float(roc_auc_score(labels, margins)),
            "auprc": float(average_precision_score(labels, margins)),
            "margin_mean": float(margins.mean()),
            "margin_std": float(margins.std()),
            "nll": float(-np.mean(labels * np.log(probabilities + 1e-12) + (1 - labels) * np.log(1 - probabilities + 1e-12))),
        }
    macro = {
        metric: float(np.mean([value[metric] for value in per_finding.values()]))
        for metric in ("auroc", "auprc", "margin_std", "nll")
    }
    return {"per_finding": per_finding, "macro": macro}


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("variant") != "arise_patch_mil_dense_sc_v1":
        raise ValueError("unexpected ARISE MIL variant")
    cache_dir = ROOT / config["cache_dir"]
    output_dir = args.output_dir or (ROOT / config["output_dir"])
    if (output_dir / "result.json").exists():
        raise FileExistsError(output_dir / "result.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    set_seed(seed)
    device = torch.device(args.device or config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ARISE MIL training requires local CUDA")
    train = CachedSCDataset(cache_dir, "train")
    val = CachedSCDataset(cache_dir, "val")
    if train.statement_to_index != val.statement_to_index:
        raise ValueError("ARISE MIL train/val statement vocabularies differ")
    first = train[0]
    model_config = MILVerifierConfig(
        visual_dim=int(first["patch_tokens"].shape[-1]),
        num_statements=len(train.statement_to_index),
        temperature=float(config["temperature"]),
        max_pool_weight=float(config["max_pool_weight"]),
    )
    model = PatchMILVerifier(model_config).to(device)
    strata = [(str(row["canonical_statement_id"]), int(row["binary_label"])) for row in train.rows]
    counts = {key: strata.count(key) for key in set(strata)}
    weights = torch.tensor([1.0 / counts[key] for key in strata], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, len(train), replacement=True, generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train, batch_size=int(config["batch_size"]), sampler=sampler, collate_fn=collate_cached_sc, num_workers=0)
    val_loader = DataLoader(val, batch_size=int(config["batch_size"]), shuffle=False, collate_fn=collate_cached_sc, num_workers=0)
    optimizer = AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    best = None
    events = []
    step = 0
    max_steps = int(args.max_steps if args.max_steps is not None else config["max_steps"])
    if max_steps <= 0:
        raise ValueError("ARISE MIL max_steps must be positive")
    while step < max_steps:
        model.train()
        for batch in train_loader:
            output = model(batch["patch_tokens"].to(device), batch["valid_mask"].to(device), batch["statement_indices"].to(device))
            loss = mil_binary_loss(output["margin"], batch["binary_labels"].to(device))
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"])))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % int(config["eval_interval"]) == 0 or step == max_steps:
                metrics = evaluate(model, val_loader, device)
                event = {"step": step, "train_loss": float(loss.detach().cpu()), "preclip_grad_norm": grad_norm, "validation": metrics}
                events.append(event)
                print(json.dumps(event, sort_keys=True), flush=True)
                score = metrics["macro"]["auroc"]
                if best is None or score > best["score"]:
                    checkpoint = {
                        "schema_version": "arise-patch-mil-checkpoint-v1",
                        "variant": config["variant"],
                        "model_config": model_config.to_dict(),
                        "model": model.state_dict(),
                        "statement_to_index": train.statement_to_index,
                        "cache_lock_sha256": file_sha256(cache_dir / "cache_lock.json"),
                        "step": step,
                        "validation": metrics,
                        "test_opened": False,
                    }
                    torch.save(checkpoint, output_dir / "best.pt")
                    best = {"score": score, "step": step, "validation": metrics}
                model.train()
            if step >= max_steps:
                break
    if best is None:
        raise RuntimeError("ARISE MIL did not produce a checkpoint")
    result = {
        "schema_version": "arise-patch-mil-training-result-v1",
        "status": "complete_development",
        "formal_result": False,
        "confirmatory_evidence": False,
        "test_opened": False,
        "model_family": "Qwen3.5-2B frozen patch tokens",
        "cache_lock_sha256": file_sha256(cache_dir / "cache_lock.json"),
        "config_sha256": file_sha256(args.config),
        "checkpoint_sha256": file_sha256(output_dir / "best.pt"),
        "best_step": best["step"],
        "validation": best["validation"],
        "events": events,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    result["canonical_sha256"] = canonical_json_sha256(result)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
