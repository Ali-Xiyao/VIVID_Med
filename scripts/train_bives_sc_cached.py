"""Train B1 dense or B2 sparse bipolar polarity from frozen Qwen tokens."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bives_cxr.expert_sc import file_sha256  # noqa: E402
from bives_cxr.polarity import (  # noqa: E402
    BipolarPolarityModel,
    CachedSCDataset,
    PolarityModelConfig,
    collate_cached_sc,
    polarity_loss,
)
from bives_cxr.polarity_metrics import polarity_metrics  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: BipolarPolarityModel,
    loader: DataLoader,
    device: torch.device,
    tau_p: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    losses = []
    for batch in loader:
        output = model(
            batch["patch_tokens"].to(device),
            batch["valid_mask"].to(device),
            batch["statement_indices"].to(device),
        )
        labels = batch["binary_labels"].to(device)
        losses.append(float(polarity_loss(output["signed_evidence"], labels, tau_p)))
        for index, sample_id in enumerate(batch["sample_ids"]):
            rows.append(
                {
                    "sample_id": sample_id,
                    "unit_id": batch["unit_ids"][index],
                    "patient_id": batch["patient_ids"][index],
                    "canonical_statement_id": batch["canonical_statement_ids"][index],
                    "binary_label": int(labels[index].cpu()),
                    "evidence_pos": float(output["evidence_pos"][index].cpu()),
                    "evidence_neg": float(output["evidence_neg"][index].cpu()),
                    "signed_evidence": float(output["signed_evidence"][index].cpu()),
                    "support_probability": float(output["support_probability"][index].cpu()),
                    "evidence_topk_indices": torch.where(output["gate"][index].cpu() > 0.5)[0].tolist(),
                }
            )
    metrics, _ = polarity_metrics(rows)
    metrics["polarity_loss"] = float(np.mean(losses))
    return metrics, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["variant"] not in {"B1_dense", "B2_sparse_exact_k"}:
        raise ValueError("variant must be B1_dense or B2_sparse_exact_k")
    cache_dir = Path(config["cache_dir"])
    output_dir = Path(config["output_dir"])
    if (output_dir / "metrics_final.json").exists():
        raise FileExistsError(output_dir / "metrics_final.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 17))
    set_seed(seed)
    device = torch.device(config.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
    train_dataset = CachedSCDataset(cache_dir, "train")
    val_dataset = CachedSCDataset(cache_dir, "val")
    if train_dataset.statement_to_index != val_dataset.statement_to_index:
        raise ValueError("train/val cached statement vocabularies differ")
    first = train_dataset[0]
    model_config = PolarityModelConfig(
        visual_dim=int(first["patch_tokens"].shape[-1]),
        num_statements=len(train_dataset.statement_to_index),
        statement_dim=int(config.get("statement_dim", 128)),
        fusion_dim=int(config.get("fusion_dim", 256)),
        evidence_max=float(config.get("evidence_max", 8.0)),
        mode="dense" if config["variant"] == "B1_dense" else "sparse_exact_k",
        topk=int(config.get("topk", 16)),
        gate_temperature=float(config.get("gate_temperature", 0.5)),
        tau_p=float(config.get("tau_p", 1.0)),
        contextual_layers=int(config.get("contextual_layers", 1)),
        contextual_heads=int(config.get("contextual_heads", 4)),
        contextual_dropout=float(config.get("contextual_dropout", 0.0)),
    )
    model = BipolarPolarityModel(model_config).to(device)
    strata = [
        (str(row["canonical_statement_id"]), int(row["binary_label"]))
        for row in train_dataset.rows
    ]
    counts = {key: strata.count(key) for key in set(strata)}
    weights = torch.tensor([1.0 / counts[key] for key in strata], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(train_dataset),
        replacement=True,
        generator=generator,
    )
    batch_size = int(config.get("batch_size", 32))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_cached_sc,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_cached_sc,
        num_workers=0,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    max_steps = int(config.get("max_steps", 500))
    eval_interval = int(config.get("eval_interval", 25))
    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    step = 0
    epoch = 0
    best_auroc = float("-inf")
    best_step = None
    events = []
    model.train()
    while step < max_steps:
        for batch in train_loader:
            output = model(
                batch["patch_tokens"].to(device),
                batch["valid_mask"].to(device),
                batch["statement_indices"].to(device),
            )
            loss = polarity_loss(
                output["signed_evidence"],
                batch["binary_labels"].to(device),
                model_config.tau_p,
            )
            loss.backward()
            preclip = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % eval_interval == 0 or step == max_steps:
                validation, prediction_rows = evaluate(model, val_loader, device, model_config.tau_p)
                event = {
                    "step": step,
                    "train_loss": float(loss.detach().cpu()),
                    "preclip_total_norm": preclip,
                    "clip_coefficient": min(1.0, max_grad_norm / max(preclip, 1e-12)),
                    "validation": validation,
                }
                events.append(event)
                print(json.dumps(event, ensure_ascii=False))
                macro_auroc = float(validation["macro"]["auroc"])
                if macro_auroc > best_auroc:
                    best_auroc = macro_auroc
                    best_step = step
                    torch.save(
                        {
                            "variant": config["variant"],
                            "model_config": model_config.__dict__,
                            "model": model.state_dict(),
                            "statement_to_index": train_dataset.statement_to_index,
                            "cache_lock_sha256": file_sha256(cache_dir / "cache_lock.json"),
                            "step": step,
                            "validation": validation,
                        },
                        output_dir / "best.pt",
                    )
                    with (output_dir / "best_val_predictions.jsonl").open("w", encoding="utf-8") as handle:
                        for row in prediction_rows:
                            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                model.train()
            if step >= max_steps:
                break
        epoch += 1
    if best_step is None:
        raise RuntimeError("no validation checkpoint was selected")
    best = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    validation, prediction_rows = evaluate(model, val_loader, device, model_config.tau_p)
    validation, thresholds = polarity_metrics(
        prediction_rows,
        target_specificity=float(config.get("target_specificity", 0.9)),
    )
    threshold_payload = {
        finding: {
            "support_probability_threshold": threshold,
            "target_specificity": float(config.get("target_specificity", 0.9)),
            "source": "weak_sc_validation_only",
        }
        for finding, threshold in thresholds.items()
    }
    (output_dir / "locked_thresholds.json").write_text(
        json.dumps(threshold_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    result = {
        "variant": config["variant"],
        "formal_result": False,
        "model_family": "Qwen3.5-2B frozen patch tokens",
        "cache_lock_sha256": file_sha256(cache_dir / "cache_lock.json"),
        "seed": seed,
        "best_step": best_step,
        "validation": validation,
        "events": events,
        "threshold_source": "weak_sc_validation_only",
        "external_test_used_for_selection": False,
    }
    (output_dir / "metrics_final.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
