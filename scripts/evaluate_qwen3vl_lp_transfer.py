"""Evaluate a trained Qwen3-VL linear-probe head on an external UMS split."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_qwen3vl_vision_lp import (  # noqa: E402
    Qwen3VLLPCollator,
    Qwen3VLVisionLinearProbe,
    UMSPILDataset,
    compute_classification_metrics,
    compute_loss,
    load_config,
    load_model_and_processor,
    move_tensors_to_device,
    prepare_labels_for_metrics,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lp-config", required=True, type=Path)
    parser.add_argument("--probe-dir", type=Path)
    parser.add_argument("--probe-name", default="best_probe.pt")
    parser.add_argument("--val-ums-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=1000, help="Use 0 or a negative value to evaluate all available rows.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device")
    parser.add_argument("--verify-images", action="store_true")
    return parser.parse_args()


def verify_images(dataset: UMSPILDataset) -> dict[str, Any]:
    missing: list[str] = []
    for idx, sample in enumerate(dataset.samples):
        path = dataset._image_path(sample)  # noqa: SLF001 - transfer audit needs the resolved path.
        if not path.exists():
            missing.append(str(path))
            if len(missing) >= 20:
                break
    return {"checked": len(dataset.samples), "missing_count": len(missing), "missing_examples": missing}


def subset_metrics(y_true: np.ndarray, y_prob: np.ndarray, label_names: list[str], sizes: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    total = int(y_true.shape[0])
    for size in sizes:
        if total < size:
            out[f"nih_{size}"] = {"records": total, "status": "insufficient_records"}
            continue
        probs = y_prob[:size]
        labels = y_true[:size]
        pred = (probs >= 0.5).astype(int)
        out[f"nih_{size}"] = {
            "records": size,
            "metrics": compute_classification_metrics(
                y_true=labels,
                y_pred=pred,
                y_prob=probs,
                label_names=label_names,
                threshold=0.5,
            ),
        }
    pred_all = (y_prob >= 0.5).astype(int)
    out["all_available"] = {
        "records": total,
        "metrics": compute_classification_metrics(
            y_true=y_true,
            y_pred=pred_all,
            y_prob=y_prob,
            label_names=label_names,
            threshold=0.5,
        ),
    }
    return out


@torch.no_grad()
def evaluate_collect(
    model: Qwen3VLVisionLinearProbe,
    dataloader: DataLoader,
    device: torch.device,
    policy: str,
    label_names: list[str],
    progress_path: Path | None = None,
) -> dict[str, Any]:
    model.eval()
    losses = []
    probs = []
    labels = []
    processed = 0
    total = len(dataloader.dataset) if hasattr(dataloader, "dataset") else None
    if progress_path:
        save_json(progress_path, {"event": "start", "processed": 0, "total": total, "time": time.time()})
    for batch in tqdm(dataloader, desc="Transfer validating", leave=False):
        batch = move_tensors_to_device(batch, device)
        logits = model(pixel_values=batch["pixel_values"], image_grid_thw=batch["image_grid_thw"])
        loss = compute_loss(logits, batch["labels"], policy)
        losses.append(float(loss.detach().cpu()))
        probs.append(torch.sigmoid(logits).cpu().numpy())
        prepared = prepare_labels_for_metrics(batch["labels"].cpu().numpy(), policy)
        labels.append(prepared)
        processed += int(prepared.shape[0])
        if progress_path and (processed == total or processed % 500 == 0):
            save_json(progress_path, {"event": "running", "processed": processed, "total": total, "time": time.time()})
    y_prob = np.concatenate(probs, axis=0)
    y_true = np.concatenate(labels, axis=0)
    all_pred = (y_prob >= 0.5).astype(int)
    if progress_path:
        save_json(progress_path, {"event": "metrics", "processed": processed, "total": total, "time": time.time()})
    return {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "metrics": compute_classification_metrics(
            y_true=y_true,
            y_pred=all_pred,
            y_prob=y_prob,
            label_names=label_names,
            threshold=0.5,
        ),
        "subset_metrics": subset_metrics(y_true=y_true, y_prob=y_prob, label_names=label_names, sizes=[1000, 5000]),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.lp_config)
    if args.device:
        config["device"] = args.device
    set_seed(int(config.get("seed", 42)))
    requested_device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    probe_dir = args.probe_dir or Path(config["training"]["output_dir"])
    probe_path = probe_dir / args.probe_name
    probe = torch.load(probe_path, map_location="cpu")
    label_names = list(probe["label_names"])
    feature_dim = int(probe["feature_dim"])

    model, processor = load_model_and_processor(config, device)
    dataset = UMSPILDataset(
        data_root=args.data_root,
        ums_jsonl_path=args.val_ums_path,
        label_names=label_names,
        max_samples=None if args.max_samples <= 0 else args.max_samples,
    )
    image_audit = verify_images(dataset)
    if args.verify_images and image_audit["missing_count"]:
        raise SystemExit(json.dumps({"error": "missing_images", **image_audit}, indent=2))

    collator = Qwen3VLLPCollator(processor=processor, prompt=config["data"].get("processor_prompt", "Classify the chest X-ray findings."))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
        pin_memory=True,
    )
    probe_model = Qwen3VLVisionLinearProbe(
        visual=model.model.visual,
        feature_dim=feature_dim,
        num_labels=len(label_names),
        freeze_backbone=True,
    ).to(device)
    probe_model.head.load_state_dict(probe["head"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    result = evaluate_collect(
        model=probe_model,
        dataloader=dataloader,
        device=device,
        policy=config["training"].get("uncertain_policy", "ignore"),
        label_names=label_names,
        progress_path=args.output_dir / "progress.json",
    )
    metrics = {
        "lp_config": str(args.lp_config),
        "probe_path": str(probe_path),
        "val_ums_path": args.val_ums_path,
        "data_root": args.data_root,
        "max_samples": "all_available" if args.max_samples <= 0 else args.max_samples,
        "evaluated_records": len(dataset),
        "feature_dim": feature_dim,
        "label_names": label_names,
        "image_audit": image_audit,
        "elapsed_seconds": time.time() - started,
        "val_loss": result["val_loss"],
        "metrics": result["metrics"],
        "subset_metrics": result["subset_metrics"],
    }
    shutil.copy2(args.lp_config, args.output_dir / "source_lp_config.yaml")
    save_json(args.output_dir / "transfer_metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
